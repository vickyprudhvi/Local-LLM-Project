"""Phase H.6 — AT&T (T) regression, from REAL live SEC EDGAR + Yahoo Finance
data (tests/fixtures/t_regression.json, captured 2026-08-17).

THE BUG THIS FILE EXISTS TO PIN DOWN
====================================
Running a valuation of AT&T in August 2026, the report stated FY2026 revenue
growth guidance of 3%-4% and fed it into the DCF's revenue-growth assumption.
AT&T published no such guidance. What its January 2026 earnings release
actually says, in one flattened run of bullet points, is:

    The Company's long-term outlook for 2026-2028 includes:
    Service revenue growth in the low-single-digit range annually.
    Advanced Connectivity service revenue growth in the mid-single-digit
      range annually, including expected growth of 5%+ in 2026.
    Legacy service revenue decline of 20%+ in 2026 ...
    Adjusted EBITDA* growth in the 3% to 4% range in 2026, improving to
      5% or better in 2028.
    Adjusted EPS* of $2.25 to $2.35 in 2026 ...

The 3%-4% is ADJUSTED EBITDA growth. The extractor's `revenue_growth`
keyword matched "Service revenue growth", found no number in that bullet,
and its fixed 110-character window ran on into the next bullet and took the
first range it found. Nothing stopped it, because EBITDA was not in the
metric table at all — so there was no competing metric keyword in between to
act as a boundary. The result was stored on a GAAP basis (it is a non-GAAP
measure), under the wrong metric, for the wrong quantity, and went straight
into `dcf.assumption.revenue_growth`.

Three further defects in the same run, all pinned here:

* Operating cash flow produced a "trailing twelve months" ending 2025-12-31
  while revenue's ended 2026-06-30 — T moved to the ...ContinuingOperations
  tag in fiscal 2026 — and free cash flow was derived by subtracting a capex
  window ending 2026-06-30 from it.
* A historical revenue CAGR spanning the WarnerMedia and DirecTV separations
  was weighed against management's current outlook as if the two described
  the same company.
* "net debt / FCF declines below 5x while margins remain stable" was filed
  under DOWNGRADE conditions. Falling leverage with stable margins is
  favourable.

Nothing here asserts a desired RECOMMENDATION for AT&T, and nothing forces
the valuation toward any external analysis — the assertions are about which
METRIC each guidance figure belongs to, which PERIOD each input covers, and
whether the provenance says so.
"""

import json
from pathlib import Path

import pytest

import tools.finance_tools as finance_tools
from finance import guidance as G
from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.dcf import AssumptionSourceType
from finance.freshness import ValuationFreshness
from finance.provider import AlphaVantageClient, ProviderResponse
from finance.quota import AlphaVantageQuotaLedger, QuotaLedger
from finance.research_pipeline import (
    CONDITION_BIDIRECTIONAL,
    CONDITION_FAVOURABLE,
    CONDITION_UNFAVOURABLE,
    classify_condition_direction,
)
from finance.sec_datasets import resolve_sec_dataset
from finance.structural_breaks import HistoricalComparability
from finance.workflow import (
    build_compact_synthesis_payload,
    render_compact_report,
    run_full_stock_analysis,
)
from finance.yahoo_datasets import resolve_yahoo_dataset
from tests.test_finance_cache import FakeClock
from tools.base import ToolFailure
from tools.executor import ToolExecutor
from tools.models import MARKET_DATA_INVALID_SYMBOL
from tools.registry import ToolRegistry

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "t_regression.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

SYMBOL = "T"

# ---- ground truth, straight from the captured filings ----
Q2_2026_PERIOD_END = "2026-06-30"
FY2025_PERIOD_END = "2025-12-31"

# The January 2026 release's outlook, as AT&T actually stated it.
GUIDED_ADJUSTED_EBITDA_GROWTH = (0.03, 0.04)      # "3% to 4% range in 2026"
GUIDED_ADJUSTED_EPS = (2.25, 2.35)                # "$2.25 to $2.35 in 2026"
# ...and what it did NOT state: a consolidated revenue-growth range.

# The separations that make a five-year CAGR non-comparable.
STRUCTURAL_BREAK_YEARS = ("2021", "2022")


class YahooFixtureClient:
    provider_id = "yahoo"

    def fetch(self, dataset, arguments):
        payload = FIXTURE["yahoo"].get(dataset.dataset_id)
        if payload is None:
            raise ToolFailure(MARKET_DATA_INVALID_SYMBOL,
                              f"no fixture data for {dataset.dataset_id}")
        return ProviderResponse(payload=payload, provider_metadata={},
                                byte_count=len(json.dumps(payload)))


class SecFixtureClient:
    provider_id = "sec"

    def __init__(self):
        self.calls = []

    def fetch(self, dataset, arguments):
        self.calls.append(dataset.dataset_id)
        if dataset.dataset_id == "filing_document":
            key = f"{arguments.get('accession')}/{arguments.get('document')}"
            text = (FIXTURE["sec"].get("filing_documents") or {}).get(key)
            if text is None:
                raise ToolFailure(MARKET_DATA_INVALID_SYMBOL,
                                  f"no fixture document for {key}")
            return ProviderResponse(
                payload={"document_text": text, "byte_count": len(text),
                         "accession": arguments.get("accession"),
                         "document": arguments.get("document"), "truncated": False},
                provider_metadata={}, byte_count=len(text))
        payload = FIXTURE["sec"].get(dataset.dataset_id)
        if payload is None:
            raise ToolFailure(MARKET_DATA_INVALID_SYMBOL,
                              f"no fixture data for {dataset.dataset_id}")
        return ProviderResponse(payload=payload, provider_metadata={},
                                byte_count=len(json.dumps(payload)))


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def wired(tmp_path, clock, monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "")
    monkeypatch.setenv("YAHOO_FINANCE_ENABLED", "true")
    monkeypatch.setenv("YAHOO_PERSONAL_USE_ACKNOWLEDGED", "true")
    monkeypatch.setenv("SEC_EDGAR_ENABLED", "true")
    monkeypatch.setenv("SEC_USER_AGENT", "TestApp/1.0 (contact: t@example.com)")
    for var in ("FINANCE_QUOTE_PROVIDER", "FINANCE_PRICE_HISTORY_PROVIDER",
                "FINANCE_CORPORATE_ACTIONS_PROVIDER", "FINANCE_COMPANY_PROFILE_PROVIDER",
                "FINANCE_ANALYST_ESTIMATES_PROVIDER"):
        monkeypatch.setenv(var, "yahoo")
    monkeypatch.setenv("FINANCE_US_FUNDAMENTALS_PROVIDER", "sec")
    monkeypatch.setenv("GUIDANCE_INGESTION_ENABLED", "true")

    yahoo_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "yahoo_m.sqlite3"), clock=clock),
        ledger=QuotaLedger(path=str(tmp_path / "yahoo_q.sqlite3"), clock=clock,
                           daily_limit=1_000_000),
        client=YahooFixtureClient(), provider_id="yahoo",
        dataset_resolver=resolve_yahoo_dataset,
        clock=clock, sleeper=lambda _s: None, min_request_interval_ms=0)
    finance_tools.set_yahoo_coordinator(yahoo_coordinator)

    sec_client = SecFixtureClient()
    sec_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "sec_m.sqlite3"), clock=clock),
        ledger=QuotaLedger(path=str(tmp_path / "sec_q.sqlite3"), clock=clock,
                           daily_limit=1_000_000),
        client=sec_client, provider_id="sec", dataset_resolver=resolve_sec_dataset,
        clock=clock, sleeper=lambda _s: None, min_request_interval_ms=0)
    finance_tools.set_sec_coordinator(sec_coordinator)

    av_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "av_m.sqlite3"), clock=clock),
        ledger=AlphaVantageQuotaLedger(path=str(tmp_path / "av_q.sqlite3"), clock=clock,
                                       daily_limit=100),
        client=AlphaVantageClient(), clock=clock, sleeper=lambda _s: None,
        jitter=lambda: 0.5)
    finance_tools.set_coordinator(av_coordinator)

    registry = ToolRegistry()
    for tool_cls in finance_tools.ALL_FINANCE_TOOL_CLASSES:
        registry.register(tool_cls())
    executor = ToolExecutor(registry)

    yield executor, sec_client
    finance_tools.set_coordinator(None)
    finance_tools.set_yahoo_coordinator(None)
    finance_tools.set_sec_coordinator(None)


def run_t(wired):
    executor, _sec = wired
    return run_full_stock_analysis(executor, SYMBOL)


def _state(result):
    return result.facts.get("current_financial_state") or {}


def _guidance(result):
    return (result.facts.get("management_guidance") or {}).get("metrics") or {}


def _base(result):
    return next(s for s in result.facts["dcf"]["scenarios"] if s["scenario"] == "base")


# ---------------------------------------------------------------------------
# Section 26, check 1: revenue guidance is NOT populated from EBITDA guidance
# ---------------------------------------------------------------------------

def test_1_revenue_growth_guidance_is_not_populated_from_ebitda_guidance(wired):
    """The headline failure. AT&T's 3%-4% is adjusted EBITDA growth, and the
    report presented it as FY2026 revenue growth guidance."""
    metrics = _guidance(run_t(wired))
    revenue_growth = metrics.get("revenue_growth")
    if revenue_growth is not None:
        assert not (revenue_growth["low"] == pytest.approx(0.03)
                    and revenue_growth["high"] == pytest.approx(0.04)), (
            "AT&T's adjusted-EBITDA growth range was stored as revenue growth")


def test_1b_the_dcf_revenue_growth_assumption_never_cites_an_ebitda_metric(wired):
    result = run_t(wired)
    provenance = (_base(result).get("assumptions") or {}).get("assumption_provenance") or {}
    entry = provenance.get("revenue_growth") or {}
    shared = (result.facts["dcf"].get("shared_assumption_provenance") or {})
    entry = entry or shared.get("revenue_growth") or {}
    derivation = (entry.get("derivation") or "").lower()
    assert "ebitda" not in derivation, (
        f"the revenue-growth assumption cites EBITDA: {derivation}")
    for evidence_id in (entry.get("source_evidence_ids") or []):
        assert "ebitda" not in str(evidence_id).lower()


# ---------------------------------------------------------------------------
# Section 26, checks 2-4: each guided metric keeps its own identity
# ---------------------------------------------------------------------------

def test_2_service_revenue_guidance_stays_labelled_service_revenue(wired):
    """AT&T guides service revenue and segment service revenue. Whatever is
    extracted from those bullets must carry a service/segment scope — never
    the consolidated `revenue_growth` name."""
    metrics = _guidance(run_t(wired))
    for name, entry in metrics.items():
        if "service" in name or name == "segment_revenue_growth":
            assert entry["scope"] in ("service", "segment"), (name, entry["scope"])
            assert name != G.GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH


def test_3_ebitda_growth_remains_ebitda_growth(wired):
    metrics = _guidance(run_t(wired))
    ebitda = metrics.get("adjusted_ebitda_growth")
    assert ebitda is not None, "AT&T's adjusted-EBITDA growth guidance was not extracted"
    assert (ebitda["low"], ebitda["high"]) == pytest.approx(GUIDED_ADJUSTED_EBITDA_GROWTH)
    # A non-GAAP measure must never be recorded on a GAAP basis (section 6).
    assert ebitda["basis"] == G.BASIS_ADJUSTED
    assert ebitda["fiscal_period"] == "FY2026"
    assert "EBITDA" in ebitda["source_excerpt"]


def test_4_free_cash_flow_guidance_remains_free_cash_flow_guidance(wired):
    """AT&T states free cash flow as a FLOOR ("$18 billion+ in 2026"), not a
    range. A floor is real guidance and must be captured as a floor — never
    as a midpoint expectation, and never under another metric's name."""
    metrics = _guidance(run_t(wired))
    fcf = metrics.get("free_cash_flow")
    if fcf is not None:
        assert fcf["bound_type"] == G.GuidanceBound.AT_LEAST
        assert "cash flow" in fcf["source_excerpt"].lower()


def test_4b_every_extracted_guidance_metric_passes_its_own_validation(wired):
    """Section 11: no number without a metric identity, period, units, basis
    and source filing reaches the report."""
    for name, entry in _guidance(run_t(wired)).items():
        assert entry["fiscal_period"], name
        assert entry["units"] in ("ratio", "currency", "currency_per_share"), name
        assert entry["basis"] in (G.BASIS_GAAP, G.BASIS_ADJUSTED,
                                  G.BASIS_COMPANY_DEFINED, G.BASIS_NONE), name
        assert entry["source_accession"], name
        assert entry["source_evidence_ids"], name
        assert entry["guidance_id"], name
        assert entry["status"] == G.GuidanceStatus.CURRENT, name


def test_4c_gaap_and_adjusted_measures_are_never_mixed(wired):
    metrics = _guidance(run_t(wired))
    for name, entry in metrics.items():
        if name.startswith("adjusted_"):
            assert entry["basis"] == G.BASIS_ADJUSTED, name


# ---------------------------------------------------------------------------
# Section 26, checks 5-6: structural breaks and what history may anchor
# ---------------------------------------------------------------------------

def test_5_historical_comparability_is_reduced_by_the_separations(wired):
    """AT&T tagged discontinued operations through the DirecTV and
    WarnerMedia separations, inside the window a five-year CAGR spans."""
    comparability = _state(run_t(wired))["historical_comparability"]
    assert comparability["historical_comparability_status"] == \
        HistoricalComparability.STRUCTURAL_BREAK
    affected = " ".join(comparability["affected_periods"])
    assert any(year in affected for year in STRUCTURAL_BREAK_YEARS), affected
    assert comparability["summary"]


def test_6_the_long_period_cagr_is_not_the_forecast_anchor(wired):
    """Section 13: when the history spans a break, the CAGR is retained as
    context and something comparable anchors the forecast instead."""
    result = run_t(wired)
    provenance = ((_base(result).get("assumptions") or {})
                  .get("assumption_provenance") or {}).get("revenue_growth") or {}
    provenance = provenance or (result.facts["dcf"].get("shared_assumption_provenance")
                                or {}).get("revenue_growth") or {}
    derivation = provenance.get("derivation") or ""
    # Either a forward or a comparable-period source anchored it; what must
    # NOT happen is the full-history CAGR being copied in as the forecast.
    assert provenance.get("source_type") in (
        AssumptionSourceType.MANAGEMENT_GUIDANCE,
        AssumptionSourceType.TTM_CALCULATION,
        AssumptionSourceType.HISTORICAL_CALCULATION,
        AssumptionSourceType.LLM_PROPOSED,
    )
    if "CAGR" in derivation:
        assert "STRUCTURAL BREAK" in derivation or "structural break" in derivation


def test_6b_the_year_one_growth_assumption_is_not_the_broken_cagr(wired):
    """AT&T's five-year reported CAGR is negative, dominated by the
    separations. A forecast anchored on it would assert the continuing
    business shrinks at the rate the disposals did."""
    result = run_t(wired)
    growth = (_base(result).get("assumptions") or {}).get("revenue_growth")
    year_one = growth[0] if isinstance(growth, list) else growth
    assert year_one > -0.05, (
        f"year-1 growth of {year_one:.2%} looks like the separation-driven CAGR")


# ---------------------------------------------------------------------------
# Section 26, check 7: condition DIRECTION
# ---------------------------------------------------------------------------

def test_7_falling_leverage_with_stable_margins_is_favorable_not_a_downgrade():
    """The exact string the live T report filed under downgrade conditions."""
    assert classify_condition_direction(
        "net debt / FCF declines below 5x while margins remain stable"
    ) == CONDITION_FAVOURABLE


def test_7b_rising_leverage_is_still_unfavorable():
    for text in ("net debt to EBITDA rises above 3.5x",
                 "leverage increases while free cash flow declines",
                 "total debt grows faster than EBITDA"):
        assert classify_condition_direction(text) == CONDITION_UNFAVOURABLE, text


def test_7c_falling_revenue_is_still_unfavorable():
    """The subject-aware rule must not flip every 'declines' to favourable."""
    for text in ("revenue declines for two consecutive quarters",
                 "free cash flow falls below $15 billion",
                 "operating margin declines below 18%"):
        assert classify_condition_direction(text) == CONDITION_UNFAVOURABLE, text


def test_7d_an_information_change_is_still_bidirectional():
    assert classify_condition_direction(
        "a company-specific WACC becomes available") == CONDITION_BIDIRECTIONAL


# ---------------------------------------------------------------------------
# Section 26, check 8: post-quarter material events
# ---------------------------------------------------------------------------

def test_8_post_quarter_filings_produce_a_freshness_warning(wired):
    """AT&T files debt-offering prospectus supplements continuously. Any that
    land after the balance-sheet date mean the equity bridge may already
    describe a capital structure the company has changed."""
    result = run_t(wired)
    state = _state(result)
    events = state["post_balance_sheet_events"]
    if events:
        assert all(e["filed"] > state["financial_as_of"] for e in events)
        assert state["valuation_freshness"] in (
            ValuationFreshness.MOSTLY_CURRENT_WITH_EVENT_WARNING,
            ValuationFreshness.STALE_INPUT_WARNING)
        readiness = result.facts["research_readiness"]
        assert any("balance sheet" in reason for reason in readiness["reasons"])


def test_8b_the_balance_sheet_itself_is_still_the_latest_filed_one(wired):
    """A post-quarter event warning must never be confused with using a stale
    balance sheet: the bridge still uses the newest one that exists."""
    state = _state(run_t(wired))
    assert state["financial_as_of"] == Q2_2026_PERIOD_END
    assert state["latest_quarterly_period"] == Q2_2026_PERIOD_END
    assert state["latest_annual_period"] == FY2025_PERIOD_END


# ---------------------------------------------------------------------------
# Period integrity — the free-cash-flow window
# ---------------------------------------------------------------------------

def test_free_cash_flow_covers_exactly_one_period(wired):
    """T renamed its operating-cash-flow tag for fiscal 2026. The old code
    produced an operating-cash-flow TTM ending 2025-12-31, a capex TTM ending
    2026-06-30, and published the difference as free cash flow."""
    flows = _state(run_t(wired))["flows"]
    ocf, capex, fcf = (flows["operating_cash_flow"], flows["capital_expenditure"],
                       flows["free_cash_flow"])
    if fcf["value"] is not None:
        assert ocf["as_of_date"] == capex["as_of_date"] == fcf["as_of_date"]
        assert ocf["period_start"] == capex["period_start"] == fcf["period_start"]
        assert fcf["value"] == pytest.approx(ocf["value"] - abs(capex["value"]))


def test_the_revenue_ttm_ends_at_the_latest_reported_quarter(wired):
    revenue = _state(run_t(wired))["flows"]["revenue"]
    assert revenue["source"] == "ttm_calculation"
    assert revenue["as_of_date"] == Q2_2026_PERIOD_END
    assert revenue["ttm"]["validation_status"] == "valid"


# ---------------------------------------------------------------------------
# Section 26, check 9: the recommendation stays the model's
# ---------------------------------------------------------------------------

def test_9_no_recommendation_is_produced_deterministically(wired):
    """Nothing in the deterministic pipeline decides BUY/HOLD/SELL/AVOID —
    that stays with the FinalInvestmentSynthesizer, which does not run here."""
    result = run_t(wired)
    assert "recommendation" not in result.facts
    report = render_compact_report(result, build_compact_synthesis_payload(result),
                                   pipeline_result=None)
    assert "Recommendation: unavailable" in report


def test_the_report_names_the_guidance_period_and_the_guided_metrics(wired):
    """Section 24: a reader must be able to see WHAT was guided, so a mapping
    error like this one is visible on the page rather than only in the JSON."""
    result = run_t(wired)
    report = render_compact_report(result, build_compact_synthesis_payload(result),
                                   pipeline_result=None)
    assert "Management guidance:" in report
    if _guidance(result):
        assert "guided metrics:" in report
        # The line must not claim consolidated revenue guidance AT&T never gave.
        guidance_line = next(line for line in report.splitlines()
                             if "guided metrics:" in line)
        assert "revenue_growth" not in guidance_line.replace("service_revenue_growth", "") \
            .replace("segment_revenue_growth", "")
