"""Phase H.4 — AOS regression, from REAL live SEC EDGAR + Yahoo Finance data
(tests/fixtures/aos_regression.json, captured 2026-08-12; see the fixture's
own "note" field for exactly what was trimmed).

THE BUG THIS FILE EXISTS TO PIN DOWN
====================================
Running a valuation of A. O. Smith in August 2026, the DCF used FY2025
annual balance-sheet values even though two 10-Qs had been filed since:

    total debt   FY2025 (2025-12-31)   ~$155M     <- what the DCF used
                 Q2 2026 (2026-06-30)  ~$637M     <- what was available

AOS financed the Leonard Valve acquisition in January 2026 (there is an 8-K
item 2.03, "Creation of a Direct Financial Obligation", filed 2026-01-06).
The equity bridge was subtracting a net debt figure that predated the
acquisition by six months, understating it by roughly $482M — and nothing
warned, because `_dcf_inputs_from_facts` read
`statements["annual"]["balance_sheet"][0]` and FY2025 genuinely WAS the
latest ANNUAL filing. Separately, the historical revenue CAGR was copied
into every forecast year's growth assumption while AOS had publicly guided
to 2-3% sales growth for 2026.

This fixture is the whole failure in one company: an annual filing, two
newer quarterly filings whose balance sheets differ materially, and three
successive earnings releases whose guidance was lowered twice.

Nothing here asserts a desired RECOMMENDATION for AOS, and nothing forces
the valuation toward any external analysis — the assertions are about which
PERIOD each input came from and whether the provenance says so.
"""

import json
from pathlib import Path

import pytest

import tools.finance_tools as finance_tools
from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.dcf import AssumptionSourceType
from finance.freshness import (
    DCF_STALE_DEBT_INPUT,
    DataCompleteness,
    ValuationFreshness,
)
from finance.provider import AlphaVantageClient, ProviderResponse
from finance.quota import AlphaVantageQuotaLedger, QuotaLedger
from finance.sec_datasets import resolve_sec_dataset
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

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "aos_regression.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

SYMBOL = "AOS"

# ---- ground truth, straight from the captured filings (raw reported
# figures, independent of anything this codebase calculates) ----
FY2025_PERIOD_END = "2025-12-31"
Q2_2026_PERIOD_END = "2026-06-30"

FY2025_LONG_TERM_DEBT = 112_700_000.0
FY2025_CURRENT_PORTION_LTD = 42_300_000.0
FY2025_TOTAL_DEBT = FY2025_LONG_TERM_DEBT + FY2025_CURRENT_PORTION_LTD   # 155.0M
FY2025_CASH = 174_500_000.0

Q2_2026_LONG_TERM_DEBT = 598_000_000.0
Q2_2026_CURRENT_PORTION_LTD = 39_500_000.0
Q2_2026_TOTAL_DEBT = Q2_2026_LONG_TERM_DEBT + Q2_2026_CURRENT_PORTION_LTD  # 637.5M
Q2_2026_CASH = 181_300_000.0

# FY2026 guidance as stated in the 2026-07-30 earnings release.
GUIDED_SALES_GROWTH = (0.02, 0.03)
GUIDED_DILUTED_EPS = (3.60, 3.75)
GUIDED_ADJUSTED_EPS = (3.70, 3.85)
# ...and as stated in the SUPERSEDED 2026-04-30 release, before it was
# narrowed. Present in the fixture precisely so supersession is testable.
SUPERSEDED_SALES_GROWTH = (0.02, 0.04)


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


def run_aos(wired):
    executor, _sec = wired
    return run_full_stock_analysis(executor, SYMBOL)


def _state(result):
    return result.facts.get("current_financial_state") or {}


def _base(result):
    return next(s for s in result.facts["dcf"]["scenarios"] if s["scenario"] == "base")


# ---------------------------------------------------------------------------
# Section 14, checks 1-3: the equity bridge uses the CURRENT balance sheet
# ---------------------------------------------------------------------------

def test_1_dcf_does_not_use_fy2025_cash_when_q2_2026_cash_exists(wired):
    result = run_aos(wired)
    cash = _state(result)["balance_sheet"]["cash_and_cash_equivalents"]
    assert cash["value"] == pytest.approx(Q2_2026_CASH)
    assert cash["as_of_date"] == Q2_2026_PERIOD_END
    assert cash["value"] != pytest.approx(FY2025_CASH)
    assert _base(result)["cash_and_cash_equivalents"] == pytest.approx(Q2_2026_CASH)


def test_2_dcf_does_not_use_fy2025_debt_when_q2_2026_debt_exists(wired):
    """The headline failure. A 4.1x difference, silently, for six months."""
    result = run_aos(wired)
    total_debt = _state(result)["total_debt"]
    assert total_debt["value"] == pytest.approx(Q2_2026_TOTAL_DEBT)
    assert total_debt["as_of_date"] == Q2_2026_PERIOD_END
    assert total_debt["value"] != pytest.approx(FY2025_TOTAL_DEBT)
    assert _base(result)["total_debt"] == pytest.approx(Q2_2026_TOTAL_DEBT)


def test_3_net_debt_uses_the_current_quarter_balance_sheet(wired):
    result = run_aos(wired)
    expected = Q2_2026_TOTAL_DEBT - Q2_2026_CASH
    assert _base(result)["net_debt"] == pytest.approx(expected)
    # ...and is materially different from what the annual-only bridge gave.
    assert abs(expected - (FY2025_TOTAL_DEBT - FY2025_CASH)) > 400_000_000


def test_3b_the_material_debt_change_is_reported_not_silently_absorbed(wired):
    """Using the right number is necessary but not sufficient: a reader also
    has to be told the capital structure moved, or every ratio computed off
    annual data silently means something different."""
    result = run_aos(wired)
    findings = _state(result)["findings"]
    debt_finding = next((f for f in findings if f["code"] == DCF_STALE_DEBT_INPUT), None)
    assert debt_finding is not None
    assert debt_finding["prior_as_of"] == FY2025_PERIOD_END
    assert debt_finding["current_as_of"] == Q2_2026_PERIOD_END
    assert debt_finding["change_ratio"] > 1.0  # more than a doubling


# ---------------------------------------------------------------------------
# Section 14, checks 4-7: guidance reaches the builder, CAGR stays context
# ---------------------------------------------------------------------------

def test_4_current_management_guidance_reaches_the_assumption_builder(wired):
    result = run_aos(wired)
    guidance = result.facts.get("management_guidance") or {}
    growth = (guidance.get("metrics") or {}).get("revenue_growth")
    assert growth is not None, "FY2026 sales-growth guidance was not extracted"
    assert (growth["low"], growth["high"]) == pytest.approx(GUIDED_SALES_GROWTH)
    assert growth["fiscal_year"] == 2026

    provenance = _base(result)["assumptions"]["assumption_provenance"]["revenue_growth"]
    assert provenance["source_type"] == AssumptionSourceType.MANAGEMENT_GUIDANCE
    assert "dcf.guidance.revenue_growth.current" in (
        provenance.get("source_evidence_ids") or [])


def test_5_historical_cagr_remains_available_as_context(wired):
    result = run_aos(wired)
    cagr = (result.facts.get("fundamental_metrics") or {}).get("revenue_cagr") or {}
    assert cagr.get("value") is not None, "historical CAGR must remain available"


def test_6_historical_cagr_is_not_automatically_copied_into_year_one_growth(wired):
    """Section 7's separation, as an assertion.

    The old behaviour set year-1 growth (and every other year) to the
    historical CAGR. It must now be the guidance midpoint instead, and the
    provenance must SAY the CAGR was deliberately not used.
    """
    result = run_aos(wired)
    base = _base(result)
    cagr = (result.facts["fundamental_metrics"]["revenue_cagr"] or {})["value"]
    year_one_growth = base["assumptions"]["revenue_growth"][0]

    assert year_one_growth != pytest.approx(cagr, abs=1e-6)
    guided_midpoint = sum(GUIDED_SALES_GROWTH) / 2
    assert year_one_growth == pytest.approx(guided_midpoint, abs=1e-6)

    derivation = base["assumptions"]["assumption_provenance"]["revenue_growth"]["derivation"]
    assert "guidance" in derivation.lower()
    assert "context" in derivation.lower()


def test_7_base_year_one_assumption_sits_inside_stated_guidance(wired):
    result = run_aos(wired)
    year_one = _base(result)["assumptions"]["revenue_growth"][0]
    low, high = GUIDED_SALES_GROWTH
    assert low <= year_one <= high


def test_growth_is_a_per_year_path_that_converges_on_terminal_growth(wired):
    """Growth is a PATH, and it hands off to the perpetuity continuously.

    Asserted as the rule rather than "the numbers must differ", because on
    AOS's base case they legitimately do not: the guidance midpoint is 2.5%
    and base terminal growth is also 2.5%, so a correct fade from one to the
    other is flat. Bull and bear, whose anchors are moved by the scenario
    deltas, do vary — and every scenario must land exactly on its own
    terminal rate, which is the property that actually matters (before this,
    the last forecast year and the perpetuity could disagree outright).
    """
    result = run_aos(wired)
    varied = 0
    for scenario in result.facts["dcf"]["scenarios"]:
        growth = scenario["assumptions"]["revenue_growth"]
        terminal_growth = scenario["assumptions"]["terminal_growth"]
        assert len(growth) == len(scenario["forecast"])
        assert growth[-1] == pytest.approx(terminal_growth, abs=1e-6), scenario["scenario"]
        if len(set(growth)) > 1:
            varied += 1
            # A fade is monotone: it never wanders on the way to the target.
            deltas = [b - a for a, b in zip(growth, growth[1:])]
            assert all(d >= -1e-9 for d in deltas) or all(d <= 1e-9 for d in deltas), \
                scenario["scenario"]
    assert varied >= 2, "bull and bear anchors differ from their terminal rates and must fade"


# ---------------------------------------------------------------------------
# Section 14, check 8: provenance on every forecast assumption
# ---------------------------------------------------------------------------

def test_8_every_forecast_assumption_carries_provenance(wired):
    result = run_aos(wired)
    for scenario in result.facts["dcf"]["scenarios"]:
        provenance = scenario["assumptions"]["assumption_provenance"]
        for field in ("revenue_growth", "operating_margin", "capex_pct_revenue",
                      "depreciation_pct_revenue", "working_capital_pct_revenue",
                      "tax_rate", "wacc", "terminal_growth"):
            entry = provenance.get(field)
            assert entry is not None, f"{scenario['scenario']}.{field}"
            assert entry.get("source_type") in AssumptionSourceType.ALL, field
            assert entry.get("derivation"), f"{field} has no derivation"
            assert entry.get("approval_status") == "proposed", field


def test_8b_per_year_forecast_path_provenance_survives_the_dcf_engine(wired):
    """Per-year provenance has to make it THROUGH finance/dcf.py's validator,
    which rebuilds provenance entries from an allowlist — a field missing
    from that allowlist is silently dropped and the audit trail disappears
    without any error. That has happened before in this codebase."""
    result = run_aos(wired)
    entry = _base(result)["assumptions"]["assumption_provenance"]["revenue_growth"]
    path = entry.get("forecast_path")
    assert path, "per-year forecast provenance was dropped by the DCF validator"
    assert len(path) == len(_base(result)["assumptions"]["revenue_growth"])
    for step in path:
        assert step["forecast_year"] >= 1
        assert step["units"] == "ratio"
        assert step["source_type"] == AssumptionSourceType.MANAGEMENT_GUIDANCE
        assert step["applied_value"] is not None


# ---------------------------------------------------------------------------
# Section 14, check 9: the DCF stays deterministic
# ---------------------------------------------------------------------------

def test_9_dcf_is_deterministic_after_assumptions_are_validated(wired):
    """Same inputs, byte-identical valuation — twice through the whole
    workflow, not merely twice through the engine."""
    executor, _sec = wired
    first = run_full_stock_analysis(executor, SYMBOL)
    second = run_full_stock_analysis(executor, SYMBOL)
    assert (json.dumps(first.facts["dcf"]["scenarios"], sort_keys=True)
            == json.dumps(second.facts["dcf"]["scenarios"], sort_keys=True))


# ---------------------------------------------------------------------------
# Section 14, check 10: freshness is visible to the research pipeline
# ---------------------------------------------------------------------------

def test_10_research_pipeline_sees_valuation_freshness(wired):
    from finance.evidence import build_evidence_index

    result = run_aos(wired)
    compact = build_compact_synthesis_payload(result)
    assert compact["valuation_freshness"] in ValuationFreshness.ALL
    index = build_evidence_index(compact)
    assert "dcf.valuation_freshness" in index
    assert "dcf.basis.balance_sheet_as_of" in index
    assert index["dcf.basis.balance_sheet_as_of"].value == Q2_2026_PERIOD_END


def test_guidance_is_citable_and_labelled_forward_looking(wired):
    """Section 17: a stage cannot distinguish guidance from reported history
    unless guidance is separately citable AND says what kind of thing it is."""
    from finance.evidence import build_evidence_index

    result = run_aos(wired)
    index = build_evidence_index(build_compact_synthesis_payload(result))
    # Section 17/claim-horizon phase: the key is now period-qualified
    # ("dcf.guidance.revenue_growth.fy2026.current", not the bare
    # "...revenue_growth.current") so a quarterly and an annual statement of
    # the same metric are never the same citable id -- find it by prefix.
    item = next((v for k, v in index.items()
                if k.startswith("dcf.guidance.revenue_growth.")), None)
    assert item is not None
    assert item.source_type == AssumptionSourceType.MANAGEMENT_GUIDANCE
    assert "forward-looking" in (item.derivation or "").lower()
    assert "not a reported historical fact" in (item.derivation or "").lower()


# ---------------------------------------------------------------------------
# Freshness classification and the compact report (sections 13 and 19)
# ---------------------------------------------------------------------------

def test_data_completeness_and_valuation_freshness_are_separate_signals(wired):
    """The distinction the AOS failure turned on: every dataset present, and
    the valuation still potentially built on stale periods. One combined
    status cannot express that."""
    result = run_aos(wired)
    state = _state(result)
    assert state["data_completeness"] in DataCompleteness.ALL
    assert state["valuation_freshness"] in ValuationFreshness.ALL
    # AOS has a newer quarterly balance sheet AND trailing-twelve-month
    # flows, and the valuation uses both, so freshness is CURRENT.
    assert state["valuation_freshness"] == ValuationFreshness.CURRENT


def test_compact_report_states_the_financial_base_concisely(wired):
    result = run_aos(wired)
    compact = build_compact_synthesis_payload(result)
    text = render_compact_report(result, compact, pipeline_result=None)
    # CASY corrective patch: explicit dates, not calendar-quarter shorthand.
    # AOS runs on a calendar fiscal year, so both readings coincided here --
    # which is exactly why the bug hid until an April-year-end issuer
    # (Casey's) rendered its fiscal year end as "Q2 2026".
    assert "Financial base: trailing twelve months to 30 Jun 2026" in text
    assert "Balance sheet: 30 Jun 2026 (latest quarterly filing)" in text
    assert "Management guidance: FY2026 current guidance" in text
    assert "Q2 2026" not in text, "calendar-quarter labels are ambiguous for non-calendar years"


def test_compact_report_does_not_dump_a_freshness_audit(wired):
    """Section 19: compact mode gets the base, not the audit. The per-field
    provenance stays in the full facts."""
    result = run_aos(wired)
    compact = build_compact_synthesis_payload(result)
    text = render_compact_report(result, compact, pipeline_result=None)
    assert "retrieval_timestamp" not in text
    assert "freshness_status" not in text
    # ...while the full facts DO carry it.
    assert _state(result)["balance_sheet"]["long_term_debt"]["freshness_status"]


def test_ttm_flows_are_used_and_differ_from_the_last_fiscal_year(wired):
    result = run_aos(wired)
    revenue = _state(result)["flows"]["revenue"]
    assert revenue["source"] == "ttm_calculation"
    assert revenue["as_of_date"] == Q2_2026_PERIOD_END
    assert result.facts["dcf"]["base_revenue"] == pytest.approx(revenue["value"])
    # AOS's trailing twelve months genuinely differs from FY2025.
    fy2025_revenue = (result.facts["statements"]["annual"]["income_statement"][0]
                      ["values"]["revenue"])
    assert revenue["value"] != pytest.approx(fy2025_revenue)


# ---------------------------------------------------------------------------
# Guidance supersession on real successive filings (section 5)
# ---------------------------------------------------------------------------

def test_superseded_guidance_is_preserved_but_not_used_as_current(wired):
    """AOS lowered its 2026 outlook twice. The newest release is the
    guidance; the earlier ones are kept for comparison and must never be
    mistaken for current."""
    result = run_aos(wired)
    current = (result.facts.get("management_guidance") or {})
    superseded = result.facts.get("superseded_guidance") or []

    assert current["guidance_date"] == "2026-07-30"
    assert superseded, "earlier releases must be retained for comparison"
    assert all(r["guidance_date"] < current["guidance_date"] for r in superseded)

    current_growth = current["metrics"]["revenue_growth"]
    assert (current_growth["low"], current_growth["high"]) == pytest.approx(
        GUIDED_SALES_GROWTH)
    # The April release guided 2-4%; that number must not be the live one.
    april = next((r for r in superseded if r["guidance_date"] == "2026-04-30"), None)
    if april and "revenue_growth" in april["metrics"]:
        april_growth = april["metrics"]["revenue_growth"]
        assert (april_growth["low"], april_growth["high"]) == pytest.approx(
            SUPERSEDED_SALES_GROWTH)
        assert april_growth["high"] != current_growth["high"]


def test_gaap_and_adjusted_eps_guidance_are_kept_separate(wired):
    """Section 5: mixing the two bases is rejected. AOS states both, and
    they differ ($3.60-3.75 GAAP vs $3.70-3.85 adjusted)."""
    result = run_aos(wired)
    metrics = (result.facts.get("management_guidance") or {}).get("metrics") or {}
    gaap = metrics.get("earnings_per_share")
    adjusted = metrics.get("adjusted_earnings_per_share")
    assert gaap is not None and adjusted is not None
    assert (gaap["low"], gaap["high"]) == pytest.approx(GUIDED_DILUTED_EPS)
    assert (adjusted["low"], adjusted["high"]) == pytest.approx(GUIDED_ADJUSTED_EPS)
    assert gaap["basis"] == "GAAP"
    assert adjusted["basis"] == "adjusted"


def test_every_guidance_value_is_traceable_to_source_evidence(wired):
    """Section 5: no unsupported numeric guidance. Every value must name the
    filing it came from and quote the text supporting it."""
    result = run_aos(wired)
    guidance = result.facts.get("management_guidance") or {}
    assert guidance.get("accession")
    assert guidance.get("source_document")
    for name, metric in (guidance.get("metrics") or {}).items():
        assert metric.get("source_excerpt"), f"{name} has no supporting excerpt"
        assert metric.get("evidence_id"), f"{name} has no evidence id"
        assert metric.get("fiscal_year") == 2026, name


# ---------------------------------------------------------------------------
# Provider routing is unchanged (section 21, checks 28-32)
# ---------------------------------------------------------------------------

def test_only_reviewed_sec_datasets_are_requested(wired):
    """No generic URL fetcher and no browser fallback: the guidance path adds
    exactly two dataset ids, both pinned to one already-accepted filing."""
    executor, sec_client = wired
    run_full_stock_analysis(executor, SYMBOL)
    assert set(sec_client.calls) <= {
        "ticker_cik_map", "company_facts", "company_submissions", "filing_document",
    }


def test_guidance_absence_does_not_fail_the_workflow(wired, monkeypatch):
    """Section 20 / section 16's "no guidance still works". Disabling
    ingestion must degrade confidence, never the run."""
    executor, _sec = wired
    monkeypatch.setenv("GUIDANCE_INGESTION_ENABLED", "false")
    result = run_full_stock_analysis(executor, SYMBOL)

    assert result.facts["dcf"]["available"] is True
    assert result.facts.get("management_guidance") is None
    # The forecast falls back down the precedence ladder rather than failing.
    provenance = _base(result)["assumptions"]["assumption_provenance"]["revenue_growth"]
    assert provenance["source_type"] in (
        AssumptionSourceType.TTM_CALCULATION,
        AssumptionSourceType.HISTORICAL_CALCULATION,
        AssumptionSourceType.CONFIGURED_DEFAULT,
    )
    compact = build_compact_synthesis_payload(result)
    text = render_compact_report(result, compact, pipeline_result=None)
    # DIS/CASY corrective patch: the report states what it actually knows.
    # Ingestion is disabled here, so no release was read at all -- distinct
    # from having read some and extracted nothing, and distinct again from
    # the company having published none.
    assert "Management guidance: not retrieved" in text
    # Scoped to the guidance line: "unavailable" legitimately appears
    # elsewhere in this report (research stance, valuation view) because no
    # pipeline ran.
    assert "Management guidance: unavailable" not in text


# ---------------------------------------------------------------------------
# CASY corrective patch: non-calendar fiscal years must not be mislabelled
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("date_text,expected", [
    ("2026-04-30", "30 Apr 2026"),   # Casey's fiscal year end -- calendar Q2
    ("2026-06-30", "30 Jun 2026"),
    ("2025-12-31", "31 Dec 2025"),
    ("2026-01-31", "31 Jan 2026"),   # a January fiscal year end
    ("2026-08-31", "31 Aug 2026"),   # Costco's fiscal year end
])
def test_period_labels_are_explicit_dates_not_calendar_quarters(date_text, expected):
    """The label must not encode a quarter.

    Casey's FY2026 ended 2026-04-30 with no newer quarterly filing. Rendering
    that as "Q2 2026" turned a fiscal year end into a quarter and produced
    "Balance sheet: Q2 2026 (latest annual filing)" -- self-contradictory, and
    readable as interim data the analysis did not have.
    """
    from finance.workflow import _period_label
    assert _period_label(date_text) == expected
    assert "Q" not in _period_label(date_text)


@pytest.mark.parametrize("bad", [None, "", "not-a-date", "2026-13-45", 20260430])
def test_a_malformed_period_label_degrades_to_none(bad):
    """The caller falls back to the raw value rather than rendering junk."""
    from finance.workflow import _period_label
    assert _period_label(bad) is None


def test_an_annual_balance_sheet_is_labelled_as_a_fiscal_year_end():
    """The half that actually broke: the KIND of period is stated in words,
    so a non-calendar fiscal year cannot be misread as an interim period."""
    from finance.report_model import valuation_basis_from
    from finance.workflow import _valuation_basis_lines

    lines = _valuation_basis_lines(valuation_basis_from({
        "dcf_financial_basis": {
            "base_revenue_basis": "ttm_calculation",
            "flow_period_end": "2026-04-30",
            "balance_sheet_as_of": "2026-04-30",
            "balance_sheet_source": "annual_sec_filing",
            "valuation_freshness": "MOSTLY_CURRENT",
        },
        "management_guidance": None,
    }))
    text = "\n".join(lines)
    assert "Balance sheet: 30 Apr 2026 (fiscal year end, latest annual filing)" in text
    assert "trailing twelve months to 30 Apr 2026" in text
    assert "Q2" not in text


# ---------------------------------------------------------------------------
# DIS/CASY corrective patch: say what is known, not more
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("examined,expected", [
    (None, "Management guidance: not retrieved"),
    (0, "Management guidance: no SEC earnings release found"),
    (1, "Management guidance: none extracted (1 SEC earnings release examined)"),
    (3, "Management guidance: none extracted (3 SEC earnings releases examined)"),
])
def test_absent_guidance_states_only_what_is_known(examined, expected):
    """"Unavailable" asserted the COMPANY published no guidance. What is
    actually known is that the EXTRACTOR found none.

    Both live counter-examples had published guidance the report denied:

        DIS   "We continue to expect fiscal 2026 adjusted EPS growth of
               approximately 12%"  -- a single value; this extractor requires
               a range, which is the guard that keeps reported actuals out
        CASY  "inside same-store sales to increase 2% to 5%"  -- a genuine
               range, missed because the metric vocabulary did not cover
               same-store sales

    Neither is a wording quibble: the report fell back to history while
    stating something that reads as a fact about the company.
    """
    from finance.report_model import valuation_basis_from
    from finance.workflow import _valuation_basis_lines

    lines = _valuation_basis_lines(valuation_basis_from({
        "dcf_financial_basis": {"base_revenue_basis": "ttm_calculation",
                                "flow_period_end": "2026-06-30",
                                "balance_sheet_as_of": "2026-06-30",
                                "balance_sheet_source": "quarterly_sec_filing"},
        "management_guidance": None,
        "guidance_releases_examined": examined,
    }))
    text = "\n".join(lines)
    assert expected in text
    assert "Management guidance: unavailable" not in text


def test_none_and_zero_releases_are_not_conflated():
    """None means ingestion never ran; 0 means it ran and found no earnings
    release. Collapsing them erases the distinction the field exists for."""
    from finance.report_model import valuation_basis_from
    from finance.workflow import _valuation_basis_lines

    def line(examined):
        return "\n".join(_valuation_basis_lines(valuation_basis_from({
            "dcf_financial_basis": {"balance_sheet_as_of": "2026-06-30",
                                    "balance_sheet_source": "quarterly_sec_filing"},
            "management_guidance": None,
            "guidance_releases_examined": examined})))

    assert line(None) != line(0)


def test_present_guidance_is_still_stated_plainly(wired):
    """The positive case is unchanged -- this patch only touches absence."""
    result = run_aos(wired)
    compact = build_compact_synthesis_payload(result)
    text = render_compact_report(result, compact, pipeline_result=None)
    assert "Management guidance: FY2026 current guidance" in text
    assert "none extracted" not in text
