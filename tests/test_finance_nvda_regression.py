"""Phase H.6 — NVIDIA (NVDA) regression, from REAL live SEC EDGAR + Yahoo
Finance data (tests/fixtures/nvda_regression.json, captured 2026-08-17).

THE BUGS THIS FILE EXISTS TO PIN DOWN
=====================================
Running a valuation of NVIDIA in August 2026, the report said

    Financial base: TTM through 2026-04-26
    Revenue                              $215.94B

$215.94B is NVDA's FY2026 ANNUAL revenue, for the year ended 2026-01-25. The
actual trailing twelve months through 2026-04-26 is $253.49B. The label and
the number came from different places — the DCF used the TTM, the Snapshot
table read the annual statements — and nothing checked that they agreed. One
quarter of NVDA's growth is $37B, so the headline understated the base by
15% while telling the reader it did not.

    Management guidance: none extracted

NVDA's May-2026 release guides in detail. Three independent reasons nothing
was found, all of them now fixed:

    * the expected fiscal year was hard-coded to the CALENDAR year (2026)
      and NVDA guides fiscal 2027 — its fiscal year began 2026-01-26;
    * guidance was assumed ANNUAL and this is a next-QUARTER outlook;
    * the value is a point with a tolerance ("$91.0 billion, plus or minus
      2%"), not the two-ended range the extractor required.

    Net debt                              -$1.14B (snapshot)
                                          -$3.77B (valuation state)

Neither reconciles with the balance sheet. NVDA reports `DebtCurrent`
$1,000M and `LongTermDebtCurrent` $1,000M — the SAME obligation, since
us-gaap defines `DebtCurrent` as short-term AND current-portion-of-long-term
debt — and both were added, overstating total debt by $1.0B against NVDA's
own reported `LongTermDebt` of $8,470M. Separately, NVDA renamed its current
securities tag from `MarketableSecuritiesCurrent` to `DebtSecuritiesCurrent`
in fiscal 2027, so $37.1B of liquidity resolved to nothing at all.

    Revenue growth: 25.0%

NVDA's derived growth is far higher; 25.0% is this engine's configured
bound. A clamp is a safety limit, not an estimate, and presenting it as the
near-term forecast is a claim the evidence does not make.

Nothing here asserts a desired RECOMMENDATION for NVIDIA, and nothing forces
the valuation toward any external analysis.
"""

import json
from pathlib import Path

import pytest

import tools.finance_tools as finance_tools
from finance import guidance as G
from finance import net_debt as ND
from finance import ttm as TTM
from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.provider import AlphaVantageClient, ProviderResponse
from finance.quota import AlphaVantageQuotaLedger, QuotaLedger
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

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "nvda_regression.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

SYMBOL = "NVDA"

# ---- ground truth, straight from the captured filings ----
FY2026_PERIOD_END = "2026-01-25"          # fiscal year ended
Q1_FY2027_PERIOD_END = "2026-04-26"       # the one quarter reported since
FY2026_REVENUE = 215_938_000_000.0         # what the report called "TTM"
TTM_REVENUE = 253_491_000_000.0            # what a trailing twelve months is

# The two quarters the roll-forward needs: the newly reported one, and the
# SAME quarter of the prior fiscal year that it replaces.
Q1_FY2026_REVENUE = 44_062_000_000.0       # 2025-01-27..2025-04-27
Q1_FY2027_REVENUE = 81_615_000_000.0       # 2026-01-26..2026-04-26

# Balance sheet at 2026-04-26.
CASH = 13_237_000_000.0
MARKETABLE_SECURITIES = 37_098_000_000.0   # DebtSecuritiesCurrent
EQUITY_SECURITIES = 30_237_000_000.0       # EquitySecuritiesFvNi
CURRENT_DEBT = 1_000_000_000.0             # DebtCurrent == LongTermDebtCurrent
LONG_TERM_DEBT = 7_470_000_000.0           # LongTermDebtNoncurrent
TOTAL_DEBT = 8_470_000_000.0               # NVDA's own reported LongTermDebt
CURRENT_ASSETS = 150_995_000_000.0

# Q2 FY2027 guidance, as NVDA stated it.
GUIDED_REVENUE_MIDPOINT = 91.0             # billions, "plus or minus 2%"
GUIDED_REVENUE_RANGE = (89.18, 92.82)
GUIDED_NON_GAAP_GROSS_MARGIN = 0.75


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


def run_nvda(wired):
    executor, _sec = wired
    return run_full_stock_analysis(executor, SYMBOL)


def _state(result):
    return result.facts.get("current_financial_state") or {}


def _guidance(result):
    return (result.facts.get("management_guidance") or {}).get("metrics") or {}


def _base(result):
    return next(s for s in result.facts["dcf"]["scenarios"] if s["scenario"] == "base")


def _company_facts():
    return FIXTURE["sec"]["company_facts"]


# ---------------------------------------------------------------------------
# Section 27, checks 1-3: TTM construction
# ---------------------------------------------------------------------------

def test_1_the_fiscal_year_value_is_not_mislabelled_ttm(wired):
    """The headline failure: $215.94B (FY2026) presented under a TTM label."""
    revenue = _state(run_nvda(wired))["flows"]["revenue"]
    assert revenue["source"] == "ttm_calculation"
    assert revenue["value"] != pytest.approx(FY2026_REVENUE)
    assert revenue["value"] == pytest.approx(TTM_REVENUE)


def test_2_one_quarter_past_fiscal_year_end_rolls_the_window_forward(wired):
    """NVDA has reported exactly ONE quarter since FY2026 ended, which is the
    case section 1 specifies: FY + current quarter - prior-year quarter."""
    revenue = _state(run_nvda(wired))["flows"]["revenue"]
    ttm = revenue["ttm"]
    assert ttm["validation_status"] == "valid"
    assert ttm["construction_method"] in (
        TTM.TtmConstruction.FOUR_DISCRETE_QUARTERS,
        TTM.TtmConstruction.ANNUAL_ROLL_FORWARD)
    assert len(ttm["quarters_included"]) >= 3
    assert ttm["source_accessions"]
    # The arithmetic identity section 1 states, whichever route was taken.
    assert revenue["value"] == pytest.approx(
        FY2026_REVENUE + Q1_FY2027_REVENUE - Q1_FY2026_REVENUE, rel=1e-6)


def test_2b_the_annual_roll_forward_produces_the_same_answer_directly():
    """The construction section 1 names, tested on its own."""
    facts = _company_facts()
    result = TTM._from_annual_roll_forward(facts, "revenue")[0]  # noqa: SLF001
    assert result is not None
    assert result.value == pytest.approx(TTM_REVENUE)
    assert result.end_date == Q1_FY2027_PERIOD_END
    assert result.construction_method == TTM.TtmConstruction.ANNUAL_ROLL_FORWARD


def test_3_the_ttm_end_date_is_the_latest_reported_quarter(wired):
    revenue = _state(run_nvda(wired))["flows"]["revenue"]
    assert revenue["as_of_date"] == Q1_FY2027_PERIOD_END
    assert revenue["ttm"]["end_date"] == Q1_FY2027_PERIOD_END
    assert 350 <= revenue["ttm"]["span_days"] <= 380


def test_3b_a_balance_sheet_field_is_never_given_a_ttm(wired):
    """Section 4: point-in-time values stay point-in-time."""
    for field_name in ("cash_and_cash_equivalents", "long_term_debt",
                       "stockholders_equity"):
        result = TTM.build_ttm(_company_facts(), field_name)
        assert result.validation_status == TTM.TtmValidation.INVALID
        assert "point-in-time" in (result.reason or "")


def test_3c_the_snapshot_and_the_valuation_quote_the_same_revenue(wired):
    """The two numbers were on the same page and disagreed by $37B."""
    result = run_nvda(wired)
    report = render_compact_report(result, build_compact_synthesis_payload(result),
                                   pipeline_result=None)
    assert "Revenue (TTM)" in report
    assert "$215.94B" not in report
    assert "$253.49B" in report


# ---------------------------------------------------------------------------
# Section 27, checks 4-6: guidance extraction
# ---------------------------------------------------------------------------

def test_4_current_quarterly_revenue_guidance_is_extracted(wired):
    """"Revenue is expected to be $91.0 billion, plus or minus 2%" — a point
    with a tolerance, for the SECOND QUARTER of fiscal 2027."""
    metrics = _guidance(run_nvda(wired))
    revenue = metrics.get("revenue")
    assert revenue is not None, "NVDA's next-quarter revenue guidance was not extracted"
    assert (revenue["low"], revenue["high"]) == pytest.approx(GUIDED_REVENUE_RANGE)
    assert revenue["midpoint"] == pytest.approx(GUIDED_REVENUE_MIDPOINT)
    assert revenue["scale"] == "billion"


def test_4b_the_guided_period_is_a_quarter_of_a_non_calendar_fiscal_year(wired):
    """NVDA's fiscal 2027 began 2026-01-26. A calendar-year filter is exactly
    what discarded every value in this release."""
    revenue = _guidance(run_nvda(wired))["revenue"]
    assert revenue["fiscal_period"] == "Q2 FY2027"
    assert revenue["period_type"] == G.GuidancePeriodType.QUARTER
    assert revenue["fiscal_year"] == 2027


def test_5_gross_margin_guidance_is_not_confused_with_revenue_guidance(wired):
    """"GAAP and non-GAAP gross margins are expected to be 74.9% and 75.0%"
    is TWO point values for two bases — never a 74.9%-75.0% range, and never
    a revenue figure."""
    metrics = _guidance(run_nvda(wired))
    margin = metrics.get("adjusted_gross_margin")
    assert margin is not None
    assert margin["midpoint"] == pytest.approx(GUIDED_NON_GAAP_GROSS_MARGIN, abs=0.005)
    assert margin["units"] == G.GuidanceUnit.RATIO
    assert margin["basis"] == G.BASIS_ADJUSTED
    # The revenue guidance is a currency amount; the margin is a ratio. If the
    # two were confused, one of these would be in the other's units.
    assert metrics["revenue"]["units"] == G.GuidanceUnit.CURRENCY
    assert metrics["revenue"]["midpoint"] > 1.0


def test_6_guidance_source_evidence_is_preserved(wired):
    for name, entry in _guidance(run_nvda(wired)).items():
        assert entry["source_accession"], name
        assert entry["source_evidence_ids"], name
        assert entry["guidance_id"], name
        assert entry["issued_at"], name
        assert entry["source_excerpt"], name
        assert entry["status"] == G.GuidanceStatus.CURRENT, name


def test_6b_older_releases_are_superseded_not_used(wired):
    """Four releases are in the fixture, each guiding a different quarter.
    Only the newest statement of each metric may be current."""
    result = run_nvda(wired)
    current = _guidance(result)
    superseded = result.facts.get("superseded_guidance") or []
    assert superseded, "older releases were not retained"
    assert current["revenue"]["fiscal_period"] == "Q2 FY2027"
    for release in superseded:
        assert release["filed"] <= (result.facts["management_guidance"] or {})["filed"]


# ---------------------------------------------------------------------------
# Section 27, checks 7-11: net debt
# ---------------------------------------------------------------------------

def test_7_cash_investments_and_equity_securities_stay_distinct(wired):
    balance = _state(run_nvda(wired))["balance_sheet"]
    assert balance["cash_and_cash_equivalents"]["value"] == pytest.approx(CASH)
    assert balance["short_term_investments"]["value"] == pytest.approx(
        MARKETABLE_SECURITIES)
    assert balance["equity_securities_at_fair_value"]["value"] == pytest.approx(
        EQUITY_SECURITIES)
    # The tag NVDA switched TO, not the one it abandoned.
    assert balance["short_term_investments"]["concept"] == "DebtSecuritiesCurrent"


def test_7b_current_assets_reconcile_against_the_selected_components(wired):
    """The check that would have CAUGHT the missing $37.1B: a company with
    $151.0B of current assets, of which the bridge can account for $13.2B,
    is not a reconciled balance sheet."""
    balance = _state(run_nvda(wired))["balance_sheet"]
    accounted = (balance["cash_and_cash_equivalents"]["value"]
                 + balance["short_term_investments"]["value"]
                 + balance["equity_securities_at_fair_value"]["value"])
    assert balance["current_assets"]["value"] == pytest.approx(CURRENT_ASSETS)
    assert accounted / CURRENT_ASSETS > 0.5


def test_8_debt_components_stay_distinct(wired):
    balance = _state(run_nvda(wired))["balance_sheet"]
    assert balance["short_term_debt"]["value"] == pytest.approx(CURRENT_DEBT)
    assert balance["long_term_debt"]["value"] == pytest.approx(LONG_TERM_DEBT)
    assert balance["short_term_debt"]["concept"] == "DebtCurrent"
    assert balance["long_term_debt"]["concept"] == "LongTermDebtNoncurrent"


def test_11_overlapping_debt_concepts_are_not_double_counted(wired):
    """`DebtCurrent` and `LongTermDebtCurrent` are both $1.0B because they are
    the same obligation. Summing all three components gave $9.47B against
    NVDA's own reported $8.47B."""
    result = run_nvda(wired)
    state = _state(result)
    assert state["total_debt"]["value"] == pytest.approx(TOTAL_DEBT)
    assert state["total_debt"]["value"] != pytest.approx(TOTAL_DEBT + CURRENT_DEBT)

    detail = state["net_debt_detail"]["components"]
    assert detail["reported_total_debt"] == pytest.approx(TOTAL_DEBT)
    excluded = {e["field"] for e in detail["excluded"]}
    assert "current_portion_of_long_term_debt" in excluded
    codes = {f["code"] for f in detail["findings"]}
    assert ND.DCF_NET_DEBT_COMPONENT_OVERLAP in codes
    # The exclusion must be EXPLAINED, not silent.
    assert any("count the same obligation twice" in e["reason"]
               for e in detail["excluded"])


def test_9_cash_only_net_debt_reconciles(wired):
    result = run_nvda(wired)
    detail = _state(result)["net_debt_detail"]
    assert detail["net_debt_policy"] == ND.NetDebtPolicyName.CASH_ONLY
    assert detail["net_debt"] == pytest.approx(TOTAL_DEBT - CASH)
    assert detail["reconciled"] is True
    assert _base(result)["net_debt"] == pytest.approx(TOTAL_DEBT - CASH)


def test_10_cash_plus_securities_policy_reconciles():
    """The other supported policy, applied to the same components."""
    from finance.freshness import DcfFreshnessPlanner

    planner = DcfFreshnessPlanner(_company_facts(), SYMBOL)
    balance, _warnings = planner.select_balance_sheet()
    _total, components = planner.derive_total_debt(balance)
    result = ND.compute_net_debt(
        components, ND.NetDebtPolicyName.CASH_AND_MARKETABLE_SECURITIES,
        marketable_securities_eligible=True)
    assert result.value == pytest.approx(TOTAL_DEBT - CASH - MARKETABLE_SECURITIES)
    assert result.eligible_marketable_securities == pytest.approx(MARKETABLE_SECURITIES)
    # And the two policies must produce DIFFERENT numbers, or the switch is
    # not doing anything.
    cash_only = ND.compute_net_debt(components, ND.NetDebtPolicyName.CASH_ONLY)
    assert cash_only.value != pytest.approx(result.value)


def test_10b_equity_securities_are_never_netted_by_either_policy():
    """$30.2B of equity securities at fair value are reported but are not
    cash under any policy this project supports."""
    from finance.freshness import DcfFreshnessPlanner

    planner = DcfFreshnessPlanner(_company_facts(), SYMBOL)
    balance, _warnings = planner.select_balance_sheet()
    _total, components = planner.derive_total_debt(balance)
    for policy in ND.NetDebtPolicyName.ALL:
        result = ND.compute_net_debt(components, policy)
        assert result.value > TOTAL_DEBT - CASH - MARKETABLE_SECURITIES - 1.0
        assert components.equity_securities_at_fair_value == pytest.approx(
            EQUITY_SECURITIES)


def test_the_net_debt_policy_is_recorded_on_the_valuation(wired):
    """Section 14: the policy must never switch silently."""
    basis = run_nvda(wired).facts["dcf_financial_basis"]
    assert basis["net_debt_policy"] in ND.NetDebtPolicyName.ALL
    reconciliation = basis["net_debt_reconciliation"]
    assert reconciliation["reconciled"] is True
    assert reconciliation["evidence_ids"]


# ---------------------------------------------------------------------------
# Section 27, checks 12-15: assumptions and what may reach the recommendation
# ---------------------------------------------------------------------------

def test_12_the_growth_clamp_is_exposed_not_treated_as_the_forecast(wired):
    """A clamp is a safety bound. NVDA's derived growth is far above it, and
    the applied 25.0% must be labelled as this engine's limit."""
    result = run_nvda(wired)
    provenance = ((_base(result).get("assumptions") or {})
                  .get("assumption_provenance") or {}).get("revenue_growth") or {}
    provenance = provenance or (result.facts["dcf"].get("shared_assumption_provenance")
                                or {}).get("revenue_growth") or {}
    assert provenance["clamped"] is True
    # `applied_value` is the per-year PATH; the clamp is about its year-1
    # anchor, which is what `raw_value` was derived for.
    applied_year_one = provenance["applied_value"]
    if isinstance(applied_year_one, list):
        applied_year_one = applied_year_one[0]
    assert provenance["raw_value"] > applied_year_one
    assert "not an estimate" in (provenance["clamp_reason"] or "").lower()

    report = render_compact_report(result, build_compact_synthesis_payload(result),
                                   pipeline_result=None)
    assert "CLAMPED" in report
    assert "not an estimate of the company's growth" in report


def test_13_the_forward_assumption_builder_receives_the_current_guidance(wired):
    """Section 8: quarterly guidance is near-term evidence and must reach the
    builder, labelled as one quarter rather than as a five-year forecast."""
    from finance import forward_assumptions as FA

    result = run_nvda(wired)
    state = result.facts["_current_financial_state"]
    evidence = FA.collect_growth_evidence(
        state, result.facts.get("_sec_company_facts"),
        comparability=state.historical_comparability)
    assert evidence.guidance_implied_next_period_growth is not None
    assert evidence.guidance_implied_comparison_period

    paths, _ = FA.build_forward_assumptions(
        state, 5, company_facts=result.facts.get("_sec_company_facts"),
        comparability=state.historical_comparability)
    block = FA.build_evidence_block(state, evidence, paths)
    assert "ONE QUARTER" in block
    assert "not a five-year forecast" in block.lower() or "NOT a five-year forecast" in block


def test_13b_quarterly_guidance_does_not_become_the_five_year_anchor(wired):
    """A +90% next-quarter number applied flat to five years would be an
    extraordinary claim. The path must fade rather than hold."""
    result = run_nvda(wired)
    growth = (_base(result).get("assumptions") or {}).get("revenue_growth")
    assert isinstance(growth, list) and len(growth) == 5
    assert growth[-1] < growth[0], "the growth path is flat across the horizon"


def test_14_the_dcf_uses_the_corrected_ttm_base(wired):
    result = run_nvda(wired)
    basis = result.facts["dcf_financial_basis"]
    assert basis["base_revenue_basis"] == "ttm_calculation"
    assert basis["flow_period_end"] == Q1_FY2027_PERIOD_END
    assert basis["flow_base_validation"] == "valid"
    assert _base(result)["base_revenue"] == pytest.approx(TTM_REVENUE) \
        if _base(result).get("base_revenue") is not None else True
    # The DCF's own forecast must start from the TTM, not the fiscal year.
    forecast = _base(result)["forecast"]
    first_year_revenue = forecast[0]["revenue"]
    assert first_year_revenue > FY2026_REVENUE


def test_15_a_mislabelled_ttm_would_block_valuation_conclusions():
    """Section 25: an invalid TTM base is NOT_READY, because everything built
    on it describes a period that does not exist."""
    from finance.workflow import _freshness_readiness_signals

    facts = {"current_financial_state": {
        "flows": {"revenue": {
            "source": "ttm_calculation",
            "ttm": {"validation_status": "invalid",
                    "reason": "TTM_INVALID_PERIOD_RECONSTRUCTION: only 2 quarters"},
        }},
    }}
    blocking, _limiting = _freshness_readiness_signals(facts)
    assert blocking and "withheld" in blocking[0]


def test_15b_nvidia_history_is_comparable_so_growth_is_not_discounted(wired):
    """NVDA's revenue rose 114% and then 65% selling more of the same product.
    That is this company's own reported growth, not a break in comparability
    — flagging it as one would discard the only real information there is."""
    comparability = _state(run_nvda(wired))["historical_comparability"]
    assert comparability["historical_comparability_status"] == \
        HistoricalComparability.COMPARABLE


def test_no_recommendation_is_produced_deterministically(wired):
    result = run_nvda(wired)
    assert "recommendation" not in result.facts
    report = render_compact_report(result, build_compact_synthesis_payload(result),
                                   pipeline_result=None)
    assert "Recommendation: unavailable" in report
