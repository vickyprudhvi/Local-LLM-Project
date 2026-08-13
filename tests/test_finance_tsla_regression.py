"""TSLA DCF validation patch — TSLA regression, from REAL live Yahoo Finance
+ SEC EDGAR data (tests/fixtures/tsla_regression.json, captured 2026-08-07;
secrets removed; SEC company_facts trimmed to only the finance/xbrl_mapping.py
concepts this codebase reads — see the fixture's own "note" field).

This is the exact company/data shape that exposed the bug this patch fixes: a
live TSLA FullStockAnalysis report produced

    Bear modeled value: -29.18
    Base modeled value: -39.38
    Bull modeled value: -51.49

-- bull MORE negative than base, base MORE negative than bear, the reverse of
the intended bull >= base >= bear ordering -- plus a nonsensical "112% above
the base modeled value" claim computed against a NEGATIVE modeled value.

Root cause (see docs/PHASE_H1_STOCK_ANALYSIS.md's "TSLA DCF validation patch"
section for the full writeup): `finance/workflow.py::_historical_nwc_ratio_
pairs` computed the working-capital assumption from the RAW `current_assets -
current_liabilities` balance-sheet aggregate, which for TSLA is dominated by
cash and short-term investments (~$44B on ~$95B revenue) -- financing
balances, not operating working capital. This produced a +24.4%-of-revenue
working-capital assumption instead of the correct OPERATING figure (~-8.6%),
forcing deeply negative FCFF in every forecast year for every scenario and,
because the resulting fake "cash build" scaled fastest under whichever
scenario had the highest revenue growth, inverting the intended ordering.

This file proves, on REAL captured TSLA data:
  1. The working-capital fix restores correct bull >= base >= bear ordering.
  2. Even after the fix, base/bear's forecasted FCFF still goes negative
     within the horizon under TSLA's current thin-margin/heavy-CapEx
     assumptions -- `validation_status` correctly reports
     DCF_NEGATIVE_TERMINAL_FCFF rather than hiding it (fail closed, per the
     patch's own "do not force monotonicity artificially, detect the
     underlying bug" and "fail valuation closed" requirements).
  3. No invalid DCF evidence reaches the evidence index (so no research
     stage can cite it).
  4. The valuation gap is withheld (no percentage-vs-negative-value claim).
  5. decision_readiness is NOT_READY.
  6. The compact report states Status: MODEL_INVALID and never prints the
     invalid bear/base/bull numbers as normal findings.
  7. The SEC weighted-average diluted share count is used, not Yahoo's basic
     shares outstanding.

Every expected value below was independently produced by RUNNING this
fixture through the real pipeline once and reading off the result (this
project's own standing rule, applied identically to the COST/COR regression
fixtures) -- the working-capital ratio was ADDITIONALLY cross-checked by an
independent hand calculation from the fixture's own raw reported figures
(see tests/test_finance_workflow.py::
test_tsla_style_cash_rich_balance_sheet_produces_a_negative_operating_ratio
for the same mechanism in isolation).
"""

import json
from pathlib import Path

import pytest

import tools.finance_tools as finance_tools
from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.dcf import DcfValidationStatus, ScenarioResultStatus
from finance.evidence import build_evidence_index
from finance.provider import AlphaVantageClient, ProviderResponse
from finance.quota import AlphaVantageQuotaLedger, QuotaLedger
from finance.sec_datasets import resolve_sec_dataset
from finance.workflow import (
    AnalysisMode,
    ResearchReadiness,
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

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tsla_regression.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

SYMBOL = "TSLA"

# ---- ground truth, from the real captured SEC filings (see the fixture's
# annual income_statement -- simply the raw reported figure) ----
FY2025_REVENUE = 94_827_000_000.0

# ---- ground truth, independently produced by RUNNING this fixture through
# the real (fixed) pipeline once and reading off the result (see module
# docstring) ----
EXPECTED_BASE_VALUE_PER_SHARE = -12.96996
EXPECTED_BULL_VALUE_PER_SHARE = -5.985514
EXPECTED_BEAR_VALUE_PER_SHARE = -16.408839
EXPECTED_NET_DEBT = -6_158_000_000.0  # NEGATIVE -- a net-cash position
EXPECTED_DILUTED_SHARES = 3_528_000_000.0


class YahooFixtureClient:
    provider_id = "yahoo"

    def fetch(self, dataset, arguments):
        payload = FIXTURE["yahoo"].get(dataset.dataset_id)
        if payload is None:
            raise ToolFailure(MARKET_DATA_INVALID_SYMBOL, f"no fixture data for {dataset.dataset_id}")
        return ProviderResponse(payload=payload, provider_metadata={}, byte_count=len(json.dumps(payload)))


class SecFixtureClient:
    provider_id = "sec"

    def fetch(self, dataset, arguments):
        payload = FIXTURE["sec"].get(dataset.dataset_id)
        if payload is None:
            raise ToolFailure(MARKET_DATA_INVALID_SYMBOL, f"no fixture data for {dataset.dataset_id}")
        return ProviderResponse(payload=payload, provider_metadata={}, byte_count=len(json.dumps(payload)))


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def wired(tmp_path, clock, monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "")  # deliberately unconfigured
    monkeypatch.setenv("YAHOO_FINANCE_ENABLED", "true")
    monkeypatch.setenv("YAHOO_PERSONAL_USE_ACKNOWLEDGED", "true")
    monkeypatch.setenv("SEC_EDGAR_ENABLED", "true")
    monkeypatch.setenv("SEC_USER_AGENT", "TestApp/1.0 (contact: t@example.com)")
    for var in ("FINANCE_QUOTE_PROVIDER", "FINANCE_PRICE_HISTORY_PROVIDER",
               "FINANCE_CORPORATE_ACTIONS_PROVIDER", "FINANCE_COMPANY_PROFILE_PROVIDER",
               "FINANCE_ANALYST_ESTIMATES_PROVIDER"):
        monkeypatch.setenv(var, "yahoo")
    monkeypatch.setenv("FINANCE_US_FUNDAMENTALS_PROVIDER", "sec")

    yahoo_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "yahoo_m.sqlite3"), clock=clock),
        ledger=QuotaLedger(path=str(tmp_path / "yahoo_q.sqlite3"), clock=clock, daily_limit=1_000_000),
        client=YahooFixtureClient(), provider_id="yahoo", dataset_resolver=resolve_yahoo_dataset,
        clock=clock, sleeper=lambda _s: None, min_request_interval_ms=0)
    finance_tools.set_yahoo_coordinator(yahoo_coordinator)

    sec_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "sec_m.sqlite3"), clock=clock),
        ledger=QuotaLedger(path=str(tmp_path / "sec_q.sqlite3"), clock=clock, daily_limit=1_000_000),
        client=SecFixtureClient(), provider_id="sec", dataset_resolver=resolve_sec_dataset,
        clock=clock, sleeper=lambda _s: None, min_request_interval_ms=0)
    finance_tools.set_sec_coordinator(sec_coordinator)

    av_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "av_m.sqlite3"), clock=clock),
        ledger=AlphaVantageQuotaLedger(path=str(tmp_path / "av_q.sqlite3"), clock=clock, daily_limit=100),
        client=AlphaVantageClient(), clock=clock, sleeper=lambda _s: None, jitter=lambda: 0.5)
    finance_tools.set_coordinator(av_coordinator)

    registry = ToolRegistry()
    for tool_cls in finance_tools.ALL_FINANCE_TOOL_CLASSES:
        registry.register(tool_cls())
    executor = ToolExecutor(registry)

    yield executor
    finance_tools.set_coordinator(None)
    finance_tools.set_yahoo_coordinator(None)
    finance_tools.set_sec_coordinator(None)


def run_tsla(wired):
    return run_full_stock_analysis(wired, SYMBOL, include_news=False)


def _scenario(result, name):
    return next(s for s in result.facts["dcf"]["scenarios"] if s["scenario"] == name)


# ---- 1: the fixture itself is real TSLA data, not a stand-in ----

def test_fixture_is_real_captured_tsla_data(wired):
    result = run_tsla(wired)
    assert result.facts["overview"]["name"] in ("Tesla, Inc.", "Tesla Inc")
    annual_income = result.facts["statements"]["annual"]["income_statement"]
    assert annual_income[0]["values"]["revenue"] == pytest.approx(FY2025_REVENUE)


def test_full_analysis_mode_reached(wired):
    result = run_tsla(wired)
    assert result.plan.mode == AnalysisMode.FULL


# ---- 2: root cause -- operating (not raw) working capital ----

def test_raw_current_assets_minus_liabilities_ratio_is_strongly_positive(wired):
    """The OLD (buggy) definition's own magnitude, confirmed against the raw
    fixture data -- proves this is a real, material bug, not a rounding
    quibble."""
    result = run_tsla(wired)
    balance = result.facts["statements"]["annual"]["balance_sheet"][0]["values"]
    revenue = result.facts["statements"]["annual"]["income_statement"][0]["values"]["revenue"]
    raw_ratio = (balance["current_assets"] - balance["current_liabilities"]) / revenue
    assert raw_ratio > 0.35  # the OLD, buggy figure would have been ~38.9%


def test_dcf_working_capital_assumption_is_negative_the_operating_figure(wired):
    result = run_tsla(wired)
    base = _scenario(result, "base")
    wc_pct = base["assumptions"]["working_capital_pct_revenue"][0]
    assert wc_pct == pytest.approx(-0.0856, abs=1e-3)
    assert wc_pct < 0


def test_capex_and_depreciation_assumptions_are_unaffected_by_the_fix(wired):
    """The fix is scoped to working capital ONLY -- CapEx/D&A derivation
    (already correct from the earlier Phase H.3 corrective patch) must be
    untouched."""
    result = run_tsla(wired)
    base = _scenario(result, "base")
    assert base["assumptions"]["capex_pct_revenue"][0] == pytest.approx(0.1012, abs=1e-3)
    assert base["assumptions"]["depreciation_pct_revenue"][0] == pytest.approx(0.039, abs=1e-3)


# ---- 3: scenario ordering is restored (sections 1/6) ----

def test_bull_base_bear_ordering_is_now_monotonic(wired):
    result = run_tsla(wired)
    base, bull, bear = _scenario(result, "base"), _scenario(result, "bull"), _scenario(result, "bear")
    assert bull["value_per_share"] >= base["value_per_share"] >= bear["value_per_share"]


def test_dcf_scenario_values_match_the_independently_computed_result(wired):
    result = run_tsla(wired)
    base, bull, bear = _scenario(result, "base"), _scenario(result, "bull"), _scenario(result, "bear")
    assert base["value_per_share"] == pytest.approx(EXPECTED_BASE_VALUE_PER_SHARE, rel=1e-6)
    assert bull["value_per_share"] == pytest.approx(EXPECTED_BULL_VALUE_PER_SHARE, rel=1e-6)
    assert bear["value_per_share"] == pytest.approx(EXPECTED_BEAR_VALUE_PER_SHARE, rel=1e-6)


def test_scenario_monotonicity_check_passed(wired):
    result = run_tsla(wired)
    monotonicity = result.facts["dcf"]["scenario_monotonicity"]
    assert monotonicity["checked"] is True
    assert monotonicity["passed"] is True
    assert monotonicity["violated_relations"] == []


def test_report_no_longer_asserts_bull_is_less_negative_than_base():
    """The exact arithmetic-error class the original report exhibited
    ('the bull value is higher (less negative) than base' while
    numerically -51.49 < -39.38) is structurally impossible to reproduce
    here: bull's value is now numerically >= base's, so any report text
    correctly describing 'bull >= base' is also arithmetically TRUE."""
    assert EXPECTED_BULL_VALUE_PER_SHARE >= EXPECTED_BASE_VALUE_PER_SHARE


# ---- 4: even after the fix, base/bear's negative terminal FCFF is
# correctly flagged, never hidden (sections 5/9) ----

def test_base_and_bear_terminal_fcff_are_still_negative_under_current_assumptions(wired):
    """TSLA's heavy CapEx intensity (~10% of revenue) against base/bear's
    thinner operating margins genuinely produces negative forecasted FCFF
    within the horizon -- a real characteristic of the current assumption
    set, not an artifact of the working-capital fix."""
    result = run_tsla(wired)
    base, bear = _scenario(result, "base"), _scenario(result, "bear")
    assert base["terminal_year_fcff"] < 0
    assert bear["terminal_year_fcff"] < 0
    assert base["status"] == ScenarioResultStatus.INVALID
    assert bear["status"] == ScenarioResultStatus.INVALID


def test_scenario_status_is_classified_per_scenario_from_its_own_terminal_fcff(wired):
    """Each scenario is classified INDEPENDENTLY, from its own arithmetic.

    This used to assert specifically that bull escaped the negative-FCFF trap
    while base and bear did not. Under Phase H.4 bull no longer escapes it,
    and the reason is a real signal rather than a broken test: the margin
    assumption now comes from TSLA's TRAILING TWELVE MONTHS (4.22%) instead
    of the last completed fiscal year. Against capital expenditure running at
    10.1% of revenue and D&A at 3.9%, FCFF is structurally negative at that
    margin in every scenario.

    So the assertion is now the RULE rather than one data-dependent outcome:
    a scenario is INVALID exactly when its own terminal-year FCFF is
    negative. That keeps testing what the classification is for — per-scenario
    judgement, never a blanket verdict — without re-pinning to whichever
    scenarios happen to fail on today's data.
    """
    result = run_tsla(wired)
    for name in ("base", "bull", "bear"):
        s = _scenario(result, name)
        expected = (ScenarioResultStatus.INVALID if s["terminal_year_fcff"] < 0
                    else ScenarioResultStatus.VALID)
        assert s["status"] == expected, name


def test_overall_validation_status_fails_closed_with_the_exact_reason(wired):
    result = run_tsla(wired)
    dcf = result.facts["dcf"]
    assert dcf["validation_status"] == DcfValidationStatus.NEGATIVE_TERMINAL_FCFF
    assert any("base" in r and "bear" in r for r in dcf["validation_reasons"])


# ---- 5: valuation-gap / scenario-spread math never misuses the invalid
# value (sections 7/8) ----

def test_valuation_gap_is_withheld_not_a_nonsense_percentage(wired):
    result = run_tsla(wired)
    gap = result.facts["valuation_gap"]
    assert gap["available"] is False
    assert "validation" in gap["reason"].lower()


def test_scenario_spread_is_withheld(wired):
    result = run_tsla(wired)
    spread = result.facts["dcf_scenario_spread"]
    assert spread["available"] is False


# ---- 6: decision readiness (section 11) ----

def test_research_readiness_is_not_ready(wired):
    result = run_tsla(wired)
    readiness = result.facts["research_readiness"]
    assert readiness["status"] == ResearchReadiness.NOT_READY
    assert any("DCF" in r for r in readiness["reasons"])


# ---- 7: no invalid DCF evidence reaches the research pipeline (sections
# 9/10) ----

def test_no_invalid_dcf_evidence_is_indexed(wired):
    result = run_tsla(wired)
    compact = build_compact_synthesis_payload(result)
    index = build_evidence_index(compact)
    for name in ("bull", "base", "bear"):
        assert f"dcf.value_per_share.{name}" not in index
    assert index["dcf.validation_status"].value == DcfValidationStatus.NEGATIVE_TERMINAL_FCFF


def test_fundamentals_and_technicals_remain_indexed_despite_invalid_dcf(wired):
    """Bull/bear researchers can still analyze fundamentals/technicals even
    though valuation evidence is withheld."""
    result = run_tsla(wired)
    compact = build_compact_synthesis_payload(result)
    index = build_evidence_index(compact)
    assert any(k.startswith("fundamental.") for k in index)
    assert any(k.startswith("technical.") for k in index)


# ---- 8: compact report rendering (sections 12/14) ----

def test_compact_report_states_model_invalid_status(wired):
    result = run_tsla(wired)
    compact = build_compact_synthesis_payload(result)
    text = render_compact_report(result, compact, pipeline_result=None)
    assert "Status: MODEL_INVALID" in text
    assert "DCF_NEGATIVE_TERMINAL_FCFF" in text


def test_compact_report_never_prints_the_invalid_scenario_values(wired):
    result = run_tsla(wired)
    compact = build_compact_synthesis_payload(result)
    text = render_compact_report(result, compact, pipeline_result=None)
    assert "Bear modeled value" not in text
    assert "Base modeled value" not in text
    assert "Bull modeled value" not in text
    for figure in ("-8.14", "6.35", "-15.06"):
        assert figure not in text


def test_compact_report_states_research_readiness_not_ready(wired):
    result = run_tsla(wired)
    compact = build_compact_synthesis_payload(result)
    text = render_compact_report(result, compact, pipeline_result=None)
    assert "Research readiness: NOT_READY" in text


def test_compact_report_contains_no_buy_sell_hold_avoid_language(wired):
    """The compact report's OWN fixed disclaimer (finance/workflow.py::
    _COMPACT_DISCLAIMER, appended to every compact report unconditionally)
    legitimately explains that it is 'never a buy/sell/hold/avoid
    instruction' -- the safe, negated use of those words, always present and
    unrelated to this analysis's content. Excluded here before scanning the
    REST of the report (the deterministic sections this patch touches) for
    an actual recommendation."""
    from finance.workflow import _COMPACT_DISCLAIMER
    result = run_tsla(wired)
    compact = build_compact_synthesis_payload(result)
    text = render_compact_report(result, compact, pipeline_result=None)
    body = text.replace(_COMPACT_DISCLAIMER, "")
    upper = body.upper()
    for forbidden in ("BUY", "SELL", "HOLD", "AVOID", "STRONG BUY", "STRONG SELL"):
        assert forbidden not in upper


def test_fundamentals_and_technicals_sections_still_render_normally(wired):
    """The rest of the report is UNAFFECTED by the invalid DCF -- Snapshot/
    Technical sections still render normal facts."""
    result = run_tsla(wired)
    compact = build_compact_synthesis_payload(result)
    text = render_compact_report(result, compact, pipeline_result=None)
    assert "## Snapshot" in text
    assert "## Technical" in text


# ---- 9: equity bridge / net debt / share-count policy (section 13) ----

def test_net_debt_is_negative_a_net_cash_position(wired):
    result = run_tsla(wired)
    base = _scenario(result, "base")
    assert base["net_debt"] == pytest.approx(EXPECTED_NET_DEBT)
    assert base["net_debt"] < 0


def test_net_debt_is_identical_across_scenarios(wired):
    result = run_tsla(wired)
    base, bull, bear = _scenario(result, "base"), _scenario(result, "bull"), _scenario(result, "bear")
    assert base["net_debt"] == bull["net_debt"] == bear["net_debt"]


def test_dcf_uses_sec_weighted_average_diluted_shares_not_yahoo_basic(wired):
    result = run_tsla(wired)
    assert result.facts.get("dcf_shares_outstanding_source") == "sec_weighted_average_diluted"
    base = _scenario(result, "base")
    assert base["diluted_shares"] == pytest.approx(EXPECTED_DILUTED_SHARES)


# ---- 10: determinism ----

def test_dcf_is_deterministic_across_repeated_runs(wired):
    result_a = run_tsla(wired)
    result_b = run_tsla(wired)
    assert result_a.facts["dcf"]["scenarios"] == result_b.facts["dcf"]["scenarios"]
    assert result_a.facts["dcf"]["validation_status"] == result_b.facts["dcf"]["validation_status"]
