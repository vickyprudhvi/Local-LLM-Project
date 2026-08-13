"""TSLA DCF validation patch — AMZN regression, a SECOND real-company
confirmation of the working-capital fix, from REAL live Yahoo Finance + SEC
EDGAR data (tests/fixtures/amzn_regression.json, captured 2026-08-08; secrets
removed; SEC company_facts trimmed to only the finance/xbrl_mapping.py
concepts this codebase reads — see the fixture's own "note" field).

Where the TSLA fixture (tests/test_finance_tsla_regression.py) exercises the
FAIL-CLOSED path (the working-capital fix restores correct scenario ordering,
but base/bear's forecasted FCFF still goes negative under TSLA's thin-margin/
heavy-CapEx assumptions, so `validation_status` correctly reports
DCF_NEGATIVE_TERMINAL_FCFF), this fixture exercises the CLEAN path: AMZN's
raw (current_assets - current_liabilities)/revenue is a modest +1.5%, but the
corrected OPERATING figure is -14.3% -- Amazon's well-documented negative
cash-conversion-cycle business model (collects from customers before paying
suppliers), now correctly reflected instead of masked by the old, buggy
definition -- and the resulting DCF passes full validation cleanly
(DCF_VALID, monotonic bull >= base >= bear, decision_readiness READY). This
proves the fix does not just "make TSLA fail correctly" — it produces a
genuinely valid, usable result whenever the underlying assumptions actually
support one.

Every expected value below was independently produced by RUNNING this
fixture through the real pipeline once and reading off the result (this
project's own standing rule, applied identically to the COST/COR/TSLA
regression fixtures).
"""

import json
from pathlib import Path

import pytest

import tools.finance_tools as finance_tools
from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.dcf import DcfValidationStatus, ScenarioResultStatus
from finance.provider import AlphaVantageClient, ProviderResponse
from finance.quota import AlphaVantageQuotaLedger, QuotaLedger
from finance.sec_datasets import resolve_sec_dataset
from finance.workflow import AnalysisMode, ResearchReadiness, run_full_stock_analysis
from finance.yahoo_datasets import resolve_yahoo_dataset
from tests.test_finance_cache import FakeClock
from tools.base import ToolFailure
from tools.executor import ToolExecutor
from tools.models import MARKET_DATA_INVALID_SYMBOL
from tools.registry import ToolRegistry

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "amzn_regression.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

SYMBOL = "AMZN"

# ---- ground truth, from the real captured SEC filings ----
FY2025_REVENUE_APPROX = 700_000_000_000.0  # sanity floor, not an exact match target

# ---- ground truth, independently produced by RUNNING this fixture through
# the real (fixed) pipeline once and reading off the result ----
#
# Phase H.4 (DCF input freshness) moved these deliberately. The valuation is
# now built on:
#   * base revenue  = trailing twelve months to 2026-06-30 ($775.7B), not
#     FY2025 ($700B)
#   * the equity bridge = the 2026-06-30 quarterly balance sheet, not the
#     2025-12-31 annual one -- AMZN's total debt nearly doubled in between
#     ($68.9B -> $132.5B), which the annual-only bridge could not see
#   * revenue growth = a per-year path fading 15.8% -> 3.0%, not one
#     historical CAGR (11.1%) held flat across all five years
EXPECTED_BASE_VALUE_PER_SHARE = 197.279832
EXPECTED_BULL_VALUE_PER_SHARE = 314.95767
EXPECTED_BEAR_VALUE_PER_SHARE = 99.396438
EXPECTED_WORKING_CAPITAL_PCT_REVENUE = -0.1430


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


def run_amzn(wired):
    return run_full_stock_analysis(wired, SYMBOL, include_news=False)


def _scenario(result, name):
    return next(s for s in result.facts["dcf"]["scenarios"] if s["scenario"] == name)


# ---- 1: fixture sanity ----

def test_fixture_is_real_captured_amzn_data(wired):
    result = run_amzn(wired)
    assert "AMAZON" in result.facts["overview"]["name"].upper()
    revenue = result.facts["statements"]["annual"]["income_statement"][0]["values"]["revenue"]
    assert revenue > FY2025_REVENUE_APPROX


def test_full_analysis_mode_reached(wired):
    result = run_amzn(wired)
    assert result.plan.mode == AnalysisMode.FULL


# ---- 2: the working-capital fix, confirmed on a SECOND real company ----

def test_raw_ratio_is_modestly_positive_but_operating_ratio_is_strongly_negative(wired):
    """The core regression: AMZN's raw balance-sheet ratio looks unremarkable
    (~+1.5%), which is exactly why the old bug was easy to miss on this
    company -- the OPERATING figure reveals Amazon's real, well-known
    negative working-capital position once cash/short-term investments and
    short-term debt are correctly excluded."""
    result = run_amzn(wired)
    balance = result.facts["statements"]["annual"]["balance_sheet"][0]["values"]
    revenue = result.facts["statements"]["annual"]["income_statement"][0]["values"]["revenue"]
    raw_ratio = (balance["current_assets"] - balance["current_liabilities"]) / revenue
    assert -0.05 < raw_ratio < 0.05  # modestly positive, not obviously wrong-looking

    base = _scenario(result, "base")
    operating_ratio = base["assumptions"]["working_capital_pct_revenue"][0]
    assert operating_ratio == pytest.approx(EXPECTED_WORKING_CAPITAL_PCT_REVENUE, abs=1e-3)
    assert operating_ratio < -0.10


# ---- 3: DCF passes full validation cleanly (the confirmatory contrast with
# TSLA's fail-closed path) ----

def test_dcf_scenario_values_match_the_independently_computed_result(wired):
    result = run_amzn(wired)
    base, bull, bear = _scenario(result, "base"), _scenario(result, "bull"), _scenario(result, "bear")
    assert base["value_per_share"] == pytest.approx(EXPECTED_BASE_VALUE_PER_SHARE, rel=1e-6)
    assert bull["value_per_share"] == pytest.approx(EXPECTED_BULL_VALUE_PER_SHARE, rel=1e-6)
    assert bear["value_per_share"] == pytest.approx(EXPECTED_BEAR_VALUE_PER_SHARE, rel=1e-6)


def test_bull_base_bear_ordering_is_monotonic(wired):
    result = run_amzn(wired)
    base, bull, bear = _scenario(result, "base"), _scenario(result, "bull"), _scenario(result, "bear")
    assert bull["value_per_share"] >= base["value_per_share"] >= bear["value_per_share"]


def test_every_scenario_status_is_valid(wired):
    result = run_amzn(wired)
    for name in ("base", "bull", "bear"):
        s = _scenario(result, name)
        assert s["status"] == ScenarioResultStatus.VALID
        assert s["terminal_year_fcff"] > 0
        assert s["warnings"] == []


def test_overall_validation_status_is_valid(wired):
    result = run_amzn(wired)
    dcf = result.facts["dcf"]
    assert dcf["validation_status"] == DcfValidationStatus.VALID
    assert dcf["validation_reasons"] == []
    assert dcf["scenario_monotonicity"]["passed"] is True


# ---- 4: valuation gap is meaningful (positive base value) and decision
# readiness is READY -- the OTHER side of sections 8/11 from the TSLA test ----

def test_valuation_gap_is_available_and_meaningful(wired):
    result = run_amzn(wired)
    gap = result.facts["valuation_gap"]
    assert gap["available"] is True
    assert gap["valuation_gap_status"] == "meaningful"
    assert gap["difference_pct"] is not None


def test_research_readiness_is_limited_by_assumption_quality(wired):
    """MLI corrective patch: this asserted a blanket READY before assumption
    QUALITY was part of readiness. A DCF can pass deterministic validation
    (bull > base > bear, finite, positive terminal FCFF) while resting on
    inputs that are not company-derived -- validation checks the arithmetic,
    not whether the inputs mean anything. On this AMZN fixture the
    bull-to-bear range is 127% of the base value and a company-derived
    assumption fell back to a configured default, so LIMITED is the honest
    label. Asserting the specific reasons is a STRONGER guarantee than the
    old blanket READY."""
    result = run_amzn(wired)
    readiness = result.facts["research_readiness"]
    assert readiness["status"] == ResearchReadiness.LIMITED
    joined = " ".join(readiness["reasons"]).lower()
    assert "scenario sensitivity is high" in joined or "configured default" in joined
    # Still NOT_READY-free: the data and DCF themselves are sound.
    assert readiness["status"] != ResearchReadiness.NOT_READY


# ---- 5: equity bridge / share-count policy, net-cash position ----

def test_net_debt_is_a_real_net_debt_position_on_the_current_balance_sheet(wired):
    """Phase H.4 inverted this, and the inversion is the finding.

    Against the FY2025 annual balance sheet AMZN showed a net CASH position.
    Against the 2026-06-30 quarterly balance sheet it does not: total debt
    went from $68.9B to $132.5B in six months (AI capital spending, funded
    with debt — trailing-twelve-month capex is $173B), against $78.2B of
    cash. Net debt is +$54.3B.

    Both figures are correct for their own date. The point is that the
    valuation was subtracting the December one in August, and nothing said
    so — which is exactly what `DCF_STALE_DEBT_INPUT` now reports.
    """
    result = run_amzn(wired)
    base = _scenario(result, "base")
    assert base["net_debt"] > 0
    findings = (result.facts.get("current_financial_state") or {}).get("findings") or []
    assert any(f["code"] == "DCF_STALE_DEBT_INPUT" for f in findings)


def test_net_debt_is_identical_across_scenarios(wired):
    result = run_amzn(wired)
    base, bull, bear = _scenario(result, "base"), _scenario(result, "bull"), _scenario(result, "bear")
    assert base["net_debt"] == bull["net_debt"] == bear["net_debt"]


def test_dcf_uses_sec_weighted_average_diluted_shares_not_yahoo_basic(wired):
    result = run_amzn(wired)
    assert result.facts.get("dcf_shares_outstanding_source") == "sec_weighted_average_diluted"


# ---- 6: determinism ----

def test_dcf_is_deterministic_across_repeated_runs(wired):
    result_a = run_amzn(wired)
    result_b = run_amzn(wired)
    assert result_a.facts["dcf"]["scenarios"] == result_b.facts["dcf"]["scenarios"]
    assert result_a.facts["dcf"]["validation_status"] == result_b.facts["dcf"]["validation_status"]
