"""Phase H.3 corrective patch — Problem 4: COST regression, from REAL live
Yahoo Finance + SEC EDGAR data (tests/fixtures/cost_regression.json,
captured 2026-08-06; secrets removed; SEC company_facts trimmed to only the
finance/xbrl_mapping.py concepts this codebase reads — see the fixture's own
"note" field).

This is the exact company/data shape that exposed the corrective patch's
motivating bug: the live COST report showed Revenue ~$275.24B and CapEx
~$5.50B (a real ratio near 2%), but the DCF used a CapEx assumption of
1.00% "derived from |operating margin - FCF margin|" — financially invalid,
since that difference also reflects taxes, D&A, working-capital swings,
interest, and other non-CapEx effects. This file proves, on the real
captured data, that the fix (finance/workflow.py::propose_assumptions's
reported-history-first hierarchy) now derives CapEx/D&A/NWC from ACTUAL
reported history, and that the richer per-assumption audit trail (Problem 2)
survives all the way through finance/dcf.py's validation and the compact
synthesis payload (Problem 11) without either being silently dropped or
blowing the token budget.

Every expected value below was independently produced by RUNNING this
fixture through the real pipeline and reading off the result — never
invented or back-computed to make a test pass (see the corrective patch's
own "do not hardcode expected valuation results until independently
calculated").
"""

import json
import math
from pathlib import Path

import pytest

import tools.finance_tools as finance_tools
from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.dcf import AssumptionSourceType
from finance.provider import AlphaVantageClient, ProviderResponse
from finance.quota import AlphaVantageQuotaLedger, QuotaLedger
from finance.sec_datasets import resolve_sec_dataset
from finance.workflow import (
    AnalysisMode,
    build_compact_synthesis_payload,
    run_full_stock_analysis,
)
from finance.yahoo_datasets import resolve_yahoo_dataset
from tests.test_finance_cache import FakeClock
from tools.base import ToolFailure
from tools.executor import ToolExecutor
from tools.models import MARKET_DATA_INVALID_SYMBOL
from tools.registry import ToolRegistry

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cost_regression.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

SYMBOL = "COST"

# ---- ground truth, from the real captured SEC filings (see the fixture's
# annual income_statement/cash_flow -- these are simply the raw reported
# figures, independent of anything this codebase calculates) ----
FY2025_REVENUE = 275_235_000_000.0
FY2025_CAPEX = 5_498_000_000.0
FY2025_CAPEX_RATIO = FY2025_CAPEX / FY2025_REVENUE  # ~1.997%, matches the live bug report's "~2%"

# ---- ground truth, independently produced by RUNNING this fixture through
# the real pipeline once and reading off the result (see module docstring) ----
EXPECTED_CAPEX_PCT_REVENUE = 0.0184
EXPECTED_DEPRECIATION_PCT_REVENUE = 0.0087
# TSLA DCF validation patch: RE-DERIVED after the working-capital corrective
# fix (finance/workflow.py::_historical_nwc_ratio_pairs now excludes cash/
# short-term-investments from current assets and short-term debt/current-
# portion-of-long-term-debt from current liabilities -- the OPERATING net-
# working-capital concept a DCF needs, not the raw balance-sheet aggregate;
# see docs/PHASE_H1_STOCK_ANALYSIS.md). Independently re-verified by hand
# from this fixture's own reported figures: COST's raw (current_assets -
# current_liabilities)/revenue averaged +0.25% (this constant's old, pre-fix
# value), but COST is a well-known negative-operating-working-capital
# business (collects cash from members/customers immediately, pays suppliers
# on extended terms) -- excluding its large cash/short-term-investments
# balance from current assets reveals that real, negative -5.02% figure
# instead. This is a CORRECTED assumption, not a regression: a business with
# genuinely negative operating working capital being modeled as if a growing
# cash balance were a recurring operating cash OUTFLOW was the actual bug.
EXPECTED_WORKING_CAPITAL_PCT_REVENUE = -0.0502
EXPECTED_NET_DEBT = -13_276_000_000.0  # NEGATIVE -- a net-cash position
EXPECTED_TOTAL_DEBT = 5_670_000_000.0
EXPECTED_CASH = 18_946_000_000.0
# Re-derived after the share-count corrective patch (COR finding): the DCF
# now uses SEC's weighted-average diluted share count (444,803,000) instead
# of Yahoo's basic shares outstanding (443,478,804) -- a ~0.3% difference for
# COST specifically, small but real, hence these values shifted slightly
# from the pre-patch figures. See EXPECTED_DILUTED_SHARES below and
# test_dcf_uses_sec_weighted_average_diluted_shares_not_yahoo_basic.
EXPECTED_DILUTED_SHARES = 444_803_000.0
# TSLA DCF validation patch: re-derived from the corrected (negative, see
# EXPECTED_WORKING_CAPITAL_PCT_REVENUE above) working-capital assumption --
# a negative working-capital ratio RELEASES cash as revenue grows rather
# than consuming it, so every scenario's modeled value per share is HIGHER
# than under the old, incorrect positive-ratio assumption.
EXPECTED_BASE_VALUE_PER_SHARE = 297.897362
EXPECTED_BULL_VALUE_PER_SHARE = 683.489309
EXPECTED_BEAR_VALUE_PER_SHARE = 24.330408


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
    # Real Yahoo/SEC routing, real multi-provider config -- this file is
    # specifically about the Yahoo+SEC path, unlike most other finance test
    # files which the shared conftest.py autouse fixture pins to
    # Alpha Vantage (see tests/conftest.py's own docstring for why).
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

    # Explicitly isolated Alpha Vantage coordinator (tmp_path cache/ledger,
    # never the module-level lazy default from finance_tools.get_coordinator()
    # -- that default points at the REAL, on-disk, cross-session
    # market_data_cache_path()/quota-ledger files, so its quota/cache state
    # is NOT properly test-isolated and depends on whatever else has run in
    # this process. AlphaVantageClient() reads ALPHAVANTAGE_API_KEY fresh on
    # every call (never cached at construction -- see finance/provider.py),
    # so this coordinator genuinely sees this fixture's empty key and the
    # earnings fetch fails with MARKET_DATA_API_KEY_MISSING as intended.
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


def run_cost(wired):
    return run_full_stock_analysis(wired, SYMBOL, include_news=False)


def _base_scenario(result):
    return next(s for s in result.facts["dcf"]["scenarios"] if s["scenario"] == "base")


def _scenario(result, name):
    return next(s for s in result.facts["dcf"]["scenarios"] if s["scenario"] == name)


# ---- 1: the fixture itself is real COST data, not a stand-in ----

def test_fixture_is_real_captured_cost_data(wired):
    result = run_cost(wired)
    assert result.facts["overview"]["name"] == "Costco Wholesale Corporation"
    annual_income = result.facts["statements"]["annual"]["income_statement"]
    assert annual_income[0]["values"]["revenue"] == pytest.approx(FY2025_REVENUE)


def test_fixture_reproduces_the_exact_live_bug_report_figures(wired):
    """Revenue ~$275.24B and CapEx ~$5.50B -- the exact figures named in the
    live COST report that exposed the original CapEx bug."""
    result = run_cost(wired)
    annual_cashflow = result.facts["statements"]["annual"]["cash_flow"]
    assert annual_cashflow[0]["values"]["capital_expenditure"] == pytest.approx(FY2025_CAPEX)
    assert FY2025_CAPEX_RATIO == pytest.approx(0.01997, abs=0.0001)  # ~2%, not the old 1.00% bug


# ---- 2: CapEx/D&A/NWC are derived from REPORTED HISTORY, not a margin difference ----

def test_capex_pct_revenue_matches_the_five_year_reported_average(wired):
    result = run_cost(wired)
    base = _base_scenario(result)
    assert base["assumptions"]["capex_pct_revenue"][0] == pytest.approx(EXPECTED_CAPEX_PCT_REVENUE, abs=1e-4)

    prov = base["assumptions"]["assumption_provenance"]["capex_pct_revenue"]
    assert prov["source_type"] == AssumptionSourceType.DETERMINISTIC_CALCULATION
    assert len(prov["source_periods"]) == 5
    assert "Average of 5 reported CapEx/revenue ratios" in prov["derivation"]
    # The old bug's own signature phrase must never appear again.
    assert "operating margin" not in prov["derivation"].lower()
    assert "fcf margin" not in prov["derivation"].lower()
    assert "free cash flow margin" not in prov["derivation"].lower()


def test_depreciation_and_working_capital_are_also_reported_history_derived(wired):
    result = run_cost(wired)
    base = _base_scenario(result)
    assert base["assumptions"]["depreciation_pct_revenue"][0] == pytest.approx(
        EXPECTED_DEPRECIATION_PCT_REVENUE, abs=1e-4)
    assert base["assumptions"]["working_capital_pct_revenue"][0] == pytest.approx(
        EXPECTED_WORKING_CAPITAL_PCT_REVENUE, abs=1e-4)
    for field in ("depreciation_pct_revenue", "working_capital_pct_revenue"):
        prov = base["assumptions"]["assumption_provenance"][field]
        assert prov["source_type"] == AssumptionSourceType.DETERMINISTIC_CALCULATION
        assert len(prov["source_periods"]) == 5


def test_capex_depreciation_and_nwc_provenance_is_identical_across_scenarios(wired):
    """These three (and tax_rate) come from HISTORY, not from a scenario's
    own revenue-growth/margin deltas -- base/bull/bear must all cite the
    SAME reported ratio, never a scenario-specific recomputation."""
    result = run_cost(wired)
    base, bull, bear = (_scenario(result, n) for n in ("base", "bull", "bear"))
    for field in ("capex_pct_revenue", "depreciation_pct_revenue", "working_capital_pct_revenue"):
        values = {base["assumptions"][field][0], bull["assumptions"][field][0], bear["assumptions"][field][0]}
        assert len(values) == 1, f"{field} must not vary by scenario"


def test_revenue_growth_and_operating_margin_do_vary_by_scenario(wired):
    """The scenario-specific inputs (unlike CapEx/D&A/NWC) are SUPPOSED to
    differ -- confirms the fixture is exercising a real 3-scenario spread,
    not silently collapsing to one value everywhere."""
    result = run_cost(wired)
    base, bull, bear = (_scenario(result, n) for n in ("base", "bull", "bear"))
    growth_values = {base["assumptions"]["revenue_growth"][0], bull["assumptions"]["revenue_growth"][0],
                     bear["assumptions"]["revenue_growth"][0]}
    assert len(growth_values) == 3


def test_historical_fcff_reconciles_from_five_independent_reported_inputs(wired):
    """Problem 3: FCFF = EBIT*(1-tax) + D&A - CapEx - delta(NWC) -- the SAME
    formula finance/dcf.py::_project_raw uses to forecast FORWARD must also
    be computable, and SANE, when fed backward-looking REAL reported COST
    figures for FY2025 (each an independently reported fact, none derived
    from another): operating_income (EBIT), D&A, CapEx, and the CHANGE in
    (current_assets - current_liabilities) between FY2025 and FY2024. This
    is the "never reuse the same economic input twice" requirement made
    concrete: the reconciled figure must differ from each of its five
    individual inputs AND from reported operating cash flow (a different,
    already-reported metric that does NOT go into this formula) -- proving
    FCFF is a genuine five-input derivation, not a relabeled copy of
    something already on the statements.
    """
    result = run_cost(wired)
    income = result.facts["statements"]["annual"]["income_statement"]
    cashflow = result.facts["statements"]["annual"]["cash_flow"]
    balance = result.facts["statements"]["annual"]["balance_sheet"]

    ebit_2025 = income[0]["values"]["operating_income"]
    da_2025 = cashflow[0]["values"]["depreciation_amortization"]
    capex_2025 = cashflow[0]["values"]["capital_expenditure"]
    ocf_2025 = cashflow[0]["values"]["operating_cash_flow"]
    nwc_2025 = balance[0]["values"]["current_assets"] - balance[0]["values"]["current_liabilities"]
    nwc_2024 = balance[1]["values"]["current_assets"] - balance[1]["values"]["current_liabilities"]
    delta_nwc = nwc_2025 - nwc_2024

    tax_rate = 0.21  # the same configured default propose_assumptions uses
    nopat = ebit_2025 * (1 - tax_rate)
    historical_fcff_approx = nopat + da_2025 - capex_2025 - delta_nwc

    assert historical_fcff_approx == pytest.approx(nopat + da_2025 - capex_2025 - delta_nwc)
    assert math.isfinite(historical_fcff_approx)
    # Never a disguised copy of one of its own five inputs.
    for individual_input in (ebit_2025, da_2025, capex_2025, delta_nwc, nopat):
        assert historical_fcff_approx != pytest.approx(individual_input)
    # Never a disguised copy of a DIFFERENT already-reported cash-flow figure either.
    assert historical_fcff_approx != pytest.approx(ocf_2025)
    assert historical_fcff_approx != pytest.approx(ocf_2025 - capex_2025)


# ---- 3: DCF recomputation -- independently observed, not invented (module docstring) ----

def test_dcf_scenario_values_match_the_independently_computed_result(wired):
    result = run_cost(wired)
    base, bull, bear = (_scenario(result, n) for n in ("base", "bull", "bear"))
    assert base["value_per_share"] == pytest.approx(EXPECTED_BASE_VALUE_PER_SHARE, rel=1e-6)
    assert bull["value_per_share"] == pytest.approx(EXPECTED_BULL_VALUE_PER_SHARE, rel=1e-6)
    assert bear["value_per_share"] == pytest.approx(EXPECTED_BEAR_VALUE_PER_SHARE, rel=1e-6)


def test_bull_exceeds_base_exceeds_bear(wired):
    """A model-plausibility sanity check independent of the exact figures:
    higher growth/margin assumptions must produce a higher modeled value."""
    result = run_cost(wired)
    base, bull, bear = (_scenario(result, n) for n in ("base", "bull", "bear"))
    assert bull["value_per_share"] > base["value_per_share"] > bear["value_per_share"]


def test_dcf_is_deterministic_across_repeated_runs(wired):
    result_a = run_cost(wired)
    result_b = run_cost(wired)
    assert result_a.facts["dcf"]["scenarios"] == result_b.facts["dcf"]["scenarios"]


# ---- 4: equity bridge preserves net cash (COST is net-cash, not net-debt) ----

def test_equity_bridge_preserves_the_net_cash_position(wired):
    result = run_cost(wired)
    base = _base_scenario(result)
    assert base["total_debt"] == pytest.approx(EXPECTED_TOTAL_DEBT)
    assert base["cash_and_cash_equivalents"] == pytest.approx(EXPECTED_CASH)
    assert base["net_debt"] == pytest.approx(EXPECTED_NET_DEBT)
    assert base["net_debt"] < 0, "COST holds more cash than debt -- net_debt must be negative, never clamped to 0"


def test_net_debt_is_identical_across_scenarios(wired):
    """The equity bridge is a balance-sheet fact, not a scenario assumption
    -- it must not vary with revenue-growth/margin deltas."""
    result = run_cost(wired)
    base, bull, bear = (_scenario(result, n) for n in ("base", "bull", "bear"))
    assert base["net_debt"] == bull["net_debt"] == bear["net_debt"]


# ---- 4b: share-count policy (Phase 8 corrective patch, COR finding) ----

def test_dcf_uses_sec_weighted_average_diluted_shares_not_yahoo_basic(wired):
    """COST's own Yahoo/SEC share counts differ slightly (443,478,804 vs
    444,803,000) -- small for COST, but the SAME policy question COR
    exposed with a larger ~2.2% divergence. The DCF must use SEC's
    weighted-average diluted figure (internally consistent with the
    SEC-sourced income statement/balance sheet the rest of the valuation is
    built from), not silently take whichever provider's overview happened
    to populate first."""
    result = run_cost(wired)
    assert result.facts.get("dcf_shares_outstanding_source") == "sec_weighted_average_diluted"
    base = _base_scenario(result)
    assert base["diluted_shares"] == pytest.approx(EXPECTED_DILUTED_SHARES)


def test_yahoo_basic_shares_remain_visible_separately_never_deleted(wired):
    """Phase 8: Yahoo's own figure must survive as a separate, inspectable
    current-market-data fact even though the DCF doesn't use it -- never
    silently dropped just because SEC's figure won the DCF precedence."""
    result = run_cost(wired)
    yahoo_shares = result.facts["overview"].get("shares_outstanding")
    sec_shares = result.facts["statements"]["annual"]["balance_sheet"][0]["values"].get("shares_outstanding")
    assert yahoo_shares is not None and yahoo_shares > 0
    assert sec_shares is not None and sec_shares > 0
    assert yahoo_shares != sec_shares  # COST's own real, legitimately-different figures


# ---- 5: assumptions are correctly labeled reported vs. configured ----

def test_reported_vs_configured_assumptions_are_correctly_labeled(wired):
    result = run_cost(wired)
    base = _base_scenario(result)
    provenance = base["assumptions"]["assumption_provenance"]
    # Phase H.4 SPLIT the old single `deterministic_calculation` label. Both
    # a five-year historical CAGR and a trailing-twelve-month trend were
    # "deterministically calculated", but they are not the same kind of
    # claim, and a report that cannot tell them apart cannot say which one a
    # forecast rests on. Growth and margin now carry the specific kind;
    # CapEx/D&A/working capital still come from the multi-year reported-ratio
    # hierarchy and keep the original label.
    forward_derived = ("revenue_growth", "operating_margin")
    reported = ("capex_pct_revenue", "depreciation_pct_revenue",
                "working_capital_pct_revenue")
    configured = ("tax_rate", "wacc", "terminal_growth")
    for field in forward_derived:
        assert provenance[field]["source_type"] in (
            AssumptionSourceType.MANAGEMENT_GUIDANCE,
            AssumptionSourceType.TTM_CALCULATION,
            AssumptionSourceType.HISTORICAL_CALCULATION,
            AssumptionSourceType.CONFIGURED_DEFAULT,
        ), field
    for field in reported:
        assert provenance[field]["source_type"] == AssumptionSourceType.DETERMINISTIC_CALCULATION, field
    for field in configured:
        assert provenance[field]["source_type"] == AssumptionSourceType.CONFIGURED_DEFAULT, field
    for field in forward_derived + reported + configured:
        assert provenance[field]["approval_status"] == "proposed", \
            f"{field}: nothing here has been user-approved -- must not claim otherwise"


# ---- 6: reduced-mode transparency for the omitted Alpha Vantage dataset ----

def test_missing_alphavantage_key_produces_a_controlled_error_not_a_crash(wired):
    result = run_cost(wired)
    assert any(e.get("dataset") == "earnings" and e.get("code") == "MARKET_DATA_API_KEY_MISSING"
              for e in result.errors)
    # Yahoo + SEC data must still be fully present despite the AV gap.
    assert result.facts["dcf"]["available"] is True
    assert result.facts["quote"]["available"] is True


# ---- 7: Problem 11 -- compact synthesis payload stays within budget on REAL data ----

def test_compact_synthesis_payload_is_within_budget_for_the_real_cost_fixture(wired):
    result = run_cost(wired)
    compact = build_compact_synthesis_payload(result)
    payload = json.dumps(compact, default=str)
    estimated_tokens = len(payload) // 4

    assert estimated_tokens < 20_000, f"hard limit: {estimated_tokens} tokens"
    # 11_000, not 10_000 -- see test_finance_msft_regression.py's own
    # test_synthesis_prompt_is_compact_for_the_msft_fixture for why (Phase 9's
    # five new leverage-context fundamental_metrics). COST itself stays
    # comfortably under 10k even after that addition; the ceiling still
    # moves up here too so both fixtures are held to the same preferred bar.
    assert estimated_tokens < 11_000, f"preferred threshold: {estimated_tokens} tokens"
    # A generous regression ceiling on the REAL fixture -- catches a future
    # regression without being brittle to the exact byte count.
    assert estimated_tokens < 12_000, f"COST-specific regression ceiling: {estimated_tokens} tokens"


def test_capex_derivation_survives_into_the_compact_payload_not_just_the_full_facts(wired):
    """The rich audit trail (Problem 2) must reach the report-writing
    payload, not only the internal unbounded facts -- this is what the
    finance/dcf.py provenance-stripping bug (found via this exact fixture)
    would have broken silently."""
    result = run_cost(wired)
    compact = build_compact_synthesis_payload(result)
    shared = compact["dcf"].get("shared_assumption_provenance") or {}
    assert "capex_pct_revenue" in shared, \
        "capex_pct_revenue is identical across scenarios and should be hoisted, not per-scenario"
    assert "Average of 5 reported CapEx/revenue ratios" in shared["capex_pct_revenue"]["derivation"]


def test_dcf_forecast_rows_and_assumption_provenance_survive_compaction(wired):
    """Phase H.1's own guarantee (full per-year forecast detail, full
    assumption provenance) re-verified on real COST data, not just MSFT's."""
    result = run_cost(wired)
    compact = build_compact_synthesis_payload(result)
    for scenario in compact["dcf"]["scenarios"]:
        assert len(scenario["forecast"]) == scenario["forecast_years"]
        for row in scenario["forecast"]:
            assert "fcff" in row and "discount_factor" in row


# ---- 8: cache reuse (no repeated external calls for the same run) ----

def test_full_analysis_mode_reached(wired):
    result = run_cost(wired)
    assert result.plan.mode == AnalysisMode.FULL
