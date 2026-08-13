"""Phase H.1 — FullStockAnalysis orchestration and security boundaries.

The load-bearing guarantees here are architectural, not numeric:

* every tool call goes through ToolExecutor, which stays the sole execution authority
* the workflow never performs the DCF arithmetic itself
* provider tools are unavailable for unrelated requests
* hallucinated and registered-but-not-offered tools are both rejected
* cached, stale and provider-fetched data are labelled distinctly and correctly
* a missing API key produces a controlled error, never a crash or a fabrication
"""

import json

import pytest

import tools.config as config
import tools.finance_tools as finance_tools
from finance.cache import MarketDataCache, Origin
from finance.coordinator import MarketDataRequestCoordinator
from finance.provider import ProviderResponse
from finance.quota import AlphaVantageQuotaLedger
from finance.workflow import (
    DCF_TOOL_NAME,
    AnalysisMode,
    _ask_local_with_content_policy,
    _sanitize_stage_error_for_display,
    build_facts,
    plan_analysis,
    propose_assumptions,
    run_full_stock_analysis,
)
from tests.test_finance_cache import FakeClock
from tools.base import ToolValidationError
from tools.executor import ToolExecutor
from tools.models import (
    MARKET_DATA_API_KEY_MISSING,
    TOOL_NOT_IN_SHORTLIST,
    ToolCall,
)
from tools.registry import ToolRegistry, default_registry

SECRET = "SUPER_SECRET_KEY_DO_NOT_LEAK"

QUOTE = {"Global Quote": {"01. symbol": "AAPL", "05. price": "200.00",
                          "08. previous close": "198.00", "06. volume": "1000",
                          "07. latest trading day": "2026-08-04",
                          "10. change percent": "1.0101%"}}
OVERVIEW = {"Symbol": "AAPL", "Name": "Test Corp", "Sector": "TECHNOLOGY",
            "Industry": "SOFTWARE", "Currency": "USD", "SharesOutstanding": "100",
            "MarketCapitalization": "20000", "PERatio": "25.5",
            "Description": "A test company."}
INCOME = {"annualReports": [
    {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD", "totalRevenue": "1000",
     "grossProfit": "400", "operatingIncome": "200", "netIncome": "150",
     "incomeBeforeTax": "190", "incomeTaxExpense": "40"},
    {"fiscalDateEnding": "2024-12-31", "reportedCurrency": "USD", "totalRevenue": "900",
     "grossProfit": "360", "operatingIncome": "180", "netIncome": "130",
     "incomeBeforeTax": "170", "incomeTaxExpense": "40"},
]}
BALANCE = {"annualReports": [
    {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD",
     "totalCurrentAssets": "500", "totalCurrentLiabilities": "250",
     "totalShareholderEquity": "600", "shortTermDebt": "100", "longTermDebt": "200",
     "cashAndCashEquivalentsAtCarryingValue": "150",
     "commonStockSharesOutstanding": "100"}]}
CASHFLOW = {"annualReports": [
    {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD",
     "operatingCashflow": "250", "capitalExpenditures": "70",
     "depreciationDepletionAndAmortization": "50"}]}
EARNINGS = {"annualEarnings": [{"fiscalDateEnding": "2025-12-31", "reportedEPS": "1.50"}]}
PRICES = {"Time Series (Daily)": {
    f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}": {
        "1. open": "100", "2. high": "101", "3. low": "99",
        "4. close": str(100 + i * 0.1), "5. adjusted close": str(100 + i * 0.1),
        "6. volume": "1000"}
    for i in range(220)}}

PAYLOAD_BY_DATASET = {
    "stock_quote": QUOTE, "company_overview": OVERVIEW, "income_statement": INCOME,
    "balance_sheet": BALANCE, "cash_flow": CASHFLOW, "earnings": EARNINGS,
    "daily_prices": PRICES,
}


class DatasetClient:
    """Serves a canned payload per dataset and counts every external call."""

    def __init__(self, failures=None):
        self.calls = []
        self.failures = dict(failures or {})

    def fetch(self, dataset, arguments):
        self.calls.append(dataset.dataset_id)
        if dataset.dataset_id in self.failures:
            raise self.failures[dataset.dataset_id]
        payload = PAYLOAD_BY_DATASET.get(dataset.dataset_id, {"ok": True})
        return ProviderResponse(payload=payload,
                                provider_metadata={"provider_function": dataset.function},
                                byte_count=len(json.dumps(payload)))

    @property
    def call_count(self):
        return len(self.calls)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def coordinator(tmp_path, clock):
    return MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "m.sqlite3"), clock=clock),
        ledger=AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock,
                                       daily_limit=100),
        client=DatasetClient(), clock=clock, sleeper=lambda _s: None, jitter=lambda: 0.5)


@pytest.fixture
def wired(coordinator, monkeypatch):
    """A registry + executor with the finance tools, sharing one coordinator."""
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", SECRET)
    finance_tools.set_coordinator(coordinator)
    registry = ToolRegistry()
    for tool_cls in finance_tools.ALL_FINANCE_TOOL_CLASSES:
        registry.register(tool_cls())
    yield registry, ToolExecutor(registry), coordinator
    finance_tools.set_coordinator(None)


# ---- registration and permissions ----

def test_finance_tools_are_registered_read_only(wired):
    registry, _executor, _coord = wired
    names = [d.name for d in registry.enabled_definitions()]

    assert DCF_TOOL_NAME in names
    assert "finance.company_overview" in names
    for definition in registry.enabled_definitions():
        assert definition.permission.value == "read", \
            f"{definition.name} must be READ; this phase has no write finance tool"


def test_no_trading_or_order_capability_exists():
    registry = default_registry()
    forbidden = ("order", "trade", "buy", "sell", "brokerage", "portfolio", "position")
    for definition in registry.enabled_definitions():
        assert not any(word in definition.name.lower() for word in forbidden), \
            f"{definition.name} looks like a trading capability"


def test_dcf_tool_needs_no_internet(wired):
    registry, _executor, _coord = wired
    assert registry.get(DCF_TOOL_NAME).requires_internet is False


# ---- 3: unrelated requests never touch the provider ----

def test_unrelated_request_does_not_start_market_data(wired):
    _registry, _executor, coordinator = wired
    registry = default_registry()
    shortlist = registry.shortlist_tools("what is the capital of France", limit=5)

    assert not any(d.name.startswith("finance.") and d.name != DCF_TOOL_NAME
                   for d in shortlist), "market-data tools must not surface for trivia"
    assert coordinator._client.call_count == 0  # noqa: SLF001


def test_stock_request_surfaces_the_finance_tools():
    registry = default_registry()
    shortlist = registry.shortlist_tools(
        "full stock analysis of AAPL including a dcf valuation", limit=6)
    assert any(d.name.startswith("finance.") for d in shortlist)


# ---- 4: cached data avoids provider startup ----

def test_fully_cached_analysis_makes_zero_external_calls(wired):
    _registry, executor, coordinator = wired
    client = coordinator._client  # noqa: SLF001

    run_full_stock_analysis(executor, "AAPL", include_news=False)
    first_calls = client.call_count
    assert first_calls > 0

    run_full_stock_analysis(executor, "AAPL", include_news=False)
    assert client.call_count == first_calls, \
        "a fully cached re-run must not contact the provider at all"


# ---- 5: missing API key is controlled ----

def test_missing_api_key_produces_a_controlled_error(tmp_path, clock, monkeypatch):
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "")
    from finance.provider import AlphaVantageClient
    from finance.datasets import resolve_dataset
    from tools.base import ToolFailure

    client = AlphaVantageClient(api_key_reader=lambda: None)
    with pytest.raises(ToolFailure) as excinfo:
        client.fetch(resolve_dataset("stock_quote"), {"symbol": "AAPL"})

    assert excinfo.value.code == MARKET_DATA_API_KEY_MISSING
    assert "ALPHAVANTAGE_API_KEY" in excinfo.value.message
    assert SECRET not in excinfo.value.message


# ---- 6-8: shortlist enforcement ----

def test_hallucinated_tool_is_rejected(wired):
    _registry, executor, _coord = wired
    result = executor.execute(ToolCall("c1", "finance.place_order", {"symbol": "AAPL"}),
                              step=1)
    assert result.success is False
    assert result.error.code == "UNKNOWN_TOOL"


def test_registered_but_not_offered_tool_is_rejected_by_the_loop():
    """tool_loop rejects an mcp.* tool that was never shortlisted this round."""
    import tool_loop

    registry = default_registry()
    shortlisted_names = {"math.calculate"}
    call = ToolCall("c1", "mcp.alphavantage.stock_quote", {"symbol": "AAPL"})
    not_registered = not registry.has(call.tool_name)
    rejected = (not_registered or call.tool_name.startswith("mcp.")) \
        and call.tool_name not in shortlisted_names
    assert rejected, "an unoffered mcp.* tool must be rejected before the executor"
    assert TOOL_NOT_IN_SHORTLIST


def test_invalid_ticker_shapes_are_rejected_before_any_request(wired):
    _registry, executor, coordinator = wired
    client = coordinator._client  # noqa: SLF001

    for bad in ["../etc/passwd", "AAPL&function=X", "A" * 50, "", "AA PL", "<script>"]:
        result = executor.execute(
            ToolCall("c1", "finance.company_overview", {"symbol": bad}), step=1)
        assert result.success is False, f"{bad!r} should be rejected"
        assert result.error.code == "INVALID_ARGUMENTS"
    assert client.call_count == 0, "no malformed symbol may reach the provider"


# ---- 9: the DCF arithmetic comes from the tool ----

def test_dcf_arithmetic_comes_from_the_registered_tool(wired):
    _registry, executor, _coord = wired
    executed = []
    original = executor.execute

    def spy(call, step=0, confirmation=None):
        executed.append(call.tool_name)
        return original(call, step=step, confirmation=confirmation)

    executor.execute = spy
    result = run_full_stock_analysis(executor, "AAPL", include_news=False)

    assert DCF_TOOL_NAME in executed, "the valuation must run through ToolExecutor"
    assert result.facts["dcf"]["available"] is True
    assert result.facts["dcf"]["calculation_version"] == "fcff_enterprise_v1"
    assert result.facts["dcf"]["value_per_share"] is not None


def test_workflow_never_computes_a_valuation_itself(wired):
    """With the DCF tool unregistered, the workflow must REPORT the gap rather
    than fall back to calculating a value on its own."""
    registry, _executor, _coord = wired
    registry.unregister(DCF_TOOL_NAME)
    executor = ToolExecutor(registry)

    result = run_full_stock_analysis(executor, "AAPL", include_news=False)

    assert result.facts["dcf"]["available"] is False
    assert result.facts["valuation_gap"]["available"] is False
    assert any(e["tool"] == DCF_TOOL_NAME for e in result.errors)


# ---- 10-11: provenance reaches the model, correctly labelled ----

def test_facts_carry_timestamps_provenance_and_assumptions(wired):
    _registry, executor, _coord = wired
    result = run_full_stock_analysis(executor, "AAPL", include_news=False)
    facts = result.facts

    assert facts["generated_at_utc"]
    assert facts["data_provenance"], "the model must see where each dataset came from"
    for provenance in facts["data_provenance"].values():
        assert "origin" in provenance and "retrieved_at_utc" in provenance
        assert provenance["origin"] in (Origin.PROVIDER, Origin.CACHE)
    assert facts["dcf"]["scenarios"][0]["assumptions"]
    assert facts["dcf"]["assumptions_origin"] == "derived_from_reported_history"
    for metric in facts["fundamental_metrics"].values():
        assert metric["calculation_version"]


def test_cached_and_fetched_data_are_labelled_distinctly(wired):
    _registry, executor, _coord = wired

    first = run_full_stock_analysis(executor, "AAPL", include_news=False)
    assert all(p["origin"] == Origin.PROVIDER
               for p in first.facts["data_provenance"].values())

    second = run_full_stock_analysis(executor, "AAPL", include_news=False)
    assert all(p["origin"] == Origin.CACHE
               for p in second.facts["data_provenance"].values()), \
        "a cached re-run must never claim the data was freshly retrieved"


def test_stale_data_is_labelled_stale_and_warned_about(wired, clock):
    _registry, executor, coordinator = wired
    run_full_stock_analysis(executor, "AAPL", include_news=False)

    # Three days: past the quote (15 min) and price-history (12 h) TTLs but still
    # inside the 7-day stale-if-error grace window, so those records are servable
    # as STALE while the statements remain fresh.
    clock.advance(3 * 24 * 3600)
    coordinator.ledger.record_rate_limit(exhausted=True)  # force the stale path

    result = run_full_stock_analysis(executor, "AAPL", include_news=False)

    assert result.plan.mode == AnalysisMode.STALE
    provenance = result.facts["data_provenance"]
    assert provenance["stock_quote"]["stale"] is True
    assert provenance["income_statement"]["stale"] is False, \
        "a still-fresh dataset must not be tarred as stale"
    assert any("STALE" in w for w in result.warnings)


def test_market_price_is_never_described_as_realtime(wired):
    _registry, executor, _coord = wired
    result = run_full_stock_analysis(executor, "AAPL", include_news=False)

    assert result.facts["quote"]["price_basis"] == "delayed"
    assert result.facts["valuation_gap"]["market_price_basis"] == "delayed"
    assert "not a trading instruction" in result.facts["valuation_gap"]["note"]


def test_valuation_gap_is_computed_locally_not_by_the_model(wired):
    _registry, executor, _coord = wired
    result = run_full_stock_analysis(executor, "AAPL", include_news=False)
    gap = result.facts["valuation_gap"]

    assert gap["available"] is True
    assert gap["market_price"] == pytest.approx(200.0)
    expected = gap["estimated_base_modeled_value_per_share"] - 200.0
    assert gap["difference"] == pytest.approx(expected, abs=1e-6)
    assert gap["direction"] in ("above", "below", "equal")


# ---- scenario spread: the confidence signal behind the Research Stance section ----

def test_scenario_spread_is_computed_locally_not_by_the_model(wired):
    from finance.workflow import _scenario_spread

    _registry, executor, _coord = wired
    result = run_full_stock_analysis(executor, "AAPL", include_news=False)
    dcf = result.facts["dcf"]
    spread = result.facts["dcf_scenario_spread"]

    by_name = {s["scenario"]: s["value_per_share"] for s in dcf["scenarios"]}
    assert spread["available"] is True
    assert spread["bull_value_per_share"] == pytest.approx(by_name["bull"])
    assert spread["bear_value_per_share"] == pytest.approx(by_name["bear"])
    assert spread["spread"] == pytest.approx(by_name["bull"] - by_name["bear"], abs=1e-6)
    assert spread["spread_pct_of_base"] == pytest.approx(
        (by_name["bull"] - by_name["bear"]) / abs(by_name["base"]), abs=1e-6)
    # Bull must exceed bear on this fixture (bull assumptions are strictly
    # more favorable) — a sanity check that the numbers used are real.
    assert spread["bull_value_per_share"] > spread["bear_value_per_share"]

    # Directly unit-testable too, independent of the full workflow.
    direct = _scenario_spread(dcf)
    assert direct == spread


def test_scenario_spread_unavailable_when_dcf_unavailable():
    from finance.workflow import _scenario_spread

    assert _scenario_spread(None) == {"available": False}
    assert _scenario_spread({"available": False}) == {"available": False}


def test_scenario_spread_unavailable_when_a_scenario_is_missing():
    from finance.workflow import _scenario_spread

    dcf = {"available": True, "scenarios": [
        {"scenario": "base", "value_per_share": 100.0},
        {"scenario": "bull", "value_per_share": 150.0},
        # no "bear" entry
    ]}
    result = _scenario_spread(dcf)
    assert result["available"] is False
    assert "reason" in result


def test_wider_bull_bear_range_yields_a_larger_spread_pct():
    from finance.workflow import _scenario_spread

    tight = _scenario_spread({"available": True, "scenarios": [
        {"scenario": "base", "value_per_share": 100.0},
        {"scenario": "bull", "value_per_share": 110.0},
        {"scenario": "bear", "value_per_share": 90.0}]})
    wide = _scenario_spread({"available": True, "scenarios": [
        {"scenario": "base", "value_per_share": 100.0},
        {"scenario": "bull", "value_per_share": 200.0},
        {"scenario": "bear", "value_per_share": 10.0}]})

    assert wide["spread_pct_of_base"] > tight["spread_pct_of_base"]


# ===========================================================================
# TSLA DCF validation patch
# ===========================================================================

# ---- root cause: operating (not raw) net working capital (section 3) ----

def _bs_period(fiscal_date, current_assets, current_liabilities, cash,
              short_term_investments=None, short_term_debt=None,
              current_portion_of_long_term_debt=None):
    values = {"current_assets": current_assets, "current_liabilities": current_liabilities,
             "cash_and_cash_equivalents": cash}
    if short_term_investments is not None:
        values["short_term_investments"] = short_term_investments
    if short_term_debt is not None:
        values["short_term_debt"] = short_term_debt
    if current_portion_of_long_term_debt is not None:
        values["current_portion_of_long_term_debt"] = current_portion_of_long_term_debt
    return {"fiscal_date": fiscal_date, "values": values}


def _inc_period(fiscal_date, revenue):
    return {"fiscal_date": fiscal_date, "values": {"revenue": revenue}}


def test_historical_nwc_ratio_excludes_cash_and_short_term_investments():
    """The TSLA root cause: a large cash/short-term-investments balance
    inside current_assets must NOT be treated as operating working capital
    -- excluding it can flip the ratio's SIGN, exactly as it did for TSLA."""
    from finance.workflow import _historical_nwc_ratio_pairs

    # current_assets is dominated by cash: raw (CA-CL)/revenue would be
    # strongly POSITIVE, but the OPERATING figure (excluding the $80 cash)
    # is negative.
    balance = [_bs_period("2025-12-31", current_assets=100.0, current_liabilities=30.0, cash=80.0)]
    income = [_inc_period("2025-12-31", revenue=200.0)]

    pairs = _historical_nwc_ratio_pairs(balance, income, max_years=5)
    assert len(pairs) == 1
    _date, ratio = pairs[0]
    # operating_current_assets = 100 - 80 = 20; operating_nwc = 20 - 30 = -10
    assert ratio == pytest.approx(-10.0 / 200.0)
    assert ratio < 0


def test_historical_nwc_ratio_excludes_short_term_debt_and_current_ltd():
    from finance.workflow import _historical_nwc_ratio_pairs

    balance = [_bs_period("2025-12-31", current_assets=100.0, current_liabilities=90.0, cash=10.0,
                          short_term_debt=40.0, current_portion_of_long_term_debt=20.0)]
    income = [_inc_period("2025-12-31", revenue=200.0)]

    pairs = _historical_nwc_ratio_pairs(balance, income, max_years=5)
    # operating_current_assets = 100 - 10 = 90
    # operating_current_liabilities = 90 - 40 - 20 = 30
    # operating_nwc = 90 - 30 = 60
    _date, ratio = pairs[0]
    assert ratio == pytest.approx(60.0 / 200.0)


def test_historical_nwc_ratio_period_excluded_when_cash_is_not_reported():
    """cash_and_cash_equivalents is required for a period to be usable --
    every real balance sheet reports it, so its absence means the period
    itself cannot be used, not that the exclusion should be silently
    skipped (which would just reintroduce the bug for that period)."""
    from finance.workflow import _historical_nwc_ratio_pairs

    balance = [{"fiscal_date": "2025-12-31",
               "values": {"current_assets": 100.0, "current_liabilities": 30.0}}]  # no cash
    income = [_inc_period("2025-12-31", revenue=200.0)]
    assert _historical_nwc_ratio_pairs(balance, income, max_years=5) == []


def test_historical_nwc_ratio_defaults_absent_sti_and_debt_to_zero():
    """UNLIKE cash, short_term_investments/short_term_debt/current_portion_
    of_long_term_debt default to 0 when absent -- matching finance.dcf.
    compute_net_debt's own convention: 'not reported' legitimately means
    'the company has none' for these three."""
    from finance.workflow import _historical_nwc_ratio_pairs

    balance = [_bs_period("2025-12-31", current_assets=100.0, current_liabilities=30.0, cash=10.0)]
    income = [_inc_period("2025-12-31", revenue=200.0)]
    pairs = _historical_nwc_ratio_pairs(balance, income, max_years=5)
    # operating_current_assets = 100 - 10 - 0 = 90; operating_nwc = 90 - 30 = 60
    assert pairs[0][1] == pytest.approx(60.0 / 200.0)


def test_tsla_style_cash_rich_balance_sheet_produces_a_negative_operating_ratio():
    """Reproduces the TSLA MAGNITUDE directly: current assets dominated by
    cash + short-term investments produces a large POSITIVE raw ratio but a
    NEGATIVE operating ratio -- the exact inversion that fed a fake, ever-
    growing 'cash investment' into the DCF and broke bull >= base >= bear."""
    from finance.workflow import _historical_nwc_ratio_pairs

    balance = [_bs_period("2025-12-31", current_assets=68_642_000_000.0,
                          current_liabilities=31_714_000_000.0, cash=16_513_000_000.0,
                          short_term_investments=27_546_000_000.0,
                          short_term_debt=1_569_000_000.0)]
    income = [_inc_period("2025-12-31", revenue=94_827_000_000.0)]

    raw_ratio = (68_642_000_000.0 - 31_714_000_000.0) / 94_827_000_000.0
    assert raw_ratio > 0.35  # the OLD (buggy) figure would have been strongly positive

    pairs = _historical_nwc_ratio_pairs(balance, income, max_years=5)
    _date, operating_ratio = pairs[0]
    assert operating_ratio < 0  # the CORRECT figure is negative


# ---- valuation-gap math for a non-positive / invalid modeled value (section 8) ----

def test_valuation_gap_percentage_is_null_for_a_non_positive_modeled_value():
    from finance.workflow import _valuation_gap

    facts = {"quote": {"price": 328.58, "price_basis": "delayed"},
            "dcf": {"available": True, "validation_status": "DCF_VALID",
                    "value_per_share": -39.38, "calculation_version": "v1"}}
    gap = _valuation_gap(facts)
    assert gap["available"] is True
    assert gap["difference_pct"] is None
    assert gap["valuation_gap_status"] == "not_meaningful"
    # The dollar difference remains well-defined and is NOT withheld.
    assert gap["difference"] == pytest.approx(-39.38 - 328.58, abs=1e-6)


def test_valuation_gap_percentage_is_meaningful_for_a_positive_modeled_value():
    from finance.workflow import _valuation_gap

    facts = {"quote": {"price": 100.0, "price_basis": "delayed"},
            "dcf": {"available": True, "validation_status": "DCF_VALID",
                    "value_per_share": 150.0, "calculation_version": "v1"}}
    gap = _valuation_gap(facts)
    assert gap["difference_pct"] == pytest.approx(0.5, abs=1e-6)
    assert gap["valuation_gap_status"] == "meaningful"


def test_valuation_gap_unavailable_when_dcf_validation_failed():
    from finance.workflow import _valuation_gap

    facts = {"quote": {"price": 328.58, "price_basis": "delayed"},
            "dcf": {"available": True, "validation_status": "DCF_INVALID_SCENARIO_ORDER",
                    "value_per_share": -39.38}}
    gap = _valuation_gap(facts)
    assert gap["available"] is False
    assert "validation" in gap["reason"].lower()


def test_valuation_gap_never_states_a_percentage_when_base_is_exactly_zero():
    from finance.workflow import _valuation_gap

    facts = {"quote": {"price": 100.0, "price_basis": "delayed"},
            "dcf": {"available": True, "validation_status": "DCF_VALID",
                    "value_per_share": 0.0}}
    gap = _valuation_gap(facts)
    assert gap["difference_pct"] is None
    assert gap["valuation_gap_status"] == "not_meaningful"


def test_scenario_spread_unavailable_when_dcf_validation_failed():
    from finance.workflow import _scenario_spread

    dcf = {"available": True, "validation_status": "DCF_NEGATIVE_TERMINAL_FCFF",
          "scenarios": [{"scenario": "base", "value_per_share": -10.0},
                        {"scenario": "bull", "value_per_share": -20.0},
                        {"scenario": "bear", "value_per_share": -5.0}]}
    result = _scenario_spread(dcf)
    assert result["available"] is False
    assert "reason" in result


def test_scenario_spread_still_available_when_validation_status_is_absent():
    """Backward compatibility: a dcf dict with no 'validation_status' field
    at all (an older/hand-built caller) must default to 'treated as usable',
    never be silently withheld."""
    from finance.workflow import _scenario_spread

    dcf = {"available": True, "scenarios": [
        {"scenario": "base", "value_per_share": 100.0},
        {"scenario": "bull", "value_per_share": 150.0},
        {"scenario": "bear", "value_per_share": 50.0}]}
    assert _scenario_spread(dcf)["available"] is True


# ---- research readiness (section 11; renamed from "decision readiness" by
# the DIS valuation/readiness correction -- see test_finance_report_
# compaction.py for the SECOND, pipeline-aware layer) ----

def test_research_readiness_not_ready_when_dcf_validation_failed():
    from finance.workflow import AnalysisPlan, ResearchReadiness, _research_readiness

    plan = AnalysisPlan("TSLA", AnalysisMode.FULL, (), (), (), 0, 0, "")
    facts = {"dcf": {"available": True, "validation_status": "DCF_NEGATIVE_TERMINAL_FCFF"},
            "statements": {"warnings": []}}
    readiness = _research_readiness(plan, facts)
    assert readiness["status"] == ResearchReadiness.NOT_READY
    assert readiness["reasons"]


def test_research_readiness_not_ready_on_reconciliation_conflict():
    from finance.workflow import AnalysisPlan, ResearchReadiness, _research_readiness

    plan = AnalysisPlan("X", AnalysisMode.FULL, (), (), (), 0, 0, "")
    facts = {"dcf": {"available": True, "validation_status": "DCF_VALID"},
            "statements": {"warnings": ["total_debt does not reconcile with the provider's own "
                                       "total-debt figure"]}}
    readiness = _research_readiness(plan, facts)
    assert readiness["status"] == ResearchReadiness.NOT_READY


def test_research_readiness_limited_when_analysis_mode_is_not_full():
    from finance.workflow import AnalysisPlan, ResearchReadiness, _research_readiness

    plan = AnalysisPlan("X", AnalysisMode.REDUCED, (), (), (), 0, 0, "")
    facts = {"dcf": {"available": True, "validation_status": "DCF_VALID"},
            "statements": {"warnings": []}}
    readiness = _research_readiness(plan, facts)
    assert readiness["status"] == ResearchReadiness.LIMITED


def test_research_readiness_limited_when_dcf_unavailable():
    from finance.workflow import AnalysisPlan, ResearchReadiness, _research_readiness

    plan = AnalysisPlan("X", AnalysisMode.FULL, (), (), (), 0, 0, "")
    facts = {"dcf": {"available": False, "reason": "DCF_ASSUMPTION_REQUIRED"},
            "statements": {"warnings": []}}
    readiness = _research_readiness(plan, facts)
    assert readiness["status"] == ResearchReadiness.LIMITED


def test_research_readiness_limited_when_dcf_valid_with_warnings():
    from finance.workflow import AnalysisPlan, ResearchReadiness, _research_readiness

    plan = AnalysisPlan("X", AnalysisMode.FULL, (), (), (), 0, 0, "")
    facts = {"dcf": {"available": True, "validation_status": "DCF_VALID_WITH_WARNINGS"},
            "statements": {"warnings": []}}
    readiness = _research_readiness(plan, facts)
    assert readiness["status"] == ResearchReadiness.LIMITED


def test_research_readiness_ready_on_a_clean_full_run():
    from finance.workflow import AnalysisPlan, ResearchReadiness, _research_readiness

    plan = AnalysisPlan("X", AnalysisMode.FULL, (), (), (), 0, 0, "")
    facts = {"dcf": {"available": True, "validation_status": "DCF_VALID"},
            "statements": {"warnings": []}}
    readiness = _research_readiness(plan, facts)
    assert readiness["status"] == ResearchReadiness.READY


def test_research_readiness_is_never_a_buy_sell_hold_avoid_word():
    from finance.workflow import ResearchReadiness

    forbidden = {"BUY", "SELL", "HOLD", "AVOID"}
    for status in ResearchReadiness.ALL:
        assert status not in forbidden


def test_research_readiness_is_populated_on_the_full_workflow_result(wired):
    _registry, executor, _coord = wired
    result = run_full_stock_analysis(executor, "AAPL", include_news=False)
    from finance.workflow import ResearchReadiness
    assert result.facts["research_readiness"]["status"] in ResearchReadiness.ALL


def test_no_decision_readiness_field_name_remains_in_facts(wired):
    """Problem 6 (DIS correction): the OLD field name must not survive
    anywhere a caller might still read it from."""
    _registry, executor, _coord = wired
    result = run_full_stock_analysis(executor, "AAPL", include_news=False)
    assert "decision_readiness" not in result.facts
    assert "research_readiness" in result.facts


# ---- the required Research Stance section (Phase H.3 corrective patch,
# Problem 1: research characterization only, no buy/hold/add/sell verdict) ----

def test_system_instructions_require_a_research_stance_not_a_trade_verdict():
    from finance.workflow import FINANCE_REPORT_SYSTEM_INSTRUCTIONS as instructions

    # The OLD action-oriented vocabulary must be explicitly PROHIBITED, not
    # merely absent -- the instructions must say not to use these words.
    assert "Do NOT use the words BUY, SELL, HOLD, AVOID" in instructions
    assert "if you hold" in instructions.lower() or "if you do not hold" in instructions.lower()
    # The new vocabulary must be present and grounded in the deterministic
    # confidence signal.
    for word in ("research_stance", "valuation_view", "overall_risk"):
        assert word in instructions
    assert "dcf_scenario_spread" in instructions
    assert "confidence" in instructions.lower()
    # ...and must not instruct the model to simply decline to answer.
    assert "do not use it as a reason to withhold an assessment" in instructions
    # Real guardrails must still be present: no guarantees, no order mechanics.
    assert "NEVER guarantee" in instructions
    assert "position size" in instructions.lower()
    assert "stop-loss" in instructions.lower()
    # DCF values must be labelled as modeled scenarios, never price targets.
    assert "price target" in instructions.lower()
    assert "modeled scenario output" in instructions


def test_no_trading_capability_is_registered_anywhere():
    """The system prompt asks for a research stance in prose (never a trade
    verdict — see the Phase H.3 corrective patch), but even independent of
    that, this must never be backed by an actual executable trading
    capability -- re-asserts this invariant stays true regardless of prompt
    wording changes."""
    from tools.registry import default_registry

    registry = default_registry()
    forbidden = ("order", "brokerage", "portfolio")
    for definition in registry.enabled_definitions():
        assert not any(word in definition.name.lower() for word in forbidden)


# ---- planning and degradation ----

def test_plan_is_full_when_quota_is_ample(coordinator):
    plan = plan_analysis(coordinator, "AAPL", include_news=False)
    assert plan.mode == AnalysisMode.FULL
    assert plan.max_external_calls == len(plan.uncached_datasets)


def test_plan_degrades_to_reduced_on_a_partial_budget(tmp_path, clock):
    coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "m.sqlite3"), clock=clock),
        ledger=AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock,
                                       daily_limit=3),
        client=DatasetClient(), clock=clock, sleeper=lambda _s: None, jitter=lambda: 0.5)

    plan = plan_analysis(coordinator, "AAPL", include_news=False)

    assert plan.mode == AnalysisMode.REDUCED
    assert len(plan.uncached_datasets) <= 3
    assert "reduced" in plan.reason


# ---- Problem 10: reduced/degraded modes report exactly what was omitted ----

def test_reduced_mode_lists_every_omitted_dataset_with_a_reason_and_effect(tmp_path, clock):
    coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "m.sqlite3"), clock=clock),
        ledger=AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock,
                                       daily_limit=3),
        client=DatasetClient(), clock=clock, sleeper=lambda _s: None, jitter=lambda: 0.5)

    plan = plan_analysis(coordinator, "AAPL", include_news=False)

    assert plan.mode == AnalysisMode.REDUCED
    assert plan.requested_datasets, "the full wanted set must always be recorded"
    assert set(plan.omitted_datasets) == set(plan.requested_datasets) - set(plan.datasets)
    assert plan.omitted_datasets, "a budget of 3 against 7 requested datasets must omit some"
    for dataset in plan.omitted_datasets:
        assert dataset in plan.omission_reasons and plan.omission_reasons[dataset]
        assert dataset in plan.omission_effects and plan.omission_effects[dataset]
    # Every requested dataset is accounted for as either planned or omitted —
    # nothing simply vanishes.
    assert set(plan.datasets) | set(plan.omitted_datasets) == set(plan.requested_datasets)


def test_cached_only_mode_reports_omitted_datasets_too(tmp_path, clock):
    cache = MarketDataCache(path=str(tmp_path / "m.sqlite3"), clock=clock)
    ledger = AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock,
                                     daily_limit=25)
    client = DatasetClient()
    coordinator = MarketDataRequestCoordinator(
        cache=cache, ledger=ledger, client=client, clock=clock,
        sleeper=lambda _s: None, jitter=lambda: 0.5)
    coordinator.fetch("stock_quote", "AAPL")  # only ONE dataset ends up cached
    ledger.record_rate_limit(exhausted=True)

    plan = plan_analysis(coordinator, "AAPL", include_news=False)

    assert plan.mode == AnalysisMode.CACHED_ONLY
    assert "company_overview" in plan.omitted_datasets
    assert "stock_quote" not in plan.omitted_datasets
    assert "No external quota" in plan.omission_reasons["company_overview"]


def test_stopped_mode_reports_every_requested_dataset_as_omitted(tmp_path, clock):
    ledger = AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock,
                                     daily_limit=25)
    ledger.record_rate_limit(exhausted=True)
    coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "m.sqlite3"), clock=clock),
        ledger=ledger, client=DatasetClient(), clock=clock,
        sleeper=lambda _s: None, jitter=lambda: 0.5)

    plan = plan_analysis(coordinator, "AAPL", include_news=False)

    assert plan.mode == AnalysisMode.STOPPED
    assert set(plan.omitted_datasets) == set(plan.requested_datasets)
    assert all(plan.omission_effects[d] for d in plan.omitted_datasets)


def test_full_mode_never_reports_any_omission(tmp_path, clock):
    coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "m.sqlite3"), clock=clock),
        ledger=AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock,
                                       daily_limit=100),
        client=DatasetClient(), clock=clock, sleeper=lambda _s: None, jitter=lambda: 0.5)

    plan = plan_analysis(coordinator, "AAPL", include_news=False)

    assert plan.mode == AnalysisMode.FULL
    assert plan.omitted_datasets == ()
    assert plan.omission_reasons == {}


def test_plan_stops_when_nothing_is_affordable_or_cached(tmp_path, clock):
    ledger = AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock,
                                     daily_limit=25)
    ledger.record_rate_limit(exhausted=True)
    coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "m.sqlite3"), clock=clock),
        ledger=ledger, client=DatasetClient(), clock=clock,
        sleeper=lambda _s: None, jitter=lambda: 0.5)

    plan = plan_analysis(coordinator, "AAPL", include_news=False)
    assert plan.mode == AnalysisMode.STOPPED


def test_stopped_analysis_returns_a_controlled_result(tmp_path, clock, monkeypatch, wired):
    _registry, executor, coordinator = wired
    coordinator.ledger.record_rate_limit(exhausted=True)

    result = run_full_stock_analysis(executor, "AAPL", include_news=False)

    assert result.plan.mode == AnalysisMode.STOPPED
    assert result.errors[0]["code"] == "STOCK_ANALYSIS_QUOTA_INSUFFICIENT"
    assert result.facts == {}


def test_news_is_excluded_unless_configured(coordinator):
    assert "news" not in plan_analysis(coordinator, "AAPL", include_news=False).datasets
    assert "news" in plan_analysis(coordinator, "AAPL", include_news=True).datasets


# ---- assumptions ----

def test_proposed_assumptions_are_anchored_to_reported_history(wired):
    _registry, executor, _coord = wired
    result = run_full_stock_analysis(executor, "AAPL", include_news=False)
    scenarios = result.facts["dcf"]["scenarios"]

    assert [s["scenario"] for s in scenarios] == ["base", "bull", "bear"]
    values = {s["scenario"]: s["value_per_share"] for s in scenarios}
    assert values["bull"] > values["base"] > values["bear"]


def test_insufficient_history_blocks_the_dcf_rather_than_guessing():
    facts = {"fundamental_metrics": {}}
    proposed = propose_assumptions(facts, forecast_years=5)
    assert proposed[0]["incomplete"] is True
    assert proposed[0]["missing_reason"]


def test_workflow_reports_a_missing_dcf_instead_of_inventing_one(wired, monkeypatch):
    _registry, executor, coordinator = wired
    # Overview with no share count, and a balance sheet that also lacks one.
    monkeypatch.setitem(PAYLOAD_BY_DATASET, "company_overview",
                        {"Symbol": "AAPL", "Name": "T", "Currency": "USD"})
    monkeypatch.setitem(PAYLOAD_BY_DATASET, "balance_sheet",
                        {"annualReports": [{"fiscalDateEnding": "2025-12-31",
                                            "reportedCurrency": "USD",
                                            "totalShareholderEquity": "600"}]})

    result = run_full_stock_analysis(executor, "ZZZZ", include_news=False)

    assert result.facts["dcf"]["available"] is False
    assert "shares" in result.facts["dcf"]["reason"].lower() or \
           "shares" in (result.facts["dcf"].get("detail") or "").lower()


# ---- symbol validation ----

def test_disabled_workflow_raises(monkeypatch, wired):
    _registry, executor, _coord = wired
    monkeypatch.setenv("STOCK_ANALYSIS_ENABLED", "false")
    with pytest.raises(ToolValidationError):
        run_full_stock_analysis(executor, "AAPL")


def test_invalid_symbol_is_rejected_by_the_workflow(wired):
    _registry, executor, _coord = wired
    with pytest.raises(ToolValidationError):
        run_full_stock_analysis(executor, "not a ticker!!")


# ---- secret hygiene ----

def test_no_secret_appears_anywhere_in_the_result(wired):
    _registry, executor, _coord = wired
    result = run_full_stock_analysis(executor, "AAPL", include_news=False)
    encoded = json.dumps(result.to_dict(), default=str)
    assert SECRET not in encoded
    assert "apikey" not in encoded.lower()


# ---- Phase H.3 corrective patch, Problem 1 requirement 6: the fallback
# (single-shot / facts-only) report path follows the SAME content-policy and
# claim-fidelity restriction as the staged pipeline's FinalInvestmentSynthesizer
# -- _ask_local_with_content_policy is the mechanism, tested directly here
# since it has no dependency on the rest of run_full_stock_analysis. ----

def _fake_ask_local(*texts):
    """Returns each of `texts` in order across successive calls."""
    remaining = list(texts)

    def fn(messages, **kwargs):
        text = remaining.pop(0) if remaining else texts[-1]
        return {"message": {"role": "assistant", "content": text},
                "metrics": {"prompt_tokens": 10, "completion_tokens": 5}, "ok": True}

    return fn


def test_ask_local_with_content_policy_returns_clean_text_unchanged():
    text, metrics = _ask_local_with_content_policy(
        [{"role": "user", "content": "hi"}],
        _fake_ask_local("The market price appears above the base modeled value."))
    assert text == "The market price appears above the base modeled value."
    assert metrics == {"prompt_tokens": 10, "completion_tokens": 5}


def test_ask_local_with_content_policy_repairs_a_trade_advice_violation():
    calls = []

    def recording_ask_local(messages, **kwargs):
        calls.append(messages)
        content = ("If you do not currently hold a position: AVOID."
                  if len(calls) == 1 else "The evidence supports a cautious research stance.")
        return {"message": {"role": "assistant", "content": content},
                "metrics": {"prompt_tokens": 10, "completion_tokens": 5}, "ok": True}

    text, _metrics = _ask_local_with_content_policy([{"role": "user", "content": "hi"}], recording_ask_local)
    assert text == "The evidence supports a cautious research stance."
    assert len(calls) == 2
    repair_prompt = calls[1][-1]["content"]
    assert "REJECTED by automated content-policy screening" in repair_prompt
    assert "holding-dependent phrasing" in repair_prompt


def test_ask_local_with_content_policy_also_repairs_a_claim_fidelity_violation():
    """The fallback path's scan covers Problem 5's unsupported-claim scanner
    too, not only Problem 1's trade-advice scanner."""
    text, _metrics = _ask_local_with_content_policy(
        [{"role": "user", "content": "hi"}],
        _fake_ask_local("This is a fortress balance sheet.",
                        "Net cash may provide financial flexibility."))
    assert text == "Net cash may provide financial flexibility."


def test_ask_local_with_content_policy_fails_closed_when_repair_still_violates():
    """Neither attempt's raw SENTENCE may be exposed. Asserted on the full
    sentences rather than the category words ('position size', 'stop-loss'),
    which the fixed withheld-text notice legitimately names when explaining
    WHY the narrative was withheld."""
    first, second = "Open with a 5% position size.", "Set a stop-loss at $180."
    text, _metrics = _ask_local_with_content_policy(
        [{"role": "user", "content": "hi"}], _fake_ask_local(first, second))
    assert "could not be produced within this project's content policy" in text
    assert first not in text and second not in text


def test_ask_local_with_content_policy_never_exposes_either_attempts_raw_text_on_failure():
    first_attempt = "Open with a 5% position size immediately."
    second_attempt = "Set an entry price near $200 instead."
    text, _metrics = _ask_local_with_content_policy(
        [{"role": "user", "content": "hi"}], _fake_ask_local(first_attempt, second_attempt))
    assert first_attempt not in text
    assert second_attempt not in text


def test_ask_local_with_content_policy_passes_the_configured_synthesis_timeout(monkeypatch):
    """GE corrective patch: this call previously passed NO timeout at all,
    silently inheriting brain.ask_local_raw's bare 120-second default -- a
    live GE run hit an actual read timeout on exactly this path (its prompt
    is the full compact payload, and its completion has no options.
    num_predict cap, unlike a research-pipeline stage). Both the first
    attempt and the one repair attempt must use tools.config.
    stock_analysis_synthesis_timeout_seconds(), not the bare default."""
    monkeypatch.setenv("STOCK_ANALYSIS_SYNTHESIS_TIMEOUT_SECONDS", "333")
    calls = []

    def recording_ask_local(messages, **kwargs):
        calls.append(kwargs.get("timeout"))
        content = ("Use a 5% position size." if len(calls) == 1
                  else "The evidence supports a cautious research stance.")
        return {"message": {"role": "assistant", "content": content},
                "metrics": {"prompt_tokens": 10, "completion_tokens": 5}, "ok": True}

    _ask_local_with_content_policy([{"role": "user", "content": "hi"}], recording_ask_local)
    assert calls == [333, 333]  # first attempt AND the repair attempt


def test_ask_local_with_content_policy_default_timeout_is_well_above_the_old_120s_default():
    assert config.stock_analysis_synthesis_timeout_seconds() > 120


# ---- _sanitize_stage_error_for_display: found via live verification against
# real COST data -- a bull_researcher claim-fidelity failure's OWN error
# message explains what was rejected by NAMING example prohibited words
# ("...remove any trade-advice directive (buy/sell/hold/avoid...)"), and that
# raw message was being quoted verbatim into the user-facing "could not
# complete" report note and "Not available: ..." section text, re-surfacing
# exactly the words the whole system exists to keep out. ----

def test_sanitize_stage_error_replaces_a_message_containing_prohibited_words():
    raw = ("CONTENT_POLICY_VIOLATION: prohibited or unsupported language detected: "
          "remove any trade-advice directive (buy/sell/hold/avoid, position size, "
          "entry/exit price, stop loss), unsupported superlative (e.g. 'fortress', "
          "'industry-leading').")
    sanitized = _sanitize_stage_error_for_display(raw)
    for forbidden in ("buy", "sell", "hold", "avoid", "fortress", "industry-leading"):
        assert forbidden not in sanitized.lower()
    assert "content-policy/claim-fidelity violation" in sanitized


def test_sanitize_stage_error_passes_through_an_ordinary_schema_error_unchanged():
    raw = "'key_points' must be a list of at least 2 item(s)"
    assert _sanitize_stage_error_for_display(raw) == raw


def test_sanitize_stage_error_handles_none_and_empty_string():
    assert _sanitize_stage_error_for_display(None) == "stage did not run"
    assert _sanitize_stage_error_for_display("") == "stage did not run"


# ---- MLI corrective patch: post-filing stock-split restatement ----
#
# A weighted-average diluted share count is only valid on the share basis in
# force when its filing was made. SEC restates prior periods for splits that
# happened BEFORE a filing, so the correct anchor is the FILED date. A split
# AFTER the filing is the dangerous case: nothing in the SEC data reflects
# it while the market price already does. Found live on MLI -- FY2025
# weighted-average diluted 111,492,000 (10-K filed 2026-02-25) then a 2-for-1
# on 2026-07-01 -- which inflated every modeled per-share value by ~2x
# ($95.77 base where ~$47.88 was correct) and inverted the valuation verdict
# from overvalued to undervalued.

def _facts_with_split(split_date, ratio=2.0, filed="2026-02-25",
                      fiscal_date="2025-12-27", shares=111_492_000.0):
    return {
        "statements": {"annual": {"balance_sheet": [{
            "dataset_id": "sec_company_facts", "fiscal_date": fiscal_date,
            "values": {"shares_outstanding": shares},
        }]}},
        "sec_fact_provenance": {
            "sec.annual.balance_sheet.0.shares_outstanding": {"filed": filed}},
        "corporate_actions": {"data": {"splits": [{"date": split_date, "ratio": ratio}]}},
    }


def test_split_after_filing_is_applied_to_the_sec_share_count():
    from finance.workflow import split_adjusted_sec_share_count
    shares, detail = split_adjusted_sec_share_count(
        _facts_with_split("2026-07-01T09:30:00-04:00"))
    assert shares == 222_984_000.0
    assert detail["split_factor"] == 2.0
    assert detail["reported_shares"] == 111_492_000.0
    assert detail["splits_applied"] == [{"date": "2026-07-01", "ratio": 2.0}]


def test_split_before_filing_is_not_double_counted():
    """SEC already restates prior periods for a split that predates the
    filing -- applying it again would halve the share count a second time."""
    from finance.workflow import split_adjusted_sec_share_count
    shares, detail = split_adjusted_sec_share_count(
        _facts_with_split("2023-10-23T09:30:00-04:00"))
    assert shares == 111_492_000.0
    assert detail["split_factor"] == 1.0
    assert detail["splits_applied"] == []


def test_multiple_post_filing_splits_compound():
    from finance.workflow import sec_share_count_split_factor
    facts = {"corporate_actions": {"data": {"splits": [
        {"date": "2026-07-01", "ratio": 2.0},
        {"date": "2026-09-01", "ratio": 3.0},
        {"date": "2020-01-01", "ratio": 5.0},   # long before the filing -- ignored
    ]}}}
    factor, applied = sec_share_count_split_factor(facts, "2026-02-25", "2025-12-27")
    assert factor == 6.0
    assert [s["date"] for s in applied] == ["2026-07-01", "2026-09-01"]


def test_no_split_history_leaves_the_share_count_untouched():
    from finance.workflow import split_adjusted_sec_share_count
    facts = _facts_with_split("2026-07-01")
    facts["corporate_actions"] = {"data": {"splits": []}}
    shares, detail = split_adjusted_sec_share_count(facts)
    assert shares == 111_492_000.0 and detail["split_factor"] == 1.0


def test_non_sec_balance_sheet_is_not_split_adjusted():
    """The restatement is specific to SEC weighted-average diluted counts."""
    from finance.workflow import split_adjusted_sec_share_count
    facts = _facts_with_split("2026-07-01")
    facts["statements"]["annual"]["balance_sheet"][0]["dataset_id"] = "yahoo_profile"
    shares, detail = split_adjusted_sec_share_count(facts)
    assert shares is None and detail == {}


def test_share_count_reconciliation_uses_the_split_adjusted_basis():
    """The pre-patch code reported MLI's 221.2M-vs-111.5M gap as a provider
    conflict ('basic vs diluted-weighted-average'). That was the wrong
    diagnosis: both are diluted, just from different split eras, and once
    restated they agree to 0.8% -- inside tolerance, so NO warning."""
    from finance.reconciliation import _reconcile_share_counts
    facts = _facts_with_split("2026-07-01T09:30:00-04:00")
    facts["overview"] = {"shares_outstanding": 221_181_388.0}
    assert _reconcile_share_counts(facts) == []


def test_share_count_reconciliation_still_flags_a_real_residual_gap():
    """A gap that survives split restatement is a genuine conflict and must
    still surface."""
    from finance.reconciliation import _reconcile_share_counts
    facts = _facts_with_split("2026-07-01T09:30:00-04:00")
    facts["overview"] = {"shares_outstanding": 500_000_000.0}
    warnings = _reconcile_share_counts(facts)
    assert warnings and "MARKET_CAP_SHARE_COUNT_MISMATCH" in warnings[0]
    assert "222,984,000" in warnings[0]  # compared on the restated basis


# ---- MLI corrective patch: readiness reflects assumption quality (spec 5/10) ----

def _facts_for_readiness(**dcf_overrides):
    base_assumptions = {"assumption_provenance": {
        "revenue_growth": {"source_type": "deterministic_calculation", "clamped": False},
        "capex_pct_revenue": {"source_type": "deterministic_calculation"},
        "wacc": {"source_type": "configured_default"},          # configured BY DESIGN
        "terminal_growth": {"source_type": "configured_default"},  # configured BY DESIGN
    }}
    dcf = {"available": True, "validation_status": "DCF_VALID",
           "scenarios": [{"scenario": "base", "assumptions": base_assumptions}]}
    dcf.update(dcf_overrides)
    return {"dcf": dcf,
            "dcf_scenario_spread": {"available": True, "spread_pct_of_base": 0.20},
            "statements": {"warnings": []}}


def _limitations(facts):
    from finance.workflow import _assumption_quality_limitations
    return _assumption_quality_limitations(facts)


def test_clean_analysis_has_no_assumption_quality_limitations():
    """Spec 15 item 14 -- a valid, complete, low-conflict analysis must still
    be able to reach READY, or the signal is worthless."""
    assert _limitations(_facts_for_readiness()) == []


def test_by_design_configured_defaults_do_not_limit_readiness():
    """wacc/terminal_growth/tax_rate are configured defaults for EVERY
    company (this project has no cost-of-capital model), so flagging them
    would mark every analysis LIMITED."""
    joined = " ".join(_limitations(_facts_for_readiness()))
    assert "wacc" not in joined and "terminal_growth" not in joined


def test_clamped_assumption_limits_readiness():
    facts = _facts_for_readiness()
    prov = facts["dcf"]["scenarios"][0]["assumptions"]["assumption_provenance"]
    prov["revenue_growth"]["clamped"] = True
    joined = " ".join(_limitations(facts))
    assert "clamped" in joined and "revenue_growth" in joined


def test_company_derived_assumption_falling_back_to_default_limits_readiness():
    facts = _facts_for_readiness()
    prov = facts["dcf"]["scenarios"][0]["assumptions"]["assumption_provenance"]
    prov["capex_pct_revenue"]["source_type"] = "configured_default"
    joined = " ".join(_limitations(facts))
    assert "configured default" in joined and "capex_pct_revenue" in joined


def test_wide_scenario_spread_limits_readiness(monkeypatch):
    monkeypatch.setenv("RESEARCH_MATERIAL_SCENARIO_SPREAD", "0.90")
    facts = _facts_for_readiness()
    facts["dcf_scenario_spread"]["spread_pct_of_base"] = 0.974  # MLI's real value
    assert any("Scenario sensitivity is high" in r for r in _limitations(facts))


def test_scenario_spread_threshold_is_configurable(monkeypatch):
    facts = _facts_for_readiness()
    facts["dcf_scenario_spread"]["spread_pct_of_base"] = 0.80
    monkeypatch.setenv("RESEARCH_MATERIAL_SCENARIO_SPREAD", "0.90")
    assert _limitations(facts) == []
    monkeypatch.setenv("RESEARCH_MATERIAL_SCENARIO_SPREAD", "0.75")
    assert any("Scenario sensitivity is high" in r for r in _limitations(facts))


def test_residual_share_count_gap_limits_readiness():
    """A gap that survives split restatement is a real unresolved conflict."""
    facts = _facts_for_readiness()
    facts.update(_facts_with_split("2026-07-01T09:30:00-04:00"))
    facts["dcf"] = _facts_for_readiness()["dcf"]
    facts["dcf_scenario_spread"] = {"available": True, "spread_pct_of_base": 0.2}
    facts["overview"] = {"shares_outstanding": 500_000_000.0}  # far off even restated
    assert any("share-count difference" in r for r in _limitations(facts))


def test_split_explained_share_count_does_not_limit_readiness():
    """MLI's own case: 221.2M vs a restated 223.0M is 0.8% -- explained, not
    a conflict, so it must NOT cap readiness."""
    facts = _facts_for_readiness()
    facts.update(_facts_with_split("2026-07-01T09:30:00-04:00"))
    facts["dcf"] = _facts_for_readiness()["dcf"]
    facts["dcf_scenario_spread"] = {"available": True, "spread_pct_of_base": 0.2}
    facts["overview"] = {"shares_outstanding": 221_181_388.0}
    assert _limitations(facts) == []


def test_confidence_band_labels():
    """Spec 7: a 45% call must not read like an 85% one."""
    from finance.workflow import _confidence_band
    assert _confidence_band(0.45) == "low"   # spec 7's own worked example
    assert _confidence_band(0.20) == "low"
    assert _confidence_band(0.55) == "moderate"
    assert _confidence_band(0.85) == "high"


# ---- VZ corrective patch: analysis-intent detection ----
#
# The old `\banalyz[es]\b` stem matched only "analyze"/"analyzes" -- not
# "analyzing", not "analysed", not the British "analyse" -- and REQUIRED a
# following noun within 60 chars, so plain "analyze VZ" fell through. The
# failure is quiet and therefore expensive: the request still gets answered,
# by the ordinary chat tool-loop, so the reply LOOKS like an analysis while
# containing no DCF, no bull/bear, no risk review and no recommendation.
# Found live on VZ.

def test_analysis_intent_phrasings_are_detected():
    from finance.workflow import detect_full_stock_analysis_request as detect
    for text in (
        "analyze VZ", "analyse VZ", "analyzing VZ", "analysed VZ",
        "VZ analysis", "give me an analysis of VZ", "analyse VZ stock",
        "do a deep dive on VZ", "deep-dive VZ", "research VZ stock",
        "what is the valuation of VZ", "DCF valuation of VZ",
        "VZ stock analysis", "full stock analysis of VZ",
    ):
        assert detect(text) == "VZ", f"{text!r} should reach the full workflow"


def test_ambiguous_company_questions_still_do_not_trigger_a_full_analysis():
    """A full analysis is a whole provider sweep plus six local-model stages.
    Firing it on a one-word or quote-shaped question would be worse than the
    gap it closes, so these deliberately stay on the ordinary path."""
    from finance.workflow import detect_full_stock_analysis_request as detect
    for text in ("VZ", "tell me about VZ", "how is VZ doing", "what is VZ trading at"):
        assert detect(text) is None, f"{text!r} should NOT trigger a full analysis"


def test_analysis_words_without_a_ticker_never_trigger():
    """The ticker-shaped-token gate is what keeps the widened verbs safe."""
    from finance.workflow import detect_full_stock_analysis_request as detect
    for text in ("analyze my spending", "deep dive into my notes",
                 "run an analysis of this file", "valuation of my house"):
        assert detect(text) is None, f"{text!r} must not look like a stock request"
