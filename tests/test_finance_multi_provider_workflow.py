"""Phase H.3 — finance/workflow.py's multi-provider integration: dataset-
specific routing (tools.config.finance_*_provider()), Yahoo/SEC gathering
merged into the SAME plan/omission-transparency machinery Alpha Vantage
already had, and the DCF/metrics pipeline running unchanged on
provider-agnostic facts.

Uses fake CLIENTS wired into real MarketDataRequestCoordinator instances
(the SAME pattern tests/test_finance_workflow.py already uses for Alpha
Vantage), injected via tools.finance_tools.set_coordinator/
set_yahoo_coordinator/set_sec_coordinator, through the REAL ToolRegistry ->
ToolExecutor -> finance.workflow.run_full_stock_analysis path.
"""

import json

import pytest

import tools.config as config
import tools.finance_tools as finance_tools
from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.provider import ProviderResponse
from finance.quota import AlphaVantageQuotaLedger, QuotaLedger
from finance.sec_datasets import resolve_sec_dataset
from finance.workflow import AnalysisMode, run_full_stock_analysis
from finance.yahoo_datasets import resolve_yahoo_dataset
from tests.test_finance_cache import FakeClock
from tests.test_finance_workflow import BALANCE, CASHFLOW, EARNINGS, INCOME, OVERVIEW, PRICES, QUOTE
from tools.base import ToolFailure
from tools.executor import ToolExecutor
from tools.models import MARKET_DATA_INVALID_SYMBOL, SEC_CIK_NOT_FOUND
from tools.registry import ToolRegistry

YAHOO_QUOTE_PAYLOAD = {"quote": {"lastPrice": 311.0, "previousClose": 309.5, "currency": "USD",
                                 "exchange": "NMS", "marketCap": 4_500_000_000_000,
                                 "open": 310.0, "dayHigh": 312.0, "dayLow": 308.0, "lastVolume": 1000}}
YAHOO_PROFILE_PAYLOAD = {"profile": {"symbol": "TEST", "shortName": "Test Corp", "sector": "Technology",
                                     "industry": "Software", "country": "United States",
                                     "fullExchangeName": "NasdaqGS", "currency": "USD",
                                     "marketCap": 4_500_000_000_000, "sharesOutstanding": 100.0}}
YAHOO_PRICE_HISTORY_PAYLOAD = {"bars": [
    {"Date": "2026-08-01T00:00:00-04:00", "Open": 300.0, "High": 305.0, "Low": 299.0,
     "Close": 304.0, "Adj Close": 304.0, "Volume": 1000, "Dividends": 0.0, "Stock Splits": 0.0},
    {"Date": "2026-08-02T00:00:00-04:00", "Open": 304.0, "High": 309.0, "Low": 303.0,
     "Close": 308.0, "Adj Close": 308.0, "Volume": 1100, "Dividends": 0.0, "Stock Splits": 0.0},
], "period": "1y", "interval": "1d"}
YAHOO_ACTIONS_PAYLOAD = {"dividends": [{"date": "2026-02-01", "amount": 0.25}], "splits": []}
YAHOO_ESTIMATES_PAYLOAD = {"analyst_price_targets": {"current": 320.0, "high": 400.0, "low": 250.0}}

SEC_TICKER_MAP_PAYLOAD = {"0": {"cik_str": 1000000, "ticker": "TEST", "title": "Test Corp"}}


def _two_year_fact(concept_val_2025, concept_val_2024, unit="USD", has_start=True):
    def fact(val, fy, filed, start=None, end=None):
        f = {"val": val, "accn": f"0001-{fy}-000001", "fy": fy, "fp": "FY",
            "form": "10-K", "filed": filed, "end": end or f"{fy}-12-31"}
        if has_start:
            f["start"] = start or f"{fy}-01-01"
        return f
    return {"units": {unit: [
        fact(concept_val_2025, 2025, "2026-02-01"),
        fact(concept_val_2024, 2024, "2025-02-01"),
    ]}}


# Two fiscal years so revenue_growth_yoy is computable (propose_assumptions
# correctly refuses to propose DCF assumptions from a single period alone —
# omitting OperatingIncomeLoss or the prior year here would make the DCF
# legitimately unavailable, not a bug).
SEC_COMPANY_FACTS_PAYLOAD = {"facts": {"us-gaap": {
    "RevenueFromContractWithCustomerExcludingAssessedTax":
        _two_year_fact(100_000_000_000, 90_000_000_000),
    "OperatingIncomeLoss": _two_year_fact(25_000_000_000, 22_000_000_000),
    "NetIncomeLoss": _two_year_fact(20_000_000_000, 18_000_000_000),
    "CashAndCashEquivalentsAtCarryingValue": _two_year_fact(10_000_000_000, 9_000_000_000, has_start=False),
    "LongTermDebtNoncurrent": _two_year_fact(5_000_000_000, 5_500_000_000, has_start=False),
    "Assets": _two_year_fact(200_000_000_000, 190_000_000_000, has_start=False),
    "StockholdersEquity": _two_year_fact(80_000_000_000, 72_000_000_000, has_start=False),
    "WeightedAverageNumberOfDilutedSharesOutstanding":
        _two_year_fact(100.0, 101.0, unit="shares"),
    "NetCashProvidedByUsedInOperatingActivities": _two_year_fact(25_000_000_000, 23_000_000_000),
}}}


class FakeYahooClient:
    provider_id = "yahoo"

    def __init__(self, fail_dataset_ids=frozenset()):
        self._fail = fail_dataset_ids
        self.calls = []

    def fetch(self, dataset, arguments):
        self.calls.append(dataset.dataset_id)
        if dataset.dataset_id in self._fail:
            raise ToolFailure(MARKET_DATA_INVALID_SYMBOL, f"no data for {dataset.dataset_id}")
        payload = {
            "stock_quote": YAHOO_QUOTE_PAYLOAD, "company_profile": YAHOO_PROFILE_PAYLOAD,
            "price_history": YAHOO_PRICE_HISTORY_PAYLOAD, "corporate_actions": YAHOO_ACTIONS_PAYLOAD,
            "analyst_estimates": YAHOO_ESTIMATES_PAYLOAD,
        }[dataset.dataset_id]
        return ProviderResponse(payload=payload, provider_metadata={}, byte_count=len(json.dumps(payload)))


class FakeSecClient:
    provider_id = "sec"

    def __init__(self, fail=False):
        self._fail = fail
        self.calls = []

    def fetch(self, dataset, arguments):
        self.calls.append(dataset.dataset_id)
        if self._fail:
            raise ToolFailure(SEC_CIK_NOT_FOUND, "no CIK mapping for this ticker")
        payload = {"ticker_cik_map": SEC_TICKER_MAP_PAYLOAD,
                  "company_facts": SEC_COMPANY_FACTS_PAYLOAD}[dataset.dataset_id]
        return ProviderResponse(payload=payload, provider_metadata={}, byte_count=len(json.dumps(payload)))


class AVClient:
    """Serves canned Alpha Vantage payloads for whatever this project's
    existing test_finance_workflow.py fixtures already define."""

    def __init__(self):
        self.calls = []

    def fetch(self, dataset, arguments):
        self.calls.append(dataset.dataset_id)
        payload = {"stock_quote": QUOTE, "company_overview": OVERVIEW, "income_statement": INCOME,
                  "balance_sheet": BALANCE, "cash_flow": CASHFLOW, "earnings": EARNINGS,
                  "daily_prices": PRICES}.get(dataset.dataset_id, {"ok": True})
        return ProviderResponse(payload=payload, provider_metadata={}, byte_count=len(json.dumps(payload)))


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def wired(tmp_path, clock, monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "TEST_KEY_NOT_REAL")
    monkeypatch.setenv("YAHOO_FINANCE_ENABLED", "true")
    monkeypatch.setenv("YAHOO_PERSONAL_USE_ACKNOWLEDGED", "true")
    monkeypatch.setenv("SEC_EDGAR_ENABLED", "true")
    monkeypatch.setenv("SEC_USER_AGENT", "TestApp/1.0 (contact: t@example.com)")
    # Restore the DEFAULT (non-alphavantage) provider routing this whole file
    # is about -- the shared conftest.py autouse fixture pins everything to
    # "alphavantage" for every OTHER test file; this one explicitly opts back
    # into the real Phase H.3 defaults.
    for var in ("FINANCE_QUOTE_PROVIDER", "FINANCE_PRICE_HISTORY_PROVIDER",
               "FINANCE_CORPORATE_ACTIONS_PROVIDER", "FINANCE_COMPANY_PROFILE_PROVIDER",
               "FINANCE_ANALYST_ESTIMATES_PROVIDER"):
        monkeypatch.setenv(var, "yahoo")
    monkeypatch.setenv("FINANCE_US_FUNDAMENTALS_PROVIDER", "sec")
    monkeypatch.setenv("FINANCE_NEWS_PROVIDER", "alphavantage")

    av_client = AVClient()
    av_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "av_m.sqlite3"), clock=clock),
        ledger=AlphaVantageQuotaLedger(path=str(tmp_path / "av_q.sqlite3"), clock=clock, daily_limit=100),
        client=av_client, clock=clock, sleeper=lambda _s: None, jitter=lambda: 0.5)
    finance_tools.set_coordinator(av_coordinator)

    yahoo_client = FakeYahooClient()
    yahoo_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "yahoo_m.sqlite3"), clock=clock),
        ledger=QuotaLedger(path=str(tmp_path / "yahoo_q.sqlite3"), clock=clock, daily_limit=1_000_000),
        client=yahoo_client, provider_id="yahoo", dataset_resolver=resolve_yahoo_dataset,
        clock=clock, sleeper=lambda _s: None, min_request_interval_ms=0)
    finance_tools.set_yahoo_coordinator(yahoo_coordinator)

    sec_client = FakeSecClient()
    sec_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "sec_m.sqlite3"), clock=clock),
        ledger=QuotaLedger(path=str(tmp_path / "sec_q.sqlite3"), clock=clock, daily_limit=1_000_000),
        client=sec_client, provider_id="sec", dataset_resolver=resolve_sec_dataset,
        clock=clock, sleeper=lambda _s: None, min_request_interval_ms=0)
    finance_tools.set_sec_coordinator(sec_coordinator)

    registry = ToolRegistry()
    for tool_cls in finance_tools.ALL_FINANCE_TOOL_CLASSES:
        registry.register(tool_cls())
    executor = ToolExecutor(registry)

    yield executor, av_client, yahoo_client, sec_client, av_coordinator
    finance_tools.set_coordinator(None)
    finance_tools.set_yahoo_coordinator(None)
    finance_tools.set_sec_coordinator(None)


def run(wired, symbol="TEST", **kwargs):
    executor, *_rest = wired
    return run_full_stock_analysis(executor, symbol, include_news=False, **kwargs)


# ---- default routing: each capability actually goes to its configured provider ----

def test_full_run_routes_each_capability_to_its_configured_provider(wired):
    result = run(wired)
    assert result.plan.mode == AnalysisMode.FULL
    prov = result.facts["data_provenance"]
    assert prov["quote"]["provider"] == "yahoo"
    assert prov["company_profile"]["provider"] == "yahoo"
    assert prov["price_history"]["provider"] == "yahoo"
    assert prov["corporate_actions"]["provider"] == "yahoo"
    assert prov["analyst_estimates"]["provider"] == "yahoo"
    assert prov["income_statement"]["provider"] == "sec"
    assert prov["balance_sheet"]["provider"] == "sec"
    assert prov["cash_flow"]["provider"] == "sec"
    assert prov["earnings"]["provider"] == "alphavantage"


def test_facts_are_correctly_shaped_regardless_of_source_provider(wired):
    result = run(wired)
    assert result.facts["quote"]["available"] is True
    assert result.facts["quote"]["price"] == 311.0
    assert result.facts["overview"]["name"] == "Test Corp"
    bs = result.facts["statements"]["annual"]["balance_sheet"][0]
    assert bs["dataset_id"] == "sec_company_facts"
    assert bs["values"]["cash_and_cash_equivalents"] == 10_000_000_000.0
    assert bs["values"]["total_debt"] == 5_000_000_000.0  # only long-term component known


def test_dcf_runs_successfully_on_sec_plus_yahoo_sourced_facts(wired):
    result = run(wired)
    assert result.facts["dcf"]["available"] is True
    base = next(s for s in result.facts["dcf"]["scenarios"] if s["scenario"] == "base")
    assert base["value_per_share"] > 0


def test_only_finance_dot_sec_and_yahoo_tools_were_actually_called(wired):
    _executor, av_client, yahoo_client, sec_client, _coord = wired
    run(wired)
    assert set(yahoo_client.calls) == {"stock_quote", "company_profile", "price_history",
                                       "corporate_actions", "analyst_estimates"}
    # `company_submissions` joins the set in Phase H.4: current management
    # guidance lives in the earnings-release exhibit attached to an
    # item-2.02 8-K, and the submissions index is how that filing is found.
    # This fixture's submissions payload lists no 8-K, so no filing_index /
    # filing_document call follows — the guidance path is skipped cleanly
    # rather than failing, which is the section-20 "no guidance" behaviour.
    assert set(sec_client.calls) == {"ticker_cik_map", "company_facts", "company_submissions"}
    assert av_client.calls == ["earnings"]  # AV's only remaining job under default routing


# ---- the STOPPED-early-return bug: AV quota exhausted must not abort Yahoo/SEC ----

def test_alpha_vantage_quota_exhaustion_degrades_to_reduced_not_stopped(wired):
    """The exact bug found live this session: run_full_stock_analysis used to
    return early on plan.mode == STOPPED before Yahoo/SEC were even
    attempted. Alpha Vantage's quota (used only for "earnings" by default)
    being exhausted must not abort an analysis Yahoo/SEC can still serve."""
    _executor, _av, _yahoo, _sec, av_coordinator = wired
    av_coordinator.ledger.record_rate_limit(exhausted=True)

    result = run(wired)

    assert result.plan.mode == AnalysisMode.REDUCED
    assert "earnings" in result.plan.omitted_datasets
    assert result.facts, "facts must still be built from Yahoo/SEC even though AV is exhausted"
    assert result.facts["quote"]["available"] is True
    assert result.facts["dcf"]["available"] is True
    assert "earnings" not in result.facts


def test_truly_nothing_available_from_any_provider_is_still_stopped(tmp_path, clock, monkeypatch):
    """The OTHER half of the same fix: STOPPED must still fire, correctly,
    when NO provider can supply anything -- it just must not fire simply
    because Alpha Vantage alone had nothing."""
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "TEST_KEY_NOT_REAL")
    monkeypatch.setenv("YAHOO_FINANCE_ENABLED", "false")
    monkeypatch.setenv("SEC_EDGAR_ENABLED", "false")
    for var in ("FINANCE_QUOTE_PROVIDER", "FINANCE_PRICE_HISTORY_PROVIDER",
               "FINANCE_CORPORATE_ACTIONS_PROVIDER", "FINANCE_COMPANY_PROFILE_PROVIDER",
               "FINANCE_ANALYST_ESTIMATES_PROVIDER", "FINANCE_US_FUNDAMENTALS_PROVIDER"):
        monkeypatch.setenv(var, "alphavantage")

    av_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "m.sqlite3"), clock=clock),
        ledger=AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock, daily_limit=100),
        client=AVClient(), clock=clock, sleeper=lambda _s: None, jitter=lambda: 0.5)
    av_coordinator.ledger.record_rate_limit(exhausted=True)
    finance_tools.set_coordinator(av_coordinator)
    registry = ToolRegistry()
    for tool_cls in finance_tools.ALL_FINANCE_TOOL_CLASSES:
        registry.register(tool_cls())
    executor = ToolExecutor(registry)

    result = run_full_stock_analysis(executor, "TEST", include_news=False)
    finance_tools.set_coordinator(None)

    assert result.plan.mode == AnalysisMode.STOPPED
    assert result.facts == {}
    assert result.errors[0]["code"] == "STOCK_ANALYSIS_QUOTA_INSUFFICIENT"


# ---- Yahoo/SEC failure modes degrade gracefully with transparent omissions ----

def test_yahoo_not_acknowledged_omits_only_yahoo_capabilities(wired, monkeypatch):
    monkeypatch.setenv("YAHOO_PERSONAL_USE_ACKNOWLEDGED", "false")
    result = run(wired)
    assert result.plan.mode == AnalysisMode.REDUCED
    for capability in ("quote", "price_history", "corporate_actions",
                      "company_profile", "analyst_estimates"):
        assert capability in result.plan.omitted_datasets
        assert "acknowledg" in result.plan.omission_reasons[capability].lower() or \
              "disabled" in result.plan.omission_reasons[capability].lower()
    # SEC and Alpha Vantage are unaffected.
    assert result.facts["statements"]["annual"]["balance_sheet"][0]["dataset_id"] == "sec_company_facts"
    assert "earnings" in result.facts


def test_sec_cik_not_found_omits_fundamentals_with_a_clear_reason(wired):
    _executor, _av, _yahoo, sec_client, _coord = wired
    sec_client._fail = True

    result = run(wired)

    assert "us_fundamentals" in result.plan.omitted_datasets
    assert "CIK" in result.plan.omission_reasons["us_fundamentals"]
    assert result.facts["dcf"]["available"] is False
    # Yahoo (quote/profile/price/etc.) is unaffected by SEC's failure.
    assert result.facts["quote"]["available"] is True


def test_sec_disabled_entirely_omits_fundamentals(wired, monkeypatch):
    monkeypatch.setenv("SEC_EDGAR_ENABLED", "false")
    result = run(wired)
    assert "us_fundamentals" in result.plan.omitted_datasets
    assert result.facts["dcf"]["available"] is False


# ---- cross-provider reconciliation fires on real divergent data ----

def test_share_count_reconciliation_warning_appears_when_sources_diverge(wired):
    # Yahoo's fixture reports 100 shares; SEC's fixture ALSO reports 100 --
    # equal, so no warning. Confirm the wiring by checking the warning is
    # ABSENT here (the live AAPL run already proved the divergent case; see
    # test_finance_reconciliation.py for the isolated divergence logic).
    result = run(wired)
    assert not any("MARKET_CAP_SHARE_COUNT_MISMATCH" in w for w in result.warnings)


def test_share_count_reconciliation_warning_fires_on_real_divergence(wired, monkeypatch, tmp_path, clock):
    divergent_profile = {"profile": dict(YAHOO_PROFILE_PAYLOAD["profile"], sharesOutstanding=150.0)}

    class DivergentYahooClient(FakeYahooClient):
        def fetch(self, dataset, arguments):
            if dataset.dataset_id == "company_profile":
                self.calls.append(dataset.dataset_id)
                return ProviderResponse(payload=divergent_profile, provider_metadata={},
                                        byte_count=len(json.dumps(divergent_profile)))
            return super().fetch(dataset, arguments)

    executor, *_rest = wired
    # Swap in a coordinator with the divergent client for this one test --
    # own explicit tmp_path cache/ledger, never the real default path.
    yahoo_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "divergent_yahoo_m.sqlite3"), clock=clock),
        ledger=QuotaLedger(path=str(tmp_path / "divergent_yahoo_q.sqlite3"), clock=clock,
                          daily_limit=1_000_000),
        client=DivergentYahooClient(), provider_id="yahoo", dataset_resolver=resolve_yahoo_dataset,
        clock=clock, min_request_interval_ms=0)
    finance_tools.set_yahoo_coordinator(yahoo_coordinator)

    result = run_full_stock_analysis(executor, "TEST", include_news=False)

    assert any("MARKET_CAP_SHARE_COUNT_MISMATCH" in w for w in result.warnings)
