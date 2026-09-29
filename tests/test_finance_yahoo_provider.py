"""Phase H.3 — finance/yahoo_provider.py: the Yahoo Finance (yfinance) client.

Uses a FAKE Ticker-like object injected via `ticker_factory` (no real network,
no real yfinance import needed) so the client's own logic — payload shaping,
invalid-symbol detection, error classification — is tested in isolation. See
tests/test_finance_multi_provider_workflow.py for the real, end-to-end proof
through run_full_stock_analysis, and the manual live verification in
docs/security/YAHOO_SEC_PROVIDER_REVIEW.md for real-network confirmation.
"""

import math

import pandas as pd
import pytest

from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.quota import QuotaLedger
from finance.yahoo_datasets import resolve_yahoo_dataset
from finance.yahoo_provider import YahooFinanceClient, _jsonable
from tools.base import ToolFailure
from tools.models import (
    MARKET_DATA_INVALID_SYMBOL,
    MARKET_DATA_PROVIDER_ERROR,
    MARKET_DATA_RATE_LIMITED,
    MARKET_DATA_YAHOO_SCHEMA_CHANGED,
)


class FakeFastInfo(dict):
    """yfinance's fast_info supports __getitem__ and raises KeyError for an
    absent field -- a plain dict already behaves this way."""


class FakeTicker:
    def __init__(self, fast_info=None, info=None, history_df=None, dividends=None,
                splits=None, analyst_price_targets=None, raise_on=None):
        self.fast_info = FakeFastInfo(fast_info or {})
        self.info = info or {}
        self._history_df = history_df
        self.dividends = dividends if dividends is not None else pd.Series(dtype=float)
        self.splits = splits if splits is not None else pd.Series(dtype=float)
        self.analyst_price_targets = analyst_price_targets or {}
        self._raise_on = raise_on or {}

    def history(self, period=None, interval=None, auto_adjust=None):
        if "history" in self._raise_on:
            raise self._raise_on["history"]
        return self._history_df if self._history_df is not None else pd.DataFrame()


def _client_for(ticker):
    return YahooFinanceClient(ticker_factory=lambda symbol: ticker)


def _dataset(name):
    return resolve_yahoo_dataset(name)


# ---- quote (fast_info) ----

def test_quote_happy_path_returns_jsonable_dict():
    ticker = FakeTicker(fast_info={"lastPrice": 311.0, "previousClose": 309.5,
                                   "currency": "USD", "marketCap": 4_500_000_000_000})
    client = _client_for(ticker)
    result = client.fetch(_dataset("stock_quote"), {"symbol": "AAPL"})
    assert result.payload["quote"]["lastPrice"] == 311.0
    assert result.payload["quote"]["currency"] == "USD"


def test_quote_missing_fields_default_to_none_not_a_crash():
    ticker = FakeTicker(fast_info={"lastPrice": 100.0})  # only one field populated
    client = _client_for(ticker)
    result = client.fetch(_dataset("stock_quote"), {"symbol": "X"})
    assert result.payload["quote"]["lastPrice"] == 100.0
    assert result.payload["quote"]["marketCap"] is None


def test_quote_with_no_lastprice_is_invalid_symbol_not_a_fabricated_empty_quote():
    """The exact bug found and fixed live this session: fast_info does NOT
    raise for an invalid/delisted ticker, it silently returns an all-None
    object. lastPrice absent is the real signal."""
    ticker = FakeTicker(fast_info={})
    client = _client_for(ticker)
    with pytest.raises(ToolFailure) as excinfo:
        client.fetch(_dataset("stock_quote"), {"symbol": "ZZZZZZZ"})
    assert excinfo.value.code == MARKET_DATA_INVALID_SYMBOL


def test_quote_nan_values_are_converted_to_none():
    ticker = FakeTicker(fast_info={"lastPrice": 100.0, "dayHigh": float("nan")})
    client = _client_for(ticker)
    result = client.fetch(_dataset("stock_quote"), {"symbol": "X"})
    assert result.payload["quote"]["dayHigh"] is None


# ---- company profile (.info) ----

def test_profile_happy_path_uses_only_curated_fields():
    ticker = FakeTicker(info={"symbol": "AAPL", "shortName": "Apple Inc.", "sector": "Technology",
                              "someRandomUnreviewedField": "should not appear"})
    client = _client_for(ticker)
    result = client.fetch(_dataset("company_profile"), {"symbol": "AAPL"})
    profile = result.payload["profile"]
    assert profile["shortName"] == "Apple Inc."
    assert "someRandomUnreviewedField" not in profile


def test_profile_with_no_symbol_or_name_is_invalid_symbol():
    ticker = FakeTicker(info={})
    client = _client_for(ticker)
    with pytest.raises(ToolFailure) as excinfo:
        client.fetch(_dataset("company_profile"), {"symbol": "ZZZZZZZ"})
    assert excinfo.value.code == MARKET_DATA_INVALID_SYMBOL


# ---- price history ----

def test_price_history_converts_dataframe_to_jsonable_records():
    df = pd.DataFrame({
        "Open": [100.0, 101.0], "High": [102.0, 103.0], "Low": [99.0, 100.0],
        "Close": [101.0, 102.0], "Adj Close": [101.0, 102.0], "Volume": [1000, 1100],
        "Dividends": [0.0, 0.0], "Stock Splits": [0.0, 0.0],
    }, index=pd.to_datetime(["2026-01-01", "2026-01-02"]).tz_localize("America/New_York"))
    df.index.name = "Date"
    ticker = FakeTicker(history_df=df)
    client = _client_for(ticker)
    result = client.fetch(_dataset("price_history"), {"symbol": "AAPL", "period": "5d", "interval": "1d"})
    bars = result.payload["bars"]
    assert len(bars) == 2
    assert bars[0]["Close"] == 101.0
    assert isinstance(bars[0]["Date"], str)  # Timestamp -> ISO string, never a raw pandas object


def test_price_history_empty_dataframe_is_not_a_hard_error():
    """For a genuinely INVALID ticker, yfinance's .history() raises
    YFPricesMissingError (see test_ticker_missing_error_is_classified_as_
    invalid_symbol) -- confirmed live this session, not assumed. An empty
    DataFrame with no exception (e.g. a valid ticker queried over a range
    with no sessions) is a different, non-error case: the provider layer
    returns it as-is, and finance/normalization.py::normalize_yahoo_price_
    history (already downstream-tested) is what correctly reports
    available=False -- the provider layer must not double-guess that."""
    ticker = FakeTicker(history_df=pd.DataFrame())
    client = _client_for(ticker)
    result = client.fetch(_dataset("price_history"), {"symbol": "AAPL", "period": "1y", "interval": "1d"})
    assert result.payload["bars"] == []


# ---- corporate actions ----

def test_corporate_actions_converts_series_to_records():
    dividends = pd.Series([0.24, 0.25], index=pd.to_datetime(["2025-01-01", "2025-04-01"]))
    ticker = FakeTicker(dividends=dividends)
    client = _client_for(ticker)
    result = client.fetch(_dataset("corporate_actions"), {"symbol": "AAPL"})
    assert len(result.payload["dividends"]) == 2
    assert result.payload["dividends"][0]["amount"] == 0.24
    assert result.payload["splits"] == []


def test_corporate_actions_with_no_history_is_a_valid_empty_result():
    """A real company that has simply never paid a dividend or split its
    stock legitimately has empty dividends AND splits -- this is a valid,
    complete result, not an invalid-symbol signal (unlike fast_info's
    lastPrice or .info's symbol/shortName, which every real active ticker
    always has)."""
    ticker = FakeTicker()  # empty dividends AND splits
    client = _client_for(ticker)
    result = client.fetch(_dataset("corporate_actions"), {"symbol": "AAPL"})
    assert result.payload == {"dividends": [], "splits": []}


# ---- analyst estimates ----

def test_analyst_estimates_happy_path():
    ticker = FakeTicker(analyst_price_targets={"current": 310.0, "high": 400.0, "low": 250.0})
    client = _client_for(ticker)
    result = client.fetch(_dataset("analyst_estimates"), {"symbol": "AAPL"})
    assert result.payload["analyst_price_targets"]["high"] == 400.0


# ---- error classification ----

def test_rate_limit_error_is_classified_and_retryable():
    from yfinance.exceptions import YFRateLimitError

    ticker = FakeTicker(raise_on={"history": YFRateLimitError()})
    client = _client_for(ticker)
    with pytest.raises(ToolFailure) as excinfo:
        client.fetch(_dataset("price_history"), {"symbol": "AAPL", "period": "1y", "interval": "1d"})
    assert excinfo.value.code == MARKET_DATA_RATE_LIMITED
    assert excinfo.value.retryable is True


def test_ticker_missing_error_is_classified_as_invalid_symbol():
    from yfinance.exceptions import YFPricesMissingError

    ticker = FakeTicker(raise_on={"history": YFPricesMissingError("ZZZZZZZ", "")})
    client = _client_for(ticker)
    with pytest.raises(ToolFailure) as excinfo:
        client.fetch(_dataset("price_history"), {"symbol": "ZZZZZZZ", "period": "1y", "interval": "1d"})
    assert excinfo.value.code == MARKET_DATA_INVALID_SYMBOL


def test_unexpected_keyerror_is_classified_as_schema_changed():
    """The concrete, live version of "Yahoo response schema change" -- an
    internal parsing failure that is NOT a known yfinance exception type."""
    class ExplodingTicker(FakeTicker):
        def history(self, period=None, interval=None, auto_adjust=None):
            raise KeyError("unexpected_field")

    client = _client_for(ExplodingTicker())
    with pytest.raises(ToolFailure) as excinfo:
        client.fetch(_dataset("price_history"), {"symbol": "AAPL", "period": "1y", "interval": "1d"})
    assert excinfo.value.code == MARKET_DATA_YAHOO_SCHEMA_CHANGED


def test_generic_exception_is_classified_as_provider_error_and_retryable():
    class ExplodingTicker(FakeTicker):
        def history(self, period=None, interval=None, auto_adjust=None):
            raise ConnectionError("network blip")

    client = _client_for(ExplodingTicker())
    with pytest.raises(ToolFailure) as excinfo:
        client.fetch(_dataset("price_history"), {"symbol": "AAPL", "period": "1y", "interval": "1d"})
    assert excinfo.value.code == MARKET_DATA_PROVIDER_ERROR
    assert excinfo.value.retryable is True


# ---- structural: no raw pandas/numpy objects ever escape ----

def test_jsonable_strips_numpy_and_pandas_types():
    import numpy as np

    assert _jsonable(np.float64(1.5)) == 1.5
    assert _jsonable(np.int64(5)) == 5
    assert _jsonable(float("nan")) is None
    assert _jsonable(pd.Timestamp("2026-01-01")) == "2026-01-01T00:00:00"
    nested = _jsonable({"a": np.float64(1.0), "b": [np.int64(2), None]})
    assert nested == {"a": 1.0, "b": [2, None]}


# ---- through the shared coordinator: cache reuse, single-flight, provider identity ----

def test_fetch_through_coordinator_reuses_cache_on_second_call(tmp_path):
    ticker = FakeTicker(fast_info={"lastPrice": 311.0, "previousClose": 309.5})
    coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "m.sqlite3")),
        ledger=QuotaLedger(path=str(tmp_path / "q.sqlite3"), daily_limit=1_000_000),
        client=_client_for(ticker), provider_id="yahoo",
        dataset_resolver=resolve_yahoo_dataset, min_request_interval_ms=0)

    first = coordinator.fetch("stock_quote", "AAPL")
    second = coordinator.fetch("stock_quote", "AAPL")
    assert first.origin == "provider"
    assert second.origin == "cache"
    assert second.external_calls == 0


def test_cache_record_is_tagged_with_the_yahoo_provider_identity(tmp_path):
    ticker = FakeTicker(fast_info={"lastPrice": 311.0})
    cache = MarketDataCache(path=str(tmp_path / "m.sqlite3"))
    coordinator = MarketDataRequestCoordinator(
        cache=cache, ledger=QuotaLedger(path=str(tmp_path / "q.sqlite3"), daily_limit=1_000_000),
        client=_client_for(ticker), provider_id="yahoo",
        dataset_resolver=resolve_yahoo_dataset, min_request_interval_ms=0)

    coordinator.fetch("stock_quote", "AAPL")
    key = coordinator.cache_key_for("stock_quote", "AAPL")
    record = cache.get(key).record
    assert record.provider == "yahoo"
