"""Phase H.3 — the Yahoo Finance client, wrapping the pinned `yfinance==1.5.2`
package (reviewed at docs/security/YAHOO_SEC_PROVIDER_REVIEW.md §1 — UNOFFICIAL
personal-use scraping, no sanctioned API, no key, requires explicit
YAHOO_PERSONAL_USE_ACKNOWLEDGED opt-in before any tool is even registered).

Unlike Alpha Vantage/SEC, `yfinance` owns its own HTTP session, cookie/CSRF
handling, and retries internally (curl_cffi) — it is NOT routed through
`tools/http_safety.py::safe_get`, because there is no caller-supplied URL to
guard: the target is always Yahoo's own fixed internal endpoints, determined
entirely by the ticker argument, never by a model- or user-supplied URL. This
is a deliberate, documented difference from the Alpha Vantage/SEC path, not
an oversight — see the review doc §"exact hosts" discussion.

`fetch(dataset, arguments) -> ProviderResponse` mirrors AlphaVantageClient's
and SecEdgarClient's interface exactly, so the SAME MarketDataRequestCoordinator
(cache, single-flight, retry, quota-ledger-as-instrumentation) serves Yahoo
too. Every yfinance return value (DataFrame, Series, dict of numpy scalars) is
converted to a plain JSON-serializable dict/list before being placed in
`ProviderResponse.payload` — nothing pandas- or numpy-typed, and never a raw
yfinance object, ever reaches the cache or the rest of the system.
"""

import math

from finance.datasets import Dataset
from finance.provider import ProviderResponse
from tools.base import ToolFailure
from tools.models import (
    MARKET_DATA_INVALID_SYMBOL,
    MARKET_DATA_PROVIDER_ERROR,
    MARKET_DATA_RATE_LIMITED,
    MARKET_DATA_TIMEOUT,
    MARKET_DATA_YAHOO_SCHEMA_CHANGED,
)

# Curated, reviewed subset of yfinance's ~180-key `.info` dict — "use only
# reviewed stable-enough fields", never the raw dict wholesale.
_PROFILE_FIELDS = (
    "symbol", "shortName", "longName", "sector", "industry", "country",
    "exchange", "fullExchangeName", "currency", "website",
    "longBusinessSummary", "fullTimeEmployees", "marketCap",
    "sharesOutstanding", "quoteType",
)

# fast_info exposes these via __getitem__; not every ticker populates every
# key, so each is read defensively (see _fast_info_to_dict).
_QUOTE_FIELDS = (
    "lastPrice", "previousClose", "open", "dayHigh", "dayLow", "yearHigh",
    "yearLow", "currency", "exchange", "marketCap", "shares",
    "fiftyDayAverage", "twoHundredDayAverage", "tenDayAverageVolume",
    "threeMonthAverageVolume", "lastVolume",
)

TRANSIENT_CODES = frozenset({MARKET_DATA_TIMEOUT, MARKET_DATA_PROVIDER_ERROR, MARKET_DATA_RATE_LIMITED})
NON_RETRYABLE_CODES = frozenset({MARKET_DATA_INVALID_SYMBOL, MARKET_DATA_YAHOO_SCHEMA_CHANGED})


def is_transient(code) -> bool:
    return code in TRANSIENT_CODES


def is_quota_signal(code) -> bool:
    return code == MARKET_DATA_RATE_LIMITED


def is_quota_exhausted(code) -> bool:
    """Yahoo publishes no daily allowance at all — only YFRateLimitError
    (mapped to MARKET_DATA_RATE_LIMITED) exists, treated as transient/paced,
    never as 'done for today'."""
    return False


def is_negative_cacheable(code) -> bool:
    return code == MARKET_DATA_INVALID_SYMBOL


def _jsonable(value):
    """Recursively strip numpy/pandas scalar types down to plain
    int/float/str/bool/None so nothing pandas-typed ever reaches the cache."""
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    # numpy scalar / pandas Timestamp / anything else with a native equivalent
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (ValueError, TypeError):
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _dataframe_to_records(df):
    if df is None or df.empty:
        return []
    reset = df.reset_index()
    reset.columns = [str(c) for c in reset.columns]
    return [{k: _jsonable(v) for k, v in row.items()} for row in reset.to_dict(orient="records")]


def _series_to_records(series, value_key):
    if series is None or series.empty:
        return []
    return [{"date": _jsonable(idx), value_key: _jsonable(val)} for idx, val in series.items()]


def _fast_info_to_dict(fast_info):
    out = {}
    for key in _QUOTE_FIELDS:
        try:
            out[key] = _jsonable(fast_info[key])
        except (KeyError, IndexError, TypeError, ValueError):
            out[key] = None
    return out


class YahooFinanceClient:
    """Bounded access to six reviewed yfinance operations. No generic
    dispatcher — `fetch` only ever calls the ONE yfinance method a given
    dataset.function names."""

    provider_id = "yahoo"

    def __init__(self, ticker_factory=None):
        # Injected so tests can supply a fake yfinance.Ticker-like object
        # without any real network access.
        self._ticker_factory = ticker_factory

    def _ticker(self, symbol):
        if self._ticker_factory is not None:
            return self._ticker_factory(symbol)
        import yfinance as yf
        return yf.Ticker(symbol)

    def describe_request(self, dataset: Dataset, arguments: dict) -> dict:
        return {
            "provider": self.provider_id,
            "function": dataset.function,
            "dataset_id": dataset.dataset_id,
            "arguments": {k: v for k, v in sorted((arguments or {}).items())},
        }

    def fetch(self, dataset: Dataset, arguments: dict) -> ProviderResponse:
        symbol = (arguments or {}).get("symbol")
        if not symbol:
            raise ToolFailure(MARKET_DATA_PROVIDER_ERROR, "A symbol is required for this Yahoo dataset.")

        try:
            ticker = self._ticker(symbol)
            payload = self._fetch_one(dataset, ticker, arguments or {})
        except ToolFailure:
            raise
        except Exception as e:  # noqa: BLE001 -- classify below, never crash the coordinator
            raise self._classify(symbol, e)

        if not payload:
            raise ToolFailure(MARKET_DATA_INVALID_SYMBOL,
                              f"Yahoo Finance returned no data for {symbol!r}.")
        return ProviderResponse(payload=payload,
                                provider_metadata={"provider_function": dataset.function},
                                byte_count=len(str(payload)))

    def _fetch_one(self, dataset: Dataset, ticker, arguments: dict) -> dict:
        if dataset.function == "fast_info":
            quote = _fast_info_to_dict(ticker.fast_info)
            # fast_info does NOT raise cleanly for an invalid/delisted ticker
            # (verified live: every field just comes back None instead of an
            # exception) -- lastPrice is the one field a real, live ticker
            # always has, so its absence is the actual invalid-symbol signal.
            if quote.get("lastPrice") is None:
                raise ToolFailure(MARKET_DATA_INVALID_SYMBOL,
                                  "Yahoo Finance returned no quote data for that symbol.")
            return {"quote": quote}
        if dataset.function == "info":
            info = ticker.info or {}
            if not info.get("symbol") and not info.get("shortName"):
                raise ToolFailure(MARKET_DATA_INVALID_SYMBOL,
                                  "Yahoo Finance returned no profile data for that symbol.")
            return {"profile": {k: _jsonable(info.get(k)) for k in _PROFILE_FIELDS}}
        if dataset.function == "history":
            period = arguments.get("period") or dataset.defaults.get("period")
            interval = arguments.get("interval") or dataset.defaults.get("interval")
            df = ticker.history(period=period, interval=interval, auto_adjust=False)
            return {"bars": _dataframe_to_records(df), "period": period, "interval": interval}
        if dataset.function == "actions":
            return {
                "dividends": _series_to_records(ticker.dividends, "amount"),
                "splits": _series_to_records(ticker.splits, "ratio"),
            }
        if dataset.function == "analyst_price_targets":
            targets = ticker.analyst_price_targets or {}
            return {"analyst_price_targets": {k: _jsonable(v) for k, v in targets.items()}}
        raise ToolFailure(MARKET_DATA_PROVIDER_ERROR, f"Unsupported Yahoo operation {dataset.function!r}.")

    @staticmethod
    def _classify(symbol, e: Exception) -> ToolFailure:
        from yfinance.exceptions import YFException, YFRateLimitError, YFTickerMissingError

        if isinstance(e, YFRateLimitError):
            return ToolFailure(MARKET_DATA_RATE_LIMITED,
                               "Yahoo Finance is rate limiting requests.", retryable=True)
        if isinstance(e, YFTickerMissingError):
            return ToolFailure(MARKET_DATA_INVALID_SYMBOL,
                               f"Yahoo Finance reports {symbol!r} as missing or delisted.")
        if isinstance(e, YFException):
            return ToolFailure(MARKET_DATA_PROVIDER_ERROR, f"Yahoo Finance reported a problem: {e}")
        if isinstance(e, (KeyError, AttributeError, TypeError, IndexError)):
            # yfinance's own internal parsing hit something it didn't expect
            # -- the concrete, live version of the "Yahoo response schema
            # change" failure mode this project's spec asked to plan for.
            return ToolFailure(MARKET_DATA_YAHOO_SCHEMA_CHANGED,
                               f"Yahoo Finance's response shape did not match what this "
                               f"integration expects ({type(e).__name__}).")
        return ToolFailure(MARKET_DATA_PROVIDER_ERROR,
                           f"The Yahoo Finance request failed ({type(e).__name__}).", retryable=True)
