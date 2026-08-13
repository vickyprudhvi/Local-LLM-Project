"""Phase H.1 — market-data cache behaviour.

Covers the eleven cache requirements: provider-then-store, zero external calls
within TTL, argument-order independence, interval/output-size separation, expiry
refresh, policy-gated staleness, corruption rejection, no caching of invalid
responses, single-flight, retry-rechecks-cache, and credential absence.
"""

import json
import os
import sqlite3
import threading

import pytest

import tools.config as config
from finance.cache import (
    CacheStatus,
    MarketDataCache,
    Origin,
    build_cache_key,
    canonical_arguments,
    normalize_symbol,
    payload_digest,
)
from finance.coordinator import FetchMode, MarketDataRequestCoordinator
from finance.datasets import resolve_dataset
from finance.provider import ProviderResponse
from finance.quota import AlphaVantageQuotaLedger
from tools.base import ToolFailure
from tools.models import (
    MARKET_DATA_CACHE_ONLY_UNAVAILABLE,
    MARKET_DATA_INVALID_RESPONSE,
    MARKET_DATA_INVALID_SYMBOL,
    MARKET_DATA_PROVIDER_ERROR,
    MARKET_DATA_QUOTA_EXCEEDED,
    MARKET_DATA_TIMEOUT,
)

SECRET = "SUPER_SECRET_KEY_DO_NOT_LEAK"


class FakeClock:
    def __init__(self, now=1_700_000_000):
        self.now = int(now)

    def __call__(self):
        return int(self.now)

    def advance(self, seconds):
        self.now += int(seconds)


class FakeClient:
    """A provider stand-in that counts calls and never touches the network."""

    def __init__(self, payload=None, failures=None):
        self.payload = payload if payload is not None else {"Global Quote": {"01. symbol": "AAPL"}}
        self.failures = list(failures or [])
        self.calls = []

    def fetch(self, dataset, arguments):
        self.calls.append((dataset.dataset_id, dict(arguments)))
        if self.failures:
            failure = self.failures.pop(0)
            if failure is not None:
                raise failure
        return ProviderResponse(payload=dict(self.payload),
                                provider_metadata={"provider_function": dataset.function},
                                byte_count=len(json.dumps(self.payload)))

    @property
    def call_count(self):
        return len(self.calls)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cache(tmp_path, clock):
    return MarketDataCache(path=str(tmp_path / "market.sqlite3"), clock=clock)


@pytest.fixture
def ledger(tmp_path, clock):
    return AlphaVantageQuotaLedger(path=str(tmp_path / "quota.sqlite3"), clock=clock,
                                   daily_limit=100)


def make_coordinator(cache, ledger, client, clock, **kwargs):
    return MarketDataRequestCoordinator(
        cache=cache, ledger=ledger, client=client, clock=clock,
        sleeper=lambda _s: None, jitter=lambda: 0.5, **kwargs)


# ---- 1-2: provider once, then cache ----

def test_first_request_calls_provider_and_stores(cache, ledger, clock):
    client = FakeClient()
    coord = make_coordinator(cache, ledger, client, clock)

    outcome = coord.fetch("stock_quote", "AAPL")

    assert client.call_count == 1
    assert outcome.origin == Origin.PROVIDER
    assert outcome.external_calls == 1
    assert cache.count() == 1


def test_second_identical_request_within_ttl_makes_zero_external_calls(cache, ledger, clock):
    client = FakeClient()
    coord = make_coordinator(cache, ledger, client, clock)

    coord.fetch("stock_quote", "AAPL")
    clock.advance(60)
    second = coord.fetch("stock_quote", "AAPL")

    assert client.call_count == 1, "the cached result must not cost a provider call"
    assert second.origin == Origin.CACHE
    assert second.external_calls == 0
    assert second.stale is False


# ---- 3-4: cache-key identity ----

def test_argument_order_does_not_change_the_cache_key():
    a = build_cache_key("daily_prices", "TIME_SERIES_DAILY_ADJUSTED", "AAPL",
                        {"outputsize": "full", "symbol": "AAPL"}, "end_of_day")
    b = build_cache_key("daily_prices", "TIME_SERIES_DAILY_ADJUSTED", "AAPL",
                        {"symbol": "AAPL", "outputsize": "full"}, "end_of_day")
    assert a == b


def test_symbol_normalization_collides_but_different_symbols_do_not():
    lower = build_cache_key("stock_quote", "GLOBAL_QUOTE", " aapl ", {}, "delayed")
    upper = build_cache_key("stock_quote", "GLOBAL_QUOTE", "AAPL", {}, "delayed")
    other = build_cache_key("stock_quote", "GLOBAL_QUOTE", "MSFT", {}, "delayed")
    assert lower == upper
    assert lower != other


def test_different_interval_or_output_size_produce_different_keys():
    compact = build_cache_key("daily_prices", "TIME_SERIES_DAILY_ADJUSTED", "AAPL",
                              {"outputsize": "compact"}, "end_of_day")
    full = build_cache_key("daily_prices", "TIME_SERIES_DAILY_ADJUSTED", "AAPL",
                           {"outputsize": "full"}, "end_of_day")
    hourly = build_cache_key("intraday_prices", "TIME_SERIES_INTRADAY", "AAPL",
                             {"interval": "60min"}, "delayed")
    five = build_cache_key("intraday_prices", "TIME_SERIES_INTRADAY", "AAPL",
                           {"interval": "5min"}, "delayed")
    assert compact != full
    assert hourly != five


def test_cache_key_never_contains_the_api_key(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", SECRET)
    key = build_cache_key("stock_quote", "GLOBAL_QUOTE", "AAPL", {"symbol": "AAPL"}, "delayed")
    assert SECRET not in key
    assert len(key) == 64  # a bare SHA-256 hex digest carries nothing else


# ---- 5-6: expiry and staleness ----

def test_expired_record_is_refreshed(cache, ledger, clock):
    client = FakeClient()
    coord = make_coordinator(cache, ledger, client, clock)
    dataset = resolve_dataset("stock_quote")

    coord.fetch("stock_quote", "AAPL")
    clock.advance(dataset.ttl_seconds + 1)
    coord.fetch("stock_quote", "AAPL")

    assert client.call_count == 2, "an expired record must trigger a refresh"


def test_stale_data_is_only_served_under_the_configured_policy(cache, ledger, clock):
    client = FakeClient()
    coord = make_coordinator(cache, ledger, client, clock)
    dataset = resolve_dataset("stock_quote")

    coord.fetch("stock_quote", "AAPL")
    clock.advance(dataset.ttl_seconds + 1)

    # Refresh now fails; NORMAL mode falls back to stale-if-error and LABELS it.
    client.failures = [ToolFailure(MARKET_DATA_TIMEOUT, "timeout", retryable=True)] * 5
    outcome = coord.fetch("stock_quote", "AAPL")
    assert outcome.stale is True
    assert outcome.status == CacheStatus.STALE
    assert outcome.origin == Origin.CACHE


def test_stale_record_beyond_the_grace_window_is_not_served(cache, ledger, clock, monkeypatch):
    monkeypatch.setenv("MARKET_DATA_STALE_IF_ERROR_SECONDS", "100")
    client = FakeClient()
    coord = make_coordinator(cache, ledger, client, clock)

    coord.fetch("stock_quote", "AAPL")
    clock.advance(10_000)
    client.failures = [ToolFailure(MARKET_DATA_TIMEOUT, "timeout", retryable=True)] * 5

    with pytest.raises(ToolFailure) as excinfo:
        coord.fetch("stock_quote", "AAPL")
    assert excinfo.value.code == MARKET_DATA_TIMEOUT


def test_cached_only_mode_never_calls_the_provider(cache, ledger, clock):
    client = FakeClient()
    coord = make_coordinator(cache, ledger, client, clock)

    with pytest.raises(ToolFailure) as excinfo:
        coord.fetch("stock_quote", "AAPL", mode=FetchMode.CACHED_ONLY)

    assert excinfo.value.code == MARKET_DATA_CACHE_ONLY_UNAVAILABLE
    assert client.call_count == 0


def test_offline_mode_serves_stale_without_calling_the_provider(cache, ledger, clock):
    client = FakeClient()
    coord = make_coordinator(cache, ledger, client, clock)
    coord.fetch("stock_quote", "AAPL")
    clock.advance(resolve_dataset("stock_quote").ttl_seconds + 1)

    outcome = coord.fetch("stock_quote", "AAPL", mode=FetchMode.OFFLINE)

    assert client.call_count == 1, "offline mode must not reach the provider"
    assert outcome.stale is True


def test_force_refresh_bypasses_a_fresh_cache_entry(cache, ledger, clock):
    client = FakeClient()
    coord = make_coordinator(cache, ledger, client, clock)
    coord.fetch("stock_quote", "AAPL")

    outcome = coord.fetch("stock_quote", "AAPL", mode=FetchMode.FORCE_REFRESH)

    assert client.call_count == 2
    assert outcome.origin == Origin.PROVIDER


# ---- 7-8: integrity ----

def test_corrupt_record_is_rejected_and_deleted(cache, clock):
    key = build_cache_key("stock_quote", "GLOBAL_QUOTE", "AAPL", {"symbol": "AAPL"}, "delayed")
    cache.put(key, "stock_quote", "GLOBAL_QUOTE", "AAPL", {"symbol": "AAPL"},
              {"Global Quote": {"01. symbol": "AAPL"}}, 3600, "delayed")

    # Tamper with the stored payload so it no longer matches its recorded digest.
    with sqlite3.connect(cache.path) as conn:
        conn.execute("UPDATE market_data SET payload = ? WHERE cache_key = ?",
                     (json.dumps({"Global Quote": {"01. symbol": "TAMPERED"}}), key))

    lookup = cache.get(key)
    assert lookup.status == CacheStatus.CORRUPT
    assert cache.count() == 0, "a corrupt row must be removed, not returned"


def test_payload_digest_detects_any_change():
    original = {"a": 1, "b": [1, 2, 3]}
    assert payload_digest(original) == payload_digest({"b": [1, 2, 3], "a": 1})
    assert payload_digest(original) != payload_digest({"a": 1, "b": [1, 2, 4]})


def test_invalid_provider_data_is_not_cached_as_success(cache, ledger, clock):
    client = FakeClient(failures=[ToolFailure(MARKET_DATA_INVALID_RESPONSE, "bad shape")])
    coord = make_coordinator(cache, ledger, client, clock)

    with pytest.raises(ToolFailure) as excinfo:
        coord.fetch("stock_quote", "AAPL")

    assert excinfo.value.code == MARKET_DATA_INVALID_RESPONSE
    assert client.call_count == 1, "a malformed response is deterministic, not retryable"
    assert cache.count() == 0, "a failed response must never become a cache entry"


def test_non_dict_payload_is_refused_by_the_cache(cache):
    with pytest.raises(ValueError):
        cache.put("k", "stock_quote", "GLOBAL_QUOTE", "AAPL", {}, ["not", "a", "dict"],
                  3600, "delayed")


def test_invalid_symbol_is_negatively_cached_and_not_refetched(cache, ledger, clock):
    client = FakeClient(failures=[ToolFailure(MARKET_DATA_INVALID_SYMBOL, "no data")])
    coord = make_coordinator(cache, ledger, client, clock)

    with pytest.raises(ToolFailure):
        coord.fetch("company_overview", "NOTATICKER")
    with pytest.raises(ToolFailure) as excinfo:
        coord.fetch("company_overview", "NOTATICKER")

    assert client.call_count == 1, "a negatively cached symbol must not be re-requested"
    assert excinfo.value.code == MARKET_DATA_INVALID_SYMBOL


# ---- 9-10: concurrency ----

def test_concurrent_identical_requests_produce_one_external_call(cache, ledger, clock):
    """Three threads ask for the same dataset at once.

    The first to win the per-key lock holds it while the other two block; they
    must then find the cache populated rather than each paying for a call.
    """
    import time as real_time

    start = threading.Barrier(3, timeout=15)

    class SlowClient(FakeClient):
        def fetch(self, dataset, arguments):
            # Hold the single-flight lock long enough for the other two threads
            # to pile up behind it.
            real_time.sleep(0.4)
            return super().fetch(dataset, arguments)

    client = SlowClient()
    coord = make_coordinator(cache, ledger, client, clock)
    results, errors = [], []
    results_guard = threading.Lock()

    def worker():
        try:
            start.wait()
            outcome = coord.fetch("stock_quote", "AAPL")
            with results_guard:
                results.append(outcome)
        except Exception as exc:  # noqa: BLE001 — surfaced via `errors` below
            with results_guard:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert not errors, f"worker errors: {errors}"
    assert client.call_count == 1, "single-flight must collapse identical requests"
    assert len(results) == 3
    assert sum(r.external_calls for r in results) == 1
    assert sum(1 for r in results if r.from_cache) == 2


def test_retry_checks_the_cache_before_calling_the_provider_again(cache, ledger, clock):
    """A transient failure retries — but if another request populated the cache
    in the meantime, the retry must serve that instead of spending a second call."""
    dataset = resolve_dataset("stock_quote")
    key = build_cache_key(dataset.dataset_id, dataset.function, "AAPL",
                          {"symbol": "AAPL"}, dataset.freshness)

    class PopulatingClient(FakeClient):
        def fetch(self, inner_dataset, arguments):
            self.calls.append((inner_dataset.dataset_id, dict(arguments)))
            # Simulate a concurrent writer landing between attempt 1 and 2.
            cache.put(key, dataset.dataset_id, dataset.function, "AAPL",
                      {"symbol": "AAPL"}, {"Global Quote": {"01. symbol": "AAPL"}},
                      dataset.ttl_seconds, dataset.freshness)
            raise ToolFailure(MARKET_DATA_TIMEOUT, "timeout", retryable=True)

    client = PopulatingClient()
    coord = make_coordinator(cache, ledger, client, clock)

    outcome = coord.fetch("stock_quote", "AAPL")

    assert client.call_count == 1, "the retry must find the cache populated"
    assert outcome.origin == Origin.CACHE


# ---- 11: no credential anywhere ----

def test_api_key_never_appears_in_cache_contents(cache, ledger, clock, monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", SECRET)
    client = FakeClient()
    coord = make_coordinator(cache, ledger, client, clock)
    coord.fetch("stock_quote", "AAPL")

    with open(cache.path, "rb") as handle:
        raw = handle.read()
    assert SECRET.encode() not in raw

    # Also check the WAL sidecar, which is where a recent write actually lives.
    wal = cache.path + "-wal"
    if os.path.exists(wal):
        with open(wal, "rb") as handle:
            assert SECRET.encode() not in handle.read()


def test_schema_version_change_invalidates_old_records(tmp_path, clock, monkeypatch):
    path = str(tmp_path / "market.sqlite3")
    cache = MarketDataCache(path=path, clock=clock)
    key = build_cache_key("stock_quote", "GLOBAL_QUOTE", "AAPL", {}, "delayed")
    cache.put(key, "stock_quote", "GLOBAL_QUOTE", "AAPL", {}, {"x": 1}, 3600, "delayed")

    # A record written under a different payload-schema version is not reused.
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE market_data SET schema_version = 999 WHERE cache_key = ?", (key,))

    assert cache.get(key).status == CacheStatus.MISS
