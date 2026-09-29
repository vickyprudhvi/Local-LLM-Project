"""Phase H.1 — the single path from a dataset request to a payload.

`MarketDataRequestCoordinator.fetch` is the ONLY place in the project that may
cause an Alpha Vantage call. It enforces, in order:

    1. cache lookup            (a fresh hit costs zero quota and returns here)
    2. mode policy             (cached-only / offline never reach the network)
    3. quota affordability     (refuse before spending, not after)
    4. per-key single-flight   (concurrent identical requests -> ONE call)
    5. re-check the cache      (another thread may have just populated it)
    6. bounded retry           (transient codes only, backoff + jitter,
                                cache re-checked before every attempt)
    7. validated cache write   (never a malformed or failed response)
    8. stale-if-error fallback (explicitly labelled stale, never "fresh")

Every dependency — clock, cache, quota ledger, provider client, sleeper, and the
random source used for jitter — is injected, so the whole thing is deterministic
under test with no network, no real timing, and no randomness.
"""

import random
import threading
import time
from dataclasses import dataclass
from typing import Optional

import tools.config as config
from finance import provider as provider_module
from finance.cache import (
    PROVIDER_ID as ALPHAVANTAGE_PROVIDER_ID,
)
from finance.cache import (
    CacheRecord,
    CacheStatus,
    MarketDataCache,
    Origin,
    build_cache_key,
)
from finance.datasets import resolve_dataset
from finance.provider import AlphaVantageClient
from finance.quota import AlphaVantageQuotaLedger
from tools.base import ToolFailure
from tools.models import (
    MARKET_DATA_CACHE_ONLY_UNAVAILABLE,
    MARKET_DATA_DISABLED,
    STOCK_ANALYSIS_QUOTA_INSUFFICIENT,
)


class FetchMode:
    """How much freedom this request has to reach the provider."""

    NORMAL = "normal"            # cache first, then provider if needed
    FORCE_REFRESH = "force_refresh"  # skip the cache read, still write the result
    CACHED_ONLY = "cached_only"  # never call the provider; fresh cache or fail
    OFFLINE = "offline"          # never call the provider; fresh OR stale cache
    ALLOW_STALE = "allow_stale"  # normal, but serve stale rather than fail


_MODES_WITHOUT_NETWORK = frozenset({FetchMode.CACHED_ONLY, FetchMode.OFFLINE})


@dataclass(frozen=True)
class FetchOutcome:
    """One resolved dataset request."""

    dataset_id: str
    symbol: str
    payload: dict
    origin: str          # Origin.PROVIDER | Origin.CACHE
    status: str          # CacheStatus.*
    stale: bool
    record: Optional[CacheRecord]
    external_calls: int
    cache_key: str

    @property
    def from_cache(self) -> bool:
        return self.origin == Origin.CACHE

    def freshness_dict(self, now) -> dict:
        """The provenance block handed to the report and the local LLM.

        This is what makes "cached" vs "fetched" vs "stale" impossible to blur.
        """
        base = {
            "dataset_id": self.dataset_id,
            "symbol": self.symbol,
            "origin": self.origin,
            "cache_status": self.status,
            "stale": self.stale,
            "external_calls": self.external_calls,
        }
        if self.record is not None:
            base.update(self.record.describe(now))
            base["origin"] = self.origin  # keep the ORIGIN of THIS request
        return base


class MarketDataRequestCoordinator:
    # Process-wide PER PROVIDER, because the provider's throttle is per
    # provider, not per coordinator instance — keyed by provider_id so a
    # Yahoo/SEC coordinator's pacing never waits on Alpha Vantage's (or vice
    # versa), even though they may run in the same process. A single lock
    # guards the whole dict; pacing operations are quick, so contention
    # between providers is a non-issue.
    _pace_guard = threading.Lock()
    _last_external_call_by_provider = {}

    def __init__(self, cache=None, ledger=None, client=None, clock=None,
                 sleeper=None, jitter=None, base_dir=None, provider_id=None,
                 dataset_resolver=None, min_request_interval_ms=None):
        self._clock = clock or (lambda: int(time.time()))
        self._cache = cache if cache is not None else MarketDataCache(
            clock=self._clock, base_dir=base_dir)
        self._ledger = ledger if ledger is not None else AlphaVantageQuotaLedger(
            clock=self._clock, base_dir=base_dir)
        self._client = client if client is not None else AlphaVantageClient()
        self._sleep = sleeper if sleeper is not None else time.sleep
        self._jitter = jitter if jitter is not None else random.random
        # Phase H.3: which provider this instance serves, and where it looks
        # up dataset definitions — both default to the original Alpha Vantage
        # behavior, so every existing caller/test that omits them is
        # completely unaffected. A second coordinator instance (Yahoo, SEC)
        # is constructed with these set explicitly, sharing the SAME cache/
        # single-flight/retry machinery rather than a parallel one.
        self._provider_id = provider_id or ALPHAVANTAGE_PROVIDER_ID
        self._resolve_dataset = dataset_resolver or resolve_dataset
        self._min_request_interval_ms = min_request_interval_ms

    @property
    def cache(self):
        return self._cache

    @property
    def ledger(self):
        return self._ledger

    # ---- planning (used before a workflow commits to anything) ----

    def cache_key_for(self, dataset_id, symbol, arguments=None) -> str:
        dataset = self._resolve_dataset(dataset_id)
        return build_cache_key(dataset.dataset_id, dataset.function, symbol,
                               self._merged_arguments(dataset, symbol, arguments),
                               dataset.freshness, provider=self._provider_id)

    @staticmethod
    def _merged_arguments(dataset, symbol, arguments) -> dict:
        merged = dict(dataset.defaults)
        merged.update({k: v for k, v in (arguments or {}).items() if v is not None})
        if "symbol" in dataset.argument_names:
            merged["symbol"] = symbol
        if "keywords" in dataset.required_arguments and "keywords" not in merged:
            merged["keywords"] = symbol
        return {k: v for k, v in merged.items()
                if k in dataset.argument_names or k in dataset.required_arguments}

    def plan(self, requests) -> dict:
        """Estimate what a batch of dataset requests would cost.

        `requests` is an iterable of (dataset_id, symbol, arguments). Returns the
        cached/uncached split and whether the estimated quota covers it — the
        input to the workflow's full / reduced / cached-only decision.
        """
        cached, uncached, stale = [], [], []
        for dataset_id, symbol, arguments in requests:
            key = self.cache_key_for(dataset_id, symbol, arguments)
            lookup = self._cache.get(key)
            if lookup.status == CacheStatus.FRESH:
                cached.append(dataset_id)
                continue
            uncached.append(dataset_id)
            # An expired-but-servable record is not "cached" for planning (it
            # still wants a refresh), but it IS the difference between a degraded
            # analysis and no analysis at all when quota runs out.
            if self._cache.get(key, allow_stale=True).status == CacheStatus.STALE:
                stale.append(dataset_id)
        snapshot = self._ledger.snapshot()
        return {
            "cached_datasets": cached,
            "uncached_datasets": uncached,
            "stale_datasets": stale,
            "max_external_calls": len(uncached),
            "estimated_remaining_quota": snapshot.estimated_remaining,
            "provider_exhausted": snapshot.provider_exhausted,
            "sufficient": (not snapshot.provider_exhausted
                          and snapshot.estimated_remaining >= len(uncached)),
            "quota": snapshot.to_dict(),
        }

    # ---- the one fetch path ----

    def fetch(self, dataset_id, symbol, arguments=None, mode=FetchMode.NORMAL) -> FetchOutcome:
        if not config.market_data_enabled():
            raise ToolFailure(MARKET_DATA_DISABLED, "Market data access is disabled.")

        dataset = self._resolve_dataset(dataset_id)
        merged = self._merged_arguments(dataset, symbol, arguments)
        key = build_cache_key(dataset.dataset_id, dataset.function, symbol, merged,
                              dataset.freshness, provider=self._provider_id)

        # 1-2. Cache first, unless this is an explicit forced refresh.
        if mode != FetchMode.FORCE_REFRESH:
            outcome = self._serve_from_cache(dataset, symbol, key, mode)
            if outcome is not None:
                return outcome

        if mode in _MODES_WITHOUT_NETWORK:
            raise ToolFailure(
                MARKET_DATA_CACHE_ONLY_UNAVAILABLE,
                f"No usable cached {dataset.label or dataset.dataset_id} is available "
                "and this request may not contact the provider.",
            )

        # 3. Refuse to start if the estimated quota cannot cover one call.
        if not self._ledger.can_spend(1):
            stale = self._stale_fallback(dataset, symbol, key)
            if stale is not None:
                return stale
            raise ToolFailure(
                STOCK_ANALYSIS_QUOTA_INSUFFICIENT,
                "The estimated market-data quota for today is exhausted.",
            )

        # 4-7. One external call per key at a time.
        lock = self._cache.lock_for(key)
        with lock:
            # 5. Another thread may have populated the cache while we waited.
            if mode != FetchMode.FORCE_REFRESH:
                outcome = self._serve_from_cache(dataset, symbol, key, mode)
                if outcome is not None:
                    return outcome
            return self._fetch_with_retries(dataset, symbol, merged, key, mode)

    # ---- internals ----

    def _serve_from_cache(self, dataset, symbol, key, mode) -> Optional[FetchOutcome]:
        allow_stale = mode in (FetchMode.OFFLINE, FetchMode.ALLOW_STALE)
        lookup = self._cache.get(key, allow_stale=allow_stale)

        if lookup.status == CacheStatus.NEGATIVE and lookup.record is not None:
            self._ledger.record_cache_hit()
            raise ToolFailure(
                lookup.record.negative_code or "MARKET_DATA_INVALID_SYMBOL",
                f"The provider previously reported no data for {symbol!r}.",
            )
        if lookup.status == CacheStatus.FRESH and lookup.record is not None:
            self._ledger.record_cache_hit()
            return self._outcome(dataset, symbol, key, lookup.record, Origin.CACHE,
                                 CacheStatus.FRESH, stale=False, external_calls=0)
        if lookup.status == CacheStatus.STALE and lookup.record is not None:
            self._ledger.record_cache_hit()
            self._ledger.record_stale_use()
            return self._outcome(dataset, symbol, key, lookup.record, Origin.CACHE,
                                 CacheStatus.STALE, stale=True, external_calls=0)
        return None

    def _fetch_with_retries(self, dataset, symbol, merged, key, mode) -> FetchOutcome:
        attempts = max(1, config.market_data_max_retries() + 1)
        base_delay_ms = config.market_data_retry_base_delay_ms()
        external_calls = 0
        last_failure = None

        for attempt in range(attempts):
            # 6. Every retry re-checks the cache first: a concurrent request may
            # have populated it, and paying for the same data twice is the exact
            # thing this coordinator exists to prevent.
            if attempt > 0:
                served = self._serve_from_cache(dataset, symbol, key, mode)
                if served is not None:
                    return served

            self._pace()
            self._ledger.record_attempt()
            external_calls += 1
            try:
                response = self._client.fetch(dataset, merged)
            except ToolFailure as failure:
                last_failure = failure
                self._ledger.record_failure()

                if provider_module.is_quota_exhausted(failure.code):
                    # Authoritative and terminal: the daily allowance is spent.
                    self._ledger.record_rate_limit(exhausted=True)
                    break

                if provider_module.is_quota_signal(failure.code):
                    # A per-second burst throttle. It clears in about a second,
                    # so retry it — but only a fetch that gives up entirely
                    # advances the throttle streak. Counting each retry would
                    # let ONE unlucky request close the window for the day.
                    giving_up = attempt + 1 >= attempts
                    self._ledger.record_throttle(final=giving_up)
                    if giving_up:
                        break
                    self._ledger.record_retry()
                    self._backoff(attempt, base_delay_ms)
                    continue
                if provider_module.is_negative_cacheable(failure.code):
                    self._cache_negative(dataset, symbol, merged, key, failure.code)
                    break
                if not provider_module.is_transient(failure.code):
                    break
                if attempt + 1 >= attempts:
                    break
                self._ledger.record_retry()
                self._backoff(attempt, base_delay_ms)
                continue

            # 7. Success: validated payload, so it is safe to store.
            self._ledger.record_success()
            record = self._cache.put(
                cache_key=key,
                dataset_id=dataset.dataset_id,
                provider_function=dataset.function,
                symbol=symbol,
                arguments=merged,
                payload=response.payload,
                ttl_seconds=dataset.ttl_seconds,
                freshness=dataset.freshness,
                provider_metadata=response.provider_metadata,
                origin=Origin.PROVIDER,
                provider=self._provider_id,
            )
            return self._outcome(dataset, symbol, key, record, Origin.PROVIDER,
                                 CacheStatus.FRESH, stale=False,
                                 external_calls=external_calls)

        # 8. Everything failed — an expired record is better than nothing, but
        # only when explicitly permitted, and always labelled stale.
        stale = self._stale_fallback(dataset, symbol, key, external_calls)
        if stale is not None:
            return stale
        raise last_failure if last_failure is not None else ToolFailure(
            "MARKET_DATA_PROVIDER_ERROR", "The market-data request failed.")

    def _stale_fallback(self, dataset, symbol, key, external_calls=0) -> Optional[FetchOutcome]:
        """stale-if-error: serve an expired record rather than fail the analysis."""
        lookup = self._cache.get(key, allow_stale=True)
        if lookup.status != CacheStatus.STALE or lookup.record is None:
            return None
        self._ledger.record_stale_use()
        return self._outcome(dataset, symbol, key, lookup.record, Origin.CACHE,
                             CacheStatus.STALE, stale=True, external_calls=external_calls)

    def _cache_negative(self, dataset, symbol, merged, key, code):
        """Negative caching for a CONTROLLED invalid-symbol response only."""
        self._cache.put(
            cache_key=key, dataset_id=dataset.dataset_id,
            provider_function=dataset.function, symbol=symbol, arguments=merged,
            payload={"negative": True},
            ttl_seconds=config.market_data_negative_ttl_seconds(),
            freshness=dataset.freshness, provider_metadata={},
            origin=Origin.PROVIDER, negative=True, negative_code=code,
            provider=self._provider_id,
        )

    def _pace(self):
        """Hold a minimum interval between EXTERNAL calls, PER PROVIDER,
        process-wide.

        Alpha Vantage's free tier allows roughly one request per second, and a
        full analysis fires several datasets back to back — spacing them is
        strictly better than tripping the throttle and retrying. SEC EDGAR's
        pacing need (well under its 10 req/s ceiling) is unrelated and much
        shorter; Yahoo publishes no rate at all. Keyed by `self._provider_id`
        so these never wait on each other. Cache hits never reach here, so a
        fully cached run is still instant.
        """
        interval_ms = (self._min_request_interval_ms if self._min_request_interval_ms is not None
                       else config.market_data_min_request_interval_ms())
        interval = interval_ms / 1000.0
        if interval <= 0:
            return
        with MarketDataRequestCoordinator._pace_guard:
            last = MarketDataRequestCoordinator._last_external_call_by_provider.get(self._provider_id)
            now = time.monotonic()
            wait = (last + interval) - now if last is not None else 0.0
            if wait > 0:
                self._sleep(min(wait, interval))
                now = time.monotonic()
            MarketDataRequestCoordinator._last_external_call_by_provider[self._provider_id] = now

    def _backoff(self, attempt, base_delay_ms):
        """Exponential backoff with jitter, bounded. Sleeper + jitter injected."""
        delay = (base_delay_ms * (2 ** attempt)) / 1000.0
        self._sleep(min(delay, 8.0) * (0.5 + 0.5 * self._jitter()))

    def _outcome(self, dataset, symbol, key, record, origin, status, stale,
                 external_calls) -> FetchOutcome:
        return FetchOutcome(
            dataset_id=dataset.dataset_id,
            symbol=record.symbol if record is not None else symbol,
            payload=record.payload if record is not None else {},
            origin=origin, status=status, stale=stale, record=record,
            external_calls=external_calls, cache_key=key,
        )
