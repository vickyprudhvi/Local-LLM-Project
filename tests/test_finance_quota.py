"""Phase H.1 — quota ledger and retry policy.

Covers the six quota requirements: cache hits are free, an external call is
counted exactly once, retries are counted, quota-exceeded is never retried, the
workflow can degrade to reduced / cached-only, and the reset window is
deterministic.
"""

import pytest

from finance.cache import MarketDataCache, Origin
from finance.coordinator import FetchMode, MarketDataRequestCoordinator
from finance.provider import (
    NON_RETRYABLE_CODES,
    TRANSIENT_CODES,
    is_negative_cacheable,
    is_quota_signal,
    is_transient,
)
from finance.quota import AlphaVantageQuotaLedger
from tests.test_finance_cache import FakeClient, FakeClock, make_coordinator
from tools.base import ToolFailure
from tools.models import (
    MARKET_DATA_AUTHENTICATION_FAILED,
    MARKET_DATA_ENTITLEMENT_REQUIRED,
    MARKET_DATA_INVALID_SYMBOL,
    MARKET_DATA_QUOTA_EXCEEDED,
    MARKET_DATA_RATE_LIMITED,
    MARKET_DATA_TIMEOUT,
    STOCK_ANALYSIS_QUOTA_INSUFFICIENT,
)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cache(tmp_path, clock):
    return MarketDataCache(path=str(tmp_path / "market.sqlite3"), clock=clock)


@pytest.fixture
def ledger(tmp_path, clock):
    return AlphaVantageQuotaLedger(path=str(tmp_path / "quota.sqlite3"), clock=clock,
                                   daily_limit=25)


# ---- 1: cache hits are free ----

def test_cache_hits_do_not_increase_external_call_count(cache, ledger, clock):
    client = FakeClient()
    coord = make_coordinator(cache, ledger, client, clock)

    coord.fetch("stock_quote", "AAPL")
    after_first = ledger.snapshot()
    for _ in range(5):
        coord.fetch("stock_quote", "AAPL")
    after_repeats = ledger.snapshot()

    assert after_first.attempted == 1
    assert after_repeats.attempted == 1, "cache hits must not consume quota"
    assert after_repeats.cache_hits == 5
    assert after_repeats.estimated_remaining == 24


# ---- 2: one external call, one ledger entry ----

def test_external_call_updates_the_ledger_once(cache, ledger, clock):
    client = FakeClient()
    coord = make_coordinator(cache, ledger, client, clock)

    coord.fetch("stock_quote", "AAPL")
    coord.fetch("company_overview", "AAPL")

    snap = ledger.snapshot()
    assert snap.attempted == 2
    assert snap.succeeded == 2
    assert snap.failed == 0


# ---- 3: retries are counted ----

def test_retries_are_counted_correctly(cache, ledger, clock, monkeypatch):
    monkeypatch.setenv("MARKET_DATA_MAX_RETRIES", "2")
    transient = ToolFailure(MARKET_DATA_TIMEOUT, "timeout", retryable=True)
    client = FakeClient(failures=[transient, transient, None])
    coord = make_coordinator(cache, ledger, client, clock)

    outcome = coord.fetch("stock_quote", "AAPL")

    snap = ledger.snapshot()
    assert client.call_count == 3, "two retries after the initial attempt"
    assert snap.attempted == 3
    assert snap.retries == 2
    assert snap.failed == 2
    assert snap.succeeded == 1
    assert outcome.origin == Origin.PROVIDER


def test_retries_stop_at_the_configured_limit(cache, ledger, clock, monkeypatch):
    monkeypatch.setenv("MARKET_DATA_MAX_RETRIES", "1")
    transient = ToolFailure(MARKET_DATA_TIMEOUT, "timeout", retryable=True)
    client = FakeClient(failures=[transient] * 10)
    coord = make_coordinator(cache, ledger, client, clock)

    with pytest.raises(ToolFailure):
        coord.fetch("stock_quote", "AAPL")

    assert client.call_count == 2, "initial attempt plus exactly one retry"


# ---- 4: deterministic rejections are never retried ----

@pytest.mark.parametrize("code", [
    MARKET_DATA_QUOTA_EXCEEDED,
    MARKET_DATA_AUTHENTICATION_FAILED,
    MARKET_DATA_ENTITLEMENT_REQUIRED,
    MARKET_DATA_INVALID_SYMBOL,
])
def test_deterministic_failures_are_not_retried(cache, ledger, clock, code, monkeypatch):
    monkeypatch.setenv("MARKET_DATA_MAX_RETRIES", "3")
    client = FakeClient(failures=[ToolFailure(code, "no")] * 10)
    coord = make_coordinator(cache, ledger, client, clock)

    with pytest.raises(ToolFailure):
        coord.fetch("stock_quote", "AAPL")

    assert client.call_count == 1, f"{code} must never be retried"
    assert ledger.snapshot().retries == 0


def test_quota_exceeded_marks_the_window_exhausted(cache, ledger, clock):
    client = FakeClient(failures=[ToolFailure(MARKET_DATA_QUOTA_EXCEEDED, "daily limit")])
    coord = make_coordinator(cache, ledger, client, clock)

    with pytest.raises(ToolFailure):
        coord.fetch("stock_quote", "AAPL")

    snap = ledger.snapshot()
    assert snap.provider_exhausted is True
    assert snap.rate_limit_events == 1
    assert snap.estimated_remaining == 0, "the provider is authoritative over the estimate"
    assert ledger.can_spend(1) is False


def test_provider_signal_overrides_a_generous_local_estimate(tmp_path, clock):
    generous = AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock,
                                       daily_limit=100_000)
    assert generous.can_spend(1) is True
    generous.record_rate_limit(exhausted=True)
    assert generous.can_spend(1) is False


# ---- provider message classification (regression: live run, 2026-08-05) ----

# Both bodies below are captured VERBATIM from live Alpha Vantage responses
# (2026-08), not guessed — with the real key substituted for a fake one.
# THROTTLE is the per-second burst advisory; it merely quotes the daily
# allowance in passing, and misreading that as exhaustion locked market data
# out for the rest of the day (fixed once already). EXHAUSTED is the genuine
# daily-cap-reached message, discovered to contain NO "exceeded"/"exhaust"
# verb at all — an earlier fabricated fixture assumed one existed and so this
# case was, until corrected, itself misclassified as a retryable throttle.
THROTTLE_BODY = (
    "Thank you for using Alpha Vantage! Please consider spreading out your free "
    "API requests more sparingly (1 request per second). You may subscribe to any "
    "of the premium plans at https://www.alphavantage.co/premium/ to lift the free "
    "key rate limit (25 requests per day), raise the per-second burst limit, and "
    "instantly unlock all premium endpoints."
)
EXHAUSTED_BODY = (
    "We have detected your API key as FAKEKEY1234567890 and our standard API rate "
    "limit is 25 requests per day. Please subscribe to any of the premium plans at "
    "https://www.alphavantage.co/premium/ to instantly remove all daily rate limits."
)


@pytest.mark.parametrize("body,expected", [
    (THROTTLE_BODY, MARKET_DATA_RATE_LIMITED),
    (EXHAUSTED_BODY, MARKET_DATA_QUOTA_EXCEEDED),
])
def test_throttle_and_exhaustion_are_classified_differently(body, expected):
    from finance.provider import AlphaVantageClient

    with pytest.raises(ToolFailure) as excinfo:
        AlphaVantageClient._raise_for_inband_error({"Note": body})
    assert excinfo.value.code == expected


def test_a_burst_throttle_is_retried_and_then_succeeds(cache, ledger, clock, monkeypatch):
    """A per-second throttle clears in about a second, so it IS retried."""
    monkeypatch.setenv("MARKET_DATA_MAX_RETRIES", "2")
    from finance.provider import AlphaVantageClient

    throttle = None
    try:
        AlphaVantageClient._raise_for_inband_error({"Note": THROTTLE_BODY})
    except ToolFailure as exc:
        throttle = exc

    client = FakeClient(failures=[throttle, None])
    coord = make_coordinator(cache, ledger, client, clock)

    outcome = coord.fetch("stock_quote", "IBM")

    assert outcome.origin == Origin.PROVIDER
    assert client.call_count == 2, "the throttle must be retried, not abandoned"
    snap = ledger.snapshot()
    assert snap.rate_limit_events == 1
    assert snap.provider_exhausted is False


def test_one_fully_throttled_fetch_does_not_zero_the_days_budget(cache, ledger, clock):
    """Even when every retry inside ONE fetch throttles, the day stays open."""
    from finance.provider import AlphaVantageClient

    def throttling_fetch(dataset, arguments):
        AlphaVantageClient._raise_for_inband_error({"Note": THROTTLE_BODY})

    client = FakeClient()
    client.fetch = throttling_fetch
    coord = make_coordinator(cache, ledger, client, clock)

    with pytest.raises(ToolFailure) as excinfo:
        coord.fetch("stock_quote", "IBM")

    assert excinfo.value.code == MARKET_DATA_RATE_LIMITED
    snap = ledger.snapshot()
    assert snap.consecutive_throttles == 1, \
        "the STREAK counts failed fetches, not individual retries"
    assert snap.provider_exhausted is False, \
        "one unlucky request must not lock out the rest of the day"
    assert ledger.can_spend(1) is True


def test_genuine_exhaustion_does_close_the_window(cache, ledger, clock):
    from finance.provider import AlphaVantageClient

    def exhausted_fetch(dataset, arguments):
        AlphaVantageClient._raise_for_inband_error({"Note": EXHAUSTED_BODY})

    client = FakeClient()
    client.fetch = exhausted_fetch
    coord = make_coordinator(cache, ledger, client, clock)

    with pytest.raises(ToolFailure):
        coord.fetch("stock_quote", "IBM")

    assert ledger.snapshot().provider_exhausted is True
    assert ledger.can_spend(1) is False


def test_repeated_throttles_eventually_close_the_window(tmp_path, clock):
    """One burst throttle must not lock the day; an unbroken run of them must,
    otherwise a genuinely capped key is retried futilely all day."""
    led = AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock,
                                  daily_limit=25)
    limit = AlphaVantageQuotaLedger.CONSECUTIVE_THROTTLES_MEANING_EXHAUSTED

    for _ in range(limit - 1):
        led.record_rate_limit(exhausted=False)
    assert led.snapshot().provider_exhausted is False
    assert led.can_spend(1) is True

    led.record_rate_limit(exhausted=False)
    assert led.snapshot().provider_exhausted is True
    assert led.can_spend(1) is False


def test_a_success_clears_the_throttle_streak(tmp_path, clock):
    led = AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock,
                                  daily_limit=25)
    limit = AlphaVantageQuotaLedger.CONSECUTIVE_THROTTLES_MEANING_EXHAUSTED

    for _ in range(limit - 1):
        led.record_rate_limit(exhausted=False)
    led.record_success()
    assert led.snapshot().consecutive_throttles == 0

    # The streak restarts from zero, so the window stays open.
    led.record_rate_limit(exhausted=False)
    assert led.snapshot().provider_exhausted is False


def test_a_pre_existing_ledger_is_migrated_not_dropped(tmp_path, clock):
    """A ledger written by an earlier version must gain new columns in place.

    These counters ARE the record of quota already spent today; dropping them
    would let the same daily budget be spent twice.
    """
    import sqlite3

    path = str(tmp_path / "old.sqlite3")
    with sqlite3.connect(path) as conn:  # the pre-migration schema
        conn.executescript("""
            CREATE TABLE quota_ledger (
                window_key TEXT PRIMARY KEY,
                attempted INTEGER NOT NULL DEFAULT 0,
                succeeded INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                retries INTEGER NOT NULL DEFAULT 0,
                cache_hits INTEGER NOT NULL DEFAULT 0,
                stale_uses INTEGER NOT NULL DEFAULT 0,
                rate_limit_events INTEGER NOT NULL DEFAULT 0,
                provider_exhausted INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL DEFAULT 0
            );""")
        conn.execute(
            "INSERT INTO quota_ledger(window_key, attempted, succeeded) VALUES (?, 7, 6)",
            (AlphaVantageQuotaLedger(path=":memory:", clock=clock).window_key(),))

    led = AlphaVantageQuotaLedger(path=path, clock=clock, daily_limit=25)
    snap = led.snapshot()

    assert snap.attempted == 7, "already-spent quota must survive the migration"
    assert snap.succeeded == 6
    assert snap.consecutive_throttles == 0
    assert snap.estimated_remaining == 18


def test_provider_messages_never_leak_a_url_or_credential():
    from finance.provider import AlphaVantageClient

    with pytest.raises(ToolFailure) as excinfo:
        AlphaVantageClient._raise_for_inband_error({"Note": THROTTLE_BODY})
    assert "https://" not in excinfo.value.message
    assert "<url>" in excinfo.value.message


def test_the_literal_api_key_is_stripped_from_the_exhaustion_message():
    """Regression: the REAL exhausted-key message embeds the credential as bare
    text ("...API key as <KEY>..."), a shape the apikey=/URL patterns alone do
    not cover — this leaked a live key into an exception message once. The
    caller must pass the exact key as `secret` so it is stripped by literal
    match regardless of how the provider chooses to phrase the message."""
    from finance.provider import AlphaVantageClient

    fake_key = "FAKEKEY1234567890"
    assert fake_key in EXHAUSTED_BODY, "fixture must actually contain the token to strip"

    with pytest.raises(ToolFailure) as excinfo:
        AlphaVantageClient._raise_for_inband_error({"Note": EXHAUSTED_BODY}, secret=fake_key)
    assert fake_key not in excinfo.value.message
    assert "REDACTED" in excinfo.value.message


def test_redaction_without_a_secret_still_never_raises():
    """Callers that cannot supply a key (e.g. it was already known-empty) must
    still get a bounded, safe message rather than an exception of their own."""
    from finance.provider import AlphaVantageClient

    with pytest.raises(ToolFailure) as excinfo:
        AlphaVantageClient._raise_for_inband_error({"Note": EXHAUSTED_BODY})
    assert isinstance(excinfo.value.message, str)


def test_retry_classification_fails_closed():
    assert is_transient(MARKET_DATA_TIMEOUT) is True
    assert is_transient(MARKET_DATA_QUOTA_EXCEEDED) is False
    assert is_transient("SOME_BRAND_NEW_UNKNOWN_CODE") is False
    assert is_quota_signal(MARKET_DATA_RATE_LIMITED) is True
    assert is_negative_cacheable(MARKET_DATA_INVALID_SYMBOL) is True
    assert is_negative_cacheable(MARKET_DATA_TIMEOUT) is False, \
        "a transport failure must never poison the cache"
    assert TRANSIENT_CODES.isdisjoint(NON_RETRYABLE_CODES)


# ---- 5: degradation ----

def test_exhausted_quota_refuses_before_spending(cache, ledger, clock):
    client = FakeClient()
    coord = make_coordinator(cache, ledger, client, clock)
    ledger.record_rate_limit(exhausted=True)

    with pytest.raises(ToolFailure) as excinfo:
        coord.fetch("stock_quote", "AAPL")

    assert excinfo.value.code == STOCK_ANALYSIS_QUOTA_INSUFFICIENT
    assert client.call_count == 0, "no call may be made once quota is known exhausted"


def test_plan_reports_the_cached_uncached_split(cache, ledger, clock):
    client = FakeClient()
    coord = make_coordinator(cache, ledger, client, clock)
    coord.fetch("stock_quote", "AAPL")

    plan = coord.plan([
        ("stock_quote", "AAPL", None),
        ("company_overview", "AAPL", None),
        ("income_statement", "AAPL", None),
    ])

    assert plan["cached_datasets"] == ["stock_quote"]
    assert sorted(plan["uncached_datasets"]) == ["company_overview", "income_statement"]
    assert plan["max_external_calls"] == 2
    assert plan["sufficient"] is True


def test_plan_reports_insufficient_quota(cache, ledger, clock):
    client = FakeClient()
    coord = make_coordinator(cache, ledger, client, clock)
    ledger.record_rate_limit(exhausted=True)

    plan = coord.plan([("stock_quote", "AAPL", None), ("company_overview", "AAPL", None)])

    assert plan["sufficient"] is False
    assert plan["provider_exhausted"] is True
    assert plan["estimated_remaining_quota"] == 0


def test_exhausted_quota_still_serves_stale_cache(cache, ledger, clock):
    client = FakeClient()
    coord = make_coordinator(cache, ledger, client, clock)
    coord.fetch("stock_quote", "AAPL")
    clock.advance(10_000)
    ledger.record_rate_limit(exhausted=True)

    outcome = coord.fetch("stock_quote", "AAPL")

    assert outcome.stale is True
    assert client.call_count == 1, "degradation must prefer stale cache over failing"


# ---- 6: deterministic reset window ----

def test_reset_window_is_a_deterministic_utc_day(tmp_path):
    clock = FakeClock(1_700_000_000)  # 2023-11-14T22:13:20Z
    led = AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock,
                                  daily_limit=25)
    assert led.window_key() == "2023-11-14"

    led.record_attempt()
    assert led.snapshot().attempted == 1

    clock.advance(2 * 3600)  # crosses UTC midnight into 2023-11-15
    assert led.window_key() == "2023-11-15"
    assert led.snapshot().attempted == 0, "a new UTC day starts a fresh window"
    assert led.snapshot().provider_exhausted is False


def test_exhaustion_does_not_leak_into_the_next_window(tmp_path):
    clock = FakeClock(1_700_000_000)
    led = AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock,
                                  daily_limit=25)
    led.record_rate_limit(exhausted=True)
    assert led.can_spend(1) is False

    clock.advance(24 * 3600)
    assert led.can_spend(1) is True
