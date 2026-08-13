"""Phase H.1 — the local Alpha Vantage quota ledger.

Counts what actually happened, so the workflow can decide how much analysis it
can afford BEFORE it starts making calls.

Two deliberate choices:

* The daily limit is a CONFIGURABLE ESTIMATE (`ALPHAVANTAGE_DAILY_CALL_LIMIT`),
  not a claim about the user's plan. Nothing here hardcodes a tier.
* A provider rate-limit response is AUTHORITATIVE. When the provider says the
  quota is exhausted, the ledger records it and `can_spend` refuses further
  external calls for the rest of the window regardless of what the local
  estimate says. The estimate can be wrong; the provider cannot.

The reset window is a UTC calendar day, computed from the injected clock, so the
rollover is deterministic in tests.
"""

import datetime
import json
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass

import tools.config as config
from finance.cache import utc_now

_SCHEMA = """
CREATE TABLE IF NOT EXISTS quota_ledger (
    window_key            TEXT PRIMARY KEY,
    attempted             INTEGER NOT NULL DEFAULT 0,
    succeeded             INTEGER NOT NULL DEFAULT 0,
    failed                INTEGER NOT NULL DEFAULT 0,
    retries               INTEGER NOT NULL DEFAULT 0,
    cache_hits            INTEGER NOT NULL DEFAULT 0,
    stale_uses            INTEGER NOT NULL DEFAULT 0,
    rate_limit_events     INTEGER NOT NULL DEFAULT 0,
    provider_exhausted    INTEGER NOT NULL DEFAULT 0,
    consecutive_throttles INTEGER NOT NULL DEFAULT 0,
    updated_at            INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass(frozen=True)
class QuotaSnapshot:
    """A point-in-time view of one reset window."""

    window_key: str
    attempted: int
    succeeded: int
    failed: int
    retries: int
    cache_hits: int
    stale_uses: int
    rate_limit_events: int
    provider_exhausted: bool
    estimated_daily_limit: int
    consecutive_throttles: int = 0

    @property
    def estimated_remaining(self) -> int:
        """Remaining external calls under the LOCAL ESTIMATE.

        Zero once the provider has told us the quota is exhausted, whatever the
        estimate says.
        """
        if self.provider_exhausted:
            return 0
        return max(0, self.estimated_daily_limit - self.attempted)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["estimated_remaining"] = self.estimated_remaining
        return out


class AlphaVantageQuotaLedger:
    """Durable per-window counters for external provider usage."""

    def __init__(self, path=None, clock=None, base_dir=None, daily_limit=None):
        self._clock = clock or utc_now
        self._daily_limit = daily_limit
        self._path = self._resolve_path(path, base_dir)
        self._guard = threading.Lock()
        self._ensure_schema()

    @staticmethod
    def _resolve_path(path, base_dir):
        if path == ":memory:":
            return path
        if path is None:
            cache_path = config.market_data_cache_path()
            path = os.path.join(os.path.dirname(cache_path), "quota_ledger.sqlite3")
        if not os.path.isabs(path):
            root = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(root, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    @property
    def path(self):
        return self._path

    def _connect(self):
        conn = sqlite3.connect(self._path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    # Columns added after the first release, with their SQL definition. A ledger
    # created by an earlier version is migrated in place rather than dropped:
    # unlike the cache (whose contents are always re-fetchable), these counters
    # ARE the record of what quota has already been spent today, so losing them
    # would let the day's budget be silently spent twice.
    _ADDED_COLUMNS = (
        ("consecutive_throttles", "INTEGER NOT NULL DEFAULT 0"),
    )

    def _ensure_schema(self):
        with self._guard, self._connect() as conn:
            conn.executescript(_SCHEMA)
            existing = {row["name"] for row in
                        conn.execute("PRAGMA table_info(quota_ledger)")}
            for name, definition in self._ADDED_COLUMNS:
                if name not in existing:
                    conn.execute(
                        f"ALTER TABLE quota_ledger ADD COLUMN {name} {definition}")

    @property
    def daily_limit(self) -> int:
        return (config.alphavantage_daily_call_limit()
                if self._daily_limit is None else int(self._daily_limit))

    def window_key(self, now=None) -> str:
        """The UTC calendar day this call falls in. Alpha Vantage's free-tier
        daily counter resets on UTC midnight; a provider that publishes a
        different window would need this changed, which is why it is one method."""
        moment = datetime.datetime.fromtimestamp(
            int(now if now is not None else self._clock()), datetime.timezone.utc)
        return moment.strftime("%Y-%m-%d")

    # ---- recording ----

    def _bump(self, **deltas):
        key = self.window_key()
        now = self._clock()
        columns = ("attempted", "succeeded", "failed", "retries", "cache_hits",
                   "stale_uses", "rate_limit_events", "provider_exhausted",
                   "consecutive_throttles")
        with self._guard, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO quota_ledger(window_key) VALUES (?)", (key,))
            sets = ", ".join(f"{c} = {c} + ?" for c in columns if deltas.get(c))
            values = [int(deltas[c]) for c in columns if deltas.get(c)]
            if sets:
                conn.execute(
                    f"UPDATE quota_ledger SET {sets}, updated_at = ? WHERE window_key = ?",
                    (*values, now, key))
            else:
                conn.execute(
                    "UPDATE quota_ledger SET updated_at = ? WHERE window_key = ?", (now, key))

    def record_attempt(self):
        """One external provider request is about to be made."""
        self._bump(attempted=1)

    def record_success(self):
        """A success proves the provider is answering, so any throttle streak is
        cleared — only UNBROKEN throttles indicate a spent window."""
        self._bump(succeeded=1)
        self._reset_throttle_streak()

    def _reset_throttle_streak(self):
        key = self.window_key()
        with self._guard, self._connect() as conn:
            conn.execute(
                "UPDATE quota_ledger SET consecutive_throttles = 0 WHERE window_key = ?",
                (key,))

    def record_failure(self):
        self._bump(failed=1)

    def record_retry(self):
        self._bump(retries=1)

    def record_cache_hit(self):
        """A served-from-cache result. Deliberately does NOT touch `attempted`:
        a cache hit costs no quota, and the tests assert exactly that."""
        self._bump(cache_hits=1)

    def record_stale_use(self):
        self._bump(stale_uses=1)

    # Alpha Vantage returns the SAME advisory for a per-second burst throttle and
    # for a spent daily allowance, so a single message cannot tell them apart.
    # A burst throttle clears within seconds; a spent allowance does not. After
    # this many consecutive throttles with no intervening success, we conclude the
    # window really is spent — which avoids both a false all-day lockout on one
    # burst and a futile retry loop against a capped key.
    CONSECUTIVE_THROTTLES_MEANING_EXHAUSTED = 3

    def record_rate_limit(self, exhausted=True):
        """The provider reported an UNAMBIGUOUS quota message. Closes the window
        immediately when `exhausted`."""
        if exhausted:
            self._bump(rate_limit_events=1, provider_exhausted=1,
                       consecutive_throttles=1)
            return
        self.record_throttle(final=True)

    def record_throttle(self, final=False):
        """A per-second burst throttle.

        `final=False` just counts the event (the request will be retried).
        `final=True` means the whole fetch gave up, which advances the streak —
        so the window closes only after several FETCHES fail this way, never
        because one request's retries all happened to be throttled.
        """
        if not final:
            self._bump(rate_limit_events=1)
            return
        self._bump(rate_limit_events=1, consecutive_throttles=1)
        if self.snapshot().consecutive_throttles >= self.CONSECUTIVE_THROTTLES_MEANING_EXHAUSTED:
            self._bump(provider_exhausted=1)

    # ---- reading ----

    def snapshot(self) -> QuotaSnapshot:
        key = self.window_key()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM quota_ledger WHERE window_key = ?", (key,)).fetchone()
        if row is None:
            return QuotaSnapshot(key, 0, 0, 0, 0, 0, 0, 0, False, self.daily_limit, 0)
        return QuotaSnapshot(
            window_key=key,
            attempted=int(row["attempted"]),
            succeeded=int(row["succeeded"]),
            failed=int(row["failed"]),
            retries=int(row["retries"]),
            cache_hits=int(row["cache_hits"]),
            stale_uses=int(row["stale_uses"]),
            rate_limit_events=int(row["rate_limit_events"]),
            provider_exhausted=bool(row["provider_exhausted"]),
            estimated_daily_limit=self.daily_limit,
            consecutive_throttles=int(row["consecutive_throttles"]),
        )

    def can_spend(self, calls=1) -> bool:
        """Whether `calls` more external requests are affordable right now."""
        snap = self.snapshot()
        if snap.provider_exhausted:
            return False
        return snap.estimated_remaining >= max(0, int(calls))


# Phase H.3: Yahoo/SEC have no scarce daily-call allowance the way Alpha
# Vantage's free tier does (Yahoo publishes no quota at all; SEC's only
# published constraint is the 10 req/s pace, handled by the coordinator's
# existing `_pace()`, not by a daily ledger) — but the attempt/success/
# failure/cache-hit counters this class already keeps are still useful
# instrumentation for them. Reusing the class under a provider-neutral alias
# (rather than adding a near-duplicate class) is the "reuse the existing
# quota ledger" instruction applied literally; callers for those providers
# just construct it with a high `daily_limit` so `can_spend` never blocks in
# practice. The class itself and every existing AlphaVantageQuotaLedger
# call site/test are completely unchanged.
QuotaLedger = AlphaVantageQuotaLedger
