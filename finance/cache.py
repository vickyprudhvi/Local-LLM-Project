"""Phase H.1 — the quota-aware market-data cache.

A standard-library SQLite store that sits in FRONT of the Alpha Vantage provider.
Every lookup happens before a provider call is even considered, so a fresh hit
costs zero external quota.

Design rules this file enforces:

* The cache key is a SHA-256 over canonical JSON of (provider, dataset, function,
  normalized symbol, normalized arguments, freshness mode, schema version). The
  API key is NEVER an input — a rotated credential must not invalidate the cache,
  and a cache file must never be able to leak one.
* A record is only written after the payload has been validated. A malformed or
  incomplete provider response is never stored as a successful result.
* Every stored payload carries its own SHA-256. A row whose payload no longer
  hashes to its recorded digest is treated as corrupt, deleted, and reported as a
  miss rather than returned.
* Reads and writes go through short transactions; writes are `INSERT OR REPLACE`
  inside a transaction so a concurrent reader never sees a half-written row.
* All timestamps are UTC epoch seconds. Presentation-layer conversion to the
  user's timezone happens in the workflow, never here.
* The clock is injected so expiry, staleness, and TTL behaviour are deterministic
  in tests.
"""

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional

import tools.config as config
from finance.datasets import CACHE_SCHEMA_VERSION

PROVIDER_ID = "alphavantage"

# Bumped independently of CACHE_SCHEMA_VERSION when the TABLE shape changes.
_DB_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_data (
    cache_key           TEXT PRIMARY KEY,
    provider            TEXT NOT NULL,
    dataset_id          TEXT NOT NULL,
    provider_function   TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    normalized_arguments TEXT NOT NULL,
    retrieved_at        INTEGER NOT NULL,
    expires_at          INTEGER NOT NULL,
    freshness           TEXT NOT NULL,
    schema_version      INTEGER NOT NULL,
    payload_hash        TEXT NOT NULL,
    payload             TEXT NOT NULL,
    origin              TEXT NOT NULL,
    provider_metadata   TEXT NOT NULL DEFAULT '{}',
    negative            INTEGER NOT NULL DEFAULT 0,
    negative_code       TEXT
);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_expires ON market_data(expires_at);
CREATE TABLE IF NOT EXISTS cache_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class CacheStatus:
    FRESH = "fresh"
    STALE = "stale"
    MISS = "miss"
    EXPIRED = "expired"
    NEGATIVE = "negative"
    CORRUPT = "corrupt"


class Origin:
    PROVIDER = "provider"
    CACHE = "cache"


def utc_now() -> int:
    """Injected everywhere as the default clock. UTC epoch seconds."""
    return int(time.time())


def normalize_symbol(symbol) -> str:
    """Uppercase, trimmed ticker. Deterministic so 'aapl' and ' AAPL ' collide."""
    if not isinstance(symbol, str):
        return ""
    return symbol.strip().upper()


def canonical_arguments(arguments) -> dict:
    """Lowercase keys, string values, sorted — so argument ORDER never changes
    the cache key while a different interval or output size always does."""
    if not isinstance(arguments, dict):
        return {}
    out = {}
    for key in sorted(arguments):
        value = arguments[key]
        if value is None:
            continue
        out[str(key).strip().lower()] = str(value).strip()
    return out


def build_cache_key(dataset_id, provider_function, symbol, arguments, freshness,
                    schema_version=CACHE_SCHEMA_VERSION, provider=PROVIDER_ID) -> str:
    """Deterministic SHA-256 cache key.

    Deliberately excludes the API key: the credential is an authentication
    detail, not part of the identity of the data, and must never be derivable
    from a cache file.
    """
    material = {
        "provider": provider,
        "dataset": dataset_id,
        "function": provider_function,
        "symbol": normalize_symbol(symbol),
        "arguments": canonical_arguments(arguments),
        "freshness": freshness,
        "schema_version": int(schema_version),
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def payload_digest(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheRecord:
    """One cached provider response plus everything needed to describe its age."""

    cache_key: str
    provider: str
    dataset_id: str
    provider_function: str
    symbol: str
    normalized_arguments: dict
    retrieved_at: int
    expires_at: int
    freshness: str
    schema_version: int
    payload_hash: str
    payload: dict
    origin: str
    provider_metadata: dict
    negative: bool = False
    negative_code: Optional[str] = None

    def age_seconds(self, now) -> int:
        return max(0, int(now) - int(self.retrieved_at))

    def is_expired(self, now) -> bool:
        return int(now) >= int(self.expires_at)

    def describe(self, now) -> dict:
        """Non-sensitive freshness description for the report and the LLM.

        Never includes the payload, a URL, or a credential.
        """
        return {
            "provider": self.provider,
            "dataset_id": self.dataset_id,
            "symbol": self.symbol,
            "retrieved_at_utc": _iso(self.retrieved_at),
            "expires_at_utc": _iso(self.expires_at),
            "age_seconds": self.age_seconds(now),
            "expired": self.is_expired(now),
            "source_freshness": self.freshness,
            "origin": self.origin,
            "schema_version": self.schema_version,
        }


def _iso(epoch) -> str:
    import datetime

    return datetime.datetime.fromtimestamp(int(epoch), datetime.timezone.utc).isoformat()


@dataclass(frozen=True)
class CacheLookup:
    """The outcome of one cache read."""

    status: str
    record: Optional[CacheRecord] = None

    @property
    def hit(self) -> bool:
        return self.status in (CacheStatus.FRESH, CacheStatus.NEGATIVE)


class MarketDataCache:
    """SQLite-backed cache with per-key single-flight locking.

    `clock` is injected (defaults to `utc_now`) so TTL, expiry and stale-if-error
    behaviour are exactly reproducible in tests.
    """

    def __init__(self, path=None, clock=None, base_dir=None):
        self._clock = clock or utc_now
        self._path = self._resolve_path(path, base_dir)
        self._locks = {}
        self._locks_guard = threading.Lock()
        self._connection_guard = threading.Lock()
        self._ensure_schema()

    # ---- location / schema ----

    @staticmethod
    def _resolve_path(path, base_dir):
        path = path or config.market_data_cache_path()
        if path == ":memory:":
            return path
        if not os.path.isabs(path):
            root = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(root, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    @property
    def path(self) -> str:
        return self._path

    def _connect(self):
        conn = sqlite3.connect(self._path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_schema(self):
        with self._connection_guard, self._connect() as conn:
            conn.executescript(_SCHEMA)
            row = conn.execute(
                "SELECT value FROM cache_meta WHERE key = 'db_schema_version'").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO cache_meta(key, value) VALUES ('db_schema_version', ?)",
                    (str(_DB_SCHEMA_VERSION),))
            elif str(row["value"]) != str(_DB_SCHEMA_VERSION):
                # Migration policy: drop and rebuild. Cached market data is always
                # re-fetchable, so invalidating is strictly safer than guessing how
                # to reinterpret rows written under a different contract.
                conn.execute("DROP TABLE IF EXISTS market_data")
                conn.executescript(_SCHEMA)
                conn.execute(
                    "UPDATE cache_meta SET value = ? WHERE key = 'db_schema_version'",
                    (str(_DB_SCHEMA_VERSION),))

    # ---- single-flight ----

    def lock_for(self, cache_key) -> threading.Lock:
        """The per-key lock that makes concurrent identical requests collapse
        into ONE external provider call. Callers must re-check the cache after
        acquiring it — another thread may have populated it while they waited."""
        with self._locks_guard:
            lock = self._locks.get(cache_key)
            if lock is None:
                lock = threading.Lock()
                self._locks[cache_key] = lock
            return lock

    # ---- reads ----

    def get(self, cache_key, allow_stale=False, stale_grace_seconds=None) -> CacheLookup:
        """Read one record.

        Returns FRESH only while within TTL. An expired record is reported as
        EXPIRED (and carried on the lookup) so the caller can still choose to
        serve it under an explicit stale-if-error policy — it is never silently
        presented as fresh.
        """
        now = self._clock()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM market_data WHERE cache_key = ?", (cache_key,)).fetchone()
        if row is None:
            return CacheLookup(CacheStatus.MISS)

        try:
            record = self._row_to_record(row)
        except (ValueError, TypeError, json.JSONDecodeError):
            self.delete(cache_key)
            return CacheLookup(CacheStatus.CORRUPT)

        # Integrity: a payload that no longer matches its digest is corrupt.
        if payload_digest(record.payload) != record.payload_hash:
            self.delete(cache_key)
            return CacheLookup(CacheStatus.CORRUPT)

        if record.schema_version != CACHE_SCHEMA_VERSION:
            self.delete(cache_key)
            return CacheLookup(CacheStatus.MISS)

        if not record.is_expired(now):
            status = CacheStatus.NEGATIVE if record.negative else CacheStatus.FRESH
            return CacheLookup(status, record)

        if allow_stale:
            grace = (config.market_data_stale_if_error_seconds()
                     if stale_grace_seconds is None else stale_grace_seconds)
            if record.age_seconds(now) <= int(grace):
                return CacheLookup(CacheStatus.STALE, record)
        return CacheLookup(CacheStatus.EXPIRED, record)

    @staticmethod
    def _row_to_record(row) -> CacheRecord:
        return CacheRecord(
            cache_key=row["cache_key"],
            provider=row["provider"],
            dataset_id=row["dataset_id"],
            provider_function=row["provider_function"],
            symbol=row["symbol"],
            normalized_arguments=json.loads(row["normalized_arguments"]),
            retrieved_at=int(row["retrieved_at"]),
            expires_at=int(row["expires_at"]),
            freshness=row["freshness"],
            schema_version=int(row["schema_version"]),
            payload_hash=row["payload_hash"],
            payload=json.loads(row["payload"]),
            origin=row["origin"],
            provider_metadata=json.loads(row["provider_metadata"] or "{}"),
            negative=bool(row["negative"]),
            negative_code=row["negative_code"],
        )

    # ---- writes ----

    def put(self, cache_key, dataset_id, provider_function, symbol, arguments, payload,
            ttl_seconds, freshness, provider_metadata=None, origin=Origin.PROVIDER,
            negative=False, negative_code=None, provider=PROVIDER_ID) -> CacheRecord:
        """Store one VALIDATED payload atomically.

        The caller is responsible for having validated `payload` first — see
        `finance/provider.py`. This method refuses a non-dict payload outright so
        a malformed response cannot become a successful cache entry.

        `provider` defaults to the original Alpha-Vantage-only identity for
        100% backward compatibility with every existing call site; a
        multi-provider caller (`finance/coordinator.py`) passes its own
        provider id explicitly so a row is never mislabeled — this is the
        other half of `build_cache_key`'s existing `provider` parameter,
        which the cache KEY already supported but this write path did not.
        """
        if not isinstance(payload, dict):
            raise ValueError("A cache payload must be a JSON object.")
        now = self._clock()
        record = CacheRecord(
            cache_key=cache_key,
            provider=provider,
            dataset_id=dataset_id,
            provider_function=provider_function,
            symbol=normalize_symbol(symbol),
            normalized_arguments=canonical_arguments(arguments),
            retrieved_at=now,
            expires_at=now + max(0, int(ttl_seconds)),
            freshness=freshness,
            schema_version=CACHE_SCHEMA_VERSION,
            payload_hash=payload_digest(payload),
            payload=payload,
            origin=origin,
            provider_metadata=dict(provider_metadata or {}),
            negative=bool(negative),
            negative_code=negative_code,
        )
        with self._connection_guard, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO market_data
                   (cache_key, provider, dataset_id, provider_function, symbol,
                    normalized_arguments, retrieved_at, expires_at, freshness,
                    schema_version, payload_hash, payload, origin, provider_metadata,
                    negative, negative_code)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.cache_key, record.provider, record.dataset_id,
                    record.provider_function, record.symbol,
                    json.dumps(record.normalized_arguments, sort_keys=True),
                    record.retrieved_at, record.expires_at, record.freshness,
                    record.schema_version, record.payload_hash,
                    json.dumps(record.payload, sort_keys=True, default=str),
                    record.origin,
                    json.dumps(record.provider_metadata, sort_keys=True, default=str),
                    1 if record.negative else 0, record.negative_code,
                ),
            )
        return record

    def delete(self, cache_key) -> bool:
        with self._connection_guard, self._connect() as conn:
            cur = conn.execute("DELETE FROM market_data WHERE cache_key = ?", (cache_key,))
            return cur.rowcount > 0

    def clear(self) -> int:
        with self._connection_guard, self._connect() as conn:
            cur = conn.execute("DELETE FROM market_data")
            return cur.rowcount

    # ---- introspection (used for quota planning) ----

    def status_for(self, cache_key) -> str:
        return self.get(cache_key).status

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) AS c FROM market_data").fetchone()["c"])
