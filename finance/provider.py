"""Phase H.1 — the Alpha Vantage HTTP client.

A trusted LOCAL provider rather than an MCP server. Both candidate Alpha Vantage
MCP servers were reviewed and rejected; see
`docs/security/ALPHAVANTAGE_PROVIDER_REVIEW.md` for the evidence.

Every request goes through the project's existing SSRF-safe fetch path
(`tools/http_safety.py`), so scheme, port, credential-in-URL, DNS and private-IP
rules are enforced identically to `browser.fetch_page` — including on redirects.

Credential handling:

* The key is read from config at call time and placed only in the query string of
  the outbound request.
* The constructed URL is never returned, logged, cached, or attached to an
  exception. `_redact` is applied to anything derived from it.
* `describe_request()` exists so callers can log WHAT was requested without ever
  holding the URL that carries the key.

Alpha Vantage signals failure with HTTP 200 plus a JSON body containing an
"Error Message", "Note", or "Information" field. Those are classified here into
controlled, deterministically-triggered error codes, and — critically — into
retryable vs. non-retryable, because retrying a quota error just burns the
remaining quota faster.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlencode

import requests

import tools.config as config
from finance.datasets import Dataset
from tools.base import ToolFailure
from tools.http_safety import FetchError, read_limited, safe_get
from tools.models import (
    MARKET_DATA_API_KEY_MISSING,
    MARKET_DATA_AUTHENTICATION_FAILED,
    MARKET_DATA_ENTITLEMENT_REQUIRED,
    MARKET_DATA_INVALID_RESPONSE,
    MARKET_DATA_INVALID_SYMBOL,
    MARKET_DATA_PROVIDER_ERROR,
    MARKET_DATA_QUOTA_EXCEEDED,
    MARKET_DATA_RATE_LIMITED,
    MARKET_DATA_TIMEOUT,
)

# Failures worth a bounded retry: the request never reached a decision, or the
# condition clears on its own.
#
# MARKET_DATA_RATE_LIMITED is a PER-SECOND burst throttle ("1 request per
# second" on the free tier), which clears in about a second — genuinely
# transient, and distinct from MARKET_DATA_QUOTA_EXCEEDED (the daily allowance,
# which does not clear until the window resets and is never retried).
TRANSIENT_CODES = frozenset({
    MARKET_DATA_TIMEOUT,
    MARKET_DATA_PROVIDER_ERROR,
    MARKET_DATA_RATE_LIMITED,
})

# Failures that are DETERMINISTIC provider rejections — retrying cannot help and,
# for the quota case, actively harms. Never retried.
NON_RETRYABLE_CODES = frozenset({
    MARKET_DATA_API_KEY_MISSING,
    MARKET_DATA_AUTHENTICATION_FAILED,
    MARKET_DATA_INVALID_SYMBOL,
    MARKET_DATA_ENTITLEMENT_REQUIRED,
    MARKET_DATA_QUOTA_EXCEEDED,
    MARKET_DATA_INVALID_RESPONSE,
})

# Alpha Vantage uses TWO DIFFERENT messages for two different conditions, both
# mentioning "25 requests per day" — the daily number alone cannot distinguish
# them. Both wordings below are captured VERBATIM from live responses (2026-08),
# not guessed:
#
#   throttle  "Thank you for using Alpha Vantage! Please consider spreading
#              out your free API requests more sparingly (1 request per
#              second). You may subscribe to any of the premium plans at ...
#              to lift the free key rate limit (25 requests per day), raise
#              the per-second burst limit, and instantly unlock all premium
#              endpoints."                            <- transient, recovers
#   exhausted "We have detected your API key as <KEY> and our standard API
#              rate limit is 25 requests per day. Please subscribe to any of
#              the premium plans at ... to instantly remove all daily rate
#              limits."                                <- done for the window
#
# Note the exhausted message contains NO "exceeded"/"exhaust"/"reached" verb —
# an earlier version of this pattern required one and so misclassified genuine
# daily exhaustion as a retryable throttle. The lead-in phrase and closing
# phrase below are each distinctive to ONE of the two messages.
_QUOTA_RE = re.compile(
    r"we have detected your api key"
    r"|instantly remove all daily rate limits"
    r"|(?:exceeded|exhaust|reached|used up|no (?:more|remaining))"
    r"[^.]{0,60}(?:daily|per day|rate limit)"
    r"|daily (?:quota|limit) (?:has been )?(?:reached|exceeded)",
    re.IGNORECASE)
_THROTTLE_RE = re.compile(
    r"(spreading out|more sparingly|per second|call frequency|"
    r"requests per minute|higher API call frequency)", re.IGNORECASE)
_RATE_LIMIT_RE = re.compile(
    r"(rate limit|requests per (minute|day)|higher API call)", re.IGNORECASE)
_PREMIUM_RE = re.compile(r"(premium|subscribe|paid plan|entitle)", re.IGNORECASE)
_INVALID_SYMBOL_RE = re.compile(r"(invalid api call|invalid.{0,20}symbol)", re.IGNORECASE)


@dataclass(frozen=True)
class ProviderResponse:
    """A validated provider payload plus safe, non-sensitive metadata."""

    payload: dict
    provider_metadata: dict
    byte_count: int


def _redact(text, secret=None) -> str:
    """Strip anything that could carry the credential out of a message.

    Applied to every provider-derived string that could reach a log, an
    exception, or the local LLM. `secret`, when given, is the EXACT credential
    in use for this call and is stripped by literal match FIRST — Alpha
    Vantage's own daily-exhaustion message embeds the raw key as plain text
    ("We have detected your API key as <KEY> and our standard API rate limit
    is..."), a shape the apikey=/URL patterns below do not cover. Regex
    patterns alone are not sufficient; every caller that has the key in scope
    must pass it here.
    """
    if not isinstance(text, str):
        return ""
    cleaned = text
    if secret:
        cleaned = cleaned.replace(secret, "REDACTED")
    cleaned = re.sub(r"(?i)(apikey|api_key|token)=[^&\s\"']+", r"\1=REDACTED", cleaned)
    cleaned = re.sub(r"https?://\S+", "<url>", cleaned)
    return cleaned[:300]


class AlphaVantageClient:
    """Bounded, SSRF-safe access to the reviewed Alpha Vantage endpoints."""

    provider_id = "alphavantage"

    def __init__(self, session=None, api_key_reader=None, endpoint=None):
        self._session = session
        # Injected so tests never need a real credential and so the key is
        # resolved fresh on each call rather than captured at construction.
        self._api_key_reader = api_key_reader or config.alphavantage_api_key
        self._endpoint = endpoint

    @property
    def session(self):
        if self._session is None:
            self._session = requests.Session()
        return self._session

    @property
    def endpoint(self):
        return self._endpoint or config.alphavantage_endpoint()

    def describe_request(self, dataset: Dataset, arguments: dict) -> dict:
        """What a call WOULD request — safe to log. Contains no credential."""
        return {
            "provider": self.provider_id,
            "function": dataset.function,
            "dataset_id": dataset.dataset_id,
            "arguments": {k: v for k, v in sorted((arguments or {}).items())},
        }

    def _build_url(self, dataset: Dataset, arguments: dict, api_key: str) -> str:
        """Construct the outbound URL. NEVER returned to a caller or logged."""
        params = {"function": dataset.function}
        for name in dataset.argument_names:
            value = arguments.get(name, dataset.defaults.get(name))
            if value in (None, ""):
                continue
            params[name] = str(value)
        params["apikey"] = api_key
        return f"{self.endpoint}?{urlencode(params)}"

    def fetch(self, dataset: Dataset, arguments: dict) -> ProviderResponse:
        """Make ONE external request and return a validated payload.

        Raises ToolFailure with a controlled code on any problem. The exception
        message never contains the URL or the credential.
        """
        api_key = self._api_key_reader()
        if not api_key:
            raise ToolFailure(
                MARKET_DATA_API_KEY_MISSING,
                "Market data is unavailable: ALPHAVANTAGE_API_KEY is not configured.",
            )

        url = self._build_url(dataset, arguments, api_key)
        try:
            response, _final_url = safe_get(
                url,
                session=self.session,
                user_agent=config.http_user_agent(),
                connect_timeout=config.market_data_connect_timeout(),
                read_timeout=config.market_data_read_timeout(),
                max_redirects=0,  # the documented endpoint never legitimately redirects
                allow_http=False,
            )
        except FetchError as e:
            # Re-raised with a market-data code so the retry classifier and the
            # quota ledger see one vocabulary. The message is already safe, but
            # redact defensively (including the literal key) in case a URL was
            # interpolated upstream.
            code = MARKET_DATA_TIMEOUT if "timed out" in e.message.lower() else MARKET_DATA_PROVIDER_ERROR
            raise ToolFailure(code, _redact(e.message, secret=api_key) or "The market-data request failed.",
                              retryable=code in TRANSIENT_CODES)
        except requests.exceptions.RequestException as e:
            raise ToolFailure(MARKET_DATA_PROVIDER_ERROR,
                              f"The market-data request failed ({type(e).__name__}).",
                              retryable=True)

        status = response.status_code
        if status in (401, 403):
            response.close()
            raise ToolFailure(MARKET_DATA_AUTHENTICATION_FAILED,
                              "Market-data authentication failed.")
        if status == 429:
            response.close()
            raise ToolFailure(MARKET_DATA_RATE_LIMITED,
                              "The market-data provider is rate limiting requests.")
        if status >= 500:
            response.close()
            raise ToolFailure(MARKET_DATA_PROVIDER_ERROR,
                              f"The market-data provider returned status {status}.",
                              retryable=True)
        if status >= 400:
            response.close()
            raise ToolFailure(MARKET_DATA_PROVIDER_ERROR,
                              f"The market-data provider returned status {status}.")

        try:
            body, byte_count = read_limited(response, config.max_market_data_bytes())
        except FetchError as e:
            raise ToolFailure(MARKET_DATA_PROVIDER_ERROR,
                              _redact(e.message, secret=api_key)
                              or "Reading the market-data response failed.")

        return self._validate(body, byte_count, dataset, api_key)

    # ---- validation ----

    def _validate(self, body: bytes, byte_count: int, dataset: Dataset,
                  api_key: str = None) -> ProviderResponse:
        """Turn raw bytes into a validated payload, or a controlled failure.

        An unparseable body, a non-object body, or an Alpha Vantage in-band error
        is NEVER returned as a success — so it can never be cached as one.
        `api_key` is passed through only for redaction (see `_redact`); it is
        never placed in the payload, the cache, or any returned metadata.
        """
        try:
            payload = json.loads(body.decode("utf-8", errors="replace"))
        except (ValueError, UnicodeDecodeError):
            raise ToolFailure(MARKET_DATA_INVALID_RESPONSE,
                              "The market-data provider returned an unparseable response.")
        if not isinstance(payload, dict):
            raise ToolFailure(MARKET_DATA_INVALID_RESPONSE,
                              "The market-data provider returned an unexpected response shape.")

        self._raise_for_inband_error(payload, secret=api_key)

        if not payload:
            # Alpha Vantage returns `{}` for an unknown symbol on several
            # fundamental endpoints. That is a CONTROLLED negative result.
            raise ToolFailure(MARKET_DATA_INVALID_SYMBOL,
                              "The provider returned no data for that symbol.")

        metadata = {}
        for key in ("Meta Data", "metaData"):
            if isinstance(payload.get(key), dict):
                metadata = {str(k): _redact(str(v), secret=api_key)
                           for k, v in payload[key].items()}
                break
        metadata["provider_function"] = dataset.function
        return ProviderResponse(payload=payload, provider_metadata=metadata,
                                byte_count=byte_count)

    @staticmethod
    def _raise_for_inband_error(payload: dict, secret: str = None):
        """Alpha Vantage reports failure with HTTP 200 + a message field.

        Ordered most-specific first, and each branch maps to a deterministic code
        so the retry policy and the quota ledger both behave predictably.
        `secret`, when given, is stripped from the message before it is placed
        in the raised exception (see `_redact`).
        """
        message = None
        for key in ("Error Message", "Note", "Information"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                message = value.strip()
                break
        if message is None:
            return

        safe = _redact(message, secret=secret)
        # Order matters: an explicit exhaustion phrase wins, but a throttle
        # advisory is checked BEFORE the generic rate-limit pattern so the
        # daily allowance merely being quoted in it cannot escalate to QUOTA.
        if _QUOTA_RE.search(message):
            raise ToolFailure(MARKET_DATA_QUOTA_EXCEEDED,
                              f"The market-data daily quota is exhausted: {safe}")
        if _THROTTLE_RE.search(message) or _RATE_LIMIT_RE.search(message):
            raise ToolFailure(MARKET_DATA_RATE_LIMITED,
                              f"The market-data provider is rate limiting requests: {safe}")
        if _PREMIUM_RE.search(message):
            raise ToolFailure(MARKET_DATA_ENTITLEMENT_REQUIRED,
                              f"That endpoint needs a premium entitlement: {safe}")
        if _INVALID_SYMBOL_RE.search(message):
            raise ToolFailure(MARKET_DATA_INVALID_SYMBOL,
                              f"The provider rejected that symbol or request: {safe}")
        raise ToolFailure(MARKET_DATA_PROVIDER_ERROR,
                          f"The market-data provider reported a problem: {safe}")


def is_transient(code) -> bool:
    """Whether a failure code may be retried. Fails closed: anything unknown is
    treated as NON-retryable so a new code never silently becomes retry-eligible."""
    return code in TRANSIENT_CODES


def is_quota_signal(code) -> bool:
    """A provider-authoritative quota/throttle signal worth recording on the
    ledger. Whether it STOPS the window is `is_quota_exhausted`; whether it may
    be retried is `is_transient`."""
    return code in (MARKET_DATA_QUOTA_EXCEEDED, MARKET_DATA_RATE_LIMITED)


def is_quota_exhausted(code) -> bool:
    """Whether the signal means the RESET WINDOW is spent, as opposed to a
    transient throttle. Only this closes the ledger's window."""
    return code == MARKET_DATA_QUOTA_EXCEEDED


def is_negative_cacheable(code) -> bool:
    """Only a provider-CONFIRMED invalid symbol is negatively cached. A transport
    failure must never poison the cache."""
    return code == MARKET_DATA_INVALID_SYMBOL
