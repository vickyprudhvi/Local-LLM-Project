"""Phase H.3 — the SEC EDGAR HTTP client.

An official, documented, keyless US-government JSON API — the opposite risk
profile from Yahoo (see docs/security/YAHOO_SEC_PROVIDER_REVIEW.md §2). No
credential; the only requirement is a descriptive `User-Agent` naming the
application and a real contact, which SEC's own fair-access guidance asks
for and which this project treats with the same "never in a cache key, log,
exception, or snapshot" discipline as `ALPHAVANTAGE_API_KEY`, even though a
User-Agent is not a secret in the same sense a credential is.

Every request goes through the project's existing SSRF-safe fetch path
(`tools/http_safety.py::safe_get`), restricted to exactly the two documented
SEC hosts via `_ALLOWED_HOSTS` below — `safe_get` itself blocks private/
loopback/non-routable IPs but has no domain allowlist, so this file adds one
locally rather than changing that shared module.
"""

import json
import re
from urllib.parse import urlsplit

import requests

import tools.config as config
from finance.cache import normalize_symbol
from finance.datasets import Dataset
from tools.base import ToolFailure
from tools.http_safety import FetchError, read_limited, safe_get
from tools.models import (
    MARKET_DATA_INVALID_RESPONSE,
    MARKET_DATA_INVALID_SYMBOL,
    MARKET_DATA_PROVIDER_ERROR,
    MARKET_DATA_RATE_LIMITED,
    MARKET_DATA_TIMEOUT,
    SEC_CIK_NOT_FOUND,
    SEC_HOST_NOT_ALLOWED,
    SEC_USER_AGENT_MISSING,
)
from finance.provider import ProviderResponse

# The exact two documented SEC hosts this client will ever contact. No other
# host is reachable through this client no matter what a dataset/argument
# says — "restrict SEC requests to approved SEC hosts" as a hard allowlist,
# not just an expectation.
_ALLOWED_HOSTS = frozenset({"www.sec.gov", "data.sec.gov"})

# SEC returns 403 both for a missing/malformed User-Agent AND for a fair-
# access-policy IP block. This client always sends a real User-Agent
# (validated non-empty before the call), so a 403 in practice means the
# latter — classified as a rate-limit signal so the SAME retry/backoff and
# quota-ledger throttle-streak machinery Alpha Vantage already uses applies
# here too, rather than a second bespoke policy.
TRANSIENT_CODES = frozenset({MARKET_DATA_TIMEOUT, MARKET_DATA_PROVIDER_ERROR, MARKET_DATA_RATE_LIMITED})
NON_RETRYABLE_CODES = frozenset({SEC_USER_AGENT_MISSING, SEC_HOST_NOT_ALLOWED,
                                 MARKET_DATA_INVALID_SYMBOL, MARKET_DATA_INVALID_RESPONSE})


# Phase H.4 -- argument validation for the two EDGAR Archives operations.
#
# These are the only SEC operations whose URL contains a caller-supplied path
# SEGMENT rather than just a zero-padded CIK, so they are the only place a
# malformed argument could try to escape the intended path. Both patterns are
# anchored, reject every path/authority metacharacter by construction (no
# "/", "\", ":", "?", "#", "%" or "..") and are applied BEFORE the string is
# ever concatenated into a URL.
_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
# EDGAR document names are flat filenames: letters, digits, dot, underscore,
# hyphen. No directory component is ever legitimate here.
_DOCUMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DOCUMENT_EXTENSIONS = (".htm", ".html", ".txt")

# EDGAR document types this project will read. An exhibit is fetched only to
# extract MANAGEMENT GUIDANCE from an earnings release, so the allowlist is
# the exhibit types that carry one.
GUIDANCE_EXHIBIT_TYPES = ("EX-99.1", "EX-99")


def _validate_accession(value) -> str:
    text = str(value or "").strip()
    if not _ACCESSION_RE.match(text):
        raise ToolFailure(
            MARKET_DATA_INVALID_SYMBOL,
            "An SEC accession number must look like 0000091142-26-000096.")
    return text


def _validate_document_name(value) -> str:
    text = str(value or "").strip()
    if not _DOCUMENT_RE.match(text) or ".." in text:
        raise ToolFailure(
            MARKET_DATA_INVALID_SYMBOL,
            "An SEC filing document name must be a plain EDGAR filename.")
    if not text.lower().endswith(_DOCUMENT_EXTENSIONS):
        raise ToolFailure(
            MARKET_DATA_INVALID_SYMBOL,
            f"Only {', '.join(_DOCUMENT_EXTENSIONS)} filing documents are read.")
    return text


def is_transient(code) -> bool:
    return code in TRANSIENT_CODES


def is_quota_signal(code) -> bool:
    return code == MARKET_DATA_RATE_LIMITED


def is_quota_exhausted(code) -> bool:
    """SEC has no daily allowance to exhaust — only the per-second pace
    matters, handled by the coordinator's _pace(), not by declaring the
    window spent. Always False: a 403 here is retried/backed-off, never
    treated as 'done for today'."""
    return False


def is_negative_cacheable(code) -> bool:
    return code == MARKET_DATA_INVALID_SYMBOL


def _redact(text) -> str:
    """No credential exists to strip here (SEC needs no key), but a
    User-Agent or a raw URL still has no business in a log/exception —
    same discipline as finance/provider.py::_redact, applied defensively."""
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"https?://\S+", "<url>", text)
    return cleaned[:300]


class SecEdgarClient:
    """Bounded, SSRF-safe, host-allowlisted access to the three reviewed SEC
    EDGAR endpoints. Mirrors AlphaVantageClient's fetch(dataset, arguments)
    interface exactly, so the SAME MarketDataRequestCoordinator serves both —
    see finance/coordinator.py's provider_id/dataset_resolver injection."""

    provider_id = "sec"

    def __init__(self, session=None, user_agent_reader=None):
        self._session = session
        self._user_agent_reader = user_agent_reader or config.sec_user_agent

    @property
    def session(self):
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def describe_request(self, dataset: Dataset, arguments: dict) -> dict:
        return {
            "provider": self.provider_id,
            "function": dataset.function,
            "dataset_id": dataset.dataset_id,
            "arguments": {k: v for k, v in sorted((arguments or {}).items())},
        }

    def _build_url(self, dataset: Dataset, arguments: dict) -> str:
        if dataset.function == "company_tickers":
            return "https://www.sec.gov/files/company_tickers.json"
        cik = (arguments or {}).get("cik")
        if not cik:
            raise ToolFailure(SEC_CIK_NOT_FOUND, "A resolved SEC CIK is required for this dataset.")
        cik10 = str(cik).strip().zfill(10)
        if dataset.function == "submissions":
            return f"https://data.sec.gov/submissions/CIK{cik10}.json"
        if dataset.function == "companyfacts":
            return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
        if dataset.function in ("filing_index", "filing_document"):
            # Phase H.4. Both operations address ONE already-accepted filing
            # under the EDGAR Archives path. Every component is validated to a
            # strict pattern before it reaches the URL, so no argument can
            # introduce a traversal segment, a query string, an authority, or
            # a second path root -- the resulting URL is always exactly
            # www.sec.gov/Archives/edgar/data/<digits>/<digits>[/<name>].
            accession = _validate_accession((arguments or {}).get("accession"))
            base = (f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/"
                    f"{accession.replace('-', '')}")
            if dataset.function == "filing_index":
                return f"{base}/index.json"
            document = _validate_document_name((arguments or {}).get("document"))
            return f"{base}/{document}"
        raise ToolFailure(MARKET_DATA_PROVIDER_ERROR, f"Unsupported SEC operation {dataset.function!r}.")

    def fetch(self, dataset: Dataset, arguments: dict) -> ProviderResponse:
        user_agent = self._user_agent_reader()
        if not user_agent or not user_agent.strip():
            raise ToolFailure(
                SEC_USER_AGENT_MISSING,
                "SEC EDGAR access requires SEC_USER_AGENT to be configured "
                "(an application name and contact, per SEC's fair-access policy).",
            )

        url = self._build_url(dataset, arguments)
        host = urlsplit(url).hostname
        if host not in _ALLOWED_HOSTS:
            raise ToolFailure(SEC_HOST_NOT_ALLOWED, f"{host!r} is not an approved SEC host.")

        try:
            response, _final_url = safe_get(
                url,
                session=self.session,
                user_agent=user_agent,
                connect_timeout=config.market_data_connect_timeout(),
                read_timeout=config.market_data_read_timeout(),
                max_redirects=0,
                allow_http=False,
            )
        except FetchError as e:
            code = MARKET_DATA_TIMEOUT if "timed out" in e.message.lower() else MARKET_DATA_PROVIDER_ERROR
            raise ToolFailure(code, _redact(e.message) or "The SEC EDGAR request failed.",
                              retryable=code in TRANSIENT_CODES)
        except requests.exceptions.RequestException as e:
            raise ToolFailure(MARKET_DATA_PROVIDER_ERROR,
                              f"The SEC EDGAR request failed ({type(e).__name__}).", retryable=True)

        status = response.status_code
        if status == 404:
            response.close()
            raise ToolFailure(MARKET_DATA_INVALID_SYMBOL, "SEC EDGAR returned no data for that CIK.")
        if status in (403, 429):
            response.close()
            raise ToolFailure(MARKET_DATA_RATE_LIMITED,
                              "SEC EDGAR is rate limiting or temporarily blocking requests.")
        if status >= 500:
            response.close()
            raise ToolFailure(MARKET_DATA_PROVIDER_ERROR,
                              f"SEC EDGAR returned status {status}.", retryable=True)
        if status >= 400:
            response.close()
            raise ToolFailure(MARKET_DATA_PROVIDER_ERROR, f"SEC EDGAR returned status {status}.")

        try:
            body, byte_count = read_limited(response, config.max_market_data_bytes())
        except FetchError as e:
            raise ToolFailure(MARKET_DATA_PROVIDER_ERROR,
                              _redact(e.message) or "Reading the SEC EDGAR response failed.")

        if dataset.function == "filing_document":
            # A filed exhibit is HTML or plain text, not JSON. It is wrapped in
            # a dict so the payload shape stays uniform for the shared
            # coordinator/cache, and it is explicitly UNTRUSTED CONTENT: it is
            # a document the issuer wrote, so nothing downstream may treat it
            # as instructions. finance/guidance.py only ever pattern-matches
            # numbers out of it.
            return ProviderResponse(
                payload={
                    "document_text": body.decode("utf-8", errors="replace"),
                    "byte_count": byte_count,
                    "accession": (arguments or {}).get("accession"),
                    "document": (arguments or {}).get("document"),
                    "truncated": byte_count >= config.max_market_data_bytes(),
                },
                provider_metadata={"provider_function": dataset.function},
                byte_count=byte_count)

        try:
            payload = json.loads(body.decode("utf-8", errors="replace"))
        except (ValueError, UnicodeDecodeError):
            raise ToolFailure(MARKET_DATA_INVALID_RESPONSE, "SEC EDGAR returned an unparseable response.")
        if not isinstance(payload, dict):
            raise ToolFailure(MARKET_DATA_INVALID_RESPONSE, "SEC EDGAR returned an unexpected response shape.")

        return ProviderResponse(payload=payload,
                                provider_metadata={"provider_function": dataset.function},
                                byte_count=byte_count)


_GLOBAL_LOOKUP_SYMBOL = "GLOBAL"


def resolve_cik(coordinator, ticker: str):
    """Ticker -> (cik_10digit, company_name), via the SAME cache/coordinator
    path every other dataset uses.

    Always requests the ticker_cik_map dataset under the CONSTANT symbol
    "GLOBAL" — the file is the same ~800KB blob regardless of which ticker is
    being resolved, so resolving the first ticker of a run fetches+caches it
    ONCE (7-day TTL); resolving every ticker after that, including a
    different one, is a pure cache hit with zero external calls.
    """
    outcome = coordinator.fetch("ticker_cik_map", _GLOBAL_LOOKUP_SYMBOL)
    normalized = normalize_symbol(ticker)
    for entry in (outcome.payload or {}).values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("ticker", "")).strip().upper() == normalized:
            cik = str(entry.get("cik_str", "")).strip().zfill(10)
            return cik, entry.get("title")
    raise ToolFailure(SEC_CIK_NOT_FOUND, f"No SEC CIK mapping found for ticker {ticker!r}.")
