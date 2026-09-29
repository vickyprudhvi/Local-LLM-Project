"""Phase H.3 — finance/sec_provider.py: the SEC EDGAR HTTP client.

Mocked at the same requests.Session-compatible boundary
tests/test_browser_fetch.py already establishes for tools/http_safety.py,
plus a mocked DNS resolution so the real SSRF guard runs unmodified against
a fake public IP. See docs/security/YAHOO_SEC_PROVIDER_REVIEW.md for the
real, live-network verification of the actual SEC EDGAR API shapes this
file's fakes are modeled on.
"""

import json

import pytest

import tools.http_safety as hs
from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.quota import QuotaLedger
from finance.sec_datasets import resolve_sec_dataset
from finance.sec_provider import SecEdgarClient, resolve_cik
from tools.base import ToolFailure
from tools.models import (
    MARKET_DATA_INVALID_SYMBOL,
    MARKET_DATA_INVALID_RESPONSE,
    MARKET_DATA_RATE_LIMITED,
    SEC_CIK_NOT_FOUND,
    SEC_HOST_NOT_ALLOWED,
    SEC_USER_AGENT_MISSING,
)


class FakeResp:
    def __init__(self, status_code=200, body=b"{}", headers=None):
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "application/json"}
        self._body = body
        self.closed = False

    def iter_content(self, chunk_size=1):
        yield self._body

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self._responses.pop(0)


TICKER_MAP_PAYLOAD = json.dumps({
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
}).encode()

SUBMISSIONS_PAYLOAD = json.dumps({
    "name": "Apple Inc.",
    "filings": {"recent": {
        "form": ["10-K", "8-K"],
        "filingDate": ["2025-10-31", "2025-10-30"],
        "accessionNumber": ["0000320193-25-000079", "0000320193-25-000078"],
        "reportDate": ["2025-09-27", "2025-10-30"],
        "primaryDocument": ["aapl-10k.htm", "aapl-8k.htm"],
    }},
}).encode()


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    monkeypatch.setattr(hs, "_resolve", lambda host: ["93.184.216.34"])


@pytest.fixture
def user_agent_reader():
    return lambda: "TestApp/1.0 (contact: test@example.com)"


# ---- credential/config gating ----

def test_missing_user_agent_fails_closed():
    client = SecEdgarClient(session=FakeSession([]), user_agent_reader=lambda: None)
    with pytest.raises(ToolFailure) as excinfo:
        client.fetch(resolve_sec_dataset("ticker_cik_map"), {})
    assert excinfo.value.code == SEC_USER_AGENT_MISSING


def test_blank_user_agent_fails_closed():
    client = SecEdgarClient(session=FakeSession([]), user_agent_reader=lambda: "   ")
    with pytest.raises(ToolFailure) as excinfo:
        client.fetch(resolve_sec_dataset("ticker_cik_map"), {})
    assert excinfo.value.code == SEC_USER_AGENT_MISSING


# ---- URL building / host allowlist ----

def test_ticker_cik_map_url():
    client = SecEdgarClient()
    url = client._build_url(resolve_sec_dataset("ticker_cik_map"), {})
    assert url == "https://www.sec.gov/files/company_tickers.json"


def test_submissions_url_uses_zero_padded_cik():
    client = SecEdgarClient()
    url = client._build_url(resolve_sec_dataset("company_submissions"), {"cik": "320193"})
    assert url == "https://data.sec.gov/submissions/CIK0000320193.json"


def test_company_facts_url_uses_zero_padded_cik():
    client = SecEdgarClient()
    url = client._build_url(resolve_sec_dataset("company_facts"), {"cik": 320193})
    assert url == "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"


def test_missing_cik_for_a_dataset_that_needs_one_fails_closed():
    client = SecEdgarClient()
    with pytest.raises(ToolFailure) as excinfo:
        client._build_url(resolve_sec_dataset("company_submissions"), {})
    assert excinfo.value.code == SEC_CIK_NOT_FOUND


# ---- Phase H.4: the EDGAR Archives document surface ----
#
# These are the only SEC operations whose URL contains a caller-supplied path
# SEGMENT rather than just a zero-padded CIK, so they are the only place a
# malformed argument could try to escape the intended path. Every one of the
# rejections below is enforced BEFORE the string is concatenated into a URL.

def test_filing_index_url_is_pinned_to_one_accession():
    client = SecEdgarClient()
    url = client._build_url(resolve_sec_dataset("filing_index"),
                            {"cik": "91142", "accession": "0000091142-26-000096"})
    assert url == ("https://www.sec.gov/Archives/edgar/data/91142/"
                   "000009114226000096/index.json")


def test_filing_document_url_is_pinned_to_one_document_in_one_accession():
    client = SecEdgarClient()
    url = client._build_url(resolve_sec_dataset("filing_document"),
                            {"cik": "91142", "accession": "0000091142-26-000096",
                             "document": "a6302026exhibit991.htm"})
    assert url == ("https://www.sec.gov/Archives/edgar/data/91142/"
                   "000009114226000096/a6302026exhibit991.htm")


@pytest.mark.parametrize("accession", [
    "", "not-an-accession", "0000091142-26-00009", "../../../etc/passwd",
    "0000091142-26-000096/../../other", "0000091142%2D26%2D000096",
    "0000091142-26-000096?x=1", "0000091142-26-000096#frag",
])
def test_a_malformed_accession_never_reaches_a_url(accession):
    client = SecEdgarClient()
    with pytest.raises(ToolFailure):
        client._build_url(resolve_sec_dataset("filing_index"),
                          {"cik": "91142", "accession": accession})


@pytest.mark.parametrize("document", [
    "", "../../../../etc/passwd", "sub/dir/file.htm", "..\\windows\\win.ini",
    "file.htm?x=1", "file.htm#frag", "evil.example.com/a.htm",
    "//evil.example.com/a.htm", "a..b/../c.htm",
    "exhibit.exe", "exhibit.js", "exhibit",          # wrong / missing extension
    "a" * 200 + ".htm",                                # implausibly long
])
def test_a_malformed_document_name_never_reaches_a_url(document):
    client = SecEdgarClient()
    with pytest.raises(ToolFailure):
        client._build_url(resolve_sec_dataset("filing_document"),
                          {"cik": "91142", "accession": "0000091142-26-000096",
                           "document": document})


def test_the_archives_surface_stays_on_an_approved_host():
    """Whatever the arguments, the host is a literal in the format string —
    there is no argument through which another host can be reached."""
    from urllib.parse import urlsplit

    from finance.sec_provider import _ALLOWED_HOSTS

    client = SecEdgarClient()
    for dataset_id, arguments in (
        ("filing_index", {"cik": "91142", "accession": "0000091142-26-000096"}),
        ("filing_document", {"cik": "91142", "accession": "0000091142-26-000096",
                             "document": "x.htm"}),
    ):
        url = client._build_url(resolve_sec_dataset(dataset_id), arguments)
        assert urlsplit(url).hostname in _ALLOWED_HOSTS


def test_only_the_two_approved_sec_hosts_are_ever_reachable(user_agent_reader):
    """Defense in depth: even though _build_url only ever produces
    www.sec.gov/data.sec.gov URLs today, fetch() independently re-checks the
    host against the allowlist before calling safe_get."""
    client = SecEdgarClient(session=FakeSession([]), user_agent_reader=user_agent_reader)
    client._build_url = lambda dataset, arguments: "https://evil.example.com/CIK.json"
    with pytest.raises(ToolFailure) as excinfo:
        client.fetch(resolve_sec_dataset("ticker_cik_map"), {})
    assert excinfo.value.code == SEC_HOST_NOT_ALLOWED


# ---- happy path ----

def test_ticker_cik_map_happy_path(user_agent_reader):
    client = SecEdgarClient(session=FakeSession([FakeResp(200, TICKER_MAP_PAYLOAD)]),
                            user_agent_reader=user_agent_reader)
    result = client.fetch(resolve_sec_dataset("ticker_cik_map"), {})
    assert result.payload["0"]["ticker"] == "AAPL"


def test_submissions_happy_path(user_agent_reader):
    client = SecEdgarClient(session=FakeSession([FakeResp(200, SUBMISSIONS_PAYLOAD)]),
                            user_agent_reader=user_agent_reader)
    result = client.fetch(resolve_sec_dataset("company_submissions"), {"cik": "0000320193"})
    assert result.payload["name"] == "Apple Inc."
    assert result.payload["filings"]["recent"]["form"][0] == "10-K"


# ---- error classification ----

def test_404_is_invalid_symbol(user_agent_reader):
    client = SecEdgarClient(session=FakeSession([FakeResp(404, b"not found")]),
                            user_agent_reader=user_agent_reader)
    with pytest.raises(ToolFailure) as excinfo:
        client.fetch(resolve_sec_dataset("company_submissions"), {"cik": "9999999999"})
    assert excinfo.value.code == MARKET_DATA_INVALID_SYMBOL


def test_403_is_rate_limited_not_authentication_failed(user_agent_reader):
    """SEC returns 403 for both a bad User-Agent AND a fair-access IP block;
    this client always sends a real User-Agent, so 403 in practice means the
    latter -- classified as rate-limited so the existing throttle/backoff
    machinery applies (see finance/sec_provider.py's module docstring)."""
    client = SecEdgarClient(session=FakeSession([FakeResp(403, b"blocked")]),
                            user_agent_reader=user_agent_reader)
    with pytest.raises(ToolFailure) as excinfo:
        client.fetch(resolve_sec_dataset("ticker_cik_map"), {})
    assert excinfo.value.code == MARKET_DATA_RATE_LIMITED


def test_429_is_rate_limited(user_agent_reader):
    client = SecEdgarClient(session=FakeSession([FakeResp(429, b"too many")]),
                            user_agent_reader=user_agent_reader)
    with pytest.raises(ToolFailure) as excinfo:
        client.fetch(resolve_sec_dataset("ticker_cik_map"), {})
    assert excinfo.value.code == MARKET_DATA_RATE_LIMITED


def test_malformed_json_is_invalid_response(user_agent_reader):
    client = SecEdgarClient(session=FakeSession([FakeResp(200, b"not json{{{")]),
                            user_agent_reader=user_agent_reader)
    with pytest.raises(ToolFailure) as excinfo:
        client.fetch(resolve_sec_dataset("ticker_cik_map"), {})
    assert excinfo.value.code == MARKET_DATA_INVALID_RESPONSE


def test_non_object_json_is_invalid_response(user_agent_reader):
    client = SecEdgarClient(session=FakeSession([FakeResp(200, b"[1, 2, 3]")]),
                            user_agent_reader=user_agent_reader)
    with pytest.raises(ToolFailure) as excinfo:
        client.fetch(resolve_sec_dataset("ticker_cik_map"), {})
    assert excinfo.value.code == MARKET_DATA_INVALID_RESPONSE


# ---- resolve_cik: the "one fetch serves every ticker" cache-reuse guarantee ----

def test_resolve_cik_finds_the_right_entry(tmp_path):
    client = SecEdgarClient(session=FakeSession([FakeResp(200, TICKER_MAP_PAYLOAD)]),
                            user_agent_reader=lambda: "TestApp/1.0 (contact: t@example.com)")
    coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "m.sqlite3")),
        ledger=QuotaLedger(path=str(tmp_path / "q.sqlite3"), daily_limit=1_000_000),
        client=client, provider_id="sec", dataset_resolver=resolve_sec_dataset,
        min_request_interval_ms=0)

    cik, name = resolve_cik(coordinator, "AAPL")
    assert cik == "0000320193"
    assert name == "Apple Inc."


def test_resolve_cik_unknown_ticker_fails_closed(tmp_path):
    client = SecEdgarClient(session=FakeSession([FakeResp(200, TICKER_MAP_PAYLOAD)]),
                            user_agent_reader=lambda: "TestApp/1.0 (contact: t@example.com)")
    coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "m.sqlite3")),
        ledger=QuotaLedger(path=str(tmp_path / "q.sqlite3"), daily_limit=1_000_000),
        client=client, provider_id="sec", dataset_resolver=resolve_sec_dataset,
        min_request_interval_ms=0)

    with pytest.raises(ToolFailure) as excinfo:
        resolve_cik(coordinator, "NOTREAL")
    assert excinfo.value.code == SEC_CIK_NOT_FOUND


def test_resolve_cik_for_a_second_different_ticker_is_a_pure_cache_hit(tmp_path):
    """The whole point of the constant "GLOBAL" lookup symbol: resolving
    100 different tickers costs exactly ONE external call, not 100."""
    client = SecEdgarClient(session=FakeSession([FakeResp(200, TICKER_MAP_PAYLOAD)]),
                            user_agent_reader=lambda: "TestApp/1.0 (contact: t@example.com)")
    coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "m.sqlite3")),
        ledger=QuotaLedger(path=str(tmp_path / "q.sqlite3"), daily_limit=1_000_000),
        client=client, provider_id="sec", dataset_resolver=resolve_sec_dataset,
        min_request_interval_ms=0)

    resolve_cik(coordinator, "AAPL")
    cik, name = resolve_cik(coordinator, "MSFT")  # only ONE FakeResp was queued
    assert cik == "0000789019"
    assert name == "MICROSOFT CORP"
    assert len(client.session.calls) == 1
