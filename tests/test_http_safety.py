"""SSRF guard + bounded fetch (tools/http_safety.py)."""

import pytest

import tools.http_safety as hs
from tools.http_safety import FetchError, _blocked_ip, read_limited, safe_get, validate_url


class FakeResp:
    def __init__(self, status_code=200, headers=None, chunks=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks if chunks is not None else [b"<html></html>"]
        self.closed = False

    def iter_content(self, chunk_size=1):
        for c in self._chunks:
            yield c

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return self._responses.pop(0)


# ---- IP blocklist ----

@pytest.mark.parametrize("ip", [
    "127.0.0.1", "10.0.0.5", "172.16.4.4", "192.168.1.10", "169.254.169.254",
    "0.0.0.0", "::1", "fc00::1", "fe80::1", "::ffff:10.0.0.1",
])
def test_blocked_ips(ip):
    assert _blocked_ip(ip) is True


@pytest.mark.parametrize("ip", ["93.184.216.34", "8.8.8.8", "1.1.1.1", "140.82.112.3"])
def test_public_ips_allowed(ip):
    assert _blocked_ip(ip) is False


# ---- URL validation ----

def test_https_ok():
    parts = validate_url("https://example.com/article", allow_http=False)
    assert parts.hostname == "example.com"


def test_http_rejected_unless_allowed():
    with pytest.raises(FetchError) as e:
        validate_url("http://example.com", allow_http=False)
    assert e.value.code == "UNSUPPORTED_URL_SCHEME"
    assert validate_url("http://example.com", allow_http=True).scheme == "http"


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com", "data:text/html,hi",
    "javascript:alert(1)", "gopher://x", "ssh://x",
])
def test_forbidden_schemes(url):
    with pytest.raises(FetchError) as e:
        validate_url(url, allow_http=True)
    assert e.value.code == "UNSUPPORTED_URL_SCHEME"


def test_embedded_credentials_rejected():
    with pytest.raises(FetchError) as e:
        validate_url("https://user:pass@example.com", allow_http=False)
    assert e.value.code == "INVALID_URL"


def test_unusual_port_rejected():
    with pytest.raises(FetchError) as e:
        validate_url("https://example.com:8443/x", allow_http=False)
    assert e.value.code == "INVALID_URL"


def test_missing_host_rejected():
    with pytest.raises(FetchError):
        validate_url("https:///nohost", allow_http=False)


# ---- safe_get ----

def _public(monkeypatch, ips=("93.184.216.34",)):
    monkeypatch.setattr(hs, "_resolve", lambda host: list(ips))


def test_safe_get_success(monkeypatch):
    _public(monkeypatch)
    session = FakeSession([FakeResp(200, {"Content-Type": "text/html"})])
    resp, final = safe_get("https://example.com", session, "UA", 5, 20, 5, False)
    assert resp.status_code == 200
    assert final == "https://example.com"


def test_safe_get_blocks_private_resolution(monkeypatch):
    monkeypatch.setattr(hs, "_resolve", lambda host: ["10.0.0.5"])
    session = FakeSession([FakeResp(200)])
    with pytest.raises(FetchError) as e:
        safe_get("https://internal.example.com", session, "UA", 5, 20, 5, False)
    assert e.value.code == "PRIVATE_NETWORK_BLOCKED"
    assert session.calls == []  # no request was ever made


def test_safe_get_public_redirect_to_private_blocked(monkeypatch):
    # First host public, redirect target resolves private.
    def fake_resolve(host):
        return ["93.184.216.34"] if host == "example.com" else ["169.254.169.254"]
    monkeypatch.setattr(hs, "_resolve", fake_resolve)
    session = FakeSession([FakeResp(302, {"Location": "http://169.254.169.254/latest/meta-data"})])
    with pytest.raises(FetchError) as e:
        safe_get("https://example.com", session, "UA", 5, 20, 5, True)
    assert e.value.code == "REDIRECT_BLOCKED"


def test_safe_get_too_many_redirects(monkeypatch):
    _public(monkeypatch)
    responses = [FakeResp(302, {"Location": "https://example.com/next"}) for _ in range(6)]
    session = FakeSession(responses)
    with pytest.raises(FetchError) as e:
        safe_get("https://example.com", session, "UA", 5, 20, 2, False)
    assert e.value.code == "TOO_MANY_REDIRECTS"


def test_safe_get_follows_valid_redirect(monkeypatch):
    _public(monkeypatch)
    session = FakeSession([
        FakeResp(301, {"Location": "https://example.com/final"}),
        FakeResp(200, {"Content-Type": "text/html"}),
    ])
    resp, final = safe_get("https://example.com/start", session, "UA", 5, 20, 5, False)
    assert resp.status_code == 200
    assert final == "https://example.com/final"


def test_safe_get_timeout(monkeypatch):
    import requests
    _public(monkeypatch)

    class TimeoutSession:
        def get(self, *a, **k):
            raise requests.exceptions.Timeout()

    with pytest.raises(FetchError) as e:
        safe_get("https://example.com", TimeoutSession(), "UA", 5, 20, 5, False)
    assert e.value.code == "FETCH_TIMEOUT"


# ---- read_limited ----

def test_read_limited_content_length_over_max():
    resp = FakeResp(200, {"Content-Length": "5000"}, chunks=[b"x" * 5000])
    with pytest.raises(FetchError) as e:
        read_limited(resp, max_bytes=1000)
    assert e.value.code == "RESPONSE_TOO_LARGE"


def test_read_limited_stream_over_max_without_content_length():
    resp = FakeResp(200, {}, chunks=[b"x" * 600, b"x" * 600])
    with pytest.raises(FetchError) as e:
        read_limited(resp, max_bytes=1000)
    assert e.value.code == "RESPONSE_TOO_LARGE"


def test_read_limited_ok():
    resp = FakeResp(200, {}, chunks=[b"hello ", b"world"])
    body, total = read_limited(resp, max_bytes=1000)
    assert body == b"hello world"
    assert total == 11
    assert resp.closed is True
