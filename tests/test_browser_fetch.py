"""browser.fetch_page tool (network + DNS mocked)."""

import pytest

import tools.http_safety as hs
from tools.base import ToolFailure, ToolValidationError
from tools.browser import FetchPageTool


class FakeResp:
    def __init__(self, status_code=200, headers=None, chunks=None):
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "text/html"}
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


@pytest.fixture
def public_dns(monkeypatch):
    monkeypatch.setattr(hs, "_resolve", lambda host: ["93.184.216.34"])


def _tool(responses):
    return FetchPageTool(session=FakeSession(responses))


HTML = (b"<html><head><title>  My Title </title></head>"
        b"<body><script>evil()</script><style>x{}</style>"
        b"<h1>Hello</h1>   <p>World\n\n\ntext</p></body></html>")


def test_successful_html(public_dns):
    tool = _tool([FakeResp(200, {"Content-Type": "text/html; charset=utf-8"}, [HTML])])
    data = tool.execute(tool.validate_arguments({"url": "https://example.com/a"}))
    assert data["title"] == "My Title"
    assert "Hello" in data["text"] and "World" in data["text"] and "text" in data["text"]
    assert "evil" not in data["text"] and "x{}" not in data["text"]  # script/style removed
    assert "\n\n" not in data["text"]  # whitespace normalized
    assert data["untrusted_content"] is True
    assert data["source_type"] == "web_page"
    assert data["final_url"] == "https://example.com/a"


def test_plain_text(public_dns):
    tool = _tool([FakeResp(200, {"Content-Type": "text/plain"}, [b"just   text\n\nhere"])])
    data = tool.execute(tool.validate_arguments({"url": "https://example.com/t"}))
    assert data["content_type"] == "text/plain"
    assert data["text"] == "just text\nhere"


def test_char_truncation(public_dns):
    tool = _tool([FakeResp(200, {"Content-Type": "text/plain"}, [b"abcdefghij"])])
    data = tool.execute(tool.validate_arguments({"url": "https://example.com", "max_chars": 4}))
    assert data["truncated"] is True
    assert len(data["text"]) == 4


def test_byte_limit_enforced(public_dns, monkeypatch):
    monkeypatch.setenv("MAX_PAGE_BYTES", "10")
    tool = _tool([FakeResp(200, {"Content-Type": "text/plain"}, [b"x" * 50])])
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"url": "https://example.com"}))
    assert e.value.code == "RESPONSE_TOO_LARGE"


def test_unsupported_binary_content(public_dns):
    tool = _tool([FakeResp(200, {"Content-Type": "application/octet-stream"}, [b"\x00\x01"])])
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"url": "https://example.com/bin"}))
    assert e.value.code == "UNSUPPORTED_CONTENT_TYPE"


def test_invalid_url_rejected():
    tool = _tool([])
    with pytest.raises(ToolValidationError):
        tool.validate_arguments({"url": "   "})


def test_file_scheme_rejected(public_dns):
    tool = _tool([FakeResp(200)])
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"url": "file:///etc/passwd"}))
    assert e.value.code == "UNSUPPORTED_URL_SCHEME"


def test_embedded_credentials_rejected(public_dns):
    tool = _tool([FakeResp(200)])
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"url": "https://user:pw@example.com"}))
    assert e.value.code == "INVALID_URL"


def test_localhost_blocked(monkeypatch):
    monkeypatch.setattr(hs, "_resolve", lambda host: ["127.0.0.1"])
    tool = _tool([FakeResp(200)])
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"url": "https://localhost/x"}))
    assert e.value.code == "PRIVATE_NETWORK_BLOCKED"
    assert tool.session.calls == []  # no request made


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.0.1",
                                 "169.254.169.254", "::1", "fc00::1"])
def test_private_targets_blocked(monkeypatch, ip):
    monkeypatch.setattr(hs, "_resolve", lambda host: [ip])
    tool = _tool([FakeResp(200)])
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"url": "https://target.example/x"}))
    assert e.value.code == "PRIVATE_NETWORK_BLOCKED"


def test_public_redirect_to_private_blocked(monkeypatch):
    def fake_resolve(host):
        return ["93.184.216.34"] if host == "example.com" else ["169.254.169.254"]
    monkeypatch.setattr(hs, "_resolve", fake_resolve)
    monkeypatch.setenv("ALLOW_HTTP_FETCH", "true")
    tool = _tool([FakeResp(302, {"Location": "http://169.254.169.254/latest/meta-data/"})])
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"url": "https://example.com"}))
    assert e.value.code == "REDIRECT_BLOCKED"


def test_final_url_preserved_across_redirect(public_dns):
    tool = _tool([
        FakeResp(301, {"Location": "https://example.com/final"}),
        FakeResp(200, {"Content-Type": "text/html"}, [b"<title>Final</title><p>hi</p>"]),
    ])
    data = tool.execute(tool.validate_arguments({"url": "https://example.com/start"}))
    assert data["final_url"] == "https://example.com/final"
    assert data["title"] == "Final"


def test_ollama_loopback_demo_blocked_no_request():
    """Example 3: fetching Ollama's local endpoint never reaches the network."""
    tool = _tool([FakeResp(200)])
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"url": "http://127.0.0.1:11434/api/tags"}))
    # Blocked before any connection (scheme/port/loopback layers all apply).
    assert e.value.code in ("UNSUPPORTED_URL_SCHEME", "INVALID_URL", "PRIVATE_NETWORK_BLOCKED")
    assert tool.session.calls == []
