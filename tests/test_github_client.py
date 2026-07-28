"""GitHubClient: auth header, error mapping, rate-limit parsing, token secrecy."""

import pytest

from tools.base import ToolFailure
from tools.github_client import GitHubClient


class FakeResp:
    def __init__(self, status_code=200, payload=None, headers=None, raise_json=False):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("bad")
        return self._payload


class FakeSession:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "params": params})
        if self._exc:
            raise self._exc
        return self._response


def test_token_added_to_authorization_header(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    session = FakeSession(FakeResp(200, {"ok": True}))
    client = GitHubClient(session=session, token="secret-token")
    client.get("/repos/a/b")
    assert session.calls[0]["headers"]["Authorization"] == "Bearer secret-token"


def test_no_token_still_makes_public_request(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    session = FakeSession(FakeResp(200, {"ok": True}))
    client = GitHubClient(session=session)
    client.get("/repos/a/b")
    assert "Authorization" not in session.calls[0]["headers"]


def test_token_never_appears_in_returned_data(monkeypatch):
    session = FakeSession(FakeResp(200, {"ok": True}))
    client = GitHubClient(session=session, token="secret-token")
    resp = client.get("/repos/a/b")
    assert "secret-token" not in str(resp.data)


def test_401_authentication_failed():
    client = GitHubClient(session=FakeSession(FakeResp(401)))
    with pytest.raises(ToolFailure) as e:
        client.get("/x")
    assert e.value.code == "GITHUB_AUTHENTICATION_FAILED"


def test_403_rate_limited():
    resp = FakeResp(403, headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1893456000"})
    client = GitHubClient(session=FakeSession(resp))
    with pytest.raises(ToolFailure) as e:
        client.get("/x")
    assert e.value.code == "GITHUB_RATE_LIMITED"
    assert e.value.retryable is True


def test_404_returned_not_raised():
    client = GitHubClient(session=FakeSession(FakeResp(404)))
    resp = client.get("/x")
    assert resp.status_code == 404
    assert resp.data is None


def test_500_github_error():
    client = GitHubClient(session=FakeSession(FakeResp(500)))
    with pytest.raises(ToolFailure) as e:
        client.get("/x")
    assert e.value.code == "GITHUB_API_ERROR"


def test_invalid_json():
    client = GitHubClient(session=FakeSession(FakeResp(200, raise_json=True)))
    with pytest.raises(ToolFailure) as e:
        client.get("/x")
    assert e.value.code == "INVALID_RESPONSE"


def test_timeout():
    import requests
    client = GitHubClient(session=FakeSession(exc=requests.exceptions.Timeout()))
    with pytest.raises(ToolFailure) as e:
        client.get("/x")
    assert e.value.code == "GITHUB_API_ERROR"


def test_rate_limit_headers_parsed():
    resp = FakeResp(200, {"ok": 1},
                    headers={"X-RateLimit-Remaining": "42", "X-RateLimit-Reset": "1893456000"})
    client = GitHubClient(session=FakeSession(resp))
    r = client.get("/x")
    assert r.rate_limit["remaining"] == 42
    assert r.rate_limit["reset_at"].startswith("2030")  # 1893456000 -> 2030-01-01


def test_malformed_rate_limit_headers_safe():
    resp = FakeResp(200, {"ok": 1},
                    headers={"X-RateLimit-Remaining": "abc", "X-RateLimit-Reset": "xyz"})
    client = GitHubClient(session=FakeSession(resp))
    r = client.get("/x")
    assert r.rate_limit["remaining"] is None
    assert r.rate_limit["reset_at"] is None
