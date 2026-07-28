"""browser.search tool + Tavily provider (network mocked)."""

import json

import pytest

import tools.config as config
from tools.base import ToolFailure, ToolValidationError
from tools.browser import SearchTool
from tools.search_provider import TavilyProvider


class FakeResp:
    def __init__(self, status_code=200, payload=None, raise_json=False):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"results": []}
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("bad json")
        return self._payload


class FakeSession:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.posts = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if self._exc:
            raise self._exc
        return self._response


class FakeProvider:
    name = "tavily"

    def __init__(self, results=None, exc=None):
        self._results = results or []
        self._exc = exc

    def search(self, query, limit):
        if self._exc:
            raise self._exc
        return self._results[:limit]


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")


def test_successful_search():
    provider = FakeProvider(results=[
        {"title": "A", "url": "https://a.com", "snippet": "sa", "source": "a.com"},
        {"title": "B", "url": "https://b.com", "snippet": "sb", "source": "b.com"},
    ])
    tool = SearchTool(provider=provider)
    args = tool.validate_arguments({"query": "  finance mcp  ", "limit": 5})
    data = tool.execute(args)
    assert data["query"] == "finance mcp"
    assert data["result_count"] == 2
    assert data["untrusted_content"] is True
    assert data["provider"] == "tavily"
    json.dumps({k: v for k, v in data.items() if k != "_log_meta"})  # serializable


def test_empty_results():
    tool = SearchTool(provider=FakeProvider(results=[]))
    data = tool.execute(tool.validate_arguments({"query": "nothing"}))
    assert data["result_count"] == 0
    assert data["results"] == []


def test_invalid_query_empty():
    tool = SearchTool(provider=FakeProvider())
    with pytest.raises(ToolValidationError):
        tool.validate_arguments({"query": "   "})


def test_oversized_query(monkeypatch):
    monkeypatch.setenv("MAX_SEARCH_QUERY_CHARS", "10")
    tool = SearchTool(provider=FakeProvider())
    with pytest.raises(ToolValidationError):
        tool.validate_arguments({"query": "x" * 50})


def test_invalid_limit_type():
    tool = SearchTool(provider=FakeProvider())
    with pytest.raises(ToolValidationError):
        tool.validate_arguments({"query": "q", "limit": "five"})


def test_limit_capped(monkeypatch):
    monkeypatch.setenv("SEARCH_MAX_RESULTS", "3")
    tool = SearchTool(provider=FakeProvider())
    args = tool.validate_arguments({"query": "q", "limit": 100})
    assert args["limit"] == 3


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    provider = TavilyProvider(session=FakeSession(FakeResp()))
    with pytest.raises(ToolFailure) as e:
        provider.search("q", 5)
    assert e.value.code == "SEARCH_API_KEY_MISSING"


def test_authentication_failure():
    provider = TavilyProvider(session=FakeSession(FakeResp(status_code=401)))
    with pytest.raises(ToolFailure) as e:
        provider.search("q", 5)
    assert e.value.code == "SEARCH_AUTHENTICATION_FAILED"


def test_rate_limited():
    provider = TavilyProvider(session=FakeSession(FakeResp(status_code=429)))
    with pytest.raises(ToolFailure) as e:
        provider.search("q", 5)
    assert e.value.code == "SEARCH_RATE_LIMITED"
    assert e.value.retryable is True


def test_provider_timeout():
    import requests
    provider = TavilyProvider(session=FakeSession(exc=requests.exceptions.Timeout()))
    with pytest.raises(ToolFailure) as e:
        provider.search("q", 5)
    assert e.value.code == "SEARCH_PROVIDER_ERROR"


def test_malformed_provider_response():
    provider = TavilyProvider(session=FakeSession(FakeResp(raise_json=True)))
    with pytest.raises(ToolFailure) as e:
        provider.search("q", 5)
    assert e.value.code == "SEARCH_PROVIDER_ERROR"


def test_provider_maps_fields_and_source():
    payload = {"results": [{"title": "T", "url": "https://x.com/p", "content": "snip"}]}
    provider = TavilyProvider(session=FakeSession(FakeResp(payload=payload)))
    results = provider.search("q", 5)
    assert results[0] == {"title": "T", "url": "https://x.com/p", "snippet": "snip", "source": "x.com"}


def test_provider_error_status():
    provider = TavilyProvider(session=FakeSession(FakeResp(status_code=500)))
    with pytest.raises(ToolFailure) as e:
        provider.search("q", 5)
    assert e.value.code == "SEARCH_PROVIDER_ERROR"
