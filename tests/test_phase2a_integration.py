"""Phase 2A integration: fake-LLM tool-loop flows, memory exclusion, Claude route,
config toggles, and the shared MAX_TOOL_STEPS behavior — no live network."""

import json

import pytest

import tool_loop
import tools.http_safety as hs
from tools.base import BaseTool
from tools.browser import FetchPageTool, SearchTool
from tools.calculator import CalculatorTool
from tools.echo import EchoTool
from tools.executor import ToolExecutor
from tools.github_client import GitHubResponse
from tools.github_tools import (
    GetRepositoryTool,
    ListDirectoryTool,
    ListReleasesTool,
    ReadFileTool,
    SearchRepositoriesTool,
)
from tools.models import ToolCall
from tools.registry import ToolRegistry, default_registry


# ---- fakes ----

class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, messages, tools=None, timeout=120):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        resp = dict(self._responses.pop(0))
        resp.setdefault("metrics", {"prompt_tokens": 1, "completion_tokens": 1})
        resp.setdefault("ok", True)
        return resp


class FakeProvider:
    name = "tavily"

    def __init__(self, results):
        self._results = results

    def search(self, query, limit):
        return self._results[:limit]


class FakePageSession:
    def __init__(self, responses):
        self._responses = list(responses)

    def get(self, url, **kwargs):
        return self._responses.pop(0)


class FakePageResp:
    def __init__(self, status_code=200, headers=None, chunks=None):
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "text/html"}
        self._chunks = chunks if chunks is not None else [b"<title>T</title><p>body</p>"]

    def iter_content(self, chunk_size=1):
        for c in self._chunks:
            yield c

    def close(self):
        pass


class SeqClient:
    """A GitHub client returning scripted GitHubResponses in call order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, path, params=None):
        self.calls.append(path)
        return self._responses.pop(0)


def _gh(data, status=200):
    return GitHubResponse(status, data, {"remaining": 50, "reset_at": None})


def _tool_call(name, args, call_id="c1"):
    return {"message": {"role": "assistant", "content": "",
                        "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": args}}]}}


def _final(text):
    return {"message": {"role": "assistant", "content": text}}


def _tool_messages(call):
    return [m for m in call["messages"] if m.get("role") == "tool"]


@pytest.fixture
def wire(monkeypatch):
    """Install a registry + executor with fake-backed internet tools into tool_loop."""
    def _install(*, search_results=None, pages=None, github=None):
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.register(CalculatorTool())
        reg.register(SearchTool(provider=FakeProvider(search_results or [])))
        reg.register(FetchPageTool(session=FakePageSession(pages or [])))
        client = SeqClient(github or [])
        for cls in (SearchRepositoriesTool, GetRepositoryTool, ReadFileTool,
                    ListDirectoryTool, ListReleasesTool):
            reg.register(cls(client=client))
        monkeypatch.setattr(tool_loop, "REGISTRY", reg)
        monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
        monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
        monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)
        monkeypatch.setattr(hs, "_resolve", lambda host: ["93.184.216.34"])
        return reg
    return _install


def _llm(monkeypatch, responses):
    fake = FakeLLM(responses)
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)
    return fake


# ---- browser flow ----

def test_search_then_fetch_then_answer(wire, monkeypatch):
    wire(search_results=[{"title": "Docs", "url": "https://ex.com/p", "snippet": "s", "source": "ex.com"}],
         pages=[FakePageResp(chunks=[b"<title>Official</title><p>the content</p>"])])
    fake = _llm(monkeypatch, [
        _tool_call("browser.search", {"query": "project"}),
        _tool_call("browser.fetch_page", {"url": "https://ex.com/p"}),
        _final("Per ex.com/p (Official), here is the summary."),
    ])
    text, metrics = tool_loop.run_local_tool_loop("summarize the project page", [], "sys")
    assert text.startswith("Per ex.com/p")
    # Both tool results were returned to the LLM as untrusted content.
    search_msg = json.loads(_tool_messages(fake.calls[1])[-1]["content"])
    assert search_msg["data"]["untrusted_content"] is True
    fetch_msg = json.loads(_tool_messages(fake.calls[2])[-1]["content"])
    assert fetch_msg["data"]["source_type"] == "web_page"
    assert metrics["prompt_tokens"] == 3


def test_safety_instructions_added_to_system_prompt(wire, monkeypatch):
    wire()
    fake = _llm(monkeypatch, [_final("hi")])
    tool_loop.run_local_tool_loop("hello", [], "PERSONA")
    system_msg = fake.calls[0]["messages"][0]
    assert system_msg["role"] == "system"
    assert "PERSONA" in system_msg["content"]
    assert "UNTRUSTED" in system_msg["content"]


# ---- github flow ----

def test_github_search_get_read_compare(wire, monkeypatch):
    wire(github=[
        _gh({"items": [{"full_name": "a/fin", "stargazers_count": 10, "html_url": "https://github.com/a/fin"}]}),
        _gh({"full_name": "a/fin", "stargazers_count": 10, "license": {"spdx_id": "MIT"}}),
        _gh({"type": "file", "encoding": "base64",
             "content": "IyBGaW4=", "size": 5, "sha": "s", "html_url": "https://github.com/a/fin/blob/main/README.md"}),
    ])
    fake = _llm(monkeypatch, [
        _tool_call("github.search_repositories", {"query": "financial analysis language:Python"}),
        _tool_call("github.get_repository", {"repository": "a/fin"}),
        _tool_call("github.read_file", {"repository": "a/fin", "path": "README.md"}),
        _final("a/fin (https://github.com/a/fin) looks best; I read its README."),
    ])
    text, _ = tool_loop.run_local_tool_loop("find financial analysis repos", [], "sys")
    assert "a/fin" in text
    read_msg = json.loads(_tool_messages(fake.calls[3])[-1]["content"])
    assert read_msg["data"]["source_type"] == "github_file"
    assert read_msg["data"]["text"] == "# Fin"


def test_tool_error_returns_to_llm(wire, monkeypatch):
    wire(github=[_gh(None, status=404)])
    fake = _llm(monkeypatch, [
        _tool_call("github.get_repository", {"repository": "a/missing"}),
        _final("That repository could not be found."),
    ])
    text, _ = tool_loop.run_local_tool_loop("look it up", [], "sys")
    assert "could not be found" in text
    payload = json.loads(_tool_messages(fake.calls[1])[0]["content"])
    assert payload["success"] is False
    assert payload["error"]["code"] == "GITHUB_REPOSITORY_NOT_FOUND"


def test_raw_payload_not_returned_as_final_answer(wire, monkeypatch):
    wire(github=[_gh({"full_name": "a/b", "stargazers_count": 7})])
    _llm(monkeypatch, [
        _tool_call("github.get_repository", {"repository": "a/b"}),
        _final("The repo a/b has 7 stars."),
    ])
    text, _ = tool_loop.run_local_tool_loop("stars?", [], "sys")
    assert text == "The repo a/b has 7 stars."
    assert "{" not in text and "full_name" not in text


# ---- shared framework behavior ----

def test_max_tool_steps_with_internet_tools(wire, monkeypatch):
    wire(search_results=[])
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 2)
    fake = _llm(monkeypatch, [
        _tool_call("browser.search", {"query": "a"}),
        _tool_call("browser.search", {"query": "b"}),
        _tool_call("browser.search", {"query": "c"}),  # exceeds limit
        _final("Answering with what I have."),
    ])
    text, _ = tool_loop.run_local_tool_loop("keep searching", [], "sys")
    assert text == "Answering with what I have."
    assert fake.calls[-1]["tools"] is None
    joined = " ".join(m["content"] for c in fake.calls for m in _tool_messages(c))
    assert "TOOL_STEP_LIMIT_REACHED" in joined


def test_internet_results_excluded_from_memory(wire, monkeypatch):
    from unittest.mock import patch
    wire(search_results=[{"title": "x", "url": "https://x.com", "snippet": "s", "source": "x.com"}])
    _llm(monkeypatch, [
        _tool_call("browser.search", {"query": "secret query text"}),
        _final("done"),
    ])
    history = []
    with patch("memory_store.remember") as mock_remember:
        text, _ = tool_loop.run_local_tool_loop("search", history, "sys")
    assert text == "done"
    mock_remember.assert_not_called()
    assert history == []  # no tool/tool-result messages leaked into caller history


def test_echo_and_calculate_still_work(wire, monkeypatch):
    wire()
    _llm(monkeypatch, [
        _tool_call("math.calculate", {"expression": "2+2"}),
        _final("4"),
    ])
    text, _ = tool_loop.run_local_tool_loop("2+2", [], "sys")
    assert text == "4"


# ---- config toggles ----

def test_internet_tools_disabled_excludes_from_schemas(monkeypatch):
    monkeypatch.setenv("INTERNET_TOOLS_ENABLED", "false")
    reg = default_registry()
    names = [d.name for d in reg.enabled_definitions()]
    assert names == ["math.calculate", "system.echo"]
    assert not any(n.startswith(("browser.", "github.")) for n in names)


def test_internet_read_disabled_blocks_execution(monkeypatch):
    monkeypatch.setenv("INTERNET_READ_ENABLED", "false")
    reg = ToolRegistry()
    reg.register(SearchTool(provider=FakeProvider([])))
    ex = ToolExecutor(reg)
    result = ex.execute(ToolCall("c1", "browser.search", {"query": "x"}), step=1)
    assert result.success is False
    assert result.error.code == "INTERNET_DISABLED"


def test_tool_calling_disabled_preserves_single_shot(wire, monkeypatch):
    wire()
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", False)
    fake = _llm(monkeypatch, [_final("plain answer")])
    text, _ = tool_loop.run_local_tool_loop("hi", [], "sys")
    assert text == "plain answer"
    assert len(fake.calls) == 1
    assert fake.calls[0]["tools"] is None


def test_claude_route_does_not_use_internet_tools(monkeypatch):
    import assistant
    from unittest.mock import patch
    from router import RouteDecision
    decision = RouteDecision(mode="claude", payload="q")
    with patch("assistant.ask_claude", return_value=("claude answer", {})) as mock_claude, \
         patch("tool_loop.run_local_tool_loop") as mock_loop:
        reply, _ = assistant.dispatch(decision, "user q", "enriched q", [], "sys")
    assert reply == "claude answer"
    mock_claude.assert_called_once()
    mock_loop.assert_not_called()
