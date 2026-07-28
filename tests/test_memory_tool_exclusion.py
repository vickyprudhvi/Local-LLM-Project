"""Internal tool messages must never reach memory_store (ChromaDB)."""

import json
from unittest.mock import patch

import pytest

import tool_loop
from tools.executor import ToolExecutor
from tools.registry import default_registry


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, messages, tools=None, timeout=120):
        self.calls.append([dict(m) for m in messages])
        resp = dict(self._responses.pop(0))
        resp.setdefault("metrics", {"prompt_tokens": 1, "completion_tokens": 1})
        resp.setdefault("ok", True)
        return resp


@pytest.fixture
def fresh_tools(monkeypatch):
    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)


def _tool_call(name, args):
    return {"message": {"role": "assistant", "content": "",
                        "tool_calls": [{"id": "c1", "function": {"name": name, "arguments": args}}]}}


def _final(text):
    return {"message": {"role": "assistant", "content": text}}


def test_tool_turn_never_calls_memory_store_remember(fresh_tools, monkeypatch):
    monkeypatch.setattr(tool_loop, "ask_local_raw",
                        FakeLLM([_tool_call("math.calculate", {"expression": "(17*23)+5"}), _final("396.")]))
    with patch("memory_store.remember") as mock_remember:
        text, _ = tool_loop.run_local_tool_loop("compute", [], "sys")
    assert text == "396."
    mock_remember.assert_not_called()


def test_only_final_text_is_returned_no_internal_messages(fresh_tools, monkeypatch):
    monkeypatch.setattr(tool_loop, "ask_local_raw",
                        FakeLLM([_tool_call("system.echo", {"text": "secret-args"}), _final("done")]))
    history = []
    text, _ = tool_loop.run_local_tool_loop("echo", history, "sys")
    # The loop returns only the final assistant text — the caller (assistant.main)
    # persists exactly that, so tool_calls / tool-result JSON never enter history.
    assert text == "done"
    assert history == []  # nothing internal was pushed into the caller's history


def test_persisted_turn_shape_excludes_tool_messages(fresh_tools, monkeypatch):
    """Simulate assistant.main's persistence step after a tool turn."""
    monkeypatch.setattr(tool_loop, "ask_local_raw",
                        FakeLLM([_tool_call("math.calculate", {"expression": "2+2"}), _final("4")]))
    history = []
    user_text = "what is 2+2"
    reply, _ = tool_loop.run_local_tool_loop(user_text, history, "sys")
    # This mirrors assistant.main lines 211-212.
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    assert history == [
        {"role": "user", "content": "what is 2+2"},
        {"role": "assistant", "content": "4"},
    ]
    assert not any(m.get("role") == "tool" or m.get("tool_calls") for m in history)
