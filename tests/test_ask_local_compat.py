"""Backward-compat: ask_local delegates to ask_local_raw; Claude path unchanged."""

from unittest.mock import patch

import brain
from router import RouteDecision


def test_ask_local_returns_text_and_metrics_tuple():
    raw = {"message": {"role": "assistant", "content": "  hi there  "},
           "metrics": {"prompt_tokens": 3, "completion_tokens": 4}, "ok": True}
    with patch("brain.ask_local_raw", return_value=raw) as mock_raw:
        text, metrics = brain.ask_local("q", [], "sys")
    assert text == "hi there"  # stripped
    assert metrics == {"prompt_tokens": 3, "completion_tokens": 4}
    mock_raw.assert_called_once()


def test_ask_local_delegates_with_system_history_user_messages():
    raw = {"message": {"content": "ok"}, "metrics": {}, "ok": True}
    history = [{"role": "user", "content": "prev"}, {"role": "assistant", "content": "ans"}]
    with patch("brain.ask_local_raw", return_value=raw) as mock_raw:
        brain.ask_local("newq", history, "SYSTEM")
    (messages,), kwargs = mock_raw.call_args
    assert messages[0] == {"role": "system", "content": "SYSTEM"}
    assert messages[-1] == {"role": "user", "content": "newq"}
    assert {"role": "user", "content": "prev"} in messages
    # ask_local must not pass tools (normal chat).
    assert kwargs.get("tools") in (None, [], {}) or "tools" not in kwargs


def test_ask_local_single_request_builder_is_ask_local_raw():
    # ask_local must not call requests.post directly — it goes through ask_local_raw.
    raw = {"message": {"content": "x"}, "metrics": {}, "ok": True}
    with patch("brain.ask_local_raw", return_value=raw), patch("brain.requests.post") as mock_post:
        brain.ask_local("q", [], "sys")
    mock_post.assert_not_called()


def test_claude_tools_web_search_unchanged():
    assert brain.CLAUDE_TOOLS == [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}]


def test_dispatch_claude_route_calls_ask_claude_not_tool_loop():
    import assistant
    decision = RouteDecision(mode="claude", payload="q")
    with patch("assistant.ask_claude", return_value=("claude answer", {})) as mock_claude, \
         patch("tool_loop.run_local_tool_loop") as mock_loop:
        reply, _ = assistant.dispatch(decision, "user q", "enriched q", [], "sys")
    assert reply == "claude answer"
    mock_claude.assert_called_once_with("enriched q", [], "sys")
    mock_loop.assert_not_called()


def test_dispatch_local_route_uses_tool_loop():
    import assistant
    decision = RouteDecision(mode="local", payload="q")
    with patch("tool_loop.run_local_tool_loop", return_value=("local answer", {})) as mock_loop, \
         patch("assistant.ask_claude") as mock_claude:
        reply, _ = assistant.dispatch(decision, "user q", "enriched q", [], "sys")
    assert reply == "local answer"
    mock_loop.assert_called_once_with("enriched q", [], "sys")
    mock_claude.assert_not_called()
