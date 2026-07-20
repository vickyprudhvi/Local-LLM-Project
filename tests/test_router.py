"""Unit tests for router.py's deterministic parts: response parsing and fallback behavior.

Classification accuracy of the LLM itself is not something pytest can assert on reliably —
that's a live/manual concern (see the model comparison notes from the router redesign).
These tests mock the Ollama call and verify route_and_answer() maps responses to the right
RouteDecision, and fails safe (mode="local") on any error.
"""

import json
from unittest.mock import MagicMock, patch

import requests

from router import RouteDecision, route_and_answer


def _mock_response(tool_calls=None, content="", status_code=200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": {"content": content, "tool_calls": tool_calls or []}}
    return mock_resp


@patch("router.requests.post")
def test_answer_locally_tool_call_routes_to_local(mock_post):
    mock_post.return_value = _mock_response(tool_calls=[{"function": {"name": "answer_locally", "arguments": {}}}])
    decision = route_and_answer("how are you today", [])
    assert decision.mode == "local"
    assert decision.payload == "how are you today"


@patch("router.requests.post")
def test_no_tool_call_falls_back_to_local(mock_post):
    mock_post.return_value = _mock_response(tool_calls=None, content="I'm doing well, thanks!")
    decision = route_and_answer("how are you today", [])
    assert decision.mode == "local"
    assert decision.payload == "how are you today"


@patch("router.requests.post")
def test_remember_tool_call_extracts_fact(mock_post):
    mock_post.return_value = _mock_response(
        tool_calls=[{"function": {"name": "remember", "arguments": {"fact": "dentist on July 14"}}}]
    )
    decision = route_and_answer("remember: dentist on July 14", [])
    assert decision.mode == "tool"
    assert decision.tool == "remember"
    assert decision.payload == "dentist on July 14"


@patch("router.requests.post")
def test_recall_tool_call_without_topic(mock_post):
    mock_post.return_value = _mock_response(tool_calls=[{"function": {"name": "recall", "arguments": {}}}])
    decision = route_and_answer("what do you remember about me", [])
    assert decision.mode == "tool"
    assert decision.tool == "recall"
    assert json.loads(decision.payload) == {"topic": None}


@patch("router.requests.post")
def test_recall_tool_call_with_topic(mock_post):
    mock_post.return_value = _mock_response(
        tool_calls=[{"function": {"name": "recall", "arguments": {"topic": "dentist appointment"}}}]
    )
    decision = route_and_answer("recall my dentist appointment", [])
    assert decision.mode == "tool"
    assert decision.tool == "recall"
    assert json.loads(decision.payload) == {"topic": "dentist appointment"}


@patch("router.requests.post")
def test_get_time_tool_call_maps_to_time(mock_post):
    mock_post.return_value = _mock_response(tool_calls=[{"function": {"name": "get_time", "arguments": {}}}])
    decision = route_and_answer("what time is it", [])
    assert decision.mode == "tool"
    assert decision.tool == "time"


@patch("router.requests.post")
def test_look_tool_call(mock_post):
    mock_post.return_value = _mock_response(tool_calls=[{"function": {"name": "look", "arguments": {}}}])
    decision = route_and_answer("what do you see", [])
    assert decision.mode == "tool"
    assert decision.tool == "look"


@patch("router.requests.post")
def test_escalate_to_claude_maps_to_claude_mode(mock_post):
    mock_post.return_value = _mock_response(
        tool_calls=[{"function": {"name": "escalate_to_claude", "arguments": {}}}]
    )
    decision = route_and_answer("should I refinance my mortgage", [])
    assert decision.mode == "claude"
    assert decision.payload == "should I refinance my mortgage"


@patch("router.requests.post")
def test_unknown_tool_name_falls_back_to_local(mock_post):
    mock_post.return_value = _mock_response(tool_calls=[{"function": {"name": "delete_everything", "arguments": {}}}])
    decision = route_and_answer("do something weird", [])
    assert decision.mode == "local"


@patch("router.requests.post")
def test_request_exception_falls_back_to_local(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("Ollama not reachable")
    decision = route_and_answer("anything", [])
    assert decision.mode == "local"
    assert decision.payload == "anything"


@patch("router.requests.post")
def test_timeout_falls_back_to_local(mock_post):
    mock_post.side_effect = requests.exceptions.ReadTimeout("timed out")
    decision = route_and_answer("anything", [])
    assert decision.mode == "local"


@patch("router.requests.post")
def test_http_error_falls_back_to_local(mock_post):
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 server error")
    mock_post.return_value = mock_resp
    decision = route_and_answer("anything", [])
    assert decision.mode == "local"


@patch("router.requests.post")
def test_remember_falls_back_to_stripped_text_if_fact_missing(mock_post):
    mock_post.return_value = _mock_response(tool_calls=[{"function": {"name": "remember", "arguments": {}}}])
    decision = route_and_answer("remember something", [])
    assert decision.mode == "tool"
    assert decision.tool == "remember"
    assert decision.payload == "remember something"


@patch("router.requests.post")
def test_history_is_trimmed_and_passed_through(mock_post):
    mock_post.return_value = _mock_response(tool_calls=None, content="ok")
    long_history = [{"role": "user", "content": f"turn {i}"} for i in range(30)]
    route_and_answer("hello", long_history)
    sent_messages = mock_post.call_args.kwargs["json"]["messages"]
    # system + up to 24 trimmed history messages + the new user turn
    assert len(sent_messages) <= 1 + 24 + 1
