"""Unit tests for router.py's deterministic parts: response parsing and fallback behavior.

Classification accuracy of the LLM itself is not something pytest can assert on reliably —
that's a live/manual concern (see the model comparison notes from the router redesign).
These tests mock the Ollama call and verify route_intent() maps responses to the right
RouteDecision, and fails safe (mode="local") on any error.
"""

from unittest.mock import MagicMock, patch

import requests

from router import RouteDecision, route_intent


def _mock_response(tool_calls=None, status_code=200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "", "tool_calls": tool_calls or []}}
    return mock_resp


@patch("router.requests.post")
def test_no_tool_call_routes_to_local(mock_post):
    mock_post.return_value = _mock_response(tool_calls=None)
    decision = route_intent("how are you today")
    assert decision == RouteDecision(mode="local", payload="how are you today")


@patch("router.requests.post")
def test_remember_tool_call_extracts_fact(mock_post):
    mock_post.return_value = _mock_response(
        tool_calls=[{"function": {"name": "remember", "arguments": {"fact": "dentist on July 14"}}}]
    )
    decision = route_intent("remember: dentist on July 14")
    assert decision.mode == "tool"
    assert decision.tool == "remember"
    assert decision.payload == "dentist on July 14"


@patch("router.requests.post")
def test_recall_tool_call(mock_post):
    mock_post.return_value = _mock_response(tool_calls=[{"function": {"name": "recall", "arguments": {}}}])
    decision = route_intent("what do you remember about me")
    assert decision == RouteDecision(mode="tool", tool="recall", payload="what do you remember about me")


@patch("router.requests.post")
def test_get_time_tool_call_maps_to_time(mock_post):
    mock_post.return_value = _mock_response(tool_calls=[{"function": {"name": "get_time", "arguments": {}}}])
    decision = route_intent("what time is it")
    assert decision.mode == "tool"
    assert decision.tool == "time"


@patch("router.requests.post")
def test_look_tool_call(mock_post):
    mock_post.return_value = _mock_response(tool_calls=[{"function": {"name": "look", "arguments": {}}}])
    decision = route_intent("what do you see")
    assert decision.mode == "tool"
    assert decision.tool == "look"


@patch("router.requests.post")
def test_escalate_to_claude_maps_to_claude_mode(mock_post):
    mock_post.return_value = _mock_response(
        tool_calls=[{"function": {"name": "escalate_to_claude", "arguments": {}}}]
    )
    decision = route_intent("should I refinance my mortgage")
    assert decision == RouteDecision(mode="claude", payload="should I refinance my mortgage")


@patch("router.requests.post")
def test_unknown_tool_name_falls_back_to_local(mock_post):
    mock_post.return_value = _mock_response(tool_calls=[{"function": {"name": "delete_everything", "arguments": {}}}])
    decision = route_intent("do something weird")
    assert decision.mode == "local"


@patch("router.requests.post")
def test_request_exception_falls_back_to_local(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("Ollama not reachable")
    decision = route_intent("anything")
    assert decision == RouteDecision(mode="local", payload="anything")


@patch("router.requests.post")
def test_timeout_falls_back_to_local(mock_post):
    mock_post.side_effect = requests.exceptions.ReadTimeout("timed out")
    decision = route_intent("anything")
    assert decision.mode == "local"


@patch("router.requests.post")
def test_http_error_falls_back_to_local(mock_post):
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 server error")
    mock_post.return_value = mock_resp
    decision = route_intent("anything")
    assert decision.mode == "local"


@patch("router.requests.post")
def test_remember_falls_back_to_stripped_text_if_fact_missing(mock_post):
    mock_post.return_value = _mock_response(tool_calls=[{"function": {"name": "remember", "arguments": {}}}])
    decision = route_intent("remember something")
    assert decision.mode == "tool"
    assert decision.tool == "remember"
    assert decision.payload == "remember something"
