"""Phase F.1 Task 1 — the router may choose local/claude/one of its OWN offered
built-ins, and nothing else. It must never dispatch an MCP, provisioning, or
filesystem-access tool, even if the underlying model hallucinates one.
"""

from unittest.mock import MagicMock, patch

from router import route_and_answer


def _mock_response(tool_calls=None, content=""):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": {"content": content, "tool_calls": tool_calls or []}}
    return mock_resp


@patch("router.requests.post")
def test_hallucinated_mcp_tool_name_falls_back_to_local(mock_post):
    """A tool_calls entry naming a tool that was never OFFERED to the router
    (e.g. an MCP provisioning tool) must never be dispatched — mode resolves to
    local, the tool name is discarded, and nothing about it reaches execution."""
    mock_post.return_value = _mock_response(
        tool_calls=[{"function": {"name": "mcp.provision", "arguments": {}}}]
    )
    decision = route_and_answer("read a file for me", [])
    assert decision.mode == "local"
    assert decision.tool is None


@patch("router.requests.post")
def test_hallucinated_filesystem_access_tool_name_falls_back_to_local(mock_post):
    mock_post.return_value = _mock_response(
        tool_calls=[{"function": {"name": "mcp.filesystem.access.add", "arguments": {}}}]
    )
    decision = route_and_answer("give access to this folder", [])
    assert decision.mode == "local"
    assert decision.tool is None


@patch("router.requests.post")
def test_malformed_router_output_with_only_a_tool_field_is_safe(mock_post):
    """No 'mode' at all, just a tool name — the fixed Ollama tools contract means
    this arrives as a tool_calls entry; it must still fail safe to local with no
    tool execution and no MCP configuration change."""
    mock_post.return_value = _mock_response(
        tool_calls=[{"function": {"name": "mcp.provision", "arguments": {"plan_id": "x"}}}]
    )
    decision = route_and_answer("do the thing", [])
    assert decision.mode == "local"
    assert decision.tool is None
    assert decision.payload == "do the thing"


@patch("router.requests.post")
def test_offered_builtin_tool_still_dispatches_normally(mock_post):
    """The hardening must not regress routing to the router's OWN declared tools."""
    mock_post.return_value = _mock_response(
        tool_calls=[{"function": {"name": "get_time", "arguments": {}}}]
    )
    decision = route_and_answer("what time is it", [])
    assert decision.mode == "tool"
    assert decision.tool == "time"


@patch("router.requests.post")
def test_local_and_claude_routing_are_unaffected(mock_post):
    mock_post.return_value = _mock_response(
        tool_calls=[{"function": {"name": "escalate_to_claude", "arguments": {}}}]
    )
    decision = route_and_answer("should I refinance my mortgage", [])
    assert decision.mode == "claude"

    mock_post.return_value = _mock_response(
        tool_calls=[{"function": {"name": "answer_locally", "arguments": {}}}]
    )
    decision = route_and_answer("how are you", [])
    assert decision.mode == "local"


def test_offered_function_names_never_include_mcp_tools():
    """A structural guarantee independent of any mocked response: the router's own
    declared tool schema can never contain an MCP/provisioning/access tool name,
    so the model is never even offered one to hallucinate a variant of."""
    import router

    assert not any(name.startswith("mcp.") for name in router._OFFERED_FUNCTION_NAMES)
    assert "answer_locally" in router._OFFERED_FUNCTION_NAMES
    assert "escalate_to_claude" in router._OFFERED_FUNCTION_NAMES
