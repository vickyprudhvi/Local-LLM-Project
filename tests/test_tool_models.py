"""Tool data models: serialization shape and JSON-safety."""

import json

from tools.models import ToolDefinition, ToolError, ToolResult


def test_tool_definition_ollama_schema_shape():
    d = ToolDefinition(
        name="math.calculate",
        description="Evaluate arithmetic.",
        input_schema={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    )
    schema = d.to_ollama_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "math.calculate"
    assert schema["function"]["parameters"]["required"] == ["expression"]


def test_result_ok_provider_json_matches_documented_shape():
    r = ToolResult.ok("math.calculate", "abc123", {"expression": "(17 * 23) + 5", "result": 396}, execution_time_ms=1.2)
    payload = json.loads(r.to_provider_json())
    assert payload == {
        "success": True,
        "tool_name": "math.calculate",
        "call_id": "abc123",
        "data": {"expression": "(17 * 23) + 5", "result": 396},
        "error": None,
    }
    # execution_time_ms is intentionally kept off the wire.
    assert "execution_time_ms" not in payload
    assert r.execution_time_ms == 1.2


def test_result_fail_provider_json_matches_documented_shape():
    r = ToolResult.fail("math.calculate", "abc123", "INVALID_ARGUMENTS", "The expression is invalid.")
    payload = json.loads(r.to_provider_json())
    assert payload["success"] is False
    assert payload["data"] == {}
    assert payload["error"] == {"code": "INVALID_ARGUMENTS", "message": "The expression is invalid."}


def test_provider_json_is_a_string():
    r = ToolResult.ok("system.echo", "id", {"echo": "hi"})
    assert isinstance(r.to_provider_json(), str)


def test_tool_error_to_dict():
    assert ToolError("X", "msg", retryable=True).to_dict() == {"code": "X", "message": "msg"}
