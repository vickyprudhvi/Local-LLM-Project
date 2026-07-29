"""Phase D — MCP client unit tests (no subprocess).

Covers result normalization, error translation, permission coercion (fail closed),
and that MCP tools obey the Phase B prompt budget alongside many built-ins.
"""

import json

import pytest

import tools.config as config
from mcp_layer.client import _extract_text, _normalize_result
from mcp_layer.errors import McpError
from mcp_layer.tool import McpTool
from tools.base import BaseTool, ToolFailure
from tools.models import ToolPermission
from tools.registry import ToolRegistry, bounded_ollama_schema

READ, WRITE, DENIED = ToolPermission.READ, ToolPermission.WRITE, ToolPermission.DENIED


# ---- result normalization ----

def test_normalize_prefers_structured_content():
    result = {"structuredContent": {"sum": 5},
              "content": [{"type": "text", "text": "ignored"}]}
    assert _normalize_result(result) == {"sum": 5}


def test_normalize_parses_json_text_block():
    assert _normalize_result({"content": [{"type": "text", "text": '{"a": 1}'}]}) == {"a": 1}


def test_normalize_wraps_plain_text():
    assert _normalize_result({"content": [{"type": "text", "text": "hello"}]}) == {"text": "hello"}


def test_extract_text_joins_text_blocks():
    result = {"content": [{"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}]}
    assert _extract_text(result) == "a\nb"


# ---- permission coercion (fail closed) ----

def test_mcp_tool_permission_coercion():
    client = object()
    assert McpTool("mcp.x.a", "a", "d", {}, "read", client).permission is READ
    assert McpTool("mcp.x.b", "b", "d", {}, "write", client).permission is WRITE
    assert McpTool("mcp.x.c", "c", "d", {}, None, client).permission is DENIED
    assert McpTool("mcp.x.d", "d", "d", {}, "bogus", client).permission is DENIED


# ---- error translation ----

class _RaisingClient:
    def __init__(self, error):
        self._error = error

    def call_tool(self, name, arguments, timeout=None):
        raise self._error


def test_mcp_tool_translates_mcp_error_into_tool_failure():
    tool = McpTool("mcp.x.f", "f", "d", {}, "read",
                   _RaisingClient(McpError("MCP_TIMEOUT", "no response", retryable=True)))
    with pytest.raises(ToolFailure) as e:
        tool.execute({})
    assert e.value.code == "MCP_TIMEOUT"
    assert e.value.retryable is True
    assert "Traceback" not in e.value.message


def test_mcp_tool_confirmation_summary_is_deterministic():
    tool = McpTool("mcp.test.write_test_file", "write_test_file", "d", {}, "write", object())
    summary = tool.confirmation_summary({"path": "notes.txt", "content": "x"})
    assert "write_test_file" in summary and "test" in summary


# ---- Phase B budget: MCP tools + many built-ins stay bounded ----

class _Dummy(BaseTool):
    def __init__(self, i):
        self.name = f"dummy.tool_{i:02d}"
        self.description = "A dummy built-in tool. " + "detail " * 6
        self.input_schema = {"type": "object", "properties": {}}
        self.permission = READ

    def execute(self, arguments):
        return {}


def test_prompt_budget_holds_with_6_mcp_plus_50_builtins():
    reg = ToolRegistry()
    for name in ("echo_text", "add_numbers", "read_test_file",
                 "write_test_file", "fail_tool", "slow_tool"):
        reg.register(McpTool(f"mcp.test.{name}", name, f"MCP tool {name}",
                             {"type": "object", "properties": {}}, "read", object()))
    for i in range(50):
        reg.register(_Dummy(i))

    assert len(reg.enabled_definitions()) == 56  # 6 MCP + 50 built-in, all offered
    limit = config.max_shortlist_tools()
    shortlisted = reg.shortlist_tools("please echo hello", limit)
    assert len(shortlisted) <= limit
    schemas = [bounded_ollama_schema(d) for d in shortlisted]
    assert len(json.dumps(schemas)) < config.max_selection_prompt_chars()
