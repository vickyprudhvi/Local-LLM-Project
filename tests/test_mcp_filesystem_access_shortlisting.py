"""Phase F.1 Task 6 — filesystem access-management tools are ordinary registry
entries. Shortlisting them needs NO changes to tools/registry.py or tool_loop.py:
the existing lexical shortlist_tools() already surfaces a tool whose name/description
tokens overlap the user's words. These tests prove that holds for the new tools,
and that an unrelated request or a request with no pending state leaves them out.
"""

import tools.config as config
from mcp_management.filesystem_access_tools import (
    ALL_FILESYSTEM_ACCESS_TOOL_CLASSES,
    register_filesystem_access_tools,
)
from tests.test_tool_selection import _DummyTool
from tools.executor import ToolExecutor
from tools.models import UNKNOWN_TOOL, ToolCall
from tools.registry import ToolRegistry


def _registry_with_access_tools_and_filler(n=20, manager=None):
    reg = ToolRegistry()
    register_filesystem_access_tools(reg, manager or object())
    for i in range(n):
        reg.register(_DummyTool(i))
    return reg


def test_all_four_tools_register_with_expected_names():
    reg = ToolRegistry()
    tools = register_filesystem_access_tools(reg, object())
    assert len(tools) == 4
    for name in ("mcp.filesystem.access.list", "mcp.filesystem.access.plan",
                "mcp.filesystem.access.add", "mcp.filesystem.access.remove"):
        assert reg.has(name)


def test_registering_twice_does_not_duplicate():
    reg = ToolRegistry()
    register_filesystem_access_tools(reg, object())
    second = register_filesystem_access_tools(reg, object())
    assert second == []
    assert len(reg.enabled_definitions()) == len([c for c in ALL_FILESYSTEM_ACCESS_TOOL_CLASSES])


def test_access_expansion_language_shortlists_the_plan_tool():
    reg = _registry_with_access_tools_and_filler()
    limit = config.max_shortlist_tools()
    for phrase in (
        "give access to this folder",
        "allow this path please",
        "add this folder to the approved directories",
        "reconfigure filesystem access",
        "approve the folder for the filesystem server",
        "use this directory for filesystem access",
        "expand allowed roots for the server",
    ):
        names = [d.name for d in reg.shortlist_tools(phrase, limit)]
        assert "mcp.filesystem.access.plan" in names, phrase


def test_remove_access_language_shortlists_the_remove_tool():
    reg = _registry_with_access_tools_and_filler()
    limit = config.max_shortlist_tools()
    names = [d.name for d in reg.shortlist_tools("remove folder access from the server", limit)]
    assert "mcp.filesystem.access.remove" in names


def test_unrelated_requests_do_not_include_access_tools():
    reg = _registry_with_access_tools_and_filler()
    limit = config.max_shortlist_tools()
    for phrase in ("what's the weather like today", "tell me a joke", "calculate 2 plus 2"):
        names = [d.name for d in reg.shortlist_tools(phrase, limit)]
        assert "mcp.filesystem.access.plan" not in names
        assert "mcp.filesystem.access.add" not in names
        assert "mcp.filesystem.access.remove" not in names


def test_shortlist_stays_bounded_even_with_access_tools_registered():
    reg = _registry_with_access_tools_and_filler(n=50)
    limit = config.max_shortlist_tools()
    names = reg.shortlist_tools("give access to this folder please", limit)
    assert len(names) <= limit


def test_full_catalog_is_never_serialized_into_the_prompt():
    import json

    reg = _registry_with_access_tools_and_filler(n=50)
    limit = config.max_shortlist_tools()
    from tools.registry import bounded_ollama_schema

    shortlisted = reg.shortlist_tools("give access to this folder please", limit)
    schemas = [bounded_ollama_schema(d) for d in shortlisted]
    assert len(schemas) <= limit
    assert len(json.dumps(schemas)) < len(json.dumps([bounded_ollama_schema(d)
                                                       for d in reg.enabled_definitions()]))


def test_the_model_cannot_invent_mcp_provision_it_is_simply_unregistered():
    reg = ToolRegistry()
    register_filesystem_access_tools(reg, object())
    assert not reg.has("mcp.provision")


def test_calling_an_unregistered_tool_name_fails_safely():
    reg = ToolRegistry()
    register_filesystem_access_tools(reg, object())
    executor = ToolExecutor(reg)
    result = executor.execute(ToolCall(call_id="c1", tool_name="mcp.provision", arguments={}))
    assert result.success is False
    assert result.error.code == UNKNOWN_TOOL
