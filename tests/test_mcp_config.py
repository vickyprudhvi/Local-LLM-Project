"""Phase E — MCP configuration validation + discovery policy (no subprocess)."""

import json

import pytest

import tools.config as app_config
from mcp_layer.config import build_config, load_config
from mcp_layer.discovery import (
    MAX_SCHEMA_BYTES,
    is_valid_input_schema,
    plan_registration,
    sanitize_description,
)
from mcp_layer.errors import McpError
from tools.models import MCP_CONFIGURATION_INVALID, ToolPermission

READ, WRITE, DENIED = ToolPermission.READ, ToolPermission.WRITE, ToolPermission.DENIED

FULL_TOOLS = {
    "echo_text": {"enabled": True, "permission": "read"},
    "write_test_file": {"enabled": True, "permission": "write"},
}


def _raw(**over):
    base = {
        "enabled": True, "required": False, "server_id": "test", "transport": "stdio",
        "command": "python", "args": ["-m", "test_mcp_server"],
        "working_directory": "./mcp_workspaces/test",
        "startup_timeout_seconds": 10, "call_timeout_seconds": 5, "shutdown_timeout_seconds": 5,
        "environment_allowlist": [], "tool_policy": {"default_permission": "denied", "tools": FULL_TOOLS},
    }
    base.update(over)
    return base


# ---- valid ----

def test_valid_configuration_loads():
    cfg = build_config(_raw())
    assert cfg.enabled is True and cfg.server_id == "test" and cfg.transport == "stdio"
    assert cfg.command == "python" and cfg.args == ("-m", "test_mcp_server")
    assert cfg.namespace == "mcp.test"
    assert cfg.tool_policy.tools["write_test_file"].permission is WRITE


def test_repository_default_config_is_disabled():
    cfg = load_config("config/mcp_server.json")
    assert cfg is not None and cfg.enabled is False


def test_missing_config_file_returns_none():
    assert load_config("config/does_not_exist.json") is None


def test_invalid_json_is_configuration_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json", encoding="utf-8")
    with pytest.raises(McpError) as e:
        load_config(str(p))
    assert e.value.code == MCP_CONFIGURATION_INVALID


# ---- fail closed ----

def test_enabled_requires_command():
    with pytest.raises(McpError) as e:
        build_config(_raw(command=""))
    assert e.value.code == MCP_CONFIGURATION_INVALID


def test_unsupported_transport_rejected():
    for bad in ("http", "https", "sse", "websocket"):
        with pytest.raises(McpError) as e:
            build_config(_raw(transport=bad))
        assert e.value.code == MCP_CONFIGURATION_INVALID


def test_invalid_server_id_rejected():
    for bad in ("bad id", "has/slash", "", "dot.dot"):
        with pytest.raises(McpError):
            build_config(_raw(server_id=bad))


def test_zero_and_negative_timeouts_rejected():
    for field in ("startup_timeout_seconds", "call_timeout_seconds", "shutdown_timeout_seconds"):
        with pytest.raises(McpError):
            build_config(_raw(**{field: 0}))
        with pytest.raises(McpError):
            build_config(_raw(**{field: -1}))


def test_command_with_newline_or_null_rejected():
    with pytest.raises(McpError):
        build_config(_raw(command="python\n"))
    with pytest.raises(McpError):
        build_config(_raw(command="py\x00thon"))


def test_invalid_permission_fails_closed_to_denied():
    cfg = build_config(_raw(tool_policy={"default_permission": "bogus",
                                         "tools": {"echo_text": {"enabled": True, "permission": "nonsense"}}}))
    assert cfg.tool_policy.default_permission is DENIED
    assert cfg.tool_policy.tools["echo_text"].permission is DENIED


def test_missing_tool_policy_defaults_denied():
    cfg = build_config(_raw(tool_policy=None))
    assert cfg.tool_policy.default_permission is DENIED
    assert cfg.tool_policy.tools == {}


def test_bad_environment_allowlist_rejected():
    with pytest.raises(McpError):
        build_config(_raw(environment_allowlist=["ok", "bad name"]))


# ---- discovery policy (no live server) ----

def _cfg_with_tools(tools):
    return build_config(_raw(tool_policy={"default_permission": "denied", "tools": tools}))


def _spec(name, schema=None, description="d"):
    return {"name": name, "description": description,
            "inputSchema": schema or {"type": "object", "properties": {}}}


def test_plan_registers_only_enabled_in_policy_tools():
    cfg = _cfg_with_tools({
        "echo_text": {"enabled": True, "permission": "read"},
        "write_test_file": {"enabled": True, "permission": "write"},
        "slow_tool": {"enabled": False, "permission": "read"},
    })
    raw = [_spec("echo_text"), _spec("write_test_file"), _spec("slow_tool"),
           _spec("unknown_tool")]
    regs, denied = plan_registration(raw, cfg)
    names = {r["remote_name"]: r["permission"] for r in regs}
    assert names == {"echo_text": READ, "write_test_file": WRITE}  # slow disabled, unknown denied
    assert "unknown_tool" in denied
    assert "slow_tool" not in denied  # disabled != denied


def test_plan_ignores_server_advertised_permission():
    # Server would advertise echo_text as read; local policy says WRITE. Local wins.
    cfg = _cfg_with_tools({"echo_text": {"enabled": True, "permission": "write"}})
    spec = _spec("echo_text")
    spec["annotations"] = {"permission": "read"}
    regs, _ = plan_registration([spec], cfg)
    assert regs[0]["permission"] is WRITE


def test_plan_rejects_invalid_tool_names():
    cfg = _cfg_with_tools({"echo_text": {"enabled": True, "permission": "read"}})
    regs, denied = plan_registration([_spec("bad name!"), _spec("echo_text")], cfg)
    assert [r["remote_name"] for r in regs] == ["echo_text"]
    assert "bad name!" in denied


def test_plan_skips_oversized_schema():
    cfg = _cfg_with_tools({"echo_text": {"enabled": True, "permission": "read"}})
    huge = {"type": "object", "properties": {"x": {"type": "string", "description": "z" * (MAX_SCHEMA_BYTES + 100)}}}
    regs, denied = plan_registration([_spec("echo_text", schema=huge)], cfg)
    assert regs == [] and "echo_text" in denied


def test_schema_validation_rejects_non_object_and_deep():
    assert is_valid_input_schema({"type": "object", "properties": {}}) is True
    assert is_valid_input_schema({"type": "array"}) is False
    assert is_valid_input_schema("not a dict") is False
    deep = {"type": "object"}
    node = deep
    for _ in range(15):
        node["properties"] = {"n": {"type": "object"}}
        node = node["properties"]["n"]
    assert is_valid_input_schema(deep) is False


def test_sanitize_description_strips_control_and_bounds():
    dirty = "hello\x00\x07 world" + "x" * 5000
    clean = sanitize_description(dirty)
    assert "\x00" not in clean and "\x07" not in clean
    assert len(clean) <= 1000
