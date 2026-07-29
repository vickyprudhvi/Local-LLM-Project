"""Phase E — external configured server, server-backed integration.

Launches the real test server via a config (command = sys.executable) against a
tmp workspace under a tmp approved root, then drives tools through the EXISTING
executor. Proves discovery, local permission authority, read/write/denied paths,
and the Phase B budget with MCP tools present.
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest

import confirmation
from mcp_layer.config import build_config
from mcp_layer.external import bootstrap_from_config
from mcp_layer.tool import McpTool
from tools.base import BaseTool
from tools.executor import ToolExecutor
from tools.models import ToolCall, ToolPermission
from tools.registry import ToolRegistry, bounded_ollama_schema

READ, WRITE = ToolPermission.READ, ToolPermission.WRITE

DEFAULT_TOOLS = {
    "echo_text": {"enabled": True, "permission": "read"},
    "add_numbers": {"enabled": True, "permission": "read"},
    "read_test_file": {"enabled": True, "permission": "read"},
    "write_test_file": {"enabled": True, "permission": "write"},
    "fail_tool": {"enabled": True, "permission": "read"},
    "slow_tool": {"enabled": True, "permission": "read"},
}


@pytest.fixture
def configured(tmp_path):
    approved = tmp_path / "mcp_workspaces"
    workdir = approved / "test"
    workdir.mkdir(parents=True)
    (workdir / "hello.txt").write_text("Hello from MCP!", encoding="utf-8")
    sessions = []

    def _start(tools=None, call_timeout=5, slow_seconds=3, default_permission="denied"):
        raw = {
            "enabled": True, "required": False, "server_id": "test", "transport": "stdio",
            "command": sys.executable, "args": [], "internal_test_server": True,
            "working_directory": str(workdir),
            "startup_timeout_seconds": 10, "call_timeout_seconds": call_timeout,
            "shutdown_timeout_seconds": 5, "environment_allowlist": ["TEST_MCP_SLOW_SECONDS"],
            "tool_policy": {"default_permission": default_permission,
                            "tools": tools if tools is not None else DEFAULT_TOOLS},
        }
        os.environ["TEST_MCP_SLOW_SECONDS"] = str(slow_seconds)
        reg = ToolRegistry()
        session = bootstrap_from_config(reg, config=build_config(raw), approved_root=str(approved))
        sessions.append(session)
        return SimpleNamespace(session=session, registry=reg, executor=ToolExecutor(reg),
                               workdir=str(workdir))

    yield _start
    os.environ.pop("TEST_MCP_SLOW_SECONDS", None)
    for s in sessions:
        s.shutdown()


def _run(c, name, args, confirmer=None):
    call = ToolCall("c1", name, args)
    if confirmer is not None:
        return confirmation.resolve_with_confirmation(c.executor, call, confirmer=confirmer)
    return c.executor.execute(call)


# ---- discovery + health ----

def test_startup_discovers_and_registers_with_health(configured):
    c = configured()
    assert c.session.health.state.value == "healthy"
    assert c.session.health.server_id == "test"
    assert set(c.session.tool_names()) == {f"mcp.test.{n}" for n in DEFAULT_TOOLS}
    assert c.session.health.registered_tool_count == 6


# ---- tool-count reconciliation (Phase E closeout Task 2) ----

def test_full_six_tool_policy_registers_all_six(configured):
    """FULL configuration: 6 discovered, 6 registered, 0 denied/skipped/disabled."""
    h = configured().session.health
    assert h.discovered_tool_count == 6
    assert h.registered_tool_count == 6
    assert h.denied_tool_count == 0
    assert h.skipped_tool_count == 0
    assert h.disabled_tool_count == 0


def test_reduced_three_tool_policy_registers_only_three(configured):
    """REDUCED policy (explicitly not the full config): 3 registered, 3 denied."""
    reduced = {
        "echo_text": {"enabled": True, "permission": "read"},
        "read_test_file": {"enabled": True, "permission": "read"},
        "write_test_file": {"enabled": True, "permission": "write"},
    }
    h = configured(tools=reduced).session.health
    assert h.discovered_tool_count == 6      # server still advertises all six
    assert h.registered_tool_count == 3
    assert h.denied_tool_count == 3          # add_numbers, fail_tool, slow_tool: not_in_local_policy
    reasons = {name: reason for name, reason, cat in h.diagnostics}
    assert reasons["add_numbers"] == "not_in_local_policy"


def test_locally_disabled_tool_is_disabled_not_denied(configured):
    tools = dict(DEFAULT_TOOLS)
    tools["slow_tool"] = {"enabled": False, "permission": "read"}
    h = configured(tools=tools).session.health
    assert h.registered_tool_count == 5
    assert h.disabled_tool_count == 1
    assert h.denied_tool_count == 0
    reasons = {name: (reason, cat) for name, reason, cat in h.diagnostics}
    assert reasons["slow_tool"] == ("locally_disabled", "disabled")


def test_health_counters_are_internally_consistent(configured):
    tools = dict(DEFAULT_TOOLS)
    tools["slow_tool"] = {"enabled": False, "permission": "read"}  # disabled
    del tools["fail_tool"]                                         # denied (not in policy)
    h = configured(tools=tools).session.health
    total = (h.registered_tool_count + h.denied_tool_count
             + h.skipped_tool_count + h.disabled_tool_count)
    assert total == h.discovered_tool_count
    assert h.registered_tool_count + h.denied_tool_count <= h.discovered_tool_count


def test_duplicate_local_name_does_not_overwrite_existing_tool(tmp_path):
    from tools.base import BaseTool
    from tools.models import ToolPermission

    class _Pre(BaseTool):
        name = "mcp.test.echo_text"
        description = "pre-existing"
        input_schema = {"type": "object", "properties": {}}
        permission = ToolPermission.READ

        def execute(self, arguments):
            return {"pre": True}

    approved = tmp_path / "mcp_workspaces"
    workdir = approved / "test"
    workdir.mkdir(parents=True)
    reg = ToolRegistry()
    pre = _Pre()
    reg.register(pre)  # occupy the name before MCP discovery
    raw = {
        "enabled": True, "required": False, "server_id": "test", "transport": "stdio",
        "command": sys.executable, "args": [], "internal_test_server": True,
        "working_directory": str(workdir),
        "startup_timeout_seconds": 10, "call_timeout_seconds": 5, "shutdown_timeout_seconds": 5,
        "environment_allowlist": [],
        "tool_policy": {"default_permission": "denied",
                        "tools": {"echo_text": {"enabled": True, "permission": "read"}}},
    }
    session = bootstrap_from_config(reg, config=build_config(raw), approved_root=str(approved))
    try:
        assert reg.get("mcp.test.echo_text") is pre  # original not overwritten
        reasons = {name: (reason, cat) for name, reason, cat in session.health.diagnostics}
        assert reasons["mcp.test.echo_text"] == ("registration_collision", "skipped")
    finally:
        session.shutdown()


# ---- executable server round-trip (Phase E closeout Task 3) ----

def test_executable_server_full_roundtrip(configured):
    c = configured()  # start -> initialize -> tools/list already done
    assert c.executor.execute(ToolCall("1", "mcp.test.echo_text", {"text": "hi"})).data == {"text": "hi"}
    assert c.executor.execute(ToolCall("2", "mcp.test.add_numbers", {"a": 17, "b": 25})).data == {"sum": 42}
    assert c.executor.execute(ToolCall("3", "mcp.test.read_test_file", {"path": "hello.txt"})).data == {"content": "Hello from MCP!"}
    c.session.shutdown()
    assert c.session.client._proc is None  # clean shutdown


def test_namespaced_names_use_server_id(configured):
    c = configured()
    assert c.registry.has("mcp.test.echo_text")


# ---- local permission authority ----

def test_local_policy_overrides_server_advertised_permission(configured):
    # Server advertises echo_text as read; local policy marks it WRITE -> confirmation.
    tools = dict(DEFAULT_TOOLS)
    tools["echo_text"] = {"enabled": True, "permission": "write"}
    c = configured(tools=tools)
    assert c.registry.get("mcp.test.echo_text").permission is WRITE
    r = _run(c, "mcp.test.echo_text", {"text": "hi"})  # no confirmation
    assert r.success is False and r.error.code == "TOOL_CONFIRMATION_REQUIRED"


# ---- read ----

def test_echo_runs_without_confirmation(configured):
    r = _run(configured(), "mcp.test.echo_text", {"text": "Phase E works."})
    assert r.success is True and r.data == {"text": "Phase E works."}


def test_read_file_from_workspace(configured):
    r = _run(configured(), "mcp.test.read_test_file", {"path": "hello.txt"})
    assert r.success is True and r.data == {"content": "Hello from MCP!"}


# ---- write (Phase C confirmation) ----

def test_write_requires_confirmation_no_server_call(configured):
    c = configured()
    r = _run(c, "mcp.test.write_test_file", {"path": "phase_e.txt", "content": "Phase E write works"})
    assert r.error.code == "TOOL_CONFIRMATION_REQUIRED"
    assert not os.path.exists(os.path.join(c.workdir, "phase_e.txt"))


def test_write_declined_creates_no_file(configured):
    c = configured()
    r = _run(c, "mcp.test.write_test_file", {"path": "phase_e.txt", "content": "x"},
             confirmer=lambda s: False)
    assert r.error.code == "TOOL_CONFIRMATION_DECLINED"
    assert not os.path.exists(os.path.join(c.workdir, "phase_e.txt"))


def test_write_approved_creates_file_once(configured):
    c = configured()
    r = _run(c, "mcp.test.write_test_file", {"path": "phase_e.txt", "content": "Phase E write works"},
             confirmer=lambda s: True)
    assert r.success is True
    path = os.path.join(c.workdir, "phase_e.txt")
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as f:
        assert f.read() == "Phase E write works"


# ---- denied ----

def test_tool_missing_from_policy_is_denied_and_never_reaches_server(configured):
    tools = {"echo_text": {"enabled": True, "permission": "read"}}  # only echo allowed
    c = configured(tools=tools)
    assert not c.registry.has("mcp.test.write_test_file")   # denied -> not registered
    offered = {d.name for d in c.registry.enabled_definitions()}
    assert "mcp.test.write_test_file" not in offered        # not shortlisted
    r = c.executor.execute(ToolCall("c1", "mcp.test.write_test_file", {"path": "x"}))
    assert r.error.code == "UNKNOWN_TOOL"                   # never reaches the server


# ---- prompt budget with MCP + many built-ins ----

class _Dummy(BaseTool):
    def __init__(self, i):
        self.name = f"dummy.tool_{i:02d}"
        self.description = "A dummy built-in. " + "detail " * 6
        self.input_schema = {"type": "object", "properties": {}}
        self.permission = READ

    def execute(self, arguments):
        return {}


def test_prompt_budget_bounded_with_mcp_plus_50_builtins(configured):
    c = configured()
    for i in range(50):
        c.registry.register(_Dummy(i))
    import tools.config as cfg
    assert len(c.registry.enabled_definitions()) == 56
    shortlisted = c.registry.shortlist_tools("please echo hello", cfg.max_shortlist_tools())
    assert len(shortlisted) <= cfg.max_shortlist_tools()
    schemas = [bounded_ollama_schema(d) for d in shortlisted]
    assert len(json.dumps(schemas)) < cfg.max_selection_prompt_chars()
