"""Phase E — lifecycle: disabled, timeouts, server exit, optional/required failure, shutdown."""

import os
import sys
import time
from types import SimpleNamespace

import pytest

from mcp_layer.config import build_config
from mcp_layer.errors import McpError
from mcp_layer.external import bootstrap_from_config
from tools.executor import ToolExecutor
from tools.models import MCP_EXECUTABLE_NOT_FOUND, ToolCall
from tools.registry import ToolRegistry

DEFAULT_TOOLS = {
    "echo_text": {"enabled": True, "permission": "read"},
    "read_test_file": {"enabled": True, "permission": "read"},
    "slow_tool": {"enabled": True, "permission": "read"},
    "fail_tool": {"enabled": True, "permission": "read"},
}


def _raw(tmp_path, **over):
    approved = tmp_path / "mcp_workspaces"
    workdir = approved / "test"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "hello.txt").write_text("Hello from MCP!", encoding="utf-8")
    raw = {
        "enabled": True, "required": False, "server_id": "test", "transport": "stdio",
        "command": sys.executable, "args": [], "internal_test_server": True,
        "working_directory": str(workdir),
        "startup_timeout_seconds": 10, "call_timeout_seconds": 5, "shutdown_timeout_seconds": 5,
        "environment_allowlist": ["TEST_MCP_SLOW_SECONDS"],
        "tool_policy": {"default_permission": "denied", "tools": DEFAULT_TOOLS},
    }
    raw.update(over)
    return raw, str(approved), str(workdir)


@pytest.fixture
def bootstrap(tmp_path):
    sessions = []

    def _start(**over):
        raw, approved, workdir = _raw(tmp_path, **over)
        reg = ToolRegistry()
        session = bootstrap_from_config(reg, config=build_config(raw), approved_root=approved)
        sessions.append(session)
        return SimpleNamespace(session=session, registry=reg, executor=ToolExecutor(reg), workdir=workdir)

    yield _start
    os.environ.pop("TEST_MCP_SLOW_SECONDS", None)
    for s in sessions:
        s.shutdown()


# ---- disabled ----

def test_disabled_config_launches_no_process(tmp_path):
    raw, approved, _ = _raw(tmp_path, enabled=False)
    reg = ToolRegistry()
    session = bootstrap_from_config(reg, config=build_config(raw), approved_root=approved)
    assert session.health.state.value == "disabled"
    assert session.client is None
    assert session.tools == []
    assert not any(n.startswith("mcp.") for n in [t.name for t in reg.enabled_definitions()])
    session.shutdown()  # no-op, must not raise


# ---- optional vs required failure ----

def test_optional_server_missing_executable_does_not_break_startup(tmp_path):
    raw, approved, _ = _raw(tmp_path, command="definitely_not_real_zzz", required=False)
    reg = ToolRegistry()
    session = bootstrap_from_config(reg, config=build_config(raw), approved_root=approved)
    assert session.health.state.value == "failed"
    assert session.health.last_error_code == MCP_EXECUTABLE_NOT_FOUND
    assert session.tools == []


def test_required_server_missing_executable_raises(tmp_path):
    raw, approved, _ = _raw(tmp_path, command="definitely_not_real_zzz", required=True)
    reg = ToolRegistry()
    with pytest.raises(McpError) as e:
        bootstrap_from_config(reg, config=build_config(raw), approved_root=approved)
    assert e.value.code == MCP_EXECUTABLE_NOT_FOUND


# ---- timeout ----

def test_call_timeout_is_normalized_and_assistant_stays_usable(bootstrap):
    os.environ["TEST_MCP_SLOW_SECONDS"] = "2"
    b = bootstrap(call_timeout_seconds=1)
    r = b.executor.execute(ToolCall("c1", "mcp.test.slow_tool", {}))
    assert r.error.code == "MCP_TIMEOUT"
    # Let the (single-threaded) server finish its sleep and emit the now-stale slow
    # response, which the client must discard by request-id. The assistant then
    # recovers: a fresh echo returns cleanly, never the stale slow payload.
    time.sleep(2.5)
    r2 = b.executor.execute(ToolCall("c2", "mcp.test.echo_text", {"text": "still here"}))
    assert r2.success is True and r2.data == {"text": "still here"}


# ---- server exit ----

def test_server_exit_during_request_is_normalized(bootstrap):
    b = bootstrap()
    b.session.client._proc.kill()
    b.session.client._proc.wait(timeout=5)
    r = b.executor.execute(ToolCall("c1", "mcp.test.echo_text", {"text": "hi"}))
    assert r.error.code in ("MCP_SERVER_EXITED", "MCP_TIMEOUT")


def test_builtin_tools_keep_working_after_server_exit(bootstrap):
    from tools.calculator import CalculatorTool
    b = bootstrap()
    b.registry.register(CalculatorTool())
    b.session.client._proc.kill()
    b.session.client._proc.wait(timeout=5)
    r = b.executor.execute(ToolCall("c1", "math.calculate", {"expression": "2+2"}))
    assert r.success is True and r.data["result"] == 4


# ---- shutdown ----

def test_shutdown_terminates_process_and_is_idempotent(bootstrap):
    b = bootstrap()
    proc = b.session.client._proc
    assert proc.poll() is None
    b.session.shutdown()
    for _ in range(60):
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    assert proc.poll() is not None
    b.session.shutdown()  # second call must not raise
    b.session.shutdown()
