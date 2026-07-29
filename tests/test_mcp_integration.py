"""Phase D — MCP integration, server-backed.

Each test launches the real internal test MCP server subprocess against a tmp
workspace, discovers + registers its tools, and drives them through the EXISTING
ToolExecutor. Proves discovery, execution, permission gating, confirmation,
timeout, error normalization, sandboxing, and lifecycle.
"""

import os
import time
from types import SimpleNamespace

import pytest

import confirmation
from mcp_layer.integration import discover_and_register, start_test_server
from tools.executor import ToolExecutor
from tools.models import ToolCall, ToolPermission
from tools.registry import ToolRegistry

READ, WRITE = ToolPermission.READ, ToolPermission.WRITE

EXPECTED = {
    "mcp.test.echo_text": READ,
    "mcp.test.add_numbers": READ,
    "mcp.test.read_test_file": READ,
    "mcp.test.write_test_file": WRITE,
    "mcp.test.fail_tool": READ,
    "mcp.test.slow_tool": READ,
}


@pytest.fixture
def mcp(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "hello.txt").write_text("Hello from MCP!", encoding="utf-8")
    clients = []

    def _start(call_timeout=10.0, slow_seconds=3):
        client = start_test_server(str(ws), call_timeout=call_timeout, slow_seconds=slow_seconds)
        clients.append(client)
        reg = ToolRegistry()
        tools = discover_and_register(reg, client, call_timeout=call_timeout)
        return SimpleNamespace(client=client, registry=reg, executor=ToolExecutor(reg),
                               tools=tools, workspace=str(ws))

    yield _start
    for c in clients:
        c.shutdown()


def _run(m, name, args, confirmer=None):
    call = ToolCall("c1", name, args)
    if confirmer is not None:
        return confirmation.resolve_with_confirmation(m.executor, call, confirmer=confirmer)
    return m.executor.execute(call)


# ---- discovery ----

def test_discovery_registers_all_tools_with_permissions(mcp):
    m = mcp()
    assert {t.name for t in m.tools} == set(EXPECTED)
    for name, perm in EXPECTED.items():
        assert m.registry.has(name)
        assert m.registry.get(name).permission is perm


def test_discovered_tools_are_llm_callable_and_offered(mcp):
    m = mcp()
    offered = {d.name for d in m.registry.enabled_definitions()}
    assert set(EXPECTED) <= offered  # MCP tools reach the local LLM like any tool


# ---- read-class execution ----

def test_echo_text(mcp):
    r = _run(mcp(), "mcp.test.echo_text", {"text": "hello"})
    assert r.success is True and r.data == {"text": "hello"}


def test_add_numbers(mcp):
    r = _run(mcp(), "mcp.test.add_numbers", {"a": 2, "b": 3})
    assert r.success is True and r.data == {"sum": 5}


def test_read_test_file(mcp):
    r = _run(mcp(), "mcp.test.read_test_file", {"path": "hello.txt"})
    assert r.success is True and r.data == {"content": "Hello from MCP!"}


# ---- write-class: confirmation ----

def test_write_requires_confirmation_and_does_not_create_file(mcp):
    m = mcp()
    r = _run(m, "mcp.test.write_test_file", {"path": "notes.txt", "content": "hi"})
    assert r.success is False and r.error.code == "TOOL_CONFIRMATION_REQUIRED"
    assert not os.path.exists(os.path.join(m.workspace, "notes.txt"))


def test_write_creates_file_after_approval(mcp):
    m = mcp()
    r = _run(m, "mcp.test.write_test_file", {"path": "notes.txt", "content": "hi there"},
             confirmer=lambda s: True)
    assert r.success is True
    path = os.path.join(m.workspace, "notes.txt")
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as f:
        assert f.read() == "hi there"


def test_write_declined_does_not_create_file(mcp):
    m = mcp()
    r = _run(m, "mcp.test.write_test_file", {"path": "declined.txt", "content": "x"},
             confirmer=lambda s: False)
    assert r.success is False and r.error.code == "TOOL_CONFIRMATION_DECLINED"
    assert not os.path.exists(os.path.join(m.workspace, "declined.txt"))


# ---- timeout / failure normalization ----

def test_slow_tool_times_out_with_mcp_timeout(mcp):
    m = mcp(call_timeout=1.0, slow_seconds=3)
    r = _run(m, "mcp.test.slow_tool", {})
    assert r.success is False and r.error.code == "MCP_TIMEOUT"


def test_fail_tool_is_normalized(mcp):
    r = _run(mcp(), "mcp.test.fail_tool", {})
    assert r.success is False and r.error.code == "MCP_CALL_FAILED"
    assert "fail" in r.error.message.lower()


def test_no_stack_trace_leaks_on_failure(mcp):
    r = _run(mcp(), "mcp.test.fail_tool", {})
    assert "Traceback" not in (r.error.message or "")


# ---- sandbox ----

def test_read_outside_workspace_is_rejected(mcp):
    r = _run(mcp(), "mcp.test.read_test_file", {"path": "../escape.txt"})
    assert r.success is False and r.error.code == "MCP_CALL_FAILED"


def test_write_outside_workspace_is_rejected(mcp):
    m = mcp()
    r = _run(m, "mcp.test.write_test_file", {"path": "../evil.txt", "content": "x"},
             confirmer=lambda s: True)
    assert r.success is False and r.error.code == "MCP_CALL_FAILED"
    assert not os.path.exists(os.path.join(os.path.dirname(m.workspace), "evil.txt"))


# ---- lifecycle ----

def test_shutdown_terminates_the_server_process(mcp):
    m = mcp()
    proc = m.client._proc
    assert proc is not None and proc.poll() is None
    m.client.shutdown()
    assert m.client._proc is None
    for _ in range(60):
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    assert proc.poll() is not None
