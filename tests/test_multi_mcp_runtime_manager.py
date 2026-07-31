"""Phase G.2 Task 2/3/4/6/9/13/16 — MultiMcpRuntimeManager core behavior.

Uses the real Node fixture filesystem server (never a fake production catalog
entry) for genuine bootstrap/verify/reuse/stop mechanics, and lightweight raw
config writes for the not-installed/disabled/invalid-config error paths.
"""

import json
import os

import pytest

from mcp_layer.errors import McpError
from mcp_layer.runtime_manager import MultiMcpRuntimeManager, RuntimeState
from tests.mcp_multi_runtime_helpers import manager_paths, node_available, write_fixture_server_config
from tools.models import (
    MCP_EXPECTED_TOOL_MISSING,
    MCP_SERVER_CONFIG_INVALID,
    MCP_SERVER_DISABLED,
    MCP_SERVER_NOT_INSTALLED,
)
from tools.registry import ToolRegistry

pytestmark = pytest.mark.skipif(not node_available(), reason="node/npm not available")


@pytest.fixture
def env(tmp_path):
    paths = manager_paths(tmp_path)
    root = tmp_path / "root_a"
    root.mkdir()
    (root / "hello.txt").write_text("hi", encoding="utf-8")
    return paths, str(root)


def _manager(reg, paths):
    return MultiMcpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])


# ---- A/B: nothing starts merely by constructing the manager ----

def test_constructing_the_manager_starts_nothing(env):
    paths, root = env
    reg = ToolRegistry()
    rm = _manager(reg, paths)
    status = rm.get_status("filesystem-test")
    assert status.state == RuntimeState.NOT_INSTALLED
    assert rm.get_session("filesystem-test") is None


# ---- C: lazy startup on first ensure_started ----

def test_ensure_started_lazily_bootstraps_and_registers_tools(env):
    paths, root = env
    write_fixture_server_config(paths, "filesystem-test", [root])
    reg = ToolRegistry()
    rm = _manager(reg, paths)

    # A config exists but nothing was started yet — INACTIVE, not NOT_INSTALLED.
    assert rm.get_status("filesystem-test").state == RuntimeState.INACTIVE

    session = rm.ensure_started("filesystem-test", expected_allowed_roots=(os.path.realpath(root),))
    assert session.health.state.value == "healthy"
    assert rm.get_status("filesystem-test").state == RuntimeState.HEALTHY
    assert reg.has("mcp.filesystem-test.read_text_file")
    session.shutdown()


# ---- D: reuse on second call, no re-bootstrap ----

def test_second_ensure_started_reuses_the_healthy_session(env):
    paths, root = env
    write_fixture_server_config(paths, "filesystem-test", [root])
    reg = ToolRegistry()
    rm = _manager(reg, paths)

    first = rm.ensure_started("filesystem-test", expected_allowed_roots=(os.path.realpath(root),))
    first_pid = first.client._proc.pid
    second = rm.ensure_started("filesystem-test", expected_allowed_roots=(os.path.realpath(root),))
    assert second is first
    assert second.client._proc.pid == first_pid
    first.shutdown()


# ---- E/F: not-installed / unsupported paths never start a process ----

def test_ensure_started_for_missing_config_raises_not_installed(env):
    paths, root = env
    reg = ToolRegistry()
    rm = _manager(reg, paths)
    # No config was ever written: NOT_INSTALLED before any attempt...
    assert rm.get_status("document-test").state == RuntimeState.NOT_INSTALLED
    with pytest.raises(McpError) as exc:
        rm.ensure_started("document-test")
    assert exc.value.code == MCP_SERVER_NOT_INSTALLED
    # ...and FAILED (with the same error code recorded) after the attempt.
    status = rm.get_status("document-test")
    assert status.state == RuntimeState.FAILED
    assert status.last_error_code == MCP_SERVER_NOT_INSTALLED


# ---- O: disabled server ----

def test_ensure_started_for_disabled_server_raises_disabled(env):
    paths, root = env
    write_fixture_server_config(paths, "filesystem-test", [root])
    config_path = os.path.join(paths["base_dir"], paths["managed_root"], "filesystem-test", "server.json")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["enabled"] = False
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(raw, f)

    reg = ToolRegistry()
    rm = _manager(reg, paths)
    with pytest.raises(McpError) as exc:
        rm.ensure_started("filesystem-test")
    assert exc.value.code == MCP_SERVER_DISABLED
    assert rm.get_session("filesystem-test") is None


# ---- Q: invalid config ----

def test_ensure_started_for_invalid_config_raises_config_invalid(env):
    paths, root = env
    server_root = os.path.join(paths["base_dir"], paths["managed_root"], "filesystem-test")
    os.makedirs(server_root, exist_ok=True)
    with open(os.path.join(server_root, "server.json"), "w", encoding="utf-8") as f:
        f.write("{not valid json")

    reg = ToolRegistry()
    rm = _manager(reg, paths)
    with pytest.raises(McpError) as exc:
        rm.ensure_started("filesystem-test")
    assert exc.value.code == MCP_SERVER_CONFIG_INVALID


# ---- generic validator: expected-tool check for a non-filesystem server ----

def test_generic_validator_flags_a_missing_expected_tool(env):
    paths, root = env
    write_fixture_server_config(paths, "document-test", [root])
    reg = ToolRegistry()
    rm = _manager(reg, paths)
    with pytest.raises(McpError) as exc:
        rm.ensure_started("document-test", expected_tools=("convert_to_markdown",))
    assert exc.value.code == MCP_EXPECTED_TOOL_MISSING
    assert rm.get_status("document-test").state == RuntimeState.FAILED


def test_generic_validator_passes_when_expected_tools_present(env):
    paths, root = env
    write_fixture_server_config(paths, "document-test", [root])
    reg = ToolRegistry()
    rm = _manager(reg, paths)
    session = rm.ensure_started("document-test", expected_tools=("read_text_file",))
    assert session.health.state.value == "healthy"
    session.shutdown()


# ---- replace_session ----

def test_replace_session_swaps_only_the_named_server(env):
    paths, root = env
    write_fixture_server_config(paths, "filesystem-test", [root])
    reg = ToolRegistry()
    rm = _manager(reg, paths)
    old = rm.ensure_started("filesystem-test", expected_allowed_roots=(os.path.realpath(root),))
    old_proc = old.client._proc  # capture now: shutdown() clears client._proc
    old_pid = old_proc.pid

    root_b = os.path.dirname(root)  # any second directory; just needs to differ
    new_roots = write_fixture_server_config(paths, "filesystem-test", [root, root_b])
    new_session = rm.replace_session("filesystem-test", expected_allowed_roots=new_roots)

    assert new_session is not old
    assert new_session.client._proc.pid != old_pid
    assert old_proc.poll() is not None  # old process actually stopped
    assert rm.get_status("filesystem-test").state == RuntimeState.HEALTHY
    new_session.shutdown()


# ---- stop / stop_all ----

def test_stop_terminates_process_and_unregisters_tools(env):
    paths, root = env
    write_fixture_server_config(paths, "filesystem-test", [root])
    reg = ToolRegistry()
    rm = _manager(reg, paths)
    session = rm.ensure_started("filesystem-test", expected_allowed_roots=(os.path.realpath(root),))
    proc = session.client._proc
    assert reg.has("mcp.filesystem-test.read_text_file")

    rm.stop("filesystem-test")

    assert proc.poll() is not None
    assert not reg.has("mcp.filesystem-test.read_text_file")
    assert rm.get_session("filesystem-test") is None
    assert rm.get_status("filesystem-test").state == RuntimeState.STOPPED


def test_stop_is_idempotent(env):
    paths, root = env
    write_fixture_server_config(paths, "filesystem-test", [root])
    reg = ToolRegistry()
    rm = _manager(reg, paths)
    rm.stop("filesystem-test")  # never started — must not raise
    session = rm.ensure_started("filesystem-test", expected_allowed_roots=(os.path.realpath(root),))
    rm.stop("filesystem-test")
    rm.stop("filesystem-test")  # second stop is a no-op
    assert rm.get_session("filesystem-test") is None


def test_stop_all_stops_every_active_server(env, tmp_path):
    paths, root = env
    root_b = tmp_path / "root_b"
    root_b.mkdir()
    write_fixture_server_config(paths, "filesystem-test", [root])
    write_fixture_server_config(paths, "document-test", [str(root_b)])
    reg = ToolRegistry()
    rm = _manager(reg, paths)
    s1 = rm.ensure_started("filesystem-test", expected_allowed_roots=(os.path.realpath(root),))
    s2 = rm.ensure_started("document-test", expected_allowed_roots=(os.path.realpath(str(root_b)),))
    proc1, proc2 = s1.client._proc, s2.client._proc

    errors = rm.stop_all()

    assert errors == ()
    assert proc1.poll() is not None
    assert proc2.poll() is not None
    assert rm.get_session("filesystem-test") is None
    assert rm.get_session("document-test") is None
