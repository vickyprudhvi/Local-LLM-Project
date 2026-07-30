"""Phase F.1 hotfix Task 5/6/8 — McpRuntimeManager.replace_active_session.

Uses the REAL Node fixture filesystem server end to end: a real old child process,
a real new child process, and a real `list_allowed_directories` call against the
live new server — never trusting the config file, registry, or plan state alone
for root verification. This is the exact defect the bug report described: the
managed config is updated correctly, but the running assistant kept using the old
McpTool objects bound to the old, now-stale client.
"""

import json
import os

import pytest

from mcp_layer.errors import McpError
from mcp_layer.external import bootstrap_from_config
from mcp_layer.runtime_manager import ActiveMcpRuntime, McpRuntimeManager
from mcp_management.registry import STATUS_INSTALLED, InstalledServer, upsert
from tests.mcp_provisioning_helpers import FIXTURE_SERVER, manager_paths, node_available
from tools.models import MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH
from tools.registry import ToolRegistry

pytestmark = pytest.mark.skipif(not node_available(), reason="node/npm not available")


def _write_managed_config(paths, roots, server_id="filesystem"):
    server_root = os.path.join(paths["base_dir"], paths["managed_root"], server_id)
    os.makedirs(server_root, exist_ok=True)
    config_path = os.path.join(server_root, "server.json")
    workspace = os.path.join(paths["base_dir"], "mcp_workspaces", server_id)
    os.makedirs(workspace, exist_ok=True)
    raw = {
        "enabled": True, "required": False, "server_id": server_id,
        "display_name": "Filesystem MCP Server", "transport": "stdio",
        "command": "node", "args": [FIXTURE_SERVER, *roots],
        "working_directory": workspace,
        "startup_timeout_seconds": 15, "call_timeout_seconds": 10,
        "shutdown_timeout_seconds": 5, "environment_allowlist": [],
        "tool_policy": {"default_permission": "denied", "tools": {
            "list_allowed_directories": {"enabled": True, "permission": "read"},
            "read_text_file": {"enabled": True, "permission": "read"},
        }},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    upsert(server_id, InstalledServer(
        catalog_id="official-filesystem", installed_version="2026.7.10",
        status=STATUS_INSTALLED, install_directory=server_root,
        configuration_path=config_path, installed_at="now",
        approved_directories=tuple(roots)),
        None, paths["base_dir"], paths["managed_root"])
    return config_path


@pytest.fixture
def env(tmp_path):
    paths = manager_paths(tmp_path)
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "hello.txt").write_text("from root a", encoding="utf-8")
    (root_b / "hello.txt").write_text("from root b", encoding="utf-8")
    return paths, str(root_a), str(root_b)


def _bootstrap_old_session(reg, paths, roots):
    from mcp_layer.config import load_config

    config_path = _write_managed_config(paths, roots)
    config = load_config(config_path)
    return bootstrap_from_config(reg, config=config, base_dir=paths["base_dir"])


def test_replace_active_session_stops_old_starts_new_and_verifies_roots(env):
    paths, root_a, root_b = env
    reg = ToolRegistry()
    old_session = _bootstrap_old_session(reg, paths, [root_a])
    assert old_session.health.state.value == "healthy"
    old_client = old_session.client
    old_proc = old_client._proc
    runtime = ActiveMcpRuntime(old_session)

    # Simulate what mcp_management.filesystem_access_update already did: the
    # managed config on disk now carries BOTH roots.
    _write_managed_config(paths, [root_a, root_b])

    coordinator = McpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])
    new_session = coordinator.replace_active_session(
        runtime, "filesystem", (os.path.realpath(root_a), os.path.realpath(root_b)))

    # Old process actually stopped — no orphan.
    assert old_proc.poll() is not None
    # The active runtime reference was atomically replaced.
    assert runtime.session is new_session
    assert new_session is not old_session
    assert new_session.client is not old_client

    # The registered tool now answers via the NEW client and sees the NEW root.
    tool = reg.get("mcp.filesystem.read_text_file")
    assert tool.session_owner == new_session.session_id
    result = tool.execute({"path": os.path.join(root_b, "hello.txt")})
    assert result["content"] == "from root b"

    new_session.shutdown()


def test_missing_expected_root_is_rejected(env):
    paths, root_a, root_b = env
    reg = ToolRegistry()
    old_session = _bootstrap_old_session(reg, paths, [root_a])
    runtime = ActiveMcpRuntime(old_session)
    _write_managed_config(paths, [root_a, root_b])

    coordinator = McpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])
    with pytest.raises(McpError) as exc:
        # Expect only root_a — the live server actually reports BOTH -> mismatch.
        coordinator.replace_active_session(runtime, "filesystem", (os.path.realpath(root_a),))
    assert exc.value.code == MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH


def test_unexpected_extra_root_is_rejected(env, tmp_path):
    paths, root_a, root_b = env
    root_c = tmp_path / "root_c"
    root_c.mkdir()
    reg = ToolRegistry()
    old_session = _bootstrap_old_session(reg, paths, [root_a])
    runtime = ActiveMcpRuntime(old_session)
    _write_managed_config(paths, [root_a, root_b])

    coordinator = McpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])
    with pytest.raises(McpError) as exc:
        # Expect root_a, root_b, AND root_c — the live server only reports the
        # first two -> mismatch (an unexpected extra root was never granted).
        coordinator.replace_active_session(
            runtime, "filesystem",
            (os.path.realpath(root_a), os.path.realpath(root_b), os.path.realpath(root_c)))
    assert exc.value.code == MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH


def test_broader_parent_root_is_rejected(env):
    paths, root_a, root_b = env
    reg = ToolRegistry()
    old_session = _bootstrap_old_session(reg, paths, [root_a])
    runtime = ActiveMcpRuntime(old_session)
    _write_managed_config(paths, [root_a, root_b])

    coordinator = McpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])
    with pytest.raises(McpError) as exc:
        # Expect the PARENT of root_a/root_b instead of the actual granted roots.
        coordinator.replace_active_session(
            runtime, "filesystem", (os.path.dirname(os.path.realpath(root_a)),))
    assert exc.value.code == MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH


def test_config_not_active_for_this_server_is_rejected(env):
    paths, root_a, _root_b = env
    reg = ToolRegistry()
    runtime = ActiveMcpRuntime(None)
    coordinator = McpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])
    with pytest.raises(McpError):
        coordinator.replace_active_session(runtime, "not-installed", (root_a,))
