"""Phase F.1 hotfix Task 7 — restart failure and rollback.

When the NEW runtime fails to come up healthy or to report the expected live
roots, and `previous_allowed_roots` is available, the coordinator restores the
previous managed config + registry approved-root state and restarts the PREVIOUS
(known-good) configuration so the assistant is never left without a working
session — but the original blocked request is never resumed on this path (the
caller, not this module, decides that; see assistant.py).
"""

import json
import os

import pytest

from mcp_layer.errors import McpError
from mcp_layer.external import bootstrap_from_config
from mcp_layer.runtime_manager import ActiveMcpRuntime, McpRuntimeManager
from mcp_management.registry import STATUS_INSTALLED, InstalledServer, get_installed, upsert
from tests.mcp_provisioning_helpers import FIXTURE_SERVER, manager_paths, node_available
from tools.models import MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH, MCP_RUNTIME_ROLLBACK_FAILED
from tools.registry import ToolRegistry

pytestmark = pytest.mark.skipif(not node_available(), reason="node/npm not available")


def _write_config(paths, roots, command="node", server_id="filesystem"):
    server_root = os.path.join(paths["base_dir"], paths["managed_root"], server_id)
    os.makedirs(server_root, exist_ok=True)
    config_path = os.path.join(server_root, "server.json")
    workspace = os.path.join(paths["base_dir"], "mcp_workspaces", server_id)
    os.makedirs(workspace, exist_ok=True)
    raw = {
        "enabled": True, "required": False, "server_id": server_id,
        "display_name": "Filesystem MCP Server", "transport": "stdio",
        "command": command, "args": [FIXTURE_SERVER, *roots],
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
    return paths, str(os.path.realpath(root_a)), str(os.path.realpath(root_b))


def _bootstrap(reg, paths, roots):
    from mcp_layer.config import load_config

    config_path = _write_config(paths, roots)
    return bootstrap_from_config(reg, config=load_config(config_path), base_dir=paths["base_dir"])


def test_root_mismatch_rolls_back_to_previous_config_and_restarts_it(env):
    paths, root_a, root_b = env
    reg = ToolRegistry()
    old_session = _bootstrap(reg, paths, [root_a])
    runtime = ActiveMcpRuntime(old_session)

    # The "new" config on disk deliberately does NOT include root_b, so asking for
    # root_b as an expected root fails live-root verification.
    _write_config(paths, [root_a])

    coordinator = McpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])
    with pytest.raises(McpError) as exc:
        coordinator.replace_active_session(runtime, "filesystem", (root_a, root_b),
                                           previous_allowed_roots=(root_a,))
    assert exc.value.code == MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH

    # The assistant is never left without a session: the restored one serves the
    # PREVIOUS (proven-good) root live.
    restored = runtime.session
    assert restored is not None
    assert restored is not old_session
    tool = reg.get("mcp.filesystem.read_text_file")
    result = tool.execute({"path": os.path.join(root_a, "hello.txt")})
    assert result["content"] == "from root a"

    # The managed config and registry were both restored to the previous root set.
    config_path = os.path.join(paths["base_dir"], paths["managed_root"], "filesystem", "server.json")
    with open(config_path, encoding="utf-8") as f:
        restored_raw = json.load(f)
    assert restored_raw["args"][1:] == [root_a]
    installed = get_installed("filesystem", None, paths["base_dir"], paths["managed_root"])
    assert installed.approved_directories == (root_a,)

    restored.shutdown()


def test_restart_failure_without_previous_roots_leaves_no_active_session(env):
    paths, root_a, root_b = env
    reg = ToolRegistry()
    old_session = _bootstrap(reg, paths, [root_a])
    runtime = ActiveMcpRuntime(old_session)
    _write_config(paths, [root_a])  # still only root_a -> mismatch, same as above

    coordinator = McpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])
    with pytest.raises(McpError) as exc:
        coordinator.replace_active_session(runtime, "filesystem", (root_a, root_b))
    assert exc.value.code == MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH
    assert runtime.session is None  # no rollback context given -> no session left, never silently stale


def test_rollback_itself_failing_is_reported_distinctly(env):
    paths, root_a, root_b = env
    reg = ToolRegistry()
    old_session = _bootstrap(reg, paths, [root_a])
    runtime = ActiveMcpRuntime(old_session)
    # Both the "new" restart AND any rollback attempt use the same broken command,
    # so the rollback attempt fails too.
    _write_config(paths, [root_a], command="this-command-does-not-exist-anywhere")

    coordinator = McpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])
    with pytest.raises(McpError) as exc:
        coordinator.replace_active_session(runtime, "filesystem", (root_a, root_b),
                                           previous_allowed_roots=(root_a,))
    # The rollback attempt (same broken command) also fails -> reported distinctly
    # so the assistant knows there is now NO working session, rather than silently
    # reporting the original restart failure as if a fallback were in place.
    assert exc.value.code == MCP_RUNTIME_ROLLBACK_FAILED
