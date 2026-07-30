"""Phase F.1 hotfix Task 4/11 — exact remote-tool ownership.

A runtime replacement must remove ONLY the remote tools the OLD session itself
registered — never the built-in access-management tools (mcp.filesystem.access.*),
and never a tool a NEWER session already re-registered under the same name. Uses
the REAL Node fixture filesystem server (tests/fixtures/fake_filesystem_server.js)
via mcp_layer.external.bootstrap_from_config, exactly the production code path —
two independent real sessions stand in for "the old session" and "the new one".
"""

import pytest

from mcp_layer.config import build_config
from mcp_layer.external import bootstrap_from_config
from mcp_management.filesystem_access_tools import register_filesystem_access_tools
from tests.mcp_provisioning_helpers import FIXTURE_SERVER, make_manager, node_available
from tools.registry import ToolRegistry

pytestmark = pytest.mark.skipif(not node_available(), reason="node/npm not available")


def _config(server_id, roots, workspace, call_timeout=10, startup_timeout=15):
    return build_config({
        "enabled": True, "required": False, "server_id": server_id, "transport": "stdio",
        "command": "node", "args": [FIXTURE_SERVER, *roots],
        "working_directory": workspace, "internal_test_server": False,
        "startup_timeout_seconds": startup_timeout, "call_timeout_seconds": call_timeout,
        "shutdown_timeout_seconds": 5, "environment_allowlist": [],
        "tool_policy": {"default_permission": "denied", "tools": {
            "list_allowed_directories": {"enabled": True, "permission": "read"},
            "read_text_file": {"enabled": True, "permission": "read"},
            "list_directory": {"enabled": True, "permission": "read"},
        }},
    })


@pytest.fixture
def two_roots(tmp_path):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()
    return str(root_a), str(root_b)


def test_unregister_owned_removes_only_this_sessions_tools(tmp_path, two_roots):
    root_a, root_b = two_roots
    reg = ToolRegistry()
    manager, _paths = make_manager(tmp_path)
    register_filesystem_access_tools(reg, manager)
    builtin_names = {"mcp.filesystem.access.list", "mcp.filesystem.access.plan",
                     "mcp.filesystem.access.add", "mcp.filesystem.access.remove"}
    assert builtin_names <= set(n for n in reg._tools)

    workspace = str(tmp_path / "mcp_workspaces" / "filesystem")
    import os
    os.makedirs(workspace, exist_ok=True)
    session_a = bootstrap_from_config(reg, config=_config("filesystem", [root_a], workspace),
                                      approved_root=str(tmp_path / "mcp_workspaces"))
    try:
        assert session_a.health.state.value == "healthy"
        old_remote_names = set(session_a.registered_remote_tool_names)
        assert "mcp.filesystem.read_text_file" in old_remote_names
        assert old_remote_names.isdisjoint(builtin_names)

        removed = reg.unregister_owned(session_a.registered_remote_tool_names, session_a.session_id)
        assert set(removed) == old_remote_names
        for name in old_remote_names:
            assert not reg.has(name)
        # Built-ins survive untouched.
        for name in builtin_names:
            assert reg.has(name)
    finally:
        session_a.shutdown()

    session_b = bootstrap_from_config(reg, config=_config("filesystem", [root_b], workspace),
                                      approved_root=str(tmp_path / "mcp_workspaces"))
    try:
        assert session_b.health.state.value == "healthy"
        assert reg.has("mcp.filesystem.read_text_file")
        # The tool object now bound is session B's, not a leftover from A.
        live_tool = reg.get("mcp.filesystem.read_text_file")
        assert live_tool.session_owner == session_b.session_id
        assert live_tool.session_owner != session_a.session_id

        # A stale attempt to remove session A's (already-gone) names must never
        # touch session B's replacement registered under the same name.
        reg.unregister_owned(session_a.registered_remote_tool_names, session_a.session_id)
        assert reg.has("mcp.filesystem.read_text_file")
        assert reg.get("mcp.filesystem.read_text_file").session_owner == session_b.session_id

        for name in builtin_names:
            assert reg.has(name)
    finally:
        session_b.shutdown()


def test_unregister_many_and_unregister_are_simple_primitives():
    reg = ToolRegistry()

    class _T:
        def __init__(self, name):
            self.name = name
            self.enabled = True

    reg.register(_T("a"))
    reg.register(_T("b"))
    assert reg.unregister("missing") is False
    assert reg.unregister("a") is True
    assert not reg.has("a")
    assert reg.unregister_many(["b", "missing"]) == 1
    assert not reg.has("b")
