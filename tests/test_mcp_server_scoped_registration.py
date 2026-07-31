"""Phase G.2 Task 7/9/16(G/H/I/N) — server-scoped tool ownership and validators.

Two independently active servers (filesystem-test, document-test): stopping or
restarting one must only ever touch ITS OWN remote tools, never the other
server's, and never the built-in access-management tools a totally separate
registration path owns.
"""

import os

import pytest

from mcp_layer.runtime_manager import MultiMcpRuntimeManager
from mcp_management.filesystem_access_tools import register_filesystem_access_tools
from mcp_management.manager import McpProvisioningManager
from tests.mcp_multi_runtime_helpers import manager_paths, node_available, write_fixture_server_config
from tests.mcp_provisioning_helpers import make_catalog
from tools.registry import ToolRegistry

pytestmark = pytest.mark.skipif(not node_available(), reason="node/npm not available")


@pytest.fixture
def env(tmp_path):
    paths = manager_paths(tmp_path)
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()
    reg = ToolRegistry()
    # Built-ins: registered once, at startup, by a totally separate path — never
    # touched by any server's session lifecycle.
    manager = McpProvisioningManager(catalog=make_catalog(), base_dir=paths["base_dir"],
                                     managed_root=paths["managed_root"])
    register_filesystem_access_tools(reg, manager)
    rm = MultiMcpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])
    write_fixture_server_config(paths, "filesystem-test", [root_a])
    write_fixture_server_config(paths, "document-test", [root_b])
    return {"paths": paths, "reg": reg, "rm": rm, "root_a": str(root_a), "root_b": str(root_b)}


_BUILTINS = ("mcp.filesystem.access.list", "mcp.filesystem.access.plan",
            "mcp.filesystem.access.add", "mcp.filesystem.access.remove")


def test_two_servers_have_independently_owned_tools(env):
    reg, rm = env["reg"], env["rm"]
    fs = rm.ensure_started("filesystem-test", expected_allowed_roots=(os.path.realpath(env["root_a"]),))
    doc = rm.ensure_started("document-test", expected_allowed_roots=(os.path.realpath(env["root_b"]),))

    assert fs.session_id != doc.session_id
    fs_tool = reg.get("mcp.filesystem-test.read_text_file")
    doc_tool = reg.get("mcp.document-test.read_text_file")
    assert fs_tool.session_owner == fs.session_id
    assert doc_tool.session_owner == doc.session_id
    for name in _BUILTINS:
        assert reg.has(name)

    fs.shutdown()
    doc.shutdown()


def test_stopping_one_server_removes_only_its_own_tools(env):
    reg, rm = env["reg"], env["rm"]
    fs = rm.ensure_started("filesystem-test", expected_allowed_roots=(os.path.realpath(env["root_a"]),))
    doc = rm.ensure_started("document-test", expected_allowed_roots=(os.path.realpath(env["root_b"]),))
    doc_proc = doc.client._proc

    rm.stop("filesystem-test")

    assert not any(name.startswith("mcp.filesystem-test.") for name in reg._tools)
    assert reg.has("mcp.document-test.read_text_file")  # untouched
    for name in _BUILTINS:
        assert reg.has(name)  # untouched
    assert doc_proc.poll() is None  # document-test process still alive
    assert rm.get_session("document-test") is doc

    doc.shutdown()


def test_restarting_one_server_rebinds_only_its_own_tools(env):
    reg, rm = env["reg"], env["rm"]
    fs = rm.ensure_started("filesystem-test", expected_allowed_roots=(os.path.realpath(env["root_a"]),))
    doc = rm.ensure_started("document-test", expected_allowed_roots=(os.path.realpath(env["root_b"]),))
    doc_proc = doc.client._proc

    new_root = os.path.dirname(env["root_a"])
    new_roots = write_fixture_server_config(env["paths"], "filesystem-test", [env["root_a"], new_root])
    new_fs = rm.replace_session("filesystem-test", expected_allowed_roots=new_roots)

    assert new_fs is not fs
    fs_tool = reg.get("mcp.filesystem-test.read_text_file")
    assert fs_tool.session_owner == new_fs.session_id

    # document-test: completely unaffected by filesystem-test's restart.
    assert rm.get_session("document-test") is doc
    assert doc_proc.poll() is None
    doc_tool = reg.get("mcp.document-test.read_text_file")
    assert doc_tool.session_owner == doc.session_id
    for name in _BUILTINS:
        assert reg.has(name)

    new_fs.shutdown()
    doc.shutdown()


def test_server_specific_validator_used_only_for_its_own_server(env):
    """The Filesystem root validator must never be applied to a different
    server_id — verified indirectly: document-test starts successfully with NO
    expected_allowed_roots even though filesystem-test's validator is registered
    for "filesystem-test" only, not for "document-test"."""
    from mcp_layer.runtime_manager import FilesystemRootValidator

    reg, rm = env["reg"], MultiMcpRuntimeManager(
        env["reg"], base_dir=env["paths"]["base_dir"], managed_root=env["paths"]["managed_root"],
        validators={"filesystem-test": FilesystemRootValidator()})

    # document-test has NO registered validator -> falls back to the generic one,
    # which does not require list_allowed_directories at all.
    doc = rm.ensure_started("document-test")
    assert doc.health.state.value == "healthy"
    doc.shutdown()
