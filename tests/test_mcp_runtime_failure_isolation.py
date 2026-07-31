"""Phase G.2 Task 10/16(J) — one server's failure never affects another.

filesystem-test comes up HEALTHY; document-test's startup is then forced to
fail. filesystem-test's process, session, and registered tools must be
completely unaffected — no global MCP health flag, no blanket tool wipe, no
cross-server process termination.
"""

import os

import pytest

from mcp_layer.errors import McpError
from mcp_layer.runtime_manager import MultiMcpRuntimeManager, RuntimeState
from tests.mcp_multi_runtime_helpers import manager_paths, node_available, write_fixture_server_config
from tools.models import MCP_EXPECTED_TOOL_MISSING
from tools.registry import ToolRegistry

pytestmark = pytest.mark.skipif(not node_available(), reason="node/npm not available")


@pytest.fixture
def env(tmp_path):
    paths = manager_paths(tmp_path)
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()
    return paths, str(root_a), str(root_b)


def test_document_test_startup_failure_leaves_filesystem_test_untouched(env):
    paths, root_a, root_b = env
    write_fixture_server_config(paths, "filesystem-test", [root_a])
    write_fixture_server_config(paths, "document-test", [root_b])

    reg = ToolRegistry()
    rm = MultiMcpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])

    fs_session = rm.ensure_started("filesystem-test", expected_allowed_roots=(os.path.realpath(root_a),))
    assert fs_session.health.state.value == "healthy"
    fs_proc = fs_session.client._proc
    fs_tools_before = set(reg._tools) & {"mcp.filesystem-test.read_text_file"}
    assert fs_tools_before

    with pytest.raises(McpError) as exc:
        rm.ensure_started("document-test", expected_tools=("this_tool_does_not_exist",))
    assert exc.value.code == MCP_EXPECTED_TOOL_MISSING

    # filesystem-test: completely unaffected.
    assert rm.get_status("filesystem-test").state == RuntimeState.HEALTHY
    assert fs_proc.poll() is None  # still running
    assert reg.has("mcp.filesystem-test.read_text_file")
    assert rm.get_session("filesystem-test") is fs_session

    # document-test: isolated failure, no lingering tools/process.
    assert rm.get_status("document-test").state == RuntimeState.FAILED
    assert rm.get_status("document-test").last_error_code == MCP_EXPECTED_TOOL_MISSING
    assert not any(name.startswith("mcp.document-test.") for name in reg._tools)

    fs_session.shutdown()


def test_filesystem_test_restart_failure_does_not_touch_document_test(env):
    """The inverse: restarting one server and having IT fail must not disturb an
    already-healthy, unrelated server."""
    paths, root_a, root_b = env
    write_fixture_server_config(paths, "filesystem-test", [root_a])
    write_fixture_server_config(paths, "document-test", [root_b])

    reg = ToolRegistry()
    rm = MultiMcpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])

    doc_session = rm.ensure_started("document-test", expected_allowed_roots=None)
    fs_session = rm.ensure_started("filesystem-test", expected_allowed_roots=(os.path.realpath(root_a),))
    doc_proc = doc_session.client._proc

    # Force the "filesystem-test" restart to fail root verification.
    with pytest.raises(McpError):
        rm.replace_session("filesystem-test", expected_allowed_roots=("C:\\nonexistent\\root",))

    assert rm.get_status("filesystem-test").state == RuntimeState.FAILED
    # document-test remains completely healthy and untouched.
    assert rm.get_status("document-test").state == RuntimeState.HEALTHY
    assert doc_proc.poll() is None
    assert rm.get_session("document-test") is doc_session
    assert reg.has("mcp.document-test.read_text_file")

    doc_session.shutdown()
    fs_session_after = rm.get_session("filesystem-test")
    if fs_session_after is not None:
        fs_session_after.shutdown()
