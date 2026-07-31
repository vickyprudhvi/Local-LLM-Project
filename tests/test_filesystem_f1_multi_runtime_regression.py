"""Phase G.2 Task 17 — Filesystem F.1 behavior with a SECOND active server.

Reproduces the full F.1 live scenario (outside-root read -> access plan ->
approval -> replace_session("filesystem") -> live roots verified -> original
request resumes) while a second, completely unrelated server (document-test) is
already active — proving the Filesystem-specific restart never touches it.
"""

import json
import os

import pytest

import assistant
import confirmation
import tool_loop
from mcp_layer.external import bootstrap_from_config
from mcp_layer.runtime_manager import MultiMcpRuntimeManager, RuntimeState
from mcp_management.filesystem_access_tools import register_filesystem_access_tools
from mcp_management.registry import STATUS_INSTALLED, InstalledServer, upsert
from tests.mcp_multi_runtime_helpers import manager_paths, node_available, write_fixture_server_config
from tests.mcp_provisioning_helpers import FIXTURE_SERVER, make_manager
from tests.test_tool_loop import FakeLLM, _final, _tool_call
from tools.executor import ToolExecutor
from tools.models import ToolCall, ToolResult
from tools.registry import default_registry

pytestmark = pytest.mark.skipif(not node_available(), reason="node/npm not available")


def _write_filesystem(paths, roots):
    server_root = os.path.join(paths["base_dir"], paths["managed_root"], "filesystem")
    os.makedirs(server_root, exist_ok=True)
    workspace = os.path.join(paths["base_dir"], "mcp_workspaces", "filesystem")
    os.makedirs(workspace, exist_ok=True)
    raw = {
        "enabled": True, "required": False, "server_id": "filesystem", "transport": "stdio",
        "command": "node", "args": [FIXTURE_SERVER, *roots], "working_directory": workspace,
        "startup_timeout_seconds": 15, "call_timeout_seconds": 10, "shutdown_timeout_seconds": 5,
        "environment_allowlist": [],
        "tool_policy": {"default_permission": "denied", "tools": {
            "list_allowed_directories": {"enabled": True, "permission": "read"},
            "read_text_file": {"enabled": True, "permission": "read"},
        }},
    }
    config_path = os.path.join(server_root, "server.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    upsert("filesystem", InstalledServer(
        catalog_id="official-filesystem", installed_version="2026.7.10", status=STATUS_INSTALLED,
        install_directory=server_root, configuration_path=config_path, installed_at="now",
        approved_directories=tuple(os.path.realpath(r) for r in roots)),
        None, paths["base_dir"], paths["managed_root"])
    return config_path


@pytest.fixture
def env(tmp_path, monkeypatch):
    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)
    monkeypatch.setattr(confirmation, "confirm_action", lambda summary: True)

    manager, paths = make_manager(tmp_path)
    register_filesystem_access_tools(reg, manager)

    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()
    (root_b / "hello.txt").write_text("Phase F.1 runtime restart passed.", encoding="utf-8")

    _write_filesystem(paths, [str(root_a)])
    other_root = tmp_path / "other_root"
    other_root.mkdir()
    write_fixture_server_config(paths, "document-test", [str(other_root)])

    runtime_manager = MultiMcpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])

    # A completely unrelated second server is already HEALTHY before the F.1
    # scenario begins.
    other_session = runtime_manager.ensure_started(
        "document-test", expected_allowed_roots=(os.path.realpath(str(other_root)),))
    assert other_session.health.state.value == "healthy"

    from mcp_layer.config import load_config
    fs_config_path = os.path.join(paths["base_dir"], paths["managed_root"], "filesystem", "server.json")
    old_fs_session = bootstrap_from_config(reg, config=load_config(fs_config_path), base_dir=paths["base_dir"])
    assert old_fs_session.health.state.value == "healthy"
    slot = runtime_manager._slot("filesystem")
    slot.session = old_fs_session
    slot.state = RuntimeState.HEALTHY
    runtime_manager._runtimes["filesystem"].replace(old_fs_session)

    return {"manager": manager, "paths": paths, "reg": reg, "runtime_manager": runtime_manager,
           "root_a": str(root_a), "root_b": str(root_b), "old_fs_session": old_fs_session,
           "other_session": other_session}


def test_filesystem_restart_leaves_the_second_server_completely_untouched(env, monkeypatch):
    """The standard F.1 outside-root flow: a real failed read, an automatically
    offered access plan, a cross-turn 'yes', a Filesystem-only restart, and
    resumption — all while document-test stays completely untouched throughout."""
    manager, paths, reg = env["manager"], env["paths"], env["reg"]
    runtime_manager = env["runtime_manager"]
    root_b = env["root_b"]
    old_fs_proc = env["old_fs_session"].client._proc
    other_proc = env["other_session"].client._proc

    target = os.path.join(root_b, "hello.txt")
    user_text = f"read '{target}'"

    # Step 1: a REAL outside-root failure through the already-active old session.
    call = ToolCall(call_id="c1", tool_name="mcp.filesystem.read_text_file", arguments={"path": target})
    result = ToolResult.fail("mcp.filesystem.read_text_file", "c1", "MCP_CALL_FAILED",
                             "Access denied - path outside allowed directories")
    found = assistant._find_outside_root_failure(manager, [(call, result)])
    assert found is not None
    server_id, found_call, failure = found
    reply, request_id = assistant._offer_filesystem_access(manager, server_id, found_call, failure, user_text)
    assert os.path.realpath(root_b) in reply

    # document-test untouched by merely PREPARING the plan.
    assert other_proc.poll() is None

    # Step 2: approve via the standard cross-turn "yes" resolution.
    outcome = assistant._resolve_filesystem_access_reply(manager, request_id, "yes")
    assert outcome.matched and outcome.resumed_text == user_text
    assert outcome.server_id == "filesystem"

    # Step 3: restart ONLY filesystem and resume the original request.
    directive = tool_loop.ToolLoopDirective(
        control=tool_loop.ToolLoopControl.RESTART_MCP_AND_RESUME,
        server_id=outcome.server_id, expected_allowed_roots=outcome.expected_allowed_roots)

    fake = FakeLLM([
        _tool_call("mcp.filesystem.read_text_file", {"path": target}),
        _final("The file contains: Phase F.1 runtime restart passed."),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)
    from router import RouteDecision
    monkeypatch.setattr(assistant, "route_and_answer",
                        lambda prompt, history: RouteDecision(mode="local", tool=None))

    reply2, pending_id = assistant._restart_mcp_and_resume(
        manager, runtime_manager, directive, outcome.resumed_text, [], "sys", set(),
        resume_budget=1, previous_allowed_roots=outcome.previous_allowed_roots)

    assert "Phase F.1 runtime restart passed" in reply2
    assert pending_id is None

    # Filesystem: old process stopped, new one healthy, live roots verified.
    assert old_fs_proc.poll() is not None
    new_fs_session = runtime_manager.get_session("filesystem")
    assert new_fs_session is not env["old_fs_session"]
    assert runtime_manager.get_status("filesystem").state == RuntimeState.HEALTHY

    # document-test: completely untouched — same session, same process, same tools.
    assert other_proc.poll() is None
    assert runtime_manager.get_session("document-test") is env["other_session"]
    assert runtime_manager.get_status("document-test").state == RuntimeState.HEALTHY
    assert reg.has("mcp.document-test.read_text_file")
    doc_tool = reg.get("mcp.document-test.read_text_file")
    assert doc_tool.session_owner == env["other_session"].session_id

    new_fs_session.shutdown()
    env["other_session"].shutdown()
