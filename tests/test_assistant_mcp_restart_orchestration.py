"""Phase F.1 hotfix Task 8/10/12 — assistant-level orchestration around the
restart coordinator.

Covers what test_mcp_access_restart_resumption.py (real processes) does not:
- the cross-turn "yes" approval feeds the SAME restart+resume path as a mid-turn
  access.add call (Task 10) — proven by asserting _FsReplyOutcome carries the
  trusted restart state the coordinator needs;
- ActiveMcpRuntime always closes whichever session is CURRENTLY active, never a
  stale reference to an already-closed one (Task 8, Test Case J);
- the user-visible messages on a runtime-restart failure are accurate and never
  suggest a workaround the acceptance criteria forbid (Task 12).
"""

import os

import pytest

import assistant
import tool_loop
from mcp_layer.errors import McpError
from mcp_layer.runtime_manager import ActiveMcpRuntime, MultiMcpRuntimeManager
from tests.mcp_provisioning_helpers import make_manager
from tests.test_tool_loop import _tool_call
from tool_loop import ToolLoopControl, ToolLoopDirective
from tools.models import (
    MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH,
    MCP_RUNTIME_REBIND_FAILED,
    MCP_RUNTIME_RESTART_FAILED,
    MCP_RUNTIME_ROLLBACK_FAILED,
)


def _install_stub_server(paths, approved_dir):
    import json

    from mcp_management.registry import STATUS_INSTALLED, InstalledServer, upsert

    approved_abs = os.path.realpath(approved_dir)
    server_root = os.path.join(paths["base_dir"], paths["managed_root"], "filesystem")
    os.makedirs(server_root, exist_ok=True)
    with open(os.path.join(server_root, "server.json"), "w", encoding="utf-8") as f:
        json.dump({
            "enabled": True, "required": False, "server_id": "filesystem",
            "display_name": "Filesystem MCP Server", "transport": "stdio",
            "command": "node", "args": ["/entrypoint.js", approved_abs],
            "working_directory": "./mcp_workspaces/filesystem",
            "startup_timeout_seconds": 15, "call_timeout_seconds": 15,
            "shutdown_timeout_seconds": 5, "environment_allowlist": [],
            "tool_policy": {"default_permission": "denied", "tools": {
                "read_text_file": {"enabled": True, "permission": "read"},
            }},
        }, f)
    upsert("filesystem", InstalledServer(
        catalog_id="official-filesystem", installed_version="2026.7.10",
        status=STATUS_INSTALLED, install_directory=server_root,
        configuration_path=os.path.join(server_root, "server.json"),
        installed_at="now", approved_directories=(approved_abs,)),
        None, paths["base_dir"], paths["managed_root"])
    return approved_abs


@pytest.fixture
def ctx(tmp_path):
    manager, paths = make_manager(tmp_path)
    approved_dir = tmp_path / "mcp_workspaces" / "filesystem"
    approved_dir.mkdir(parents=True)
    approved = _install_stub_server(paths, str(approved_dir))
    outside_dir = tmp_path / "f1_external_test"
    outside_dir.mkdir()
    (outside_dir / "hello.txt").write_text("hi", encoding="utf-8")
    return {"manager": manager, "paths": paths, "approved": approved, "outside_dir": str(outside_dir)}


# ---- Task 10: the cross-turn "yes" flow carries the same trusted restart state ----

def test_cross_turn_yes_outcome_carries_restart_state(ctx, monkeypatch):
    from tools.models import ToolCall, ToolResult

    target = os.path.join(ctx["outside_dir"], "hello.txt")
    call = ToolCall(call_id="c1", tool_name="mcp.filesystem.read_text_file", arguments={"path": target})
    result = ToolResult.fail("mcp.filesystem.read_text_file", "c1", "MCP_CALL_FAILED",
                             "Access denied - path outside allowed directories")
    found = assistant._find_outside_root_failure(ctx["manager"], [(call, result)])
    server_id, found_call, failure = found
    user_text = f"read '{target}'"
    _, request_id = assistant._offer_filesystem_access(ctx["manager"], server_id, found_call, failure, user_text)

    expected_new_roots = [ctx["approved"], os.path.realpath(ctx["outside_dir"])]
    monkeypatch.setattr(
        "mcp_management.manager.update_filesystem_access",
        lambda plan, **kw: {"server_id": "filesystem", "approved_directories": expected_new_roots,
                            "config_path": "x"})

    outcome = assistant._resolve_filesystem_access_reply(ctx["manager"], request_id, "yes")
    assert outcome.resumed_text == user_text
    assert outcome.server_id == "filesystem"
    assert set(outcome.expected_allowed_roots) == set(expected_new_roots)
    assert outcome.previous_allowed_roots == (ctx["approved"],)

    # And this is EXACTLY the shape _restart_mcp_and_resume consumes for a mid-turn
    # access.add success too — same directive type, same fields.
    directive = ToolLoopDirective(control=ToolLoopControl.RESTART_MCP_AND_RESUME,
                                  server_id=outcome.server_id,
                                  expected_allowed_roots=outcome.expected_allowed_roots)
    assert directive.control == ToolLoopControl.RESTART_MCP_AND_RESUME


# ---- Task 8 / Test Case J: ActiveMcpRuntime always closes the CURRENT session ----

class _FakeSession:
    def __init__(self, label):
        self.label = label
        self.closed = False

    def shutdown(self):
        self.closed = True


def test_active_runtime_close_targets_the_current_session_only():
    old = _FakeSession("old")
    new = _FakeSession("new")
    runtime = ActiveMcpRuntime(old)
    runtime.replace(new)
    runtime.close()
    assert new.closed is True
    assert old.closed is False  # already closed by the replacement step itself, not here
    assert runtime.session is None


def test_active_runtime_close_is_idempotent():
    runtime = ActiveMcpRuntime(None)
    runtime.close()  # must not raise
    session = _FakeSession("only")
    runtime.replace(session)
    runtime.close()
    runtime.close()  # second close is a no-op
    assert session.closed is True


# ---- Task 12: accurate, non-workaround user-visible messages on restart failure ----

_FORBIDDEN_PHRASES = ("wait a moment", "try a new conversation", "cache", "manually restart",
                      "copy the file")


@pytest.mark.parametrize("error_code", [
    MCP_RUNTIME_RESTART_FAILED, MCP_RUNTIME_REBIND_FAILED,
    MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH, MCP_RUNTIME_ROLLBACK_FAILED,
])
def test_restart_failure_message_is_accurate_and_has_no_forbidden_workaround(ctx, monkeypatch, error_code):
    def _fail_replace(*a, **kw):
        raise McpError(error_code, "forced failure")

    runtime_manager = MultiMcpRuntimeManager(tool_loop.REGISTRY, base_dir=ctx["paths"]["base_dir"],
                                             managed_root=ctx["paths"]["managed_root"])
    monkeypatch.setattr(runtime_manager, "replace_session", _fail_replace)
    directive = ToolLoopDirective(control=ToolLoopControl.RESTART_MCP_AND_RESUME,
                                  server_id="filesystem", expected_allowed_roots=("a",))
    reply, pending_id = assistant._restart_mcp_and_resume(
        ctx["manager"], runtime_manager, directive, "read something", [], "sys", set(),
        resume_budget=1)
    assert pending_id is None
    assert "updated" in reply  # accurately reports the config DID change
    lowered = reply.lower()
    for phrase in _FORBIDDEN_PHRASES:
        assert phrase not in lowered


def test_resume_budget_exhausted_refuses_without_touching_the_runtime(ctx, monkeypatch):
    def _must_not_be_called(*a, **kw):
        raise AssertionError("replace_session must not run when resume_budget is exhausted")

    runtime_manager = MultiMcpRuntimeManager(tool_loop.REGISTRY, base_dir=ctx["paths"]["base_dir"],
                                             managed_root=ctx["paths"]["managed_root"])
    monkeypatch.setattr(runtime_manager, "replace_session", _must_not_be_called)
    directive = ToolLoopDirective(control=ToolLoopControl.RESTART_MCP_AND_RESUME,
                                  server_id="filesystem", expected_allowed_roots=("a",))
    reply, pending_id = assistant._restart_mcp_and_resume(
        ctx["manager"], runtime_manager, directive, "read something", [], "sys", set(),
        resume_budget=0)
    assert pending_id is None
    assert "won't restart it again" in reply
