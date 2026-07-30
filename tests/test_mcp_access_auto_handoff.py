"""Phase F.1 hotfix Task 2/13(A) — automatic first-request access-plan handoff.

A direct outside-root read (`read 'C:\\...\\f1_external_test\\hello.txt'`) must
produce the access plan immediately, WITHOUT the user ever needing to rephrase as
"use the filesystem MCP server...". Proven by asserting the local LLM is called
EXACTLY ONCE — the tool loop halts before it can be asked for a second, generic
fallback answer.
"""

import os

import pytest

import assistant
import confirmation
import tool_loop
from tests.mcp_provisioning_helpers import make_manager
from tests.test_tool_loop import FakeLLM, _tool_call
from mcp_layer.runtime_manager import ActiveMcpRuntime
from mcp_management.registry import STATUS_INSTALLED, InstalledServer, upsert
from tools.executor import ToolExecutor
from tools.registry import default_registry


def _install_stub_server(paths, approved_dir):
    import json

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
def ctx(tmp_path, monkeypatch):
    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)

    manager, paths = make_manager(tmp_path)
    approved_dir = tmp_path / "mcp_workspaces" / "filesystem"
    approved_dir.mkdir(parents=True, exist_ok=True)
    approved = _install_stub_server(paths, str(approved_dir))
    outside_dir = tmp_path / "f1_external_test"
    outside_dir.mkdir()
    (outside_dir / "hello.txt").write_text("hi", encoding="utf-8")

    # A fake McpTool standing in for the real remote one, registered under the
    # same name the classifier expects — this test is about tool-LOOP control
    # flow, not the live MCP transport (already proven for real in
    # tests/test_mcp_runtime_replacement.py).
    from mcp_layer.errors import McpError
    from tools.base import BaseTool, ToolFailure
    from tools.models import MCP_CALL_FAILED, ToolPermission

    class _StubReadTool(BaseTool):
        name = "mcp.filesystem.read_text_file"
        description = "stub"
        input_schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        permission = ToolPermission.READ
        llm_callable = True

        def execute(self, arguments):
            path = os.path.realpath(arguments["path"])
            if not (path == approved or path.startswith(approved + os.sep)):
                raise ToolFailure(MCP_CALL_FAILED, "Access denied - path outside allowed directories")
            with open(path, encoding="utf-8") as f:
                return {"content": f.read()}

    reg.register(_StubReadTool())

    return {"manager": manager, "paths": paths, "approved": approved, "outside_dir": str(outside_dir),
           "runtime": ActiveMcpRuntime(None)}


def test_outside_root_read_produces_plan_without_a_second_llm_call(ctx, monkeypatch):
    target = os.path.join(ctx["outside_dir"], "hello.txt")
    user_text = f"read '{target}'"

    fake = FakeLLM([
        _tool_call("mcp.filesystem.read_text_file", {"path": target}),
        {"message": {"role": "assistant", "content": "I can't read that; try copying the file instead."}},
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    reply, extra_metrics, pending_id = assistant._run_local_turn(
        ctx["manager"], ctx["runtime"], user_text, user_text, [], "sys", set())

    assert len(fake.calls) == 1  # the model was NEVER asked for the generic fallback
    assert pending_id is not None
    assert "hello.txt" in reply or os.path.basename(ctx["outside_dir"]) in reply
    assert "copying the file" not in reply  # the generic fallback text never appears

    from mcp_management.registry import get_installed
    installed = get_installed("filesystem", None, ctx["paths"]["base_dir"], ctx["paths"]["managed_root"])
    assert installed.approved_directories == (ctx["approved"],)  # nothing changed yet


def test_loop_prevented_on_second_attempt_for_the_same_request(ctx, monkeypatch):
    target = os.path.join(ctx["outside_dir"], "hello.txt")
    user_text = f"read '{target}'"
    fake = FakeLLM([_tool_call("mcp.filesystem.read_text_file", {"path": target})])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    already_attempted = {user_text}
    reply, extra_metrics, pending_id = assistant._run_local_turn(
        ctx["manager"], ctx["runtime"], user_text, user_text, [], "sys", already_attempted)

    assert pending_id is None
    assert "already tried" in reply
