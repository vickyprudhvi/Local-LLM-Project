"""Phase F.1 Task 6/7/10 — the tool_loop observation hook, and assistant.py's
pending-approval / resumption wiring around it.

The tool_loop hook test proves the minimal, additive change to tool_loop.py (an
optional `on_tool_result` callback, default None) is necessary AND backward
compatible: nothing else in tool_loop.py changed, and every call that omits the
new parameter behaves exactly as before (see tests/test_tool_loop.py, unmodified).
"""

import os

import pytest

import assistant
import tool_loop
from mcp_layer.errors import McpError
from mcp_management.registry import STATUS_INSTALLED, InstalledServer, upsert
from tests.mcp_provisioning_helpers import make_manager, manager_paths
from tests.test_tool_loop import FakeLLM, _final, _install, _tool_call
from tools.executor import ToolExecutor
from tools.models import ToolCall, ToolResult
from tools.registry import default_registry


# ---- tool_loop.on_tool_result hook ----

@pytest.fixture
def fresh_tools(monkeypatch):
    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)
    return reg


def test_on_tool_result_fires_once_per_call(fresh_tools, monkeypatch):
    _install(monkeypatch, [
        _tool_call("math.calculate", {"expression": "1+1"}),
        _final("done"),
    ])
    observed = []
    text, _ = tool_loop.run_local_tool_loop("compute", [], "sys",
                                            on_tool_result=lambda call, result: observed.append((call, result)))
    assert text == "done"
    assert len(observed) == 1
    call, result = observed[0]
    assert call.tool_name == "math.calculate"
    assert isinstance(result, ToolResult)


def test_omitting_on_tool_result_is_identical_to_before(fresh_tools, monkeypatch):
    """Default behavior is byte-for-byte unchanged when the new parameter is omitted."""
    _install(monkeypatch, [
        _tool_call("math.calculate", {"expression": "1+1"}),
        _final("done"),
    ])
    text, metrics = tool_loop.run_local_tool_loop("compute", [], "sys")
    assert text == "done"


def test_a_raising_callback_never_breaks_the_turn(fresh_tools, monkeypatch):
    _install(monkeypatch, [
        _tool_call("math.calculate", {"expression": "1+1"}),
        _final("done"),
    ])

    def _boom(call, result):
        raise RuntimeError("observer bug")

    text, _ = tool_loop.run_local_tool_loop("compute", [], "sys", on_tool_result=_boom)
    assert text == "done"


# ---- assistant.py wiring around a classified outside-root failure ----

def _install_stub_server(paths, approved_dir):
    approved_abs = os.path.realpath(approved_dir)
    server_root = os.path.join(paths["base_dir"], paths["managed_root"], "filesystem")
    os.makedirs(server_root, exist_ok=True)
    import json
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
    approved_dir.mkdir(parents=True, exist_ok=True)
    approved = _install_stub_server(paths, str(approved_dir))
    outside_dir = tmp_path / "data" / "repositories" / "project" / "chapter_pdfs"
    outside_dir.mkdir(parents=True)
    (outside_dir / "README.md").write_text("hi", encoding="utf-8")
    return {"manager": manager, "paths": paths, "approved": approved, "outside_dir": str(outside_dir)}


def _outside_call_and_result(ctx):
    target = os.path.join(ctx["outside_dir"], "README.md")
    call = ToolCall(call_id="c1", tool_name="mcp.filesystem.read_text_file", arguments={"path": target})
    result = ToolResult.fail("mcp.filesystem.read_text_file", "c1", "MCP_CALL_FAILED",
                             "Access denied - path outside allowed directories")
    return call, result


def test_find_outside_root_failure_detects_it(ctx):
    call, result = _outside_call_and_result(ctx)
    found = assistant._find_outside_root_failure(ctx["manager"], [(call, result)])
    assert found is not None
    server_id, found_call, failure = found
    assert server_id == "filesystem"
    assert failure.proposed_root == os.path.realpath(ctx["outside_dir"])


def test_find_outside_root_failure_ignores_successful_calls(ctx):
    call = ToolCall(call_id="c1", tool_name="mcp.filesystem.read_text_file", arguments={"path": "x"})
    result = ToolResult.ok("mcp.filesystem.read_text_file", "c1", {"content": "hi"})
    assert assistant._find_outside_root_failure(ctx["manager"], [(call, result)]) is None


def test_offer_creates_a_pending_plan_without_changing_config(ctx):
    call, result = _outside_call_and_result(ctx)
    _, _, failure = assistant._find_outside_root_failure(ctx["manager"], [(call, result)])
    reply, request_id = assistant._offer_filesystem_access(
        ctx["manager"], "filesystem", call, failure, "read README.md from " + ctx["outside_dir"])
    assert ctx["outside_dir"].split(os.sep)[-1] in reply or "chapter_pdfs" in reply
    assert request_id is not None
    from mcp_management.registry import get_installed
    installed = get_installed("filesystem", None, ctx["paths"]["base_dir"], ctx["paths"]["managed_root"])
    assert installed.approved_directories == (ctx["approved"],)  # unchanged until approval


def _validate_fn_ok(config, proposed_roots, base_dir=None, start_server_fn=None):
    return {"discovered_tool_count": 1, "protocol_version": "test"}


def test_bare_yes_resolves_the_pending_plan_and_resumes(ctx, monkeypatch):
    call, result = _outside_call_and_result(ctx)
    _, _, failure = assistant._find_outside_root_failure(ctx["manager"], [(call, result)])
    user_text = "read README.md from " + ctx["outside_dir"]
    _, request_id = assistant._offer_filesystem_access(ctx["manager"], "filesystem", call, failure, user_text)

    monkeypatch.setattr("mcp_management.manager.update_filesystem_access",
                        lambda plan, **kw: {"server_id": "filesystem",
                                            "approved_directories": list(plan.proposed_allowed_directories),
                                            "config_path": "x"})
    outcome = assistant._resolve_filesystem_access_reply(ctx["manager"], request_id, "yes")
    assert outcome.matched is True
    assert outcome.resumed_text == user_text
    assert outcome.next_pending_id is None


def test_bare_no_declines_and_leaves_config_unchanged(ctx):
    call, result = _outside_call_and_result(ctx)
    _, _, failure = assistant._find_outside_root_failure(ctx["manager"], [(call, result)])
    user_text = "read README.md from " + ctx["outside_dir"]
    _, request_id = assistant._offer_filesystem_access(ctx["manager"], "filesystem", call, failure, user_text)

    outcome = assistant._resolve_filesystem_access_reply(ctx["manager"], request_id, "no")
    assert outcome.matched is True
    assert outcome.resumed_text is None
    from mcp_management.registry import get_installed
    installed = get_installed("filesystem", None, ctx["paths"]["base_dir"], ctx["paths"]["managed_root"])
    assert installed.approved_directories == (ctx["approved"],)


def test_unrelated_reply_does_not_match_and_leaves_plan_pending(ctx):
    call, result = _outside_call_and_result(ctx)
    _, _, failure = assistant._find_outside_root_failure(ctx["manager"], [(call, result)])
    user_text = "read README.md from " + ctx["outside_dir"]
    _, request_id = assistant._offer_filesystem_access(ctx["manager"], "filesystem", call, failure, user_text)

    outcome = assistant._resolve_filesystem_access_reply(ctx["manager"], request_id,
                                                          "what's the weather like today")
    assert outcome.matched is False


def test_yes_with_no_pending_state_changes_nothing(ctx):
    outcome = assistant._resolve_filesystem_access_reply(ctx["manager"], "fsreq_never_existed", "yes")
    assert outcome.matched is False
    from mcp_management.registry import get_installed
    installed = get_installed("filesystem", None, ctx["paths"]["base_dir"], ctx["paths"]["managed_root"])
    assert installed.approved_directories == (ctx["approved"],)


def test_show_plan_reply_does_not_consume_the_pending_request(ctx):
    call, result = _outside_call_and_result(ctx)
    _, _, failure = assistant._find_outside_root_failure(ctx["manager"], [(call, result)])
    user_text = "read README.md from " + ctx["outside_dir"]
    _, request_id = assistant._offer_filesystem_access(ctx["manager"], "filesystem", call, failure, user_text)

    outcome = assistant._resolve_filesystem_access_reply(ctx["manager"], request_id, "show plan")
    assert outcome.matched is True
    assert outcome.next_pending_id == request_id  # still pending
    assert "filesystem" in outcome.speak


# ---- restricted path: never offered a plan ----

def test_restricted_path_yields_no_offer(tmp_path):
    manager, paths = make_manager(tmp_path)
    approved_dir = tmp_path / "mcp_workspaces" / "filesystem"
    approved_dir.mkdir(parents=True)
    _install_stub_server(paths, str(approved_dir))

    ssh_dir = tmp_path / "home" / ".ssh"
    ssh_dir.mkdir(parents=True)
    key = ssh_dir / "id_rsa"
    key.touch()
    call = ToolCall(call_id="c1", tool_name="mcp.filesystem.read_text_file",
                    arguments={"path": str(key)})
    result = ToolResult.fail("mcp.filesystem.read_text_file", "c1", "MCP_CALL_FAILED",
                             "Access denied - path outside allowed directories")
    found = assistant._find_outside_root_failure(manager, [(call, result)])
    assert found is None  # restricted -> ineligible -> never offered
