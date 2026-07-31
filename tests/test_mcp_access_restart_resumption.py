"""Phase F.1 hotfix Task 3/5/6/8/9 and 13(B/G) — end-to-end mid-turn recovery.

The exact scenario from the bug report: the local model calls
`mcp.filesystem.access.add` itself (write-permission, Phase C confirmation) and it
succeeds. The tool loop must halt IMMEDIATELY — no stale `read_text_file` call
through the old client — the runtime must be replaced with a REAL new process
(the Node fixture filesystem server), and the ORIGINAL request must then resume
through the real router / shortlist / executor / McpTool pipeline and return the
REAL file content, all inside one call chain, with no manual restart.
"""

import json
import os

import pytest

import assistant
import confirmation
import tool_loop
from mcp_layer.external import bootstrap_from_config
from mcp_layer.runtime_manager import MultiMcpRuntimeManager, RuntimeState
from mcp_management.registry import STATUS_INSTALLED, InstalledServer, upsert
from mcp_management.filesystem_access_tools import register_filesystem_access_tools
from router import RouteDecision
from tests.mcp_provisioning_helpers import FIXTURE_SERVER, make_manager, node_available
from tests.test_tool_loop import FakeLLM, _final, _tool_call
from tools.executor import ToolExecutor
from tools.registry import default_registry

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
def env(tmp_path, monkeypatch):
    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)
    # Write-tool confirmation ("Proceed? [yes/No]") auto-approved — tests never
    # block on input() (confirmation.py's own documented contract).
    monkeypatch.setattr(confirmation, "confirm_action", lambda summary: True)

    manager, paths = make_manager(tmp_path)
    register_filesystem_access_tools(reg, manager)

    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()
    (root_b / "hello.txt").write_text("Phase F.1 runtime restart passed.", encoding="utf-8")

    _write_managed_config(paths, [str(root_a)])
    from mcp_layer.config import load_config
    config_path = os.path.join(paths["base_dir"], paths["managed_root"], "filesystem", "server.json")
    old_session = bootstrap_from_config(reg, config=load_config(config_path), base_dir=paths["base_dir"])
    assert old_session.health.state.value == "healthy"

    runtime = MultiMcpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])
    # Seed the "filesystem" slot as already HEALTHY with the pre-bootstrapped
    # session — simulating a server that was already active before this turn,
    # exactly like Task 17's "Filesystem starts lazily if inactive" companion
    # scenario (already-active case).
    slot = runtime._slot("filesystem")
    slot.session = old_session
    slot.state = RuntimeState.HEALTHY
    runtime._runtimes["filesystem"].replace(old_session)

    return {"manager": manager, "paths": paths, "reg": reg, "runtime": runtime,
           "root_a": str(root_a), "root_b": str(root_b), "old_session": old_session}


def test_access_add_halts_then_restarts_and_resumes_with_real_content(env, monkeypatch):
    manager, paths, reg, runtime = env["manager"], env["paths"], env["reg"], env["runtime"]
    root_a, root_b = env["root_a"], env["root_b"]
    old_proc = env["old_session"].client._proc

    plan = manager.prepare_filesystem_access_plan("filesystem", root_b)

    target = os.path.join(root_b, "hello.txt")
    user_text = f"read '{target}'"

    fake = FakeLLM([
        _tool_call("mcp.filesystem.access.add",
                  {"server_id": "filesystem", "plan_id": plan.plan_id, "plan_hash": plan.plan_hash}),
        _tool_call("mcp.filesystem.read_text_file", {"path": target}),
        _final("The file contains: Phase F.1 runtime restart passed."),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)
    monkeypatch.setattr(assistant, "route_and_answer",
                        lambda prompt, history: RouteDecision(mode="local", tool=None))

    reply, extra_metrics, pending_id = assistant._run_local_turn(
        manager, runtime, user_text, user_text, [], "sys", set(), resume_budget=1)

    # Exactly 3 LLM calls: access.add, then (after a REAL restart) read_text_file,
    # then the final answer. No stale call through the old client in between, and
    # no extra "generic answer" call anywhere.
    assert len(fake.calls) == 3
    assert "Phase F.1 runtime restart passed" in reply
    assert pending_id is None

    # Old process actually terminated; runtime now points at a NEW session.
    new_session = runtime.get_session("filesystem")
    assert old_proc.poll() is not None
    assert new_session is not env["old_session"]
    assert new_session.client is not None

    # The read tool the model used in step 2 is bound to the NEW client.
    live_tool = reg.get("mcp.filesystem.read_text_file")
    assert live_tool.session_owner == new_session.session_id

    # Built-ins remain registered and callable.
    for name in ("mcp.filesystem.access.list", "mcp.filesystem.access.plan",
                "mcp.filesystem.access.add", "mcp.filesystem.access.remove"):
        assert reg.has(name)

    new_session.shutdown()


def test_no_stale_read_reaches_the_old_client_after_access_add(env, monkeypatch):
    """Even if the model's SECOND scripted response were a stale-session call, the
    loop must never reach it in the same batch as the access.add success — proven
    by never consuming the second FakeLLM response for anything except the resumed
    (post-restart) round."""
    manager, paths, reg, runtime = env["manager"], env["paths"], env["reg"], env["runtime"]
    root_b = env["root_b"]
    plan = manager.prepare_filesystem_access_plan("filesystem", root_b)

    fake = FakeLLM([
        _tool_call("mcp.filesystem.access.add",
                  {"server_id": "filesystem", "plan_id": plan.plan_id, "plan_hash": plan.plan_hash}),
        _final("done"),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)
    monkeypatch.setattr(assistant, "route_and_answer",
                        lambda prompt, history: RouteDecision(mode="local", tool=None))

    text, metrics = tool_loop.run_local_tool_loop(
        "grant access to " + root_b, [], "sys",
        on_tool_result=lambda call, result: assistant._classify_access_apply_success(
            manager, call, result)[0])
    assert text is None  # the loop itself halted; it never reached the "_final" response
    assert len(fake.calls) == 1
