"""Assistant tweak — a directory-access offer must trigger even when the model
checks `list_allowed_directories` first (a normal, SUCCESSFUL call) instead of
attempting the real read/list operation and letting THAT fail.

Reproduces the reported live scenario: "list files in <not-yet-approved
directory>" — the model calls list_allowed_directories, sees the directory
isn't covered, and (without this fix) just tells the user it can't help
instead of the automatic Phase F.1 access-offer ever triggering.
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
            "list_directory": {"enabled": True, "permission": "read"},
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
    monkeypatch.setattr(confirmation, "confirm_action", lambda summary: True)

    manager, paths = make_manager(tmp_path)
    register_filesystem_access_tools(reg, manager)

    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()
    (root_b / "notes.txt").write_text("hello from root_b", encoding="utf-8")

    _write_managed_config(paths, [str(root_a)])
    from mcp_layer.config import load_config
    config_path = os.path.join(paths["base_dir"], paths["managed_root"], "filesystem", "server.json")
    session = bootstrap_from_config(reg, config=load_config(config_path), base_dir=paths["base_dir"])
    assert session.health.state.value == "healthy"

    runtime = MultiMcpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])
    slot = runtime._slot("filesystem")
    slot.session = session
    slot.state = RuntimeState.HEALTHY
    runtime._runtimes["filesystem"].replace(session)

    return {"manager": manager, "paths": paths, "reg": reg, "runtime": runtime,
           "root_a": str(root_a), "root_b": str(root_b), "session": session}


def test_checking_allowed_directories_first_still_offers_access(env, monkeypatch):
    manager, runtime = env["manager"], env["runtime"]
    root_b = env["root_b"]
    user_text = f"list files in {root_b}"

    fake = FakeLLM([
        _tool_call("mcp.filesystem.list_allowed_directories", {}),
        _final("this should never be reached — the loop must halt first"),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    reply, extra_metrics, pending_id = assistant._run_local_turn(
        manager, runtime, user_text, user_text, [], "sys", set(), resume_budget=1)

    # Halted after the FIRST call — the model never got a second turn to
    # write "I can't access that" from the list_allowed_directories result.
    assert len(fake.calls) == 1
    assert pending_id is not None and pending_id.startswith("fsreq_")
    assert os.path.realpath(root_b) in reply
    assert "yes" in reply.lower()

    env["session"].shutdown()


def test_approving_the_offer_resumes_and_lists_the_real_directory(env, monkeypatch):
    manager, runtime = env["manager"], env["runtime"]
    root_b = env["root_b"]
    old_proc = env["session"].client._proc
    user_text = f"list files in {root_b}"

    fake = FakeLLM([
        _tool_call("mcp.filesystem.list_allowed_directories", {}),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    _reply, _metrics, pending_id = assistant._run_local_turn(
        manager, runtime, user_text, user_text, [], "sys", set(), resume_budget=1)
    assert pending_id is not None

    # Approve, then resume with a fresh script that actually lists root_b.
    resume_fake = FakeLLM([
        _tool_call("mcp.filesystem.list_directory", {"path": root_b}),
        _final("root_b contains: notes.txt"),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", resume_fake)
    monkeypatch.setattr(assistant, "route_and_answer",
                        lambda prompt, history: RouteDecision(mode="local", tool=None))

    outcome = assistant._resolve_filesystem_access_reply(manager, pending_id, "yes")
    assert outcome.matched and outcome.resumed_text == user_text

    directive = tool_loop.ToolLoopDirective(
        control=tool_loop.ToolLoopControl.RESTART_MCP_AND_RESUME,
        server_id=outcome.server_id, expected_allowed_roots=outcome.expected_allowed_roots)
    reply2, pending_id2 = assistant._restart_mcp_and_resume(
        manager, runtime, directive, outcome.resumed_text, [], "sys", set(),
        resume_budget=1, previous_allowed_roots=outcome.previous_allowed_roots)

    assert "notes.txt" in reply2
    assert pending_id2 is None
    assert old_proc.poll() is not None  # old process replaced

    new_session = runtime.get_session("filesystem")
    new_session.shutdown()


def test_no_offer_when_requested_directory_already_covered(env, monkeypatch):
    """A list_allowed_directories check for a directory that IS already
    approved must not spuriously offer anything."""
    manager, runtime = env["manager"], env["runtime"]
    root_a = env["root_a"]
    user_text = f"list files in {root_a}"

    fake = FakeLLM([
        _tool_call("mcp.filesystem.list_allowed_directories", {}),
        _final(f"{root_a} contains no files yet."),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    reply, _metrics, pending_id = assistant._run_local_turn(
        manager, runtime, user_text, user_text, [], "sys", set(), resume_budget=1)

    assert pending_id is None
    assert len(fake.calls) == 2  # both the tool call and the final answer ran

    env["session"].shutdown()


def test_no_offer_when_user_text_names_no_directory(env, monkeypatch):
    """A bare 'what directories can you access?' must not trigger any offer —
    extract_directory_candidate finds nothing to propose."""
    manager, runtime = env["manager"], env["runtime"]
    user_text = "what directories can you access?"

    fake = FakeLLM([
        _tool_call("mcp.filesystem.list_allowed_directories", {}),
        _final("I can access root_a."),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    reply, _metrics, pending_id = assistant._run_local_turn(
        manager, runtime, user_text, user_text, [], "sys", set(), resume_budget=1)

    assert pending_id is None
    assert len(fake.calls) == 2

    env["session"].shutdown()
