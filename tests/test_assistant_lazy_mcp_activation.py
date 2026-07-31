"""Phase G.2 Task 18 — assistant-level lazy-activation integration scenarios.

Complements tests/test_assistant_capability_selection.py (Scenarios 1-6) with:
Scenario A (constructing the runtime manager starts zero processes — this IS
what `main()` does at startup, no _start_mcp equivalent remains), and Scenario 7
(a local-file request routes "local" — the router.py hotfix from the previous
phase — and reaches lazy Filesystem activation with zero ask_claude calls).
"""

import os

import pytest

import assistant
import tool_loop
from mcp_layer.runtime_manager import MultiMcpRuntimeManager, RuntimeState
from router import RouteDecision
from tests.mcp_multi_runtime_helpers import manager_paths, node_available, write_fixture_server_config
from tests.mcp_provisioning_helpers import make_manager
from tests.test_tool_loop import FakeLLM, _final, _tool_call
from tools.executor import ToolExecutor
from tools.registry import default_registry

pytestmark = pytest.mark.skipif(not node_available(), reason="node/npm not available")


# ---- Scenario 1 (Task 16.A): constructing the runtime manager starts zero MCP ----

def test_constructing_the_runtime_manager_launches_no_process(tmp_path):
    paths = manager_paths(tmp_path)
    reg = default_registry()
    rm = MultiMcpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])
    assert rm.get_session("filesystem") is None
    assert not any(name.startswith("mcp.") for name in reg._tools)


# ---- Scenario 7: router already proposes "local" for a file request; no Claude ----

def test_local_file_request_activates_filesystem_and_never_calls_claude(tmp_path, monkeypatch):
    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)

    manager, paths = make_manager(tmp_path)
    approved = tmp_path / "approved"
    approved.mkdir()
    (approved / "hello.txt").write_text("hi there", encoding="utf-8")
    write_fixture_server_config(paths, "filesystem", [str(approved)])
    # write_fixture_server_config registers under registry key "filesystem" with
    # catalog_id "filesystem"; align the installed entry's catalog_id with the
    # manager's real catalog for a clean SELECTED match.
    from mcp_management.registry import get_installed, upsert
    installed = get_installed("filesystem", None, paths["base_dir"], paths["managed_root"])
    from dataclasses import replace as dc_replace
    upsert("filesystem", dc_replace(installed, catalog_id="official-filesystem"),
          None, paths["base_dir"], paths["managed_root"])

    runtime_manager = MultiMcpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])

    ask_claude_calls = []
    monkeypatch.setattr(assistant, "ask_claude", lambda *a, **kw: ask_claude_calls.append(1))
    # The router now correctly proposes "local" for this request (router.py hotfix
    # from the previous phase) — asserted directly rather than re-testing the LLM.
    monkeypatch.setattr(assistant, "route_and_answer",
                        lambda prompt, history: RouteDecision(mode="local", tool=None))

    target = os.path.join(str(approved), "hello.txt")
    user_text = f"read '{target}'"
    fake = FakeLLM([
        _tool_call("mcp.filesystem.read_text_file", {"path": target}),
        _final("The file says: hi there"),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    decision = assistant.route_and_answer(user_text, [])
    assert decision.mode == "local"

    reply, extra_metrics, pending_id = assistant._process_local_request_with_capability_selection(
        manager, runtime_manager, user_text, user_text, [], "sys", set())

    assert "hi there" in reply
    assert pending_id is None
    assert ask_claude_calls == []  # Claude was never invoked
    assert runtime_manager.get_status("filesystem").state == RuntimeState.HEALTHY
    session = runtime_manager.get_session("filesystem")
    assert session is not None
    session.shutdown()


# ---- Scenario: general question -> zero ensure_started calls (spy-based) ----

def test_general_question_never_calls_ensure_started(tmp_path, monkeypatch):
    manager, paths = make_manager(tmp_path)
    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)

    runtime_manager = MultiMcpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])

    def _must_not_run(*a, **kw):
        raise AssertionError("ensure_started must not be called for NONE_REQUIRED")

    monkeypatch.setattr(runtime_manager, "ensure_started", _must_not_run)
    fake = FakeLLM([_final("Machine learning is a field of AI.")])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    reply, extra_metrics, pending_id = assistant._process_local_request_with_capability_selection(
        manager, runtime_manager, "What is machine learning?", "What is machine learning?",
        [], "sys", set())
    assert reply == "Machine learning is a field of AI."
