"""Phase G.3 Task 20 (scenarios F, T, X) — cross-turn approval resolution at the
assistant boundary: request -> plan -> yes -> install -> activate -> resume,
"show plan", a bare yes with no pending request, and resume-at-most-once.
"""

import pytest

import assistant
import tool_loop
from mcp_layer.runtime_manager import MultiMcpRuntimeManager
from mcp_management.auto_provisioning import AutoProvisioningManager
from mcp_management.manager import McpProvisioningManager
from router import RouteDecision
from tests.auto_provisioning_helpers import catalog_with_calculator_test
from tests.test_tool_loop import FakeLLM, _final, _tool_call
from tools.executor import ToolExecutor
from tools.registry import default_registry


@pytest.fixture
def env(tmp_path, monkeypatch):
    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)

    catalog = catalog_with_calculator_test()
    manager = McpProvisioningManager(catalog=catalog, base_dir=str(tmp_path),
                                     managed_root="app_data/mcp_servers")
    manager.auto_provisioning = AutoProvisioningManager(
        manager.catalog, base_dir=manager.base_dir, managed_root=manager.managed_root,
        registry_path=manager.registry_path)
    runtime_manager = MultiMcpRuntimeManager(reg, base_dir=manager.base_dir,
                                             managed_root=manager.managed_root)
    return {"manager": manager, "runtime_manager": runtime_manager, "reg": reg}


def test_bare_yes_without_a_pending_plan_does_nothing(env):
    outcome = assistant._resolve_auto_provisioning_reply(
        env["manager"].auto_provisioning, env["runtime_manager"], "autoreq_doesnotexist", "yes")
    assert outcome.matched is False


def test_show_plan_returns_the_same_stored_plan_text(env):
    manager, runtime_manager = env["manager"], env["runtime_manager"]
    user_text = "add 10 and 20 using the calculation capability"
    reply1, _m, request_id = assistant._process_local_request_with_capability_selection(
        manager, runtime_manager, user_text, user_text, [], "sys", set())

    outcome = assistant._resolve_auto_provisioning_reply(
        manager.auto_provisioning, runtime_manager, request_id, "show plan")
    assert outcome.matched
    assert outcome.resumed_text is None
    assert outcome.next_pending_id == request_id
    assert outcome.speak == reply1


def test_unrelated_reply_does_not_match_and_leaves_plan_pending(env):
    manager, runtime_manager = env["manager"], env["runtime_manager"]
    user_text = "add 10 and 20 using the calculation capability"
    _reply, _m, request_id = assistant._process_local_request_with_capability_selection(
        manager, runtime_manager, user_text, user_text, [], "sys", set())

    outcome = assistant._resolve_auto_provisioning_reply(
        manager.auto_provisioning, runtime_manager, request_id, "what's the weather tomorrow")
    assert outcome.matched is False
    # The plan is still there, untouched.
    assert manager.auto_provisioning.pending(request_id) is not None


def test_repeated_yes_after_success_does_not_reinstall(env, monkeypatch):
    manager, runtime_manager, reg = env["manager"], env["runtime_manager"], env["reg"]
    user_text = "add 10 and 20 using the calculation capability"
    _reply, _m, request_id = assistant._process_local_request_with_capability_selection(
        manager, runtime_manager, user_text, user_text, [], "sys", set())

    fake = FakeLLM([_final("ok")])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)
    monkeypatch.setattr(assistant, "route_and_answer",
                        lambda prompt, history: RouteDecision(mode="local", tool=None))

    first = assistant._resolve_auto_provisioning_reply(
        manager.auto_provisioning, runtime_manager, request_id, "yes")
    assert first.matched and first.resumed_text == user_text

    # The SAME request_id again ("repeated yes"): attempts already reached
    # MAX_PROVISIONING_ATTEMPTS on the first success, so this must be rejected
    # (MCP_PROVISIONING_ALREADY_IN_PROGRESS) rather than reinstalling.
    from mcp_management.registry import get_installed

    installed_before = get_installed("calculator-test", None, manager.base_dir, manager.managed_root)
    second = assistant._resolve_auto_provisioning_reply(
        manager.auto_provisioning, runtime_manager, request_id, "yes")
    assert second.matched is True
    assert second.resumed_text is None
    assert "already" in (second.speak or "").lower() or "progress" in (second.speak or "").lower()
    installed_after = get_installed("calculator-test", None, manager.base_dir, manager.managed_root)
    assert installed_before == installed_after
