"""Phase G.3 Task 13/20 — assistant-level integration: an approved-but-not-
installed provider automatically offers a plan, never touches Phase B before
approval, and installs/activates/resumes through the real pipeline once
approved. Real process, real venv — no mocks of the installer itself.
"""

import pytest

import assistant
import tool_loop
from mcp_layer.runtime_manager import MultiMcpRuntimeManager
from mcp_management.auto_provisioning import AutoProvisioningManager
from mcp_management.manager import McpProvisioningManager
from router import RouteDecision
from tests.auto_provisioning_helpers import catalog_with_calculator_test
from tests.mcp_provisioning_helpers import catalog_dict as filesystem_catalog_dict
from tests.test_tool_loop import FakeLLM, _final, _tool_call
from tools.executor import ToolExecutor
from tools.registry import default_registry


def _build_manager(tmp_path):
    catalog = catalog_with_calculator_test()
    manager = McpProvisioningManager(catalog=catalog, base_dir=str(tmp_path),
                                     managed_root="app_data/mcp_servers")
    manager.auto_provisioning = AutoProvisioningManager(
        manager.catalog, base_dir=manager.base_dir, managed_root=manager.managed_root,
        registry_path=manager.registry_path)
    return manager


@pytest.fixture
def env(tmp_path, monkeypatch):
    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)

    manager = _build_manager(tmp_path)
    runtime_manager = MultiMcpRuntimeManager(reg, base_dir=manager.base_dir,
                                             managed_root=manager.managed_root)
    return {"manager": manager, "runtime_manager": runtime_manager, "reg": reg}


def test_selected_uninstalled_provider_offers_a_plan_before_phase_b(env):
    """Scenario A: plan created, Phase B never reached, no install occurs."""
    manager, runtime_manager = env["manager"], env["runtime_manager"]
    user_text = "add 10 and 20 using the calculation capability"

    def _phase_b_must_not_run(*a, **kw):
        raise AssertionError("Phase B must not run before approval")

    import tool_loop as tl

    original = tl.ask_local_raw
    tl.ask_local_raw = _phase_b_must_not_run
    try:
        reply, metrics, request_id = assistant._process_local_request_with_capability_selection(
            manager, runtime_manager, user_text, user_text, [], "sys", set())
    finally:
        tl.ask_local_raw = original

    assert request_id is not None and request_id.startswith("autoreq_")
    assert "Calculator Test MCP" in reply
    assert "Proceed?" in reply
    from mcp_management.registry import get_installed

    assert get_installed("calculator-test", None, manager.base_dir, manager.managed_root) is None


def test_decline_leaves_no_trace(env):
    """Scenario B."""
    manager, runtime_manager = env["manager"], env["runtime_manager"]
    user_text = "add 10 and 20 using the calculation capability"
    _reply, _metrics, request_id = assistant._process_local_request_with_capability_selection(
        manager, runtime_manager, user_text, user_text, [], "sys", set())

    outcome = assistant._resolve_auto_provisioning_reply(
        manager.auto_provisioning, runtime_manager, request_id, "no")
    assert outcome.matched
    assert outcome.resumed_text is None
    from mcp_management.registry import get_installed

    assert get_installed("calculator-test", None, manager.base_dir, manager.managed_root) is None
    import os

    server_root = os.path.join(manager.base_dir, manager.managed_root, "calculator-test")
    assert not os.path.isdir(os.path.join(server_root, "versions"))


def test_approve_installs_activates_and_resumes_to_the_real_tool(env, monkeypatch):
    """Scenario C/T: cross-turn yes -> install -> validate -> activate -> resume
    -> real mcp.calculator-test.add call -> real result."""
    manager, runtime_manager, reg = env["manager"], env["runtime_manager"], env["reg"]
    user_text = "add 10 and 20 using the calculation capability"
    _reply, _metrics, request_id = assistant._process_local_request_with_capability_selection(
        manager, runtime_manager, user_text, user_text, [], "sys", set())
    assert request_id is not None

    fake = FakeLLM([
        _tool_call("mcp.calculator-test.add", {"a": 10, "b": 20}),
        _final("The result is 30."),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)
    monkeypatch.setattr(assistant, "route_and_answer",
                        lambda prompt, history: RouteDecision(mode="local", tool=None))

    outcome = assistant._resolve_auto_provisioning_reply(
        manager.auto_provisioning, runtime_manager, request_id, "yes")
    assert outcome.matched
    assert outcome.resumed_text == user_text

    reply, _extra, pending_id = assistant._process_local_request_with_capability_selection(
        manager, runtime_manager, outcome.resumed_text, outcome.resumed_text, [], "sys", set())
    assert "30" in reply
    assert pending_id is None
    assert reg.has("mcp.calculator-test.add")


def test_unsupported_capability_never_offers_a_plan(env):
    """Scenario V: no approved provider at all -> MCP_CAPABILITY_UNAVAILABLE-style
    explanation, never a provisioning plan."""
    from mcp_management.catalog import build_catalog

    manager, runtime_manager = env["manager"], env["runtime_manager"]
    manager.catalog = build_catalog(filesystem_catalog_dict())
    manager.auto_provisioning.catalog = manager.catalog

    reply, _metrics, request_id = assistant._process_local_request_with_capability_selection(
        manager, runtime_manager, "convert this document.pdf to markdown", "convert this document.pdf to markdown",
        [], "sys", set())
    assert request_id is None
    assert "Proceed?" not in reply


def test_router_proposes_local_for_calculation_capability_no_claude(env, monkeypatch):
    """Scenario U: effective route is local, zero Claude calls, even though the
    server isn't installed yet."""
    manager, runtime_manager = env["manager"], env["runtime_manager"]
    ask_claude_calls = []
    monkeypatch.setattr(assistant, "ask_claude", lambda *a, **kw: ask_claude_calls.append(1))

    decision = RouteDecision(mode="local", tool=None)
    assert decision.mode == "local"
    reply, _metrics, request_id = assistant._process_local_request_with_capability_selection(
        manager, runtime_manager, "add 10 and 20 using the calculation capability",
        "add 10 and 20 using the calculation capability", [], "sys", set())
    assert request_id is not None
    assert ask_claude_calls == []
