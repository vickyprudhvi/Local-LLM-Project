"""Phase G.3 Task 16/20 (scenarios N, P, Q, R, W) — rollback on failure, before
and after activation, with multi-server failure isolation.
"""

from dataclasses import replace

import pytest

from mcp_layer.errors import McpError
from mcp_layer.runtime_manager import RuntimeState
from mcp_management.auto_provisioning import AutoProvisioningApproval
from mcp_management.registry import get_installed
from mcp_management.provisioning_models import PendingAutoProvisioningState
from tests.auto_provisioning_helpers import build_auto_provisioning_env
from tests.mcp_multi_runtime_helpers import node_available, write_fixture_server_config
from tools.models import MCP_EXPECTED_TOOL_MISSING, MCP_PROVISIONING_RESUME_FAILED


def _catalog_with_impossible_expected_tool():
    from tests.auto_provisioning_helpers import calculator_test_catalog_entry_raw
    from mcp_management.catalog import build_catalog
    import copy

    raw = copy.deepcopy(calculator_test_catalog_entry_raw())
    raw["expected_tools"] = ["add", "echo", "multiply"]
    raw["default_tool_policy"]["tools"]["multiply"] = {"enabled": True, "permission": "read"}
    return build_catalog({"catalog_version": 1, "servers": {"calculator-test": raw}})


def test_candidate_validation_failure_leaves_no_installed_state(tmp_path):
    env = build_auto_provisioning_env(tmp_path)
    env["catalog"] = _catalog_with_impossible_expected_tool()
    env["manager"].catalog = env["catalog"]
    entry = env["catalog"].get("calculator-test")

    request = env["manager"].begin_request("add 10 and 20", "arithmetic_calculation", entry)
    plan = env["manager"].prepare_plan(request.request_id)
    approval = AutoProvisioningApproval(approved=True, plan_id=plan.plan_id, plan_hash=plan.plan_hash)
    with pytest.raises(McpError) as exc:
        env["manager"].provision_and_activate(request.request_id, env["runtime_manager"], approval=approval)
    assert exc.value.code == MCP_EXPECTED_TOOL_MISSING

    assert get_installed("calculator-test", None, env["base_dir"], "app_data/mcp_servers") is None
    assert env["manager"].pending(request.request_id).state == PendingAutoProvisioningState.FAILED
    # No orphan candidate directory survives.
    import os

    server_root = os.path.join(env["base_dir"], "app_data/mcp_servers", "calculator-test")
    candidates_root = os.path.join(server_root, "candidates")
    assert not os.path.isdir(candidates_root) or not os.listdir(candidates_root)
    # And no lingering process/tools.
    assert env["runtime_manager"].get_session("calculator-test") is None
    assert not env["reg"].has("mcp.calculator-test.add")


def test_activation_failure_after_install_restores_previous_absent_state(tmp_path, monkeypatch):
    env = build_auto_provisioning_env(tmp_path)
    entry = env["catalog"].get("calculator-test")
    request = env["manager"].begin_request("add 10 and 20", "arithmetic_calculation", entry)
    plan = env["manager"].prepare_plan(request.request_id)
    approval = AutoProvisioningApproval(approved=True, plan_id=plan.plan_id, plan_hash=plan.plan_hash)

    def _boom(*a, **kw):
        raise McpError("MCP_RUNTIME_START_FAILED", "simulated activation failure")

    monkeypatch.setattr(env["runtime_manager"], "ensure_started", _boom)

    with pytest.raises(McpError) as exc:
        env["manager"].provision_and_activate(request.request_id, env["runtime_manager"], approval=approval)
    assert exc.value.code == MCP_PROVISIONING_RESUME_FAILED

    # Rolled back to "not installed" (there was no PREVIOUS state — this was a
    # first install), not left as a partial installed record.
    assert get_installed("calculator-test", None, env["base_dir"], "app_data/mcp_servers") is None
    import os

    config_path = os.path.join(env["base_dir"], "app_data/mcp_servers", "calculator-test", "server.json")
    assert not os.path.isfile(config_path)


@pytest.mark.skipif(not node_available(), reason="node/npm not available")
def test_calculator_install_failure_does_not_touch_an_already_healthy_second_server(tmp_path):
    from tests.mcp_multi_runtime_helpers import manager_paths

    env = build_auto_provisioning_env(tmp_path)
    paths = {"base_dir": env["base_dir"], "managed_root": "app_data/mcp_servers"}
    other_root = tmp_path / "other_root"
    other_root.mkdir()
    write_fixture_server_config(paths, "document-test", [str(other_root)])
    other_session = env["runtime_manager"].ensure_started(
        "document-test", expected_allowed_roots=(str(other_root.resolve()),))
    other_proc = other_session.client._proc
    assert other_session.health.state.value == "healthy"

    env["catalog"] = _catalog_with_impossible_expected_tool()
    env["manager"].catalog = env["catalog"]
    entry = env["catalog"].get("calculator-test")
    request = env["manager"].begin_request("add 10 and 20", "arithmetic_calculation", entry)
    plan = env["manager"].prepare_plan(request.request_id)
    approval = AutoProvisioningApproval(approved=True, plan_id=plan.plan_id, plan_hash=plan.plan_hash)
    with pytest.raises(McpError):
        env["manager"].provision_and_activate(request.request_id, env["runtime_manager"], approval=approval)

    assert other_proc.poll() is None
    assert env["runtime_manager"].get_session("document-test") is other_session
    assert env["runtime_manager"].get_status("document-test").state == RuntimeState.HEALTHY
    other_session.shutdown()
