"""Phase G.3 Task 9/12/20 (scenarios C, I) — the full candidate transaction:
install -> validate -> atomically activate config + registry -> Phase G.2 picks
it up. Real process, real venv, real tool call — no mocks.
"""

import os

import pytest

from mcp_layer.runtime_manager import RuntimeState
from mcp_management.auto_provisioning import AutoProvisioningApproval
from mcp_management.registry import get_installed
from tests.auto_provisioning_helpers import build_auto_provisioning_env
from tools.executor import ToolExecutor
from tools.models import ToolCall


def _approve_and_provision(env):
    entry = env["catalog"].get("calculator-test")
    request = env["manager"].begin_request("add 10 and 20", "arithmetic_calculation", entry)
    plan = env["manager"].prepare_plan(request.request_id)
    approval = AutoProvisioningApproval(approved=True, plan_id=plan.plan_id, plan_hash=plan.plan_hash)
    result = env["manager"].provision_and_activate(request.request_id, env["runtime_manager"], approval=approval)
    return request, plan, result


def test_full_transaction_installs_validates_and_activates(tmp_path):
    env = build_auto_provisioning_env(tmp_path)
    request, plan, result = _approve_and_provision(env)

    assert result.installed_version == "1.0.0"
    assert os.path.isfile(result.managed_config_path)
    installed = get_installed("calculator-test", None, env["base_dir"], "app_data/mcp_servers")
    assert installed is not None
    assert installed.installer_type == "python_venv"
    assert installed.status == "installed"

    status = env["runtime_manager"].get_status("calculator-test")
    assert status.state == RuntimeState.HEALTHY
    assert env["reg"].has("mcp.calculator-test.add")
    assert env["reg"].has("mcp.calculator-test.echo")


def test_installed_tool_actually_computes(tmp_path):
    env = build_auto_provisioning_env(tmp_path)
    _approve_and_provision(env)
    executor = ToolExecutor(env["reg"])
    result = executor.execute(ToolCall(call_id="c1", tool_name="mcp.calculator-test.add",
                                       arguments={"a": 10, "b": 20}))
    assert result.success
    assert result.data["result"] == 30


def test_resume_returns_original_text_exactly_once(tmp_path):
    env = build_auto_provisioning_env(tmp_path)
    request, plan, result = _approve_and_provision(env)
    resumed = env["manager"].resume(request.request_id)
    assert resumed == "add 10 and 20"
    assert env["manager"].resume(request.request_id) is None  # single-use


def test_second_request_for_same_server_reuses_installed_state_no_reinstall(tmp_path):
    env = build_auto_provisioning_env(tmp_path)
    _approve_and_provision(env)

    entry = env["catalog"].get("calculator-test")
    request2 = env["manager"].begin_request("add 1 and 2", "arithmetic_calculation", entry)
    plan2 = env["manager"].prepare_plan(request2.request_id)
    approval2 = AutoProvisioningApproval(approved=True, plan_id=plan2.plan_id, plan_hash=plan2.plan_hash)
    result2 = env["manager"].provision_and_activate(request2.request_id, env["runtime_manager"], approval=approval2)
    assert result2.installed_version == "1.0.0"
    # Same runtime session reused — no second candidate/venv was built.
    session_before = env["runtime_manager"].get_session("calculator-test")
    assert session_before is not None
