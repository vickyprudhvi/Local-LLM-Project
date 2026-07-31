"""Phase G.3 Task 20 — AutoProvisioningPlan model behavior."""

from datetime import datetime, timedelta, timezone

from mcp_management.provisioning_models import (
    AutoProvisioningApproval,
    AutoProvisioningPlan,
    PendingAutoProvisioningRequest,
    PendingAutoProvisioningState,
    ProvisioningPlanStatus,
)


def _base_kwargs(**overrides):
    now = datetime.now(timezone.utc)
    kwargs = dict(
        plan_id="", plan_hash="", request_id="autoreq_1", original_user_text="add 1 and 2",
        catalog_id="calculator-test", server_id="calculator-test", display_name="Calculator Test MCP",
        installer_type="python_venv", exact_package="calculator-test-mcp", exact_version="1.0.0",
        lock_file_hash="abc123", executable_identity="python",
        expected_tools=("add", "echo"), tool_policy_hash="policyhash",
        environment_allowlist=(), install_network_hosts=(), runtime_network_policy="disabled",
        target_install_directory="/tmp/calculator-test/versions/1.0.0",
        candidate_config_hash="confighash",
        created_at=now.isoformat(timespec="seconds"),
        expires_at=(now + timedelta(seconds=900)).isoformat(timespec="seconds"),
    )
    kwargs.update(overrides)
    return kwargs


def test_plan_hash_is_deterministic_for_identical_inputs():
    p1 = AutoProvisioningPlan(**_base_kwargs()).with_hash()
    p2 = AutoProvisioningPlan(**_base_kwargs()).with_hash()
    assert p1.plan_hash == p2.plan_hash
    assert p1.plan_hash


def test_plan_hash_changes_when_version_changes():
    p1 = AutoProvisioningPlan(**_base_kwargs()).with_hash()
    p2 = AutoProvisioningPlan(**_base_kwargs(exact_version="1.0.1")).with_hash()
    assert p1.plan_hash != p2.plan_hash


def test_plan_hash_changes_when_tool_policy_hash_changes():
    p1 = AutoProvisioningPlan(**_base_kwargs()).with_hash()
    p2 = AutoProvisioningPlan(**_base_kwargs(tool_policy_hash="different")).with_hash()
    assert p1.plan_hash != p2.plan_hash


def test_plan_hash_changes_when_lock_file_hash_changes():
    p1 = AutoProvisioningPlan(**_base_kwargs()).with_hash()
    p2 = AutoProvisioningPlan(**_base_kwargs(lock_file_hash="different")).with_hash()
    assert p1.plan_hash != p2.plan_hash


def test_plan_hash_changes_when_request_id_changes():
    p1 = AutoProvisioningPlan(**_base_kwargs()).with_hash()
    p2 = AutoProvisioningPlan(**_base_kwargs(request_id="autoreq_2")).with_hash()
    assert p1.plan_hash != p2.plan_hash


def test_plan_is_expired_past_expiry():
    now = datetime.now(timezone.utc)
    plan = AutoProvisioningPlan(**_base_kwargs(
        expires_at=(now - timedelta(seconds=1)).isoformat(timespec="seconds"))).with_hash()
    assert plan.is_expired()


def test_plan_not_expired_before_expiry():
    plan = AutoProvisioningPlan(**_base_kwargs()).with_hash()
    assert not plan.is_expired()


def test_summary_lines_contain_no_secrets_and_expected_fields():
    plan = AutoProvisioningPlan(**_base_kwargs()).with_hash()
    text = "\n".join(plan.summary_lines())
    assert "Calculator Test MCP" in text
    assert "calculator-test-mcp==1.0.0" in text
    assert "add" in text and "echo" in text
    assert "Proceed?" in text
    for secret_marker in ("password", "token", "secret", "api_key"):
        assert secret_marker not in text.lower()


def test_pending_request_advanced_preserves_other_fields():
    request = PendingAutoProvisioningRequest(
        request_id="autoreq_1", original_user_text="add 1 and 2", capability="arithmetic_calculation",
        catalog_id="calculator-test", server_id="calculator-test")
    advanced = request.advanced(PendingAutoProvisioningState.AWAITING_APPROVAL, plan_id="autoplan_x")
    assert advanced.state == PendingAutoProvisioningState.AWAITING_APPROVAL
    assert advanced.plan_id == "autoplan_x"
    assert advanced.original_user_text == request.original_user_text
    assert advanced.attempts == 0


def test_approval_is_a_distinct_type_from_plain_dict():
    approval = AutoProvisioningApproval(approved=True, plan_id="p1", plan_hash="h1")
    assert approval.approved is True
    assert not isinstance(approval, dict)


def test_status_enum_has_all_required_values():
    values = {s.value for s in ProvisioningPlanStatus}
    assert values == {
        "prepared", "approved", "installing", "validating", "activating",
        "completed", "declined", "failed", "expired", "invalidated",
    }
