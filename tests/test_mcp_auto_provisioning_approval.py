"""Phase G.3 Task 20 — auto-provisioning approval enforcement (Task 4).

A DELIBERATELY differently-named file from the pre-existing Phase F
`tests/test_mcp_provisioning_approval.py`, which covers the manual/LLM-tool
`ProvisioningApproval` flow and is untouched by this phase.
"""

import pytest

from mcp_layer.errors import McpError
from mcp_management.auto_provisioning import build_auto_plan, require_auto_provisioning_approval
from mcp_management.provisioning_models import AutoProvisioningApproval
from tests.auto_provisioning_helpers import calculator_test_catalog_entry
from tools.models import (
    MCP_PROVISIONING_CONFIRMATION_MISMATCH,
    MCP_PROVISIONING_CONFIRMATION_REQUIRED,
    MCP_PROVISIONING_DECLINED,
    MCP_PROVISIONING_PLAN_EXPIRED,
)


def _plan(base_dir="/tmp/g3", ttl_seconds=900):
    entry = calculator_test_catalog_entry()
    return build_auto_plan(entry, "autoreq_1", "add 1 and 2", base_dir=base_dir, ttl_seconds=ttl_seconds)


def test_missing_approval_required():
    plan = _plan()
    with pytest.raises(McpError) as exc:
        require_auto_provisioning_approval(plan, None)
    assert exc.value.code == MCP_PROVISIONING_CONFIRMATION_REQUIRED


def test_wrong_type_approval_rejected():
    plan = _plan()
    with pytest.raises(McpError) as exc:
        require_auto_provisioning_approval(plan, {"approved": True})
    assert exc.value.code == MCP_PROVISIONING_CONFIRMATION_MISMATCH


def test_declined_approval_rejected():
    plan = _plan()
    approval = AutoProvisioningApproval(approved=False, plan_id=plan.plan_id, plan_hash=plan.plan_hash)
    with pytest.raises(McpError) as exc:
        require_auto_provisioning_approval(plan, approval)
    assert exc.value.code == MCP_PROVISIONING_DECLINED


def test_mismatched_plan_hash_rejected():
    plan = _plan()
    approval = AutoProvisioningApproval(approved=True, plan_id=plan.plan_id, plan_hash="wrong")
    with pytest.raises(McpError) as exc:
        require_auto_provisioning_approval(plan, approval)
    assert exc.value.code == MCP_PROVISIONING_CONFIRMATION_MISMATCH


def test_mismatched_plan_id_rejected():
    plan = _plan()
    approval = AutoProvisioningApproval(approved=True, plan_id="different", plan_hash=plan.plan_hash)
    with pytest.raises(McpError) as exc:
        require_auto_provisioning_approval(plan, approval)
    assert exc.value.code == MCP_PROVISIONING_CONFIRMATION_MISMATCH


def test_expired_plan_rejected_even_with_matching_hash():
    plan = _plan(ttl_seconds=-1)
    assert plan.is_expired()
    approval = AutoProvisioningApproval(approved=True, plan_id=plan.plan_id, plan_hash=plan.plan_hash)
    with pytest.raises(McpError) as exc:
        require_auto_provisioning_approval(plan, approval)
    assert exc.value.code == MCP_PROVISIONING_PLAN_EXPIRED


def test_valid_approval_passes():
    plan = _plan()
    approval = AutoProvisioningApproval(approved=True, plan_id=plan.plan_id, plan_hash=plan.plan_hash)
    require_auto_provisioning_approval(plan, approval)  # does not raise


def test_filesystem_access_approval_type_not_accepted():
    """A DIFFERENT approval type (Phase F.1's own) must never satisfy this check."""
    from mcp_management.filesystem_access import FilesystemAccessApproval

    plan = _plan()
    approval = FilesystemAccessApproval(approved=True, plan_id=plan.plan_id, plan_hash=plan.plan_hash)
    with pytest.raises(McpError) as exc:
        require_auto_provisioning_approval(plan, approval)
    assert exc.value.code == MCP_PROVISIONING_CONFIRMATION_MISMATCH
