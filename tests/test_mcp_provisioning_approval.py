"""Phase F — installation approval is explicit, hash-bound, and single-purpose."""

import os

import pytest

from mcp_layer.errors import McpError
from mcp_management.approval import (
    collect_approval,
    render_plan,
    require_approval,
)
from mcp_management.models import ProvisioningApproval
from mcp_management.planner import build_plan
from tests.mcp_provisioning_helpers import make_catalog, workspace_with_file
from tools.models import (
    MCP_PROVISIONING_CONFIRMATION_MISMATCH,
    MCP_PROVISIONING_CONFIRMATION_REQUIRED,
    MCP_PROVISIONING_DECLINED,
    ToolConfirmation,
    hash_arguments,
)


@pytest.fixture
def plan(tmp_path):
    entry = make_catalog().get("official-filesystem")
    return build_plan(entry, requested_directories=[workspace_with_file(tmp_path)],
                      base_dir=str(tmp_path / "repo"))


def _approval_for(plan, approved=True):
    return ProvisioningApproval(approved=approved, plan_id=plan.plan_id,
                                plan_hash=plan.compute_hash())


def test_missing_approval_is_required(plan):
    with pytest.raises(McpError) as e:
        require_approval(plan, None)
    assert e.value.code == MCP_PROVISIONING_CONFIRMATION_REQUIRED


def test_matching_approval_passes(plan):
    assert require_approval(plan, _approval_for(plan)) is None


def test_declined_approval_is_declined(plan):
    with pytest.raises(McpError) as e:
        require_approval(plan, _approval_for(plan, approved=False))
    assert e.value.code == MCP_PROVISIONING_DECLINED


def test_wrong_hash_is_mismatch(plan):
    bad = ProvisioningApproval(True, plan.plan_id, "0" * 64)
    with pytest.raises(McpError) as e:
        require_approval(plan, bad)
    assert e.value.code == MCP_PROVISIONING_CONFIRMATION_MISMATCH


def test_wrong_plan_id_is_mismatch(plan):
    bad = ProvisioningApproval(True, "plan_somethingelse", plan.compute_hash())
    with pytest.raises(McpError) as e:
        require_approval(plan, bad)
    assert e.value.code == MCP_PROVISIONING_CONFIRMATION_MISMATCH


def test_phase_c_tool_confirmation_is_not_installation_approval(plan):
    # A write-tool confirmation must never authorize an installation.
    tool_confirmation = ToolConfirmation(approved=True, tool_name="mcp.provision.install",
                                        arguments_hash=hash_arguments({}))
    with pytest.raises(McpError) as e:
        require_approval(plan, tool_confirmation)
    assert e.value.code == MCP_PROVISIONING_CONFIRMATION_MISMATCH


def test_approval_cannot_be_reused_for_a_different_plan(tmp_path, plan):
    other_dir = tmp_path / "other_files"
    other_dir.mkdir()
    entry = make_catalog().get("official-filesystem")
    other_plan = build_plan(entry, requested_directories=[str(other_dir)],
                            base_dir=str(tmp_path / "repo"))
    approval = _approval_for(plan)
    with pytest.raises(McpError) as e:
        require_approval(other_plan, approval)
    assert e.value.code == MCP_PROVISIONING_CONFIRMATION_MISMATCH


def test_approval_invalidated_when_any_security_field_changes(tmp_path, plan):
    """Version, directory, install path, env names, and policy all rebind approval."""
    approval = _approval_for(plan)
    workspace = str(plan.requested_directories[0])
    variants = [
        build_plan(make_catalog(version="1.2.3").get("official-filesystem"),
                   requested_directories=[workspace], base_dir=str(tmp_path / "repo")),
        build_plan(make_catalog().get("official-filesystem"),
                   requested_directories=[workspace], base_dir=str(tmp_path / "repo"),
                   managed_root="app_data/other_root"),
        build_plan(make_catalog(tools={
            "default_permission": "denied",
            "tools": {"read_text_file": {"enabled": True, "permission": "read"},
                      "list_directory": {"enabled": True, "permission": "read"},
                      "list_allowed_directories": {"enabled": True, "permission": "read"},
                      "write_file": {"enabled": True, "permission": "write"},
                      "move_file": {"enabled": True, "permission": "write"}},
        }).get("official-filesystem"),
            requested_directories=[workspace], base_dir=str(tmp_path / "repo")),
    ]
    for variant in variants:
        with pytest.raises(McpError) as e:
            require_approval(variant, approval)
        assert e.value.code == MCP_PROVISIONING_CONFIRMATION_MISMATCH


def test_collect_approval_defaults_to_no(plan):
    declined = collect_approval(plan, confirmer=lambda p: False)
    assert declined.approved is False
    with pytest.raises(McpError) as e:
        require_approval(plan, declined)
    assert e.value.code == MCP_PROVISIONING_DECLINED


def test_collect_approval_binds_to_the_shown_plan(plan):
    shown = []
    approval = collect_approval(plan, confirmer=lambda p: shown.append(p) or True)
    assert shown and shown[0] is plan
    assert require_approval(plan, approval) is None


def test_rendered_plan_is_deterministic_and_complete(plan):
    text = render_plan(plan)
    assert text == render_plan(plan)
    assert plan.package_name in text
    assert plan.package_version in text
    assert str(plan.install_directory) in text
    assert str(plan.requested_directories[0]) in text
    for token in ("read tools", "write tools", "denied tools"):
        assert token in text
