"""Phase F.1 Task 7 — the pending approval state machine on McpProvisioningManager.

Uses the manager directly (no live MCP process): `apply_filesystem_access` is
exercised with a `validate_fn` stub so these tests stay hermetic and fast, the same
pattern test_mcp_post_install_validation.py uses for Phase F's own validator.
"""

import os

import pytest

from mcp_layer.errors import McpError
from mcp_management.filesystem_access import FilesystemAccessApproval, FilesystemAccessOperation
from mcp_management.registry import STATUS_INSTALLED, InstalledServer, get_installed, upsert
from tests.mcp_provisioning_helpers import make_manager, manager_paths


def _install_stub_server(paths, approved=("workspace",)):
    approved_abs = []
    for name in approved:
        d = os.path.join(paths["base_dir"], name)
        os.makedirs(d, exist_ok=True)
        approved_abs.append(os.path.realpath(d))
    server_root = os.path.join(paths["base_dir"], paths["managed_root"], "filesystem")
    os.makedirs(server_root, exist_ok=True)
    import json
    with open(os.path.join(server_root, "server.json"), "w", encoding="utf-8") as f:
        json.dump({
            "enabled": True, "required": False, "server_id": "filesystem",
            "display_name": "Filesystem MCP Server", "transport": "stdio",
            "command": "node", "args": ["/entrypoint.js", *approved_abs],
            "working_directory": "./mcp_workspaces/filesystem",
            "startup_timeout_seconds": 15, "call_timeout_seconds": 15,
            "shutdown_timeout_seconds": 5, "environment_allowlist": [],
            "tool_policy": {"default_permission": "denied", "tools": {
                "read_text_file": {"enabled": True, "permission": "read"},
            }},
        }, f)
    upsert("filesystem", InstalledServer(
        catalog_id="official-filesystem", installed_version="2026.7.10",
        status=STATUS_INSTALLED, install_directory=server_root,
        configuration_path=os.path.join(server_root, "server.json"),
        installed_at="now", approved_directories=tuple(approved_abs)),
        None, paths["base_dir"], paths["managed_root"])
    return approved_abs


def _ok_validate(config, proposed_roots, base_dir=None, start_server_fn=None):
    return {"discovered_tool_count": 1, "protocol_version": "test"}


@pytest.fixture
def ctx(tmp_path):
    manager, paths = make_manager(tmp_path)
    approved = _install_stub_server(paths)
    new_dir = tmp_path / "chapter_pdfs"
    new_dir.mkdir()
    return {"manager": manager, "paths": paths, "approved": approved, "new_dir": str(new_dir)}


def test_no_config_change_before_approval(ctx):
    manager, paths = ctx["manager"], ctx["paths"]
    plan = manager.prepare_filesystem_access_plan("filesystem", ctx["new_dir"])
    installed = get_installed("filesystem", None, paths["base_dir"], paths["managed_root"])
    assert list(installed.approved_directories) == ctx["approved"]
    assert plan.plan_id  # a plan exists, but nothing on disk changed


def test_decline_leaves_roots_unchanged(ctx):
    manager, paths = ctx["manager"], ctx["paths"]
    plan = manager.prepare_filesystem_access_plan("filesystem", ctx["new_dir"])
    approval = FilesystemAccessApproval(approved=False, plan_id=plan.plan_id, plan_hash=plan.plan_hash)
    with pytest.raises(McpError) as exc:
        manager.apply_filesystem_access(plan, approval=approval, validate_fn=_ok_validate)
    assert exc.value.code == "MCP_FILESYSTEM_ACCESS_DECLINED"
    installed = get_installed("filesystem", None, paths["base_dir"], paths["managed_root"])
    assert list(installed.approved_directories) == ctx["approved"]


def test_approval_applies_and_extends_the_roots(ctx):
    manager, paths = ctx["manager"], ctx["paths"]
    plan = manager.prepare_filesystem_access_plan("filesystem", ctx["new_dir"])
    approval = FilesystemAccessApproval(approved=True, plan_id=plan.plan_id, plan_hash=plan.plan_hash)
    result = manager.apply_filesystem_access(plan, approval=approval, validate_fn=_ok_validate)
    assert os.path.realpath(ctx["new_dir"]) in result["approved_directories"]
    assert set(ctx["approved"]) <= set(result["approved_directories"])
    installed = get_installed("filesystem", None, paths["base_dir"], paths["managed_root"])
    assert set(installed.approved_directories) == set(result["approved_directories"])


def test_changed_proposed_root_invalidates_the_approval(ctx):
    manager = ctx["manager"]
    plan = manager.prepare_filesystem_access_plan("filesystem", ctx["new_dir"])
    tampered = plan.__class__(**{**plan.__dict__, "requested_directory": "/somewhere/else"})
    approval = FilesystemAccessApproval(approved=True, plan_id=plan.plan_id, plan_hash=plan.plan_hash)
    with pytest.raises(McpError) as exc:
        manager.apply_filesystem_access(tampered, approval=approval, validate_fn=_ok_validate)
    assert exc.value.code == "MCP_FILESYSTEM_ACCESS_CONFIRMATION_MISMATCH"


def test_changed_existing_root_set_invalidates_the_approval(ctx):
    manager, paths = ctx["manager"], ctx["paths"]
    plan = manager.prepare_filesystem_access_plan("filesystem", ctx["new_dir"])
    approval = FilesystemAccessApproval(approved=True, plan_id=plan.plan_id, plan_hash=plan.plan_hash)
    # The registry's approved roots drift after the plan was prepared.
    extra_dir = os.path.realpath(str(ctx["approved"][0]))
    installed = get_installed("filesystem", None, paths["base_dir"], paths["managed_root"])
    upsert("filesystem", installed.__class__(**{**installed.__dict__,
                                                "approved_directories": (extra_dir, "/some/drifted/root")}),
          None, paths["base_dir"], paths["managed_root"])
    with pytest.raises(McpError) as exc:
        manager.apply_filesystem_access(plan, approval=approval, validate_fn=_ok_validate)
    assert exc.value.code == "MCP_FILESYSTEM_ACCESS_PLAN_INVALID"


def test_expired_approval_fails(ctx):
    import datetime

    manager = ctx["manager"]
    plan = manager.prepare_filesystem_access_plan("filesystem", ctx["new_dir"])
    past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=1)).isoformat(
        timespec="seconds")
    expired_plan = plan.__class__(**{**plan.__dict__, "expires_at": past})
    approval = FilesystemAccessApproval(approved=True, plan_id=expired_plan.plan_id,
                                        plan_hash=expired_plan.plan_hash)
    with pytest.raises(McpError) as exc:
        manager.apply_filesystem_access(expired_plan, approval=approval, validate_fn=_ok_validate)
    assert exc.value.code == "MCP_FILESYSTEM_ACCESS_EXPIRED"


def test_approval_is_single_use(ctx):
    """Re-applying the same plan a second time must fail: the registry has already
    moved on, so the plan's `current_allowed_directories` snapshot is stale."""
    manager = ctx["manager"]
    plan = manager.prepare_filesystem_access_plan("filesystem", ctx["new_dir"])
    approval = FilesystemAccessApproval(approved=True, plan_id=plan.plan_id, plan_hash=plan.plan_hash)
    manager.apply_filesystem_access(plan, approval=approval, validate_fn=_ok_validate)
    with pytest.raises(McpError) as exc:
        manager.apply_filesystem_access(plan, approval=approval, validate_fn=_ok_validate)
    assert exc.value.code == "MCP_FILESYSTEM_ACCESS_PLAN_INVALID"


def test_a_provisioning_approval_cannot_authorize_a_filesystem_access_change(ctx):
    from mcp_management.models import ProvisioningApproval

    manager = ctx["manager"]
    plan = manager.prepare_filesystem_access_plan("filesystem", ctx["new_dir"])
    wrong_type = ProvisioningApproval(approved=True, plan_id=plan.plan_id, plan_hash=plan.plan_hash)
    with pytest.raises(McpError) as exc:
        manager.apply_filesystem_access(plan, approval=wrong_type, validate_fn=_ok_validate)
    assert exc.value.code == "MCP_FILESYSTEM_ACCESS_CONFIRMATION_MISMATCH"


def test_bare_yes_only_approves_when_a_matching_pending_request_exists(ctx):
    manager = ctx["manager"]
    assert manager.pending_filesystem_access("fsreq_does_not_exist") is None


def test_full_pending_request_lifecycle_resolves_with_bare_yes(ctx):
    manager = ctx["manager"]
    request = manager.begin_filesystem_access_request(
        original_user_text="read hello.txt from " + ctx["new_dir"],
        requested_path=os.path.join(ctx["new_dir"], "hello.txt"),
        proposed_root=ctx["new_dir"], server_id="filesystem")
    plan = manager.prepare_filesystem_access_plan(
        "filesystem", ctx["new_dir"], request_id=request.request_id,
        requested_path=request.requested_path, original_user_text=request.original_user_text)
    manager.apply_filesystem_access(
        plan, request_id=request.request_id, confirmer=lambda p: True, validate_fn=_ok_validate)
    resumed = manager.resume_filesystem_access(request.request_id)
    assert resumed == request.original_user_text


def test_decline_marks_request_declined_and_resume_returns_none(ctx):
    manager = ctx["manager"]
    request = manager.begin_filesystem_access_request(
        original_user_text="read x", requested_path="x", proposed_root=ctx["new_dir"],
        server_id="filesystem")
    manager.decline_filesystem_access(request.request_id)
    assert manager.resume_filesystem_access(request.request_id) is None


def test_removing_the_last_approved_root_is_refused(tmp_path):
    manager, paths = make_manager(tmp_path)
    _install_stub_server(paths, approved=("only_root",))
    installed = get_installed("filesystem", None, paths["base_dir"], paths["managed_root"])
    only_root = installed.approved_directories[0]
    with pytest.raises(McpError) as exc:
        manager.prepare_filesystem_access_plan(
            "filesystem", only_root, operation=FilesystemAccessOperation.REMOVE_ROOT)
    assert exc.value.code == "MCP_FILESYSTEM_LAST_ROOT_REQUIRED"


def test_loop_prevention_blocks_a_repeat_attempt_on_the_same_pending_request(ctx):
    from mcp_management.manager import MAX_PROVISIONING_ATTEMPTS

    manager = ctx["manager"]
    request = manager.begin_filesystem_access_request(
        original_user_text="read x", requested_path="x", proposed_root=ctx["new_dir"],
        server_id="filesystem")
    plan = manager.prepare_filesystem_access_plan(
        "filesystem", ctx["new_dir"], request_id=request.request_id)
    # Simulate this pending request having already used up its one allowed attempt.
    manager._filesystem_pending[request.request_id] = manager._filesystem_pending[
        request.request_id].advanced(manager._filesystem_pending[request.request_id].state,
                                     attempts=MAX_PROVISIONING_ATTEMPTS)
    approval = FilesystemAccessApproval(approved=True, plan_id=plan.plan_id, plan_hash=plan.plan_hash)
    with pytest.raises(McpError) as exc:
        manager.apply_filesystem_access(plan, approval=approval, request_id=request.request_id,
                                        validate_fn=_ok_validate)
    assert exc.value.code == "MCP_FILESYSTEM_ACCESS_LOOP_PREVENTED"


def test_removing_a_root_that_is_not_approved_is_reported(ctx):
    manager = ctx["manager"]
    with pytest.raises(McpError) as exc:
        manager.prepare_filesystem_access_plan(
            "filesystem", ctx["new_dir"], operation=FilesystemAccessOperation.REMOVE_ROOT)
    assert exc.value.code == "MCP_FILESYSTEM_ROOT_NOT_FOUND"
