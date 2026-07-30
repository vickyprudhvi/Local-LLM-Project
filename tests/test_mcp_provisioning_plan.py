"""Phase F — provisioning plans: determinism, hashing, and directory screening."""

import json
import os

import pytest

from mcp_layer.errors import McpError
from mcp_management.planner import build_plan, validate_approved_directory
from tests.mcp_provisioning_helpers import make_catalog, workspace_with_file
from tools.models import (
    MCP_DIRECTORY_NOT_APPROVED,
    MCP_PROVISIONING_PLAN_INVALID,
    ToolPermission,
)


@pytest.fixture
def entry():
    return make_catalog().get("official-filesystem")


def _plan(entry, tmp_path, directory, **kwargs):
    return build_plan(entry, requested_directories=[directory],
                      base_dir=str(tmp_path / "repo"), **kwargs)


# ---- determinism + hashing ----

def test_plan_is_deterministic(entry, tmp_path):
    workspace = workspace_with_file(tmp_path)
    first = _plan(entry, tmp_path, workspace)
    second = _plan(entry, tmp_path, workspace)
    assert first.plan_hash == second.plan_hash
    assert first.plan_id == second.plan_id
    assert first.plan_id.startswith("plan_")


def test_changed_directory_changes_hash(entry, tmp_path):
    a = workspace_with_file(tmp_path)
    other = tmp_path / "other_files"
    other.mkdir()
    assert _plan(entry, tmp_path, a).plan_hash != _plan(entry, tmp_path, str(other)).plan_hash


def test_changed_permission_changes_hash(entry, tmp_path):
    workspace = workspace_with_file(tmp_path)
    baseline = _plan(entry, tmp_path, workspace)

    escalated = make_catalog(tools={
        "default_permission": "denied",
        "tools": {
            "list_allowed_directories": {"enabled": True, "permission": "read"},
            "list_directory": {"enabled": True, "permission": "read"},
            "read_text_file": {"enabled": True, "permission": "read"},
            "get_file_info": {"enabled": True, "permission": "read"},
            "search_files": {"enabled": True, "permission": "read"},
            "write_file": {"enabled": True, "permission": "write"},
            "move_file": {"enabled": True, "permission": "write"},  # changed
            "edit_file": {"enabled": False, "permission": "denied"},
        },
    }).get("official-filesystem")
    assert _plan(escalated, tmp_path, workspace).plan_hash != baseline.plan_hash


def test_changed_version_changes_hash(entry, tmp_path):
    workspace = workspace_with_file(tmp_path)
    other = make_catalog(version="1.2.3").get("official-filesystem")
    assert _plan(other, tmp_path, workspace).plan_hash != _plan(entry, tmp_path, workspace).plan_hash


def test_hash_covers_every_security_field(entry, tmp_path):
    plan = _plan(entry, tmp_path, workspace_with_file(tmp_path))
    fields = set(plan.security_fields())
    assert fields == {
        "catalog_id", "server_id", "package_manager", "package_name", "package_version",
        "package_source", "entrypoint_relative", "install_directory", "runtime_workspace",
        "transport", "requested_directories", "requested_environment_variables",
        "tool_policy",
    }
    assert plan.plan_hash == plan.compute_hash()


def test_plan_contains_no_secret_values(entry, tmp_path, monkeypatch):
    monkeypatch.setenv("PHASE_F_SECRET", "do-not-expose")
    plan = _plan(entry, tmp_path, workspace_with_file(tmp_path))
    blob = json.dumps(plan.security_fields()) + repr(plan)
    assert "do-not-expose" not in blob


# ---- layout ----

def test_install_directory_is_under_managed_root(entry, tmp_path):
    plan = _plan(entry, tmp_path, workspace_with_file(tmp_path))
    install = str(plan.install_directory)
    assert os.path.join("app_data", "mcp_servers", "filesystem", "versions") in install
    assert install.endswith(entry.package_version)
    # Never the repo root or a virtualenv.
    assert os.path.realpath(install) != os.path.realpath(str(tmp_path / "repo"))
    assert "venv" not in install


def test_runtime_workspace_is_under_mcp_workspaces(entry, tmp_path):
    plan = _plan(entry, tmp_path, workspace_with_file(tmp_path))
    assert os.path.join("mcp_workspaces", "filesystem") in str(plan.runtime_workspace)


def test_policy_comes_from_catalog(entry, tmp_path):
    plan = _plan(entry, tmp_path, workspace_with_file(tmp_path))
    assert "write_file" in plan.write_tools()
    assert "read_text_file" in plan.read_tools()
    assert set(plan.denied_tools()) >= {"move_file", "edit_file"}
    assert plan.proposed_tool_policy.default_permission is ToolPermission.DENIED


def test_summary_lines_describe_the_action(entry, tmp_path):
    workspace = workspace_with_file(tmp_path)
    text = "\n".join(_plan(entry, tmp_path, workspace).summary_lines())
    assert entry.package_name in text and entry.package_version in text
    assert workspace in text
    assert "write_file" in text and "move_file" in text
    assert "after approval" in text


# ---- required inputs ----

def test_missing_required_directory_rejected(entry, tmp_path):
    with pytest.raises(McpError) as e:
        build_plan(entry, requested_directories=[], base_dir=str(tmp_path / "repo"))
    assert e.value.code == MCP_PROVISIONING_PLAN_INVALID


# ---- directory screening ----

def test_nonexistent_directory_rejected(tmp_path):
    with pytest.raises(McpError) as e:
        validate_approved_directory(str(tmp_path / "nope"), base_dir=str(tmp_path))
    assert e.value.code == MCP_DIRECTORY_NOT_APPROVED


def test_directory_can_be_created_when_allowed(tmp_path):
    target = tmp_path / "fresh"
    resolved = validate_approved_directory(str(target), base_dir=str(tmp_path), allow_create=True)
    assert os.path.isdir(resolved)


@pytest.mark.parametrize("bad", ["with\x00null", "with\nnewline"])
def test_illegal_characters_rejected(tmp_path, bad):
    with pytest.raises(McpError) as e:
        validate_approved_directory(bad, base_dir=str(tmp_path))
    assert e.value.code == MCP_DIRECTORY_NOT_APPROVED


def test_traversal_is_resolved_and_screened(tmp_path):
    workspace = tmp_path / "user_files"
    workspace.mkdir()
    # Resolves back inside tmp_path, so it is allowed but canonicalized.
    resolved = validate_approved_directory("user_files/../user_files", base_dir=str(tmp_path))
    assert resolved == os.path.realpath(str(workspace))


def test_filesystem_root_rejected(tmp_path):
    root = os.path.abspath(os.sep)
    with pytest.raises(McpError) as e:
        validate_approved_directory(root, base_dir=str(tmp_path))
    assert e.value.code == MCP_DIRECTORY_NOT_APPROVED


def test_home_root_requires_explicit_broad_approval():
    home = os.path.expanduser("~")
    with pytest.raises(McpError) as e:
        validate_approved_directory(home, base_dir=home)
    assert e.value.code == MCP_DIRECTORY_NOT_APPROVED
    # Explicit broad approval is honoured.
    assert validate_approved_directory(home, base_dir=home, allow_broad=True)


def test_repository_root_requires_explicit_broad_approval(tmp_path):
    base = tmp_path / "repo"
    base.mkdir()
    with pytest.raises(McpError) as e:
        validate_approved_directory(str(base), base_dir=str(base))
    assert e.value.code == MCP_DIRECTORY_NOT_APPROVED


def test_credential_directory_never_granted(tmp_path):
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    with pytest.raises(McpError) as e:
        validate_approved_directory(str(ssh), base_dir=str(tmp_path), allow_broad=True)
    assert e.value.code == MCP_DIRECTORY_NOT_APPROVED


def test_system_directory_never_granted(tmp_path):
    system_root = os.environ.get("SYSTEMROOT") or "/etc"
    if not os.path.isdir(system_root):
        pytest.skip("no system directory available to test")
    with pytest.raises(McpError) as e:
        validate_approved_directory(system_root, base_dir=str(tmp_path), allow_broad=True)
    assert e.value.code == MCP_DIRECTORY_NOT_APPROVED


def test_narrow_subdirectory_of_documents_is_allowed(tmp_path):
    # A narrow subdirectory is the preferred scope and needs no broad approval.
    narrow = tmp_path / "Documents" / "Project"
    narrow.mkdir(parents=True)
    assert validate_approved_directory(str(narrow), base_dir=str(tmp_path))
