"""Phase F — installation is transactional: validated, promoted, recorded, or rolled back.

These tests launch the REAL Node fixture server, so the install path is exercised
end to end (start, initialize, tools/list, policy, smoke test, shutdown) with no
network access.
"""

import json
import os

import pytest

from mcp_layer.errors import McpError
from mcp_management import audit
from mcp_management.approval import collect_approval
from mcp_management.installer import install
from mcp_management.models import ProvisioningApproval
from mcp_management.planner import build_plan
from mcp_management.registry import get_installed, load_registry
from tests.mcp_provisioning_helpers import (
    FakeNpm,
    make_catalog,
    node_available,
    workspace_with_file,
)
from tools.models import (
    MCP_INSTALLATION_FAILED,
    MCP_POST_INSTALL_VALIDATION_FAILED,
    MCP_PROVISIONING_CONFIRMATION_REQUIRED,
)

pytestmark = pytest.mark.skipif(not node_available(), reason="node/npm not available")


@pytest.fixture
def ctx(tmp_path):
    catalog = make_catalog()
    entry = catalog.get("official-filesystem")
    base_dir = str(tmp_path / "repo")
    os.makedirs(os.path.join(base_dir, "mcp_workspaces"), exist_ok=True)
    workspace = workspace_with_file(tmp_path)
    plan = build_plan(entry, requested_directories=[workspace], base_dir=base_dir)
    approval = ProvisioningApproval(True, plan.plan_id, plan.compute_hash())
    return {
        "entry": entry, "plan": plan, "approval": approval, "base_dir": base_dir,
        "workspace": workspace, "managed_root": "app_data/mcp_servers",
    }


def _install(ctx, npm=None, **kwargs):
    return install(ctx["plan"], ctx["entry"], ctx["approval"], base_dir=ctx["base_dir"],
                   managed_root=ctx["managed_root"], npm_runner=npm or FakeNpm(),
                   **kwargs)


# ---- happy path ----

def test_successful_install_validates_promotes_and_registers(ctx):
    result = _install(ctx)
    assert result["version"] == ctx["plan"].package_version
    assert os.path.isdir(result["install_directory"])
    # The version directory is the promoted final location, not a staging dir.
    assert result["install_directory"] == str(ctx["plan"].install_directory)
    assert "staging" not in result["install_directory"]

    validation = result["validation"]
    assert validation["ok"] is True
    assert validation["registered_tool_count"] >= 3
    # The fixture advertises an undocumented tool; local policy must deny it.
    assert validation["denied_tool_count"] >= 1

    installed = get_installed("filesystem", None, ctx["base_dir"], ctx["managed_root"])
    assert installed is not None
    assert installed.installed_version == ctx["plan"].package_version
    assert installed.status == "installed"
    assert installed.last_validation_result == "healthy"


def test_generated_artifacts_are_written(ctx):
    result = _install(ctx)
    server_root = os.path.dirname(os.path.dirname(result["install_directory"]))
    for name in ("server.json", "permissions.json", "install-record.json", "current.json"):
        assert os.path.isfile(os.path.join(server_root, name)), name

    record = audit.read_install_record(server_root)
    assert record["package_version"] == ctx["plan"].package_version
    assert record["plan_hash"] == ctx["plan"].compute_hash()
    assert record["package_lock_sha256"]
    assert record["approved_directories"] == [str(ctx["plan"].requested_directories[0])]
    assert record["validation"]["ok"] is True


def test_install_record_has_no_secrets(ctx, monkeypatch):
    monkeypatch.setenv("PHASE_F_SECRET", "do-not-expose")
    result = _install(ctx)
    server_root = os.path.dirname(os.path.dirname(result["install_directory"]))
    blob = json.dumps(audit.read_install_record(server_root))
    assert "do-not-expose" not in blob
    assert "PHASE_F_SECRET" not in blob


def test_no_approval_installs_nothing(ctx):
    with pytest.raises(McpError) as e:
        install(ctx["plan"], ctx["entry"], None, base_dir=ctx["base_dir"],
                managed_root=ctx["managed_root"], npm_runner=FakeNpm())
    assert e.value.code == MCP_PROVISIONING_CONFIRMATION_REQUIRED
    assert not os.path.isdir(str(ctx["plan"].install_directory))
    assert load_registry(None, ctx["base_dir"], ctx["managed_root"]) == {}


def test_declined_approval_leaves_no_files(ctx):
    declined = collect_approval(ctx["plan"], confirmer=lambda p: False)
    with pytest.raises(McpError):
        install(ctx["plan"], ctx["entry"], declined, base_dir=ctx["base_dir"],
                managed_root=ctx["managed_root"], npm_runner=FakeNpm())
    server_root = os.path.join(ctx["base_dir"], "app_data", "mcp_servers", "filesystem")
    assert not os.path.isdir(str(ctx["plan"].install_directory))
    assert not os.path.isfile(os.path.join(server_root, "server.json"))
    assert load_registry(None, ctx["base_dir"], ctx["managed_root"]) == {}


# ---- idempotency ----

def test_same_version_reinstall_is_idempotent(ctx):
    first = _install(ctx)
    npm = FakeNpm()
    second = _install(ctx, npm=npm)
    assert second["reused_existing_installation"] is True
    assert npm.calls == []  # npm was not run again
    assert second["install_directory"] == first["install_directory"]
    registry = load_registry(None, ctx["base_dir"], ctx["managed_root"])
    assert list(registry) == ["filesystem"]


def test_force_reinstall_runs_npm_again(ctx):
    _install(ctx)
    npm = FakeNpm()
    result = _install(ctx, npm=npm, force_reinstall=True)
    assert npm.calls, "force_reinstall must run the installer again"
    assert result["reused_existing_installation"] is False


# ---- rollback ----

def test_npm_failure_rolls_back_completely(ctx):
    with pytest.raises(McpError) as e:
        _install(ctx, npm=FakeNpm(fail=True))
    assert e.value.code == MCP_INSTALLATION_FAILED
    assert not os.path.isdir(str(ctx["plan"].install_directory))
    assert load_registry(None, ctx["base_dir"], ctx["managed_root"]) == {}
    server_root = os.path.join(ctx["base_dir"], "app_data", "mcp_servers", "filesystem")
    assert not any(n.startswith(".staging") for n in _safe_listdir(server_root))


def test_validation_failure_rolls_back_and_leaves_no_registry_entry(ctx):
    def failing_validate(*args, **kwargs):
        raise McpError(MCP_POST_INSTALL_VALIDATION_FAILED, "forced validation failure")

    with pytest.raises(McpError) as e:
        _install(ctx, validate_fn=failing_validate)
    assert e.value.code == MCP_POST_INSTALL_VALIDATION_FAILED
    assert not os.path.isdir(str(ctx["plan"].install_directory))
    assert load_registry(None, ctx["base_dir"], ctx["managed_root"]) == {}


def test_failed_reinstall_preserves_the_previous_healthy_install(ctx):
    good = _install(ctx)
    entrypoint = os.path.join(good["install_directory"], *"node_modules/@modelcontextprotocol/server-filesystem/dist/index.js".split("/"))
    assert os.path.isfile(entrypoint)

    def failing_validate(*args, **kwargs):
        raise McpError(MCP_POST_INSTALL_VALIDATION_FAILED, "forced failure")

    with pytest.raises(McpError):
        _install(ctx, validate_fn=failing_validate, force_reinstall=True)

    # The previously installed, validated version is still recorded and intact.
    installed = get_installed("filesystem", None, ctx["base_dir"], ctx["managed_root"])
    assert installed is not None and installed.installed_version == ctx["plan"].package_version


def test_no_staging_directory_is_left_behind(ctx):
    _install(ctx)
    server_root = os.path.join(ctx["base_dir"], "app_data", "mcp_servers", "filesystem")
    assert not any(n.startswith(".staging") for n in _safe_listdir(server_root))


def _safe_listdir(path):
    try:
        return os.listdir(path)
    except OSError:
        return []
