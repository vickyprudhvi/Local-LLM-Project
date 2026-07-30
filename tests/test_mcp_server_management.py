"""Phase F — disable, enable, repair, uninstall, update reporting, and the registry."""

import json
import os

import pytest

from mcp_layer.errors import McpError
from mcp_management.models import ProvisioningApproval
from mcp_management.registry import (
    STATUS_DISABLED,
    STATUS_INSTALLED,
    InstalledServer,
    load_registry,
    upsert,
)
from mcp_layer.config_resolver import McpConfigSource, resolve_config
from tests.mcp_provisioning_helpers import (
    FakeNpm,
    make_catalog,
    make_manager,
    managed_config_file,
    node_available,
    workspace_with_file,
    write_template,
)
from tools.models import (
    MCP_NOT_INSTALLED,
    MCP_REGISTRY_CORRUPT,
    MCP_UPDATE_AVAILABLE,
)


@pytest.fixture
def provisioned(tmp_path):
    if not node_available():
        pytest.skip("node/npm not available")
    manager, paths = make_manager(tmp_path)
    workspace = workspace_with_file(tmp_path)
    plan = manager.prepare_plan("official-filesystem", [workspace])
    approval = ProvisioningApproval(True, plan.plan_id, plan.compute_hash())
    result = manager.provision(plan, approval=approval, npm_runner=FakeNpm())
    return {"manager": manager, "paths": paths, "plan": plan, "result": result,
            "workspace": workspace}


def _managed_config(paths):
    path = managed_config_file(paths)
    if not os.path.isfile(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def _template_bytes(paths):
    path = os.path.join(paths["base_dir"], paths["template_path"])
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return f.read()


# ---- disable / enable ----

def test_disable_marks_managed_config_disabled_but_keeps_files(provisioned):
    manager, paths = provisioned["manager"], provisioned["paths"]
    before = _template_bytes(paths)
    result = manager.disable("filesystem")
    assert result["status"] == STATUS_DISABLED
    assert _managed_config(paths)["enabled"] is False
    # Installed files are preserved so re-enabling needs no reinstall.
    assert os.path.isdir(str(provisioned["plan"].install_directory))
    registry = load_registry(None, paths["base_dir"], paths["managed_root"])
    assert registry["filesystem"].status == STATUS_DISABLED
    # The committed template is never touched.
    assert _template_bytes(paths) == before


def test_disabled_server_is_not_selected_by_the_resolver(provisioned):
    manager, paths = provisioned["manager"], provisioned["paths"]
    write_template(paths, enabled=False)
    manager.disable("filesystem")
    resolved = resolve_config(base_dir=paths["base_dir"], managed_root=paths["managed_root"],
                              override="", template_path=paths["template_path"])
    assert resolved.source is McpConfigSource.DEFAULT_TEMPLATE


def test_enable_reactivates_without_reinstalling(provisioned, monkeypatch):
    manager, paths = provisioned["manager"], provisioned["paths"]
    manager.disable("filesystem")

    # Any npm invocation during enable would be a bug.
    import mcp_management.npm_installer as npm_mod
    monkeypatch.setattr(npm_mod, "install_package",
                        lambda *a, **k: pytest.fail("enable must not reinstall"))

    result = manager.enable("filesystem")
    assert result["status"] == STATUS_INSTALLED
    assert result["reinstalled"] is False
    assert _managed_config(paths)["enabled"] is True
    resolved = resolve_config(base_dir=paths["base_dir"], managed_root=paths["managed_root"],
                              override="", template_path=paths["template_path"])
    assert resolved.source is McpConfigSource.MANAGED_ACTIVE


def test_disable_unknown_server_reports_not_installed(tmp_path):
    manager, _ = make_manager(tmp_path)
    with pytest.raises(McpError) as e:
        manager.disable("filesystem")
    assert e.value.code == MCP_NOT_INSTALLED


# ---- repair ----

def test_repair_of_intact_install_does_not_reinstall(provisioned):
    result = provisioned["manager"].repair("filesystem")
    assert result["reinstalled"] is False
    assert result["record_present"] is True


def test_repair_uses_the_same_pinned_version(provisioned):
    """A catalog that now pins a different version must not be silently applied."""
    manager, paths = provisioned["manager"], provisioned["paths"]
    manager.catalog = make_catalog(version="9.9.9")
    with pytest.raises(McpError) as e:
        manager.repair("filesystem")
    assert e.value.code == MCP_UPDATE_AVAILABLE
    assert "does not upgrade" in e.value.message


def test_repair_reinstalls_when_files_are_missing(provisioned):
    import shutil

    manager = provisioned["manager"]
    shutil.rmtree(str(provisioned["plan"].install_directory), ignore_errors=True)
    called = {}

    def reinstall(catalog_entry, installed):
        called["version"] = installed.installed_version
        return {"status": STATUS_INSTALLED}

    result = manager.repair("filesystem", reinstall_fn=reinstall)
    assert result["reinstalled"] is True
    # Repair restores the SAME version it had, never a newer one.
    assert called["version"] == provisioned["plan"].package_version


# ---- uninstall ----

def test_uninstall_removes_managed_files_only(provisioned):
    manager, paths = provisioned["manager"], provisioned["paths"]
    workspace = provisioned["workspace"]
    install_dir = str(provisioned["plan"].install_directory)
    write_template(paths, enabled=False)
    template_before = _template_bytes(paths)

    result = manager.uninstall("filesystem")
    assert result["removed"] is True
    assert not os.path.isdir(install_dir)
    assert _managed_config(paths) is None
    assert load_registry(None, paths["base_dir"], paths["managed_root"]) == {}
    # The committed template survives untouched and becomes the effective config.
    assert _template_bytes(paths) == template_before
    resolved = resolve_config(base_dir=paths["base_dir"], managed_root=paths["managed_root"],
                              override="", template_path=paths["template_path"])
    assert resolved.source is McpConfigSource.DEFAULT_TEMPLATE

    # The user's approved directory and its contents are untouched.
    assert os.path.isdir(workspace)
    assert sorted(os.listdir(workspace)) == ["hello.txt"]
    with open(os.path.join(workspace, "hello.txt"), encoding="utf-8") as f:
        assert "Hello from automatic MCP provisioning!" in f.read()


def test_uninstall_preserves_a_bounded_audit_record(provisioned):
    manager, paths = provisioned["manager"], provisioned["paths"]
    manager.uninstall("filesystem")
    server_root = os.path.join(paths["base_dir"], "app_data", "mcp_servers", "filesystem")
    record_path = os.path.join(server_root, "uninstall-record.json")
    assert os.path.isfile(record_path)
    record = json.load(open(record_path, encoding="utf-8"))
    assert record["package_version"] == provisioned["plan"].package_version
    assert record["plan_hash"]


def test_double_uninstall_is_safe(provisioned):
    manager = provisioned["manager"]
    assert manager.uninstall("filesystem")["removed"] is True
    second = manager.uninstall("filesystem")
    assert second["removed"] is False
    assert second["already_absent"] is True


# ---- updates are never automatic ----

def test_update_is_reported_not_applied(provisioned):
    manager = provisioned["manager"]
    manager.catalog = make_catalog(version="9.9.9")
    status = manager.check_for_update("filesystem")
    assert status["update_available"] is True
    assert status["error_code"] == MCP_UPDATE_AVAILABLE
    assert status["installed_version"] == provisioned["plan"].package_version
    # Nothing changed on disk.
    assert os.path.isdir(str(provisioned["plan"].install_directory))


def test_no_update_reported_when_versions_match(provisioned):
    assert provisioned["manager"].check_for_update("filesystem")["update_available"] is False


# ---- registry integrity ----

def test_registry_writes_are_atomic_and_reloadable(tmp_path):
    base_dir = str(tmp_path / "repo")
    entry = InstalledServer(
        catalog_id="official-filesystem", installed_version="1.0.0",
        status=STATUS_INSTALLED, install_directory="/tmp/x",
        configuration_path="/tmp/x/server.json", installed_at="now",
    )
    upsert("filesystem", entry, None, base_dir, "app_data/mcp_servers")
    reloaded = load_registry(None, base_dir, "app_data/mcp_servers")
    assert reloaded["filesystem"].installed_version == "1.0.0"
    # No temporary artifacts left behind.
    managed = os.path.join(base_dir, "app_data", "mcp_servers")
    assert not [n for n in os.listdir(managed) if n.startswith(".tmp_")]


def test_corrupt_registry_is_not_silently_overwritten(tmp_path):
    base_dir = str(tmp_path / "repo")
    managed = os.path.join(base_dir, "app_data", "mcp_servers")
    os.makedirs(managed, exist_ok=True)
    path = os.path.join(managed, "installed_servers.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{ corrupt")

    with pytest.raises(McpError) as e:
        load_registry(None, base_dir, "app_data/mcp_servers")
    assert e.value.code == MCP_REGISTRY_CORRUPT
    with open(path, encoding="utf-8") as f:
        assert f.read() == "{ corrupt"  # untouched


def test_registry_missing_field_is_reported(tmp_path):
    base_dir = str(tmp_path / "repo")
    managed = os.path.join(base_dir, "app_data", "mcp_servers")
    os.makedirs(managed, exist_ok=True)
    with open(os.path.join(managed, "installed_servers.json"), "w", encoding="utf-8") as f:
        json.dump({"registry_version": 1, "servers": {"filesystem": {"catalog_id": "x"}}}, f)
    with pytest.raises(McpError) as e:
        load_registry(None, base_dir, "app_data/mcp_servers")
    assert e.value.code == MCP_REGISTRY_CORRUPT


def test_status_reports_catalog_and_installed_state(provisioned):
    status = provisioned["manager"].status()
    assert status["catalog_entries"] == ["official-filesystem"]
    assert "filesystem" in status["installed"]
    assert status["installed"]["filesystem"]["installed_version"]
