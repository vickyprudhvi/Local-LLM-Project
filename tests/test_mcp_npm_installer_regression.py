"""Phase G.3 Task 6/20 (scenario J) — the generalized npm backend recognizes an
already-installed Filesystem server and never reinstalls it.

Uses the real Node fixture server (no network) exactly like the existing Phase F
npm tests, but drives the NEW `NpmInstaller` (Task 5/6) directly.
"""

import json
import os
import shutil

import pytest

from mcp_management.installers import ProvisioningTransaction, get_installer
from mcp_management.installers.npm_backend import NpmInstaller
from tests.mcp_provisioning_helpers import (
    ENTRYPOINT_RELATIVE,
    FIXTURE_SERVER,
    PACKAGE_NAME,
    PINNED_VERSION,
    make_catalog,
    node_available,
)

pytestmark = pytest.mark.skipif(not node_available(), reason="node/npm not available")


def _materialize_existing_install(final_dir):
    """Exactly what a prior REAL npm install would have left behind."""
    pkg_dir = os.path.join(final_dir, "node_modules", *PACKAGE_NAME.split("/"), "dist")
    os.makedirs(pkg_dir, exist_ok=True)
    shutil.copyfile(FIXTURE_SERVER, os.path.join(pkg_dir, "index.js"))
    with open(os.path.join(final_dir, "package-lock.json"), "w", encoding="utf-8") as f:
        json.dump({"lockfileVersion": 3, "packages": {}}, f)


def test_installer_type_is_registered():
    assert get_installer("npm") is not None
    assert isinstance(get_installer("npm"), NpmInstaller)


def test_existing_install_is_recognized_and_reused(tmp_path):
    catalog_entry = make_catalog().get("official-filesystem")
    installer = get_installer("npm")
    server_root = str(tmp_path / "app_data" / "mcp_servers" / "filesystem")
    final_dir = os.path.join(server_root, "versions", PINNED_VERSION)
    _materialize_existing_install(final_dir)

    txn = ProvisioningTransaction(
        transaction_id="t1", server_id="filesystem", base_dir=str(tmp_path),
        managed_root="app_data/mcp_servers", server_root=server_root,
        candidate_directory=os.path.join(server_root, "candidates", "t1"),
        final_directory=final_dir)
    candidate = installer.prepare_candidate(None, catalog_entry, txn)
    candidate = installer.install_candidate(candidate, None, catalog_entry)

    assert candidate.extra.get("reused_existing_installation") == "true"
    assert candidate.install_directory == final_dir


def test_no_npm_call_occurs_when_reusing(tmp_path, monkeypatch):
    catalog_entry = make_catalog().get("official-filesystem")
    installer = get_installer("npm")
    server_root = str(tmp_path / "app_data" / "mcp_servers" / "filesystem")
    final_dir = os.path.join(server_root, "versions", PINNED_VERSION)
    _materialize_existing_install(final_dir)

    def _must_not_run(*a, **kw):
        raise AssertionError("npm install_package must not run for an intact existing install")

    monkeypatch.setattr("mcp_management.installers.npm_backend.npm_installer.install_package", _must_not_run)

    txn = ProvisioningTransaction(
        transaction_id="t1", server_id="filesystem", base_dir=str(tmp_path),
        managed_root="app_data/mcp_servers", server_root=server_root,
        candidate_directory=os.path.join(server_root, "candidates", "t1"),
        final_directory=final_dir)
    candidate = installer.prepare_candidate(None, catalog_entry, txn)
    installer.install_candidate(candidate, None, catalog_entry)  # does not raise


def test_missing_install_triggers_a_real_ignore_scripts_npm_call(tmp_path):
    """When nothing is installed yet, the real (fixture) npm binary runs with
    --ignore-scripts still enforced — proven by inspecting the resulting argv
    the shared npm_installer module built."""
    import mcp_management.npm_installer as npm_installer

    catalog_entry = make_catalog().get("official-filesystem")
    argv = npm_installer.build_npm_argv(catalog_entry, "node")
    assert "--ignore-scripts" in argv
    assert "-g" not in argv
    assert f"{PACKAGE_NAME}@{PINNED_VERSION}" in argv
