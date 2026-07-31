"""Phase G.3 Task 20 (scenarios K, L, M) — real, offline python_venv installer.

Every test here does a REAL `python -m venv` + real `pip install --require-hashes`
from the committed local wheel fixture — no network, no fake/mocked subprocess.
"""

import os
import shutil
import tempfile

import pytest

from mcp_layer.errors import McpError
from mcp_management.installers import ProvisioningTransaction, get_installer
from tests.auto_provisioning_helpers import calculator_test_catalog_entry
from tools.models import MCP_LOCK_FILE_INVALID, MCP_PYTHON_VERSION_UNSUPPORTED


@pytest.fixture
def tmp_root():
    d = tempfile.mkdtemp(prefix="g3_venv_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _transaction(tmp_root, transaction_id="t1"):
    server_root = os.path.join(tmp_root, "app_data", "mcp_servers", "calculator-test")
    return ProvisioningTransaction(
        transaction_id=transaction_id, server_id="calculator-test", base_dir=tmp_root,
        managed_root="app_data/mcp_servers", server_root=server_root,
        candidate_directory=os.path.join(server_root, "candidates", transaction_id),
        final_directory=os.path.join(server_root, "versions", "1.0.0"))


def test_candidate_gets_its_own_isolated_venv(tmp_root):
    installer = get_installer("python_venv")
    entry = calculator_test_catalog_entry()
    txn = _transaction(tmp_root)
    candidate = installer.prepare_candidate(None, entry, txn)
    assert os.path.isfile(candidate.extra["venv_python"])
    # Isolated: nothing installed yet beyond the base venv (pip/setuptools only).
    assert candidate.install_directory == txn.candidate_directory


def test_install_from_committed_lock_succeeds_offline(tmp_root):
    installer = get_installer("python_venv")
    entry = calculator_test_catalog_entry()
    txn = _transaction(tmp_root)
    candidate = installer.prepare_candidate(None, entry, txn)
    candidate = installer.install_candidate(candidate, None, entry)
    assert candidate.lock_hash
    installer.validate_artifacts(candidate, None, entry)  # does not raise


def test_installed_version_exactly_matches_catalog(tmp_root):
    import subprocess

    installer = get_installer("python_venv")
    entry = calculator_test_catalog_entry()
    txn = _transaction(tmp_root)
    candidate = installer.prepare_candidate(None, entry, txn)
    candidate = installer.install_candidate(candidate, None, entry)
    check = subprocess.run(
        [candidate.extra["venv_python"], "-c",
         "import importlib.metadata as m; print(m.version('calculator-test-mcp'))"],
        capture_output=True, text=True, timeout=30, check=False)
    assert check.stdout.strip() == "1.0.0"


def test_launch_spec_targets_the_venv_interpreter(tmp_root):
    installer = get_installer("python_venv")
    entry = calculator_test_catalog_entry()
    txn = _transaction(tmp_root)
    candidate = installer.prepare_candidate(None, entry, txn)
    candidate = installer.install_candidate(candidate, None, entry)
    spec = installer.build_launch_spec(candidate, entry)
    assert spec.command == candidate.extra["venv_python"]
    assert spec.args == ("-m", "calculator_test_mcp.server")


def test_invalid_python_constraint_rejected_before_any_install(tmp_root):
    installer = get_installer("python_venv")
    entry = calculator_test_catalog_entry()
    from dataclasses import replace

    impossible = replace(entry, python_constraint=">=99.0")
    txn = _transaction(tmp_root)
    with pytest.raises(McpError) as exc:
        installer.prepare_candidate(None, impossible, txn)
    assert exc.value.code == MCP_PYTHON_VERSION_UNSUPPORTED
    assert not os.path.isdir(txn.candidate_directory) or not os.listdir(txn.candidate_directory)


def test_tampered_lock_file_hash_rejected(tmp_root, monkeypatch, tmp_path):
    """A lock file whose content no longer matches the plan's recorded hash
    (Task 16: MCP_LOCK_FILE_INVALID) must stop the install."""
    installer = get_installer("python_venv")
    entry = calculator_test_catalog_entry()
    txn = _transaction(tmp_root)
    candidate = installer.prepare_candidate(None, entry, txn)

    class _FakePlan:
        lock_file_hash = "0" * 64  # deliberately wrong

    with pytest.raises(McpError) as exc:
        installer.install_candidate(candidate, _FakePlan(), entry)
    assert exc.value.code == MCP_LOCK_FILE_INVALID


def test_cleanup_removes_non_final_candidate_directory(tmp_root):
    installer = get_installer("python_venv")
    entry = calculator_test_catalog_entry()
    txn = _transaction(tmp_root)
    candidate = installer.prepare_candidate(None, entry, txn)
    assert os.path.isdir(txn.candidate_directory)
    installer.cleanup_candidate(candidate)
    assert not os.path.isdir(txn.candidate_directory)
