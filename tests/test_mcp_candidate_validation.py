"""Phase G.3 Task 10/20 (scenarios N, O) — real candidate MCP process validation."""

import pytest

from mcp_layer.errors import McpError
from mcp_management.auto_provisioning import _validate_candidate_process
from mcp_layer.config import build_config
from tests.auto_provisioning_helpers import calculator_test_catalog_entry
from tools.models import MCP_EXPECTED_TOOL_MISSING


def _config_for(entry, base_dir, python_exe):
    import os

    workspace = os.path.join(base_dir, "mcp_workspaces", "calculator-test")
    os.makedirs(workspace, exist_ok=True)
    return build_config({
        "enabled": True, "required": False, "server_id": entry.server_id,
        "display_name": entry.display_name, "transport": "stdio",
        "command": python_exe, "args": ["-m", entry.launch_module],
        "working_directory": workspace,
        "startup_timeout_seconds": 15, "call_timeout_seconds": 10, "shutdown_timeout_seconds": 5,
        "environment_allowlist": [], "tool_policy": {"default_permission": "denied", "tools": {
            name: {"enabled": e.enabled, "permission": e.permission.value}
            for name, e in entry.default_tool_policy.tools.items()}},
    })


@pytest.fixture
def installed_calculator(tmp_path):
    """A REAL, installed (but not yet production-registered) calculator-test
    venv — reused across validation scenarios."""
    from mcp_management.installers import ProvisioningTransaction, get_installer

    entry = calculator_test_catalog_entry()
    installer = get_installer("python_venv")
    server_root = str(tmp_path / "app_data" / "mcp_servers" / "calculator-test")
    txn = ProvisioningTransaction(
        transaction_id="t1", server_id="calculator-test", base_dir=str(tmp_path),
        managed_root="app_data/mcp_servers", server_root=server_root,
        candidate_directory=server_root + "/candidates/t1", final_directory=server_root + "/versions/1.0.0")
    candidate = installer.prepare_candidate(None, entry, txn)
    candidate = installer.install_candidate(candidate, None, entry)
    return entry, candidate, str(tmp_path)


def test_candidate_process_exposes_exactly_the_expected_tools(installed_calculator):
    entry, candidate, base_dir = installed_calculator
    config = _config_for(entry, base_dir, candidate.extra["venv_python"])
    report = _validate_candidate_process(config, entry, base_dir)
    assert set(report["expected_tools_present"]) == {"add", "echo"}
    assert report["discovered_tool_count"] == 2


def test_missing_expected_tool_rejected(installed_calculator):
    from dataclasses import replace

    entry, candidate, base_dir = installed_calculator
    stricter = replace(entry, expected_tools=("add", "echo", "multiply"))
    config = _config_for(stricter, base_dir, candidate.extra["venv_python"])
    with pytest.raises(McpError) as exc:
        _validate_candidate_process(config, stricter, base_dir)
    assert exc.value.code == MCP_EXPECTED_TOOL_MISSING


def test_exact_name_comparison_not_suffix_matching(installed_calculator):
    """A catalog that (hypothetically) expects a NAMESPACED or suffixed tool
    name must never match the candidate's bare 'add'/'echo' — proving there is
    no suffix-only matching anywhere in this check."""
    from dataclasses import replace

    entry, candidate, base_dir = installed_calculator
    mismatched = replace(entry, expected_tools=("calculator-test.add",))
    config = _config_for(mismatched, base_dir, candidate.extra["venv_python"])
    with pytest.raises(McpError) as exc:
        _validate_candidate_process(config, mismatched, base_dir)
    assert exc.value.code == MCP_EXPECTED_TOOL_MISSING
