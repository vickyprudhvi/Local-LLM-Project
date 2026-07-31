"""Phase G.3 Task 20 — generalized catalog installer schema validation.

Proves: the python_venv installer type loads and validates correctly, the
existing Filesystem (npm) entry keeps loading completely unchanged, and every
listed catalog-validation rule fails closed.
"""

import copy

import pytest

from mcp_layer.errors import McpError
from mcp_management.catalog import build_catalog, load_catalog
from tests.auto_provisioning_helpers import calculator_test_catalog_entry_raw


def _catalog_with(entry_raw):
    return {"catalog_version": 1, "servers": {"calculator-test": entry_raw}}


def test_python_venv_entry_loads():
    catalog = build_catalog(_catalog_with(calculator_test_catalog_entry_raw()))
    entry = catalog.get("calculator-test")
    assert entry.installer_type == "python_venv"
    assert entry.launch_module == "calculator_test_mcp.server"
    assert entry.lock_file_relative == "config/mcp_locks/calculator-test-mcp-1.0.0.txt"
    assert entry.expected_tools == ("add", "echo")


def test_existing_filesystem_entry_still_loads_unchanged():
    catalog = load_catalog()
    entry = catalog.get("official-filesystem")
    assert entry.installer_type == "npm"
    assert entry.entrypoint_relative.endswith("index.js")
    assert entry.lock_file_relative is None
    assert entry.launch_module is None


def test_unknown_installer_type_rejected():
    raw = copy.deepcopy(calculator_test_catalog_entry_raw())
    raw["installer"]["type"] = "docker"
    with pytest.raises(McpError):
        build_catalog(_catalog_with(raw))


def test_missing_lock_file_rejected():
    raw = copy.deepcopy(calculator_test_catalog_entry_raw())
    del raw["installer"]["lock_file"]
    with pytest.raises(McpError):
        build_catalog(_catalog_with(raw))


def test_lock_file_path_traversal_rejected():
    raw = copy.deepcopy(calculator_test_catalog_entry_raw())
    raw["installer"]["lock_file"] = "../outside/lock.txt"
    with pytest.raises(McpError):
        build_catalog(_catalog_with(raw))


def test_wildcard_version_rejected():
    raw = copy.deepcopy(calculator_test_catalog_entry_raw())
    raw["installer"]["version"] = "1.0.*"
    with pytest.raises(McpError):
        build_catalog(_catalog_with(raw))


def test_latest_version_rejected():
    raw = copy.deepcopy(calculator_test_catalog_entry_raw())
    raw["installer"]["version"] = "latest"
    with pytest.raises(McpError):
        build_catalog(_catalog_with(raw))


def test_git_branch_version_rejected():
    raw = copy.deepcopy(calculator_test_catalog_entry_raw())
    raw["installer"]["version"] = "main"
    with pytest.raises(McpError):
        build_catalog(_catalog_with(raw))


def test_malformed_python_constraint_rejected():
    raw = copy.deepcopy(calculator_test_catalog_entry_raw())
    raw["installer"]["python_constraint"] = ">=3.9; rm -rf /"
    with pytest.raises(McpError):
        build_catalog(_catalog_with(raw))


def test_missing_launch_module_rejected():
    raw = copy.deepcopy(calculator_test_catalog_entry_raw())
    del raw["launch"]["module"]
    with pytest.raises(McpError):
        build_catalog(_catalog_with(raw))


def test_wrong_entrypoint_type_rejected():
    raw = copy.deepcopy(calculator_test_catalog_entry_raw())
    raw["launch"]["entrypoint_type"] = "shell_command"
    with pytest.raises(McpError):
        build_catalog(_catalog_with(raw))


def test_empty_expected_tools_rejected_for_python_venv():
    raw = copy.deepcopy(calculator_test_catalog_entry_raw())
    raw["expected_tools"] = []
    with pytest.raises(McpError):
        build_catalog(_catalog_with(raw))


def test_enabled_tool_missing_from_expected_tools_rejected():
    raw = copy.deepcopy(calculator_test_catalog_entry_raw())
    raw["default_tool_policy"]["tools"]["multiply"] = {"enabled": True, "permission": "read"}
    with pytest.raises(McpError):
        build_catalog(_catalog_with(raw))


def test_permission_outside_allowed_values_defaults_denied_not_rejected():
    raw = copy.deepcopy(calculator_test_catalog_entry_raw())
    raw["default_tool_policy"]["tools"]["add"]["permission"] = "sudo"
    catalog = build_catalog(_catalog_with(raw))
    entry = catalog.get("calculator-test")
    from tools.models import ToolPermission

    assert entry.default_tool_policy.tools["add"].permission == ToolPermission.DENIED


def test_duplicate_server_id_rejected():
    raw = _catalog_with(calculator_test_catalog_entry_raw())
    raw["servers"]["calculator-test-2"] = copy.deepcopy(raw["servers"]["calculator-test"])
    with pytest.raises(McpError):
        build_catalog(raw)


def test_invalid_install_host_rejected():
    raw = copy.deepcopy(calculator_test_catalog_entry_raw())
    raw["network_policy"] = {"install_hosts": ["not a host/with spaces"]}
    with pytest.raises(McpError):
        build_catalog(_catalog_with(raw))


def test_shell_fragment_in_launch_arguments_is_a_plain_argv_string_not_executed():
    raw = copy.deepcopy(calculator_test_catalog_entry_raw())
    raw["launch"]["arguments"] = ["--flag=1 && rm -rf /"]
    catalog = build_catalog(_catalog_with(raw))
    entry = catalog.get("calculator-test")
    # Accepted as a literal argv element (never interpreted by a shell — every
    # installer backend launches with shell=False), not rejected outright.
    assert entry.launch_arguments == ("--flag=1 && rm -rf /",)
