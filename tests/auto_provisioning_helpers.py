"""Phase G.3 — shared test fixtures for the generalized auto-provisioning suite.

`calculator_test_catalog_entry()` is a TEST-ONLY trusted-catalog entry (never
written to `config/mcp_catalog.json`) for the `calculator-test` fixture MCP
server built from `tests/fixtures/calculator_mcp_pkg` — a real, local, offline
Python package (no network) exposing `add`/`echo` tools, used to prove the
generalized python_venv installer end-to-end without adding a second
production provider.
"""

import os

from mcp_management.catalog import build_catalog

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CALCULATOR_PACKAGE_NAME = "calculator-test-mcp"
CALCULATOR_PACKAGE_VERSION = "1.0.0"
CALCULATOR_LOCK_FILE_RELATIVE = "config/mcp_locks/calculator-test-mcp-1.0.0.txt"


def calculator_test_catalog_entry_raw():
    return {
        "server_id": "calculator-test",
        "display_name": "Calculator Test MCP",
        "description": "Test-only fixture MCP server exposing add/echo tools.",
        "capabilities": ["arithmetic_calculation"],
        "risk_category": "test_fixture",
        "transport": "stdio",
        "required_runtimes": [],
        "installer": {
            "type": "python_venv",
            "package": CALCULATOR_PACKAGE_NAME,
            "version": CALCULATOR_PACKAGE_VERSION,
            "lock_file": CALCULATOR_LOCK_FILE_RELATIVE,
            "python_constraint": ">=3.9",
        },
        "launch": {
            "transport": "stdio",
            "entrypoint_type": "python_module",
            "module": "calculator_test_mcp.server",
            "arguments": [],
        },
        "expected_tools": ["add", "echo"],
        "default_tool_policy": {
            "default_permission": "denied",
            "tools": {
                "add": {"enabled": True, "permission": "read"},
                "echo": {"enabled": True, "permission": "read"},
            },
        },
        "granular_capabilities": ["arithmetic_calculation"],
        "selection_hints": {
            "explicit_names": ["calculator", "calculator test"],
            "actions": {"arithmetic_calculation": ["add", "calculate", "compute", "sum"]},
        },
        "network_policy": {"install_hosts": []},
    }


def calculator_test_catalog_entry():
    catalog = build_catalog({
        "catalog_version": 1,
        "servers": {"calculator-test": calculator_test_catalog_entry_raw()},
    })
    return catalog.get("calculator-test")


def catalog_with_calculator_test(extra_entries=None, catalog_version=1):
    """A full McpCatalog containing ONLY the calculator-test fixture (plus any
    extra raw entries the caller supplies, e.g. a Filesystem/document-test entry
    for multi-server isolation tests)."""
    servers = {"calculator-test": calculator_test_catalog_entry_raw()}
    servers.update(extra_entries or {})
    return build_catalog({"catalog_version": catalog_version, "servers": servers})


def build_auto_provisioning_env(tmp_path, extra_entries=None, registry=None):
    """A fully wired (manager, runtime_manager, reg, catalog) triple under an
    isolated tmp_path base_dir — the standard fixture for end-to-end Phase G.3
    tests that drive the REAL candidate transaction."""
    from mcp_layer.runtime_manager import MultiMcpRuntimeManager
    from mcp_management.auto_provisioning import AutoProvisioningManager
    from tools.registry import default_registry

    base_dir = str(tmp_path)
    catalog = catalog_with_calculator_test(extra_entries=extra_entries)
    reg = registry if registry is not None else default_registry()
    manager = AutoProvisioningManager(catalog, base_dir=base_dir, managed_root="app_data/mcp_servers")
    runtime_manager = MultiMcpRuntimeManager(reg, base_dir=base_dir, managed_root="app_data/mcp_servers")
    return {"catalog": catalog, "reg": reg, "manager": manager, "runtime_manager": runtime_manager,
           "base_dir": base_dir}
