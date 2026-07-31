"""Phase G.2 shared test helpers (not a test module).

Builds two INDEPENDENT fixture MCP servers ("filesystem-test", "document-test")
using the same real Node fixture server binary Phase F/F.1 already uses
(tests/fixtures/fake_filesystem_server.js), each with its own server_id, config
path, and approved root — never the production catalog or a fake server added
to it (Task 15).
"""

import json
import os

from mcp_management.registry import STATUS_INSTALLED, InstalledServer, upsert
from tests.mcp_provisioning_helpers import FIXTURE_SERVER, manager_paths, node_available

__all__ = ["FIXTURE_SERVER", "manager_paths", "node_available", "write_fixture_server_config"]


def write_fixture_server_config(paths, server_id, roots, tools=None):
    """Write a real, bootstrap-able managed config for `server_id` and register
    its installed-server registry entry. Returns the approved root(s) (realpath)."""
    server_root = os.path.join(paths["base_dir"], paths["managed_root"], server_id)
    os.makedirs(server_root, exist_ok=True)
    workspace = os.path.join(paths["base_dir"], "mcp_workspaces", server_id)
    os.makedirs(workspace, exist_ok=True)
    approved = tuple(os.path.realpath(str(r)) for r in roots)
    raw = {
        "enabled": True, "required": False, "server_id": server_id, "transport": "stdio",
        "command": "node", "args": [FIXTURE_SERVER, *approved], "working_directory": workspace,
        "startup_timeout_seconds": 15, "call_timeout_seconds": 10, "shutdown_timeout_seconds": 5,
        "environment_allowlist": [],
        "tool_policy": {"default_permission": "denied", "tools": tools or {
            "list_allowed_directories": {"enabled": True, "permission": "read"},
            "read_text_file": {"enabled": True, "permission": "read"},
            "list_directory": {"enabled": True, "permission": "read"},
        }},
    }
    config_path = os.path.join(server_root, "server.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    upsert(server_id, InstalledServer(
        catalog_id=server_id, installed_version="1.0.0", status=STATUS_INSTALLED,
        install_directory=server_root, configuration_path=config_path, installed_at="now",
        approved_directories=approved), None, paths["base_dir"], paths["managed_root"])
    return approved
