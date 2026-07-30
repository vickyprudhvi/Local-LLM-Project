"""Phase F.1 Task 4/13 — restricted-path screening, path validation edge cases, and
permission behavior after an access-root update. Reuses planner.validate_approved_directory
(already exercised by Phase F's own security suite, test_mcp_security.py) so the same
forbidden/broad-location rules apply here with no duplicated logic.
"""

import os

import pytest

from mcp_layer.errors import McpError
from mcp_management.access_classifier import propose_root
from tools.models import MCP_DIRECTORY_NOT_APPROVED


# ---- restricted locations are never proposed ----

@pytest.mark.parametrize("leaf", [".ssh", ".aws", ".gnupg", ".azure", ".kube", ".docker",
                                  "credentials", "secrets", ".password-store"])
def test_credential_directories_are_never_proposed(tmp_path, leaf):
    restricted = tmp_path / "home" / leaf
    restricted.mkdir(parents=True)
    target = restricted / "a_file"
    target.touch()
    proposal = propose_root([str(target)], remote_name="read_text_file", base_dir=str(tmp_path))
    assert proposal.ok is False
    assert proposal.restricted is True


def test_browser_profile_fragment_is_never_proposed(tmp_path):
    restricted = tmp_path / "AppData" / "Roaming" / "Mozilla" / "Firefox" / "Profiles" / "x"
    restricted.mkdir(parents=True)
    target = restricted / "cookies.sqlite"
    target.touch()
    proposal = propose_root([str(target)], remote_name="read_text_file", base_dir=str(tmp_path))
    assert proposal.ok is False
    assert proposal.restricted is True


def test_the_repo_root_itself_is_not_proposed(tmp_path):
    (tmp_path / "setup.py").touch()
    proposal = propose_root([str(tmp_path / "setup.py")], remote_name="read_text_file",
                            base_dir=str(tmp_path))
    assert proposal.ok is False


def test_a_narrow_child_of_documents_is_still_screened_as_broad_home(tmp_path):
    """Home itself (the base_dir here) is broad; a file directly inside it never
    proposes home as the root."""
    home_child = tmp_path / "notes.txt"
    home_child.touch()
    proposal = propose_root([str(home_child)], remote_name="read_text_file", base_dir=str(tmp_path))
    assert proposal.ok is False


def test_the_filesystem_root_is_never_proposed():
    root = os.path.abspath(os.sep)
    proposal = propose_root([os.path.join(root, "file.txt")], remote_name="read_text_file")
    assert proposal.ok is False


# ---- path validation edge cases ----

def test_null_byte_in_path_is_rejected():
    from mcp_management.access_classifier import classify_outside_root_failure
    from tools.models import ToolResult

    result = ToolResult.fail("mcp.filesystem.read_text_file", "c1", "MCP_CALL_FAILED", "denied")
    failure = classify_outside_root_failure(
        "mcp.filesystem.read_text_file", {"path": "C:\\a\\b\x00\\c.txt"}, result, ())
    assert failure is None


def test_traversal_component_does_not_escape_the_screen(tmp_path):
    from mcp_management.planner import validate_approved_directory

    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    # Traversing back up resolves (via realpath canonicalization) to base_dir
    # itself, which IS the broad-location screen's own base_dir entry — so a
    # traversal component cannot be used to sneak past the broad-location check.
    traversal = str(nested / ".." / "..")
    with pytest.raises(McpError):
        validate_approved_directory(traversal, base_dir=str(tmp_path), allow_broad=False)


def test_windows_style_path_resolves_correctly(tmp_path):
    nested = tmp_path / "project" / "chapter_pdfs"
    nested.mkdir(parents=True)
    windows_style = str(nested).replace("/", "\\") + "\\README.md"
    proposal = propose_root([windows_style], remote_name="read_text_file", base_dir=str(tmp_path))
    assert proposal.ok is True
    assert proposal.directory == os.path.realpath(str(nested))


# ---- permission behavior after an access update ----

def test_read_tools_stay_read_and_write_tools_stay_write_after_a_root_add(tmp_path):
    """The generated config's tool_policy is carried through untouched by an
    access-root update — server-advertised permissions were never consulted for
    it in the first place (Phase E discovery re-applies the LOCAL policy), and
    filesystem_access_update.py never touches `tool_policy` at all."""
    import json

    from mcp_layer.config import build_config
    from mcp_management.filesystem_access import FilesystemAccessPlan
    from mcp_management.filesystem_access_update import update_filesystem_access
    from mcp_management.registry import STATUS_INSTALLED, InstalledServer, upsert
    from tests.mcp_provisioning_helpers import manager_paths

    paths = manager_paths(tmp_path)
    approved = os.path.realpath(str(tmp_path / "root_a"))
    os.makedirs(approved, exist_ok=True)
    server_root = os.path.join(paths["base_dir"], paths["managed_root"], "filesystem")
    os.makedirs(server_root, exist_ok=True)
    config_path = os.path.join(server_root, "server.json")
    policy = {
        "default_permission": "denied",
        "tools": {
            "read_text_file": {"enabled": True, "permission": "read"},
            "write_file": {"enabled": True, "permission": "write"},
            "move_file": {"enabled": False, "permission": "denied"},
            "edit_file": {"enabled": False, "permission": "denied"},
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "enabled": True, "required": False, "server_id": "filesystem",
            "display_name": "Filesystem MCP Server", "transport": "stdio",
            "command": "node", "args": ["/entrypoint.js", approved],
            "working_directory": "./mcp_workspaces/filesystem",
            "startup_timeout_seconds": 15, "call_timeout_seconds": 15,
            "shutdown_timeout_seconds": 5, "environment_allowlist": [],
            "tool_policy": policy,
        }, f)
    upsert("filesystem", InstalledServer(
        catalog_id="official-filesystem", installed_version="2026.7.10",
        status=STATUS_INSTALLED, install_directory=server_root,
        configuration_path=config_path, installed_at="now",
        approved_directories=(approved,)), None, paths["base_dir"], paths["managed_root"])

    new_dir = os.path.realpath(str(tmp_path / "root_b"))
    os.makedirs(new_dir, exist_ok=True)
    proposed = tuple(sorted((approved, new_dir)))
    plan = FilesystemAccessPlan(
        plan_id="fsplan_1", server_id="filesystem", catalog_id="official-filesystem",
        operation="add_root", requested_directory=new_dir,
        current_allowed_directories=(approved,), proposed_allowed_directories=proposed,
    ).with_hash()

    class _Client:
        protocol_version = "test"

        def list_tools(self, timeout=None):
            return [{"name": "list_allowed_directories",
                    "inputSchema": {"type": "object", "properties": {}}}]

        def call_tool(self, name, arguments, timeout=None):
            return {"directories": list(proposed)}

        def shutdown(self):
            pass

    update_filesystem_access(plan, base_dir=paths["base_dir"], managed_root=paths["managed_root"],
                             start_server_fn=lambda config, **kw: _Client())

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    config = build_config(raw)
    assert config.tool_policy.tools["read_text_file"].permission.value == "read"
    assert config.tool_policy.tools["write_file"].permission.value == "write"
    assert config.tool_policy.tools["move_file"].enabled is False
    assert config.tool_policy.tools["edit_file"].enabled is False


def test_server_advertised_permissions_cannot_override_local_policy():
    """Same guarantee Phase E's discovery already provides (plan_registration uses
    ONLY the local config.tool_policy) — an access-root update reuses the exact
    same discovery path on the next real bootstrap, so this is unaffected."""
    from mcp_layer.discovery import plan_registration
    from mcp_layer.config import McpToolPolicy, McpToolPolicyEntry
    from tools.models import ToolPermission

    config = type("C", (), {"tool_policy": McpToolPolicy(
        default_permission=ToolPermission.DENIED,
        tools={"write_file": McpToolPolicyEntry(enabled=True, permission=ToolPermission.WRITE)})})()
    raw_tools = [{"name": "write_file", "description": "", "annotations": {"readOnlyHint": True},
                 "inputSchema": {"type": "object", "properties": {}}}]
    registrations, _ = plan_registration(raw_tools, config)
    assert registrations[0]["permission"] == ToolPermission.WRITE  # server's hint is ignored
