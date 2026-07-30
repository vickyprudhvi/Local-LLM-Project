"""Phase F.1 Task 8-9 — the transactional in-place config update, and rollback.

Uses a `_FakeClient` (same shape as test_mcp_post_install_validation.py's) instead
of a real Node process, so these tests are fast and hermetic while still exercising
the real `update_filesystem_access` transaction end to end.
"""

import json
import os

import pytest

from mcp_layer.errors import McpError
from mcp_management.filesystem_access import FilesystemAccessOperation, FilesystemAccessPlan
from mcp_management.filesystem_access_update import update_filesystem_access
from mcp_management.registry import STATUS_INSTALLED, InstalledServer, get_installed, upsert
from tests.mcp_provisioning_helpers import manager_paths
from tools.models import MCP_FILESYSTEM_ACCESS_VALIDATION_FAILED


class _FakeClient:
    def __init__(self, tools=None, call_results=None, fail_call=None, fail_start=False):
        self._tools = tools if tools is not None else []
        self._call_results = call_results or {}
        self._fail_call = fail_call
        self.fail_start = fail_start
        self.shutdown_called = False
        self.protocol_version = "2024-11-05"

    def list_tools(self, timeout=None):
        return self._tools

    def call_tool(self, name, arguments, timeout=None):
        if self._fail_call == name:
            raise McpError("MCP_CALL_FAILED", "forced failure")
        return self._call_results.get(name, {})

    def shutdown(self):
        self.shutdown_called = True


def _tool(name):
    return {"name": name, "description": "", "inputSchema": {"type": "object", "properties": {}}}


def _setup(tmp_path, approved=("root_a",)):
    paths = manager_paths(tmp_path)
    approved_abs = []
    for name in approved:
        d = os.path.join(paths["base_dir"], name)
        os.makedirs(d, exist_ok=True)
        approved_abs.append(os.path.realpath(d))
    server_root = os.path.join(paths["base_dir"], paths["managed_root"], "filesystem")
    os.makedirs(server_root, exist_ok=True)
    config_path = os.path.join(server_root, "server.json")
    raw = {
        "enabled": True, "required": False, "server_id": "filesystem",
        "display_name": "Filesystem MCP Server", "transport": "stdio",
        "command": "node", "args": ["/entrypoint.js", *approved_abs],
        "working_directory": "./mcp_workspaces/filesystem",
        "startup_timeout_seconds": 15, "call_timeout_seconds": 15,
        "shutdown_timeout_seconds": 5, "environment_allowlist": [],
        "tool_policy": {"default_permission": "denied", "tools": {
            "read_text_file": {"enabled": True, "permission": "read"},
            "list_allowed_directories": {"enabled": True, "permission": "read"},
        }},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    upsert("filesystem", InstalledServer(
        catalog_id="official-filesystem", installed_version="2026.7.10",
        status=STATUS_INSTALLED, install_directory=server_root,
        configuration_path=config_path, installed_at="now",
        approved_directories=tuple(approved_abs)),
        None, paths["base_dir"], paths["managed_root"])
    return paths, approved_abs, config_path


def _plan(server_id, current, proposed, operation=FilesystemAccessOperation.ADD_ROOT):
    return FilesystemAccessPlan(
        plan_id="fsplan_1", server_id=server_id, catalog_id="official-filesystem",
        operation=operation, requested_directory=proposed[-1] if proposed else "",
        current_allowed_directories=tuple(current), proposed_allowed_directories=tuple(proposed),
    ).with_hash()


def test_managed_config_is_updated_with_the_new_root(tmp_path):
    paths, approved, config_path = _setup(tmp_path)
    new_dir = os.path.realpath(str(tmp_path / "root_b"))
    os.makedirs(new_dir, exist_ok=True)
    proposed = tuple(sorted(approved + [new_dir]))
    plan = _plan("filesystem", approved, proposed)

    client = _FakeClient(tools=[_tool("list_allowed_directories")],
                         call_results={"list_allowed_directories": {"directories": list(proposed)}})
    update_filesystem_access(plan, base_dir=paths["base_dir"], managed_root=paths["managed_root"],
                             start_server_fn=lambda config, **kw: client)

    with open(config_path, "r", encoding="utf-8") as f:
        written = json.load(f)
    assert written["args"][1:] == sorted(proposed)
    assert client.shutdown_called is True  # no orphan process


def test_tracked_template_is_never_touched(tmp_path):
    paths, approved, config_path = _setup(tmp_path)
    template_path = os.path.join(paths["base_dir"], paths["template_path"])
    os.makedirs(os.path.dirname(template_path), exist_ok=True)
    with open(template_path, "w", encoding="utf-8") as f:
        f.write('{"enabled": false}')
    before = os.path.getmtime(template_path)

    new_dir = os.path.realpath(str(tmp_path / "root_b"))
    os.makedirs(new_dir, exist_ok=True)
    proposed = tuple(sorted(approved + [new_dir]))
    plan = _plan("filesystem", approved, proposed)
    client = _FakeClient(tools=[_tool("list_allowed_directories")],
                         call_results={"list_allowed_directories": {"directories": list(proposed)}})
    update_filesystem_access(plan, base_dir=paths["base_dir"], managed_root=paths["managed_root"],
                             start_server_fn=lambda config, **kw: client)
    assert os.path.getmtime(template_path) == before


def test_npm_installer_is_never_called_during_an_access_root_change(tmp_path, monkeypatch):
    import mcp_management.npm_installer as npm_installer

    def _forbidden(*args, **kwargs):
        raise AssertionError("npm_installer.install_package must never run for an access-root change")

    monkeypatch.setattr(npm_installer, "install_package", _forbidden)

    paths, approved, config_path = _setup(tmp_path)
    new_dir = os.path.realpath(str(tmp_path / "root_b"))
    os.makedirs(new_dir, exist_ok=True)
    proposed = tuple(sorted(approved + [new_dir]))
    plan = _plan("filesystem", approved, proposed)
    client = _FakeClient(tools=[_tool("list_allowed_directories")],
                         call_results={"list_allowed_directories": {"directories": list(proposed)}})
    update_filesystem_access(plan, base_dir=paths["base_dir"], managed_root=paths["managed_root"],
                             start_server_fn=lambda config, **kw: client)  # must not raise


def test_existing_roots_remain_and_new_root_is_added_once(tmp_path):
    paths, approved, config_path = _setup(tmp_path, approved=("root_a", "root_b"))
    new_dir = os.path.realpath(str(tmp_path / "root_c"))
    os.makedirs(new_dir, exist_ok=True)
    proposed = tuple(sorted(set(approved) | {new_dir}))
    plan = _plan("filesystem", approved, proposed)
    client = _FakeClient(tools=[_tool("list_allowed_directories")],
                         call_results={"list_allowed_directories": {"directories": list(proposed)}})
    result = update_filesystem_access(plan, base_dir=paths["base_dir"], managed_root=paths["managed_root"],
                                      start_server_fn=lambda config, **kw: client)
    assert set(approved) <= set(result["approved_directories"])
    assert result["approved_directories"].count(new_dir) == 1


def test_generated_config_passes_phase_e_validation(tmp_path):
    from mcp_layer.config import build_config

    paths, approved, config_path = _setup(tmp_path)
    new_dir = os.path.realpath(str(tmp_path / "root_b"))
    os.makedirs(new_dir, exist_ok=True)
    proposed = tuple(sorted(approved + [new_dir]))
    plan = _plan("filesystem", approved, proposed)
    client = _FakeClient(tools=[_tool("list_allowed_directories")],
                         call_results={"list_allowed_directories": {"directories": list(proposed)}})
    update_filesystem_access(plan, base_dir=paths["base_dir"], managed_root=paths["managed_root"],
                             start_server_fn=lambda config, **kw: client)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    build_config(raw)  # must not raise


# ---- rollback / failure paths ----

def test_validation_failure_leaves_the_previous_config_untouched(tmp_path):
    paths, approved, config_path = _setup(tmp_path)
    with open(config_path, "r", encoding="utf-8") as f:
        before = f.read()

    new_dir = os.path.realpath(str(tmp_path / "root_b"))
    os.makedirs(new_dir, exist_ok=True)
    proposed = tuple(sorted(approved + [new_dir]))
    plan = _plan("filesystem", approved, proposed)
    # The server reports roots beyond what was approved -> the verifier must fail
    # BEFORE anything is written.
    client = _FakeClient(tools=[_tool("list_allowed_directories")],
                         call_results={"list_allowed_directories": {"directories": [os.path.abspath(os.sep)]}})

    with pytest.raises(McpError) as exc:
        update_filesystem_access(plan, base_dir=paths["base_dir"], managed_root=paths["managed_root"],
                                 start_server_fn=lambda config, **kw: client)
    assert exc.value.code == MCP_FILESYSTEM_ACCESS_VALIDATION_FAILED
    with open(config_path, "r", encoding="utf-8") as f:
        after = f.read()
    assert after == before
    installed = get_installed("filesystem", None, paths["base_dir"], paths["managed_root"])
    assert list(installed.approved_directories) == approved
    assert client.shutdown_called is True  # no orphan process even on failure


def test_start_failure_leaves_the_previous_config_untouched(tmp_path):
    paths, approved, config_path = _setup(tmp_path)
    with open(config_path, "r", encoding="utf-8") as f:
        before = f.read()

    new_dir = os.path.realpath(str(tmp_path / "root_b"))
    os.makedirs(new_dir, exist_ok=True)
    proposed = tuple(sorted(approved + [new_dir]))
    plan = _plan("filesystem", approved, proposed)

    def _failing_start(config, **kw):
        raise McpError("MCP_STARTUP_FAILED", "forced")

    with pytest.raises(McpError) as exc:
        update_filesystem_access(plan, base_dir=paths["base_dir"], managed_root=paths["managed_root"],
                                 start_server_fn=_failing_start)
    assert exc.value.code == MCP_FILESYSTEM_ACCESS_VALIDATION_FAILED
    with open(config_path, "r", encoding="utf-8") as f:
        after = f.read()
    assert after == before


def test_tools_list_failure_leaves_the_previous_config_untouched(tmp_path):
    paths, approved, config_path = _setup(tmp_path)
    with open(config_path, "r", encoding="utf-8") as f:
        before = f.read()
    new_dir = os.path.realpath(str(tmp_path / "root_b"))
    os.makedirs(new_dir, exist_ok=True)
    proposed = tuple(sorted(approved + [new_dir]))
    plan = _plan("filesystem", approved, proposed)

    class _BrokenListToolsClient(_FakeClient):
        def list_tools(self, timeout=None):
            raise McpError("MCP_DISCOVERY_FAILED", "forced")

    client = _BrokenListToolsClient()
    with pytest.raises(McpError) as exc:
        update_filesystem_access(plan, base_dir=paths["base_dir"], managed_root=paths["managed_root"],
                                 start_server_fn=lambda config, **kw: client)
    assert exc.value.code == MCP_FILESYSTEM_ACCESS_VALIDATION_FAILED
    with open(config_path, "r", encoding="utf-8") as f:
        after = f.read()
    assert after == before
    assert client.shutdown_called is True


def test_not_installed_server_is_reported(tmp_path):
    paths = manager_paths(tmp_path)
    plan = _plan("filesystem", (), ("x",))
    with pytest.raises(McpError) as exc:
        update_filesystem_access(plan, base_dir=paths["base_dir"], managed_root=paths["managed_root"])
    assert exc.value.code == "MCP_FILESYSTEM_ACCESS_NOT_INSTALLED"
