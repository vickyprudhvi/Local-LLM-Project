"""Shared Phase F test helpers (not a test module).

`FakeNpm` stands in for npm: it records the argv it was given and materializes the
package layout the catalog entry declares, using the real Node fixture server.
Installation and validation tests therefore drive a genuine child process with no
network access.
"""

import json
import os
import shutil

from mcp_management.catalog import build_catalog

FIXTURE_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "fixtures", "fake_filesystem_server.js")
PINNED_VERSION = "2026.7.10"
PACKAGE_NAME = "@modelcontextprotocol/server-filesystem"
ENTRYPOINT_RELATIVE = f"node_modules/{PACKAGE_NAME}/dist/index.js"

DEFAULT_POLICY = {
    "default_permission": "denied",
    "tools": {
        "list_allowed_directories": {"enabled": True, "permission": "read"},
        "list_directory": {"enabled": True, "permission": "read"},
        "read_text_file": {"enabled": True, "permission": "read"},
        "get_file_info": {"enabled": True, "permission": "read"},
        "search_files": {"enabled": True, "permission": "read"},
        "write_file": {"enabled": True, "permission": "write"},
        "move_file": {"enabled": False, "permission": "denied"},
        "edit_file": {"enabled": False, "permission": "denied"},
    },
}


def catalog_dict(version=PINNED_VERSION, package=PACKAGE_NAME, tools=None,
                 allow_lifecycle_scripts=None, capabilities=None, transport="stdio",
                 installer_type="npm"):
    """A valid single-entry catalog document (override pieces per test).

    `allow_lifecycle_scripts` is emitted only when explicitly set, so tests can
    build a catalog that REQUESTS lifecycle scripts and assert it is rejected.
    """
    installer = {
        "type": installer_type,
        "package": package,
        "version": version,
        "entrypoint": ENTRYPOINT_RELATIVE,
    }
    if allow_lifecycle_scripts is not None:
        installer["allow_lifecycle_scripts"] = allow_lifecycle_scripts
    return {
        "catalog_version": 1,
        "servers": {
            "official-filesystem": {
                "server_id": "filesystem",
                "display_name": "Filesystem MCP Server",
                "description": "Read and modify files inside explicitly approved directories.",
                "capabilities": capabilities or ["filesystem", "read_files", "write_files"],
                "risk_category": "local_filesystem",
                "transport": transport,
                "required_runtimes": ["node", "npm"],
                "installer": installer,
                "required_inputs": [
                    {"name": "allowed_directory", "type": "directory",
                     "required": True, "user_approval_required": True},
                ],
                "expected_tools": ["list_allowed_directories", "list_directory", "read_text_file"],
                "default_tool_policy": tools if tools is not None else DEFAULT_POLICY,
            }
        },
    }


def make_catalog(**kwargs):
    return build_catalog(catalog_dict(**kwargs))


class FakeNpm:
    """Records install calls and materializes the package from the Node fixture."""

    def __init__(self, fail=False, returncode=1, stdout="", stderr="", timeout=False):
        self.calls = []
        self.fail = fail
        self.returncode = returncode
        self.stdout = stdout or "added 1 package"
        self.stderr = stderr
        self.timeout = timeout

    def __call__(self, argv, cwd, env, timeout):
        self.calls.append({"argv": list(argv), "cwd": cwd, "env": dict(env or {}),
                           "timeout": timeout})
        if self.timeout:
            import subprocess
            raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)
        if self.fail:
            return self.returncode, self.stdout, self.stderr or "npm ERR! install failed"

        pkg_dir = os.path.join(cwd, "node_modules", *PACKAGE_NAME.split("/"), "dist")
        os.makedirs(pkg_dir, exist_ok=True)
        shutil.copyfile(FIXTURE_SERVER, os.path.join(pkg_dir, "index.js"))
        manifest_dir = os.path.dirname(pkg_dir)
        with open(os.path.join(manifest_dir, "package.json"), "w", encoding="utf-8") as f:
            json.dump({"name": PACKAGE_NAME, "version": PINNED_VERSION,
                       "bin": {"mcp-server-filesystem": "dist/index.js"}}, f)
        with open(os.path.join(cwd, "package-lock.json"), "w", encoding="utf-8") as f:
            json.dump({"lockfileVersion": 3, "packages": {}}, f)
        return 0, self.stdout, self.stderr


def node_available():
    return bool(shutil.which("node")) and bool(shutil.which("npm"))


def workspace_with_file(tmp_path, name="hello.txt",
                        content="Hello from automatic MCP provisioning!"):
    """A disposable approved directory containing one seed file."""
    workspace = tmp_path / "user_files"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / name).write_text(content, encoding="utf-8")
    return str(workspace)


def manager_paths(tmp_path):
    """Isolated base_dir / managed_root / template paths for a manager under test."""
    base = tmp_path / "repo"
    (base / "mcp_workspaces").mkdir(parents=True, exist_ok=True)
    (base / "config").mkdir(parents=True, exist_ok=True)
    return {
        "base_dir": str(base),
        "managed_root": "app_data/mcp_servers",
        # The committed portable template. Phase F must never write this.
        "template_path": "config/mcp_server.json",
    }


def managed_config_file(paths, server_id="filesystem"):
    """Path of the generated managed configuration Phase F activates."""
    return os.path.join(paths["base_dir"], paths["managed_root"], server_id, "server.json")


def write_template(paths, enabled=False, server_id="filesystem"):
    """Write a portable disabled template, as the repository ships."""
    target = os.path.join(paths["base_dir"], paths["template_path"])
    os.makedirs(os.path.dirname(target), exist_ok=True)
    document = {
        "enabled": enabled, "required": False, "server_id": server_id,
        "display_name": "Filesystem MCP Server", "transport": "stdio",
        "command": "node", "args": [],
        "working_directory": "./mcp_workspaces/filesystem",
        "startup_timeout_seconds": 15, "call_timeout_seconds": 15,
        "shutdown_timeout_seconds": 5, "environment_allowlist": [],
        "tool_policy": {"default_permission": "denied", "tools": {}},
    }
    with open(target, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
    return target


def make_manager(tmp_path, catalog=None, **catalog_kwargs):
    """An McpProvisioningManager wired to isolated paths."""
    from mcp_management.manager import McpProvisioningManager

    paths = manager_paths(tmp_path)
    return McpProvisioningManager(
        catalog=catalog if catalog is not None else make_catalog(**catalog_kwargs),
        base_dir=paths["base_dir"],
        managed_root=paths["managed_root"],
    ), paths
