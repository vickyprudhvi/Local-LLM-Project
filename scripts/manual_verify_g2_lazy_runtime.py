"""Phase G.2 Task 19 — real-process lazy-activation verification.

Drives the REAL, already-installed `@modelcontextprotocol/server-filesystem`
package (no reinstall, no network) through `MultiMcpRuntimeManager` exactly as
`assistant.py` does: build the runtime manager with zero MCP processes, activate
Filesystem lazily on the first selected request, reuse the session on a second
request, then stop it and confirm no orphan process and no lingering remote
tools (built-ins survive).

Isolated under a temp base_dir/managed_root — never touches the real
app_data/mcp_servers/ state.

Run: venv/Scripts/python.exe scripts/manual_verify_g2_lazy_runtime.py
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_layer.runtime_manager import MultiMcpRuntimeManager, RuntimeState
from mcp_management.filesystem_access_tools import register_filesystem_access_tools
from mcp_management.manager import McpProvisioningManager
from mcp_management.registry import STATUS_INSTALLED, InstalledServer, upsert
from tests.mcp_provisioning_helpers import make_catalog
from tools.executor import ToolExecutor
from tools.models import ToolCall
from tools.registry import default_registry

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REAL_ENTRYPOINT = os.path.join(
    _REPO_ROOT, "app_data", "mcp_servers", "filesystem", "versions", "2026.7.10",
    "node_modules", "@modelcontextprotocol", "server-filesystem", "dist", "index.js")


def _ok(label):
    print(f"[OK] {label}")


def main():
    if not os.path.isfile(_REAL_ENTRYPOINT):
        print(f"[SKIP] real filesystem MCP package not found at {_REAL_ENTRYPOINT}; "
              "run the normal provisioning flow first.")
        return 1

    tmp_root = tempfile.mkdtemp(prefix="g2_lazy_runtime_manual_")
    try:
        managed_root = "app_data/mcp_servers"
        approved_root = os.path.join(tmp_root, "approved")
        os.makedirs(approved_root, exist_ok=True)
        with open(os.path.join(approved_root, "hello.txt"), "w", encoding="utf-8") as f:
            f.write("Phase G.2 lazy runtime verification passed.")

        server_root = os.path.join(tmp_root, managed_root, "filesystem")
        os.makedirs(server_root, exist_ok=True)
        # Must live under <base_dir>/mcp_workspaces/... — that's the approved
        # sandbox root mcp_layer.external.start_server enforces working
        # directories against (independent of managed_root).
        workspace = os.path.join(tmp_root, "mcp_workspaces", "filesystem")
        os.makedirs(workspace, exist_ok=True)
        raw = {
            "enabled": True, "required": False, "server_id": "filesystem", "transport": "stdio",
            "command": shutil.which("node") or "node", "args": [_REAL_ENTRYPOINT, approved_root],
            "working_directory": workspace,
            "startup_timeout_seconds": 15, "call_timeout_seconds": 15, "shutdown_timeout_seconds": 5,
            "environment_allowlist": [],
            "tool_policy": {"default_permission": "denied", "tools": {
                "list_allowed_directories": {"enabled": True, "permission": "read"},
                "read_text_file": {"enabled": True, "permission": "read"},
            }},
        }
        config_path = os.path.join(server_root, "server.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        upsert("filesystem", InstalledServer(
            catalog_id="official-filesystem", installed_version="2026.7.10", status=STATUS_INSTALLED,
            install_directory=server_root, configuration_path=config_path, installed_at="now",
            approved_directories=(os.path.realpath(approved_root),)), None, tmp_root, managed_root)
        _ok(f"Filesystem MCP config written (real npm package, no reinstall); root: {approved_root}")

        reg = default_registry()
        manager = McpProvisioningManager(catalog=make_catalog(), base_dir=tmp_root, managed_root=managed_root)
        register_filesystem_access_tools(reg, manager)
        rm = MultiMcpRuntimeManager(reg, base_dir=tmp_root, managed_root=managed_root)

        # 1-3: build the runtime — zero process, zero remote tools.
        assert rm.get_session("filesystem") is None
        assert not any(name.startswith("mcp.filesystem.") and "access" not in name for name in reg._tools)
        _ok("runtime manager built: no Filesystem process, no remote tools registered")

        # 4-7: lazy activation on the first selected request.
        executor = ToolExecutor(reg)
        session = rm.ensure_started("filesystem",
                                    expected_allowed_roots=(os.path.realpath(approved_root),))
        first_pid = session.client._proc.pid
        assert session.health.state.value == "healthy"
        assert rm.get_status("filesystem").state == RuntimeState.HEALTHY
        assert reg.has("mcp.filesystem.read_text_file")
        _ok(f"Filesystem started lazily (PID {first_pid}); session HEALTHY; remote tools registered")

        # 8: read the approved test file through the real pipeline.
        call = ToolCall(call_id="c1", tool_name="mcp.filesystem.read_text_file",
                        arguments={"path": os.path.join(approved_root, "hello.txt")})
        result = executor.execute(call)
        assert result.success, result.error
        assert "Phase G.2 lazy runtime verification passed." in result.data.get("content", "")
        _ok(f"read succeeded through the real pipeline: {result.data}")

        # 9-10: a second request reuses the same session/PID.
        second_session = rm.ensure_started("filesystem",
                                           expected_allowed_roots=(os.path.realpath(approved_root),))
        assert second_session is session
        assert second_session.client._proc.pid == first_pid
        _ok(f"second request reused the SAME session (PID {first_pid} unchanged)")

        # 11-15: stop, confirm exit, tools unregistered, built-ins remain, no orphan.
        proc = session.client._proc  # capture now: stop() clears client._proc
        rm.stop("filesystem")
        assert proc.poll() is not None
        assert not reg.has("mcp.filesystem.read_text_file")
        for name in ("mcp.filesystem.access.list", "mcp.filesystem.access.plan",
                    "mcp.filesystem.access.add", "mcp.filesystem.access.remove"):
            assert reg.has(name)
        _ok("Filesystem stopped: process exited, remote tools unregistered, built-ins remain")

        print("\nAll Phase G.2 lazy-runtime manual verification steps passed.")
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
