"""Phase F.1 hotfix Task 14 — real-process runtime-restart verification.

Drives the REAL code path end to end: the actual already-installed
`@modelcontextprotocol/server-filesystem` npm package (no reinstall, no network),
real Node child processes, the real ToolExecutor/McpTool/ToolRegistry pipeline, and
`assistant._run_local_turn` / `assistant._restart_mcp_and_resume` exactly as
`python assistant.py` would call them for a "local" turn — with a scripted FakeLLM
standing in for Ollama so the scenario is deterministic, and everything isolated
under a temp base_dir/managed_root so the REAL app_data/mcp_servers state is never
touched.

Scenario (mirrors the bug report):
  1. Start with exactly one approved root.
  2. Ask to read a file in a second, NOT-yet-approved root.
  3. The model calls read_text_file; it fails outside-root; the tool loop halts
     BEFORE a second, generic LLM answer — no rephrase needed.
  4. Approve the plan ("yes" — the synchronous Phase F.1 approval mechanism).
  5. The MCP runtime is replaced: old process stops, old remote tools are
     unregistered, a NEW process starts, its live `list_allowed_directories` is
     verified against the expected two roots, and only THEN is the original
     request resumed through the real router / shortlist / executor / McpTool
     pipeline (no manual restart, no npm call, no sleep/retry loop).

Run: venv/Scripts/python.exe scripts/manual_verify_f1_restart.py
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assistant
import confirmation
import tool_loop
from mcp_layer.external import bootstrap_from_config
from mcp_layer.runtime_manager import ActiveMcpRuntime
from mcp_management.filesystem_access_tools import register_filesystem_access_tools
from mcp_management.manager import McpProvisioningManager
from mcp_management.registry import STATUS_INSTALLED, InstalledServer, upsert
from router import RouteDecision
from tests.mcp_provisioning_helpers import make_catalog
from tests.test_tool_loop import FakeLLM, _final, _tool_call
from tools.executor import ToolExecutor
from tools.registry import default_registry

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REAL_ENTRYPOINT = os.path.join(
    _REPO_ROOT, "app_data", "mcp_servers", "filesystem", "versions", "2026.7.10",
    "node_modules", "@modelcontextprotocol", "server-filesystem", "dist", "index.js")


def _ok(label):
    print(f"[OK] {label}")


def _write_config(base_dir, managed_root, roots, workspace):
    server_root = os.path.join(base_dir, managed_root, "filesystem")
    os.makedirs(server_root, exist_ok=True)
    config_path = os.path.join(server_root, "server.json")
    raw = {
        "enabled": True, "required": False, "server_id": "filesystem",
        "display_name": "Filesystem MCP Server", "transport": "stdio",
        "command": shutil.which("node") or "node", "args": [_REAL_ENTRYPOINT, *roots],
        "working_directory": workspace,
        "startup_timeout_seconds": 15, "call_timeout_seconds": 15,
        "shutdown_timeout_seconds": 5, "environment_allowlist": [],
        "tool_policy": {"default_permission": "denied", "tools": {
            "list_allowed_directories": {"enabled": True, "permission": "read"},
            "read_text_file": {"enabled": True, "permission": "read"},
            "list_directory": {"enabled": True, "permission": "read"},
        }},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    upsert("filesystem", InstalledServer(
        catalog_id="official-filesystem", installed_version="2026.7.10",
        status=STATUS_INSTALLED, install_directory=server_root,
        configuration_path=config_path, installed_at="now",
        approved_directories=tuple(roots)), None, base_dir, managed_root)
    return config_path


def main():
    if not os.path.isfile(_REAL_ENTRYPOINT):
        print(f"[SKIP] real filesystem MCP package not found at {_REAL_ENTRYPOINT}; "
              "run the normal provisioning flow first.")
        return 1

    tmp_root = tempfile.mkdtemp(prefix="f1_restart_manual_")
    old_session = None
    try:
        managed_root = "app_data/mcp_servers"
        initial_root = os.path.join(tmp_root, "mcp_workspaces", "filesystem")
        external_root = os.path.join(tmp_root, "f1_external_test")
        os.makedirs(initial_root, exist_ok=True)
        os.makedirs(external_root, exist_ok=True)
        with open(os.path.join(external_root, "hello.txt"), "w", encoding="utf-8") as f:
            f.write("Phase F.1 runtime restart passed.")

        _write_config(tmp_root, managed_root, [initial_root], initial_root)
        _ok(f"Filesystem MCP installed (real npm package, no reinstall); initial root: {initial_root}")

        reg = default_registry()
        tool_loop.REGISTRY = reg
        tool_loop.EXECUTOR = ToolExecutor(reg)
        confirmation.confirm_action = lambda summary: True  # auto-approve write confirmations

        manager = McpProvisioningManager(catalog=make_catalog(), base_dir=tmp_root, managed_root=managed_root)
        register_filesystem_access_tools(reg, manager)
        # This script isolates MCP state under tmp_root; memory recall is unrelated
        # to what's being verified here, so skip it rather than touch the real
        # ChromaDB store.
        assistant._enrich_with_memory = lambda text: text

        from mcp_layer.config import load_config
        config_path = os.path.join(tmp_root, managed_root, "filesystem", "server.json")
        old_session = bootstrap_from_config(reg, config=load_config(config_path), base_dir=tmp_root)
        assert old_session.health.state.value == "healthy"
        old_proc = old_session.client._proc  # captured now: shutdown() clears client._proc
        old_pid = old_proc.pid
        _ok(f"Old MCP server started (PID {old_pid}); root: {initial_root}")

        runtime = ActiveMcpRuntime(old_session)

        target = os.path.join(external_root, "hello.txt")
        user_text = f"read '{target}'"
        fake = FakeLLM([
            _tool_call("mcp.filesystem.read_text_file", {"path": target}),
            _final("SHOULD NEVER BE REACHED — the loop must halt before this."),
        ])
        tool_loop.ask_local_raw = fake

        reply, extra_metrics, pending_id = assistant._run_local_turn(
            manager, runtime, user_text, user_text, [], "sys", set())
        assert len(fake.calls) == 1, "the tool loop must halt before a second, generic LLM call"
        assert pending_id is not None
        assert "SHOULD NEVER" not in reply
        _ok("mcp.filesystem.read_text_file -> failed")
        _ok("filesystem access required; access plan prepared automatically (no rephrase needed)")
        print(reply)

        outcome = assistant._resolve_filesystem_access_reply(manager, pending_id, "yes")
        assert outcome.matched and outcome.resumed_text == user_text
        _ok("access approved")

        directive = tool_loop.ToolLoopDirective(
            control=tool_loop.ToolLoopControl.RESTART_MCP_AND_RESUME,
            server_id=outcome.server_id, expected_allowed_roots=outcome.expected_allowed_roots)

        fake2 = FakeLLM([
            _tool_call("mcp.filesystem.read_text_file", {"path": target}),
            _final("The file contains: Phase F.1 runtime restart passed."),
        ])
        tool_loop.ask_local_raw = fake2
        assistant.route_and_answer = lambda prompt, history: RouteDecision(mode="local", tool=None)

        reply2, pending_id2 = assistant._restart_mcp_and_resume(
            manager, runtime, directive, outcome.resumed_text, [], "sys", set(), resume_budget=1,
            previous_allowed_roots=outcome.previous_allowed_roots)
        _ok("runtime restart requested")

        new_session = runtime.session
        new_pid = new_session.client._proc.pid
        assert old_proc.poll() is not None, "old MCP process did not stop"
        _ok(f"old MCP session stopped (PID {old_pid} exited)")
        assert new_pid != old_pid
        _ok(f"new MCP session HEALTHY (PID {new_pid})")

        old_remote_names = set(old_session.registered_remote_tool_names)
        for name in old_remote_names:
            live = reg.get(name)
            assert live.session_owner == new_session.session_id, "old remote tools unregistered; " \
                "new remote tools bound to the new client"
        _ok("old remote tools unregistered; new remote tools registered and bound to the new client")

        expected_roots = {os.path.realpath(initial_root), os.path.realpath(external_root)}
        assert set(directive.expected_allowed_roots) == expected_roots
        _ok(f"allowed roots verified live: {len(expected_roots)}")

        assert pending_id2 is None
        assert "Phase F.1 runtime restart passed" in reply2
        _ok("original request resumed automatically")
        _ok(f"mcp.filesystem.read_text_file -> ok: {reply2!r}")

        new_proc = new_session.client._proc
        new_session.shutdown()
        assert new_proc.poll() is not None
        _ok("new MCP session shut down cleanly; no orphan process")

        print("\nAll Phase F.1 runtime-restart manual verification steps passed.")
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
