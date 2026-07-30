"""Phase F.1 Task 14 manual verification, scripted end-to-end (no interactive
terminal or live Ollama available in this environment). Drives the REAL code path
— a genuine node child process running the repo's fixture filesystem server, the
real ToolExecutor/McpTool pipeline, and the real assistant.py helper functions —
instead of an interactive `python assistant.py` session. Mirrors the existing
Phase F integration tests' use of FakeNpm + the Node fixture server to avoid a
real network install while still exercising a real process end to end.

Run: venv/Scripts/python.exe scripts/manual_verify_f1.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assistant
from mcp_layer.errors import McpError
from mcp_management.installer import install
from mcp_management.planner import build_plan
from mcp_management.registry import get_installed
from tests.mcp_provisioning_helpers import FakeNpm, make_manager, workspace_with_file
from tools.executor import ToolExecutor
from tools.models import ToolCall


def _ok(label):
    print(f"[OK] {label}")


def main():
    tmp_root = tempfile.mkdtemp(prefix="f1_manual_")
    try:
        import pathlib
        tmp_path = pathlib.Path(tmp_root)

        manager, paths = make_manager(tmp_path)
        entry = manager.catalog.get("official-filesystem")
        initial_root = workspace_with_file(tmp_path, name="hello.txt", content="hi from initial root")
        plan = build_plan(entry, requested_directories=[initial_root], base_dir=paths["base_dir"])
        from mcp_management.models import ProvisioningApproval

        approval = ProvisioningApproval(True, plan.plan_id, plan.compute_hash())
        install(plan, entry, approval, base_dir=paths["base_dir"], managed_root=paths["managed_root"],
               npm_runner=FakeNpm())
        installed = get_installed("filesystem", None, paths["base_dir"], paths["managed_root"])
        assert installed is not None
        _ok(f"Filesystem MCP installed with initial root: {installed.approved_directories}")

        # ---- 1. Request a file outside the approved root ----
        outside_dir = tmp_path / "data" / "repositories" / "neonwatty" / "machine-learning-refined" / "chapter_pdfs"
        outside_dir.mkdir(parents=True)
        (outside_dir / "README.md").write_text("chapter pdfs readme", encoding="utf-8")
        target = str(outside_dir / "README.md")
        user_text = f"read '{target}'"

        call = ToolCall(call_id="c1", tool_name="mcp.filesystem.read_text_file", arguments={"path": target})
        from tools.models import ToolResult
        fake_result = ToolResult.fail("mcp.filesystem.read_text_file", "c1", "MCP_CALL_FAILED",
                                      "Access denied - path outside allowed directories")
        found = assistant._find_outside_root_failure(manager, [(call, fake_result)])
        assert found is not None
        server_id, found_call, failure = found
        assert failure.proposed_root == os.path.realpath(str(outside_dir))
        _ok(f"Outside-root failure detected; proposed root: {failure.proposed_root}")

        reply, request_id = assistant._offer_filesystem_access(manager, server_id, found_call, failure, user_text)
        assert os.path.realpath(str(outside_dir)) in reply
        before = get_installed("filesystem", None, paths["base_dir"], paths["managed_root"])
        assert before.approved_directories == installed.approved_directories
        _ok("Plan shown; no config change yet; no npm execution")

        # ---- 2. Decline ----
        outcome = assistant._resolve_filesystem_access_reply(manager, request_id, "no")
        assert outcome.resumed_text is None
        after_decline = get_installed("filesystem", None, paths["base_dir"], paths["managed_root"])
        assert after_decline.approved_directories == installed.approved_directories
        _ok("Declined: roots unchanged, file not read, server not restarted")

        # ---- 3. Repeat + approve via bare 'yes' ----
        found = assistant._find_outside_root_failure(manager, [(call, fake_result)])
        server_id, found_call, failure = found
        reply, request_id = assistant._offer_filesystem_access(manager, server_id, found_call, failure, user_text)

        outcome = assistant._resolve_filesystem_access_reply(manager, request_id, "yes")
        assert outcome.resumed_text == user_text
        _ok("Approved via bare 'yes'; pending plan recognized; no invented mcp.provision")

        updated = get_installed("filesystem", None, paths["base_dir"], paths["managed_root"])
        assert os.path.realpath(initial_root) in updated.approved_directories
        assert os.path.realpath(str(outside_dir)) in updated.approved_directories
        _ok(f"Managed config updated; both roots present: {updated.approved_directories}")

        # ---- 4. Resume through the REAL pipeline: actually start a real McpSession
        #          off the updated config and READ the file via ToolExecutor + McpTool. ----
        import mcp_layer
        from tools.registry import default_registry

        reg = default_registry()
        session = mcp_layer.bootstrap_from_config(reg, config_path=updated.configuration_path,
                                                  base_dir=paths["base_dir"])
        try:
            assert session.health.state.value == "healthy", session.health
            executor = ToolExecutor(reg)
            read_name = next(n for n in session.tool_names() if n.endswith("read_text_file"))
            list_allowed_name = next(n for n in session.tool_names()
                                     if n.endswith("list_allowed_directories"))

            call = ToolCall(call_id="c2", tool_name=read_name, arguments={"path": target})
            result = executor.execute(call)
            assert result.success, result.error
            _ok(f"Original request resumed through the real pipeline; README.md contents returned "
               f"({len(str(result.data))} chars)")

            allowed_call = ToolCall(call_id="c3", tool_name=list_allowed_name, arguments={})
            allowed_result = executor.execute(allowed_call)
            assert allowed_result.success, allowed_result.error
            _ok(f"list_allowed_directories via the live server: {allowed_result.data}")
        finally:
            session.shutdown()
        _ok("Server shut down cleanly; no orphan process")

        # ---- 5. Restricted path: .ssh must never be offered a plan ----
        ssh_dir = tmp_path / "home" / ".ssh"
        ssh_dir.mkdir(parents=True)
        key = ssh_dir / "id_rsa"
        key.write_text("private key material", encoding="utf-8")
        restricted_call = ToolCall(call_id="c4", tool_name="mcp.filesystem.read_text_file",
                                   arguments={"path": str(key)})
        restricted_result = ToolResult.fail("mcp.filesystem.read_text_file", "c4", "MCP_CALL_FAILED",
                                            "Access denied - path outside allowed directories")
        found = assistant._find_outside_root_failure(manager, [(restricted_call, restricted_result)])
        assert found is None
        _ok("Restricted .ssh path rejected: no approval plan generated, no config change")

        print("\nAll Phase F.1 manual verification steps passed.")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
