"""Phase G.3 Task 21 — real-process auto-provisioning verification.

Drives the REAL calculator-test fixture (a genuine, offline, locally-built
Python wheel — no network) through the full G.3 pipeline exactly as
assistant.py does: capability selection finds it SELECTED but not installed,
a deterministic plan is prepared and shown, approval installs it into an
isolated candidate venv, the real candidate process is validated, the install
is atomically activated, Phase G.2 lazily starts the real production runtime,
the original request resumes, and the actual `add` tool computes a real
result. A second, unrelated, already-HEALTHY Filesystem-shaped server is kept
running throughout to prove failure/success isolation between servers.

Isolated under a temp base_dir/managed_root — never touches the real
app_data/mcp_servers/ state or config/mcp_catalog.json.

Run: venv/Scripts/python.exe scripts/manual_verify_g3_auto_provisioning.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_layer.runtime_manager import MultiMcpRuntimeManager, RuntimeState  # noqa: E402
from mcp_management.auto_provisioning import AutoProvisioningManager  # noqa: E402
from mcp_management.provisioning_models import AutoProvisioningApproval  # noqa: E402
from mcp_management.registry import get_installed  # noqa: E402
from tests.auto_provisioning_helpers import catalog_with_calculator_test  # noqa: E402
from tests.mcp_multi_runtime_helpers import node_available, write_fixture_server_config  # noqa: E402
from tools.executor import ToolExecutor  # noqa: E402
from tools.models import ToolCall  # noqa: E402
from tools.registry import default_registry  # noqa: E402


def _ok(label):
    print(f"[OK] {label}")


def main():
    tmp_root = tempfile.mkdtemp(prefix="g3_auto_provisioning_manual_")
    try:
        managed_root = "app_data/mcp_servers"
        extra = {}
        if node_available():
            other_root = os.path.join(tmp_root, "other_root")
            os.makedirs(other_root, exist_ok=True)
        catalog = catalog_with_calculator_test()
        reg = default_registry()
        manager = AutoProvisioningManager(catalog, base_dir=tmp_root, managed_root=managed_root)
        runtime_manager = MultiMcpRuntimeManager(reg, base_dir=tmp_root, managed_root=managed_root)
        _ok(f"catalog + registries isolated under {tmp_root}")

        # 1: start with calculator-test approved but not installed, zero processes.
        assert get_installed("calculator-test", None, tmp_root, managed_root) is None
        assert runtime_manager.get_session("calculator-test") is None
        _ok("baseline: calculator-test approved, not installed, zero processes")

        other_session = None
        if node_available():
            write_fixture_server_config({"base_dir": tmp_root, "managed_root": managed_root},
                                        "document-test", [other_root])
            other_session = runtime_manager.ensure_started(
                "document-test", expected_allowed_roots=(os.path.realpath(other_root),))
            assert other_session.health.state.value == "healthy"
            _ok(f"unrelated second server 'document-test' already HEALTHY (PID "
               f"{other_session.client._proc.pid}) — isolation witness")

        # 2-3: a request requiring arithmetic_calculation selects calculator-test
        # and prepares a plan — with NO install yet.
        entry = catalog.get("calculator-test")
        user_text = "add 10 and 20 using the calculation capability"
        request = manager.begin_request(user_text, "arithmetic_calculation", entry)
        assert request is not None
        plan = manager.prepare_plan(request.request_id)
        assert get_installed("calculator-test", None, tmp_root, managed_root) is None
        print("\n".join(plan.summary_lines()))
        _ok("provisioning plan prepared; nothing installed before approval")

        # 4-5: approve -> install -> validate -> activate -> Phase G.2 healthy.
        approval = AutoProvisioningApproval(approved=True, plan_id=plan.plan_id, plan_hash=plan.plan_hash)
        result = manager.provision_and_activate(request.request_id, runtime_manager, approval=approval)
        assert result.installed_version == "1.0.0"
        installed = get_installed("calculator-test", None, tmp_root, managed_root)
        assert installed is not None and installed.installer_type == "python_venv"
        status = runtime_manager.get_status("calculator-test")
        assert status.state == RuntimeState.HEALTHY
        _ok(f"installed + validated + activated; runtime {status.state.value}, "
           f"{status.registered_tool_count} tool(s) registered")

        # 6-7: original request resumes; the real tool computes a real result.
        resumed_text = manager.resume(request.request_id)
        assert resumed_text == user_text
        executor = ToolExecutor(reg)
        call_result = executor.execute(ToolCall(call_id="c1", tool_name="mcp.calculator-test.add",
                                                arguments={"a": 10, "b": 20}))
        assert call_result.success and call_result.data["result"] == 30
        _ok(f"original request resumed; mcp.calculator-test.add -> {call_result.data}")

        # 8: Filesystem-shaped second server unaffected throughout.
        if other_session is not None:
            assert other_session.client._proc.poll() is None
            assert runtime_manager.get_session("document-test") is other_session
            _ok("unrelated second server: still the SAME process, unaffected")

        # 9-10: a second arithmetic request reuses the installed server, no reinstall.
        request2 = manager.begin_request("add 1 and 2", "arithmetic_calculation", entry)
        plan2 = manager.prepare_plan(request2.request_id)
        approval2 = AutoProvisioningApproval(approved=True, plan_id=plan2.plan_id, plan_hash=plan2.plan_hash)
        session_before = runtime_manager.get_session("calculator-test")
        result2 = manager.provision_and_activate(request2.request_id, runtime_manager, approval=approval2)
        assert result2.installed_version == "1.0.0"
        assert runtime_manager.get_session("calculator-test") is session_before
        _ok("second request reused the installed server; no reinstall, same runtime")

        # 11-12: stop everything, confirm no orphan, tools removed, built-ins remain.
        # Capture PIDs BEFORE stop_all() — McpClient.shutdown() clears client._proc.
        proc = session_before.client._proc
        other_proc = other_session.client._proc if other_session is not None else None
        runtime_manager.stop_all()
        assert proc.poll() is not None
        assert not reg.has("mcp.calculator-test.add")
        assert reg.has("math.calculate")  # a built-in survives
        _ok("stopped: process exited, remote tools unregistered, built-ins remain")
        if other_proc is not None:
            assert other_proc.poll() is not None

        print("\nAll Phase G.3 auto-provisioning manual verification steps passed.")
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
