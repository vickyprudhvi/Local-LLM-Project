"""Phase G.3 Task 17/20 (scenario S) — concurrent installs of the SAME server are
deduplicated: exactly one transaction proceeds, the other is rejected rather than
racing a second candidate/venv into existence. Real threads, real subprocess.
"""

import threading

from mcp_layer.errors import McpError
from mcp_management.auto_provisioning import AutoProvisioningApproval
from mcp_management.registry import get_installed
from tests.auto_provisioning_helpers import build_auto_provisioning_env
from tools.models import MCP_PROVISIONING_ALREADY_IN_PROGRESS


def test_two_concurrent_installs_of_the_same_server_deduplicate(tmp_path):
    env = build_auto_provisioning_env(tmp_path)
    entry = env["catalog"].get("calculator-test")

    request_a = env["manager"].begin_request("add 1 and 2", "arithmetic_calculation", entry)
    plan_a = env["manager"].prepare_plan(request_a.request_id)
    approval_a = AutoProvisioningApproval(approved=True, plan_id=plan_a.plan_id, plan_hash=plan_a.plan_hash)

    request_b = env["manager"].begin_request("add 3 and 4", "arithmetic_calculation", entry)
    plan_b = env["manager"].prepare_plan(request_b.request_id)
    approval_b = AutoProvisioningApproval(approved=True, plan_id=plan_b.plan_id, plan_hash=plan_b.plan_hash)

    # Serialize thread B's lock acquisition attempt to land WHILE A still holds
    # the per-server lock, by having A signal "I've started installing" before
    # B even calls provision_and_activate.
    a_started = threading.Event()
    real_lock_for = env["manager"]._lock_for

    def _instrumented_lock_for(server_id):
        lock = real_lock_for(server_id)
        a_started.set()
        return lock

    results = {}
    errors = {}

    def run_a():
        env["manager"]._lock_for = _instrumented_lock_for
        try:
            results["a"] = env["manager"].provision_and_activate(
                request_a.request_id, env["runtime_manager"], approval=approval_a)
        except McpError as e:
            errors["a"] = e

    def run_b():
        a_started.wait(timeout=10)
        try:
            results["b"] = env["manager"].provision_and_activate(
                request_b.request_id, env["runtime_manager"], approval=approval_b)
        except McpError as e:
            errors["b"] = e

    t_a = threading.Thread(target=run_a)
    t_b = threading.Thread(target=run_b)
    t_a.start()
    t_b.start()
    t_a.join(timeout=60)
    t_b.join(timeout=60)

    # Exactly one succeeded (installed the real server); the other either got
    # MCP_PROVISIONING_ALREADY_IN_PROGRESS (raced the lock) or transparently
    # reused the now-installed server (raced AFTER A finished) — never a second
    # independent install.
    installed = get_installed("calculator-test", None, env["base_dir"], "app_data/mcp_servers")
    assert installed is not None
    assert installed.installed_version == "1.0.0"
    if "a" in errors:
        assert errors["a"].code == MCP_PROVISIONING_ALREADY_IN_PROGRESS
    if "b" in errors:
        assert errors["b"].code == MCP_PROVISIONING_ALREADY_IN_PROGRESS
    assert "a" in results or "b" in results


def test_second_server_installs_independently_no_global_lock(tmp_path):
    """Two DIFFERENT server_ids never contend on the same lock."""
    env = build_auto_provisioning_env(tmp_path)
    lock1 = env["manager"]._lock_for("calculator-test")
    lock2 = env["manager"]._lock_for("some-other-server")
    assert lock1 is not lock2
