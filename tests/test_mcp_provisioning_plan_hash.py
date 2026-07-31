"""Phase G.3 Task 20 (scenario D) — a catalog change invalidates a prepared plan.

Complements tests/test_generalized_mcp_provisioning_models.py (which hand-builds
plan variants) by building plans through `build_auto_plan` from real, slightly
modified catalog entries — proving the hash actually reacts to what a catalog
edit changes, not just to hand-constructed field diffs.
"""

import copy

from mcp_management.auto_provisioning import build_auto_plan
from mcp_management.catalog import build_catalog
from tests.auto_provisioning_helpers import calculator_test_catalog_entry_raw


def _plan_from(raw_entry, base_dir="/tmp/g3"):
    catalog = build_catalog({"catalog_version": 1, "servers": {"calculator-test": raw_entry}})
    entry = catalog.get("calculator-test")
    return build_auto_plan(entry, "autoreq_1", "add 1 and 2", base_dir=base_dir)


def test_version_change_invalidates_hash():
    base = calculator_test_catalog_entry_raw()
    bumped = copy.deepcopy(base)
    bumped["installer"]["version"] = "1.0.1"
    p1, p2 = _plan_from(base), _plan_from(bumped)
    assert p1.plan_hash != p2.plan_hash


def test_package_name_change_invalidates_hash():
    base = calculator_test_catalog_entry_raw()
    changed = copy.deepcopy(base)
    changed["installer"]["package"] = "totally-different-package"
    p1, p2 = _plan_from(base), _plan_from(changed)
    assert p1.plan_hash != p2.plan_hash


def test_tool_policy_change_invalidates_hash():
    base = calculator_test_catalog_entry_raw()
    changed = copy.deepcopy(base)
    changed["default_tool_policy"]["tools"]["echo"]["permission"] = "write"
    p1, p2 = _plan_from(base), _plan_from(changed)
    assert p1.plan_hash != p2.plan_hash


def test_expected_tools_change_invalidates_hash():
    base = calculator_test_catalog_entry_raw()
    changed = copy.deepcopy(base)
    changed["expected_tools"] = ["add", "echo", "subtract"]
    changed["default_tool_policy"]["tools"]["subtract"] = {"enabled": True, "permission": "read"}
    p1, p2 = _plan_from(base), _plan_from(changed)
    assert p1.plan_hash != p2.plan_hash


def test_launch_module_change_invalidates_hash():
    base = calculator_test_catalog_entry_raw()
    changed = copy.deepcopy(base)
    changed["launch"]["module"] = "calculator_test_mcp.other_server"
    p1, p2 = _plan_from(base), _plan_from(changed)
    assert p1.plan_hash != p2.plan_hash


def test_unchanged_catalog_reproduces_identical_hash():
    base = calculator_test_catalog_entry_raw()
    p1, p2 = _plan_from(base), _plan_from(copy.deepcopy(base))
    assert p1.plan_hash == p2.plan_hash


def test_install_directory_differs_by_base_dir():
    base = calculator_test_catalog_entry_raw()
    p1 = _plan_from(base, base_dir="/tmp/g3_one")
    p2 = _plan_from(base, base_dir="/tmp/g3_two")
    assert p1.plan_hash != p2.plan_hash
