"""Phase G.3 Task 20 (scenario D) — a catalog change invalidates a prepared plan.

Complements tests/test_generalized_mcp_provisioning_models.py (which hand-builds
plan variants) by building plans through `build_auto_plan` from real, slightly
modified catalog entries — proving the hash actually reacts to what a catalog
edit changes, not just to hand-constructed field diffs.
"""

import copy
import os

import pytest

from mcp_management.auto_provisioning import build_auto_plan
from mcp_management.catalog import build_catalog
from mcp_management.document_authorization import DocumentInputSnapshot
from tests.auto_provisioning_helpers import calculator_test_catalog_entry_raw
from tools.models import ToolPermission


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


def _snapshot(path: str) -> DocumentInputSnapshot:
    import hashlib

    with open(path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    return DocumentInputSnapshot(
        source_uri=path,
        local_path=path,
        size_bytes=os.path.getsize(path),
        ctime_ns=int(os.stat(path).st_ctime_ns),
        sha256=sha256,
        permission=ToolPermission.READ,
    )


def test_document_snapshots_included_in_plan_hash(tmp_path):
    base = calculator_test_catalog_entry_raw()
    catalog = build_catalog({"catalog_version": 1, "servers": {"calculator-test": base}})
    entry = catalog.get("calculator-test")

    p1 = _plan_from(base, base_dir=str(tmp_path))
    assert p1.document_snapshots == ()

    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4 fake fixture")
    snap = _snapshot(str(doc))
    p2 = build_auto_plan(entry, "autoreq_1", "summarize report.pdf",
                         base_dir=str(tmp_path), document_snapshots=(snap,))

    assert p2.document_snapshots == (snap,)
    assert p1.plan_hash != p2.plan_hash


def test_document_snapshots_revalidation_affects_hash(tmp_path):
    """Changing the file content between plan builds must change the snapshot
    fingerprint and therefore the plan hash."""
    base = calculator_test_catalog_entry_raw()
    catalog = build_catalog({"catalog_version": 1, "servers": {"calculator-test": base}})
    entry = catalog.get("calculator-test")

    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4 version one")
    snap1 = _snapshot(str(doc))
    p1 = build_auto_plan(entry, "autoreq_1", "summarize report.pdf",
                         base_dir=str(tmp_path), document_snapshots=(snap1,))

    doc.write_bytes(b"%PDF-1.4 version two")
    snap2 = _snapshot(str(doc))
    p2 = build_auto_plan(entry, "autoreq_1", "summarize report.pdf",
                         base_dir=str(tmp_path), document_snapshots=(snap2,))

    assert p1.plan_hash != p2.plan_hash
