"""Phase G.4 — document snapshots travel through the auto-provisioning plan
and become single-use authorizations on resumption.
"""

import os

import pytest

from mcp_management.auto_provisioning import AutoProvisioningManager, build_auto_plan
from mcp_management.catalog import build_catalog
from mcp_management.document_authorization import (
    DocumentAuthorizationStore,
    DocumentInputSnapshot,
)
from tests.auto_provisioning_helpers import calculator_test_catalog_entry_raw
from tools.models import ToolPermission


@pytest.fixture(autouse=True)
def _reset_default_store():
    original = DocumentAuthorizationStore._default
    DocumentAuthorizationStore._default = DocumentAuthorizationStore()
    yield
    DocumentAuthorizationStore._default = original


@pytest.fixture
def env(tmp_path):
    raw = calculator_test_catalog_entry_raw()
    catalog = build_catalog({"catalog_version": 1, "servers": {"calculator-test": raw}})
    manager = AutoProvisioningManager(catalog, base_dir=str(tmp_path),
                                      managed_root="app_data/mcp_servers")
    return {"manager": manager, "base_dir": str(tmp_path), "catalog": catalog}


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


def test_begin_request_carries_document_snapshots(env, tmp_path):
    manager = env["manager"]
    entry = env["catalog"].get("calculator-test")
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4 fixture")
    snap = _snapshot(str(doc))

    request = manager.begin_request("summarize report.pdf", "document_to_markdown", entry,
                                    document_snapshots=(snap,))
    assert request is not None
    assert request.document_snapshots == (snap,)


def test_plan_hash_includes_document_snapshots(env, tmp_path):
    manager = env["manager"]
    entry = env["catalog"].get("calculator-test")
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4 fixture")
    snap = _snapshot(str(doc))

    request = manager.begin_request("summarize report.pdf", "document_to_markdown", entry,
                                    document_snapshots=(snap,))
    plan = manager.prepare_plan(request.request_id)
    assert plan.document_snapshots == (snap,)
    assert plan.plan_hash != build_auto_plan(
        entry, request.request_id, "summarize report.pdf",
        base_dir=env["base_dir"]).plan_hash


def test_resume_creates_document_authorizations(env, tmp_path):
    manager = env["manager"]
    entry = env["catalog"].get("calculator-test")
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4 fixture")
    snap = _snapshot(str(doc))

    request = manager.begin_request("summarize report.pdf", "document_to_markdown", entry,
                                    document_snapshots=(snap,))
    plan = manager.prepare_plan(request.request_id)
    # Fake the request into the READY state without running the real install.
    from mcp_management.provisioning_models import PendingAutoProvisioningState
    manager._pending[request.request_id] = request.advanced(
        PendingAutoProvisioningState.READY, plan_id=plan.plan_id)

    resumed = manager.resume(request.request_id)
    assert resumed == request.original_user_text

    auth = DocumentAuthorizationStore.default().find_and_reserve_for_path(str(doc))
    assert auth is not None
    assert auth.snapshot.local_path == str(doc)
    assert not auth.consumed


def test_resume_fails_when_document_no_longer_exists(env, tmp_path):
    manager = env["manager"]
    entry = env["catalog"].get("calculator-test")
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4 fixture")
    snap = _snapshot(str(doc))
    # Record the snapshot, then delete the file before resumption.
    snap_stale = DocumentInputSnapshot(
        source_uri=str(doc),
        local_path=str(doc),
        size_bytes=snap.size_bytes,
        ctime_ns=snap.ctime_ns,
        permission=ToolPermission.READ,
    )

    request = manager.begin_request("summarize report.pdf", "document_to_markdown", entry,
                                    document_snapshots=(snap_stale,))
    plan = manager.prepare_plan(request.request_id)
    from mcp_management.provisioning_models import PendingAutoProvisioningState
    manager._pending[request.request_id] = request.advanced(
        PendingAutoProvisioningState.READY, plan_id=plan.plan_id)

    doc.unlink()
    from mcp_layer.errors import McpError
    with pytest.raises(McpError):
        manager.resume(request.request_id)
