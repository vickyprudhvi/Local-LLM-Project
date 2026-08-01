"""Phase G.4 — document authorization, snapshot binding, and exact-file policy.

All tests here are deterministic and use the local filesystem only; no MCP
servers are started.
"""

import os
import tempfile
import threading
import time

import pytest

from mcp_layer.errors import McpError
from mcp_management.document_authorization import (
    DocumentAuthorizationStore,
    DocumentInputSnapshot,
    LocalDocumentExactFilePolicy,
)
from tools.models import (
    MCP_DOCUMENT_AUTHORIZATION_CONSUMED,
    MCP_DOCUMENT_AUTHORIZATION_EXPIRED,
    MCP_DOCUMENT_AUTHORIZATION_RESERVED,
    MCP_DOCUMENT_NOT_FOUND,
    MCP_DOCUMENT_PATH_INVALID,
    MCP_DOCUMENT_PATH_RESTRICTED,
    MCP_DOCUMENT_PERMISSION_DENIED,
    MCP_DOCUMENT_SNAPSHOT_MISMATCH,
    ToolPermission,
)


@pytest.fixture
def exact_file(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("G4_VERIFY_TXT_2026", encoding="utf-8")
    return p


def _read_snapshot(path: str) -> DocumentInputSnapshot:
    stat = os.stat(path)
    return DocumentInputSnapshot(
        source_uri=path,
        local_path=path,
        size_bytes=stat.st_size,
        ctime_ns=stat.st_ctime_ns,
        permission=ToolPermission.READ,
    )


def test_explicit_read_permission_required(tmp_path):
    p = tmp_path / "locked.txt"
    p.write_text("x", encoding="utf-8")
    snapshot = DocumentInputSnapshot(
        source_uri=str(p), local_path=str(p), permission=ToolPermission.DENIED
    )
    policy = LocalDocumentExactFilePolicy()
    with pytest.raises(McpError) as exc:
        policy.validate(snapshot)
    assert exc.value.code == MCP_DOCUMENT_PERMISSION_DENIED


def test_empty_path_rejected():
    snapshot = DocumentInputSnapshot(source_uri="", permission=ToolPermission.READ)
    with pytest.raises(McpError) as exc:
        LocalDocumentExactFilePolicy().validate(snapshot)
    assert exc.value.code == MCP_DOCUMENT_PATH_INVALID


def test_http_url_rejected():
    snapshot = DocumentInputSnapshot(
        source_uri="https://example.com/file.pdf", permission=ToolPermission.READ
    )
    with pytest.raises(McpError) as exc:
        LocalDocumentExactFilePolicy().validate(snapshot)
    assert exc.value.code == MCP_DOCUMENT_PATH_RESTRICTED


def test_file_url_rejected():
    snapshot = DocumentInputSnapshot(
        source_uri="file:///C:/Users/x/file.pdf", permission=ToolPermission.READ
    )
    with pytest.raises(McpError) as exc:
        LocalDocumentExactFilePolicy().validate(snapshot)
    assert exc.value.code == MCP_DOCUMENT_PATH_RESTRICTED


def test_unc_path_rejected():
    snapshot = DocumentInputSnapshot(
        source_uri="\\\\server\\share\\file.pdf", permission=ToolPermission.READ
    )
    with pytest.raises(McpError) as exc:
        LocalDocumentExactFilePolicy().validate(snapshot)
    assert exc.value.code == MCP_DOCUMENT_PATH_RESTRICTED


def test_relative_path_rejected():
    snapshot = DocumentInputSnapshot(
        source_uri="fixtures/markitdown_sample.pdf", permission=ToolPermission.READ
    )
    with pytest.raises(McpError) as exc:
        LocalDocumentExactFilePolicy().validate(snapshot)
    assert exc.value.code == MCP_DOCUMENT_PATH_INVALID


def test_glob_characters_rejected(tmp_path):
    snapshot = DocumentInputSnapshot(
        source_uri=str(tmp_path / "*.txt"), permission=ToolPermission.READ
    )
    with pytest.raises(McpError) as exc:
        LocalDocumentExactFilePolicy().validate(snapshot)
    assert exc.value.code == MCP_DOCUMENT_PATH_RESTRICTED


def test_nonexistent_file_rejected(tmp_path):
    snapshot = DocumentInputSnapshot(
        source_uri=str(tmp_path / "missing.pdf"), permission=ToolPermission.READ
    )
    with pytest.raises(McpError) as exc:
        LocalDocumentExactFilePolicy().validate(snapshot)
    assert exc.value.code == MCP_DOCUMENT_NOT_FOUND


def test_directory_rejected(tmp_path):
    d = tmp_path / "folder"
    d.mkdir()
    snapshot = DocumentInputSnapshot(
        source_uri=str(d), permission=ToolPermission.READ
    )
    with pytest.raises(McpError) as exc:
        LocalDocumentExactFilePolicy().validate(snapshot)
    assert exc.value.code == MCP_DOCUMENT_NOT_FOUND


def test_traversal_components_rejected(tmp_path):
    target = tmp_path / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    # Absolute path containing .. is rejected by exact-file policy.
    tricky = str(tmp_path / "subdir" / ".." / "secret.txt")
    snapshot = DocumentInputSnapshot(source_uri=tricky, permission=ToolPermission.READ)
    with pytest.raises(McpError) as exc:
        LocalDocumentExactFilePolicy().validate(snapshot)
    assert exc.value.code == MCP_DOCUMENT_PATH_INVALID

    # Escape to a sibling directory is also rejected.
    outside = (tmp_path.parent / "outside_g4.txt")
    outside.write_text("x", encoding="utf-8")
    escape = str(tmp_path / ".." / "outside_g4.txt")
    snapshot2 = DocumentInputSnapshot(source_uri=escape, permission=ToolPermission.READ)
    with pytest.raises(McpError) as exc:
        LocalDocumentExactFilePolicy().validate(snapshot2)
    assert exc.value.code == MCP_DOCUMENT_PATH_INVALID


def test_absolute_local_file_accepted(exact_file):
    snapshot = _read_snapshot(str(exact_file))
    bound = LocalDocumentExactFilePolicy().bind(snapshot)
    assert bound.local_path == str(exact_file.resolve())


def test_store_creates_authorization(exact_file):
    store = DocumentAuthorizationStore()
    snapshot = _read_snapshot(str(exact_file))
    auth = store.create_authorization(snapshot)
    assert auth.auth_id.startswith("doc-auth-")
    assert not auth.consumed
    assert auth.snapshot.local_path == str(exact_file.resolve())


def test_reserve_then_consume_sequence(exact_file):
    store = DocumentAuthorizationStore()
    auth = store.create_authorization(_read_snapshot(str(exact_file)))
    reserved = store.reserve_authorization(auth.auth_id)
    assert reserved._reserved
    consumed = store.consume_authorization(auth.auth_id)
    assert consumed.consumed
    assert not consumed._reserved


def test_consumed_authorization_cannot_be_reused(exact_file):
    store = DocumentAuthorizationStore()
    auth = store.create_authorization(_read_snapshot(str(exact_file)))
    store.consume_authorization(auth.auth_id)
    with pytest.raises(McpError) as exc:
        store.reserve_authorization(auth.auth_id)
    assert exc.value.code == MCP_DOCUMENT_AUTHORIZATION_CONSUMED


def test_reserved_authorization_cannot_be_doubly_reserved(exact_file):
    store = DocumentAuthorizationStore()
    auth = store.create_authorization(_read_snapshot(str(exact_file)))
    store.reserve_authorization(auth.auth_id)
    with pytest.raises(McpError) as exc:
        store.reserve_authorization(auth.auth_id)
    assert exc.value.code == MCP_DOCUMENT_AUTHORIZATION_RESERVED


def test_expired_authorization_rejected(exact_file):
    store = DocumentAuthorizationStore(default_ttl_seconds=0)
    auth = store.create_authorization(_read_snapshot(str(exact_file)), ttl_seconds=0)
    time.sleep(0.05)
    with pytest.raises(McpError) as exc:
        store.reserve_authorization(auth.auth_id)
    assert exc.value.code == MCP_DOCUMENT_AUTHORIZATION_EXPIRED


def test_snapshot_mismatch_detected(exact_file):
    store = DocumentAuthorizationStore()
    auth = store.create_authorization(_read_snapshot(str(exact_file)))
    store.consume_authorization(auth.auth_id)
    with pytest.raises(McpError) as exc:
        store.verify_snapshot_matches(auth.auth_id, str(exact_file), size_bytes=99999)
    assert exc.value.code == MCP_DOCUMENT_SNAPSHOT_MISMATCH


def test_snapshot_verification_succeeds_when_unchanged(exact_file):
    store = DocumentAuthorizationStore()
    snapshot = _read_snapshot(str(exact_file))
    auth = store.create_authorization(snapshot)
    store.consume_authorization(auth.auth_id)
    verified = store.verify_snapshot_matches(
        auth.auth_id, snapshot.local_path, snapshot.size_bytes
    )
    assert verified.snapshot.local_path == snapshot.local_path


def test_create_authorization_without_read_fails(exact_file):
    store = DocumentAuthorizationStore()
    snapshot = DocumentInputSnapshot(
        source_uri=str(exact_file), permission=ToolPermission.DENIED
    )
    with pytest.raises(McpError) as exc:
        store.create_authorization(snapshot)
    assert exc.value.code == MCP_DOCUMENT_PERMISSION_DENIED


def test_store_thread_safety(exact_file):
    store = DocumentAuthorizationStore()
    auth = store.create_authorization(_read_snapshot(str(exact_file)))
    errors = []

    def reserve():
        try:
            store.reserve_authorization(auth.auth_id)
        except McpError as e:
            errors.append(e.code)

    threads = [threading.Thread(target=reserve) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one thread wins the reservation; the rest see RESERVED.
    assert errors.count(MCP_DOCUMENT_AUTHORIZATION_RESERVED) == 9
    assert len(errors) == 9


def test_bound_snapshot_uses_resolved_path(tmp_path):
    p = tmp_path / "real.txt"
    p.write_text("x", encoding="utf-8")
    snapshot = DocumentInputSnapshot(source_uri=str(p), permission=ToolPermission.READ)
    bound = LocalDocumentExactFilePolicy().bind(snapshot)
    assert bound.local_path == str(p.resolve())
    assert bound.local_path == bound.source_uri
