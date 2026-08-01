"""Phase G.4 — atomic, single-use authorization for local document conversion.

A document may be converted to markdown by an MCP server only after:

1. An explicit `ToolPermission.READ` classification is recorded on the snapshot.
2. The path is an exact, absolute local file path (no URLs, no UNC, no relative
   segments, no globs).
3. The file exists at the time the authorization is created.
4. A single-use authorization is reserved before the MCP call and consumed
   immediately after the single conversion attempt — success or failure.

The authorization store is intentionally in-process and non-persistent. It does
not mutate the filesystem and it never rewrites the supplied path.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, Optional
from urllib.parse import urlparse

from mcp_layer.errors import McpError
from tools.models import (
    MCP_DOCUMENT_AUTHORIZATION_CONSUMED,
    MCP_DOCUMENT_AUTHORIZATION_EXPIRED,
    MCP_DOCUMENT_AUTHORIZATION_REQUIRED,
    MCP_DOCUMENT_AUTHORIZATION_RESERVED,
    MCP_DOCUMENT_NOT_FOUND,
    MCP_DOCUMENT_PATH_INVALID,
    MCP_DOCUMENT_PATH_RESTRICTED,
    MCP_DOCUMENT_PERMISSION_DENIED,
    MCP_DOCUMENT_SNAPSHOT_MISMATCH,
    ToolPermission,
)


def build_document_snapshots_from_text(user_text):
    """Extract local document paths from `user_text` and return READ snapshots.

    This is the canonical bridge from the assistant/selection flow to the
    document authorization store: every path the user explicitly mentioned is
    recorded as a snapshot with explicit READ permission.  When the file already
    exists, its size, SHA-256, and creation timestamp are captured so the plan
    hash binds to the exact bytes the user showed intent to convert.

    The policy layer later validates existence, absoluteness, and exact-file
    constraints before any MCP server sees the path.
    """
    import hashlib

    from mcp_management.capability_detector import extract_document_paths

    paths = extract_document_paths(user_text)
    snapshots = []
    for path in paths:
        size = None
        sha256 = None
        ctime_ns = None
        if os.path.isfile(path):
            size = os.path.getsize(path)
            ctime_ns = int(os.stat(path).st_ctime_ns)
            with open(path, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()
        snapshots.append(DocumentInputSnapshot(
            source_uri=path,
            permission=ToolPermission.READ,
            size_bytes=size,
            sha256=sha256,
            ctime_ns=ctime_ns,
        ))
    return tuple(snapshots)

_GLOB_CHARS = re.compile(r"[*?\[\]]")
# Explicit remote/URL-like schemes that are never local files.
_REMOTE_SCHEMES = {"http", "https", "ftp", "ftps", "sftp", "file", "data", "blob"}


@dataclass(frozen=True)
class DocumentInputSnapshot:
    """Immutable description of a local document the user wants converted.

    `source_uri` is the canonical reference passed through planning.  For a
    local file it is the absolute path.  The `permission` field must be
    `ToolPermission.READ`; anything else fails closed.
    """

    source_uri: str
    local_path: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    ctime_ns: Optional[int] = None
    permission: ToolPermission = ToolPermission.DENIED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def content_fingerprint(self) -> str:
        """Stable, deterministic identity of this snapshot."""
        return hashlib.sha256(
            f"{self.source_uri}|{self.size_bytes}|{self.sha256}|{self.ctime_ns}|{self.permission}".encode(
                "utf-8"
            )
        ).hexdigest()


@dataclass(frozen=True)
class DocumentInputAuthorization:
    """Single-use binding to a snapshot.

    `consumed` is set by the store after the one conversion attempt.
    `reserved` is an internal transient flag that prevents concurrent use of the
    same authorization while the MCP call is in flight.
    """

    auth_id: str
    snapshot: DocumentInputSnapshot
    expires_at: datetime
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    consumed: bool = False
    _reserved: bool = False


class LocalDocumentExactFilePolicy:
    """Fail-closed policy: only exact, absolute, existing local file paths.

    Rejects every non-local reference: URLs, UNC paths, relative paths, paths
    containing glob characters, and paths with traversal components that resolve
    differently than the literal absolute form.  The file must exist and must
    carry explicit READ permission.
    """

    def validate(self, snapshot: DocumentInputSnapshot) -> None:
        if snapshot.permission != ToolPermission.READ:
            raise McpError(
                MCP_DOCUMENT_PERMISSION_DENIED,
                "Document snapshot lacks explicit READ permission.",
            )

        source = (snapshot.local_path or snapshot.source_uri or "").strip()
        if not source:
            raise McpError(MCP_DOCUMENT_PATH_INVALID, "Document path is empty.")

        parsed = urlparse(source)
        if parsed.scheme and parsed.scheme.lower() in _REMOTE_SCHEMES:
            raise McpError(
                MCP_DOCUMENT_PATH_RESTRICTED,
                f"Remote document URI scheme not allowed: {parsed.scheme}.",
            )

        # Reject UNC/network paths (Windows \server\share or forward-slash variants).
        if source.startswith("\\\\") or source.startswith("//"):
            raise McpError(
                MCP_DOCUMENT_PATH_RESTRICTED,
                "UNC or network paths are not allowed for local document conversion.",
            )

        # Detect URLs the urllib parser may have missed.
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", source):
            raise McpError(
                MCP_DOCUMENT_PATH_RESTRICTED,
                "URL-style document source is not allowed.",
            )

        # Cross-platform glob detection before parsing.
        if _GLOB_CHARS.search(source):
            raise McpError(
                MCP_DOCUMENT_PATH_RESTRICTED,
                "Glob characters are not allowed in document path.",
            )

        # Parse strictly: we do not want POSIX to silently reinterpret a Windows path.
        if _looks_like_windows_path(source):
            pure = PureWindowsPath(source)
        else:
            pure = PurePosixPath(source)

        if not pure.is_absolute():
            raise McpError(
                MCP_DOCUMENT_PATH_INVALID,
                "Document path must be absolute.",
            )

        # Exact-file policy: no relative components anywhere in the path.
        if any(part in (".", "..") for part in pure.parts):
            raise McpError(
                MCP_DOCUMENT_PATH_INVALID,
                "Document path contains disallowed relative components.",
            )

        # Resolve strictly on the real filesystem. resolve() with strict=True fails if the
        # target does not exist.
        try:
            resolved = Path(source).resolve(strict=True)
        except FileNotFoundError as exc:
            raise McpError(
                MCP_DOCUMENT_NOT_FOUND,
                f"Document does not exist: {source}",
            ) from exc
        except (OSError, RuntimeError) as exc:
            raise McpError(
                MCP_DOCUMENT_PATH_INVALID,
                f"Document path could not be resolved: {exc}",
            ) from exc

        # The resolved path must still be absolute and must not have become a UNC path.
        if not resolved.is_absolute() or str(resolved).startswith("\\\\"):
            raise McpError(
                MCP_DOCUMENT_PATH_INVALID,
                "Resolved document path is not a local absolute path.",
            )

        if not resolved.is_file():
            raise McpError(
                MCP_DOCUMENT_NOT_FOUND,
                f"Document path is not a file: {source}",
            )

    def bind(self, snapshot: DocumentInputSnapshot) -> DocumentInputSnapshot:
        """Validate then return a snapshot whose local_path is resolved absolutely."""
        self.validate(snapshot)
        resolved = Path(snapshot.local_path or snapshot.source_uri).resolve()
        return DocumentInputSnapshot(
            source_uri=snapshot.source_uri,
            local_path=str(resolved),
            mime_type=snapshot.mime_type,
            size_bytes=snapshot.size_bytes,
            sha256=snapshot.sha256,
            ctime_ns=snapshot.ctime_ns,
            permission=snapshot.permission,
            created_at=snapshot.created_at,
        )


def _looks_like_windows_path(source: str) -> bool:
    """Heuristic: drive letter (C:) or backslashes indicate Windows form."""
    if "\\" in source:
        return True
    if len(source) >= 2 and source[1] == ":" and source[0].isalpha():
        return True
    return False


class DocumentAuthorizationStore:
    """In-process store for document authorizations.

    Thread-safe.  Authorizations expire after `default_ttl_seconds`.  Once
    consumed they can never be reused.  A transient reservation prevents two
    threads from using the same authorization concurrently.

    A single default instance is shared across the process; callers may also
    create isolated instances for tests.
    """

    _default: Optional["DocumentAuthorizationStore"] = None
    _default_lock = threading.Lock()

    def __init__(self, default_ttl_seconds: int = 300):
        self._lock = threading.Lock()
        self._auths: Dict[str, DocumentInputAuthorization] = {}
        self.default_ttl_seconds = default_ttl_seconds

    @classmethod
    def default(cls) -> "DocumentAuthorizationStore":
        with cls._default_lock:
            if cls._default is None:
                cls._default = cls()
            return cls._default

    def create_authorization(
        self,
        snapshot: DocumentInputSnapshot,
        ttl_seconds: Optional[int] = None,
    ) -> DocumentInputAuthorization:
        """Validate the snapshot under exact-file policy and store an authorization."""
        policy = LocalDocumentExactFilePolicy()
        bound = policy.bind(snapshot)
        auth_id = "doc-auth-" + secrets.token_urlsafe(24)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        auth = DocumentInputAuthorization(
            auth_id=auth_id,
            snapshot=bound,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl),
        )
        with self._lock:
            self._auths[auth_id] = auth
        return auth

    def get_authorization(self, auth_id: str) -> Optional[DocumentInputAuthorization]:
        """Read-only access; does not change consumed/reserved state."""
        with self._lock:
            return self._auths.get(auth_id)

    def reserve_authorization(self, auth_id: str) -> DocumentInputAuthorization:
        """Mark an authorization as in-use; raises if consumed, expired, or reserved."""
        with self._lock:
            auth = self._auths.get(auth_id)
            if auth is None:
                raise McpError(
                    MCP_DOCUMENT_AUTHORIZATION_REQUIRED,
                    "Document authorization not found.",
                )
            if auth.consumed:
                raise McpError(
                    MCP_DOCUMENT_AUTHORIZATION_CONSUMED,
                    "Document authorization has already been used.",
                )
            if auth._reserved:
                raise McpError(
                    MCP_DOCUMENT_AUTHORIZATION_RESERVED,
                    "Document authorization is already reserved.",
                )
            if datetime.now(timezone.utc) > auth.expires_at:
                raise McpError(
                    MCP_DOCUMENT_AUTHORIZATION_EXPIRED,
                    "Document authorization has expired.",
                )
            updated = DocumentInputAuthorization(
                auth_id=auth.auth_id,
                snapshot=auth.snapshot,
                expires_at=auth.expires_at,
                created_at=auth.created_at,
                consumed=False,
                _reserved=True,
            )
            self._auths[auth_id] = updated
            return updated

    def consume_authorization(self, auth_id: str) -> DocumentInputAuthorization:
        """Mark an authorization as consumed.  Safe to call on reserved or unreserved auth."""
        with self._lock:
            auth = self._auths.get(auth_id)
            if auth is None:
                raise McpError(
                    MCP_DOCUMENT_AUTHORIZATION_REQUIRED,
                    "Document authorization not found.",
                )
            updated = DocumentInputAuthorization(
                auth_id=auth.auth_id,
                snapshot=auth.snapshot,
                expires_at=auth.expires_at,
                created_at=auth.created_at,
                consumed=True,
                _reserved=False,
            )
            self._auths[auth_id] = updated
            return updated

    def verify_snapshot_matches(
        self,
        auth_id: str,
        local_path: str,
        size_bytes: int,
        sha256: Optional[str] = None,
    ) -> DocumentInputAuthorization:
        """Verify the on-disk file still matches the authorized snapshot.

        Called before creating a fresh authorization for a retry; the original
        authorization must already be consumed.
        """
        with self._lock:
            auth = self._auths.get(auth_id)
        if auth is None:
            raise McpError(
                MCP_DOCUMENT_AUTHORIZATION_REQUIRED,
                "Document authorization not found.",
            )
        snapshot = auth.snapshot
        if snapshot.local_path != local_path:
            raise McpError(
                MCP_DOCUMENT_SNAPSHOT_MISMATCH,
                "Document path does not match authorized snapshot.",
            )
        if snapshot.size_bytes is not None and snapshot.size_bytes != size_bytes:
            raise McpError(
                MCP_DOCUMENT_SNAPSHOT_MISMATCH,
                "Document size does not match authorized snapshot.",
            )
        if sha256 is not None and snapshot.sha256 is not None and snapshot.sha256 != sha256:
            raise McpError(
                MCP_DOCUMENT_SNAPSHOT_MISMATCH,
                "Document hash does not match authorized snapshot.",
            )
        return auth

    def find_and_reserve_for_path(
        self,
        local_path: str,
    ) -> Optional[DocumentInputAuthorization]:
        """Find an unconsumed authorization matching `local_path` and reserve it.

        The path is compared after canonical absolute resolution.  Returns None
        if no unconsumed, unreserved, unexpired authorization matches.
        """
        target = _canonical_local_path(local_path)
        with self._lock:
            now = datetime.now(timezone.utc)
            for auth in list(self._auths.values()):
                if auth.consumed or auth._reserved or now > auth.expires_at:
                    continue
                if _canonical_local_path(auth.snapshot.local_path or "") == target:
                    updated = DocumentInputAuthorization(
                        auth_id=auth.auth_id,
                        snapshot=auth.snapshot,
                        expires_at=auth.expires_at,
                        created_at=auth.created_at,
                        consumed=False,
                        _reserved=True,
                    )
                    self._auths[auth.auth_id] = updated
                    return updated
        return None


def _canonical_local_path(path: str) -> str:
    """Absolute, normalized local path for deterministic comparison."""
    try:
        return str(Path(path).resolve())
    except (OSError, ValueError):
        return os.path.abspath(path)


def _path_to_file_uri(path: str) -> str:
    """Convert an absolute local path to a file:// URI."""
    import urllib.parse
    import urllib.request

    return urllib.parse.urljoin("file:", urllib.request.pathname2url(os.path.abspath(path)))


def _file_uri_to_path(uri: str) -> Optional[str]:
    """Extract the local path from a file:// URI, or None if it is not one."""
    import urllib.parse
    import urllib.request

    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme and parsed.scheme.lower() != "file":
        return None
    # url2pathname expects the path component; strip a leading slash on Windows.
    path = urllib.request.url2pathname(parsed.path)
    return path
