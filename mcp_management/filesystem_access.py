"""Phase F.1 — expand an already-installed server's approved filesystem roots.

Mirrors the Phase F provisioning models (McpProvisioningPlan / PendingCapabilityRequest,
see mcp_management/models.py) but for a narrower, cheaper change: adding or removing one
approved directory on a server that is ALREADY installed. No package is (re)installed here.

A FilesystemAccessPlan is immutable and hash-bound, exactly like a provisioning plan, so
an approval can never silently drift to a different set of roots: change the server, the
operation, the requested directory, or the current/proposed root sets and the hash changes,
so a stale approval no longer matches. `expires_at` bounds how long an unapproved plan stays
valid, and PendingFilesystemAccessRequest carries the ORIGINAL blocked user request across the
single approval turn so it can be resumed through the normal router / shortlist / executor
pipeline afterward — never answered directly by this module.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Tuple

from tools.models import hash_arguments

DEFAULT_PLAN_TTL_SECONDS = 15 * 60


class FilesystemAccessOperation(str, Enum):
    ADD_ROOT = "add_root"
    REMOVE_ROOT = "remove_root"
    REPLACE_ROOTS = "replace_roots"


@dataclass(frozen=True)
class FilesystemAccessPlan:
    """What will change, and to exactly which server. Immutable once built."""

    plan_id: str
    server_id: str
    catalog_id: str
    operation: FilesystemAccessOperation
    requested_directory: str
    current_allowed_directories: Tuple[str, ...]
    proposed_allowed_directories: Tuple[str, ...]
    requested_path: Optional[str] = None
    original_user_text: str = ""
    risk_summary: Tuple[str, ...] = ()
    plan_hash: str = ""
    created_at: str = ""
    expires_at: str = ""

    def security_fields(self) -> dict:
        """Exactly the fields the approval is bound to."""
        return {
            "server_id": self.server_id,
            "catalog_id": self.catalog_id,
            "operation": (self.operation.value if isinstance(self.operation, FilesystemAccessOperation)
                         else str(self.operation)),
            "requested_directory": self.requested_directory,
            "current_allowed_directories": list(self.current_allowed_directories),
            "proposed_allowed_directories": list(self.proposed_allowed_directories),
            "expires_at": self.expires_at,
        }

    def compute_hash(self) -> str:
        return hash_arguments(self.security_fields())

    def with_hash(self) -> "FilesystemAccessPlan":
        return replace(self, plan_hash=self.compute_hash())

    def is_expired(self, now=None) -> bool:
        if not self.expires_at:
            return False
        now = now or datetime.now(timezone.utc)
        try:
            expires = datetime.fromisoformat(self.expires_at)
        except ValueError:
            return False
        return now >= expires

    def summary_lines(self) -> Tuple[str, ...]:
        """Deterministic, human-readable approval-prompt text (trusted data only)."""
        verb = "Add" if self.operation == FilesystemAccessOperation.ADD_ROOT else "Remove"
        lines = [f"Filesystem access change ({verb.lower()}) for server: {self.server_id}"]
        if self.requested_path:
            lines.append(f"  requested file: {self.requested_path}")
        lines.append(f"  {verb.lower()} directory: {self.requested_directory}")
        lines.append("  current allowed directories:")
        for d in self.current_allowed_directories:
            lines.append(f"    - {d}")
        lines.append("  new allowed directories after approval:")
        for d in self.proposed_allowed_directories:
            lines.append(f"    - {d}")
        for risk in self.risk_summary:
            lines.append(f"  risk: {risk}")
        lines.append("  after approval:")
        lines.append("    - the existing filesystem MCP server will restart")
        lines.append("    - the MCP package will not be reinstalled")
        lines.append("    - read operations will be allowed inside the new directory")
        lines.append("    - write operations will still require confirmation")
        lines.append("    - move/edit remain denied")
        return tuple(lines)


@dataclass(frozen=True)
class FilesystemAccessApproval:
    """A user's decision on one exact filesystem-access plan (single-use, hash-bound)."""

    approved: bool
    plan_id: str
    plan_hash: str


class PendingFilesystemAccessState(str, Enum):
    DETECTED = "detected"
    PLAN_PREPARED = "plan_prepared"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    APPLYING = "applying"
    VALIDATING = "validating"
    READY = "ready"
    RESUMED = "resumed"
    DECLINED = "declined"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass(frozen=True)
class PendingFilesystemAccessRequest:
    """The original blocked user request, preserved across approval so it can resume."""

    request_id: str
    original_user_text: str
    requested_path: str
    proposed_root: str
    server_id: str
    access_plan_id: Optional[str] = None
    state: PendingFilesystemAccessState = PendingFilesystemAccessState.DETECTED
    provisioning_attempts: int = 0

    def advanced(self, state: PendingFilesystemAccessState, plan_id=None, attempts=None):
        return replace(
            self,
            state=state,
            access_plan_id=plan_id if plan_id is not None else self.access_plan_id,
            provisioning_attempts=self.provisioning_attempts if attempts is None else attempts,
        )
