"""Phase G.3 — generalized, installer-agnostic auto-provisioning models.

Deliberately a SEPARATE set of types from Phase F's `mcp_management.models`
(`McpProvisioningPlan`, `PendingCapabilityRequest`, ...), which remain unchanged
and continue to back the manual, LLM-callable `mcp.provision.*` tools
(`mcp_management/provisioning_tools.py`). These types back the AUTOMATIC flow: a
request needs an approved-but-uninstalled provider, so the assistant itself
proposes a plan (never the model), the user approves once, and the ORIGINAL
request resumes. Naming them distinctly (`AutoProvisioningPlan`, not a second
`McpProvisioningPlan`) means a plan or approval from one flow can never be
mistaken for the other's — `require_auto_provisioning_approval` rejects the
wrong type outright, exactly like Phase F.1 keeps `FilesystemAccessApproval`
distinct from `ProvisioningApproval`.

`AutoProvisioningPlan` is deliberately generic across installer types (npm,
python_venv, ...): nothing here knows HOW to install a package, only WHAT will
be installed, from where, and with what access — derived entirely from a
trusted `McpCatalogEntry` plus the original request. `plan_hash` covers every
security-relevant field (Task 3), so approval binds to exactly one plan; change
the version, the lock file, the tool policy, or the original request and the old
approval no longer matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple

from tools.models import hash_arguments

if TYPE_CHECKING:
    from mcp_management.document_authorization import DocumentInputSnapshot


DEFAULT_PLAN_TTL_SECONDS = 15 * 60


class ProvisioningPlanStatus(str, Enum):
    PREPARED = "prepared"
    APPROVED = "approved"
    INSTALLING = "installing"
    VALIDATING = "validating"
    ACTIVATING = "activating"
    COMPLETED = "completed"
    DECLINED = "declined"
    FAILED = "failed"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"


@dataclass(frozen=True)
class AutoProvisioningPlan:
    """What will be installed, from where, with what access — for the automatic
    (approved-but-not-installed) flow. Immutable once built."""

    plan_id: str
    plan_hash: str
    request_id: str
    original_user_text: str

    catalog_id: str
    server_id: str
    display_name: str

    installer_type: str
    exact_package: str
    exact_version: str
    lock_file_hash: Optional[str]
    executable_identity: str

    expected_tools: Tuple[str, ...]
    tool_policy_hash: str
    environment_allowlist: Tuple[str, ...]
    install_network_hosts: Tuple[str, ...]
    runtime_network_policy: str

    target_install_directory: str
    candidate_config_hash: str

    created_at: str
    expires_at: str
    status: ProvisioningPlanStatus = ProvisioningPlanStatus.PREPARED
    risk_summary: Tuple[str, ...] = ()
    # Phase G.4 — document snapshots bound into approval hash; revalidated before use.
    document_snapshots: Tuple["DocumentInputSnapshot", ...] = ()

    def security_fields(self) -> dict:
        """Exactly the fields the approval is bound to (Task 3)."""
        return {
            "request_id": self.request_id,
            "catalog_id": self.catalog_id,
            "server_id": self.server_id,
            "installer_type": self.installer_type,
            "exact_package": self.exact_package,
            "exact_version": self.exact_version,
            "lock_file_hash": self.lock_file_hash,
            "executable_identity": self.executable_identity,
            "expected_tools": list(self.expected_tools),
            "tool_policy_hash": self.tool_policy_hash,
            "environment_allowlist": list(self.environment_allowlist),
            "install_network_hosts": list(self.install_network_hosts),
            "runtime_network_policy": self.runtime_network_policy,
            "target_install_directory": self.target_install_directory,
            "candidate_config_hash": self.candidate_config_hash,
            "document_snapshots": [
                snapshot.content_fingerprint() for snapshot in self.document_snapshots
            ],
        }

    def compute_hash(self) -> str:
        return hash_arguments(self.security_fields())

    def with_hash(self) -> "AutoProvisioningPlan":
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
        """Deterministic, human-readable approval-prompt text (Task 15) — built
        only from trusted catalog + validated plan data, never model output."""
        installer_label = {"npm": "npm", "python_venv": "Python virtual environment"}.get(
            self.installer_type, self.installer_type)
        lines = [
            "Install approved MCP server",
            "",
            "Server:",
            f"  {self.display_name}",
            "",
            "Server ID:",
            f"  {self.server_id}",
            "",
            "Catalog:",
            f"  {self.catalog_id}",
            "",
            "Installer:",
            f"  {installer_label}",
            "",
            "Package:",
            f"  {self.exact_package}=={self.exact_version}" if self.installer_type == "python_venv"
            else f"  {self.exact_package}@{self.exact_version}",
            "",
            "Installation destination:",
            f"  {self.target_install_directory}",
            "",
            "Expected tools:",
        ]
        lines.extend(f"  - {t}" for t in self.expected_tools) if self.expected_tools else lines.append("  none")
        lines.append("")
        lines.append("Installation network:")
        lines.extend(f"  - {h}" for h in self.install_network_hosts) if self.install_network_hosts \
            else lines.append("  none")
        lines.append("")
        lines.append("Runtime network:")
        lines.append(f"  {self.runtime_network_policy}")
        lines.append("")
        lines.append("Environment variables:")
        lines.extend(f"  - {e}" for e in self.environment_allowlist) if self.environment_allowlist \
            else lines.append("  none")
        for risk in self.risk_summary:
            lines.append(f"  risk: {risk}")
        lines.append("")
        lines.append("After approval:")
        lines.append("  - install in an isolated environment")
        lines.append("  - validate the package and MCP tools")
        lines.append("  - start the server")
        lines.append("  - resume the original request")
        lines.append("")
        lines.append("Proceed?")
        return tuple(lines)


@dataclass(frozen=True)
class AutoProvisioningApproval:
    """A user's decision on one exact auto-provisioning plan (single-use,
    hash-bound). A DISTINCT type from Phase F's `ProvisioningApproval` and
    Phase F.1's `FilesystemAccessApproval` — never interchangeable."""

    approved: bool
    plan_id: str
    plan_hash: str


class PendingAutoProvisioningState(str, Enum):
    DETECTED = "detected"
    PLAN_PREPARED = "plan_prepared"
    AWAITING_APPROVAL = "awaiting_approval"
    INSTALLING = "installing"
    VALIDATING = "validating"
    ACTIVATING = "activating"
    READY = "ready"
    RESUMED = "resumed"
    DECLINED = "declined"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass(frozen=True)
class PendingAutoProvisioningRequest:
    """The original blocked user request, preserved across provisioning so it
    can resume exactly once (Task 4)."""

    request_id: str
    original_user_text: str
    capability: str
    catalog_id: str
    server_id: str
    plan_id: Optional[str] = None
    state: PendingAutoProvisioningState = PendingAutoProvisioningState.DETECTED
    attempts: int = 0
    # Phase G.4: document snapshots the user showed intent to convert, carried
    # through detection/plan/approval/resumption and bound into the plan hash.
    document_snapshots: Tuple["DocumentInputSnapshot", ...] = ()

    def advanced(self, state: PendingAutoProvisioningState, plan_id=None, attempts=None):
        return replace(
            self,
            state=state,
            plan_id=plan_id if plan_id is not None else self.plan_id,
            attempts=self.attempts if attempts is None else attempts,
        )


@dataclass(frozen=True)
class ProvisioningResult:
    """What actually happened after a successful install + activation."""

    server_id: str
    catalog_id: str
    installed_version: str
    managed_config_path: str
    installed_state_hash: str
    validation_summary: dict
    runtime_activation_required: bool = True
