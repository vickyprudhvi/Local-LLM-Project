"""Phase F — immutable provisioning models.

A provisioning plan is the single, deterministic description of what will be
installed and what access it will receive. Its `plan_hash` covers every
security-relevant field, so approval binds to exactly one plan: change the
version, the install directory, the approved directory, the environment names, or
the tool policy and the old approval no longer matches.

Nothing here holds secret VALUES — only environment-variable names.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

from mcp_layer.config import McpToolPolicy
from tools.models import ToolPermission, hash_arguments


def policy_fingerprint(policy: McpToolPolicy) -> dict:
    """Stable, JSON-serializable view of a tool policy (used inside the plan hash)."""
    return {
        "default_permission": ToolPermission.coerce(policy.default_permission).value,
        "tools": {
            name: {
                "enabled": bool(entry.enabled),
                "permission": ToolPermission.coerce(entry.permission).value,
            }
            for name, entry in sorted(policy.tools.items())
        },
    }


@dataclass(frozen=True)
class McpProvisioningPlan:
    """What will be installed, where, with what access. Immutable once built."""

    plan_id: str
    catalog_id: str
    server_id: str
    display_name: str
    package_manager: str
    package_name: str
    package_version: str
    package_source: str
    entrypoint_relative: str
    install_directory: Path
    runtime_workspace: Path
    transport: str
    requested_directories: Tuple[Path, ...]
    requested_environment_variables: Tuple[str, ...]
    proposed_tool_policy: McpToolPolicy
    required_runtimes: Tuple[str, ...] = ()
    requires_network_install: bool = True
    requires_credentials: bool = False
    risk_summary: Tuple[str, ...] = ()
    plan_hash: str = ""

    def security_fields(self) -> dict:
        """Exactly the fields the approval is bound to."""
        return {
            "catalog_id": self.catalog_id,
            "server_id": self.server_id,
            "package_manager": self.package_manager,
            "package_name": self.package_name,
            "package_version": self.package_version,
            "package_source": self.package_source,
            "entrypoint_relative": self.entrypoint_relative,
            "install_directory": str(self.install_directory),
            "runtime_workspace": str(self.runtime_workspace),
            "transport": self.transport,
            "requested_directories": [str(p) for p in self.requested_directories],
            "requested_environment_variables": list(self.requested_environment_variables),
            "tool_policy": policy_fingerprint(self.proposed_tool_policy),
        }

    def compute_hash(self) -> str:
        return hash_arguments(self.security_fields())

    def with_hash(self) -> "McpProvisioningPlan":
        from dataclasses import replace

        return replace(self, plan_hash=self.compute_hash())

    def read_tools(self) -> Tuple[str, ...]:
        return tuple(sorted(n for n, e in self.proposed_tool_policy.tools.items()
                            if e.enabled and e.permission is ToolPermission.READ))

    def write_tools(self) -> Tuple[str, ...]:
        return tuple(sorted(n for n, e in self.proposed_tool_policy.tools.items()
                            if e.enabled and e.permission is ToolPermission.WRITE))

    def denied_tools(self) -> Tuple[str, ...]:
        return tuple(sorted(n for n, e in self.proposed_tool_policy.tools.items()
                            if not e.enabled or e.permission is ToolPermission.DENIED))

    def summary_lines(self) -> Tuple[str, ...]:
        """Deterministic, human-readable plan summary for the approval prompt.

        Built only from trusted catalog + validated plan data — never model output.
        """
        lines = [
            f"Install MCP server: {self.display_name} ({self.catalog_id})",
            f"  package:      {self.package_name}@{self.package_version} via {self.package_manager}",
            f"  source:       {self.package_source}",
            f"  install into: {self.install_directory}",
            f"  workspace:    {self.runtime_workspace}",
            f"  transport:    {self.transport}",
        ]
        for directory in self.requested_directories:
            lines.append(f"  grants access to directory: {directory}")
        if self.requested_environment_variables:
            lines.append("  environment variables (names only): "
                         + ", ".join(self.requested_environment_variables))
        lines.append(f"  read tools:   {', '.join(self.read_tools()) or 'none'}")
        lines.append(f"  write tools (each still needs confirmation): "
                     f"{', '.join(self.write_tools()) or 'none'}")
        lines.append(f"  denied tools: {', '.join(self.denied_tools()) or 'none'}")
        for risk in self.risk_summary:
            lines.append(f"  risk: {risk}")
        lines.append(f"  network access during install: {'yes' if self.requires_network_install else 'no'}")
        lines.append(f"  credentials required: {'yes' if self.requires_credentials else 'no'}")
        lines.append("  after approval: install, generate configuration, validate, then activate.")
        return tuple(lines)


@dataclass(frozen=True)
class ProvisioningApproval:
    """A user's decision on one exact provisioning plan (single-use, hash-bound)."""

    approved: bool
    plan_id: str
    plan_hash: str


class PendingRequestState(str, Enum):
    CAPABILITY_DETECTED = "capability_detected"
    PLAN_PREPARED = "plan_prepared"
    AWAITING_APPROVAL = "awaiting_approval"
    INSTALLING = "installing"
    VALIDATING = "validating"
    READY = "ready"
    RESUMED = "resumed"
    DECLINED = "declined"
    FAILED = "failed"


@dataclass(frozen=True)
class PendingCapabilityRequest:
    """The original user request, preserved across provisioning so it can resume."""

    request_id: str
    original_user_text: str
    required_capability: str
    selected_catalog_id: str
    provisioning_plan_id: Optional[str] = None
    state: PendingRequestState = PendingRequestState.CAPABILITY_DETECTED
    attempts: int = 0

    def advanced(self, state: PendingRequestState, plan_id=None, attempts=None):
        from dataclasses import replace

        return replace(
            self,
            state=state,
            provisioning_plan_id=plan_id if plan_id is not None else self.provisioning_plan_id,
            attempts=self.attempts if attempts is None else attempts,
        )


@dataclass(frozen=True)
class CapabilityDetection:
    """Structured detector output, already validated against the trusted catalog."""

    requires_mcp: bool
    capability: Optional[str] = None
    recommended_catalog_id: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""
    error_code: Optional[str] = None

    def to_dict(self) -> dict:
        out = {
            "requires_mcp": self.requires_mcp,
            "capability": self.capability,
            "recommended_catalog_id": self.recommended_catalog_id,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }
        if self.error_code:
            out["error_code"] = self.error_code
        return out
