"""Phase F — installation approval, separate from Phase C tool confirmation.

Installing a server is a different decision from calling a write tool, so it uses
its own type (`ProvisioningApproval`) bound to a specific plan_id AND plan_hash.
A Phase C `ToolConfirmation` can never be mistaken for installation approval, and
an approval for one plan can never authorize a different one: change the version,
the directory, the install path, the environment names, or the tool policy and the
hash changes, so the approval no longer matches.

Collection (asking the user) is separate from enforcement (`require_approval`), so
the installer can never block on input() and tests never need a terminal.
"""

from rich.console import Console

from mcp_layer.errors import McpError
from mcp_management.filesystem_access import FilesystemAccessApproval, FilesystemAccessPlan
from mcp_management.models import McpProvisioningPlan, ProvisioningApproval
from tools.models import (
    MCP_FILESYSTEM_ACCESS_CONFIRMATION_MISMATCH,
    MCP_FILESYSTEM_ACCESS_CONFIRMATION_REQUIRED,
    MCP_FILESYSTEM_ACCESS_DECLINED,
    MCP_FILESYSTEM_ACCESS_EXPIRED,
    MCP_PROVISIONING_CONFIRMATION_MISMATCH,
    MCP_PROVISIONING_CONFIRMATION_REQUIRED,
    MCP_PROVISIONING_DECLINED,
)

console = Console()

_APPROVALS = {"y", "yes"}


def require_approval(plan: McpProvisioningPlan, approval: ProvisioningApproval):
    """Raise unless `approval` authorizes exactly `plan`. Returns None when valid."""
    if approval is None:
        raise McpError(
            MCP_PROVISIONING_CONFIRMATION_REQUIRED,
            "Installing this MCP server requires explicit approval.",
        )
    if not isinstance(approval, ProvisioningApproval):
        # A Phase C tool confirmation (or anything else) is not installation approval.
        raise McpError(MCP_PROVISIONING_CONFIRMATION_MISMATCH,
                       "That confirmation does not authorize an installation.")
    if not approval.approved:
        raise McpError(MCP_PROVISIONING_DECLINED, "The user declined the installation.")
    expected = plan.compute_hash()
    if approval.plan_id != plan.plan_id or approval.plan_hash != expected:
        raise McpError(
            MCP_PROVISIONING_CONFIRMATION_MISMATCH,
            "The approval does not match this provisioning plan; review the plan again.",
        )
    return None


def require_filesystem_access_approval(plan: FilesystemAccessPlan, approval: FilesystemAccessApproval):
    """Raise unless `approval` authorizes exactly `plan`, and it hasn't expired.

    Same shape as `require_approval`, deliberately kept a DISTINCT type: a Phase F
    ProvisioningApproval (or a Phase C ToolConfirmation) can never be mistaken for a
    filesystem-access approval, so a bare 'yes' can never be replayed to authorize
    a different kind of change than the one it was collected for.
    """
    if approval is None:
        raise McpError(
            MCP_FILESYSTEM_ACCESS_CONFIRMATION_REQUIRED,
            "Changing filesystem access requires explicit approval.",
        )
    if not isinstance(approval, FilesystemAccessApproval):
        raise McpError(MCP_FILESYSTEM_ACCESS_CONFIRMATION_MISMATCH,
                       "That confirmation does not authorize a filesystem access change.")
    if not approval.approved:
        raise McpError(MCP_FILESYSTEM_ACCESS_DECLINED,
                       "The user declined the filesystem access change.")
    if plan.is_expired():
        raise McpError(MCP_FILESYSTEM_ACCESS_EXPIRED,
                       "This filesystem access plan has expired; prepare a new one.")
    expected = plan.compute_hash()
    if approval.plan_id != plan.plan_id or approval.plan_hash != expected:
        raise McpError(
            MCP_FILESYSTEM_ACCESS_CONFIRMATION_MISMATCH,
            "The approval does not match this filesystem access plan; review the plan again.",
        )
    return None


def render_plan(plan: McpProvisioningPlan) -> str:
    """The deterministic approval prompt text (trusted data only)."""
    return "\n".join(plan.summary_lines())


def confirm_provisioning(plan: McpProvisioningPlan) -> bool:
    """Show the plan and read a y/N decision. Default is NO."""
    console.print(f"[yellow]{render_plan(plan)}[/yellow]")
    console.print(r"[yellow]Install this MCP server? \[yes/No][/yellow]")
    try:
        answer = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in _APPROVALS


def collect_approval(plan: McpProvisioningPlan, confirmer=None) -> ProvisioningApproval:
    """Ask the user about `plan` and return an approval bound to it."""
    if confirmer is None:
        confirmer = confirm_provisioning
    approved = bool(confirmer(plan))
    return ProvisioningApproval(approved=approved, plan_id=plan.plan_id,
                                plan_hash=plan.compute_hash())
