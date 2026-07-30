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
from mcp_management.models import McpProvisioningPlan, ProvisioningApproval
from tools.models import (
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
