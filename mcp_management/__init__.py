"""Phase F — automatic MCP provisioning: trusted catalog, plan, approval, install.

The assistant can determine which approved MCP capability a request needs and then
prepare, install, configure, validate, and activate that server — but only after
explicit user approval, and only from the application-maintained catalog. The LLM
never generates or executes an installation command.

Provisioned servers enter the system through the existing Phase E path: a generated
Phase E configuration is activated, and the ordinary bootstrap registers the remote
tools as McpTool(BaseTool) for the existing ToolExecutor.
"""

from mcp_management.approval import (
    collect_approval,
    confirm_provisioning,
    render_plan,
    require_approval,
)
from mcp_management.capability_detector import detect_capability, validate_detection
from mcp_management.catalog import (
    McpCatalog,
    McpCatalogEntry,
    build_catalog,
    load_catalog,
)
from mcp_management.installer import install
from mcp_management.manager import MAX_PROVISIONING_ATTEMPTS, McpProvisioningManager
from mcp_management.models import (
    CapabilityDetection,
    McpProvisioningPlan,
    PendingCapabilityRequest,
    PendingRequestState,
    ProvisioningApproval,
)
from mcp_management.planner import build_plan, validate_approved_directory
from mcp_management.provisioning_tools import (
    ALL_PROVISIONING_TOOL_CLASSES,
    register_provisioning_tools,
)

__all__ = [
    "ALL_PROVISIONING_TOOL_CLASSES",
    "CapabilityDetection",
    "MAX_PROVISIONING_ATTEMPTS",
    "McpCatalog",
    "McpCatalogEntry",
    "McpProvisioningManager",
    "McpProvisioningPlan",
    "PendingCapabilityRequest",
    "PendingRequestState",
    "ProvisioningApproval",
    "build_catalog",
    "build_plan",
    "collect_approval",
    "confirm_provisioning",
    "detect_capability",
    "install",
    "load_catalog",
    "register_provisioning_tools",
    "render_plan",
    "require_approval",
    "validate_approved_directory",
    "validate_detection",
]
