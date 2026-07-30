"""Phase F — provisioning exposed as ordinary built-in tools.

These are plain BaseTool subclasses, so they run through the existing
ToolExecutor with the existing Phase C permission and confirmation logic. They
accept only TRUSTED IDENTIFIERS — a catalog id, a plan id/hash, a server id, and a
directory that deterministic code then screens. They never accept a package name,
a command, a URL, an executable path, or shell arguments, so the LLM cannot
construct an installation.

The install tool is WRITE: the Phase C prompt shows the deterministic plan summary
and binds to the exact arguments, and the installer additionally re-checks the plan
hash before touching the filesystem.
"""

from mcp_layer.errors import McpError
from mcp_management.models import ProvisioningApproval
from tools.base import BaseTool, ToolFailure, ToolValidationError
from tools.models import (
    MCP_PROVISIONING_PLAN_INVALID,
    MCP_SERVER_NOT_APPROVED,
    ToolPermission,
)

_ID_MAX = 128


def _identifier(arguments, key, required=True):
    value = arguments.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ToolValidationError(f"'{key}' must be a non-empty string.")
    # Validate the RAW value: reject control characters before any normalization,
    # so a trailing newline or null byte can never be silently stripped away.
    if len(value) > _ID_MAX or any(c in value for c in ("\x00", "\n", "\r", "\t")):
        raise ToolValidationError(f"'{key}' is not a valid identifier.")
    return value.strip()


class _ManagerTool(BaseTool):
    """Base for provisioning tools; holds the manager and normalizes McpError."""

    llm_callable = True
    timeout_seconds = 600.0

    def __init__(self, manager):
        self._manager = manager

    def _run(self, arguments):
        raise NotImplementedError

    def execute(self, arguments: dict) -> dict:
        try:
            return self._run(arguments)
        except McpError as e:
            # Controlled, coded failure -> normalized ToolResult by the executor.
            raise ToolFailure(e.code, e.message, retryable=e.retryable)


class CatalogSearchTool(_ManagerTool):
    name = "mcp.catalog.search"
    description = ("List the approved MCP servers in the trusted catalog, optionally "
                   "filtered by capability. Read-only; installs nothing.")
    input_schema = {
        "type": "object",
        "properties": {
            "capability": {"type": "string",
                           "description": "Optional capability filter, e.g. 'filesystem'."}
        },
    }
    permission = ToolPermission.READ
    timeout_seconds = 15.0

    def _run(self, arguments):
        capability = _identifier(arguments, "capability", required=False)
        summaries = list(self._manager.catalog.capability_summaries())
        if capability:
            summaries = [s for s in summaries if capability in s["capabilities"]]
        return {"catalog_version": self._manager.catalog.catalog_version,
                "servers": summaries, "count": len(summaries)}


class ProvisionPlanTool(_ManagerTool):
    name = "mcp.provision.plan"
    description = ("Prepare an installation plan for an approved catalog server. "
                   "Read-only: it installs nothing and requires approval to proceed.")
    input_schema = {
        "type": "object",
        "properties": {
            "catalog_id": {"type": "string", "description": "Trusted catalog id."},
            "directory": {"type": "string",
                          "description": "Directory the server should be allowed to access."},
        },
        "required": ["catalog_id"],
    }
    permission = ToolPermission.READ
    timeout_seconds = 30.0

    def _run(self, arguments):
        catalog_id = _identifier(arguments, "catalog_id")
        directory = arguments.get("directory")
        directories = [directory] if isinstance(directory, str) and directory.strip() else []
        plan = self._manager.prepare_plan(catalog_id, requested_directories=directories)
        return {
            "plan_id": plan.plan_id,
            "plan_hash": plan.plan_hash,
            "catalog_id": plan.catalog_id,
            "server_id": plan.server_id,
            "package": f"{plan.package_name}@{plan.package_version}",
            "install_directory": str(plan.install_directory),
            "approved_directories": [str(d) for d in plan.requested_directories],
            "read_tools": list(plan.read_tools()),
            "write_tools": list(plan.write_tools()),
            "denied_tools": list(plan.denied_tools()),
            "summary": list(plan.summary_lines()),
            "next_step": "Call mcp.provision.install with this plan_id and plan_hash to "
                         "request installation approval.",
        }


class ProvisionInstallTool(_ManagerTool):
    name = "mcp.provision.install"
    description = ("Install a previously planned MCP server. Requires confirmation; "
                   "only a plan_id/plan_hash from mcp.provision.plan is accepted.")
    input_schema = {
        "type": "object",
        "properties": {
            "plan_id": {"type": "string", "description": "Plan id from mcp.provision.plan."},
            "plan_hash": {"type": "string", "description": "Plan hash from mcp.provision.plan."},
        },
        "required": ["plan_id", "plan_hash"],
    }
    permission = ToolPermission.WRITE

    def confirmation_summary(self, arguments: dict) -> str:
        """Deterministic plan text — built from the stored plan, never model output."""
        plan_id = arguments.get("plan_id")
        plan = self._manager.get_plan(plan_id) if isinstance(plan_id, str) else None
        if plan is None:
            return f"Install an MCP server for unknown plan {plan_id!r} (this will fail)."
        return "\n".join(plan.summary_lines())

    def _run(self, arguments):
        plan_id = _identifier(arguments, "plan_id")
        plan_hash = _identifier(arguments, "plan_hash")
        plan = self._manager.get_plan(plan_id)
        if plan is None:
            raise McpError(MCP_PROVISIONING_PLAN_INVALID,
                           "That provisioning plan is unknown or has expired; prepare a new one.")
        # The user already approved this exact plan text through the Phase C prompt,
        # and the executor bound that approval to these arguments; the installer
        # still re-validates the hash.
        approval = ProvisioningApproval(approved=True, plan_id=plan.plan_id,
                                        plan_hash=plan_hash)
        result = self._manager.provision(plan, approval=approval)
        validation = result.get("validation") or {}
        return {
            "installed": True,
            "server_id": result["server_id"],
            "catalog_id": result["catalog_id"],
            "version": result["version"],
            "registered_tool_count": validation.get("registered_tool_count"),
            "denied_tool_count": validation.get("denied_tool_count"),
            "reused_existing_installation": result.get("reused_existing_installation", False),
            "note": "The server is active. Re-run the original request so the normal "
                    "tool pipeline can use the new tools.",
        }


class ProvisionStatusTool(_ManagerTool):
    name = "mcp.provision.status"
    description = "Report installed MCP servers and provisioning state. Read-only."
    input_schema = {"type": "object", "properties": {}}
    permission = ToolPermission.READ
    timeout_seconds = 15.0

    def _run(self, arguments):
        return self._manager.status()


class _ServerActionTool(_ManagerTool):
    input_schema = {
        "type": "object",
        "properties": {"server_id": {"type": "string", "description": "Installed server id."}},
        "required": ["server_id"],
    }
    permission = ToolPermission.WRITE
    timeout_seconds = 120.0
    _action_verb = "act on"

    def confirmation_summary(self, arguments: dict) -> str:
        server_id = arguments.get("server_id")
        return f"{self._action_verb} the managed MCP server '{server_id}'."


class ServerEnableTool(_ServerActionTool):
    name = "mcp.server.enable"
    description = "Re-activate an already-installed MCP server. Never reinstalls."
    _action_verb = "Enable"

    def _run(self, arguments):
        server_id = _identifier(arguments, "server_id")
        result = self._manager.enable(server_id)
        return {"server_id": server_id, "status": result["status"], "reinstalled": False}


class ServerDisableTool(_ServerActionTool):
    name = "mcp.server.disable"
    description = "Stop using an installed MCP server. Installed files are preserved."
    _action_verb = "Disable"

    def _run(self, arguments):
        server_id = _identifier(arguments, "server_id")
        result = self._manager.disable(server_id)
        return {"server_id": server_id, "status": result["status"]}


class ServerRepairTool(_ServerActionTool):
    name = "mcp.server.repair"
    description = ("Verify an installed MCP server and restore the SAME pinned version "
                   "if files are missing. Never upgrades.")
    _action_verb = "Repair"

    def _run(self, arguments):
        server_id = _identifier(arguments, "server_id")
        result = self._manager.repair(server_id)
        return {"server_id": server_id, "reinstalled": bool(result.get("reinstalled")),
                "status": result.get("status")}


class ServerUninstallTool(_ServerActionTool):
    name = "mcp.server.uninstall"
    description = ("Remove a managed MCP server installation. Files in approved "
                   "directories are never deleted.")
    _action_verb = "Uninstall"

    def confirmation_summary(self, arguments: dict) -> str:
        server_id = arguments.get("server_id")
        return (f"Uninstall the managed MCP server '{server_id}'. This removes the managed "
                "installation only — files in your approved directories are not touched.")

    def _run(self, arguments):
        server_id = _identifier(arguments, "server_id")
        result = self._manager.uninstall(server_id)
        return {"server_id": server_id, "removed": bool(result.get("removed")),
                "audit_preserved": bool(result.get("audit_preserved"))}


ALL_PROVISIONING_TOOL_CLASSES = (
    CatalogSearchTool,
    ProvisionPlanTool,
    ProvisionInstallTool,
    ProvisionStatusTool,
    ServerEnableTool,
    ServerDisableTool,
    ServerRepairTool,
    ServerUninstallTool,
)


def register_provisioning_tools(registry, manager):
    """Register the provisioning tools into a ToolRegistry. Returns the tools."""
    tools = []
    for tool_cls in ALL_PROVISIONING_TOOL_CLASSES:
        tool = tool_cls(manager)
        if registry.has(tool.name):
            continue
        registry.register(tool)
        tools.append(tool)
    return tools
