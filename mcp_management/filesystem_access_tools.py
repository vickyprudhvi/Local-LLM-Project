"""Phase F.1 — filesystem access-management exposed as ordinary built-in tools.

Plain BaseTool subclasses, run through the existing ToolExecutor with the existing
Phase C permission/confirmation logic — same pattern as mcp_management.provisioning_tools.
They accept only structured, trusted identifiers (a server id, a plan id/hash, a
directory) that deterministic code then screens; never a command, package name,
npm argument, environment variable, or raw config object. Adding or removing a root
is a CONFIGURATION CHANGE on an already-installed server, never package provisioning —
these tools never call npm and are distinct from mcp.provision.*.

Descriptions deliberately reuse the vocabulary a user would say ("give access to
this folder", "allow this path", "expand allowed roots", "remove folder access")
so the existing lexical Phase B shortlist (tools.registry.shortlist_tools) surfaces
them for a matching request with no change to the shortlisting mechanism itself.
"""

from mcp_layer.errors import McpError
from mcp_management.filesystem_access import FilesystemAccessApproval, FilesystemAccessOperation
from tools.base import BaseTool, ToolFailure, ToolValidationError
from tools.models import MCP_FILESYSTEM_ACCESS_PLAN_INVALID, ToolPermission

_ID_MAX = 1024


def _identifier(arguments, key, required=True, max_len=_ID_MAX):
    value = arguments.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ToolValidationError(f"'{key}' must be a non-empty string.")
    if len(value) > max_len or any(c in value for c in ("\x00", "\n", "\r", "\t")):
        raise ToolValidationError(f"'{key}' is not a valid value.")
    return value.strip()


class _AccessManagerTool(BaseTool):
    """Base for filesystem access-management tools; normalizes McpError."""

    llm_callable = True
    timeout_seconds = 120.0

    def __init__(self, manager):
        self._manager = manager

    def _run(self, arguments):
        raise NotImplementedError

    def execute(self, arguments: dict) -> dict:
        try:
            return self._run(arguments)
        except McpError as e:
            raise ToolFailure(e.code, e.message, retryable=e.retryable)


class FilesystemAccessListTool(_AccessManagerTool):
    name = "mcp.filesystem.access.list"
    description = ("List the directories an already-installed filesystem MCP server is "
                   "currently approved to access. Read-only; changes nothing.")
    input_schema = {
        "type": "object",
        "properties": {
            "server_id": {"type": "string", "description": "Installed server id, e.g. 'filesystem'."},
        },
        "required": ["server_id"],
    }
    permission = ToolPermission.READ
    timeout_seconds = 15.0

    def _run(self, arguments):
        server_id = _identifier(arguments, "server_id")
        directories = self._manager.list_filesystem_access(server_id)
        return {"server_id": server_id, "approved_directories": list(directories)}


class FilesystemAccessPlanTool(_AccessManagerTool):
    name = "mcp.filesystem.access.plan"
    description = (
        "Prepare a plan to give access to a folder, allow a path, add a folder, "
        "reconfigure filesystem access, approve a directory, use a directory, or "
        "expand the allowed roots for an ALREADY-INSTALLED filesystem MCP server. "
        "Use this — never mcp.provision.plan — when the server is already installed "
        "and a request was rejected for being outside its approved directories. "
        "Read-only: nothing changes until mcp.filesystem.access.add is approved."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "server_id": {"type": "string", "description": "Installed server id, e.g. 'filesystem'."},
            "directory": {"type": "string", "description": "The directory to grant (or revoke) access to."},
            "operation": {"type": "string", "description": "'add' (default) or 'remove'."},
        },
        "required": ["server_id", "directory"],
    }
    permission = ToolPermission.READ
    timeout_seconds = 30.0

    def _run(self, arguments):
        server_id = _identifier(arguments, "server_id")
        directory = _identifier(arguments, "directory")
        operation_raw = arguments.get("operation") or "add"
        if operation_raw not in ("add", "remove"):
            raise ToolValidationError("'operation' must be 'add' or 'remove'.")
        operation = (FilesystemAccessOperation.ADD_ROOT if operation_raw == "add"
                    else FilesystemAccessOperation.REMOVE_ROOT)
        plan = self._manager.prepare_filesystem_access_plan(server_id, directory, operation=operation)
        return {
            "plan_id": plan.plan_id,
            "plan_hash": plan.plan_hash,
            "server_id": plan.server_id,
            "operation": operation_raw,
            "requested_directory": plan.requested_directory,
            "current_allowed_directories": list(plan.current_allowed_directories),
            "proposed_allowed_directories": list(plan.proposed_allowed_directories),
            "summary": list(plan.summary_lines()),
            "next_step": "Call mcp.filesystem.access.add with this plan_id and plan_hash to "
                         "request approval.",
        }


class FilesystemAccessAddTool(_AccessManagerTool):
    name = "mcp.filesystem.access.add"
    description = (
        "Apply a previously prepared filesystem-access plan (allow this path, approve the "
        "folder, expand allowed roots) — adds or removes a directory from an ALREADY-INSTALLED "
        "server's approved roots, per the plan's own operation. Requires confirmation. Never "
        "reinstalls the MCP package or touches npm."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "server_id": {"type": "string"},
            "plan_id": {"type": "string", "description": "Plan id from mcp.filesystem.access.plan."},
            "plan_hash": {"type": "string", "description": "Plan hash from mcp.filesystem.access.plan."},
        },
        "required": ["server_id", "plan_id", "plan_hash"],
    }
    permission = ToolPermission.WRITE

    def confirmation_summary(self, arguments: dict) -> str:
        plan_id = arguments.get("plan_id")
        plan = self._manager.get_filesystem_access_plan(plan_id) if isinstance(plan_id, str) else None
        if plan is None:
            return f"Change filesystem access for unknown plan {plan_id!r} (this will fail)."
        return "\n".join(plan.summary_lines())

    def _run(self, arguments):
        server_id = _identifier(arguments, "server_id")
        plan_id = _identifier(arguments, "plan_id")
        plan_hash = _identifier(arguments, "plan_hash")
        plan = self._manager.get_filesystem_access_plan(plan_id)
        if plan is None or plan.server_id != server_id:
            raise McpError(MCP_FILESYSTEM_ACCESS_PLAN_INVALID,
                           "That filesystem access plan is unknown or has expired; prepare a new one.")
        approval = FilesystemAccessApproval(approved=True, plan_id=plan.plan_id, plan_hash=plan_hash)
        result = self._manager.apply_filesystem_access(plan, approval=approval)
        return {
            "server_id": result["server_id"],
            "approved_directories": result["approved_directories"],
            "note": "The server's approved directories were updated; the MCP package was NOT "
                    "reinstalled. Re-run the original request so the normal tool pipeline can "
                    "use the new access.",
        }


class FilesystemAccessRemoveTool(_AccessManagerTool):
    name = "mcp.filesystem.access.remove"
    description = (
        "Remove folder access: revoke a previously approved directory from an "
        "already-installed filesystem MCP server. Requires confirmation. Refuses to "
        "remove the last remaining approved directory. Never reinstalls the package."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "server_id": {"type": "string"},
            "directory": {"type": "string", "description": "The currently approved directory to revoke."},
        },
        "required": ["server_id", "directory"],
    }
    permission = ToolPermission.WRITE

    def confirmation_summary(self, arguments: dict) -> str:
        server_id = arguments.get("server_id")
        directory = arguments.get("directory")
        return f"Remove approved directory '{directory}' from the '{server_id}' MCP server."

    def _run(self, arguments):
        server_id = _identifier(arguments, "server_id")
        directory = _identifier(arguments, "directory")
        plan = self._manager.prepare_filesystem_access_plan(
            server_id, directory, operation=FilesystemAccessOperation.REMOVE_ROOT)
        approval = FilesystemAccessApproval(approved=True, plan_id=plan.plan_id,
                                            plan_hash=plan.plan_hash)
        result = self._manager.apply_filesystem_access(plan, approval=approval)
        return {"server_id": result["server_id"], "approved_directories": result["approved_directories"]}


ALL_FILESYSTEM_ACCESS_TOOL_CLASSES = (
    FilesystemAccessListTool,
    FilesystemAccessPlanTool,
    FilesystemAccessAddTool,
    FilesystemAccessRemoveTool,
)


def register_filesystem_access_tools(registry, manager):
    """Register the filesystem access-management tools into a ToolRegistry."""
    tools = []
    for tool_cls in ALL_FILESYSTEM_ACCESS_TOOL_CLASSES:
        tool = tool_cls(manager)
        if registry.has(tool.name):
            continue
        registry.register(tool)
        tools.append(tool)
    return tools
