"""McpTool — an MCP tool exposed as an ordinary BaseTool.

Discovery builds one McpTool per remote tool. It carries the standard BaseTool
surface (name, description, input_schema, permission) so the registry shortlists
it and the executor runs it with no MCP awareness. execute() delegates to the MCP
client and translates McpError into ToolFailure, which the executor normalizes
into a ToolResult exactly like any built-in tool's controlled failure.
"""

from mcp_layer.errors import McpError
from tools.base import BaseTool, ToolFailure
from tools.models import ToolPermission


class McpTool(BaseTool):
    def __init__(self, registry_name, remote_name, description, input_schema, permission,
                 client, server_label="test", call_timeout=20.0, timeout_seconds=60.0,
                 session_owner=None):
        self.name = registry_name
        self.description = description or f"MCP tool {remote_name}"
        self.input_schema = input_schema or {"type": "object", "properties": {}}
        self.permission = ToolPermission.coerce(permission)
        self.llm_callable = True
        self.timeout_seconds = timeout_seconds  # executor thread backstop
        self.call_timeout = call_timeout        # per-call MCP timeout (fires first)
        # Which McpSession discovered/registered this tool (an opaque session id, or
        # None for tools not owned by any session). Lets a runtime replacement remove
        # exactly the tools a stale session registered — see
        # tools.registry.ToolRegistry.unregister_owned and mcp_layer.runtime_manager.
        self.session_owner = session_owner
        self._remote_name = remote_name
        self._server_label = server_label
        self._client = client

    def confirmation_summary(self, arguments: dict) -> str:
        # Deterministic, built only from the tool identity + argument keys — never
        # from server/tool output.
        keys = ", ".join(sorted(arguments)) if isinstance(arguments, dict) and arguments else "no arguments"
        return (f"Run the MCP tool '{self._remote_name}' on the '{self._server_label}' "
                f"server ({keys}).")

    def execute(self, arguments: dict) -> dict:
        try:
            return self._client.call_tool(self._remote_name, arguments, timeout=self.call_timeout)
        except McpError as e:
            raise ToolFailure(e.code, e.message, retryable=e.retryable)
