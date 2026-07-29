"""Phase D — MCP (Model Context Protocol) client layer.

A minimal, synchronous MCP client that speaks newline-delimited JSON-RPC 2.0 over
a subprocess's stdio. It plugs MCP tools into the EXISTING architecture: each
discovered MCP tool becomes an McpTool (a BaseTool) registered in the shared
ToolRegistry, so the existing ToolExecutor runs it through the exact same path
(permission gate, confirmation, timeout, ToolResult) with no MCP-specific branch.

Synchronous by design: the executor is synchronous and thread-based, so a sync
client is a cleaner fit than the async official SDK. Only the protocol subset the
test server needs (initialize, tools/list, tools/call) is implemented.
"""

from mcp_layer.client import McpClient
from mcp_layer.errors import McpError
from mcp_layer.integration import (
    McpSession,
    bootstrap_test_server,
    discover_and_register,
    start_test_server,
)
from mcp_layer.tool import McpTool

__all__ = [
    "McpClient",
    "McpError",
    "McpSession",
    "McpTool",
    "bootstrap_test_server",
    "discover_and_register",
    "start_test_server",
]
