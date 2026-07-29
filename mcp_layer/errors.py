"""A single controlled MCP error type carrying one of the normalized MCP codes.

McpClient raises McpError(code, message); McpTool.execute translates it into a
ToolFailure so the existing executor produces a structured ToolResult. Messages
are always safe for the model — no stack traces, no secrets.
"""


class McpError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
