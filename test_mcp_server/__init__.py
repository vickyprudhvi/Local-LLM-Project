"""Internal deterministic MCP test server (Phase D).

NOT production code. It implements just enough of the Model Context Protocol
(initialize, tools/list, tools/call) over newline-delimited JSON-RPC on stdio to
exercise the assistant's MCP client. It never touches files outside its configured
workspace.
"""
