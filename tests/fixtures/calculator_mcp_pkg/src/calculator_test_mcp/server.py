"""Phase G.3 Task 19 — test-only fixture MCP server: `add` and `echo` over
stdio JSON-RPC 2.0 (newline-delimited), matching the same wire protocol
`mcp_layer.client.McpClient` speaks to every other MCP server in this project.
Never a production capability — installed only from a committed local wheel,
never a network package index.
"""

import json
import sys

PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "add",
        "description": "Add two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "echo",
        "description": "Echo a string back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]


def _ok(structured):
    return {
        "content": [{"type": "text", "text": json.dumps(structured)}],
        "structuredContent": structured,
        "isError": False,
    }


def _tool_error(message):
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _call_tool(name, args):
    args = args or {}
    if name == "add":
        return _ok({"result": args.get("a", 0) + args.get("b", 0)})
    if name == "echo":
        return _ok({"result": args.get("text", "")})
    return None


def _handle(message):
    method = message.get("method")
    msg_id = message.get("id")
    params = message.get("params") or {}
    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "calculator-test-mcp", "version": "1.0.0"},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        try:
            result = _call_tool(params.get("name"), params.get("arguments"))
        except Exception as e:  # noqa: BLE001 — report as a normal tool error
            return {"jsonrpc": "2.0", "id": msg_id, "result": _tool_error(str(e))}
        if result is None:
            return {"jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {params.get('name')}"}}
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}
    if msg_id is None:
        return None
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": "Method not found"}}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        response = _handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
