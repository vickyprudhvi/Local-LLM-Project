"""The internal test MCP server: initialize / tools/list / tools/call over stdio.

Transport: one JSON-RPC 2.0 message per line on stdin/stdout (the MCP stdio
framing). Six deterministic tools. All filesystem access is confined to the
workspace directory (TEST_MCP_WORKSPACE); any path escaping it is rejected.
"""

import json
import os
import sys
import time

PROTOCOL_VERSION = "2024-11-05"


def _workspace():
    # Phase E launches with cwd = the isolated workspace (no env var). Phase D sets
    # TEST_MCP_WORKSPACE explicitly. Fall back to cwd, never a hardcoded subdir.
    return os.path.realpath(os.environ.get("TEST_MCP_WORKSPACE") or os.getcwd())


# name, description, permission (advertised via annotations), inputSchema
TOOLS = [
    {
        "name": "echo_text",
        "description": "Echo the provided text back unchanged.",
        "permission": "read",
        "inputSchema": {"type": "object",
                        "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
    {
        "name": "add_numbers",
        "description": "Add two numbers and return their sum.",
        "permission": "read",
        "inputSchema": {"type": "object",
                        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                        "required": ["a", "b"]},
    },
    {
        "name": "read_test_file",
        "description": "Read a text file from the test workspace.",
        "permission": "read",
        "inputSchema": {"type": "object",
                        "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    {
        "name": "write_test_file",
        "description": "Write a text file into the test workspace.",
        "permission": "write",
        "inputSchema": {"type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path"]},
    },
    {
        "name": "fail_tool",
        "description": "Always fails; used to exercise error normalization.",
        "permission": "read",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "slow_tool",
        "description": "Sleeps longer than the client timeout; used for timeout tests.",
        "permission": "read",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _resolve_in_workspace(path):
    if not isinstance(path, str) or not path.strip():
        raise ValueError("A relative 'path' is required.")
    if os.path.isabs(path):
        raise ValueError("Absolute paths are not allowed.")
    base = _workspace()
    full = os.path.realpath(os.path.join(base, path))
    if full != base and not full.startswith(base + os.sep):
        raise ValueError("Path escapes the test workspace.")
    return full


def _ok(structured):
    """A tools/call success result: a text block plus structured content."""
    return {"content": [{"type": "text", "text": json.dumps(structured)}],
            "structuredContent": structured, "isError": False}


def _tool_error(message):
    """A tools/call error result (isError=true), per MCP tool-error convention."""
    return {"content": [{"type": "text", "text": message}], "isError": True}


def call_tool(name, args):
    """Return an MCP tools/call result dict, or None if the tool is unknown."""
    args = args or {}
    if name == "echo_text":
        return _ok({"text": args.get("text", "")})
    if name == "add_numbers":
        return _ok({"sum": (args.get("a", 0) or 0) + (args.get("b", 0) or 0)})
    if name == "read_test_file":
        full = _resolve_in_workspace(args.get("path"))
        if not os.path.isfile(full):
            return _tool_error(f"File not found: {args.get('path')}")
        with open(full, "r", encoding="utf-8") as f:
            return _ok({"content": f.read()})
    if name == "write_test_file":
        full = _resolve_in_workspace(args.get("path"))
        os.makedirs(os.path.dirname(full) or _workspace(), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(args.get("content", ""))
        return _ok({"written": True, "path": args.get("path")})
    if name == "fail_tool":
        return _tool_error("fail_tool always fails.")
    if name == "slow_tool":
        time.sleep(float(os.environ.get("TEST_MCP_SLOW_SECONDS", "3")))
        return _ok({"done": True})
    return None


def handle(message):
    """Return a JSON-RPC response dict, or None for notifications."""
    method = message.get("method")
    mid = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "test-mcp-server", "version": "0.1.0"},
        }}
    if method == "notifications/initialized":
        return None  # notification: no response
    if method == "tools/list":
        tools = [{"name": t["name"], "description": t["description"],
                  "inputSchema": t["inputSchema"],
                  "annotations": {"permission": t["permission"]}} for t in TOOLS]
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": tools}}
    if method == "tools/call":
        name = params.get("name")
        try:
            result = call_tool(name, params.get("arguments"))
        except Exception as e:  # noqa: BLE001 — surface as a tool error, never crash
            return {"jsonrpc": "2.0", "id": mid, "result": _tool_error(f"{type(e).__name__}: {e}")}
        if result is None:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32601, "message": f"Unknown tool: {name}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    if mid is not None:
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return None  # unknown notification


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
