"""McpClient — a minimal synchronous MCP client over a subprocess's stdio.

Speaks newline-delimited JSON-RPC 2.0. A background reader thread drains the
server's stdout onto a queue; request/response is correlated by JSON-RPC id with a
deadline. Every failure mode is surfaced as a controlled McpError code:
MCP_STARTUP_FAILED, MCP_TIMEOUT, MCP_SERVER_EXITED, MCP_TOOL_NOT_FOUND,
MCP_CALL_FAILED, MCP_INVALID_RESPONSE.
"""

import json
import queue
import subprocess
import threading
import time

from mcp_layer.errors import McpError
from tools.models import (
    MCP_CALL_FAILED,
    MCP_INVALID_RESPONSE,
    MCP_SERVER_EXITED,
    MCP_STARTUP_FAILED,
    MCP_TIMEOUT,
    MCP_TOOL_NOT_FOUND,
)

PROTOCOL_VERSION = "2024-11-05"
_EOF = object()  # sentinel: the server's stdout closed (server exited)


class McpClient:
    def __init__(self, command, cwd=None, env=None, default_call_timeout=20.0):
        self._command = list(command)
        self._cwd = cwd
        self._env = env
        self.default_call_timeout = default_call_timeout
        self._proc = None
        self._queue = queue.Queue()
        self._reader = None
        self._id = 0
        self._write_lock = threading.Lock()

    # ---- lifecycle ----

    def start(self, timeout=15.0):
        """Launch the server subprocess and run the MCP initialize handshake."""
        try:
            self._proc = subprocess.Popen(
                self._command, cwd=self._cwd, env=self._env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except (OSError, ValueError) as e:
            raise McpError(MCP_STARTUP_FAILED, "Failed to start the MCP server.") from e

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        try:
            self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "local-llm-project", "version": "0.1.0"},
            }, timeout)
        except McpError as e:
            # Any handshake failure is a startup failure from the caller's view.
            self.shutdown()
            if e.code in (MCP_TIMEOUT, MCP_SERVER_EXITED):
                raise McpError(MCP_STARTUP_FAILED, "The MCP server did not initialize.") from e
            raise
        self._notify("notifications/initialized", {})

    def shutdown(self):
        """Close stdin (signaling the server to exit) and reap the process."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    # ---- MCP methods ----

    def list_tools(self, timeout=15.0):
        result = self._request("tools/list", {}, timeout)
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise McpError(MCP_INVALID_RESPONSE, "The MCP server returned an invalid tools list.")
        return tools

    def call_tool(self, name, arguments, timeout=None):
        """Call an MCP tool and return the normalized result dict (ToolResult.data)."""
        timeout = self.default_call_timeout if timeout is None else timeout
        result = self._request("tools/call", {"name": name, "arguments": arguments or {}}, timeout)
        if result.get("isError"):
            raise McpError(MCP_CALL_FAILED, _extract_text(result) or "The MCP tool reported an error.")
        return _normalize_result(result)

    # ---- transport internals ----

    def _read_loop(self):
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._queue.put(json.loads(line))
                except json.JSONDecodeError:
                    continue
        finally:
            self._queue.put(_EOF)

    def _next_id(self):
        self._id += 1
        return self._id

    def _write(self, obj):
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise McpError(MCP_SERVER_EXITED, "The MCP server is not running.")
        try:
            with self._write_lock:
                proc.stdin.write(json.dumps(obj) + "\n")
                proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise McpError(MCP_SERVER_EXITED, "The MCP server has exited.") from e

    def _notify(self, method, params):
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method, params, timeout):
        rid = self._next_id()
        self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise McpError(MCP_TIMEOUT, f"The MCP server did not respond within {timeout:g}s.",
                               retryable=True)
            try:
                message = self._queue.get(timeout=remaining)
            except queue.Empty:
                raise McpError(MCP_TIMEOUT, f"The MCP server did not respond within {timeout:g}s.",
                               retryable=True)
            if message is _EOF:
                raise McpError(MCP_SERVER_EXITED, "The MCP server exited unexpectedly.")
            if not isinstance(message, dict) or message.get("id") != rid:
                continue  # a notification or a stale/other-id message — ignore
            if "error" in message and message["error"] is not None:
                err = message["error"] or {}
                code = (MCP_TOOL_NOT_FOUND if err.get("code") == -32601 and method == "tools/call"
                        else MCP_CALL_FAILED)
                raise McpError(code, err.get("message", "The MCP request failed."))
            result = message.get("result")
            if not isinstance(result, dict):
                raise McpError(MCP_INVALID_RESPONSE, "The MCP server returned an invalid response.")
            return result


def _extract_text(result):
    parts = [b.get("text", "") for b in result.get("content", [])
             if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(p for p in parts if p).strip()


def _normalize_result(result):
    """Turn an MCP tools/call result into a plain dict for ToolResult.data."""
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    text = _extract_text(result)
    if text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            pass
    return {"text": text}
