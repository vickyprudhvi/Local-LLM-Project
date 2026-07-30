"""McpClient — a minimal synchronous MCP client over a subprocess's stdio.

Speaks newline-delimited JSON-RPC 2.0. A background reader thread drains the
server's stdout onto a queue; request/response is correlated by JSON-RPC id with a
deadline. Every failure mode is surfaced as a controlled McpError code:
MCP_STARTUP_FAILED, MCP_TIMEOUT, MCP_SERVER_EXITED, MCP_TOOL_NOT_FOUND,
MCP_CALL_FAILED, MCP_INVALID_RESPONSE.
"""

import collections
import json
import queue
import re
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
MAX_OUTPUT_BYTES = 100 * 1024   # cap on a normalized tool result
_STDERR_MAX_LINES = 200         # bounded stderr retention for diagnostics
_STDERR_MAX_CHARS = 8 * 1024
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class McpClient:
    def __init__(self, command, cwd=None, env=None, default_call_timeout=20.0, shutdown_timeout=5.0):
        self._command = list(command)
        self._cwd = str(cwd) if cwd is not None else None
        self._env = env
        self.default_call_timeout = default_call_timeout
        self.shutdown_timeout = shutdown_timeout
        self._proc = None
        self._queue = queue.Queue()
        self._reader = None
        self._stderr_reader = None
        self._stderr_buf = collections.deque(maxlen=_STDERR_MAX_LINES)
        self._id = 0
        self._write_lock = threading.Lock()
        # Populated by start(): the server's initialize result (Phase F validates the
        # negotiated protocol version and records serverInfo). No secrets.
        self.initialize_result = None

    @property
    def protocol_version(self):
        return (self.initialize_result or {}).get("protocolVersion")

    @property
    def server_info(self):
        return (self.initialize_result or {}).get("serverInfo") or {}

    # ---- lifecycle ----

    def start(self, timeout=15.0):
        """Launch the server subprocess (shell=False) and run the initialize handshake."""
        try:
            self._proc = subprocess.Popen(
                self._command, cwd=self._cwd, env=self._env, shell=False,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )
        except (OSError, ValueError) as e:
            raise McpError(MCP_STARTUP_FAILED, "Failed to start the MCP server.") from e

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        # Drain stderr onto a bounded buffer so it never fills the OS pipe (which
        # would block the child) and is never mistaken for a protocol message.
        self._stderr_reader = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_reader.start()
        try:
            self.initialize_result = self._request("initialize", {
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
        """Terminate the child and join reader threads. Idempotent and never raises."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=self.shutdown_timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=self.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    pass
        for thread in (self._reader, self._stderr_reader):
            if thread is not None:
                thread.join(timeout=self.shutdown_timeout)
        # Drop any late/queued messages so a stale response can't satisfy a future call.
        self._drain_queue()

    def recent_stderr(self):
        """Bounded, sanitized recent stderr for diagnostics only — never sent to the LLM."""
        joined = "".join(self._stderr_buf)
        joined = _CONTROL_RE.sub("", joined)
        return joined[-_STDERR_MAX_CHARS:]

    def _drain_queue(self):
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

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
        data = _normalize_result(result)
        try:
            if len(json.dumps(data)) > MAX_OUTPUT_BYTES:
                raise McpError(MCP_INVALID_RESPONSE, "The MCP tool returned an oversized result.")
        except (TypeError, ValueError):
            raise McpError(MCP_INVALID_RESPONSE, "The MCP tool returned a non-serializable result.")
        return data

    # ---- transport internals ----

    def _stderr_loop(self):
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                self._stderr_buf.append(line)
        except (OSError, ValueError):
            pass

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
