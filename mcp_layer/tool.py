"""McpTool — an MCP tool exposed as an ordinary BaseTool.

Discovery builds one McpTool per remote tool. It carries the standard BaseTool
surface (name, description, input_schema, permission) so the registry shortlists
it and the executor runs it with no MCP awareness. execute() delegates to the MCP
client and translates McpError into ToolFailure, which the executor normalizes
into a ToolResult exactly like any built-in tool's controlled failure.
"""

import os
from typing import Optional

from mcp_layer.errors import McpError
from tools.base import BaseTool, ToolFailure
from tools.models import ToolPermission


_MAX_MARKDOWN_OUTPUT_BYTES = 1 * 1024 * 1024


class McpTool(BaseTool):
    def __init__(self, registry_name, remote_name, description, input_schema, permission,
                 client, server_label="test", call_timeout=20.0, timeout_seconds=60.0,
                 session_owner=None, invocation_policy=None):
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
        self._invocation_policy = invocation_policy or {}

    def confirmation_summary(self, arguments: dict) -> str:
        # Deterministic, built only from the tool identity + argument keys — never
        # from server/tool output.
        keys = ", ".join(sorted(arguments)) if isinstance(arguments, dict) and arguments else "no arguments"
        return (f"Run the MCP tool '{self._remote_name}' on the '{self._server_label}' "
                f"server ({keys}).")

    def execute(self, arguments: dict) -> dict:
        transformed, auth_id = self._apply_invocation_policy(arguments)
        try:
            result = self._client.call_tool(self._remote_name, transformed, timeout=self.call_timeout)
            return self._normalize_result(result)
        except McpError as e:
            raise ToolFailure(e.code, e.message, retryable=e.retryable)
        finally:
            if auth_id:
                try:
                    from mcp_management.document_authorization import DocumentAuthorizationStore
                    DocumentAuthorizationStore.default().consume_authorization(auth_id)
                except Exception:  # noqa: BLE001 — never mask the real call outcome
                    pass

    def _normalize_result(self, result) -> dict:
        """Normalize and enforce size for document-conversion results.

        For exact_file_uri policy, only the verified {'text': '<markdown>'} shape is
        accepted; oversized output is rejected without truncation.
        """
        if self._invocation_policy.get("argument_mode") != "exact_file_uri":
            return result if isinstance(result, dict) else {}
        text = _extract_text(result)
        if text is None:
            from tools.models import MCP_DOCUMENT_NORMALIZATION_FAILED
            raise ToolFailure(
                MCP_DOCUMENT_NORMALIZATION_FAILED,
                "Document conversion result did not contain a text content item.",
            )
        if len(text.encode("utf-8")) > _MAX_MARKDOWN_OUTPUT_BYTES:
            from tools.models import MCP_DOCUMENT_OUTPUT_TOO_LARGE
            raise ToolFailure(
                MCP_DOCUMENT_OUTPUT_TOO_LARGE,
                "Document conversion result exceeds the maximum allowed size.",
            )
        return {"text": text}

    def _apply_invocation_policy(self, arguments: dict):
        """Fail-closed pre-call transformation for tools with a catalog invocation policy.

        Returns (transformed_arguments, reserved_auth_id).  auth_id is None when no
        policy applies so the caller can skip consume_authorization.
        """
        if not self._invocation_policy:
            return arguments, None
        mode = self._invocation_policy.get("argument_mode")
        if mode == "exact_file_uri":
            from mcp_management.document_authorization import (
                DocumentAuthorizationStore,
                _file_uri_to_path,
            )
            from tools.models import (
                MCP_DOCUMENT_AUTHORIZATION_REQUIRED,
                MCP_DOCUMENT_PATH_INVALID,
                MCP_DOCUMENT_PATH_RESTRICTED,
            )

            if not isinstance(arguments, dict) or set(arguments.keys()) != {"uri"}:
                raise ToolFailure(
                    MCP_DOCUMENT_PATH_INVALID,
                    "convert_to_markdown requires exactly one 'uri' argument.",
                )
            supplied = arguments["uri"]
            if not isinstance(supplied, str) or not supplied.strip():
                raise ToolFailure(
                    MCP_DOCUMENT_PATH_INVALID,
                    "Document URI must be a non-empty string.",
                )
            # Reject true remote URIs.  file:// is accepted but its local path is
            # checked against an active authorization before it reaches the server.
            lower = supplied.lower()
            if any(lower.startswith(p) for p in ("http://", "https://", "ftp://", "ftps://")):
                raise ToolFailure(
                    MCP_DOCUMENT_PATH_RESTRICTED,
                    "Remote document URIs are not allowed.",
                )
            if supplied.startswith("\\\\") or supplied.startswith("//"):
                raise ToolFailure(
                    MCP_DOCUMENT_PATH_RESTRICTED,
                    "UNC paths are not allowed for document conversion.",
                )

            supplied_path = _file_uri_to_path(supplied) or supplied
            if not os.path.isabs(supplied_path):
                raise ToolFailure(
                    MCP_DOCUMENT_PATH_INVALID,
                    "Document URI must resolve to an absolute local path.",
                )

            # Find a matching, unconsumed authorization and reserve it for this call.
            store = DocumentAuthorizationStore.default()
            auth = store.find_and_reserve_for_path(supplied_path)
            if auth is None:
                raise ToolFailure(
                    MCP_DOCUMENT_AUTHORIZATION_REQUIRED,
                    "No active authorization for the requested document.",
                )
            # Replace the model-provided value with the trusted file:// URI from the snapshot.
            from mcp_management.document_authorization import _path_to_file_uri
            trusted_uri = _path_to_file_uri(auth.snapshot.local_path)
            return {"uri": trusted_uri}, auth.auth_id
        return arguments, None


def _extract_text(result) -> Optional[str]:
    """Extract text from the verified {'text': '<markdown>'} result shape."""
    if not isinstance(result, dict):
        return None
    if "text" in result and isinstance(result["text"], str):
        return result["text"]
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                return item["text"]
    return None
