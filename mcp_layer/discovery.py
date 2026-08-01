"""Phase E — validate/sanitize discovered MCP tools and apply the LOCAL tool policy.

Tool metadata from the server (names, descriptions, schemas) is untrusted data.
This module bounds and sanitizes it, then decides — using ONLY the local
configuration — which tools register and with what permission. Server-advertised
permissions are ignored entirely.
"""

import json
import re

from mcp_layer.tool import McpTool
from tools.models import ToolPermission

NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
MAX_TOOLS = 100
MAX_DESCRIPTION_CHARS = 1000
MAX_SCHEMA_BYTES = 20 * 1024
MAX_SCHEMA_NESTING = 10
_VALID_SCHEMA_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


def sanitize_description(description) -> str:
    if not isinstance(description, str):
        return ""
    text = description.replace("\x00", "")
    # Drop control characters except tab/newline.
    text = "".join(ch for ch in text if ch in "\t\n" or ord(ch) >= 32)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:MAX_DESCRIPTION_CHARS]


def _schema_depth(node, depth=0):
    if depth > MAX_SCHEMA_NESTING:
        return depth
    if isinstance(node, dict):
        return max([depth] + [_schema_depth(v, depth + 1) for v in node.values()])
    if isinstance(node, list):
        return max([depth] + [_schema_depth(v, depth + 1) for v in node])
    return depth


def schema_rejection_reason(schema):
    """Return a skip reason for an invalid input schema, or None if it is valid."""
    if not isinstance(schema, dict):
        return "invalid_schema"
    declared = schema.get("type", "object")
    if declared != "object":
        return "invalid_schema"  # MCP tool input schemas are objects
    if declared is not None and declared not in _VALID_SCHEMA_TYPES:
        return "invalid_schema"
    if "properties" in schema and not isinstance(schema["properties"], dict):
        return "invalid_schema"
    try:
        serialized = json.dumps(schema)
    except (TypeError, ValueError):
        return "invalid_schema"
    if len(serialized) > MAX_SCHEMA_BYTES:
        return "oversized_schema"
    if _schema_depth(schema) > MAX_SCHEMA_NESTING:
        return "excessive_schema_depth"
    return None


def is_valid_input_schema(schema) -> bool:
    return schema_rejection_reason(schema) is None


# Diagnostic categories: denied (policy rejection), skipped (metadata invalid), or
# disabled (present in policy but turned off). Registered tools are none of these.
_DENY_REASONS = {"not_in_local_policy", "permission_denied"}
_SKIP_REASONS = {"invalid_name", "invalid_schema", "oversized_schema",
                 "excessive_schema_depth", "tool_limit_exceeded", "registration_collision"}
_DISABLED_REASONS = {"locally_disabled"}


def _category(reason):
    if reason in _DENY_REASONS:
        return "denied"
    if reason in _DISABLED_REASONS:
        return "disabled"
    return "skipped"


def plan_registration(raw_tools, config):
    """Return (registrations, diagnostics).

    registrations: list of dicts {remote_name, description, input_schema, permission}.
    diagnostics:   list of (name, reason, category) for every discovered tool NOT
                   registered — so discovered = registered + len(diagnostics), and
                   the counts stay internally consistent.

    A tool registers only when it is valid AND present in the local policy AND
    locally enabled AND its LOCAL permission is not denied. Everything else is
    denied / skipped / disabled and never registered — so it can never be
    shortlisted or reach the server. Server-advertised permissions are ignored.
    """
    registrations = []
    diagnostics = []
    if not isinstance(raw_tools, list):
        return registrations, diagnostics

    def note(name, reason):
        diagnostics.append((str(name)[:64], reason, _category(reason)))

    for index, spec in enumerate(raw_tools):
        if index >= MAX_TOOLS:
            note(spec.get("name") if isinstance(spec, dict) else spec, "tool_limit_exceeded")
            continue
        if not isinstance(spec, dict):
            note(spec, "invalid_name")
            continue
        remote_name = spec.get("name")
        if not isinstance(remote_name, str) or not NAME_RE.match(remote_name):
            note(remote_name, "invalid_name")
            continue
        reason = schema_rejection_reason(spec.get("inputSchema"))
        if reason:
            note(remote_name, reason)
            continue
        entry = config.tool_policy.tools.get(remote_name)
        if entry is None:
            note(remote_name, "not_in_local_policy")
            continue
        if not entry.enabled:
            note(remote_name, "locally_disabled")
            continue
        permission = ToolPermission.coerce(entry.permission)  # LOCAL authority
        if permission is ToolPermission.DENIED:
            note(remote_name, "permission_denied")
            continue

        registrations.append({
            "remote_name": remote_name,
            "description": sanitize_description(spec.get("description", "")),
            "input_schema": spec.get("inputSchema"),
            "permission": permission,
        })
    return registrations, diagnostics


def build_tools(registrations, config, client, session_owner=None):
    """Create McpTool instances from a registration plan (does not register them)."""
    tools = []
    for reg in registrations:
        tools.append(McpTool(
            registry_name=f"{config.namespace}.{reg['remote_name']}",
            remote_name=reg["remote_name"],
            description=reg["description"],
            input_schema=reg["input_schema"],
            permission=reg["permission"],
            client=client,
            server_label=config.server_id,
            call_timeout=config.call_timeout_seconds,
            session_owner=session_owner,
            invocation_policy=config.invocation_policy,
        ))
    return tools
