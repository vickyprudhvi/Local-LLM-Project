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


def is_valid_input_schema(schema) -> bool:
    if not isinstance(schema, dict):
        return False
    stype = schema.get("type", "object")
    if stype != "object":
        return False  # MCP tool input schemas are objects
    if "properties" in schema and not isinstance(schema["properties"], dict):
        return False
    declared = schema.get("type")
    if declared is not None and declared not in _VALID_SCHEMA_TYPES:
        return False
    try:
        serialized = json.dumps(schema)
    except (TypeError, ValueError):
        return False
    if len(serialized) > MAX_SCHEMA_BYTES:
        return False
    if _schema_depth(schema) > MAX_SCHEMA_NESTING:
        return False
    return True


def plan_registration(raw_tools, config):
    """Return (registrations, denied_names).

    registrations: list of dicts {remote_name, description, input_schema, permission}
    denied_names:  discovered tools rejected by local policy or validation (diagnostics).

    A tool registers only when it is valid AND present in the local policy AND
    locally enabled; its permission is the LOCAL one. Anything else is denied
    (missing from policy / invalid name / bad schema) or silently skipped (locally
    disabled) — never registered, so it can never be shortlisted or reach the server.
    """
    registrations = []
    denied = []
    if not isinstance(raw_tools, list):
        return registrations, denied

    for spec in raw_tools[:MAX_TOOLS]:
        if not isinstance(spec, dict):
            continue
        remote_name = spec.get("name")
        if not isinstance(remote_name, str) or not NAME_RE.match(remote_name):
            denied.append(str(remote_name))
            continue
        schema = spec.get("inputSchema")
        if not is_valid_input_schema(schema):
            denied.append(remote_name)  # oversized/invalid schema: skip safely
            continue

        entry = config.tool_policy.tools.get(remote_name)
        if entry is None:
            denied.append(remote_name)  # missing from local policy -> DENIED
            continue
        if not entry.enabled:
            continue  # locally disabled -> not registered (and not counted as denied)

        registrations.append({
            "remote_name": remote_name,
            "description": sanitize_description(spec.get("description", "")),
            "input_schema": schema,
            "permission": ToolPermission.coerce(entry.permission),  # LOCAL authority
        })
    return registrations, denied


def build_tools(registrations, config, client):
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
        ))
    return tools
