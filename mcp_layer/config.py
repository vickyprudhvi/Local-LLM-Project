"""Phase E — trusted MCP server configuration: immutable models + validating loader.

Configuration is loaded once at startup and treated as immutable for the session.
Everything fails closed: unknown transport, invalid names, bad timeouts, or a
missing command on an enabled server all raise MCP_CONFIGURATION_INVALID. The LLM
never contributes to any of these values — they come only from the trusted file.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from mcp_layer.errors import McpError
from tools.models import MCP_CONFIGURATION_INVALID, ToolPermission

NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SUPPORTED_TRANSPORTS = ("stdio",)


@dataclass(frozen=True)
class McpToolPolicyEntry:
    enabled: bool
    permission: ToolPermission


@dataclass(frozen=True)
class McpToolPolicy:
    default_permission: ToolPermission
    tools: Dict[str, McpToolPolicyEntry] = field(default_factory=dict)


@dataclass(frozen=True)
class McpServerConfig:
    enabled: bool
    required: bool
    server_id: str
    display_name: str
    transport: str
    command: str
    args: Tuple[str, ...]
    working_directory: Path
    startup_timeout_seconds: float
    call_timeout_seconds: float
    shutdown_timeout_seconds: float
    environment_allowlist: Tuple[str, ...]
    tool_policy: McpToolPolicy
    invocation_policy: Dict[str, str] = field(default_factory=dict)
    # Internal-development-only: launch the repository's bundled test server by its
    # absolute script path (resolved deterministically from the repo), so it needs
    # no PYTHONPATH injection. Defaults to false; ordinary external servers never
    # get the repo root on their path. Set only via the trusted config file.
    internal_test_server: bool = False

    @property
    def namespace(self) -> str:
        return f"mcp.{self.server_id}"


def _invalid(message: str) -> McpError:
    return McpError(MCP_CONFIGURATION_INVALID, f"Invalid MCP configuration: {message}")


def _require_type(value, types, name):
    if not isinstance(value, types):
        raise _invalid(f"'{name}' has the wrong type.")
    return value


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(f"'{name}' must be a number.")
    if value <= 0:
        raise _invalid(f"'{name}' must be greater than zero.")
    return float(value)


def _permission(value, name):
    # Fail closed: an unknown/invalid permission becomes DENIED rather than READ.
    coerced = ToolPermission.coerce(value)
    return coerced


def _tool_policy(raw) -> McpToolPolicy:
    raw = raw or {}
    _require_type(raw, dict, "tool_policy")
    default_permission = _permission(raw.get("default_permission", "denied"), "tool_policy.default_permission")
    entries = {}
    tools = raw.get("tools", {})
    _require_type(tools, dict, "tool_policy.tools")
    for name, spec in tools.items():
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise _invalid(f"tool name {name!r} is not a valid identifier.")
        _require_type(spec, dict, f"tool_policy.tools.{name}")
        enabled = bool(spec.get("enabled", False))
        permission = _permission(spec.get("permission", "denied"), f"tool_policy.tools.{name}.permission")
        entries[name] = McpToolPolicyEntry(enabled=enabled, permission=permission)
    return McpToolPolicy(default_permission=default_permission, tools=entries)


def build_config(raw: dict) -> McpServerConfig:
    """Validate a raw dict into an immutable McpServerConfig (fail closed)."""
    _require_type(raw, dict, "<root>")
    enabled = bool(raw.get("enabled", False))
    required = bool(raw.get("required", False))

    server_id = raw.get("server_id", "")
    if not isinstance(server_id, str) or not NAME_RE.match(server_id):
        raise _invalid("'server_id' must match ^[a-zA-Z0-9_-]+$.")

    transport = raw.get("transport", "stdio")
    if transport not in SUPPORTED_TRANSPORTS:
        raise _invalid(f"transport {transport!r} is not supported (only 'stdio').")

    command = raw.get("command", "")
    if enabled:
        if not isinstance(command, str) or not command.strip():
            raise _invalid("'command' is required when the server is enabled.")
        if "\x00" in command or "\n" in command or "\r" in command:
            raise _invalid("'command' contains illegal characters.")
    command = command if isinstance(command, str) else ""

    args_raw = raw.get("args", [])
    _require_type(args_raw, list, "args")
    args = []
    for a in args_raw:
        if not isinstance(a, str) or "\x00" in a:
            raise _invalid("every entry in 'args' must be a string without null bytes.")
        args.append(a)

    working_directory = raw.get("working_directory", "")
    if enabled and (not isinstance(working_directory, str) or not working_directory.strip()):
        raise _invalid("'working_directory' is required when the server is enabled.")
    working_directory = Path(working_directory) if isinstance(working_directory, str) else Path("")

    startup = _positive_number(raw.get("startup_timeout_seconds", 10), "startup_timeout_seconds")
    call = _positive_number(raw.get("call_timeout_seconds", 5), "call_timeout_seconds")
    shutdown = _positive_number(raw.get("shutdown_timeout_seconds", 5), "shutdown_timeout_seconds")

    allowlist_raw = raw.get("environment_allowlist", [])
    _require_type(allowlist_raw, list, "environment_allowlist")
    allowlist = []
    for name in allowlist_raw:
        if not isinstance(name, str) or not ENV_NAME_RE.match(name):
            raise _invalid(f"environment_allowlist entry {name!r} is not a valid variable name.")
        allowlist.append(name)

    display_name = raw.get("display_name")
    if not isinstance(display_name, str) or not display_name:
        display_name = server_id

    invocation_policy_raw = raw.get("invocation_policy", {})
    if not isinstance(invocation_policy_raw, dict):
        raise _invalid("'invocation_policy' must be an object.")
    invocation_policy = {}
    for k, v in invocation_policy_raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise _invalid("'invocation_policy' entries must be strings.")
        if any(c in k for c in ("\x00", "\n", "\r")) or any(c in v for c in ("\x00", "\n", "\r")):
            raise _invalid("'invocation_policy' entry contains illegal characters.")
        invocation_policy[k] = v

    return McpServerConfig(
        enabled=enabled,
        required=required,
        server_id=server_id,
        display_name=display_name,
        transport=transport,
        command=command,
        args=tuple(args),
        working_directory=working_directory,
        startup_timeout_seconds=startup,
        call_timeout_seconds=call,
        shutdown_timeout_seconds=shutdown,
        environment_allowlist=tuple(allowlist),
        tool_policy=_tool_policy(raw.get("tool_policy")),
        invocation_policy=invocation_policy,
        internal_test_server=bool(raw.get("internal_test_server", False)),
    )


def load_config(path) -> Optional[McpServerConfig]:
    """Load + validate the config file. Returns None if the file is absent.

    Raises McpError(MCP_CONFIGURATION_INVALID) on malformed or invalid content.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        raise _invalid("the file could not be read or parsed as JSON.") from e
    return build_config(raw)
