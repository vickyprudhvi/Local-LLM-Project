"""Which MCP configuration is in effect — one deterministic resolver.

Phase F installs machine-specific configurations (absolute executable, entrypoint,
workspace, and approved-directory paths). Those must never be written into the
committed `config/mcp_server.json`, which stays a portable, disabled-by-default
template. Instead the effective configuration is RESOLVED at startup:

  1. MCP_CONFIG_PATH             — explicit operator override
  2. managed active server       — a Phase F installation that is enabled
  3. config/mcp_server.json      — the committed portable template (disabled)

The override is validated (canonical, regular file, no null/newline) and produces a
structured MCP_CONFIGURATION_INVALID error when it is not usable. The LLM never
selects or changes it — it comes only from the environment.

Only the resolved SOURCE and a basename are safe to log; the path is never logged
together with configuration contents.
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import tools.config as app_config
from mcp_layer.errors import McpError
from tools.models import MCP_CONFIGURATION_INVALID

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANAGED_CONFIG_FILENAME = "server.json"
REGISTRY_FILENAME = "installed_servers.json"
# The registry's on-disk status value for an installed (not disabled) server. Kept
# as a literal so this layer reads the managed state file without importing the
# Phase F package (mcp_management depends on mcp_layer, not the other way round).
_STATUS_INSTALLED = "installed"


class McpConfigSource(str, Enum):
    ENVIRONMENT_OVERRIDE = "environment_override"
    MANAGED_ACTIVE = "managed_active"
    DEFAULT_TEMPLATE = "default_template"
    NONE = "none"


@dataclass(frozen=True)
class ResolvedMcpConfig:
    path: Optional[Path]
    source: McpConfigSource
    server_id: Optional[str] = None

    @property
    def exists(self) -> bool:
        return self.path is not None and os.path.isfile(str(self.path))

    def describe(self) -> str:
        """Safe one-line description for logs: source + basename only, never contents."""
        name = os.path.basename(str(self.path)) if self.path else "none"
        return f"{self.source.value}:{name}"


def _validate_override(raw, base_dir):
    """Canonicalize and check an MCP_CONFIG_PATH value."""
    if any(c in raw for c in ("\x00", "\n", "\r")):
        raise McpError(MCP_CONFIGURATION_INVALID,
                       "MCP_CONFIG_PATH contains illegal characters.")
    resolved = os.path.realpath(os.path.join(base_dir, raw))
    if not os.path.exists(resolved):
        raise McpError(MCP_CONFIGURATION_INVALID,
                       "MCP_CONFIG_PATH does not point to an existing file.")
    if not os.path.isfile(resolved):
        raise McpError(MCP_CONFIGURATION_INVALID,
                       "MCP_CONFIG_PATH must point to a regular JSON file.")
    return Path(resolved)


def _managed_active(base_dir, managed_root=None):
    """The generated config of an ENABLED managed server, or None.

    "Enabled" means the registry lists it as installed AND its generated
    configuration exists with `enabled: true`. Disabling flips that flag, so the
    managed configuration simply stops being selected at the next startup.
    """
    import json

    root = os.path.join(base_dir, str(managed_root or app_config.mcp_managed_root()))
    registry_file = os.path.join(root, REGISTRY_FILENAME)
    if not os.path.isfile(registry_file):
        return None, None
    try:
        with open(registry_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        servers = raw.get("servers") or {}
    except (OSError, ValueError):
        # A corrupt registry is reported by the registry loader itself; resolution
        # must not crash startup, so fall through to the template.
        return None, None

    root_abs = os.path.realpath(root)
    for server_id in sorted(servers):
        spec = servers[server_id]
        if not isinstance(spec, dict) or spec.get("status") != _STATUS_INSTALLED:
            continue

        # A tampered registry must never redirect startup to an arbitrary file:
        # only a configuration_path that canonically resolves INSIDE the managed
        # root is honoured; otherwise fall back to the canonical location.
        candidate = os.path.join(root, server_id, MANAGED_CONFIG_FILENAME)
        configured = spec.get("configuration_path")
        if isinstance(configured, str) and configured:
            configured_abs = os.path.realpath(configured)
            if configured_abs == root_abs or configured_abs.startswith(root_abs + os.sep):
                candidate = configured_abs
        candidate = os.path.realpath(candidate)
        if candidate != root_abs and not candidate.startswith(root_abs + os.sep):
            continue
        if not os.path.isfile(candidate):
            continue

        try:
            with open(candidate, "r", encoding="utf-8") as f:
                document = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(document, dict) or document.get("enabled") is not True:
            continue
        # The generated document must belong to the registry entry that names it.
        if document.get("server_id") != server_id:
            continue
        return Path(candidate), server_id
    return None, None


def resolve_config(base_dir=None, managed_root=None, override=None,
                   template_path=None) -> ResolvedMcpConfig:
    """Resolve the effective MCP configuration by the documented precedence."""
    base_dir = base_dir or _REPO_ROOT

    raw_override = override if override is not None else os.environ.get("MCP_CONFIG_PATH")
    if raw_override is not None and str(raw_override).strip():
        return ResolvedMcpConfig(
            path=_validate_override(str(raw_override), base_dir),
            source=McpConfigSource.ENVIRONMENT_OVERRIDE,
        )

    managed_path, server_id = _managed_active(base_dir, managed_root)
    if managed_path is not None:
        return ResolvedMcpConfig(path=managed_path, source=McpConfigSource.MANAGED_ACTIVE,
                                 server_id=server_id)

    template = os.path.realpath(os.path.join(
        base_dir, str(template_path or app_config.mcp_config_path())))
    if os.path.isfile(template):
        return ResolvedMcpConfig(path=Path(template), source=McpConfigSource.DEFAULT_TEMPLATE)
    return ResolvedMcpConfig(path=None, source=McpConfigSource.NONE)
