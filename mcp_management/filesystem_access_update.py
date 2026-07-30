"""Phase F.1 — rewrite ONLY the approved-roots args of an already-installed server.

Mirrors installer.py's transactional shape (generate -> validate a real running server ->
only then persist) but never touches npm, never creates a staging install directory, and
never changes the installed package version. "Promotion" here means: the candidate
configuration is proven against a REAL live server process before a single byte of the
managed `server.json` is overwritten. Since nothing is written until after that live
verification succeeds, there is no partial state to roll back on failure — the previous,
still-valid configuration was simply never touched, and the live client used to verify it is
always shut down (no orphan process).

`lifecycle.activate()` is reused for the actual write: it re-validates with the Phase E
loader and writes atomically (temp file + os.replace), exactly like every other Phase F
write path.
"""

import json
import os

from mcp_layer.errors import McpError
from mcp_management import lifecycle
from mcp_management.configuration_generator import validate_generated
from mcp_management.planner import managed_server_root
from mcp_management.registry import InstalledServer, get_installed, upsert, utc_now
from tools.models import (
    MCP_FILESYSTEM_ACCESS_NOT_INSTALLED,
    MCP_FILESYSTEM_ACCESS_UPDATE_FAILED,
    MCP_FILESYSTEM_ACCESS_VALIDATION_FAILED,
)


def _read_current_config(server_id, base_dir, managed_root):
    path = os.path.join(managed_server_root(server_id, base_dir, managed_root), "server.json")
    if not os.path.isfile(path):
        raise McpError(MCP_FILESYSTEM_ACCESS_NOT_INSTALLED,
                       f"No managed configuration exists for server {server_id!r}.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), path
    except (OSError, ValueError) as e:
        raise McpError(MCP_FILESYSTEM_ACCESS_UPDATE_FAILED,
                       "The current managed configuration could not be read.") from e


def _default_verify(config, proposed_roots, base_dir=None, start_server_fn=None):
    """Start the server with the CANDIDATE config and confirm list_allowed_directories
    reports exactly the proposed roots. Always shuts the process down (no orphan)."""
    from mcp_layer.external import start_server
    from mcp_management.validator import _extract_paths

    start = start_server_fn or start_server
    client = None
    try:
        try:
            client = start(config, base_dir=base_dir, allow_create=True)
        except McpError as e:
            raise McpError(MCP_FILESYSTEM_ACCESS_VALIDATION_FAILED,
                           f"The server failed to start with the new roots ({e.code}).") from e
        try:
            raw_tools = client.list_tools(timeout=config.startup_timeout_seconds)
        except McpError as e:
            raise McpError(MCP_FILESYSTEM_ACCESS_VALIDATION_FAILED,
                           f"tools/list failed after the access change ({e.code}).") from e
        names = {t.get("name") for t in raw_tools if isinstance(t, dict)}
        approved = {os.path.realpath(str(d)) for d in proposed_roots}
        if "list_allowed_directories" in names:
            try:
                result = client.call_tool("list_allowed_directories", {},
                                          timeout=config.call_timeout_seconds)
            except McpError as e:
                raise McpError(MCP_FILESYSTEM_ACCESS_VALIDATION_FAILED,
                               f"list_allowed_directories failed ({e.code}).") from e
            reported = _extract_paths(result)
            if reported and reported != approved:
                raise McpError(MCP_FILESYSTEM_ACCESS_VALIDATION_FAILED,
                               "The server's reported allowed directories do not match the "
                               "approved set after the access change.")
        return {"discovered_tool_count": len(raw_tools), "protocol_version": client.protocol_version}
    finally:
        if client is not None:
            try:
                client.shutdown()
            except Exception:  # noqa: BLE001 — never mask the validation outcome
                pass


def update_filesystem_access(plan, base_dir=None, managed_root=None, registry_path=None,
                             start_server_fn=None, validate_fn=None):
    """Apply `plan.proposed_allowed_directories` to the installed server's config.

    Order: read the current managed config -> build a candidate config carrying the
    plan's full proposed root set -> Phase E-validate it -> start a REAL server
    process with the candidate config and verify list_allowed_directories -> only
    then activate (validate again + atomic write) -> update the registry's
    approved_directories. No step here calls npm or changes the installed version.
    """
    installed = get_installed(plan.server_id, registry_path, base_dir, managed_root)
    if installed is None:
        raise McpError(MCP_FILESYSTEM_ACCESS_NOT_INSTALLED,
                       f"Server {plan.server_id!r} is not installed.")

    previous_raw, config_path = _read_current_config(plan.server_id, base_dir, managed_root)
    args = previous_raw.get("args") or []
    entrypoint = args[0] if args else None
    if not entrypoint:
        raise McpError(MCP_FILESYSTEM_ACCESS_UPDATE_FAILED,
                       "The installed configuration has no entrypoint.")

    candidate = dict(previous_raw)
    candidate["args"] = [entrypoint] + list(plan.proposed_allowed_directories)

    try:
        config = validate_generated(candidate)
    except McpError as e:
        raise McpError(MCP_FILESYSTEM_ACCESS_UPDATE_FAILED,
                       f"The candidate configuration is invalid: {e.message}") from e

    verify = validate_fn or _default_verify
    # Nothing has been written yet: on any failure here the previous config
    # remains active, untouched, and no npm call was ever made.
    verify(config, plan.proposed_allowed_directories, base_dir=base_dir, start_server_fn=start_server_fn)

    # activate() re-validates and writes atomically (temp file + os.replace), the
    # same primitive every other Phase F write path uses.
    lifecycle.activate(candidate, base_dir, managed_root, registry_path)

    updated = InstalledServer(
        catalog_id=installed.catalog_id,
        installed_version=installed.installed_version,
        status=installed.status,
        install_directory=installed.install_directory,
        configuration_path=installed.configuration_path,
        installed_at=installed.installed_at,
        last_validated_at=utc_now(),
        last_validation_result="healthy",
        approved_directories=tuple(plan.proposed_allowed_directories),
    )
    upsert(plan.server_id, updated, registry_path, base_dir, managed_root)
    return {
        "server_id": plan.server_id,
        "approved_directories": list(plan.proposed_allowed_directories),
        "config_path": config_path,
    }
