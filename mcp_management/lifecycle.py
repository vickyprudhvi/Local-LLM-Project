"""Phase F — activate, disable, enable, repair, and uninstall a managed server.

Activation writes ONLY inside the managed root: the generated document lives at
`app_data/mcp_servers/<server_id>/server.json` and is marked enabled, and the
registry records the server as installed. The startup resolver
(`mcp_layer.config_resolver`) then selects it ahead of the committed template, so
the ordinary Phase E bootstrap registers the tools as McpTool(BaseTool). No
separate runtime path is introduced.

The committed `config/mcp_server.json` is a portable, disabled-by-default template
and is NEVER written by Phase F — not on activate, disable, repair, or uninstall.

Uninstall removes ONLY managed files (the version directory and generated
configuration). Directories the user approved — and everything in them — are never
touched. A bounded audit record is preserved.
"""

import json
import os
import shutil

import tools.config as app_config
from mcp_layer.errors import McpError
from mcp_management import audit, npm_installer
from mcp_management.configuration_generator import validate_generated
from mcp_management.installer import GENERATED_CONFIG_FILENAME
from mcp_management.planner import managed_server_root
from mcp_management.registry import (
    STATUS_DISABLED,
    STATUS_INSTALLED,
    atomic_write_json,
    get_installed,
    load_registry,
    remove as registry_remove,
    set_status,
)
from tools.models import (
    MCP_ACTIVATION_FAILED,
    MCP_NOT_INSTALLED,
    MCP_REPAIR_FAILED,
    MCP_UNINSTALL_FAILED,
    MCP_UPDATE_AVAILABLE,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def template_config_path(base_dir=None, path=None):
    """The committed portable template. Read for reference only; never written here."""
    base_dir = base_dir or _REPO_ROOT
    return os.path.join(base_dir, str(path or app_config.mcp_config_path()))


def managed_config_path(server_id, base_dir=None, managed_root=None):
    """Where a managed server's generated configuration lives."""
    return os.path.join(managed_server_root(server_id, base_dir, managed_root),
                        GENERATED_CONFIG_FILENAME)


def _read_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def activate(raw_config, base_dir=None, managed_root=None, registry_path=None):
    """Validate and mark a generated configuration active — inside the managed root.

    Writes `app_data/mcp_servers/<server_id>/server.json` with `enabled: true` and
    sets the registry status, which is what the resolver keys off. The committed
    template is untouched.
    """
    try:
        config = validate_generated(raw_config)
    except McpError as e:
        raise McpError(MCP_ACTIVATION_FAILED,
                       f"The configuration could not be activated: {e.message}") from e

    server_id = raw_config.get("server_id")
    document = dict(raw_config)
    document["enabled"] = True
    atomic_write_json(managed_config_path(server_id, base_dir, managed_root), document)
    if get_installed(server_id, registry_path, base_dir, managed_root) is not None:
        set_status(server_id, STATUS_INSTALLED, registry_path, base_dir, managed_root,
                   validation_result="healthy")
    return config


def _generated_config(server_id, base_dir=None, managed_root=None):
    path = os.path.join(managed_server_root(server_id, base_dir, managed_root),
                        GENERATED_CONFIG_FILENAME)
    raw = _read_json(path)
    if raw is None:
        raise McpError(MCP_NOT_INSTALLED,
                       f"No generated configuration exists for server {server_id!r}.")
    return raw, path


def disable(server_id, base_dir=None, managed_root=None, registry_path=None):
    """Stop using the server: mark the MANAGED config disabled and update the registry.

    Installed files are preserved so re-enabling needs no reinstall, and the
    resolver stops selecting the managed configuration at the next startup (it
    falls back to the committed disabled template). The caller is responsible for
    shutting down a running session (tools unregister with it). The committed
    template is never modified.
    """
    entry = get_installed(server_id, registry_path, base_dir, managed_root)
    if entry is None:
        raise McpError(MCP_NOT_INSTALLED, f"Server {server_id!r} is not installed.")

    target = managed_config_path(server_id, base_dir, managed_root)
    raw = _read_json(target)
    if raw is not None:
        raw = dict(raw)
        raw["enabled"] = False
        atomic_write_json(target, raw)
    set_status(server_id, STATUS_DISABLED, registry_path, base_dir, managed_root)
    return {"server_id": server_id, "status": STATUS_DISABLED,
            "install_directory": entry.install_directory}


def enable(server_id, base_dir=None, managed_root=None, registry_path=None):
    """Re-activate an already-installed server. Never reinstalls anything."""
    entry = get_installed(server_id, registry_path, base_dir, managed_root)
    if entry is None:
        raise McpError(MCP_NOT_INSTALLED, f"Server {server_id!r} is not installed.")

    raw, _ = _generated_config(server_id, base_dir, managed_root)
    command = raw.get("command")
    args = raw.get("args") or []
    entrypoint = args[0] if args else None
    if not entrypoint or not os.path.isfile(str(entrypoint)):
        raise McpError(MCP_NOT_INSTALLED,
                       "The installed entrypoint is missing; run repair before enabling.")
    if not command or not os.path.isfile(str(command)):
        # The runtime may have moved since installation; re-resolve it.
        raw = dict(raw)
        raw["command"] = npm_installer.resolve_runtime("node").replace(os.sep, "/")

    raw = dict(raw)
    raw["enabled"] = True
    config = activate(raw, base_dir, managed_root, registry_path)
    return {"server_id": server_id, "status": STATUS_INSTALLED, "config": config,
            "raw_config": raw, "reinstalled": False}


def repair(server_id, catalog, base_dir=None, managed_root=None, registry_path=None,
           reinstall_fn=None):
    """Verify the installation and reinstall the SAME pinned version if needed.

    Never upgrades: the version comes from the recorded installation, and the
    catalog entry must still pin that exact version.
    """
    entry = get_installed(server_id, registry_path, base_dir, managed_root)
    if entry is None:
        raise McpError(MCP_NOT_INSTALLED, f"Server {server_id!r} is not installed.")
    catalog_entry = catalog.get(entry.catalog_id)
    if catalog_entry is None:
        raise McpError(MCP_REPAIR_FAILED,
                       "The installed server is no longer in the trusted catalog.")
    if catalog_entry.package_version != entry.installed_version:
        # A newer approved version exists: report it, never silently upgrade.
        raise McpError(MCP_UPDATE_AVAILABLE,
                       f"The catalog now pins {catalog_entry.package_version}; the installed "
                       f"version is {entry.installed_version}. Repair does not upgrade — "
                       "approve a new provisioning plan to change versions.")

    record = audit.read_install_record(managed_server_root(server_id, base_dir, managed_root))
    raw, _ = _generated_config(server_id, base_dir, managed_root)
    args = raw.get("args") or []
    entrypoint = args[0] if args else None
    intact = bool(entrypoint) and os.path.isfile(str(entrypoint))

    if intact:
        result = enable(server_id, base_dir, managed_root, registry_path)
        result["reinstalled"] = False
        result["record_present"] = record is not None
        return result
    if reinstall_fn is None:
        raise McpError(MCP_REPAIR_FAILED,
                       "The installation is incomplete and no reinstaller was provided.")
    result = reinstall_fn(catalog_entry, entry)
    result["reinstalled"] = True
    result["record_present"] = record is not None
    return result


def uninstall(server_id, base_dir=None, managed_root=None, registry_path=None):
    """Remove managed files and registry state. User workspaces are never touched."""
    server_root = managed_server_root(server_id, base_dir, managed_root)
    entry = get_installed(server_id, registry_path, base_dir, managed_root)
    # Idempotent: uninstalling twice is safe. Preserved audit records do NOT count
    # as an installation, so only real artifacts make this a live uninstall.
    has_artifacts = (os.path.isdir(os.path.join(server_root, "versions"))
                     or os.path.isfile(os.path.join(server_root, GENERATED_CONFIG_FILENAME)))
    if entry is None and not has_artifacts:
        return {"server_id": server_id, "removed": False, "already_absent": True}

    # Deactivate by removing the MANAGED configuration only. The committed
    # template at config/mcp_server.json is never read or written here.
    managed_config = managed_config_path(server_id, base_dir, managed_root)
    if os.path.isfile(managed_config):
        try:
            os.unlink(managed_config)
        except OSError as e:
            raise McpError(MCP_UNINSTALL_FAILED,
                           "The managed configuration could not be removed.") from e

    preserved_record = audit.read_install_record(server_root)
    removed_paths = []
    for name in ("versions", GENERATED_CONFIG_FILENAME, "permissions.json", "current.json"):
        path = os.path.join(server_root, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            removed_paths.append(path)
        elif os.path.isfile(path):
            try:
                os.unlink(path)
                removed_paths.append(path)
            except OSError:
                pass

    if preserved_record is not None:
        # Keep a bounded audit trail of what had been installed.
        atomic_write_json(os.path.join(server_root, "uninstall-record.json"), {
            "server_id": server_id,
            "catalog_id": preserved_record.get("catalog_id"),
            "package_version": preserved_record.get("package_version"),
            "plan_hash": preserved_record.get("plan_hash"),
            "uninstalled_from": preserved_record.get("install_directory"),
        })
    registry_remove(server_id, registry_path, base_dir, managed_root)
    return {"server_id": server_id, "removed": True, "removed_paths": removed_paths,
            "audit_preserved": preserved_record is not None}


def check_for_update(server_id, catalog, base_dir=None, managed_root=None,
                     registry_path=None):
    """Report (never apply) a newer approved catalog version."""
    entry = get_installed(server_id, registry_path, base_dir, managed_root)
    if entry is None:
        raise McpError(MCP_NOT_INSTALLED, f"Server {server_id!r} is not installed.")
    catalog_entry = catalog.get(entry.catalog_id)
    if catalog_entry is None:
        return {"update_available": False, "installed_version": entry.installed_version}
    available = catalog_entry.package_version != entry.installed_version
    return {
        "update_available": available,
        "installed_version": entry.installed_version,
        "catalog_version": catalog_entry.package_version,
        "error_code": MCP_UPDATE_AVAILABLE if available else None,
    }


def list_installed(base_dir=None, managed_root=None, registry_path=None):
    return {sid: entry.to_dict()
            for sid, entry in load_registry(registry_path, base_dir, managed_root).items()}
