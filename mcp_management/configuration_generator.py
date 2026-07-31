"""Phase F — generate the Phase E server configuration automatically.

The user never edits JSON. Everything comes from the validated plan plus paths
resolved on this machine: an absolute runtime executable, the absolute managed
entrypoint, canonical approved directories, an isolated workspace under
mcp_workspaces/, and the tool policy from the TRUSTED catalog (so local permission
authority is preserved).

The generated document is always validated with the Phase E loader
(`mcp_layer.config.build_config`) before it is written or activated, and it never
contains secret values — only environment-variable names.
"""

import os

from mcp_layer.config import build_config
from mcp_layer.errors import McpError
from mcp_management.models import McpProvisioningPlan, policy_fingerprint
from mcp_management.registry import atomic_write_json
from tools.models import MCP_CONFIGURATION_GENERATION_FAILED, ToolPermission

DEFAULT_STARTUP_TIMEOUT = 15
DEFAULT_CALL_TIMEOUT = 15
DEFAULT_SHUTDOWN_TIMEOUT = 5


def _portable(path):
    """Absolute path with forward slashes — valid JSON and accepted on Windows."""
    return str(path).replace(os.sep, "/")


def generate_config_dict(plan: McpProvisioningPlan, runtime_executable, entrypoint_path,
                         enabled=True, required=False):
    """Build the Phase E configuration document for an installed server."""
    if not runtime_executable or not os.path.isabs(str(runtime_executable)):
        raise McpError(MCP_CONFIGURATION_GENERATION_FAILED,
                       "The runtime executable must be an absolute path.")
    if not entrypoint_path or not os.path.isabs(str(entrypoint_path)):
        raise McpError(MCP_CONFIGURATION_GENERATION_FAILED,
                       "The entrypoint must be an absolute path.")

    args = [_portable(entrypoint_path)]
    # The approved directories are arguments to the server itself (its allowed roots).
    args.extend(_portable(os.path.realpath(str(d))) for d in plan.requested_directories)

    policy = policy_fingerprint(plan.proposed_tool_policy)
    return {
        "enabled": bool(enabled),
        "required": bool(required),
        "server_id": plan.server_id,
        "display_name": plan.display_name,
        "transport": plan.transport,
        "command": _portable(runtime_executable),
        "args": args,
        "working_directory": _portable(plan.runtime_workspace),
        "startup_timeout_seconds": DEFAULT_STARTUP_TIMEOUT,
        "call_timeout_seconds": DEFAULT_CALL_TIMEOUT,
        "shutdown_timeout_seconds": DEFAULT_SHUTDOWN_TIMEOUT,
        # Names only; values are read from the parent process at launch time.
        "environment_allowlist": list(plan.requested_environment_variables),
        "tool_policy": policy,
        "generated_by": "mcp_management.configuration_generator",
        "catalog_id": plan.catalog_id,
        "package_version": plan.package_version,
    }


def generate_config_dict_from_launch_spec(server_id, display_name, transport, launch_spec,
                                          working_directory, environment_allowlist, tool_policy,
                                          catalog_id, package_version, installer_type,
                                          enabled=True, required=False):
    """Phase G.3 (Task 11) — generalize configuration generation for ANY installer
    type. Unlike `generate_config_dict` (npm-only: derives `args` from an
    entrypoint file plus approved directories), this takes the installer's own
    `McpLaunchSpec` (already-resolved absolute command + argv) directly, so a
    python_venv candidate's `<venv>/python -m <module>` launch is expressed
    exactly as naturally as npm's `node <entrypoint.js>` — no server-specific
    branch here. Still always Phase-E-validated before being written or activated,
    and never contains secret values — only environment-variable names.
    """
    if not launch_spec.command or not os.path.isabs(str(launch_spec.command)):
        raise McpError(MCP_CONFIGURATION_GENERATION_FAILED,
                       "The launch command must be an absolute path.")

    policy = policy_fingerprint(tool_policy)
    return {
        "enabled": bool(enabled),
        "required": bool(required),
        "server_id": server_id,
        "display_name": display_name,
        "transport": transport,
        "command": _portable(launch_spec.command),
        "args": [_portable(a) for a in launch_spec.args],
        "working_directory": _portable(working_directory),
        "startup_timeout_seconds": DEFAULT_STARTUP_TIMEOUT,
        "call_timeout_seconds": DEFAULT_CALL_TIMEOUT,
        "shutdown_timeout_seconds": DEFAULT_SHUTDOWN_TIMEOUT,
        # Names only; values are read from the parent process at launch time.
        "environment_allowlist": list(environment_allowlist),
        "tool_policy": policy,
        "generated_by": "mcp_management.configuration_generator",
        "catalog_id": catalog_id,
        "package_version": package_version,
        "installer_type": installer_type,
    }


def validate_generated(raw):
    """Validate with the Phase E loader. Returns the McpServerConfig."""
    try:
        return build_config(raw)
    except McpError as e:
        raise McpError(MCP_CONFIGURATION_GENERATION_FAILED,
                       f"The generated configuration is not valid: {e.message}") from e


def write_config(raw, path):
    """Validate then atomically write the configuration. Returns the McpServerConfig."""
    config = validate_generated(raw)
    atomic_write_json(path, raw)
    return config


def write_permissions_snapshot(plan: McpProvisioningPlan, path):
    """A standalone record of the applied policy, for auditability."""
    atomic_write_json(path, {
        "catalog_id": plan.catalog_id,
        "server_id": plan.server_id,
        "source": "trusted_catalog",
        "note": "Server-advertised permissions are ignored; this local policy is authoritative.",
        "policy": policy_fingerprint(plan.proposed_tool_policy),
        "read_tools": list(plan.read_tools()),
        "write_tools": list(plan.write_tools()),
        "denied_tools": list(plan.denied_tools()),
        "default_permission": ToolPermission.coerce(
            plan.proposed_tool_policy.default_permission).value,
    })
    return path
