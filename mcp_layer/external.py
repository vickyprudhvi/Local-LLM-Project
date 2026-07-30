"""Phase E — bootstrap ONE externally-configured stdio MCP server.

Flow: load config -> (disabled? stop) -> validate executable + working directory ->
build isolated env -> launch (shell=False) -> initialize -> tools/list -> apply
LOCAL tool policy -> register McpTools -> report health.

Registered tools are plain McpTool(BaseTool)s, so the existing ToolExecutor runs
them with no MCP-specific branch. Optional-server failures (required=false) are
swallowed so built-in tools keep working.
"""

import os
import shutil
import uuid

import tools.config as app_config
from mcp_layer.client import McpClient
from mcp_layer.config import McpServerConfig, load_config
from mcp_layer.config_resolver import resolve_config
from mcp_layer.discovery import build_tools, plan_registration
from mcp_layer.environment import build_child_environment
from mcp_layer.errors import McpError
from mcp_layer.health import McpHealth, McpHealthState
from mcp_layer.integration import McpSession
from tools.models import (
    MCP_CONFIGURATION_INVALID,
    MCP_DISCOVERY_FAILED,
    MCP_EXECUTABLE_NOT_FOUND,
    MCP_WORKING_DIRECTORY_INVALID,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_config_path(base_dir=None):
    """The effective configuration path, by the documented precedence.

    MCP_CONFIG_PATH override -> enabled managed (Phase F) server -> committed
    portable template. Phase F never writes the committed template.
    """
    base_dir = base_dir or _REPO_ROOT
    resolved = resolve_config(base_dir=base_dir)
    if resolved.path is not None:
        return str(resolved.path)
    return os.path.join(base_dir, app_config.mcp_config_path())


def validate_executable(command):
    """Resolve `command` to a real executable path without installing anything."""
    if not isinstance(command, str) or not command.strip():
        raise McpError(MCP_CONFIGURATION_INVALID, "The MCP command is empty.")
    if any(c in command for c in ("\x00", "\n", "\r")):
        raise McpError(MCP_CONFIGURATION_INVALID, "The MCP command contains illegal characters.")
    resolved = shutil.which(command)
    if resolved:
        return resolved
    if os.path.isabs(command) and os.path.isfile(command) and os.access(command, os.X_OK):
        return command
    raise McpError(MCP_EXECUTABLE_NOT_FOUND, f"The MCP executable was not found: {command!r}. "
                                             "It will not be installed automatically.")


def resolve_workspaces_root(approved_root=None, base_dir=None):
    base_dir = base_dir or _REPO_ROOT
    approved_root = approved_root or app_config.mcp_workspaces_root()
    return os.path.realpath(os.path.join(base_dir, str(approved_root)))


def validate_working_directory(working_directory, approved_root_abs, base_dir=None, allow_create=True):
    """Resolve the working directory and enforce that it stays under the approved root."""
    base_dir = base_dir or _REPO_ROOT
    raw = str(working_directory)
    if not raw.strip():
        raise McpError(MCP_WORKING_DIRECTORY_INVALID, "No working directory is configured.")
    wd_abs = os.path.realpath(os.path.join(base_dir, raw))
    if wd_abs != approved_root_abs and not wd_abs.startswith(approved_root_abs + os.sep):
        raise McpError(MCP_WORKING_DIRECTORY_INVALID,
                       "The MCP working directory is outside the approved mcp_workspaces root.")
    if not os.path.isdir(wd_abs):
        if not allow_create:
            raise McpError(MCP_WORKING_DIRECTORY_INVALID, "The MCP working directory does not exist.")
        try:
            os.makedirs(wd_abs, exist_ok=True)
        except OSError as e:
            raise McpError(MCP_WORKING_DIRECTORY_INVALID,
                           "The MCP working directory could not be created.") from e
    return wd_abs


def internal_test_server_script(base_dir=None):
    """Absolute path to the bundled test server's executable script."""
    base_dir = base_dir or _REPO_ROOT
    return os.path.join(base_dir, "test_mcp_server", "server.py")


def build_launch_argv(config: McpServerConfig, executable, base_dir=None):
    """The argv list for Popen. For the internal test server, launch its absolute
    script path directly (no `-m`, so no repository PYTHONPATH is ever needed).
    Ordinary external servers get their configured args verbatim."""
    if config.internal_test_server:
        script = internal_test_server_script(base_dir)
        if not os.path.isfile(script):
            raise McpError(MCP_CONFIGURATION_INVALID, "The internal test server script was not found.")
        return [executable, script]
    return [executable, *config.args]


def start_server(config: McpServerConfig, approved_root=None, base_dir=None, allow_create=True):
    """Validate, isolate, and launch the configured server; run the initialize handshake."""
    base_dir = base_dir or _REPO_ROOT
    executable = validate_executable(config.command)
    approved_root_abs = resolve_workspaces_root(approved_root, base_dir)
    workdir = validate_working_directory(config.working_directory, approved_root_abs, base_dir, allow_create)
    argv = build_launch_argv(config, executable, base_dir)

    # Minimal child environment: platform vars + explicitly allowlisted names only.
    # No repository root is injected into PYTHONPATH; the parent PYTHONPATH is
    # inherited only if the operator allowlists it.
    env = build_child_environment(config.environment_allowlist)

    client = McpClient(argv, cwd=workdir, env=env,
                       default_call_timeout=config.call_timeout_seconds,
                       shutdown_timeout=config.shutdown_timeout_seconds)
    try:
        client.start(timeout=config.startup_timeout_seconds)
    except McpError:
        client.shutdown()
        raise
    return client


def bootstrap_from_config(registry, config=None, config_path=None, approved_root=None,
                          base_dir=None, allow_create=True, session_id=None):
    """Full Phase E startup. Returns an McpSession (with .health). Raises only when
    the configured server is `required` and startup fails.

    `session_id` (opaque; auto-generated when omitted) is stamped onto every
    McpTool this call registers, via `McpTool.session_owner` — see
    `mcp_layer.runtime_manager` for why a runtime replacement needs it.
    """
    base_dir = base_dir or _REPO_ROOT
    if config is None:
        config = load_config(config_path or default_config_path(base_dir))

    if config is None or not config.enabled:
        sid = config.server_id if config else None
        ns = config.namespace if config else "mcp"
        return McpSession(None, [], namespace=ns,
                          health=McpHealth(McpHealthState.DISABLED, sid))

    session_id = session_id or uuid.uuid4().hex
    client = None
    try:
        client = start_server(config, approved_root=approved_root, base_dir=base_dir,
                              allow_create=allow_create)
        try:
            raw_tools = client.list_tools(timeout=config.startup_timeout_seconds)
        except McpError as e:
            raise McpError(MCP_DISCOVERY_FAILED, "MCP tool discovery failed.", retryable=e.retryable) from e

        registrations, diagnostics = plan_registration(raw_tools, config)
        diagnostics = list(diagnostics)
        planned = build_tools(registrations, config, client, session_owner=session_id)

        registered = []
        for tool in planned:
            if registry.has(tool.name):
                # Never silently overwrite an existing tool (built-in or another MCP tool).
                diagnostics.append((tool.name, "registration_collision", "skipped"))
                continue
            registry.register(tool)
            registered.append(tool)

        denied_count = sum(1 for _, _, cat in diagnostics if cat == "denied")
        skipped_count = sum(1 for _, _, cat in diagnostics if cat == "skipped")
        disabled_count = sum(1 for _, _, cat in diagnostics if cat == "disabled")
        health = McpHealth(
            state=McpHealthState.HEALTHY,
            server_id=config.server_id,
            discovered_tool_count=len(registered) + len(diagnostics),
            registered_tool_count=len(registered),
            denied_tool_count=denied_count,
            skipped_tool_count=skipped_count,
            disabled_tool_count=disabled_count,
            diagnostics=tuple(diagnostics),
        )
        return McpSession(client, registered, namespace=config.namespace, health=health,
                          session_id=session_id)
    except McpError as e:
        if client is not None:
            try:
                client.shutdown()
            except Exception:  # noqa: BLE001 — shutdown must never mask the original error
                pass
        if config.required:
            raise
        return McpSession(None, [], namespace=config.namespace,
                          health=McpHealth(McpHealthState.FAILED, config.server_id,
                                           last_error_code=e.code))
