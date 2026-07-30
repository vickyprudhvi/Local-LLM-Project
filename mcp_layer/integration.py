"""Start the internal test MCP server, discover its tools, and register them.

This is the glue that plugs MCP into the existing architecture:

    start server -> initialize -> tools/list -> McpTool per tool -> register

Registered McpTools are namespaced (mcp.<server>.<tool>) and marked with the
permission the server advertises (coerced via Phase C; unknown/missing -> DENIED).
Nothing here touches the executor or the router.
"""

import os
import sys
import uuid

import tools.config as config
from mcp_layer.client import McpClient
from mcp_layer.errors import McpError
from mcp_layer.tool import McpTool
from tools.models import MCP_STARTUP_FAILED

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_NAMESPACE = "mcp.test"


class McpSession:
    """A running MCP server + the tools discovered from it. Shut it down when done.

    `client` may be None (e.g. a disabled or failed Phase E server), in which case
    there is nothing to shut down. `health` carries Phase E diagnostics or None.

    `session_id` is an opaque identifier stamped onto every McpTool this session
    registers (`McpTool.session_owner`); `registered_remote_tool_names` is exactly
    those tools' registry names. Together they let a runtime replacement
    (mcp_layer.runtime_manager) remove precisely this session's remote tools —
    never a built-in management tool, and never a replacement tool a NEWER session
    already re-registered under the same name.
    """

    def __init__(self, client, tools, namespace=DEFAULT_NAMESPACE, health=None, session_id=None):
        self.client = client
        self.tools = tools
        self.namespace = namespace
        self.health = health
        self.session_id = session_id or uuid.uuid4().hex
        self.registered_remote_tool_names = tuple(t.name for t in tools)

    def tool_names(self):
        return [t.name for t in self.tools]

    def shutdown(self):
        if self.client is not None:
            self.client.shutdown()


def start_test_server(workspace, call_timeout=20.0, startup_timeout=15.0, slow_seconds=None):
    """Launch `python -m test_mcp_server` against `workspace` and initialize it."""
    workspace = os.path.abspath(workspace)
    os.makedirs(workspace, exist_ok=True)
    env = dict(os.environ)
    env["TEST_MCP_WORKSPACE"] = workspace
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    if slow_seconds is not None:
        env["TEST_MCP_SLOW_SECONDS"] = str(slow_seconds)
    client = McpClient([sys.executable, "-m", "test_mcp_server"],
                       cwd=_REPO_ROOT, env=env, default_call_timeout=call_timeout)
    client.start(timeout=startup_timeout)
    return client


def discover_and_register(registry, client, namespace=DEFAULT_NAMESPACE,
                          call_timeout=20.0, list_timeout=15.0, session_owner=None):
    """tools/list -> McpTool per tool -> register into `registry`. Returns the McpTools."""
    server_label = namespace.split(".")[-1]
    registered = []
    for spec in client.list_tools(timeout=list_timeout):
        remote_name = spec.get("name")
        if not isinstance(remote_name, str) or not remote_name:
            continue
        permission = (spec.get("annotations") or {}).get("permission")
        tool = McpTool(
            registry_name=f"{namespace}.{remote_name}",
            remote_name=remote_name,
            description=spec.get("description", ""),
            input_schema=spec.get("inputSchema"),
            permission=permission,  # coerced in McpTool (unknown/missing -> DENIED)
            client=client,
            server_label=server_label,
            call_timeout=call_timeout,
            session_owner=session_owner,
        )
        registry.register(tool)
        registered.append(tool)
    return registered


def bootstrap_test_server(registry, workspace=None, namespace=DEFAULT_NAMESPACE):
    """Config-driven start + discover + register. Returns an McpSession, or raises McpError.

    Used by the assistant at startup. Callers should treat a failure as non-fatal
    (log MCP_STARTUP_FAILED and continue without MCP tools).
    """
    workspace = workspace or config.mcp_test_workspace()
    call_timeout = float(config.mcp_call_timeout())
    startup_timeout = float(config.mcp_startup_timeout())
    try:
        client = start_test_server(workspace, call_timeout=call_timeout,
                                   startup_timeout=startup_timeout)
    except McpError:
        raise
    except Exception as e:  # noqa: BLE001 — normalize any unexpected startup fault
        raise McpError(MCP_STARTUP_FAILED, "Failed to start the MCP server.") from e
    session_id = uuid.uuid4().hex
    tools = discover_and_register(registry, client, namespace=namespace, call_timeout=call_timeout,
                                  session_owner=session_id)
    return McpSession(client, tools, namespace=namespace, session_id=session_id)
