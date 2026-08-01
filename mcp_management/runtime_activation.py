"""Phase G.2 — bridge Phase G.1's server selection to lazy MCP runtime activation.

Strictly a thin orchestration layer: looks up trusted-catalog + installed-server
metadata (mcp_management's job — trust and policy) and then asks the caller's
`MultiMcpRuntimeManager` (mcp_layer's job — process/session mechanics) to
`ensure_started` exactly the ONE server Phase G.1 selected. Never starts,
installs, or touches any OTHER server, and never picks an exact tool — that
remains Phase B's job.
"""

from dataclasses import dataclass
from typing import Optional

from mcp_layer.errors import McpError
from mcp_management.registry import get_installed
from tools.models import MCP_SERVER_NOT_INSTALLED

_FILESYSTEM_SERVER_ID = "filesystem"


@dataclass(frozen=True)
class ServerActivationResult:
    """The outcome of trying to lazily activate ONE selected server."""

    activated: bool
    server_id: Optional[str]
    error_code: Optional[str] = None
    message: Optional[str] = None


def ensure_selected_server_active(selection, runtime_manager, catalog, base_dir=None, managed_root=None,
                                  registry_path=None) -> ServerActivationResult:
    """Lazily activate `selection.selected_server_id` (Task 4/5/6).

    Only meaningful when `selection.status` is SELECTED — the caller
    (assistant.py) is expected to have already branched on that; this function
    does not re-check it and does not know about the other statuses.

    Root verification is Filesystem-specific (Task 8/9): `expected_allowed_roots`
    is populated only for the "filesystem" server_id, straight from the
    installed-server registry's own `approved_directories` — never guessed, and
    never applied to any other server.
    """
    server_id = selection.selected_server_id
    catalog_entry = catalog.get(selection.selected_catalog_id) if catalog is not None else None
    installed = get_installed(server_id, registry_path, base_dir, managed_root)

    if installed is None:
        return ServerActivationResult(
            activated=False, server_id=server_id, error_code=MCP_SERVER_NOT_INSTALLED,
            message=(
                f"The approved MCP server {server_id!r} can provide this capability, but it is not "
                "installed yet. Approval-driven provisioning can install it if you approve the plan."
            ),
        )

    expected_allowed_roots = installed.approved_directories if server_id == _FILESYSTEM_SERVER_ID else None
    expected_tools = catalog_entry.expected_tools if catalog_entry is not None else ()

    try:
        runtime_manager.ensure_started(server_id, expected_allowed_roots=expected_allowed_roots,
                                       expected_tools=expected_tools)
    except McpError as e:
        return ServerActivationResult(activated=False, server_id=server_id, error_code=e.code,
                                      message=f"The MCP server {server_id!r} could not be started ({e.code}).")
    return ServerActivationResult(activated=True, server_id=server_id)
