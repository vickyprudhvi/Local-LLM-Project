"""Phase F — build the deterministic provisioning plan.

The plan is derived ONLY from the trusted catalog entry plus a caller-supplied
directory request. Nothing here consults the LLM. Directory requests are resolved
canonically and screened against forbidden locations (credentials, system paths)
and broad locations (home root, Documents) that need explicit opt-in.

`plan_id` is derived from the plan hash, so identical inputs always yield an
identical plan — the property approval matching depends on.
"""

import os
from pathlib import Path

import tools.config as app_config
from mcp_layer.errors import McpError
from mcp_management.catalog import McpCatalogEntry
from mcp_management.models import McpProvisioningPlan
from tools.models import (
    MCP_DIRECTORY_NOT_APPROVED,
    MCP_PROVISIONING_PLAN_INVALID,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Never grantable, with or without explicit broad approval.
_FORBIDDEN_LEAF_NAMES = (
    ".ssh", ".aws", ".gnupg", ".azure", ".kube", ".docker", "gcloud",
    "credentials", "secrets", ".password-store", "keychains",
)
_FORBIDDEN_PATH_FRAGMENTS = (
    os.path.join("appdata", "roaming", "mozilla"),
    os.path.join("appdata", "local", "google", "chrome", "user data"),
    os.path.join("appdata", "roaming", "microsoft", "edge"),
    os.path.join("library", "application support", "google", "chrome"),
    os.path.join(".config", "gcloud"),
    os.path.join(".mozilla", "firefox"),
)


def _forbidden_roots():
    roots = []
    for candidate in (
        os.environ.get("SYSTEMROOT"), os.environ.get("WINDIR"),
        r"C:\Windows", r"C:\Program Files", r"C:\Program Files (x86)",
        "/etc", "/usr", "/bin", "/sbin", "/boot", "/dev", "/proc", "/sys",
        "/System", "/Library",
    ):
        if candidate:
            roots.append(os.path.realpath(candidate))
    # The assistant's own virtualenv is never a grantable workspace.
    roots.append(os.path.realpath(os.path.join(_REPO_ROOT, "venv")))
    return tuple(roots)


def _broad_roots(base_dir):
    """Locations a user may approve only by explicitly opting into a broad scope."""
    home = os.path.realpath(os.path.expanduser("~"))
    broad = [home, os.path.realpath(base_dir)]
    for name in ("Documents", "Desktop", "Downloads", "OneDrive"):
        broad.append(os.path.join(home, name))
    return tuple(broad)


def _is_filesystem_root(path):
    return os.path.dirname(path) == path


def validate_approved_directory(requested, base_dir=None, allow_broad=False,
                                allow_create=False):
    """Canonicalize and screen a requested directory. Returns an absolute path.

    Raises MCP_DIRECTORY_NOT_APPROVED for illegal characters, non-existent paths,
    forbidden locations, or broad locations without an explicit opt-in.
    """
    base_dir = base_dir or _REPO_ROOT
    if not isinstance(requested, (str, Path)) or not str(requested).strip():
        raise McpError(MCP_DIRECTORY_NOT_APPROVED, "No directory was requested.")
    raw = str(requested)
    if any(c in raw for c in ("\x00", "\n", "\r")):
        raise McpError(MCP_DIRECTORY_NOT_APPROVED, "The requested directory contains illegal characters.")

    resolved = os.path.realpath(os.path.join(base_dir, raw))

    if _is_filesystem_root(resolved):
        raise McpError(MCP_DIRECTORY_NOT_APPROVED, "The filesystem root cannot be granted.")

    lowered = resolved.lower()
    if os.path.basename(lowered) in _FORBIDDEN_LEAF_NAMES:
        raise McpError(MCP_DIRECTORY_NOT_APPROVED,
                       "That directory holds credentials and cannot be granted.")
    for fragment in _FORBIDDEN_PATH_FRAGMENTS:
        if fragment in lowered:
            raise McpError(MCP_DIRECTORY_NOT_APPROVED,
                           "That directory holds browser or cloud credentials and cannot be granted.")
    for root in _forbidden_roots():
        if resolved == root or resolved.startswith(root + os.sep):
            raise McpError(MCP_DIRECTORY_NOT_APPROVED,
                           "That directory is a protected system location and cannot be granted.")

    if not allow_broad:
        for root in _broad_roots(base_dir):
            if resolved == os.path.realpath(root):
                raise McpError(
                    MCP_DIRECTORY_NOT_APPROVED,
                    "That directory is very broad. Request a narrower subdirectory, or "
                    "approve the broad scope explicitly.",
                )

    if not os.path.isdir(resolved):
        if not allow_create:
            raise McpError(MCP_DIRECTORY_NOT_APPROVED, "The requested directory does not exist.")
        try:
            os.makedirs(resolved, exist_ok=True)
        except OSError as e:
            raise McpError(MCP_DIRECTORY_NOT_APPROVED,
                           "The requested directory could not be created.") from e
    return resolved


def managed_server_root(server_id, base_dir=None, managed_root=None):
    base_dir = base_dir or _REPO_ROOT
    managed_root = managed_root or app_config.mcp_managed_root()
    return os.path.join(base_dir, str(managed_root), server_id)


def install_directory_for(entry: McpCatalogEntry, base_dir=None, managed_root=None):
    return os.path.join(managed_server_root(entry.server_id, base_dir, managed_root),
                        "versions", entry.package_version)


def runtime_workspace_for(entry: McpCatalogEntry, base_dir=None, workspaces_root=None):
    """The server's cwd — always an isolated directory under mcp_workspaces/,
    which is what the Phase E loader requires."""
    base_dir = base_dir or _REPO_ROOT
    workspaces_root = workspaces_root or app_config.mcp_workspaces_root()
    return os.path.join(base_dir, str(workspaces_root), entry.server_id)


def build_plan(entry: McpCatalogEntry, requested_directories=(), base_dir=None,
               managed_root=None, workspaces_root=None, allow_broad=False,
               allow_create=False) -> McpProvisioningPlan:
    """Build the immutable, hashed provisioning plan for one catalog entry."""
    base_dir = base_dir or _REPO_ROOT
    if entry.installer_type != "npm":
        raise McpError(MCP_PROVISIONING_PLAN_INVALID,
                       f"Unsupported installer type {entry.installer_type!r}.")

    approved = []
    for requested in requested_directories:
        approved.append(Path(validate_approved_directory(
            requested, base_dir=base_dir, allow_broad=allow_broad, allow_create=allow_create)))
    if entry.requires_directory() and not approved:
        raise McpError(MCP_PROVISIONING_PLAN_INVALID,
                       f"{entry.display_name} requires an approved directory.")

    env_names = entry.required_environment_variables()
    risks = [f"risk category: {entry.risk_category}"]
    if approved:
        risks.append("The server can read the approved directory; write tools still "
                     "require per-call confirmation.")
    risks.append("npm lifecycle scripts are disabled for this installation.")

    plan = McpProvisioningPlan(
        plan_id="",
        catalog_id=entry.catalog_id,
        server_id=entry.server_id,
        display_name=entry.display_name,
        package_manager=entry.installer_type,
        package_name=entry.package_name,
        package_version=entry.package_version,
        package_source=entry.package_source,
        entrypoint_relative=entry.entrypoint_relative,
        install_directory=Path(install_directory_for(entry, base_dir, managed_root)),
        runtime_workspace=Path(runtime_workspace_for(entry, base_dir, workspaces_root)),
        transport=entry.transport,
        requested_directories=tuple(approved),
        requested_environment_variables=tuple(env_names),
        proposed_tool_policy=entry.default_tool_policy,
        required_runtimes=entry.required_runtimes,
        requires_network_install=True,
        requires_credentials=bool(env_names),
        risk_summary=tuple(risks),
    ).with_hash()

    # Deterministic id derived from the hash: identical inputs -> identical plan.
    from dataclasses import replace

    return replace(plan, plan_id=f"plan_{plan.plan_hash[:16]}")
