"""Phase G.3 — installer backend registry.

Two backends are registered: `npm` (generalized from Phase F) and `python_venv`
(new). A `preinstalled_executable` type was scoped out (Task 8) rather than
shipped as an unsafe partial implementation — `config/mcp_catalog.json` entries
declaring it are rejected at catalog-load time (not in `SUPPORTED_INSTALLERS`),
and any other unregistered/unknown installer type fails closed with
MCP_INSTALLER_UNSUPPORTED via `get_installer()` returning None.
"""

from mcp_management.installers.base import (
    CandidateInstallation,
    McpInstaller,
    McpLaunchSpec,
    ProvisioningTransaction,
    get_installer,
    register_installer,
)
from mcp_management.installers.npm_backend import NpmInstaller
from mcp_management.installers.python_venv import PythonVenvInstaller

register_installer(NpmInstaller())
register_installer(PythonVenvInstaller())

__all__ = [
    "CandidateInstallation",
    "McpInstaller",
    "McpLaunchSpec",
    "ProvisioningTransaction",
    "get_installer",
    "register_installer",
    "NpmInstaller",
    "PythonVenvInstaller",
]
