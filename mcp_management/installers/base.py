"""Phase G.3 Task 5 — the deterministic installer strategy interface.

Every installer backend (npm, python_venv, ...) implements exactly this
Protocol. Nothing outside an installer module ever branches on installer type
again — the candidate-transaction orchestrator (`mcp_management.auto_provisioning`)
only ever calls these five methods, dispatched once via `INSTALLERS[plan.installer_type]`.

All process execution MUST use argument arrays (`subprocess.run([...], shell=False)`)
— never a shell command string, never `shell=True`. A backend that cannot satisfy
this contract safely must not be registered; an unknown/unregistered type fails
closed with MCP_INSTALLER_UNSUPPORTED (Task 8).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass(frozen=True)
class ProvisioningTransaction:
    """Filesystem coordinates for ONE candidate installation attempt.

    `candidate_directory` is always under `<server_root>/candidates/<transaction_id>/`
    — never the live `versions/<version>/` directory — so a failure at any point
    before atomic promotion leaves the previously-installed (or absent) version
    completely untouched.
    """

    transaction_id: str
    server_id: str
    base_dir: str
    managed_root: str
    server_root: str
    candidate_directory: str
    final_directory: str


@dataclass(frozen=True)
class CandidateInstallation:
    """What a backend produced inside its candidate directory. `extra` carries
    installer-specific detail (e.g. the venv's python executable path) that only
    that same backend's later methods need to understand."""

    transaction: ProvisioningTransaction
    install_directory: str
    lock_hash: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class McpLaunchSpec:
    """The exact, absolute argv this candidate (or, after promotion, the
    installed server) launches with. Never derived from model output — only
    from validated installer output plus the trusted catalog entry."""

    command: str
    args: Tuple[str, ...]


class McpInstaller(Protocol):
    installer_type: str

    def prepare_candidate(self, plan, catalog_entry, transaction: ProvisioningTransaction) -> CandidateInstallation:
        ...

    def install_candidate(self, candidate: CandidateInstallation, plan, catalog_entry) -> CandidateInstallation:
        ...

    def validate_artifacts(self, candidate: CandidateInstallation, plan, catalog_entry) -> None:
        ...

    def build_launch_spec(self, candidate: CandidateInstallation, catalog_entry) -> McpLaunchSpec:
        ...

    def cleanup_candidate(self, candidate: CandidateInstallation) -> None:
        ...


_INSTALLERS: Dict[str, McpInstaller] = {}


def register_installer(installer: McpInstaller) -> None:
    _INSTALLERS[installer.installer_type] = installer


def get_installer(installer_type: str) -> Optional[McpInstaller]:
    return _INSTALLERS.get(installer_type)
