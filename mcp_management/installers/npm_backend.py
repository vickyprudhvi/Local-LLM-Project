"""Phase G.3 Task 6 — the npm installer backend, generalized from Phase F.

Reuses `mcp_management.npm_installer` verbatim rather than duplicating it:
`McpCatalogEntry` already carries the exact same attributes Phase F's
`npm_installer` functions read from a plan (`package_name`, `package_version`,
`server_id`, `entrypoint_relative`, `required_runtimes`), so the catalog entry is
passed directly wherever those functions expect a "plan" — no adapter object,
no duplicated argv-building logic. `--ignore-scripts`, the isolated per-server
directory, `shell=False`, and the exact-version pin are all still enforced by
that shared module, unchanged.

`install_candidate` reuses an intact, already-installed version directory
instead of reinstalling (Task 6's compatibility requirement): the existing
Filesystem installation is recognized and never triggers a second npm call.
"""

import os
import shutil

from mcp_layer.errors import McpError
from mcp_management import npm_installer
from mcp_management.installers.base import CandidateInstallation, McpLaunchSpec, ProvisioningTransaction

INSTALLER_TYPE = "npm"


class NpmInstaller:
    installer_type = INSTALLER_TYPE

    def prepare_candidate(self, plan, catalog_entry, transaction: ProvisioningTransaction) -> CandidateInstallation:
        shutil.rmtree(transaction.candidate_directory, ignore_errors=True)
        os.makedirs(transaction.candidate_directory, exist_ok=True)
        return CandidateInstallation(transaction=transaction, install_directory=transaction.candidate_directory)

    def install_candidate(self, candidate: CandidateInstallation, plan, catalog_entry) -> CandidateInstallation:
        # Reuse an already-installed, intact version instead of reinstalling —
        # this is what makes re-approving a plan for an already-installed
        # Filesystem server a no-op with zero npm calls (Task 6 regression).
        final_dir = candidate.transaction.final_directory
        if _entrypoint_ok(final_dir, catalog_entry):
            lock_hash = npm_installer.lockfile_hash(final_dir)
            return CandidateInstallation(
                transaction=candidate.transaction, install_directory=final_dir, lock_hash=lock_hash,
                extra={"reused_existing_installation": "true"})

        runtimes = npm_installer.check_runtimes(catalog_entry)
        npm_installer.install_package(catalog_entry, candidate.install_directory,
                                      npm_executable=runtimes.get("npm"))
        npm_installer.validate_entrypoint(catalog_entry, candidate.install_directory)
        lock_hash = npm_installer.lockfile_hash(candidate.install_directory)
        return CandidateInstallation(
            transaction=candidate.transaction, install_directory=candidate.install_directory,
            lock_hash=lock_hash, extra={"reused_existing_installation": "false"})

    def validate_artifacts(self, candidate: CandidateInstallation, plan, catalog_entry) -> None:
        npm_installer.validate_entrypoint(catalog_entry, candidate.install_directory)

    def build_launch_spec(self, candidate: CandidateInstallation, catalog_entry) -> McpLaunchSpec:
        node = npm_installer.resolve_runtime("node")
        entrypoint = npm_installer.validate_entrypoint(catalog_entry, candidate.install_directory)
        return McpLaunchSpec(command=node, args=(entrypoint,))

    def cleanup_candidate(self, candidate: CandidateInstallation) -> None:
        if candidate.install_directory != candidate.transaction.final_directory:
            shutil.rmtree(candidate.install_directory, ignore_errors=True)


def _entrypoint_ok(directory, catalog_entry):
    try:
        npm_installer.validate_entrypoint(catalog_entry, directory)
        return True
    except McpError:
        return False
