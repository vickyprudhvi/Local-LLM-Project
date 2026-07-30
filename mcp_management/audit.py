"""Phase F — the immutable installation record.

Written once per successful installation next to the managed install. It captures
what was installed, from where, under which approved plan, and how validation went
— enough to audit or reproduce the install.

Deliberately excluded: secret values, authentication tokens, the child
environment, file contents, and raw user conversations. Environment variables are
recorded by NAME only.
"""

import os

from mcp_management.models import McpProvisioningPlan, policy_fingerprint
from mcp_management.registry import atomic_write_json, utc_now
from tools.models import hash_arguments

APP_VERSION = "phase-f"
INSTALL_RECORD_FILENAME = "install-record.json"


def build_install_record(plan: McpProvisioningPlan, install_directory, lock_hash=None,
                         validation=None, installer_result=None, previous_version=None):
    """Assemble the installation record (names and hashes only — no secrets)."""
    return {
        "catalog_id": plan.catalog_id,
        "server_id": plan.server_id,
        "package_name": plan.package_name,
        "package_version": plan.package_version,
        "package_source": plan.package_source,
        "package_lock_sha256": lock_hash,
        "installed_at": utc_now(),
        "application_version": APP_VERSION,
        "plan_id": plan.plan_id,
        "plan_hash": plan.plan_hash or plan.compute_hash(),
        "install_directory": str(install_directory),
        "runtime_workspace": str(plan.runtime_workspace),
        "approved_directories": [str(d) for d in plan.requested_directories],
        # Names only. Values are never stored.
        "approved_environment_variable_names": list(plan.requested_environment_variables),
        "permission_policy_hash": hash_arguments(policy_fingerprint(plan.proposed_tool_policy)),
        "permission_policy": policy_fingerprint(plan.proposed_tool_policy),
        "validation": validation or {},
        "installer_result": installer_result or {},
        "previous_version": previous_version,
        # Always false in Phase F: --ignore-scripts is unconditional.
        "lifecycle_scripts_allowed": False,
    }


def write_install_record(record, server_root):
    path = os.path.join(server_root, INSTALL_RECORD_FILENAME)
    atomic_write_json(path, record)
    return path


def read_install_record(server_root):
    import json

    path = os.path.join(server_root, INSTALL_RECORD_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
