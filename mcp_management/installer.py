"""Phase F — installation as a transaction.

Order: approve -> check runtimes -> install into a STAGING directory -> validate the
entrypoint -> generate the configuration -> start and validate the real server ->
atomically promote staging to the immutable version directory -> write the audit
record -> update the registry.

Any failure removes the staging directory, leaves the registry untouched, does not
activate anything, and preserves a previously installed healthy version. Nothing
here calls a newly installed MCP tool: runtime execution always goes through the
normal ToolExecutor pipeline.
"""

import os
import shutil

from mcp_layer.errors import McpError
from mcp_management import audit, npm_installer
from mcp_management.approval import require_approval
from mcp_management.catalog import McpCatalogEntry
from mcp_management.configuration_generator import (
    generate_config_dict,
    validate_generated,
    write_config,
    write_permissions_snapshot,
)
from mcp_management.models import McpProvisioningPlan
from mcp_management.planner import managed_server_root
from mcp_management.registry import (
    STATUS_INSTALLED,
    InstalledServer,
    upsert,
    utc_now,
)
from tools.models import MCP_INSTALLATION_FAILED

GENERATED_CONFIG_FILENAME = "server.json"
PERMISSIONS_FILENAME = "permissions.json"
CURRENT_FILENAME = "current.json"

# The runtime that launches the server for each installer type.
_LAUNCH_RUNTIME = {"npm": "node"}


def _rmtree(path):
    shutil.rmtree(path, ignore_errors=True)


def _entrypoint_ok(directory, plan):
    try:
        npm_installer.validate_entrypoint(plan, directory)
        return True
    except McpError:
        return False


def install(plan: McpProvisioningPlan, entry: McpCatalogEntry, approval,
            base_dir=None, registry_path=None, managed_root=None,
            npm_runner=None, install_timeout=None, run_write_smoke_test=False,
            validate_fn=None, start_server_fn=None, force_reinstall=False):
    """Run the full installation transaction. Returns a result dict on success."""
    # 1. Approval first — nothing is created before this passes.
    require_approval(plan, approval)

    # 2. Required runtimes must already exist; nothing is auto-installed.
    runtimes = npm_installer.check_runtimes(plan)
    launch_runtime_name = _LAUNCH_RUNTIME.get(plan.package_manager, "node")
    launch_executable = runtimes.get(launch_runtime_name) or npm_installer.resolve_runtime(
        launch_runtime_name)

    server_root = managed_server_root(plan.server_id, base_dir, managed_root)
    final_dir = str(plan.install_directory)
    staging_dir = os.path.join(server_root, f".staging-{plan.package_version}-{os.getpid()}")
    os.makedirs(server_root, exist_ok=True)

    previous_version = None
    from mcp_management.registry import get_installed

    existing = get_installed(plan.server_id, registry_path, base_dir, managed_root)
    if existing is not None and existing.installed_version != plan.package_version:
        previous_version = existing.installed_version

    installer_result = {}
    lock_hash = None
    promoted = False
    reused_existing = False

    try:
        if _entrypoint_ok(final_dir, plan) and not force_reinstall:
            # Idempotent: this exact version is already installed and intact.
            reused_existing = True
            install_dir = final_dir
            lock_hash = npm_installer.lockfile_hash(final_dir)
            installer_result = {"reused_existing_installation": True}
        else:
            _rmtree(staging_dir)
            installer_result = npm_installer.install_package(
                plan, staging_dir, npm_executable=runtimes.get("npm"),
                runner=npm_runner, timeout=install_timeout,
            )
            npm_installer.validate_entrypoint(plan, staging_dir)
            lock_hash = npm_installer.lockfile_hash(staging_dir)
            install_dir = staging_dir

        # 3. Generate + Phase E-validate the configuration for the current location.
        entrypoint = npm_installer.validate_entrypoint(plan, install_dir)
        raw_config = generate_config_dict(plan, launch_executable, entrypoint)
        config = validate_generated(raw_config)

        # 4. Start the real server and validate it (always shuts down afterwards).
        validate = validate_fn or _default_validate
        report = validate(config, plan, entry.expected_tools, entrypoint, base_dir,
                          run_write_smoke_test, start_server_fn)

        # 5. Promote staging -> immutable version directory.
        if not reused_existing:
            _rmtree(final_dir)
            os.makedirs(os.path.dirname(final_dir), exist_ok=True)
            os.rename(staging_dir, final_dir)
            promoted = True
            entrypoint = npm_installer.validate_entrypoint(plan, final_dir)
            raw_config = generate_config_dict(plan, launch_executable, entrypoint)
            config = validate_generated(raw_config)

        # 6. Persist generated artifacts inside the managed server root.
        generated_config_path = os.path.join(server_root, GENERATED_CONFIG_FILENAME)
        write_config(raw_config, generated_config_path)
        write_permissions_snapshot(plan, os.path.join(server_root, PERMISSIONS_FILENAME))

        record = audit.build_install_record(
            plan, final_dir, lock_hash=lock_hash, validation=report.summary(),
            installer_result=installer_result, previous_version=previous_version,
        )
        audit.write_install_record(record, server_root)
        from mcp_management.registry import atomic_write_json

        atomic_write_json(os.path.join(server_root, CURRENT_FILENAME), {
            "server_id": plan.server_id,
            "catalog_id": plan.catalog_id,
            "version": plan.package_version,
            "install_directory": final_dir,
            "configuration_path": generated_config_path,
            "updated_at": utc_now(),
        })

        # 7. Registry last: only a fully validated install is recorded.
        installed = InstalledServer(
            catalog_id=plan.catalog_id,
            installed_version=plan.package_version,
            status=STATUS_INSTALLED,
            install_directory=final_dir,
            configuration_path=generated_config_path,
            installed_at=utc_now(),
            last_validated_at=utc_now(),
            last_validation_result="healthy",
            approved_directories=tuple(str(d) for d in plan.requested_directories),
        )
        upsert(plan.server_id, installed, registry_path, base_dir, managed_root)

        return {
            "server_id": plan.server_id,
            "catalog_id": plan.catalog_id,
            "version": plan.package_version,
            "install_directory": final_dir,
            "generated_config_path": generated_config_path,
            "raw_config": raw_config,
            "config": config,
            "validation": report.summary(),
            "install_record": record,
            "installed": installed,
            "reused_existing_installation": reused_existing,
        }
    except McpError:
        _rollback(staging_dir, final_dir if promoted and not reused_existing else None)
        raise
    except Exception as e:  # noqa: BLE001 — normalize anything unexpected
        _rollback(staging_dir, final_dir if promoted and not reused_existing else None)
        raise McpError(MCP_INSTALLATION_FAILED,
                       f"The installation failed unexpectedly ({type(e).__name__}).") from e
    finally:
        _rmtree(staging_dir)


def _rollback(staging_dir, promoted_dir):
    """Remove partial artifacts. A previously installed version is left untouched."""
    _rmtree(staging_dir)
    if promoted_dir:
        _rmtree(promoted_dir)


def _default_validate(config, plan, expected_tools, entrypoint, base_dir,
                      run_write_smoke_test, start_server_fn):
    from mcp_management.validator import validate_installation

    return validate_installation(
        config, plan, expected_tools=expected_tools, entrypoint=entrypoint,
        base_dir=base_dir, run_write_smoke_test=run_write_smoke_test,
        start_server_fn=start_server_fn,
    )
