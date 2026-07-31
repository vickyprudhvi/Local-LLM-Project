"""Phase G.3 — automatic, approval-driven provisioning for an approved-but-not-
installed MCP provider (Tasks 4, 9, 10, 16, 17).

Mirrors the shape of Phase F.1's `filesystem_access` flow (a plan, a pending
request carrying the ORIGINAL blocked request, single-use hash-bound approval,
a resume() that hands text back to the caller rather than answering directly)
but drives the generalized, installer-agnostic candidate-transaction pipeline
from Tasks 5-12 instead of a directory-only change.

`AutoProvisioningManager` never calls the user's requested business action —
`resume()` only returns the original text so the normal router / Phase G.1 /
Phase G.2 / Phase B / ToolExecutor pipeline can re-run it, exactly like every
other resumption path in this project.
"""

import os
import shutil
import threading
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Optional

import tools.config as app_config
from interaction_log import log_mcp_event
from mcp_layer.errors import McpError
from mcp_management.configuration_generator import (
    generate_config_dict_from_launch_spec,
    validate_generated,
    write_config,
)
from mcp_management.installer import GENERATED_CONFIG_FILENAME
from mcp_management.installers import ProvisioningTransaction, get_installer
from mcp_management.installers.python_venv import lock_file_hash as _python_lock_file_hash
from mcp_management.lifecycle import managed_config_path
from mcp_management.models import policy_fingerprint
from mcp_management.planner import (
    install_directory_for,
    managed_server_root,
    runtime_workspace_for,
)
from mcp_management.provisioning_models import (
    AutoProvisioningApproval,
    AutoProvisioningPlan,
    PendingAutoProvisioningRequest,
    PendingAutoProvisioningState,
    ProvisioningPlanStatus,
    ProvisioningResult,
)
from mcp_management.registry import (
    STATUS_INSTALLED,
    InstalledServer,
    atomic_write_json,
    get_installed,
    remove as registry_remove,
    upsert,
    utc_now,
)
from tools.models import (
    MCP_CANDIDATE_START_FAILED,
    MCP_EXPECTED_TOOL_MISSING,
    MCP_INSTALLATION_FAILED,
    MCP_INSTALLER_UNSUPPORTED,
    MCP_PROVISIONING_ALREADY_IN_PROGRESS,
    MCP_PROVISIONING_CONFIRMATION_MISMATCH,
    MCP_PROVISIONING_CONFIRMATION_REQUIRED,
    MCP_PROVISIONING_DECLINED,
    MCP_PROVISIONING_PLAN_EXPIRED,
    MCP_PROVISIONING_PLAN_INVALID,
    MCP_PROVISIONING_RESUME_FAILED,
    MCP_SERVER_NOT_APPROVED,
    hash_arguments,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_PROVISIONING_ATTEMPTS = 1


def require_auto_provisioning_approval(plan: AutoProvisioningPlan, approval):
    """Raise unless `approval` authorizes exactly `plan`, and it hasn't expired.

    A DISTINCT approval type from Phase F's `ProvisioningApproval` and Phase
    F.1's `FilesystemAccessApproval` (Task 4) — a Phase C tool confirmation or
    either of those can never be mistaken for this one.
    """
    if approval is None:
        raise McpError(MCP_PROVISIONING_CONFIRMATION_REQUIRED,
                       "Installing this MCP server requires explicit approval.")
    if not isinstance(approval, AutoProvisioningApproval):
        raise McpError(MCP_PROVISIONING_CONFIRMATION_MISMATCH,
                       "That confirmation does not authorize this installation.")
    if not approval.approved:
        raise McpError(MCP_PROVISIONING_DECLINED, "The user declined the installation.")
    if plan.is_expired():
        raise McpError(MCP_PROVISIONING_PLAN_EXPIRED,
                       "This provisioning plan has expired; prepare a new one.")
    expected = plan.compute_hash()
    if approval.plan_id != plan.plan_id or approval.plan_hash != expected:
        raise McpError(MCP_PROVISIONING_CONFIRMATION_MISMATCH,
                       "The approval does not match this provisioning plan; review it again.")


# ---- deterministic plan construction (Task 3) ----

def _executable_identity(catalog_entry):
    return {"npm": "node", "python_venv": "python"}.get(catalog_entry.installer_type,
                                                         catalog_entry.installer_type)


def _install_network_hosts(catalog_entry):
    if catalog_entry.installer_type == "npm":
        return ("registry.npmjs.org",)
    return catalog_entry.install_hosts


def _candidate_config_template(catalog_entry):
    """The deterministic, machine-independent shape of the config this entry
    will generate — everything the plan hash binds EXCEPT absolute paths
    resolved only at install time (Task 3: 'candidate managed configuration')."""
    return {
        "server_id": catalog_entry.server_id,
        "transport": catalog_entry.transport,
        "installer_type": catalog_entry.installer_type,
        "package_name": catalog_entry.package_name,
        "package_version": catalog_entry.package_version,
        "entrypoint_relative": catalog_entry.entrypoint_relative,
        "launch_module": catalog_entry.launch_module,
        "launch_arguments": list(catalog_entry.launch_arguments),
        "environment_allowlist": list(catalog_entry.required_environment_variables()),
        "tool_policy": policy_fingerprint(catalog_entry.default_tool_policy),
    }


def build_auto_plan(catalog_entry, request_id, original_user_text, base_dir=None,
                    managed_root=None, ttl_seconds=None) -> AutoProvisioningPlan:
    """Build the immutable, hashed auto-provisioning plan for one catalog entry.

    Derived ONLY from the trusted catalog entry plus the request id/text — never
    from model output (Task 3, Task 18: no user- or model-controlled package
    name, version, path, or permission).
    """
    if get_installer(catalog_entry.installer_type) is None:
        raise McpError(MCP_INSTALLER_UNSUPPORTED,
                       f"Installer type {catalog_entry.installer_type!r} is not supported.")
    if catalog_entry.requires_directory():
        # Out of scope for G.3 (Task: "no combined install-and-folder-access
        # plans"): a server needing a directory grant is not auto-provisioned.
        raise McpError(MCP_PROVISIONING_PLAN_INVALID,
                       f"{catalog_entry.display_name} requires a directory grant and cannot be "
                       "auto-provisioned.")

    base_dir = base_dir or _REPO_ROOT
    ttl_seconds = ttl_seconds if ttl_seconds is not None else _default_ttl_seconds()

    lock_hash = None
    if catalog_entry.installer_type == "python_venv":
        lock_hash = _python_lock_file_hash(os.path.join(_REPO_ROOT, catalog_entry.lock_file_relative))

    now = datetime.now(timezone.utc)
    plan = AutoProvisioningPlan(
        plan_id="", plan_hash="",
        request_id=request_id, original_user_text=original_user_text,
        catalog_id=catalog_entry.catalog_id, server_id=catalog_entry.server_id,
        display_name=catalog_entry.display_name,
        installer_type=catalog_entry.installer_type,
        exact_package=catalog_entry.package_name, exact_version=catalog_entry.package_version,
        lock_file_hash=lock_hash,
        executable_identity=_executable_identity(catalog_entry),
        expected_tools=catalog_entry.expected_tools,
        tool_policy_hash=hash_arguments(policy_fingerprint(catalog_entry.default_tool_policy)),
        environment_allowlist=catalog_entry.required_environment_variables(),
        install_network_hosts=_install_network_hosts(catalog_entry),
        runtime_network_policy="disabled",
        target_install_directory=install_directory_for(catalog_entry, base_dir, managed_root),
        candidate_config_hash=hash_arguments(_candidate_config_template(catalog_entry)),
        created_at=now.isoformat(timespec="seconds"),
        expires_at=(now + timedelta(seconds=ttl_seconds)).isoformat(timespec="seconds"),
    ).with_hash()
    return replace(plan, plan_id=f"autoplan_{plan.plan_hash[:16]}")


def _default_ttl_seconds():
    from mcp_management.provisioning_models import DEFAULT_PLAN_TTL_SECONDS

    return DEFAULT_PLAN_TTL_SECONDS


# ---- candidate MCP process validation (Task 10) ----

def _validate_candidate_process(config, catalog_entry, base_dir):
    """Start the ACTUAL candidate process, exactly like a real server, but
    never register its tools into the production ToolRegistry. Exact-name
    comparison only — never suffix matching."""
    from mcp_layer.external import start_server

    client = None
    try:
        try:
            client = start_server(config, base_dir=base_dir, allow_create=True)
        except McpError as e:
            raise McpError(MCP_CANDIDATE_START_FAILED,
                           f"The candidate MCP server failed to start ({e.code}).") from e
        try:
            raw_tools = client.list_tools(timeout=config.startup_timeout_seconds)
        except McpError as e:
            raise McpError(MCP_CANDIDATE_START_FAILED,
                           f"The candidate MCP server did not respond to tools/list ({e.code}).") from e
    finally:
        if client is not None:
            try:
                client.shutdown()
            except Exception:  # noqa: BLE001 — never mask the validation outcome
                pass

    discovered = {t.get("name") for t in raw_tools if isinstance(t, dict)}
    expected = set(catalog_entry.expected_tools)
    missing = expected - discovered
    if missing:
        raise McpError(MCP_EXPECTED_TOOL_MISSING,
                       f"The candidate server did not expose expected tool(s): {sorted(missing)}.")
    # default_permission is structurally forced to DENIED at catalog-load time
    # (mcp_management.catalog._build_policy), so any tool the candidate exposes
    # that the trusted policy never explicitly enabled is denied by construction
    # — never silently granted READ. Here we only need every catalog-ENABLED
    # tool to actually be present, so activation never claims a tool it cannot
    # deliver.
    enabled_policy_tools = {name for name, entry in catalog_entry.default_tool_policy.tools.items()
                            if entry.enabled}
    unmet = enabled_policy_tools - discovered
    if unmet:
        raise McpError(MCP_EXPECTED_TOOL_MISSING,
                       f"Catalog-enabled tool(s) {sorted(unmet)} were not exposed by the candidate.")
    return {"discovered_tool_count": len(discovered),
           "expected_tools_present": sorted(expected & discovered)}


# ---- the candidate installation transaction (Task 9) ----

def _run_transaction(plan: AutoProvisioningPlan, catalog_entry, base_dir, managed_root, registry_path):
    installer = get_installer(plan.installer_type)
    if installer is None:
        raise McpError(MCP_INSTALLER_UNSUPPORTED,
                       f"Installer type {plan.installer_type!r} is not supported.")

    transaction_id = uuid.uuid4().hex[:16]
    server_root = managed_server_root(plan.server_id, base_dir, managed_root)
    candidates_root = os.path.join(server_root, "candidates", transaction_id)
    final_dir = install_directory_for(catalog_entry, base_dir, managed_root)
    transaction = ProvisioningTransaction(
        transaction_id=transaction_id, server_id=plan.server_id, base_dir=base_dir or _REPO_ROOT,
        managed_root=str(managed_root or app_config.mcp_managed_root()),
        server_root=server_root, candidate_directory=candidates_root, final_directory=final_dir)

    candidate = installer.prepare_candidate(plan, catalog_entry, transaction)
    promoted = False
    try:
        candidate = installer.install_candidate(candidate, plan, catalog_entry)
        installer.validate_artifacts(candidate, plan, catalog_entry)

        workspace = runtime_workspace_for(catalog_entry, base_dir)
        launch_spec = installer.build_launch_spec(candidate, catalog_entry)
        raw_config = generate_config_dict_from_launch_spec(
            plan.server_id, plan.display_name, catalog_entry.transport, launch_spec, workspace,
            plan.environment_allowlist, catalog_entry.default_tool_policy,
            plan.catalog_id, plan.exact_version, plan.installer_type)
        config = validate_generated(raw_config)
        report = _validate_candidate_process(config, catalog_entry, base_dir)

        if os.path.realpath(candidate.install_directory) != os.path.realpath(transaction.final_directory):
            os.makedirs(os.path.dirname(final_dir), exist_ok=True)
            shutil.rmtree(final_dir, ignore_errors=True)
            os.rename(candidate.install_directory, final_dir)
            promoted = True
            candidate = replace(candidate, install_directory=final_dir,
                                extra={**candidate.extra,
                                      **({"venv_python": _reroot_venv_python(candidate, final_dir)}
                                         if "venv_python" in candidate.extra else {})})
            launch_spec = installer.build_launch_spec(candidate, catalog_entry)
            raw_config = generate_config_dict_from_launch_spec(
                plan.server_id, plan.display_name, catalog_entry.transport, launch_spec, workspace,
                plan.environment_allowlist, catalog_entry.default_tool_policy,
                plan.catalog_id, plan.exact_version, plan.installer_type)
            config = validate_generated(raw_config)

        generated_config_path = os.path.join(server_root, GENERATED_CONFIG_FILENAME)
        write_config(raw_config, generated_config_path)

        installed = InstalledServer(
            catalog_id=plan.catalog_id, installed_version=plan.exact_version, status=STATUS_INSTALLED,
            install_directory=final_dir, configuration_path=generated_config_path,
            installed_at=utc_now(), last_validated_at=utc_now(), last_validation_result="healthy",
            approved_directories=(), installer_type=plan.installer_type,
            catalog_entry_hash=plan.candidate_config_hash,
            lock_hash=candidate.lock_hash or plan.lock_file_hash,
            expected_tools_hash=hash_arguments(list(plan.expected_tools)),
            tool_policy_hash=plan.tool_policy_hash, last_known_good_version=plan.exact_version,
        )
        upsert(plan.server_id, installed, registry_path, base_dir, managed_root)
        installer.cleanup_candidate(candidate)

        return ProvisioningResult(
            server_id=plan.server_id, catalog_id=plan.catalog_id, installed_version=plan.exact_version,
            managed_config_path=generated_config_path,
            installed_state_hash=hash_arguments(installed.to_dict()),
            validation_summary=report, runtime_activation_required=True)
    except McpError:
        installer.cleanup_candidate(candidate)
        if promoted:
            shutil.rmtree(final_dir, ignore_errors=True)
        raise
    except Exception as e:  # noqa: BLE001 — normalize anything unexpected
        installer.cleanup_candidate(candidate)
        if promoted:
            shutil.rmtree(final_dir, ignore_errors=True)
        raise McpError(MCP_INSTALLATION_FAILED,
                       f"Provisioning failed unexpectedly ({type(e).__name__}).") from e
    finally:
        shutil.rmtree(candidates_root, ignore_errors=True)


def _reroot_venv_python(candidate, new_root):
    """After promoting a python_venv candidate directory, its venv python path
    still points at the OLD candidate location; recompute it under `new_root`."""
    old_python = candidate.extra["venv_python"]
    relative = os.path.relpath(old_python, candidate.install_directory)
    return os.path.join(new_root, relative)


# ---- the pending-request / approval facade (Tasks 4, 13, 16, 17) ----

class AutoProvisioningManager:
    """Coordinates Phase G.3. All paths are injectable so tests stay hermetic."""

    def __init__(self, catalog, base_dir=None, managed_root=None, registry_path=None):
        self.catalog = catalog
        self.base_dir = base_dir or _REPO_ROOT
        self.managed_root = managed_root
        self.registry_path = registry_path
        self._pending = {}
        self._plans = {}
        self._locks_guard = threading.Lock()
        self._locks = {}

    def _lock_for(self, server_id):
        with self._locks_guard:
            lock = self._locks.get(server_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[server_id] = lock
            return lock

    def is_eligible(self, catalog_entry) -> bool:
        return (get_installer(catalog_entry.installer_type) is not None
                and not catalog_entry.requires_directory())

    # ---- detection -> plan -> approval ----

    def begin_request(self, user_text, capability, catalog_entry) -> Optional[PendingAutoProvisioningRequest]:
        if not self.is_eligible(catalog_entry):
            return None
        request = PendingAutoProvisioningRequest(
            request_id=f"autoreq_{uuid.uuid4().hex[:12]}", original_user_text=user_text,
            capability=capability, catalog_id=catalog_entry.catalog_id, server_id=catalog_entry.server_id)
        self._pending[request.request_id] = request
        log_mcp_event("auto_provisioning_detected", capability=capability,
                     catalog_id=catalog_entry.catalog_id, server_id=catalog_entry.server_id,
                     state=request.state.value)
        return request

    def prepare_plan(self, request_id) -> Optional[AutoProvisioningPlan]:
        request = self._pending.get(request_id)
        if request is None:
            return None
        catalog_entry = self.catalog.get(request.catalog_id)
        if catalog_entry is None:
            raise McpError(MCP_SERVER_NOT_APPROVED,
                           f"{request.catalog_id!r} is not in the trusted MCP catalog.")
        plan = build_auto_plan(catalog_entry, request.request_id, request.original_user_text,
                               base_dir=self.base_dir, managed_root=self.managed_root)
        self._plans[plan.plan_id] = plan
        self._pending[request_id] = request.advanced(PendingAutoProvisioningState.AWAITING_APPROVAL,
                                                      plan_id=plan.plan_id)
        log_mcp_event("auto_provisioning_plan_prepared", catalog_id=plan.catalog_id,
                     server_id=plan.server_id, plan_id=plan.plan_id, plan_hash=plan.plan_hash,
                     installer_type=plan.installer_type, package_version=plan.exact_version)
        return plan

    def get_plan(self, plan_id) -> Optional[AutoProvisioningPlan]:
        return self._plans.get(plan_id)

    def pending(self, request_id) -> Optional[PendingAutoProvisioningRequest]:
        return self._pending.get(request_id)

    def decline(self, request_id):
        request = self._pending.get(request_id)
        if request is None:
            return None
        self._pending[request_id] = request.advanced(PendingAutoProvisioningState.DECLINED)
        log_mcp_event("auto_provisioning_declined", server_id=request.server_id)
        return request

    # ---- install -> validate -> activate -> resume ----

    def provision_and_activate(self, request_id, runtime_manager, approval=None,
                               confirmer=None) -> Optional[ProvisioningResult]:
        """Revalidate, install, validate, atomically activate, then hand off to
        Phase G.2 (`runtime_manager.ensure_started`). Returns None only when
        `request_id` is unknown (a bare 'yes' with no matching pending plan)."""
        request = self._pending.get(request_id)
        if request is None:
            return None
        plan = self._plans.get(request.plan_id) if request.plan_id else None
        if plan is None:
            raise McpError(MCP_PROVISIONING_PLAN_INVALID,
                           "That provisioning plan is unknown or has expired; prepare a new one.")

        if request.attempts >= MAX_PROVISIONING_ATTEMPTS:
            raise McpError(MCP_PROVISIONING_ALREADY_IN_PROGRESS,
                           "This request already attempted provisioning once.")

        if approval is None and confirmer is not None:
            approved = bool(confirmer(plan))
            approval = AutoProvisioningApproval(approved=approved, plan_id=plan.plan_id,
                                                plan_hash=plan.compute_hash())
        try:
            require_auto_provisioning_approval(plan, approval)
        except McpError as e:
            self._pending[request_id] = request.advanced(
                PendingAutoProvisioningState.DECLINED if e.code == MCP_PROVISIONING_DECLINED
                else PendingAutoProvisioningState.FAILED)
            log_mcp_event("auto_provisioning_approval", error_code=e.code,
                         catalog_id=plan.catalog_id, plan_id=plan.plan_id)
            raise

        # Revalidate: the catalog and installed state must still match what the
        # plan was built from (Task 4/18) — a catalog edit or a race with
        # another installer invalidates a stale plan rather than using it.
        catalog_entry = self.catalog.get(plan.catalog_id)
        if catalog_entry is None or catalog_entry.server_id != plan.server_id:
            self._pending[request_id] = request.advanced(PendingAutoProvisioningState.FAILED)
            raise McpError(MCP_PROVISIONING_PLAN_INVALID,
                           "The plan no longer references a trusted catalog entry.")
        if hash_arguments(_candidate_config_template(catalog_entry)) != plan.candidate_config_hash:
            self._pending[request_id] = request.advanced(PendingAutoProvisioningState.INVALIDATED)
            raise McpError(MCP_PROVISIONING_PLAN_INVALID,
                           "The trusted catalog changed since this plan was approved; prepare a new one.")

        already = get_installed(plan.server_id, self.registry_path, self.base_dir, self.managed_root)
        if already is not None:
            # Installed since the plan was prepared (e.g. a concurrent approval
            # elsewhere) — reuse it, never reinstall.
            self._pending[request_id] = request.advanced(PendingAutoProvisioningState.READY)
            return ProvisioningResult(
                server_id=plan.server_id, catalog_id=plan.catalog_id,
                installed_version=already.installed_version,
                managed_config_path=already.configuration_path,
                installed_state_hash=hash_arguments(already.to_dict()),
                validation_summary={}, runtime_activation_required=True)

        lock = self._lock_for(plan.server_id)
        if not lock.acquire(blocking=False):
            raise McpError(MCP_PROVISIONING_ALREADY_IN_PROGRESS,
                           f"Installation of {plan.server_id!r} is already in progress.")
        try:
            request = request.advanced(PendingAutoProvisioningState.INSTALLING, attempts=request.attempts + 1)
            self._pending[request_id] = request
            log_mcp_event("auto_provisioning_approval", catalog_id=plan.catalog_id,
                         plan_id=plan.plan_id, approval_result="approved")

            pre_installed, config_path, pre_config_raw = self._capture_pre_state(plan.server_id)
            try:
                result = _run_transaction(plan, catalog_entry, self.base_dir, self.managed_root,
                                          self.registry_path)
            except McpError as e:
                self._pending[request_id] = request.advanced(PendingAutoProvisioningState.FAILED)
                log_mcp_event("auto_provisioning_installation", error_code=e.code,
                             catalog_id=plan.catalog_id, server_id=plan.server_id, plan_id=plan.plan_id)
                raise

            request = request.advanced(PendingAutoProvisioningState.ACTIVATING)
            self._pending[request_id] = request
            try:
                runtime_manager.ensure_started(plan.server_id, expected_tools=catalog_entry.expected_tools)
            except McpError as e:
                self._restore_pre_state(plan.server_id, pre_installed, config_path, pre_config_raw)
                self._pending[request_id] = request.advanced(PendingAutoProvisioningState.FAILED)
                log_mcp_event("auto_provisioning_activation", error_code=e.code,
                             catalog_id=plan.catalog_id, server_id=plan.server_id, plan_id=plan.plan_id)
                raise McpError(MCP_PROVISIONING_RESUME_FAILED,
                               f"The server installed but could not be activated ({e.code}).") from e

            self._pending[request_id] = request.advanced(PendingAutoProvisioningState.READY)
            log_mcp_event("auto_provisioning_installation", catalog_id=plan.catalog_id,
                         server_id=plan.server_id, plan_id=plan.plan_id,
                         package_version=plan.exact_version, installation_result="installed")
            return result
        finally:
            lock.release()

    def _capture_pre_state(self, server_id):
        pre_installed = get_installed(server_id, self.registry_path, self.base_dir, self.managed_root)
        config_path = managed_config_path(server_id, self.base_dir, self.managed_root)
        pre_config_raw = _read_json(config_path)
        return pre_installed, config_path, pre_config_raw

    def _restore_pre_state(self, server_id, pre_installed, config_path, pre_config_raw):
        try:
            if pre_installed is not None:
                upsert(server_id, pre_installed, self.registry_path, self.base_dir, self.managed_root)
            else:
                registry_remove(server_id, self.registry_path, self.base_dir, self.managed_root)
            if pre_config_raw is not None:
                atomic_write_json(config_path, pre_config_raw)
            elif os.path.isfile(config_path):
                os.unlink(config_path)
        except OSError:
            pass

    # ---- resumption ----

    def resume(self, request_id):
        """Return the original request text so the NORMAL pipeline can re-run
        it. Never calls the newly installed MCP tool itself."""
        request = self._pending.get(request_id)
        if request is None or request.state is not PendingAutoProvisioningState.READY:
            return None
        self._pending[request_id] = request.advanced(PendingAutoProvisioningState.RESUMED)
        log_mcp_event("auto_provisioning_resumed", server_id=request.server_id,
                     original_request_id=request_id)
        return request.original_user_text


def _read_json(path):
    import json

    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
