"""Phase F — the provisioning facade: detect, plan, approve, install, activate, resume.

Holds the pending-request state machine so the ORIGINAL user request survives
provisioning and can be re-run afterwards. Resumption deliberately returns the
original text to the caller instead of invoking the new MCP tool directly: the
request goes back through routing, the Phase B shortlist, and the ToolExecutor, so
the newly registered McpTool is selected and executed by the normal pipeline.

At most ONE provisioning attempt per original request, which makes a
detect -> install -> detect loop impossible.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone

from interaction_log import log_mcp_event
from mcp_layer.errors import McpError
from mcp_management import lifecycle
from mcp_management.approval import (
    collect_approval,
    require_approval,
    require_filesystem_access_approval,
)
from mcp_management.capability_detector import detect_capability
from mcp_management.catalog import load_catalog
from mcp_management.filesystem_access import (
    DEFAULT_PLAN_TTL_SECONDS,
    FilesystemAccessApproval,
    FilesystemAccessOperation,
    FilesystemAccessPlan,
    PendingFilesystemAccessRequest,
    PendingFilesystemAccessState,
)
from mcp_management.filesystem_access_update import update_filesystem_access
from mcp_management.installer import install
from mcp_management.models import PendingCapabilityRequest, PendingRequestState
from mcp_management.planner import build_plan, validate_approved_directory
from mcp_management.registry import get_installed, load_registry
from tools.models import (
    MCP_FILESYSTEM_ACCESS_ALREADY_GRANTED,
    MCP_FILESYSTEM_ACCESS_LOOP_PREVENTED,
    MCP_FILESYSTEM_ACCESS_NOT_INSTALLED,
    MCP_FILESYSTEM_ACCESS_PLAN_INVALID,
    MCP_FILESYSTEM_LAST_ROOT_REQUIRED,
    MCP_FILESYSTEM_ROOT_NOT_FOUND,
    MCP_NOT_INSTALLED,
    MCP_PROVISIONING_LOOP_PREVENTED,
    MCP_SERVER_NOT_APPROVED,
)

MAX_PROVISIONING_ATTEMPTS = 1
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class McpProvisioningManager:
    """Coordinates Phase F. All paths are injectable so tests stay hermetic."""

    def __init__(self, catalog=None, base_dir=None, managed_root=None,
                 registry_path=None, active_path=None, catalog_path=None):
        self.base_dir = base_dir or _REPO_ROOT
        self.managed_root = managed_root
        self.registry_path = registry_path
        self.active_path = active_path
        self.catalog = catalog if catalog is not None else load_catalog(
            catalog_path, self.base_dir)
        self._pending = {}
        self._plans = {}
        self._filesystem_pending = {}
        self._filesystem_plans = {}

    # ---- state ----

    def installed_server_ids(self):
        return tuple(load_registry(self.registry_path, self.base_dir, self.managed_root))

    def status(self):
        return {
            "catalog_version": self.catalog.catalog_version,
            "catalog_entries": [e.catalog_id for e in self.catalog.entries.values()],
            "installed": lifecycle.list_installed(self.base_dir, self.managed_root,
                                                  self.registry_path),
            "pending_requests": {rid: req.state.value for rid, req in self._pending.items()},
        }

    # ---- detection ----

    def detect(self, user_text):
        return detect_capability(user_text, self.catalog, self.installed_server_ids())

    def begin_request(self, user_text):
        """Detect the capability and open a pending request when provisioning is needed.

        Returns (detection, pending_request_or_None). A pending request is only
        created when an approved catalog entry exists and is NOT already installed.
        """
        detection = self.detect(user_text)
        if not detection.requires_mcp or detection.error_code or not detection.recommended_catalog_id:
            return detection, None

        entry = self.catalog.get(detection.recommended_catalog_id)
        installed = get_installed(entry.server_id, self.registry_path, self.base_dir,
                                 self.managed_root)
        if installed is not None:
            # Already provisioned: reuse it, no plan or approval needed.
            return detection, None

        request = PendingCapabilityRequest(
            request_id=f"req_{uuid.uuid4().hex[:12]}",
            original_user_text=user_text,
            required_capability=detection.capability,
            selected_catalog_id=entry.catalog_id,
        )
        self._pending[request.request_id] = request
        log_mcp_event("capability_detected", catalog_id=entry.catalog_id,
                      server_id=entry.server_id, state=request.state.value)
        return detection, request

    # ---- planning ----

    def prepare_plan(self, catalog_id, requested_directories=(), allow_broad=False,
                     allow_create=False, request_id=None):
        """Build (and remember) a provisioning plan for an approved catalog entry."""
        entry = self.catalog.get(catalog_id)
        if entry is None:
            raise McpError(MCP_SERVER_NOT_APPROVED,
                           f"{catalog_id!r} is not in the trusted MCP catalog.")
        plan = build_plan(entry, requested_directories=requested_directories,
                          base_dir=self.base_dir, managed_root=self.managed_root,
                          allow_broad=allow_broad, allow_create=allow_create)
        self._plans[plan.plan_id] = (plan, entry)
        if request_id and request_id in self._pending:
            self._pending[request_id] = self._pending[request_id].advanced(
                PendingRequestState.AWAITING_APPROVAL, plan_id=plan.plan_id)
        log_mcp_event("plan_prepared", catalog_id=entry.catalog_id, server_id=entry.server_id,
                      plan_id=plan.plan_id, plan_hash=plan.plan_hash,
                      package_version=plan.package_version,
                      approved_directory_count=len(plan.requested_directories),
                      environment_variable_names=list(plan.requested_environment_variables))
        return plan

    def get_plan(self, plan_id):
        return self._plans.get(plan_id, (None, None))[0]

    # ---- provisioning ----

    def provision(self, plan, approval=None, request_id=None, confirmer=None,
                  npm_runner=None, validate_fn=None, start_server_fn=None,
                  run_write_smoke_test=False, install_timeout=None,
                  force_reinstall=False, activate=True):
        """Install and (optionally) activate a planned server.

        `approval` must authorize this exact plan. When omitted, `confirmer` is used
        to collect one — so a caller that supplies neither gets
        MCP_PROVISIONING_CONFIRMATION_REQUIRED and nothing is installed.
        """
        stored_plan, entry = self._plans.get(plan.plan_id, (None, None))
        if entry is None:
            entry = self.catalog.get(plan.catalog_id)
        if entry is None:
            raise McpError(MCP_SERVER_NOT_APPROVED,
                           "The plan does not reference a trusted catalog entry.")

        request = self._pending.get(request_id) if request_id else None
        if request is not None:
            if request.attempts >= MAX_PROVISIONING_ATTEMPTS:
                log_mcp_event("provisioning_blocked", error_code=MCP_PROVISIONING_LOOP_PREVENTED,
                              catalog_id=entry.catalog_id, plan_id=plan.plan_id)
                raise McpError(MCP_PROVISIONING_LOOP_PREVENTED,
                               "This request already attempted provisioning once.")
            request = request.advanced(PendingRequestState.INSTALLING,
                                       attempts=request.attempts + 1)
            self._pending[request_id] = request

        if approval is None and confirmer is not None:
            approval = collect_approval(plan, confirmer)
        # Enforce before touching the filesystem (also re-checked inside install()).
        try:
            require_approval(plan, approval)
        except McpError as e:
            if request is not None:
                self._pending[request_id] = request.advanced(PendingRequestState.DECLINED)
            log_mcp_event("provisioning_approval", error_code=e.code,
                          catalog_id=entry.catalog_id, plan_id=plan.plan_id,
                          approval_result="declined_or_missing")
            raise

        log_mcp_event("provisioning_approval", catalog_id=entry.catalog_id,
                      plan_id=plan.plan_id, approval_result="approved")
        try:
            result = install(
                plan, entry, approval, base_dir=self.base_dir,
                registry_path=self.registry_path, managed_root=self.managed_root,
                npm_runner=npm_runner, install_timeout=install_timeout,
                run_write_smoke_test=run_write_smoke_test, validate_fn=validate_fn,
                start_server_fn=start_server_fn, force_reinstall=force_reinstall,
            )
        except McpError as e:
            if request is not None:
                self._pending[request_id] = request.advanced(PendingRequestState.FAILED)
            log_mcp_event("installation", error_code=e.code, catalog_id=entry.catalog_id,
                          server_id=entry.server_id, plan_id=plan.plan_id,
                          installation_result="failed")
            raise

        if activate:
            # Writes only inside the managed root; the committed template is untouched.
            lifecycle.activate(result["raw_config"], self.base_dir, self.managed_root,
                               self.registry_path)
        validation = result.get("validation") or {}
        log_mcp_event("installation", catalog_id=entry.catalog_id, server_id=entry.server_id,
                      plan_id=plan.plan_id, package_version=plan.package_version,
                      installation_result="installed", validation_result="healthy",
                      registered_tool_count=validation.get("registered_tool_count"),
                      denied_tool_count=validation.get("denied_tool_count"),
                      discovered_tool_count=validation.get("discovered_tool_count"))
        if request is not None:
            self._pending[request_id] = request.advanced(PendingRequestState.READY)
        return result

    # ---- resumption ----

    def resume(self, request_id):
        """Return the original request text so the NORMAL pipeline can re-run it.

        The manager never calls the newly installed MCP tool itself.
        """
        request = self._pending.get(request_id)
        if request is None:
            return None
        if request.state is not PendingRequestState.READY:
            return None
        self._pending[request_id] = request.advanced(PendingRequestState.RESUMED)
        log_mcp_event("request_resumed", catalog_id=request.selected_catalog_id,
                      state=PendingRequestState.RESUMED.value)
        return request.original_user_text

    def pending(self, request_id):
        return self._pending.get(request_id)

    # ---- management passthroughs ----

    def disable(self, server_id):
        return lifecycle.disable(server_id, self.base_dir, self.managed_root,
                                 self.registry_path)

    def enable(self, server_id):
        return lifecycle.enable(server_id, self.base_dir, self.managed_root,
                                self.registry_path)

    def uninstall(self, server_id):
        return lifecycle.uninstall(server_id, self.base_dir, self.managed_root,
                                   self.registry_path)

    def repair(self, server_id, reinstall_fn=None):
        return lifecycle.repair(server_id, self.catalog, self.base_dir, self.managed_root,
                                self.registry_path, reinstall_fn)

    def check_for_update(self, server_id):
        return lifecycle.check_for_update(server_id, self.catalog, self.base_dir,
                                          self.managed_root, self.registry_path)

    def require_installed(self, server_id):
        entry = get_installed(server_id, self.registry_path, self.base_dir, self.managed_root)
        if entry is None:
            raise McpError(MCP_NOT_INSTALLED, f"Server {server_id!r} is not installed.")
        return entry

    # ---- Phase F.1: filesystem access-root expansion on an already-installed server ----

    def _require_installed_for_access(self, server_id):
        installed = get_installed(server_id, self.registry_path, self.base_dir, self.managed_root)
        if installed is None:
            raise McpError(MCP_FILESYSTEM_ACCESS_NOT_INSTALLED,
                           f"Server {server_id!r} is not installed.")
        return installed

    def list_filesystem_access(self, server_id):
        installed = self._require_installed_for_access(server_id)
        return tuple(sorted({os.path.realpath(d) for d in installed.approved_directories}))

    def begin_filesystem_access_request(self, original_user_text, requested_path,
                                        proposed_root, server_id):
        """Open a pending filesystem-access request for an outside-root failure
        already classified by the caller. Mirrors begin_request()'s shape."""
        request = PendingFilesystemAccessRequest(
            request_id=f"fsreq_{uuid.uuid4().hex[:12]}",
            original_user_text=original_user_text,
            requested_path=requested_path,
            proposed_root=proposed_root,
            server_id=server_id,
        )
        self._filesystem_pending[request.request_id] = request
        log_mcp_event("filesystem_access_detected", server_id=server_id,
                      state=request.state.value)
        return request

    def prepare_filesystem_access_plan(self, server_id, directory,
                                       operation=FilesystemAccessOperation.ADD_ROOT,
                                       request_id=None, requested_path=None,
                                       original_user_text="",
                                       ttl_seconds=DEFAULT_PLAN_TTL_SECONDS):
        """Build (and remember) a filesystem-access plan for an installed server.

        ADD_ROOT screens `directory` through the same forbidden/broad-location
        rules Phase F install-time uses. REMOVE_ROOT requires `directory` to be
        an existing approved root and refuses to remove the last one.
        """
        installed = self._require_installed_for_access(server_id)
        current = tuple(sorted({os.path.realpath(d) for d in installed.approved_directories}))

        if operation == FilesystemAccessOperation.ADD_ROOT:
            validated_dir = validate_approved_directory(directory, base_dir=self.base_dir,
                                                         allow_broad=False, allow_create=False)
            if validated_dir in current:
                raise McpError(MCP_FILESYSTEM_ACCESS_ALREADY_GRANTED,
                               "That directory is already approved.")
            proposed = tuple(sorted(set(current) | {validated_dir}))
        elif operation == FilesystemAccessOperation.REMOVE_ROOT:
            validated_dir = os.path.realpath(str(directory))
            if validated_dir not in current:
                raise McpError(MCP_FILESYSTEM_ROOT_NOT_FOUND,
                               "That directory is not currently approved.")
            proposed = tuple(d for d in current if d != validated_dir)
            if not proposed:
                raise McpError(MCP_FILESYSTEM_LAST_ROOT_REQUIRED,
                               "At least one approved directory must remain.")
        else:
            raise McpError(MCP_FILESYSTEM_ACCESS_PLAN_INVALID,
                           f"Unsupported filesystem access operation {operation!r}.")

        now = datetime.now(timezone.utc)
        plan = FilesystemAccessPlan(
            plan_id=f"fsplan_{uuid.uuid4().hex[:16]}",
            server_id=server_id,
            catalog_id=installed.catalog_id,
            operation=operation,
            requested_directory=validated_dir,
            current_allowed_directories=current,
            proposed_allowed_directories=proposed,
            requested_path=requested_path,
            original_user_text=original_user_text,
            created_at=now.isoformat(timespec="seconds"),
            expires_at=(now + timedelta(seconds=ttl_seconds)).isoformat(timespec="seconds"),
        ).with_hash()
        self._filesystem_plans[plan.plan_id] = plan
        if request_id and request_id in self._filesystem_pending:
            self._filesystem_pending[request_id] = self._filesystem_pending[request_id].advanced(
                PendingFilesystemAccessState.AWAITING_APPROVAL, plan_id=plan.plan_id)
        log_mcp_event("filesystem_access_plan_prepared", server_id=server_id, plan_id=plan.plan_id,
                      plan_hash=plan.plan_hash, operation=operation.value,
                      added_directory_count=max(0, len(proposed) - len(current)),
                      removed_directory_count=max(0, len(current) - len(proposed)),
                      resulting_directory_count=len(proposed))
        return plan

    def get_filesystem_access_plan(self, plan_id):
        return self._filesystem_plans.get(plan_id)

    def apply_filesystem_access(self, plan, approval=None, request_id=None, confirmer=None,
                                start_server_fn=None, validate_fn=None):
        """Apply an approved filesystem-access plan. Never calls npm.

        `approval` must authorize this exact plan (hash-bound, not expired). The
        server's CURRENT approved directories are re-read and must still match
        `plan.current_allowed_directories` — if they drifted since the plan was
        prepared (including from a prior application of this same plan), the plan
        is stale and must be re-prepared. This also makes every plan single-use.
        """
        request = self._filesystem_pending.get(request_id) if request_id else None
        if request is not None:
            if request.provisioning_attempts >= MAX_PROVISIONING_ATTEMPTS:
                log_mcp_event("filesystem_access_blocked",
                              error_code=MCP_FILESYSTEM_ACCESS_LOOP_PREVENTED,
                              server_id=plan.server_id, plan_id=plan.plan_id)
                raise McpError(MCP_FILESYSTEM_ACCESS_LOOP_PREVENTED,
                               "This request already attempted a filesystem access change once.")
            request = request.advanced(PendingFilesystemAccessState.APPLYING,
                                       attempts=request.provisioning_attempts + 1)
            self._filesystem_pending[request_id] = request

        if approval is None and confirmer is not None:
            approved = bool(confirmer(plan))
            approval = FilesystemAccessApproval(approved=approved, plan_id=plan.plan_id,
                                                plan_hash=plan.compute_hash())
        try:
            require_filesystem_access_approval(plan, approval)

            installed = self._require_installed_for_access(plan.server_id)
            current_now = tuple(sorted({os.path.realpath(d) for d in installed.approved_directories}))
            if current_now != plan.current_allowed_directories:
                raise McpError(MCP_FILESYSTEM_ACCESS_PLAN_INVALID,
                               "The server's approved directories changed since this plan was "
                               "prepared; prepare a new plan.")
        except McpError as e:
            if request is not None:
                self._filesystem_pending[request_id] = request.advanced(
                    PendingFilesystemAccessState.DECLINED if e.code.endswith("_DECLINED")
                    else PendingFilesystemAccessState.FAILED)
            log_mcp_event("filesystem_access_approval", error_code=e.code,
                          server_id=plan.server_id, plan_id=plan.plan_id)
            raise

        log_mcp_event("filesystem_access_approval", server_id=plan.server_id, plan_id=plan.plan_id,
                      approval_result="approved")
        try:
            result = update_filesystem_access(
                plan, base_dir=self.base_dir, managed_root=self.managed_root,
                registry_path=self.registry_path, start_server_fn=start_server_fn,
                validate_fn=validate_fn,
            )
        except McpError as e:
            if request is not None:
                self._filesystem_pending[request_id] = request.advanced(PendingFilesystemAccessState.FAILED)
            log_mcp_event("filesystem_access_update", error_code=e.code,
                          server_id=plan.server_id, plan_id=plan.plan_id)
            raise

        log_mcp_event("filesystem_access_update", server_id=plan.server_id, plan_id=plan.plan_id,
                      operation=(plan.operation.value if isinstance(plan.operation, FilesystemAccessOperation)
                                else str(plan.operation)),
                      resulting_directory_count=len(plan.proposed_allowed_directories))
        if request is not None:
            self._filesystem_pending[request_id] = request.advanced(PendingFilesystemAccessState.READY)
        return result

    def resume_filesystem_access(self, request_id):
        """Return the original request text so the NORMAL pipeline can re-run it.

        The manager never calls the newly-accessible file/tool itself.
        """
        request = self._filesystem_pending.get(request_id)
        if request is None or request.state is not PendingFilesystemAccessState.READY:
            return None
        self._filesystem_pending[request_id] = request.advanced(PendingFilesystemAccessState.RESUMED)
        log_mcp_event("filesystem_access_resumed", server_id=request.server_id,
                      original_request_id=request_id)
        return request.original_user_text

    def decline_filesystem_access(self, request_id):
        request = self._filesystem_pending.get(request_id)
        if request is None:
            return None
        self._filesystem_pending[request_id] = request.advanced(PendingFilesystemAccessState.DECLINED)
        log_mcp_event("filesystem_access_declined", server_id=request.server_id)
        return request

    def pending_filesystem_access(self, request_id):
        return self._filesystem_pending.get(request_id)
