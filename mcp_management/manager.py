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

from interaction_log import log_mcp_event
from mcp_layer.errors import McpError
from mcp_management import lifecycle
from mcp_management.approval import collect_approval, require_approval
from mcp_management.capability_detector import detect_capability
from mcp_management.catalog import load_catalog
from mcp_management.installer import install
from mcp_management.models import PendingCapabilityRequest, PendingRequestState
from mcp_management.planner import build_plan
from mcp_management.registry import get_installed, load_registry
from tools.models import (
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
