"""Phase F.1 hotfix — deterministic MCP runtime session replacement.

Persisting a new approved root in the managed `server.json` (mcp_management's job)
is only half of an access change. The already-running assistant process holds an
`McpSession` whose `McpClient` talks to an already-launched child process — that
process was started with the OLD argv, and no amount of rewriting the config file
on disk changes what that live process already agreed to serve. Every `McpTool`
discovered from it is a plain Python object bound to that SAME client. Restarting
the runtime is therefore a distinct step from updating the config: the old session
must be stopped, its tools unregistered, a fresh process started from the NEW
config, fresh tools registered against the NEW client, and the live server asked
directly (`list_allowed_directories`) to prove it actually agrees with what the
config file now says — never assumed from the write alone.

`ActiveMcpRuntime` is the one authoritative mutable holder for "the current
session" so callers (assistant.py) never keep a stale local variable pointing at an
already-shutdown session. `McpRuntimeManager.replace_active_session` is the single
deterministic coordinator for the stop -> unregister -> bootstrap -> verify ->
swap sequence, with a best-effort rollback (previous config + registry state +
previous session restarted) when the NEW runtime fails to come up healthy or to
report the exact expected roots. At most one replacement attempt is made per call;
there is no sleep/retry loop anywhere in this module.
"""

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional, Protocol, Sequence

from mcp_layer.config import load_config
from mcp_layer.config_resolver import MANAGED_CONFIG_FILENAME
from mcp_layer.errors import McpError
from mcp_layer.external import bootstrap_from_config
from tools.models import (
    MCP_EXPECTED_TOOL_MISSING,
    MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH,
    MCP_RUNTIME_REBIND_FAILED,
    MCP_RUNTIME_RESTART_FAILED,
    MCP_RUNTIME_ROLLBACK_FAILED,
    MCP_RUNTIME_VALIDATION_FAILED,
    MCP_SERVER_CONFIG_INVALID,
    MCP_SERVER_DISABLED,
    MCP_SERVER_NOT_INSTALLED,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ActiveMcpRuntime:
    """The single authoritative mutable reference to "the current MCP session".

    Every consumer (tool registration, shutdown, health checks, diagnostics) reads
    `.session` at the moment it needs it rather than caching it in a local variable,
    so a runtime replacement is visible everywhere immediately and uniformly — no
    component can be left holding a reference to an already-closed session.
    """

    def __init__(self, session=None):
        self._session = session

    @property
    def session(self):
        return self._session

    def replace(self, session) -> None:
        self._session = session

    def close(self) -> None:
        """Shut down whatever session is currently active (idempotent)."""
        session, self._session = self._session, None
        if session is not None:
            session.shutdown()


def _canonical_root(path) -> str:
    real = os.path.realpath(str(path))
    return real.lower() if os.name == "nt" else real


def _canonical_roots(roots) -> set:
    return {_canonical_root(r) for r in (roots or ())}


def _server_config_path(server_id, base_dir, managed_root) -> str:
    """Where EXACTLY `server_id`'s managed config lives — never "whichever
    managed server happens to be active" (Phase G.2: several may be at once)."""
    import tools.config as app_config

    base_dir = base_dir or _REPO_ROOT
    managed_root = managed_root or app_config.mcp_managed_root()
    return os.path.join(base_dir, str(managed_root), server_id, MANAGED_CONFIG_FILENAME)


def _unregister_session_tools(registry, session) -> tuple:
    if session is None:
        return ()
    return registry.unregister_owned(
        getattr(session, "registered_remote_tool_names", ()),
        getattr(session, "session_id", None),
    )


def _verify_live_roots(client, expected_roots, call_timeout):
    """Ask the LIVE new server for its allowed directories and compare canonically.

    Never trusts the config file, the registry, or plan state for this — only the
    real running process's own answer counts.
    """
    from mcp_management.validator import _extract_paths

    result = client.call_tool("list_allowed_directories", {}, timeout=call_timeout)
    reported = _canonical_roots(_extract_paths(result))
    expected = _canonical_roots(expected_roots)
    if reported != expected:
        raise McpError(
            MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH,
            "The restarted MCP server's live allowed directories do not exactly match "
            "the expected approved roots.",
        )


# ============================================================================
# Phase G.2 — server-keyed runtime state, pluggable per-server validation, and
# the multi-server manager. Everything above this point (ActiveMcpRuntime,
# McpRuntimeManager) is Phase F.1 and is REUSED, not replaced: a
# MultiMcpRuntimeManager holds one ActiveMcpRuntime + McpRuntimeManager per
# server_id internally rather than re-implementing bootstrap/verify/rollback.
# ============================================================================

class RuntimeState(str, Enum):
    NOT_INSTALLED = "not_installed"
    INACTIVE = "inactive"
    STARTING = "starting"
    HEALTHY = "healthy"
    RESTARTING = "restarting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ActiveMcpRuntimeSlot:
    """One server's runtime bookkeeping. Never shared across server_ids."""

    server_id: str
    session: object = None
    state: RuntimeState = RuntimeState.INACTIVE
    last_error_code: Optional[str] = None
    started_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None


@dataclass(frozen=True)
class McpRuntimeStatus:
    """A read-only snapshot for status/debug reporting — never a control handle."""

    server_id: str
    state: RuntimeState
    config_found: bool
    enabled: bool
    last_error_code: Optional[str] = None
    registered_tool_count: int = 0


@dataclass(frozen=True)
class RuntimeValidationContext:
    expected_allowed_roots: Optional[Sequence[str]] = None
    expected_tools: Sequence[str] = ()


class McpRuntimeValidator(Protocol):
    """A per-server (or per-capability) live-validation hook. Raises McpError on
    failure; returns None on success. Never starts/stops anything itself."""

    def validate(self, server_id: str, session, context: RuntimeValidationContext) -> None: ...


class FilesystemRootValidator:
    """The Filesystem MCP server's own live-verification rule (Task 9): its
    `list_allowed_directories` answer must exactly match the expected roots.
    Registered for server_id "filesystem" only — never applied to another server."""

    def validate(self, server_id, session, context):
        if context.expected_allowed_roots is None:
            return
        _verify_live_roots(session.client, context.expected_allowed_roots,
                           session.client.default_call_timeout)


class GenericRuntimeValidator:
    """The default validator for a server with no server-specific live check:
    confirms the session actually reports healthy and, when the caller supplied
    expected tool names (e.g. from the trusted catalog's `expected_tools`), that
    they were actually registered — never a Filesystem-only concept."""

    def validate(self, server_id, session, context):
        if session.client is None or session.health is None or session.health.state.value != "healthy":
            raise McpError(MCP_RUNTIME_VALIDATION_FAILED,
                           f"The {server_id!r} MCP server did not report healthy.")
        registered = tuple(getattr(session, "registered_remote_tool_names", ()))
        missing = [t for t in context.expected_tools
                  if not any(name.endswith("." + t) for name in registered)]
        if missing:
            raise McpError(MCP_EXPECTED_TOOL_MISSING,
                           f"The {server_id!r} MCP server did not register expected tool(s): "
                           f"{', '.join(sorted(missing))}.")


class McpRuntimeManager:
    """Coordinates replacing one active MCP session with a freshly bootstrapped one."""

    def __init__(self, registry, base_dir=None, managed_root=None, registry_path=None):
        self.registry = registry
        self.base_dir = base_dir
        self.managed_root = managed_root
        self.registry_path = registry_path

    # ---- internal: bootstrap a fresh session and prove it matches expected_roots ----

    def _bootstrap_and_verify(self, config, expected_allowed_roots, server_id=None, validator=None,
                              expected_tools=()):
        """Bootstrap once and validate before ever calling it healthy.

        Default (`validator=None`): EXACT prior behavior — live
        `list_allowed_directories` must match `expected_allowed_roots`. Phase G.2
        callers may instead pass a `McpRuntimeValidator` (e.g. a generic
        expected-tools check for a non-filesystem server) via `validator`; every
        pre-existing caller omits it and is therefore byte-for-byte unaffected.
        """
        session_id = uuid.uuid4().hex
        try:
            new_session = bootstrap_from_config(self.registry, config=config, base_dir=self.base_dir,
                                                session_id=session_id)
        except McpError as e:
            raise McpError(MCP_RUNTIME_RESTART_FAILED,
                           f"The MCP runtime failed to restart: {e.message}") from e

        healthy = (new_session.client is not None and new_session.health is not None
                  and new_session.health.state.value == "healthy")
        if not healthy:
            last_code = new_session.health.last_error_code if new_session.health else None
            _unregister_session_tools(self.registry, new_session)
            raise McpError(MCP_RUNTIME_REBIND_FAILED,
                           f"The restarted MCP server did not report healthy ({last_code}).")

        try:
            if validator is not None:
                context = RuntimeValidationContext(expected_allowed_roots=expected_allowed_roots,
                                                    expected_tools=tuple(expected_tools))
                validator.validate(server_id or config.server_id, new_session, context)
            else:
                _verify_live_roots(new_session.client, expected_allowed_roots, config.call_timeout_seconds)
        except McpError:
            try:
                new_session.shutdown()
            finally:
                _unregister_session_tools(self.registry, new_session)
            raise
        return new_session

    # ---- internal: best-effort rollback to the previous config/registry/session ----

    def _rollback(self, server_id, current_raw_config, previous_allowed_roots, validator=None):
        from mcp_management import lifecycle
        from mcp_management.configuration_generator import validate_generated
        from mcp_management.registry import get_installed, upsert

        args = current_raw_config.get("args") or []
        entrypoint = args[0] if args else None
        if not entrypoint:
            raise McpError(MCP_RUNTIME_ROLLBACK_FAILED,
                           "No entrypoint was available to rebuild the previous configuration.")

        rollback_raw = dict(current_raw_config)
        rollback_raw["args"] = [entrypoint] + [str(d) for d in previous_allowed_roots]
        try:
            rollback_config = validate_generated(rollback_raw)
        except McpError as e:
            raise McpError(MCP_RUNTIME_ROLLBACK_FAILED,
                           f"The rollback configuration is invalid: {e.message}") from e

        lifecycle.activate(rollback_raw, self.base_dir, self.managed_root, self.registry_path)
        installed = get_installed(server_id, self.registry_path, self.base_dir, self.managed_root)
        if installed is not None:
            restored = dc_replace(installed, approved_directories=tuple(previous_allowed_roots))
            upsert(server_id, restored, self.registry_path, self.base_dir, self.managed_root)

        try:
            return self._bootstrap_and_verify(rollback_config, previous_allowed_roots, server_id=server_id,
                                              validator=validator)
        except McpError as e:
            raise McpError(MCP_RUNTIME_ROLLBACK_FAILED,
                           f"The previous configuration could not be restarted either: "
                           f"{e.message}") from e

    # ---- the coordinator ----

    def replace_active_session(self, runtime: ActiveMcpRuntime, server_id, expected_allowed_roots,
                               previous_allowed_roots=None, validator=None, expected_tools=()):
        """Replace `runtime.session` with a fresh session for `server_id`.

        Reads the ALREADY-WRITTEN managed configuration for EXACTLY `server_id`
        (mcp_management writes it before this is ever called) — never "whichever
        managed server happens to be globally active" (Phase G.2: several servers
        may be installed and enabled at once, so that single-server notion no
        longer identifies one server_id). Confirms the config exists and is
        enabled, stops the old session and unregisters exactly its own remote
        tools, starts a fresh process, and verifies the live server reports
        `expected_allowed_roots` exactly (or passes `validator`, for Phase G.2's
        non-filesystem servers) before swapping `runtime` over. On any failure
        after the old session is stopped, attempts one rollback to
        `previous_allowed_roots` (when given) so the assistant is never left
        without a working session. Returns the new (or, on rollback, restored)
        McpSession; raises McpError on unrecoverable failure.
        """
        old_session = runtime.session

        config_path = _server_config_path(server_id, self.base_dir, self.managed_root)
        if not os.path.isfile(config_path):
            raise McpError(MCP_RUNTIME_RESTART_FAILED,
                           "The managed configuration is not active for this server.")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                current_raw_config = json.load(f)
        except (OSError, ValueError) as e:
            raise McpError(MCP_RUNTIME_RESTART_FAILED,
                           "The updated managed configuration could not be read.") from e
        try:
            config = load_config(config_path)
        except McpError as e:
            raise McpError(MCP_RUNTIME_RESTART_FAILED,
                           f"The updated configuration failed validation: {e.message}") from e
        if not config.enabled:
            raise McpError(MCP_RUNTIME_RESTART_FAILED,
                           "The managed configuration is not active for this server.")

        # Stop accepting new calls through the old session and close it. Its remote
        # tools are removed by exact ownership — built-ins (mcp.<server>.access.*)
        # and any tool a NEWER session already re-registered are never touched.
        if old_session is not None:
            try:
                old_session.shutdown()
            except Exception:  # noqa: BLE001 — never let a close failure block replacement
                pass
            _unregister_session_tools(self.registry, old_session)

        try:
            new_session = self._bootstrap_and_verify(config, expected_allowed_roots, server_id=server_id,
                                                      validator=validator, expected_tools=expected_tools)
        except McpError as restart_error:
            runtime.replace(None)
            if previous_allowed_roots is None:
                raise
            restored_session = self._rollback(server_id, current_raw_config, previous_allowed_roots,
                                              validator=validator)
            runtime.replace(restored_session)
            raise restart_error

        runtime.replace(new_session)
        return new_session


def _utc_now():
    return datetime.now(timezone.utc)


class MultiMcpRuntimeManager:
    """Phase G.2 — server-keyed runtime manager: more than one MCP session may be
    HEALTHY simultaneously, each started lazily, independently owned, and
    restartable without touching any other server_id's slot.

    Internally holds one `ActiveMcpRuntime` + shares one stateless
    `McpRuntimeManager` coordinator per server — reusing the exact Phase F.1
    bootstrap/verify/rollback mechanics (Task 1: no parallel runtime stack) and
    adding only: server-keyed slots, per-server locks (Task 11), lazy first
    activation (Task 4), and pluggable per-server validators (Task 9).
    """

    def __init__(self, registry, base_dir=None, managed_root=None, registry_path=None, validators=None):
        self.registry = registry
        self.base_dir = base_dir or _REPO_ROOT
        self.managed_root = managed_root
        self.registry_path = registry_path
        self._validators: Dict[str, McpRuntimeValidator] = dict(validators or {})
        self._coordinator = McpRuntimeManager(registry, base_dir=self.base_dir, managed_root=managed_root,
                                              registry_path=registry_path)
        self._slots: Dict[str, ActiveMcpRuntimeSlot] = {}
        self._runtimes: Dict[str, ActiveMcpRuntime] = {}
        self._locks_guard = threading.Lock()
        self._locks: Dict[str, threading.Lock] = {}

    # ---- internal bookkeeping ----

    def _lock_for(self, server_id) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(server_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[server_id] = lock
            return lock

    def _slot(self, server_id) -> ActiveMcpRuntimeSlot:
        slot = self._slots.get(server_id)
        if slot is None:
            slot = ActiveMcpRuntimeSlot(server_id=server_id)
            self._slots[server_id] = slot
            self._runtimes[server_id] = ActiveMcpRuntime(None)
        return slot

    def _validator_for(self, server_id, expected_allowed_roots) -> McpRuntimeValidator:
        configured = self._validators.get(server_id)
        if configured is not None:
            return configured
        if expected_allowed_roots is not None:
            return FilesystemRootValidator()
        return GenericRuntimeValidator()

    def _server_config_path(self, server_id) -> str:
        return _server_config_path(server_id, self.base_dir, self.managed_root)

    def _resolve_server_config(self, server_id):
        """Load + validate the config for EXACTLY this server_id — never "whichever
        managed server happens to be active" (that was the single-server Phase
        E/F.1 world's notion, and is unsuitable here since several
        servers may be installed at once). Raises MCP_SERVER_NOT_INSTALLED /
        MCP_SERVER_DISABLED / MCP_SERVER_CONFIG_INVALID."""
        path = self._server_config_path(server_id)
        if not os.path.isfile(path):
            raise McpError(MCP_SERVER_NOT_INSTALLED, f"Server {server_id!r} is not installed.")
        try:
            config = load_config(path)
        except McpError as e:
            raise McpError(MCP_SERVER_CONFIG_INVALID,
                           f"The configuration for {server_id!r} is invalid: {e.message}") from e
        if not config.enabled:
            raise McpError(MCP_SERVER_DISABLED, f"Server {server_id!r} is disabled.")
        return config

    # ---- read-only status ----

    def get_session(self, server_id):
        slot = self._slots.get(server_id)
        return slot.session if slot else None

    def get_status(self, server_id) -> McpRuntimeStatus:
        slot = self._slot(server_id)
        config_found = os.path.isfile(self._server_config_path(server_id))
        enabled = False
        if config_found:
            try:
                config = load_config(self._server_config_path(server_id))
                enabled = bool(config and config.enabled)
            except McpError:
                enabled = False
        state = slot.state
        if state == RuntimeState.INACTIVE and not config_found:
            state = RuntimeState.NOT_INSTALLED
        tool_count = len(getattr(slot.session, "registered_remote_tool_names", ())) if slot.session else 0
        return McpRuntimeStatus(server_id=server_id, state=state, config_found=config_found, enabled=enabled,
                                last_error_code=slot.last_error_code, registered_tool_count=tool_count)

    # ---- lazy activation ----

    def ensure_started(self, server_id, expected_allowed_roots=None, expected_tools=()) -> object:
        """Return a HEALTHY session for `server_id`, starting it lazily if needed.

        Idempotent and safe under concurrent calls for the SAME server_id (a
        per-server lock serializes them — Task 11): only one bootstrap happens,
        every caller receives the resulting healthy session (or the same raised
        error). A call for a DIFFERENT server_id never blocks on this one.
        """
        lock = self._lock_for(server_id)
        with lock:
            slot = self._slot(server_id)
            if slot.state == RuntimeState.HEALTHY and slot.session is not None:
                slot.last_used_at = _utc_now()
                return slot.session

            slot.state = RuntimeState.STARTING
            slot.last_error_code = None
            try:
                config = self._resolve_server_config(server_id)
                validator = self._validator_for(server_id, expected_allowed_roots)
                session = self._coordinator._bootstrap_and_verify(
                    config, expected_allowed_roots, server_id=server_id, validator=validator,
                    expected_tools=expected_tools)
            except McpError as e:
                slot.state = RuntimeState.FAILED
                slot.last_error_code = e.code
                slot.session = None
                raise

            slot.session = session
            self._runtimes[server_id].replace(session)
            slot.state = RuntimeState.HEALTHY
            now = _utc_now()
            slot.started_at = now
            slot.last_used_at = now
            return session

    # ---- restart (Task 8: generalizes the F.1 Filesystem-restart mechanism) ----

    def replace_session(self, server_id, expected_allowed_roots=None, previous_allowed_roots=None,
                        expected_tools=()) -> object:
        """Replace ONLY `server_id`'s session. Every other slot is untouched —
        no other server's process, session, or registered tools are affected."""
        lock = self._lock_for(server_id)
        with lock:
            slot = self._slot(server_id)
            slot.state = RuntimeState.RESTARTING
            slot.last_error_code = None
            runtime = self._runtimes[server_id]
            validator = self._validator_for(server_id, expected_allowed_roots)
            try:
                new_session = self._coordinator.replace_active_session(
                    runtime, server_id, expected_allowed_roots, previous_allowed_roots=previous_allowed_roots,
                    validator=validator, expected_tools=expected_tools)
            except McpError as e:
                slot.state = RuntimeState.FAILED
                slot.last_error_code = e.code
                slot.session = runtime.session  # None, or the rolled-back session
                raise
            slot.session = new_session
            slot.state = RuntimeState.HEALTHY
            now = _utc_now()
            slot.started_at = now
            slot.last_used_at = now
            return new_session

    # ---- shutdown ----

    def stop(self, server_id) -> None:
        """Idempotent: stopping an inactive or already-stopped server is a no-op."""
        lock = self._lock_for(server_id)
        with lock:
            slot = self._slots.get(server_id)
            if slot is None or slot.session is None:
                return
            slot.state = RuntimeState.STOPPING
            runtime = self._runtimes[server_id]
            session = runtime.session
            try:
                if session is not None:
                    session.shutdown()
            except Exception:  # noqa: BLE001 — never block cleanup on a close failure
                pass
            finally:
                _unregister_session_tools(self.registry, session)
                runtime.replace(None)
                slot.session = None
                slot.state = RuntimeState.STOPPED

    def stop_all(self) -> tuple:
        """Stop every active slot from a STABLE snapshot of server IDs (Task 12):
        one server's stop failure never prevents attempting the rest. Returns a
        tuple of (server_id, sanitized_message) for any that failed to stop
        cleanly — never a raw exception or stack trace."""
        errors = []
        for server_id in tuple(self._slots.keys()):
            try:
                self.stop(server_id)
            except Exception as e:  # noqa: BLE001 — aggregate, never abort the sweep
                errors.append((server_id, str(e)))
        return tuple(errors)
