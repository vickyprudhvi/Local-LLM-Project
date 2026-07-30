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
import uuid
from dataclasses import replace as dc_replace

from mcp_layer.config import load_config
from mcp_layer.config_resolver import McpConfigSource, resolve_config
from mcp_layer.errors import McpError
from mcp_layer.external import bootstrap_from_config
from tools.models import (
    MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH,
    MCP_RUNTIME_REBIND_FAILED,
    MCP_RUNTIME_RESTART_FAILED,
    MCP_RUNTIME_ROLLBACK_FAILED,
)


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


class McpRuntimeManager:
    """Coordinates replacing one active MCP session with a freshly bootstrapped one."""

    def __init__(self, registry, base_dir=None, managed_root=None, registry_path=None):
        self.registry = registry
        self.base_dir = base_dir
        self.managed_root = managed_root
        self.registry_path = registry_path

    # ---- internal: bootstrap a fresh session and prove it matches expected_roots ----

    def _bootstrap_and_verify(self, config, expected_allowed_roots):
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
            _verify_live_roots(new_session.client, expected_allowed_roots, config.call_timeout_seconds)
        except McpError:
            try:
                new_session.shutdown()
            finally:
                _unregister_session_tools(self.registry, new_session)
            raise
        return new_session

    # ---- internal: best-effort rollback to the previous config/registry/session ----

    def _rollback(self, server_id, current_raw_config, previous_allowed_roots):
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
            return self._bootstrap_and_verify(rollback_config, previous_allowed_roots)
        except McpError as e:
            raise McpError(MCP_RUNTIME_ROLLBACK_FAILED,
                           f"The previous configuration could not be restarted either: "
                           f"{e.message}") from e

    # ---- the coordinator ----

    def replace_active_session(self, runtime: ActiveMcpRuntime, server_id, expected_allowed_roots,
                               previous_allowed_roots=None):
        """Replace `runtime.session` with a fresh session for `server_id`.

        Reads the ALREADY-WRITTEN managed configuration (mcp_management writes it
        before this is ever called), confirms it is the active managed source for
        this exact server, stops the old session and unregisters exactly its own
        remote tools, starts a fresh process, and verifies the live server reports
        `expected_allowed_roots` exactly before swapping `runtime` over. On any
        failure after the old session is stopped, attempts one rollback to
        `previous_allowed_roots` (when given) so the assistant is never left
        without a working session. Returns the new (or, on rollback, restored)
        McpSession; raises McpError on unrecoverable failure.
        """
        old_session = runtime.session

        resolved = resolve_config(base_dir=self.base_dir, managed_root=self.managed_root)
        if resolved.source != McpConfigSource.MANAGED_ACTIVE or resolved.server_id != server_id:
            raise McpError(MCP_RUNTIME_RESTART_FAILED,
                           "The managed configuration is not active for this server.")
        config_path = str(resolved.path)
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
            new_session = self._bootstrap_and_verify(config, expected_allowed_roots)
        except McpError as restart_error:
            runtime.replace(None)
            if previous_allowed_roots is None:
                raise
            restored_session = self._rollback(server_id, current_raw_config, previous_allowed_roots)
            runtime.replace(restored_session)
            raise restart_error

        runtime.replace(new_session)
        return new_session
