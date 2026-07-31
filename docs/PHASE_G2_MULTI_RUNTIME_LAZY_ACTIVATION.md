# Phase G.2 — Multi-Server MCP Runtime Foundation and Lazy Activation

> Capability selection identifies the approved provider. Lazy activation starts
> that provider only when the current request needs it.

> Remote MCP tools must be registered before Phase B builds its shortlist.

> A runtime failure is scoped to one server ID. It must not remove tools, stop
> processes, or change health state for another server.

> Assistant startup loads metadata but launches no MCP child process.

## Why MCP servers no longer start at assistant startup

Every phase through F.1 assumed exactly one MCP session could ever be active,
and `assistant.main()` started it EAGERLY (`_start_mcp()`) before the first
request was even read. That assumption breaks the moment a second approved
server exists (Phase G.1 already detects `document_to_markdown` — it just has
no provider yet): eagerly starting every installed server on startup would mean
a general knowledge question pays the cost of launching child processes it will
never use, and two servers competing for one `ActiveMcpRuntime` slot is not
representable at all.

Phase G.2 replaces eager, single-slot startup with **server-keyed runtime
slots**, each activated **lazily** — only when Phase G.1 actually selects that
server for the current request.

## Server-keyed runtime models

`mcp_layer/runtime_manager.py` (Phase F.1's `ActiveMcpRuntime` /
`McpRuntimeManager` are unmodified in their default behavior — see below —
these are ADDED alongside them):

```python
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
    server_id: str
    session: object = None
    state: RuntimeState = RuntimeState.INACTIVE
    last_error_code: str | None = None
    started_at: datetime | None = None
    last_used_at: datetime | None = None

class MultiMcpRuntimeManager:
    def ensure_started(self, server_id, expected_allowed_roots=None, expected_tools=()): ...
    def get_session(self, server_id): ...
    def get_status(self, server_id) -> McpRuntimeStatus: ...
    def replace_session(self, server_id, expected_allowed_roots=None,
                        previous_allowed_roots=None, expected_tools=()): ...
    def stop(self, server_id) -> None: ...
    def stop_all(self) -> tuple: ...
```

A `dict[str, ActiveMcpRuntimeSlot]` backs the manager: at most one active
session per `server_id`, and any number of DIFFERENT `server_id`s may be
`HEALTHY` simultaneously — each with its own process, its own `McpSession`, and
its own tools.

**Reuse, not a parallel stack (Task 1).** Internally, `MultiMcpRuntimeManager`
holds one `ActiveMcpRuntime` per `server_id` and shares ONE stateless
`McpRuntimeManager` coordinator across all of them — the exact bootstrap /
verify / rollback mechanics Phase F.1 already had and already has full test
coverage for. `McpRuntimeManager.replace_active_session` and its internal
`_bootstrap_and_verify` gained two new, fully-optional parameters
(`validator`, `expected_tools`); every existing call site that omits them gets
byte-for-byte the old behavior — `tests/test_mcp_runtime_replacement.py` and
`tests/test_mcp_runtime_rollback.py` pass completely unmodified.

**A real bug this generalization caught and fixed:** `McpRuntimeManager.
replace_active_session` used to confirm the config it was about to swap in via
`mcp_layer.config_resolver.resolve_config()` — "whichever managed server
happens to be globally active." That question has no single answer once two
servers are installed at once (it silently picked whichever server_id sorted
first alphabetically and rejected the actual target as "not the active
server"). Fixed to resolve the config for the EXACT `server_id` being replaced,
by reading `<base_dir>/<managed_root>/<server_id>/server.json` directly — this
was caught by `tests/test_mcp_server_scoped_registration.py`'s two-server
restart test, not assumed safe by inspection.

## Lazy activation: exactly where `ensure_started` runs

```
Router: local or Claude
  -> Phase G.1 capability selection (select_for_request)
       NONE_REQUIRED  -> _run_local_turn unchanged, no MCP call
       SELECTED       -> ensure_selected_server_active(selection, runtime_manager, ...)
                            activated  -> _run_local_turn (Phase B now sees the tools)
                            not activated (MCP_SERVER_NOT_INSTALLED/...) -> normalized reply, no Phase B
       UNSUPPORTED / AMBIGUOUS / MULTI_SERVER_REQUIRED -> normalized reply, no MCP call at all
```

This all happens inside `assistant._process_local_request_with_capability_
selection` — the SAME single authoritative local-request entrypoint Phase G.1
established (Task 1's invariant carries forward unchanged: `_run_local_turn`,
and therefore Phase B, is still called from nowhere else in production code).
Activation always completes — successfully or not — strictly BEFORE Phase B's
shortlist is ever built.

### The bridge: `mcp_management/runtime_activation.py`

`ensure_selected_server_active(selection, runtime_manager, catalog, ...)` is
the ONLY place trust (catalog + installed-server registry — `mcp_management`'s
job) meets mechanics (`MultiMcpRuntimeManager.ensure_started` —
`mcp_layer`'s job). It:

1. Looks up `get_installed(server_id, ...)`. `None` → `MCP_SERVER_NOT_INSTALLED`
   immediately, with the exact message Task 6 specifies — no runtime call at
   all.
2. For `server_id == "filesystem"` only, reads `expected_allowed_roots` from
   the installed registry's own `approved_directories` (never guessed, never
   applied to any other server — Task 8/9's "Filesystem root verification may
   remain a server-specific validator").
3. Calls `runtime_manager.ensure_started(server_id, expected_allowed_roots=...,
   expected_tools=catalog_entry.expected_tools)`, translating any `McpError`
   into a normalized, non-activating result.

## Runtime states and validation (Task 3/9)

`ensure_started` never marks a slot `HEALTHY` until: `bootstrap_from_config`
returns a session whose `health.state == "healthy"`, AND the configured
validator's `.validate(server_id, session, context)` raises nothing.

- **`FilesystemRootValidator`** — registered for `server_id="filesystem"` only
  (wired once, at `assistant.main()` startup, via
  `MultiMcpRuntimeManager(..., validators={"filesystem": FilesystemRootValidator()})`).
  Calls the live `list_allowed_directories` and requires an EXACT match against
  `expected_allowed_roots`.
- **`GenericRuntimeValidator`** — the default for every other server: confirms
  the session is genuinely healthy and, when the caller supplied
  `expected_tools` (from the catalog entry), that they were actually
  registered (`MCP_EXPECTED_TOOL_MISSING` otherwise). No Filesystem-specific
  concept leaks into a generic server's lifecycle.

A failed bootstrap or failed validation always: shuts down the half-started
session, unregisters exactly the tools IT registered, and marks the slot
`FAILED` with the triggering error code recorded — never leaves a
partially-initialized session reachable as if it were healthy.

## Tool ownership stays server- and session-scoped (Task 7)

Unchanged from Phase F.1, now proven across TWO simultaneously-active
sessions: every `McpTool` carries `session_owner` (the `McpSession.session_id`
that discovered it), and `ToolRegistry.unregister_owned(names, owner)` removes
a name only when the CURRENTLY registered tool for that name still belongs to
`owner`. Stopping or restarting `document-test` can never touch
`mcp.filesystem.*` names, and vice versa — proven directly in
`tests/test_mcp_server_scoped_registration.py` and
`tests/test_mcp_runtime_failure_isolation.py`.

## Failure isolation (Task 10)

There is no global MCP health flag anywhere in this design — `get_status`
is always parametrized by `server_id`, `ensure_started`/`replace_session`
operate on exactly one slot under exactly one per-server lock, and a failure
path only ever touches that slot's own session/tools. A `document-test`
startup failure leaves an already-`HEALTHY` `filesystem-test` completely
untouched: same PID, same session object, same registered tools
(`tests/test_mcp_runtime_failure_isolation.py`).

## Concurrency (Task 11)

`MultiMcpRuntimeManager` keeps one `threading.Lock` per `server_id` (created
lazily, guarded by a small top-level lock). `ensure_started` acquires that
server's lock for its ENTIRE bootstrap-or-reuse decision: concurrent calls for
the SAME `server_id` serialize into exactly one bootstrap, and every caller
receives the identical resulting session (`tests/test_mcp_runtime_concurrency.
py`, real threads + real subprocesses). Two DIFFERENT `server_id`s never
contend on the same lock and start fully independently.

## Filesystem F.1 compatibility (Task 17)

`assistant._restart_mcp_and_resume` now calls
`runtime_manager.replace_session(directive.server_id, ...)` instead of
constructing a one-off `McpRuntimeManager` for a single global session. The
entire F.1 live scenario — outside-root read, automatic access plan, cross-turn
"yes", Filesystem-only restart, live-root re-verification, exactly-once
resumption — is unchanged in behavior and re-verified with a SECOND,
unrelated, already-`HEALTHY` server present throughout
(`tests/test_filesystem_f1_multi_runtime_regression.py`): its process, session,
and tools never move.

## Shutdown (Task 12)

`assistant.main()`'s `finally` block calls `runtime_manager.stop_all()`, which
iterates a STABLE snapshot of active `server_id`s (`tuple(self._slots.keys())`
taken once, before any stop runs) so one server's stop failure can never skip
the rest. Each `stop()` is idempotent — stopping an inactive or already-stopped
server is a no-op — so the same slot is never closed twice.

## Normal vs. debug output (Task 14)

Normal mode prints nothing at startup beyond the existing Phase F/F.1
provisioning-tool availability lines — no per-server health dump, no "MCP
config source," no "MCP process started." With `MCP_CAPABILITY_DEBUG=true`,
`assistant._log_runtime_activation` additionally prints, right after an
activation attempt:

```
MCP filesystem:
  state: healthy
  tools registered: 12
```

or, on a failed activation, the same plus an `error: <code>` line. Never a raw
path, environment value, or credential.

## Out of scope (unchanged from the task boundary)

MarkItDown installation, automatic MCP installation, public MCP discovery,
remote HTTP MCP, multi-server workflows in one request, idle-timeout shutdown,
uninstall/repair UI, and LLM-driven server selection are all explicitly not
part of this phase — see the task's own OUT OF SCOPE list.

## Tests

```
pytest -q tests/test_multi_mcp_runtime_manager.py
pytest -q tests/test_mcp_runtime_failure_isolation.py
pytest -q tests/test_mcp_runtime_concurrency.py
pytest -q tests/test_mcp_server_scoped_registration.py
pytest -q tests/test_filesystem_f1_multi_runtime_regression.py
pytest -q tests/test_assistant_lazy_mcp_activation.py
pytest -q tests/test_assistant_capability_selection.py   # Phase G.1's suite, now lazily-activating
pytest -q   # full suite
```

For a fully scripted, real-process equivalent of a manual CLI session (old
PID/new PID, live activation, reuse, clean stop, no orphan) against the
ACTUAL, already-installed `@modelcontextprotocol/server-filesystem` package,
see `scripts/manual_verify_g2_lazy_runtime.py` — isolated under a temp
directory, never touches `app_data/mcp_servers/`.
