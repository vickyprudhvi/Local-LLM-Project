# Phase F.1 — Filesystem MCP Access Expansion, Pending Approval, and Resumption

> **Adding filesystem access is a configuration change, not MCP package
> provisioning.** The existing managed Filesystem MCP installation is reused and
> never reinstalled merely to add another approved root.
>
> **The router decides only Local versus Claude (plus its own small fixed set of
> built-ins).** Tool and MCP-management selection remain inside the normal local
> tool-selection pipeline — the router can never dispatch an MCP or provisioning
> tool, including a hallucinated one.
>
> **A bare confirmation such as "yes" is accepted only when it matches a pending,
> unexpired, hashed filesystem-access plan.**

## The problem this phase fixes

Phase F installs the Filesystem MCP server with exactly one approved root, taken
from whatever directory the FIRST request that triggered installation happened to
name. Every later request for a file outside that one root was rejected by the
live server with no path forward: the assistant could only ask "should I
reconfigure?", and a plain "yes" reply had nowhere to attach itself — it went back
through the router as a brand-new, unrelated request, the local model had no tool
that could act on it, and in the observed failure the model invented a
non-existent `mcp.provision` tool name that the router's own contract never
offered it in the first place.

## Why Filesystem MCP roots are restricted

`@modelcontextprotocol/server-filesystem` enforces its allowed roots from its own
argv — every directory after the entrypoint script. `mcp_management`'s Phase F
install path only ever passes the ONE directory approved at install time (see
`mcp_management/configuration_generator.py:generate_config_dict`). This is
intentional defense in depth: even if the local LLM is tricked into requesting an
arbitrary path, the live server process itself refuses anything outside its argv,
independent of any local policy bug.

## How an outside-root request is detected

Detection is **structural**, not textual. `mcp_management/access_classifier.py`'s
`classify_outside_root_failure(tool_name, arguments, result, allowed_roots)`:

1. Only looks at MCP tool calls (`tool_name` starts with `"mcp."`) that failed.
2. Ignores failure codes that are never an access problem — timeouts, transport
   errors, malformed/invalid arguments, confirmation states. A generic fault never
   spawns an access plan.
3. Extracts the path argument(s) from the call itself (`path`, `file_path`,
   `paths`, `source`, `destination` — whichever the specific MCP tool uses).
4. Canonicalizes each path and checks whether it falls **structurally inside**
   any of the server's currently registered approved roots.
5. If every path IS inside an approved root, the failure is something else
   entirely (not found, OS permission, a crash) and is never classified as an
   access problem — this is what keeps "file not found inside an approved
   directory" from spawning a bogus plan.
6. Only when a path resolves outside every approved root does the classifier
   propose a plan — server error TEXT is never consulted to make this decision,
   so it is robust to whatever wording a given filesystem server happens to use.

`assistant.py`'s `_find_outside_root_failure` runs this classifier against every
individual tool call made during a "local" turn (see the observation hook below),
using the target server's `approved_directories` straight from the Phase F
registry (`mcp_management/registry.py`) — never a guess.

## Narrowest-root selection

`mcp_management/access_classifier.py:propose_root` computes the directory that
would be proposed, and reuses `mcp_management/planner.py:validate_approved_directory`
— the SAME forbidden/broad-location screen Phase F already applies at install
time — so a proposed root is never:

- the requested file itself (a single file proposes its **parent** directory)
- a system directory, drive root, or the filesystem root
- a credential store (`.ssh`, `.aws`, `.gnupg`, `.azure`, `.kube`, `.docker`,
  `credentials`, `secrets`, `.password-store`, browser profile directories)
- the repository root, the user's home directory, or an entire broad folder
  (`Documents`, `Desktop`, `Downloads`, `OneDrive`) without an explicit
  broad-scope opt-in the interactive flow never grants

For a directory-listing style call (`list_directory`, `directory_tree`,
`search_files`, `create_directory`) the requested directory itself is proposed,
not its parent. For several files under one parent, the nearest common parent is
proposed (`os.path.commonpath`). Files with no common parent (different drives, or
otherwise unrelated) are reported as ineligible — the caller must approve each
separately rather than the proposal silently guessing something broad.

## Pending approval and plan models

`mcp_management/filesystem_access.py` mirrors Phase F's own provisioning models:

- `FilesystemAccessPlan` — immutable, `plan_hash`-bound over server id, catalog
  id, operation, requested directory, and BOTH the current and proposed full root
  sets. Changing any of those invalidates a prior approval. `expires_at` bounds
  how long an unapproved plan stays valid (15 minutes by default).
- `PendingFilesystemAccessRequest` — carries the ORIGINAL blocked user request
  text across the approval turn, with a `PendingFilesystemAccessState` state
  machine (`detected -> plan_prepared -> awaiting_approval -> applying ->
  ready -> resumed`, or `declined`/`failed`/`expired`).
- `FilesystemAccessApproval` — a distinct type from Phase F's `ProvisioningApproval`
  and from Phase C's `ToolConfirmation`, so neither can ever be replayed to
  authorize a filesystem-access change (`mcp_management/approval.py:
  require_filesystem_access_approval`).

Both live on `McpProvisioningManager` (`mcp_management/manager.py`) as small,
in-memory dictionaries (`_filesystem_pending`, `_filesystem_plans`) — the exact
same pattern Phase F already uses for `_pending`/`_plans`, scoped to the one
manager instance created once per process.

## How a bare "yes" is resolved safely

`assistant.py`'s `main()` loop checks for an active pending filesystem-access
request **before** calling the router at all:

```
pending filesystem-access request?
    -> "yes"/"approve"/"ok"  : apply the plan, then RESUME the original request
    -> "no"/"decline"/"cancel": decline, leave roots unchanged, stop
    -> "show plan"/"what folder": re-show the plan, stay pending
    -> anything else          : fall through to normal routing; plan stays pending
```

`_resolve_filesystem_access_reply` only recognizes an exact match against small,
fixed word sets (`_FS_YES_WORDS`, `_FS_NO_WORDS`, `_FS_SHOW_PLAN_WORDS`) after
stripping trailing punctuation — a sentence that merely *contains* the word "yes"
does not match and is treated as an unrelated new request, per the acceptance
criteria. Because the router is never consulted for a matching reply, it never has
the opportunity to invent a tool name for this step, and `mcp.provision` is never
needed here at all — approval is resolved entirely by
`McpProvisioningManager.apply_filesystem_access`, a plain Python method call.

## Router hardening (Task 1)

`router.py` already used a fixed Ollama function-calling contract (`TOOLS`), which
never included any MCP/provisioning tool — the observed `mcp.provision` name was a
model **hallucination**: a `tool_calls` entry naming a function that was never
offered. The existing fallback already resolved this safely to `mode="local"`, but
the check was implicit (an unknown name fell through `_TOOL_NAME_MAP.get()`).
Phase F.1 makes the boundary explicit and structural:

```python
_OFFERED_FUNCTION_NAMES = frozenset(entry["function"]["name"] for entry in TOOLS ...)
...
if name not in _OFFERED_FUNCTION_NAMES:
    # never trust an out-of-contract name — same as no tool call at all
    return RouteDecision(mode="local", ...)
```

This closes the boundary independent of whatever `_TOOL_NAME_MAP` happens to
contain, so a future MCP-shaped name can never be routed just because it
collides with something else. `router.py`'s three-way contract (local / claude /
one of its own small fixed built-ins) is otherwise unchanged — existing
Local-vs-Claude and built-in routing tests all still pass unmodified. See
`tests/test_router_tool_boundary.py`.

## Registering and shortlisting the access-management tools

`mcp_management/filesystem_access_tools.py` adds four ordinary `BaseTool`s,
registered exactly like Phase F's provisioning tools:

| Tool | Permission | Purpose |
| --- | --- | --- |
| `mcp.filesystem.access.list` | READ | List an installed server's currently approved directories. |
| `mcp.filesystem.access.plan` | READ | Prepare an add/remove plan for an installed server; changes nothing. |
| `mcp.filesystem.access.add` | WRITE (Phase C confirmation) | Apply a prepared plan; never reinstalls the package. |
| `mcp.filesystem.access.remove` | WRITE (Phase C confirmation) | Revoke a previously approved directory; refuses to remove the last one. |

**No changes to `tools/registry.py` or `tool_loop.py`'s shortlist call were
needed.** The existing lexical `ToolRegistry.shortlist_tools()` already ranks a
tool by token overlap between the user's words and the tool's name/description —
so tool descriptions were written using the vocabulary a user would actually say
("give access to this folder", "allow this path", "expand allowed roots",
"remove folder access", etc.), and the existing mechanism surfaces them for a
matching request with no core-file changes at all (`tests/
test_mcp_filesystem_access_shortlisting.py`).

The one true pending-approval bypass (a bare "yes") is handled entirely in
`assistant.py`, **before** the router or shortlist ever run — not by changing
what gets shortlisted.

## Multiple roots for one Filesystem server

The Phase F registry's `InstalledServer.approved_directories` already supports an
arbitrary tuple of directories; Phase F.1 is the first thing that grows it past
one. `FilesystemAccessPlan.proposed_allowed_directories` is always the FULL
resulting set (existing roots plus the new one, deduplicated via `set`, ordered
via `sorted()` for determinism), and `filesystem_access_update.py` writes that
whole list into the generated config's `args`. Removing a root is a separate,
explicitly approved plan (`FilesystemAccessOperation.REMOVE_ROOT`) that refuses to
drop the last remaining root (`MCP_FILESYSTEM_LAST_ROOT_REQUIRED`). A restart of
the assistant re-resolves the SAME managed `server.json`, so all approved roots
persist across restarts with no extra bookkeeping.

## No npm reinstall; the managed config update

`mcp_management/filesystem_access_update.py:update_filesystem_access` never
imports or calls anything in `npm_installer.py`. Order:

```
read current managed server.json
    -> build a candidate document: SAME command/entrypoint, args = [entrypoint] + full proposed root list
    -> Phase E-validate the candidate (mcp_layer.config.build_config)
    -> start a REAL server process with the candidate config
    -> tools/list, then list_allowed_directories -> must match the proposed set exactly
    -> shut the verification process down (always, even on failure)
    -> lifecycle.activate(candidate) -> re-validates + atomic write (temp file + os.replace)
    -> registry.upsert(..., approved_directories=<the new full set>)
```

Because nothing is written to `server.json` until AFTER the live verification
above succeeds, there is no partial state to roll back on failure — the previous,
still-valid configuration was simply never touched. `config/mcp_server.json` (the
committed, portable template) is never read or written by this path, exactly like
every other Phase F write.

`tests/test_mcp_filesystem_access_update.py::test_npm_installer_is_never_called_
during_an_access_root_change` monkeypatches `npm_installer.install_package` to
raise if it is ever invoked, and asserts the update still succeeds.

## Restart, validation, and resumption

After `apply_filesystem_access` succeeds, `assistant.py`'s `main()` shuts down the
live `McpSession` and calls `_start_mcp()` again — the same restart primitive
`_provision_if_needed` already used for first-time installs — so the newly
written `server.json` (now covering all approved roots) is what the fresh
bootstrap picks up. It then re-injects the **original** blocked request text
(`manager.resume_filesystem_access`) and lets it fall through to the normal
`route_and_answer -> dispatch -> tool_loop.run_local_tool_loop -> ToolExecutor ->
McpTool` pipeline exactly once. `mcp_management` never reads the file itself and
never calls the newly-accessible tool directly — resumption is always "hand the
original text back to the normal pipeline," the identical contract
`McpProvisioningManager.resume()` already uses for first-time provisioning.

## Loop prevention

Each ORIGINAL blocked request gets at most one expansion attempt:
`McpProvisioningManager.apply_filesystem_access` raises
`MCP_FILESYSTEM_ACCESS_LOOP_PREVENTED` if the SAME pending request's plan is
applied a second time (mirroring Phase F's `MAX_PROVISIONING_ATTEMPTS`), and
`assistant.py`'s `main()` additionally tracks which original request texts have
already been offered a plan once — if the resumed call is STILL outside the
(newly expanded) roots, the assistant reports this plainly rather than looping.

## Restricted paths

A path whose narrowest proposed root fails `validate_approved_directory`'s
forbidden/broad screen is reported as `restricted=True`, `eligible=False` by the
classifier, and `_find_outside_root_failure` never offers a plan for it — no
approval prompt, no config change, no file content. Requesting
`~/.ssh/id_rsa` is exactly this case (`tests/
test_mcp_filesystem_access_resumption.py::test_restricted_path_yields_no_offer`).

## Known limitations

- The pending-approval store is in-memory, scoped to one `McpProvisioningManager`
  instance for the life of the process — it does not survive a full assistant
  restart mid-approval (an in-flight plan is simply lost, which is safe: nothing
  was written).
- `mcp.filesystem.access.remove`'s plan and apply steps are combined into a single
  WRITE-permission tool call (Phase C's confirmation gate is still the real
  approval boundary) rather than the two-step plan/approve flow `add` uses; this
  keeps removal simple since it is not part of the primary manual scenario.
- Narrow-root proposal for multiple files with no common parent reports the
  failure as ineligible rather than automatically producing several plans; the
  caller (or a future turn) must approve each root separately.
- Detection only runs for the "local" routing mode, where the local model's own
  tool-calling loop is what can reach an MCP filesystem tool; the router's own
  small fixed built-in set and the Claude escalation path never touch MCP tools,
  so no equivalent hook exists (or is needed) there.

## Tests

```
pytest -q tests/test_router_tool_boundary.py
pytest -q tests/test_mcp_filesystem_access_plan.py
pytest -q tests/test_mcp_filesystem_access_approval.py
pytest -q tests/test_mcp_filesystem_access_update.py
pytest -q tests/test_mcp_filesystem_access_resumption.py
pytest -q tests/test_mcp_filesystem_access_security.py
pytest -q tests/test_mcp_filesystem_access_shortlisting.py
pytest -q   # full suite
```

==================================================
## Runtime hotfix: same-process MCP session replacement (Phase F.1.1)
==================================================

> A successful filesystem access update is a control-plane event. The current
> tool loop stops immediately, the old MCP session is replaced, remote tools are
> rebound to the new client, and only then is the original request resumed.

> Persisting the new root in `server.json` is not sufficient because existing
> `McpTool` instances remain bound to the MCP client that discovered them.

> No sleep-based delay, manual assistant restart, or npm reinstall is required
> after adding an approved root.

### Why persisted config alone is insufficient

Everything documented above this section — `update_filesystem_access`, the plan
model, the approval flow — updates `server.json` on disk and proves (via a
throwaway verification client) that a FRESH process would honour the new roots.
None of that touches the assistant process's LIVE state:

- the existing `McpSession.client` still talks to the OLD child process, started
  with the OLD argv;
- every `McpTool` discovered from that session is a plain Python object whose
  `execute()` calls `self._client.call_tool(...)` — `self._client` is the SAME
  old client, forever, unless something replaces it;
- the old process never re-reads `server.json`; it only ever knew the roots it
  was launched with.

The original restart primitive (`mcp_session.shutdown(); mcp_session =
_start_mcp()`) looked sufficient but had two separate bugs:

1. **Silent registration collision.** `mcp_layer.external.bootstrap_from_config`
   registers a new `McpTool` only when `registry.has(tool.name)` is false. Since
   the OLD tools (e.g. `mcp.filesystem.read_text_file`) were never unregistered,
   every rediscovered tool from the new session was silently skipped
   (`registration_collision`) — the stale tool, bound to the now-terminated
   client, stayed registered and kept being called.
2. **No live verification.** Nothing ever asked the new process's own
   `list_allowed_directories` to confirm it actually reports the roots the config
   file claims — a config/runtime mismatch (or a race) would go unnoticed.

This is why the bug report showed `mcp.filesystem.access.add -> ok` immediately
followed by `mcp.filesystem.read_text_file -> failed` twice: the config was
correct, but the live call kept reaching the dead client.

### Tool-loop control: how the loop halts

`tool_loop.py` defines a small typed contract (`ToolLoopControl`,
`ToolLoopDirective`) that `on_tool_result` may optionally return. Returning `None`
— the only thing every pre-existing caller ever did (`list.append` returns
`None`) — is byte-for-byte identical to before: the loop simply continues.
Returning a directive whose `control` is not `CONTINUE` stops
`run_local_tool_loop` IMMEDIATELY: no further tool call in the same batch
executes, and the local LLM is never asked again (`_final_after_exhaustion` is
never reached). `text` in that case is `None` — the caller already built and
holds the directive, so it drives whatever happens next itself; the exact same
approach the module already used for observation (an exception from the callback
is swallowed, never a source of control flow).

### Automatic first-request handling (Task 2)

Previously, `_find_outside_root_failure` only ran AFTER the whole local turn
finished, scanning everything the model did. By the time it ran, the model had
already been given the failed tool result and asked to answer — which is exactly
how it could write a generic "copy the file" answer, or even decide on its own to
call `mcp.filesystem.access.plan` / `.access.add` itself using the STALE session.

`assistant._run_local_turn` now classifies each tool result THE INSTANT it comes
back, via the same `on_tool_result` hook, and returns a
`HALT_FOR_FILESYSTEM_ACCESS` directive right away. The loop stops before the
model's next turn, so a plain `read 'C:\...\f1_external_test\hello.txt'` produces
the access plan directly — no "use the filesystem MCP server..." rephrase is ever
needed.

### Session replacement (Tasks 3, 5, 8)

`mcp_layer/runtime_manager.py` adds:

- **`ActiveMcpRuntime`** — the one authoritative mutable holder for "the current
  session". Every consumer (registration, shutdown, diagnostics) reads
  `runtime.session` at the moment it needs it, so nothing can hold a stale local
  variable across a replacement. `assistant.main()`'s `finally` block calls
  `runtime.close()`, which always targets whatever session is CURRENT.
- **`McpRuntimeManager.replace_active_session(runtime, server_id,
  expected_allowed_roots, previous_allowed_roots=None)`** — the single
  deterministic coordinator: stop the old session, unregister exactly its own
  remote tools, re-resolve and re-validate the managed configuration (must
  already be `MANAGED_ACTIVE` for this exact server), bootstrap a fresh process,
  verify its LIVE roots, and only then atomically swap `runtime` over.

`assistant._classify_access_apply_success` recognizes a successful
`mcp.<server>.access.add`/`.remove` call structurally (a built-in registry name
always has 4 dot-separated segments; a remote tool always has 3) and returns a
`RESTART_MCP_AND_RESUME` directive carrying the trusted `server_id` and
`expected_allowed_roots` straight from the tool's own result data — never from
generated text. Both entry points — a mid-turn `access.add` call from the model
itself, and the cross-turn "yes" approval (`assistant._resolve_filesystem_access_reply`)
— build the same directive shape and feed the same `assistant._restart_mcp_and_resume`
function (Task 10), so the outcome never depends on how the approval was
collected.

### Exact remote-tool ownership (Tasks 4, 11)

`McpSession` now carries a `session_id` (opaque, auto-generated) and
`registered_remote_tool_names` — exactly the remote tools THIS session
registered, which is structurally disjoint from the built-in
`mcp.filesystem.access.*` tools (those are registered once, at startup, by a
completely separate path — `register_filesystem_access_tools` — and never touched
by a session replacement). Every `McpTool` a session builds is stamped with
`session_owner = session_id`. `ToolRegistry.unregister_owned(names, owner)`
removes a name only when the CURRENTLY registered tool's `session_owner` still
matches `owner` — a name a newer session already re-registered, or a built-in
with no session owner, is left untouched even if it happens to appear in a stale
name list.

### Live root verification (Task 6)

After the new process starts, `McpRuntimeManager` calls the new client's
`list_allowed_directories` directly — never trusting the config file, the
registry, or plan state — and canonically compares the result against
`expected_allowed_roots` (case-insensitive on Windows, case-sensitive on POSIX,
both `os.path.realpath`-resolved). Any mismatch — missing root, unexpected extra
root, or an unexpectedly broader parent root — raises
`MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH` and the original request is never resumed.

### Rollback (Task 7)

If the new runtime fails at any point — bootstrap, health, or root verification —
and the caller supplied `previous_allowed_roots` (available whenever a
`FilesystemAccessPlan` was involved, since the plan already records
`current_allowed_directories`), `McpRuntimeManager` rebuilds a candidate
configuration with the SAME command/entrypoint but the previous root list,
re-activates it (`lifecycle.activate`, the same atomic-write primitive every
other Phase F write path uses), restores the registry's `approved_directories`,
and starts THAT known-good configuration instead — registering tools bound to the
restored client and replacing `runtime` with it. The ORIGINAL failure code
(`MCP_RUNTIME_RESTART_FAILED` / `MCP_RUNTIME_REBIND_FAILED` /
`MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH`) is still raised so the caller knows a
restart failed even though recovery succeeded; if the rollback attempt ALSO
fails, `MCP_RUNTIME_ROLLBACK_FAILED` is raised instead so the assistant never
reports success when no working session remains. When no
`previous_allowed_roots` is available, the runtime reference is simply cleared
(`runtime.session is None`) rather than left pointing at anything stale.

### Resumption, exactly once (Task 9)

`assistant._restart_mcp_and_resume` takes a `resume_budget` (1 by default) that
is checked BEFORE touching the runtime at all — not after. A resumed request that
itself triggers another `RESTART_MCP_AND_RESUME` recurses with
`resume_budget - 1`, so at most one runtime replacement ever happens per original
request; a second attempt is refused outright (`MCP_RESUME_ABORTED`) without
touching the MCP process again. The resumption itself re-enters the REAL pipeline
— `route_and_answer` -> `_run_local_turn` -> `tool_loop.run_local_tool_loop` ->
`ToolExecutor` -> the freshly-registered `McpTool` — never a direct file read from
`mcp_management` or the runtime manager.

### Why sleeps and cache workarounds are prohibited

Every step above is a direct, synchronous consequence of the previous one: the
old process is confirmed stopped before the new one starts (`McpClient.shutdown()`
blocks up to its configured timeout, escalating terminate -> kill), the new
process's OWN `list_allowed_directories` answer is what gates resumption, and the
registry mutation (`unregister_owned` then `register`) happens on the same thread,
synchronously, with the interpreter's GIL as the only concurrency primitive
involved. There is no state that a delay could ever resolve that determinism
doesn't already resolve immediately — a `time.sleep` would only mask a bug (or
add latency to the common, already-working case) rather than fix one.

### Interactive CLI verification

```
venv/Scripts/python.exe assistant.py
```

then, with the managed config starting at a single approved root:

```
read 'C:\Users\<you>\...\f1_external_test\hello.txt'
```

Expect the access plan to appear automatically (no rephrase), and after replying
`yes`: `access approved -> runtime restart requested -> old MCP session stopped ->
new MCP session HEALTHY -> allowed roots verified -> original request resumed ->
mcp.filesystem.read_text_file -> ok`, with the real file content in the final
answer — no second approval, no manual restart, no npm call.

For a fully scripted (non-interactive) equivalent that exercises the SAME code
path against the real, already-installed `@modelcontextprotocol/server-filesystem`
package end to end (old PID vs. new PID, live root verification, real file
content), see `scripts/manual_verify_f1_restart.py` — it isolates all MCP state
under a temp directory, so it never touches `app_data/mcp_servers/` or requires a
live Ollama endpoint.

### Tests

```
pytest -q tests/test_mcp_stale_tool_prevention.py
pytest -q tests/test_mcp_tool_registration_ownership.py
pytest -q tests/test_mcp_runtime_replacement.py
pytest -q tests/test_mcp_runtime_rollback.py
pytest -q tests/test_mcp_access_auto_handoff.py
pytest -q tests/test_mcp_access_restart_resumption.py
pytest -q tests/test_assistant_mcp_restart_orchestration.py
```
