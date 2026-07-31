# Phase G.3 — Generalized Trusted MCP Provisioning: Approval, Activation, and Request Resumption

> The assistant may automatically identify that an approved MCP provider is needed.
> Installation still uses only deterministic metadata from the trusted catalog.

> The model never constructs an npm, pip, uv, shell, Git, or Docker installation command.

> A server is not marked installed until its candidate process initializes
> successfully and exposes the exact expected tools.

> Provisioning completes before Phase B receives the server's tools.

> Public MCP directories are discovery sources for developer review, not
> runtime trust sources.

## Approved vs. installed

Phase G.1's trusted catalog already distinguishes "approved" (a `McpCatalogEntry`
exists) from "selectable" (`granular_capabilities`/`selection_hints` match the
request). Phase G.2 added a third state: "installed but not yet running"
(lazy activation). Phase G.3 closes the last gap — **approved but never
installed at all**. Until this phase, `ensure_selected_server_active` could only
report `MCP_SERVER_NOT_INSTALLED` and stop; there was no path from that error
back to a working tool call without a human manually running the Phase F
`mcp.provision.*` chat tools (Filesystem's own directory-input flow) or editing
files by hand. G.3 adds the missing, generalized, non-Filesystem-specific link:
detect → plan → approve → install → validate → activate → resume, for ANY
catalog entry whose installer type is registered and which needs no directory
grant.

## Architecture

```
mcp_management/
  provisioning_models.py     Task 3 — AutoProvisioningPlan, ProvisioningPlanStatus,
                              PendingAutoProvisioningRequest, AutoProvisioningApproval,
                              ProvisioningResult. A DISTINCT set of types from Phase
                              F's own mcp_management.models (never a second
                              `McpProvisioningPlan` class — different name, so a
                              plan/approval from one flow can never be mistaken for
                              the other's).
  auto_provisioning.py       Tasks 4/9/10/16/17 — AutoProvisioningManager (the
                              pending-request/approval facade), build_auto_plan,
                              _run_transaction (the candidate-transaction
                              orchestrator), _validate_candidate_process,
                              require_auto_provisioning_approval.
  installers/
    base.py                  Task 5 — McpInstaller Protocol, ProvisioningTransaction,
                              CandidateInstallation, McpLaunchSpec, the
                              type -> installer registry.
    npm_backend.py            Task 6 — generalizes Phase F's npm_installer.py.
    python_venv.py            Task 7 — new: isolated venv + hash-verified pip install.
  catalog.py                  Task 2 — extended (not replaced): installer_type
                              "python_venv", lock_file/python_constraint/launch.
  configuration_generator.py  Task 11 — generate_config_dict_from_launch_spec(),
                              installer-agnostic (added alongside the existing
                              npm-shaped generate_config_dict, which Filesystem
                              still uses unmodified).
  registry.py                 Task 12 — InstalledServer gained optional fields
                              (installer_type, catalog_entry_hash, lock_hash,
                              expected_tools_hash, tool_policy_hash,
                              last_known_good_version); every field is Optional
                              with a default, so an old registry entry (including
                              the real installed Filesystem server) keeps loading.
assistant.py                  Task 13/14/15 — _start_auto_provisioning,
                              _offer_mcp_provisioning, _resolve_auto_provisioning_reply,
                              wired into _process_local_request_with_capability_
                              selection and main()'s cross-turn reply dispatch.
```

Nothing above **duplicates** Phase F: `_run_transaction` reuses
`mcp_layer.external.start_server` (the same real-process launch/initialize path
Phase E/F/F.1 already use) for candidate validation, `configuration_generator.
validate_generated`/`write_config` for Phase-E-checked config writes,
`mcp_management.registry.upsert`/`atomic_write_json` for atomic state, and the
npm backend calls Phase F's own `npm_installer.py` functions directly — a
`McpCatalogEntry` already carries every attribute those functions read from a
"plan" (`package_name`, `package_version`, `server_id`, `entrypoint_relative`,
`required_runtimes`), so it is passed in unchanged, no adapter object needed.

## Catalog schema (Task 2)

```json
{
  "catalog_id": "calculator-test",
  "server_id": "calculator-test",
  "installer": {
    "type": "python_venv",
    "package": "calculator-test-mcp",
    "version": "1.0.0",
    "lock_file": "config/mcp_locks/calculator-test-mcp-1.0.0.txt",
    "python_constraint": ">=3.9"
  },
  "launch": {
    "transport": "stdio",
    "entrypoint_type": "python_module",
    "module": "calculator_test_mcp.server",
    "arguments": []
  },
  "expected_tools": ["add", "echo"],
  "default_tool_policy": { "default_permission": "denied", "tools": {
    "add": {"enabled": true, "permission": "read"},
    "echo": {"enabled": true, "permission": "read"}
  }},
  "network_policy": {"install_hosts": []}
}
```

`SUPPORTED_INSTALLERS` is now `("npm", "python_venv")`; any other value fails
closed with `MCP_CATALOG_INVALID` at load time — including
`"preinstalled_executable"` (Task 8: scoped OUT rather than shipped as an
unsafe partial backend; `mcp_management.installers.get_installer()` returning
`None` for an unregistered type is the second, defense-in-depth fail-closed
gate, surfaced at runtime as `MCP_INSTALLER_UNSUPPORTED`).

**Backward compatibility is structural, not a special case.** An npm entry
(including the real Filesystem entry) keeps its old flat `entrypoint` field and
none of the new python_venv-only fields are ever populated for it — the two
installer shapes are simply disjoint branches of the same `build_entry()`
function, so `config/mcp_catalog.json`'s existing entry byte-for-byte validates
the same as before Phase G.3
(`tests/test_generalized_mcp_catalog_installers.py::
test_existing_filesystem_entry_still_loads_unchanged`). The stricter new rule
— every catalog-enabled tool must be one of the exact `expected_tools` — is
applied ONLY to python_venv (new-schema) entries; Filesystem's own
`default_tool_policy` deliberately enables more tools than its narrow
`expected_tools` health check, and that stays exempt.

## Plan security (Task 3)

`AutoProvisioningPlan.security_fields()` covers: `request_id`, `catalog_id`,
`server_id`, `installer_type`, `exact_package`, `exact_version`,
`lock_file_hash`, `executable_identity`, `expected_tools`, `tool_policy_hash`,
`environment_allowlist`, `install_network_hosts`, `runtime_network_policy`,
`target_install_directory`, `candidate_config_hash` — `plan_hash =
hash_arguments(security_fields())`. Changing ANY of these (a catalog edit, a
different version, a modified tool policy) produces a different hash, so a
stale approval no longer matches
(`tests/test_mcp_provisioning_plan_hash.py`). `expires_at` bounds how long an
unapproved plan is honored (`AutoProvisioningPlan.is_expired()`), and
`require_auto_provisioning_approval` re-checks the type (never a Phase F
`ProvisioningApproval` or a Phase F.1 `FilesystemAccessApproval`), the approval
flag, expiry, and the exact hash — every check independently enforced, not
merely implied by control flow
(`tests/test_mcp_auto_provisioning_approval.py`).

`AutoProvisioningManager.provision_and_activate` re-validates the plan against
the LIVE catalog and installed-server registry immediately before installing
(not only at prepare-time): a catalog edit since the plan was shown invalidates
it (`MCP_PROVISIONING_PLAN_INVALID`), and a server that became installed in the
meantime (e.g. a concurrent approval) short-circuits to reuse it instead of a
second install. `attempts >= MAX_PROVISIONING_ATTEMPTS` (1) makes approval
single-use — repeating "yes" after a success is rejected with
`MCP_PROVISIONING_ALREADY_IN_PROGRESS`, never a second transaction.

## The installation transaction (Task 9)

```
app_data/mcp_servers/<server_id>/
├── candidates/<transaction_id>/   # created, installed into, torn down here
├── versions/<version>/            # the ONLY thing ever promoted into
├── server.json                    # generated, Phase-E-validated managed config
└── installed_servers.json         # (shared, at the managed root) registry
```

`_run_transaction` (`mcp_management/auto_provisioning.py`): resolve the
installer for `plan.installer_type` → `prepare_candidate` → `install_candidate`
→ `validate_artifacts` → build a launch spec → generate + Phase-E-validate a
config pointed at the CANDIDATE directory → start the real candidate process
and validate it (Task 10, never registered into the production `ToolRegistry`)
→ atomically `os.rename` candidate → `versions/<version>` (skipped entirely
when the npm backend detects an already-intact installed version — Task 6's
reuse path) → regenerate the config against the now-final path → write
`server.json` → `upsert` the registry entry → hand off to Phase G.2. Any
failure at any step removes the candidate directory and — if a partial
promotion already happened — the just-created version directory too, and
never touches a previously-installed healthy version or another server's
state (`tests/test_mcp_provisioning_rollback.py`).

## Candidate validation (Task 10)

`_validate_candidate_process` calls `mcp_layer.external.start_server` on the
CANDIDATE's own generated config — the exact same real launch/initialize path
every other MCP server in this project uses — then `tools/list`, then always
shuts the process down (`finally`). Tool-name comparison is EXACT set
membership against `catalog_entry.expected_tools` and every catalog-enabled
policy tool: `"calculator-test.add"` in a catalog does NOT match a candidate
that exposes bare `"add"` (`tests/test_mcp_candidate_validation.py::
test_exact_name_comparison_not_suffix_matching`). Because
`default_tool_policy.default_permission` is structurally forced to `denied` at
catalog-LOAD time (unchanged Phase F rule), any tool the candidate exposes that
the catalog never explicitly enabled is denied by construction — never
silently READ.

## npm compatibility (Task 6)

`NpmInstaller.install_candidate` checks whether `versions/<version>/` already
has an intact entrypoint (via the SAME `npm_installer.validate_entrypoint` Phase
F always used) before doing anything else; if so, it returns immediately with
`reused_existing_installation="true"` and never calls `npm install`
(`tests/test_mcp_npm_installer_regression.py::
test_no_npm_call_occurs_when_reusing` monkeypatches `npm_installer.
install_package` to raise `AssertionError` if called, and it never fires). The
real, already-installed Filesystem server on this machine is unaffected by
Phase G.3 — no reinstall, same `app_data/mcp_servers/filesystem/` contents,
same lazy activation.

## Python isolation (Task 7)

Every candidate gets its own `venv` (`python -m venv`, from `sys.executable` —
the interpreter already running the assistant, never downloaded, never
model-provided), verified against `catalog_entry.python_constraint` via
`packaging.specifiers.SpecifierSet` BEFORE creating anything. Dependencies come
only from a committed, catalog-named lock file
(`config/mcp_locks/calculator-test-mcp-1.0.0.txt`) via
`pip install --require-hashes --no-input --disable-pip-version-check
--no-cache-dir -r <lock file>`, invoked with `cwd` set to the repository root so
the lock file's own relative wheel path resolves portably across machines/clones
— no absolute path is ever committed. `PYTHONNOUSERSITE=1` in the child
environment blocks user-site bleed-through; the venv itself is never created
with `--system-site-packages`. After install, the installed distribution's
EXACT version is verified via `importlib.metadata` inside the candidate venv
itself, and no executable this backend will ever launch resolves outside that
venv's own directory tree (`tests/test_mcp_python_venv_installer.py`).

The `calculator-test-mcp` package (Task 19's test-only fixture — never a
production catalog entry) is a genuine local Python package
(`tests/fixtures/calculator_mcp_pkg/`) exposing `add`/`echo` over the same
newline-delimited JSON-RPC 2.0 wire protocol every other MCP server in this
project speaks, built once into a committed wheel
(`tests/fixtures/calculator_mcp_pkg/dist/*.whl`) and installed purely from that
local file — zero network access at test or install time.

## Assistant integration (Tasks 13/14/15)

```
selection.status == SELECTED
  -> ensure_selected_server_active (Phase G.2)
       activated                          -> Phase B (unchanged)
       MCP_SERVER_NOT_INSTALLED
         + auto_provisioning wired up
         + catalog entry eligible (no directory grant required)
                                           -> _offer_mcp_provisioning (Task 13/15)
                                                -> deterministic plan text, HALT
                                                   (no Phase B, no LLM install tool)
         otherwise                        -> the pre-G.3 message, unchanged
```

`_offer_mcp_provisioning`/`_resolve_auto_provisioning_reply` mirror Phase
F.1's `_offer_filesystem_access`/`_resolve_filesystem_access_reply` shape
exactly (a plan, a pending request keyed by an opaque id, exact-word
yes/no/show-plan matching — never "contains yes"). The pending id itself
(`"autoreq_..."` vs. `"fsreq_..."`) is what `main()`'s REPL loop uses to pick
the right resolver — never guessed from the reply text — so the two cross-turn
flows can never be confused with each other, and `_restart_mcp_and_resume`
(the F.1 runtime-REPLACE coordinator) is never invoked here: a freshly
installed server has no prior session to replace, so the G.3 resumption path
re-enters routing and `_process_local_request_with_capability_selection`
directly instead. There is no `mcp.server.install` tool the local model could
call or hallucinate — the entire decision
(`capability selected + approved provider + not installed = offer a plan`) is
deterministic Python control flow in `assistant.py`, reached BEFORE Phase B's
shortlist is ever built.

`AutoProvisioningManager` is attached to the existing `McpProvisioningManager`
as a plain attribute (`manager.auto_provisioning = ...`) in
`_start_auto_provisioning`, not a new constructor parameter — so
`McpProvisioningManager.__init__` is completely unchanged and the many existing
tests that build one directly are unaffected;
`getattr(manager, "auto_provisioning", None)` is how the rest of the module
finds it, tolerating "not wired up" exactly like the existing `manager is None`
tolerance throughout this file.

## Request resumption (Task 13 step 11, Task 20 scenario X)

`AutoProvisioningManager.resume(request_id)` only returns the ORIGINAL text —
it never calls the newly installed tool itself. The caller re-enters
`route_and_answer` → `_process_local_request_with_capability_selection`, the
SAME single authoritative entrypoint every other local request uses, so the
resumed request goes through capability selection again, Phase G.2 activation
again (now `HEALTHY`, so it's a no-op reuse), the Phase B shortlist, and the
ToolExecutor — exactly once. `request.attempts >= MAX_PROVISIONING_ATTEMPTS`
(checked before any lock is acquired) makes a second automatic install attempt
for the same original request impossible.

## Concurrency (Task 17)

One `threading.Lock()` per `server_id` (`AutoProvisioningManager._lock_for`,
lazily created). `provision_and_activate` attempts a **non-blocking**
`lock.acquire(blocking=False)`: the losing thread gets
`MCP_PROVISIONING_ALREADY_IN_PROGRESS` immediately rather than silently
queuing behind a slow install, and the winning thread's completed install is
what every subsequent caller for that `server_id` observes via the
already-installed short-circuit (`tests/test_mcp_provisioning_concurrency.py`,
real threads, real subprocess installs). Two DIFFERENT `server_id`s never
share a lock.

## Security (Task 18)

Every subprocess call across both installer backends uses `subprocess.run([...],
shell=False)` — never a shell string, never `shell=True`. No package name,
version, path, or permission is ever read from user text or model output — the
entire plan is derived from `McpCatalogEntry` fields, which are themselves
schema-validated at catalog-load time (exact-version regex, safe-relative-path
checks, a fixed `python_constraint` charset, `default_permission` forced to
`denied`). npm lifecycle scripts remain unconditionally disabled
(`--ignore-scripts`, unchanged from Phase F). The python_venv backend never
allows an editable install, extras, a Git/HTTP URL, or a path outside the
repository's own committed lock file — `pip install --require-hashes` fails
closed the moment ANY resolved package (direct or transitive) lacks a verified
hash.

## Files added

- `mcp_management/provisioning_models.py`
- `mcp_management/auto_provisioning.py`
- `mcp_management/installers/__init__.py`, `base.py`, `npm_backend.py`, `python_venv.py`
- `tests/fixtures/calculator_mcp_pkg/` (pyproject.toml, package source, built wheel)
- `config/mcp_locks/calculator-test-mcp-1.0.0.txt`
- `tests/auto_provisioning_helpers.py`
- `scripts/manual_verify_g3_auto_provisioning.py`
- `tests/test_generalized_mcp_provisioning_models.py`
- `tests/test_generalized_mcp_catalog_installers.py`
- `tests/test_mcp_provisioning_plan_hash.py`
- `tests/test_mcp_auto_provisioning_approval.py`
- `tests/test_mcp_python_venv_installer.py`
- `tests/test_mcp_npm_installer_regression.py`
- `tests/test_mcp_candidate_validation.py`
- `tests/test_mcp_atomic_activation.py`
- `tests/test_mcp_provisioning_rollback.py`
- `tests/test_mcp_provisioning_concurrency.py`
- `tests/test_assistant_auto_mcp_provisioning.py`
- `tests/test_mcp_provisioning_request_resumption.py`

## Files modified

- `mcp_management/catalog.py` — python_venv installer schema + validation (Task 2), additive.
- `mcp_management/registry.py` — optional new `InstalledServer` fields (Task 12), additive.
- `mcp_management/configuration_generator.py` — added
  `generate_config_dict_from_launch_spec` (Task 11), additive; the existing
  `generate_config_dict` is untouched.
- `mcp_management/capability_detector.py` — added a generic, catalog-driven
  capability classifier for granular capabilities outside the
  filesystem/document families (needed for `arithmetic_calculation` to be
  detectable at all); explicitly excludes every capability id the existing
  filesystem/document classifiers already own, so their behavior is unchanged.
- `assistant.py` — `_start_auto_provisioning`, `_offer_mcp_provisioning`,
  `_resolve_auto_provisioning_reply`, wired into
  `_process_local_request_with_capability_selection` and `main()`.
- `tools/models.py` — new `MCP_PROVISIONING_*`/`MCP_INSTALLATION_*`/etc. error codes (Task 16).

## Manual CLI acceptance (Task 22) — real transcript

Run against a temporary, isolated catalog/managed-root/workspaces-root (the
existing `MCP_CATALOG_PATH`/`MCP_MANAGED_ROOT`/`MCP_WORKSPACES_ROOT` env
overrides already support absolute paths — no new override was needed). This
transcript is a REAL run against the real local Ollama model and a real
`calculator-test-mcp` install — not a simulation:

```
home-ai (LLM router v2 - consolidated)
MCP provisioning: 8 tool(s) available; catalog has 1 approved server(s).
Filesystem access management: 4 tool(s) available.
mode [t=text, p=push-to-talk, q=quit]: > routing: mode=local tool=None
capability detection:
  capability: arithmetic_calculation
  evidence:
    - action_object: add
trusted provider lookup:
  capability: arithmetic_calculation
  candidates: 1
  result: selected
  selected: calculator-test
MCP calculator-test:
  state: not_installed
  tools registered: 0
  error: MCP_SERVER_NOT_INSTALLED
Install approved MCP server

Server:
  Calculator Test MCP
...
Expected tools:
  - add
  - echo
...
Proceed?
mode [t=text, p=push-to-talk, q=quit]: > routing (resumed): mode=local tool=None
MCP calculator-test:
  state: healthy
  tools registered: 2
local llm tool: mcp.calculator-test.add
local llm tool result: mcp.calculator-test.add -> ok
local llm tool: none (answered directly)
10 + 20 = 30
mode [t=text, p=push-to-talk, q=quit]: > routing: mode=local tool=None
MCP calculator-test:
  state: healthy
  tools registered: 2
local llm tool: mcp.calculator-test.add
local llm tool: none (answered directly)
1 + 2 = 3
mode [t=text, p=push-to-talk, q=quit]: q
```

Note the second "add 1 and 2" turn shows NO approval prompt at all — the
already-installed server is reused directly. Confirmed after the run:
`git status` on `config/mcp_catalog.json` and `app_data/mcp_servers/` showed no
changes (the real production catalog and Filesystem installation were never
touched), and no orphan `calculator-test` process remained after quitting.

## Out of scope (unchanged from the task boundary)

Production MarkItDown entry, document conversion, server-specific folder
access for non-Filesystem servers, combined install-and-folder-access plans,
remote HTTP MCP, OAuth, Composio, Playwright, finance MCP, multi-server
workflows, automatic public MCP discovery, package updates, automatic repair,
uninstall workflow, and idle runtime shutdown are all still out of scope — see
the task's own OUT OF SCOPE list. A `preinstalled_executable` installer type
was deliberately NOT implemented (Task 8) rather than shipped unsafely; the
typed schema boundary (`SUPPORTED_INSTALLERS`, `get_installer()` returning
`None`) is in place so it can be added later without touching this phase's
other code.

## Tests

```
pytest -q tests/test_generalized_mcp_provisioning_models.py
pytest -q tests/test_generalized_mcp_catalog_installers.py
pytest -q tests/test_mcp_provisioning_plan_hash.py
pytest -q tests/test_mcp_auto_provisioning_approval.py
pytest -q tests/test_mcp_python_venv_installer.py
pytest -q tests/test_mcp_npm_installer_regression.py
pytest -q tests/test_mcp_candidate_validation.py
pytest -q tests/test_mcp_atomic_activation.py
pytest -q tests/test_mcp_provisioning_rollback.py
pytest -q tests/test_mcp_provisioning_concurrency.py
pytest -q tests/test_assistant_auto_mcp_provisioning.py
pytest -q tests/test_mcp_provisioning_request_resumption.py
pytest -q   # full suite
```

For a fully scripted, real-process equivalent of the manual CLI session (plan,
install, validate, activate, resume, reuse, clean stop, second-server
isolation), see `scripts/manual_verify_g3_auto_provisioning.py` — isolated
under a temp directory, never touches `app_data/mcp_servers/` or
`config/mcp_catalog.json`.
