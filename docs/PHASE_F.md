# Phase F — Automatic MCP Provisioning and Configuration

> **Phase F allows the assistant to determine which approved MCP capability is
> required and automatically prepare, install, configure, validate, and activate
> that server after explicit user approval.**
>
> **The LLM never generates or executes arbitrary installation commands.
> Installation details come only from the trusted MCP catalog and deterministic
> installer code.**
>
> **Installation approval, directory access, credential access, and write
> permissions remain user-controlled.**

## What automatic provisioning means

Phase E could talk to one MCP server, but a human had to install it, find its
executable, and hand-edit `config/mcp_server.json`. Phase F removes that work:

```
request -> capability detected -> approved catalog entry selected
        -> provisioning plan -> USER APPROVAL -> isolated install (exact pin)
        -> configuration generated -> server validated -> activated
        -> original request re-runs through the normal pipeline
```

Nothing about the runtime architecture changes. A provisioned server is activated
as an ordinary Phase E configuration, so the normal bootstrap registers its tools
as `McpTool(BaseTool)` and the existing `ToolExecutor` runs them under the existing
Phase C permission and confirmation rules. `router.py`, `tools/executor.py`,
`tools/registry.py`, and `tool_loop.py` are unchanged.

## The trusted catalog

`config/mcp_catalog.json` is the **only** source of installable servers. It is
application-maintained data, schema-validated at load time, and fails closed:

| Rejected | Why |
| --- | --- |
| `latest`, `*`, `^1.2.3`, `~1.2.3`, `>=1.0.0`, `1.x` | versions must be an exact pin |
| installer types other than `npm` | only one installer is implemented |
| transports other than `stdio` | no HTTP/SSE/WebSocket |
| `default_permission` other than `denied` | no blanket access to undeclared tools |
| absolute or `..`-containing entrypoints | must stay inside the install directory |
| duplicate `server_id`s, bad ids, empty capabilities | ambiguous or incomplete entries |

Catalog `description` text is sanitized and length-bounded before it is displayed
or summarized, so metadata cannot smuggle instructions into a prompt.

Phase F ships exactly one production entry, `official-filesystem`
(`@modelcontextprotocol/server-filesystem`, pinned to an exact version). GitHub,
database, Docker, Slack, and browser servers are deliberately absent.

### Why arbitrary package installation is prohibited

If the model could name a package, a prompt-injected instruction inside a web page
or repository README could become `npm install <anything>`. Provisioning therefore
accepts only a `catalog_id` that already exists in the trusted file; anything else
returns `MCP_SERVER_NOT_APPROVED` and no process starts.

## Capability detection

`mcp_management/capability_detector.py` decides whether a request needs an MCP
capability. It is deterministic (patterns over the request plus the catalog's own
capability list), so it is reproducible and testable. Its output is a typed
`CapabilityDetection` that structurally cannot carry a command, package, path, URL,
environment value, or permission override.

- filesystem-shaped requests ("Read notes.txt", "List files in this folder") select
  `official-filesystem`
- knowledge questions ("What is the capital of France?", "Explain SQL joins",
  "Write a poem") need no MCP at all
- a recognized capability with no approved entry ("Check GitHub pull requests")
  returns `MCP_CAPABILITY_UNAVAILABLE` — it explains why nothing happened rather
  than silently doing nothing

`validate_detection` re-checks any detector output against the catalog, so even a
future LLM-assisted detector cannot introduce an unapproved server.

## The provisioning plan

A plan is immutable and describes exactly what will happen: package and exact
version, install directory, isolated runtime workspace, requested directories,
requested environment-variable **names**, the proposed read/write/denied tools, the
risk category, whether install needs network, and whether credentials are needed.
It holds no secret values.

`plan_hash` covers every security-relevant field, and `plan_id` is derived from it,
so identical inputs always produce an identical plan.

## Approval

Installation approval is its own type (`ProvisioningApproval`), bound to a specific
`plan_id` **and** `plan_hash`. A Phase C write-tool confirmation can never be
mistaken for installation approval. Changing the catalog id, package, version,
install directory, approved directory, environment names, or tool policy changes
the hash, so a prior approval no longer matches
(`MCP_PROVISIONING_CONFIRMATION_MISMATCH`).

The prompt shows the deterministic plan summary and defaults to **No**. Declining
returns `MCP_PROVISIONING_DECLINED` and leaves no directory, configuration, or
registry entry behind.

## Installation layout

Installs are isolated under the managed root — never global, never the repository
root, never the assistant's virtualenv, never with elevated privileges:

```
app_data/mcp_servers/
├── installed_servers.json          # application-owned registry (atomic writes)
└── filesystem/
    ├── versions/
    │   └── 2026.7.10/              # immutable after a successful install
    │       ├── package.json        # private manifest (isolates npm)
    │       ├── package-lock.json
    │       └── node_modules/
    ├── server.json                 # generated Phase E configuration
    ├── permissions.json            # applied policy snapshot
    ├── install-record.json         # immutable audit record
    ├── current.json
    └── uninstall-record.json       # bounded audit trail after removal
```

The npm invocation is built by `build_npm_argv` from the plan alone:
`npm install <package>@<exact-version> --save-exact --omit=dev --no-audit --no-fund
--ignore-scripts`, run with `shell=False`, an argv list, a bounded timeout, bounded
sanitized output, and no `-g`. A failed install is never retried with another
version or tag. Node and npm must already exist; a missing runtime returns
`MCP_RUNTIME_MISSING` and nothing is installed on the user's behalf.

### npm lifecycle scripts

> **Phase F does not permit npm lifecycle scripts. All managed npm installations
> use `--ignore-scripts`.**

This is structural, not a default:

- `--ignore-scripts` is appended unconditionally in `build_npm_argv`; there is no
  branch that omits it.
- A catalog entry containing `"allow_lifecycle_scripts": true` is **rejected** at
  load time with `MCP_CATALOG_INVALID`. The key may only be absent or exactly
  `false`.
- The field exists on neither `McpCatalogEntry` nor `McpProvisioningPlan`, so
  there is nothing for a caller, a plan, or model output to flip.
- There is no environment variable, config flag, CLI switch, or development
  override that re-enables them.

The official filesystem server ships a prebuilt `dist/`, so it installs and
validates correctly with scripts disabled.

## Directory-access approval

The requested directory is canonicalized and screened before it can appear in a
plan. Always refused: the filesystem root, system directories, `.ssh`, `.aws`,
`.gnupg`, cloud credential folders, browser profiles, and the assistant's venv.
Refused unless the user explicitly approves a broad scope: the home root, the
repository root, and whole `Documents`/`Desktop`/`Downloads`/`OneDrive` folders —
prefer the narrowest directory that satisfies the request. Illegal characters
(null, newline) and non-existent paths are rejected.

## Permission defaults

Permissions come from the trusted catalog, never from the server. Read-only tools
are `read`; create/write tools are `write` (and still hit the Phase C confirmation
on every call); move/delete/edit and **any tool not in the policy** are denied and
never registered, so a denied tool cannot reach the server at all.

The official server advertises `write_file` with read-ish annotations; the local
policy still makes it `write`. Server-advertised permissions are ignored.

## Configuration generation

`configuration_generator.py` produces the Phase E document automatically: absolute
runtime executable, absolute managed entrypoint, canonical approved directories as
server arguments, an isolated `mcp_workspaces/<server_id>` working directory, safe
default timeouts, environment **names** only, and the catalog policy. It is always
validated with the real Phase E loader (`mcp_layer.config.build_config`) before it
is written or activated, and written atomically. You never hand-edit JSON.

## Which configuration is in effect

A generated configuration is machine-specific (absolute executable, entrypoint,
workspace, and approved-directory paths), so it must never land in source control.
**Phase F never writes `config/mcp_server.json`.** That file stays a portable,
committed, disabled-by-default template.

Instead the effective configuration is resolved at startup by
`mcp_layer/config_resolver.py`, in this order:

| # | Source | Where |
| --- | --- | --- |
| 1 | `ENVIRONMENT_OVERRIDE` | `MCP_CONFIG_PATH` — an explicit operator override |
| 2 | `MANAGED_ACTIVE` | `app_data/mcp_servers/<server_id>/server.json` while that server is enabled |
| 3 | `DEFAULT_TEMPLATE` | the committed `config/mcp_server.json` |

A server counts as *managed active* when the registry lists it as `installed` **and**
its generated configuration has `enabled: true`. Disabling flips that flag, so the
next startup falls back to the committed (disabled) template — no file in `config/`
is ever rewritten. Uninstalling removes the managed configuration and registry
entry, again leaving `config/mcp_server.json` untouched.

`MCP_CONFIG_PATH` is validated: canonicalized, required to be an existing regular
file, and rejected if it contains null or newline characters
(`MCP_CONFIGURATION_INVALID`). It comes only from the environment — the LLM can
neither read nor set it. Logs record just the source and the file's basename
(`managed_active:server.json`), never the full path alongside configuration
contents.

### The committed template

`config/mcp_server.json` ships disabled, with `server_id: "disabled"`, an empty
`command`, no arguments, an empty environment allowlist, and an empty tool policy.
It contains no absolute paths, usernames, entrypoints, workspaces, credentials, or
managed installation state, so it is safe to commit. While disabled, no executable
is resolved and no subprocess is started.

### The managed configuration

A provisioned server's configuration is **local generated state** and is never
committed:

```
app_data/mcp_servers/<server_id>/server.json
```

### The environment override

`MCP_CONFIG_PATH` is for trusted local administration and testing. It is **not**
selected by the LLM, not produced by the capability detector, not stored in the
catalog, and not part of any provisioning request — it is read only from the
process environment. An invalid override fails visibly
(`MCP_CONFIGURATION_INVALID`) instead of silently falling back to the template, so
a typo cannot quietly run a different server.

To point at a hand-installed server without editing the committed template, set
`MCP_CONFIG_PATH` to your own JSON file (`config/mcp_server.local.json` is
gitignored for exactly this).

### Managed-state integrity

The resolver treats the registry as untrusted-ish local state: a
`configuration_path` is honoured only if it canonically resolves **inside** the
managed root, and the generated document's `server_id` must match its registry key.
A tampered or corrupt registry therefore cannot redirect startup to an arbitrary
executable — resolution falls back to the disabled template.

### Resetting local state

To remove all managed MCP state without touching your files:

```bash
# stops using and removes the managed installation (user workspaces untouched)
rm -rf app_data/mcp_servers/<server_id>
rm -f  app_data/mcp_servers/installed_servers.json
```

`config/mcp_server.json` needs no repair — it is never modified. Directories you
approved, and everything inside them, are never deleted by uninstall or by a reset.

## Post-install validation

Before anything is promoted or recorded, the freshly installed server is started
and checked: entrypoint present, process starts (`shell=False`), `initialize`,
protocol version reported, `notifications/initialized`, `tools/list`, at least one
expected core tool, schema limits and local policy applied, unknown tools denied,
the server's reported allowed roots match the approved directories, and a read
smoke test. The write smoke test is off by default and, when enabled, only touches
a disposable installer-owned file that is deleted afterwards. The server is always
shut down, so no orphan process remains.

Installation is a transaction: install into a staging directory, validate, then
atomically promote. Any failure removes the staging directory, writes no registry
entry, activates nothing, and preserves a previously installed healthy version.

## Original-request resumption

The original request is preserved in a `PendingCapabilityRequest`. After successful
provisioning, `manager.resume()` returns the original text so it re-runs through
routing, the Phase B shortlist, and the `ToolExecutor` — the installer never calls
the newly installed MCP tool itself. At most **one** provisioning attempt is allowed
per original request (`MCP_PROVISIONING_LOOP_PREVENTED`), so a
detect → install → detect loop is impossible.

## Disable, repair, uninstall, update

- **provision** — installs, generates the managed configuration, validates it, and
  activates it. The committed template is not touched.
- **disable** — marks the *managed* configuration disabled and updates the
  registry; installed files are preserved and resolution falls back to the
  committed disabled template.
- **enable** — reactivates the existing installation and never reinstalls (npm is
  not invoked).
- **repair** — verifies the recorded installation and restores the **same** pinned
  version when files are missing. If the catalog now pins a different version it
  reports `MCP_UPDATE_AVAILABLE` instead of upgrading.
- **uninstall** — removes only managed files (version directory, generated
  configuration) plus the registry entry, keeps a bounded audit record, and is safe
  to run twice. **Files in approved directories are never deleted, and
  `config/mcp_server.json` is left untouched.**
- **updates** are never automatic; a newer approved version is only reported.

Through every one of these operations the tracked `config/mcp_server.json` is
byte-identical — a property asserted by tests and verified end to end against a
real npm install.

## Provisioning tools

Provisioning is exposed as ordinary built-in tools, so it runs through the existing
executor and permission model: `mcp.catalog.search`, `mcp.provision.plan`,
`mcp.provision.status` (read) and `mcp.provision.install`, `mcp.server.enable`,
`mcp.server.disable`, `mcp.server.repair`, `mcp.server.uninstall` (write, so each
needs confirmation). Their schemas accept only trusted identifiers — a catalog id,
a plan id/hash, a server id, and a directory that deterministic code then screens.
None accepts a package name, command, URL, executable path, or shell argument.

## Running tests

```bash
# Configuration resolution + portable-template checks:
pytest tests/test_mcp_config_resolution.py -q

# Phase F only:
pytest tests/test_mcp_catalog.py tests/test_mcp_capability_detector.py \
       tests/test_mcp_provisioning_plan.py tests/test_mcp_provisioning_approval.py \
       tests/test_mcp_npm_installer.py tests/test_mcp_install_transaction.py \
       tests/test_mcp_configuration_generation.py tests/test_mcp_config_resolution.py \
       tests/test_mcp_post_install_validation.py tests/test_mcp_request_resumption.py \
       tests/test_mcp_server_management.py tests/test_mcp_phase_f_security.py -q

# Everything:
pytest -q
```

Git hygiene — no generated MCP state may ever be staged:

```bash
git status --short          # expect no app_data/ or node_modules entries
git check-ignore -q app_data/mcp_servers/installed_servers.json && echo ignored
git diff --cached --check   # whitespace/conflict markers
```

Installation tests never touch the network: a fake npm materializes the package
from `tests/fixtures/fake_filesystem_server.js`, a **real** Node stdio MCP server,
so start/initialize/tools-list/policy/smoke-test/shutdown are exercised for real.

## Manual smoke test

1. Create a disposable directory with a file:
   `mcp_workspaces/user_files/hello.txt` containing
   `Hello from automatic MCP provisioning!`
2. Ask to read that file. The assistant detects the filesystem capability, finds no
   installed server, selects `official-filesystem`, and presents a plan showing the
   package, exact version, install directory, approved directory, and the
   read/write/denied tools. **Nothing is installed yet.**
3. Decline once and confirm no install directory, no configuration, and no registry
   entry exist.
4. Ask again and approve. The exact pinned version installs, the configuration is
   generated and validated, the server starts, tools register, and the original
   request resumes and returns the file contents.
5. Ask to create a file — the Phase C write confirmation appears; declining creates
   nothing, approving creates it exactly once.
6. Ask to move a file out of the approved directory — denied, and the server is
   never called.
7. Disable (process stops, tools unregister, files preserved), re-enable (no npm
   run), then uninstall (managed install removed, **your files untouched**).

## Known limitations

- **One catalog entry** (`official-filesystem`) and one active server. Choosing
  among several installed servers at runtime is Phase G.
- Only `stdio`; only the `npm` installer.
- `npm install` needs network access at install time; nothing else does.
- Installing does not verify a package signature or checksum beyond npm's own
  integrity checking; the `package-lock.json` hash is recorded for audit.
- `--ignore-scripts` is the default. A package that genuinely requires lifecycle
  scripts must opt in through the catalog, which is a real trust escalation.
- The capability detector is pattern-based, so an unusually phrased request may not
  be recognized (it then simply behaves as before, with no MCP).
- Only one managed server can be *active* at a time. The resolver picks the first
  enabled managed server in id order, so enabling a second one is not meaningful
  until multi-server runtime selection (Phase G).
- A managed configuration records absolute paths captured at install time. Moving
  the repository or the managed root invalidates it; `repair` (or re-provisioning)
  regenerates it.
