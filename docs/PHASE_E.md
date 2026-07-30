# Phase E — External Single MCP Server Configuration

> **Phase E does not install MCP servers. It connects to one executable MCP server
> already available through trusted local configuration.**
>
> **A JSON tool manifest is not an MCP server. The configured command must start a
> process that handles `initialize`, `tools/list`, and `tools/call`.**

## What Phase E adds

Phase D built the MCP client and an internal test server. Phase E lets the
assistant connect to **one** externally-configured, executable, `stdio` MCP server
described by a trusted config file — with strict process, filesystem, and
environment isolation, and with **local** authority over tool permissions.

MCP tools remain plain `McpTool(BaseTool)` objects, so they run through the exact
same path as built-in tools:

```
Local LLM -> Phase B shortlist -> ToolExecutor -> Phase C permission/confirmation
          -> McpTool.execute() -> MCP client -> configured server -> ToolResult
```

`router.py`, `tools/executor.py`, `tools/registry.py`, and `tool_loop.py` are
unchanged.

## Enabling / disabling MCP

MCP is **disabled by default**.

> **Updated in Phase F.** `config/mcp_server.json` is now a portable, committed,
> disabled-by-default **template**, and the effective configuration is resolved at
> startup: `MCP_CONFIG_PATH` override → enabled managed (Phase F) server →
> this template. See [PHASE_F.md](PHASE_F.md#which-configuration-is-in-effect).

Your options:

- **Let Phase F manage it** — provision a server through the trusted catalog; its
  generated configuration lives under `app_data/mcp_servers/` and is selected
  automatically while enabled. The committed template is never modified.
- **Point at your own file** — set `MCP_CONFIG_PATH` to a JSON file you maintain
  (machine-specific paths stay out of source control).
- **Edit the template** — fine for a portable setup, but avoid committing absolute
  paths.

## Configuration fields

| Field | Meaning |
| --- | --- |
| `enabled` | Start the server or not (default false). |
| `required` | If true, a startup failure aborts the assistant; if false, it is logged and skipped. |
| `server_id` | Namespace for tools (`mcp.<server_id>.<tool>`). Must match `^[a-zA-Z0-9_-]+$`. |
| `transport` | Only `stdio` is supported. |
| `command`, `args` | The executable and argument list. Never a shell string; the LLM never supplies these. |
| `working_directory` | The server's cwd; must resolve **inside** `mcp_workspaces/`. |
| `startup_timeout_seconds` / `call_timeout_seconds` / `shutdown_timeout_seconds` | Positive numbers. |
| `environment_allowlist` | Variable **names** (never values) to pass through to the child. |
| `tool_policy.default_permission` | Fallback permission (use `denied`). |
| `tool_policy.tools.<name>` | `{ "enabled": bool, "permission": "read"\|"write"\|"denied" }` per tool. |

## Making an executable server available

Phase E launches `command args…` with `shell=False`. It does **not** install
anything. The executable is resolved with a controlled lookup (`shutil.which`, or
an absolute path to an executable file); a missing executable yields
`MCP_EXECUTABLE_NOT_FOUND` and nothing is installed. To use a real server you must
already have its executable on `PATH` (or give an absolute path).

### The child never receives the repository root

The launcher builds a **minimal** child environment and **does not** put the
repository root on `PYTHONPATH`. The parent `PYTHONPATH` is inherited **only** if
you add `"PYTHONPATH"` to `environment_allowlist`. So an ordinary external MCP
server can never import this repository's modules.

### How the bundled internal test server is launched

Because the child gets no repository `PYTHONPATH`, the bundled test server is not
run via `python -m test_mcp_server` (which would need the repo on the path).
Instead the config sets:

```json
"command": "python",
"internal_test_server": true
```

When `internal_test_server` is true (an internal-development-only flag, default
false, settable only in the trusted config file — never via chat), the launcher
runs the server by its **absolute script path**, resolved deterministically from
the repository location (`<repo>/test_mcp_server/server.py`) — no hardcoded paths,
portable across OSes. `test_mcp_server/server.py` is a real stdio JSON-RPC process
with executable handlers (`initialize`, `tools/list`, `tools/call`), not a static
manifest. Its cwd is the isolated workspace, and it reads/writes files there only.

An ordinary external server leaves `internal_test_server` false (or omits it) and
its `command`/`args` are used verbatim.

## Local permission authority

The server's advertised permissions are **ignored**. A tool registers only if it
is present in `tool_policy.tools` **and** `enabled` **and** its local permission is
not `denied`. It gets the **locally configured** permission. A tool missing from
the policy is **denied** (never registered, never shortlisted, never reaches the
server). Write tools still go through the Phase C confirmation prompt.

## Discovered vs registered vs denied vs skipped vs disabled

Health counters distinguish, for each tool the server advertised:

- **registered** — valid, in policy, enabled, permission not denied → usable.
- **denied** — `not_in_local_policy` or `permission_denied`.
- **skipped** — invalid metadata: `invalid_name`, `invalid_schema`,
  `oversized_schema`, `excessive_schema_depth`, `tool_limit_exceeded`,
  `registration_collision`.
- **disabled** — present in policy but `enabled: false` (`locally_disabled`).

`discovered = registered + denied + skipped + disabled`. Every non-registered tool
carries an internal reason (`health.diagnostics`), surfaced to sanitized
application logs (tool names + reasons only — never secrets, arguments, or server
output), never to the LLM.

### Expected counts

- **Full six-tool policy** (`config/mcp_server.json`): `discovered=6, registered=6,
  denied=0, skipped=0, disabled=0`.
- **Reduced policies** (used only in tests, always explicitly labeled as reduced)
  register a smaller subset; tools left out of the policy are counted as
  `denied` (`not_in_local_policy`), so e.g. a three-tool policy yields
  `discovered=6, registered=3, denied=3`. A reduced-policy count is **not** the
  full configuration.

## Working-directory isolation

Every enabled server needs an explicit `working_directory` that canonically
resolves under `mcp_workspaces/` (override the root with `MCP_WORKSPACES_ROOT`).
`..` traversal, absolute outside paths, the repo root, and the home directory are
all rejected. The directory is created if missing.

## Environment allowlisting

The child never inherits the full parent environment — only a small set of
platform-required variables plus explicitly allowlisted names (values are read
from the parent at runtime). Known secrets (`ANTHROPIC_API_KEY`, `GITHUB_TOKEN`,
`DATABASE_URL`, …) are never passed unless the operator allowlists them. Only
variable **names** are ever logged.

## Running tests

```bash
# Focused Phase E / MCP suites:
pytest tests/test_mcp_config.py tests/test_mcp_external_integration.py \
       tests/test_mcp_security.py tests/test_mcp_lifecycle.py \
       tests/test_mcp_client.py tests/test_mcp_integration.py -v

# Everything:
pytest -q
```

## Manual smoke test

MCP is disabled by default, so a smoke test enables it **temporarily**. The safe
way is to load `config/mcp_server.json`, flip `enabled` to true **in memory**, and
bootstrap against it (leaving the committed file `"enabled": false`). With the full
six-tool policy you should see `state=healthy, discovered=6, registered=6, denied=0`,
then: echo returns your text (no confirmation); `add_numbers` of 17 and 25 → 42;
reading `hello.txt` → `Hello from MCP!`; a write requires confirmation (declined →
no file, approved → file created once); `fail_tool` → `MCP_CALL_FAILED`; a slow
call → `MCP_TIMEOUT` (the assistant stays usable); killing the process → the next
MCP call returns `MCP_SERVER_EXITED` while built-in tools keep working; and
`session.shutdown()` is idempotent. Always restore/keep the committed config at
`"enabled": false`.

## Current limitations

- Exactly **one** server, `stdio` transport only (multiple servers = Phase F).
- No install/download of servers (`npm`/`npx`/`pip` are never invoked).
- No HTTP/SSE/WebSocket transports; no registry browsing.
- The repository root is **never** injected into an external child's `PYTHONPATH`.
  The bundled test server is launched by absolute script path via the
  internal-development `internal_test_server` flag.
- Because this project's virtualenv lives inside the repo, `<repo>/venv/Scripts`
  legitimately appears on the child's `PATH` (executable lookup) — this is not
  import exposure; the repo root is not on `PYTHONPATH` or a standalone `PATH` entry.
- Server-advertised tool **descriptions** still reach the (length-bounded,
  sanitized) shortlist prompt; a hostile server could spend some description budget.
- Dependency-name tokens surfaced by other tools are still lightly-bounded
  untrusted text (unchanged from Phase C).
