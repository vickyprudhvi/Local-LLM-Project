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

MCP is **disabled by default**. Edit `config/mcp_server.json`:

- `"enabled": false` — no subprocess starts, no MCP tools register, built-ins work.
- `"enabled": true` — the configured server is started at assistant startup.

Set `MCP_CONFIG_PATH` to use a different config file.

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
anything. To use a server you must already have its executable on `PATH` (or give
an absolute path). For the bundled test server the command is:

```json
"command": "python",
"args": ["-m", "test_mcp_server"]
```

`test_mcp_server` is a real process (`python -m test_mcp_server`) that speaks
JSON-RPC on stdio — not a static manifest.

## Local permission authority

The server's advertised permissions are **ignored**. A tool registers only if it
is present in `tool_policy.tools` **and** `enabled`, and it gets the **locally
configured** permission. A tool missing from the policy is **denied** (never
registered, never shortlisted, never reaches the server). Write tools still go
through the Phase C confirmation prompt.

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
pytest -q
# Phase E only:
pytest tests/test_mcp_config.py tests/test_mcp_external_integration.py \
       tests/test_mcp_security.py tests/test_mcp_lifecycle.py -q
```

## Current limitations

- Exactly **one** server, `stdio` transport only (multiple servers = Phase F).
- No install/download of servers (`npm`/`npx`/`pip` are never invoked).
- No HTTP/SSE/WebSocket transports; no registry browsing.
- The launcher sets `PYTHONPATH` to the repo root so `python -m <pkg>` resolves —
  harmless for a non-Python external server, but noted.
- Dependency-name tokens surfaced by other tools are still lightly-bounded
  untrusted text (unchanged from Phase C).
