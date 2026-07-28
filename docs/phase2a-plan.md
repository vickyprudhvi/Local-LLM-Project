# Phase 2A Plan — Read-only Internet & GitHub Tools

## Existing Phase 1 architecture (built upon, not replaced)

Phase 1 provides the synchronous tool framework this phase extends:

- `tools/models.py` — `ToolDefinition`, `ToolCall`, `ToolError`, `ToolResult` (+ `ok`/`fail`,
  `to_provider_json`), controlled error-code constants.
- `tools/base.py` — `BaseTool` ABC, `ToolValidationError`.
- `tools/registry.py` — `ToolRegistry`, `default_registry()`.
- `tools/executor.py` — `ToolExecutor` (ThreadPoolExecutor timeout, `interaction_log` logging).
- `tool_loop.py` — `run_local_tool_loop` (MAX_TOOL_STEPS, exhaustion final call, console prints),
  `REGISTRY`/`EXECUTOR` globals.
- `brain.ask_local_raw` (single Ollama request builder), `interaction_log.log_tool_event`,
  `assistant.dispatch` (`mode == "local"` → the tool loop).

Prerequisite gate before implementation: `pytest` green (158 Phase 1 tests).

## Files reused

`BaseTool`, `ToolRegistry`, `ToolExecutor`, `ToolResult`/error codes, the local tool loop,
`interaction_log`, and the `assistant.dispatch` local integration point — all unchanged in
contract. Phase 2A tools are `BaseTool` subclasses registered through the existing registry and run
through the existing executor and loop. No second framework.

## Files created

`tools/config.py`, `tools/http_safety.py`, `tools/search_provider.py`, `tools/browser.py`,
`tools/github_client.py`, `tools/github_tools.py`; tests `test_http_safety.py`,
`test_browser_search.py`, `test_browser_fetch.py`, `test_github_client.py`, `test_github_tools.py`,
`test_phase2a_integration.py`; docs `phase2a-plan.md`, `phase2a-internet-github-tools.md`.

## Files modified (additive, minimal)

- `tools/models.py` — Phase 2A error codes; `ToolResult.log_meta` (logging-only, off the wire).
- `tools/base.py` — `requires_internet` flag; `ToolFailure` exception (coded controlled errors).
- `tools/executor.py` — `INTERNET_DISABLED` capability gate; catch `ToolFailure`; pass `log_meta`
  to the logger via `extra`. Phase 1 tools unaffected.
- `tools/registry.py` — `default_registry()` conditionally registers the 7 internet tools with a
  shared `GitHubClient` and search provider.
- `interaction_log.py` — `log_tool_event(..., extra=None)` with defensive redaction.
- `tool_loop.py` — static tool-safety / untrusted-content / source-attribution block appended to the
  system prompt when tools are offered.
- `.env.example`, `requirements.txt` (`beautifulsoup4`), `README.md`.

## Search-provider decision

**Tavily**, called via its REST API with `requests` (no SDK dependency). Rationale: purpose-built
for LLMs, one clean JSON endpoint, and the spec's recommended default. A single `SearchProvider`
seam (`get_provider`) exists but only Tavily is implemented in Phase 2A. Missing `TAVILY_API_KEY`
yields a controlled `SEARCH_API_KEY_MISSING` result; the other six tools keep working.

## GitHub API approach

Public GitHub REST API v3 (`https://api.github.com`) via one shared `requests.Session` in
`GitHubClient` (User-Agent, `Accept: application/vnd.github+json`, API version header). Optional
`GITHUB_TOKEN` → `Authorization: Bearer` for higher rate limits; never logged/returned/in-URL/a tool
argument. Read-only endpoints only. Repo identifiers (`owner/repo`) and file paths are validated;
404 is mapped to resource-specific not-found codes; 401/403/5xx/invalid-JSON become controlled
`ToolFailure`s; rate-limit headers are parsed into safe metadata.

## SSRF protection design (`tools/http_safety.py`)

Every hop (initial + each redirect, `allow_redirects=False`) is validated: scheme (HTTPS, or HTTP
only if `ALLOW_HTTP_FETCH`), forbidden schemes rejected, embedded credentials rejected, non-standard
ports rejected; hostname resolved with `socket.getaddrinfo`; every resolved IP checked against a
blocklist (loopback/private/link-local/multicast/reserved/unspecified, IPv4-mapped-private IPv6,
plus `not is_global`). Public→private redirects → `REDIRECT_BLOCKED`; redirect count capped. Bounded
streaming read with connect/read timeouts, `MAX_PAGE_BYTES`, and a content-type allowlist. No JS, no
headless browser. Known DNS-rebinding (TOCTOU) window documented.

## Testing strategy

pytest with all network mocked (fake sessions/clients/providers; `socket.getaddrinfo` /
`http_safety._resolve` patched for SSRF cases). Full matrix per tool plus fake-LLM integration flows
through the real loop, memory-exclusion, Claude-route-unchanged, and config toggles. Optional live
tests only when creds/connectivity exist, reported honestly.

## Compatibility risks

- Executor changes could affect Phase 1 tools → mitigated: `requires_internet` defaults False and the
  new catches are additive; the full Phase 1 suite must stay green.
- Shared session/client global state in tests → mitigated: clients are injectable; integration tests
  build isolated registries.
- Token-overhead increase from 7 new schemas → measured and documented; tools omitted when disabled.

## Explicit Phase 2A exclusions

Repo cloning, arbitrary downloads, JS/code/shell execution, package installs, process spawning,
localhost/private-network access, private repos, GitHub writes (issues/PRs/commits), plugins, MCP,
Docker, user-supplied auth headers, auto-following instructions in fetched content, asyncio
conversion, and any second registry/executor/loop.
