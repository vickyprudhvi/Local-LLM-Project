# Phase 1 Tool-Framework — Implementation Plan

Narrow Phase 1: prove the local LLM can request a safe tool, have it execute, get a
structured JSON result back, and write the final natural-language answer itself. Two
built-in tools only: `system.echo`, `math.calculate`. Native Ollama tool calling only
(verified, no JSON fallback). No Phase 2 (no internet/GitHub/plugins/MCP/filesystem/
shell/camera/finance/etc.).

## Current repository architecture (as discovered)

Flat module layout, Python 3.13, pytest, talks to Ollama over HTTP via `requests`.

- `assistant.py` — entry point (`python assistant.py`); `main()` loop → `_enrich_with_memory`
  → `route_and_answer` → `dispatch` → `ask_local` / `ask_claude`. History is an in-memory
  list of `{"role","content"}` dicts; per turn only the user text and the final assistant
  reply are appended (lines 211-212).
- `router.py` — LLM classifier. Already uses native Ollama tool calling (`/api/chat` with
  `"tools"`, reads `message["tool_calls"]`). Returns `RouteDecision(mode ∈ local|claude|tool)`.
- `brain.py` — `ask_local(prompt, history, system_prompt) -> (text, metrics)`, `ask_claude`
  (server-side `web_search`), `trim_history`, `load_system_prompt`.
- `memory_store.py` — ChromaDB; the only writer is `remember()`, only called by the explicit
  `remember` tool branch.
- `interaction_log.py` — `log_turn()` appends one JSON line per turn.

## Relevant existing files

`assistant.py` (integration site), `brain.py` (request builder + trimming),
`interaction_log.py` (logging), `router.py`/`memory_store.py` (must remain untouched
behaviorally), `tests/` (pytest + `unittest.mock`, empty root `conftest.py` for flat imports).

## Exact integration point

`assistant.py`, the `mode == "local"` fallthrough (was line 161):
`return ask_local(prompt, history, system_prompt)` → now
`return tool_loop.run_local_tool_loop(prompt, history, system_prompt)`. This is the ONLY
behavioral change to the routing/dispatch flow. Claude and tool branches are unchanged.

## Files created

`tool_loop.py`; `tools/{__init__,models,base,registry,executor,echo,calculator}.py`;
`tests/{test_tool_models,test_tool_registry,test_tool_executor,test_calculator,test_tool_loop,
test_history_trimming,test_memory_tool_exclusion,test_ask_local_compat,test_tool_config}.py`;
`tests/conftest.py` (redirects interaction log to tmp); `docs/*.md`; `README.md`.

## Files modified

- `brain.py` — `OLLAMA_URL` from env (same default); new `ask_local_raw()` single request
  builder; `ask_local()` becomes a thin wrapper over it; new `trim_history_tool_aware()`.
- `interaction_log.py` — new `log_tool_event()` (same module/file, safe metadata only).
- `assistant.py` — one-line local-path change + `import tool_loop`.
- `.env.example` — `OLLAMA_URL`, `TOOL_CALLING_ENABLED`, `MAX_TOOL_STEPS`,
  `DEFAULT_TOOL_TIMEOUT_SECONDS`.

## Compatibility risks & mitigations

- `ask_local` callers expect `(text, metrics)` → preserved via thin wrapper (`test_ask_local_compat`).
- Router/Claude behavior → untouched; tests assert Claude route still calls `ask_claude` and
  `CLAUDE_TOOLS` web_search config is unchanged.
- Cloud model served through local daemon with no auth headers → replicated exactly.
- Tool messages polluting memory/history → they live only inside the loop's local message list,
  never returned or persisted (`test_memory_tool_exclusion`).
- Token overhead from schemas every local call → measured & documented; schemas omitted when
  disabled / none enabled / on the final exhaustion call.

## Test plan

Deterministic pytest, no live Ollama (FakeLLM replaces `ask_local_raw`). Covers models,
registry, executor (all controlled errors + timeout + logging), calculator (ops + every
rejection + limits), the loop (round-trips, multi-call, error-to-LLM, step-limit accounting,
exhaustion single final call, no infinite loop), history trimming (no orphaned tool messages),
memory exclusion, ask_local compatibility, Claude-route-unchanged, and config defaults.

## Tool-calling verification plan

Before writing protocol code, hit the live endpoint/model with a `math.calculate` tool def and
an arithmetic prompt several times; confirm `message.tool_calls` shape, argument type, multi-call
behavior, and the accepted tool-result message shape. Stop if it can't be verified. Results in
`docs/phase1-tool-call-test.md`.
