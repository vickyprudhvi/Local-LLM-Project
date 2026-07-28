# Phase 1 Tool Framework

A minimal, synchronous, native-Ollama tool-calling loop wired into the **local** answering path
only. The local LLM is always the component that interprets tool results and writes the final
answer — a tool never produces the user-facing response.

## Integration point

`assistant.py`, `dispatch()`, `mode == "local"`:

```python
# was: return ask_local(prompt, history, system_prompt)
return tool_loop.run_local_tool_loop(prompt, history, system_prompt)
```

Both return `(text, metrics)`, so `dispatch()` and `main()` are otherwise unchanged. The
`claude` and `tool` branches are untouched.

## Architecture

```
user
  │
assistant.main ──► _enrich_with_memory ──► router.route_and_answer ──► dispatch
                                                                         │
                                        mode=="claude" ─► ask_claude (unchanged, web_search)
                                        mode=="tool"   ─► existing tool branches (unchanged)
                                        mode=="local"  ─► tool_loop.run_local_tool_loop
                                                                │
                          ┌─────────────────────────────────────┘
                          ▼
        brain.ask_local_raw(messages, tools) ──► Ollama /api/chat
                          │  assistant message (may contain tool_calls)
                          ▼
        no tool_calls? ─► return final text to user
        tool_calls?    ─► ToolExecutor.execute(ToolCall) ─► ToolResult
                          ─► append {"role":"tool","content":json.dumps(result),"tool_name":…}
                          ─► call ask_local_raw again … (up to MAX_TOOL_STEPS)
```

## `ask_local_raw()` — the single request builder

`brain.ask_local_raw(messages, tools=None, timeout=120)` is the **only** place that builds and
sends an Ollama `/api/chat` request for the local path. It:

- builds `{model, messages, stream:False, keep_alive:"10m"}` and adds `"tools"` only when truthy;
- shares URL, model, timeout, `keep_alive`, metrics extraction, and error handling with normal chat;
- returns the **complete** assistant message (including `tool_calls`) plus metrics:
  `{"message": {...}, "metrics": {"prompt_tokens", "completion_tokens", "eval_duration"}, "ok": bool}`;
- on a network error returns `ok=False` with the original fallback content string.

## `ask_local()` — backward compatibility

`ask_local(prompt, history, system_prompt)` is now a thin wrapper: it assembles
`[system] + trim_history(history,12) + [user]`, calls `ask_local_raw(messages)`, and returns
`(message.content.strip(), metrics)`. Signature and return shape are unchanged, so existing
callers (recall/calendar/scan_room summaries, tests) keep working. There is exactly **one**
request builder.

## Tool models (`tools/models.py`)

Plain dataclasses (no new dependency): `ToolDefinition`, `ToolCall`, `ToolError`, `ToolResult`.
`ToolResult.to_provider_json()` serializes `{success, tool_name, call_id, data, error}` — always
JSON-serializable, never exceptions/traces/secrets. Controlled error codes: `UNKNOWN_TOOL`,
`TOOL_DISABLED`, `INVALID_ARGUMENTS`, `TOOL_TIMEOUT`, `TOOL_EXECUTION_ERROR`,
`INVALID_TOOL_OUTPUT`, `MALFORMED_TOOL_CALL`, `TOOL_STEP_LIMIT_REACHED`.

## Registry (`tools/registry.py`)

Register (reject duplicates), get by exact name, enable/disable, and `enabled_definitions()` /
`enabled_ollama_schemas()` in deterministic (name-sorted) order. Disabled tools are excluded from
the schemas sent to the model and cannot be executed. `default_registry()` registers the two
built-ins. No dynamic discovery.

## Executor (`tools/executor.py`)

`ToolExecutor.execute(ToolCall, step)` checks exists → enabled → validate → execute →
validate-output, records `execution_time_ms`, returns a `ToolResult`, and logs start/complete via
`interaction_log.log_tool_event`. Execution runs in a `ThreadPoolExecutor` with
`future.result(timeout=tool.timeout_seconds)`. No `eval`/`exec`/subprocess/dynamic import. All
failures become structured `ToolResult`s — never raw stack traces.

### Thread-timeout limitation

A Python thread that exceeds its timeout may keep running in the background — Python cannot
forcibly kill it. This is acceptable in Phase 1 because the only tools are bounded and
side-effect-free (`system.echo`, `math.calculate`). We do **not** claim the thread is terminated.

## Built-in tools

- **`system.echo`** — `{"text": "..."}` → `{"echo": "..."}`. Proves the round-trip.
- **`math.calculate`** — `{"expression": "(17 * 23) + 5"}` → `{"expression": ..., "result": 396}`.
  Restricted-AST evaluator (never `eval`). Allowed: `+ - * / **`, unary `+/-`, parentheses,
  int/float literals. Everything else is rejected with `INVALID_ARGUMENTS`. Documented limits:
  expression ≤ 200 chars, AST depth ≤ 20, `|exponent|` ≤ 100, `|value|` ≤ 1e15, division by zero
  rejected.

## Tool-result serialization & native message sequence

The provider-facing tool message is `{"role":"tool", "content": json.dumps(...), "tool_name": name}`.
`content` is always a JSON **string**. `tool_name` is included (verified supported/optional). No
OpenAI `tool_call_id` is sent (verified not required). The exact assistant tool-call message is
appended verbatim before its tool results, preserving order. See `phase1-tool-call-test.md`.

## Maximum-step behavior

`MAX_TOOL_STEPS` (default 5). **Every** requested tool call consumes one step — success, unknown,
disabled, invalid arguments, malformed, timeout, or execution error. When the limit is exceeded:
a `TOOL_STEP_LIMIT_REACHED` tool result is appended, and **exactly one** final `ask_local_raw`
call is made **with the tools array omitted** plus a short instruction telling the model no more
tools are available and to answer with what it has (stating any limitation). An outer hard cap
(`MAX_TOOL_STEPS + 2` iterations) guarantees the loop cannot run indefinitely.

## History trimming

`brain.trim_history` (plain user/assistant pairs) is unchanged and still used by the router and
Claude paths. `brain.trim_history_tool_aware(history, turns)` groups messages into logical turns
by user-message boundaries and trims **whole turns**, so a retained `tool` result is never split
from the assistant `tool_calls` message that produced it, multiple tool results stay attached to
their call, and the final assistant answer stays with its turn. It degrades to plain-pair behavior
for tool-free history.

## ChromaDB exclusion

Tool-call and tool-result messages exist only inside the loop's local `messages` list. They are
never returned to the caller and never appended to the persistent `history` (only the final
assistant text is), so `memory_store` — whose sole writer is the explicit `remember` tool — never
sees them.

## Configuration

Read from the environment with safe defaults (app still starts if unset):

| Variable | Default | Effect |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint (now configurable; same default). |
| `TOOL_CALLING_ENABLED` | `true` | When false, the local path sends no tools (original behavior). |
| `MAX_TOOL_STEPS` | `5` | Max tool calls per turn (every call counts). |
| `DEFAULT_TOOL_TIMEOUT_SECONDS` | `10` | Default per-tool timeout. |

## `interaction_log.py` integration

`log_tool_event(tool_name, call_id, step, status, duration_ms=None, error_code=None)` appends one
JSON line to the same `logs/interactions.jsonl` (not a second logger). It logs only safe metadata —
tool name, internal call id, step number, status (start/complete/rejected/timeout/error),
duration, controlled error code. It never logs arguments, output, secrets, or stack traces.
Normal per-turn `log_turn` logging is unchanged.

## Token-overhead measurement

Same non-tool prompt ("Say hello in one short sentence."), tools off vs on, via the live model:

| Run | Enabled tools | Serialized schema size | prompt_tokens |
|---|---|---|---|
| tools OFF | 0 | — | 28 |
| tools ON | 2 | 682 bytes | 408 |

**Prompt-token overhead from the 2 tool schemas: ≈ 380 tokens/call.** (Ollama expands tool
schemas into template instructions, so the overhead exceeds the raw serialized bytes.) Completion
tokens and latency varied independently of tools because the model is a reasoning model. Keep
descriptions/schemas concise. The code already omits the tools array when tool calling is disabled,
no tools are enabled, or on the final exhaustion call, so non-tool turns pay nothing when
`TOOL_CALLING_ENABLED=false`.

## How to run tests

```
python -m pytest -q            # full suite (deterministic, no live Ollama)
python -m pytest tests/test_tool_loop.py -q
```

## Phase 1 limitations

- Two bounded, side-effect-free tools only.
- Synchronous; thread timeouts don't forcibly kill a runaway thread (safe here).
- Verified against `qwen3.5:397b-cloud` via the local daemon; re-verify if the model/endpoint
  changes.
- No JSON-prompt fallback (native only). Fallback is a possible future enhancement.

## Recommended Phase 2 starting point

Add a new `BaseTool` subclass and register it in `default_registry()` — the registry/executor/loop
need no changes for a new bounded tool. The first Phase 2 candidate is a **read-only, allow-listed**
capability (e.g., a constrained HTTP GET or a scoped file read) with its own permission checks and
timeout, plus per-tool output-schema validation. A JSON-prompt fallback protocol (for models
without native tool calling) and tool-schema caching are the other natural extensions.
