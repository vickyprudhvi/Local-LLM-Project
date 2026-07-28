# Phase 1 — Native Ollama Tool-Calling Verification

This records the live verification performed **before** implementing the tool protocol, using
the exact same transport as `brain.ask_local` (same `OLLAMA_URL`, model, `stream=False`,
`keep_alive="10m"`, `timeout=120`, and **no auth headers**). The verification script is not
committed; it lives outside the repo.

## Configuration tested

- **Model:** `qwen3.5:397b-cloud` (the live `LOCAL_MODEL`).
- **Endpoint category:** local Ollama daemon on `localhost` (a cloud-suffixed model served
  through the local daemon). No authentication headers were sent — same as the existing
  `ask_local`/`router.py` code. (Actual URL/token are not reproduced here.)
- **Request:** `POST {OLLAMA_URL}/api/chat` with `stream=False`, `keep_alive="10m"`, and one
  `math.calculate` tool definition (`{"type":"function","function":{name,description,parameters}}`,
  ~296 bytes serialized).

## Reliability

The arithmetic prompt "Use the calculator to compute (17 * 23) + 5" was sent **4/4 times** with
tools attached; the model returned a `math.calculate` tool call **every time** (0 direct answers).
A two-operation prompt returned **2 tool calls in a single assistant message on 3/3 runs**.
Native tool calling is reliable for this model → **native protocol only, no JSON fallback.**

## Sanitized request payload

```json
{
  "model": "qwen3.5:397b-cloud",
  "messages": [{"role": "user", "content": "Use the calculator to compute (17 * 23) + 5. Call the math.calculate tool."}],
  "stream": false,
  "keep_alive": "10m",
  "tools": [
    {"type": "function",
     "function": {"name": "math.calculate",
                  "description": "Evaluate a basic arithmetic expression (+ - * / ** and parentheses).",
                  "parameters": {"type": "object",
                                 "properties": {"expression": {"type": "string"}},
                                 "required": ["expression"]}}}
  ]
}
```

## Sanitized response — assistant tool-call message

```json
{
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "id": "call_8f554d7628a14c74abb4cad6",
        "function": {
          "name": "math.calculate",
          "arguments": {"expression": "(17 * 23) + 5"}
        }
      }
    ]
  }
}
```

### Confirmed structure

| Question | Answer |
|---|---|
| Is `message.tool_calls` returned? | **Yes**, reliably (4/4). |
| Location of the function name | `tool_calls[i]["function"]["name"]` |
| Structure of arguments | `tool_calls[i]["function"]["arguments"]` |
| Arguments: dict or encoded string? | **dict** (a JSON object, not a string) |
| Call identifier | `tool_calls[i]["id"]`, a string like `"call_8f55…"` |
| Keys present on each tool_call | `["function", "id"]` |
| Multiple tool calls in one message? | **Yes** — 2 calls returned on 3/3 runs |
| Does it ever answer directly instead? | Not observed for a clear arithmetic prompt (0/4) |
| Run-to-run consistency | Consistent across all runs |

## Verified tool-result message format

A follow-up call including the assistant tool-call message plus a tool-result message produced a
correct final answer ("The result of (17 * 23) + 5 is 396.").

- **Role:** `role = "tool"`.
- **Content:** must be a **string** — we send `json.dumps(...)` of the `ToolResult`.
- **`tool_name`:** **supported and optional.** The round-trip succeeded both with and without a
  `tool_name` field. We include it (harmless, and it helps the model associate the result).
- **OpenAI-style `tool_call_id`:** **not used / not required.** The round-trip worked without any
  `tool_call_id` matching. We do **not** send it.
- **Sequencing:** tool results are appended **after** the exact assistant tool-call message, in
  order, before the next model call. The assistant message is preserved verbatim (including its
  `id`s), so the sequence stays well-formed.

### Tool-result message we send

```json
{
  "role": "tool",
  "content": "{\"success\": true, \"tool_name\": \"math.calculate\", \"call_id\": \"call_8f55…\", \"data\": {\"expression\": \"(17 * 23) + 5\", \"result\": 396}, \"error\": null}",
  "tool_name": "math.calculate"
}
```

## Limitations of this verification

- Verified against one model (`qwen3.5:397b-cloud`) via the local daemon. A different model/tag or
  a direct `https://ollama.com` cloud endpoint (with `Authorization: Bearer`) may differ; re-verify
  if `LOCAL_MODEL` or `OLLAMA_URL` changes.
- Reliability was sampled over a handful of runs, not exhaustively.
- The model is a reasoning model, so `completion_tokens` and latency vary run-to-run independently
  of tool use.
