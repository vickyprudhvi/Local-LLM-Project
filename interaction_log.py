"""Structured logging of each user turn: routing decision, token usage, and timing.

Appends one JSON object per line to logs/interactions.jsonl so usage can be
grepped/analyzed later without parsing rich console output.
"""

import json
import os
from datetime import datetime

LOG_PATH = os.environ.get("INTERACTION_LOG_PATH", "logs/interactions.jsonl")


def log_turn(question, mode, tool, prompt_tokens, completion_tokens, total_time_sec):
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)

    total_tokens = None
    if prompt_tokens is not None or completion_tokens is not None:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "mode": mode,
        "tool": tool,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "total_time_sec": round(total_time_sec, 3),
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def log_tool_event(tool_name, call_id, step, status, duration_ms=None, error_code=None, extra=None):
    """Append one JSON line describing a single tool-execution event.

    Logs ONLY safe execution metadata — never tool arguments, tool output,
    secrets, or stack traces. `status` is one of: start, complete, rejected,
    timeout, error. `extra` is an optional dict of safe, non-content metadata
    (e.g. http_status_category, bytes_read, result_count, rate_limit_remaining);
    a small allowlist of known-sensitive keys is dropped defensively.
    """
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": "tool",
        "tool": tool_name,
        "call_id": call_id,
        "step": step,
        "status": status,
        "duration_ms": round(duration_ms, 3) if duration_ms is not None else None,
        "error_code": error_code,
    }
    if extra:
        _REDACT = {"token", "authorization", "api_key", "apikey", "password", "cookie",
                   "secret", "query", "url", "text", "content"}
        for key, value in extra.items():
            if key.lower() in _REDACT:
                continue
            record[key] = value
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record
