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


def log_tool_selection(registered_tools, shortlisted_tools, selection_prompt_size,
                       prompt_eval_count, completion_tokens):
    """Append one JSON line describing the Phase B tool-selection step of a local turn.

    Records only safe telemetry: the count of registered (candidate) tools, the
    shortlisted tool NAMES (not sensitive), the selection prompt size in chars, and
    the selection call's token counts. Lets prompt growth be tracked as the registry
    grows — the whole point of the selection budget.
    """
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": "tool_selection",
        "registered_tools": registered_tools,
        "shortlisted_tools": shortlisted_tools,
        "selection_prompt_size": selection_prompt_size,
        "prompt_eval_count": prompt_eval_count,
        "completion_tokens": completion_tokens,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


_MCP_LOG_ALLOWED_KEYS = (
    "catalog_id", "server_id", "package_version", "plan_id", "plan_hash",
    "approval_result", "state", "installation_result", "validation_result",
    "discovered_tool_count", "registered_tool_count", "denied_tool_count",
    "previous_version", "approved_directory_count", "environment_variable_names",
)


def log_mcp_event(action, error_code=None, **fields):
    """Append one JSON line describing a Phase F MCP provisioning/lifecycle action.

    Only an allowlist of safe metadata is recorded (ids, versions, counts, hashes,
    state, result). Secret values, environment values, raw package output, full
    file contents, and conversation text are never logged — anything not on the
    allowlist is dropped.
    """
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": "mcp_management",
        "action": action,
        "error_code": error_code,
    }
    for key, value in (fields or {}).items():
        if key in _MCP_LOG_ALLOWED_KEYS:
            record[key] = value
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
