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
