"""LLM-based routing only. route_and_answer(...) -> RouteDecision; callers run
ask_local() separately for mode == "local".

Previously this call also generated the final answer itself when no tool applied,
to avoid a second round trip — needed back when ROUTER_MODEL and LOCAL_MODEL were
both local models competing for the same 4GB GPU. Now that LOCAL_MODEL is an
Ollama Cloud model, that VRAM constraint no longer applies, so routing stays a
pure classification step: the model must always call a tool, including
answer_locally for an ordinary chat turn, so this call never spends tokens
generating prose that would just be thrown away.

Deliberately uses its own ROUTER_SYSTEM_PROMPT, not system_prompt.txt (the
persona prompt passed to ask_local/ask_claude) — classification shouldn't
inherit persona instructions ("be concise", memory rules, etc.) that have
nothing to do with picking a tool, and shouldn't change behavior just because
the persona prompt gets tweaked.

Not pure — makes an Ollama tool-calling call. Falls back to mode="local" on any
failure (timeout, malformed response, unknown tool), per the same rule as every
other Ollama/Anthropic call; the actual error message comes from ask_local's own
failure path when the caller invokes it.
"""

import os
from dataclasses import dataclass
from typing import Optional

import requests
from dotenv import load_dotenv
from rich.console import Console

from brain import trim_history

load_dotenv()
console = Console()

OLLAMA_URL = "http://localhost:11434"
ROUTER_MODEL = os.environ.get("LOCAL_MODEL", "qwen3:30b-a3b")
ROUTER_TIMEOUT = 180

ROUTER_SYSTEM_PROMPT = (
    "You are the routing layer for a personal assistant. Your only job is to pick the right tool "
    "for the user's message — you do not answer questions, hold a persona, or explain your choice.\n\n"
    "For every message, call exactly one tool: pick the one whose description matches, or call "
    "answer_locally for an ordinary conversational turn that doesn't match any other tool. If the "
    "user explicitly says to use Claude or use the local model, respect that override regardless of "
    "topic. Always call a tool — never answer directly in this response."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "answer_locally",
            "description": "Handle an ordinary conversational turn — call this when no other tool applies and escalation to Claude isn't needed.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save a fact the user wants remembered for later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to remember, cleaned up, with no leading phrases like 'remember' or 'remember that'.",
                    }
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Look up what has been remembered about the user. Call this when the user asks what you remember or know about them.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look",
            "description": "Take a photo with the camera and describe what is in it. Call this when the user asks what you see, to look at something, or to use the camera.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_claude",
            "description": (
                "Escalate to Claude. Always call this for money, investing, tax, mortgages, insurance, "
                "medical/health, legal, resume/salary/career questions, or anything requiring careful/deep "
                "reasoning or high accuracy — even if you think you know the answer, do not answer these yourself."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_TOOL_NAME_MAP = {
    "remember": "remember",
    "recall": "recall",
    "get_time": "time",
    "look": "look",
}


@dataclass(frozen=True)
class RouteDecision:
    mode: str  # "tool" | "claude" | "local"
    tool: Optional[str] = None
    payload: Optional[str] = None
    prompt_tokens: Optional[int] = None  # from the routing call to ROUTER_MODEL
    completion_tokens: Optional[int] = None


def route_and_answer(text: str, history) -> RouteDecision:
    stripped = text.strip()
    trimmed = trim_history(history, 12)
    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
    messages.extend(trimmed)
    messages.append({"role": "user", "content": stripped})

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": ROUTER_MODEL,
                "messages": messages,
                "tools": TOOLS,
                "stream": False,
                "keep_alive": -1,
            },
            timeout=ROUTER_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Local model call failed: {e}[/red]")
        return RouteDecision(mode="local", payload=stripped)

    data = resp.json()
    prompt_tokens = data.get("prompt_eval_count")
    completion_tokens = data.get("eval_count")
    message = data.get("message", {})
    tool_calls = message.get("tool_calls") or []

    if not tool_calls:
        console.print("[red]Router didn't call a tool as instructed, falling back to local.[/red]")
        return RouteDecision(
            mode="local", payload=stripped,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )

    call = tool_calls[0].get("function", {})
    name = call.get("name")
    args = call.get("arguments") or {}

    if name == "answer_locally":
        return RouteDecision(
            mode="local", payload=stripped,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )

    if name == "escalate_to_claude":
        return RouteDecision(
            mode="claude", payload=stripped,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )

    tool = _TOOL_NAME_MAP.get(name)
    if tool is None:
        console.print(f"[red]Router returned unknown tool '{name}', falling back to local.[/red]")
        return RouteDecision(
            mode="local", payload=stripped,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )

    payload = args.get("fact", stripped) if tool == "remember" else stripped
    return RouteDecision(
        mode="tool", tool=tool, payload=payload,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )
