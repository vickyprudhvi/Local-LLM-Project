"""LLM-based intent routing via LOCAL_MODEL_TINY (qwen3:4b). route_intent(text) -> RouteDecision.

Not pure — makes an Ollama tool-calling call. Falls back to mode="local" on any
failure (timeout, malformed response, unknown tool name), per the same
readable-error-instead-of-crash rule as every other Ollama/Anthropic call.
"""

import os
from dataclasses import dataclass
from typing import Optional

import requests
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

OLLAMA_URL = "http://localhost:11434"
ROUTER_MODEL = os.environ.get("LOCAL_MODEL_TINY", "qwen3:4b")
ROUTER_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a strict intent router for a personal voice assistant. For every message, decide whether to "
    "call exactly one tool, or respond normally as plain conversation if none apply. If the user explicitly "
    "says to use Claude or use the local model, respect that override regardless of topic. Do not explain "
    "your choice."
)

TOOLS = [
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


def route_intent(text: str) -> RouteDecision:
    stripped = text.strip()

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": ROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": stripped},
                ],
                "tools": TOOLS,
                "stream": False,
                "keep_alive": "10m",
            },
            timeout=ROUTER_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Router model call failed, falling back to local: {e}[/red]")
        return RouteDecision(mode="local", payload=stripped)

    message = resp.json().get("message", {})
    tool_calls = message.get("tool_calls") or []

    if not tool_calls:
        return RouteDecision(mode="local", payload=stripped)

    call = tool_calls[0].get("function", {})
    name = call.get("name")
    args = call.get("arguments") or {}

    if name == "escalate_to_claude":
        return RouteDecision(mode="claude", payload=stripped)

    tool = _TOOL_NAME_MAP.get(name)
    if tool is None:
        console.print(f"[red]Router returned unknown tool '{name}', falling back to local.[/red]")
        return RouteDecision(mode="local", payload=stripped)

    payload = args.get("fact", stripped) if tool == "remember" else stripped
    return RouteDecision(mode="tool", tool=tool, payload=payload)
