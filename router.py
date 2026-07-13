"""LLM-based routing + answering, unified into one call to LOCAL_MODEL. route_and_answer(...) -> RouteDecision.

Originally used a separate tiny model (qwen3:4b) just for routing, with a second
call to the big model for the actual answer. On this machine's 4GB GPU, those two
models can't both stay resident — every turn evicted one to load the other, which
was slow enough to blow past ask_local's timeout. Consolidated to one model: the
same tool-calling call that decides whether a tool applies also produces the final
answer when no tool is needed, so only one model is ever loaded and local chat
turns need no second round trip.

Not pure — makes an Ollama tool-calling call. Falls back to mode="local" with a
readable error message on any failure (timeout, malformed response, unknown tool),
per the same rule as every other Ollama/Anthropic call.
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

ROUTER_INSTRUCTIONS = (
    "\n\nYou also have tools available. For every message, call exactly one tool if it matches a "
    "tool description below; otherwise just answer normally, in character. If the user explicitly "
    "says to use Claude or use the local model, respect that override regardless of topic. Do not "
    "explain your tool choice — either call the tool, or just answer."
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
    answer: Optional[str] = None  # pre-generated answer text, populated when mode == "local"
    prompt_tokens: Optional[int] = None  # from the routing/answering call to ROUTER_MODEL
    completion_tokens: Optional[int] = None


def route_and_answer(text: str, history, system_prompt: str) -> RouteDecision:
    stripped = text.strip()
    trimmed = trim_history(history, 12)
    messages = [{"role": "system", "content": system_prompt + ROUTER_INSTRUCTIONS}]
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
        return RouteDecision(mode="local", payload=stripped, answer="Sorry, I couldn't reach the local model just now.")

    data = resp.json()
    prompt_tokens = data.get("prompt_eval_count")
    completion_tokens = data.get("eval_count")
    message = data.get("message", {})
    tool_calls = message.get("tool_calls") or []

    if not tool_calls:
        answer = (message.get("content") or "").strip()
        return RouteDecision(
            mode="local", payload=stripped, answer=answer,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )

    call = tool_calls[0].get("function", {})
    name = call.get("name")
    args = call.get("arguments") or {}

    if name == "escalate_to_claude":
        return RouteDecision(
            mode="claude", payload=stripped,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )

    tool = _TOOL_NAME_MAP.get(name)
    if tool is None:
        console.print(f"[red]Router returned unknown tool '{name}', falling back to local.[/red]")
        return RouteDecision(
            mode="local", payload=stripped, answer="",
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )

    payload = args.get("fact", stripped) if tool == "remember" else stripped
    return RouteDecision(
        mode="tool", tool=tool, payload=payload,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )
