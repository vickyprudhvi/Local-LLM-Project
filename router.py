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

import json
import os
from dataclasses import dataclass
from datetime import datetime
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

ROUTER_SYSTEM_PROMPT_TEMPLATE = (
    "You are the routing layer for a personal assistant. Your only job is to pick the right tool "
    "for the user's message — you do not answer questions, hold a persona, or explain your choice.\n\n"
    "For every message, call exactly one tool: pick the one whose description matches, or call "
    "answer_locally for an ordinary conversational turn that doesn't match any other tool. If the "
    "user explicitly says to use Claude or use the local model, respect that override regardless of "
    "topic. Always call a tool — never answer directly in this response.\n\n"
    "Today's date is {today} ({weekday}). Use it to resolve any relative dates a tool needs."
)


def _router_system_prompt():
    now = datetime.now()
    return ROUTER_SYSTEM_PROMPT_TEMPLATE.format(today=now.strftime("%Y-%m-%d"), weekday=now.strftime("%A"))

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
            "description": (
                "Look up what has been remembered about the user. Call this when the user asks what "
                "you remember or know about them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": (
                            "The specific subject to recall, e.g. 'dentist appointment' for 'recall my "
                            "dentist appointment'. Omit entirely for a generic request with no specific "
                            "subject, like 'what do you remember about me' or 'what do you know about me'."
                        ),
                    }
                },
            },
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
            "name": "look_carefully",
            "description": (
                "Take a photo and send it to Claude for careful, accurate reading. Use "
                "whenever the user wants text read accurately, or asks you to answer a "
                "question, multiple choice question, form, receipt, document, chart, or "
                "diagram visible on screen or on paper — including follow-ups like 'what's "
                "the correct answer' after a photo was just discussed. Always prefer this "
                "over 'look' when accuracy matters or there is text to read."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_camera",
            "description": (
                "Capture a still frame from a networked room camera (the Tapo IP camera, e.g. the "
                "'office' camera) and describe it. Call this when the user refers to a named or fixed "
                "room camera — 'look through the office camera', 'capture the office camera', 'check "
                "the room', 'what is on my desk'. This is the fixed IP camera, distinct from 'look', "
                "which uses the laptop's own webcam; prefer 'look' when the user just says 'take a "
                "picture' with no room/camera name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "camera_name": {
                        "type": "string",
                        "description": "Name of the camera to capture, e.g. 'office'. Omit for the default camera.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_room",
            "description": (
                "Pan the room camera left/right and tilt it up/down to survey the whole room, "
                "capturing and describing multiple views, then answer the user's question using "
                "them. Call this for 'look at the complete room', 'scan the room', 'look around', "
                "'show me the whole room' — anything asking for more than what a single fixed "
                "camera view shows. This physically moves the camera; prefer 'capture_camera' for "
                "a single still shot."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar_events",
            "description": (
                "Look up events on the user's Google Calendar. Call this when the user asks what's on "
                "their calendar, what they have coming up, or about a past appointment. Resolve any "
                "relative date the user gives ('today', 'next Tuesday', 'June 15') into a concrete "
                "YYYY-MM-DD date yourself, using the current date given in your instructions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": (
                            "Earliest date to include, YYYY-MM-DD. Omit for an open-ended future query "
                            "('upcoming') — this is the default when the user gives no date. For a single "
                            "day, set start_date and end_date to the same date. For 'everything, past and "
                            "future', set start_date to several years ago and leave end_date empty."
                        ),
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Latest date to include, YYYY-MM-DD (inclusive). Omit for no upper bound.",
                    },
                },
            },
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
    "look_carefully": "look_carefully",
    "capture_camera": "capture_camera",
    "scan_room": "scan_room",
    "get_calendar_events": "calendar",
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
    messages = [{"role": "system", "content": _router_system_prompt()}]
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

    if tool == "remember":
        payload = args.get("fact", stripped)
    elif tool == "calendar":
        # start_date/end_date (YYYY-MM-DD), resolved by the router LLM itself; both optional.
        payload = json.dumps({"start_date": args.get("start_date"), "end_date": args.get("end_date")})
    elif tool == "recall":
        # topic is optional: empty/omitted means a generic "what do you remember" style request.
        payload = json.dumps({"topic": args.get("topic")})
    elif tool == "capture_camera":
        # camera_name is optional; None means the default configured camera.
        payload = json.dumps({"camera_name": args.get("camera_name")})
    else:
        payload = stripped
    return RouteDecision(
        mode="tool", tool=tool, payload=payload,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )
