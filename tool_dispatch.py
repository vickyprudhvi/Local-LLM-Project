"""Single dispatch path for router-selected built-in tools (Phase A).

assistant.py no longer knows how any specific tool works. When the router picks a
tool (RouteDecision.mode == "tool"), assistant hands the decision here, and this
module:

  1. maps the router's short tool name to the registered tool + parses its payload
     into an arguments dict,
  2. executes it through the shared ToolExecutor (the same registry/executor the
     local tool-calling loop uses — one source of truth), and
  3. renders the resulting ToolResult into a spoken (reply, metrics) pair.

The render step dispatches on a small, generic response PROTOCOL (the "render"
key a tool puts in its result data) — never on the tool's identity. Adding a new
built-in reuses an existing directive without changing this file. The vision- and
summarization-LLM calls live here (not inside the tools) because they need the
conversation's user_text / history / system_prompt, which tools deliberately
never see.

Token metrics come only from the finishing LLM calls (summarize / describe /
synthesize), exactly as the former assistant.dispatch branches reported them; a
tool that speaks directly contributes no metrics.
"""

import json
import uuid

import eyes
import tool_loop
from brain import ask_local
from tools.models import ToolCall

# Router short-name -> (registered tool name, payload -> arguments dict).
# The payload encodings mirror what router.py produces today; this is the one
# place that bridges the router's naming/payload convention to the registry.


def _args_fact(payload):
    return {"fact": payload if payload is not None else ""}


def _args_json(payload):
    return json.loads(payload) if payload else {}


def _args_none(payload):
    return {}


_ROUTE = {
    "remember": ("memory.remember", _args_fact),
    "recall": ("memory.recall", _args_json),
    "time": ("system.time", _args_none),
    "look": ("camera.look", _args_none),
    "look_carefully": ("camera.look_carefully", _args_none),
    "capture_camera": ("camera.capture", _args_json),
    "scan_room": ("camera.scan", _args_none),
    "calendar": ("calendar.read", _args_json),
}


def execute_and_render(decision, user_text, history, system_prompt):
    """Execute the router-selected tool and render its result. Returns (reply, metrics)."""
    entry = _ROUTE.get(decision.tool)
    if entry is None:
        # Defensive: a routed tool with no registered mapping (kept from the old
        # dispatch's "coming in a later phase" fallback).
        return f"[{decision.tool} isn't wired up yet — coming in a later phase]", {}

    tool_name, parse = entry
    try:
        arguments = parse(decision.payload)
    except (ValueError, TypeError):
        arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}

    call = ToolCall(call_id=f"call_{uuid.uuid4().hex[:16]}", tool_name=tool_name, arguments=arguments)
    result = tool_loop.EXECUTOR.execute(call)
    return _render(result, user_text, history, system_prompt)


def _render(result, user_text, history, system_prompt):
    """Turn a ToolResult into (reply, metrics) via the generic render protocol."""
    if not result.success:
        # Domain errors are already returned by tools as a "speak" directive on a
        # successful result; reaching here means an unexpected fault (timeout/bug).
        return "Sorry, I couldn't do that just now.", {}

    data = result.data
    kind = data.get("render")

    if kind == "speak":
        return data.get("text", ""), {}

    if kind == "summarize":
        return ask_local(data["content"], history, system_prompt=data["instructions"])

    if kind == "describe_local":
        if "max_side" in data:
            return eyes.describe_local(data["image_path"], user_text, max_side=data["max_side"])
        return eyes.describe_local(data["image_path"], user_text)

    if kind == "describe_claude":
        return eyes.describe_claude(data["image_path"], user_text, history, system_prompt)

    if kind == "scan_synthesize":
        return _synthesize_scan(data["images"], user_text, history, system_prompt)

    return "Sorry, I couldn't complete that.", {}


def _synthesize_scan(images, user_text, history, system_prompt):
    """Describe each scan frame separately, then synthesize one answer.

    Reproduces the former assistant.dispatch scan_room fallback exactly, including
    the summed token accounting across the per-frame vision calls and the final
    synthesis call.
    """
    prompt_tokens = 0
    completion_tokens = 0
    descriptions = []
    for image in images:
        desc, metrics = eyes.describe_local(image["image_path"], user_text)
        descriptions.append(f"{image['position']}: {desc}")
        prompt_tokens += metrics.get("prompt_tokens") or 0
        completion_tokens += metrics.get("completion_tokens") or 0

    synth_prompt = (
        "Here's what was seen across five views of the same room (center, and panned up, "
        "down, left, and right):\n\n" + "\n".join(descriptions)
    )
    synth_system_prompt = (
        f'The user asked: "{user_text}". You\'re answering them out loud, based on five '
        "partial camera views (center, up, down, left, right) of the same room. Answer their "
        "actual question naturally and concisely using whichever view(s) are relevant — don't "
        "mention the view labels or that there were multiple photos. If what they asked about "
        "isn't visible in any view, say so plainly."
    )
    summary, synth_metrics = ask_local(synth_prompt, history, system_prompt=synth_system_prompt)
    prompt_tokens += synth_metrics.get("prompt_tokens") or 0
    completion_tokens += synth_metrics.get("completion_tokens") or 0
    return summary, {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
