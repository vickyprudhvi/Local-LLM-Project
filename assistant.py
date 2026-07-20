"""Main loop. v2: single-model unified routing + answering to avoid VRAM swap-thrashing."""

import json
import re
import time
from datetime import datetime

from rich.console import Console

import calendar_reader
import eyes
import memory_store
from brain import ask_claude, ask_local, load_system_prompt
from ears import listen_push_to_talk
from interaction_log import log_turn
from router import route_and_answer
from voice import speak

console = Console()

CLAUDE_VISION_RE = re.compile(r"\b(read|text|question|document|screen)\b|ask claude", re.IGNORECASE)


def _use_claude_vision(text):
    return bool(CLAUDE_VISION_RE.search(text))


def _enrich_with_memory(user_text):
    facts = memory_store.recall(user_text, n_results=3)
    if not facts:
        return user_text
    facts_block = "Remembered facts that may be relevant:\n" + "\n".join(f"- {f['text']}" for f in facts)
    return f"{facts_block}\n\n{user_text}"


def dispatch(decision, user_text, prompt, history, system_prompt):
    """Returns (reply, metrics dict). metrics covers only calls made *beyond* routing —
    e.g. an escalation to Claude or a vision model call — since the router's own
    token usage is already on `decision`."""
    if decision.mode == "tool" and decision.tool == "time":
        return datetime.now().strftime("%A, %B %d %Y, %I:%M %p"), {}

    if decision.mode == "tool" and decision.tool == "remember":
        fact_id = memory_store.remember(decision.payload)
        if fact_id is None:
            return "Sorry, I couldn't save that.", {}
        return f"Got it, I'll remember: {decision.payload}", {}

    if decision.mode == "tool" and decision.tool == "recall":
        facts = memory_store.list_all(n_results=10)
        if not facts:
            return "I don't have anything remembered yet.", {}
        return "Here's what I remember: " + "; ".join(f["text"] for f in facts), {}

    if decision.mode == "tool" and decision.tool == "calendar":
        date_args = json.loads(decision.payload) if decision.payload else {}
        try:
            events = calendar_reader.get_events(
                start_date=date_args.get("start_date"), end_date=date_args.get("end_date"), n=10
            )
        except Exception as e:
            return f"Sorry, I couldn't reach your calendar: {e}", {}
        if not events:
            return "Nothing found on your calendar for that range.", {}
        calendar_prompt = "Here's what's on your calendar: " + "; ".join(f"{e['start']} - {e['summary']}" for e in events)
        calendar_summary_prompt = (
            "You're telling someone what's on their calendar, out loud. Summarize the events in a "
            "natural, conversational sentence or two — no bullet points, no markdown, no raw "
            "timestamps. Use plain phrasing for dates and times, like 'today at 2pm' or 'next "
            "Wednesday'. Be concise."
        )
        return ask_local(calendar_prompt, history, system_prompt=calendar_summary_prompt)

    if decision.mode == "tool" and decision.tool == "look":
        try:
            path = eyes.snapshot()
        except RuntimeError as e:
            return f"Sorry, I couldn't use the camera: {e}", {}
        return eyes.describe_local(path, user_text), {}

    if decision.mode == "tool" and decision.tool == "look_carefully":
        try:
            path = eyes.snapshot()
        except RuntimeError as e:
            return f"Sorry, I couldn't use the camera: {e}", {}
        return eyes.describe_claude(path, user_text, history, system_prompt)

    if decision.mode == "tool":
        return f"[{decision.tool} isn't wired up yet — coming in a later phase]", {}

    if decision.mode == "claude":
        return ask_claude(prompt, history, system_prompt)

    # mode == "local" — routing only decided; generate the answer separately
    return ask_local(prompt, history, system_prompt)


def get_user_text(mode):
    if mode == "p":
        text = listen_push_to_talk()
        console.print(f"[dim]heard: {text}[/dim]")
        return text
    return input("> ").strip()


def main():
    system_prompt = load_system_prompt()
    history = []

    console.print("[bold]home-ai (LLM router v2 — consolidated)[/bold]")

    while True:
        mode = input("mode [t=text, p=push-to-talk, q=quit]: ").strip().lower()
        if mode == "q":
            break
        if mode not in ("t", "p"):
            continue

        user_text = get_user_text(mode)
        if not user_text:
            continue

        turn_start = time.perf_counter()

        prompt = _enrich_with_memory(user_text)
        decision = route_and_answer(prompt, history)
        console.print(f"[dim]routing: mode={decision.mode} tool={decision.tool}[/dim]")

        reply, extra_metrics = dispatch(decision, user_text, prompt, history, system_prompt)
        console.print(f"[cyan]{reply}[/cyan]")
        speak(reply)

        total_time_sec = time.perf_counter() - turn_start
        prompt_tokens = (decision.prompt_tokens or 0) + (extra_metrics.get("prompt_tokens") or 0)
        completion_tokens = (decision.completion_tokens or 0) + (extra_metrics.get("completion_tokens") or 0)
        log_turn(
            question=user_text,
            mode=decision.mode,
            tool=decision.tool,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_time_sec=total_time_sec,
        )

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
