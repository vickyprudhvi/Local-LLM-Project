"""Main loop. Phase 7: system_prompt polish."""

import re
from datetime import datetime

from rich.console import Console

import eyes
import memory_store
from brain import ask_claude, ask_local, load_system_prompt
from ears import listen_push_to_talk
from router import route_intent
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


def dispatch(decision, user_text, history, system_prompt):
    if decision.mode == "tool" and decision.tool == "time":
        return datetime.now().strftime("%A, %B %d %Y, %I:%M %p")

    if decision.mode == "tool" and decision.tool == "remember":
        fact_id = memory_store.remember(decision.payload)
        if fact_id is None:
            return "Sorry, I couldn't save that."
        return f"Got it, I'll remember: {decision.payload}"

    if decision.mode == "tool" and decision.tool == "recall":
        facts = memory_store.recall(user_text, n_results=3)
        if not facts:
            return "I don't have anything remembered yet."
        return "Here's what I remember: " + "; ".join(f["text"] for f in facts)

    if decision.mode == "tool" and decision.tool == "look":
        try:
            path = eyes.snapshot()
        except RuntimeError as e:
            return f"Sorry, I couldn't use the camera: {e}"

        if _use_claude_vision(user_text):
            return eyes.describe_claude(path, user_text, history, system_prompt)
        return eyes.describe_local(path, user_text)

    if decision.mode == "tool":
        return f"[{decision.tool} isn't wired up yet — coming in a later phase]"

    prompt = _enrich_with_memory(user_text)
    if decision.mode == "claude":
        return ask_claude(prompt, history, system_prompt)

    reply, _metrics = ask_local(prompt, history, system_prompt)
    return reply


def get_user_text(mode):
    if mode == "p":
        text = listen_push_to_talk()
        console.print(f"[dim]heard: {text}[/dim]")
        return text
    return input("> ").strip()


def main():
    system_prompt = load_system_prompt()
    history = []

    console.print("[bold]home-ai (Phase 7)[/bold]")

    while True:
        mode = input("mode [t=text, p=push-to-talk, q=quit]: ").strip().lower()
        if mode == "q":
            break
        if mode not in ("t", "p"):
            continue

        user_text = get_user_text(mode)
        if not user_text:
            continue

        decision = route_intent(user_text)
        console.print(f"[dim]routing: mode={decision.mode} tool={decision.tool}[/dim]")

        reply = dispatch(decision, user_text, history, system_prompt)
        console.print(f"[cyan]{reply}[/cyan]")
        speak(reply)

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
