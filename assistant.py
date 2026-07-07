"""Main loop. Phase 3: push-to-talk + spoken answers wired in."""

from datetime import datetime

from rich.console import Console

from brain import ask_local, load_system_prompt
from ears import listen_push_to_talk
from router import route_intent
from voice import speak

console = Console()


def dispatch(decision, user_text, history, system_prompt):
    if decision.mode == "tool" and decision.tool == "time":
        return datetime.now().strftime("%A, %B %d %Y, %I:%M %p")
    if decision.mode == "tool":
        return f"[{decision.tool} isn't wired up yet — coming in a later phase]"

    # claude escalation goes live in Phase 5; for now every non-tool route answers locally
    reply, _metrics = ask_local(user_text, history, system_prompt)
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

    console.print("[bold]home-ai (Phase 3)[/bold]")

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
