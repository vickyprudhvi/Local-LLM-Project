"""Main loop. Phase 2: router wired in, routing decisions printed each turn."""

from datetime import datetime

from rich.console import Console

from brain import ask_local, load_system_prompt
from router import route_intent

console = Console()


def dispatch(decision, user_text, history, system_prompt):
    if decision.mode == "tool" and decision.tool == "time":
        return datetime.now().strftime("%A, %B %d %Y, %I:%M %p")
    if decision.mode == "tool":
        return f"[{decision.tool} isn't wired up yet — coming in a later phase]"

    # claude escalation goes live in Phase 5; for now every non-tool route answers locally
    reply, _metrics = ask_local(user_text, history, system_prompt)
    return reply


def main():
    system_prompt = load_system_prompt()
    history = []

    console.print("[bold]home-ai — text loop (Phase 2)[/bold]  (type 'q' to quit)")

    while True:
        user_text = input("> ").strip()
        if not user_text:
            continue
        if user_text.lower() in ("q", "quit", "exit"):
            break

        decision = route_intent(user_text)
        console.print(f"[dim]routing: mode={decision.mode} tool={decision.tool}[/dim]")

        reply = dispatch(decision, user_text, history, system_prompt)
        console.print(f"[cyan]{reply}[/cyan]")

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
