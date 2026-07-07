"""Main loop. Phase 1: text-only, local model, conversation history."""

from rich.console import Console

from brain import ask_local, load_system_prompt

console = Console()


def main():
    system_prompt = load_system_prompt()
    history = []

    console.print("[bold]home-ai — text loop (Phase 1)[/bold]  (type 'q' to quit)")

    while True:
        user_text = input("> ").strip()
        if not user_text:
            continue
        if user_text.lower() in ("q", "quit", "exit"):
            break

        reply, _metrics = ask_local(user_text, history, system_prompt)
        console.print(f"[cyan]{reply}[/cyan]")

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
