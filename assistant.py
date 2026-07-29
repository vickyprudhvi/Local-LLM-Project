"""Main loop. v2: single-model unified routing + answering to avoid VRAM swap-thrashing.

Phase A: assistant.py is a generic orchestration layer only. It routes a turn
(Claude / local / tool), and for a tool turn hands the RouteDecision to
tool_dispatch, which runs the selected built-in through the shared ToolRegistry /
ToolExecutor and renders the result. assistant.py knows nothing about how any
specific tool works — there are no per-tool branches here.
"""

import time

from rich.console import Console

import mcp_layer
import memory_store
import tool_dispatch
import tool_loop
from brain import ask_claude, load_system_prompt
from ears import listen_push_to_talk
from interaction_log import log_turn
from mcp_layer import McpError
from router import route_and_answer
from voice import speak

console = Console()


def _start_mcp():
    """Start the configured external MCP server and register its tools (Phase E).

    Generic subsystem bootstrap — not a per-tool branch. Reads config/mcp_server.json
    (MCP disabled by default). An optional server (`required: false`) that fails to
    start is logged and skipped so built-in tools keep working. Returns an McpSession.
    """
    try:
        session = mcp_layer.bootstrap_from_config(tool_loop.REGISTRY)
    except McpError as e:
        # Only a `required` server reaches here; keep startup clean and continue.
        console.print(f"[yellow]Required MCP server failed to start ({e.code}).[/yellow]")
        return None

    health = session.health
    if health is not None and health.state.value == "healthy":
        console.print(f"[dim]MCP '{health.server_id}': discovered={health.discovered_tool_count} "
                      f"registered={health.registered_tool_count} denied={health.denied_tool_count} "
                      f"skipped={health.skipped_tool_count} disabled={health.disabled_tool_count} — "
                      f"{', '.join(session.tool_names())}[/dim]")
        # Sanitized diagnostics: tool names + skip reasons only (never secrets/args).
        for name, reason, category in health.diagnostics:
            console.print(f"[dim]  MCP tool '{name}' {category}: {reason}[/dim]")
    elif health is not None and health.state.value == "failed":
        console.print(f"[yellow]MCP server unavailable ({health.last_error_code}); "
                      f"continuing without MCP tools.[/yellow]")
    else:
        console.print("[dim]MCP: disabled.[/dim]")
    return session


def _enrich_with_memory(user_text):
    facts = memory_store.recall(user_text, n_results=3)
    if not facts:
        return user_text
    facts_block = "Remembered facts that may be relevant:\n" + "\n".join(f"- {f['text']}" for f in facts)
    return f"{facts_block}\n\n{user_text}"


def dispatch(decision, user_text, prompt, history, system_prompt):
    """Generic orchestration: turn a RouteDecision into (reply, metrics dict).

    metrics covers only calls made *beyond* routing — e.g. an escalation to Claude,
    a vision-model call, or a tool's finishing summarization — since the router's
    own token usage is already on `decision`.

    There are no tool-specific branches here: the only decision is which of the
    three routes to take. Tool-mode requests are executed through the shared
    ToolRegistry / ToolExecutor and rendered by tool_dispatch, which is the single
    place that knows how a built-in's result becomes a spoken answer.
    """
    if decision.mode == "claude":
        return ask_claude(prompt, history, system_prompt)

    if decision.mode == "tool":
        return tool_dispatch.execute_and_render(decision, user_text, history, system_prompt)

    # mode == "local" (and any unexpected mode) — routing only decided; generate the
    # answer separately. The local tool loop lets the model call its LLM-selectable
    # tools and then writes the final answer itself. With TOOL_CALLING_ENABLED=false
    # it falls back to the original single-shot ask_local behavior.
    return tool_loop.run_local_tool_loop(prompt, history, system_prompt)


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
    mcp_session = _start_mcp()

    try:
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
    finally:
        if mcp_session is not None:
            mcp_session.shutdown()


if __name__ == "__main__":
    main()
