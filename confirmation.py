"""Phase C — collecting a user's yes/no for write-class tool calls.

Deliberately separate from ToolExecutor: the executor only DECIDES that a write
tool needs confirmation (returns TOOL_CONFIRMATION_REQUIRED with a deterministic
action_summary) and VALIDATES a supplied confirmation. This module COLLECTS the
decision from the user and re-invokes the executor with a matching, single-use
ToolConfirmation.

Consequences of this split:
  - Direct executor callers cannot accidentally run a write tool (no confirmation
    => TOOL_CONFIRMATION_REQUIRED, handler never runs).
  - Tests never block on input(): they inject a `confirmer`.
  - The confirmation is bound to the exact tool + arguments (hash), so a stale or
    unrelated 'yes' cannot approve a different action.

The default answer is NO. Only 'y'/'yes' (case-insensitive) approve. The summary
shown to the user is application/tool text — never model output or repo content.
"""

from rich.console import Console

from tools.models import (
    TOOL_CONFIRMATION_REQUIRED,
    ToolConfirmation,
    hash_arguments,
)

console = Console()

_APPROVALS = {"y", "yes"}


def confirm_action(summary: str) -> bool:
    """Print the deterministic action summary and read a y/N decision. Default: no."""
    console.print(f"[yellow]{summary}[/yellow]")
    console.print(r"[yellow]Proceed? \[yes/No][/yellow]")
    try:
        answer = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in _APPROVALS


def resolve_with_confirmation(executor, call, step: int = 0, confirmer=None):
    """Execute `call`, collecting confirmation if the executor requires it.

    Returns the final ToolResult. For a read tool this is a single execute(). For a
    write tool: the first execute() returns TOOL_CONFIRMATION_REQUIRED (handler NOT
    run), we ask the user, then re-run with a matching ToolConfirmation — which the
    executor validates before allowing (or, on decline, rejects without running).

    `confirmer` is resolved at call time (default: confirm_action) so it stays
    patchable in tests and callers never block on input() unexpectedly.
    """
    if confirmer is None:
        confirmer = confirm_action
    result = executor.execute(call, step=step)
    if not (result.error and result.error.code == TOOL_CONFIRMATION_REQUIRED):
        return result

    summary = (result.error.details or {}).get("action_summary") or f"Run '{call.tool_name}'."
    approved = bool(confirmer(summary))
    confirmation = ToolConfirmation(
        approved=approved,
        tool_name=call.tool_name,
        arguments_hash=hash_arguments(call.arguments),
    )
    return executor.execute(call, step=step, confirmation=confirmation)
