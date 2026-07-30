"""Main loop. v2: single-model unified routing + answering to avoid VRAM swap-thrashing.

Phase A: assistant.py is a generic orchestration layer only. It routes a turn
(Claude / local / tool), and for a tool turn hands the RouteDecision to
tool_dispatch, which runs the selected built-in through the shared ToolRegistry /
ToolExecutor and renders the result. assistant.py knows nothing about how any
specific tool works — there are no per-tool branches here.
"""

import time
from dataclasses import dataclass
from typing import Optional

from rich.console import Console

import mcp_layer
import memory_store
import tool_dispatch
import tool_loop
import tools.config as app_config
from brain import ask_claude, load_system_prompt
from ears import listen_push_to_talk
from interaction_log import log_turn
from mcp_layer import ActiveMcpRuntime, McpError, McpRuntimeManager
from mcp_layer.config_resolver import resolve_config
from mcp_management.access_classifier import classify_outside_root_failure
from mcp_management.approval import confirm_provisioning
from mcp_management.capability_detector import extract_directory_candidate
from mcp_management.registry import get_installed
from router import route_and_answer
from tools.models import (
    MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH,
    MCP_RUNTIME_REBIND_FAILED,
    MCP_RUNTIME_RESTART_FAILED,
    MCP_RUNTIME_ROLLBACK_FAILED,
)
from voice import speak

console = Console()

_FS_YES_WORDS = {"y", "yes", "approve", "approved", "ok", "okay"}
_FS_NO_WORDS = {"n", "no", "decline", "declined", "cancel"}
_FS_SHOW_PLAN_WORDS = {
    "show plan", "show the plan", "what folder", "what directory",
    "what folder will be added", "what folder will be added?",
    "what directory will be added", "what directory will be added?",
}


def _start_provisioning():
    """Register the Phase F provisioning tools so MCP servers can be set up in-chat.

    Generic subsystem bootstrap — the tools are ordinary BaseTools, so they run
    through the existing executor with the existing permission/confirmation logic.
    A catalog problem is non-fatal: the assistant simply cannot provision.
    """
    if not app_config.mcp_provisioning_enabled():
        return None
    try:
        from mcp_management import McpProvisioningManager, register_provisioning_tools

        manager = McpProvisioningManager()
        tools = register_provisioning_tools(tool_loop.REGISTRY, manager)
    except McpError as e:
        console.print(f"[yellow]MCP provisioning unavailable ({e.code}).[/yellow]")
        return None
    console.print(f"[dim]MCP provisioning: {len(tools)} tool(s) available; "
                  f"catalog has {len(manager.catalog.entries)} approved server(s).[/dim]")
    return manager


def _start_filesystem_access(manager):
    """Register the Phase F.1 filesystem access-management tools (in-chat).

    A peer bootstrap step to _start_provisioning: same manager, same registry, same
    ordinary BaseTool/ToolExecutor path. A missing manager (provisioning disabled)
    simply means these tools are unavailable too.
    """
    if manager is None:
        return
    from mcp_management.filesystem_access_tools import register_filesystem_access_tools

    tools = register_filesystem_access_tools(tool_loop.REGISTRY, manager)
    console.print(f"[dim]Filesystem access management: {len(tools)} tool(s) available.[/dim]")


def _provision_if_needed(manager, mcp_session, user_text, confirmer=None):
    """Generic capability step: offer to install an approved MCP server when the
    request needs one that isn't present. Returns the (possibly new) McpSession.

    This is the Phase F entry point into a live turn — a peer of
    _enrich_with_memory, not a tool-specific branch. Detection, the plan, and the
    approval prompt are all deterministic; nothing is installed without an explicit
    yes. On success the server is started and its tools registered, so the ORIGINAL
    request is then answered by the normal router / shortlist / executor path.
    """
    if manager is None:
        return mcp_session
    try:
        detection, request = manager.begin_request(user_text)
    except McpError as e:
        console.print(f"[yellow]MCP capability check failed ({e.code}).[/yellow]")
        return mcp_session

    if request is None:
        # Nothing to do: no MCP needed, already installed, or no approved server.
        if detection.requires_mcp and detection.error_code:
            console.print(f"[dim]MCP: {detection.reason}[/dim]")
        return mcp_session

    directory = extract_directory_candidate(user_text)
    if not directory:
        # Never guess a directory to grant — say so and answer normally instead.
        console.print("[dim]MCP: an approved server could help, but no directory was "
                      "named to grant access to.[/dim]")
        return mcp_session

    try:
        plan = manager.prepare_plan(request.selected_catalog_id, [directory],
                                    request_id=request.request_id)
        console.print(f"[yellow]This needs the '{detection.capability}' capability, "
                      f"which isn't installed yet.[/yellow]")
        manager.provision(plan, request_id=request.request_id,
                          confirmer=confirmer or confirm_provisioning)
        manager.resume(request.request_id)
    except McpError as e:
        # Always surface WHY, plus the directory that was tried — a bare error code
        # ("MCP_DIRECTORY_NOT_APPROVED") reads like a bug when the real cause is a
        # path that does not exist or is too broad.
        console.print(f"[yellow]MCP provisioning stopped: {e.message}[/yellow]")
        console.print(f"[dim]  directory: {directory}[/dim]")
        console.print(f"[dim]  code: {e.code}[/dim]")
        return mcp_session

    # Restart MCP so the newly installed server's tools register for this turn.
    if mcp_session is not None:
        mcp_session.shutdown()
    return _start_mcp()


def _find_outside_root_failure(manager, calls_and_results):
    """From the (call, result) pairs observed during one local turn, return the
    first eligible "outside the server's approved roots" failure as
    (server_id, call, FilesystemAccessFailure), or None.

    Deterministic: relies only on classify_outside_root_failure (structural path
    comparison against the server's currently registered approved roots), never on
    the model's own prose. A failure the classifier judges ineligible (unrelated
    fault, or a restricted/unjustified proposed root) is skipped, not offered.
    """
    if manager is None:
        return None
    for call, result in calls_and_results:
        if result.success or not call.tool_name.startswith("mcp."):
            continue
        parts = call.tool_name.split(".", 2)
        if len(parts) != 3:
            continue
        server_id = parts[1]
        installed = get_installed(server_id, manager.registry_path, manager.base_dir, manager.managed_root)
        if installed is None:
            continue
        failure = classify_outside_root_failure(
            call.tool_name, call.arguments, result, installed.approved_directories,
            base_dir=manager.base_dir)
        if failure is not None and failure.eligible:
            return server_id, call, failure
    return None


def _offer_filesystem_access(manager, server_id, call, failure, user_text):
    """Open a pending filesystem-access request and prepare its plan. Returns
    (reply_text, request_id). Nothing is changed on disk — only a plan is built
    and shown; the change is applied only after an explicit approval reply."""
    request = manager.begin_filesystem_access_request(
        original_user_text=user_text,
        requested_path=", ".join(failure.requested_paths),
        proposed_root=failure.proposed_root,
        server_id=server_id,
    )
    plan = manager.prepare_filesystem_access_plan(
        server_id, failure.proposed_root, request_id=request.request_id,
        requested_path=request.requested_path, original_user_text=user_text)
    lines = ["This needs access outside the currently approved directory.", ""]
    lines.extend(plan.summary_lines())
    lines.append("")
    lines.append("Reply 'yes' to approve, 'no' to decline, or 'show plan' to see this again.")
    return "\n".join(lines), request.request_id


@dataclass
class _FsReplyOutcome:
    matched: bool
    speak: Optional[str] = None
    next_pending_id: Optional[str] = None
    resumed_text: Optional[str] = None
    # Populated only when an approval was just applied (resumed_text is not None):
    # trusted state for the deterministic MCP runtime restart (Task 10 — the
    # cross-turn "yes" flow and the mid-turn access.add flow both feed the same
    # restart coordinator with the same shape of state).
    server_id: Optional[str] = None
    expected_allowed_roots: tuple = ()
    previous_allowed_roots: tuple = ()


def _resolve_filesystem_access_reply(manager, request_id, user_text):
    """Interpret `user_text` as a reply to a pending filesystem-access approval.

    `matched=False` means the text does not look like a reply to the pending plan
    at all (e.g. an unrelated new request) — the caller should fall through to
    normal routing and leave the pending plan untouched for a later turn. Only an
    exact 'yes'/'no'/'show plan'-style reply is ever treated as a decision, so
    unrelated text that merely contains the word "yes" is never misread as approval.
    """
    if manager is None:
        return _FsReplyOutcome(matched=False)
    request = manager.pending_filesystem_access(request_id)
    if request is None:
        return _FsReplyOutcome(matched=False)

    normalized = user_text.strip().lower().rstrip("?.! ")

    if normalized in _FS_SHOW_PLAN_WORDS:
        plan = manager.get_filesystem_access_plan(request.access_plan_id)
        if plan is None:
            return _FsReplyOutcome(matched=True, speak="That plan is no longer available.",
                                   next_pending_id=None)
        return _FsReplyOutcome(matched=True, speak="\n".join(plan.summary_lines()),
                               next_pending_id=request_id)

    if normalized in _FS_NO_WORDS:
        manager.decline_filesystem_access(request_id)
        return _FsReplyOutcome(matched=True, speak="Okay, I've left filesystem access unchanged.",
                               next_pending_id=None)

    if normalized in _FS_YES_WORDS:
        plan = manager.get_filesystem_access_plan(request.access_plan_id)
        if plan is None:
            return _FsReplyOutcome(matched=True,
                                   speak="That plan is no longer available; please ask again.",
                                   next_pending_id=None)
        try:
            result = manager.apply_filesystem_access(plan, request_id=request_id, confirmer=lambda p: True)
        except McpError as e:
            return _FsReplyOutcome(matched=True,
                                   speak=f"I couldn't update filesystem access ({e.code}): {e.message}",
                                   next_pending_id=None)
        resumed_text = manager.resume_filesystem_access(request_id)
        return _FsReplyOutcome(
            matched=True, speak=None, next_pending_id=None, resumed_text=resumed_text,
            server_id=result.get("server_id"),
            expected_allowed_roots=tuple(result.get("approved_directories") or ()),
            previous_allowed_roots=plan.current_allowed_directories,
        )

    return _FsReplyOutcome(matched=False)


def _classify_access_apply_success(manager, call, result):
    """Recognize a successful mcp.<server>.access.add/remove call — structurally,
    from the BUILT-IN tool's own registry name (4 dot-separated segments; a remote
    MCP tool always has exactly 3, e.g. "mcp.filesystem.read_text_file") and its
    own result data, never from freshly generated LLM text.

    Returns (ToolLoopDirective_or_None, previous_allowed_roots_or_None). The
    "previous" root set — needed only for a best-effort rollback if the runtime
    restart itself fails — is recovered from the plan the `add` call referenced
    (its `plan_id` argument); a `remove` call carries no plan id, so rollback
    context is unavailable for that path (documented limitation).
    """
    if not result.success:
        return None, None
    parts = call.tool_name.split(".")
    if len(parts) != 4 or parts[0] != "mcp" or parts[2] != "access" or parts[3] not in ("add", "remove"):
        return None, None
    approved = result.data.get("approved_directories") if isinstance(result.data, dict) else None
    if not isinstance(approved, list):
        return None, None

    previous_roots = None
    plan_id = call.arguments.get("plan_id") if isinstance(call.arguments, dict) else None
    if isinstance(plan_id, str) and manager is not None:
        plan = manager.get_filesystem_access_plan(plan_id)
        if plan is not None:
            previous_roots = plan.current_allowed_directories

    directive = tool_loop.ToolLoopDirective(
        control=tool_loop.ToolLoopControl.RESTART_MCP_AND_RESUME,
        server_id=parts[1],
        expected_allowed_roots=tuple(str(d) for d in approved),
    )
    return directive, previous_roots


_RUNTIME_RESTART_FAILED_MESSAGE = (
    "Filesystem access was updated, but the MCP runtime could not be restarted safely. "
    "The previous configuration was restored."
)
_RUNTIME_ERROR_MESSAGES = {
    MCP_RUNTIME_RESTART_FAILED: _RUNTIME_RESTART_FAILED_MESSAGE,
    MCP_RUNTIME_REBIND_FAILED: _RUNTIME_RESTART_FAILED_MESSAGE,
    MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH: (
        "Filesystem access was updated, but the restarted MCP server did not report the "
        "expected approved directories. The previous configuration was restored."
    ),
    MCP_RUNTIME_ROLLBACK_FAILED: (
        "Filesystem access was updated, but the MCP runtime failed to restart and the "
        "previous configuration could not be restored automatically. Please restart the "
        "assistant."
    ),
}


def _restart_mcp_and_resume(manager, runtime, directive, user_text, history, system_prompt,
                            attempted_fs_requests, resume_budget, previous_allowed_roots=None):
    """Phase F.1 hotfix — the one deterministic path from an applied access change
    to the original request actually succeeding.

    Replaces the active MCP session (mcp_layer.runtime_manager.McpRuntimeManager),
    verifies the LIVE new server's allowed roots, and — only then — resumes
    `user_text` through the normal router / Phase B shortlist / local LLM
    selection / ToolExecutor / freshly-registered McpTool pipeline exactly once.
    Both the cross-turn "yes" approval and a mid-turn access.add call from the
    model itself route through this same function (Task 10), so the outcome never
    depends on how the approval was collected. `resume_budget` bounds the total
    number of runtime replacements this call chain may still perform — checked
    BEFORE touching the runtime, so at most one replacement ever happens
    (MCP_RESUME_ABORTED otherwise). Returns (reply, pending_fs_request_id_or_None).
    """
    if resume_budget <= 0:
        console.print("[yellow]MCP_RESUME_ABORTED: a runtime replacement already occurred "
                      "for this request.[/yellow]")
        return ("Filesystem access was already updated and the MCP runtime already restarted "
                "once for this request; I won't restart it again automatically.", None)

    coordinator = McpRuntimeManager(
        tool_loop.REGISTRY,
        base_dir=manager.base_dir if manager else None,
        managed_root=manager.managed_root if manager else None,
        registry_path=manager.registry_path if manager else None,
    )
    try:
        coordinator.replace_active_session(
            runtime, directive.server_id, directive.expected_allowed_roots,
            previous_allowed_roots=previous_allowed_roots)
    except McpError as e:
        message = _RUNTIME_ERROR_MESSAGES.get(
            e.code, f"Filesystem access was updated, but the MCP runtime restart failed ({e.code}).")
        console.print(f"[yellow]{message}[/yellow]")
        return message, None

    console.print(f"[dim]MCP '{directive.server_id}': runtime restarted; "
                  f"{len(directive.expected_allowed_roots)} allowed root(s) verified live.[/dim]")

    resumed_prompt = _enrich_with_memory(user_text)
    resumed_decision = route_and_answer(resumed_prompt, history)
    console.print(f"[dim]routing (resumed): mode={resumed_decision.mode} tool={resumed_decision.tool}[/dim]")
    if resumed_decision.mode != "local":
        reply, _extra = dispatch(resumed_decision, user_text, resumed_prompt, history, system_prompt)
        return reply, None

    reply, _extra, pending_fs_request_id = _run_local_turn(
        manager, runtime, user_text, resumed_prompt, history, system_prompt,
        attempted_fs_requests, resume_budget=resume_budget - 1)
    return reply, pending_fs_request_id


def _run_local_turn(manager, runtime, user_text, prompt, history, system_prompt,
                    attempted_fs_requests, resume_budget=1):
    """Run one local-mode turn with immediate Phase F.1 hotfix interception.

    Unlike the old post-hoc scan (which only looked at what happened after the
    WHOLE turn finished), the observer here classifies each tool result the
    instant it comes back, so the tool loop halts BEFORE the local LLM gets a
    chance to write a generic fallback answer or place another call through a
    soon-to-be-stale MCP session. Returns
    (reply, extra_metrics, pending_fs_request_id_or_None).
    """
    mcp_session = _provision_if_needed(manager, runtime.session, user_text)
    runtime.replace(mcp_session)

    halt = {}

    def on_result(call, result):
        directive, previous_roots = _classify_access_apply_success(manager, call, result)
        if directive is None:
            found = _find_outside_root_failure(manager, [(call, result)])
            if found is not None:
                server_id, found_call, failure = found
                directive = tool_loop.ToolLoopDirective(
                    control=tool_loop.ToolLoopControl.HALT_FOR_FILESYSTEM_ACCESS,
                    server_id=server_id)
                halt["outside_root"] = (server_id, found_call, failure)
        if directive is not None:
            halt["directive"] = directive
            halt["previous_allowed_roots"] = previous_roots
        return directive

    reply, extra_metrics = tool_loop.run_local_tool_loop(
        prompt, history, system_prompt, on_tool_result=on_result)

    directive = halt.get("directive")
    if directive is None:
        return reply, extra_metrics, None

    if directive.control == tool_loop.ToolLoopControl.HALT_FOR_FILESYSTEM_ACCESS:
        server_id, found_call, failure = halt["outside_root"]
        if user_text in attempted_fs_requests:
            console.print("[yellow]MCP_FILESYSTEM_ACCESS_LOOP_PREVENTED: already "
                          "attempted an access expansion for this request.[/yellow]")
            reply = ("I already tried expanding filesystem access for this exact "
                     "request once, and it's still outside the approved directories. "
                     "Please check the path and ask again.")
            return reply, extra_metrics, None
        attempted_fs_requests.add(user_text)
        reply, pending_fs_request_id = _offer_filesystem_access(
            manager, server_id, found_call, failure, user_text)
        return reply, extra_metrics, pending_fs_request_id

    # RESTART_MCP_AND_RESUME — the model itself called access.add/remove and it
    # succeeded; the loop already halted before any stale follow-up call.
    reply, pending_fs_request_id = _restart_mcp_and_resume(
        manager, runtime, directive, user_text, history, system_prompt,
        attempted_fs_requests, resume_budget,
        previous_allowed_roots=halt.get("previous_allowed_roots"))
    return reply, extra_metrics, pending_fs_request_id


def _start_mcp():
    """Start the configured external MCP server and register its tools (Phase E).

    Generic subsystem bootstrap — not a per-tool branch. Reads config/mcp_server.json
    (MCP disabled by default). An optional server (`required: false`) that fails to
    start is logged and skipped so built-in tools keep working. Returns an McpSession.
    """
    try:
        # Which configuration is in effect: env override -> managed (Phase F) -> template.
        # Only the source + basename are logged, never the path with its contents.
        resolved = resolve_config()
        console.print(f"[dim]MCP config: {resolved.describe()}[/dim]")
        session = mcp_layer.bootstrap_from_config(tool_loop.REGISTRY)
    except McpError as e:
        # Only a `required` server (or a bad MCP_CONFIG_PATH) reaches here.
        console.print(f"[yellow]MCP startup problem ({e.code}); continuing without MCP tools.[/yellow]")
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


def dispatch(decision, user_text, prompt, history, system_prompt, on_tool_result=None):
    """Generic orchestration: turn a RouteDecision into (reply, metrics dict).

    metrics covers only calls made *beyond* routing — e.g. an escalation to Claude,
    a vision-model call, or a tool's finishing summarization — since the router's
    own token usage is already on `decision`.

    There are no tool-specific branches here: the only decision is which of the
    three routes to take. Tool-mode requests are executed through the shared
    ToolRegistry / ToolExecutor and rendered by tool_dispatch, which is the single
    place that knows how a built-in's result becomes a spoken answer.

    `on_tool_result` is an optional, generic pass-through to the local tool loop
    (see tool_loop.run_local_tool_loop) for observing individual tool-call results
    within a "local" turn; dispatch itself has no opinion on what it's used for.
    """
    if decision.mode == "claude":
        return ask_claude(prompt, history, system_prompt)

    if decision.mode == "tool":
        return tool_dispatch.execute_and_render(decision, user_text, history, system_prompt)

    # mode == "local" (and any unexpected mode) — routing only decided; generate the
    # answer separately. The local tool loop lets the model call its LLM-selectable
    # tools and then writes the final answer itself. With TOOL_CALLING_ENABLED=false
    # it falls back to the original single-shot ask_local behavior.
    return tool_loop.run_local_tool_loop(prompt, history, system_prompt, on_tool_result=on_tool_result)


def get_user_text(mode):
    if mode == "p":
        text = listen_push_to_talk()
        console.print(f"[dim]heard: {text}[/dim]")
        return text
    return input("> ").strip()


def main():
    system_prompt = load_system_prompt()
    history = []
    # Phase F.1 cross-turn state: at most one filesystem-access plan awaits a bare
    # yes/no reply at a time, and each ORIGINAL blocked request gets at most one
    # expansion attempt (loop prevention — MCP_FILESYSTEM_ACCESS_LOOP_PREVENTED).
    pending_fs_request_id = None
    attempted_fs_requests = set()

    console.print("[bold]home-ai (LLM router v2 — consolidated)[/bold]")
    # The one authoritative mutable reference to "the current MCP session" (Phase
    # F.1 hotfix Task 8) — every consumer reads runtime.session at the moment it
    # needs it, so a runtime replacement can never leave a component holding a
    # stale local variable that points at an already-closed session.
    runtime = ActiveMcpRuntime(_start_mcp())
    provisioning_manager = _start_provisioning()
    _start_filesystem_access(provisioning_manager)

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

            # A pending filesystem-access approval takes priority over normal
            # routing: a bare yes/no/show-plan reply resolves it directly, never
            # reaching the router or local tool-selection. Anything else falls
            # through to normal routing below and leaves the plan pending.
            if pending_fs_request_id is not None:
                outcome = _resolve_filesystem_access_reply(
                    provisioning_manager, pending_fs_request_id, user_text)
                if outcome.matched:
                    if outcome.resumed_text is None:
                        if outcome.speak is not None:
                            console.print(f"[cyan]{outcome.speak}[/cyan]")
                            speak(outcome.speak)
                            history.append({"role": "user", "content": user_text})
                            history.append({"role": "assistant", "content": outcome.speak})
                        pending_fs_request_id = outcome.next_pending_id
                        continue

                    # Approved and applied: replace the MCP runtime deterministically
                    # (Task 5/6/8) and resume the ORIGINAL blocked request through the
                    # normal pipeline (Task 9) — the SAME coordinator a mid-turn
                    # access.add call uses, so the outcome never depends on how the
                    # approval was collected (Task 10).
                    directive = tool_loop.ToolLoopDirective(
                        control=tool_loop.ToolLoopControl.RESTART_MCP_AND_RESUME,
                        server_id=outcome.server_id,
                        expected_allowed_roots=outcome.expected_allowed_roots)
                    turn_start = time.perf_counter()
                    reply, pending_fs_request_id = _restart_mcp_and_resume(
                        provisioning_manager, runtime, directive, outcome.resumed_text, history,
                        system_prompt, attempted_fs_requests, resume_budget=1,
                        previous_allowed_roots=outcome.previous_allowed_roots)
                    console.print(f"[cyan]{reply}[/cyan]")
                    speak(reply)
                    total_time_sec = time.perf_counter() - turn_start
                    log_turn(question=outcome.resumed_text, mode="local", tool=None,
                            prompt_tokens=0, completion_tokens=0, total_time_sec=total_time_sec)
                    history.append({"role": "user", "content": user_text})
                    history.append({"role": "assistant", "content": reply})
                    continue

            turn_start = time.perf_counter()

            prompt = _enrich_with_memory(user_text)
            decision = route_and_answer(prompt, history)
            console.print(f"[dim]routing: mode={decision.mode} tool={decision.tool}[/dim]")

            if decision.mode == "local":
                # Phase F.1 hotfix: provisioning-if-needed, immediate outside-root
                # interception, and a successful access.add/remove all run inside
                # _run_local_turn — see its docstring for why "immediate" matters.
                reply, extra_metrics, pending_fs_request_id = _run_local_turn(
                    provisioning_manager, runtime, user_text, prompt, history, system_prompt,
                    attempted_fs_requests, resume_budget=1)
            else:
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
        # Closes whichever session is CURRENTLY active — never a stale reference to
        # an already-shutdown session from before a runtime replacement.
        runtime.close()


if __name__ == "__main__":
    main()
