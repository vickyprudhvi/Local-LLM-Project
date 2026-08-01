"""Main loop. v2: single-model unified routing + answering to avoid VRAM swap-thrashing.

Phase A: assistant.py is a generic orchestration layer only. It routes a turn
(Claude / local / tool), and for a tool turn hands the RouteDecision to
tool_dispatch, which runs the selected built-in through the shared ToolRegistry /
ToolExecutor and renders the result. assistant.py knows nothing about how any
specific tool works — there are no per-tool branches here.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

from rich.console import Console

import memory_store
import tool_dispatch
import tool_loop
import tools.config as app_config
from brain import ask_claude, load_system_prompt
from ears import listen_push_to_talk
from interaction_log import log_turn
from mcp_layer import FilesystemRootValidator, McpError, MultiMcpRuntimeManager
from mcp_management.access_classifier import (
    FilesystemAccessFailure,
    _is_within,
    classify_outside_root_failure,
    propose_root,
)
from mcp_management.approval import confirm_provisioning
from mcp_management.auto_provisioning import AutoProvisioningManager
from mcp_management.capabilities import CapabilitySelectionStatus
from mcp_management.capability_detector import extract_directory_candidate
from mcp_management.capability_service import select_for_request
from mcp_management.document_authorization import (
    DocumentAuthorizationStore,
    build_document_snapshots_from_text,
)
from mcp_management.provisioning_models import AutoProvisioningApproval
from mcp_management.registry import get_installed
from mcp_management.runtime_activation import ensure_selected_server_active
from router import route_and_answer
from tools.models import (
    MCP_DOCUMENT_AUTHORIZATION_CONSUMED,
    MCP_DOCUMENT_AUTHORIZATION_EXPIRED,
    MCP_DOCUMENT_AUTHORIZATION_REQUIRED,
    MCP_DOCUMENT_AUTHORIZATION_RESERVED,
    MCP_DOCUMENT_SNAPSHOT_MISMATCH,
    MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH,
    MCP_RUNTIME_REBIND_FAILED,
    MCP_RUNTIME_RESTART_FAILED,
    MCP_RUNTIME_ROLLBACK_FAILED,
    MCP_SERVER_NOT_INSTALLED,
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

# Phase G.3 — auto-provisioning cross-turn reply words. A DISTINCT constant set
# from the Phase F.1 filesystem-access ones above (even though the words
# overlap) because which resolver runs is chosen by the pending request id's
# prefix ("autoreq_" vs "fsreq_"), never by guessing from the reply text alone.
_AP_YES_WORDS = {"y", "yes", "approve", "approved", "proceed"}
_AP_NO_WORDS = {"n", "no", "decline", "declined", "cancel"}
_AP_SHOW_PLAN_WORDS = {"show plan", "show the plan"}
_AUTO_PROVISIONING_REQUEST_PREFIX = "autoreq_"

# Phase G.4 Defect 4 — every word/phrase that resolves an approval workflow,
# across both the Phase F.1 filesystem-access and Phase G.3 auto-provisioning
# reply vocabularies. A bare one of these with NO matching pending request is
# never a normal request (see `_is_bare_approval_token` / `main()`).
_ALL_APPROVAL_TOKEN_WORDS = (
    _FS_YES_WORDS | _FS_NO_WORDS | _FS_SHOW_PLAN_WORDS
    | _AP_YES_WORDS | _AP_NO_WORDS | _AP_SHOW_PLAN_WORDS
)

# Phase G.4 Defect 2/9 — internal document-authorization control-plane
# failures. These mean the REQUEST-BOUND authorization state itself is
# missing/expired/consumed/reserved/stale — never a reason for the model to
# see raw error text and improvise (e.g. inventing a Filesystem approval
# offer). A DIFFERENT failure (a malformed/malicious model-supplied URI, a
# genuine remote conversion failure, an oversized result) is deliberately NOT
# in this set — those remain visible to the model exactly as before.
_DOCUMENT_AUTH_CONTROL_PLANE_CODES = frozenset({
    MCP_DOCUMENT_AUTHORIZATION_REQUIRED,
    MCP_DOCUMENT_AUTHORIZATION_EXPIRED,
    MCP_DOCUMENT_AUTHORIZATION_CONSUMED,
    MCP_DOCUMENT_AUTHORIZATION_RESERVED,
    MCP_DOCUMENT_SNAPSHOT_MISMATCH,
})
_DOCUMENT_TO_MARKDOWN_CAPABILITY = "document_to_markdown"


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


def _start_auto_provisioning(manager):
    """Register the Phase G.3 automatic-provisioning subsystem on `manager`.

    Deliberately an ATTRIBUTE on the existing `McpProvisioningManager`, not a
    new constructor parameter: `McpProvisioningManager.__init__` stays exactly
    as Phase F left it, so every existing test that builds one directly (there
    are many) is unaffected, and `getattr(manager, "auto_provisioning", None)`
    is how the rest of this module (and any caller that doesn't wire this up)
    finds it — `None` simply means "automatic provisioning unavailable," the
    same tolerance already used for `manager is None` throughout this file.
    """
    if manager is None:
        return
    manager.auto_provisioning = AutoProvisioningManager(
        manager.catalog, base_dir=manager.base_dir, managed_root=manager.managed_root,
        registry_path=manager.registry_path)


def _provision_if_needed(manager, user_text, confirmer=None):
    """Generic capability step: offer to install an approved MCP server when the
    request needs one that isn't present.

    This is the Phase F entry point into a live turn — a peer of
    _enrich_with_memory, not a tool-specific branch. Detection, the plan, and the
    approval prompt are all deterministic; nothing is installed without an explicit
    yes. Installation itself never starts the new server (Phase G.2: startup is
    always lazy) — the ORIGINAL request is re-answered by the normal router /
    Phase G.1 capability-selection / shortlist / executor path, and THAT lazily
    activates the newly-installed server the moment it is actually selected.
    """
    if manager is None:
        return
    try:
        detection, request = manager.begin_request(user_text)
    except McpError as e:
        console.print(f"[yellow]MCP capability check failed ({e.code}).[/yellow]")
        return

    if request is None:
        # Nothing to do: no MCP needed, already installed, or no approved server.
        if detection.requires_mcp and detection.error_code:
            console.print(f"[dim]MCP: {detection.reason}[/dim]")
        return

    directory = extract_directory_candidate(user_text)
    if not directory:
        # Never guess a directory to grant — say so and answer normally instead.
        console.print("[dim]MCP: an approved server could help, but no directory was "
                      "named to grant access to.[/dim]")
        return

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


def _find_directory_request_outside_roots(manager, call, result, user_text):
    """A SUCCESSFUL mcp.<server>.list_allowed_directories call paired with a
    directory the ORIGINAL request named — one not covered by any currently
    approved root — is functionally the same situation as an outside-root
    failure: the user asked to work in a directory the server can't reach.
    This covers the case where the model checks the allowed list first
    (a normal, successful call) instead of attempting the real read/list
    operation and letting THAT fail, which is the only case
    `_find_outside_root_failure` can classify. Detected deterministically
    from the server's own registry state and a path shape extracted from the
    user's own words (`extract_directory_candidate`) — never from anything
    the model generated. Returns (server_id, FilesystemAccessFailure) or None.
    """
    if manager is None or not result.success:
        return None
    parts = call.tool_name.split(".", 2)
    if len(parts) != 3 or parts[2] != "list_allowed_directories":
        return None
    server_id = parts[1]
    installed = get_installed(server_id, manager.registry_path, manager.base_dir, manager.managed_root)
    if installed is None:
        return None
    candidate = extract_directory_candidate(user_text)
    if not candidate:
        return None
    try:
        resolved = os.path.realpath(candidate)
    except (OSError, ValueError):
        return None
    allowed = [os.path.realpath(str(r)) for r in (installed.approved_directories or ())]
    if _is_within(resolved, allowed):
        return None  # already covered — nothing to offer
    proposal = propose_root([resolved], remote_name="list_directory", base_dir=manager.base_dir)
    if not proposal.ok:
        return None
    failure = FilesystemAccessFailure(
        requested_paths=(resolved,), proposed_root=proposal.directory,
        restricted=proposal.restricted, eligible=proposal.ok, reason=proposal.reason)
    return server_id, failure


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


def _restart_mcp_and_resume(manager, runtime_manager, directive, user_text, history, system_prompt,
                            attempted_fs_requests, resume_budget, previous_allowed_roots=None):
    """Phase F.1 hotfix — the one deterministic path from an applied access change
    to the original request actually succeeding.

    Phase G.2: replaces ONLY `directive.server_id`'s session
    (`MultiMcpRuntimeManager.replace_session` — every other server's slot is
    untouched), verifies the LIVE new server's allowed roots, and — only then —
    resumes `user_text` through the normal router / Phase B shortlist / local LLM
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

    try:
        runtime_manager.replace_session(
            directive.server_id, expected_allowed_roots=directive.expected_allowed_roots,
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

    # Task 8 (G.1) / Task 8 (G.2): a resumed request re-enters the FULL pipeline —
    # router already ran above; this re-runs Phase G.1 capability selection (and,
    # for a SELECTED result, Phase G.2 lazy activation) before Phase B, exactly
    # like a fresh request. A resumed document-conversion request with no
    # approved provider must still report MCP_CAPABILITY_UNAVAILABLE here, not
    # fall through into Phase B or another Filesystem access cycle.
    reply, _extra, pending_fs_request_id = _process_local_request_with_capability_selection(
        manager, runtime_manager, user_text, resumed_prompt, history, system_prompt,
        attempted_fs_requests, resume_budget=resume_budget - 1)
    return reply, pending_fs_request_id


def _run_local_turn(manager, runtime_manager, user_text, prompt, history, system_prompt,
                    attempted_fs_requests, resume_budget=1, selection=None):
    """Run one local-mode turn with immediate Phase F.1 hotfix interception.

    Unlike the old post-hoc scan (which only looked at what happened after the
    WHOLE turn finished), the observer here classifies each tool result the
    instant it comes back, so the tool loop halts BEFORE the local LLM gets a
    chance to write a generic fallback answer or place another call through a
    soon-to-be-stale MCP session. Returns
    (reply, extra_metrics, pending_fs_request_id_or_None).

    By the time this runs, Phase G.2 has already lazily activated whichever
    server Phase G.1 selected (see _process_local_request_with_capability_
    selection) — this function no longer starts anything itself.

    `selection`, when given, carries the Phase G capability decision into the
    tool loop so a REQUIRED request can be fail-closed if the model refuses to
    select a tool from the offered shortlist.
    """
    _provision_if_needed(manager, user_text)

    tool_requirement = tool_loop.ToolRequirement.NONE
    preferred_mcp_server_id = None
    if selection is not None:
        tool_requirement = getattr(selection, "tool_requirement", tool_loop.ToolRequirement.NONE)
        preferred_mcp_server_id = getattr(selection, "preferred_mcp_server_id", None)

    halt = {}

    def on_result(call, result):
        directive, previous_roots = _classify_access_apply_success(manager, call, result)
        if directive is None and not result.success and result.error is not None \
                and result.error.code in _DOCUMENT_AUTH_CONTROL_PLANE_CODES:
            # Phase G.4 Defect 2 — an internal document-authorization
            # control-plane failure (missing/expired/consumed/reserved/stale)
            # must never reach the local LLM: it is not something the model
            # can reason about, and letting it see the raw code is exactly
            # what led the model to fabricate a Filesystem approval message
            # in the reported live failure. Halt immediately, before the model
            # gets another turn.
            directive = tool_loop.ToolLoopDirective(
                control=tool_loop.ToolLoopControl.HALT_WITH_ERROR,
                error_code=result.error.code)
        if directive is None:
            found = _find_outside_root_failure(manager, [(call, result)])
            if found is not None:
                server_id, found_call, failure = found
                directive = tool_loop.ToolLoopDirective(
                    control=tool_loop.ToolLoopControl.HALT_FOR_FILESYSTEM_ACCESS,
                    server_id=server_id)
                halt["outside_root"] = (server_id, found_call, failure)
        if directive is None:
            # The model checked list_allowed_directories (a normal, successful
            # call) instead of attempting the real operation and letting THAT
            # fail — the only case the check above can catch. Detect the same
            # "directory not covered" situation from the user's own words.
            found = _find_directory_request_outside_roots(manager, call, result, user_text)
            if found is not None:
                server_id, failure = found
                directive = tool_loop.ToolLoopDirective(
                    control=tool_loop.ToolLoopControl.HALT_FOR_FILESYSTEM_ACCESS,
                    server_id=server_id)
                halt["outside_root"] = (server_id, call, failure)
        if directive is not None:
            halt["directive"] = directive
            halt["previous_allowed_roots"] = previous_roots
        return directive

    loop_result = tool_loop.run_local_tool_loop(
        prompt, history, system_prompt, on_tool_result=on_result,
        tool_requirement=tool_requirement, preferred_mcp_server_id=preferred_mcp_server_id)
    reply = loop_result.text
    extra_metrics = dict(loop_result.metrics)

    directive = halt.get("directive")

    # Phase G.4 Defects 2/9 — reconstruct exactly once (fresh snapshot + fresh
    # authorization — never the same authorization id) and retry tool
    # selection/execution exactly once. A SECOND control-plane failure of the
    # same kind returns a controlled internal error instead of trying again or
    # exposing the model to it.
    document_auth_retried = False
    while (directive is not None and directive.control == tool_loop.ToolLoopControl.HALT_WITH_ERROR
          and not document_auth_retried):
        document_auth_retried = True
        console.print(f"[dim]MCP document authorization control-plane failure "
                      f"({directive.error_code}); attempting one deterministic "
                      "reconstruction.[/dim]")
        halt.pop("directive", None)
        if not _reconstruct_document_authorization(user_text):
            return ("I couldn't re-establish access to that document for this request. "
                    "Please try again.", extra_metrics, None)
        loop_result = tool_loop.run_local_tool_loop(
            prompt, history, system_prompt, on_tool_result=on_result,
            tool_requirement=tool_requirement, preferred_mcp_server_id=preferred_mcp_server_id)
        reply = loop_result.text
        extra_metrics = {
            "prompt_tokens": extra_metrics.get("prompt_tokens", 0)
                            + loop_result.metrics.get("prompt_tokens", 0),
            "completion_tokens": extra_metrics.get("completion_tokens", 0)
                                + loop_result.metrics.get("completion_tokens", 0),
        }
        directive = halt.get("directive")

    if directive is not None and directive.control == tool_loop.ToolLoopControl.HALT_WITH_ERROR:
        return ("I couldn't convert that document right now because of an internal "
                "authorization issue. Please try again.", extra_metrics, None)

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
        manager, runtime_manager, directive, user_text, history, system_prompt,
        attempted_fs_requests, resume_budget,
        previous_allowed_roots=halt.get("previous_allowed_roots"))
    return reply, extra_metrics, pending_fs_request_id


_ZERO_METRICS = {"prompt_tokens": 0, "completion_tokens": 0}


def _log_capability_selection(selection):
    """Debug-only diagnostics (Task 9/12): silent on a normal request
    (MCP_CAPABILITY_DEBUG unset) and even then never prints raw request text,
    file contents, credentials, or full paths — only capability ids, coarse
    evidence labels (a verb, a path SHAPE like "windows_absolute", an
    extension), confidence, and the selection outcome.
    """
    if not app_config.mcp_capability_debug_enabled():
        return
    if selection.status == CapabilitySelectionStatus.NONE_REQUIRED:
        return
    for req in selection.required_capabilities:
        console.print("[dim]capability detection:[/dim]")
        console.print(f"[dim]  capability: {req.capability_id}[/dim]")
        console.print("[dim]  evidence:[/dim]")
        for ev in req.evidence:
            console.print(f"[dim]    - {ev.evidence_type.value}: {ev.value}[/dim]")

    console.print("[dim]trusted provider lookup:[/dim]")
    required_ids = ", ".join(r.capability_id for r in selection.required_capabilities)
    console.print(f"[dim]  capability: {required_ids}[/dim]")
    console.print(f"[dim]  candidates: {len(selection.candidates)}[/dim]")
    console.print(f"[dim]  result: {selection.status.value}[/dim]")
    if selection.selected_server_id:
        console.print(f"[dim]  selected: {selection.selected_server_id}[/dim]")


def _capability_selection_reply(selection):
    """Normalized user-visible text (Task 11) for a status that skips Phase B
    entirely. None for NONE_REQUIRED/SELECTED, which proceed exactly as before."""
    if selection.status in (
        CapabilitySelectionStatus.UNSUPPORTED,
        CapabilitySelectionStatus.AMBIGUOUS,
        CapabilitySelectionStatus.MULTI_SERVER_REQUIRED,
    ):
        return selection.explanation
    if selection.status == CapabilitySelectionStatus.INVALID_CATALOG:
        return "The trusted MCP catalog could not be used right now; continuing without it."
    return None


def _log_runtime_activation(runtime_manager, server_id, activation):
    """Debug-only diagnostics (Phase G.2 Task 14): silent unless
    MCP_CAPABILITY_DEBUG is set. Reports state + tool count only — never a raw
    path, environment value, or credential."""
    if not app_config.mcp_capability_debug_enabled():
        return
    status = runtime_manager.get_status(server_id)
    console.print(f"[dim]MCP {server_id}:[/dim]")
    console.print(f"[dim]  state: {status.state.value}[/dim]")
    console.print(f"[dim]  tools registered: {status.registered_tool_count}[/dim]")
    if not activation.activated:
        console.print(f"[dim]  error: {activation.error_code}[/dim]")


def _offer_mcp_provisioning(auto_manager, capability, catalog_entry, user_text):
    """Phase G.3 — deterministic control-plane branch, not an LLM decision
    (Task 14): reached only when Phase G.1 already selected an APPROVED
    catalog entry and Phase G.2 already found it not installed. Opens a
    pending auto-provisioning request and prepares its plan. Returns
    (reply_text, request_id), or None when this catalog entry is not eligible
    for automatic provisioning (e.g. it requires a directory grant, which
    stays on the existing Filesystem-specific manual/heuristic path).

    Phase G.4: for the `document_to_markdown` capability, the local document
    paths extracted from the user's text are captured as READ snapshots and
    bound into the provisioning plan hash."""
    document_snapshots = ()
    if capability == "document_to_markdown":
        document_snapshots = build_document_snapshots_from_text(user_text)
    request = auto_manager.begin_request(user_text, capability, catalog_entry,
                                          document_snapshots=document_snapshots)
    if request is None:
        return None
    plan = auto_manager.prepare_plan(request.request_id)
    lines = list(plan.summary_lines())
    return "\n".join(lines), request.request_id


@dataclass
class _AutoProvisioningReplyOutcome:
    matched: bool
    speak: Optional[str] = None
    next_pending_id: Optional[str] = None
    resumed_text: Optional[str] = None


def _resolve_auto_provisioning_reply(auto_manager, runtime_manager, request_id, user_text):
    """Interpret `user_text` as a reply to a pending Phase G.3 provisioning
    approval. Mirrors `_resolve_filesystem_access_reply`'s exact-reply-only
    matching (Task 4): unrelated text that merely contains "yes" is never
    misread as approval, and only the caller's own request id (already
    disambiguated by its "autoreq_" prefix — see `main()`) reaches here."""
    if auto_manager is None:
        return _AutoProvisioningReplyOutcome(matched=False)
    request = auto_manager.pending(request_id)
    if request is None:
        return _AutoProvisioningReplyOutcome(matched=False)

    normalized = user_text.strip().lower().rstrip("?.! ")

    if normalized in _AP_SHOW_PLAN_WORDS:
        plan = auto_manager.get_plan(request.plan_id) if request.plan_id else None
        if plan is None:
            return _AutoProvisioningReplyOutcome(matched=True, speak="That plan is no longer available.")
        return _AutoProvisioningReplyOutcome(matched=True, speak="\n".join(plan.summary_lines()),
                                             next_pending_id=request_id)

    if normalized in _AP_NO_WORDS:
        auto_manager.decline(request_id)
        return _AutoProvisioningReplyOutcome(matched=True, speak="Okay, I won't install that MCP server.")

    if normalized in _AP_YES_WORDS:
        plan = auto_manager.get_plan(request.plan_id) if request.plan_id else None
        if plan is None:
            return _AutoProvisioningReplyOutcome(
                matched=True, speak="That plan is no longer available; please ask again.")
        approval = AutoProvisioningApproval(approved=True, plan_id=plan.plan_id, plan_hash=plan.compute_hash())
        try:
            auto_manager.provision_and_activate(request_id, runtime_manager, approval=approval)
        except McpError as e:
            return _AutoProvisioningReplyOutcome(
                matched=True, speak=f"I couldn't install that MCP server ({e.code}): {e.message}")
        resumed_text = auto_manager.resume(request_id)
        return _AutoProvisioningReplyOutcome(matched=True, resumed_text=resumed_text)

    return _AutoProvisioningReplyOutcome(matched=False)


def _reconstruct_document_authorization(user_text):
    """Phase G.4 Defects 1/2/7/9 — deterministically (re)capture and validate
    document snapshots from `user_text` and create a FRESH, single-use
    `DocumentInputAuthorization` for each. Reuses the exact same primitives
    (`build_document_snapshots_from_text`, `DocumentAuthorizationStore.
    create_authorization`) the first-time-provisioning resume path already
    uses (`AutoProvisioningManager._create_document_authorizations_for_
    resumption`) — never a parallel authorization mechanism.

    Returns True when at least one snapshot was found and authorized; False
    when no local document evidence exists at all, or revalidation fails
    (file missing, changed, wrong type, etc.) — the caller must fail closed
    rather than let Phase B run with no live authorization. Every call
    creates a brand-new authorization id; a prior one (from an earlier
    request, or a just-consumed one) is never reused.
    """
    snapshots = build_document_snapshots_from_text(user_text)
    if not snapshots:
        return False
    store = DocumentAuthorizationStore.default()
    try:
        for snapshot in snapshots:
            store.create_authorization(snapshot)
    except McpError:
        return False
    return True


def _is_bare_approval_token(user_text):
    """Phase G.4 Defect 4 — 'yes'/'no'/'show plan' (and synonyms) are
    approval-workflow replies, never ordinary requests. Normalized identically
    to `_resolve_filesystem_access_reply`/`_resolve_auto_provisioning_reply`
    so this can never disagree with them about what counts as a bare token."""
    normalized = user_text.strip().lower().rstrip("?.! ")
    return normalized in _ALL_APPROVAL_TOKEN_WORDS


def _process_local_request_with_capability_selection(manager, runtime_manager, user_text, prompt, history,
                                                      system_prompt, attempted_fs_requests,
                                                      resume_budget=1):
    """THE single authoritative entrypoint for every local request (Task 1).

    Every local request — the normal typed/push-to-talk turn, a router
    fallback-to-local, and every resumption (Filesystem approval, runtime
    restart, a future provisioning approval) — must call this function, never
    `_run_local_turn` directly. It is the only place `_run_local_turn` is
    called from in production code.

    A read-only capability/server-selection check runs BEFORE Phase B. For
    SELECTED, Phase G.2 then lazily activates ONLY the selected server_id
    (Task 4/5) — no other server is touched, and nothing starts merely because
    it is installed. NONE_REQUIRED and SELECTED-and-activated are strictly
    additive — behavior is EXACTLY the pre-G.1 call to `_run_local_turn`
    (invariant 15). UNSUPPORTED/AMBIGUOUS/MULTI_SERVER_REQUIRED/a failed
    activation (e.g. MCP_SERVER_NOT_INSTALLED) return immediately: no Phase B
    shortlist, no local LLM completion, no ToolExecutor call.
    """
    if manager is None:
        # No trusted catalog available in this configuration at all.
        return _run_local_turn(manager, runtime_manager, user_text, prompt, history, system_prompt,
                               attempted_fs_requests, resume_budget=resume_budget)

    selection = select_for_request(
        user_text, manager.catalog, base_dir=manager.base_dir, managed_root=manager.managed_root,
        registry_path=manager.registry_path, runtime=runtime_manager)
    _log_capability_selection(selection)

    reply = _capability_selection_reply(selection)
    if reply is not None:
        return reply, dict(_ZERO_METRICS), None

    if selection.status == CapabilitySelectionStatus.SELECTED:
        activation = ensure_selected_server_active(
            selection, runtime_manager, manager.catalog, base_dir=manager.base_dir,
            managed_root=manager.managed_root, registry_path=manager.registry_path)
        _log_runtime_activation(runtime_manager, selection.selected_server_id, activation)
        if not activation.activated:
            if activation.error_code == MCP_SERVER_NOT_INSTALLED:
                # Phase G.3: an APPROVED provider that simply isn't installed
                # yet is a deterministic auto-provisioning opportunity, not a
                # dead end — offered only when a G.3 subsystem is wired up
                # (`_start_auto_provisioning`) AND the entry is eligible (no
                # directory grant required; Filesystem stays on its existing
                # manual/heuristic path unchanged).
                auto_manager = getattr(manager, "auto_provisioning", None)
                catalog_entry = manager.catalog.get(selection.selected_catalog_id)
                if auto_manager is not None and catalog_entry is not None:
                    capability_id = (selection.required_capabilities[0].capability_id
                                     if selection.required_capabilities else "")
                    offer = _offer_mcp_provisioning(auto_manager, capability_id, catalog_entry, user_text)
                    if offer is not None:
                        reply, request_id = offer
                        return reply, dict(_ZERO_METRICS), request_id
            return activation.message, dict(_ZERO_METRICS), None

        # Phase G.4 Defects 1/7/8 — for document_to_markdown, a HEALTHY,
        # ALREADY-installed provider (the common, repeat-use case) still needs
        # a fresh, request-bound DocumentInputAuthorization created before
        # Phase B ever sees the conversion tool. Any authorization created
        # during a PAST provisioning approval belongs to that earlier request
        # and must never be assumed valid for this one — this runs on every
        # call into this single authoritative entrypoint, fresh request or
        # resumed one alike, so resumption can never skip it (Defect 7).
        # Reuses the exact snapshot-capture + create_authorization primitives
        # the first-time-install resume path already uses — no new mechanism.
        capability_id = (selection.required_capabilities[0].capability_id
                         if selection.required_capabilities else "")
        if capability_id == _DOCUMENT_TO_MARKDOWN_CAPABILITY:
            if not _reconstruct_document_authorization(user_text):
                return ("I couldn't find a valid local document to convert for this request. "
                        "Please give the exact local file path.", dict(_ZERO_METRICS), None)

    # NONE_REQUIRED, or SELECTED-and-now-active: existing behavior, completely
    # unchanged. Phase B remains responsible for the exact-tool shortlist; the
    # selection above is not yet consumed by it (a later Phase G stage does that).
    return _run_local_turn(manager, runtime_manager, user_text, prompt, history, system_prompt,
                           attempted_fs_requests, resume_budget=resume_budget,
                           selection=selection)


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
    loop_result = tool_loop.run_local_tool_loop(
        prompt, history, system_prompt, on_tool_result=on_tool_result)
    return loop_result.text, loop_result.metrics


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
    provisioning_manager = _start_provisioning()
    _start_filesystem_access(provisioning_manager)
    _start_auto_provisioning(provisioning_manager)
    # Phase G.2: a server-keyed runtime manager, built with metadata only — no
    # MCP child process is launched here (Task 4). Every server starts lazily,
    # the first time Phase G.1 actually selects it as the preferred provider.
    runtime_manager = MultiMcpRuntimeManager(
        tool_loop.REGISTRY,
        base_dir=provisioning_manager.base_dir if provisioning_manager else None,
        managed_root=provisioning_manager.managed_root if provisioning_manager else None,
        registry_path=provisioning_manager.registry_path if provisioning_manager else None,
        validators={"filesystem": FilesystemRootValidator()},
    )

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

            # A pending filesystem-access OR auto-provisioning approval takes
            # priority over normal routing: a bare yes/no/show-plan reply
            # resolves it directly, never reaching the router or local
            # tool-selection. Which resolver runs is chosen by the pending id's
            # own prefix ("autoreq_" is Phase G.3; anything else is Phase F.1's
            # filesystem-access id shape) — never guessed from the reply text.
            # Anything else falls through to normal routing and leaves the
            # plan pending.
            if pending_fs_request_id is not None and pending_fs_request_id.startswith(
                    _AUTO_PROVISIONING_REQUEST_PREFIX):
                outcome = _resolve_auto_provisioning_reply(
                    getattr(provisioning_manager, "auto_provisioning", None), runtime_manager,
                    pending_fs_request_id, user_text)
                if outcome.matched:
                    if outcome.resumed_text is None:
                        if outcome.speak is not None:
                            console.print(f"[cyan]{outcome.speak}[/cyan]")
                            speak(outcome.speak)
                            history.append({"role": "user", "content": user_text})
                            history.append({"role": "assistant", "content": outcome.speak})
                        pending_fs_request_id = outcome.next_pending_id
                        continue

                    # Approved, installed, validated, and activated (Phase G.2's
                    # ensure_started already ran inside provision_and_activate —
                    # there is no prior session to "replace", unlike the
                    # filesystem-restart path below). Resume the ORIGINAL
                    # blocked request by re-entering the SAME authoritative
                    # pipeline as a fresh turn.
                    turn_start = time.perf_counter()
                    resumed_prompt = _enrich_with_memory(outcome.resumed_text)
                    resumed_decision = route_and_answer(resumed_prompt, history)
                    console.print(f"[dim]routing (resumed): mode={resumed_decision.mode} "
                                 f"tool={resumed_decision.tool}[/dim]")
                    if resumed_decision.mode == "local":
                        reply, _extra, pending_fs_request_id = _process_local_request_with_capability_selection(
                            provisioning_manager, runtime_manager, outcome.resumed_text, resumed_prompt,
                            history, system_prompt, attempted_fs_requests)
                    else:
                        reply, _extra = dispatch(resumed_decision, outcome.resumed_text, resumed_prompt,
                                                 history, system_prompt)
                        pending_fs_request_id = None
                    console.print(f"[cyan]{reply}[/cyan]")
                    speak(reply)
                    total_time_sec = time.perf_counter() - turn_start
                    log_turn(question=outcome.resumed_text, mode="local", tool=None,
                            prompt_tokens=0, completion_tokens=0, total_time_sec=total_time_sec)
                    history.append({"role": "user", "content": user_text})
                    history.append({"role": "assistant", "content": reply})
                    continue

            elif pending_fs_request_id is not None:
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
                        provisioning_manager, runtime_manager, directive, outcome.resumed_text, history,
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

            # Phase G.4 Defect 4 — a bare approval-workflow reply ("yes", "no",
            # "show plan", ...) with NO matching pending request (fs-access or
            # auto-provisioning — pending_fs_request_id is None here) is never
            # treated as an ordinary new request: no Router call, no capability
            # selection, no Phase B, no local LLM call, no tool execution. This
            # is what stops an LLM-fabricated approval-looking answer (which
            # never creates real pending state) from having any effect when the
            # user replies to it.
            if pending_fs_request_id is None and _is_bare_approval_token(user_text):
                reply = "No pending approval request."
                console.print(f"[cyan]{reply}[/cyan]")
                speak(reply)
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": reply})
                continue

            turn_start = time.perf_counter()

            prompt = _enrich_with_memory(user_text)
            decision = route_and_answer(prompt, history)
            console.print(f"[dim]routing: mode={decision.mode} tool={decision.tool}[/dim]")

            if decision.mode == "local":
                # Every local request (including a router fallback-to-local —
                # decision.mode is "local" either way) enters through the ONE
                # authoritative capability-selection entrypoint (Task 1).
                # NONE_REQUIRED/SELECTED fall straight through to the unchanged
                # Phase F.1 hotfix path (provisioning-if-needed, immediate
                # outside-root interception, access.add/remove restart — see
                # _run_local_turn's docstring for why "immediate" matters).
                reply, extra_metrics, pending_fs_request_id = _process_local_request_with_capability_selection(
                    provisioning_manager, runtime_manager, user_text, prompt, history, system_prompt,
                    attempted_fs_requests)
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
        # Stops every server that is currently active — one server's shutdown
        # failure never prevents attempting the rest (Task 12).
        runtime_manager.stop_all()


if __name__ == "__main__":
    main()
