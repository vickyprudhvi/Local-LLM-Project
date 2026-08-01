"""Phase G.4 live-orchestration defect fixes — focused regression tests.

Reproduces (at the assistant.py orchestration level, without a real installed
MarkItDown package) the exact live failure reported: a document_to_markdown
request against an ALREADY-installed, healthy provider used to reach Phase B
with no authorization at all, causing MCP_DOCUMENT_AUTHORIZATION_REQUIRED to
leak to the local LLM, which then fabricated a Filesystem approval message
that a later bare "yes" incorrectly turned into a normal routed request
selecting a non-existent tool.

Covers scenarios A, C, D, E, G from the defect report; F is covered by
tests/test_capability_gating_regression.py and tests/test_tool_loop.py; B and
H are covered by tests/test_mcp_auto_provisioning_document_auth.py and
tests/test_mcp_document_authorization.py / test_mcp_invocation_policy.py.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional, Tuple
from unittest.mock import MagicMock

import pytest

import assistant
import tool_loop
from mcp_layer.tool import McpTool
from mcp_management.capabilities import CapabilityRequirement, CapabilitySelectionStatus, ToolRequirement
from mcp_management.document_authorization import DocumentAuthorizationStore
from mcp_management.runtime_activation import ServerActivationResult
from tests.test_tool_loop import FakeLLM, _final, _tool_call
from tools.executor import ToolExecutor
from tools.models import MCP_DOCUMENT_AUTHORIZATION_REQUIRED, ToolPermission
from tools.registry import default_registry


@pytest.fixture(autouse=True)
def _reset_default_store():
    original = DocumentAuthorizationStore._default
    DocumentAuthorizationStore._default = DocumentAuthorizationStore()
    yield
    DocumentAuthorizationStore._default = original


def _selected_markitdown_selection(user_text):
    return SimpleNamespace(
        status=CapabilitySelectionStatus.SELECTED,
        required_capabilities=(CapabilityRequirement(capability_id="document_to_markdown", confidence=0.9),),
        selected_server_id="markitdown",
        selected_catalog_id="official-markitdown",
        candidates=(),
        explanation="Selected 'markitdown' for capability match (document_to_markdown).",
        error_code=None,
        tool_requirement=ToolRequirement.REQUIRED,
        preferred_mcp_server_id="markitdown",
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)

    client = MagicMock()
    client.call_tool.return_value = {"text": "# Hands on LLM\nG4_VERIFY_PDF_2026"}
    tool = McpTool(
        registry_name="mcp.markitdown.convert_to_markdown",
        remote_name="convert_to_markdown",
        description="Convert a local document to markdown.",
        input_schema={"type": "object", "properties": {"uri": {"type": "string"}}},
        permission=ToolPermission.READ,
        client=client,
        invocation_policy={"argument_mode": "exact_file_uri"},
    )
    reg.register(tool)

    doc = tmp_path / "Hands_on_LLM.pdf"
    doc.write_bytes(b"%PDF-1.4 fixture content")

    manager = SimpleNamespace(
        catalog=None, base_dir=str(tmp_path), managed_root=None, registry_path=None,
        # _run_local_turn calls _provision_if_needed(manager, user_text) unconditionally
        # (a peer Phase F legacy heuristic, orthogonal to Phase G.1 selection); this
        # stub reports "nothing to provision" so it is a no-op for these tests.
        begin_request=lambda user_text: (SimpleNamespace(requires_mcp=False, error_code=None), None),
    )
    runtime_manager = SimpleNamespace()

    monkeypatch.setattr(assistant, "select_for_request",
                        lambda user_text, *a, **kw: _selected_markitdown_selection(user_text))
    monkeypatch.setattr(assistant, "ensure_selected_server_active",
                        lambda *a, **kw: ServerActivationResult(activated=True, server_id="markitdown"))

    return {"manager": manager, "runtime_manager": runtime_manager, "reg": reg,
           "client": client, "doc_path": str(doc)}


# ---- Scenario A: installed MarkItDown request ----

def test_authorization_created_before_phase_b_and_conversion_succeeds(env, monkeypatch):
    user_text = f"summarize {env['doc_path']}"
    fake = FakeLLM([
        _tool_call("mcp.markitdown.convert_to_markdown", {"uri": env["doc_path"]}),
        _final("The document is about hands-on LLMs. G4_VERIFY_PDF_2026"),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    reply, metrics, pending_id = assistant._process_local_request_with_capability_selection(
        env["manager"], env["runtime_manager"], user_text, user_text, [], "sys", set())

    assert pending_id is None
    assert "G4_VERIFY_PDF_2026" in reply
    # The trusted local file:// URI was sent, never the model's raw path string.
    sent_args = env["client"].call_tool.call_args[0][1]
    assert sent_args["uri"].startswith("file:")
    assert env["client"].call_tool.call_count == 1
    # Authorization was consumed after the single call attempt.
    from mcp_management.document_authorization import DocumentAuthorizationStore

    assert DocumentAuthorizationStore.default().find_and_reserve_for_path(env["doc_path"]) is None


def test_no_filesystem_plan_created_for_markitdown_request(env, monkeypatch):
    """Scenario G — even on a real conversion, nothing resembling a Filesystem
    access plan is ever created for a document_to_markdown request."""
    user_text = f"summarize {env['doc_path']}"
    fake = FakeLLM([
        _tool_call("mcp.markitdown.convert_to_markdown", {"uri": env["doc_path"]}),
        _final("Summary. G4_VERIFY_PDF_2026"),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    _reply, _metrics, pending_id = assistant._process_local_request_with_capability_selection(
        env["manager"], env["runtime_manager"], user_text, user_text, [], "sys", set())
    assert pending_id is None  # no fsreq_/autoreq_ pending plan of any kind


# ---- Scenario C: missing/consumed authorization is recovered once, never shown to the LLM ----

def test_control_plane_authorization_error_is_recovered_without_llm_exposure(env, monkeypatch):
    """The FIRST tool call fails with the raw control-plane code (simulating a
    stale/consumed authorization); the retry must succeed with a FRESH
    authorization and the model must never see the raw error code."""
    user_text = f"summarize {env['doc_path']}"
    call_count = {"n": 0}
    real_call_tool = env["client"].call_tool

    def _flaky_call_tool(name, arguments, timeout=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Consume the just-created authorization out from under the call to
            # force MCP_DOCUMENT_AUTHORIZATION_REQUIRED on the (simulated) retry
            # inside McpTool.execute's own invocation-policy resolution.
            from mcp_management.document_authorization import DocumentAuthorizationStore

            store = DocumentAuthorizationStore.default()
            auth = store.find_and_reserve_for_path(env["doc_path"])
            if auth is not None:
                store.consume_authorization(auth.auth_id)
            raise assistant.McpError(MCP_DOCUMENT_AUTHORIZATION_REQUIRED, "forced")
        return {"text": "Summary after recovery. G4_VERIFY_PDF_2026"}

    env["client"].call_tool.side_effect = _flaky_call_tool

    fake = FakeLLM([
        _tool_call("mcp.markitdown.convert_to_markdown", {"uri": env["doc_path"]}),
        _tool_call("mcp.markitdown.convert_to_markdown", {"uri": env["doc_path"]}),
        _final("Summary after recovery. G4_VERIFY_PDF_2026"),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    reply, _metrics, pending_id = assistant._process_local_request_with_capability_selection(
        env["manager"], env["runtime_manager"], user_text, user_text, [], "sys", set())

    assert pending_id is None
    assert "G4_VERIFY_PDF_2026" in reply
    # The model was never shown the raw internal error code in any message
    # the FakeLLM received.
    for call in fake.calls:
        for m in call.get("messages", []):
            content = m.get("content") or ""
            assert MCP_DOCUMENT_AUTHORIZATION_REQUIRED not in str(content)


def test_unrecoverable_authorization_failure_returns_controlled_error(env, monkeypatch):
    """When even a fresh authorization cannot resolve the failure (simulated by
    always failing), a controlled internal-error message is returned — never a
    raw internal code, never a Filesystem fallback, never a second retry."""
    env["client"].call_tool.side_effect = assistant.McpError(
        MCP_DOCUMENT_AUTHORIZATION_REQUIRED, "forced, always fails")

    user_text = f"summarize {env['doc_path']}"
    fake = FakeLLM([
        _tool_call("mcp.markitdown.convert_to_markdown", {"uri": env["doc_path"]}),
        _tool_call("mcp.markitdown.convert_to_markdown", {"uri": env["doc_path"]}),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    reply, _metrics, pending_id = assistant._process_local_request_with_capability_selection(
        env["manager"], env["runtime_manager"], user_text, user_text, [], "sys", set())

    assert pending_id is None
    assert MCP_DOCUMENT_AUTHORIZATION_REQUIRED not in reply
    assert "internal" in reply.lower() or "try again" in reply.lower()
    # Exactly two attempts total (original + one reconstruction retry).
    assert env["client"].call_tool.call_count == 2


# ---- Scenario D/E: fabricated approval text and bare tokens with no pending plan ----

def test_fabricated_approval_text_creates_no_pending_state():
    """Scenario D — free-form LLM text that merely LOOKS like an approval
    prompt never creates real pending state anywhere the resolvers can find."""
    fabricated = (
        "Filesystem access change...\nReply yes to approve, no to decline."
    )
    # No deterministic offer function was ever called, so there is nothing to
    # resolve: both resolvers report no match for any request id.
    fs_outcome = assistant._resolve_filesystem_access_reply(None, "fsreq_doesnotexist", "yes")
    ap_outcome = assistant._resolve_auto_provisioning_reply(None, None, "autoreq_doesnotexist", "yes")
    assert fs_outcome.matched is False
    assert ap_outcome.matched is False
    assert fabricated  # the fabricated text itself is never parsed into state


def test_bare_yes_without_pending_plan_is_a_bare_token():
    assert assistant._is_bare_approval_token("yes")
    assert assistant._is_bare_approval_token(" Yes! ")
    assert assistant._is_bare_approval_token("show plan")
    assert assistant._is_bare_approval_token("no")
    assert not assistant._is_bare_approval_token("yes please summarize the other file too")
    assert not assistant._is_bare_approval_token("summarize C:\\file.pdf")
