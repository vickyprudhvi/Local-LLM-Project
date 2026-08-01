"""Phase G.1 hotfix — reproduces and locks down the real CLI defect report.

Observed request:
    summarize C:\\Users\\Prudhvi\\OneDrive\\learn stuff\\Hands_on_LLM.pdf

Covers: exact-string regression (quoted and unquoted), the full Task 2 document-
intent matrix, Task 5 zero-call spies, Task 1/8 single-entrypoint enforcement
(including router fallback-to-local and resumption), and Task 7 shortlist-
membership enforcement for a hallucinated MCP tool name.
"""

import os

import pytest

import assistant
import tool_loop
from mcp_layer.runtime_manager import MultiMcpRuntimeManager
from mcp_management.capabilities import CapabilitySelectionStatus, ToolRequirement
from mcp_management.manager import McpProvisioningManager
from mcp_management.registry import STATUS_INSTALLED, InstalledServer, upsert
from router import RouteDecision
from tests.mcp_provisioning_helpers import make_manager
from tests.test_tool_loop import FakeLLM, _final, _tool_call
from tool_loop import ToolLoopControl, ToolLoopDirective, ToolLoopResultType
from tools.base import BaseTool
from tools.executor import ToolExecutor
from tools.models import (
    MCP_SELECTED_PROVIDER_TOOL_UNAVAILABLE,
    SELECTED_PROVIDER_TOOL_NOT_SHORTLISTED,
    TOOL_NOT_IN_SHORTLIST,
    TOOL_REQUIRED_NOT_SELECTED,
    ToolPermission,
)
from tools.registry import ToolRegistry, default_registry

REPORTED_TEXT = r"summarize C:\Users\Prudhvi\OneDrive\learn stuff\Hands_on_LLM.pdf"
REPORTED_TEXT_QUOTED = r'summarize "C:\Users\Prudhvi\OneDrive\learn stuff\Hands_on_LLM.pdf"'


# ---- exact reported string: real production catalog + real registry ----

@pytest.fixture
def real_manager():
    import json
    import os

    from mcp_management.catalog import build_catalog, load_catalog

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "config", "mcp_catalog.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Phase G.4 regression isolation: these G.1 hotfix tests verify behavior
    # when no document provider is installed.  Keep the MarkItDown entry disabled
    # in this fixture so enabling the production entry does not change the test
    # semantics.
    data["servers"]["official-markitdown"]["enabled"] = False
    return McpProvisioningManager(catalog=build_catalog(data))


@pytest.mark.parametrize("text", [REPORTED_TEXT, REPORTED_TEXT_QUOTED])
def test_reported_string_is_unsupported_with_full_path_before_extension(real_manager, text):
    runtime = MultiMcpRuntimeManager(tool_loop.REGISTRY)

    def _boom(*a, **kw):
        raise AssertionError("Phase B (the local tool loop) must not run")

    tool_loop.run_local_tool_loop, saved = _boom, tool_loop.run_local_tool_loop
    try:
        reply, metrics, pending_id = assistant._process_local_request_with_capability_selection(
            real_manager, runtime, text, text, [], "sys", set())
    finally:
        tool_loop.run_local_tool_loop = saved

    assert "document_to_markdown" in reply
    assert "no approved MCP server currently provides it" in reply
    assert metrics == {"prompt_tokens": 0, "completion_tokens": 0}
    assert pending_id is None
    assert runtime.get_session("filesystem") is None  # no MCP process, no runtime call


def test_reported_string_full_path_is_not_truncated():
    from mcp_management.capability_detector import _find_local_paths

    paths, types, has_url = _find_local_paths(REPORTED_TEXT)
    assert paths == [r"C:\Users\Prudhvi\OneDrive\learn stuff\Hands_on_LLM.pdf"]
    assert types == ["windows_absolute"]
    assert has_url is False


# ---- Task 5: zero calls to shortlist/model/executor/classifier/planner/runtime ----

def test_unsupported_document_request_makes_zero_downstream_calls(real_manager, monkeypatch):
    def _fail(name):
        def _inner(*a, **kw):
            raise AssertionError(f"{name} must not be called for an unsupported MCP capability")
        return _inner

    monkeypatch.setattr(ToolRegistry, "shortlist_tools", _fail("Phase B shortlist construction"))
    monkeypatch.setattr(tool_loop, "ask_local_raw", _fail("local model completion"))
    monkeypatch.setattr(ToolExecutor, "execute", _fail("ToolExecutor"))
    monkeypatch.setattr("mcp_management.access_classifier.classify_outside_root_failure",
                        _fail("filesystem access classifier"))
    monkeypatch.setattr(McpProvisioningManager, "prepare_filesystem_access_plan",
                        _fail("filesystem access planner"))
    monkeypatch.setattr(MultiMcpRuntimeManager, "ensure_started", _fail("MCP runtime manager"))
    monkeypatch.setattr(MultiMcpRuntimeManager, "replace_session", _fail("MCP runtime manager"))

    runtime = MultiMcpRuntimeManager(tool_loop.REGISTRY)
    reply, metrics, pending_id = assistant._process_local_request_with_capability_selection(
        real_manager, runtime, REPORTED_TEXT, REPORTED_TEXT, [], "sys", set())

    assert "document_to_markdown" in reply
    assert pending_id is None


# ---- Task 2: document-intent matrix ----

@pytest.fixture(scope="module")
def catalog():
    from mcp_management.catalog import load_catalog
    return load_catalog()


@pytest.fixture(scope="module")
def detector():
    from mcp_management.capability_detector import McpCapabilityDetector
    return McpCapabilityDetector()


@pytest.mark.parametrize("text,expected", [
    (r"summarize C:\Docs\report.pdf", "document_to_markdown"),
    (r"review C:\Docs\slides.pptx", "document_to_markdown"),
    (r"analyze C:\Docs\budget.xlsx", "document_to_markdown"),
    (r"extract text from C:\Docs\contract.docx", "document_to_markdown"),
    (r"copy C:\Docs\report.pdf to C:\Archive", "manage_local_files"),
    (r"read C:\Docs\notes.txt", "read_local_text_file"),
])
def test_document_intent_matrix(detector, catalog, text, expected):
    reqs = detector.detect(text, catalog)
    assert {r.capability_id for r in reqs} == {expected}


@pytest.mark.parametrize("text", [
    "What is a PDF?",
    "read README.md",
])
def test_document_intent_matrix_none_required(detector, catalog, text):
    assert detector.detect(text, catalog) == ()


def test_remote_document_url_is_not_filesystem(detector, catalog):
    reqs = detector.detect("summarize https://example.com/report.pdf", catalog)
    assert all(r.capability_id != "read_local_text_file" for r in reqs)
    assert all(r.capability_id != "manage_local_files" for r in reqs)


# ---- Task 4: fixture catalog with an approved document provider -> SELECTED ----

def test_fixture_document_provider_is_selected_no_tool_no_server(tmp_path):
    from mcp_management.catalog import build_catalog

    doc = {
        "catalog_version": 1,
        "servers": {
            "document-test": {
                "server_id": "document-test", "display_name": "Document Test Server",
                "description": "test fixture", "capabilities": ["filesystem"],
                "risk_category": "local_filesystem", "transport": "stdio",
                "required_runtimes": ["node"],
                "installer": {"type": "npm", "package": "@test/document-test", "version": "1.0.0",
                              "entrypoint": "dist/index.js"},
                "expected_tools": [], "default_tool_policy": {"default_permission": "denied", "tools": {}},
                "granular_capabilities": ["document_to_markdown"],
            }
        },
    }
    catalog = build_catalog(doc)
    manager, paths = make_manager(tmp_path, catalog=catalog)
    runtime = MultiMcpRuntimeManager(tool_loop.REGISTRY)

    def _boom(*a, **kw):
        raise AssertionError("Phase B must not run when SELECTED — G.1 never picks a tool")

    saved = tool_loop.run_local_tool_loop
    tool_loop.run_local_tool_loop = _boom
    try:
        from mcp_management.capability_service import select_for_request
        selection = select_for_request(
            REPORTED_TEXT, catalog, base_dir=paths["base_dir"], managed_root=paths["managed_root"],
            registry_path=None, runtime=runtime)
    finally:
        tool_loop.run_local_tool_loop = saved

    assert selection.status == CapabilitySelectionStatus.SELECTED
    assert selection.selected_server_id == "document-test"
    assert runtime.get_session("document-test") is None  # no server was started to make this selection


# ---- Task 1/8: single authoritative entrypoint ----

def test_router_fallback_to_local_still_invokes_capability_wrapper(real_manager, monkeypatch):
    """A router fallback still produces mode='local'; main()'s handling of it is
    identical to a normal local decision — both call the SAME wrapper."""
    decision = RouteDecision(mode="local", payload=REPORTED_TEXT)  # what a fallback returns
    assert decision.mode == "local"

    runtime = MultiMcpRuntimeManager(tool_loop.REGISTRY)
    called = []
    real_select = assistant.select_for_request

    def _spy(*a, **kw):
        called.append(True)
        return real_select(*a, **kw)

    monkeypatch.setattr(assistant, "select_for_request", _spy)
    assistant._process_local_request_with_capability_selection(
        real_manager, runtime, decision.payload, decision.payload, [], "sys", set())
    assert called  # capability selection ran for this fallback-shaped decision


def test_resumed_request_reinvokes_capability_selection(real_manager, monkeypatch):
    """_restart_mcp_and_resume's resumed call must re-run capability selection —
    a resumed document-conversion request with no provider must still be
    UNSUPPORTED, never falling through into Phase B or Filesystem access."""
    monkeypatch.setattr(assistant, "route_and_answer",
                        lambda prompt, history: RouteDecision(mode="local", tool=None))

    def _boom(*a, **kw):
        raise AssertionError("Phase B must not run for a resumed unsupported request")

    monkeypatch.setattr(tool_loop, "run_local_tool_loop", _boom)

    runtime = MultiMcpRuntimeManager(tool_loop.REGISTRY)
    monkeypatch.setattr(runtime, "replace_session", lambda *a, **kw: None)

    directive = ToolLoopDirective(control=ToolLoopControl.RESTART_MCP_AND_RESUME,
                                  server_id="filesystem", expected_allowed_roots=())
    reply, pending_id = assistant._restart_mcp_and_resume(
        real_manager, runtime, directive, REPORTED_TEXT, [], "sys", set(), resume_budget=1)

    assert "document_to_markdown" in reply
    assert pending_id is None


# ---- Task 7: hallucinated MCP tool rejected before ToolExecutor ----

def test_unregistered_non_namespaced_hallucination_is_rejected_before_executor(monkeypatch):
    """Phase G.4 Defect 5 — the exact observed hallucination,
    "filesystem.read_file", carries no "mcp." prefix at all. It must be
    rejected BEFORE ToolExecutor ever runs (TOOL_NOT_IN_SHORTLIST), never
    reach the executor's UNKNOWN_TOOL path — "do not let invalid selection
    degrade into UNKNOWN_TOOL"."""
    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)

    fake = FakeLLM([
        _tool_call("filesystem.read_file", {"path": r"C:\approved\hello.txt"}),
        _final("done"),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    text, metrics = tool_loop.run_local_tool_loop("read something", [], "sys")
    assert text == "done"
    tool_messages = [m for m in fake.calls[1]["messages"] if m.get("role") == "tool"]
    assert "TOOL_NOT_IN_SHORTLIST" in tool_messages[0]["content"]
    assert "UNKNOWN_TOOL" not in tool_messages[0]["content"]


def test_registered_but_not_offered_mcp_tool_is_rejected(monkeypatch):
    """A REGISTERED mcp.* tool that simply wasn't in this round's shortlist is
    rejected before ToolExecutor runs — not just outright-unregistered names."""
    from tools.base import BaseTool
    from tools.models import ToolPermission

    class _NotOfferedTool(BaseTool):
        name = "mcp.filesystem.convert_pdf"
        description = "a real registered tool the shortlist simply did not offer"
        input_schema = {"type": "object", "properties": {}}
        permission = ToolPermission.READ
        llm_callable = True

        def execute(self, arguments):
            raise AssertionError("must never execute")

    reg = default_registry()
    reg.register(_NotOfferedTool())
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)
    # Deterministically exclude the tool from what's offered, independent of
    # lexical scoring — this test is about ENFORCEMENT, not ranking.
    monkeypatch.setattr(reg, "shortlist_tools", lambda *a, **kw: [
        d for d in reg.enabled_definitions() if d.name != "mcp.filesystem.convert_pdf"
    ][:1])

    fake = FakeLLM([
        _tool_call("mcp.filesystem.convert_pdf", {"path": r"C:\approved\report.pdf"}),
        _final("done"),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    text, metrics = tool_loop.run_local_tool_loop("convert this pdf", [], "sys")
    assert text == "done"
    tool_messages = [m for m in fake.calls[1]["messages"] if m.get("role") == "tool"]
    assert TOOL_NOT_IN_SHORTLIST in tool_messages[0]["content"]


# ---- Task 11: valid Filesystem behavior is not regressed ----

def _install_stub_server(paths, approved_dir):
    import json

    approved_abs = os.path.realpath(approved_dir)
    from tests.mcp_provisioning_helpers import FIXTURE_SERVER

    server_root = os.path.join(paths["base_dir"], paths["managed_root"], "filesystem")
    os.makedirs(server_root, exist_ok=True)
    workspace = os.path.join(paths["base_dir"], "mcp_workspaces", "filesystem")
    os.makedirs(workspace, exist_ok=True)
    with open(os.path.join(server_root, "server.json"), "w", encoding="utf-8") as f:
        json.dump({
            "enabled": True, "required": False, "server_id": "filesystem",
            "display_name": "Filesystem MCP Server", "transport": "stdio",
            "command": "node", "args": [FIXTURE_SERVER, approved_abs],
            "working_directory": workspace,
            "startup_timeout_seconds": 15, "call_timeout_seconds": 15,
            "shutdown_timeout_seconds": 5, "environment_allowlist": [],
            "tool_policy": {"default_permission": "denied", "tools": {
                "read_text_file": {"enabled": True, "permission": "read"},
                "list_allowed_directories": {"enabled": True, "permission": "read"},
            }},
        }, f)
    upsert("filesystem", InstalledServer(
        catalog_id="official-filesystem", installed_version="2026.7.10",
        status=STATUS_INSTALLED, install_directory=server_root,
        configuration_path=os.path.join(server_root, "server.json"),
        installed_at="now", approved_directories=(approved_abs,)),
        None, paths["base_dir"], paths["managed_root"])
    return approved_abs


@pytest.mark.skipif(not __import__("tests.mcp_provisioning_helpers", fromlist=["node_available"])
                    .node_available(), reason="node/npm not available")
def test_valid_local_read_still_selects_filesystem_and_reaches_phase_b(tmp_path, monkeypatch):
    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)

    manager, paths = make_manager(tmp_path)
    approved_dir = tmp_path / "approved"
    approved_dir.mkdir()
    approved = _install_stub_server(paths, str(approved_dir))
    (approved_dir / "hello.txt").write_text("hi", encoding="utf-8")

    target = os.path.join(approved, "hello.txt")
    user_text = f"read '{target}'"
    fake = FakeLLM([
        _tool_call("mcp.filesystem.read_text_file", {"path": target}),
        _final("The file says: hi"),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    runtime = MultiMcpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])
    reply, metrics, pending_id = assistant._process_local_request_with_capability_selection(
        manager, runtime, user_text, user_text, [], "sys", set())

    assert "hi" in reply
    assert pending_id is None
    assert len(fake.calls) == 2  # the tool call round + the final answer — Phase B ran
    session = runtime.get_session("filesystem")
    assert session is not None
    session.shutdown()


# ---- Phase G.4 fail-close: REQUIRED tool-selection semantics ----

class _StubMcpTool(BaseTool):
    name = "mcp.stub-server.read_text_file"
    description = "read a file"
    input_schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    permission = ToolPermission.READ
    llm_callable = True

    def execute(self, arguments):
        return {"content": "hello"}


def _stub_registry(monkeypatch):
    """Registry with a single MCP stub provider tool; caller may override shortlist."""
    reg = default_registry()
    reg.register(_StubMcpTool())
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)
    return reg


def test_required_no_provider_tools_fails_closed(monkeypatch):
    """A: REQUIRED + preferred provider with zero enabled tools -> controlled error."""
    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)

    result = tool_loop.run_local_tool_loop(
        "read x", [], "sys",
        tool_requirement=ToolRequirement.REQUIRED,
        preferred_mcp_server_id="missing-server")

    assert result.result_type == ToolLoopResultType.NO_TOOL_INVALID_REQUIRED
    assert MCP_SELECTED_PROVIDER_TOOL_UNAVAILABLE in result.text
    assert result.metrics == {"prompt_tokens": 0, "completion_tokens": 0}


def test_required_provider_tools_not_shortlisted_gets_injected(monkeypatch):
    """B: REQUIRED + provider exists but its tools missed Phase B -> one is injected."""
    _stub_registry(monkeypatch)
    reg = tool_loop.REGISTRY
    # Deterministically exclude the stub tool from the lexical shortlist.
    monkeypatch.setattr(reg, "shortlist_tools", lambda *a, **kw: [
        d for d in reg.enabled_definitions()
        if not d.name.startswith("mcp.stub-server.")
    ][:5])

    fake = FakeLLM([
        _tool_call("mcp.stub-server.read_text_file", {"path": "x"}),
        _final("File says hello."),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    result = tool_loop.run_local_tool_loop(
        "read x", [], "sys",
        tool_requirement=ToolRequirement.REQUIRED,
        preferred_mcp_server_id="stub-server")

    assert result.result_type == ToolLoopResultType.TOOL_SELECTED
    assert result.text == "File says hello."
    # The injected provider tool is present in the first LLM call's tools.
    first_tools = fake.calls[0]["tools"] or []
    offered_names = [t.get("function", {}).get("name") for t in first_tools]
    assert "mcp.stub-server.read_text_file" in offered_names


def test_required_model_refusal_triggers_one_retry_then_error(monkeypatch):
    """C: REQUIRED + model refuses to select a tool -> one nudge, then controlled error."""
    _stub_registry(monkeypatch)
    fake = FakeLLM([
        _final("I can answer directly."),
        _final("Still no tool."),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    result = tool_loop.run_local_tool_loop(
        "read x", [], "sys",
        tool_requirement=ToolRequirement.REQUIRED,
        preferred_mcp_server_id="stub-server")

    assert result.result_type == ToolLoopResultType.NO_TOOL_INVALID_REQUIRED
    assert TOOL_REQUIRED_NOT_SELECTED in result.text
    assert result.retry_count == 1
    assert len(fake.calls) == 2
    # The nudge is appended as a user message on the retry call.
    assert any(
        "requires tool-backed data" in m.get("content", "")
        for m in fake.calls[1]["messages"]
        if m.get("role") == "user"
    )
    # The model's initial direct answer is discarded, never surfaced to the user.
    assert "I can answer directly" not in result.text
    assert "Still no tool" not in result.text


def test_required_model_selects_tool_on_retry_succeeds(monkeypatch):
    """D: REQUIRED + model refuses once, then selects the tool on retry -> success."""
    _stub_registry(monkeypatch)
    fake = FakeLLM([
        _final("I can answer directly."),
        _tool_call("mcp.stub-server.read_text_file", {"path": "x"}),
        _final("File says hello."),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    result = tool_loop.run_local_tool_loop(
        "read x", [], "sys",
        tool_requirement=ToolRequirement.REQUIRED,
        preferred_mcp_server_id="stub-server")

    assert result.result_type == ToolLoopResultType.TOOL_SELECTED
    assert result.text == "File says hello."
    assert result.retry_count == 1
    assert result.selected_tool_name == "mcp.stub-server.read_text_file"


def test_required_immediate_tool_selection_succeeds(monkeypatch):
    """E: REQUIRED + model selects the tool immediately -> no retry, success."""
    _stub_registry(monkeypatch)
    fake = FakeLLM([
        _tool_call("mcp.stub-server.read_text_file", {"path": "x"}),
        _final("File says hello."),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    result = tool_loop.run_local_tool_loop(
        "read x", [], "sys",
        tool_requirement=ToolRequirement.REQUIRED,
        preferred_mcp_server_id="stub-server")

    assert result.result_type == ToolLoopResultType.TOOL_SELECTED
    assert result.text == "File says hello."
    assert result.retry_count == 0
    assert result.selected_tool_name == "mcp.stub-server.read_text_file"


def test_none_request_direct_answer_is_valid(monkeypatch):
    """F: NONE/OPTIONAL request + model answers directly -> valid final answer."""
    _stub_registry(monkeypatch)
    fake = FakeLLM([_final("Plain answer.")])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    result = tool_loop.run_local_tool_loop("hello", [], "sys")

    assert result.result_type == ToolLoopResultType.NO_TOOL_VALID
    assert result.text == "Plain answer."
    assert result.retry_count == 0


def test_required_direct_answer_after_tool_success_is_allowed(monkeypatch):
    """G: REQUIRED + tool already succeeded -> direct final answer is allowed."""
    _stub_registry(monkeypatch)
    fake = FakeLLM([
        _tool_call("mcp.stub-server.read_text_file", {"path": "x"}),
        _final("Done."),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    result = tool_loop.run_local_tool_loop(
        "read x", [], "sys",
        tool_requirement=ToolRequirement.REQUIRED,
        preferred_mcp_server_id="stub-server")

    assert result.result_type == ToolLoopResultType.TOOL_SELECTED
    assert result.text == "Done."
    assert result.retry_count == 0
