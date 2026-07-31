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
from mcp_layer.runtime_manager import ActiveMcpRuntime
from mcp_management.capabilities import CapabilitySelectionStatus
from mcp_management.manager import McpProvisioningManager
from mcp_management.registry import STATUS_INSTALLED, InstalledServer, upsert
from router import RouteDecision
from tests.mcp_provisioning_helpers import make_manager
from tests.test_tool_loop import FakeLLM, _final, _tool_call
from tool_loop import ToolLoopControl, ToolLoopDirective
from tools.executor import ToolExecutor
from tools.models import TOOL_NOT_IN_SHORTLIST
from tools.registry import ToolRegistry, default_registry

REPORTED_TEXT = r"summarize C:\Users\Prudhvi\OneDrive\learn stuff\Hands_on_LLM.pdf"
REPORTED_TEXT_QUOTED = r'summarize "C:\Users\Prudhvi\OneDrive\learn stuff\Hands_on_LLM.pdf"'


# ---- exact reported string: real production catalog + real registry ----

@pytest.fixture
def real_manager():
    return McpProvisioningManager()


@pytest.mark.parametrize("text", [REPORTED_TEXT, REPORTED_TEXT_QUOTED])
def test_reported_string_is_unsupported_with_full_path_before_extension(real_manager, text):
    runtime = ActiveMcpRuntime(None)

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
    assert runtime.session is None  # no MCP process, no runtime call


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
    monkeypatch.setattr(assistant.McpRuntimeManager, "replace_active_session",
                        _fail("MCP runtime manager"))

    runtime = ActiveMcpRuntime(None)
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
    runtime = ActiveMcpRuntime(None)

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
    assert runtime.session is None  # no server was started to make this selection


# ---- Task 1/8: single authoritative entrypoint ----

def test_router_fallback_to_local_still_invokes_capability_wrapper(real_manager, monkeypatch):
    """A router fallback still produces mode='local'; main()'s handling of it is
    identical to a normal local decision — both call the SAME wrapper."""
    decision = RouteDecision(mode="local", payload=REPORTED_TEXT)  # what a fallback returns
    assert decision.mode == "local"

    runtime = ActiveMcpRuntime(None)
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

    runtime = ActiveMcpRuntime(None)

    class _NoOpCoordinator:
        def __init__(self, *a, **kw):
            pass

        def replace_active_session(self, *a, **kw):
            return None

    monkeypatch.setattr(assistant, "McpRuntimeManager", _NoOpCoordinator)

    directive = ToolLoopDirective(control=ToolLoopControl.RESTART_MCP_AND_RESUME,
                                  server_id="filesystem", expected_allowed_roots=())
    reply, pending_id = assistant._restart_mcp_and_resume(
        real_manager, runtime, directive, REPORTED_TEXT, [], "sys", set(), resume_budget=1)

    assert "document_to_markdown" in reply
    assert pending_id is None


# ---- Task 7: hallucinated MCP tool rejected before ToolExecutor ----

def test_unregistered_non_namespaced_hallucination_is_already_safe(monkeypatch):
    """The exact observed hallucination, "filesystem.read_file", carries no
    "mcp." prefix at all — it is rejected by ToolExecutor's own pre-existing
    UNKNOWN_TOOL check (never registered under any name), with no execution."""
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
    assert "UNKNOWN_TOOL" in tool_messages[0]["content"]


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
    server_root = os.path.join(paths["base_dir"], paths["managed_root"], "filesystem")
    os.makedirs(server_root, exist_ok=True)
    with open(os.path.join(server_root, "server.json"), "w", encoding="utf-8") as f:
        json.dump({
            "enabled": True, "required": False, "server_id": "filesystem",
            "display_name": "Filesystem MCP Server", "transport": "stdio",
            "command": "node", "args": ["/entrypoint.js", approved_abs],
            "working_directory": "./mcp_workspaces/filesystem",
            "startup_timeout_seconds": 15, "call_timeout_seconds": 15,
            "shutdown_timeout_seconds": 5, "environment_allowlist": [],
            "tool_policy": {"default_permission": "denied", "tools": {
                "read_text_file": {"enabled": True, "permission": "read"},
            }},
        }, f)
    upsert("filesystem", InstalledServer(
        catalog_id="official-filesystem", installed_version="2026.7.10",
        status=STATUS_INSTALLED, install_directory=server_root,
        configuration_path=os.path.join(server_root, "server.json"),
        installed_at="now", approved_directories=(approved_abs,)),
        None, paths["base_dir"], paths["managed_root"])
    return approved_abs


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

    from tools.base import BaseTool, ToolFailure
    from tools.models import MCP_CALL_FAILED, ToolPermission

    class _StubReadTool(BaseTool):
        name = "mcp.filesystem.read_text_file"
        description = "stub"
        input_schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        permission = ToolPermission.READ
        llm_callable = True

        def execute(self, arguments):
            path = os.path.realpath(arguments["path"])
            if not (path == approved or path.startswith(approved + os.sep)):
                raise ToolFailure(MCP_CALL_FAILED, "outside root")
            with open(path, encoding="utf-8") as f:
                return {"content": f.read()}

    reg.register(_StubReadTool())

    target = os.path.join(approved, "hello.txt")
    user_text = f"read '{target}'"
    fake = FakeLLM([
        _tool_call("mcp.filesystem.read_text_file", {"path": target}),
        _final("The file says: hi"),
    ])
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)

    runtime = ActiveMcpRuntime(None)
    reply, metrics, pending_id = assistant._process_local_request_with_capability_selection(
        manager, runtime, user_text, user_text, [], "sys", set())

    assert "hi" in reply
    assert pending_id is None
    assert len(fake.calls) == 2  # the tool call round + the final answer — Phase B ran
