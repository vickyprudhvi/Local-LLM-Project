"""Phase G.1 Task 10/14 — assistant-level capability-selection integration.

Drives assistant._process_local_request_with_capability_selection directly (the exact
function main() calls for a "local" RouteDecision) with a fixture catalog and a
FakeLLM, so these tests never need live Ollama or a real MCP process.
"""

import os

import pytest

import assistant
import tool_loop
from mcp_layer.runtime_manager import ActiveMcpRuntime
from mcp_management.capabilities import CapabilitySelectionStatus
from mcp_management.registry import STATUS_INSTALLED, InstalledServer, upsert
from tests.mcp_provisioning_helpers import catalog_dict, make_manager
from tests.test_tool_loop import FakeLLM, _final, _tool_call
from tools.base import BaseTool, ToolFailure
from tools.executor import ToolExecutor
from tools.models import MCP_CALL_FAILED, ToolPermission
from tools.registry import default_registry


def _catalog_with_g1_filesystem():
    doc = catalog_dict()
    doc["servers"]["official-filesystem"]["granular_capabilities"] = ["read_local_text_file"]
    doc["servers"]["official-filesystem"]["selection_hints"] = {
        "explicit_names": ["filesystem"],
        "actions": {"read_local_text_file": ["read"]},
    }
    from mcp_management.catalog import build_catalog

    return build_catalog(doc)


@pytest.fixture
def env(tmp_path, monkeypatch):
    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)

    manager, paths = make_manager(tmp_path, catalog=_catalog_with_g1_filesystem())
    approved_dir = tmp_path / "approved"
    approved_dir.mkdir()
    (approved_dir / "hello.txt").write_text("hi there", encoding="utf-8")
    approved_abs = os.path.realpath(str(approved_dir))

    server_root = os.path.join(paths["base_dir"], paths["managed_root"], "filesystem")
    os.makedirs(server_root, exist_ok=True)
    upsert("filesystem", InstalledServer(
        catalog_id="official-filesystem", installed_version="1.0.0",
        status=STATUS_INSTALLED, install_directory=server_root,
        configuration_path=os.path.join(server_root, "server.json"),
        installed_at="now", approved_directories=(approved_abs,)),
        None, paths["base_dir"], paths["managed_root"])

    class _StubReadTool(BaseTool):
        name = "mcp.filesystem.read_text_file"
        description = "stub"
        input_schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        permission = ToolPermission.READ
        llm_callable = True

        def execute(self, arguments):
            path = os.path.realpath(arguments["path"])
            if not (path == approved_abs or path.startswith(approved_abs + os.sep)):
                raise ToolFailure(MCP_CALL_FAILED, "outside approved roots")
            with open(path, encoding="utf-8") as f:
                return {"content": f.read()}

    reg.register(_StubReadTool())

    return {"manager": manager, "paths": paths, "approved_dir": str(approved_dir),
           "runtime": ActiveMcpRuntime(None)}


def _install(monkeypatch, responses):
    fake = FakeLLM(responses)
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)
    return fake


# ---- Scenario 1: general question -> NONE_REQUIRED, existing behavior ----

def test_general_question_selects_nothing_and_behaves_normally(env, monkeypatch):
    fake = _install(monkeypatch, [_final("Machine learning is a field of AI.")])
    captured = []
    monkeypatch.setattr(assistant, "_log_capability_selection", captured.append)

    reply, extra_metrics, pending_id = assistant._process_local_request_with_capability_selection(
        env["manager"], env["runtime"], "What is machine learning?", "What is machine learning?",
        [], "sys", set())

    assert reply == "Machine learning is a field of AI."
    assert pending_id is None
    assert len(captured) == 1
    assert captured[0].status == CapabilitySelectionStatus.NONE_REQUIRED
    assert env["runtime"].session is None  # no MCP lifecycle call happened


# ---- Scenario 2: local file read -> SELECTED, existing pipeline continues ----

def test_local_file_read_is_selected_and_pipeline_continues(env, monkeypatch):
    target = os.path.join(env["approved_dir"], "hello.txt")
    user_text = f"read '{target}'"
    fake = _install(monkeypatch, [
        _tool_call("mcp.filesystem.read_text_file", {"path": target}),
        _final("The file says: hi there"),
    ])
    captured = []
    monkeypatch.setattr(assistant, "_log_capability_selection", captured.append)

    reply, extra_metrics, pending_id = assistant._process_local_request_with_capability_selection(
        env["manager"], env["runtime"], user_text, user_text, [], "sys", set())

    assert "hi there" in reply
    assert pending_id is None
    assert captured[0].status == CapabilitySelectionStatus.SELECTED
    assert captured[0].selected_server_id == "filesystem"
    # No runtime replacement/lifecycle call: the stub session stays untouched.
    assert env["runtime"].session is None


# ---- Scenario 3: unsupported capability -> Phase B never invoked ----

def test_unsupported_capability_short_circuits_before_phase_b(env, monkeypatch):
    def _must_not_run(*args, **kwargs):
        raise AssertionError("the local tool loop (Phase B) must not run for an "
                             "impossible MCP requirement")

    monkeypatch.setattr(tool_loop, "run_local_tool_loop", _must_not_run)
    # This fixture catalog only declares the filesystem server's granular
    # capabilities — no document-conversion provider is approved, matching the
    # real production catalog's current state.
    text = "summarize C:\\Documents\\report.pdf"
    reply, extra_metrics, pending_id = assistant._process_local_request_with_capability_selection(
        env["manager"], env["runtime"], text, text, [], "sys", set())

    assert "document_to_markdown" in reply
    assert pending_id is None
    assert extra_metrics == {"prompt_tokens": 0, "completion_tokens": 0}
    assert env["runtime"].session is None  # no runtime/installation call


# ---- Scenario 4: fixture catalog WITH an approved document provider ----

def test_fixture_document_provider_is_selected_with_status_reported(tmp_path, monkeypatch):
    from mcp_management.catalog import build_catalog

    doc = catalog_dict()
    doc["servers"]["document-test"] = {
        "server_id": "document-test", "display_name": "Document Test Server",
        "description": "test fixture", "capabilities": ["filesystem"],
        "risk_category": "local_filesystem", "transport": "stdio",
        "required_runtimes": ["node"],
        "installer": {"type": "npm", "package": "@test/document-test", "version": "1.0.0",
                      "entrypoint": "dist/index.js"},
        "expected_tools": [], "default_tool_policy": {"default_permission": "denied", "tools": {}},
        "granular_capabilities": ["document_to_markdown"],
    }
    catalog = build_catalog(doc)
    manager, paths = make_manager(tmp_path, catalog=catalog)
    server_root = os.path.join(paths["base_dir"], paths["managed_root"], "document-test")
    os.makedirs(server_root, exist_ok=True)
    upsert("document-test", InstalledServer(
        catalog_id="document-test", installed_version="1.0.0",
        status=STATUS_INSTALLED, install_directory=server_root,
        configuration_path=os.path.join(server_root, "server.json"),
        installed_at="now", approved_directories=()),
        None, paths["base_dir"], paths["managed_root"])

    runtime = ActiveMcpRuntime(None)
    captured = []
    monkeypatch.setattr(assistant, "_log_capability_selection", captured.append)

    text = "summarize C:\\Documents\\report.pdf"
    reply, extra_metrics, pending_id = assistant._process_local_request_with_capability_selection(
        manager, runtime, text, text, [], "sys", set())

    assert captured[0].status == CapabilitySelectionStatus.UNSUPPORTED or \
        captured[0].selected_server_id == "document-test"
    # This fixture server IS approved, so the request must select it, not report
    # it as unavailable.
    assert captured[0].selected_server_id == "document-test"
    winner = next(c for c in captured[0].candidates if c.server_id == "document-test")
    assert winner.installed is True
    assert winner.active is False  # no runtime session exists; read-only report
    assert runtime.session is None  # selection alone never starts anything
