"""Phase G.1/G.2 — assistant-level capability-selection + lazy-activation
integration.

Drives assistant._process_local_request_with_capability_selection directly (the
exact function main() calls for a "local" RouteDecision) with a fixture catalog
and a FakeLLM, so these tests never need live Ollama. SELECTED scenarios use the
REAL Node fixture filesystem server (Task 15: no fake server in production, but a
genuine child process in tests) since Phase G.2 lazily activates a real session
before Phase B runs.
"""

import json
import os

import pytest

import assistant
import tool_loop
from mcp_layer.runtime_manager import MultiMcpRuntimeManager
from mcp_management.capabilities import CapabilitySelectionStatus
from mcp_management.registry import STATUS_INSTALLED, InstalledServer, upsert
from tests.mcp_provisioning_helpers import FIXTURE_SERVER, catalog_dict, make_manager, node_available
from tests.test_tool_loop import FakeLLM, _final, _tool_call
from tools.executor import ToolExecutor
from tools.registry import default_registry

pytestmark = pytest.mark.skipif(not node_available(), reason="node/npm not available")


def _catalog_with_g1_filesystem():
    doc = catalog_dict()
    doc["servers"]["official-filesystem"]["granular_capabilities"] = ["read_local_text_file"]
    doc["servers"]["official-filesystem"]["selection_hints"] = {
        "explicit_names": ["filesystem"],
        "actions": {"read_local_text_file": ["read"]},
    }
    from mcp_management.catalog import build_catalog

    return build_catalog(doc)


def _write_filesystem_config(paths, roots):
    server_root = os.path.join(paths["base_dir"], paths["managed_root"], "filesystem")
    os.makedirs(server_root, exist_ok=True)
    workspace = os.path.join(paths["base_dir"], "mcp_workspaces", "filesystem")
    os.makedirs(workspace, exist_ok=True)
    raw = {
        "enabled": True, "required": False, "server_id": "filesystem", "transport": "stdio",
        "command": "node", "args": [FIXTURE_SERVER, *roots], "working_directory": workspace,
        "startup_timeout_seconds": 15, "call_timeout_seconds": 10, "shutdown_timeout_seconds": 5,
        "environment_allowlist": [],
        "tool_policy": {"default_permission": "denied", "tools": {
            "list_allowed_directories": {"enabled": True, "permission": "read"},
            "read_text_file": {"enabled": True, "permission": "read"},
        }},
    }
    config_path = os.path.join(server_root, "server.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    upsert("filesystem", InstalledServer(
        catalog_id="official-filesystem", installed_version="1.0.0", status=STATUS_INSTALLED,
        install_directory=server_root, configuration_path=config_path, installed_at="now",
        approved_directories=tuple(roots)), None, paths["base_dir"], paths["managed_root"])


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
    _write_filesystem_config(paths, [approved_abs])

    runtime_manager = MultiMcpRuntimeManager(reg, base_dir=paths["base_dir"],
                                             managed_root=paths["managed_root"])
    return {"manager": manager, "paths": paths, "approved_dir": approved_abs,
           "runtime": runtime_manager, "reg": reg}


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
    assert env["runtime"].get_session("filesystem") is None  # no MCP lifecycle call happened


# ---- Scenario 2/3: local file read -> SELECTED, lazy activation, Phase B runs ----

def test_local_file_read_is_selected_activates_lazily_and_reaches_phase_b(env, monkeypatch):
    target = os.path.join(env["approved_dir"], "hello.txt")
    user_text = f"read '{target}'"
    fake = _install(monkeypatch, [
        _tool_call("mcp.filesystem.read_text_file", {"path": target}),
        _final("The file says: hi there"),
    ])
    captured = []
    monkeypatch.setattr(assistant, "_log_capability_selection", captured.append)

    assert env["runtime"].get_session("filesystem") is None  # inactive before the request

    reply, extra_metrics, pending_id = assistant._process_local_request_with_capability_selection(
        env["manager"], env["runtime"], user_text, user_text, [], "sys", set())

    assert "hi there" in reply
    assert pending_id is None
    assert captured[0].status == CapabilitySelectionStatus.SELECTED
    assert captured[0].selected_server_id == "filesystem"
    session = env["runtime"].get_session("filesystem")
    assert session is not None
    assert session.health.state.value == "healthy"
    session.shutdown()


def test_second_request_reuses_the_same_session(env, monkeypatch):
    target = os.path.join(env["approved_dir"], "hello.txt")
    user_text = f"read '{target}'"
    fake = _install(monkeypatch, [
        _tool_call("mcp.filesystem.read_text_file", {"path": target}),
        _final("first answer"),
        _tool_call("mcp.filesystem.read_text_file", {"path": target}),
        _final("second answer"),
    ])

    assistant._process_local_request_with_capability_selection(
        env["manager"], env["runtime"], user_text, user_text, [], "sys", set())
    first_session = env["runtime"].get_session("filesystem")
    first_pid = first_session.client._proc.pid

    assistant._process_local_request_with_capability_selection(
        env["manager"], env["runtime"], user_text, user_text, [], "sys", set())
    second_session = env["runtime"].get_session("filesystem")

    assert second_session is first_session
    assert second_session.client._proc.pid == first_pid
    second_session.shutdown()


# ---- Scenario: unsupported capability -> Phase B never invoked, no activation ----

def test_unsupported_capability_short_circuits_before_phase_b_and_activation(env, monkeypatch):
    def _must_not_run(*args, **kwargs):
        raise AssertionError("the local tool loop (Phase B) must not run for an "
                             "impossible MCP requirement")

    monkeypatch.setattr(tool_loop, "run_local_tool_loop", _must_not_run)
    text = "summarize C:\\Documents\\report.pdf"
    reply, extra_metrics, pending_id = assistant._process_local_request_with_capability_selection(
        env["manager"], env["runtime"], text, text, [], "sys", set())

    assert "document_to_markdown" in reply
    assert pending_id is None
    assert extra_metrics == {"prompt_tokens": 0, "completion_tokens": 0}
    assert env["runtime"].get_session("filesystem") is None  # no activation attempted


# ---- Scenario: approved provider selected but NOT installed ----

def test_approved_uninstalled_provider_reports_not_installed_no_phase_b(tmp_path, monkeypatch):
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
    # No registry entry, no server.json: a genuinely uninstalled approved provider.

    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)

    def _must_not_run(*a, **kw):
        raise AssertionError("Phase B must not run for an uninstalled selected provider")

    monkeypatch.setattr(tool_loop, "run_local_tool_loop", _must_not_run)

    runtime_manager = MultiMcpRuntimeManager(reg, base_dir=paths["base_dir"],
                                             managed_root=paths["managed_root"])
    captured = []
    monkeypatch.setattr(assistant, "_log_capability_selection", captured.append)

    text = "summarize C:\\Documents\\report.pdf"
    reply, extra_metrics, pending_id = assistant._process_local_request_with_capability_selection(
        manager, runtime_manager, text, text, [], "sys", set())

    assert captured[0].status == CapabilitySelectionStatus.SELECTED
    assert captured[0].selected_server_id == "document-test"
    assert "document-test" in reply
    assert "not installed" in reply
    assert pending_id is None
    assert runtime_manager.get_session("document-test") is None
