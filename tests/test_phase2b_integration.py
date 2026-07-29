"""Phase 2B integration: fake-LLM clone→inspect→scan→capability flow, memory,
Claude route, config toggles, path attack — no live GitHub/clone."""

import json
import os

import pytest

import tool_loop
from tools.calculator import CalculatorTool
from tools.echo import EchoTool
from tools.executor import ToolExecutor
from tools.github_client import GitHubResponse
from tools.models import ToolCall
from tools.registry import ToolRegistry, default_registry
from tools.repo_clone import CloneRepositoryTool
from tools.repo_tools import (
    CapabilityReportTool,
    InspectTool,
    ListFilesTool,
    ReadFileTool,
    SecurityScanTool,
)


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, messages, tools=None, timeout=120):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        resp = dict(self._responses.pop(0))
        resp.setdefault("metrics", {"prompt_tokens": 1, "completion_tokens": 1})
        resp.setdefault("ok", True)
        return resp


class FakeClient:
    def __init__(self, data):
        self._resp = GitHubResponse(200, data, {"remaining": 50, "reset_at": None})

    def get(self, path, params=None):
        return self._resp


class FakeRunner:
    def __init__(self, files):
        self.files = files

    def clone(self, url, ref, destination, timeout=None):
        os.makedirs(destination, exist_ok=True)
        for rel, content in self.files.items():
            p = os.path.join(destination, rel)
            os.makedirs(os.path.dirname(p) or destination, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)

    def rev_parse_head(self, repo_dir, timeout=30):
        return "abc123def456"


def _tool_call(name, args, cid="c1"):
    return {"message": {"role": "assistant", "content": "",
                        "tool_calls": [{"id": cid, "function": {"name": name, "arguments": args}}]}}


def _final(text):
    return {"message": {"role": "assistant", "content": text}}


def _tool_msgs(call):
    return [m for m in call["messages"] if m.get("role") == "tool"]


@pytest.fixture(autouse=True)
def _auto_approve_writes(monkeypatch):
    """These tests exercise the clone→inspect flow, not the confirmation UX. Auto-
    approve write confirmations (github.clone_repository) so the loop never blocks
    on input(); Phase C confirmation behavior is covered by dedicated tests."""
    import confirmation
    monkeypatch.setattr(confirmation, "confirm_action", lambda summary: True)


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOSITORY_ROOT", str(tmp_path / "repos"))
    monkeypatch.setenv("REPOSITORY_CLONE_ENABLED", "true")  # implies inspection enabled

    def _install(repo_files):
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.register(CalculatorTool())
        client = FakeClient({"private": False, "visibility": "public",
                             "default_branch": "main", "size": 5})
        reg.register(CloneRepositoryTool(client=client, runner=FakeRunner(repo_files)))
        for cls in (ListFilesTool, ReadFileTool, InspectTool, SecurityScanTool, CapabilityReportTool):
            reg.register(cls())
        monkeypatch.setattr(tool_loop, "REGISTRY", reg)
        monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
        monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
        monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 6)
        return reg
    return _install


def _llm(monkeypatch, responses):
    fake = FakeLLM(responses)
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)
    return fake


REPO_FILES = {
    "README.md": "# Finance MCP server\n",
    "pyproject.toml": "[project]\nname='fin'\nrequires-python='>=3.11'\ndependencies=['mcp>=1.0','fastapi']\n",
    "server.py": "import subprocess\nif __name__ == '__main__':\n    subprocess.run('x', shell=True)\n",
}


def test_full_clone_inspect_scan_capability_flow(wired, monkeypatch):
    wired(REPO_FILES)
    fake = _llm(monkeypatch, [
        _tool_call("github.clone_repository", {"repository": "acme/fin"}),
        _tool_call("repo.inspect", {"repository": "acme/fin"}),
        _tool_call("repo.security_scan", {"repository": "acme/fin"}),
        _tool_call("repo.capability_report", {"repository": "acme/fin"}),
        _final("acme/fin (https://github.com/acme/fin): a Python MCP server; static review found "
               "a subprocess call needing manual review. Nothing was executed or installed."),
    ])
    text, metrics = tool_loop.run_local_tool_loop("clone and analyze acme/fin, do not run it", [], "sys")
    assert "acme/fin" in text and "executed" in text.lower()
    # Clone result returned to LLM with executed:false.
    clone_payload = json.loads(_tool_msgs(fake.calls[1])[-1]["content"])
    assert clone_payload["data"]["executed"] is False
    # Capability report reached the model and recommends manual review.
    cap_payload = json.loads(_tool_msgs(fake.calls[4])[-1]["content"])
    assert cap_payload["data"]["recommendation"]["status"] == "manual_review_required"
    assert metrics["prompt_tokens"] == 5


def test_clone_error_returns_to_llm(wired, monkeypatch):
    wired(REPO_FILES)
    fake = _llm(monkeypatch, [
        _tool_call("github.clone_repository", {"repository": "acme/fin"}),
        _tool_call("github.clone_repository", {"repository": "acme/fin"}),  # already cloned now
        _final("It was already cloned; I used the existing copy."),
    ])
    text, _ = tool_loop.run_local_tool_loop("clone acme/fin twice", [], "sys")
    payload = json.loads(_tool_msgs(fake.calls[2])[-1]["content"])
    assert payload["success"] is False
    assert payload["error"]["code"] == "REPOSITORY_ALREADY_CLONED"


def test_raw_payload_not_final_answer(wired, monkeypatch):
    wired(REPO_FILES)
    _llm(monkeypatch, [
        _tool_call("github.clone_repository", {"repository": "acme/fin"}),
        _final("Cloned acme/fin for static inspection."),
    ])
    text, _ = tool_loop.run_local_tool_loop("clone it", [], "sys")
    assert text == "Cloned acme/fin for static inspection."
    assert "{" not in text and "commit" not in text


def test_path_attack_returns_controlled_error(wired, monkeypatch):
    wired(REPO_FILES)
    fake = _llm(monkeypatch, [
        _tool_call("github.clone_repository", {"repository": "acme/fin"}),
        _tool_call("repo.read_file", {"repository": "acme/fin", "path": "../../.env"}),
        _final("I can't read that path — it escapes the repository."),
    ])
    text, _ = tool_loop.run_local_tool_loop("read ../../.env from acme/fin", [], "sys")
    payload = json.loads(_tool_msgs(fake.calls[2])[-1]["content"])
    assert payload["success"] is False
    assert payload["error"]["code"] in ("REPOSITORY_PATH_ESCAPE", "INVALID_REPOSITORY_PATH")


def test_safety_block_mentions_static_and_untrusted(wired, monkeypatch):
    wired(REPO_FILES)
    fake = _llm(monkeypatch, [_final("hi")])
    tool_loop.run_local_tool_loop("hello", [], "PERSONA")
    system_content = fake.calls[0]["messages"][0]["content"]
    assert "STATIC" in system_content and "UNTRUSTED" in system_content


def test_repo_payloads_excluded_from_memory(wired, monkeypatch):
    from unittest.mock import patch
    wired(REPO_FILES)
    _llm(monkeypatch, [
        _tool_call("github.clone_repository", {"repository": "acme/fin"}),
        _tool_call("repo.read_file", {"repository": "acme/fin", "path": "README.md"}),
        _final("done"),
    ])
    history = []
    with patch("memory_store.remember") as mock_remember:
        text, _ = tool_loop.run_local_tool_loop("inspect acme/fin", history, "sys")
    assert text == "done"
    mock_remember.assert_not_called()
    assert history == []  # no clone/inspection payloads leaked into caller history


def test_max_tool_steps_enforced(wired, monkeypatch):
    wired(REPO_FILES)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 1)
    fake = _llm(monkeypatch, [
        _tool_call("github.clone_repository", {"repository": "acme/fin"}),
        _tool_call("repo.inspect", {"repository": "acme/fin"}),  # exceeds limit
        _final("Answering with what I have."),
    ])
    text, _ = tool_loop.run_local_tool_loop("clone and inspect", [], "sys")
    assert text == "Answering with what I have."
    assert fake.calls[-1]["tools"] is None
    joined = " ".join(m["content"] for c in fake.calls for m in _tool_msgs(c))
    assert "TOOL_STEP_LIMIT_REACHED" in joined


def test_phase1_tools_still_work(wired, monkeypatch):
    wired(REPO_FILES)
    _llm(monkeypatch, [_tool_call("math.calculate", {"expression": "2+2"}), _final("4")])
    text, _ = tool_loop.run_local_tool_loop("2+2", [], "sys")
    assert text == "4"


# ---- config toggles (registry-level) ----

def test_clone_disabled_excludes_clone_tool(monkeypatch):
    monkeypatch.setenv("REPOSITORY_CLONE_ENABLED", "false")
    monkeypatch.setenv("REPOSITORY_INSPECTION_ENABLED", "false")
    reg = default_registry(include_internet=True)
    names = [d.name for d in reg.enabled_definitions()]
    assert "github.clone_repository" not in names
    assert not any(n.startswith("repo.") for n in names)


def test_inspection_enabled_registers_repo_tools(monkeypatch):
    monkeypatch.setenv("REPOSITORY_CLONE_ENABLED", "false")
    monkeypatch.setenv("REPOSITORY_INSPECTION_ENABLED", "true")
    reg = default_registry(include_internet=True)
    names = [d.name for d in reg.enabled_definitions()]
    assert "github.clone_repository" not in names  # clone still off
    assert "repo.inspect" in names and "repo.read_file" in names


def test_clone_capability_blocked_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOSITORY_ROOT", str(tmp_path / "repos"))
    monkeypatch.setenv("REPOSITORY_CLONE_ENABLED", "false")
    reg = ToolRegistry()
    reg.register(CloneRepositoryTool(client=FakeClient({"private": False}), runner=FakeRunner({})))
    ex = ToolExecutor(reg)
    result = ex.execute(ToolCall("c1", "github.clone_repository", {"repository": "a/b"}), step=1)
    assert result.success is False
    assert result.error.code == "REPOSITORY_CLONE_DISABLED"


def test_inspection_capability_blocked_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOSITORY_ROOT", str(tmp_path / "repos"))
    monkeypatch.setenv("REPOSITORY_CLONE_ENABLED", "false")
    monkeypatch.setenv("REPOSITORY_INSPECTION_ENABLED", "false")
    reg = ToolRegistry()
    reg.register(InspectTool())
    ex = ToolExecutor(reg)
    result = ex.execute(ToolCall("c1", "repo.inspect", {"repository": "a/b"}), step=1)
    assert result.success is False
    assert result.error.code == "REPOSITORY_INSPECTION_DISABLED"


def test_claude_route_unchanged(monkeypatch):
    import assistant
    from unittest.mock import patch
    from router import RouteDecision
    decision = RouteDecision(mode="claude", payload="q")
    with patch("assistant.ask_claude", return_value=("claude answer", {})) as mock_claude, \
         patch("tool_loop.run_local_tool_loop") as mock_loop:
        reply, _ = assistant.dispatch(decision, "user q", "enriched q", [], "sys")
    assert reply == "claude answer"
    mock_claude.assert_called_once()
    mock_loop.assert_not_called()
