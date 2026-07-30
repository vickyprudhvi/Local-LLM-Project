"""Phase F.1 hotfix Task 1/13(K) — the typed tool-loop control contract.

Proves the halt/continue behavior of ToolLoopDirective at the tool_loop level in
isolation (no assistant.py, no real MCP process): the loop stops IMMEDIATELY when
the callback returns a non-CONTINUE directive — no further tool calls in the same
batch execute, and the local LLM is never asked again — while a callback that
returns None (every pre-existing caller) reproduces the exact prior behavior.
"""

import pytest

import tool_loop
from tests.test_tool_loop import FakeLLM, _final, _install, _multi_tool_call, _tool_call
from tool_loop import ToolLoopControl, ToolLoopDirective
from tools.executor import ToolExecutor
from tools.registry import default_registry


@pytest.fixture
def fresh_tools(monkeypatch):
    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)
    return reg


def test_none_directive_is_identical_to_omitting_the_callback(fresh_tools, monkeypatch):
    fake = _install(monkeypatch, [
        _tool_call("math.calculate", {"expression": "1+1"}),
        _final("done"),
    ])
    text, metrics = tool_loop.run_local_tool_loop(
        "compute", [], "sys", on_tool_result=lambda call, result: None)
    assert text == "done"
    assert len(fake.calls) == 2  # selection call + final answer, exactly as before


def test_continue_directive_is_identical_to_none(fresh_tools, monkeypatch):
    fake = _install(monkeypatch, [
        _tool_call("math.calculate", {"expression": "1+1"}),
        _final("done"),
    ])
    directive = ToolLoopDirective(control=ToolLoopControl.CONTINUE)
    text, metrics = tool_loop.run_local_tool_loop(
        "compute", [], "sys", on_tool_result=lambda call, result: directive)
    assert text == "done"
    assert len(fake.calls) == 2


def test_halt_for_filesystem_access_stops_before_a_second_llm_call(fresh_tools, monkeypatch):
    fake = _install(monkeypatch, [
        _tool_call("math.calculate", {"expression": "1+1"}),
        _final("a fallback answer the model must never get to write"),
    ])
    directive = ToolLoopDirective(control=ToolLoopControl.HALT_FOR_FILESYSTEM_ACCESS,
                                  server_id="filesystem")
    text, metrics = tool_loop.run_local_tool_loop(
        "read something outside the root", [], "sys",
        on_tool_result=lambda call, result: directive)
    assert text is None  # the caller already holds the directive; text is not meaningful
    assert len(fake.calls) == 1  # only the selection call — no second (fallback) call


def test_restart_mcp_and_resume_stops_before_a_second_llm_call(fresh_tools, monkeypatch):
    fake = _install(monkeypatch, [
        _tool_call("mcp.filesystem.access.add", {"server_id": "filesystem", "plan_id": "p",
                                                  "plan_hash": "h"}),
        _final("never reached"),
    ])
    directive = ToolLoopDirective(control=ToolLoopControl.RESTART_MCP_AND_RESUME,
                                  server_id="filesystem", expected_allowed_roots=("a", "b"))
    text, metrics = tool_loop.run_local_tool_loop(
        "grant access", [], "sys", on_tool_result=lambda call, result: directive)
    assert text is None
    assert len(fake.calls) == 1


def test_halt_stops_remaining_calls_in_the_same_batch(fresh_tools, monkeypatch):
    """Two tool calls requested in ONE assistant message: a halting directive on the
    FIRST must prevent the SECOND from ever executing."""
    fake = _install(monkeypatch, [
        _multi_tool_call([
            ("math.calculate", {"expression": "1+1"}),
            ("math.calculate", {"expression": "396"}),  # must never run
        ]),
        _final("never reached"),
    ])
    seen = []

    def on_result(call, result):
        seen.append(call.arguments)
        return ToolLoopDirective(control=ToolLoopControl.HALT_FOR_FILESYSTEM_ACCESS)

    text, metrics = tool_loop.run_local_tool_loop("go", [], "sys", on_tool_result=on_result)
    assert text is None
    assert len(seen) == 1  # the second call in the batch never reached the observer
    assert len(fake.calls) == 1


def test_a_raising_callback_still_continues_normally(fresh_tools, monkeypatch):
    """An observer that raises is swallowed exactly like before (default: None),
    NOT treated as a halt."""
    _install(monkeypatch, [
        _tool_call("math.calculate", {"expression": "1+1"}),
        _final("done"),
    ])

    def _boom(call, result):
        raise RuntimeError("observer bug")

    text, _ = tool_loop.run_local_tool_loop("compute", [], "sys", on_tool_result=_boom)
    assert text == "done"
