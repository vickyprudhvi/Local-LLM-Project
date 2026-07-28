"""Local tool-conversation loop: round-trips, multi-call, errors, step limit, no infinite loop.

All tests use a deterministic FakeLLM in place of ask_local_raw — no live Ollama.
"""

import json

import pytest

import tool_loop
from tools.executor import ToolExecutor
from tools.registry import default_registry


class FakeLLM:
    """Replays scripted /api/chat responses and records each call's messages+tools."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, messages, tools=None, timeout=120):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        resp = dict(self._responses.pop(0))
        resp.setdefault("metrics", {"prompt_tokens": 1, "completion_tokens": 1})
        resp.setdefault("ok", True)
        return resp


def _tool_call(name, args, call_id="call_x"):
    return {"message": {"role": "assistant", "content": "",
                        "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": args}}]}}


def _multi_tool_call(pairs):
    return {"message": {"role": "assistant", "content": "",
                        "tool_calls": [{"id": f"call_{i}", "function": {"name": n, "arguments": a}}
                                       for i, (n, a) in enumerate(pairs)]}}


def _final(text):
    return {"message": {"role": "assistant", "content": text}}


@pytest.fixture
def fresh_tools(monkeypatch):
    """Isolate the loop's registry/executor and force tool calling on."""
    reg = default_registry()
    monkeypatch.setattr(tool_loop, "REGISTRY", reg)
    monkeypatch.setattr(tool_loop, "EXECUTOR", ToolExecutor(reg))
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 5)
    return reg


def _install(monkeypatch, responses):
    fake = FakeLLM(responses)
    monkeypatch.setattr(tool_loop, "ask_local_raw", fake)
    return fake


def _tool_messages(call):
    return [m for m in call["messages"] if m.get("role") == "tool"]


def test_no_tool_call_returns_content(fresh_tools, monkeypatch):
    fake = _install(monkeypatch, [_final("Just a normal answer.")])
    text, metrics = tool_loop.run_local_tool_loop("hello", [], "sys")
    assert text == "Just a normal answer."
    assert len(fake.calls) == 1
    assert fake.calls[0]["tools"]  # tools were offered on the first call


def test_calculate_round_trip(fresh_tools, monkeypatch):
    fake = _install(monkeypatch, [
        _tool_call("math.calculate", {"expression": "(17 * 23) + 5"}),
        _final("The result is 396."),
    ])
    text, metrics = tool_loop.run_local_tool_loop("compute it", [], "sys")
    assert text == "The result is 396."
    # The second LLM call carried a serialized tool result with the real data.
    tmsgs = _tool_messages(fake.calls[1])
    assert len(tmsgs) == 1
    assert isinstance(tmsgs[0]["content"], str)  # json.dumps'd string
    payload = json.loads(tmsgs[0]["content"])
    assert payload["success"] is True
    assert payload["data"] == {"expression": "(17 * 23) + 5", "result": 396}
    # Metrics summed across both LLM calls.
    assert metrics["prompt_tokens"] == 2
    assert metrics["completion_tokens"] == 2


def test_echo_round_trip(fresh_tools, monkeypatch):
    fake = _install(monkeypatch, [
        _tool_call("system.echo", {"text": "ping"}),
        _final("You said ping."),
    ])
    text, _ = tool_loop.run_local_tool_loop("echo ping", [], "sys")
    assert text == "You said ping."
    payload = json.loads(_tool_messages(fake.calls[1])[0]["content"])
    assert payload["data"] == {"echo": "ping"}


def test_multiple_sequential_tool_calls_one_message(fresh_tools, monkeypatch):
    fake = _install(monkeypatch, [
        _multi_tool_call([("math.calculate", {"expression": "(17*23)+5"}),
                          ("math.calculate", {"expression": "100/4"})]),
        _final("396 and 25."),
    ])
    text, _ = tool_loop.run_local_tool_loop("two sums", [], "sys")
    assert text == "396 and 25."
    tmsgs = _tool_messages(fake.calls[1])
    results = [json.loads(m["content"])["data"]["result"] for m in tmsgs]
    assert results == [396, 25]


def test_tool_error_returned_to_llm_then_final_answer(fresh_tools, monkeypatch):
    fake = _install(monkeypatch, [
        _tool_call("math.calculate", {"expression": "1/0"}),
        _final("I couldn't compute that — division by zero."),
    ])
    text, _ = tool_loop.run_local_tool_loop("divide", [], "sys")
    assert text.startswith("I couldn't compute")
    payload = json.loads(_tool_messages(fake.calls[1])[0]["content"])
    assert payload["success"] is False
    assert payload["error"]["code"] == "INVALID_ARGUMENTS"


def test_raw_tool_result_is_never_the_final_answer(fresh_tools, monkeypatch):
    fake = _install(monkeypatch, [
        _tool_call("math.calculate", {"expression": "2+2"}),
        _final("It's 4."),
    ])
    text, _ = tool_loop.run_local_tool_loop("2+2", [], "sys")
    # The final user-facing text is the LLM's, not the JSON tool result.
    assert text == "It's 4."
    assert "success" not in text and "{" not in text


def test_disabled_tool_calling_makes_single_no_tools_call(fresh_tools, monkeypatch):
    monkeypatch.setattr(tool_loop, "TOOL_CALLING_ENABLED", False)
    fake = _install(monkeypatch, [_final("Plain answer.")])
    text, _ = tool_loop.run_local_tool_loop("hi", [], "sys")
    assert text == "Plain answer."
    assert len(fake.calls) == 1
    assert fake.calls[0]["tools"] is None


@pytest.mark.parametrize("bad_call", [
    _tool_call("does.not.exist", {}),                         # unknown
    _tool_call("system.echo", {"text": "x"}),                 # will be disabled below
    _tool_call("math.calculate", {"expression": "nope("}),    # invalid args
    {"message": {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "c", "function": {"arguments": {}}}]}},  # malformed (no name)
])
def test_rejected_calls_count_toward_step_limit(fresh_tools, monkeypatch, bad_call):
    # If system.echo is the case under test, disable it so the call is rejected.
    fresh_tools.disable("system.echo")
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 1)
    # After the one allowed (rejected) call, a good calculate call must NOT execute.
    fake = _install(monkeypatch, [
        bad_call,
        _tool_call("math.calculate", {"expression": "(17*23)+5"}),  # should be blocked by step limit
        _final("Done, with limitations."),
    ])
    text, _ = tool_loop.run_local_tool_loop("go", [], "sys")
    assert text == "Done, with limitations."
    # The blocked calculate never produced a 396 result anywhere.
    all_tool_contents = " ".join(
        m["content"] for c in fake.calls for m in _tool_messages(c)
    )
    assert "396" not in all_tool_contents
    assert "TOOL_STEP_LIMIT_REACHED" in all_tool_contents


def test_step_limit_exhaustion_makes_exactly_one_final_tools_omitted_call(fresh_tools, monkeypatch):
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 2)
    fake = _install(monkeypatch, [
        _tool_call("math.calculate", {"expression": "1+1"}, "a"),
        _tool_call("math.calculate", {"expression": "2+2"}, "b"),
        _tool_call("math.calculate", {"expression": "3+3"}, "c"),  # exceeds limit
        _final("Here's my best answer with what I have."),
    ])
    text, _ = tool_loop.run_local_tool_loop("keep going", [], "sys")
    assert text == "Here's my best answer with what I have."
    # Four LLM calls total: 3 tool rounds + exactly one final.
    assert len(fake.calls) == 4
    # The final call omitted the tools array.
    assert fake.calls[-1]["tools"] is None
    # And a step-limit result was recorded.
    joined = " ".join(m["content"] for c in fake.calls for m in _tool_messages(c))
    assert "TOOL_STEP_LIMIT_REACHED" in joined


def test_no_infinite_loop_when_model_always_requests_tools(fresh_tools, monkeypatch):
    monkeypatch.setattr(tool_loop, "MAX_TOOL_STEPS", 3)
    # Model never volunteers a final answer during tool rounds; loop must still end.
    responses = [_tool_call("math.calculate", {"expression": "1+1"}) for _ in range(4)]
    responses.append(_final("Final."))
    fake = _install(monkeypatch, responses)
    text, _ = tool_loop.run_local_tool_loop("loop", [], "sys")
    assert text == "Final."
    assert fake.calls[-1]["tools"] is None


def test_caller_history_is_not_mutated(fresh_tools, monkeypatch):
    _install(monkeypatch, [
        _tool_call("math.calculate", {"expression": "2+2"}),
        _final("4"),
    ])
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]
    snapshot = [dict(m) for m in history]
    tool_loop.run_local_tool_loop("2+2", history, "sys")
    assert history == snapshot  # no tool/assistant-tool-call messages leaked into caller history
