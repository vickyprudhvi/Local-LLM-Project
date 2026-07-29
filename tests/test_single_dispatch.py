"""Phase A single-dispatch-path guarantees.

Covers the new architecture-critical behavior introduced by the refactor:

  - built-ins are registered in the shared registry and executable through the
    ToolExecutor, but are NOT offered to the local tool-calling loop
    (llm_callable=False), so the local LLM's callable tool set is unchanged;
  - the router-selected built-ins reach a spoken reply through the executor
    (system.time and memory.remember are the checkpoint capabilities not exercised
    by the recall/camera dispatch tests).
"""

from unittest.mock import patch

from assistant import dispatch
from router import RouteDecision
from tools.registry import default_registry

BUILTIN_NAMES = {
    "system.time",
    "memory.remember",
    "memory.recall",
    "calendar.read",
    "camera.look",
    "camera.look_carefully",
    "camera.capture",
    "camera.scan",
}


def test_builtins_are_registered_in_the_shared_registry():
    reg = default_registry()
    for name in BUILTIN_NAMES:
        assert reg.has(name), f"{name} should be registered"
        assert reg.is_enabled(name)


def test_builtins_are_not_offered_to_the_local_llm():
    # The local tool-calling loop offers only llm_callable tools. Registering the
    # router-dispatched built-ins must NOT expand what the local LLM can call.
    reg = default_registry()
    offered = {d.name for d in reg.enabled_definitions()}
    assert offered.isdisjoint(BUILTIN_NAMES)
    # The genuinely LLM-callable Phase 1 tools remain offered.
    assert {"system.echo", "math.calculate"} <= offered


def test_time_routes_through_executor_to_a_spoken_reply():
    decision = RouteDecision(mode="tool", tool="time", payload="what time is it")
    reply, metrics = dispatch(decision, "what time is it", "what time is it", [], "sys")
    # Same format the former time branch produced: "<Weekday>, <Month> <dd> <YYYY>, ...".
    assert "," in reply and any(ch.isdigit() for ch in reply)
    assert metrics == {}


@patch("confirmation.confirm_action", return_value=True)
@patch("memory_store.remember")
def test_remember_routes_through_executor_and_confirms(mock_remember, _approve):
    mock_remember.return_value = "fact-id-123"
    decision = RouteDecision(mode="tool", tool="remember", payload="the wifi code is swordfish")
    reply, metrics = dispatch(decision, "remember the wifi code is swordfish",
                              "remember the wifi code is swordfish", [], "sys")
    mock_remember.assert_called_once_with("the wifi code is swordfish")
    assert reply == "Got it, I'll remember: the wifi code is swordfish"
    assert metrics == {}


@patch("confirmation.confirm_action", return_value=True)
@patch("memory_store.remember")
def test_remember_save_failure_is_reported(mock_remember, _approve):
    mock_remember.return_value = None
    decision = RouteDecision(mode="tool", tool="remember", payload="something")
    reply, _ = dispatch(decision, "remember something", "remember something", [], "sys")
    assert reply == "Sorry, I couldn't save that."


def test_unknown_routed_tool_falls_back_to_placeholder():
    decision = RouteDecision(mode="tool", tool="teleport", payload=None)
    reply, metrics = dispatch(decision, "teleport me", "teleport me", [], "sys")
    assert "isn't wired up yet" in reply
    assert metrics == {}


# ---- Phase C: assistant-level confirmation integration (write route) ----

def test_assistant_write_shows_deterministic_summary_and_executes_on_approval(monkeypatch):
    import confirmation
    seen = []
    monkeypatch.setattr(confirmation, "confirm_action", lambda s: (seen.append(s), True)[1])
    decision = RouteDecision(mode="tool", tool="remember", payload="PostgreSQL is my database")
    with patch("memory_store.remember", return_value="id") as mock_remember:
        reply, _ = dispatch(decision, "remember that", "remember that", [], "sys")
    # The confirmation summary is deterministic app text built from the fact.
    assert seen and "PostgreSQL is my database" in seen[0]
    mock_remember.assert_called_once_with("PostgreSQL is my database")
    assert reply == "Got it, I'll remember: PostgreSQL is my database"


def test_assistant_write_declined_does_not_execute_and_reports_cancellation(monkeypatch):
    import confirmation
    monkeypatch.setattr(confirmation, "confirm_action", lambda s: False)
    decision = RouteDecision(mode="tool", tool="remember", payload="secret plan")
    with patch("memory_store.remember") as mock_remember:
        reply, _ = dispatch(decision, "remember that", "remember that", [], "sys")
    mock_remember.assert_not_called()
    assert "cancel" in reply.lower()


def test_assistant_read_tool_executes_without_confirmation(monkeypatch):
    import confirmation
    # If the confirmer were ever consulted for a read tool, fail loudly.
    monkeypatch.setattr(confirmation, "confirm_action",
                        lambda s: (_ for _ in ()).throw(AssertionError("read tool asked for confirmation")))
    decision = RouteDecision(mode="tool", tool="time", payload="what time is it")
    reply, _ = dispatch(decision, "what time is it", "what time is it", [], "sys")
    assert "," in reply and any(ch.isdigit() for ch in reply)
