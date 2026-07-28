"""History trimming: plain pairs still work; tool turns never orphaned."""

from brain import trim_history, trim_history_tool_aware


def _u(t):
    return {"role": "user", "content": t}


def _a(t):
    return {"role": "assistant", "content": t}


def _a_toolcall(name):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": "call_1", "function": {"name": name, "arguments": {}}}]}


def _tool(name, content="{}"):
    return {"role": "tool", "tool_name": name, "content": content}


# ---- plain history (unchanged behavior) ----

def test_plain_trim_keeps_last_n_pairs():
    history = []
    for i in range(10):
        history += [_u(f"u{i}"), _a(f"a{i}")]
    trimmed = trim_history(history, 3)
    assert len(trimmed) == 6
    assert trimmed[0] == _u("u7")


def test_tool_aware_on_plain_history_matches_last_n_turns():
    history = []
    for i in range(10):
        history += [_u(f"u{i}"), _a(f"a{i}")]
    trimmed = trim_history_tool_aware(history, 3)
    assert trimmed == history[-6:]


def test_short_history_returned_as_is():
    history = [_u("a"), _a("b")]
    assert trim_history_tool_aware(history, 5) is history


# ---- tool turns ----

def _tool_turn():
    # One logical turn containing a tool call + result + final answer.
    return [_u("calc it"), _a_toolcall("math.calculate"),
            _tool("math.calculate", '{"result": 396}'), _a("It's 396.")]


def test_tool_block_stays_intact_and_not_orphaned_at_boundary():
    # Build several plain turns, then a tool turn last; trim to keep just 1 turn.
    history = [_u("u0"), _a("a0"), _u("u1"), _a("a1")] + _tool_turn()
    trimmed = trim_history_tool_aware(history, 1)
    # Exactly the tool turn is kept, in order, with nothing orphaned.
    assert trimmed == _tool_turn()
    # A retained tool result is always preceded by its assistant tool-call message.
    for i, msg in enumerate(trimmed):
        if msg.get("role") == "tool":
            assert any(m.get("tool_calls") for m in trimmed[:i])


def test_never_starts_mid_turn_with_an_orphaned_tool_message():
    history = _tool_turn() + [_u("later"), _a("ok")]
    trimmed = trim_history_tool_aware(history, 1)
    # Keeping 1 turn drops the whole tool turn; result starts at a user message.
    assert trimmed[0]["role"] == "user"
    assert not any(m.get("role") == "tool" for m in trimmed)


def test_multiple_tool_results_stay_attached_to_their_assistant_call():
    multi = [_u("two sums"),
             {"role": "assistant", "content": "",
              "tool_calls": [{"id": "a", "function": {"name": "math.calculate", "arguments": {}}},
                             {"id": "b", "function": {"name": "math.calculate", "arguments": {}}}]},
             _tool("math.calculate", '{"result": 396}'),
             _tool("math.calculate", '{"result": 25}'),
             _a("396 and 25.")]
    history = [_u("u0"), _a("a0")] + multi
    trimmed = trim_history_tool_aware(history, 1)
    assert trimmed == multi
    assert sum(1 for m in trimmed if m.get("role") == "tool") == 2


def test_final_assistant_answer_stays_with_its_turn():
    history = [_u("u0"), _a("a0")] + _tool_turn()
    trimmed = trim_history_tool_aware(history, 1)
    assert trimmed[-1] == _a("It's 396.")
