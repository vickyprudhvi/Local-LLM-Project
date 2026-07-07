from router import route_intent


def test_look_up_mortgage_rates_routes_to_claude_not_camera():
    decision = route_intent("look up mortgage rates")
    assert decision.mode == "claude"
    assert decision.tool is None


def test_bare_look_does_not_false_trigger_camera():
    decision = route_intent("look up python tutorials")
    assert decision.tool != "look"
    assert decision.mode == "local"


def test_remember_extracts_fact():
    decision = route_intent("remember: dentist on July 14")
    assert decision.mode == "tool"
    assert decision.tool == "remember"
    assert decision.payload == "dentist on July 14"


def test_note_and_save_this_are_aliases_for_remember():
    assert route_intent("note: buy milk").tool == "remember"
    assert route_intent("save this: wifi password is hunter2").tool == "remember"


def test_claude_override_wins_over_tool_pattern():
    decision = route_intent("ask claude: what time is it")
    assert decision.mode == "claude"
    assert decision.payload == "what time is it"


def test_claude_override_wins_over_claude_domain_pattern():
    decision = route_intent("ask claude should i invest in index funds")
    assert decision.mode == "claude"


def test_local_override_wins_over_claude_domain_pattern():
    decision = route_intent("use local: should i invest in index funds")
    assert decision.mode == "local"


def test_recall_phrase():
    decision = route_intent("what do you remember")
    assert decision.mode == "tool"
    assert decision.tool == "recall"


def test_recall_phrase_variant():
    decision = route_intent("what do you know about me")
    assert decision.tool == "recall"


def test_time_phrase():
    decision = route_intent("what time is it")
    assert decision.mode == "tool"
    assert decision.tool == "time"


def test_look_phrases_route_to_camera_tool():
    for phrase in (
        "look at this",
        "take a look",
        "what do you see",
        "check the camera",
        "take a picture",
    ):
        decision = route_intent(phrase)
        assert decision.mode == "tool", phrase
        assert decision.tool == "look", phrase


def test_default_routes_to_local():
    decision = route_intent("how's the weather today")
    assert decision.mode == "local"
    assert decision.tool is None


def test_claude_domain_keywords():
    for text in (
        "should I refinance my mortgage",
        "can you help me with my resume",
        "think hard about this problem",
        "what does my insurance cover",
    ):
        decision = route_intent(text)
        assert decision.mode == "claude", text
