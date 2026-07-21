"""Unit tests for assistant.py's dispatch() recall branching (Phase 1)."""

import json
from unittest.mock import patch

from assistant import dispatch
from router import RouteDecision


def _recall_decision(topic):
    return RouteDecision(mode="tool", tool="recall", payload=json.dumps({"topic": topic}))


@patch("assistant.ask_local")
@patch("assistant.memory_store.recall")
def test_recall_with_topic_uses_semantic_search_then_phrases_via_ask_local(mock_recall, mock_ask_local):
    mock_recall.return_value = [{"text": "dentist on July 14", "metadata": {}, "distance": 0.2}]
    mock_ask_local.return_value = ("You've got a dentist appointment on July 14.", {"prompt_tokens": 10})

    reply, metrics = dispatch(
        _recall_decision("dentist"), "recall my dentist appointment",
        "recall my dentist appointment", [], "system prompt",
    )

    mock_recall.assert_called_once_with("dentist", n_results=3)
    ask_local_args, ask_local_kwargs = mock_ask_local.call_args
    assert ask_local_args[0] == "Here's what I remember: dentist on July 14"
    assert "conversational" in ask_local_kwargs["system_prompt"]
    assert reply == "You've got a dentist appointment on July 14."
    assert metrics == {"prompt_tokens": 10}


@patch("assistant.memory_store.recall")
def test_recall_with_topic_and_no_match_returns_fallback(mock_recall):
    mock_recall.return_value = []

    reply, _ = dispatch(
        _recall_decision("unicorns"), "recall unicorns", "recall unicorns", [], "system prompt",
    )

    assert reply == "I don't have a relevant memory for that."


@patch("assistant.ask_local")
@patch("assistant.memory_store.list_all")
def test_recall_without_topic_falls_back_to_list_all_then_phrases_via_ask_local(mock_list_all, mock_ask_local):
    mock_list_all.return_value = [{"text": "some fact", "metadata": {}}]
    mock_ask_local.return_value = ("Just that some fact.", {})

    reply, _ = dispatch(
        _recall_decision(None), "what do you remember about me",
        "what do you remember about me", [], "system prompt",
    )

    mock_list_all.assert_called_once_with(n_results=10)
    ask_local_args, _ = mock_ask_local.call_args
    assert ask_local_args[0] == "Here's what I remember: some fact"
    assert reply == "Just that some fact."


@patch("assistant.memory_store.list_all")
def test_recall_without_topic_and_nothing_saved(mock_list_all):
    mock_list_all.return_value = []

    reply, _ = dispatch(
        _recall_decision(None), "what do you remember about me",
        "what do you remember about me", [], "system prompt",
    )

    assert reply == "I don't have anything remembered yet."
