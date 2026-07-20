"""Unit tests for assistant.py's dispatch() recall branching (Phase 1)."""

import json
from unittest.mock import patch

from assistant import dispatch
from router import RouteDecision


def _recall_decision(topic):
    return RouteDecision(mode="tool", tool="recall", payload=json.dumps({"topic": topic}))


@patch("assistant.memory_store.recall")
def test_recall_with_topic_uses_semantic_search(mock_recall):
    mock_recall.return_value = [{"text": "dentist on July 14", "metadata": {}, "distance": 0.2}]

    reply, metrics = dispatch(
        _recall_decision("dentist"), "recall my dentist appointment",
        "recall my dentist appointment", [], "system prompt",
    )

    mock_recall.assert_called_once_with("dentist", n_results=3)
    assert reply == "Here's what I remember: dentist on July 14"
    assert metrics == {}


@patch("assistant.memory_store.recall")
def test_recall_with_topic_and_no_match_returns_fallback(mock_recall):
    mock_recall.return_value = []

    reply, _ = dispatch(
        _recall_decision("unicorns"), "recall unicorns", "recall unicorns", [], "system prompt",
    )

    assert reply == "I don't have a relevant memory for that."


@patch("assistant.memory_store.list_all")
def test_recall_without_topic_falls_back_to_list_all(mock_list_all):
    mock_list_all.return_value = [{"text": "some fact", "metadata": {}}]

    reply, _ = dispatch(
        _recall_decision(None), "what do you remember about me",
        "what do you remember about me", [], "system prompt",
    )

    mock_list_all.assert_called_once_with(n_results=10)
    assert reply == "Here's what I remember: some fact"


@patch("assistant.memory_store.list_all")
def test_recall_without_topic_and_nothing_saved(mock_list_all):
    mock_list_all.return_value = []

    reply, _ = dispatch(
        _recall_decision(None), "what do you remember about me",
        "what do you remember about me", [], "system prompt",
    )

    assert reply == "I don't have anything remembered yet."
