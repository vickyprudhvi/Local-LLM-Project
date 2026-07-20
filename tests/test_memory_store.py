"""Unit tests for memory_store.py's semantic recall() filtering.

remember()/list_all() are exercised indirectly via test_router.py and assistant
dispatch; these tests focus on recall()'s distance-threshold and top-N behavior
since that's the piece Phase 1 relies on.
"""

from unittest.mock import MagicMock, patch

import pytest

import memory_store


@pytest.fixture(autouse=True)
def _reset_module_state():
    """_client/_collection are cached module globals; keep tests isolated from each other."""
    memory_store._client = None
    memory_store._collection = None
    yield
    memory_store._client = None
    memory_store._collection = None


def _mock_collection(count, query_result=None):
    collection = MagicMock()
    collection.count.return_value = count
    collection.query.return_value = query_result or {}
    return collection


@patch("memory_store._embed", return_value=[0.1, 0.2, 0.3])
@patch("memory_store._get_collection")
def test_recall_filters_out_results_beyond_max_distance(mock_get_collection, mock_embed):
    mock_get_collection.return_value = _mock_collection(
        count=2,
        query_result={
            "documents": [["dentist on July 14", "unrelated fact"]],
            "metadatas": [[{"kind": "fact"}, {"kind": "fact"}]],
            "distances": [[0.4, 5.0]],
        },
    )

    results = memory_store.recall("dentist", n_results=3, max_distance=1.5)

    assert [r["text"] for r in results] == ["dentist on July 14"]


@patch("memory_store._embed", return_value=[0.1, 0.2, 0.3])
@patch("memory_store._get_collection")
def test_recall_requests_at_most_n_results(mock_get_collection, mock_embed):
    mock_get_collection.return_value = _mock_collection(
        count=10,
        query_result={"documents": [[]], "metadatas": [[]], "distances": [[]]},
    )

    memory_store.recall("topic", n_results=3)

    _, kwargs = mock_get_collection.return_value.query.call_args
    assert kwargs["n_results"] == 3


@patch("memory_store._get_collection")
def test_recall_returns_empty_when_collection_is_empty(mock_get_collection):
    mock_get_collection.return_value = _mock_collection(count=0)

    results = memory_store.recall("anything")

    assert results == []


@patch("memory_store._embed", return_value=None)
@patch("memory_store._get_collection")
def test_recall_returns_empty_when_embedding_fails(mock_get_collection, mock_embed):
    mock_get_collection.return_value = _mock_collection(count=3)

    results = memory_store.recall("anything")

    assert results == []
