"""remember/recall on a ChromaDB PersistentClient. Embeddings come from Ollama explicitly."""

import os
import uuid
from datetime import datetime, timezone

import chromadb
import requests
from rich.console import Console

console = Console()

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-minilm")

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path="./memory")
        _collection = _client.get_or_create_collection("facts")
    return _collection


def _embed(text):
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Embedding call failed: {e}[/red]")
        return None

    data = resp.json()
    try:
        return data["embeddings"][0]
    except (KeyError, IndexError):
        console.print("[red]Embedding response was malformed.[/red]")
        return None


def remember(text, kind="fact", source="user", confidence=0.85):
    embedding = _embed(text)
    if embedding is None:
        return None

    collection = _get_collection()
    fact_id = str(uuid.uuid4())
    metadata = {
        "kind": kind,
        "source": source,
        "confidence": confidence,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    collection.add(ids=[fact_id], embeddings=[embedding], documents=[text], metadatas=[metadata])
    return fact_id


def list_all(n_results=10):
    """Return the most recent remembered facts, unfiltered by relevance.

    For explicit "what do you remember" style requests, where there's no topic
    to match against — a similarity search against a generic query embeds far
    from every specific fact and would filter everything out. See recall()
    below for the topical-matching version used to enrich prompts.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    data = collection.get(include=["documents", "metadatas"])
    facts = [{"text": text, "metadata": metadata} for text, metadata in zip(data["documents"], data["metadatas"])]
    facts.sort(key=lambda f: f["metadata"].get("created_at", ""), reverse=True)
    return facts[:n_results]


MAX_RELEVANT_DISTANCE = 1.5


def recall(query, n_results=3, max_distance=MAX_RELEVANT_DISTANCE):
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    embedding = _embed(query)
    if embedding is None:
        return []

    results = collection.query(query_embeddings=[embedding], n_results=min(n_results, count))

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {"text": text, "metadata": metadata, "distance": distance}
        for text, metadata, distance in zip(documents, metadatas, distances)
        if distance <= max_distance
    ]
