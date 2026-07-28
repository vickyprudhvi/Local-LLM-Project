"""Shared test fixtures.

Redirect interaction-log writes to a per-test temp file so running the suite
never appends to the repo's logs/interactions.jsonl.
"""

import pytest

import interaction_log


@pytest.fixture(autouse=True)
def _redirect_interaction_log(tmp_path, monkeypatch):
    monkeypatch.setattr(interaction_log, "LOG_PATH", str(tmp_path / "interactions.jsonl"))
