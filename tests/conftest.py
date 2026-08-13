"""Shared test fixtures.

Redirect interaction-log writes to a per-test temp file so running the suite
never appends to the repo's logs/interactions.jsonl.
"""

import pytest

import interaction_log


@pytest.fixture(autouse=True)
def _redirect_interaction_log(tmp_path, monkeypatch):
    monkeypatch.setattr(interaction_log, "LOG_PATH", str(tmp_path / "interactions.jsonl"))


@pytest.fixture(autouse=True)
def _isolate_research_run_artifacts(tmp_path, monkeypatch):
    """Phase H.5, Phase 0: run artifacts are OFF by default, but a test that
    enables them must not write into the repo's logs/research_runs. Same
    discipline as the interaction-log redirect above."""
    monkeypatch.setenv("RESEARCH_RUN_ARTIFACTS_DIR", str(tmp_path / "research_runs"))


@pytest.fixture(autouse=True)
def _pin_finance_providers_to_alphavantage(monkeypatch):
    """Phase H.3 added Yahoo/SEC as the DEFAULT dataset-specific providers
    (finance_quote_provider() etc. default to "yahoo"/"sec", not
    "alphavantage") -- but every finance test written before that phase
    builds Alpha-Vantage-shaped fixtures and asserts Alpha-Vantage-specific
    behavior. That assumption was always real; this makes it explicit and
    central instead of leaving ~50 test files to each implicitly depend on
    a default that no longer holds. Also disables Yahoo/SEC outright so a
    test that forgets to mock them fails closed instead of reaching a real
    provider. Tests that actually exercise Yahoo/SEC (test_finance_yahoo_
    provider.py, test_finance_sec_provider.py, ...) override these
    explicitly via their own monkeypatch.setenv calls, same pattern as
    ALPHAVANTAGE_API_KEY is already pinned per-test today.
    """
    for var in ("FINANCE_QUOTE_PROVIDER", "FINANCE_PRICE_HISTORY_PROVIDER",
               "FINANCE_CORPORATE_ACTIONS_PROVIDER", "FINANCE_US_FUNDAMENTALS_PROVIDER",
               "FINANCE_COMPANY_PROFILE_PROVIDER", "FINANCE_ANALYST_ESTIMATES_PROVIDER",
               "FINANCE_NEWS_PROVIDER"):
        monkeypatch.setenv(var, "alphavantage")
    monkeypatch.setenv("YAHOO_FINANCE_ENABLED", "false")
    monkeypatch.setenv("YAHOO_PERSONAL_USE_ACKNOWLEDGED", "false")
    monkeypatch.setenv("SEC_EDGAR_ENABLED", "false")
