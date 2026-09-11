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


@pytest.fixture(autouse=True)
def _pin_finance_extraction_to_v1(monkeypatch):
    """The suite measures the code, not the operator's `.env`.

    Same discipline as the provider pin above, for the same reason. Once the
    semantic extractor was wired to the composition root, setting
    FINANCE_EXTRACTION_MODE=compare in this repository's own `.env` made the
    canaries issue REAL model calls -- minutes of network per run, a
    nondeterministic reader deciding whether assertions held, and a suite that
    passed or failed depending on a file nobody reads while writing tests.

    So the mode is pinned to v1 and the extractor registry is emptied. A test
    exercising compare/v2 says so explicitly, by passing `mode=` or patching
    `config.finance_extraction_mode`, and supplies its own stub extractor.
    """
    monkeypatch.setenv("FINANCE_EXTRACTION_MODE", "v1")

    from finance.extraction import runtime as _extraction_runtime

    previous = _extraction_runtime._EXTRACTOR_FACTORY  # noqa: SLF001
    _extraction_runtime.register_extractor_factory(None)
    yield
    _extraction_runtime.register_extractor_factory(previous)


@pytest.fixture(autouse=True)
def _pin_finance_actualization_to_v1(monkeypatch):
    """The suite measures the code, not the operator's `.env`.

    Same discipline as the extraction-mode pin above. Without it, switching
    FINANCE_ACTUALIZATION_MODE in this repository's own `.env` would silently
    change which period every finance test resolves as current -- and the
    canaries would start asserting against a different quarter for reasons
    nobody reading the test could see.
    """
    monkeypatch.setenv("FINANCE_ACTUALIZATION_MODE", "v1")


@pytest.fixture(autouse=True)
def _pin_reported_actuals_to_v1(monkeypatch):
    """The suite measures the code, not the operator's `.env`.

    Same discipline as the two pins above. Without it, setting
    FINANCE_REPORTED_ACTUALS_MODE in this repository's own `.env` would send
    every finance test out to SEC EDGAR for filing indexes and exhibits --
    real network per test, and a suite whose answers depend on what an issuer
    filed this morning.

    A test exercising compare/v2 says so explicitly, by passing `mode=` and
    its own fetcher.
    """
    monkeypatch.setenv("FINANCE_REPORTED_ACTUALS_MODE", "v1")
