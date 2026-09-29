"""Shared test fixtures.

Redirect interaction-log writes to a per-test temp file so running the suite
never appends to the repo's logs/interactions.jsonl.
"""

import pytest

import interaction_log


# ---------------------------------------------------------------------------
# Phase H.25 — session-scoped v1 pin, ahead of every module/session-scoped
# fixture in the suite.
#
# Root cause of the full-suite-only guidance-canary failures: this
# repository's own .env sets FINANCE_EXTRACTION_MODE=v2 (also
# FINANCE_ACTUALIZATION_MODE=v2 and FINANCE_REPORTED_ACTUALS_MODE=v2) --
# exactly the scenario `_pin_finance_extraction_to_v1` below already warns
# about in its own docstring. `assistant.py` loads that `.env` (via
# `brain.py`'s `load_dotenv()`) and registers a REAL model-backed extractor
# factory, both as import-time side effects. Once any test file imports
# `assistant` (directly, or transitively -- one file is enough for the whole
# process), `os.environ["FINANCE_EXTRACTION_MODE"]` is genuinely "v2" and
# `finance.extraction.runtime._EXTRACTOR_FACTORY` is genuinely a live,
# model-backed factory, for the rest of the pytest process.
#
# The three `_pin_*_to_v1` fixtures below are function-scoped, so pytest sets
# them up immediately before each TEST FUNCTION body runs. But
# `tests/test_finance_canaries.py::runs` (and any other module- or
# session-scoped fixture that does real analysis work) is a HIGHER-scoped
# fixture, and pytest always instantiates higher-scoped fixtures before
# lower-scoped ones for a given test. So the first time `runs` is requested,
# it runs BEFORE any function-scoped pin has applied -- and sees the real,
# leaked "v2" mode and the real extractor factory. Confirmed directly: a
# probe fixture at module scope, paired with nothing more than `import
# assistant` in a sibling file, observed config_mode == "v2" and
# extractor_registered == True at its own build time.
#
# This is not specific to canaries or to extraction: EVERY module/session
# scoped fixture that does real finance work is exposed to the identical gap
# for all three pinned modes. The fix is structural, not a canary-specific
# patch: pin all three modes at SESSION scope, autouse, declared first in
# this file, so they are set before any test-, module-, or session-scoped
# fixture anywhere in the suite ever runs. The function-scoped pins below are
# kept as-is -- once the session pin has already set "v1" as the baseline,
# they are a harmless, redundant extra layer, and they still give an
# individual test's own monkeypatch override (for a test that explicitly
# exercises v2/compare) the same per-test teardown behavior as before.
@pytest.fixture(scope="session", autouse=True)
def _pin_finance_modes_to_v1_before_any_fixture():
    from _pytest.monkeypatch import MonkeyPatch

    from finance.extraction import runtime as _extraction_runtime

    mp = MonkeyPatch()
    mp.setenv("FINANCE_EXTRACTION_MODE", "v1")
    mp.setenv("FINANCE_ACTUALIZATION_MODE", "v1")
    mp.setenv("FINANCE_REPORTED_ACTUALS_MODE", "v1")
    previous = _extraction_runtime._EXTRACTOR_FACTORY  # noqa: SLF001
    _extraction_runtime.register_extractor_factory(None)
    try:
        yield
    finally:
        _extraction_runtime.register_extractor_factory(previous)
        mp.undo()


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
