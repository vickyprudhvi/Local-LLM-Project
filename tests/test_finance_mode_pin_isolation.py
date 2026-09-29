"""Phase H.25 -- regression test for the full-suite-only guidance-canary leak.

Root cause (see the long comment on `_pin_finance_modes_to_v1_before_any_fixture`
in tests/conftest.py): this repository's own `.env` sets FINANCE_EXTRACTION_MODE
(and FINANCE_ACTUALIZATION_MODE / FINANCE_REPORTED_ACTUALS_MODE) to "v2", and
importing `assistant` (directly or transitively, anywhere in the collected
suite) loads that `.env` and registers a real, model-backed extraction factory
-- both as pure IMPORT-TIME side effects, at collection time, before any
fixture in the session has run.

The three `_pin_*_to_v1` fixtures in conftest.py are function-scoped, so they
only take effect immediately before each TEST FUNCTION body runs. A module- or
session-scoped fixture that does real work (e.g. test_finance_canaries.py's
`runs`) is a HIGHER-scoped fixture, and pytest always builds higher-scoped
fixtures before lower-scoped ones for a given test -- so such a fixture would
see the leaked "v2" mode and the real factory, with no function-scoped pin
having applied yet.

This test reproduces that exact mechanism -- a file whose mere IMPORT corrupts
os.environ and the extraction registry (mimicking assistant.py's real
import-time side effects), collected ahead of a file with a MODULE-scoped
fixture that reads the pinned config -- as a REAL, separate pytest subprocess
against this repository's actual tests/conftest.py. A subprocess is used
deliberately: this test must never corrupt the ambient state of the pytest
process it is itself running under, and it must exercise the real
session-scoped fixture, not a re-implementation of it.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_LEAK_FILE_SOURCE = textwrap.dedent("""
    import os

    from finance.extraction import runtime as _extraction_runtime

    # Mimics assistant.py's own import-time side effects: a raw os.environ
    # write (not monkeypatch, which would auto-undo) and a real mutation of
    # the process-wide extractor registry -- both happen simply by importing
    # this module, before any pytest fixture in the session has run.
    os.environ["FINANCE_EXTRACTION_MODE"] = "v2"


    def _must_never_be_called():
        raise AssertionError("the leaked fake extractor factory must never run")


    _extraction_runtime.register_extractor_factory(_must_never_be_called)


    def test_noop():
        assert True
""")

_PROBE_FILE_SOURCE = textwrap.dedent("""
    import pytest

    from finance.extraction import runtime as _extraction_runtime
    from tools import config


    @pytest.fixture(scope="module")
    def probe():
        # Built once, at THIS test's setup -- i.e. before the function-scoped
        # `_pin_finance_extraction_to_v1` autouse fixture (also function
        # scoped) has had any chance to run for this specific test.
        return {
            "mode": config.finance_extraction_mode(),
            "registered": _extraction_runtime.extractor_registered(),
        }


    def test_module_scoped_fixture_sees_v1_not_the_leak(probe):
        assert probe["mode"] == "v1", probe
        assert probe["registered"] is False, probe
""")


def test_module_scoped_fixture_is_protected_from_the_collection_time_leak(tmp_path):
    leak_file = REPO_ROOT / "tests" / "test_zzz_h25_leak_simulator.py"
    probe_file = REPO_ROOT / "tests" / "test_zzz_h25_module_probe.py"
    try:
        leak_file.write_text(_LEAK_FILE_SOURCE, encoding="utf-8")
        probe_file.write_text(_PROBE_FILE_SOURCE, encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(leak_file), str(probe_file), "-q"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
        assert "2 passed" in result.stdout, result.stdout
    finally:
        leak_file.unlink(missing_ok=True)
        probe_file.unlink(missing_ok=True)
