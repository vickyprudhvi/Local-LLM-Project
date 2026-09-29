"""Golden Document Benchmark (INITIAL, spec sections 24-27): the scored run.

Twelve cases (`tests/fixtures/document_package_benchmark.py`), not the full
25-40 spec section 24 describes -- see that fixture module's own docstring
and the phase completion summary for why this is reported as a start rather
than the finished benchmark.

Both scored quantities are DETERMINISTIC-boundary measurements (document
classification, and the event validator's accept/reject decision given a
stubbed candidate), so 100% is the right bar here -- unlike a live-model
recall benchmark, a failure on one of these hand-built cases is a bug in
`finance/documents/`, not sampling noise.
"""

import re

from tests.document_package_benchmark_harness import run_benchmark
from tests.fixtures.document_package_benchmark import all_cases, classes_covered


def test_benchmark_covers_the_classes_this_phase_targets():
    covered = set(classes_covered())
    required = {"mature_profitable", "loss_making_growth",
               "foreign_private_issuer", "guidance_heavy", "routine_corporate",
               "financing_event", "hard_safety", "actual_table_fallback",
               "guidance_near_actual_table", "bridge_facility", "debt_refinancing",
               "incomplete_structured_actuals"}
    missing = required - covered
    assert not missing, f"golden benchmark is missing classes: {missing}"


def test_actual_fact_acceptance_boundary_is_perfect_on_the_golden_set():
    """Deterministic-boundary counterpart to the live actuals-reader
    benchmark (`scripts/run_live_document_pipeline_benchmark.py`): a
    scripted candidate carrying the hand-verified ground truth must be
    accepted, and the outlook-table contamination item must be refused."""
    score = run_benchmark(all_cases())
    assert score.actual_acceptance_accuracy == 1.0, score.all_actual_errors


def test_document_classification_accuracy_is_perfect_on_the_golden_set():
    score = run_benchmark(all_cases())
    assert score.classification_accuracy == 1.0, score.all_classification_errors
    assert not score.reporting_status_errors, score.reporting_status_errors


def test_event_acceptance_boundary_is_perfect_on_the_golden_set():
    """The two negative-control cases are what make this assertion mean
    something: `financing-item-code-mismatch-negative-control` and
    `financing-undrawn-claimed-funded-negative-control` must be REFUSED, and
    every funded/committed case must be accepted with its flag intact."""
    score = run_benchmark(all_cases())
    assert score.event_acceptance_accuracy == 1.0, score.all_event_errors


def test_no_benchmark_case_id_appears_in_production_code():
    """Section 24: ticker-shaped fixture names may exist in fixture data
    only, never in production branching. This benchmark uses descriptive
    case ids rather than tickers, but the discipline is checked anyway."""
    import pathlib

    case_ids = [case.case_id for case in all_cases()]
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for directory in ("finance", "tools"):
        for path in (root / directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for case_id in case_ids:
                if case_id in text:
                    offenders.append(f"{path}: {case_id}")
    assert not offenders, offenders
