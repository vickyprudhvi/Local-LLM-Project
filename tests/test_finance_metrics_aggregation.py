"""Phase H.5, Phase 5b — the aggregation that decides which patterns stay.

47 patterns accumulated over five rounds of live failures and nobody could
say which mattered. Gating behaviour cannot answer it: a pattern that never
fires and one that fires constantly look identical from outside. Counting is
the only way to tell them apart.

This script reports; it never deletes. The coverage criterion decides when a
report is trustworthy enough to act on.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import finance.claim_validation as cv
import finance.content_policy as cp
from scripts.aggregate_validation_metrics import (
    MIN_RUNS,
    MIN_TICKERS,
    aggregate,
    all_rules,
    read_validation_records,
    report,
)


def _record(symbol, passed, findings):
    return {"event": "research_validation", "symbol": symbol, "passed": passed,
            "finding_count": len(findings), "findings": findings, "stage_statuses": {}}


def _finding(rule_id, field_path="thesis", outcome="quarantined", span="fortress"):
    return {"rule_id": rule_id, "label": "l", "severity": "overstatement",
            "field_path": field_path, "matched_span": span, "outcome": outcome}


def _write(tmp_path, records, extra_lines=()):
    path = tmp_path / "log.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        for line in extra_lines:
            handle.write(line + "\n")
    return str(path)


def test_only_validation_records_are_read(tmp_path):
    """The log is shared with tool/turn/mcp events."""
    path = _write(tmp_path, [_record("AOS", True, [])], extra_lines=[
        json.dumps({"event": "tool", "tool": "finance.dcf_model"}),
        json.dumps({"event": "turn", "question": "hi"}),
    ])
    assert len(read_validation_records(path)) == 1


def test_a_truncated_final_line_does_not_break_the_history(tmp_path):
    """An interrupted run must not make everything before it unreadable."""
    path = _write(tmp_path, [_record("AOS", True, [])], extra_lines=['{"event": "resea'])
    assert len(read_validation_records(path)) == 1


def test_a_missing_log_is_not_an_error(tmp_path):
    assert read_validation_records(str(tmp_path / "nope.jsonl")) == []


def test_a_rule_is_counted_once_per_run_not_once_per_occurrence():
    """A rule firing three times in one report is one piece of evidence about
    that rule, not three -- otherwise a single verbose report could make a
    pattern look essential."""
    records = [_record("AOS", True, [
        _finding("CV-101", field_path="thesis"),
        _finding("CV-101", field_path="claims[0].claim"),
        _finding("CV-101", field_path="balanced_assessment"),
    ])]
    stats, _coverage = aggregate(records)
    assert stats["CV-101"]["fires"] == 1
    # ...but every affected field is still recorded.
    assert len(stats["CV-101"]["fields"]) == 3


def test_distinct_tickers_are_tracked_separately_from_runs():
    """Ten fires on one ticker is much weaker evidence than one fire on ten."""
    records = [_record("AOS", True, [_finding("CV-101")]) for _ in range(5)]
    stats, coverage = aggregate(records)
    assert stats["CV-101"]["fires"] == 5
    assert len(stats["CV-101"]["symbols"]) == 1
    assert coverage["runs"] == 5
    assert len(coverage["symbols"]) == 1


def test_fatal_and_quarantined_outcomes_are_separated():
    """A pattern that kills stages is different evidence from one that only
    trims a sentence."""
    records = [
        _record("AOS", False, [_finding("CP-001", outcome="fatal")]),
        _record("WM", True, [_finding("CV-101", outcome="quarantined")]),
    ]
    stats, _ = aggregate(records)
    assert stats["CP-001"]["fatal"] == 1 and stats["CP-001"]["quarantined"] == 0
    assert stats["CV-101"]["fatal"] == 0 and stats["CV-101"]["quarantined"] == 1


def test_runs_that_survived_are_tracked():
    records = [
        _record("AOS", True, [_finding("CV-101")]),
        _record("WM", False, [_finding("CV-101")]),
    ]
    stats, coverage = aggregate(records)
    assert stats["CV-101"]["runs_that_passed"] == 1
    assert coverage["passed"] == 1


def test_never_fired_rules_are_the_headline_output(capsys):
    """The whole point: rules with no evidence behind them."""
    records = [_record("AOS", True, [_finding("CV-101")])]
    stats, coverage = aggregate(records)
    report(stats, coverage)
    out = capsys.readouterr().out

    assert "PATTERNS THAT HAVE NEVER FIRED" in out
    total = len(all_rules())
    assert f"({total - 1} of {total})" in out
    assert "CV-101" in out


def test_the_coverage_criterion_gates_action(capsys):
    """Coverage, not elapsed time -- this tool runs occasionally, so a
    calendar window says nothing about how much text the patterns saw."""
    stats, coverage = aggregate([_record("AOS", True, [])])
    report(stats, coverage)
    assert "COVERAGE NOT MET" in capsys.readouterr().out

    enough = [_record(f"T{i}", True, []) for i in range(MIN_TICKERS)]
    enough += [_record("T0", True, []) for _ in range(MIN_RUNS - MIN_TICKERS)]
    stats, coverage = aggregate(enough)
    report(stats, coverage)
    out = capsys.readouterr().out
    assert "COVERAGE MET" in out
    # Fabrication rules are never deletion candidates on absence alone.
    assert "Fabrication rules are NOT" in out


def test_every_defined_rule_appears_somewhere_in_the_report(capsys):
    """A rule missing from both lists would be invisible to the decision."""
    stats, coverage = aggregate([_record("AOS", True, [_finding("CP-001", outcome="fatal")])])
    report(stats, coverage)
    out = capsys.readouterr().out
    for rule in all_rules():
        assert rule.rule_id in out, rule.rule_id


def test_an_unknown_rule_id_in_the_log_does_not_crash(capsys):
    """Logs outlive rule tables. A record naming a since-deleted rule must
    still be readable."""
    stats, coverage = aggregate([_record("AOS", True, [_finding("CV-999")])])
    report(stats, coverage)
    assert "unknown rule" in capsys.readouterr().out


def test_aggregation_is_deterministic():
    records = [_record("AOS", True, [_finding("CV-101"), _finding("CP-001")]),
               _record("WM", False, [_finding("CV-101")])]
    first, _ = aggregate(records)
    second, _ = aggregate(records)
    assert {k: v["fires"] for k, v in first.items()} == {k: v["fires"] for k, v in second.items()}
