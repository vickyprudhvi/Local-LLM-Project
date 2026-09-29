"""Phase H.5 (validation rework) — Phase 0 (replay artifacts) + 5a (metrics).

Phase 0 exists because the WM escalation this rework responds to could not be
re-examined: it happened once, in a transient console buffer, and
`logs/interactions.jsonl` records tool calls only (a search across 4,117
records for `research_manager` returns zero). Every later question about that
failure had to be answered by re-running the model and hoping it recurred.

Phase 5a exists because gating behaviour cannot tell you which patterns earn
their place. A pattern that never fires and a pattern that fires constantly
both look identical from outside -- "no failures" -- which is how a
vocabulary list reached 47 entries with nobody able to say which mattered.

Neither phase changes any gating decision. Both are purely observational.
"""

import json
from pathlib import Path

import pytest

import interaction_log
import finance.research_pipeline as rp
from finance.run_artifact import (
    artifact_path,
    build_artifact,
    load_artifact,
    write_artifact,
)
from tests.test_finance_research_pipeline import (
    INDEX,
    _bull_response,
    _research_manager_response,
    make_ask_local,
)


def _run(**overrides):
    return rp.run_research_pipeline(INDEX, make_ask_local(**overrides))


# ---------------------------------------------------------------------------
# Phase 0 — artifacts
# ---------------------------------------------------------------------------

def test_artifacts_are_off_by_default(tmp_path, monkeypatch):
    """Diagnostic capture, not a product feature -- ~100 KB per run."""
    monkeypatch.delenv("RESEARCH_RUN_ARTIFACTS_ENABLED", raising=False)
    monkeypatch.setenv("RESEARCH_RUN_ARTIFACTS_DIR", str(tmp_path / "runs"))
    assert write_artifact("TEST", _run()) is None
    assert not (tmp_path / "runs").exists()


def test_an_enabled_run_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_RUN_ARTIFACTS_ENABLED", "true")
    monkeypatch.setenv("RESEARCH_RUN_ARTIFACTS_DIR", str(tmp_path / "runs"))
    result = _run()

    path = write_artifact("TEST", result)
    assert path is not None
    loaded = load_artifact(path)

    assert loaded["schema"] == "research_run_artifact/1"
    assert loaded["symbol"] == "TEST"
    assert loaded["available"] is True
    assert len(loaded["checkpoints"]) == 6
    assert {c["stage"] for c in loaded["checkpoints"]} == {
        "bull_researcher", "bear_researcher", "rebuttal_round",
        "research_manager", "risk_reviewer", "final_investment_synthesizer"}


def test_a_failed_run_captures_the_text_that_failed(tmp_path, monkeypatch):
    """The whole point. A clean run is the least interesting thing to keep;
    the failing field is the artifact anyone actually wants."""
    monkeypatch.setenv("RESEARCH_RUN_ARTIFACTS_ENABLED", "true")
    monkeypatch.setenv("RESEARCH_RUN_ARTIFACTS_DIR", str(tmp_path / "runs"))
    body = json.loads(_bull_response())
    body["claims"][0]["claim"] = "Use a 5% position size for this name."
    result = _run(bull=json.dumps(body))

    loaded = load_artifact(write_artifact("TEST", result))
    bull = next(c for c in loaded["checkpoints"] if c["stage"] == "bull_researcher")

    assert bull["status"] == rp.StageStatus.FAILED
    assert bull["error"]
    assert bull["findings"], "the findings must survive into the artifact"
    assert bull["failure_detail"]["field_path"] == "claims[0].claim"


def test_the_artifact_filename_cannot_escape_its_directory(tmp_path):
    """`symbol` reaches this from a caller; a malformed one must not produce
    a path outside the artifact directory."""
    for symbol in ("../../etc/passwd", "A/B", "..", "", None, "sym\\bol"):
        path = Path(artifact_path(symbol, timestamp="X", directory=str(tmp_path)))
        assert path.parent == tmp_path, symbol


def test_writing_never_raises_when_the_directory_is_unusable(tmp_path, monkeypatch):
    """A diagnostic capture failing must not take down an analysis that
    otherwise succeeded."""
    monkeypatch.setenv("RESEARCH_RUN_ARTIFACTS_ENABLED", "true")
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("RESEARCH_RUN_ARTIFACTS_DIR", str(blocker))
    assert write_artifact("TEST", _run()) is None


def test_build_artifact_is_pure():
    """No filesystem access, so it can be asserted on directly."""
    artifact = build_artifact("TEST", _run(), evidence_index_size=42)
    assert artifact["evidence_index_size"] == 42
    assert artifact["total_prompt_tokens"] is not None


# ---------------------------------------------------------------------------
# Phase 5a — metrics
# ---------------------------------------------------------------------------

def test_a_quarantined_finding_is_recorded_with_its_rule_id():
    bad = _bull_response(extra={"thesis": "This is a fortress balance sheet."})
    checkpoint = _run(bull=bad).by_stage("bull_researcher")

    assert len(checkpoint.findings) == 1
    finding = checkpoint.findings[0]
    assert finding["rule_id"].startswith("CV-1")   # superlative group
    assert finding["field_path"] == "thesis"
    assert finding["outcome"] == "quarantined"
    assert finding["severity"] == "overstatement"
    assert finding["matched_span"].lower() == "fortress"


def test_a_fatal_finding_is_recorded_too():
    """The blind spot this phase closes. A fatal finding previously survived
    only as prose inside the error message, so the rule_id and field_path
    were lost -- meaning the patterns most worth studying (the ones that kill
    stages) were exactly the ones with no data behind them."""
    body = json.loads(_bull_response())
    body["claims"][0]["claim"] = "Use a 5% position size for this name."
    checkpoint = _run(bull=json.dumps(body)).by_stage("bull_researcher")

    assert checkpoint.status == rp.StageStatus.FAILED
    assert checkpoint.findings, "a failed stage must still report its findings"
    finding = next(f for f in checkpoint.findings if f["rule_id"] == "CP-001")
    assert finding["outcome"] == "fatal"
    assert finding["severity"] == "fabrication"
    assert finding["field_path"] == "claims[0].claim"


def test_findings_are_deterministically_ordered():
    """Invariant 3."""
    body = json.loads(_research_manager_response())
    body["supported_bull_points"] = ["A fortress balance sheet.", "Best-in-class returns."]
    body["balanced_assessment"] = "The consensus value points toward more upside."
    runs = [_run(research_manager=json.dumps(body)) for _ in range(3)]
    serialized = [json.dumps(r.by_stage("research_manager").findings, sort_keys=True)
                  for r in runs]
    assert serialized[0] == serialized[1] == serialized[2]
    paths = [f["field_path"] for f in runs[0].by_stage("research_manager").findings]
    assert paths == sorted(paths)


def test_a_clean_run_records_no_findings():
    for stage in ("bull_researcher", "research_manager", "final_investment_synthesizer"):
        assert _run().by_stage(stage).findings == [], stage


def test_the_matched_span_is_truncated():
    assert rp._METRIC_SPAN_MAX == 40
    long_span = type("F", (), {"rule_id": "X", "label": "l", "severity": "overstatement",
                               "field_path": "p", "matched_span": "z" * 200})()
    assert len(rp._finding_metric(long_span, "fatal")["matched_span"]) == 40


def test_the_log_line_records_paths_and_spans_but_no_field_values(tmp_path, monkeypatch):
    monkeypatch.setattr(interaction_log, "LOG_PATH", str(tmp_path / "log.jsonl"))
    secret_prose = ("This is a fortress balance sheet and here is a great deal of "
                    "additional model-authored prose that must not be logged.")
    bad = _bull_response(extra={"thesis": secret_prose})
    result = _run(bull=bad)
    findings = [f for c in result.checkpoints for f in (c.findings or [])]

    record = interaction_log.log_research_validation("TEST", result.available, findings,
                                                     {"bull_researcher": "completed"})
    written = json.loads(Path(interaction_log.LOG_PATH).read_text(encoding="utf-8").strip())

    assert written["event"] == "research_validation"
    assert written["symbol"] == "TEST"
    assert written["passed"] is True
    assert written["finding_count"] == 1
    assert written["findings"][0]["field_path"] == "thesis"
    assert written["findings"][0]["matched_span"].lower() == "fortress"
    # The surrounding prose is NOT logged -- only the path and the span.
    assert "additional model-authored prose" not in json.dumps(written)
    assert record["stage_statuses"]["bull_researcher"] == "completed"


def test_the_run_level_passed_flag_tracks_whether_a_verdict_was_reached(tmp_path, monkeypatch):
    """`passed` means the pipeline produced a final synthesis, NOT that every
    stage succeeded -- the two genuinely differ.

    A fabrication in bull_researcher kills that stage, but research_manager
    only requires ONE of bull/bear, so the run still reaches a verdict. For
    weighing a pattern in Phase 5b that distinction matters: the per-finding
    `outcome` says whether the finding was fatal to its stage, and `passed`
    says whether the analysis survived it. A pattern that repeatedly kills a
    stage the pipeline routes around is weaker evidence than one that ends
    the run.
    """
    monkeypatch.setattr(interaction_log, "LOG_PATH", str(tmp_path / "log.jsonl"))
    body = json.loads(_bull_response())
    body["claims"][0]["claim"] = "Use a 5% position size for this name."
    result = _run(bull=json.dumps(body))

    assert result.by_stage("bull_researcher").status == rp.StageStatus.FAILED
    assert result.available is True, "the pipeline routes around a single dead researcher"

    findings = [f for c in result.checkpoints for f in (c.findings or [])]
    interaction_log.log_research_validation("TEST", result.available, findings, {})
    written = json.loads(Path(interaction_log.LOG_PATH).read_text(encoding="utf-8").strip())
    assert written["passed"] is True
    assert any(f["outcome"] == "fatal" for f in written["findings"])


def test_telemetry_never_breaks_an_analysis(monkeypatch):
    """Invariant 2, in spirit: an observational path must not be able to
    change a run's outcome."""
    from finance import workflow

    def explode(*_a, **_k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(interaction_log, "log_research_validation", explode)
    workflow._record_validation_telemetry("TEST", _run())  # must not raise
