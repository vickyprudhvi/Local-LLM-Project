"""WM corrective patch — bounded, CLASSIFIED stage recovery.

THE PROBLEM THIS FIXES
======================
`research_manager` appeared to "fail for a different stock every time". It was
never ticker-specific. It was a single-attempt stage with the largest schema in
the pipeline whose failures are STOCHASTIC, and only ONE failure class — a
content-policy violation — was ever retried. Everything else (cut off at the
token limit, one malformed brace, one hallucinated evidence id, a missing
field) killed the stage outright and cascaded through every stage downstream.

Four consecutive live WM runs measured research_manager at completion_tokens
7719, 7587, 7691 against an 8000 cap — within ~300 tokens of truncation every
time. And because Ollama's `done_reason` was being discarded, a truncated
response was indistinguishable from a malformed one, so even the diagnosis
pointed the wrong way.

Every class below is MECHANICAL: the model can fix it given the specific
correction, without being told anything new about the company.
"""

import json

import pytest

import finance.research_pipeline as rp
import tools.config as config
from tests.test_finance_research_pipeline import INDEX, make_ask_local


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error,expected", [
    (f"{rp.TRUNCATION_MARKER} the response hit the output-token limit",
     rp.StageFailureKind.TRUNCATED),
    (f"{rp.CONTENT_POLICY_VIOLATION_MARKER} prohibited language",
     rp.StageFailureKind.CONTENT_POLICY),
    ("'evidence_cited' cites unknown evidence ID(s): ['fundamental.made_up']",
     rp.StageFailureKind.UNKNOWN_EVIDENCE),
    ("response did not contain a parseable JSON object", rp.StageFailureKind.MALFORMED_JSON),
    ("call failed: ReadTimeout", rp.StageFailureKind.TRANSPORT),
    ("local model call failed", rp.StageFailureKind.TRANSPORT),
    ("'evidence_balance' must be one of ('bull_supported_more', ...)",
     rp.StageFailureKind.SCHEMA),
    ("'thesis' must be a non-empty string", rp.StageFailureKind.SCHEMA),
    ("", rp.StageFailureKind.UNRECOVERABLE),
    (None, rp.StageFailureKind.UNRECOVERABLE),
])
def test_each_failure_shape_is_classified(error, expected):
    assert rp._classify_stage_failure(error) == expected


def test_truncation_is_distinguished_from_malformed_json():
    """The same symptom (unparseable output), opposite corrections: one needs
    a SHORTER answer, the other a rewritten one. Conflating them is why the
    budget problem was invisible."""
    truncated = rp._correction_for(rp.StageFailureKind.TRUNCATED, "x")
    malformed = rp._correction_for(rp.StageFailureKind.MALFORMED_JSON, "x")
    assert "CUT OFF" in truncated and "SHORTER" in truncated
    assert "could not be parsed as JSON" in malformed
    assert "SHORTER" not in malformed


def test_the_unknown_evidence_correction_names_the_bad_ids():
    correction = rp._correction_for(
        rp.StageFailureKind.UNKNOWN_EVIDENCE,
        "'evidence_cited' cites unknown evidence ID(s): ['fundamental.made_up']")
    assert "fundamental.made_up" in correction
    assert "copied EXACTLY" in correction


def test_the_schema_correction_quotes_the_actual_error():
    correction = rp._correction_for(rp.StageFailureKind.SCHEMA,
                                    "'thesis' must be a non-empty string")
    assert "'thesis' must be a non-empty string" in correction


def test_an_unrecoverable_failure_gets_no_correction():
    """Preserves the previous fail-closed behaviour for anything unrecognized."""
    assert rp._correction_for(rp.StageFailureKind.UNRECOVERABLE, "") is None


# ---------------------------------------------------------------------------
# Recovery end to end
# ---------------------------------------------------------------------------

_GOOD_BULL = json.dumps({
    "thesis": "Growth supports the case.",
    "claims": [
        {"claim_id": "b1", "claim": "Revenue growth is healthy.",
         "evidence_ids": ["fundamental.revenue_growth_yoy"],
         "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.6},
        {"claim_id": "b2", "claim": "The quoted price is a reference point.",
         "evidence_ids": ["quote.price"],
         "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.5},
    ],
    "confidence": "medium",
})


def _sequence_ask_local(stage_prefix, responses, record=None, truncated_first=False):
    """Returns `responses` in order for one stage; canned good output elsewhere."""
    iterator = iter(responses)

    def fn(messages, **kwargs):
        if record is not None:
            record.append((messages, kwargs))
        system = messages[0]["content"]
        if system.startswith(stage_prefix):
            content = next(iterator)
            metrics = {"prompt_tokens": 100, "completion_tokens": 20}
            if truncated_first and content == responses[0]:
                metrics["truncated"] = True
            return {"message": {"role": "assistant", "content": content},
                    "metrics": metrics, "ok": True}
        return make_ask_local()(messages, **kwargs)

    return fn


def test_a_malformed_first_response_recovers_on_retry(monkeypatch):
    monkeypatch.setenv("RESEARCH_STAGE_MAX_ATTEMPTS", "3")
    calls = []
    ask = _sequence_ask_local("You are the Bull Researcher",
                              ["not json at all", _GOOD_BULL], record=calls)
    result = rp.run_research_pipeline(INDEX, ask)

    assert result.by_stage("bull_researcher").status == rp.StageStatus.COMPLETED
    bull_calls = [c for c in calls
                  if c[0][0]["content"].startswith("You are the Bull Researcher")]
    assert len(bull_calls) == 2
    assert "could not be parsed as JSON" in bull_calls[1][0][1]["content"]


def test_a_truncated_first_response_asks_for_a_shorter_one(monkeypatch):
    """The WM case. The retry must ask for LESS, not just ask again — asking
    again identically truncates again."""
    monkeypatch.setenv("RESEARCH_STAGE_MAX_ATTEMPTS", "3")
    calls = []
    ask = _sequence_ask_local("You are the Bull Researcher",
                              ['{"thesis": "cut off mid', _GOOD_BULL],
                              record=calls, truncated_first=True)
    result = rp.run_research_pipeline(INDEX, ask)

    assert result.by_stage("bull_researcher").status == rp.StageStatus.COMPLETED
    bull_calls = [c for c in calls
                  if c[0][0]["content"].startswith("You are the Bull Researcher")]
    retry_prompt = bull_calls[1][0][1]["content"]
    assert "CUT OFF" in retry_prompt
    assert "SHORTER" in retry_prompt


def test_an_unknown_evidence_id_recovers_on_retry(monkeypatch):
    monkeypatch.setenv("RESEARCH_STAGE_MAX_ATTEMPTS", "3")
    bad = json.dumps({
        "thesis": "Growth supports the case.",
        "claims": [
            {"claim_id": "b1", "claim": "Revenue growth is healthy.",
             "evidence_ids": ["fundamental.completely_made_up"],
             "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.6},
            {"claim_id": "b2", "claim": "The quoted price is a reference point.",
             "evidence_ids": ["quote.price"],
             "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.5},
        ],
        "confidence": "medium",
    })
    calls = []
    ask = _sequence_ask_local("You are the Bull Researcher", [bad, _GOOD_BULL], record=calls)
    result = rp.run_research_pipeline(INDEX, ask)

    assert result.by_stage("bull_researcher").status == rp.StageStatus.COMPLETED
    bull_calls = [c for c in calls
                  if c[0][0]["content"].startswith("You are the Bull Researcher")]
    assert "completely_made_up" in bull_calls[1][0][1]["content"]


def test_recovery_is_bounded_and_still_fails_closed(monkeypatch):
    """A stage that never produces valid output must still fail closed, with
    nothing invalid exposed — and must not grind on indefinitely."""
    monkeypatch.setenv("RESEARCH_STAGE_MAX_ATTEMPTS", "3")
    calls = []
    ask = _sequence_ask_local("You are the Bull Researcher",
                              ["not json", "still not json", "nope"], record=calls)
    result = rp.run_research_pipeline(INDEX, ask)

    checkpoint = result.by_stage("bull_researcher")
    assert checkpoint.status == rp.StageStatus.FAILED
    assert checkpoint.output is None
    assert "after 3 attempts" in checkpoint.error
    bull_calls = [c for c in calls
                  if c[0][0]["content"].startswith("You are the Bull Researcher")]
    assert len(bull_calls) == 3


def test_attempts_can_be_pinned_to_one_restoring_the_old_behaviour(monkeypatch):
    monkeypatch.setenv("RESEARCH_STAGE_MAX_ATTEMPTS", "1")
    calls = []
    ask = _sequence_ask_local("You are the Bull Researcher", ["not json"], record=calls)
    result = rp.run_research_pipeline(INDEX, ask)

    assert result.by_stage("bull_researcher").status == rp.StageStatus.FAILED
    bull_calls = [c for c in calls
                  if c[0][0]["content"].startswith("You are the Bull Researcher")]
    assert len(bull_calls) == 1


def test_token_accounting_sums_every_attempt(monkeypatch):
    monkeypatch.setenv("RESEARCH_STAGE_MAX_ATTEMPTS", "3")
    ask = _sequence_ask_local("You are the Bull Researcher",
                              ["not json", "not json", "not json"])
    result = rp.run_research_pipeline(INDEX, ask)
    checkpoint = result.by_stage("bull_researcher")
    assert checkpoint.prompt_tokens == 300   # 3 attempts x 100
    assert checkpoint.completion_tokens == 60


def test_a_recovered_stage_does_not_block_the_stages_downstream(monkeypatch):
    """The whole point: one recoverable slip must not cascade."""
    monkeypatch.setenv("RESEARCH_STAGE_MAX_ATTEMPTS", "3")
    ask = _sequence_ask_local("You are the Bull Researcher", ["not json", _GOOD_BULL])
    result = rp.run_research_pipeline(INDEX, ask)
    for stage in ("bull_researcher", "bear_researcher", "rebuttal_round",
                  "research_manager", "risk_reviewer", "final_investment_synthesizer"):
        assert result.by_stage(stage).status == rp.StageStatus.COMPLETED, stage


def test_the_correction_keeps_the_original_evidence_index(monkeypatch):
    """A correction is APPENDED to the original prompt, never a replacement —
    a retry that dropped the evidence would just fail differently."""
    monkeypatch.setenv("RESEARCH_STAGE_MAX_ATTEMPTS", "2")
    calls = []
    ask = _sequence_ask_local("You are the Bull Researcher",
                              ["not json", _GOOD_BULL], record=calls)
    rp.run_research_pipeline(INDEX, ask)
    bull_calls = [c for c in calls
                  if c[0][0]["content"].startswith("You are the Bull Researcher")]
    first, retry = bull_calls[0][0][1]["content"], bull_calls[1][0][1]["content"]
    assert retry.startswith(first)


def test_brain_surfaces_the_truncation_signal():
    """`done_reason` was being discarded, which is what made a budget problem
    look like a formatting problem."""
    import brain

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"role": "assistant", "content": "{"},
                    "prompt_eval_count": 10, "eval_count": 8000,
                    "done_reason": "length"}

    import requests
    original = requests.post
    requests.post = lambda *a, **k: _Resp()
    try:
        raw = brain.ask_local_raw([{"role": "user", "content": "x"}])
    finally:
        requests.post = original

    assert raw["metrics"]["done_reason"] == "length"
    assert raw["metrics"]["truncated"] is True
