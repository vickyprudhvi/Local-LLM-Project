"""Phase H.5 (validation rework) — Phase 3: the repair prompt.

The repair prompt used to send the forbidden vocabulary back to the model.
Measured live on WM, research_manager: attempt 1 flagged ONE violation,
attempt 2 flagged FOUR, attempt 3 flagged THREE, and the stage died. The
model was reading the list of banned phrases and writing disclaimers about
them -- "this is not a price target", "not an intrinsic value" -- which
matched the very patterns it was being told to avoid.

So the correction now shows WHERE the problem is without showing WHAT the
phrase was, and asks for something positive rather than listing prohibitions.

Two branches, decided by whether the failure carries structured detail:

  * vocabulary-bearing (`_validate_claim_fidelity`) -> field-scoped, redacted
  * structural coherence guard -> message sent verbatim, because it names
    nothing forbidden and IS the fix
"""

import json
import re

import pytest

import finance.claim_validation as cv
import finance.content_policy as cp
import finance.research_pipeline as rp
from tests.test_finance_research_pipeline import (
    INDEX,
    _bull_response,
    make_ask_local,
    make_ask_local_with_stage_sequence,
)


def _all_rule_phrases():
    """The banned PHRASE from each of the 47 rule labels.

    Phrases, not words: what harms is the model learning that "price target"
    or "fortress" is the thing to write disclaimers about. Individual words
    ("support", "value", "price") are ordinary English and appear in any
    instruction prose, so matching on them measures nothing.

    Derived from the rule tables rather than hardcoded, so a new pattern
    cannot introduce a leak this test would miss.
    """
    phrases = set()
    for rule in list(cp.RULES) + list(cv.RULES):
        label = rule.label
        inner = label.split("(", 1)[1].rsplit(")", 1)[0] if "(" in label else label
        # "intrinsic value -- use 'base/bull/bear modeled value'" -> the
        # banned part only; the suggested replacement is not itself banned.
        inner = inner.split("--")[0].strip().lower()
        if inner and len(inner) >= 5:
            phrases.add(inner)
    # Descriptive category names the model would never write.
    return phrases - {"holding-dependent phrasing", "reader-directed investment imperative",
                      "future-tense technical certainty",
                      "consensus-estimate language", "no consensus data source exists"}


def _phrases_in(text):
    lowered = (text or "").lower()
    return sorted(p for p in _all_rule_phrases() if p in lowered)


# ---------------------------------------------------------------------------
# The vocabulary-bearing branch
# ---------------------------------------------------------------------------

def test_the_correction_names_the_field_and_redacts_the_span():
    detail = {"field_path": "claims[0].claim",
              "redacted_text": f"Use a 5% {rp._REDACTION}."}
    correction = rp._correction_for(rp.StageFailureKind.CONTENT_POLICY, "irrelevant", detail)

    assert "claims[0].claim" in correction
    assert rp._REDACTION in correction
    assert "Use a 5%" in correction, "surrounding text is kept so the model has context"
    assert "position size" not in correction.lower()


def test_the_correction_gives_a_positive_instruction():
    """Not a list of prohibitions -- that is what produced the escalation."""
    detail = {"field_path": "thesis", "redacted_text": f"A {rp._REDACTION} balance sheet."}
    correction = rp._correction_for(rp.StageFailureKind.CONTENT_POLICY, "x", detail)

    assert "cite the evidence ID" in correction
    assert "State plainly what the evidence shows" in correction
    for forbidden in ("do not write", "never write", "remove any", "instead of"):
        assert forbidden not in correction.lower(), forbidden


def test_the_correction_leaks_no_rule_vocabulary():
    """Asserted against the rule TABLES, so it cannot drift as rules change."""
    detail = {"field_path": "balanced_assessment",
              "redacted_text": f"The {rp._REDACTION} points to upside."}
    correction = rp._correction_for(
        rp.StageFailureKind.CONTENT_POLICY,
        # The raised message still contains the vocabulary (workflow.py parses
        # it), so this also proves the correction is NOT built from it.
        "CONTENT_POLICY_VIOLATION: prohibited or unsupported language detected in free-text "
        "fields: price target, guarantee, fortress. Every claim must be a plain, evidence-"
        "grounded observation: remove any position size, entry/exit price, stop loss...",
        detail)
    assert _phrases_in(correction) == [], "repair prompt leaked a banned phrase"


def test_the_correction_is_not_built_from_the_error_message():
    """The message and the correction are now independent artifacts."""
    detail = {"field_path": "thesis", "redacted_text": "A [...] balance sheet."}
    noisy = "CONTENT_POLICY_VIOLATION: ... fortress ... stop loss ... price target ..."
    correction = rp._correction_for(rp.StageFailureKind.CONTENT_POLICY, noisy, detail)
    assert "fortress" not in correction.lower()
    assert "stop loss" not in correction.lower()


# ---------------------------------------------------------------------------
# The structural branch
# ---------------------------------------------------------------------------

def test_a_coherence_guard_message_is_sent_verbatim():
    """These name nothing forbidden, and the message IS the fix -- redacting
    it would leave the model with no idea what to change."""
    message = ("CONTENT_POLICY_VIOLATION: research_stance was 'insufficient_data' but "
               "'recommendation' was 'buy'")
    correction = rp._correction_for(rp.StageFailureKind.CONTENT_POLICY, message, None)
    assert message in correction
    assert "not internally consistent" in correction


def test_every_coherence_guard_message_is_free_of_rule_vocabulary():
    """If one of them ever did name a banned phrase, sending it verbatim
    would reintroduce the leak. This pins that they do not."""
    messages = [
        "research_stance was 'insufficient_data' but 'recommendation' was 'buy'",
        "material datasets were omitted from this analysis",
        "research readiness is 'NOT_READY' but confidence is 0.80",
        "the valuation freshness is 'STALE_INPUT_WARNING' but confidence is 0.90",
        "overall_risk is 'low' but the risk reviewer aggregated 'high'",
        "valuation_view is 'undervalued' but the DCF failed validation",
    ]
    for message in messages:
        assert _phrases_in(message) == [], f"{message!r} leaks a banned phrase"


# ---------------------------------------------------------------------------
# End to end, and the escalation itself
# ---------------------------------------------------------------------------

def test_a_fabrication_repair_prompt_redacts_the_span_end_to_end():
    """`claims` is a FAIL-policy field, so a fabrication there reaches the
    repair path rather than being quarantined."""
    body = json.loads(_bull_response())
    body["claims"][0]["claim"] = "Use a 5% position size for this name."
    calls = []
    rp.run_research_pipeline(INDEX, make_ask_local(record=calls, bull=json.dumps(body)))

    bull_calls = [c for c in calls
                  if c[0][0]["content"].startswith("You are the Bull Researcher")]
    assert len(bull_calls) >= 2, "the repair attempt must actually happen"
    repair = bull_calls[1][0][1]["content"]
    assert "position size" not in repair.lower()
    assert "claims[0].claim" in repair
    assert rp._REDACTION in repair


def test_the_repair_does_not_hand_the_model_more_vocabulary_each_round():
    """The WM escalation, as a property.

    A stage that keeps failing must not receive a progressively larger list
    of forbidden phrases -- that is the mechanism that turned one violation
    into four.
    """
    body = json.loads(_bull_response())
    body["claims"][0]["claim"] = "Use a 5% position size for this name."
    violating = json.dumps(body)
    calls = []
    rp.run_research_pipeline(
        INDEX, make_ask_local_with_stage_sequence({"bull": [violating, violating, violating]},
                                                  record=calls))

    bull_prompts = [c[0][1]["content"] for c in calls
                    if c[0][0]["content"].startswith("You are the Bull Researcher")]
    assert len(bull_prompts) >= 2

    # Measured as a DELTA against the first prompt, not as an absolute count.
    # The shared guardrails legitimately name several of these phrases while
    # instructing against them (see the Phase 3 report), so the property that
    # matters is that a RETRY adds nothing new -- the escalation came from
    # the correction accreting vocabulary round after round.
    baseline = set(_phrases_in(bull_prompts[0]))
    for index, prompt in enumerate(bull_prompts[1:], start=1):
        added = set(_phrases_in(prompt)) - baseline
        assert added == set(), f"retry {index} introduced new banned phrases: {sorted(added)}"
    # ...and each retry stays the same size rather than accreting.
    assert len(set(len(p) for p in bull_prompts[1:])) <= 1


def test_the_raised_message_still_carries_the_labels_for_classification():
    """Invariant on a hard coupling: finance/workflow.py::
    _classify_content_policy_violation PARSES the message, slicing between
    "free-text fields: " and ". Every claim must be", to produce the
    user-facing reason line. Phase 3 changed the PROMPT, deliberately not the
    message."""
    from finance.workflow import _classify_content_policy_violation

    body = json.loads(_bull_response())
    body["claims"][0]["claim"] = "Use a 5% position size for this name."
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull=json.dumps(body)))
    error = result.by_stage("bull_researcher").error

    assert "free-text fields: " in error
    assert ". Every claim must be" in error
    assert _classify_content_policy_violation(error) == "used prohibited trade-advice language"


def test_failure_detail_is_absent_for_non_content_policy_failures():
    """Parse errors and transport errors carry no structured detail, and must
    not accidentally take the field-scoped branch."""
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull="not json"))
    checkpoint = result.by_stage("bull_researcher")
    assert checkpoint.failure_detail is None
    assert rp._correction_for(rp.StageFailureKind.MALFORMED_JSON, checkpoint.error, None)
