"""Item 5: does `ResearchReadiness.NOT_READY` permit a directional
BUY/HOLD/SELL/AVOID recommendation?

AUDIT RESULT: yes, deliberately, per the documented "LLM-owns-the-
recommendation patch" (`finance.research_pipeline._require_recommendation_
supported_by_readiness`). An earlier hard block (recommendation forced to
`insufficient_evidence` under NOT_READY) was REMOVED on the reasoning that a
structurally broken DCF does not mean the remaining fundamentals/technicals
evidence supports no view -- only that any view drawn from it is weakly
held. `insufficient_evidence` remains the normal, unconditional answer; any
other recommendation is permitted, but ONLY when BOTH:

  1. `limiting_factors` names the readiness limitation explicitly, and
  2. `confidence` is held at or below 0.35.

This is already a deterministic, centrally-enforced gate the LLM cannot vary
case by case -- `_Invalid` is raised (the stage fails, per this module's
existing "cascade-fatal on a structural violation" convention) if either
condition is missing. This file locks that policy in with the generalized
tests the phase requests; it found no bug in the invariant itself.

A separate, narrower observation (documented, not fixed here, out of this
phase's scope): the readiness status validated against is the BASE signal
computed BEFORE the research pipeline runs
(`finance.workflow.synthesize_report` passes `compact["research_readiness"]`
into `run_research_pipeline`); the EFFECTIVE, cascade-aware status a reader
actually sees in the rendered report is computed AFTER the pipeline
completes (`_effective_research_readiness`) and can be a stricter NOT_READY
that the base signal could not have anticipated (e.g. an unrelated required
stage failing to complete). A recommendation validated against a permissive
base readiness could therefore sit beside a rendered "Research readiness:
NOT_READY" banner. This is a real, checkable timing gap, but closing it
requires validating the synthesizer's own recommendation against a signal
that depends on whether the synthesizer itself completed -- a structural
circularity, not a small central fix -- so it is named here for a dedicated
future phase rather than patched under this one's "do not invent a new
policy casually" instruction.
"""

import pytest

from finance import research_pipeline as R


def _validated(confidence=0.9, limiting_factors=None, valuation_view="fairly_valued"):
    return {"confidence": confidence, "limiting_factors": limiting_factors or [],
            "valuation_view": valuation_view}


# ---------------------------------------------------------------------------
# H. NOT_READY recommendation behaviour follows ONE deterministic policy
# ---------------------------------------------------------------------------

def test_insufficient_evidence_needs_no_justification_under_not_ready():
    """The normal, unconditional answer -- no limiting_factors, no
    confidence floor required."""
    R._require_recommendation_supported_by_readiness(  # noqa: SLF001
        "insufficient_evidence", _validated(confidence=0.9, limiting_factors=[]),
        "NOT_READY")


@pytest.mark.parametrize("recommendation", ["buy", "sell", "hold", "avoid"])
def test_a_directional_recommendation_under_not_ready_requires_limiting_factors(
        recommendation):
    with pytest.raises(R._Invalid, match="limiting_factors"):
        R._require_recommendation_supported_by_readiness(
            recommendation, _validated(confidence=0.2, limiting_factors=[]), "NOT_READY")


@pytest.mark.parametrize("recommendation", ["buy", "sell", "hold", "avoid"])
def test_a_directional_recommendation_under_not_ready_has_a_confidence_ceiling(
        recommendation):
    with pytest.raises(R._Invalid, match="confidence"):
        R._require_recommendation_supported_by_readiness(
            recommendation,
            _validated(confidence=0.36, limiting_factors=["Readiness is NOT_READY: DCF failed."]),
            "NOT_READY")


@pytest.mark.parametrize("recommendation", ["buy", "sell", "hold", "avoid"])
def test_a_directional_recommendation_under_not_ready_is_permitted_when_both_are_met(
        recommendation):
    """The policy does NOT force insufficient_evidence -- see the module
    docstring's "LLM-owns-the-recommendation" rationale."""
    R._require_recommendation_supported_by_readiness(
        recommendation,
        _validated(confidence=0.35, limiting_factors=["Readiness is NOT_READY: DCF failed."]),
        "NOT_READY")


def test_the_confidence_ceiling_is_exactly_at_the_documented_value():
    assert R._NOT_READY_MAX_CONFIDENCE == 0.35


@pytest.mark.parametrize("readiness", ["READY", "LIMITED", None])
def test_the_gate_does_not_apply_outside_not_ready(readiness):
    """READY/LIMITED analyses are unconstrained by this specific check --
    other gates (freshness, valuation-direction) apply independently."""
    R._require_recommendation_supported_by_readiness(
        "buy", _validated(confidence=0.95, limiting_factors=[]), readiness)


def test_the_policy_is_deterministic_not_a_model_judgment_call():
    """Same inputs, same verdict, every time -- no field here is read as a
    hedge or a disclaimer the way `finance.claim_validation`'s tone checks
    are; this is a structural presence/threshold check only."""
    args = ("buy", _validated(confidence=0.35,
                              limiting_factors=["Readiness is NOT_READY: DCF failed."]),
           "NOT_READY")
    for _ in range(5):
        R._require_recommendation_supported_by_readiness(*args)
