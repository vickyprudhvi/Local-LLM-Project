"""A metric whose identity already fixes its denominator.

THE INVARIANT

The spec's denominator rule (§"units and denominator are part of the
identity") governs an ABSOLUTE metric carrying a percentage. "Operating income
of 21%" is ambiguous -- 21% of revenue, or 21% growth? -- and where the
sentence settles neither, the figure is not identifiable and is refused.

`gross_margin` is not that case. It is a metric whose canonical taxonomy unit
is RATIO and whose denominator is fixed by its own definition. There is no
competing "gross margin growth of 75%" reading for a denominator clause to
disambiguate: the metric name IS the denominator statement.

`resolve_identity` already draws this line correctly -- a ratio candidate
under a RATIO metric is kept, a ratio candidate under an ABSOLUTE metric goes
through `resolve_percentage_identity` or is refused. The defect was that
`denominator_is_grounded` restated the same decision WITHOUT consulting the
taxonomy, and so contradicted it: it demanded an "of revenue" clause from
every margin candidate, including ones whose metric could not be anything
else.

FOUND BY

The live semantic benchmark, on a release stating "GAAP and non-GAAP gross
margins are expected to be 74.9% and 75.0%, respectively, plus or minus 50
bps." The reader proposed both correctly and grounded both in that sentence;
the validator refused both. That is a validator failure by the change policy
in `docs/finance_extraction_v2.md` -- the candidate was semantically correct,
its grounding was correct, and the rejection contradicted an existing
invariant -- and it is a CLASS, affecting every canonically-ratio metric, not
one issuer.
"""

import pytest

from finance import guidance as gm
from finance.extraction.schema import GuidanceCandidate, ValueType
from finance.extraction.validator import (
    GuidanceCandidateValidator,
    denominator_is_grounded,
)

# The sentence as filed: it states the percentages and names the metric, and
# never says "of revenue" -- because a gross margin does not need to.
MARGIN_SENTENCE = ("GAAP and non-GAAP gross margins are expected to be 74.9% "
                   "and 75.0%, respectively, plus or minus 50 bps.")

DOCUMENT = ("Outlook\n\nFor the second quarter of fiscal 2027: "
            + MARGIN_SENTENCE
            + " GAAP and non-GAAP operating expenses are expected to be "
              "approximately $8.5 billion and $8.3 billion.")


def _margin_candidate(metric_id="adjusted_gross_margin", basis="NON_GAAP",
                      value=75.0, sentence=MARGIN_SENTENCE):
    return GuidanceCandidate(
        metric_id=metric_id, value_type=ValueType.MARGIN, value=value,
        unit="PERCENT", target_period="Q2 FY2027",
        target_period_type="quarterly", basis=basis, prospective=True,
        source_sentence=sentence, confidence=0.95)


# ---------------------------------------------------------------------------
# The metrics this is about really are ratio metrics in the taxonomy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "gross_margin", "adjusted_gross_margin", "operating_margin",
    "adjusted_operating_margin",
])
def test_these_metrics_carry_their_denominator_in_the_taxonomy(name):
    """The premise of the fix, asserted rather than assumed."""
    assert name in gm._METRIC_BY_NAME          # noqa: SLF001
    assert gm._METRIC_BY_NAME[name][1] == gm.GuidanceUnit.RATIO  # noqa: SLF001


# ---------------------------------------------------------------------------
# The fix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric_id,basis", [
    ("gross_margin", "GAAP"),
    ("adjusted_gross_margin", "NON_GAAP"),
    ("operating_margin", "GAAP"),
])
def test_a_canonically_ratio_metric_needs_no_of_revenue_clause(metric_id, basis):
    ok, reason = denominator_is_grounded(_margin_candidate(metric_id, basis))
    assert ok, reason


def test_the_margin_survives_the_whole_boundary_and_keeps_its_basis():
    validator = GuidanceCandidateValidator(document_text=DOCUMENT,
                                           issued_at="2026-05-27")
    metric, code, reason = validator.validate(_margin_candidate())
    assert metric is not None, f"{code}: {reason}"
    assert metric.name == "adjusted_gross_margin"
    assert metric.unit == gm.GuidanceUnit.RATIO
    assert metric.basis == gm.BASIS_ADJUSTED
    # Stated as 75.0 percent, stored as a ratio. The percentage is not the
    # unit; the identity is.
    assert metric.low == pytest.approx(0.75)


def test_both_bases_survive_as_two_separate_statements():
    """GAAP 74.9% and non-GAAP 75.0% are different quantities.

    A boundary that admitted one and dropped the other would silently pick a
    basis for the reader.
    """
    validator = GuidanceCandidateValidator(document_text=DOCUMENT,
                                           issued_at="2026-05-27")
    accepted, rejected = validator.validate_all([
        _margin_candidate("gross_margin", "GAAP", 74.9),
        _margin_candidate("adjusted_gross_margin", "NON_GAAP", 75.0),
    ])
    assert not rejected, rejected
    assert {m.basis for m in accepted} == {gm.BASIS_GAAP, gm.BASIS_ADJUSTED}


# ---------------------------------------------------------------------------
# ...and what the fix must NOT loosen
# ---------------------------------------------------------------------------

def test_an_absolute_metric_claiming_a_margin_is_still_refused():
    """The failure the original rule exists to stop, unchanged.

    `operating_income` is measured in currency. A percentage under it is
    ambiguous between a margin and a growth rate, and a sentence settling
    neither leaves the figure unidentifiable -- exactly the live defect that
    stored 0.21 as `operating_income`.
    """
    candidate = GuidanceCandidate(
        metric_id="operating_income", value_type=ValueType.MARGIN, value=21.0,
        unit="PERCENT", target_period="FY2027", target_period_type="annual",
        prospective=True, confidence=0.95,
        source_sentence="Operating income is expected to be 21%.")
    metric, code, _reason = GuidanceCandidateValidator(
        document_text="Outlook. Operating income is expected to be 21%.",
        issued_at="2026-05-27").validate(candidate)
    assert metric is None and code


def test_a_ratio_candidate_from_a_sentence_with_no_percentage_is_still_refused():
    """The other half of the check, which the live run confirmed working.

    A leverage TARGET stated as a multiple is not a percentage, and a
    candidate claiming a ratio from a sentence that states none has not read
    it out of the document.
    """
    sentence = ("The company targets net leverage in the 2.5x range within "
                "approximately three years.")
    candidate = GuidanceCandidate(
        metric_id="gross_margin", value_type=ValueType.MARGIN, value=2.5,
        unit="PERCENT", target_period="FY2027", target_period_type="annual",
        prospective=True, source_sentence=sentence, confidence=0.95)
    ok, reason = denominator_is_grounded(candidate)
    assert not ok and "percentage" in reason


def test_an_unknown_metric_name_is_still_refused():
    """The taxonomy lookup must not become a way in for an invented metric."""
    candidate = _margin_candidate(metric_id="vibe_margin")
    metric, code, _reason = GuidanceCandidateValidator(
        document_text=DOCUMENT, issued_at="2026-05-27").validate(candidate)
    assert metric is None and code
