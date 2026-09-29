"""A point with a tolerance, and who is allowed to do the arithmetic.

THE PROBLEM

"Revenue is expected to be approximately $91.0 billion, plus or minus 2%"
contains 91.0, 2, and "plus or minus". It does not contain 89.18 or 92.82.

A reader that reports those endpoints has calculated them and presented the
result as something it read, and the grounding check refuses it -- correctly,
because the numbers are not in the document. So the guidance was lost, and the
live benchmark recorded it as a validator rejection when the real fault was
that the schema had no way to say "a point, and a width around it".

THE SHAPE OF THE FIX

The reader reports OPERANDS. `finance.extraction.tolerance` does the
arithmetic. The boundary grounds the operands against the sentence and marks
the result DERIVED, so nothing downstream can mistake a computed endpoint for
a published one.

WHAT IS NOT WEAKENED

Ordinary point and range candidates are still held to literal grounding: a
candidate that reports 89.18 as a `low` is claiming the company said 89.18,
and it is refused unless the company did.
"""

import math

import pytest

from finance import guidance as gm
from finance.extraction import tolerance as tol
from finance.extraction.schema import (
    GuidanceCandidate,
    RejectionCode,
    ToleranceBasis,
    ToleranceUnit,
    ValueType,
)
from finance.extraction.validator import GuidanceCandidateValidator

REVENUE_SENTENCE = ("Revenue is expected to be approximately $91.0 billion, "
                    "plus or minus 2%.")
MARGIN_SENTENCE = ("GAAP and non-GAAP gross margins are expected to be 74.9% "
                   "and 75.0%, respectively, plus or minus 50 bps.")
EPS_SENTENCE = ("Adjusted EPS is expected to be $2.50, plus or minus $0.10.")
RANGE_SENTENCE = ("Revenue is expected to be $89.0 billion to $93.0 billion.")

DOCUMENT = ("Outlook. " + REVENUE_SENTENCE + " " + MARGIN_SENTENCE + " "
            + EPS_SENTENCE + " " + RANGE_SENTENCE)


def _tolerance(metric_id="revenue", value=91.0, unit="USD_BILLION",
               tolerance_value=2.0, tolerance_unit=ToleranceUnit.PERCENT,
               tolerance_basis=ToleranceBasis.OF_POINT_VALUE,
               sentence=REVENUE_SENTENCE, period="FY2027", **kwargs):
    return GuidanceCandidate(
        metric_id=metric_id, value_type=ValueType.TOLERANCE, value=value,
        unit=unit, tolerance_value=tolerance_value,
        tolerance_unit=tolerance_unit, tolerance_basis=tolerance_basis,
        target_period=period, target_period_type="annual", prospective=True,
        source_sentence=sentence, confidence=0.95, **kwargs)


def _validate(candidate, document=DOCUMENT):
    return GuidanceCandidateValidator(
        document_text=document, issued_at="2026-05-27").validate(candidate)


# ---------------------------------------------------------------------------
# A/B/C. structured representation, operand grounding, derivation
# ---------------------------------------------------------------------------

def test_the_operands_are_what_gets_grounded():
    metric, code, reason = _validate(_tolerance())
    assert metric is not None, f"{code}: {reason}"


def test_the_derived_endpoints_are_the_arithmetic_the_document_implies():
    metric, _code, _reason = _validate(_tolerance())
    assert metric.low == pytest.approx(89.18)
    assert metric.high == pytest.approx(92.82)


def test_basis_points_against_a_margin_are_percentage_points():
    """75.0% +/- 50bps is 74.5..75.5, not 75 x 0.995.

    The relative reading moves the answer by a factor of 150, which is why
    the basis is stated and checked rather than assumed.
    """
    metric, code, reason = _validate(_tolerance(
        metric_id="adjusted_gross_margin", value=75.0, unit="PERCENT",
        tolerance_value=50.0, tolerance_unit=ToleranceUnit.BASIS_POINTS,
        tolerance_basis=ToleranceBasis.ABSOLUTE, sentence=MARGIN_SENTENCE,
        basis="NON_GAAP", period="Q2 FY2027"))
    assert metric is not None, f"{code}: {reason}"
    assert metric.low == pytest.approx(0.745)
    assert metric.high == pytest.approx(0.755)


def test_an_absolute_tolerance_in_the_points_own_unit():
    metric, code, reason = _validate(_tolerance(
        metric_id="adjusted_earnings_per_share", value=2.50, unit="PER_SHARE",
        tolerance_value=0.10, tolerance_unit=ToleranceUnit.SAME_AS_POINT,
        tolerance_basis=ToleranceBasis.ABSOLUTE, sentence=EPS_SENTENCE,
        basis="NON_GAAP"))
    assert metric is not None, f"{code}: {reason}"
    assert metric.low == pytest.approx(2.40)
    assert metric.high == pytest.approx(2.60)


# ---------------------------------------------------------------------------
# D. provenance
# ---------------------------------------------------------------------------

def test_derived_endpoints_carry_derived_provenance():
    """A computed endpoint must never be indistinguishable from a stated one."""
    metric, _code, _reason = _validate(_tolerance())
    assert metric.derivation_type == tol.DERIVATION_TYPE
    assert metric.derivation_formula
    assert metric.bound_type == gm.GuidanceBound.POINT_WITH_TOLERANCE
    operands = " ".join(metric.derivation_operands)
    assert "91.0" in operands and "2.0" in operands
    assert ToleranceBasis.OF_POINT_VALUE in operands


def test_the_reported_operands_survive_on_the_derivation():
    derivation, why = tol.derive(_tolerance())
    assert derivation is not None, why
    assert derivation.reported_point == 91.0
    assert derivation.reported_tolerance == 2.0
    assert derivation.derived_low == pytest.approx(89.18)


# ---------------------------------------------------------------------------
# E/F. a computed endpoint presented as a reported one; direct ranges
# ---------------------------------------------------------------------------

def test_a_computed_endpoint_reported_as_a_range_is_still_refused():
    """The behaviour that must NOT be relaxed.

    Claiming `low=89.18` is claiming the company said 89.18. It did not, and
    the structured-tolerance path is not a way around that.
    """
    candidate = GuidanceCandidate(
        metric_id="revenue", value_type=ValueType.RANGE, low=89.18, high=92.82,
        unit="USD_BILLION", target_period="FY2027", target_period_type="annual",
        prospective=True, source_sentence=REVENUE_SENTENCE, confidence=0.95)
    metric, code, _reason = _validate(candidate)
    assert metric is None
    assert code == RejectionCode.VALUE_NOT_IN_EVIDENCE


def test_a_stated_range_stays_a_stated_range():
    candidate = GuidanceCandidate(
        metric_id="revenue", value_type=ValueType.RANGE, low=89.0, high=93.0,
        unit="USD_BILLION", target_period="FY2027", target_period_type="annual",
        prospective=True, source_sentence=RANGE_SENTENCE, confidence=0.95)
    metric, code, reason = _validate(candidate)
    assert metric is not None, f"{code}: {reason}"
    assert metric.bound_type == gm.GuidanceBound.RANGE
    assert metric.derivation_type is None, "nothing was derived here"


def test_the_two_shapes_are_distinguishable_downstream():
    """§7: both expose low/high; their provenance must not be collapsed."""
    derived, _c, _r = _validate(_tolerance())
    stated, _c, _r = _validate(GuidanceCandidate(
        metric_id="revenue", value_type=ValueType.RANGE, low=89.0, high=93.0,
        unit="USD_BILLION", target_period="FY2027", target_period_type="annual",
        prospective=True, source_sentence=RANGE_SENTENCE, confidence=0.95))
    assert derived.bound_type != stated.bound_type
    assert bool(derived.derivation_type) != bool(stated.derivation_type)


def test_two_numbers_without_a_tolerance_phrase_are_not_a_tolerance():
    """Otherwise any sentence with two figures could be reshaped into one."""
    sentence = "Revenue was $91.0 billion and operating income was $2.0 billion."
    metric, code, _reason = _validate(
        _tolerance(sentence=sentence), document="Outlook. " + sentence)
    assert metric is None
    assert code in (RejectionCode.VALUE_NOT_IN_EVIDENCE,
                    RejectionCode.NOT_PROSPECTIVE)


# ---------------------------------------------------------------------------
# G/H/I. fail closed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,expect", [
    ({"value": None}, "point value"),
    ({"tolerance_value": None}, "tolerance value"),
    ({"tolerance_unit": "FURLONGS"}, "not supported"),
    ({"tolerance_value": -2.0}, "negative"),
])
def test_derivation_fails_closed(kwargs, expect):
    derivation, why = tol.derive(_tolerance(**kwargs))
    assert derivation is None
    assert expect in why


def test_an_absolute_percentage_tolerance_on_an_amount_is_refused():
    """"$91 billion plus or minus 2%" has no reading where 2 is subtracted."""
    derivation, why = tol.derive(_tolerance(
        tolerance_basis=ToleranceBasis.ABSOLUTE))
    assert derivation is None
    assert "not a supported combination" in why


def test_an_ambiguous_percentage_on_a_percentage_is_refused():
    """"margin of 35%, plus or minus 2%" could be 34.98-35.02 or 33-37."""
    derivation, why = tol.derive(_tolerance(
        metric_id="operating_margin", value=35.0, unit="PERCENT",
        tolerance_value=2.0, tolerance_unit=ToleranceUnit.PERCENT,
        tolerance_basis=ToleranceBasis.UNKNOWN))
    assert derivation is None
    assert "ambiguous" in why


def test_basis_points_claimed_as_relative_are_refused():
    derivation, why = tol.derive(_tolerance(
        metric_id="adjusted_gross_margin", value=75.0, unit="PERCENT",
        tolerance_value=50.0, tolerance_unit=ToleranceUnit.BASIS_POINTS,
        tolerance_basis=ToleranceBasis.OF_POINT_VALUE))
    assert derivation is None
    assert "percentage points" in why


def test_a_non_finite_operand_is_refused():
    derivation, why = tol.derive(_tolerance(value=float("inf")))
    assert derivation is None
    assert "point value" in why


def test_an_undrivable_tolerance_is_refused_by_the_boundary_with_its_own_code():
    metric, code, _reason = _validate(_tolerance(tolerance_unit="FURLONGS"))
    assert metric is None
    assert code == RejectionCode.TOLERANCE_NOT_DERIVABLE


def test_a_tolerance_never_reaches_the_metric_when_derivation_fails():
    """Fail closed: no partial statement, no midpoint, nothing."""
    accepted, rejected = GuidanceCandidateValidator(
        document_text=DOCUMENT, issued_at="2026-05-27").validate_all(
            [_tolerance(tolerance_value=None)])
    assert accepted == []
    assert len(rejected) == 1


# ---------------------------------------------------------------------------
# L. the existing refusal guards are untouched
# ---------------------------------------------------------------------------

def test_a_tolerance_row_inside_a_reported_table_is_still_refused():
    """A ROW, not a sentence -- the case block scope exists for.

    A sentence that says "is expected to be" qualifies itself wherever it
    sits, and that is correct. What must not pass is a bare row that carries
    no vocabulary and whose only governing caption says these numbers already
    happened.
    """
    row = "Revenue $91.0 billion plus or minus 2%."
    document = ("Condensed Consolidated Statements of Operations . "
                "Three Months Ended June 30 . " + row)
    metric, code, _reason = _validate(_tolerance(sentence=row),
                                      document=document)
    assert metric is None
    assert code == RejectionCode.NOT_PROSPECTIVE


def test_the_same_tolerance_row_under_a_guidance_caption_is_accepted():
    """The other half, so the test above is measuring the caption and not
    simply the absence of vocabulary in the row."""
    row = "Revenue $91.0 billion plus or minus 2%."
    document = ("The following table summarizes our guidance for the coming "
                "year. Current Guidance . Q1 Estimates . " + row)
    metric, code, reason = _validate(_tolerance(sentence=row),
                                     document=document)
    assert metric is not None, f"{code}: {reason}"
    assert metric.low == pytest.approx(89.18)


def test_a_tolerance_attributed_to_analysts_is_still_refused():
    sentence = ("Analysts expect revenue of approximately $91.0 billion, plus "
                "or minus 2%.")
    metric, code, _reason = _validate(
        _tolerance(sentence=sentence), document="Outlook. " + sentence)
    assert metric is None
    assert code == RejectionCode.ANALYST_ESTIMATE


def test_a_tolerance_whose_sentence_is_not_in_the_document_is_refused():
    metric, code, _reason = _validate(
        _tolerance(sentence="Revenue is expected to be approximately $95.0 "
                            "billion, plus or minus 2%."))
    assert metric is None
    assert code == RejectionCode.EVIDENCE_NOT_IN_SOURCE


def test_the_derivation_never_produces_a_low_above_its_high():
    for tolerance_value in (0.0, 1.0, 50.0, 99.0):
        derivation, why = tol.derive(_tolerance(tolerance_value=tolerance_value))
        if derivation is None:
            continue
        assert derivation.derived_low <= derivation.derived_high, why
        assert math.isfinite(derivation.derived_low)
        assert math.isfinite(derivation.derived_high)
