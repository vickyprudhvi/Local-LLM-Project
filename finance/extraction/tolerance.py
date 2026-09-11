"""Turning a stated point and tolerance into endpoints. Deterministically.

WHY THIS IS NOT THE MODEL'S JOB

"Revenue is expected to be approximately $91.0 billion, plus or minus 2%"
contains three facts: 91.0, 2, and "plus or minus". It does not contain 89.18
or 92.82. A model that reports those endpoints has done arithmetic and
presented the result as something it read, and the grounding check is right to
refuse it -- the numbers are not in the document.

So the reader reports the OPERANDS and this module does the arithmetic. Every
operand is grounded in the sentence before it gets here; the operation is
fixed; and the result is marked DERIVED so nothing downstream can mistake it
for a figure the company published.

WHY THE BASIS MATTERS MORE THAN IT LOOKS

    $91.0 billion +/- 2%        -> 89.18 .. 92.82     (2% OF the point)
    gross margin 75.0% +/- 50bps -> 74.5 .. 75.5      (50bps ABSOLUTE)

Read the second as relative and the answer moves by a factor of 150. The basis
is therefore stated by the reader and CHECKED here against the units, and an
ambiguous combination is refused rather than guessed at.

FAIL CLOSED

Every §6 condition returns a reason instead of a number: a missing operand, an
unresolved or incompatible unit, a negative tolerance, an unsupported basis, a
non-finite result, or a derived low above its high. A tolerance that cannot be
resolved is not a range with a caveat -- it is not a range.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from finance.extraction.schema import (
    GuidanceCandidate,
    ToleranceBasis,
    ToleranceUnit,
    ValueType,
    domain_unit,
    is_percent_unit,
)

DERIVATION_TYPE = "GUIDANCE_TOLERANCE"

_BASIS_POINTS_PER_PERCENT = 100.0
_PERCENT_PER_UNIT = 100.0


@dataclass(frozen=True)
class ToleranceDerivation:
    """What was reported, what was computed, and how.

    Mirrors the provenance `finance/canonical.py` already records for a
    derived margin (`derivation_formula`, `source_metrics`): the inputs, the
    operation, and the fact that the output is derived.
    """

    derivation_type: str
    reported_point: float
    reported_tolerance: float
    tolerance_unit: str
    tolerance_basis: str
    derived_low: float
    derived_high: float
    formula: str
    operand_evidence: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "derivation_type": self.derivation_type,
            "reported_point": self.reported_point,
            "reported_tolerance": self.reported_tolerance,
            "tolerance_unit": self.tolerance_unit,
            "tolerance_basis": self.tolerance_basis,
            "derived_low": self.derived_low,
            "derived_high": self.derived_high,
            "formula": self.formula,
            "operand_evidence": list(self.operand_evidence),
        }


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def resolve_basis(candidate: GuidanceCandidate) -> Tuple[Optional[str], str]:
    """(basis, reason). The stated basis, checked against the units.

    Where the units permit only one reading, a stated basis that contradicts
    them is refused rather than quietly corrected -- the reader and the
    document disagree, and that is worth surfacing.
    """
    stated = (candidate.tolerance_basis or ToleranceBasis.UNKNOWN).upper()
    if stated not in ToleranceBasis.ALL:
        return None, f"tolerance basis {candidate.tolerance_basis!r} is not a supported basis"

    tolerance_unit = (candidate.tolerance_unit or "").upper()
    point_is_ratio = (is_percent_unit(candidate.unit)
                      or candidate.value_type in ValueType.RATIO_SHAPES)

    # An amount cannot carry an absolute percentage tolerance: "$91 billion
    # plus or minus 2%" has no reading where 2 is subtracted from 91.
    if tolerance_unit in (ToleranceUnit.PERCENT, ToleranceUnit.BASIS_POINTS) \
            and not point_is_ratio:
        if stated == ToleranceBasis.ABSOLUTE:
            return None, ("the point is an amount and the tolerance is a "
                          "percentage, so an absolute tolerance is not a "
                          "supported combination")
        return ToleranceBasis.OF_POINT_VALUE, ""

    # Basis points against a ratio are percentage POINTS by convention:
    # "75.0%, plus or minus 50 bps" is 74.5 to 75.5.
    if tolerance_unit == ToleranceUnit.BASIS_POINTS and point_is_ratio:
        if stated == ToleranceBasis.OF_POINT_VALUE:
            return None, ("basis points against a ratio are percentage points; "
                          "a relative reading was stated and is not supported")
        return ToleranceBasis.ABSOLUTE, ""

    # A tolerance in the point's own unit is absolute by construction:
    # "EPS $2.50 plus or minus $0.10".
    if tolerance_unit == ToleranceUnit.SAME_AS_POINT:
        if stated == ToleranceBasis.OF_POINT_VALUE:
            return None, ("a tolerance stated in the point's own unit is an "
                          "absolute amount, not a proportion of it")
        return ToleranceBasis.ABSOLUTE, ""

    # A percentage tolerance on a percentage point is genuinely ambiguous --
    # "margin of 35%, plus or minus 2%" could be 34.98-35.02 or 33-37 -- so
    # the reader must say which, and silence is refused.
    if tolerance_unit == ToleranceUnit.PERCENT and point_is_ratio:
        if stated == ToleranceBasis.UNKNOWN:
            return None, ("a percentage tolerance on a percentage figure is "
                          "ambiguous and the sentence's reading was not stated")
        return stated, ""

    return None, f"tolerance unit {candidate.tolerance_unit!r} could not be resolved"


def derive(candidate: GuidanceCandidate
           ) -> Tuple[Optional[ToleranceDerivation], str]:
    """(derivation, reason). Exactly one is meaningful.

    Runs AFTER the operands have been grounded. This function does not look
    at the document: its job is arithmetic on numbers already checked against
    it.
    """
    point = candidate.value
    if point is None and candidate.low is not None and candidate.low == candidate.high:
        point = candidate.low
    if not _finite(point):
        return None, "a tolerance needs a point value and none was reported"

    tolerance = candidate.tolerance_value
    if not _finite(tolerance):
        return None, "a tolerance needs a tolerance value and none was reported"
    if tolerance < 0:
        return None, ("a negative tolerance is not a width; the sentence's "
                      "'plus or minus' already carries the sign")

    tolerance_unit = (candidate.tolerance_unit or "").upper()
    if tolerance_unit not in ToleranceUnit.ALL:
        return None, f"tolerance unit {candidate.tolerance_unit!r} is not supported"

    basis, why = resolve_basis(candidate)
    if basis is None:
        return None, why

    if basis == ToleranceBasis.OF_POINT_VALUE:
        if tolerance_unit == ToleranceUnit.BASIS_POINTS:
            fraction = tolerance / (_BASIS_POINTS_PER_PERCENT * _PERCENT_PER_UNIT)
        else:
            fraction = tolerance / _PERCENT_PER_UNIT
        low, high = point * (1.0 - fraction), point * (1.0 + fraction)
        formula = f"{point} x (1 -/+ {fraction})"
    else:
        width = tolerance
        if tolerance_unit == ToleranceUnit.BASIS_POINTS:
            # The point is stated in percent, so basis points convert to
            # percent. Converting the point instead would silently change the
            # unit the figure is carried in.
            width = tolerance / _BASIS_POINTS_PER_PERCENT
        low, high = point - width, point + width
        formula = f"{point} -/+ {width}"

    if not (_finite(low) and _finite(high)):
        return None, "the derived endpoints are not finite"
    if low > high:
        return None, "the derived low is above the derived high"

    _domain, scale = domain_unit(candidate.unit)
    return ToleranceDerivation(
        derivation_type=DERIVATION_TYPE,
        reported_point=float(point),
        reported_tolerance=float(tolerance),
        tolerance_unit=tolerance_unit,
        tolerance_basis=basis,
        derived_low=float(low), derived_high=float(high),
        formula=formula,
        operand_evidence=tuple(
            part for part in (
                f"point={point}{(' ' + scale) if scale else ''}",
                f"tolerance={tolerance} {tolerance_unit}",
                f"basis={basis}") if part)), ""
