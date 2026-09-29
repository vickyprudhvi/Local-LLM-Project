"""The candidate types V2 produces, and the vocabularies they use.

A candidate is a PROPOSAL. It carries everything the validator needs to
decide whether to accept it and everything a reader needs to check that
decision — including the exact sentence it came from, because a candidate
that cannot point at its own evidence is not checkable and is refused.

Vocabularies are imported from the existing domain model wherever one
exists (§5: "use existing project enums rather than blindly adding parallel
vocabulary"). Only two are new, and both name something the domain had no
word for: how COMPLETE a statement set is, and what KIND OF SOURCE reported
it.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from finance import guidance as gm
from finance import semantics as sem


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

class ValueType:
    """The SHAPE of a guided figure.

    A range and a floor are different claims: "$2.25 to $2.35" states both
    ends, "$18 billion+" states only a minimum, and reading the second's
    midpoint as a forecast invents a number the company withheld. This
    mirrors `guidance.GuidanceBound` and adds the shapes a margin or a growth
    rate arrives in.
    """

    POINT = "point"
    RANGE = "range"
    FLOOR = "floor"
    CEILING = "ceiling"
    GROWTH_RATE = "growth_rate"
    MARGIN = "margin"
    # "$91.0 billion, plus or minus 2%" is a POINT plus a TOLERANCE, and the
    # endpoints are ours to compute, not the issuer's to state. Keeping it a
    # distinct shape is what lets the boundary ground the OPERANDS the
    # sentence really contains (91.0, 2, "plus or minus") instead of demanding
    # 89.18 and 92.82, which appear nowhere in the document.
    #
    # It is also a different CLAIM from a stated range: "$89B to $93B" is the
    # company naming two endpoints, "$91B +/-2%" is the company naming one
    # number and an uncertainty around it. Both end up with a low and a high;
    # only one of them was reported that way.
    TOLERANCE = "tolerance"
    UNKNOWN = "unknown"

    ALL = (POINT, RANGE, FLOOR, CEILING, GROWTH_RATE, MARGIN, TOLERANCE, UNKNOWN)

    # Shapes whose value is a RATIO of something. Each must name what it is a
    # ratio OF, or it cannot be identified -- the lesson of "operating income
    # = 21% of projected revenue" being stored as $21.
    RATIO_SHAPES = (GROWTH_RATE, MARGIN)


class ToleranceUnit:
    """What a tolerance is measured in.

    Deliberately small. A tolerance is not a general quantity -- it is a
    percentage, a count of basis points, or an amount in the point's own unit.
    Anything else is refused rather than guessed at.
    """

    PERCENT = "PERCENT"
    BASIS_POINTS = "BASIS_POINTS"
    SAME_AS_POINT = "SAME_AS_POINT"
    ALL = (PERCENT, BASIS_POINTS, SAME_AS_POINT)


class ToleranceBasis:
    """What the tolerance is measured AGAINST, which changes the arithmetic.

    "$91.0 billion plus or minus 2%" is 2% OF THE POINT: the endpoints are
    91 x 0.98 and 91 x 1.02.

    "gross margin of 75.0%, plus or minus 50 bps" is 50 bps ABSOLUTE: the
    endpoints are 74.5% and 75.5%, not 75 x 0.995. Reading the second as
    relative would move the answer by a factor of 150.

    So the basis is stated, never inferred. Where the units make only one
    reading possible the deterministic layer checks the stated basis against
    them; where both readings are possible (a percentage point plus a
    percentage tolerance) an unstated basis is refused.
    """

    OF_POINT_VALUE = "OF_POINT_VALUE"
    ABSOLUTE = "ABSOLUTE"
    UNKNOWN = "UNKNOWN"
    ALL = (OF_POINT_VALUE, ABSOLUTE, UNKNOWN)


class SourceType:
    """WHERE a figure was reported, which decides how much authority it has.

    The distinction the architecture had no word for. A 10-K and an earnings
    release both state actual results; one is audited and final and the other
    is preliminary. Collapsing them meant a newer complete earnings release
    could not advance the current state at all, because the only question
    ever asked was "is this a periodic filing?".
    """

    FORMAL_PERIODIC_FILING = "FORMAL_PERIODIC_FILING"     # 10-K, 10-Q, 20-F, 6-K
    PRELIMINARY_EARNINGS_RELEASE = "PRELIMINARY_EARNINGS_RELEASE"   # 8-K item 2.02
    OTHER = "OTHER"

    ALL = (FORMAL_PERIODIC_FILING, PRELIMINARY_EARNINGS_RELEASE, OTHER)

    # Authority ranking for a TIE on period end. Lower is stronger. It breaks
    # ties only: a NEWER period beats a stronger source, because the question
    # "what is the latest reported state" is about the period first.
    AUTHORITY = {FORMAL_PERIODIC_FILING: 0, PRELIMINARY_EARNINGS_RELEASE: 1, OTHER: 2}


class StatementCompleteness:
    """How much of a statement set a source actually carries.

    §13's guard against over-correction: a newer press release may only
    advance the current state if it is COMPLETE. An incomplete one supplies
    per-metric candidates with explicit provenance and never replaces a
    complete filing wholesale.
    """

    COMPLETE = "COMPLETE"        # income statement + balance sheet + cash flow
    PARTIAL = "PARTIAL"          # some statements, or some required metrics
    HEADLINE_ONLY = "HEADLINE_ONLY"
    NONE = "NONE"

    ALL = (COMPLETE, PARTIAL, HEADLINE_ONLY, NONE)


class FinalityStatus:
    """Whether the figures are final. Provenance, never a filter."""

    AUDITED = "AUDITED"
    UNAUDITED_PRELIMINARY = "UNAUDITED_PRELIMINARY"
    UNKNOWN = "UNKNOWN"

    ALL = (AUDITED, UNAUDITED_PRELIMINARY, UNKNOWN)


class GuidanceAction:
    """What this statement does to any earlier statement of the same thing.

    A reaffirmation is not a second forecast. Without this, one outlook
    repeated across three releases looked like three independent signals
    agreeing with each other, which is a much stronger-sounding thing than
    one company saying one thing once.
    """

    NEW = "NEW"
    RAISED = "RAISED"
    LOWERED = "LOWERED"
    REITERATED = "REITERATED"
    REAFFIRMED = "REAFFIRMED"
    WITHDRAWN = "WITHDRAWN"
    SUPERSEDED = "SUPERSEDED"
    UNKNOWN = "UNKNOWN"

    ALL = (NEW, RAISED, LOWERED, REITERATED, REAFFIRMED, WITHDRAWN, SUPERSEDED,
           UNKNOWN)

    # Actions that RESTATE an existing figure rather than adding a new one.
    # These may update the current signal; they may never become a second
    # independent one (§10).
    RESTATEMENTS = (REITERATED, REAFFIRMED)


class RejectionCode:
    """Why a candidate was refused. Structured, so a benchmark can count the
    CLASSES of failure rather than listing individual sentences."""

    NO_EVIDENCE = "NO_EVIDENCE"
    EVIDENCE_NOT_IN_SOURCE = "EVIDENCE_NOT_IN_SOURCE"
    VALUE_NOT_IN_EVIDENCE = "VALUE_NOT_IN_EVIDENCE"
    UNKNOWN_METRIC = "UNKNOWN_METRIC"
    UNIT_METRIC_MISMATCH = "UNIT_METRIC_MISMATCH"
    DENOMINATOR_MISSING = "DENOMINATOR_MISSING"
    DENOMINATOR_NOT_IN_EVIDENCE = "DENOMINATOR_NOT_IN_EVIDENCE"
    NOT_PROSPECTIVE = "NOT_PROSPECTIVE"
    AMBIGUOUS_TARGET_PERIOD = "AMBIGUOUS_TARGET_PERIOD"
    TARGET_PERIOD_IMPLAUSIBLE = "TARGET_PERIOD_IMPLAUSIBLE"
    TARGET_PERIOD_COMPLETED = "TARGET_PERIOD_COMPLETED"
    INCOMPATIBLE_BASIS = "INCOMPATIBLE_BASIS"
    HISTORICAL_TABLE = "HISTORICAL_TABLE"
    ANALYST_ESTIMATE = "ANALYST_ESTIMATE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    # A point-with-tolerance whose endpoints could not be computed. Distinct
    # from VALUE_NOT_IN_EVIDENCE: the operands were found, the arithmetic was
    # refused. Fail closed -- an unresolvable tolerance is not a range with a
    # caveat, it is not a range.
    TOLERANCE_NOT_DERIVABLE = "TOLERANCE_NOT_DERIVABLE"
    # The cited evidence does not describe the metric claimed -- a component
    # measure read as the consolidated one, or a sentence about something
    # else entirely. Numeric grounding cannot catch this: the numbers, the
    # unit and the period were all correct.
    METRIC_NOT_GROUNDED = "METRIC_NOT_GROUNDED"
    MALFORMED = "MALFORMED"
    DUPLICATE_ECONOMIC_IDENTITY = "DUPLICATE_ECONOMIC_IDENTITY"

    ALL = (NO_EVIDENCE, EVIDENCE_NOT_IN_SOURCE, VALUE_NOT_IN_EVIDENCE,
           UNKNOWN_METRIC, UNIT_METRIC_MISMATCH, DENOMINATOR_MISSING,
           DENOMINATOR_NOT_IN_EVIDENCE, NOT_PROSPECTIVE,
           AMBIGUOUS_TARGET_PERIOD, TARGET_PERIOD_IMPLAUSIBLE,
           TARGET_PERIOD_COMPLETED, INCOMPATIBLE_BASIS, HISTORICAL_TABLE,
           ANALYST_ESTIMATE, LOW_CONFIDENCE, MALFORMED,
           DUPLICATE_ECONOMIC_IDENTITY)

    # Rejections that mean the model asserted something the source does not
    # support. These are the ones a benchmark must count as CRITICAL: a
    # system with fewer facts and none of these beats one with more facts and
    # any of them (§18).
    UNSUPPORTED = (EVIDENCE_NOT_IN_SOURCE, VALUE_NOT_IN_EVIDENCE,
                   DENOMINATOR_NOT_IN_EVIDENCE, HISTORICAL_TABLE,
                   ANALYST_ESTIMATE)


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuidanceCandidate:
    """One proposed forward statement, with the sentence it came from.

    Frozen: a candidate that has been validated must not be edited on the way
    to the canonical layer, which is the same rule
    `ValidatedDCFInputPacket` enforces for valuation inputs and for the same
    reason.
    """

    metric_id: Optional[str] = None
    value_type: str = ValueType.UNKNOWN
    low: Optional[float] = None
    high: Optional[float] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    denominator_metric: Optional[str] = None
    target_period: Optional[str] = None
    target_period_type: Optional[str] = None
    comparison_period: Optional[str] = None
    basis: Optional[str] = None
    # A POINT plus an uncertainty around it, kept structurally. The model
    # reports what the sentence SAYS -- the point, the tolerance and what the
    # tolerance is measured against -- and never the endpoints, which are
    # arithmetic and belong to `finance.extraction.tolerance`.
    tolerance_value: Optional[float] = None
    tolerance_unit: Optional[str] = None
    tolerance_basis: str = ToleranceBasis.UNKNOWN
    action: str = GuidanceAction.UNKNOWN
    prospective: bool = True
    source_sentence: str = ""
    source_span: Optional[Tuple[int, int]] = None
    section_label: Optional[str] = None
    confidence: float = 0.0
    # Filled by the extractor, never by the model.
    document_id: Optional[str] = None
    issued_at: Optional[str] = None

    def bounds(self) -> Tuple[Optional[float], Optional[float]]:
        """(low, high) whichever shape the model used to express it."""
        if self.low is not None or self.high is not None:
            return self.low, self.high
        return self.value, self.value

    def to_dict(self) -> dict:
        low, high = self.bounds()
        return {
            "metric_id": self.metric_id, "value_type": self.value_type,
            "low": low, "high": high, "unit": self.unit,
            "denominator_metric": self.denominator_metric,
            "target_period": self.target_period,
            "target_period_type": self.target_period_type,
            "comparison_period": self.comparison_period,
            "basis": self.basis, "action": self.action,
            "prospective": self.prospective,
            "source_sentence": self.source_sentence,
            "section_label": self.section_label,
            "confidence": self.confidence,
            "document_id": self.document_id, "issued_at": self.issued_at,
        }


@dataclass(frozen=True)
class ReportedActualCandidate:
    """A period of ACTUAL results offered by one source.

    `PRELIMINARY_REPORTED_ACTUAL` in §2's language is this with
    `source_type=PRELIMINARY_EARNINGS_RELEASE`. The type is one thing with a
    source attribute rather than two types, because everything downstream
    asks the same questions of it -- which period, how complete, how
    authoritative -- and only the answers differ.
    """

    period_end: Optional[str] = None
    period_start: Optional[str] = None
    period_type: Optional[str] = None        # sem.PeriodFrequency
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None
    issued_at: Optional[str] = None
    source_type: str = SourceType.OTHER
    form: Optional[str] = None
    accession: Optional[str] = None
    statement_completeness: str = StatementCompleteness.NONE
    finality: str = FinalityStatus.UNKNOWN
    present_metrics: Tuple[str, ...] = ()
    missing_metrics: Tuple[str, ...] = ()
    currency: Optional[str] = None
    entity_id: Optional[str] = None
    # The reported numbers, where the source supplied them. Needed to tell a
    # MATERIAL disagreement between two sources for one period from a rounding
    # difference; `present_metrics` says only that a metric was reported.
    values: Dict[str, float] = field(default_factory=dict)
    # An earnings release carries reported results AND an outlook. A candidate
    # built from the outlook is never a reported actual, whatever else is true
    # of the document it came from.
    is_prospective: bool = False

    @property
    def authority(self) -> int:
        return SourceType.AUTHORITY.get(self.source_type, 9)

    def to_dict(self) -> dict:
        return {
            "period_end": self.period_end, "period_start": self.period_start,
            "values": dict(self.values), "is_prospective": self.is_prospective,
            "period_type": self.period_type, "fiscal_year": self.fiscal_year,
            "fiscal_period": self.fiscal_period, "issued_at": self.issued_at,
            "source_type": self.source_type, "form": self.form,
            "accession": self.accession,
            "statement_completeness": self.statement_completeness,
            "finality": self.finality,
            "present_metrics": list(self.present_metrics),
            "missing_metrics": list(self.missing_metrics),
            "currency": self.currency, "entity_id": self.entity_id,
        }


@dataclass
class ExtractionOutcome:
    """What one extraction run produced, and what it refused.

    Both halves matter. A layer that reports only what it accepted cannot be
    measured: the benchmark's most important number is how often a candidate
    was refused because the source did not support it (§18), and that is
    knowable only if refusals are recorded with their reason.
    """

    accepted: List[GuidanceCandidate] = field(default_factory=list)
    rejected: List[Tuple[GuidanceCandidate, str, str]] = field(default_factory=list)
    actuals: List[ReportedActualCandidate] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    extractor_version: str = ""
    document_id: Optional[str] = None
    sections_used: Tuple[str, ...] = ()

    def rejection_codes(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _candidate, code, _reason in self.rejected:
            counts[code] = counts.get(code, 0) + 1
        return counts

    def unsupported_count(self) -> int:
        """Candidates the source did not support. The number that matters."""
        return sum(1 for _c, code, _r in self.rejected
                   if code in RejectionCode.UNSUPPORTED)

    def to_dict(self) -> dict:
        return {
            "accepted": [c.to_dict() for c in self.accepted],
            "rejected": [{"candidate": c.to_dict(), "code": code, "reason": reason}
                         for c, code, reason in self.rejected],
            "actuals": [a.to_dict() for a in self.actuals],
            "notes": list(self.notes),
            "extractor_version": self.extractor_version,
            "document_id": self.document_id,
            "sections_used": list(self.sections_used),
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "rejection_codes": self.rejection_codes(),
        }


# ---------------------------------------------------------------------------
# Unit vocabulary, mapped onto the domain's own
# ---------------------------------------------------------------------------
#
# The model is given a small, explicit unit vocabulary because "USD" and
# "USD_BILLION" are questions it can answer from the text, while
# `GuidanceUnit.CURRENCY` plus a separate `scale` field is an internal
# representation it has no way to reason about. The translation is here, in
# one place, so the domain keeps its own vocabulary.

_MODEL_UNITS = {
    "USD": (gm.GuidanceUnit.CURRENCY, None),
    "USD_MILLION": (gm.GuidanceUnit.CURRENCY, "million"),
    "USD_BILLION": (gm.GuidanceUnit.CURRENCY, "billion"),
    "PER_SHARE": (gm.GuidanceUnit.CURRENCY_PER_SHARE, None),
    "RATIO": (gm.GuidanceUnit.RATIO, None),
    "PERCENT": (gm.GuidanceUnit.RATIO, None),
    "SHARES": (gm.GuidanceUnit.SHARES, None),
    "UNKNOWN": (None, None),
    # Phase H.22 deliberately does NOT add a "CURRENCY" (or any other) key
    # here: this dict is SHARED with guidance extraction's own prompt
    # schema (`finance.extraction.semantic_extractor` imports
    # `MODEL_UNIT_NAMES` directly to build it), and touching guidance's
    # architecture is a standing cross-phase constraint this project has
    # held since it began. An earlier draft of this phase added "CURRENCY"
    # here; a full-suite pytest run then showed 4 guidance-canary failures
    # that did not reproduce in isolation or in any targeted subset. That
    # specific correlation did NOT hold up under test (reverting this key
    # alone did not make the failures go away -- they are pre-existing
    # full-suite-only test-order fragility in the canary harness, confirmed
    # unrelated to this phase's code). The key stays OUT regardless: the
    # events reader's currency-neutral "CURRENCY" unit token is a LOCAL
    # concept of `finance.documents.monetary`/`event_extractor.py` instead,
    # resolved without ever touching this shared dictionary -- not because
    # it was proven to cause the canary flakiness, but because a reader
    # this phase does not own has no reason to see a vocabulary change it
    # did not ask for.
}

MODEL_UNIT_NAMES = tuple(_MODEL_UNITS)


def domain_unit(model_unit: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """(GuidanceUnit, scale) for a unit name the model returned."""
    return _MODEL_UNITS.get((model_unit or "UNKNOWN").upper(), (None, None))


def is_percent_unit(model_unit: Optional[str]) -> bool:
    return (model_unit or "").upper() in ("PERCENT", "RATIO")


# Target-period types the model may use, mapped to the domain's own. The
# model is asked for the KIND of period (a fiscal year, a quarter) and never
# for the relative distance (CURRENT vs NEXT fiscal year) -- that is derived
# deterministically from the issue date, and asking a reader to compute it
# was one of the ways the two got confused.
_MODEL_PERIOD_TYPES = {
    "FISCAL_YEAR": sem.PeriodFrequency.ANNUAL,
    "QUARTER": sem.PeriodFrequency.QUARTER,
    "HALF_YEAR": sem.PeriodFrequency.HALF_YEAR,
    "MULTI_YEAR": sem.PeriodFrequency.MULTI_YEAR,
    "OTHER": sem.PeriodFrequency.UNKNOWN,
}

MODEL_PERIOD_TYPE_NAMES = tuple(_MODEL_PERIOD_TYPES)


def domain_period_frequency(model_period_type: Optional[str]) -> str:
    return _MODEL_PERIOD_TYPES.get((model_period_type or "OTHER").upper(),
                                   sem.PeriodFrequency.UNKNOWN)


# ---------------------------------------------------------------------------
# Shared vocabulary: where a release says its numbers already happened
# ---------------------------------------------------------------------------
#
# Used by BOTH layers, deliberately in one place. `select_sections` stops
# absorbing a table run here, and `governing_caption` stops walking back for a
# caption here -- the reader must not be SENT reported history as if it were
# an outlook, and the boundary must not ACCEPT it as one.
#
# Two definitions of the same idea is how the denominator defect happened:
# `resolve_identity` and `denominator_is_grounded` each decided what a ratio
# metric was, disagreed, and refused correct data. One pattern, both callers.
REPORTED_RESULTS_MARKER = re.compile(
    r"(?i)\b(?:three|six|nine|twelve)\s+months\s+ended\b"
    r"|\bweighted[\s-]average\s+shares\b"
    r"|\breconciliation\s+of\b"
    r"|\b(?:condensed\s+)?consolidated\s+statements?\b")
