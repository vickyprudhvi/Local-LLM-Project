"""The candidate type for an LLM-read ACTUAL statement line.

Proposes exactly one KIND of thing: a reported (not guided, not derived)
figure for a metric the taxonomy already knows, with the evidence it was
read from. It reuses `finance.extraction.schema`'s unit/period vocabulary
directly (spec section 9: "use existing types/enums whenever possible") and
`finance.semantics.ConsolidationScope` for scope, so a unit or a scope
question never has two answers depending which extractor produced it.

WHAT THIS SCHEMA CANNOT EXPRESS, ON PURPOSE

There is no `derived` shape and no arithmetic operator. Growth, margins and
per-share conversions are computed by the EXISTING deterministic modules
(`finance/growth.py`, `finance/normalization.py`) once a candidate is
accepted and merged in -- exactly as they already do for XBRL and for
`reported_actuals` table facts. Section 11 of the spec ("Allowed
derivations") is enforced by never giving the model a field to populate one
in the first place.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from finance import semantics as sem
from finance.extraction.schema import MODEL_UNIT_NAMES, domain_unit

# Metric names this layer may propose. Deliberately the SAME field vocabulary
# `finance/reported_actuals/facts.py::ROW_ALIASES` resolves prose row labels
# into, so an LLM-read fact and a deterministically-read fact land under one
# name and can be merged, deduplicated and compared identically downstream.
ACTUAL_METRIC_NAMES = (
    "revenue", "gross_profit", "operating_income", "income_before_tax",
    "income_tax_expense", "net_income", "diluted_eps",
    "cash_and_cash_equivalents", "short_term_investments", "current_assets",
    "current_liabilities", "assets", "liabilities", "stockholders_equity",
    "current_portion_of_long_term_debt", "short_term_debt", "long_term_debt",
    "operating_cash_flow", "capital_expenditure",
    "depreciation_and_amortization", "diluted_shares",
)

# Which of `ACTUAL_METRIC_NAMES` are point-in-time balances. Mirrors
# `finance/reported_actuals/facts.py::INSTANT_FIELDS`, derived from the same
# reviewed concept map, so the two never disagree about what a "balance" is.
def instant_metric_names() -> frozenset:
    from finance.reported_actuals.facts import INSTANT_FIELDS
    return frozenset(name for name in ACTUAL_METRIC_NAMES if name in INSTANT_FIELDS)


class ActualPeriodType:
    QUARTER = "QUARTER"
    ANNUAL = "ANNUAL"
    HALF_YEAR = "HALF_YEAR"
    YTD_6M = "YTD_6M"
    INSTANT = "INSTANT"
    OTHER = "OTHER"
    ALL = (QUARTER, ANNUAL, HALF_YEAR, YTD_6M, INSTANT, OTHER)


_PERIOD_TYPE_TO_FREQUENCY = {
    ActualPeriodType.QUARTER: sem.PeriodFrequency.QUARTER,
    ActualPeriodType.ANNUAL: sem.PeriodFrequency.ANNUAL,
    ActualPeriodType.HALF_YEAR: sem.PeriodFrequency.HALF_YEAR,
    ActualPeriodType.YTD_6M: sem.PeriodFrequency.YTD_6M,
    ActualPeriodType.INSTANT: sem.PeriodFrequency.INSTANT,
    ActualPeriodType.OTHER: sem.PeriodFrequency.UNKNOWN,
}


def domain_frequency(period_type: Optional[str]) -> str:
    return _PERIOD_TYPE_TO_FREQUENCY.get(
        (period_type or ActualPeriodType.OTHER).upper(), sem.PeriodFrequency.UNKNOWN)


class ActualBasis:
    GAAP = "GAAP"
    IFRS = "IFRS"
    ADJUSTED = "ADJUSTED"
    UNKNOWN = "UNKNOWN"
    ALL = (GAAP, IFRS, ADJUSTED, UNKNOWN)


_BASIS_TO_DOMAIN = {
    ActualBasis.GAAP: sem.AccountingBasis.GAAP,
    ActualBasis.IFRS: sem.AccountingBasis.IFRS,
    ActualBasis.ADJUSTED: sem.AccountingBasis.ADJUSTED,
    ActualBasis.UNKNOWN: sem.AccountingBasis.UNKNOWN,
}


def domain_basis(basis: Optional[str]) -> str:
    return _BASIS_TO_DOMAIN.get((basis or ActualBasis.UNKNOWN).upper(),
                                sem.AccountingBasis.UNKNOWN)


# A scale WORD (as `finance.extraction.schema.domain_unit` returns it) to the
# multiplier that turns a written figure into a full magnitude. Mirrors
# `finance/reported_actuals/tables.py::SCALE_FACTORS`, keyed the way this
# layer's unit vocabulary spells it (singular, from an LLM-reported unit
# rather than a parsed "(in millions)" caption).
_SCALE_MULTIPLIER = {None: 1.0, "million": 1_000_000.0, "billion": 1_000_000_000.0}


def scale_multiplier(unit: Optional[str]) -> float:
    _domain, scale = domain_unit(unit)
    return _SCALE_MULTIPLIER.get(scale, 1.0)


class ActualRejectionCode:
    """Why an actual-fact candidate was refused."""

    NO_EVIDENCE = "NO_EVIDENCE"
    EVIDENCE_NOT_IN_SOURCE = "EVIDENCE_NOT_IN_SOURCE"
    VALUE_NOT_IN_EVIDENCE = "VALUE_NOT_IN_EVIDENCE"
    UNKNOWN_METRIC = "UNKNOWN_METRIC"
    UNIT_METRIC_MISMATCH = "UNIT_METRIC_MISMATCH"
    UNKNOWN_CURRENCY = "UNKNOWN_CURRENCY"
    PROSPECTIVE_NOT_ACTUAL = "PROSPECTIVE_NOT_ACTUAL"
    SCOPE_NOT_CONSOLIDATED = "SCOPE_NOT_CONSOLIDATED"
    ADJUSTED_NOT_CANONICAL = "ADJUSTED_NOT_CANONICAL"
    PERIOD_UNRESOLVED = "PERIOD_UNRESOLVED"
    PERIOD_NOT_YET_ENDED = "PERIOD_NOT_YET_ENDED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MALFORMED = "MALFORMED"

    ALL = (NO_EVIDENCE, EVIDENCE_NOT_IN_SOURCE, VALUE_NOT_IN_EVIDENCE,
           UNKNOWN_METRIC, UNIT_METRIC_MISMATCH, UNKNOWN_CURRENCY,
           PROSPECTIVE_NOT_ACTUAL, SCOPE_NOT_CONSOLIDATED,
           ADJUSTED_NOT_CANONICAL, PERIOD_UNRESOLVED, PERIOD_NOT_YET_ENDED,
           LOW_CONFIDENCE, MALFORMED)

    # A benchmark's CRITICAL column (spec section 27): the source did not
    # support what was accepted, as opposed to a candidate that was merely
    # incomplete or low-confidence.
    UNSUPPORTED = (EVIDENCE_NOT_IN_SOURCE, VALUE_NOT_IN_EVIDENCE,
                   PROSPECTIVE_NOT_ACTUAL, SCOPE_NOT_CONSOLIDATED,
                   ADJUSTED_NOT_CANONICAL)


@dataclass(frozen=True)
class ActualFactCandidate:
    """One proposed reported figure, with the row/sentence it came from.

    `period_end` is the ISO date the candidate is FOR, resolved
    deterministically (never by the model) from `period_label` and the
    document's own filing/period context by the extractor, exactly as
    `finance.extraction.semantic_extractor` fills `document_id`/`issued_at`
    on a `GuidanceCandidate` rather than trusting the model for them.
    """

    metric_id: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    currency: Optional[str] = None
    period_label: Optional[str] = None
    period_type: str = ActualPeriodType.OTHER
    period_end: Optional[str] = None
    basis: str = ActualBasis.UNKNOWN
    scope: str = sem.ConsolidationScope.UNKNOWN
    prospective: bool = False
    source_evidence: str = ""
    section_label: Optional[str] = None
    confidence: float = 0.0
    # Filled by the extractor, never by the model. `statement_kind` is the
    # deterministic table classifier's own verdict
    # (`finance.reported_actuals.tables.StatementKind`) for the table this
    # candidate was read from -- the validator refuses anything not in
    # `StatementKind.REPORTED_STATEMENTS`, and the model never sees or sets it.
    statement_kind: Optional[str] = None
    source_document_id: Optional[str] = None
    form: Optional[str] = None
    filed: Optional[str] = None

    def resolved_value(self) -> Optional[float]:
        """The value at full magnitude, scale applied."""
        if self.value is None:
            return None
        return self.value * scale_multiplier(self.unit)

    def to_dict(self) -> dict:
        return {
            "metric_id": self.metric_id, "value": self.value, "unit": self.unit,
            "currency": self.currency, "period_label": self.period_label,
            "period_type": self.period_type, "period_end": self.period_end,
            "basis": self.basis, "scope": self.scope,
            "prospective": self.prospective,
            "source_evidence": self.source_evidence,
            "section_label": self.section_label, "confidence": self.confidence,
            "source_document_id": self.source_document_id, "form": self.form,
            "filed": self.filed,
        }


__all__ = [
    "ACTUAL_METRIC_NAMES", "ActualBasis", "ActualFactCandidate",
    "ActualPeriodType", "ActualRejectionCode", "domain_basis",
    "domain_frequency", "instant_metric_names", "scale_multiplier",
    "MODEL_UNIT_NAMES",
]
