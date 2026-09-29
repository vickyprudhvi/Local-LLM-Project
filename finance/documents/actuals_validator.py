"""The acceptance boundary for LLM-read actual statement lines.

Produces `finance.reported_actuals.facts.ActualFinancialFact` -- the EXACT
type the deterministic table reader already produces -- so an accepted
candidate rejoins the one canonical pipeline through
`finance.reported_actuals.candidates.build_candidates` /
`company_facts_overlay` rather than through a second, parallel one. Nothing
downstream of `ActualFinancialFact` ever learns whether a fact came from a
regex row-match or from a model read.

Checks, in the order a reader would want them explained:

    1  evidence grounding      the cited row line is one this run showed
    2  value grounding         the number is literally in that row line
    3  metric identity         one of the reviewed field names
    4  scope                   CONSOLIDATED only -- a segment/component row
                                is not a company-level fact
    5  basis                   GAAP/IFRS only -- an adjusted figure is real
                                but is not the canonical line by that name
    6  currency                a plausible ISO code
    7  not prospective         a hard fail-safe even though `select_
                                actual_sections` never sends a guidance table
    8  period resolution       the column header maps to a real period end
    9  period plausibility     the period must already have ENDED
   10  confidence               below the floor is refused
"""

import re
from typing import Dict, Optional, Sequence, Tuple

import tools.config as config
from finance import semantics as sem
from finance.documents.actuals_schema import (
    ACTUAL_METRIC_NAMES,
    ActualFactCandidate,
    ActualRejectionCode,
    domain_basis,
    domain_frequency,
    instant_metric_names,
)
from finance.extraction.validator import _numbers_in, evidence_is_grounded
from finance.reported_actuals.facts import ActualFinancialFact, FIELD_IDENTITY
from finance.reported_actuals.facts import _COMPONENT_ROW as COMPONENT_LANGUAGE
from finance.reported_actuals.tables import StatementKind

# Reused rather than restated -- one definition of "this text is forward-
# looking" for the whole codebase (finance/extraction/schema.py docstring).
from finance.extraction.validator import _PROSPECTIVE as PROSPECTIVE_LANGUAGE

_ISO_CURRENCY = re.compile(r"^[A-Z]{3}$")


def _matches_any(value: float, present) -> bool:
    forms = {value, value * 100.0, value / 100.0}
    return any(any(abs(f - p) <= max(abs(p) * 1e-6, 1e-9) for p in present)
               for f in forms)


class ActualFactCandidateValidator:
    """One centralized acceptance boundary for LLM-read actual facts."""

    def __init__(self, grid_text: str = "",
                 period_ends_by_label: Optional[Dict[str, str]] = None,
                 as_of: Optional[str] = None,
                 accession: Optional[str] = None, form: Optional[str] = None,
                 filed: Optional[str] = None,
                 min_confidence: Optional[float] = None):
        self.grid_text = grid_text or ""
        self.period_ends_by_label = dict(period_ends_by_label or {})
        self.as_of = as_of
        self.accession = accession
        self.form = form
        self.filed = filed
        self.min_confidence = (min_confidence if min_confidence is not None
                               else config.finance_actuals_extraction_min_confidence())

    def validate(self, candidate: ActualFactCandidate
                ) -> Tuple[Optional[ActualFinancialFact], Optional[str], str]:
        if not candidate.source_evidence.strip():
            return None, ActualRejectionCode.NO_EVIDENCE, (
                "the candidate cites no row, so nothing about it can be checked")
        if self.grid_text and not evidence_is_grounded(
                candidate.source_evidence, self.grid_text):
            return None, ActualRejectionCode.EVIDENCE_NOT_IN_SOURCE, (
                "the cited row does not appear in the table this run showed the model")

        if candidate.value is None:
            return None, ActualRejectionCode.VALUE_NOT_IN_EVIDENCE, (
                "the candidate reports no value")
        present = _numbers_in(candidate.source_evidence)
        if not present or not _matches_any(candidate.value, present):
            return None, ActualRejectionCode.VALUE_NOT_IN_EVIDENCE, (
                f"{candidate.value} does not appear in the cited row")

        if not candidate.metric_id or candidate.metric_id not in ACTUAL_METRIC_NAMES:
            return None, ActualRejectionCode.UNKNOWN_METRIC, (
                f"{candidate.metric_id!r} is not a reviewed actual-statement metric")

        # Defensive: `select_actual_sections` already excludes every table
        # that is not a reported statement, so this should never fire in
        # practice. It stays because the candidate schema is decoupled from
        # the extractor that produced it, and a future caller building
        # candidates a different way must not be able to skip this boundary.
        if candidate.statement_kind not in StatementKind.REPORTED_STATEMENTS:
            return None, ActualRejectionCode.ADJUSTED_NOT_CANONICAL, (
                f"the source table is classified {candidate.statement_kind!r}, not a "
                "reported income statement, balance sheet or cash flow statement")

        if candidate.scope and candidate.scope.upper() not in (
                sem.ConsolidationScope.CONSOLIDATED, "CONSOLIDATED", ""):
            return None, ActualRejectionCode.SCOPE_NOT_CONSOLIDATED, (
                f"the candidate's own scope is {candidate.scope!r}, not consolidated")
        # Defense in depth: the deterministic table reader refuses a row
        # whose own label names a segment/region/discontinued-operations
        # breakout regardless of what a candidate CLAIMS its scope is
        # (`finance.reported_actuals.facts._COMPONENT_ROW`). The same text
        # is checked here so a model cannot mark a component row
        # "CONSOLIDATED" and have that self-declaration believed.
        if COMPONENT_LANGUAGE.search(candidate.source_evidence or ""):
            return None, ActualRejectionCode.SCOPE_NOT_CONSOLIDATED, (
                "the cited row's own label names a segment, region or "
                "discontinued-operations breakout")

        basis = domain_basis(candidate.basis)
        if basis == sem.AccountingBasis.ADJUSTED:
            return None, ActualRejectionCode.ADJUSTED_NOT_CANONICAL, (
                "the candidate is an adjusted/non-GAAP figure, which cannot fill a "
                "canonical GAAP/IFRS field")

        if candidate.prospective or PROSPECTIVE_LANGUAGE.search(candidate.source_evidence):
            return None, ActualRejectionCode.PROSPECTIVE_NOT_ACTUAL, (
                "the cited row reads as forward-looking, not a reported result")

        currency = (candidate.currency or "").upper()
        if not _ISO_CURRENCY.match(currency):
            return None, ActualRejectionCode.UNKNOWN_CURRENCY, (
                f"{candidate.currency!r} is not a recognisable ISO currency code")

        period_end = self.period_ends_by_label.get(candidate.period_label or "")
        if not period_end:
            return None, ActualRejectionCode.PERIOD_UNRESOLVED, (
                f"column {candidate.period_label!r} does not resolve to a known period end")
        if self.as_of and period_end > self.as_of:
            return None, ActualRejectionCode.PERIOD_NOT_YET_ENDED, (
                f"{period_end} is after {self.as_of}; a period that has not ended has no "
                "reported actual")

        if candidate.confidence < self.min_confidence:
            return None, ActualRejectionCode.LOW_CONFIDENCE, (
                f"confidence {candidate.confidence:.2f} is below the floor "
                f"{self.min_confidence:.2f}")

        frequency = domain_frequency(candidate.period_type)
        is_instant = candidate.metric_id in instant_metric_names()
        resolved_value = candidate.resolved_value()
        fact = ActualFinancialFact(
            field=candidate.metric_id, value=resolved_value,
            raw_value=candidate.value, currency=currency, scale=None,
            period_end=period_end, period_start=None,
            frequency=(sem.PeriodFrequency.INSTANT if is_instant else frequency),
            flow_or_instant=(sem.FlowOrInstant.INSTANT if is_instant
                             else sem.FlowOrInstant.FLOW),
            accounting_basis=basis, statement_kind=candidate.statement_kind,
            row_label=candidate.source_evidence[:200],
            column_label=candidate.period_label or "",
            band_label=candidate.section_label or "",
            metric_identity=FIELD_IDENTITY.get(candidate.metric_id, sem.MetricIdentity.UNKNOWN),
            accession=self.accession, form=self.form, filed=self.filed,
            document=candidate.source_document_id, finality="UNAUDITED_PRELIMINARY")
        return fact, None, ""

    def validate_all(self, candidates: Sequence[ActualFactCandidate]
                     ):
        accepted, rejected = [], []
        for candidate in candidates:
            fact, code, reason = self.validate(candidate)
            if fact is None:
                rejected.append((candidate, code, reason))
            else:
                accepted.append(fact)
        return accepted, rejected
