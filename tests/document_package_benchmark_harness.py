"""Scorer for the Golden Document Benchmark (`tests/fixtures/document_package_benchmark.py`).

Two things are measured separately, matching spec section 26's own list:

    document-selection accuracy   `resolve_document_package` alone
    event acceptance behaviour    a STUBBED "model" proposing exactly the
                                   case's `ExpectedEvent` fields, run through
                                   the REAL `EventCandidateValidator`

The stub is deliberate and mirrors `tests/test_finance_extraction_benchmark.py`'s
own oracle-candidate approach: it measures the deterministic ACCEPTANCE
BOUNDARY, not a live model's reading ability, which needs a model call and is
a separate, live benchmark (see `scripts/run_live_extraction_benchmark.py`
for the guidance layer's equivalent; this phase does not add a live-model
counterpart for the document pipeline).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from finance.documents.actuals_bridge import _period_ends_by_label
from finance.documents.actuals_extractor import select_actual_sections
from finance.documents.actuals_schema import ActualFactCandidate
from finance.documents.actuals_validator import ActualFactCandidateValidator
from finance.documents.event_schema import EventCandidate
from finance.documents.event_validator import EventCandidateValidator
from finance.documents.spans import build_source_spans
from finance.documents.package import resolve_document_package
from finance.reported_actuals.tables import document_scale, parse_filing_tables
from tests.fixtures.document_package_benchmark import DocumentPackageCase


_BUCKET_ATTR = {
    "periodic_annual": "periodic_annual",
    "periodic_quarterly": "periodic_quarterly",
    "earnings_releases": "earnings_releases",
    "financing_events": "financing_events",
    "corporate_events": "corporate_events",
    "unknown": "unknown",
}


@dataclass
class CaseScore:
    case_id: str
    classification_correct: int = 0
    classification_total: int = 0
    classification_errors: List[str] = field(default_factory=list)
    reporting_status_correct: bool = True
    event_correct: int = 0
    event_total: int = 0
    event_errors: List[str] = field(default_factory=list)
    actual_correct: int = 0
    actual_total: int = 0
    actual_errors: List[str] = field(default_factory=list)


@dataclass
class BenchmarkScore:
    case_scores: List[CaseScore] = field(default_factory=list)

    @property
    def classification_accuracy(self) -> float:
        total = sum(c.classification_total for c in self.case_scores)
        correct = sum(c.classification_correct for c in self.case_scores)
        return (correct / total) if total else 1.0

    @property
    def event_acceptance_accuracy(self) -> float:
        total = sum(c.event_total for c in self.case_scores)
        correct = sum(c.event_correct for c in self.case_scores)
        return (correct / total) if total else 1.0

    @property
    def actual_acceptance_accuracy(self) -> float:
        total = sum(c.actual_total for c in self.case_scores)
        correct = sum(c.actual_correct for c in self.case_scores)
        return (correct / total) if total else 1.0

    @property
    def all_classification_errors(self) -> List[str]:
        return [e for c in self.case_scores for e in c.classification_errors]

    @property
    def all_event_errors(self) -> List[str]:
        return [e for c in self.case_scores for e in c.event_errors]

    @property
    def all_actual_errors(self) -> List[str]:
        return [e for c in self.case_scores for e in c.actual_errors]

    @property
    def reporting_status_errors(self) -> List[str]:
        return [c.case_id for c in self.case_scores if not c.reporting_status_correct]


def _bucket_of(package, accession: str) -> str:
    for bucket in _BUCKET_ATTR:
        if accession in {d.accession for d in getattr(package, bucket)}:
            return bucket
    return "NOT_SELECTED"


def score_case(case: DocumentPackageCase) -> CaseScore:
    score = CaseScore(case_id=case.case_id)
    package = resolve_document_package(case.case_id, case.submissions)

    score.reporting_status_correct = (
        package.reporting_status == case.expected_reporting_status)

    for expectation in case.expected:
        score.classification_total += 1
        actual_bucket = _bucket_of(package, expectation.accession)
        if actual_bucket == expectation.expected_bucket:
            score.classification_correct += 1
        else:
            score.classification_errors.append(
                f"{case.case_id}/{expectation.accession}: expected "
                f"{expectation.expected_bucket}, got {actual_bucket}")

    for expected_event in case.expected_events:
        score.event_total += 1
        document_text = case.event_documents.get(expected_event.accession, "")
        event_spans = build_source_spans(document_text)
        written_amount, written_unit = _as_written(expected_event.amount)
        candidate = EventCandidate(
            event_type=expected_event.event_type,
            amount=written_amount, unit=written_unit,
            # Phase H.22: the stub's currency must match the fixture's OWN
            # stated currency (a EUR-denominated case must not be validated
            # against a hardcoded USD claim); `expected_currency` defaults
            # to None on every pre-H.22 fixture, so "USD" remains the
            # fallback for all of them.
            currency=((expected_event.expected_currency or "USD")
                     if expected_event.amount is not None else None),
            funded=bool(expected_event.funded), committed=bool(expected_event.committed),
            # The stub cites EVERY span of the document -- it stands in for
            # a model that correctly identified every piece of supporting
            # text, so the validator's grounding checks see the same full
            # evidence a `source_evidence=document_text` stub gave them
            # before spans existed.
            evidence_span_ids=tuple(span.span_id for span in event_spans),
            confidence=0.9, accession=expected_event.accession, form="8-K")
        validator = EventCandidateValidator(spans=event_spans, form="8-K",
                                            items=_items_for(case, expected_event.accession))
        resolved, code, reason = validator.validate(candidate)
        accepted = resolved is not None
        if accepted == expected_event.should_be_accepted:
            score.event_correct += 1
            if accepted:
                # Accepted AND correct is not enough on its own for the two
                # hard-safety cases -- the funded flag itself must survive
                # unchanged, or a validator that "corrected" a wrong claim
                # into a right one would look identical to one that never
                # made the mistake.
                if resolved.funded != bool(expected_event.funded):
                    score.event_correct -= 1
                    score.event_errors.append(
                        f"{case.case_id}/{expected_event.accession}: accepted with "
                        f"funded={resolved.funded}, expected {expected_event.funded}")
        else:
            score.event_errors.append(
                f"{case.case_id}/{expected_event.accession}: expected accepted="
                f"{expected_event.should_be_accepted}, got {accepted} ({code}: {reason})")

    if case.actual_document_text and case.expected_actual_facts:
        tables = parse_filing_tables(
            case.actual_document_text,
            document_scale_hint=document_scale(case.actual_document_text))
        sections = select_actual_sections(tables)
        grid_text = "\n\n".join(s.text for s in sections)
        period_ends = _period_ends_by_label(sections, tables)
        by_end = {end: label for label, end in period_ends.items()}
        section_kind_by_period = {}
        for section in sections:
            for label, end in period_ends.items():
                if label in section.text:
                    section_kind_by_period[end] = section.statement_kind

        for expected_fact in case.expected_actual_facts:
            score.actual_total += 1
            period_label = by_end.get(expected_fact.period_end)
            written_amount = expected_fact.value / 1_000_000.0
            # A row line naming this exact metric/value, drawn from the
            # actual rendered grid where possible so grounding succeeds; the
            # outlook-contamination negative item names its OWN period
            # (never in `period_ends`, since that table was excluded before
            # rendering), which is exactly what should make it un-groundable.
            source_line = next(
                (line for line in grid_text.splitlines()
                 if line.startswith("Row:") and period_label and
                 f"{written_amount:.1f}" in line), "")
            candidate = ActualFactCandidate(
                metric_id=expected_fact.metric_id, value=written_amount,
                unit="USD_MILLION", currency="USD", period_label=period_label,
                period_type="QUARTER", period_end=None, basis="GAAP",
                scope="CONSOLIDATED", prospective=False,
                source_evidence=source_line, confidence=0.9,
                statement_kind=section_kind_by_period.get(expected_fact.period_end,
                                                          "INCOME_STATEMENT"))
            validator = ActualFactCandidateValidator(
                grid_text=grid_text, period_ends_by_label=period_ends,
                as_of=expected_fact.period_end, form="8-K")
            fact, code, reason = validator.validate(candidate)
            accepted = fact is not None
            if accepted == expected_fact.should_be_accepted:
                score.actual_correct += 1
            else:
                score.actual_errors.append(
                    f"{case.case_id}/{expected_fact.metric_id}@{expected_fact.period_end}: "
                    f"expected accepted={expected_fact.should_be_accepted}, got {accepted} "
                    f"({code}: {reason})")

    return score


def _as_written(amount):
    """Full magnitude back to the (point, unit) shape a reader would state --
    the same convention `finance.extraction.schema.domain_unit` uses for
    guidance, applied here so the stub candidate looks like a real one
    instead of a number no filing would ever literally contain."""
    if amount is None:
        return None, "UNKNOWN"
    if abs(amount) >= 1_000_000_000:
        return amount / 1_000_000_000.0, "USD_BILLION"
    if abs(amount) >= 1_000_000:
        return amount / 1_000_000.0, "USD_MILLION"
    return amount, "USD"


def _items_for(case: DocumentPackageCase, accession: str) -> str:
    recent = case.submissions["filings"]["recent"]
    for index, acc in enumerate(recent["accessionNumber"]):
        if acc == accession:
            return recent["items"][index]
    return ""


def run_benchmark(cases) -> BenchmarkScore:
    return BenchmarkScore(case_scores=[score_case(case) for case in cases])
