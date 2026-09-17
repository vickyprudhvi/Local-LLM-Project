"""The controlled trigger, and the rejoin back into the canonical pipeline.

WHEN THE FALLBACK MAY RUN

`should_attempt_fallback` is the ONLY gate. It looks at what the structured
SEC path and the deterministic `reported_actuals` table reader already
produced for the package's latest relevant period and says yes only when
neither could resolve it to `StatementCompleteness.COMPLETE`. Most runs
never reach the LLM extractor at all -- this is section 5's "controlled
fallback, not an uncontrolled replacement" as an actual gate, not a comment.

HOW ACCEPTED FACTS REJOIN THE CANONICAL PIPELINE

`run_fallback_on_document` calls the LLM extractor and validator and then
hands the accepted `ActualFinancialFact` objects to the EXISTING
`finance.reported_actuals.candidates.build_candidates` /
`company_facts_overlay`. Those two functions already implement same-document
conflict detection, currency-mixing refusal and completeness classification;
duplicating that here would be exactly the "second resolver with a different
name" the whole architecture is organised against.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from finance.documents.actuals_extractor import (
    ActualsExtractionFailure,
    ActualsTableSection,
    select_actual_sections,
)
from finance.documents.actuals_schema import ActualRejectionCode
from finance.documents.actuals_validator import ActualFactCandidateValidator
from finance.extraction.schema import ReportedActualCandidate, StatementCompleteness
from finance.reported_actuals.candidates import build_candidates, company_facts_overlay
from finance.reported_actuals.facts import ActualFinancialFact
from finance.reported_actuals.tables import parse_filing_tables, document_scale


def should_attempt_fallback(
        target_period_end: Optional[str],
        structured_completeness: Optional[str] = None,
        reported_actuals_candidates: Sequence[ReportedActualCandidate] = ()
) -> Tuple[bool, str]:
    """(should attempt, reason). Never runs for a period already resolved.

    `structured_completeness` is the XBRL/CompanyFacts path's own verdict
    for the target period (from `finance.extraction.document_resolver.
    classify_completeness`), passed in rather than recomputed so this
    decision is against the same answer production already reached.
    """
    if not target_period_end:
        return False, "no target period was identified to attempt a fallback for"
    if structured_completeness == StatementCompleteness.COMPLETE:
        return False, (
            f"the structured SEC path already resolves {target_period_end} as COMPLETE")
    for candidate in reported_actuals_candidates:
        if (candidate.period_end == target_period_end
                and candidate.statement_completeness == StatementCompleteness.COMPLETE):
            return False, (
                f"the deterministic table reader already resolves {target_period_end} "
                "as COMPLETE")
    return True, (
        f"neither the structured path nor the deterministic table reader resolves "
        f"{target_period_end} as COMPLETE; attempting the LLM actual-table fallback")


@dataclass
class FallbackOutcome:
    candidates: List[ReportedActualCandidate] = field(default_factory=list)
    facts: List[ActualFinancialFact] = field(default_factory=list)
    facts_overlay: dict = field(default_factory=dict)
    rejection_codes: Dict[str, int] = field(default_factory=dict)
    unsupported_count: int = 0
    sections_read: int = 0
    failure_code: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate_count": len(self.candidates),
            "fact_count": len(self.facts),
            "sections_read": self.sections_read,
            "rejection_codes": dict(self.rejection_codes),
            "unsupported_count": self.unsupported_count,
            "failure_code": self.failure_code,
            "notes": list(self.notes),
        }


def _period_ends_by_label(sections: Sequence[ActualsTableSection],
                          tables) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    by_index = {table.index: table for table in tables}
    for section in sections:
        table = by_index.get(section.table_index)
        if table is None:
            continue
        for column in table.period_columns:
            if column.label and column.period_end:
                mapping[column.label] = column.period_end
    return mapping


def run_fallback_on_document(document_text: str, *, extractor,
                             accession: Optional[str] = None,
                             form: Optional[str] = None,
                             filed: Optional[str] = None,
                             document_id: Optional[str] = None,
                             as_of: Optional[str] = None,
                             company_facts: Optional[dict] = None,
                             entity_id: Optional[str] = None) -> FallbackOutcome:
    """One document, read by the LLM fallback, validated, rejoined.

    `extractor` satisfies `ActualsSemanticExtractor` (an already-constructed
    `LocalModelActualsExtractor`, typically). This function never constructs
    one itself, matching `finance.extraction.runtime`'s composition -- the
    caller decides whether a model client is even available.
    """
    outcome = FallbackOutcome()
    tables = parse_filing_tables(document_text,
                                 document_scale_hint=document_scale(document_text))
    sections = select_actual_sections(tables)
    outcome.sections_read = len(sections)
    if not sections:
        outcome.notes.append("no reported-statement table was found to read")
        return outcome

    try:
        raw_candidates = extractor.extract(
            sections, issued_at=filed, document_id=document_id, form=form)
    except ActualsExtractionFailure as failure:
        outcome.failure_code = failure.code
        outcome.notes.append(f"the actuals fallback read did not complete: {failure}")
        return outcome

    grid_text = "\n\n".join(s.text for s in sections)
    period_ends = _period_ends_by_label(sections, tables)
    validator = ActualFactCandidateValidator(
        grid_text=grid_text, period_ends_by_label=period_ends, as_of=as_of,
        accession=accession, form=form, filed=filed)
    accepted, rejected = validator.validate_all(raw_candidates)
    for _candidate, code, _reason in rejected:
        outcome.rejection_codes[code] = outcome.rejection_codes.get(code, 0) + 1
    outcome.unsupported_count = sum(
        1 for _c, code, _r in rejected if code in ActualRejectionCode.UNSUPPORTED)
    outcome.facts = accepted

    if not accepted:
        outcome.notes.append("no candidate survived the acceptance boundary")
        return outcome

    candidates, build_rejections = build_candidates(
        accepted, accession=accession, form=form, filed=filed, entity_id=entity_id)
    outcome.candidates = candidates
    for code, _detail in build_rejections:
        outcome.rejection_codes[code] = outcome.rejection_codes.get(code, 0) + 1
    outcome.facts_overlay = company_facts_overlay(accepted, base=company_facts)
    return outcome
