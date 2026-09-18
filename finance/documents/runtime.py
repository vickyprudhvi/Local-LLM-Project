"""The seam where the document-pipeline mode is decided. One flag, section 22.

    v1        (default) nothing here runs. No package is resolved, no filing
              is fetched beyond what the existing pipeline already fetches,
              no model is called. Byte-for-byte today's behaviour.
    compare   the package is resolved, both new extractors run where
              registered, and everything is RECORDED in diagnostics. New
              candidates and events are withheld -- the existing answer is
              unchanged.
    v2        validated actual-fact candidates/overlay are OFFERED to the
              existing canonical pipeline (the caller still decides whether
              to use them, exactly as `finance.reported_actuals.runtime`
              does); validated events are attached as evidence. A failure
              fails closed and says so; it never falls back to `compare`'s
              behaviour silently.

WHO REGISTERS

`assistant.py`, at import, alongside the existing
`finance.extraction.runtime.register_model_client` call -- same composition
root, same reason: `finance/` must never import a model client directly.
Two separate registries (`register_actuals_model_client`,
`register_event_model_client`) because the two extractors read different
things and either may be registered without the other.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from tools import config

from finance.documents.diagnostics import DocumentPipelineDiagnostics
from finance.documents.event_schema import ResolvedFinancingEvent
from finance.documents.package import FinancialDocumentPackage, resolve_document_package
from finance.extraction.schema import ReportedActualCandidate, StatementCompleteness


class DocumentPipelineMode:
    V1 = "v1"
    COMPARE = "compare"
    V2 = "v2"
    ALL = (V1, COMPARE, V2)


class DocumentPipelineFailure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Registries -- one per extractor kind, same pattern as
# `finance.extraction.runtime` and for the same reason.
# ---------------------------------------------------------------------------

_ACTUALS_FACTORY: Optional[Callable] = None
_EVENT_FACTORY: Optional[Callable] = None
_ACTUALS_CACHE: Dict[str, object] = {}
_EVENT_CACHE: Dict[str, object] = {}


def register_actuals_model_client(ask_local_fn: Optional[Callable]) -> None:
    global _ACTUALS_FACTORY
    if ask_local_fn is None:
        _ACTUALS_FACTORY = None
        return

    def _factory():
        from finance.documents.actuals_extractor import LocalModelActualsExtractor
        return LocalModelActualsExtractor(ask_local_fn, cache=_ACTUALS_CACHE)

    _ACTUALS_FACTORY = _factory


def register_event_model_client(ask_local_fn: Optional[Callable]) -> None:
    global _EVENT_FACTORY
    if ask_local_fn is None:
        _EVENT_FACTORY = None
        return

    def _factory():
        from finance.documents.event_extractor import LocalModelEventExtractor
        return LocalModelEventExtractor(ask_local_fn, cache=_EVENT_CACHE)

    _EVENT_FACTORY = _factory


def actuals_extractor_registered() -> bool:
    return _ACTUALS_FACTORY is not None


def event_extractor_registered() -> bool:
    return _EVENT_FACTORY is not None


# ---------------------------------------------------------------------------
# Fetching -- reuses the existing SEC coordinator wrapper rather than a
# second one.
# ---------------------------------------------------------------------------

class DocumentFetcher:
    """Filing index pages and exhibits. Thin wrapper so a test can inject one."""

    def __init__(self, symbol: str, coordinator=None, cik: Optional[str] = None):
        from finance.reported_actuals.runtime import SecDocumentFetcher
        self._impl = SecDocumentFetcher(symbol, coordinator=coordinator, cik=cik)

    def filing_index(self, accession: str) -> str:
        return self._impl.filing_index(accession)

    def exhibit(self, accession: str, document: str) -> str:
        return self._impl.exhibit(accession, document)


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

@dataclass
class DocumentPipelineResult:
    package: Optional[FinancialDocumentPackage] = None
    candidates: Tuple[ReportedActualCandidate, ...] = ()
    facts_overlay: dict = field(default_factory=dict)
    events: Tuple[ResolvedFinancingEvent, ...] = ()
    diagnostics: DocumentPipelineDiagnostics = field(
        default_factory=DocumentPipelineDiagnostics)


def _document_counts(package: FinancialDocumentPackage) -> Dict[str, int]:
    return {
        "periodic_annual": len(package.periodic_annual),
        "periodic_quarterly": len(package.periodic_quarterly),
        "earnings_releases": len(package.earnings_releases),
        "financing_events": len(package.financing_events),
        "corporate_events": len(package.corporate_events),
        "unknown": len(package.unknown),
    }


def _run_actuals_fallback(package: FinancialDocumentPackage, fetcher,
                          *, symbol: str, company_facts: Optional[dict],
                          as_of: Optional[str],
                          target_period_end: Optional[str],
                          structured_completeness: Optional[str],
                          reported_actuals_candidates,
                          diagnostics: DocumentPipelineDiagnostics):
    from finance.documents.actuals_bridge import (
        run_fallback_on_document, should_attempt_fallback)
    from finance.reported_actuals.discovery import select_results_document

    should, reason = should_attempt_fallback(
        target_period_end, structured_completeness, reported_actuals_candidates)
    diagnostics.notes.append(reason)
    if not should or not package.earnings_releases:
        return (), {}
    if not actuals_extractor_registered():
        diagnostics.notes.append(
            "the actuals fallback would run but no model client is registered")
        return (), {}

    diagnostics.llm_actuals_attempted = True
    newest = max(package.earnings_releases, key=lambda d: d.filed or "")
    try:
        index_html = fetcher.filing_index(newest.accession)
        selection = select_results_document(
            index_html, accession=newest.accession, form=newest.form,
            filing_date=newest.filed, period_of_report=newest.period_of_report)
        if not selection.document_id:
            diagnostics.notes.append(
                f"{newest.accession}: no results exhibit found for the actuals fallback")
            return (), {}
        text = fetcher.exhibit(newest.accession, selection.document_id)
    except Exception as failure:                              # noqa: BLE001
        diagnostics.failure_code = "ACTUALS_FALLBACK_SOURCE_FAILED"
        diagnostics.notes.append(f"{type(failure).__name__}: {failure}")
        return (), {}

    extractor = _ACTUALS_FACTORY()
    outcome = run_fallback_on_document(
        text, extractor=extractor, accession=newest.accession, form=newest.form,
        filed=newest.filed, document_id=selection.document_id, as_of=as_of,
        company_facts=company_facts)
    diagnostics.llm_actuals_accepted = len(outcome.facts)
    diagnostics.llm_actuals_rejection_codes = dict(outcome.rejection_codes)
    diagnostics.llm_actuals_candidates_proposed = (
        len(outcome.facts) + sum(outcome.rejection_codes.values()))
    if outcome.failure_code:
        diagnostics.failure_code = outcome.failure_code
    diagnostics.notes.extend(outcome.notes)
    return tuple(outcome.candidates), outcome.facts_overlay


def _run_event_extraction(package: FinancialDocumentPackage, fetcher,
                          *, diagnostics: DocumentPipelineDiagnostics
                          ) -> Tuple[ResolvedFinancingEvent, ...]:
    from finance.documents.event_extractor import (
        EventExtractionFailure, select_event_sections)
    from finance.documents.event_validator import EventCandidateValidator

    if not package.financing_events:
        return ()
    if not event_extractor_registered():
        diagnostics.notes.append(
            "financing-event documents were found but no model client is registered")
        return ()

    diagnostics.llm_events_attempted = True
    resolved_events: List[ResolvedFinancingEvent] = []
    extractor = _EVENT_FACTORY()
    for doc in package.financing_events:
        if not doc.document_id:
            continue
        try:
            text = fetcher.exhibit(doc.accession, doc.document_id)
        except Exception as failure:                          # noqa: BLE001
            diagnostics.notes.append(
                f"{doc.accession}: the financing-event filing could not be read "
                f"({type(failure).__name__})")
            continue
        sections, failure_code = select_event_sections(text)
        if not sections:
            diagnostics.notes.append(
                f"{doc.accession}: no financing-eligible section could be shown to the "
                f"model ({failure_code})")
            continue
        try:
            raw_candidates = extractor.extract(
                sections, items=doc.items, filed=doc.filed, accession=doc.accession,
                document_id=doc.document_id)
        except EventExtractionFailure as failure:
            diagnostics.notes.append(
                f"{doc.accession}: the event read did not complete ({failure.code})")
            continue
        diagnostics.llm_events_candidates_proposed += len(raw_candidates)
        all_spans = tuple(span for section in sections for span in section.spans)
        validator = EventCandidateValidator(spans=all_spans, items=doc.items,
                                            form=doc.form)
        accepted, rejected = validator.validate_all(raw_candidates)
        resolved_events.extend(accepted)
        for _candidate, code, _reason in rejected:
            diagnostics.llm_events_rejection_codes[code] = \
                diagnostics.llm_events_rejection_codes.get(code, 0) + 1

    diagnostics.llm_events_accepted = len(resolved_events)
    diagnostics.events_funded_count = sum(1 for e in resolved_events if e.funded)
    diagnostics.events_committed_only_count = sum(
        1 for e in resolved_events if e.committed and not e.funded)
    return tuple(resolved_events)


def build_document_pipeline(
        symbol: str,
        submissions: Optional[dict],
        company_facts: Optional[dict] = None,
        *,
        mode: Optional[str] = None,
        fetcher=None,
        as_of: Optional[str] = None,
        target_period_end: Optional[str] = None,
        structured_completeness: Optional[str] = None,
        reported_actuals_candidates=(),
) -> DocumentPipelineResult:
    """The one entry point. Returns candidates/overlay/events + diagnostics.

    Under `v1` this returns immediately with an empty result and never
    resolves a package, never fetches a filing, never calls a model --
    identical in cost and effect to not having this module at all.
    """
    mode = (mode or config.finance_document_pipeline_mode()
            or DocumentPipelineMode.V1).strip().lower()
    diagnostics = DocumentPipelineDiagnostics(mode=mode)

    if mode == DocumentPipelineMode.V1:
        return DocumentPipelineResult(diagnostics=diagnostics)
    if mode not in DocumentPipelineMode.ALL:
        diagnostics.failure_code = "UNKNOWN_MODE"
        diagnostics.notes.append(
            f"'{mode}' is not a recognised document-pipeline mode; v1 was used")
        return DocumentPipelineResult(diagnostics=diagnostics)

    package = resolve_document_package(symbol, submissions, as_of=as_of)
    diagnostics.documents_selected = _document_counts(package)
    diagnostics.periods_found = sorted(
        {d.period_of_report for d in package.periodic_annual + package.periodic_quarterly
         if d.period_of_report}
        | ({package.prior_comparable_period_end}
           if package.prior_comparable_period_end else set()),
        reverse=True)
    diagnostics.structured_facts_available = (
        structured_completeness == StatementCompleteness.COMPLETE)

    fetcher = fetcher or DocumentFetcher(symbol)

    try:
        candidates, overlay = _run_actuals_fallback(
            package, fetcher, symbol=symbol, company_facts=company_facts,
            as_of=as_of, target_period_end=target_period_end,
            structured_completeness=structured_completeness,
            reported_actuals_candidates=reported_actuals_candidates,
            diagnostics=diagnostics)
    except Exception as failure:                               # noqa: BLE001
        candidates, overlay = (), {}
        diagnostics.failure_code = "ACTUALS_FALLBACK_FAILED"
        diagnostics.notes.append(f"{type(failure).__name__}: {failure}")

    try:
        events = _run_event_extraction(package, fetcher, diagnostics=diagnostics)
    except Exception as failure:                               # noqa: BLE001
        events = ()
        diagnostics.notes.append(
            f"event extraction did not complete ({type(failure).__name__}: {failure})")

    if mode == DocumentPipelineMode.COMPARE:
        # Records, never offers -- same rule every other seam in this
        # codebase follows. Events are pure evidence with no numeric effect
        # in either mode (see `finance.documents.event_schema.
        # ResolvedFinancingEvent`'s docstring), so they are still returned
        # here for the report to show; only `candidates`/`facts_overlay`
        # would change an existing numeric answer, and those are withheld.
        return DocumentPipelineResult(package=package, events=events,
                                      diagnostics=diagnostics)

    return DocumentPipelineResult(package=package, candidates=candidates,
                                  facts_overlay=overlay, events=events,
                                  diagnostics=diagnostics)
