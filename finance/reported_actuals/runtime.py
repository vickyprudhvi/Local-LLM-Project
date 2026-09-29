"""The seam where reported-actual SOURCES are chosen. One decision, one place.

WHAT THIS RETURNS, AND TO WHOM

Candidates. Nothing else. They are handed to
`finance/actualization_runtime.py::resolve_actual_state`, which hands them to
the existing Actualization V2 resolver, which decides which period is current.
This module has no opinion on that and must never acquire one.

    v1        nothing runs, nothing is fetched, no candidate is produced.
    compare   discovery and extraction run and are RECORDED; the candidates
              are reported in the observation and are NOT offered for use.
    v2        the candidates are offered to the resolver, and a failure is a
              failure rather than a silent fall back to what CompanyFacts
              could see.

`compare` withholding its own candidates is the point rather than a
limitation, for the same reason the extraction and actualization seams do it:
a compare mode that quietly changed which period was selected would be
v2-by-default wearing a diagnostic's name.

WHY BOTH MODES MUST BE ON BEFORE ANYTHING CHANGES

`FINANCE_ACTUALIZATION_MODE` is `v1` in production, and under `v1` the
Actualization runtime returns before it ever looks at a candidate. So a run
with `FINANCE_REPORTED_ACTUALS_MODE=v2` and the actualization default still
behaves exactly as it does today. Two switches, and the second one is the one
that changes an answer.

RETRIEVAL

Through the SAME `MarketDataRequestCoordinator` every other SEC dataset uses,
so caching, pacing, the quota ledger, the user-agent requirement and error
normalization are inherited rather than reimplemented. The fetcher is
injectable purely so a test does not need a network; production passes
nothing and gets the shared SEC coordinator.

No socket is opened anywhere in `finance/actualization.py`,
`finance/ttm.py`, `finance/freshness.py`, the DCF modules or the research
pipeline. This module is the boundary, and it holds.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from finance.extraction.schema import ReportedActualCandidate
from finance.reported_actuals.candidates import (
    ReportedActualsExtraction,
    company_facts_overlay,
    extract_reported_actuals,
)
from finance.reported_actuals.discovery import (
    DocumentSelection,
    EarningsSourceCandidate,
    SourceDiscoveryCode,
    find_reported_actual_filings,
    select_results_document,
)


class ReportedActualsMode:
    V1 = "v1"
    COMPARE = "compare"
    V2 = "v2"
    ALL = (V1, COMPARE, V2)


class ReportedActualsFailure(Exception):
    """v2 could not read a source it had decided to read. Never a fallback."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class SourceFailureCode:
    INDEX_UNAVAILABLE = "FILING_INDEX_UNAVAILABLE"
    DOCUMENT_UNAVAILABLE = "FILING_DOCUMENT_UNAVAILABLE"
    NO_RESULTS_DOCUMENT = "NO_RESULTS_DOCUMENT"
    EXTRACTION_EMPTY = "NO_ACTUAL_FACTS_EXTRACTED"
    UNKNOWN_MODE = "UNKNOWN_MODE"
    NO_SUBMISSIONS = "NO_SUBMISSIONS_INDEX"


# ---------------------------------------------------------------------------
# What a run reports about itself (section 29)
# ---------------------------------------------------------------------------

@dataclass
class ReportedActualsObservation:
    """Structured diagnostics only. No filing text, no document contents."""

    mode: str = ReportedActualsMode.V1
    candidates_offered: bool = False
    filings_inspected: int = 0
    releases_detected: int = 0
    filings_refused: int = 0
    documents_selected: List[dict] = field(default_factory=list)
    tables_read: int = 0
    tables_usable: int = 0
    facts_accepted: int = 0
    candidate_count: int = 0
    candidate_periods: List[str] = field(default_factory=list)
    candidate_completeness: Dict[str, str] = field(default_factory=dict)
    values_populated: Dict[str, int] = field(default_factory=dict)
    rejection_codes: Dict[str, int] = field(default_factory=dict)
    latest_release_period: Optional[str] = None
    latest_companyfacts_period: Optional[str] = None
    advances_period: bool = False
    same_period_overlap: List[str] = field(default_factory=list)
    failure_code: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = {
            "mode": self.mode,
            "candidates_offered": self.candidates_offered,
            "filings_inspected": self.filings_inspected,
            "releases_detected": self.releases_detected,
            "filings_refused": self.filings_refused,
            "documents_selected": list(self.documents_selected),
            "tables_read": self.tables_read,
            "tables_usable": self.tables_usable,
            "facts_accepted": self.facts_accepted,
            "candidate_count": self.candidate_count,
            "candidate_periods": list(self.candidate_periods),
            "candidate_completeness": dict(self.candidate_completeness),
            "values_populated": dict(self.values_populated),
            "rejection_codes": dict(self.rejection_codes),
            "latest_release_period": self.latest_release_period,
            "latest_companyfacts_period": self.latest_companyfacts_period,
            "advances_period": self.advances_period,
            "same_period_overlap": list(self.same_period_overlap),
        }
        if self.failure_code:
            payload["failure_code"] = self.failure_code
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class SecDocumentFetcher:
    """Filing index pages and exhibits, through the shared SEC coordinator."""

    def __init__(self, symbol: str, coordinator=None, cik: Optional[str] = None):
        self.symbol = symbol
        self._coordinator = coordinator
        self._cik = cik

    @property
    def coordinator(self):
        if self._coordinator is None:
            from tools import finance_tools
            self._coordinator = finance_tools.get_sec_coordinator()
        return self._coordinator

    @property
    def cik(self) -> str:
        if self._cik is None:
            from finance.sec_provider import resolve_cik
            self._cik, _name = resolve_cik(self.coordinator, self.symbol)
        return self._cik

    def _document(self, accession: str, document: str) -> str:
        outcome = self.coordinator.fetch(
            "filing_document", self.symbol,
            {"cik": self.cik, "accession": accession, "document": document})
        return (outcome.payload or {}).get("document_text") or ""

    def filing_index(self, accession: str) -> str:
        return self._document(accession, f"{accession}-index.html")

    def exhibit(self, accession: str, document: str) -> str:
        return self._document(accession, document)


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------

@dataclass
class ReportedActualsResult:
    """Candidates, and the record of how they were produced."""

    candidates: Tuple[ReportedActualCandidate, ...] = ()
    extractions: Tuple[ReportedActualsExtraction, ...] = ()
    # The accepted facts in the SEC company-facts shape, so the EXISTING
    # twelve-month reconstruction can follow the period the resolver selects.
    # Empty unless candidates were actually offered.
    facts_overlay: dict = field(default_factory=dict)
    observation: ReportedActualsObservation = field(
        default_factory=ReportedActualsObservation)


def discover_release_candidates(
        symbol: str,
        submissions: Optional[dict],
        *,
        mode: Optional[str] = None,
        fetcher: Optional[SecDocumentFetcher] = None,
        max_filings: Optional[int] = None,
        companyfacts_latest_period: Optional[str] = None,
        company_facts: Optional[dict] = None,
        as_of: Optional[str] = None,
) -> ReportedActualsResult:
    """Reported-actual candidates from filed earnings-release exhibits.

    `companyfacts_latest_period` is what the existing discovery path can
    already see. It is passed in rather than recomputed so the observation
    compares against the answer production actually has, and so this module
    never needs to open the CompanyFacts payload.
    """
    from tools import config

    mode = (mode or config.finance_reported_actuals_mode()
            or ReportedActualsMode.V1).strip().lower()
    observation = ReportedActualsObservation(
        mode=mode, latest_companyfacts_period=companyfacts_latest_period)

    if mode == ReportedActualsMode.V1:
        return ReportedActualsResult(observation=observation)
    if mode not in ReportedActualsMode.ALL:
        observation.failure_code = SourceFailureCode.UNKNOWN_MODE
        observation.notes.append(
            f"'{mode}' is not a recognised reported-actuals mode; v1 was used")
        return ReportedActualsResult(observation=observation)

    if not submissions:
        observation.failure_code = SourceFailureCode.NO_SUBMISSIONS
        observation.notes.append(
            "no SEC filing index was available, so no earnings release could be located")
        if mode == ReportedActualsMode.V2:
            raise ReportedActualsFailure(
                SourceFailureCode.NO_SUBMISSIONS,
                "the SEC filing index is required to locate an earnings release")
        return ReportedActualsResult(observation=observation)

    limit = max_filings if max_filings is not None \
        else config.finance_reported_actuals_max_filings()
    accepted, refused = find_reported_actual_filings(submissions, limit=limit)
    observation.filings_inspected = len(accepted) + len(refused)
    observation.releases_detected = len(accepted)
    observation.filings_refused = len(refused)

    fetcher = fetcher or SecDocumentFetcher(symbol)
    extractions: List[ReportedActualsExtraction] = []
    candidates: List[ReportedActualCandidate] = []

    for filing in accepted:
        selection, extraction = _read_one(filing, fetcher, observation, mode)
        if selection is not None:
            observation.documents_selected.append(selection.to_dict())
        if extraction is None:
            continue
        extractions.append(extraction)
        observation.tables_read += len(extraction.tables)
        observation.tables_usable += sum(1 for t in extraction.tables if t.get("usable"))
        observation.facts_accepted += len(extraction.canonical_facts)
        for code, count in extraction.rejection_codes().items():
            observation.rejection_codes[code] = \
                observation.rejection_codes.get(code, 0) + count
        candidates.extend(extraction.candidates)

    usable = [c for c in candidates
              if not as_of or (c.period_end or "") <= as_of]
    observation.candidate_count = len(usable)
    observation.candidate_periods = sorted(
        {c.period_end for c in usable if c.period_end}, reverse=True)
    observation.candidate_completeness = {
        c.period_end or "?": c.statement_completeness for c in usable}
    observation.values_populated = {
        c.period_end or "?": len(c.values or {}) for c in usable}
    observation.latest_release_period = (observation.candidate_periods[0]
                                         if observation.candidate_periods else None)
    observation.advances_period = bool(
        observation.latest_release_period and companyfacts_latest_period
        and observation.latest_release_period > companyfacts_latest_period)
    if companyfacts_latest_period:
        observation.same_period_overlap = [
            period for period in observation.candidate_periods
            if period == companyfacts_latest_period]

    if mode == ReportedActualsMode.COMPARE:
        # Records, never resolves. The existing candidate set stands, and no
        # overlay is produced either -- an overlay moves a twelve-month
        # window, which is a change to an answer.
        observation.candidates_offered = False
        return ReportedActualsResult(candidates=(), extractions=tuple(extractions),
                                     observation=observation)

    if not usable:
        observation.failure_code = SourceFailureCode.EXTRACTION_EMPTY
        observation.notes.append(
            "no reported-actual candidate could be built from any located "
            "earnings release")
    observation.candidates_offered = bool(usable)
    accepted_periods = {c.period_end for c in usable}
    overlay = company_facts_overlay(
        [fact for extraction in extractions for fact in extraction.facts
         if fact.period_end in accepted_periods],
        base=company_facts) if usable else {}
    return ReportedActualsResult(candidates=tuple(usable),
                                 extractions=tuple(extractions),
                                 facts_overlay=overlay,
                                 observation=observation)


def _read_one(filing: EarningsSourceCandidate, fetcher: SecDocumentFetcher,
              observation: ReportedActualsObservation, mode: str
              ) -> Tuple[Optional[DocumentSelection], Optional[ReportedActualsExtraction]]:
    """One filing: index -> document -> facts. Every failure is structured."""
    try:
        index_html = fetcher.filing_index(filing.accession)
    except Exception as failure:                              # noqa: BLE001
        observation.notes.append(
            f"{filing.accession}: the filing index could not be read "
            f"({type(failure).__name__})")
        observation.rejection_codes[SourceFailureCode.INDEX_UNAVAILABLE] = \
            observation.rejection_codes.get(SourceFailureCode.INDEX_UNAVAILABLE, 0) + 1
        return None, None

    selection = select_results_document(
        index_html, accession=filing.accession, form=filing.form,
        filing_date=filing.filed, period_of_report=filing.period_of_report)
    if not selection.document_id:
        observation.rejection_codes[SourceFailureCode.NO_RESULTS_DOCUMENT] = \
            observation.rejection_codes.get(SourceFailureCode.NO_RESULTS_DOCUMENT, 0) + 1
        return selection, None

    try:
        exhibit = fetcher.exhibit(filing.accession, selection.document_id)
    except Exception as failure:                              # noqa: BLE001
        observation.notes.append(
            f"{filing.accession}/{selection.document_id}: the exhibit could not "
            f"be read ({type(failure).__name__})")
        observation.rejection_codes[SourceFailureCode.DOCUMENT_UNAVAILABLE] = \
            observation.rejection_codes.get(SourceFailureCode.DOCUMENT_UNAVAILABLE, 0) + 1
        return selection, None

    extraction = extract_reported_actuals(
        exhibit, accession=filing.accession, form=filing.form,
        filed=filing.filed, document=selection.document_id)
    return selection, extraction
