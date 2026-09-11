"""Which filings plausibly report actual results, and which document inside them.

A FORM CODE IS ELIGIBILITY, NOT PROOF

Section 3 of the brief is explicit, and it is right: adding "8-K" and "6-K" to
a form allowlist would let an acquisition announcement, a debt issuance, a
director resignation and an investor deck all present themselves as reported
financial statements. An 8-K is a container for anything the issuer must
disclose promptly; roughly one in six of them is an earnings release.

So eligibility is decided from METADATA EVIDENCE, each piece named and
recorded:

    item 2.02          "Results of Operations and Financial Condition" -- the
                       SEC's own classification, and the strongest single
                       signal there is
    period of report   the filing says which period it is about
    exhibit type       EX-99.x carries the release; the 8-K body carries the
                       cover page
    document description / filename
                       "Press Release", "Q3 2026 Earnings Release",
                       "exhibit991.htm"

A filing with no earnings evidence at all is not a reported-actual source, and
that is a decision recorded with its reason rather than a silent skip.

6-K HAS NO ITEM NUMBERS

A foreign private issuer's 6-K carries no item classification, so item 2.02
cannot be the test there. The evidence available is the exhibit description
and the document itself, which is weaker -- and it is treated as weaker: a 6-K
qualifies only on a description or filename that names results, never on the
form alone.

RETRIEVAL LIVES BEHIND THE EXISTING PROVIDER

Nothing here opens a socket. `find_reported_actual_filings` reads a
submissions payload that has already been fetched, and `select_results_document`
reads a filing index page that has already been fetched. The caller
(`runtime.py`) owns the coordinator, so caching, rate limits, timeouts,
user-agent and error normalization stay exactly where they already are.
"""

import html as _html
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from finance import taxonomy as taxonomy_module


class SourceDiscoveryCode:
    """Why a filing was accepted or refused as a reported-actual source."""

    EARNINGS_ITEM = "EARNINGS_ITEM_2_02"
    DESCRIPTION_NAMES_RESULTS = "DESCRIPTION_NAMES_RESULTS"
    FILENAME_NAMES_RESULTS = "FILENAME_NAMES_RESULTS"
    PERIODIC_FILING = "PERIODIC_FILING"
    NO_EARNINGS_EVIDENCE = "NO_EARNINGS_EVIDENCE"
    FORM_NOT_ELIGIBLE = "FORM_NOT_ELIGIBLE"
    NO_PERIOD_OF_REPORT = "NO_PERIOD_OF_REPORT"
    NO_RESULTS_DOCUMENT = "NO_RESULTS_DOCUMENT"
    EXHIBIT_SELECTED = "EXHIBIT_SELECTED"


# The SEC's own classification of an earnings release. The one piece of
# evidence that is a statement by the filer rather than an inference.
EARNINGS_RELEASE_ITEM = "2.02"

# Forms that MAY carry an earnings release as an exhibit. Membership here
# makes a filing eligible to be examined; it never makes it a source.
RELEASE_CARRIER_FORMS = frozenset({"8-K", "8-K/A", "6-K", "6-K/A"})

# Forms that ARE the reported statements: the document itself is the financial
# statements, so no further evidence is needed.
#
# 6-K IS NOT ONE OF THEM, and used to be, because `taxonomy.ALL_REPORT_FORMS`
# groups it with the interim filings. A 6-K is a CONTAINER exactly like an
# 8-K -- it carries interim results, a notice of an annual general meeting, a
# change of auditor -- and treating the form as proof made every filing a
# foreign private issuer had ever made a reported-actual source, with its
# evidence never read.
PERIODIC_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A",
                            "20-F", "20-F/A", "40-F", "40-F/A"})

# Vocabulary that names REPORTED RESULTS.
#
# "PRESS RELEASE" IS NOT IN IT, and used to be. It names the VEHICLE, not the
# subject: an appointment, an acquisition and a quarter's results are all
# announced by press release, and counting the phrase as evidence admitted
# every one of them. That is a form allowlist one level down.
_RESULTS_SUBJECT = re.compile(
    r"(?i)\b(?:earnings"
    r"|(?:financial|quarterly|annual|interim|half[\s-]year|full[\s-]year"
    r"|fourth[\s-]quarter|first[\s-]quarter|second[\s-]quarter"
    r"|third[\s-]quarter|operating)\s+results"
    r"|results\s+(?:of\s+operations|for\s+the)"
    r"|reports?\s+(?:first|second|third|fourth)[\s-]quarter"
    r")\b")

# Documents that sit BESIDE an earnings release, or announce something else
# entirely. An investor deck restates the same figures with different
# roundings and no statement captions; a transcript is speech; an appointment
# is not a period. None is a filed financial statement.
_NOT_A_RESULTS_DOCUMENT = re.compile(
    r"(?i)\b(?:presentation|slides?|deck|webcast|transcript|conference\s+call"
    r"|supplemental\s+(?:slides|deck)|infographic|prospectus|indenture"
    r"|underwriting|credit\s+agreement|by[\s-]?laws?|charter"
    r"|certificat\w+|opinion|consent|subsidiaries|xbrl|cover\s+page"
    r"|appointment|resignation|retirement|leadership|succession"
    r"|acquisition|merger|divestiture|disposal|offering|notes?\s+due"
    r"|dividend|buyback|general\s+meeting|notice\s+of|amendment"
    r"|restructuring\s+plan|auditor|litigation|settlement)\b")

# A filename that names its SUBJECT. The EX-99 slot is deliberately absent:
# "exhibit991.htm" says where a document sits in a filing, not what it is
# about, and every 8-K carrying any press release has one.
_FILENAME_RESULTS = re.compile(r"(?i)(?:earnings|results|q[1-4]\d{2}|er\d{2,})")

# At the DOCUMENT level the EX-99 slot IS the right prior: by then the FILING
# has been established as a statement of results and the only question left is
# which of its exhibits carries the statements.
_FILENAME_EXHIBIT_SLOT = re.compile(r"(?i)(?:ex(?:hibit)?[\s_-]*99|release|press)")

_EXHIBIT_TYPE = re.compile(r"(?i)^ex-?99(?:\.\d+)?$")


@dataclass(frozen=True)
class EarningsSourceCandidate:
    """One filing that may report actual results, with the evidence for it."""

    accession: str
    form: str
    filed: str
    period_of_report: Optional[str]
    items: str = ""
    evidence: Tuple[str, ...] = ()
    reason: str = ""

    @property
    def is_release_carrier(self) -> bool:
        return (self.form or "").upper() in RELEASE_CARRIER_FORMS

    def to_dict(self) -> dict:
        return {"accession": self.accession, "form": self.form,
                "filed": self.filed, "period_of_report": self.period_of_report,
                "items": self.items, "evidence": list(self.evidence),
                "reason": self.reason}


@dataclass(frozen=True)
class DocumentSelection:
    """The document inside a filing that carries the reported results."""

    accession: str
    form: str
    filing_date: str
    period_of_report: Optional[str]
    document_id: Optional[str]
    document_type: Optional[str]
    document_description: Optional[str]
    selection_reason: str
    code: str

    def to_dict(self) -> dict:
        return {"filing_id": self.accession, "form": self.form,
                "filing_date": self.filing_date,
                "period_of_report": self.period_of_report,
                "document_id": self.document_id,
                "document_type": self.document_type,
                "document_description": self.document_description,
                "selection_reason": self.selection_reason, "code": self.code}


def _rows(submissions: dict) -> List[dict]:
    recent = ((submissions or {}).get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    out = []
    for index, form in enumerate(forms):
        def at(key, default=""):
            values = recent.get(key) or []
            return values[index] if index < len(values) else default
        out.append({
            "form": form, "accession": at("accessionNumber"),
            "filed": at("filingDate"), "report_date": at("reportDate"),
            "items": at("items"), "description": at("primaryDocDescription"),
            "document": at("primaryDocument"),
        })
    return out


def classify_filing(row: dict) -> Tuple[bool, Tuple[str, ...], str]:
    """(is a reported-actual source, evidence codes, reason).

    Evidence is CUMULATIVE and named. A filing qualifies on item 2.02 alone,
    because that is the SEC's classification of the filing rather than a guess
    about it; otherwise it needs its description or its primary document to
    name reported results, and must not look like a presentation or a
    transcript.
    """
    form = (row.get("form") or "").upper()
    # Carrier first. A 6-K is both an interim form and a container, and asking
    # "is it periodic?" before "is it a container?" answered the wrong
    # question for every foreign private issuer.
    if form not in RELEASE_CARRIER_FORMS:
        if form in PERIODIC_FORMS:
            return True, (SourceDiscoveryCode.PERIODIC_FILING,), \
                f"{form} is a periodic filing and reports the period directly."
        return False, (), f"{form or 'this form'} does not carry earnings exhibits."

    evidence: List[str] = []
    items = row.get("items") or ""
    description = row.get("description") or ""
    document = row.get("document") or ""
    excluded = bool(_NOT_A_RESULTS_DOCUMENT.search(description)
                    or _NOT_A_RESULTS_DOCUMENT.search(document))

    # Item 2.02 is the SEC's own classification of the filing and is evidence
    # on its own -- but not when the filing itself says it is about something
    # else. A description naming an appointment is the more specific claim.
    if EARNINGS_RELEASE_ITEM in items and not excluded:
        evidence.append(SourceDiscoveryCode.EARNINGS_ITEM)
    if _RESULTS_SUBJECT.search(description) and not excluded:
        evidence.append(SourceDiscoveryCode.DESCRIPTION_NAMES_RESULTS)
    if _FILENAME_RESULTS.search(document) and not excluded:
        evidence.append(SourceDiscoveryCode.FILENAME_NAMES_RESULTS)

    if not evidence:
        return False, (), (
            f"{form} filed {row.get('filed')} carries no earnings evidence: "
            f"items {items or 'none'}, description "
            f"{description or 'none'}.")
    return True, tuple(evidence), (
        f"{form} filed {row.get('filed')} qualifies on "
        f"{', '.join(evidence)}.")


def find_reported_actual_filings(submissions: dict, *, limit: int = 8,
                                 carriers_only: bool = True
                                 ) -> Tuple[List[EarningsSourceCandidate],
                                            List[EarningsSourceCandidate]]:
    """(accepted, refused), newest first.

    `carriers_only` keeps the result to the filings the CompanyFacts path
    cannot already see. A periodic filing is reported as accepted evidence
    only when the caller asks for the full view.
    """
    accepted: List[EarningsSourceCandidate] = []
    refused: List[EarningsSourceCandidate] = []
    for row in _rows(submissions):
        form = (row.get("form") or "").upper()
        if carriers_only and form not in RELEASE_CARRIER_FORMS:
            continue
        qualifies, evidence, reason = classify_filing(row)
        candidate = EarningsSourceCandidate(
            accession=row.get("accession") or "", form=row.get("form") or "",
            filed=row.get("filed") or "",
            period_of_report=row.get("report_date") or None,
            items=row.get("items") or "", evidence=evidence, reason=reason)
        if qualifies:
            accepted.append(candidate)
        else:
            refused.append(candidate)
        if len(accepted) >= limit:
            break
    return accepted, refused


# ---------------------------------------------------------------------------
# Which document inside the filing
# ---------------------------------------------------------------------------

_INDEX_ROW = re.compile(r"(?is)<tr[^>]*>(.*?)</tr>")
_INDEX_CELL = re.compile(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>")


def _index_rows(filing_index_html: str) -> List[List[str]]:
    rows = []
    for row_html in _INDEX_ROW.findall(filing_index_html or ""):
        cells = [re.sub(r"(?s)<[^>]+>", " ", cell)
                 for cell in _INDEX_CELL.findall(row_html)]
        cells = [re.sub(r"\s+", " ", _html.unescape(c)).strip() for c in cells]
        if len(cells) >= 4:
            rows.append(cells)
    return rows


def select_results_document(filing_index_html: str, *, accession: str = "",
                            form: str = "", filing_date: str = "",
                            period_of_report: Optional[str] = None
                            ) -> DocumentSelection:
    """The exhibit carrying the reported results, with why it was chosen.

    EX-99.1 IS NOT ASSUMED. It is the conventional slot and it is frequently
    something else -- a supplemental deck, a second press release about an
    acquisition announced the same morning, a segment-realignment schedule.
    So every EX-99.x row is SCORED on its description and its filename, and
    the best-scoring one wins; the numbering only breaks a tie.

    Columns on an EDGAR index page are Seq | Description | Document | Type.
    """
    rows = _index_rows(filing_index_html)
    best: Optional[Tuple[Tuple[int, int, int], List[str]]] = None
    for cells in rows:
        description, document, exhibit_type = cells[1], cells[2], cells[3]
        name = document.split()[0] if document else ""
        if not name.lower().endswith((".htm", ".html", ".txt")):
            continue
        if not _EXHIBIT_TYPE.match(exhibit_type.strip()):
            continue
        if _NOT_A_RESULTS_DOCUMENT.search(description) \
                or _NOT_A_RESULTS_DOCUMENT.search(name):
            continue
        score = (
            2 if _RESULTS_SUBJECT.search(description) else 0,
            1 if (_FILENAME_RESULTS.search(name)
                  or _FILENAME_EXHIBIT_SLOT.search(name)
                  or _FILENAME_EXHIBIT_SLOT.search(description)) else 0,
            # EX-99.1 before EX-99.2 before a bare EX-99, as a tie-break only.
            -_exhibit_ordinal(exhibit_type),
        )
        if best is None or score > best[0]:
            best = (score, cells)

    if best is None:
        return DocumentSelection(
            accession=accession, form=form, filing_date=filing_date,
            period_of_report=period_of_report, document_id=None,
            document_type=None, document_description=None,
            code=SourceDiscoveryCode.NO_RESULTS_DOCUMENT,
            selection_reason="No EX-99 exhibit on the filing index reads as a "
                             "statement of reported results.")

    _score, cells = best
    return DocumentSelection(
        accession=accession, form=form, filing_date=filing_date,
        period_of_report=period_of_report,
        document_id=cells[2].split()[0], document_type=cells[3].strip(),
        document_description=cells[1],
        code=SourceDiscoveryCode.EXHIBIT_SELECTED,
        selection_reason=f"{cells[3].strip()} '{cells[1]}' names reported "
                         f"results and is the best-scoring exhibit on the index.")


def _exhibit_ordinal(exhibit_type: str) -> int:
    match = re.search(r"(\d+)$", (exhibit_type or "").strip())
    return int(match.group(1)) if match and "." in exhibit_type else 99
