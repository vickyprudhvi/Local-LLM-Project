"""WHICH filings are relevant to a current stock analysis. Nothing else.

This module answers exactly one question: "what are the latest relevant
financial documents for this issuer?" It never reads a document's text and
never proposes a financial value -- that is `actuals_extractor.py` and
`event_extractor.py`, downstream of the package this module builds.

A FORM CODE IS ELIGIBILITY, NOT PROOF, same as
`finance/reported_actuals/discovery.py` (imported here rather than
restated): an 8-K is a container for anything the issuer must disclose
promptly, and only metadata evidence -- item codes, forms that ARE the
statements, filing descriptions -- decides what it is a container FOR.

ONE DOCUMENT MAY BE CLASSIFIED MULTIPLE WAYS. An earnings-release 8-K
routinely carries both reported actual results AND forward guidance in one
exhibit (spec section 3); `DocumentRef.classifications` is a tuple for
exactly this reason, never a single enum value.

WHAT THIS DELIBERATELY EXCLUDES

Director changes, ownership filings (Forms 3/4/5), proxy statements, and
other routine administrative filings are classified `CORPORATE_EVENT` and are
never fed to the actual/guidance/event extractors. They may matter to a
different kind of analysis; they are not a source of financial statements or
of financing terms.
"""

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from finance import taxonomy as taxonomy_module
from finance.reported_actuals.discovery import (
    EarningsSourceCandidate,
    PERIODIC_FORMS,
    RELEASE_CARRIER_FORMS,
    find_reported_actual_filings,
)


class DocumentClass:
    """Spec section 3. A document may carry more than one of these."""

    ACTUAL_PERIODIC_ANNUAL = "ACTUAL_PERIODIC_ANNUAL"
    ACTUAL_PERIODIC_QUARTERLY = "ACTUAL_PERIODIC_QUARTERLY"
    ACTUAL_EARNINGS_RELEASE = "ACTUAL_EARNINGS_RELEASE"
    GUIDANCE_UPDATE = "GUIDANCE_UPDATE"
    FINANCING_EVENT = "FINANCING_EVENT"
    CORPORATE_EVENT = "CORPORATE_EVENT"
    UNKNOWN = "UNKNOWN"

    ALL = (ACTUAL_PERIODIC_ANNUAL, ACTUAL_PERIODIC_QUARTERLY,
           ACTUAL_EARNINGS_RELEASE, GUIDANCE_UPDATE, FINANCING_EVENT,
           CORPORATE_EVENT, UNKNOWN)


# 8-K item codes that make a filing eligible to be read for financing/capital
# terms. Eligibility only -- `event_validator.py` decides whether the cited
# text actually supports a funded amount. Kept in sync with
# `finance/structural_breaks.py::_POST_BALANCE_SHEET_ITEMS` by listing the
# same codes rather than importing a private module dict across a boundary
# that has no public one.
FINANCING_EVENT_ITEMS = ("1.01", "2.01", "2.03", "2.04", "3.02")

# Forms whose entire content is routine and administrative -- never a source
# of financial statements, guidance or financing terms (spec section 2).
_ROUTINE_CORPORATE_FORMS = frozenset({
    "3", "4", "5", "144", "DEF 14A", "DEFA14A", "SC 13D", "SC 13D/A",
    "SC 13G", "SC 13G/A", "25-NSE",
})

# How far back a "prior comparable period" may sit relative to the latest
# one, in days. Wide enough for a 52/53-week filer's drift, never wide
# enough to reach two years back.
_PRIOR_COMPARABLE_LOW_DAYS = 340
_PRIOR_COMPARABLE_HIGH_DAYS = 390


@dataclass(frozen=True)
class DocumentRef:
    """One filing (or exhibit inside one), with why it was selected."""

    accession: str
    form: str
    filed: str
    period_of_report: Optional[str] = None
    items: str = ""
    document_id: Optional[str] = None
    classifications: Tuple[str, ...] = ()
    selection_reason: str = ""

    def to_dict(self) -> dict:
        return {"accession": self.accession, "form": self.form,
                "filed": self.filed, "period_of_report": self.period_of_report,
                "items": self.items, "document_id": self.document_id,
                "classifications": list(self.classifications),
                "selection_reason": self.selection_reason}


@dataclass
class FinancialDocumentPackage:
    """Every filing this run considers relevant, classified and nothing more."""

    symbol: str
    entity_id: Optional[str] = None
    as_of: Optional[str] = None
    reporting_status: str = "domestic_registrant"
    periodic_annual: Tuple[DocumentRef, ...] = ()
    periodic_quarterly: Tuple[DocumentRef, ...] = ()
    earnings_releases: Tuple[DocumentRef, ...] = ()
    financing_events: Tuple[DocumentRef, ...] = ()
    corporate_events: Tuple[DocumentRef, ...] = ()
    unknown: Tuple[DocumentRef, ...] = ()
    prior_comparable_period_end: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    codes: List[str] = field(default_factory=list)

    @property
    def actual_documents(self) -> Tuple[DocumentRef, ...]:
        """Documents that may carry reported actual results."""
        return self.periodic_annual + self.periodic_quarterly + self.earnings_releases

    @property
    def guidance_documents(self) -> Tuple[DocumentRef, ...]:
        return tuple(d for d in self.earnings_releases
                     if DocumentClass.GUIDANCE_UPDATE in d.classifications)

    def all_documents(self) -> Tuple[DocumentRef, ...]:
        return (self.periodic_annual + self.periodic_quarterly
                + self.earnings_releases + self.financing_events
                + self.corporate_events + self.unknown)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "entity_id": self.entity_id,
            "as_of": self.as_of, "reporting_status": self.reporting_status,
            "periodic_annual": [d.to_dict() for d in self.periodic_annual],
            "periodic_quarterly": [d.to_dict() for d in self.periodic_quarterly],
            "earnings_releases": [d.to_dict() for d in self.earnings_releases],
            "financing_events": [d.to_dict() for d in self.financing_events],
            "corporate_events": [d.to_dict() for d in self.corporate_events],
            "unknown": [d.to_dict() for d in self.unknown],
            "prior_comparable_period_end": self.prior_comparable_period_end,
            "notes": list(self.notes), "codes": list(self.codes),
        }


def _filing_rows(submissions: dict) -> List[dict]:
    """The same row shape `reported_actuals.discovery` builds internally.

    Reimplemented at this width (rather than importing the leading-underscore
    helper) because it is genuinely tiny and this module additionally needs
    `report_date` and raw `items` for forms `find_reported_actual_filings`
    does not itself examine (periodic and financing filings).
    """
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


def _detect_reporting_status(rows: Sequence[dict]) -> str:
    """Domestic vs foreign private issuer, from the forms actually filed.

    A 20-F/40-F filer never also files a 10-K; a domestic registrant never
    files a 20-F/40-F. Evidence-driven, matching spec section 12's rule for
    business-model classification: never a ticker or a name.
    """
    forms = {(r.get("form") or "").upper() for r in rows}
    if forms & {"20-F", "20-F/A", "40-F", "40-F/A"}:
        return "foreign_private_issuer"
    return "domestic_registrant"


def _top_by_recency(rows: Sequence[dict], forms: Sequence[str], limit: int
                    ) -> List[dict]:
    wanted = {f.upper() for f in forms}
    matched = [r for r in rows if (r.get("form") or "").upper() in wanted]
    matched.sort(key=lambda r: r.get("filed") or "", reverse=True)
    return matched[:limit]


def _classify_financing_row(row: dict) -> bool:
    form = (row.get("form") or "").upper()
    if form != "8-K":
        return False
    items = row.get("items") or ""
    return any(code in items for code in FINANCING_EVENT_ITEMS)


def _prior_comparable(rows: Sequence[dict], latest_period_end: Optional[str]
                      ) -> Optional[str]:
    """The periodic filing's report date landing roughly a year earlier."""
    if not latest_period_end:
        return None
    try:
        latest = datetime.date.fromisoformat(latest_period_end)
    except ValueError:
        return None
    best = None
    for row in rows:
        end = row.get("report_date")
        if not end or (row.get("form") or "").upper() not in PERIODIC_FORMS:
            continue
        try:
            candidate_date = datetime.date.fromisoformat(end)
        except ValueError:
            continue
        gap = (latest - candidate_date).days
        if _PRIOR_COMPARABLE_LOW_DAYS <= gap <= _PRIOR_COMPARABLE_HIGH_DAYS:
            if best is None or candidate_date > datetime.date.fromisoformat(best):
                best = end
    return best


def resolve_document_package(symbol: str, submissions: Optional[dict], *,
                             entity_id: Optional[str] = None,
                             reporting_status: Optional[str] = None,
                             as_of: Optional[str] = None,
                             max_periodic: int = 2,
                             max_earnings_releases: Optional[int] = None,
                             max_financing_events: int = 8
                             ) -> FinancialDocumentPackage:
    """The latest RELEVANT documents for one issuer, classified.

    `max_earnings_releases` defaults to
    `tools.config.finance_reported_actuals_max_filings()`, so a package and
    the existing reported-actuals discovery agree on how many releases are
    worth reading without either importing the other's default.
    """
    from tools import config as config_module

    package = FinancialDocumentPackage(symbol=symbol, entity_id=entity_id,
                                       as_of=as_of)
    if not submissions:
        package.codes.append("NO_SUBMISSIONS")
        package.notes.append(
            "no SEC filing index was available; no document could be selected")
        return package

    rows = _filing_rows(submissions)
    package.reporting_status = reporting_status or _detect_reporting_status(rows)

    annual_forms = [f for f in PERIODIC_FORMS if f in taxonomy_module.ANNUAL_FORMS]
    quarterly_forms = [f for f in PERIODIC_FORMS if f not in annual_forms]

    package.periodic_annual = tuple(
        DocumentRef(accession=r["accession"], form=r["form"], filed=r["filed"],
                    period_of_report=r.get("report_date") or None,
                    classifications=(DocumentClass.ACTUAL_PERIODIC_ANNUAL,),
                    selection_reason=f"{r['form']} is the annual periodic filing.")
        for r in _top_by_recency(rows, annual_forms, max_periodic))
    package.periodic_quarterly = tuple(
        DocumentRef(accession=r["accession"], form=r["form"], filed=r["filed"],
                    period_of_report=r.get("report_date") or None,
                    classifications=(DocumentClass.ACTUAL_PERIODIC_QUARTERLY,),
                    selection_reason=f"{r['form']} is the quarterly periodic filing.")
        for r in _top_by_recency(rows, quarterly_forms, max_periodic))

    limit = (max_earnings_releases if max_earnings_releases is not None
             else config_module.finance_reported_actuals_max_filings())
    accepted, refused = find_reported_actual_filings(submissions, limit=limit)
    package.earnings_releases = tuple(
        DocumentRef(accession=c.accession, form=c.form, filed=c.filed,
                    period_of_report=c.period_of_report, items=c.items,
                    # An earnings release commonly carries both reported
                    # actuals and forward guidance in one exhibit (section 3);
                    # both classifications are attached and the downstream
                    # extractors read the same document for different things.
                    classifications=(DocumentClass.ACTUAL_EARNINGS_RELEASE,
                                     DocumentClass.GUIDANCE_UPDATE),
                    selection_reason=c.reason)
        for c in accepted)
    package.codes.extend(f"EARNINGS_RELEASE_REFUSED:{c.accession}"
                         for c in refused if c.accession)

    financing_rows = [r for r in rows if _classify_financing_row(r)]
    financing_rows.sort(key=lambda r: r.get("filed") or "", reverse=True)
    package.financing_events = tuple(
        DocumentRef(accession=r["accession"], form=r["form"], filed=r["filed"],
                    period_of_report=r.get("report_date") or None,
                    items=r.get("items") or "", document_id=r.get("document") or None,
                    classifications=(DocumentClass.FINANCING_EVENT,),
                    selection_reason=(
                        f"8-K item(s) {r.get('items')} are eligible financing/"
                        "capital-structure evidence."))
        for r in financing_rows[:max_financing_events])

    classified_accessions = {d.accession for d in package.all_documents()}
    corporate, unknown = [], []
    for row in rows:
        if row.get("accession") in classified_accessions:
            continue
        form = (row.get("form") or "").upper()
        if form in _ROUTINE_CORPORATE_FORMS or form not in (
                set(RELEASE_CARRIER_FORMS) | set(PERIODIC_FORMS) | {"8-K"}):
            corporate.append(row)
        else:
            unknown.append(row)
    package.corporate_events = tuple(
        DocumentRef(accession=r["accession"], form=r["form"], filed=r["filed"],
                    classifications=(DocumentClass.CORPORATE_EVENT,),
                    selection_reason=f"{r['form']} is routine/administrative.")
        for r in corporate)
    package.unknown = tuple(
        DocumentRef(accession=r["accession"], form=r["form"], filed=r["filed"],
                    classifications=(DocumentClass.UNKNOWN,),
                    selection_reason="no eligibility rule matched this filing.")
        for r in unknown)

    latest_period = None
    if package.periodic_annual:
        latest_period = package.periodic_annual[0].period_of_report
    if package.periodic_quarterly and (
            not latest_period
            or (package.periodic_quarterly[0].period_of_report or "") > latest_period):
        latest_period = package.periodic_quarterly[0].period_of_report
    package.prior_comparable_period_end = _prior_comparable(rows, latest_period)

    return package


class FinancialDocumentResolver:
    """Thin named entry point matching the spec's vocabulary.

    `resolve_document_package` is the function every caller in this codebase
    actually imports; this class exists only so "FinancialDocumentResolver"
    is a real, discoverable name for anyone reading the architecture from the
    spec inward.
    """

    @staticmethod
    def resolve(symbol: str, submissions: Optional[dict], **kwargs
               ) -> FinancialDocumentPackage:
        return resolve_document_package(symbol, submissions, **kwargs)
