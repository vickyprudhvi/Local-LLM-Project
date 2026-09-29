"""Phase H.29 spike -- a thin EdgarTools retrieval adapter.

WHAT THIS MODULE IS

A source/retrieval boundary, nothing more. It answers "what filings exist
and what do they literally say" the same way `finance.sec_provider.
SecEdgarClient` + `finance.reported_actuals.discovery` + `finance.documents.
text_normalization` already do -- company/CIK lookup, filing discovery,
accession/form/date/period metadata, primary-document and section/item text,
and exhibit listing. It never proposes a financial FACT: no canonical value,
no currency, no scale, no event. Every one of those still comes from
`finance.documents.event_extractor` / `actuals_extractor`, validated by
`event_validator.py` / `actuals_validator.py`, completely unmodified by this
module (see spec section 3: EdgarTools must not bypass any of that).

WHY A SEPARATE MODULE, NOT A NEW CLASS INSIDE `sec_provider.py`

`finance.sec_provider.SecEdgarClient` is a reviewed, production, always-on
component (`finance.provider.AlphaVantageClient`'s sibling under the shared
`MarketDataRequestCoordinator`). This adapter is spike-only, gated behind
`tools.config.finance_sec_provider() != "current"` (default "current" -- see
that function's own docstring), and depends on the THIRD-PARTY `edgar`
package, which the rest of `finance/` must never import. Keeping it in its
own file makes that import boundary a file boundary: nothing outside this
module ever writes `import edgar`.

LAZY IMPORT, ALWAYS

`import edgar` happens inside functions, never at module load time. Importing
`finance.documents.edgartools_adapter` itself must never fail, and never
install a third-party side effect, when `edgartools` is not installed at all
-- which is the case for this project's own main venv by default; this
module exists for the spike's comparison script and for an explicit
`FINANCE_SEC_PROVIDER=edgartools`/`compare` opt-in, never for the default
`current` path production always takes.

WHAT THIS MODULE DELIBERATELY DOES NOT DO (spec section 4)

`Filing.obj().to_context()` and any other LLM-oriented formatted summary
string are NEVER returned by this module as authoritative text. Only three
kinds of output leave here: (1) plain filing/company METADATA (accession,
form, dates, item codes -- structurally identical in shape to what
`finance.reported_actuals.discovery`'s row dicts already carry), (2) VERBATIM
section/document text (`Section.text()` / `Filing.text()`), fed to OUR OWN
`finance.documents.spans.build_source_spans` exactly like our own normalized
text already is, and (3) raw structured XBRL facts, compared only, never
substituted for `finance.sec_provider`'s companyfacts path in this phase.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import tools.config as config


class EdgarToolsAdapterError(Exception):
    """A controlled failure from the adapter -- never a raw third-party
    traceback escaping into finance/ code. `code` is one of the values in
    `EdgarToolsErrorCode`."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class EdgarToolsErrorCode:
    NOT_INSTALLED = "EDGARTOOLS_NOT_INSTALLED"
    NO_IDENTITY = "EDGARTOOLS_NO_IDENTITY"
    COMPANY_NOT_FOUND = "EDGARTOOLS_COMPANY_NOT_FOUND"
    FILING_NOT_FOUND = "EDGARTOOLS_FILING_NOT_FOUND"
    FETCH_FAILED = "EDGARTOOLS_FETCH_FAILED"


class TextRepresentation:
    """What KIND of text a caller received -- spec section 7's explicit ask,
    so downstream code never guesses whether normalization already ran.

    RAW_HTML         the filed document's own HTML/inline-XBRL, unprocessed.
    NORMALIZED_TEXT  our OWN `finance.documents.text_normalization` output
                     (never produced by this module -- listed for callers
                     that compare the two).
    SECTION_TEXT     EdgarTools' own structurally-segmented section/item
                     text (`Section.text()` / `Filing.text()`) -- already
                     visible-text, already had markup stripped by EdgarTools
                     itself. This is what `filing_text`/`filing_sections`
                     below return.
    """

    RAW_HTML = "RAW_HTML"
    NORMALIZED_TEXT = "NORMALIZED_TEXT"
    SECTION_TEXT = "SECTION_TEXT"


@dataclass(frozen=True)
class EdgarToolsCompanyRef:
    cik: str
    name: str
    ticker: Optional[str] = None

    def to_dict(self) -> dict:
        return {"cik": self.cik, "name": self.name, "ticker": self.ticker}


@dataclass(frozen=True)
class EdgarToolsFilingRef:
    """Mirrors `finance.documents.package.DocumentRef`'s field NAMES where
    the concept is the same, deliberately -- a caller comparing the two
    structurally is comparing like-for-like keys, not translating a schema."""

    cik: str
    company_name: str
    accession: str
    form: str
    filed: str
    period_of_report: Optional[str] = None
    items: str = ""
    primary_document: Optional[str] = None

    def to_dict(self) -> dict:
        return {"cik": self.cik, "company_name": self.company_name,
                "accession": self.accession, "form": self.form,
                "filed": self.filed, "period_of_report": self.period_of_report,
                "items": self.items, "primary_document": self.primary_document}


@dataclass(frozen=True)
class EdgarToolsSectionRef:
    name: str
    title: Optional[str]
    text: str
    text_representation: str = TextRepresentation.SECTION_TEXT

    def to_dict(self) -> dict:
        return {"name": self.name, "title": self.title,
                "text_representation": self.text_representation,
                "char_count": len(self.text)}


@dataclass(frozen=True)
class EdgarToolsExhibitRef:
    document: str
    document_type: Optional[str]
    description: Optional[str]

    def to_dict(self) -> dict:
        return {"document": self.document, "document_type": self.document_type,
                "description": self.description}


def _edgar_module():
    """Lazy, controlled import. Raises EdgarToolsAdapterError, never a bare
    ImportError, so a caller not opted into FINANCE_SEC_PROVIDER=edgartools/
    compare never has to handle a third-party exception type."""
    try:
        import edgar  # noqa: F401 -- imported for its side effect below
    except ImportError as e:
        raise EdgarToolsAdapterError(
            EdgarToolsErrorCode.NOT_INSTALLED,
            "the 'edgartools' package is not installed in this environment") from e

    identity = config.edgar_identity() if hasattr(config, "edgar_identity") else None
    if not identity:
        import os
        identity = os.environ.get("EDGAR_IDENTITY", "").strip()
    if not identity:
        raise EdgarToolsAdapterError(
            EdgarToolsErrorCode.NO_IDENTITY,
            "EDGAR_IDENTITY is not configured -- same fair-access identity "
            "requirement as finance.sec_provider's SEC_USER_AGENT")
    edgar.set_identity(identity)
    return edgar


def lookup_company(ticker: str) -> EdgarToolsCompanyRef:
    edgar = _edgar_module()
    try:
        company = edgar.Company(ticker)
    except Exception as e:  # noqa: BLE001 -- any third-party failure becomes one controlled code
        raise EdgarToolsAdapterError(
            EdgarToolsErrorCode.COMPANY_NOT_FOUND,
            f"no EdgarTools company match for {ticker!r}: {e}") from e
    if company is None or getattr(company, "cik", None) is None:
        raise EdgarToolsAdapterError(
            EdgarToolsErrorCode.COMPANY_NOT_FOUND,
            f"no EdgarTools company match for {ticker!r}")
    return EdgarToolsCompanyRef(cik=str(company.cik), name=company.name, ticker=ticker.upper())


def _filing_ref_from(filing, cik: str, company_name: str) -> EdgarToolsFilingRef:
    items = ""
    try:
        obj = filing.obj()
        raw_items = getattr(obj, "items", None)
        if raw_items:
            items = ",".join(str(i).replace("Item", "").strip() for i in raw_items)
    except Exception:  # noqa: BLE001 -- item extraction is best-effort metadata
        items = ""
    return EdgarToolsFilingRef(
        cik=cik, company_name=company_name,
        accession=filing.accession_no, form=filing.form,
        filed=str(filing.filing_date),
        period_of_report=str(getattr(filing, "period_of_report", "") or "") or None,
        items=items,
        primary_document=getattr(filing, "primary_document", None))


def discover_filings(ticker: str, form: Optional[str] = None, limit: int = 10
                     ) -> List[EdgarToolsFilingRef]:
    """Filing discovery -- `finance.documents.package.resolve_document_package`'s
    counterpart. `form` may be a single EDGAR form string ("10-K", "8-K") or
    None for the issuer's most recent filings of any form."""
    edgar = _edgar_module()
    company_ref = lookup_company(ticker)
    try:
        company = edgar.Company(ticker)
        filings = company.get_filings(form=form) if form else company.get_filings()
    except Exception as e:  # noqa: BLE001
        raise EdgarToolsAdapterError(
            EdgarToolsErrorCode.FETCH_FAILED,
            f"EdgarTools filing discovery failed for {ticker!r}: {e}") from e
    out = []
    for filing in filings:
        out.append(_filing_ref_from(filing, company_ref.cik, company_ref.name))
        if len(out) >= limit:
            break
    return out


def get_filing(ticker: str, accession: str) -> EdgarToolsFilingRef:
    """The SAME single filing our own pipeline would address by accession --
    the basis for every same-filing A/B comparison in this phase."""
    edgar = _edgar_module()
    company_ref = lookup_company(ticker)
    try:
        company = edgar.Company(ticker)
        matches = company.get_filings(accession_number=accession)
    except Exception as e:  # noqa: BLE001
        raise EdgarToolsAdapterError(
            EdgarToolsErrorCode.FETCH_FAILED,
            f"EdgarTools filing fetch failed for {ticker!r}/{accession!r}: {e}") from e
    if not matches or len(matches) == 0:
        raise EdgarToolsAdapterError(
            EdgarToolsErrorCode.FILING_NOT_FOUND,
            f"EdgarTools found no filing {accession!r} for {ticker!r}")
    return _filing_ref_from(matches[0], company_ref.cik, company_ref.name)


def _resolve_underlying_filing(ticker: str, accession: str):
    edgar = _edgar_module()
    company = edgar.Company(ticker)
    matches = company.get_filings(accession_number=accession)
    if not matches or len(matches) == 0:
        raise EdgarToolsAdapterError(
            EdgarToolsErrorCode.FILING_NOT_FOUND,
            f"EdgarTools found no filing {accession!r} for {ticker!r}")
    return matches[0]


def filing_text(ticker: str, accession: str) -> Tuple[str, str]:
    """(text, text_representation). Verbatim visible text EdgarTools itself
    extracted from the filed document -- never a formatted/LLM-oriented
    summary (spec section 4). This is what would feed OUR OWN
    `finance.documents.spans.build_source_spans` if this provider were ever
    selected downstream of discovery."""
    filing = _resolve_underlying_filing(ticker, accession)
    try:
        text = filing.text() or ""
    except Exception as e:  # noqa: BLE001
        raise EdgarToolsAdapterError(
            EdgarToolsErrorCode.FETCH_FAILED,
            f"EdgarTools text retrieval failed: {e}") from e
    return text, TextRepresentation.SECTION_TEXT


def filing_sections(ticker: str, accession: str) -> Dict[str, EdgarToolsSectionRef]:
    """Item/section retrieval -- EdgarTools' structural counterpart to
    `finance.documents.text_normalization.select_relevant_item_blocks`.
    Keys are EdgarTools' own section names (e.g. "item_201"); every value's
    `.text` is verbatim, never summarized."""
    filing = _resolve_underlying_filing(ticker, accession)
    try:
        obj = filing.obj()
        sections = getattr(obj, "sections", None)
    except Exception as e:  # noqa: BLE001
        raise EdgarToolsAdapterError(
            EdgarToolsErrorCode.FETCH_FAILED,
            f"EdgarTools section retrieval failed: {e}") from e
    out: Dict[str, EdgarToolsSectionRef] = {}
    if not sections:
        return out
    items = sections.items() if hasattr(sections, "items") else enumerate(sections)
    for name, section in items:
        try:
            text = section.text() if hasattr(section, "text") else str(section)
        except Exception:  # noqa: BLE001 -- one bad section must not fail the whole filing
            text = ""
        title = getattr(section, "title", None)
        out[str(name)] = EdgarToolsSectionRef(name=str(name), title=title, text=text or "")
    return out


def filing_exhibits(ticker: str, accession: str) -> List[EdgarToolsExhibitRef]:
    """Exhibit discovery -- EdgarTools' structural counterpart to
    `finance.reported_actuals.discovery.select_results_document`'s HTML
    index-table parsing. Uses the filing's `.attachments` (every document in
    the filing, typed), not a formatted summary."""
    filing = _resolve_underlying_filing(ticker, accession)
    try:
        attachments = filing.attachments
    except Exception as e:  # noqa: BLE001
        raise EdgarToolsAdapterError(
            EdgarToolsErrorCode.FETCH_FAILED,
            f"EdgarTools exhibit discovery failed: {e}") from e
    out = []
    for a in attachments:
        out.append(EdgarToolsExhibitRef(
            document=str(getattr(a, "document", "") or ""),
            document_type=str(getattr(a, "document_type", "") or "") or None,
            description=str(getattr(a, "description", "") or "") or None))
    return out
