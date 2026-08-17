"""Phase H.4/H.6 — current management guidance, extracted from SEC-filed material.

WHY THIS IS NOT AN LLM READING A PRESS RELEASE
==============================================
Quantitative guidance never appears in XBRL company facts; it lives in the
earnings-release exhibit attached to an item-2.02 8-K. That is prose, and the
obvious approach — hand the document to the local model and ask for the
numbers — is exactly the approach section 5 forbids, for a good reason: a
model asked to find guidance in a document that contains none will supply
some, and a fabricated forward number is far more damaging than a missing
one because it flows straight into the DCF and looks identical to a real one.

So NOTHING here interprets. Every value is produced by a reviewed pattern
matched against the document text, and everything a pattern does not match is
DROPPED rather than guessed at. The document is untrusted input to a parser,
never instructions and never context for a model.

THE TWO BUGS PHASE H.6 EXISTS TO FIX
====================================
Both were found live, and they are opposite failures of the same missing
idea — that a guidance number without a METRIC IDENTITY is not guidance.

1. AT&T (mapped to the wrong metric). The July 2026 release says:

       Service revenue growth in the low-single-digit range annually
       Advanced Connectivity service revenue growth in the mid-single-digit ...
       Adjusted EBITDA* growth in the 3% to 4% range in 2026, improving to ...

   `revenue_growth`'s keyword matched "Service revenue growth", found no
   number in that bullet, and its 110-character window ran on into the NEXT
   bullet and captured "3% to 4%" — adjusted EBITDA growth. Nothing stopped
   it, because EBITDA was not in the metric table at all, so there was no
   "other metric keyword" in between to act as a boundary. AT&T's EBITDA
   guidance was stored as `revenue_growth`, on a GAAP basis, and fed straight
   into `dcf.assumption.revenue_growth`. Two errors in one value: the wrong
   metric AND the wrong basis.

   Fixed three ways, all of which are needed: EBITDA is now a first-class
   metric (so it acts as a boundary), keyword matching is LONGEST-MATCH-WINS
   (so "service revenue growth" can never be read as consolidated revenue
   growth), and a window now stops at a clause boundary rather than running
   a fixed number of characters into the next sentence.

2. NVIDIA (not extracted at all). The May 2026 release says:

       NVIDIA's outlook for the second quarter of fiscal 2027 is as follows:
       Revenue is expected to be $91.0 billion, plus or minus 2%.
       GAAP and non-GAAP gross margins are expected to be 74.9% and 75.0%,
       respectively, plus or minus 50 basis points.

   Three independent reasons nothing was found: the expected fiscal year was
   hard-coded to the CALENDAR year (2026) and NVDA's guidance names fiscal
   2027, so every candidate was rejected; guidance was assumed to be ANNUAL
   and this is a next-QUARTER outlook; and the value is a point with a
   tolerance rather than a range, which the range-only rule was built to
   require. All three are now handled, and the fiscal-period model records
   "Q2 FY2027" rather than flattening it to a year.

PRECISION IS STILL BOUGHT STRUCTURALLY
======================================
A guidance figure must still be expressed either as a RANGE ("between $3.60
and $3.75", "2% to 3%") or as a POINT WITH AN EXPLICIT TOLERANCE ("$91.0
billion, plus or minus 2%"). Companies state guidance that way and state
actuals as bare single values, so the requirement separates the two
automatically. It is why, in AOS's own release, the line

    Diluted EPS (GAAP)  $ 3.60-3.75   $ 3.85

yields the guidance range and silently ignores the $3.85 prior-year actual
sitting next to it. The cost is recall on bare single-point guidance, which
is accepted deliberately: a miss is visible (guidance reads as unavailable,
and the whole workflow is built to keep working that way — see section 20),
whereas a false positive is not.

Each extracted value keeps the exact excerpt it came from, so any number in
the report can be traced back to the sentence in the filing that supports it.
"""

import hashlib
import html as _html
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# The 8-K item that means "Results of Operations and Financial Condition" —
# the earnings release. Guidance in any other item type is not looked for.
EARNINGS_RELEASE_ITEM = "2.02"

# GAAP vs adjusted must never be mixed (section 5/6). A metric is "adjusted"
# only when the document says so within the matched window.
BASIS_GAAP = "GAAP"
BASIS_ADJUSTED = "adjusted"
BASIS_COMPANY_DEFINED = "company_defined"
BASIS_NONE = "not_specified"

_ADJUSTED_MARKERS = re.compile(r"(?i)\b(adjusted|non-?GAAP|core|underlying|comparable)\b")


class GuidanceUnit:
    RATIO = "ratio"                # growth rates, margins: stored as decimals
    CURRENCY = "currency"          # absolute amounts, in the document's scale
    CURRENCY_PER_SHARE = "currency_per_share"


class GuidanceStatus:
    """Section 9. A guidance figure's standing at the time of the valuation."""

    CURRENT = "CURRENT"
    SUPERSEDED = "SUPERSEDED"
    WITHDRAWN = "WITHDRAWN"
    EXPIRED = "EXPIRED"
    ALL = (CURRENT, SUPERSEDED, WITHDRAWN, EXPIRED)


class GuidancePeriodType:
    ANNUAL = "annual"
    QUARTER = "quarter"
    MULTI_YEAR = "multi_year"
    ALL = (ANNUAL, QUARTER, MULTI_YEAR)


class GuidanceBound:
    """How the company expressed the figure.

    A range and a floor are not the same claim. "Adjusted EPS of $2.25 to
    $2.35" states both ends; "Free cash flow of $18 billion+" states only a
    minimum, and reading its midpoint as a forecast would be inventing a
    number the company withheld.
    """

    RANGE = "range"
    AT_LEAST = "at_least"
    APPROXIMATELY = "approximately"
    ALL = (RANGE, AT_LEAST, APPROXIMATELY)


# ---------------------------------------------------------------------------
# Section 6 — the metric taxonomy
# ---------------------------------------------------------------------------


class GuidanceMetricName:
    """Deterministic metric categories.

    The legacy names (`revenue`, `revenue_growth`, `operating_margin`,
    `earnings_per_share`, `adjusted_earnings_per_share`,
    `capital_expenditure`, `free_cash_flow`, `operating_cash_flow`) are kept
    EXACTLY as they were and continue to mean the CONSOLIDATED, total-company
    measure. Everything else is new. Renaming the existing keys would have
    silently changed what every downstream consumer reads, which is the same
    class of failure this taxonomy exists to prevent.
    """

    # Revenue — consolidated vs its components. The distinction is the whole
    # point: AT&T guides service revenue, which is most of but not all of
    # consolidated revenue, and the two must never be treated as one number.
    CONSOLIDATED_REVENUE = "revenue"
    CONSOLIDATED_REVENUE_GROWTH = "revenue_growth"
    SERVICE_REVENUE = "service_revenue"
    SERVICE_REVENUE_GROWTH = "service_revenue_growth"
    PRODUCT_REVENUE = "product_revenue"
    PRODUCT_REVENUE_GROWTH = "product_revenue_growth"
    SEGMENT_REVENUE_GROWTH = "segment_revenue_growth"

    # Profitability
    OPERATING_INCOME = "operating_income"
    OPERATING_MARGIN = "operating_margin"
    ADJUSTED_OPERATING_MARGIN = "adjusted_operating_margin"
    GROSS_MARGIN = "gross_margin"
    ADJUSTED_GROSS_MARGIN = "adjusted_gross_margin"
    OPERATING_EXPENSES = "operating_expenses"
    ADJUSTED_OPERATING_EXPENSES = "adjusted_operating_expenses"

    # EBITDA. Separate from every revenue measure, separate from operating
    # income, and GAAP separate from adjusted.
    EBITDA = "ebitda"
    ADJUSTED_EBITDA = "adjusted_ebitda"
    EBITDA_GROWTH = "ebitda_growth"
    ADJUSTED_EBITDA_GROWTH = "adjusted_ebitda_growth"

    # Per share
    EPS = "earnings_per_share"
    ADJUSTED_EPS = "adjusted_earnings_per_share"

    # Cash and capital
    CAPEX = "capital_expenditure"
    OPERATING_CASH_FLOW = "operating_cash_flow"
    FREE_CASH_FLOW = "free_cash_flow"
    SHARE_REPURCHASES = "share_repurchases"
    NET_LEVERAGE_TARGET = "net_leverage_target"
    TAX_RATE = "tax_rate"
    # Cash items a release guides ALONGSIDE free cash flow, and which the
    # free-cash-flow keyword's window would otherwise reach into. AT&T's own
    # sentence is the case: "...free cash flow* outlook anticipates annual
    # cash taxes of $1.0 billion to $1.5 billion and cash contributions to
    # its employee benefit plans...". Both are real guided quantities and
    # both are boundaries — the same mechanism that stops EBITDA growth from
    # being read as revenue growth.
    CASH_TAXES = "cash_taxes"
    PENSION_CONTRIBUTIONS = "pension_contributions"


# Metrics that describe TOTAL-COMPANY revenue. Only these may anchor a DCF
# revenue-growth assumption directly (section 7 / section 12 precedence 1).
CONSOLIDATED_REVENUE_METRICS = frozenset({
    GuidanceMetricName.CONSOLIDATED_REVENUE,
    GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH,
})

# Metrics that describe PART of revenue. Usable as supporting forward
# evidence, never as consolidated revenue guidance (section 7: "preserve that
# distinction ... do not pretend it is exact consolidated revenue guidance").
REVENUE_COMPONENT_METRICS = frozenset({
    GuidanceMetricName.SERVICE_REVENUE,
    GuidanceMetricName.SERVICE_REVENUE_GROWTH,
    GuidanceMetricName.PRODUCT_REVENUE,
    GuidanceMetricName.PRODUCT_REVENUE_GROWTH,
    GuidanceMetricName.SEGMENT_REVENUE_GROWTH,
})

# Metrics that are NOT revenue in any form. Named explicitly so the rule
# "never map EBITDA growth to revenue growth" is a table lookup rather than a
# reviewer's memory.
NON_REVENUE_METRICS = frozenset({
    GuidanceMetricName.OPERATING_INCOME, GuidanceMetricName.OPERATING_MARGIN,
    GuidanceMetricName.ADJUSTED_OPERATING_MARGIN, GuidanceMetricName.GROSS_MARGIN,
    GuidanceMetricName.ADJUSTED_GROSS_MARGIN, GuidanceMetricName.OPERATING_EXPENSES,
    GuidanceMetricName.ADJUSTED_OPERATING_EXPENSES,
    GuidanceMetricName.EBITDA, GuidanceMetricName.ADJUSTED_EBITDA,
    GuidanceMetricName.EBITDA_GROWTH, GuidanceMetricName.ADJUSTED_EBITDA_GROWTH,
    GuidanceMetricName.EPS, GuidanceMetricName.ADJUSTED_EPS,
    GuidanceMetricName.CAPEX, GuidanceMetricName.OPERATING_CASH_FLOW,
    GuidanceMetricName.FREE_CASH_FLOW, GuidanceMetricName.SHARE_REPURCHASES,
    GuidanceMetricName.NET_LEVERAGE_TARGET, GuidanceMetricName.TAX_RATE,
    GuidanceMetricName.CASH_TAXES, GuidanceMetricName.PENSION_CONTRIBUTIONS,
})


def is_consolidated_revenue_metric(name: str) -> bool:
    return name in CONSOLIDATED_REVENUE_METRICS


def is_revenue_component_metric(name: str) -> bool:
    return name in REVENUE_COMPONENT_METRICS


def may_anchor_revenue_growth(name: str) -> bool:
    """Section 7: only consolidated revenue guidance may populate
    `dcf.assumption.revenue_growth.*` directly. Everything else — EBITDA
    growth above all — is barred here rather than in each consumer."""
    return name == GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH


# ---------------------------------------------------------------------------
# The structured guidance record (section 5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuidanceMetric:
    """One guidance figure, with everything needed to cite and validate it."""

    name: str
    low: float
    high: float
    unit: str
    basis: str
    fiscal_year: int
    evidence_id: str
    source_excerpt: str
    scale: Optional[str] = None      # "millions"/"billions" when stated
    # Phase H.6 additions (section 5).
    guidance_id: str = ""
    issued_at: Optional[str] = None
    fiscal_period: Optional[str] = None       # "FY2026", "Q2 FY2027"
    period_type: str = GuidancePeriodType.ANNUAL
    scope: str = "consolidated"               # consolidated | service | product | segment
    source_accession: Optional[str] = None
    source_evidence_ids: Tuple[str, ...] = ()
    # How the company expressed the figure. AT&T states most of its plan as
    # floors ("$18 billion+"); a floor stored as low==high is a MINIMUM, not
    # a midpoint forecast, and nothing may read it as one.
    bound_type: str = GuidanceBound.RANGE
    status: str = GuidanceStatus.CURRENT
    status_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name, "low": self.low, "high": self.high, "unit": self.unit,
            "basis": self.basis, "fiscal_year": self.fiscal_year,
            "evidence_id": self.evidence_id, "source_excerpt": self.source_excerpt,
            "scale": self.scale,
            "guidance_id": self.guidance_id,
            "issued_at": self.issued_at,
            "fiscal_period": self.fiscal_period,
            "period_type": self.period_type,
            "midpoint": self.midpoint,
            "units": self.unit,
            "scope": self.scope,
            "source_accession": self.source_accession,
            "source_evidence_ids": list(self.source_evidence_ids),
            "bound_type": self.bound_type,
            "status": self.status,
            "status_reason": self.status_reason,
        }

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0

    def with_status(self, status: str, reason: Optional[str] = None) -> "GuidanceMetric":
        return GuidanceMetric(
            **{**self.__dict__, "status": status, "status_reason": reason})


@dataclass(frozen=True)
class GuidanceRelease:
    """Guidance as stated by ONE filing."""

    symbol: str
    fiscal_year: Optional[int]
    accession: str
    document: str
    filed: str
    metrics: Dict[str, GuidanceMetric] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "fiscal_year": self.fiscal_year,
            "accession": self.accession,
            "document": self.document,
            "filed": self.filed,
            "source_document": f"SEC 8-K exhibit {self.document} (accession {self.accession}, "
                               f"filed {self.filed})",
            "guidance_date": self.filed,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Document handling
# ---------------------------------------------------------------------------


def html_to_text(document: str) -> str:
    """Filed HTML -> flat text. No interpretation, just tag removal.

    Block-level tags become a full stop as well as a space (Phase H.6): an
    outlook table or bullet list flattens to one long line otherwise, and a
    clause boundary is what stops one bullet's metric name from claiming the
    next bullet's number — the AT&T failure exactly.
    """
    if not document:
        return ""
    text = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", document)
    # Table cell boundaries carry meaning in an outlook table; keep them as
    # separators so "$3,900 $3,950" does not become "$3,900$3,950".
    text = re.sub(r"(?i)</(td|th)\s*>", " ", text)
    # Row/paragraph/list boundaries END A STATEMENT.
    text = re.sub(r"(?i)</(tr|p|div|li)\s*>", " . ", text)
    text = re.sub(r"(?i)<br\s*/?>", " . ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = _html.unescape(text)
    # Non-breaking spaces and the bullet glyphs EDGAR filings are full of.
    text = text.replace("\xa0", " ").replace("•", " . ").replace("◦", " . ")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text).strip()
    # Collapse the runs of separators the substitutions above can produce.
    text = re.sub(r"(?:\s*\.\s*){2,}", ". ", text)
    return text.strip()


def find_earnings_release_filings(submissions: dict, limit: int = 8) -> List[dict]:
    """Item-2.02 8-K filings, NEWEST FIRST.

    Newest-first ordering is what makes supersession work: guidance from the
    most recent release that states a metric replaces every earlier statement
    of the SAME metric for the SAME period (section 9).
    """
    recent = (submissions.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    items = recent.get("items") or []
    accessions = recent.get("accessionNumber") or []
    filing_dates = recent.get("filingDate") or []
    report_dates = recent.get("reportDate") or []

    out = []
    for index, form in enumerate(forms):
        if form != "8-K":
            continue
        item_text = items[index] if index < len(items) else ""
        if EARNINGS_RELEASE_ITEM not in (item_text or ""):
            continue
        out.append({
            "accession": accessions[index] if index < len(accessions) else "",
            "filed": filing_dates[index] if index < len(filing_dates) else "",
            "report_date": report_dates[index] if index < len(report_dates) else "",
            "items": item_text,
        })
        if len(out) >= limit:
            break
    return out


# EDGAR's index.json labels each file with its ICON, not its exhibit type, so
# the authoritative type table is the filing's own index page.
_FILING_INDEX_ROW = re.compile(r"(?is)<tr[^>]*>(.*?)</tr>")
_FILING_INDEX_CELL = re.compile(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>")


def select_exhibit_document(filing_index_html: str,
                            exhibit_types: Sequence[str] = ("EX-99.1", "EX-99")) -> Optional[str]:
    """The document name of the earnings-release exhibit, from the filing's
    index page. Types are matched in the order given, so EX-99.1 (the
    conventional earnings release) wins over a bare EX-99."""
    if not filing_index_html:
        return None
    rows = []
    for row_html in _FILING_INDEX_ROW.findall(filing_index_html):
        cells = [re.sub(r"(?s)<[^>]+>", " ", cell) for cell in _FILING_INDEX_CELL.findall(row_html)]
        cells = [re.sub(r"\s+", " ", _html.unescape(c)).strip() for c in cells]
        if len(cells) >= 4:
            rows.append(cells)
    for wanted in exhibit_types:
        for cells in rows:
            # Columns are Seq | Description | Document | Type | Size.
            if cells[3].upper() == wanted.upper():
                document = cells[2].split()[0] if cells[2] else ""
                if document.lower().endswith((".htm", ".html", ".txt")):
                    return document
    return None


# ---------------------------------------------------------------------------
# Numeric patterns
# ---------------------------------------------------------------------------

_NUM = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"

# A RANGE, in the spellings real releases use. Requiring two numbers is one
# of the two precision guards — see the module docstring.
#
# A SCALE WORD may sit between the number and the connector: "$23 billion to
# $24 billion" is how most large-cap guidance is written, and requiring the
# two numbers to be adjacent to the connector silently dropped every such
# range — AT&T's "Capital investment* in the $23 billion to $24 billion range
# annually" among them. The scale itself is recorded on the metric (`scale`),
# not folded into the value, so a figure stays in the units the release used.
_SCALE = r"(?:\s*(?:billion|million|bn|mm)s?)?"
_RANGE_PATTERNS = (
    # "between $3.60 and $3.75", "between 2% and 3%", "between $23 billion and $24 billion"
    re.compile(rf"(?i)between\s*\$?\s*({_NUM})\s*(%?){_SCALE}\s*(?:and|to)\s*"
               rf"\$?\s*({_NUM})\s*(%?){_SCALE}"),
    # "$3.60 to $3.75", "2% to 3%", "a range of 2% to 3%", "$23 billion to $24 billion"
    re.compile(rf"(?i)\$?\s*({_NUM})\s*(%?){_SCALE}\s*to\s*\$?\s*({_NUM})\s*(%?){_SCALE}"),
    # "$ 3.60-3.75", "2%-3%"
    re.compile(rf"(?i)\$?\s*({_NUM})\s*(%?){_SCALE}\s*-\s*\$?\s*({_NUM})\s*(%?){_SCALE}"),
)

# A POINT WITH AN EXPLICIT TOLERANCE — the other accepted guidance shape, and
# the one NVIDIA states every metric in:
#
#     "Revenue is expected to be $91.0 billion, plus or minus 2%."
#     "gross margins are expected to be 74.9% ... plus or minus 50 basis points"
#
# A bare "$91.0 billion" is still refused: without the stated tolerance there
# is nothing distinguishing a forward figure from a reported actual, which is
# the guard the range requirement exists to provide.
_TOLERANCE_PATTERN = re.compile(
    rf"(?i)\$?\s*({_NUM})\s*(%?)\s*(billion|million)?\s*,?\s*"
    r"(?:plus\s*or\s*minus|\+/-|±)\s*"
    rf"({_NUM})\s*(%|basis\s*points|bps)")

# "GAAP and non-GAAP gross margins are expected to be 74.9% and 75.0%,
# respectively" — two POINT values for two different bases, which must never
# be read as a single 74.9%-75.0% range. Captured explicitly so the GAAP and
# adjusted measures land in their own metrics rather than being lost.
_DUAL_BASIS_PATTERN = re.compile(
    rf"(?i)GAAP\s+and\s+non-?GAAP\b[^.]{{0,80}}?"
    rf"(?:are\s+)?expected\s+to\s+be\s+"
    rf"(?:approximately\s+)?\$?\s*({_NUM})\s*(%?)\s*(billion|million)?\s*"
    rf"and\s+(?:approximately\s+)?\$?\s*({_NUM})\s*(%?)\s*(billion|million)?")

# A ONE-SIDED FLOOR — "Free cash flow* of $18 billion+ in 2026", "expected
# growth of 5%+ in 2026", "$45 billion+ to shareholders". AT&T states most of
# its multi-year plan this way, and a floor is real, quantified, checkable
# guidance: dropping it left the T report with no free-cash-flow guidance at
# all while the release led with it.
#
# It is NOT flattened into a range. `bound_type` records that the company
# stated a minimum, so nothing downstream can treat the number as a midpoint
# forecast — which would be reading "at least $18B" as "we expect $18B".
_FLOOR_PATTERN = re.compile(
    rf"(?i)\$?\s*({_NUM})\s*(%?)\s*(billion|million)?\s*\+"
    rf"|(?:at\s+least|no\s+less\s+than|or\s+better|or\s+more)\s*\$?\s*({_NUM})\s*(%?)")

# How far after a metric name a one-sided floor may begin. Roughly "of", "in
# the", "at least" — a connector, not a clause.
_FLOOR_ADJACENCY = 25

_BASIS_POINTS_PER_PERCENT = 100.0



# ---------------------------------------------------------------------------
# Metric keywords (section 6)
# ---------------------------------------------------------------------------
#
# (canonical name, unit, percent required, scope, forced basis or None, regex)
#
# ORDER DOES NOT DECIDE PRECEDENCE — length of the matched text does (see
# `_metric_keyword_hits`). "Service revenue growth" must never be read as
# consolidated "revenue growth" merely because the latter is listed first,
# and relying on list order to prevent that is exactly the kind of implicit
# rule that produced the AT&T mis-mapping.

_SCOPE_CONSOLIDATED = "consolidated"
_SCOPE_SERVICE = "service"
_SCOPE_PRODUCT = "product"
_SCOPE_SEGMENT = "segment"

_METRIC_PATTERNS = (
    # -- revenue: components FIRST in the file for readability only --------
    (GuidanceMetricName.SERVICE_REVENUE_GROWTH, GuidanceUnit.RATIO, True, _SCOPE_SERVICE, None,
     re.compile(r"(?i)\b(?:consolidated\s+|total\s+)?service\s+revenues?\s+growth\b|"
                r"\bgrowth\s+in\s+(?:consolidated\s+)?service\s+revenues?\b")),
    (GuidanceMetricName.SERVICE_REVENUE, GuidanceUnit.CURRENCY, False, _SCOPE_SERVICE, None,
     re.compile(r"(?i)\b(?:consolidated\s+|total\s+)?service\s+revenues?\b")),
    (GuidanceMetricName.PRODUCT_REVENUE_GROWTH, GuidanceUnit.RATIO, True, _SCOPE_PRODUCT, None,
     re.compile(r"(?i)\bproduct\s+revenues?\s+growth\b")),
    (GuidanceMetricName.PRODUCT_REVENUE, GuidanceUnit.CURRENCY, False, _SCOPE_PRODUCT, None,
     re.compile(r"(?i)\bproduct\s+revenues?\b")),
    # A named business line ("Advanced Connectivity service revenue growth",
    # "Legacy service revenue", "Data Center revenue"). Deliberately matched
    # as SEGMENT rather than consolidated: AT&T's release states four
    # different segment revenue trajectories, none of which is the company's
    # total.
    (GuidanceMetricName.SEGMENT_REVENUE_GROWTH, GuidanceUnit.RATIO, True, _SCOPE_SEGMENT, None,
     re.compile(r"(?i)\b(?:advanced\s+connectivity|legacy|mobility|business\s+wireline|"
                r"consumer\s+wireline|data\s+center|gaming|automotive|segment)\s+"
                r"(?:service\s+)?revenues?\s+growth\b")),

    # -- revenue: consolidated --------------------------------------------
    (GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH, GuidanceUnit.RATIO, True,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\b(?:consolidated\s+|total\s+)?(?:net\s+)?sales\s+growth\b|"
                r"\b(?:consolidated\s+|total\s+)?revenues?\s+growth\b|"
                r"\bgrowth\s+in\s+(?:consolidated\s+|total\s+)?(?:net\s+)?sales\b|"
                r"\brevenues?\s+(?:is|are)\s+expected\s+to\s+grow\b")),
    (GuidanceMetricName.CONSOLIDATED_REVENUE, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     # Bare "Revenue" is accepted here (NVDA's outlook says exactly that),
     # which is only safe because a value still has to be a range or a
     # point-with-tolerance inside a forward-looking clause.
     re.compile(r"(?i)\b(?:consolidated\s+|total\s+)?net\s+sales\b|"
                r"\b(?:consolidated\s+|total\s+)?revenues?\b")),

    # -- EBITDA. The metric whose absence caused the AT&T mis-mapping. -----
    (GuidanceMetricName.ADJUSTED_EBITDA_GROWTH, GuidanceUnit.RATIO, True,
     _SCOPE_CONSOLIDATED, BASIS_ADJUSTED,
     re.compile(r"(?i)\b(?:adjusted|non-?GAAP)\s+EBITDA\s*\*?\s+growth\b|"
                r"\bgrowth\s+in\s+(?:adjusted|non-?GAAP)\s+EBITDA\b")),
    (GuidanceMetricName.EBITDA_GROWTH, GuidanceUnit.RATIO, True, _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bEBITDA\s*\*?\s+growth\b|\bgrowth\s+in\s+EBITDA\b")),
    (GuidanceMetricName.ADJUSTED_EBITDA, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, BASIS_ADJUSTED,
     re.compile(r"(?i)\b(?:adjusted|non-?GAAP)\s+EBITDA\b")),
    (GuidanceMetricName.EBITDA, GuidanceUnit.CURRENCY, False, _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bEBITDA\b")),

    # -- per share ---------------------------------------------------------
    (GuidanceMetricName.ADJUSTED_EPS, GuidanceUnit.CURRENCY_PER_SHARE, False,
     _SCOPE_CONSOLIDATED, BASIS_ADJUSTED,
     re.compile(r"(?i)\b(?:adjusted|non-?GAAP)\s+(?:diluted\s+)?"
                r"(?:earnings\s+per\s+share|EPS)\s*\*?")),
    (GuidanceMetricName.EPS, GuidanceUnit.CURRENCY_PER_SHARE, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bdiluted\s+(?:earnings\s+per\s+share|EPS)\b|"
                r"(?<!adjusted\s)\bEPS\s*\(GAAP\)")),

    # -- margins and expenses ---------------------------------------------
    (GuidanceMetricName.ADJUSTED_GROSS_MARGIN, GuidanceUnit.RATIO, True,
     _SCOPE_CONSOLIDATED, BASIS_ADJUSTED,
     re.compile(r"(?i)\b(?:adjusted|non-?GAAP)\s+gross\s+margins?\b")),
    (GuidanceMetricName.GROSS_MARGIN, GuidanceUnit.RATIO, True, _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bgross\s+margins?\b")),
    (GuidanceMetricName.ADJUSTED_OPERATING_MARGIN, GuidanceUnit.RATIO, True,
     _SCOPE_CONSOLIDATED, BASIS_ADJUSTED,
     re.compile(r"(?i)\b(?:adjusted|non-?GAAP)\s+operating\s+margins?\b")),
    (GuidanceMetricName.OPERATING_MARGIN, GuidanceUnit.RATIO, True, _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\boperating\s+margins?\b")),
    (GuidanceMetricName.ADJUSTED_OPERATING_EXPENSES, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, BASIS_ADJUSTED,
     re.compile(r"(?i)\b(?:adjusted|non-?GAAP)\s+operating\s+expenses\b")),
    (GuidanceMetricName.OPERATING_EXPENSES, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\boperating\s+expenses\b")),
    (GuidanceMetricName.OPERATING_INCOME, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\boperating\s+income\b")),

    # -- cash and capital --------------------------------------------------
    (GuidanceMetricName.FREE_CASH_FLOW, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bfree\s+cash\s+flow\s*\*?")),
    (GuidanceMetricName.OPERATING_CASH_FLOW, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bcash\s+(?:provided\s+by|from)\s+operating\s+activities\b|"
                r"\boperating\s+cash\s+flow\b")),
    (GuidanceMetricName.CAPEX, GuidanceUnit.CURRENCY, False, _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bcapital\s+expenditures?\b|\bcapital\s+investment\s*\*?|\bcapex\b")),
    (GuidanceMetricName.SHARE_REPURCHASES, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bshare\s+repurchases?\b|\bstock\s+repurchases?\b|\bbuybacks?\b")),
    (GuidanceMetricName.NET_LEVERAGE_TARGET, GuidanceUnit.RATIO, False,
     _SCOPE_CONSOLIDATED, BASIS_COMPANY_DEFINED,
     re.compile(r"(?i)\bnet\s+debt[\s-]*to[\s-]*(?:adjusted\s+)?EBITDA\b|"
                r"\bleverage\s+ratio\b|\bnet\s+leverage\b")),
    (GuidanceMetricName.TAX_RATE, GuidanceUnit.RATIO, True, _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\b(?:GAAP\s+and\s+non-?GAAP\s+)?tax\s+rates?\b|"
                r"\beffective\s+tax\s+rate\b")),
    # Cash items a release guides ALONGSIDE free cash flow. AT&T's own
    # sentence is why they are here: "...free cash flow* outlook anticipates
    # annual cash taxes of $1.0 billion to $1.5 billion and cash
    # contributions to its employee benefit plans...". Both are real guided
    # quantities, and listing them makes each a BOUNDARY for the other — the
    # same mechanism that stops EBITDA growth being read as revenue growth.
    # Without them, free-cash-flow guidance came out as $1.0-1.5B.
    (GuidanceMetricName.CASH_TAXES, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\b(?:annual\s+)?cash\s+taxes\b|\bcash\s+tax\s+payments\b")),
    (GuidanceMetricName.PENSION_CONTRIBUTIONS, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bcash\s+contributions\b|\bpension\s+contributions\b|"
                r"\bbenefit\s+plan\s+contributions\b")),
)

_METRIC_BY_NAME = {entry[0]: entry for entry in _METRIC_PATTERNS}

# How far after a metric keyword a value may appear and still be that
# metric's guidance. Tight on purpose: the further away a number is, the more
# likely it belongs to a different line of the release. Reducing this from
# 160 to 110 fixed a real false positive on AOS's January release, where a
# flattened reconciliation table put "Free cash flow (non-GAAP) $546.0
# $473.8" within reach of the EPS guidance range printed underneath it.
#
# Phase H.6: the window is now additionally cut at the first CLAUSE BOUNDARY
# (see `_clause_window`), which is what actually stops one bullet's metric
# name from claiming the next bullet's number.
_WINDOW = 110

# The narrower window used to decide GAAP vs adjusted. It looks only at the
# text IMMEDIATELY BEFORE the metric keyword, because "adjusted" is a
# modifier of the metric name ("adjusted EPS"), not a property of the
# paragraph. Using the wide context window instead made AOS's plain sales
# growth come out as "adjusted" purely because an adjusted-EPS sentence
# happened to sit nearby -- and, worse, made the GAAP diluted-EPS guidance
# get dropped entirely as a suspected duplicate of the adjusted measure.
_BASIS_LOOKBEHIND = 30

_SCALE_RE = re.compile(r"(?i)\b(millions?|billions?)\b")

# A range only counts as GUIDANCE when the surrounding text says it is
# forward-looking. Without this, a prior-year comparison range would qualify.
_FORWARD_MARKERS = re.compile(
    r"(?i)\b(outlook|guidance|expect\w*|anticipat\w*|forecast\w*|target\w*|"
    r"project\w*|estimat\w*|reaffirm\w*|reiterat\w*|narrow\w*|rais\w*|lower\w*|"
    r"updat\w*|plans?\s+to|on\s+track)\b")

# Section 9: an outlook the company has taken back is not current guidance.
_WITHDRAWAL_MARKERS = re.compile(
    r"(?i)\b(withdraw\w*|suspend\w*|no\s+longer\s+(?:providing|issuing)\s+"
    r"(?:full[\s-]year\s+)?guidance|rescind\w*)\b")


def _to_number(text: str) -> Optional[float]:
    try:
        return float(text.replace(",", ""))
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Fiscal period parsing (section 8)
# ---------------------------------------------------------------------------

_QUARTER_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4,
                  "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}

_QUARTER_PERIOD_RE = re.compile(
    r"(?i)\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+"
    r"(?:of\s+)?(?:fiscal\s+(?:year\s+)?)?(20\d{2})\b"
    r"|\b(?:fiscal\s+)?Q([1-4])\s*(?:of\s+)?(?:fiscal\s+(?:year\s+)?)?(20\d{2})\b"
    r"|\bQ([1-4])\s*FY\s*(20\d{2}|\d{2})\b")

_ANNUAL_PERIOD_RE = re.compile(
    r"(?i)\b(?:full[\s-]year|fiscal\s+year|fiscal|full\s+fiscal\s+year|"
    r"for\s+the\s+year|calendar\s+year)\s+(?:of\s+)?(20\d{2})\b"
    r"|\bFY\s*(20\d{2})\b")

_MULTI_YEAR_RE = re.compile(r"\b(20\d{2})\s*-\s*(20\d{2})\b")

_YEAR_RE = re.compile(r"\b(20\d{2})\b")


@dataclass(frozen=True)
class GuidancePeriod:
    label: str
    period_type: str
    fiscal_year: int
    quarter: Optional[int] = None


def parse_guidance_period(context: str, filed: str) -> Optional[GuidancePeriod]:
    """The fiscal period a guidance figure applies to, from its own context.

    Non-calendar fiscal years are the reason this exists. NVIDIA's May-2026
    release guides "the second quarter of fiscal 2027"; hard-coding the
    expected year to the calendar year of the filing rejected every value in
    the document. The period is now READ, and only then checked for
    plausibility against the filing date.

    Returns None when no period can be identified — which is a rejection
    (section 11: "reject a number without a period"), not a default.
    """
    quarter_match = _QUARTER_PERIOD_RE.search(context)
    if quarter_match:
        groups = quarter_match.groups()
        if groups[0] and groups[1]:
            quarter = _QUARTER_WORDS.get(groups[0].lower())
            year = int(groups[1])
        elif groups[2] and groups[3]:
            quarter, year = int(groups[2]), int(groups[3])
        else:
            quarter = int(groups[4])
            raw_year = groups[5]
            year = int(raw_year) if len(raw_year) == 4 else 2000 + int(raw_year)
        if quarter:
            return GuidancePeriod(label=f"Q{quarter} FY{year}",
                                  period_type=GuidancePeriodType.QUARTER,
                                  fiscal_year=year, quarter=quarter)

    multi = _MULTI_YEAR_RE.search(context)
    annual_match = _ANNUAL_PERIOD_RE.search(context)
    if annual_match:
        year = int(annual_match.group(1) or annual_match.group(2))
        # A multi-year outlook ("2026-2028") that also names a specific year
        # for THIS figure ("in the 3% to 4% range in 2026") applies to that
        # year; the span is recorded as the period type so a reader is not
        # told a three-year plan is a one-year outlook.
        if multi and int(multi.group(1)) <= year <= int(multi.group(2)):
            return GuidancePeriod(label=f"FY{year}", period_type=GuidancePeriodType.ANNUAL,
                                  fiscal_year=year)
        return GuidancePeriod(label=f"FY{year}", period_type=GuidancePeriodType.ANNUAL,
                              fiscal_year=year)

    if multi:
        return GuidancePeriod(label=f"FY{multi.group(1)}-FY{multi.group(2)}",
                              period_type=GuidancePeriodType.MULTI_YEAR,
                              fiscal_year=int(multi.group(1)))

    years = sorted({int(y) for y in _YEAR_RE.findall(context)})
    if len(years) == 1:
        return GuidancePeriod(label=f"FY{years[0]}", period_type=GuidancePeriodType.ANNUAL,
                              fiscal_year=years[0])
    return None


# An outlook HEADER declares the period for everything that follows it until
# the next header. NVIDIA's release is built this way and cannot be read any
# other way:
#
#     NVIDIA's outlook for the second quarter of fiscal 2027 is as follows:
#     Revenue is expected to be $91.0 billion, plus or minus 2%.
#     GAAP and non-GAAP gross margins are expected to be 74.9% and 75.0% ...
#     GAAP and non-GAAP operating expenses are expected to be ... $8.3 billion.
#     For the full year fiscal 2027, NVIDIA expects ... tax rates ...
#
# The gross-margin and operating-expense sentences name no period at all; the
# only statement of their period is the header three sentences earlier. Worse,
# a window wide enough to reach the header ALSO reaches the full-year tax
# sentence below, which is a different period — so the header must win by
# being the nearest PRECEDING declaration, not by being anywhere in range.
_OUTLOOK_HEADER_RE = re.compile(
    r"(?i)\boutlook\s+for\s+(?:the\s+)?[^.]{0,60}?"
    r"((?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+"
    r"(?:of\s+)?(?:fiscal\s+(?:year\s+)?)?20\d{2}"
    r"|(?:full[\s-]year|fiscal\s+year|fiscal)\s+20\d{2}"
    r"|20\d{2}\s*-\s*20\d{2})"
    r"|\bfor\s+the\s+(full[\s-]year\s+fiscal\s+20\d{2}|full\s+year\s+fiscal\s+20\d{2})"
    r"|\b(?:outlook|guidance)\s+for\s+(20\d{2})\b")

# How far back a preceding outlook header may sit and still govern a figure.
# Generous enough to span the three or four sentences an outlook block runs
# to, far too tight to reach the previous quarter's block.
_HEADER_REACH = 700

# A bare year inside the figure's OWN clause ("in the 3% to 4% range in 2026,
# improving to 5% or better in 2028") is the strongest signal there is: it
# sits with the number rather than in a header that covers several. Applied
# first, and only when the clause names exactly one year.
_CLAUSE_YEAR_REACH = 90


def _period_declarations(text: str) -> List[Tuple[int, GuidancePeriod]]:
    """Every outlook-header period declaration, with its position."""
    declarations: List[Tuple[int, GuidancePeriod]] = []
    for match in _OUTLOOK_HEADER_RE.finditer(text):
        phrase = next((g for g in match.groups() if g), None)
        if not phrase:
            continue
        period = parse_guidance_period(phrase, "")
        if period is not None:
            declarations.append((match.end(), period))
    return declarations


def _clause_after(text: str, position: int, reach: int = _CLAUSE_YEAR_REACH) -> str:
    """The remainder of the statement that begins at `position`.

    Cut at the first clause boundary, which is what keeps AT&T's "in the 3%
    to 4% range in 2026" from reading the "2028" of a later bullet, and keeps
    NVIDIA's "$8.3 billion, respectively." from reading the "fiscal 2027" of
    the full-year tax sentence that follows it.
    """
    tail = text[position:position + reach]
    boundary = _CLAUSE_BOUNDARY.search(tail)
    return tail[:boundary.start()] if boundary else tail


def resolve_guidance_period(text: str, keyword_start: int, value_start: int, value_end: int,
                            filed: str,
                            declarations: Optional[Sequence[Tuple[int, GuidancePeriod]]] = None
                            ) -> Optional[GuidancePeriod]:
    """Which fiscal period this particular figure applies to.

    Four sources, in strict priority order, because they disagree in real
    releases and picking the wrong one attaches a number to the wrong year:

    1. A quarter named in the figure's own statement.
    2. The FIRST year named after the value, within the same statement. AT&T's
       "Adjusted EBITDA* growth in the 3% to 4% range in 2026, improving to
       5% or better in 2028" is FY2026 guidance — the year attached to the
       number, not the far end of the multi-year plan the bullet list sits
       under.
    3. The nearest PRECEDING outlook header. This is the ONLY statement of
       period NVIDIA's gross-margin and operating-expense lines have; both
       sentences name no period at all and inherit "the second quarter of
       fiscal 2027" from the header three sentences above.
    4. The surrounding context, as a last resort.

    Returns None when none of the four identifies a period — a rejection
    (section 11), never a default.
    """
    statement = text[keyword_start:value_end] + _clause_after(text, value_end)
    if _QUARTER_PERIOD_RE.search(statement):
        parsed = parse_guidance_period(statement, filed)
        if parsed is not None:
            return parsed

    trailing_years = _YEAR_RE.findall(_clause_after(text, value_end))
    if trailing_years:
        year = int(trailing_years[0])
        return GuidancePeriod(label=f"FY{year}", period_type=GuidancePeriodType.ANNUAL,
                              fiscal_year=year)

    statement_years = {int(y) for y in _YEAR_RE.findall(statement)}
    if len(statement_years) == 1:
        year = statement_years.pop()
        return GuidancePeriod(label=f"FY{year}", period_type=GuidancePeriodType.ANNUAL,
                              fiscal_year=year)

    if declarations is None:
        declarations = _period_declarations(text)
    preceding = [(position, period) for position, period in declarations
                 if position <= value_start and value_start - position <= _HEADER_REACH]
    if preceding:
        return max(preceding, key=lambda pair: pair[0])[1]

    context = text[max(0, keyword_start - _WINDOW):value_end + _WINDOW]
    return parse_guidance_period(context, filed)


def _period_is_plausible(period: GuidancePeriod, filed: str) -> bool:
    """A guided period must be the filing's own year or the next one.

    Deliberately loose enough for every non-calendar fiscal year (NVDA files
    in May 2026 and guides fiscal 2027) and strict enough to reject the prior
    year's actuals, which is the whole reason a year check exists. A
    multi-year plan is accepted when its FIRST year clears the same test.
    """
    if not filed or len(filed) < 4:
        return True
    try:
        filed_year = int(filed[:4])
    except ValueError:
        return True
    return filed_year <= period.fiscal_year <= filed_year + 1


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

# Characters that end one statement in flattened filing text. `html_to_text`
# turns every block-level boundary into a full stop for exactly this reason.
_CLAUSE_BOUNDARY = re.compile(r"[.;:]\s")


def _clause_window(text: str, start: int, limit: int = _WINDOW) -> str:
    """The text after a keyword, cut at the first clause boundary.

    This is the fix for the AT&T mis-mapping stated as a rule: a metric name
    may only claim a value that appears in ITS OWN clause. "Service revenue
    growth in the low-single-digit range annually." ends there; the "3% to 4%"
    in the following sentence belongs to whatever that sentence names.
    """
    window = text[start:start + limit]
    boundary = _CLAUSE_BOUNDARY.search(window)
    return window[:boundary.start()] if boundary else window


def _metric_keyword_hits(text: str) -> List[Tuple[int, int, tuple]]:
    """Every metric keyword occurrence, LONGEST MATCH WINNING at each start.

    Longest-match is what makes "Advanced Connectivity service revenue
    growth" a segment metric and "Service revenue growth" a service metric,
    rather than both collapsing into consolidated `revenue_growth` because
    that pattern also matches the tail of the phrase. Relying on the order of
    the pattern table for this is precisely what failed on AT&T.
    """
    hits: List[Tuple[int, int, tuple]] = []
    for entry in _METRIC_PATTERNS:
        for match in entry[5].finditer(text):
            hits.append((match.start(), match.end(), entry))

    # Drop any hit fully contained in a longer hit that starts no later.
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    kept: List[Tuple[int, int, tuple]] = []
    for start, end, entry in hits:
        if any(k_start <= start and end <= k_end for k_start, k_end, _e in kept):
            continue
        kept.append((start, end, entry))
    kept.sort(key=lambda h: h[0])
    return kept


def _find_range(window: str, require_percent: bool
                ) -> Optional[Tuple[float, float, bool, str, str]]:
    """(low, high, was_percent, matched_text, bound_type) for the first usable value.

    Accepts a range, a point-with-tolerance, or a one-sided floor. A bare
    point with no stated tolerance and no floor marker is refused — see the
    module docstring.
    """
    tolerance = _TOLERANCE_PATTERN.search(window)
    if tolerance is not None:
        centre = _to_number(tolerance.group(1))
        spread = _to_number(tolerance.group(4))
        if centre is not None and spread is not None:
            is_percent = bool(tolerance.group(2))
            unit_word = (tolerance.group(5) or "").lower()
            if "basis" in unit_word or "bps" in unit_word:
                delta = spread / _BASIS_POINTS_PER_PERCENT
            else:
                delta = centre * spread / 100.0
            if (is_percent and require_percent) or (not is_percent and not require_percent):
                return (centre - delta, centre + delta, is_percent,
                        tolerance.group(0).strip(), GuidanceBound.RANGE)

    for pattern in _RANGE_PATTERNS:
        for match in pattern.finditer(window):
            low = _to_number(match.group(1))
            high = _to_number(match.group(3))
            if low is None or high is None:
                continue
            is_percent = bool(match.group(2) or match.group(4))
            if require_percent and not is_percent:
                continue
            if not require_percent and is_percent:
                # A percent range cannot be a per-share or currency figure.
                continue
            if high < low or low == high:
                continue
            return low, high, is_percent, match.group(0).strip(), GuidanceBound.RANGE

    # A FLOOR must sit essentially ADJACENT to the metric name — "Free cash
    # flow* of $18 billion+", not "free cash flow* through 2028, its plans to
    # return $45 billion+ to shareholders". A bare "N+" is far weaker evidence
    # than a two-ended range (it is one token, and releases are full of them),
    # so it only counts when the connector between the metric and the number
    # is a preposition or two. Without this bound, AT&T's capital-return plan
    # ($45 billion+ to shareholders) was captured as free-cash-flow guidance,
    # and a SEGMENT's "expected growth of 6%+" as consolidated EBITDA growth.
    floor = _FLOOR_PATTERN.search(window)
    if floor is not None and floor.start() <= _FLOOR_ADJACENCY:
        value = _to_number(floor.group(1) or floor.group(4))
        is_percent = bool(floor.group(2) or floor.group(5))
        if value is not None and require_percent == is_percent:
            return value, value, is_percent, floor.group(0).strip(), GuidanceBound.AT_LEAST
    return None


def _guidance_id(symbol: str, accession: str, name: str, period_label: str) -> str:
    digest = hashlib.sha256(
        f"{symbol}|{accession}|{name}|{period_label}".encode("utf-8")).hexdigest()[:12]
    return f"gd_{digest}"


def extract_guidance_from_text(text: str, symbol: str, accession: str, document: str,
                               filed: str, expected_fiscal_year: Optional[int] = None
                               ) -> GuidanceRelease:
    """Pattern-match guidance out of one earnings release. No interpretation.

    `expected_fiscal_year` is now a HINT used only to label the release, not a
    filter. Filtering on it is what silently discarded every NVIDIA value:
    the caller passed the calendar year and NVDA guides its own fiscal year,
    which is a year ahead. Each figure now carries the period IT names
    (section 8), and plausibility is checked against the FILING DATE
    (`_period_is_plausible`), which rejects prior-year actuals without
    assuming a calendar fiscal year.

    Every rejection is recorded in `warnings` so a reader can tell "this
    company published no guidance" from "a value was found and refused".
    """
    metrics: Dict[str, GuidanceMetric] = {}
    warnings: List[str] = []
    if not text:
        return GuidanceRelease(symbol=symbol, fiscal_year=expected_fiscal_year,
                               accession=accession, document=document, filed=filed,
                               warnings=("The filing document was empty.",))

    hits = _metric_keyword_hits(text)
    boundaries = [start for start, _end, _entry in hits]
    declarations = _period_declarations(text)

    for index, (start, end, entry) in enumerate(hits):
        name, unit, require_percent, scope, forced_basis, _regex = entry
        if name in metrics:
            continue

        # The window stops at the earlier of: a clause boundary, the next
        # metric keyword, or `_WINDOW` characters. All three matter — the
        # next-keyword cut is what keeps a table row from reaching the row
        # below it even when neither carries punctuation.
        next_keyword = next((b for b in boundaries if b > end), None)
        limit = _WINDOW if next_keyword is None else min(_WINDOW, next_keyword - end)
        if limit <= 0:
            continue
        window = _clause_window(text, end, limit)
        context = text[max(0, start - _WINDOW):end + _WINDOW]

        if not _FORWARD_MARKERS.search(context):
            continue

        found = _find_range(window, require_percent)
        if found is None:
            found = _dual_basis_value(text, end, name, require_percent)
            if found is None:
                continue
        low, high, is_percent, matched, bound_type = found

        # Where the matched value actually sits, so the period rules can look
        # at the text immediately after THE NUMBER rather than after the
        # metric name.
        # The dual-basis pattern searches BACKWARD from the keyword, so its
        # match can begin before `end`; search from there or the period rules
        # below would measure from the wrong place.
        value_start = text.find(matched, max(0, end - 140),
                                end + limit + len(matched) + 8)
        if value_start < 0:
            value_start = end
        value_end = value_start + len(matched)

        period = resolve_guidance_period(text, start, value_start, value_end, filed,
                                         declarations)
        if period is None:
            warnings.append(
                f"Ignored a {name} value ({matched}) because no fiscal period could be "
                "identified near it.")
            continue
        if not _period_is_plausible(period, filed):
            warnings.append(
                f"Ignored a {name} value ({matched}) for {period.label}: a release filed "
                f"{filed} cannot be stating forward guidance for that period.")
            continue

        # GAAP vs adjusted is decided from the modifier immediately before
        # the metric name, never from the wider paragraph — see
        # `_BASIS_LOOKBEHIND`. A metric whose own name carries the basis
        # (adjusted EBITDA, adjusted EPS) forces it.
        modifier = text[max(0, start - _BASIS_LOOKBEHIND):start]
        basis = forced_basis or (
            BASIS_ADJUSTED if _ADJUSTED_MARKERS.search(modifier) else BASIS_GAAP)
        if forced_basis is None and basis == BASIS_ADJUSTED:
            # "adjusted" directly modifies this mention, so it is the adjusted
            # measure. Storing it under the GAAP name would mix the two bases
            # (section 6). The adjusted variant has its own pattern and will
            # match the same text.
            adjusted_name = _ADJUSTED_TWIN.get(name)
            if adjusted_name and adjusted_name in _METRIC_BY_NAME:
                continue
            basis = BASIS_ADJUSTED

        status = GuidanceStatus.CURRENT
        status_reason = None
        if _WITHDRAWAL_MARKERS.search(context):
            status = GuidanceStatus.WITHDRAWN
            status_reason = (
                "The text around this figure states that the outlook was withdrawn or "
                "suspended, so it is retained for history and never used as current guidance.")

        scale_match = _SCALE_RE.search(context)
        if is_percent:
            low, high = low / 100.0, high / 100.0

        evidence_id = f"dcf.guidance.{name}.current"
        metrics[name] = GuidanceMetric(
            name=name, low=low, high=high, unit=unit, basis=basis,
            fiscal_year=period.fiscal_year,
            evidence_id=evidence_id,
            source_excerpt=_excerpt(text, start, end + limit),
            scale=(scale_match.group(1).lower() if scale_match and
                   unit == GuidanceUnit.CURRENCY else None),
            guidance_id=_guidance_id(symbol, accession, name, period.label),
            issued_at=filed,
            fiscal_period=period.label,
            period_type=period.period_type,
            scope=scope,
            source_accession=accession,
            source_evidence_ids=(evidence_id,),
            bound_type=bound_type,
            status=status,
            status_reason=status_reason)

    return GuidanceRelease(symbol=symbol, fiscal_year=expected_fiscal_year,
                           accession=accession, document=document, filed=filed,
                           metrics=metrics, warnings=tuple(warnings))


# GAAP metric -> its adjusted counterpart, for the "an 'adjusted' modifier
# means the adjusted metric owns this text" rule above.
_ADJUSTED_TWIN = {
    GuidanceMetricName.EPS: GuidanceMetricName.ADJUSTED_EPS,
    GuidanceMetricName.GROSS_MARGIN: GuidanceMetricName.ADJUSTED_GROSS_MARGIN,
    GuidanceMetricName.OPERATING_MARGIN: GuidanceMetricName.ADJUSTED_OPERATING_MARGIN,
    GuidanceMetricName.OPERATING_EXPENSES: GuidanceMetricName.ADJUSTED_OPERATING_EXPENSES,
    GuidanceMetricName.EBITDA: GuidanceMetricName.ADJUSTED_EBITDA,
    GuidanceMetricName.EBITDA_GROWTH: GuidanceMetricName.ADJUSTED_EBITDA_GROWTH,
}

# Which of a "GAAP and non-GAAP X are expected to be A and B" pair belongs to
# which metric. The GAAP figure is stated first in every release reviewed.
_DUAL_BASIS_ADJUSTED_METRICS = frozenset({
    GuidanceMetricName.ADJUSTED_GROSS_MARGIN,
    GuidanceMetricName.ADJUSTED_OPERATING_MARGIN,
    GuidanceMetricName.ADJUSTED_OPERATING_EXPENSES,
})


def _dual_basis_value(text: str, keyword_end: int, name: str, require_percent: bool
                      ) -> Optional[Tuple[float, float, bool, str]]:
    """Handle "GAAP and non-GAAP X are expected to be A and B, respectively".

    Two POINT values for two bases, which must never be read as one A-to-B
    range (that would invent a spread the company did not state and mix the
    two bases into one figure). Each side becomes its own metric's value,
    stored as a degenerate low==high range so the record shape stays uniform.
    """
    window = text[max(0, keyword_end - 120):keyword_end + _WINDOW]
    match = _DUAL_BASIS_PATTERN.search(window)
    if match is None:
        return None
    gaap_value = _to_number(match.group(1))
    adjusted_value = _to_number(match.group(4))
    if gaap_value is None or adjusted_value is None:
        return None
    is_percent = bool(match.group(2) or match.group(5))
    if require_percent != is_percent:
        return None
    value = adjusted_value if name in _DUAL_BASIS_ADJUSTED_METRICS else gaap_value
    return value, value, is_percent, match.group(0).strip(), GuidanceBound.APPROXIMATELY


def _excerpt(text: str, start: int, end: int, pad: int = 40) -> str:
    lo = max(0, start - pad)
    hi = min(len(text), end)
    return text[lo:hi].strip()


# ---------------------------------------------------------------------------
# Section 9 — selecting CURRENT guidance
# ---------------------------------------------------------------------------


def select_current_guidance(releases: Sequence[GuidanceRelease],
                            fiscal_year: Optional[int] = None,
                            as_of: Optional[str] = None
                            ) -> Tuple[Optional[GuidanceRelease], List[GuidanceRelease]]:
    """The current guidance for each metric, plus every release it supersedes.

    PER-METRIC, not per-release, and that is the Phase H.6 correction. The
    old rule picked the newest release that had ANY metrics and discarded the
    rest. On AT&T that chose the January release — whose "3% to 4%" was
    misread as revenue growth — while the July release, which reiterates the
    same outlook, contributed nothing. Selecting per metric means the newest
    STATEMENT OF EACH METRIC wins, which is what supersession actually means.

    Withdrawn guidance is never selected as current (section 9). Guidance for
    a period that has already ended is marked EXPIRED and likewise not
    selected. Both are still returned among the superseded releases, because
    "management withdrew its outlook" is itself material.
    """
    usable = [r for r in releases if r.metrics]
    if not usable:
        return None, []
    ordered = sorted(usable, key=lambda r: (r.filed or "", r.accession or ""), reverse=True)

    current_metrics: Dict[str, GuidanceMetric] = {}
    supplier_of: Dict[str, GuidanceRelease] = {}
    for release in ordered:
        for name, metric in release.metrics.items():
            # `fiscal_year` is the EARLIEST period still relevant, not an
            # exact match. Guidance for a LATER fiscal year is the whole
            # point of asking — NVIDIA's May-2026 release guides fiscal 2027
            # — and requiring equality against the calendar year is what
            # discarded it. Only guidance for an already-past period is
            # dropped here.
            if fiscal_year is not None and metric.fiscal_year is not None \
                    and metric.fiscal_year < fiscal_year:
                continue
            if metric.status == GuidanceStatus.WITHDRAWN:
                continue
            if _is_expired(metric, as_of):
                continue
            if name in current_metrics:
                continue
            current_metrics[name] = metric
            supplier_of[name] = release

    if not current_metrics:
        return None, list(ordered)

    # The release that supplied the most current metrics names the result —
    # a guidance record has to cite ONE primary filing, and every individual
    # metric still carries its own `source_accession`. Counted by release
    # IDENTITY rather than accession number: two releases can share an
    # accession in a synthetic fixture, and the supersession list must not
    # silently collapse when they do.
    counts: Dict[int, int] = {}
    for release in supplier_of.values():
        counts[id(release)] = counts.get(id(release), 0) + 1
    primary = max(ordered, key=lambda r: (counts.get(id(r), 0),
                                          r.filed or "", r.accession or ""))

    combined = GuidanceRelease(
        symbol=primary.symbol,
        fiscal_year=fiscal_year if fiscal_year is not None else primary.fiscal_year,
        accession=primary.accession,
        document=primary.document,
        filed=primary.filed,
        metrics=current_metrics,
        warnings=primary.warnings)
    superseded = [r for r in ordered if r is not primary]
    return combined, superseded


def _is_expired(metric: GuidanceMetric, as_of: Optional[str]) -> bool:
    """A fiscal year that ended before the valuation date is not an outlook.

    Only applied to ANNUAL and MULTI-YEAR periods, and only on the calendar
    year, because the fiscal-year-end date is not knowable from the release
    text. A quarterly outlook is never expired here: the whole point of
    next-quarter guidance is that it is the freshest forward statement there
    is, and it is superseded by the next release rather than by the calendar.
    """
    if not as_of or len(as_of) < 4:
        return False
    if metric.period_type == GuidancePeriodType.QUARTER:
        return False
    try:
        current_year = int(as_of[:4])
    except ValueError:
        return False
    return bool(metric.fiscal_year) and metric.fiscal_year < current_year


# ---------------------------------------------------------------------------
# Section 11 — validation
# ---------------------------------------------------------------------------

REQUIRED_GUIDANCE_FIELDS = ("name", "fiscal_period", "units", "basis", "source_accession")


def validate_guidance_metric(metric: GuidanceMetric) -> List[str]:
    """Section 11's checklist, as a list of problems (empty means valid).

    Every numeric guidance value must carry a source document, an evidence
    id, a metric identity, a fiscal period, units, a low/high (or an exact
    value expressed as low==high) and a GAAP/adjusted/company-defined basis.
    A value failing any of these is not published as guidance.
    """
    problems: List[str] = []
    if not metric.name or metric.name not in _METRIC_BY_NAME:
        problems.append(f"metric identity {metric.name!r} is not in the reviewed taxonomy")
    if not metric.fiscal_period:
        problems.append("no fiscal period")
    if metric.unit not in (GuidanceUnit.RATIO, GuidanceUnit.CURRENCY,
                           GuidanceUnit.CURRENCY_PER_SHARE):
        problems.append(f"unrecognized units {metric.unit!r}")
    if metric.basis not in (BASIS_GAAP, BASIS_ADJUSTED, BASIS_COMPANY_DEFINED, BASIS_NONE):
        problems.append(f"unrecognized basis {metric.basis!r}")
    if metric.low is None or metric.high is None or metric.high < metric.low:
        problems.append("no usable low/high value")
    if not metric.source_accession:
        problems.append("no source filing")
    if not metric.source_evidence_ids:
        problems.append("no evidence id")
    expected_unit = _METRIC_BY_NAME.get(metric.name, (None, None))[1]
    if expected_unit is not None and metric.unit != expected_unit:
        problems.append(f"units {metric.unit!r} do not match the {metric.name!r} taxonomy "
                        f"entry ({expected_unit!r})")
    return problems
