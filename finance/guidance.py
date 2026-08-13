"""Phase H.4 — current management guidance, extracted from SEC-filed material.

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

Precision is bought with a specific structural rule: a guidance figure must
be expressed as a RANGE ("between $3.60 and $3.75", "2% to 3%", "$3.60-3.75").
Companies state guidance as ranges and state actuals as single values, so
requiring two numbers separates the two automatically. It is why, in AOS's
own release, the line

    Diluted EPS (GAAP)  $ 3.60-3.75   $ 3.85

yields the guidance range and silently ignores the $3.85 prior-year actual
sitting next to it. The cost is recall on single-point guidance, which is
accepted deliberately: a miss is visible (guidance simply reads as
unavailable, and the whole workflow is built to keep working that way — see
section 20), whereas a false positive is not.

Each extracted value keeps the exact excerpt it came from, so any number in
the report can be traced back to the sentence in the filing that supports it.
"""

import html as _html
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# The 8-K item that means "Results of Operations and Financial Condition" —
# the earnings release. Guidance in any other item type is not looked for.
EARNINGS_RELEASE_ITEM = "2.02"

# GAAP vs adjusted must never be mixed (section 5). A metric is "adjusted"
# only when the document says so within the matched window.
BASIS_GAAP = "GAAP"
BASIS_ADJUSTED = "adjusted"
BASIS_NONE = "not_specified"

_ADJUSTED_MARKERS = re.compile(r"(?i)\b(adjusted|non-?GAAP|core|underlying)\b")


class GuidanceUnit:
    RATIO = "ratio"                # growth rates, margins: stored as decimals
    CURRENCY = "currency"          # absolute amounts, in the document's scale
    CURRENCY_PER_SHARE = "currency_per_share"


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

    def to_dict(self) -> dict:
        return {
            "name": self.name, "low": self.low, "high": self.high, "unit": self.unit,
            "basis": self.basis, "fiscal_year": self.fiscal_year,
            "evidence_id": self.evidence_id, "source_excerpt": self.source_excerpt,
            "scale": self.scale,
        }

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0


@dataclass(frozen=True)
class GuidanceRelease:
    """Guidance as stated by ONE filing, for ONE fiscal year."""

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
    """Filed HTML -> flat text. No interpretation, just tag removal."""
    if not document:
        return ""
    text = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", document)
    # Table cell/row boundaries carry meaning in an outlook table; keep them
    # as separators so "$3,900 $3,950" does not become "$3,900$3,950".
    text = re.sub(r"(?i)</(td|th|tr|p|div|li|br)\s*>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = _html.unescape(text)
    # Non-breaking spaces and the bullet glyphs EDGAR filings are full of.
    text = text.replace("\xa0", " ").replace("•", " ").replace("◦", " ")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def find_earnings_release_filings(submissions: dict, limit: int = 8) -> List[dict]:
    """Item-2.02 8-K filings, NEWEST FIRST.

    Newest-first ordering is what makes supersession work: guidance from the
    most recent release for a fiscal year replaces every earlier statement of
    guidance for that same year (section 5).
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

# A RANGE, in the spellings real releases use. Requiring two numbers is the
# main precision guard — see the module docstring.
_RANGE_PATTERNS = (
    # "between $3.60 and $3.75", "between 2% and 3%"
    re.compile(rf"(?i)between\s*\$?\s*({_NUM})\s*(%?)\s*(?:and|to)\s*\$?\s*({_NUM})\s*(%?)"),
    # "$3.60 to $3.75", "2% to 3%", "a range of 2% to 3%"
    re.compile(rf"(?i)\$?\s*({_NUM})\s*(%?)\s*to\s*\$?\s*({_NUM})\s*(%?)"),
    # "$ 3.60-3.75", "2%-3%"
    re.compile(rf"(?i)\$?\s*({_NUM})\s*(%?)\s*-\s*\$?\s*({_NUM})\s*(%?)"),
)

# Metric keyword -> (canonical name, unit, whether a percent sign is required).
# Only metrics this project actually consumes are listed; an unrecognized
# guidance line is ignored rather than stored under a guessed name.
_METRIC_PATTERNS = (
    ("revenue_growth", GuidanceUnit.RATIO, True,
     re.compile(r"(?i)\b(?:net\s+)?sales\s+growth\b|\brevenue\s+growth\b|"
                r"\bgrowth\s+in\s+(?:net\s+)?sales\b")),
    ("earnings_per_share", GuidanceUnit.CURRENCY_PER_SHARE, False,
     re.compile(r"(?i)\bdiluted\s+(?:earnings\s+per\s+share|EPS)\b|"
                r"(?<!adjusted\s)\bEPS\s*\(GAAP\)")),
    ("adjusted_earnings_per_share", GuidanceUnit.CURRENCY_PER_SHARE, False,
     re.compile(r"(?i)\badjusted\s+(?:diluted\s+)?(?:earnings\s+per\s+share|EPS)\b")),
    ("revenue", GuidanceUnit.CURRENCY, False,
     re.compile(r"(?i)\bnet\s+sales\b|\btotal\s+revenue[s]?\b")),
    ("operating_margin", GuidanceUnit.RATIO, True,
     re.compile(r"(?i)\boperating\s+margin\b")),
    ("capital_expenditure", GuidanceUnit.CURRENCY, False,
     re.compile(r"(?i)\bcapital\s+expenditures?\b|\bcapex\b")),
    ("free_cash_flow", GuidanceUnit.CURRENCY, False,
     re.compile(r"(?i)\bfree\s+cash\s+flow\b")),
    ("operating_cash_flow", GuidanceUnit.CURRENCY, False,
     re.compile(r"(?i)\bcash\s+(?:provided\s+by|from)\s+operating\s+activities\b")),
)

# How far after a metric keyword a range may appear and still be that
# metric's guidance. Tight on purpose: the further away a number is, the more
# likely it belongs to a different line of the release. Reducing this from
# 160 to 110 fixed a real false positive on AOS's January release, where a
# flattened reconciliation table put "Free cash flow (non-GAAP) $546.0
# $473.8" within reach of the EPS guidance range printed underneath it, and
# $3.85-$4.15 of earnings per share was extracted as free-cash-flow guidance.
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
_YEAR_RE = re.compile(r"\b(20\d{2})\b")

# A range only counts as GUIDANCE when the surrounding text says it is
# forward-looking. Without this, a prior-year comparison range would qualify.
_FORWARD_MARKERS = re.compile(
    r"(?i)\b(outlook|guidance|expect\w*|anticipat\w*|forecast\w*|target\w*|"
    r"project\w*|estimat\w*|reaffirm\w*|narrow\w*|rais\w*|lower\w*|updat\w*)\b")


def _to_number(text: str) -> Optional[float]:
    try:
        return float(text.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _other_metric_keyword_before(window: str, position: int, own_name: str) -> bool:
    """Does a DIFFERENT metric's keyword sit between the keyword and the range?

    Filed releases are laid out as tables, and flattening a table puts
    unrelated lines next to each other. If another metric is named in
    between, the range belongs to that metric, not to this one — this is what
    stops "Free cash flow ... $3.85-$4.15" (an EPS row printed below a cash
    row) from being read as free-cash-flow guidance.
    """
    prefix = window[:position]
    for name, _unit, _pct, keyword_re in _METRIC_PATTERNS:
        if name == own_name:
            continue
        if keyword_re.search(prefix):
            return True
    return False


def _find_range(window: str, require_percent: bool, own_name: str
                ) -> Optional[Tuple[float, float, bool, str]]:
    """(low, high, was_percent, matched_text) for the first usable range."""
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
            if _other_metric_keyword_before(window, match.start(), own_name):
                continue
            return low, high, is_percent, match.group(0).strip()
    return None


def extract_guidance_from_text(text: str, symbol: str, accession: str, document: str,
                               filed: str, expected_fiscal_year: Optional[int] = None
                               ) -> GuidanceRelease:
    """Pattern-match guidance out of one earnings release. No interpretation.

    `expected_fiscal_year` is enforced, not inferred-and-trusted: a range whose
    surrounding text names a DIFFERENT year is rejected outright (section 5's
    "wrong fiscal year rejected"), because a release routinely discusses the
    prior year's actuals alongside the current year's outlook.
    """
    metrics: Dict[str, GuidanceMetric] = {}
    warnings: List[str] = []
    if not text:
        return GuidanceRelease(symbol=symbol, fiscal_year=expected_fiscal_year,
                               accession=accession, document=document, filed=filed,
                               warnings=("The filing document was empty.",))

    for name, unit, require_percent, keyword_re in _METRIC_PATTERNS:
        if name in metrics:
            continue
        for keyword in keyword_re.finditer(text):
            window = text[keyword.end():keyword.end() + _WINDOW]
            context = text[max(0, keyword.start() - _WINDOW):keyword.end() + _WINDOW]

            if not _FORWARD_MARKERS.search(context):
                continue

            found = _find_range(window, require_percent, name)
            if found is None:
                continue
            low, high, is_percent, matched = found

            years = {int(y) for y in _YEAR_RE.findall(context)}
            if expected_fiscal_year is not None:
                if years and expected_fiscal_year not in years:
                    warnings.append(
                        f"Ignored a {name} range ({matched}) because the surrounding text names "
                        f"{sorted(years)} rather than fiscal {expected_fiscal_year}.")
                    continue
                if not years:
                    warnings.append(
                        f"Ignored a {name} range ({matched}) because no fiscal year could be "
                        "identified near it.")
                    continue

            # GAAP vs adjusted is decided from the modifier immediately before
            # the metric name, never from the wider paragraph — see
            # `_BASIS_LOOKBEHIND`.
            modifier = text[max(0, keyword.start() - _BASIS_LOOKBEHIND):keyword.start()]
            basis = BASIS_ADJUSTED if _ADJUSTED_MARKERS.search(modifier) else BASIS_GAAP
            if name == "adjusted_earnings_per_share":
                basis = BASIS_ADJUSTED
            elif name == "earnings_per_share" and basis == BASIS_ADJUSTED:
                # "adjusted" directly modifies this EPS mention, so it is the
                # adjusted measure; storing it under the GAAP name would mix
                # the two bases (section 5).
                continue

            scale_match = _SCALE_RE.search(context)
            if is_percent:
                low, high = low / 100.0, high / 100.0

            metrics[name] = GuidanceMetric(
                name=name, low=low, high=high, unit=unit, basis=basis,
                fiscal_year=expected_fiscal_year or (max(years) if years else 0),
                evidence_id=f"dcf.guidance.{name}.current",
                source_excerpt=_excerpt(text, keyword.start(), keyword.end() + _WINDOW),
                scale=(scale_match.group(1).lower() if scale_match and
                       unit == GuidanceUnit.CURRENCY else None))
            break

    return GuidanceRelease(symbol=symbol, fiscal_year=expected_fiscal_year,
                           accession=accession, document=document, filed=filed,
                           metrics=metrics, warnings=tuple(warnings))


def _excerpt(text: str, start: int, end: int, pad: int = 40) -> str:
    lo = max(0, start - pad)
    hi = min(len(text), end)
    return text[lo:hi].strip()


def select_current_guidance(releases: Sequence[GuidanceRelease], fiscal_year: int
                            ) -> Tuple[Optional[GuidanceRelease], List[GuidanceRelease]]:
    """The newest valid release for `fiscal_year`, plus the ones it supersedes.

    Superseded guidance is RETAINED and returned (section 5: "preserve prior
    guidance for comparison") but is never the current guidance — an outlook
    that has since been narrowed, raised or lowered is exactly the kind of
    stale forward evidence that would distort a valuation.
    """
    matching = [r for r in releases if r.fiscal_year == fiscal_year and r.metrics]
    if not matching:
        return None, []
    matching = sorted(matching, key=lambda r: (r.filed or "", r.accession or ""), reverse=True)
    return matching[0], matching[1:]
