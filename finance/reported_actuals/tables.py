"""A filed document's tables, with their rows and columns still meaning something.

WHY A GRID AND NOT FLAT TEXT

`finance/guidance.py::html_to_text` flattens a filing into one line, which is
right for a sentence reader and destroys a financial statement. Run over a
condensed income statement it produces

    Net revenue $ 29,591 $ 22,187 $ 15,952 $ 71,089 $ 45,872

and every one of those five numbers is a different period: three fiscal
quarters and two year-to-date columns, two of them from the PRIOR YEAR. Read
positionally, the second number is last quarter's revenue reported as this
quarter's; read greedily, the fourth is nine months of revenue reported as
three. Both are ordinary-looking wrong answers.

So this module keeps the grid. EDGAR tables are built from `colspan`-padded
cells with the currency symbol in its own column and spacer cells between the
data columns, so a cell's meaning is its GRID POSITION, not its index among
its siblings. A value belongs to the period column whose grid span contains
it, and a value that lands in no period column -- a "Change" column, a
footnote marker -- is not a fact.

WHAT A COLUMN'S PERIOD IS

Two rows of header, usually:

    band        Fiscal Quarter Ended        Three Fiscal Quarters Ended
    columns     Aug 2 2026  May 3 2026  Aug 3 2025    Aug 2 2026  Aug 3 2025

The band gives the DURATION, the column gives the END. Both are needed and
neither is enough: "August 2, 2026" appears twice above, once meaning a
quarter and once meaning nine months, and a reader that took only the date
would treat them as the same period. That is exactly the YTD-as-quarter
failure, and it is a header-alignment question rather than a semantic one.

"Three Fiscal Quarters Ended" IS NINE MONTHS. Counting quarters and counting
months are two different spellings of a duration and the words look alike;
this module keeps two separate vocabularies so "three quarters" can never be
read as "three months".

A table with no duration band is a BALANCE SHEET: its columns are instants.

FAIL CLOSED

A row whose values cannot be placed in period columns is skipped and recorded.
A header that resolves no period produces no columns and the table yields
nothing. An unresolvable scale or currency marks the whole table unusable
rather than letting a figure through in unknown units -- "23,975" is a very
different fact in thousands, millions and billions, and there is no way to
tell from the digits.
"""

import datetime
import html as _html
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from finance import semantics as sem


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

class StatementKind:
    """Which statement a table is, decided from its caption.

    The caption is the issuer's own title for the table and is the only place
    that distinguishes a condensed income statement from the GAAP-to-non-GAAP
    reconciliation that follows it -- both carry a row called "operating
    income", for the same period, with different numbers.
    """

    INCOME_STATEMENT = "INCOME_STATEMENT"
    BALANCE_SHEET = "BALANCE_SHEET"
    CASH_FLOW = "CASH_FLOW"
    NON_GAAP_RECONCILIATION = "NON_GAAP_RECONCILIATION"
    SEGMENT = "SEGMENT"
    SHARE_COUNT = "SHARE_COUNT"
    GUIDANCE = "GUIDANCE"
    OTHER = "OTHER"

    # The kinds a canonical GAAP reported fact may be taken from.
    REPORTED_STATEMENTS = (INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW)


class TableRejection:
    """Why a table, or a row in one, produced nothing. Structured, so a
    benchmark counts classes rather than reading sentences."""

    NO_PERIOD_HEADER = "NO_PERIOD_HEADER"
    UNRESOLVED_PERIOD = "UNRESOLVED_PERIOD"
    UNKNOWN_SCALE = "UNKNOWN_SCALE"
    UNKNOWN_CURRENCY = "UNKNOWN_CURRENCY"
    AMBIGUOUS_CURRENCY = "AMBIGUOUS_CURRENCY"
    VALUES_DO_NOT_ALIGN = "VALUES_DO_NOT_ALIGN"
    FORWARD_LOOKING_TABLE = "FORWARD_LOOKING_TABLE"
    NO_ROWS = "NO_ROWS"


# Scale words, and what they multiply a printed figure by. A missing scale is
# NOT 1: an issuer that prints "23,975" without saying so anywhere is a table
# this layer refuses, because the alternative is a thousand-fold error that
# looks like a plausible number.
SCALE_FACTORS = {
    "thousands": 1_000.0,
    "millions": 1_000_000.0,
    "billions": 1_000_000_000.0,
    "units": 1.0,
}

_SCALE_PHRASE = re.compile(
    r"(?i)\bin\s+(thousand|million|billion)s?\b"
    r"|\((?:[^)]*?\b)?(thousand|million|billion)s?\b[^)]*\)"
    r"|\bamounts?\s+in\s+(thousand|million|billion)s?\b")

# "except per share" tells us the per-share rows are NOT scaled. Recorded so a
# per-share row is never multiplied by a million.
_EXCEPT_PER_SHARE = re.compile(r"(?i)except\s+(?:for\s+)?per[\s-]share|except\s+per\s+share")

# Currency, from the symbol or the code. Deliberately small and explicit:
# guessing a currency is the same class of error as guessing a scale.
_CURRENCY_SYMBOLS = {"$": "USD", "US$": "USD", "€": "EUR", "£": "GBP",
                     "¥": "JPY", "R$": "BRL", "CHF": "CHF"}
_CURRENCY_CODE = re.compile(
    r"(?i)\bin\s+(?:millions|billions|thousands)\s+of\s+(USD|EUR|GBP|JPY|CHF|CAD|AUD|SEK|DKK|NOK|BRL|INR|CNY|KRW|TWD)\b"
    r"|\b(USD|EUR|GBP|JPY|CHF|CAD|AUD|SEK|DKK|NOK|BRL|INR|CNY|KRW|TWD)\s+(?:millions|billions|thousands)\b"
    r"|\((?:in\s+)?(?:millions|billions|thousands)\s+of\s+(USD|EUR|GBP|JPY|CHF|CAD|AUD|SEK|DKK|NOK|BRL|INR|CNY|KRW|TWD)\)")

# US dollars stated in words, which several foreign private issuers do because
# their reporting currency is NOT obvious from a bare dollar sign.
_US_DOLLAR_PHRASE = re.compile(r"(?i)\bU\.?S\.?\s*dollars?\b|\bin\s+U\.?S\.?\$")


# -- statement captions -----------------------------------------------------
#
# Matched against the caption ONLY. A row label never decides what statement
# it is in: the non-GAAP reconciliation's rows are spelled exactly like the
# income statement's, which is the entire reason it needs its own caption.

_CAPTION_PATTERNS = (
    # Non-GAAP first: "RECONCILIATION OF GAAP TO NON-GAAP OPERATING INCOME"
    # contains the word "income" and would otherwise read as an income
    # statement. The more specific claim wins.
    (StatementKind.NON_GAAP_RECONCILIATION, re.compile(
        r"(?i)non[\s-]?gaap|reconciliation|adjusted\s+(?:results|measures|earnings)"
        r"|supplemental\s+(?:financial\s+)?(?:information|measures)")),
    (StatementKind.GUIDANCE, re.compile(
        r"(?i)\b(?:outlook|guidance|forecast|projected|business\s+outlook)\b")),
    (StatementKind.SEGMENT, re.compile(
        r"(?i)\bby\s+segment\b|\bsegment\s+(?:results|information|data)\b"
        r"|\brevenue\s+by\b|\bdisaggregat")),
    (StatementKind.SHARE_COUNT, re.compile(
        r"(?i)weighted[\s-]average\s+(?:number\s+of\s+)?shares"
        r"|shares\s+used\s+in\s+comput")),
    (StatementKind.CASH_FLOW, re.compile(
        r"(?i)statements?\s+of\s+cash\s+flows?|cash\s+flow\s+statements?")),
    (StatementKind.BALANCE_SHEET, re.compile(
        r"(?i)balance\s+sheets?|statements?\s+of\s+financial\s+position")),
    (StatementKind.INCOME_STATEMENT, re.compile(
        r"(?i)statements?\s+of\s+(?:operations|income|earnings|profit\s+or\s+loss"
        r"|comprehensive\s+income)|income\s+statements?")),
)


# -- period bands -----------------------------------------------------------
#
# TWO VOCABULARIES, KEPT APART ON PURPOSE. "Three months ended" is a quarter
# and "three fiscal quarters ended" is nine months; they differ by one word
# and by a factor of three. A single pattern with an optional "fiscal
# quarters|months" alternation would read one as the other on the day someone
# reorders the group.

_MONTHS_BAND = re.compile(
    r"(?i)\b(one|two|three|four|five|six|nine|ten|twelve|1|2|3|4|6|9|12)\s*"
    r"(?:-|\s)?months?\s+(?:ended|ending|period)")

_QUARTERS_BAND = re.compile(
    r"(?i)\b(one|two|three|four|first|second|third|fourth|1|2|3|4)\s*"
    r"(?:-|\s)?(?:fiscal\s+)?quarters?\s+(?:ended|ending)")

_SINGLE_QUARTER_BAND = re.compile(
    r"(?i)\b(?:fiscal\s+)?quarter(?:ly)?\s+(?:ended|ending)\b"
    r"|\bfor\s+the\s+quarter\b")

_YEAR_BAND = re.compile(
    r"(?i)\b(?:fiscal\s+)?years?\s+(?:ended|ending)\b"
    r"|\btwelve\s+months?\s+(?:ended|ending)\b"
    r"|\bfull[\s-]year\s+(?:ended|ending)\b"
    r"|\bannual\s+period\b")

_HALF_BAND = re.compile(r"(?i)\b(?:six\s+months?|half[\s-]year|first\s+half)\s+(?:ended|ending)\b")

_WORD_COUNTS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "nine": 9, "ten": 10, "twelve": 12, "first": 1, "second": 2,
                "third": 3, "fourth": 4,
                "1": 1, "2": 2, "3": 3, "4": 4, "6": 6, "9": 9, "10": 10, "12": 12}

# Months to the domain's own frequency vocabulary. Anything not here is
# UNKNOWN and therefore unusable, rather than rounded to the nearest.
_MONTHS_TO_FREQUENCY = {
    3: sem.PeriodFrequency.QUARTER,
    6: sem.PeriodFrequency.YTD_6M,
    9: sem.PeriodFrequency.YTD_9M,
    12: sem.PeriodFrequency.ANNUAL,
}

# A band that says the columns are point-in-time.
_INSTANT_BAND = re.compile(r"(?i)\bas\s+(?:of|at)\b|\bbalance\s+at\b")

# Vocabulary that makes a table an OUTLOOK rather than a report. Section 6's
# hard boundary, enforced structurally: a table under a forward caption is not
# a source of reported actuals whatever its rows are called.
_FORWARD_TABLE = re.compile(
    r"(?i)\b(?:outlook|guidance|expect\w*|forecast\w*|project(?:ed|ion)s?"
    r"|estimat\w*|target\w*|anticipat\w*)\b")

# Third-party numbers. A consensus table sitting beside the results is not the
# company's report of what happened.
_ANALYST_TABLE = re.compile(
    r"(?i)\banalysts?\b|\bconsensus\b|\bstreet\s+estimate|\bsurveyed\s+by\b"
    r"|\brefinitiv\b|\bfactset\b|\bvisible\s+alpha\b")


# -- dates ------------------------------------------------------------------

_MONTH_NAMES = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_MONTH_DAY_YEAR = re.compile(
    r"(?i)\b(" + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) +
    r")\.?\s+(\d{1,2})\s*,?\s*(\d{4})\b")
_MONTH_DAY = re.compile(
    r"(?i)\b(" + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) +
    r")\.?\s+(\d{1,2})\s*,\s*$")
_DAY_MONTH_YEAR = re.compile(
    r"(?i)\b(\d{1,2})\s+(" + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) +
    r")\.?\s+(\d{4})\b")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_YEAR_ONLY = re.compile(r"^\s*(?:FY\s*)?(19|20)(\d{2})\s*$", re.IGNORECASE)


def _parse_date(text: str) -> Optional[str]:
    """A column or band label to an ISO date, or None. Never a guess."""
    if not text:
        return None
    match = _ISO_DATE.search(text)
    if match:
        return _valid(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = _MONTH_DAY_YEAR.search(text)
    if match:
        return _valid(int(match.group(3)), _MONTH_NAMES[match.group(1).lower()],
                      int(match.group(2)))
    match = _DAY_MONTH_YEAR.search(text)
    if match:
        return _valid(int(match.group(3)), _MONTH_NAMES[match.group(2).lower()],
                      int(match.group(1)))
    return None


def _valid(year: int, month: int, day: int) -> Optional[str]:
    try:
        return datetime.date(year, month, day).isoformat()
    except ValueError:
        return None


def _month_day_without_year(text: str) -> Optional[Tuple[int, int]]:
    """"June 30," from a band whose columns carry only the year."""
    match = _MONTH_DAY.search((text or "").strip())
    if match:
        return _MONTH_NAMES[match.group(1).lower()], int(match.group(2))
    return None


def band_frequency(label: str) -> str:
    """A header band to the domain's period frequency.

    Order matters and is deliberate. Quarters are counted before months
    because "Three Fiscal Quarters Ended" also contains no month word but
    would fall through to the single-quarter pattern; and the year band is
    tried before the months band so "Twelve Months Ended" resolves once.
    """
    text = " ".join((label or "").split())
    if not text:
        return sem.PeriodFrequency.UNKNOWN
    if _INSTANT_BAND.search(text):
        return sem.PeriodFrequency.INSTANT
    match = _QUARTERS_BAND.search(text)
    if match:
        count = _WORD_COUNTS.get(match.group(1).lower())
        return _MONTHS_TO_FREQUENCY.get((count or 0) * 3, sem.PeriodFrequency.UNKNOWN)
    if _YEAR_BAND.search(text):
        return sem.PeriodFrequency.ANNUAL
    match = _MONTHS_BAND.search(text)
    if match:
        count = _WORD_COUNTS.get(match.group(1).lower())
        return _MONTHS_TO_FREQUENCY.get(count or 0, sem.PeriodFrequency.UNKNOWN)
    if _HALF_BAND.search(text):
        return sem.PeriodFrequency.YTD_6M
    if _SINGLE_QUARTER_BAND.search(text):
        return sem.PeriodFrequency.QUARTER
    return sem.PeriodFrequency.UNKNOWN


# ---------------------------------------------------------------------------
# HTML to a grid
# ---------------------------------------------------------------------------

_CELL = re.compile(r"(?is)<t([dh])([^>]*)>(.*?)</t\1\s*>")
_ROW = re.compile(r"(?is)<tr[^>]*>(.*?)</tr>")
_COLSPAN = re.compile(r"(?i)colspan\s*=\s*[\"']?(\d+)")
_TABLE_OPEN = re.compile(r"(?i)<table\b[^>]*>")
_TABLE_CLOSE = re.compile(r"(?i)</table\s*>")


def _clean(fragment: str) -> str:
    text = re.sub(r"(?s)<[^>]+>", " ", fragment or "")
    text = _html.unescape(text)
    text = (text.replace("\xa0", " ").replace("—", "-").replace("–", "-")
            .replace("’", "'").replace("�", " "))
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class GridCell:
    text: str
    start: int          # inclusive grid column
    end: int            # exclusive

    @property
    def empty(self) -> bool:
        return not self.text


def iter_table_spans(document: str) -> List[Tuple[int, int]]:
    """(start, end) of every TOP-LEVEL table, nesting handled.

    A non-greedy `<table>.*?</table>` closes an outer table on an inner
    table's closing tag, which silently truncates the rows that matter --
    EDGAR filings nest tables for layout routinely.
    """
    spans: List[Tuple[int, int]] = []
    position, depth, start = 0, 0, None
    while position < len(document):
        opened = _TABLE_OPEN.search(document, position)
        closed = _TABLE_CLOSE.search(document, position)
        if closed is None:
            break
        if opened is not None and opened.start() < closed.start():
            if depth == 0:
                start = opened.start()
            depth += 1
            position = opened.end()
            continue
        depth = max(depth - 1, 0)
        position = closed.end()
        if depth == 0 and start is not None:
            spans.append((start, closed.end()))
            start = None
    return spans


def table_grid(table_html: str) -> List[List[GridCell]]:
    """Rows of cells, each carrying the grid columns it occupies."""
    grid: List[List[GridCell]] = []
    for row_html in _ROW.findall(table_html):
        cells: List[GridCell] = []
        column = 0
        for _tag, attributes, body in _CELL.findall(row_html):
            match = _COLSPAN.search(attributes)
            span = max(int(match.group(1)), 1) if match else 1
            cells.append(GridCell(_clean(body), column, column + span))
            column += span
        grid.append(cells)
    return grid


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"^\(?\s*-?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*\)?$")
_PERCENT_CELL = re.compile(r"%|\bpercent\b", re.IGNORECASE)
_CURRENCY_ONLY = re.compile(r"^[\$€£¥]$|^R\$$|^US\$$")


@dataclass(frozen=True)
class CellValue:
    """One numeric cell, before it is attached to anything."""

    raw: str
    number: float
    is_percent: bool
    negative: bool
    start: int
    end: int


def _cell_value(cell: GridCell) -> Optional[CellValue]:
    text = cell.text.strip()
    if not text or _CURRENCY_ONLY.match(text):
        return None
    percent = bool(_PERCENT_CELL.search(text))
    stripped = _PERCENT_CELL.sub("", text).strip()
    match = _NUMBER.match(stripped)
    if not match:
        return None
    negative = stripped.startswith("(") or stripped.lstrip("$( ").startswith("-")
    value = float(match.group(1).replace(",", ""))
    return CellValue(raw=text, number=-value if negative else value,
                     is_percent=percent, negative=negative,
                     start=cell.start, end=cell.end)


# ---------------------------------------------------------------------------
# The parsed table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PeriodColumn:
    """One column, and the period it covers."""

    label: str
    band_label: str
    period_end: Optional[str]
    frequency: str
    start: int
    end: int

    @property
    def usable(self) -> bool:
        return bool(self.period_end) and self.frequency != sem.PeriodFrequency.UNKNOWN

    def to_dict(self) -> dict:
        return {"label": self.label, "band_label": self.band_label,
                "period_end": self.period_end, "frequency": self.frequency}


@dataclass(frozen=True)
class TableRow:
    """One line of a statement: a label, and the values under each column."""

    label: str
    values: Dict[int, CellValue]        # period-column index -> value
    unplaced: Tuple[str, ...] = ()      # numbers that fell in no period column
    row_index: int = 0

    def to_dict(self) -> dict:
        return {"label": self.label,
                "values": {str(k): v.number for k, v in sorted(self.values.items())},
                "unplaced": list(self.unplaced)}


@dataclass
class FilingTable:
    """One table, read structurally. Nothing here is interpreted financially."""

    index: int
    caption: str
    kind: str
    period_columns: List[PeriodColumn] = field(default_factory=list)
    rows: List[TableRow] = field(default_factory=list)
    scale: Optional[str] = None
    scale_factor: Optional[float] = None
    per_share_unscaled: bool = False
    currency: Optional[str] = None
    rejections: List[Tuple[str, str]] = field(default_factory=list)   # (code, detail)
    start: int = 0

    @property
    def usable(self) -> bool:
        return bool(self.period_columns) and bool(self.rows) \
            and self.scale_factor is not None and bool(self.currency)

    def reject(self, code: str, detail: str = "") -> None:
        self.rejections.append((code, detail))

    def to_dict(self) -> dict:
        return {"index": self.index, "caption": self.caption, "kind": self.kind,
                "period_columns": [c.to_dict() for c in self.period_columns],
                "row_count": len(self.rows), "scale": self.scale,
                "currency": self.currency, "usable": self.usable,
                "rejections": [{"code": c, "detail": d} for c, d in self.rejections]}


# How far back of the document to read looking for a table's caption, and how
# many BLOCKS of it to accept.
#
# Walking up by blocks rather than by characters is what stops the RELEASE
# HEADLINE becoming a table's caption. "Example Corp Reports Fourth Quarter
# Results and Provides Outlook" sits a few hundred characters above the
# condensed income statement, contains the word "Outlook", and a fixed
# character window swept it in -- classifying the quarter's reported results
# as an outlook table and discarding every figure in it.
#
# So the walk stops as soon as the text collected NAMES A STATEMENT. The
# nearest block that says what the table is, is the caption; anything further
# up is the document around it.
# The window is generous in HTML CHARACTERS and tight in BLOCKS, because a
# filed exhibit's markup is mostly styling. Measured on a live filing: the
# words "CONDENSED CONSOLIDATED BALANCE SHEETS" sat 718 characters above their
# table and two text-blocks above it, behind inline styles. A 700-character
# window missed the title by eighteen characters, classified the balance sheet
# as an unidentified table, and discarded every figure in it -- while a reader
# looking at the page sees the title directly above the numbers.
#
# What actually bounds the search is the PREVIOUS TABLE. Text before it
# belongs to that table, so the walk stops there and can never absorb the
# preceding statement's caption however much room is left in the window.
_CAPTION_WINDOW = 8000
_CAPTION_BLOCKS = 6

_BLOCK_BOUNDARY = re.compile(r"(?i)</(?:p|div|td|tr|table|h[1-6]|li)\s*>|<br\s*/?>")


def preceding_caption(document: str, start: int, floor: int = 0) -> str:
    """The caption block(s) immediately above a table.

    Walks UP block by block and stops as soon as the text collected names a
    statement, so a release headline further up can never become a table's
    caption -- the failure that made "Reports Fourth Quarter Results and
    Provides Outlook" classify a condensed income statement as an outlook.
    """
    begin = max(floor, start - _CAPTION_WINDOW, 0)
    window = document[begin:start]
    blocks = [_clean(part) for part in _BLOCK_BOUNDARY.split(window)]
    blocks = [b for b in blocks if b]
    caption = ""
    for block in reversed(blocks[-_CAPTION_BLOCKS:]):
        caption = f"{block} {caption}".strip()
        if classify_caption(caption) != StatementKind.OTHER:
            break
    return caption


def classify_caption(caption: str) -> str:
    for kind, pattern in _CAPTION_PATTERNS:
        if pattern.search(caption or ""):
            return kind
    return StatementKind.OTHER


def _resolve_scale(text: str) -> Tuple[Optional[str], Optional[float]]:
    match = _SCALE_PHRASE.search(text or "")
    if not match:
        return None, None
    word = next((g for g in match.groups() if g), None)
    if not word:
        return None, None
    name = word.lower() + "s"
    return name, SCALE_FACTORS.get(name)


def _resolve_currency(text: str, cells_text: str) -> Tuple[Optional[str], Optional[str]]:
    """(currency, rejection code). A bare dollar sign is USD only when nothing
    contradicts it; a named non-USD currency always wins over a symbol."""
    haystack = f"{text} {cells_text}"
    codes = {g for match in _CURRENCY_CODE.finditer(haystack)
             for g in match.groups() if g}
    if len(codes) > 1:
        return None, TableRejection.AMBIGUOUS_CURRENCY
    if codes:
        return codes.pop().upper(), None
    symbols = {name for symbol, name in _CURRENCY_SYMBOLS.items()
               if symbol in haystack and symbol not in ("$",)}
    if len(symbols) > 1:
        return None, TableRejection.AMBIGUOUS_CURRENCY
    if symbols:
        only = symbols.pop()
        # A euro sign beside a dollar sign is two currencies in one table.
        if "$" in haystack and only != "USD":
            return None, TableRejection.AMBIGUOUS_CURRENCY
        return only, None
    if "$" in haystack or _US_DOLLAR_PHRASE.search(haystack):
        return "USD", None
    return None, TableRejection.UNKNOWN_CURRENCY


def _header_rows(grid: List[List[GridCell]]) -> Tuple[int, int, List[PeriodColumn]]:
    """(band row index, column row index, columns). The first alignment that
    resolves at least one period wins."""
    best: Tuple[int, int, List[PeriodColumn]] = (-1, -1, [])
    for index, cells in enumerate(grid):
        labelled = [c for c in cells if c.text]
        if not labelled:
            continue
        columns = _columns_from(grid, index)
        if len(columns) > len(best[2]):
            best = (index - 1, index, columns)
        # A full header is usually within the first rows; keep scanning only
        # while nothing has resolved, so a data row further down whose cells
        # happen to parse as dates cannot displace a real header.
        if best[2] and index > best[1] + 2:
            break
    return best


def _columns_from(grid: List[List[GridCell]], row_index: int) -> List[PeriodColumn]:
    """Period columns for one candidate header row, using the bands above it."""
    cells = [c for c in grid[row_index] if c.text]
    if not cells:
        return []

    bands = _bands_above(grid, row_index)
    columns: List[PeriodColumn] = []
    for cell in cells:
        band_label, band_frequency_name = _band_for(bands, cell)
        period_end = _parse_date(cell.text)
        frequency = band_frequency_name

        if period_end is None:
            year = _YEAR_ONLY.match(cell.text)
            month_day = _month_day_without_year(band_label)
            if year and month_day:
                # "Three Months Ended June 30," over columns "2026  2025".
                period_end = _valid(int(year.group(1) + year.group(2)),
                                    month_day[0], month_day[1])
            elif year:
                # A bare year with no month anywhere is not a period end. A
                # fiscal year does not necessarily end on 31 December and
                # assuming it does is how a September filer's year moves by a
                # quarter.
                period_end = None

        if period_end is None:
            continue
        if frequency == sem.PeriodFrequency.UNKNOWN:
            # No band at all: the columns are dated but undurated, which is
            # what a balance sheet looks like.
            frequency = (sem.PeriodFrequency.INSTANT if not band_label
                         else sem.PeriodFrequency.UNKNOWN)
        columns.append(PeriodColumn(
            label=cell.text, band_label=band_label, period_end=period_end,
            frequency=frequency, start=cell.start, end=cell.end))
    return [c for c in columns if c.usable]


def _bands_above(grid: List[List[GridCell]], row_index: int) -> List[GridCell]:
    """Labelled cells from the rows above that carry a duration phrase."""
    bands: List[GridCell] = []
    for offset in (1, 2):
        above = row_index - offset
        if above < 0:
            break
        for cell in grid[above]:
            if cell.text and band_frequency(cell.text) != sem.PeriodFrequency.UNKNOWN:
                bands.append(cell)
        if bands:
            break
    return bands


def _band_for(bands: List[GridCell], cell: GridCell) -> Tuple[str, str]:
    """The band covering a column, by grid overlap. ("", UNKNOWN) when none."""
    for band in bands:
        if band.start <= cell.start < band.end:
            return band.text, band_frequency(band.text)
    if len(bands) == 1 and bands[0].end - bands[0].start <= 1:
        # A band cell that was not padded to span its columns. One band and
        # one duration is unambiguous; two would not be, and are not accepted.
        return bands[0].text, band_frequency(bands[0].text)
    return "", sem.PeriodFrequency.UNKNOWN


def _rows_from(grid: List[List[GridCell]], header_index: int,
               columns: List[PeriodColumn]) -> List[TableRow]:
    rows: List[TableRow] = []
    for index in range(header_index + 1, len(grid)):
        cells = grid[index]
        labelled = [c for c in cells if c.text]
        if not labelled:
            continue
        label = labelled[0].text
        if _cell_value(labelled[0]) is not None:
            # A row that opens with a number has no label, so nothing
            # identifies what it is. Not a fact.
            continue

        values: Dict[int, CellValue] = {}
        unplaced: List[str] = []
        for cell in cells[1:]:
            value = _cell_value(cell)
            if value is None:
                continue
            placed = None
            for position, column in enumerate(columns):
                if column.start <= value.start < column.end:
                    placed = position
                    break
            if placed is None:
                unplaced.append(value.raw)
            elif placed not in values:
                values[placed] = value
        if values:
            rows.append(TableRow(label=label, values=values,
                                 unplaced=tuple(unplaced), row_index=index))
    return rows


def parse_filing_tables(document: str,
                        document_scale_hint: Optional[str] = None
                        ) -> List[FilingTable]:
    """Every table in a filed document, with its grid, columns and units.

    `document_scale_hint` is the scale stated once at the top of a release and
    not repeated per table. It is a HINT and never overrides a scale the table
    states itself.
    """
    if not document:
        return []

    tables: List[FilingTable] = []
    spans = iter_table_spans(document)
    for index, (start, end) in enumerate(spans):
        table_html = document[start:end]
        floor = spans[index - 1][1] if index else 0
        preceding = preceding_caption(document, start, floor=floor)
        grid = table_grid(table_html)
        band_index, header_index, columns = _header_rows(grid)

        # The caption is the text BEFORE the first data row: what precedes the
        # table, plus any title rows inside it above the header. Data rows are
        # deliberately excluded -- a row labelled "Changes in expected credit
        # losses" would otherwise make a condensed income statement read as an
        # outlook table, and a row naming a European subsidiary would make a
        # dollar-reporting issuer look ambiguous about its currency.
        title_limit = header_index if header_index >= 0 else min(len(grid), 4)
        inner_text = " ".join(c.text for row in grid[:max(title_limit, 0)]
                              for c in row if c.text)
        caption = " ".join(f"{preceding} {inner_text}".split())

        parsed = FilingTable(index=index, caption=caption,
                             kind=classify_caption(caption), start=start)

        # Units first: a table whose units cannot be established yields
        # nothing, so there is no point resolving its periods.
        scale, factor = _resolve_scale(caption)
        if factor is None and document_scale_hint:
            scale, factor = document_scale_hint, SCALE_FACTORS.get(document_scale_hint)
        parsed.scale, parsed.scale_factor = scale, factor
        parsed.per_share_unscaled = bool(_EXCEPT_PER_SHARE.search(caption))
        if factor is None:
            parsed.reject(TableRejection.UNKNOWN_SCALE, caption[:120])

        all_text = " ".join(c.text for row in grid for c in row if c.text)
        currency, currency_code = _resolve_currency(caption, all_text[:4000])
        parsed.currency = currency
        if currency_code:
            parsed.reject(currency_code, caption[:120])

        parsed.period_columns = columns
        if not columns:
            parsed.reject(TableRejection.NO_PERIOD_HEADER, caption[:120])
            tables.append(parsed)
            continue

        parsed.rows = _rows_from(grid, header_index, columns)
        if not parsed.rows:
            parsed.reject(TableRejection.NO_ROWS, caption[:120])

        if _FORWARD_TABLE.search(caption):
            parsed.kind = StatementKind.GUIDANCE
            parsed.reject(TableRejection.FORWARD_LOOKING_TABLE, caption[:120])
        if _ANALYST_TABLE.search(caption):
            parsed.kind = StatementKind.OTHER
            parsed.reject(TableRejection.FORWARD_LOOKING_TABLE, "third-party estimates")
        tables.append(parsed)
    return tables


def document_scale(document_text: str) -> Optional[str]:
    """A scale stated once near the top of a release, for tables that omit it."""
    scale, _factor = _resolve_scale((document_text or "")[:6000])
    return scale
