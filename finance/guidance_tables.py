"""Phase H.14, sections 10-11 — guidance tables keep their rows AND columns.

A guidance table states two things about every number: the ROW says which
metric it is, the COLUMN says which period it covers. Sentence-based
extraction sees neither. Run over a five-row, two-column outlook table it
returned a single metric and dropped the rest -- the full-year column, both
margin rows, the cash-flow row -- and the one value it did keep matched its
period by position rather than by parsing.

Losing a row is losing data. Losing a column is worse: it silently attaches
a quarterly figure to a full year, or the non-GAAP line to the GAAP one,
and the result looks like ordinary guidance.

So this module refuses to flatten. It reads the header for the periods, the
row label for the metric, and pairs the nth value with the nth column. When
that pairing cannot be established -- ragged rows, no header, a column count
that does not match -- it returns nothing and lets the sentence extractor
handle the passage, because a table parsed half-right is worse than a table
not parsed at all.

Nothing here is issuer-specific: it keys on the shape of a table and on the
metric vocabulary that already exists in `finance.guidance`.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# A cell holding a range or a single figure, with optional currency, scale
# and percent. Deliberately tolerant about the dash: tables use hyphen, en
# dash and the word "to" interchangeably.
_RANGE = re.compile(
    r"(?P<low>\$?\s*-?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:%|percent)?\s*"
    r"(?:[-‐-―]|to)\s*"
    r"(?P<high>\$?\s*-?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<pct>%|percent)?\s*"
    r"(?P<scale>billion|million|bn|mm|m|b)?",
    re.IGNORECASE)

_SINGLE = re.compile(
    r"(?P<value>\$?\s*-?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<pct>%|percent)?\s*"
    r"(?P<scale>billion|million|bn|mm|m|b)?",
    re.IGNORECASE)

# A cell that deliberately says "nothing here" -- a metric guided for one
# column and not the other. Common, and not a parse failure.
_EMPTY_CELL = re.compile(r"^\s*(?:-{1,3}|—|n/?a|not\s+provided|\.{3})\s*$", re.IGNORECASE)

# Column headers naming a period. Reuses the same vocabulary the sentence
# path understands so a table and a sentence cannot disagree about what
# "FY2027" means.
_HEADER_PERIOD = re.compile(
    r"(?i)\b(?:"
    r"Q([1-4])\s*(?:FY)?\s*(20\d{2})"
    r"|(?:first|second|third|fourth)\s+quarter\s+(?:of\s+)?(?:fiscal\s+)?(20\d{2})"
    r"|(?:FY|fiscal(?:\s+year)?)\s*(20\d{2})"
    r"|full[\s-]year\s+(20\d{2})"
    r")\b")

# At least this many columns before a line is treated as a header. One
# period in a line is a sentence, not a table header.
_MIN_HEADER_COLUMNS = 1

# Two or more spaces separate columns in a plain-text table. A single space
# is inside a label.
_COLUMN_GAP = re.compile(r"\s{2,}|\t+")


@dataclass
class TableCell:
    """One value, with the row and column that give it meaning."""

    metric_label: str
    period_label: str
    low: Optional[float] = None
    high: Optional[float] = None
    is_percent: bool = False
    scale: Optional[str] = None
    raw: str = ""

    def to_dict(self) -> dict:
        return {
            "metric_label": self.metric_label, "period_label": self.period_label,
            "low": self.low, "high": self.high, "is_percent": self.is_percent,
            "scale": self.scale, "raw": self.raw,
        }


@dataclass
class ParsedTable:
    """A guidance table with its row/column structure intact."""

    period_labels: List[str] = field(default_factory=list)
    cells: List[TableCell] = field(default_factory=list)
    skipped_rows: List[str] = field(default_factory=list)
    reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return bool(self.cells)

    def to_dict(self) -> dict:
        return {
            "period_labels": list(self.period_labels),
            "cells": [c.to_dict() for c in self.cells],
            "skipped_rows": list(self.skipped_rows),
            "reason": self.reason,
        }


def _normalise_period(match) -> str:
    """One header match to a canonical label the guidance parser understands."""
    quarter, q_year, word_year, fy_year, full_year = match.groups()
    if quarter and q_year:
        return f"Q{quarter} FY{q_year}"
    if word_year:
        ordinal = match.group(0).split()[0].lower()
        number = {"first": 1, "second": 2, "third": 3, "fourth": 4}.get(ordinal)
        return f"Q{number} FY{word_year}" if number else f"FY{word_year}"
    return f"FY{fy_year or full_year}"


def _header_periods(line: str) -> List[str]:
    """Every period named in a candidate header line, left to right."""
    return [_normalise_period(m) for m in _HEADER_PERIOD.finditer(line or "")]


def _split_cells(line: str) -> Tuple[str, List[str]]:
    """(row label, cell texts). Columns are separated by 2+ spaces or tabs."""
    parts = [p.strip() for p in _COLUMN_GAP.split(line.strip()) if p.strip()]
    if len(parts) < 2:
        return "", []
    return parts[0], parts[1:]


def _parse_cell(text: str) -> Optional[dict]:
    """One cell to (low, high, percent, scale), or None when it holds nothing.

    A range is tried before a single value: "20% - 22%" must not be read as
    the single figure 20 with the rest discarded.
    """
    if not text or _EMPTY_CELL.match(text):
        return None

    def _number(raw):
        return float(str(raw).replace("$", "").replace(",", "").strip())

    match = _RANGE.search(text)
    if match:
        try:
            low, high = _number(match.group("low")), _number(match.group("high"))
        except (TypeError, ValueError):
            return None
        # A "range" whose high is below its low is not a range -- most often
        # a date or a footnote reference caught by the dash.
        if high < low:
            return None
        return {"low": low, "high": high,
                "is_percent": bool(match.group("pct")) or "%" in text,
                "scale": (match.group("scale") or "").lower() or None}

    match = _SINGLE.search(text)
    if not match:
        return None
    try:
        value = _number(match.group("value"))
    except (TypeError, ValueError):
        return None
    return {"low": value, "high": value,
            "is_percent": bool(match.group("pct")) or "%" in text,
            "scale": (match.group("scale") or "").lower() or None}


def parse_guidance_table(text: str) -> ParsedTable:
    """Rows and columns preserved, or nothing at all.

    Section 11's requirement, and the reason this fails closed: a table
    parsed half-right attaches real numbers to the wrong periods, and the
    output is indistinguishable from correct guidance. When the structure
    cannot be established the sentence extractor still runs over the same
    passage, so refusing here costs nothing that was not already available.
    """
    result = ParsedTable()
    if not isinstance(text, str) or not text.strip():
        result.reason = "no text"
        return result

    lines = [l for l in text.splitlines() if l.strip()]

    # The header is the line naming the MOST distinct periods. Choosing the
    # first line that names any period picked the TITLE instead ("Fiscal
    # 2027 Financial Targets") and then read the real header as a data row.
    # Ties go to the later line, because a title precedes its table.
    header_index, periods = None, []
    for index, line in enumerate(lines):
        found = _header_periods(line)
        if len(set(found)) != len(found):
            continue  # a repeated period is a sentence, not a column header
        if len(found) < _MIN_HEADER_COLUMNS:
            continue
        # A header carries periods and no values of its own.
        stripped = _HEADER_PERIOD.sub("", line)
        if _parse_cell(stripped):
            continue
        if periods and len(found) < len(periods):
            continue
        header_index, periods = index, found

    if header_index is None:
        result.reason = "no column header naming a period was found"
        return result
    result.period_labels = periods

    for line in lines[header_index + 1:]:
        label, cells = _split_cells(line)
        if not label or not cells:
            continue
        # Section 11: the nth value belongs to the nth column. When the
        # counts do not line up the association is a guess, and a guess here
        # attaches a real number to the wrong period.
        if len(cells) != len(periods):
            result.skipped_rows.append(line.strip())
            continue
        for period, cell_text in zip(periods, cells):
            parsed = _parse_cell(cell_text)
            if parsed is None:
                continue
            result.cells.append(TableCell(
                metric_label=label, period_label=period,
                low=parsed["low"], high=parsed["high"],
                is_percent=parsed["is_percent"], scale=parsed["scale"],
                raw=cell_text))

    if not result.cells:
        result.reason = "no row associated a value with a column period"
    return result
