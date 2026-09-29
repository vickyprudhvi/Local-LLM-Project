"""One table cell to one identified financial fact, or to nothing.

WHAT DECIDES WHAT A ROW IS

The ROW LABEL says which line, and only the row label. Not its position, not
the magnitude of its number, not the row above it. "Total current assets" and
"Total assets" differ by one word and by a factor of three, and they sit six
rows apart in every balance sheet ever filed; a substring match on "total
assets" reads the first as the second without hesitating.

So every alias here is ANCHORED. A label matches a field when the whole label
is that line, allowing only the decorations issuers actually add -- a
"(loss)", a ", net", a footnote marker, a trailing colon. A label that is a
longer phrase containing the alias is a DIFFERENT LINE and matches nothing.

WHAT DECIDES WHETHER IT MAY BE A CANONICAL FACT

The TABLE, not the row. A GAAP-to-non-GAAP reconciliation contains a row
called "Operating income" for the same period as the income statement, with a
different number; both are real, one is the company's reported GAAP result and
one is not. Nothing in the row distinguishes them and nothing ever will, which
is why `tables.py` classifies the table from its caption and this module
refuses to take a canonical GAAP field from anything but a reported statement.

Adjusted figures are still EXTRACTED and labelled `ADJUSTED`. They are simply
not eligible to fill a field whose name means the GAAP line.

FLOW AND INSTANT MUST AGREE

`finance/xbrl_mapping.py::CONCEPT_MAP` already records, for every field, whether
it is a point-in-time balance or a flow over a period. That flag is reused here
rather than restated: a balance-sheet field appearing under a "Three Months
Ended" column, or revenue appearing under an undated "As of" column, is a
misread header rather than a fact, and is refused.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from finance import semantics as sem
from finance.reported_actuals.tables import (
    FilingTable,
    PeriodColumn,
    StatementKind,
    TableRow,
)


class FactRejection:
    """Why a row produced no canonical fact."""

    UNKNOWN_ROW_LABEL = "UNKNOWN_ROW_LABEL"
    NOT_A_REPORTED_STATEMENT = "NOT_A_REPORTED_STATEMENT"
    ADJUSTED_MEASURE = "ADJUSTED_MEASURE"
    FLOW_INSTANT_MISMATCH = "FLOW_INSTANT_MISMATCH"
    PERIOD_UNRESOLVED = "PERIOD_UNRESOLVED"
    UNKNOWN_SCALE = "UNKNOWN_SCALE"
    UNKNOWN_CURRENCY = "UNKNOWN_CURRENCY"
    PERCENT_NOT_AN_AMOUNT = "PERCENT_NOT_AN_AMOUNT"
    YTD_NOT_A_QUARTER = "YTD_NOT_A_QUARTER"
    DUPLICATE_ROW = "DUPLICATE_ROW"


# ---------------------------------------------------------------------------
# Row aliases
# ---------------------------------------------------------------------------
#
# Ordered MOST SPECIFIC FIRST, and the first match wins. The ordering carries
# real meaning: "total current assets" must be tried before "total assets",
# and "current portion of long-term debt" before "long-term debt".
#
# Field names are `finance/xbrl_mapping.py::CONCEPT_MAP`'s, so a fact from a
# release and a fact from XBRL land under the same key and can be compared.

# Decorations an issuer may add to any line without changing what it is.
_DECORATION = re.compile(
    r"(?i)\s*(?:\((?:loss(?:es)?|net|unaudited|note\s*\d+|\d+)\)"
    r"|,\s*net(?:\s+of\s+[a-z\s]+)?"
    r"|\s*\(\d+\)"
    r"|\s*\*+"
    r"|:)\s*$")

# A leading article or bullet residue.
_LEADING = re.compile(r"(?i)^\s*(?:[-•●]\s*)+")


def normalize_label(label: str) -> str:
    """A row label reduced to its line, with the decorations removed."""
    text = _LEADING.sub("", " ".join((label or "").split()))
    previous = None
    while previous != text:
        previous = text
        text = _DECORATION.sub("", text).strip()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _alias(*spellings: str) -> re.Pattern:
    """An anchored alternation. Anchoring is the whole precision guarantee."""
    body = "|".join(spellings)
    return re.compile(rf"^(?:{body})$", re.IGNORECASE)


ROW_ALIASES: Tuple[Tuple[str, re.Pattern], ...] = (
    # -- income statement --------------------------------------------------
    ("revenue", _alias(
        r"(?:total\s+)?(?:net\s+)?revenues?",
        r"(?:total\s+)?(?:net\s+)?sales",
        r"(?:total\s+)?revenues?\s+and\s+other\s+income",
        r"(?:total\s+)?net\s+revenues?\s+and\s+other\s+income",
        r"revenues?\s+from\s+contracts?\s+with\s+customers",
        r"total\s+revenues?\s*(?:and|&)\s*sales",
        r"sales\s+and\s+other\s+operating\s+revenues?")),
    ("gross_profit", _alias(r"gross\s+(?:profit|margin)")),
    ("operating_income", _alias(
        r"(?:total\s+)?operating\s+(?:income|profit|earnings)",
        r"(?:income|profit|earnings)\s+from\s+operations",
        r"operating\s+income\s*\(loss\)",
        # IFRS spells the line "Profit from operating activities", with or
        # without a parenthesised loss. It is the same measure, it had no
        # alias, and so every foreign issuer's operating income went unread.
        r"(?:profit|loss|income|earnings)(?:\s*(?:/|\s+or\s+)\s*\(?loss\)?)?"
        r"\s+from\s+operating\s+activities")),
    ("income_before_tax", _alias(
        r"(?:income|earnings|profit)\s+before\s+(?:income\s+)?taxe?s?",
        r"(?:income|earnings)\s+before\s+(?:provision\s+for\s+)?income\s+taxe?s?")),
    ("income_tax_expense", _alias(
        r"(?:provision|benefit)\s+for\s+income\s+taxe?s?",
        r"income\s+tax\s+(?:expense|provision)")),
    ("net_income", _alias(
        r"net\s+(?:income|earnings|profit)",
        r"net\s+(?:income|earnings|loss)\s*\(loss\)",
        r"profit\s+for\s+the\s+(?:period|year)",
        r"net\s+(?:income|earnings|profit)\s+attributable\s+to\s+(?:the\s+)?"
        r"(?:company|parent|common\s+(?:stock|share)holders?|owners?\s+of\s+the\s+parent)")),
    # NO BARE "DILUTED". A condensed income statement puts an unlabelled
    # "Diluted" row under an earnings-per-share heading and a share-count
    # table puts one under a weighted-average heading; the two are a dollar
    # amount and a share count, differing by nine orders of magnitude, and the
    # row itself says which only through a heading this reader does not
    # follow. `identify_row` resolves it from the TABLE instead, and refuses
    # where the table does not say.
    ("diluted_eps", _alias(
        r"diluted(?:\s+(?:net\s+)?(?:income|earnings|loss))?\s+per\s+(?:common\s+)?share",
        r"(?:net\s+)?(?:income|earnings|loss)\s+per\s+(?:common\s+)?share\s*[-–]?\s*diluted",
        r"diluted\s+(?:net\s+)?(?:income|earnings|loss)\s+per\s+share")),

    # -- balance sheet -----------------------------------------------------
    ("cash_and_cash_equivalents", _alias(
        r"cash\s+and\s+cash\s+equivalents",
        r"cash\s*(?:and|&)\s*equivalents",
        r"cash\s+and\s+cash\s+equivalents\s+at\s+end\s+of\s+(?:the\s+)?period")),
    ("short_term_investments", _alias(
        r"short[\s-]term\s+investments",
        r"marketable\s+securities",
        r"short[\s-]term\s+marketable\s+securities")),
    ("current_assets", _alias(r"total\s+current\s+assets")),
    ("current_liabilities", _alias(r"total\s+current\s+liabilities")),
    ("assets", _alias(r"total\s+assets")),
    ("liabilities", _alias(
        r"total\s+liabilities",
        r"total\s+liabilities\s+excluding\s+[a-z\s]+")),
    ("stockholders_equity", _alias(
        r"total\s+(?:stock|share)holders'?\s+equity",
        r"total\s+equity",
        r"total\s+equity\s+attributable\s+to\s+(?:the\s+)?(?:company|owners?\s+of\s+the\s+parent)",
        r"equity\s+attributable\s+to\s+(?:the\s+)?owners?\s+of\s+the\s+parent")),
    ("current_portion_of_long_term_debt", _alias(
        r"current\s+portion\s+of\s+long[\s-]term\s+debt",
        r"long[\s-]term\s+debt,?\s+current\s+portion",
        r"current\s+maturities\s+of\s+long[\s-]term\s+debt")),
    ("short_term_debt", _alias(
        r"short[\s-]term\s+(?:debt|borrowings?)",
        r"notes\s+payable",
        r"commercial\s+paper")),
    ("long_term_debt", _alias(
        r"long[\s-]term\s+debt",
        r"long[\s-]term\s+debt,?\s+(?:less|net\s+of)\s+current\s+(?:portion|maturities)",
        r"long[\s-]term\s+borrowings?")),

    # -- cash flow ---------------------------------------------------------
    ("operating_cash_flow", _alias(
        r"net\s+cash\s+(?:provided\s+by|from|generated\s+(?:by|from))\s+"
        r"operating\s+activities",
        r"net\s+cash\s+provided\s+by\s*\(used\s+in\)\s+operating\s+activities",
        r"cash\s+(?:flow|provided|generated)\s+from\s+operations",
        r"cash\s+flows?\s+from\s+operating\s+activities",
        r"net\s+cash\s+flows?\s+from\s+operating\s+activities")),
    ("capital_expenditure", _alias(
        r"purchases?\s+of\s+property(?:,?\s+plant)?(?:\s+and\s+equipment)?",
        r"purchases?\s+of\s+property\s+and\s+equipment",
        r"capital\s+expenditures?",
        r"additions?\s+to\s+property(?:,?\s+plant)?(?:\s+and\s+equipment)?",
        r"acquisitions?\s+of\s+property(?:,?\s+plant)?(?:\s+and\s+equipment)?",
        r"payments?\s+(?:to\s+acquire|for)\s+property(?:,?\s+plant)?"
        r"(?:\s+and\s+equipment)?")),
    ("depreciation_and_amortization", _alias(
        r"depreciation\s+and\s+amortization",
        r"depreciation,?\s+depletion\s+and\s+amortization")),

    # -- share counts ------------------------------------------------------
    ("diluted_shares", _alias(
        r"diluted\s+weighted[\s-]average\s+(?:number\s+of\s+)?(?:common\s+)?shares"
        r"(?:\s+outstanding)?",
        r"weighted[\s-]average\s+(?:number\s+of\s+)?(?:common\s+)?shares"
        r"(?:\s+outstanding)?\s*[-–]?\s*diluted",
        r"shares\s+used\s+in\s+comput\w+\s+diluted\s+[a-z\s]+per\s+share")),
)

# Rows whose label matches a canonical alias but which are NEVER the
# consolidated line. Checked before the aliases, because "revenue" appears
# inside every one of them and anchoring alone does not exclude a table whose
# rows really are named "Total net sales" per segment.
_COMPONENT_ROW = re.compile(
    r"(?i)\bsegment\b|\bby\s+(?:segment|region|geography|product)\b"
    r"|\bnoncontrolling\b|\bminority\s+interest\b"
    r"|\bdiscontinued\s+operations?\b")

# Which fields are point-in-time. Read from the reviewed mapping so there is
# ONE answer to "is this a balance?" in the codebase.
def _instant_fields() -> frozenset:
    from finance.xbrl_mapping import CONCEPT_MAP
    return frozenset(name for name, (is_instant, _c) in CONCEPT_MAP.items()
                     if is_instant)


INSTANT_FIELDS = _instant_fields()

# Field -> the domain's semantic identity, where one exists. Facts for a field
# with no identity are still produced; they simply cannot take part in the
# semantic compatibility checks, which is honest rather than convenient.
FIELD_IDENTITY = {
    "revenue": sem.MetricIdentity.REVENUE,
    "operating_income": sem.MetricIdentity.OPERATING_INCOME,
    "net_income": sem.MetricIdentity.NET_INCOME,
    "gross_profit": sem.MetricIdentity.GROSS_PROFIT,
    "cash_and_cash_equivalents": sem.MetricIdentity.CASH,
    "short_term_investments": sem.MetricIdentity.SHORT_TERM_INVESTMENTS,
    "short_term_debt": sem.MetricIdentity.SHORT_TERM_DEBT,
    "long_term_debt": sem.MetricIdentity.LONG_TERM_DEBT,
    "stockholders_equity": sem.MetricIdentity.STOCKHOLDERS_EQUITY,
    "operating_cash_flow": sem.MetricIdentity.OPERATING_CASH_FLOW,
    "capital_expenditure": sem.MetricIdentity.CAPEX,
    "diluted_shares": sem.MetricIdentity.SHARES_WEIGHTED_AVERAGE_DILUTED,
}

# Fields reported as a NEGATIVE outflow in a cash-flow statement but held as a
# positive magnitude everywhere else in this project. `finance/normalization.py`
# already owns that convention; this mirrors it for one field rather than
# re-deriving a sign policy.
_ABSOLUTE_MAGNITUDE_FIELDS = ("capital_expenditure",)

# Per-share rows are printed in units, never in the table's scale.
_PER_SHARE_FIELDS = ("diluted_eps",)
# Share counts are usually printed in the table's scale (thousands/millions).
_SHARE_COUNT_FIELDS = ("diluted_shares",)


# A bare basis row -- "Basic", "Diluted" -- means whatever its TABLE is about.
_BARE_SHARE_BASIS = re.compile(r"(?i)^(?:basic|diluted)$")


def identify_row(label: str, statement_kind: Optional[str] = None) -> Optional[str]:
    """The canonical field this row reports, or None.

    `statement_kind` resolves the one ambiguity a row label genuinely cannot:
    a row called "Diluted" is a share count in a weighted-average-shares table
    and an earnings-per-share figure in an income statement. Without the
    table, it is refused rather than guessed -- a share count read as EPS is
    off by a factor of a billion and still looks like a number.
    """
    text = normalize_label(label)
    if not text or _COMPONENT_ROW.search(text):
        return None
    if _BARE_SHARE_BASIS.match(text):
        if statement_kind == StatementKind.SHARE_COUNT and text == "diluted":
            return "diluted_shares"
        return None
    for name, pattern in ROW_ALIASES:
        if pattern.match(text):
            return name
    return None


# ---------------------------------------------------------------------------
# The fact
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActualFinancialFact:
    """One reported number, with everything needed to check and cite it."""

    field: str
    value: float
    raw_value: float
    currency: str
    scale: Optional[str]
    period_end: str
    period_start: Optional[str]
    frequency: str
    flow_or_instant: str
    accounting_basis: str
    statement_kind: str
    row_label: str
    column_label: str
    band_label: str
    metric_identity: str = sem.MetricIdentity.UNKNOWN
    table_index: int = 0
    row_index: int = 0
    # Filing provenance, stamped by the caller that knows it.
    accession: Optional[str] = None
    form: Optional[str] = None
    filed: Optional[str] = None
    document: Optional[str] = None
    finality: str = "UNKNOWN"

    @property
    def is_canonical(self) -> bool:
        """May this fill a field whose name means the reported GAAP line?"""
        return (self.accounting_basis != sem.AccountingBasis.ADJUSTED
                and self.statement_kind in StatementKind.REPORTED_STATEMENTS)

    def to_dict(self) -> dict:
        return {"field": self.field, "value": self.value,
                "raw_value": self.raw_value, "currency": self.currency,
                "scale": self.scale, "period_end": self.period_end,
                "period_start": self.period_start, "frequency": self.frequency,
                "flow_or_instant": self.flow_or_instant,
                "accounting_basis": self.accounting_basis,
                "statement_kind": self.statement_kind,
                "row_label": self.row_label, "column_label": self.column_label,
                "band_label": self.band_label,
                "metric_identity": self.metric_identity,
                "table_index": self.table_index, "row_index": self.row_index,
                "accession": self.accession, "form": self.form,
                "filed": self.filed, "document": self.document,
                "finality": self.finality, "is_canonical": self.is_canonical}


def _period_start(period_end: str, frequency: str) -> Optional[str]:
    """The start implied by an end and a duration. Approximate by design.

    Used only to describe the fact; nothing selects on it. A fiscal quarter is
    thirteen weeks and not three calendar months, so this is the month
    arithmetic a reader would do and is labelled as derived rather than
    reported.
    """
    import datetime

    months = sem.duration_months(frequency)
    if not months or not period_end:
        return None
    try:
        end = datetime.date.fromisoformat(period_end)
    except ValueError:
        return None
    year = end.year
    month = end.month - months + 1
    while month <= 0:
        month += 12
        year -= 1
    try:
        return datetime.date(year, month, 1).isoformat()
    except ValueError:
        return None


def facts_from_table(table: FilingTable, *, accession: Optional[str] = None,
                     form: Optional[str] = None, filed: Optional[str] = None,
                     document: Optional[str] = None,
                     finality: str = "UNKNOWN",
                     gaap_basis: str = sem.AccountingBasis.GAAP
                     ) -> Tuple[List[ActualFinancialFact], List[Tuple[str, str]]]:
    """(facts, rejections) for one parsed table."""
    rejections: List[Tuple[str, str]] = []

    if table.scale_factor is None:
        rejections.append((FactRejection.UNKNOWN_SCALE, table.caption[:100]))
        return [], rejections
    if not table.currency:
        rejections.append((FactRejection.UNKNOWN_CURRENCY, table.caption[:100]))
        return [], rejections
    if table.kind == StatementKind.GUIDANCE:
        rejections.append((FactRejection.NOT_A_REPORTED_STATEMENT,
                           "the table is an outlook"))
        return [], rejections

    basis = (sem.AccountingBasis.ADJUSTED
             if table.kind == StatementKind.NON_GAAP_RECONCILIATION else gaap_basis)
    canonical_table = table.kind in StatementKind.REPORTED_STATEMENTS \
        or table.kind == StatementKind.SHARE_COUNT

    facts: List[ActualFinancialFact] = []
    seen: Dict[Tuple[str, str, str], int] = {}

    for row in table.rows:
        name = identify_row(row.label, table.kind)
        if name is None:
            continue
        if not canonical_table and basis != sem.AccountingBasis.ADJUSTED:
            rejections.append((FactRejection.NOT_A_REPORTED_STATEMENT,
                               f"{row.label} in a {table.kind} table"))
            continue

        wants_instant = name in INSTANT_FIELDS
        for position, cell in sorted(row.values.items()):
            if position >= len(table.period_columns):
                continue
            column = table.period_columns[position]
            if cell.is_percent:
                rejections.append((FactRejection.PERCENT_NOT_AN_AMOUNT,
                                   f"{row.label} / {column.label}"))
                continue
            is_instant_column = column.frequency == sem.PeriodFrequency.INSTANT
            if wants_instant != is_instant_column:
                # A balance under a duration header, or a flow under an
                # instant one. The header was misread or the row was; either
                # way this is not a fact about that period.
                rejections.append((
                    FactRejection.FLOW_INSTANT_MISMATCH,
                    f"{row.label} ({'instant' if wants_instant else 'flow'}) "
                    f"under {column.band_label or column.label}"))
                continue

            factor = table.scale_factor
            if name in _PER_SHARE_FIELDS:
                factor = 1.0
            value = cell.number * factor
            if name in _ABSOLUTE_MAGNITUDE_FIELDS:
                value = abs(value)

            key = (name, column.period_end or "", column.frequency)
            if key in seen:
                # A statement may repeat a line (a subtotal restated in the
                # supplementary detail). The FIRST occurrence is the statement
                # line; a later one is not a second fact.
                rejections.append((FactRejection.DUPLICATE_ROW,
                                   f"{row.label} / {column.label}"))
                continue
            seen[key] = 1

            facts.append(ActualFinancialFact(
                field=name, value=value, raw_value=cell.number,
                currency=table.currency, scale=table.scale,
                period_end=column.period_end or "",
                period_start=(None if is_instant_column
                              else _period_start(column.period_end or "",
                                                 column.frequency)),
                frequency=column.frequency,
                flow_or_instant=(sem.FlowOrInstant.INSTANT if is_instant_column
                                 else sem.FlowOrInstant.FLOW),
                accounting_basis=basis, statement_kind=table.kind,
                row_label=row.label, column_label=column.label,
                band_label=column.band_label,
                metric_identity=FIELD_IDENTITY.get(name, sem.MetricIdentity.UNKNOWN),
                table_index=table.index, row_index=row.row_index,
                accession=accession, form=form, filed=filed, document=document,
                finality=finality))
    return facts, rejections
