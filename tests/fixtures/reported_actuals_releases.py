"""Filed earnings-release exhibits, built the way EDGAR actually builds them.

WHY THE MARKUP IS UGLY ON PURPOSE

A real filed exhibit is not a tidy `<table><tr><td>value`. It is a grid padded
with `colspan`, with the currency symbol in a column of its own, spacer columns
between every pair of data columns, and a two-level header where one row names
the DURATION and the row beneath names the END DATE. The reader under test
resolves a value's meaning from its grid position, so a fixture that skipped
the padding would test a parser this project does not have.

`statement_table` below reproduces that layout exactly -- the same
label(3) / spacer(3) / column(3) rhythm and the same band spans measured on a
live filing -- so a fixture case and a live exhibit exercise one code path.

WHAT THE FIXTURES ARE

Section 23's classes A-K, as generic issuers. No ticker appears: a case is a
calendar, a set of statements and a set of column headers, and nothing about
"a complete Q4 release filed before the 10-K" needs a company's name.

Every expected value is written beside the text that states it, so a change to
one is visibly a change to the other.
"""

from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Markup
# ---------------------------------------------------------------------------

_LABEL_WIDTH = 3
_SPACER = 3
_COLUMN_WIDTH = 3


def _cell(text: str, span: int) -> str:
    return f'<td colspan="{span}">{text}</td>'


def _row(cells: Sequence[Tuple[str, int]]) -> str:
    return "<tr>" + "".join(_cell(t, n) for t, n in cells) + "</tr>"


def _header_row(columns: Sequence[str]) -> str:
    cells: List[Tuple[str, int]] = [("", _LABEL_WIDTH)]
    for label in columns:
        cells.append(("", _SPACER))
        cells.append((label, _COLUMN_WIDTH))
    return _row(cells)


def _band_row(bands: Sequence[Tuple[str, int]]) -> str:
    """`bands` is (label, how many columns it covers), left to right."""
    cells: List[Tuple[str, int]] = [("", _LABEL_WIDTH)]
    for label, count in bands:
        cells.append(("", _SPACER))
        cells.append((label, count * (_SPACER + _COLUMN_WIDTH) - _SPACER))
    return _row(cells)


def _data_row(label: str, values: Sequence[Optional[str]],
              currency: str = "$") -> str:
    cells: List[Tuple[str, int]] = [(label, _LABEL_WIDTH)]
    for value in values:
        cells.append(("", _SPACER))
        if value is None:
            cells.append(("", _COLUMN_WIDTH))
            continue
        # The currency symbol sits in its own one-wide column, exactly as
        # filed. It is not a value and must never be read as one.
        cells.append((currency, 1))
        cells.append((value, 1))
        cells.append(("", 1))
    return _row(cells)


def statement_table(caption: str, columns: Sequence[str],
                    rows: Sequence[Tuple[str, Sequence[Optional[str]]]],
                    bands: Optional[Sequence[Tuple[str, int]]] = None,
                    currency: str = "$") -> str:
    """One filed statement table, caption above it as EDGAR places it."""
    parts = [f'<div style="text-align:center">{caption}</div>', "<table>"]
    if bands:
        parts.append(_band_row(bands))
    parts.append(_header_row(columns))
    for label, values in rows:
        parts.append(_data_row(label, values, currency=currency))
    parts.append("</table>")
    return "\n".join(parts)


def release(title: str, *blocks: str) -> str:
    return ("<html><body>\n<p>" + title + "</p>\n"
            + "\n".join(blocks) + "\n</body></html>")


# ---------------------------------------------------------------------------
# The shared issuer: fiscal year ends 31 July
# ---------------------------------------------------------------------------
#
# The same non-calendar year the Actualization benchmark uses, for the same
# reason: it is where "latest periodic filing" and "latest reported period"
# come apart, and a fixture set built entirely on December year-ends would
# never exercise the difference.

Q3_END = "April 30, 2026"
Q3_END_ISO = "2026-04-30"
Q4_END = "July 31, 2026"
Q4_END_ISO = "2026-07-31"
Q4_PRIOR_END = "July 31, 2025"
Q4_PRIOR_END_ISO = "2025-07-31"

# The reported Q4 figures every case that states them agrees on, in millions.
Q4 = {
    "revenue": 1_450.0,
    "operating_income": 302.0,
    "net_income": 236.0,
    "cash_and_cash_equivalents": 812.0,
    "short_term_investments": 190.0,
    "assets": 6_140.0,
    "current_assets": 2_050.0,
    "liabilities": 3_260.0,
    "current_liabilities": 1_120.0,
    "stockholders_equity": 2_880.0,
    "short_term_debt": 150.0,
    "long_term_debt": 1_400.0,
    "operating_cash_flow": 358.0,
    "capital_expenditure": 74.0,
}

# The FULL YEAR figures, stated in the same release under a second band. They
# share a period END with the quarter above and are a different measurement of
# it; nothing may substitute one for the other.
FY = {
    "revenue": 5_460.0,
    "operating_income": 1_106.0,
    "net_income": 861.0,
    "operating_cash_flow": 1_302.0,
    "capital_expenditure": 268.0,
}

MILLIONS = 1_000_000.0

# The prior-year comparative column, as one factor used by EVERY statement.
# Two different factors would make a fixture disagree with itself, and the
# reader would be right to refuse the figure -- a defect in the fixture that
# would read as a defect in the code.
PRIOR_FACTOR = 0.86


def _m(value: float) -> str:
    return f"{value:,.1f}"


# -- the reusable statement blocks ------------------------------------------

def income_statement(*, quarter: bool = True, annual: bool = False,
                     scale: str = "IN MILLIONS") -> str:
    """The condensed income statement, with whichever bands the case needs."""
    columns, bands, sets = [], [], []
    if quarter:
        columns += [Q4_END, Q4_PRIOR_END]
        bands.append(("Three Months Ended", 2))
        sets.append(("quarter", Q4))
    if annual:
        columns += [Q4_END, Q4_PRIOR_END]
        bands.append(("Year Ended", 2))
        sets.append(("annual", FY))

    def values(name, prior_factor=PRIOR_FACTOR):
        out = []
        for _kind, source in sets:
            current = source.get(name)
            out += [_m(current), _m(current * prior_factor)] if current is not None \
                else [None, None]
        return out

    return statement_table(
        f"CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS - UNAUDITED ({scale})",
        columns,
        [("Total revenue", values("revenue")),
         ("Cost of revenue", values("revenue") and [_m(700.0), _m(620.0)]
          + ([_m(2600.0), _m(2300.0)] if annual and quarter else [])),
         ("Operating income", values("operating_income")),
         ("Net income", values("net_income"))],
        bands=bands)


def balance_sheet(scale: str = "IN MILLIONS") -> str:
    """A balance sheet: two dated columns and NO duration band."""
    return statement_table(
        f"CONDENSED CONSOLIDATED BALANCE SHEETS - UNAUDITED ({scale})",
        [Q4_END, Q4_PRIOR_END],
        [("Cash and cash equivalents",
          [_m(Q4["cash_and_cash_equivalents"]), _m(690.0)]),
         ("Short-term investments", [_m(Q4["short_term_investments"]), _m(160.0)]),
         ("Total current assets", [_m(Q4["current_assets"]), _m(1_880.0)]),
         ("Total assets", [_m(Q4["assets"]), _m(5_710.0)]),
         ("Short-term debt", [_m(Q4["short_term_debt"]), _m(120.0)]),
         ("Total current liabilities", [_m(Q4["current_liabilities"]), _m(1_010.0)]),
         ("Long-term debt", [_m(Q4["long_term_debt"]), _m(1_480.0)]),
         ("Total liabilities", [_m(Q4["liabilities"]), _m(3_190.0)]),
         ("Total stockholders' equity",
          [_m(Q4["stockholders_equity"]), _m(2_520.0)])])


def cash_flow(*, quarter: bool = True, annual: bool = False,
              scale: str = "IN MILLIONS") -> str:
    columns, bands, sets = [], [], []
    if quarter:
        columns += [Q4_END, Q4_PRIOR_END]
        bands.append(("Three Months Ended", 2))
        sets.append(Q4)
    if annual:
        columns += [Q4_END, Q4_PRIOR_END]
        bands.append(("Year Ended", 2))
        sets.append(FY)

    def values(name):
        out = []
        for source in sets:
            current = source.get(name)
            out += [_m(current), _m(current * PRIOR_FACTOR)]
        return out

    return statement_table(
        f"CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS - UNAUDITED ({scale})",
        columns,
        [("Net income", values("net_income")),
         ("Net cash provided by operating activities", values("operating_cash_flow")),
         ("Purchases of property, plant and equipment", values("capital_expenditure"))],
        bands=bands)


def non_gaap_reconciliation() -> str:
    """Adjusted figures for the SAME period, spelled like the GAAP lines.

    The rows are called "Operating income" and "Net income" because that is
    what issuers call them. Only the caption distinguishes this table, which
    is precisely why the caption is what decides.
    """
    return statement_table(
        "RECONCILIATION OF GAAP TO NON-GAAP RESULTS - UNAUDITED (IN MILLIONS)",
        [Q4_END, Q4_PRIOR_END],
        [("Operating income", [_m(302.0), _m(260.0)]),
         ("Stock-based compensation", [_m(88.0), _m(74.0)]),
         ("Non-GAAP operating income", [_m(390.0), _m(334.0)]),
         ("Net income", [_m(310.0), _m(268.0)])],
        bands=[("Three Months Ended", 2)])


ADJUSTED_OPERATING_INCOME = 390.0 * MILLIONS
ADJUSTED_NET_INCOME = 310.0 * MILLIONS


def outlook_table() -> str:
    """An OUTLOOK table. Same shape, same row names, forward periods."""
    return statement_table(
        "FINANCIAL OUTLOOK - GUIDANCE (IN MILLIONS)",
        ["October 31, 2026", "July 31, 2027"],
        [("Total revenue", [_m(1_600.0), _m(6_300.0)]),
         ("Operating income", [_m(340.0), _m(1_310.0)])],
        bands=[("Three Months Ending", 1), ("Year Ending", 1)])


GUIDED_Q1_REVENUE = 1_600.0 * MILLIONS
GUIDED_FY_REVENUE = 6_300.0 * MILLIONS


def analyst_table() -> str:
    """Third-party consensus, printed beside the results as some issuers do."""
    return statement_table(
        "ANALYST CONSENSUS ESTIMATES SURVEYED BY A DATA PROVIDER (IN MILLIONS)",
        [Q4_END],
        [("Total revenue", [_m(1_505.0)]),
         ("Operating income", [_m(318.0)])],
        bands=[("Three Months Ended", 1)])


ANALYST_REVENUE = 1_505.0 * MILLIONS


def headline_paragraph() -> str:
    """A release that states two numbers in prose and files no statements."""
    return ("<p>The company today reported total revenue of $1,450.0 million "
            "for the fourth quarter ended July 31, 2026 and diluted earnings "
            "per share of $1.83.</p>")


def share_count_table() -> str:
    return statement_table(
        "WEIGHTED-AVERAGE SHARES USED IN COMPUTING EARNINGS PER SHARE (IN MILLIONS)",
        [Q4_END, Q4_PRIOR_END],
        [("Basic", [_m(126.0), _m(131.0)]),
         ("Diluted", [_m(129.0), _m(134.0)])],
        bands=[("Three Months Ended", 2)])


# ---------------------------------------------------------------------------
# The documents, one per section 23 class
# ---------------------------------------------------------------------------

# A. complete Q4/FY release, filed before the 10-K
COMPLETE_Q4_RELEASE = release(
    "Example Corp Reports Fourth Quarter and Full Year Fiscal 2026 Results",
    income_statement(quarter=True, annual=True),
    balance_sheet(),
    cash_flow(quarter=True, annual=True),
    non_gaap_reconciliation(),
    share_count_table())

# B. the same quarter announced with an income statement and nothing else:
# no balance sheet, no cash-flow statement. A candidate EXISTS -- the period
# was reported -- and it cannot carry the state.
HEADLINE_ONLY_RELEASE = release(
    "Example Corp Announces Fourth Quarter Fiscal 2026 Results",
    headline_paragraph(),
    statement_table(
        "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS - UNAUDITED (IN MILLIONS)",
        [Q4_END, Q4_PRIOR_END],
        [("Total revenue", [_m(1_450.0), _m(1_247.0)]),
         ("Net income", [_m(236.0), _m(203.0)])],
        bands=[("Three Months Ended", 2)]))

# The same announcement with NO table at all. Prose is not a financial
# statement and this layer reads none: the correct output is nothing.
PROSE_ONLY_RELEASE = release(
    "Example Corp Announces Fourth Quarter Fiscal 2026 Results",
    headline_paragraph())

# C/D. the periodic filing that follows, agreeing or disagreeing
PERIODIC_FILING_MATCHING = release(
    "Example Corp Annual Report on Form 10-K",
    income_statement(quarter=True, annual=True),
    balance_sheet(),
    cash_flow(quarter=True, annual=True))


def _conflicting_income_statement() -> str:
    """The same quarter, restated: revenue moved 4.5% between the release and
    the filing. Far outside a rounding difference and a different claim about
    one period."""
    return statement_table(
        "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS - UNAUDITED (IN MILLIONS)",
        [Q4_END, Q4_PRIOR_END],
        [("Total revenue", [_m(1_385.0), _m(1_247.0)]),
         ("Operating income", [_m(302.0), _m(260.0)]),
         ("Net income", [_m(236.0), _m(203.0)])],
        bands=[("Three Months Ended", 2)])


CONFLICTING_REVENUE = 1_385.0 * MILLIONS

PERIODIC_FILING_CONFLICTING = release(
    "Example Corp Annual Report on Form 10-K",
    _conflicting_income_statement(),
    balance_sheet(),
    cash_flow(quarter=True))

# E. actuals AND an outlook in one document
RELEASE_WITH_OUTLOOK = release(
    "Example Corp Reports Fourth Quarter Fiscal 2026 Results and Provides Outlook",
    income_statement(quarter=True),
    balance_sheet(),
    cash_flow(quarter=True),
    "<p>Financial Outlook</p>",
    outlook_table())

# -- canonical-integration fixtures --------------------------------------
#
# A Q4 release with an income statement and a cash-flow statement but NO
# balance sheet. Two statements of three: PARTIAL, cannot carry the whole
# period, must not advance the state wholesale.
RELEASE_WITH_OUTLOOK_NO_BS = release(
    "Example Corp Reports Fourth Quarter Fiscal 2026 Results",
    income_statement(quarter=True),
    cash_flow(quarter=True))


def _cash_flow_no_capex() -> str:
    """A COMPLETE cash-flow statement that simply omits the capex line -- some
    issuers report capital expenditure only in an investing-activities
    subtotal a deterministic reader does not decompose."""
    return statement_table(
        "CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS - UNAUDITED (IN MILLIONS)",
        [Q4_END, Q4_PRIOR_END],
        [("Net income", [_m(Q4["net_income"]), _m(Q4["net_income"] * PRIOR_FACTOR)]),
         ("Net cash provided by operating activities",
          [_m(Q4["operating_cash_flow"]),
           _m(Q4["operating_cash_flow"] * PRIOR_FACTOR)])],
        bands=[("Three Months Ended", 2)])


# A COMPLETE Q4 release -- all three statements -- that does not tag capital
# expenditure. The period advances; capex alone falls back to the prior
# quarter and is LABELLED as such.
RELEASE_NO_CASH_FLOW = release(
    "Example Corp Reports Fourth Quarter and Full Year Fiscal 2026 Results",
    income_statement(quarter=True),
    balance_sheet(),
    _cash_flow_no_capex())

# A Q3 release that RESTATES the quarter CompanyFacts already carries, with
# the SAME figures. The overlay must recognise this as one economic quarter,
# not add a near-duplicate to the twelve-month window. Numbers here match
# `canonical_actual_integration_benchmark._FLOW_QUARTERS[('FY2026','Q3')]`
# and `._BALANCE['2026-04-30']` exactly.
RELEASE_Q3_RESTATEMENT = release(
    "Example Corp Reports Third Quarter Fiscal 2026 Results",
    statement_table(
        "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS - UNAUDITED (IN MILLIONS)",
        [Q3_END, Q3_END.replace("2026", "2025")],
        [("Total revenue", ["1,340.0", "1,050.0"]),
         ("Operating income", ["272.0", "210.0"]),
         ("Net income", ["214.0", "168.0"])],
        bands=[("Three Months Ended", 2)]),
    statement_table(
        "CONDENSED CONSOLIDATED BALANCE SHEETS - UNAUDITED (IN MILLIONS)",
        [Q3_END, "July 31, 2025"],
        [("Cash and cash equivalents", ["690.0", "640.0"]),
         ("Total assets", ["5,710.0", "5,480.0"]),
         ("Total current assets", ["1,880.0", "1,790.0"]),
         ("Total current liabilities", ["1,010.0", "980.0"]),
         ("Total stockholders' equity", ["2,520.0", "2,360.0"])]),
    statement_table(
        "CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS - UNAUDITED (IN MILLIONS)",
        [Q3_END, Q3_END.replace("2026", "2025")],
        [("Net cash provided by operating activities", ["318.0", "250.0"]),
         ("Purchases of property, plant and equipment", ["66.0", "52.0"])],
        bands=[("Three Months Ended", 2)]))


# G. results with a consensus table beside them
RELEASE_WITH_ANALYST_TABLE = release(
    "Example Corp Reports Fourth Quarter Fiscal 2026 Results",
    income_statement(quarter=True),
    balance_sheet(),
    cash_flow(quarter=True),
    analyst_table())

# I. the same statements printed in THOUSANDS
THOUSANDS_RELEASE = release(
    "Example Corp Reports Fourth Quarter Fiscal 2026 Results",
    statement_table(
        "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS - UNAUDITED (IN THOUSANDS)",
        [Q4_END, Q4_PRIOR_END],
        [("Total revenue", ["1,450,000", "1,247,000"]),
         ("Operating income", ["302,000", "260,000"]),
         ("Net income", ["236,000", "203,000"])],
        bands=[("Three Months Ended", 2)]),
    statement_table(
        "CONDENSED CONSOLIDATED BALANCE SHEETS - UNAUDITED (IN THOUSANDS)",
        [Q4_END, Q4_PRIOR_END],
        [("Cash and cash equivalents", ["812,000", "690,000"]),
         ("Total assets", ["6,140,000", "5,710,000"]),
         ("Total stockholders' equity", ["2,880,000", "2,520,000"])]),
    statement_table(
        "CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS - UNAUDITED (IN THOUSANDS)",
        [Q4_END, Q4_PRIOR_END],
        [("Net cash provided by operating activities", ["358,000", "322,000"])],
        bands=[("Three Months Ended", 2)]))

# A table that states no scale anywhere. Its figures are unreadable and must
# produce nothing at all.
NO_SCALE_RELEASE = release(
    "Example Corp Reports Fourth Quarter Fiscal 2026 Results",
    statement_table(
        "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS - UNAUDITED",
        [Q4_END, Q4_PRIOR_END],
        [("Total revenue", ["1,450.0", "1,247.0"]),
         ("Operating income", ["302.0", "260.0"])],
        bands=[("Three Months Ended", 2)]))

# J. a quarter column and a YEAR-TO-DATE column sharing one period end
QUARTER_AND_YTD_RELEASE = release(
    "Example Corp Reports Third Quarter Fiscal 2026 Results",
    statement_table(
        "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS - UNAUDITED (IN MILLIONS)",
        [Q3_END, Q3_END],
        [("Total revenue", ["1,310.0", "4,010.0"]),
         ("Operating income", ["268.0", "804.0"]),
         ("Net income", ["209.0", "625.0"])],
        bands=[("Three Months Ended", 1), ("Nine Months Ended", 1)]),
    statement_table(
        "CONDENSED CONSOLIDATED BALANCE SHEETS - UNAUDITED (IN MILLIONS)",
        [Q3_END],
        [("Cash and cash equivalents", ["740.0"]),
         ("Total assets", ["5,980.0"]),
         ("Total stockholders' equity", ["2,760.0"])]),
    statement_table(
        "CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS - UNAUDITED (IN MILLIONS)",
        [Q3_END, Q3_END],
        [("Net cash provided by operating activities", ["331.0", "988.0"])],
        bands=[("Three Months Ended", 1), ("Nine Months Ended", 1)]))

Q3_QUARTER_REVENUE = 1_310.0 * MILLIONS
Q3_YTD_REVENUE = 4_010.0 * MILLIONS

# H. a foreign private issuer's interim results, filed on a 6-K under IFRS
FOREIGN_ISSUER_RELEASE = release(
    "Example International plc Announces Results for the Half Year Ended 30 June 2026",
    statement_table(
        "CONSOLIDATED STATEMENTS OF PROFIT OR LOSS - UNAUDITED "
        "(IN MILLIONS OF USD)",
        ["30 June 2026", "30 June 2025"],
        [("Revenue", ["2,410.0", "2,190.0"]),
         ("Profit from operating activities", ["486.0", "441.0"]),
         ("Profit for the period", ["372.0", "338.0"])],
        bands=[("Six Months Ended", 2)]),
    statement_table(
        "CONSOLIDATED STATEMENTS OF FINANCIAL POSITION - UNAUDITED "
        "(IN MILLIONS OF USD)",
        ["30 June 2026", "31 December 2025"],
        [("Cash and cash equivalents", ["1,180.0", "1,020.0"]),
         ("Total assets", ["11,400.0", "10,880.0"]),
         ("Total equity", ["5,120.0", "4,860.0"])]),
    statement_table(
        "CONSOLIDATED STATEMENTS OF CASH FLOWS - UNAUDITED (IN MILLIONS OF USD)",
        ["30 June 2026", "30 June 2025"],
        [("Net cash from operating activities", ["705.0", "651.0"])],
        bands=[("Six Months Ended", 2)]))

FOREIGN_HALF_YEAR_END = "2026-06-30"
FOREIGN_REVENUE = 2_410.0 * MILLIONS

# A euro-reporting release. Its figures are real and are NOT dollars, and
# nothing here converts currency.
EURO_RELEASE = release(
    "Example Europe SE Reports Fourth Quarter 2026 Results",
    statement_table(
        "CONSOLIDATED STATEMENTS OF PROFIT OR LOSS - UNAUDITED "
        "(IN MILLIONS OF EUR)",
        ["31 December 2026", "31 December 2025"],
        [("Revenue", ["3,120.0", "2,940.0"]),
         ("Profit from operating activities", ["540.0", "498.0"])],
        bands=[("Year Ended", 2)], currency="€"),
    statement_table(
        "CONSOLIDATED STATEMENTS OF FINANCIAL POSITION - UNAUDITED "
        "(IN MILLIONS OF EUR)",
        ["31 December 2026", "31 December 2025"],
        [("Cash and cash equivalents", ["980.0", "910.0"]),
         ("Total assets", ["14,200.0", "13,700.0"]),
         ("Total equity", ["6,400.0", "6,010.0"])], currency="€"),
    statement_table(
        "CONSOLIDATED STATEMENTS OF CASH FLOWS - UNAUDITED (IN MILLIONS OF EUR)",
        ["31 December 2026", "31 December 2025"],
        [("Net cash from operating activities", ["812.0", "760.0"])],
        bands=[("Year Ended", 2)], currency="€"))

EURO_REVENUE = 3_120.0 * MILLIONS

# A release whose statement titles sit behind the wall of inline styling a
# real EDGAR exhibit carries. The caption is three text-blocks above the
# table and several thousand HTML characters away.
# Measured on a live exhibit: the statement title sat 718 HTML characters
# above its table, behind exactly this kind of inline styling. A caption
# window counted in HTML characters therefore misses it while a reader looking
# at the page sees the title directly above the numbers.
_STYLE = ('style="font-family:Calibri,sans-serif;font-kerning:none;'
          'min-width:fit-content;margin-bottom:0pt;margin-top:0pt;'
          'text-align:center;font-size:9pt;line-height:120%;'
          'padding:2.62pt 0pt 1.5pt 0pt;background-color:#ffffff;'
          'vertical-align:bottom;border-top:0pt solid #000000;'
          'border-bottom:0pt solid #000000;white-space:nowrap;'
          'letter-spacing:0pt;text-indent:0pt;font-weight:700"')


def _styled_caption(*lines: str) -> str:
    return "".join(f"<div {_STYLE}>{line}</div>\n" for line in lines)


def styled_statement(caption_lines, columns, rows, bands=None) -> str:
    """A statement whose title is split across styled blocks above the table."""
    return _styled_caption(*caption_lines) + statement_table(
        "", columns, rows, bands=bands).split("\n", 1)[1]


HEAVY_MARKUP_RELEASE = release(
    "Example Corp Reports Fourth Quarter and Full Year Fiscal 2026 Results",
    styled_statement(
        ["EXAMPLE CORPORATION", "Q4 FISCAL 2026 FINANCIAL RESULTS",
         "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS",
         "($ in millions, except per share data)"],
        [Q4_END, Q4_PRIOR_END],
        [("Total revenue", [_m(Q4["revenue"]), _m(1_247.0)]),
         ("Operating income", [_m(Q4["operating_income"]), _m(260.0)]),
         ("Net income", [_m(Q4["net_income"]), _m(203.0)])],
        bands=[("Three Months Ended", 2)]),
    styled_statement(
        ["EXAMPLE CORPORATION", "FISCAL 2026 FINANCIAL RESULTS",
         "CONDENSED CONSOLIDATED BALANCE SHEETS", "($ in millions)"],
        [Q4_END, Q4_PRIOR_END],
        [("Cash and cash equivalents",
          [_m(Q4["cash_and_cash_equivalents"]), _m(690.0)]),
         ("Total assets", [_m(Q4["assets"]), _m(5_710.0)]),
         ("Total stockholders' equity",
          [_m(Q4["stockholders_equity"]), _m(2_520.0)])]),
    styled_statement(
        ["EXAMPLE CORPORATION", "FISCAL 2026 FINANCIAL RESULTS",
         "CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS", "($ in millions)"],
        [Q4_END, Q4_PRIOR_END],
        [("Net cash provided by operating activities",
          [_m(Q4["operating_cash_flow"]), _m(322.0)]),
         ("Purchases of property, plant and equipment",
          [_m(Q4["capital_expenditure"]), _m(66.0)])],
        bands=[("Three Months Ended", 2)]))


# F. an 8-K about something else entirely, with an EX-99.1 press release
LEADERSHIP_RELEASE = release(
    "Example Corp Announces Appointment of Chief Financial Officer",
    "<p>Example Corp today announced the appointment of a new Chief Financial "
    "Officer, effective September 1, 2026. The company also entered into a "
    "credit agreement providing for a revolving facility of $500.0 million.</p>")


# ---------------------------------------------------------------------------
# Filing indexes and submissions payloads
# ---------------------------------------------------------------------------

def filing_index(rows: Sequence[Tuple[str, str, str, str]]) -> str:
    """An EDGAR filing index page. Columns are Seq | Description | Document | Type."""
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows)
    return f"<html><body><table>{body}</table></body></html>"


EARNINGS_INDEX = filing_index([
    ("Seq", "Description", "Document", "Type"),
    ("1", "FORM 8-K", "form8k.htm", "8-K"),
    ("2", "PRESS RELEASE DATED SEPTEMBER 2, 2026", "exhibit991.htm", "EX-99.1"),
    ("3", "INVESTOR PRESENTATION", "exhibit992.htm", "EX-99.2"),
])

# The conventional slot holds the WRONG document here: EX-99.1 is a deck and
# the results are in EX-99.2. An index reader that assumes the numbering picks
# the presentation.
INVERTED_INDEX = filing_index([
    ("Seq", "Description", "Document", "Type"),
    ("1", "FORM 8-K", "form8k.htm", "8-K"),
    ("2", "INVESTOR PRESENTATION - FOURTH QUARTER", "deck991.htm", "EX-99.1"),
    ("3", "PRESS RELEASE - FOURTH QUARTER FISCAL 2026 RESULTS",
     "earnings992.htm", "EX-99.2"),
])

LEADERSHIP_INDEX = filing_index([
    ("Seq", "Description", "Document", "Type"),
    ("1", "FORM 8-K", "form8k.htm", "8-K"),
    ("2", "PRESS RELEASE - LEADERSHIP APPOINTMENT", "exhibit991.htm", "EX-99.1"),
])

NO_EXHIBIT_INDEX = filing_index([
    ("Seq", "Description", "Document", "Type"),
    ("1", "FORM 8-K", "form8k.htm", "8-K"),
])


def submissions(*filings: dict) -> dict:
    """The SEC submissions payload shape, from a list of filing dicts."""
    keys = {"form": "form", "accession": "accessionNumber",
            "filed": "filingDate", "report_date": "reportDate",
            "items": "items", "description": "primaryDocDescription",
            "document": "primaryDocument"}
    recent = {target: [] for target in keys.values()}
    for filing in filings:
        for source, target in keys.items():
            recent[target].append(filing.get(source, ""))
    return {"filings": {"recent": recent}}


def filing(form: str, accession: str, filed: str, *, items: str = "",
           report_date: str = "", description: str = "",
           document: str = "") -> dict:
    return {"form": form, "accession": accession, "filed": filed,
            "items": items, "report_date": report_date,
            "description": description, "document": document}
