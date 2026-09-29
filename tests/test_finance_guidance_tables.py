"""Phase H.14, sections 10-11 / 37 — a guidance table keeps rows AND columns.

Section 37 calls this critical, and the reason is asymmetric. Losing a ROW
loses data, which is visible: a metric the report does not mention. Losing a
COLUMN is worse, because it silently attaches a quarterly figure to a full
year, or the non-GAAP line to the GAAP one, and the output is
indistinguishable from correct guidance.

Measured before the fix, on the five-row two-column table below: sentence
extraction returned ONE metric and matched its period by position rather
than by parsing. Everything else -- the full-year column, both margin rows,
the cash-flow row -- was dropped.

No issuer is named in any production path exercised here; the fixtures are
generic table shapes.
"""

import pytest

from finance.guidance_tables import parse_guidance_table


TWO_COLUMN_TABLE = """Fiscal 2027 Financial Targets

                                    Q3 FY2027              FY2027
Total revenue                  $6.60 - $6.65 billion   $26.0 - $26.2 billion
Subscription revenue           $6.30 - $6.35 billion   $24.8 - $25.0 billion
GAAP operating margin                 34.5%                  35.0%
Non-GAAP operating margin             45.5%                  46.0%
Free cash flow growth                  --                  20% - 22%
"""


def _cell(table, metric, period):
    return next((c for c in table.cells
                 if c.metric_label == metric and c.period_label == period), None)


# ---------------------------------------------------------------------------
# Section 11 — row identity and column identity both survive
# ---------------------------------------------------------------------------

def test_every_row_and_column_is_recovered():
    table = parse_guidance_table(TWO_COLUMN_TABLE)
    assert table.ok
    assert table.period_labels == ["Q3 FY2027", "FY2027"]
    # Five rows, two columns, minus the one deliberately empty cell.
    assert len(table.cells) == 9
    assert table.skipped_rows == []


def test_a_value_stays_with_its_own_column():
    """The failure that matters: a quarter's figure read as the full year."""
    table = parse_guidance_table(TWO_COLUMN_TABLE)
    quarter = _cell(table, "Total revenue", "Q3 FY2027")
    annual = _cell(table, "Total revenue", "FY2027")
    assert (quarter.low, quarter.high) == (6.60, 6.65)
    assert (annual.low, annual.high) == (26.0, 26.2)


def test_a_value_stays_with_its_own_row():
    table = parse_guidance_table(TWO_COLUMN_TABLE)
    total = _cell(table, "Total revenue", "FY2027")
    subscription = _cell(table, "Subscription revenue", "FY2027")
    assert total.low == 26.0
    assert subscription.low == 24.8


def test_gaap_and_non_gaap_rows_stay_distinct():
    """Adjacent rows differing only by basis are the easiest pair to merge."""
    table = parse_guidance_table(TWO_COLUMN_TABLE)
    gaap = _cell(table, "GAAP operating margin", "FY2027")
    non_gaap = _cell(table, "Non-GAAP operating margin", "FY2027")
    assert gaap.low == 35.0
    assert non_gaap.low == 46.0


def test_percentages_and_currency_are_distinguished():
    table = parse_guidance_table(TWO_COLUMN_TABLE)
    revenue = _cell(table, "Total revenue", "FY2027")
    margin = _cell(table, "GAAP operating margin", "FY2027")
    assert revenue.is_percent is False and revenue.scale == "billion"
    assert margin.is_percent is True and margin.scale is None


def test_a_percentage_range_keeps_both_bounds():
    """"20% - 22%" must not be read as the single figure 20."""
    table = parse_guidance_table(TWO_COLUMN_TABLE)
    fcf = _cell(table, "Free cash flow growth", "FY2027")
    assert (fcf.low, fcf.high) == (20.0, 22.0)
    assert fcf.is_percent is True


def test_an_empty_cell_produces_no_value():
    """A metric guided for one column and not the other is normal."""
    table = parse_guidance_table(TWO_COLUMN_TABLE)
    assert _cell(table, "Free cash flow growth", "Q3 FY2027") is None
    assert _cell(table, "Free cash flow growth", "FY2027") is not None


def test_a_title_line_is_not_mistaken_for_the_header():
    """"Fiscal 2027 Financial Targets" names a period and is not a header.

    Taking the first line naming any period picked the title, and then read
    the real header as a data row -- producing one nonsense cell whose value
    was the year itself.
    """
    table = parse_guidance_table(TWO_COLUMN_TABLE)
    assert "Fiscal 2027 Financial Targets" not in [c.metric_label for c in table.cells]
    assert all(c.low != 2027.0 for c in table.cells)


# ---------------------------------------------------------------------------
# Section 11 — fail closed rather than guess
# ---------------------------------------------------------------------------

def test_a_ragged_row_is_skipped_not_guessed():
    """Three values under two columns: the association would be a guess, and
    a guess here attaches a real number to the wrong period."""
    table = parse_guidance_table(
        "            Q3 FY2027        FY2027\n"
        "Total revenue        $6.60 billion   $26.0 billion   $99.0 billion\n")
    assert table.cells == []
    assert table.skipped_rows


def test_a_table_with_no_period_header_is_refused():
    table = parse_guidance_table(
        "Total revenue        $6.60 billion   $26.0 billion\n"
        "Subscription         $6.30 billion   $24.8 billion\n")
    assert not table.ok
    assert "no column header" in table.reason


def test_prose_is_not_parsed_as_a_table():
    table = parse_guidance_table(
        "We expect revenue of $6.60 billion to $6.65 billion in Q3 FY2027.")
    assert not table.ok


def test_a_single_column_table_still_parses():
    table = parse_guidance_table(
        "                     FY2027\n"
        "Total revenue     $26.0 - $26.2 billion\n"
        "Free cash flow    $8.0 - $8.2 billion\n")
    assert table.ok
    assert table.period_labels == ["FY2027"]
    assert len(table.cells) == 2


def test_empty_input_is_handled():
    for value in ("", "   ", None):
        assert parse_guidance_table(value).ok is False


def test_a_repeated_period_is_not_a_header():
    """A sentence naming one period twice is not a two-column header."""
    table = parse_guidance_table(
        "FY2027 revenue and FY2027 margin are discussed below.\n"
        "Total revenue    $26.0 billion   $27.0 billion\n")
    assert not table.ok


# ---------------------------------------------------------------------------
# Metamorphic — the numbers alone decide nothing
# ---------------------------------------------------------------------------

def test_swapping_the_column_headers_swaps_the_periods():
    """Identical values, reversed header order: every cell must follow."""
    swapped = TWO_COLUMN_TABLE.replace(
        "                                    Q3 FY2027              FY2027",
        "                                    FY2027              Q3 FY2027")
    table = parse_guidance_table(swapped)
    assert table.period_labels == ["FY2027", "Q3 FY2027"]
    # The first data column is now the full year.
    assert _cell(table, "Total revenue", "FY2027").low == 6.60
    assert _cell(table, "Total revenue", "Q3 FY2027").low == 26.0
