"""Phase H.6 — TTM construction, on synthetic facts that isolate one rule each.

The live cases live in tests/test_finance_nvda_regression.py and
tests/test_finance_t_regression.py. This file exists for the shapes real
filings only occasionally produce — a broken YTD chain, an amended filing, a
53-week year, a missing quarter — where a synthetic fixture can state the
input exactly and the expected refusal precisely.

The property under test throughout is that a value is called TTM only when a
twelve-month window was actually constructed. Every refusal is a PASS.
"""

import pytest

from finance import ttm as TTM

REVENUE = "RevenueFromContractWithCustomerExcludingAssessedTax"
OCF = "NetCashProvidedByUsedInOperatingActivities"


def _duration(start, end, value, form="10-Q", filed="2026-08-01", accn="0000-26-1"):
    return {"start": start, "end": end, "val": value, "form": form, "filed": filed,
            "accn": accn, "fy": int(end[:4]), "fp": "Q1"}


def _facts(concepts):
    return {"facts": {"us-gaap": {
        concept: {"units": {"USD": rows}} for concept, rows in concepts.items()}}}


def _four_quarters(concept=REVENUE, base=100):
    return _facts({concept: [
        _duration("2025-07-01", "2025-09-30", base),
        _duration("2025-10-01", "2025-12-31", base + 10),
        _duration("2026-01-01", "2026-03-31", base + 20),
        _duration("2026-04-01", "2026-06-30", base + 30),
    ]})


# ---------------------------------------------------------------------------
# Four discrete quarters — the simple construction
# ---------------------------------------------------------------------------

def test_four_contiguous_quarters_sum_to_a_valid_ttm():
    result = TTM.build_ttm(_four_quarters(), "revenue")
    assert result.validation_status == TTM.TtmValidation.VALID
    assert result.value == pytest.approx(100 + 110 + 120 + 130)
    assert result.start_date == "2025-07-01"
    assert result.end_date == "2026-06-30"
    assert result.construction_method == TTM.TtmConstruction.FOUR_DISCRETE_QUARTERS
    assert len(result.quarters_included) == 4
    assert result.source_accessions


def test_three_quarters_are_refused_rather_than_scaled_up():
    facts = _facts({REVENUE: [
        _duration("2025-10-01", "2025-12-31", 100),
        _duration("2026-01-01", "2026-03-31", 110),
        _duration("2026-04-01", "2026-06-30", 120),
    ]})
    result = TTM.build_ttm(facts, "revenue")
    assert result.validation_status == TTM.TtmValidation.INVALID
    assert TTM.TTM_INVALID_PERIOD_RECONSTRUCTION in result.reason
    assert result.value is None


def test_a_missing_middle_quarter_is_never_silently_filled():
    facts = _facts({REVENUE: [
        _duration("2025-04-01", "2025-06-30", 90),
        _duration("2025-07-01", "2025-09-30", 100),
        # Q4 missing entirely.
        _duration("2026-01-01", "2026-03-31", 120),
        _duration("2026-04-01", "2026-06-30", 130),
    ]})
    result = TTM.build_ttm(facts, "revenue")
    assert result.validation_status == TTM.TtmValidation.INVALID
    assert "gap" in result.reason


def test_overlapping_quarters_are_refused_rather_than_double_counted():
    facts = _facts({REVENUE: [
        _duration("2025-07-01", "2025-09-30", 100),
        _duration("2025-09-01", "2025-11-30", 110),   # overlaps the previous
        _duration("2025-12-01", "2026-02-28", 120),
        _duration("2026-03-01", "2026-05-31", 130),
    ]})
    result = TTM.build_ttm(facts, "revenue")
    assert result.validation_status == TTM.TtmValidation.INVALID
    assert "OVERLAP" in result.reason or "gap" in result.reason


def test_a_fifty_three_week_year_is_still_a_valid_twelve_months():
    """52/53-week fiscal calendars stretch a year to ~371 days."""
    facts = _facts({REVENUE: [
        _duration("2025-06-30", "2025-09-28", 100),
        _duration("2025-09-29", "2025-12-28", 110),
        _duration("2025-12-29", "2026-03-29", 120),
        _duration("2026-03-30", "2026-07-05", 130),   # the 14-week quarter
    ]})
    result = TTM.build_ttm(facts, "revenue")
    assert result.validation_status == TTM.TtmValidation.VALID
    assert 350 <= result.span_days <= 380


def test_quarters_spanning_more_than_a_year_are_refused():
    facts = _facts({REVENUE: [
        _duration("2025-01-01", "2025-03-31", 100),
        _duration("2025-04-01", "2025-06-30", 110),
        _duration("2025-07-01", "2025-09-30", 120),
        _duration("2025-10-01", "2026-06-30", 130),   # nine months, not one
    ]})
    result = TTM.build_ttm(facts, "revenue")
    assert result.validation_status == TTM.TtmValidation.INVALID


# ---------------------------------------------------------------------------
# YTD cumulative facts (section 2)
# ---------------------------------------------------------------------------

def test_ytd_facts_are_differenced_rather_than_summed():
    """A 10-Q's cash-flow statement is year-to-date. Summing Q1YTD + H1YTD +
    9MYTD + FY would count Q1 four times."""
    facts = _facts({OCF: [
        _duration("2025-01-01", "2025-03-31", 100),   # Q1 YTD
        _duration("2025-01-01", "2025-06-30", 250),   # H1 YTD -> Q2 = 150
        _duration("2025-01-01", "2025-09-30", 400),   # 9M YTD -> Q3 = 150
        _duration("2025-01-01", "2025-12-31", 600, form="10-K"),   # FY -> Q4 = 200
    ]})
    result = TTM.build_ttm(facts, "operating_cash_flow")
    assert result.validation_status == TTM.TtmValidation.VALID
    # The whole fiscal year, not the sum of the cumulative columns (1350).
    assert result.value == pytest.approx(600)
    assert result.used_reconstruction is True


def test_a_broken_ytd_chain_is_refused_rather_than_bridged():
    """Q3YTD - Q1YTD is two quarters of activity, not one."""
    facts = _facts({OCF: [
        _duration("2025-01-01", "2025-03-31", 100),
        # H1 missing.
        _duration("2025-01-01", "2025-09-30", 400),
        _duration("2025-01-01", "2025-12-31", 600, form="10-K"),
    ]})
    result = TTM.build_ttm(facts, "operating_cash_flow")
    assert result.validation_status == TTM.TtmValidation.INVALID


def test_a_six_month_ytd_is_never_mistaken_for_a_quarter():
    facts = _facts({REVENUE: [
        _duration("2026-01-01", "2026-06-30", 500),
        _duration("2025-01-01", "2025-06-30", 450),
    ]})
    result = TTM.build_ttm(facts, "revenue")
    assert result.validation_status == TTM.TtmValidation.INVALID


# ---------------------------------------------------------------------------
# The annual roll-forward (section 1)
# ---------------------------------------------------------------------------

def _roll_forward_facts(quarter="Q1"):
    """A fiscal year plus one, two or three newly reported quarters — the Q1,
    Q2 and Q3 roll-forward cases section 1 requires."""
    ytd = {"Q1": [("2026-03-31", 120), ("2025-03-31", 100)],
           "Q2": [("2026-06-30", 250), ("2025-06-30", 210)],
           "Q3": [("2026-09-30", 390), ("2025-09-30", 330)]}[quarter]
    rows = [
        _duration("2025-01-01", "2025-12-31", 460, form="10-K"),   # FY2025
    ]
    for end, value in ytd:
        rows.append(_duration(f"{end[:4]}-01-01", end, value))
    return _facts({REVENUE: rows})


@pytest.mark.parametrize("quarter,expected", [
    ("Q1", 460 + 120 - 100),
    ("Q2", 460 + 250 - 210),
    ("Q3", 460 + 390 - 330),
])
def test_the_roll_forward_works_for_q1_q2_and_q3(quarter, expected):
    """TTM = latest fiscal year + current year-to-date - the SAME year-to-date
    period of the prior fiscal year."""
    result, reason = TTM._from_annual_roll_forward(  # noqa: SLF001
        _roll_forward_facts(quarter), "revenue")
    assert result is not None, reason
    assert result.value == pytest.approx(expected)
    assert result.construction_method == TTM.TtmConstruction.ANNUAL_ROLL_FORWARD
    assert result.validation_status == TTM.TtmValidation.VALID
    assert len(result.quarters_included) == 3


def test_the_roll_forward_refuses_a_mismatched_prior_period():
    """Subtracting a prior-year H1 from a current-year Q1 is not twelve
    months; it is thirteen and a half."""
    facts = _facts({REVENUE: [
        _duration("2025-01-01", "2025-12-31", 460, form="10-K"),
        _duration("2026-01-01", "2026-03-31", 120),      # current Q1
        _duration("2025-01-01", "2025-06-30", 210),      # prior H1 only
    ]})
    result, reason = TTM._from_annual_roll_forward(facts, "revenue")  # noqa: SLF001
    assert result is None
    assert "SAME year-to-date period" in reason


def test_the_roll_forward_refuses_when_nothing_has_been_reported_since():
    facts = _facts({REVENUE: [
        _duration("2025-01-01", "2025-12-31", 460, form="10-K"),
    ]})
    result, reason = TTM._from_annual_roll_forward(facts, "revenue")  # noqa: SLF001
    assert result is None
    assert "nothing to roll forward" in reason


def test_a_non_calendar_fiscal_year_rolls_forward_correctly():
    """NVIDIA's fiscal year ends in late January. Nothing here may assume a
    December year end."""
    facts = _facts({REVENUE: [
        _duration("2025-01-27", "2026-01-25", 2159, form="10-K"),
        _duration("2026-01-26", "2026-04-26", 816),
        _duration("2025-01-27", "2025-04-27", 440),
    ]})
    result, reason = TTM._from_annual_roll_forward(facts, "revenue")  # noqa: SLF001
    assert result is not None, reason
    assert result.value == pytest.approx(2159 + 816 - 440)
    assert result.end_date == "2026-04-26"


# ---------------------------------------------------------------------------
# Amended filings and duplicates
# ---------------------------------------------------------------------------

def test_an_amended_filing_supersedes_the_original_for_the_same_period():
    facts = _facts({REVENUE: [
        _duration("2025-07-01", "2025-09-30", 100),
        _duration("2025-10-01", "2025-12-31", 110),
        _duration("2026-01-01", "2026-03-31", 120, filed="2026-05-01"),
        _duration("2026-01-01", "2026-03-31", 125, form="10-Q/A", filed="2026-06-01"),
        _duration("2026-04-01", "2026-06-30", 130),
    ]})
    result = TTM.build_ttm(facts, "revenue")
    assert result.validation_status == TTM.TtmValidation.VALID
    assert result.value == pytest.approx(100 + 110 + 125 + 130)


def test_the_same_period_reported_twice_is_counted_once():
    """Every 10-Q restates the prior-year comparative; the same period
    appearing in several filings must not multiply."""
    rows = []
    for start, end, value in (("2025-07-01", "2025-09-30", 100),
                              ("2025-10-01", "2025-12-31", 110),
                              ("2026-01-01", "2026-03-31", 120),
                              ("2026-04-01", "2026-06-30", 130)):
        rows.append(_duration(start, end, value, filed="2026-05-01"))
        rows.append(_duration(start, end, value, filed="2026-08-01"))
    result = TTM.build_ttm(_facts({REVENUE: rows}), "revenue")
    assert result.validation_status == TTM.TtmValidation.VALID
    assert result.value == pytest.approx(460)


# ---------------------------------------------------------------------------
# Trailing-ness (section 3's end-date invariant)
# ---------------------------------------------------------------------------

def test_a_window_far_behind_the_latest_reported_period_is_partial():
    """A valid twelve months that ends six months early is not a CURRENT
    trailing twelve months, and must not be combined with figures that are."""
    facts = _four_quarters()
    result = TTM.build_ttm(facts, "revenue", reference_end="2026-12-31")
    assert result.validation_status == TTM.TtmValidation.PARTIAL
    assert result.ok is True                       # still usable, with a caveat
    assert "NOT a current one" in result.reason


def test_a_window_ending_at_the_reference_period_is_valid():
    result = TTM.build_ttm(_four_quarters(), "revenue", reference_end="2026-06-30")
    assert result.validation_status == TTM.TtmValidation.VALID
    assert result.reason is None


def test_the_latest_reported_period_is_taken_from_the_income_statement():
    """A company whose cash-flow tagging lags must not make its own revenue
    look stale."""
    facts = _facts({
        REVENUE: [
            _duration("2025-07-01", "2025-09-30", 100),
            _duration("2025-10-01", "2025-12-31", 110),
            _duration("2026-01-01", "2026-03-31", 120),
            _duration("2026-04-01", "2026-06-30", 130),
        ],
        OCF: [_duration("2025-01-01", "2025-12-31", 400, form="10-K")],
    })
    assert TTM.latest_reported_period_end(facts) == "2026-06-30"


# ---------------------------------------------------------------------------
# Section 4 — balance-sheet fields are refused outright
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field_name", [
    "cash_and_cash_equivalents", "short_term_debt", "long_term_debt",
    "stockholders_equity", "current_assets",
])
def test_no_balance_sheet_field_is_ever_given_a_ttm(field_name):
    result = TTM.build_ttm({}, field_name)
    assert result.validation_status == TTM.TtmValidation.INVALID
    assert "point-in-time" in result.reason


def test_an_unmapped_field_is_refused_rather_than_guessed():
    result = TTM.build_ttm({}, "not_a_real_field")
    assert result.validation_status == TTM.TtmValidation.INVALID
    assert "no reviewed XBRL concept mapping" in result.reason


# ---------------------------------------------------------------------------
# Period comparability, for callers that combine two windows
# ---------------------------------------------------------------------------

def test_two_windows_over_the_same_period_are_comparable():
    left = TTM.build_ttm(_four_quarters(REVENUE, 100), "revenue")
    right = TTM.build_ttm(_four_quarters(REVENUE, 100), "revenue")
    assert TTM.periods_are_comparable(left, right) is True


def test_two_windows_over_different_periods_are_not_comparable():
    """The AT&T free-cash-flow failure in one assertion."""
    left = TTM.build_ttm(_four_quarters(), "revenue")
    shifted = _facts({REVENUE: [
        _duration("2025-01-01", "2025-03-31", 100),
        _duration("2025-04-01", "2025-06-30", 110),
        _duration("2025-07-01", "2025-09-30", 120),
        _duration("2025-10-01", "2025-12-31", 130),
    ]})
    right = TTM.build_ttm(shifted, "revenue")
    assert left.ok and right.ok
    assert left.end_date != right.end_date
    assert TTM.periods_are_comparable(left, right) is False


def test_an_invalid_window_is_never_comparable_with_anything():
    valid = TTM.build_ttm(_four_quarters(), "revenue")
    invalid = TTM.build_ttm({}, "revenue")
    assert TTM.periods_are_comparable(valid, invalid) is False
    assert TTM.periods_are_comparable(None, valid) is False
