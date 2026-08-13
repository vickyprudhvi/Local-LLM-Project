"""Phase H.4 — period selection and TTM construction (spec sections 3, 15).

Synthetic company_facts throughout: every case below is a SHAPE that real
filings take, isolated so the behaviour is pinned without depending on any
one issuer continuing to file that way. The real-data counterparts live in
tests/test_finance_aos_regression.py.

The rule these tests exist to hold: TTM FAILS CLOSED. A trailing-twelve-month
figure that silently sums three quarters, or sums across a gap, or sums two
overlapping year-to-date columns, is indistinguishable from a correct one by
inspection — so every one of those must produce no TTM at all rather than a
plausible wrong number.
"""

import pytest

from finance import period_facts as pf
from finance.freshness import DcfFreshnessPlanner, build_ttm


def _facts(concept_rows: dict) -> dict:
    """Build a companyfacts payload from {concept: [fact, ...]}."""
    return {"facts": {"us-gaap": {
        concept: {"units": {"USD": rows}} for concept, rows in concept_rows.items()}}}


def _duration(start, end, val, *, fy=2026, fp="Q1", form="10-Q", filed=None, accn="a-1"):
    return {"start": start, "end": end, "val": val, "fy": fy, "fp": fp,
            "form": form, "filed": filed or end, "accn": accn}


def _instant(end, val, *, fy=2026, fp="Q2", form="10-Q", filed=None, accn="a-1"):
    return {"end": end, "val": val, "fy": fy, "fp": fp, "form": form,
            "filed": filed or end, "accn": accn}


REVENUE = "RevenueFromContractWithCustomerExcludingAssessedTax"
OCF = "NetCashProvidedByUsedInOperatingActivities"


# ---------------------------------------------------------------------------
# Discrete quarters: directly reported, YTD, and the Q4 gap
# ---------------------------------------------------------------------------

def test_directly_reported_quarters_are_used_without_arithmetic():
    facts = _facts({REVENUE: [
        _duration("2025-01-01", "2025-03-31", 100),
        _duration("2025-04-01", "2025-06-30", 110),
    ]})
    series = pf.discrete_quarters(facts, "revenue")
    assert [q.value for q in series.quarters] == [100, 110]
    assert series.used_reconstruction is False
    assert all(not q.reconstructed_from for q in series.quarters)


def test_q2_is_reconstructed_from_a_half_year_ytd_column():
    """A 10-Q cash-flow statement is year-to-date, not per quarter. Summing
    the YTD columns would double-count Q1."""
    facts = _facts({OCF: [
        _duration("2025-01-01", "2025-03-31", 40),    # Q1 YTD == discrete Q1
        _duration("2025-01-01", "2025-06-30", 95),    # H1 YTD
    ]})
    series = pf.discrete_quarters(facts, "operating_cash_flow")
    values = {q.end: q.value for q in series.quarters}
    assert values["2025-03-31"] == 40
    assert values["2025-06-30"] == pytest.approx(55)   # 95 - 40, not 95
    assert series.used_reconstruction is True


def test_q3_is_reconstructed_from_the_nine_month_ytd_column():
    facts = _facts({OCF: [
        _duration("2025-01-01", "2025-03-31", 40),
        _duration("2025-01-01", "2025-06-30", 95),
        _duration("2025-01-01", "2025-09-30", 150),
    ]})
    series = pf.discrete_quarters(facts, "operating_cash_flow")
    values = {q.end: q.value for q in series.quarters}
    assert values["2025-09-30"] == pytest.approx(55)   # 150 - 95


def test_q4_is_reconstructed_from_the_annual_figure():
    """No company files a Q4 10-Q. Without this, every TTM window spanning a
    year end has a hole and no TTM can be built for three quarters of the
    year — which is exactly what happened to AOS before this was added."""
    facts = _facts({REVENUE: [
        _duration("2025-01-01", "2025-03-31", 100),
        _duration("2025-01-01", "2025-06-30", 210),
        _duration("2025-01-01", "2025-09-30", 320),
        _duration("2025-01-01", "2025-12-31", 450, fp="FY", form="10-K",
                  filed="2026-02-10"),
    ]})
    series = pf.discrete_quarters(facts, "revenue")
    values = {q.end: q.value for q in series.quarters}
    assert values["2025-12-31"] == pytest.approx(130)   # 450 - 320
    q4 = next(q for q in series.quarters if q.end == "2025-12-31")
    assert q4.reconstructed_from, "a reconstructed quarter must say so"


def test_a_broken_ytd_chain_fails_closed_rather_than_differencing_two_quarters():
    """With Q1 missing, H1 minus nothing is not a quarter. The chain stops."""
    facts = _facts({OCF: [
        _duration("2025-01-01", "2025-06-30", 95),
        _duration("2025-01-01", "2025-09-30", 150),
    ]})
    series = pf.discrete_quarters(facts, "operating_cash_flow")
    ends = {q.end for q in series.quarters}
    assert "2025-06-30" not in ends, "a two-quarter span must not become one quarter"


def test_a_later_filing_supersedes_an_earlier_one_for_the_same_period():
    """How a 10-K/A restatement correctly replaces the original."""
    facts = _facts({REVENUE: [
        _duration("2025-01-01", "2025-03-31", 100, filed="2025-04-30"),
        _duration("2025-01-01", "2025-03-31", 105, filed="2025-11-01", form="10-Q/A"),
    ]})
    series = pf.discrete_quarters(facts, "revenue")
    assert [q.value for q in series.quarters] == [105]


def test_a_period_type_mismatch_is_refused_for_instant_fields():
    facts = _facts({"Assets": [_instant("2026-06-30", 500)]})
    series = pf.discrete_quarters(facts, "assets")
    assert series.quarters == []
    assert series.warnings


# ---------------------------------------------------------------------------
# Balance-sheet instants
# ---------------------------------------------------------------------------

def test_latest_instant_prefers_the_newest_period_not_the_newest_filing():
    """Every 10-Q restates the prior year end as its comparative column, so
    the newest FILING routinely contains an OLD balance."""
    facts = _facts({"CashAndCashEquivalentsAtCarryingValue": [
        _instant("2025-12-31", 174, filed="2026-07-30"),   # comparative
        _instant("2026-06-30", 181, filed="2026-07-30"),   # current
    ]})
    latest = pf.latest_instant(facts, "cash_and_cash_equivalents")
    assert latest.value == 181
    assert latest.end == "2026-06-30"


def test_instant_as_of_requires_the_exact_balance_sheet_date():
    """A balance sheet is one moment. A field not reported at that date is
    ABSENT, never back-filled from an older filing — the rule that stops a
    2010 short-term-borrowings figure entering a 2026 net-debt bridge."""
    facts = _facts({"ShortTermBorrowings": [_instant("2010-09-30", 158)]})
    assert pf.latest_instant(facts, "short_term_debt").end == "2010-09-30"
    assert pf.instant_as_of(facts, "short_term_debt", "2026-06-30") is None


def test_the_winning_concept_is_the_one_still_in_use():
    """Precedence order alone picks the first candidate that has ANY fact,
    which for a long-abandoned tag means a decade-old value. Recency wins;
    precedence only breaks ties."""
    facts = _facts({
        "ShortTermBorrowings": [_instant("2010-09-30", 158)],
        "DebtCurrent": [_instant("2026-06-30", 42)],
    })
    latest = pf.latest_instant(facts, "short_term_debt")
    assert latest.value == 42
    assert latest.concept == "DebtCurrent"


# ---------------------------------------------------------------------------
# TTM roll-forward
# ---------------------------------------------------------------------------

def _four_quarters(values, year_start=2025):
    ends = [("2025-09-30", "2025-07-01"), ("2025-12-31", "2025-10-01"),
            ("2026-03-31", "2026-01-01"), ("2026-06-30", "2026-04-01")]
    return _facts({REVENUE: [
        _duration(start, end, value) for (end, start), value in zip(ends, values)]})


def test_ttm_sums_exactly_four_contiguous_quarters():
    result = build_ttm(_four_quarters([100, 110, 120, 130]), "revenue")
    assert result.ok is True
    assert result.value == pytest.approx(460)
    assert result.period_start == "2025-07-01"
    assert result.period_end == "2026-06-30"


def test_ttm_refuses_when_only_three_quarters_exist():
    facts = _facts({REVENUE: [
        _duration("2026-01-01", "2026-03-31", 100),
        _duration("2026-04-01", "2026-06-30", 110),
    ]})
    result = build_ttm(facts, "revenue")
    assert result.ok is False
    assert "four are required" in result.reason


def test_ttm_refuses_across_a_gap_rather_than_skipping_a_quarter():
    """The dangerous case: four quarters exist, but not four CONSECUTIVE
    ones. Summing them understates a year and looks completely normal."""
    facts = _facts({REVENUE: [
        _duration("2024-01-01", "2024-03-31", 90),
        _duration("2025-10-01", "2025-12-31", 110),
        _duration("2026-01-01", "2026-03-31", 120),
        _duration("2026-04-01", "2026-06-30", 130),
    ]})
    result = build_ttm(facts, "revenue")
    assert result.ok is False
    assert "not contiguous" in result.reason


def test_ttm_offset_gives_the_prior_year_window_for_growth_comparison():
    ends = [("2024-09-30", "2024-07-01"), ("2024-12-31", "2024-10-01"),
            ("2025-03-31", "2025-01-01"), ("2025-06-30", "2025-04-01"),
            ("2025-09-30", "2025-07-01"), ("2025-12-31", "2025-10-01"),
            ("2026-03-31", "2026-01-01"), ("2026-06-30", "2026-04-01")]
    values = [80, 85, 90, 95, 100, 110, 120, 130]
    facts = _facts({REVENUE: [_duration(s, e, v) for (e, s), v in zip(ends, values)]})

    current = build_ttm(facts, "revenue")
    prior = build_ttm(facts, "revenue", offset=4)
    assert current.value == pytest.approx(460)
    assert prior.value == pytest.approx(350)
    assert prior.period_end == "2025-06-30"


def test_annual_only_filer_gets_the_annual_figure_not_a_fabricated_ttm():
    facts = _facts({REVENUE: [
        _duration("2024-01-01", "2024-12-31", 400, fp="FY", form="10-K"),
        _duration("2025-01-01", "2025-12-31", 450, fp="FY", form="10-K"),
    ]})
    planner = DcfFreshnessPlanner(facts, "TEST")
    selection = planner.select_flow("revenue", reference_date="2025-12-31")
    assert selection.source == "annual_sec_filing"
    assert selection.value == pytest.approx(450)
    assert "No trailing-twelve-month figure was built" in selection.derivation


def test_a_ttm_window_that_is_not_actually_trailing_is_refused():
    """Four contiguous quarters spanning 365 days, from a decade ago. Every
    structural check passes; "trailing" is the property that does not — and
    without this guard AMZN's capital expenditure resolved to a 2016-2017
    window as its current TTM."""
    ends = [("2016-09-30", "2016-07-01"), ("2016-12-31", "2016-10-01"),
            ("2017-03-31", "2017-01-01"), ("2017-06-30", "2017-04-01")]
    facts = _facts({REVENUE: [_duration(s, e, 100) for e, s in ends]})
    planner = DcfFreshnessPlanner(facts, "TEST")

    assert build_ttm(facts, "revenue").ok is True          # structurally fine
    selection = planner.select_flow("revenue", reference_date="2026-06-30")
    assert selection.value is None                          # ...but not trailing
    assert "not a trailing twelve months" in (selection.derivation or "")


def test_a_non_calendar_fiscal_year_rolls_normally():
    """A June-year-end filer. Nothing here keys on calendar quarters."""
    ends = [("2025-09-30", "2025-07-01"), ("2025-12-31", "2025-10-01"),
            ("2026-03-31", "2026-01-01"), ("2026-06-30", "2026-04-01")]
    facts = _facts({REVENUE: [
        _duration(s, e, 100, fy=2026, fp="Q1") for e, s in ends]})
    result = build_ttm(facts, "revenue")
    assert result.ok is True
    assert result.value == pytest.approx(400)


def test_duplicate_facts_for_one_period_are_collapsed_not_double_counted():
    facts = _facts({REVENUE: [
        _duration("2026-01-01", "2026-03-31", 100, filed="2026-04-30", accn="a-1"),
        _duration("2026-01-01", "2026-03-31", 100, filed="2026-04-30", accn="a-1"),
    ]})
    series = pf.discrete_quarters(facts, "revenue")
    assert len(series.quarters) == 1


def test_a_unit_this_project_does_not_handle_is_treated_as_absent():
    facts = {"facts": {"us-gaap": {REVENUE: {"units": {"EUR": [
        _duration("2026-01-01", "2026-03-31", 100)]}}}}}
    assert pf.discrete_quarters(facts, "revenue").quarters == []
    assert pf.annual_periods(facts, "revenue") == []
