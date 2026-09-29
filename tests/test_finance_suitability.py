"""Phase H.7 — DCF suitability, bound semantics and the period model
(sections 1, 5, 35-38, 62).

The property under test throughout: A BOUND MAY LIMIT AN INPUT, NEVER REPLACE
IT. A model that cannot represent a company must say so, not quietly model a
different company.

Nothing here is issuer-specific.
"""

import pytest

from finance import period_facts as PF
from finance import suitability as SU


# ---------------------------------------------------------------------------
# Section 1 — the period model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("start,end,expected", [
    (None, "2026-06-30", PF.DurationType.INSTANT),
    ("2026-04-01", "2026-06-30", PF.DurationType.QUARTER),
    ("2026-01-01", "2026-06-30", PF.DurationType.HALF_YEAR),
    ("2026-01-01", "2026-09-30", PF.DurationType.YTD_9M),
    ("2025-07-01", "2026-06-30", PF.DurationType.ANNUAL),
    ("2026-01-01", "2026-02-15", PF.DurationType.OTHER),
])
def test_duration_types_are_classified_from_actual_dates(start, end, expected):
    assert PF.classify_duration(start, end) == expected


def test_a_six_month_span_starting_at_the_fiscal_year_is_year_to_date():
    """The one genuinely ambiguous span: a half-year standing alone and a
    six-month year-to-date column are the same length and not the same thing."""
    assert PF.classify_duration("2026-01-01", "2026-06-30",
                                fiscal_year_start="2026-01-01") == PF.DurationType.YTD_6M
    assert PF.classify_duration("2026-01-01", "2026-06-30",
                                fiscal_year_start="2025-07-01") == PF.DurationType.HALF_YEAR


def test_a_53_week_year_is_still_annual():
    assert PF.classify_duration("2025-06-30", "2026-07-05") == PF.DurationType.ANNUAL


def _fact(start, end, value=100, unit="USD", currency="USD"):
    return PF.PeriodFact(field="revenue", value=value, unit=unit, concept="X",
                         accession="a", form="10-Q", filed="2026-08-01",
                         fiscal_year=2026, fiscal_period="Q2", start=start, end=end,
                         currency=currency)


def test_two_adjacent_quarters_are_summable():
    ok, reason = PF.periods_are_summable(
        _fact("2026-01-01", "2026-03-31"), _fact("2026-04-01", "2026-06-30"))
    assert ok is True and reason == ""


def test_overlapping_periods_are_not_summable():
    ok, reason = PF.periods_are_summable(
        _fact("2026-01-01", "2026-03-31"), _fact("2026-03-01", "2026-05-31"))
    assert ok is False and "overlap" in reason


def test_a_cumulative_period_is_never_summed():
    """Section 3: adding Q1 + 6M YTD double-counts the first quarter."""
    ok, reason = PF.periods_are_summable(
        _fact("2026-01-01", "2026-03-31"), _fact("2026-01-01", "2026-06-30"))
    assert ok is False
    assert "cumulative" in reason or "overlap" in reason


def test_an_instant_is_never_summed_with_a_period():
    ok, reason = PF.periods_are_summable(
        _fact(None, "2026-06-30"), _fact("2026-04-01", "2026-06-30"))
    assert ok is False and "point-in-time" in reason


def test_facts_in_different_currencies_are_not_summable():
    ok, reason = PF.periods_are_summable(
        _fact("2026-01-01", "2026-03-31", currency="USD"),
        _fact("2026-04-01", "2026-06-30", currency="EUR"))
    assert ok is False and "currencies" in reason


def test_facts_in_different_units_are_not_summable():
    ok, reason = PF.periods_are_summable(
        _fact("2026-01-01", "2026-03-31", unit="USD"),
        _fact("2026-04-01", "2026-06-30", unit="shares"))
    assert ok is False and "units" in reason


def test_an_amending_form_is_marked_as_amended():
    raw = {"start": "2026-01-01", "end": "2026-03-31", "val": 1, "form": "10-K/A",
           "filed": "2026-06-01", "accn": "a"}
    fact = PF._to_fact("revenue", raw, "X")  # noqa: SLF001
    assert fact.amended is True
    raw["form"] = "10-K"
    assert PF._to_fact("revenue", raw, "X").amended is False  # noqa: SLF001


# ---------------------------------------------------------------------------
# Section 36 — did a bound limit an input, or replace it?
# ---------------------------------------------------------------------------

MARGIN_BOUNDS = (-1.00, 0.60)


def test_a_value_inside_the_bounds_is_not_a_breach():
    assert SU.classify_bound_breach(0.10, MARGIN_BOUNDS, "operating_margin") is None


def test_a_small_breach_limits_the_input():
    breach = SU.classify_bound_breach(0.65, MARGIN_BOUNDS, "operating_margin")
    assert breach is not None
    assert breach["replaces_input"] is False


def test_a_large_breach_replaces_the_input():
    """The failure this exists to catch: an observed value so far outside the
    bound that applying the bound asserts a different company."""
    breach = SU.classify_bound_breach(-0.60, (0.01, 0.60), "operating_margin")
    assert breach is not None
    assert breach["replaces_input"] is True
    assert breach["observed_value"] == pytest.approx(-0.60)
    assert breach["bound"] == pytest.approx(0.01)


def test_an_absent_observation_is_not_a_breach():
    assert SU.classify_bound_breach(None, MARGIN_BOUNDS, "operating_margin") is None


# ---------------------------------------------------------------------------
# Section 37 — the normalization path
# ---------------------------------------------------------------------------

def test_a_normalization_path_starts_where_the_company_actually_is():
    path = SU.normalization_path(-0.60, 0.05, 5)
    assert path[0] == pytest.approx(-0.60)
    assert path[-1] == pytest.approx(0.05)
    assert all(earlier <= later for earlier, later in zip(path, path[1:]))


def test_a_normalization_path_of_one_year_is_the_observation():
    assert SU.normalization_path(-0.60, 0.05, 1) == [pytest.approx(-0.60)]


def test_a_floor_clips_the_path_without_moving_its_start():
    path = SU.normalization_path(0.02, 0.20, 5, floor=0.01)
    assert path[0] == pytest.approx(0.02)
    assert min(path) >= 0.01


# ---------------------------------------------------------------------------
# Section 35 — suitability classification
# ---------------------------------------------------------------------------

def test_a_mature_profitable_company_is_suitable():
    assessment = SU.assess_dcf_suitability(
        revenue=1_000, operating_margin=0.12, free_cash_flow=90,
        operating_cash_flow=140, margin_bounds=MARGIN_BOUNDS,
        terminal_value_share=0.60, historical_comparability="COMPARABLE")
    assert assessment.status == SU.DcfSuitability.SUITABLE
    assert assessment.summary


def test_a_loss_making_company_with_one_weak_year_is_not_blocked():
    """Section 35: do NOT simply block every loss-making company."""
    assessment = SU.assess_dcf_suitability(
        revenue=1_000, operating_margin=-0.05, free_cash_flow=-20,
        operating_cash_flow=10, margin_bounds=MARGIN_BOUNDS,
        terminal_value_share=0.60, negative_periods=1, total_periods=5)
    assert assessment.status in (SU.DcfSuitability.SUITABLE_WITH_HIGH_UNCERTAINTY,
                                 SU.DcfSuitability.LIMITED)
    assert assessment.status != SU.DcfSuitability.NOT_SUITABLE


def test_persistent_cash_burn_with_a_negative_margin_is_at_least_limited():
    assessment = SU.assess_dcf_suitability(
        revenue=1_000, operating_margin=-0.60, free_cash_flow=-600,
        operating_cash_flow=-300, margin_bounds=MARGIN_BOUNDS,
        terminal_value_share=0.80, negative_periods=5, total_periods=5,
        cash_runway_years=1.2)
    assert assessment.status in (SU.DcfSuitability.LIMITED,
                                 SU.DcfSuitability.NOT_SUITABLE)
    codes = {s.code for s in assessment.signals}
    assert "PERSISTENT_NEGATIVE_CASH_GENERATION" in codes


def test_a_bound_that_would_replace_the_margin_is_blocking():
    assessment = SU.assess_dcf_suitability(
        revenue=1_000, operating_margin=-0.60, margin_bounds=(0.01, 0.60))
    codes = {s.code for s in assessment.signals}
    assert SU.DCF_ASSUMPTION_EXCEEDS_MODEL_RANGE in codes
    blocking = [s for s in assessment.signals if s.severity == "blocking"]
    assert blocking


def test_an_extreme_terminal_dependence_is_blocking():
    assessment = SU.assess_dcf_suitability(
        revenue=1_000, operating_margin=0.10, terminal_value_share=0.95,
        margin_bounds=MARGIN_BOUNDS)
    codes = {s.code for s in assessment.signals}
    assert "TERMINAL_VALUE_DOMINANCE" in codes


def test_a_high_but_not_extreme_terminal_dependence_is_only_a_warning():
    assessment = SU.assess_dcf_suitability(
        revenue=1_000, operating_margin=0.10, terminal_value_share=0.80,
        margin_bounds=MARGIN_BOUNDS)
    assert assessment.status == SU.DcfSuitability.SUITABLE_WITH_HIGH_UNCERTAINTY


def test_an_unresolved_share_basis_is_blocking():
    """Every per-share figure depends on the denominator."""
    assessment = SU.assess_dcf_suitability(
        revenue=1_000, operating_margin=0.10, margin_bounds=MARGIN_BOUNDS,
        share_reconciliation_status="MATERIAL_DIFFERENCE")
    codes = {s.code for s in assessment.signals}
    assert "UNRESOLVED_SHARE_BASIS" in codes


def test_material_dilution_is_reported():
    assessment = SU.assess_dcf_suitability(
        revenue=1_000, operating_margin=0.10, margin_bounds=MARGIN_BOUNDS,
        share_dilution=0.25)
    codes = {s.code for s in assessment.signals}
    assert "MATERIAL_DILUTION" in codes


def test_a_structural_break_is_reported_as_a_suitability_signal():
    assessment = SU.assess_dcf_suitability(
        revenue=1_000, operating_margin=0.10, margin_bounds=MARGIN_BOUNDS,
        historical_comparability="STRUCTURAL_BREAK")
    codes = {s.code for s in assessment.signals}
    assert "STRUCTURALLY_CHANGED_BUSINESS" in codes


def test_two_blocking_signals_make_the_instrument_unsuitable():
    assessment = SU.assess_dcf_suitability(
        revenue=1_000, operating_margin=-0.60, margin_bounds=(0.01, 0.60),
        terminal_value_share=0.95, share_reconciliation_status="MATERIAL_DIFFERENCE")
    assert assessment.status == SU.DcfSuitability.NOT_SUITABLE
    assert assessment.blocks_valuation_conclusion is True


def test_suitability_is_independent_of_arithmetic_validity():
    """A valuation can be perfectly valid and completely unsuitable -- that
    is the state the live run was in."""
    assessment = SU.assess_dcf_suitability(
        revenue=1_000, operating_margin=-0.60, margin_bounds=(0.01, 0.60),
        terminal_value_share=0.95, share_reconciliation_status="MATERIAL_DIFFERENCE")
    assert assessment.status == SU.DcfSuitability.NOT_SUITABLE
    # Nothing in the assessment refers to arithmetic at all.
    assert all("arithmetic" not in s.detail.lower() for s in assessment.signals)


def test_the_worst_status_wins_when_several_are_combined():
    assert SU.DcfSuitability.worst([
        SU.DcfSuitability.SUITABLE, SU.DcfSuitability.LIMITED]) == \
        SU.DcfSuitability.LIMITED
    assert SU.DcfSuitability.worst([]) == SU.DcfSuitability.SUITABLE
