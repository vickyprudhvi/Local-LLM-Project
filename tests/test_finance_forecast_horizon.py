"""Forecast-horizon compatibility: comparable magnitude vs. direction only.

Spec §4 already says compatibility is a property of a PAIR **and** an
OPERATION. This file adds the pair that was missing: forward evidence against
a FORECAST ASSUMPTION, where the operation is "set or check the assumption's
NUMERIC MAGNITUDE".

A quarterly year-over-year revenue growth rate can be perfectly correct — the
right metric, the right comparison quarter, a validated derivation — and
still not be a number that may be compared against an ANNUAL model bound. A
company can grow 84% in one quarter against an easy comparable and 32% over
twelve months. Both are true. Only one of them is the same KIND of quantity
as a year-1 annual growth assumption.

The live failure this responds to: an annual model bound of 25%, trailing
twelve-month growth of 32%, and next-quarter guidance implying 84%. The bound
conflict was real — 32% > 25% — but it was reported as though the model were
59 percentage points wrong, because the quarterly figure was used as the
corroborating magnitude. The conclusion was right and the arithmetic behind
it described a different period.

So evidence now carries `forecast_compatibility`:

    ASSUMPTION_COMPARABLE       may set or check a magnitude
    DIRECTIONAL_CORROBORATION   may support "growth remains elevated"
    NOT_COMPARABLE              neither

Nothing is discarded. A quarterly guide keeps its metric, its target period,
its comparison period, its derived growth rate and its evidence ids, and
stays citable research evidence. Only the OPERATIONS it is eligible for
change.
"""

import pytest

from finance import forward_assumptions as FA
from finance import semantics as sem

FC = sem.ForecastCompatibility
BOUNDS = FA.GROWTH_BOUNDS


# ---------------------------------------------------------------------------
# 1. A quarterly guide against its own prior-year quarter is a VALID growth
# ---------------------------------------------------------------------------

def test_a_quarter_guide_against_the_prior_year_quarter_is_a_valid_growth():
    """The derivation is not in question. Same metric, same quarter, one year
    apart -- spec §5's rule for a growth rate is satisfied."""
    guided = sem.SemanticFact(
        metric_id=sem.MetricIdentity.REVENUE, value=29_400.0, units="currency",
        period_frequency=sem.PeriodFrequency.QUARTER, fiscal_quarter=3,
        flow_or_instant=sem.FlowOrInstant.FLOW,
        current_or_historical=sem.CurrentOrHistorical.FORWARD)
    prior = sem.SemanticFact(
        metric_id=sem.MetricIdentity.REVENUE, value=15_950.0, units="currency",
        period_frequency=sem.PeriodFrequency.QUARTER, fiscal_quarter=3,
        flow_or_instant=sem.FlowOrInstant.FLOW,
        current_or_historical=sem.CurrentOrHistorical.HISTORICAL)
    assert sem.compatible_for(sem.Operation.GROWTH, guided, prior).allowed


# ---------------------------------------------------------------------------
# 2-3. The same figure: rejected for magnitude, allowed for direction
# ---------------------------------------------------------------------------

def _quarterly_vs_annual():
    return sem.forecast_compatibility(
        evidence_metric=sem.MetricIdentity.REVENUE,
        evidence_frequency=sem.PeriodFrequency.QUARTER,
        assumption_metric=sem.MetricIdentity.REVENUE,
        assumption_frequency=sem.PeriodFrequency.ANNUAL)


def test_quarterly_growth_may_not_set_an_annual_assumption_magnitude():
    eligibility = _quarterly_vs_annual()
    assert eligibility.status == FC.DIRECTIONAL_CORROBORATION
    assert not eligibility.may_set_magnitude


def test_quarterly_growth_may_corroborate_a_direction():
    eligibility = _quarterly_vs_annual()
    assert eligibility.may_corroborate_direction
    assert eligibility.compatible_horizon == sem.PeriodFrequency.QUARTER


@pytest.mark.parametrize("frequency", [sem.PeriodFrequency.ANNUAL,
                                       sem.PeriodFrequency.TTM])
def test_a_twelve_month_figure_is_assumption_comparable(frequency):
    eligibility = sem.forecast_compatibility(
        evidence_metric=sem.MetricIdentity.REVENUE, evidence_frequency=frequency,
        assumption_metric=sem.MetricIdentity.REVENUE,
        assumption_frequency=sem.PeriodFrequency.ANNUAL)
    assert eligibility.status == FC.ASSUMPTION_COMPARABLE
    assert eligibility.may_set_magnitude


def test_a_different_metric_is_not_comparable_at_any_horizon():
    """Horizon compatibility is the SECOND test, not the only one. A
    twelve-month EBITDA growth rate is not a revenue-growth assumption."""
    eligibility = sem.forecast_compatibility(
        evidence_metric=sem.MetricIdentity.EBITDA,
        evidence_frequency=sem.PeriodFrequency.ANNUAL,
        assumption_metric=sem.MetricIdentity.REVENUE,
        assumption_frequency=sem.PeriodFrequency.ANNUAL)
    assert eligibility.status == FC.NOT_COMPARABLE
    assert not eligibility.may_set_magnitude
    assert not eligibility.may_corroborate_direction


def test_an_unknown_horizon_is_refused_rather_than_assumed():
    """Fail closed, as §4 requires of every unrecognised case."""
    eligibility = sem.forecast_compatibility(
        evidence_metric=sem.MetricIdentity.REVENUE,
        evidence_frequency=sem.PeriodFrequency.UNKNOWN,
        assumption_metric=sem.MetricIdentity.REVENUE,
        assumption_frequency=sem.PeriodFrequency.ANNUAL)
    assert eligibility.status == FC.NOT_COMPARABLE


# ---------------------------------------------------------------------------
# 4-7. The model-bound conflict is established on comparable evidence only
# ---------------------------------------------------------------------------

def _evidence(**kwargs):
    return FA.GrowthEvidence(**kwargs)


def test_quarterly_guidance_alone_cannot_create_an_annual_bound_conflict():
    """Case 4. The quarterly figure is far past the bound and the comparable
    twelve-month evidence is inside it. The model is not being constrained by
    anything a year-1 annual assumption is measured in."""
    evidence = _evidence(ttm_yoy=0.18,
                         guidance_implied_next_period_growth=0.84,
                         guidance_implied_comparison_period="2025-05-05..2025-08-03")
    assert FA.detect_model_bound_conflict(
        evidence, raw_value=0.18, applied_value=0.18, bounds=BOUNDS) is None


def test_comparable_ttm_growth_past_the_bound_creates_the_conflict():
    """Case 5. Twelve-month growth of 32% against a 25% ceiling: the same
    kind of quantity, and the bound is setting the forecast."""
    evidence = _evidence(ttm_yoy=0.323)
    conflict = FA.detect_model_bound_conflict(
        evidence, raw_value=0.323, applied_value=0.25, bounds=BOUNDS)
    assert conflict is not None
    assert conflict["code"] == FA.DCF_MODEL_BOUND_CONFLICT
    assert conflict["corroborating_value"] == pytest.approx(0.323)


def test_a_stronger_quarterly_guide_does_not_change_the_conflicts_magnitude():
    """Case 6, and the heart of it. The conflict stands on the comparable
    evidence; the quarterly figure appears as directional support and its
    magnitude drives nothing."""
    comparable_only = FA.detect_model_bound_conflict(
        _evidence(ttm_yoy=0.323), 0.323, 0.25, BOUNDS)
    with_quarter = FA.detect_model_bound_conflict(
        _evidence(ttm_yoy=0.323, guidance_implied_next_period_growth=0.84,
                  guidance_implied_comparison_period="2025-05-05..2025-08-03"),
        0.323, 0.25, BOUNDS)

    assert with_quarter["corroborating_value"] == comparable_only["corroborating_value"]
    assert with_quarter["severity"] == comparable_only["severity"]
    assert with_quarter["model_bound"] == comparable_only["model_bound"]
    # The quarterly figure is PRESERVED, as supporting context.
    support = with_quarter.get("directional_support")
    assert support and support["value"] == pytest.approx(0.84)
    assert support["forecast_compatibility"] == FC.DIRECTIONAL_CORROBORATION
    # ...and its magnitude never reaches the sentence a reader is given.
    assert "84" not in with_quarter["message"]


def test_full_year_guidance_past_the_bound_creates_the_conflict():
    """Case 7. An annual guide is the same horizon as the assumption."""
    evidence = _evidence(guidance_low=0.40, guidance_high=0.44,
                         guidance_period_type="annual",
                         guidance_source_metric="revenue_growth")
    conflict = FA.detect_model_bound_conflict(
        evidence, raw_value=0.42, applied_value=0.25, bounds=BOUNDS)
    assert conflict is not None
    assert conflict["corroborating_value"] == pytest.approx(0.42)


def test_quarterly_revenue_growth_guidance_is_also_only_directional():
    """A quarterly guide stated as a RATE, not derived from a level. Same
    horizon problem, so the same eligibility."""
    evidence = _evidence(guidance_low=0.80, guidance_high=0.88,
                         guidance_period_type="quarter",
                         guidance_source_metric="revenue_growth",
                         ttm_yoy=0.18)
    assert FA.detect_model_bound_conflict(evidence, 0.18, 0.18, BOUNDS) is None


def test_the_conflict_names_which_evidence_established_it():
    """A reader has to be able to check the claim, which means knowing which
    figure it rests on and over what period."""
    conflict = FA.detect_model_bound_conflict(_evidence(ttm_yoy=0.323), 0.323, 0.25, BOUNDS)
    assert conflict["corroborating_forecast_compatibility"] == FC.ASSUMPTION_COMPARABLE
    assert conflict["corroborating_horizon"] in (sem.PeriodFrequency.TTM,
                                                 sem.PeriodFrequency.ANNUAL)


# ---------------------------------------------------------------------------
# Nothing is discarded
# ---------------------------------------------------------------------------

def test_a_quarterly_guide_keeps_every_field_it_arrived_with():
    """"Restrict the operations, not the evidence." The figure stays whole."""
    evidence = _evidence(guidance_implied_next_period_growth=0.84,
                         guidance_implied_comparison_period="2025-05-05..2025-08-03",
                         guidance_evidence_id="dcf.guidance.revenue.current",
                         guidance_period_label="Q3 FY2026",
                         guidance_source_metric="revenue")
    assert evidence.guidance_implied_next_period_growth == pytest.approx(0.84)
    assert evidence.guidance_implied_comparison_period
    assert evidence.guidance_evidence_id
    assert evidence.guidance_period_label == "Q3 FY2026"
    assert evidence.implied_growth_forecast_compatibility == FC.DIRECTIONAL_CORROBORATION


def test_the_eligibility_is_structured_metadata_not_prose():
    """The request's own requirement, asserted directly: a consumer reads a
    STATUS, never a sentence."""
    eligibility = _quarterly_vs_annual()
    assert eligibility.status in FC.ALL
    assert eligibility.compatible_assumption_metric == sem.MetricIdentity.REVENUE
    assert eligibility.compatible_horizon in sem.PeriodFrequency.ALL
    assert isinstance(eligibility.to_dict(), dict)
