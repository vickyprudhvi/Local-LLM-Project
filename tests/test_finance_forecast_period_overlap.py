"""Phase H.33 — a trailing window that overlaps the period being guided is
not that period's prior comparable, however its frequency is labelled.

THE LIVE FAILURE (JNJ)

FY2026 revenue guidance ($101.1B) divided by trailing-twelve-month revenue
ending Q2 FY2026 ($97.93B) produced 3.2% "growth". Both sides are twelve-
month windows (`finance.semantics.same_duration` correctly says ANNUAL and
TTM are the same LENGTH of time), and neither carries a matching
`(fiscal_year, fiscal_quarter)` pair, so the one existing same-period guard
never fired. What makes the pair wrong is that the TTM's own calendar window
(2025-06-30..2026-06-28) already contains Q1 and Q2 of FY2026 -- the very
year being guided. Dividing by it does not answer "how much will the
company grow this year"; it answers a question nobody asked.

No ticker appears below. Every fixture is a fictional issuer.
"""

import pytest

from finance import forward_assumptions as FA
from finance import guidance as G
from finance import semantics as sem

F = sem.PeriodFrequency
M = sem.MetricIdentity
O = sem.Operation
N = G.GuidanceMetricName
FC = sem.ForecastCompatibility


def _rev(frequency, **kw):
    kw.setdefault("flow_or_instant", sem.FlowOrInstant.FLOW)
    return sem.SemanticFact(metric_id=M.REVENUE, value=100.0,
                            period_frequency=frequency, **kw)


# ---------------------------------------------------------------------------
# Section 2/5 — the general overlap rule, at the `semantics` layer
# ---------------------------------------------------------------------------

def test_periods_overlap_detects_a_shared_window():
    assert sem.periods_overlap("2025-06-30", "2026-06-28", "2025-12-28", "2026-12-29")


def test_periods_overlap_allows_adjacent_periods_touching_at_one_boundary():
    """FY2025 ending 2025-12-28 and FY2026 beginning the same day are the
    PRIOR and CURRENT year, not an overlap. A `<=` boundary test would
    reject every legitimate adjacent-period comparison this exists to
    allow."""
    assert not sem.periods_overlap("2024-12-30", "2025-12-28", "2025-12-28", "2026-12-29")


def test_periods_overlap_is_silent_without_dates_on_both_sides():
    assert not sem.periods_overlap(None, None, "2025-12-28", "2026-12-29")
    assert not sem.periods_overlap("2025-06-30", "2026-06-28", None, None)


def test_annual_guidance_over_an_overlapping_ttm_is_rejected():
    """The JNJ pair itself, at the semantic-fact level."""
    guided = _rev(F.ANNUAL, fiscal_year=2026,
                 current_or_historical=sem.CurrentOrHistorical.FORWARD,
                 start_date="2025-12-28", end_date="2026-12-29")
    ttm = _rev(F.TTM, current_or_historical=sem.CurrentOrHistorical.CURRENT,
              start_date="2025-06-30", end_date="2026-06-28")
    verdict = sem.compatible_for(O.GROWTH, guided, ttm)
    assert not verdict
    assert verdict.code == sem.OVERLAPPING_COMPARISON_PERIODS


def test_annual_guidance_over_the_true_prior_year_is_still_allowed():
    """The companion case that must stay allowed: the TTM ending RIGHT
    where the guided year begins is the genuine prior-year comparable."""
    guided = _rev(F.ANNUAL, fiscal_year=2026,
                 current_or_historical=sem.CurrentOrHistorical.FORWARD,
                 start_date="2025-12-28", end_date="2026-12-29")
    prior_ttm = _rev(F.TTM, current_or_historical=sem.CurrentOrHistorical.CURRENT,
                     start_date="2024-12-29", end_date="2025-12-28")
    assert sem.compatible_for(O.GROWTH, guided, prior_ttm)


def test_without_dates_the_existing_permissive_behaviour_is_unchanged():
    """`test_full_year_guidance_over_ttm_revenue_is_a_growth_rate` in
    test_finance_semantics.py already asserts this pair is ALLOWED when
    neither fact carries dates -- this test states why that is still
    correct: with no calendar evidence, there is nothing to find an overlap
    IN, so permissive is the only honest answer, not a loophole."""
    guided = _rev(F.ANNUAL, fiscal_year=2026,
                 current_or_historical=sem.CurrentOrHistorical.FORWARD)
    ttm = _rev(F.TTM, current_or_historical=sem.CurrentOrHistorical.CURRENT)
    assert sem.compatible_for(O.GROWTH, guided, ttm)


# ---------------------------------------------------------------------------
# JNJ-shaped generalized fixture (section 8)
# ---------------------------------------------------------------------------
#
#   TTM ending Q2 FY2026           = 98   (24 + 25 + 24 + 25, most recent 4Q)
#   FY2025 actual (reported)       = 94
#   FY2026 revenue guidance        = 101
#   explicit FY2026 growth guidance = 7%-8%   (101 / 94 - 1 = 7.45%)

REVENUE_CONCEPT = "RevenueFromContractWithCustomerExcludingAssessedTax"


def _quarter(start, end, value):
    return {"start": start, "end": end, "val": value * 1e9, "form": "10-Q",
           "filed": end, "accn": f"q-{end}", "fy": int(end[:4]), "fp": "Q1"}


def _annual(start, end, value):
    return {"start": start, "end": end, "val": value * 1e9, "form": "10-K",
           "filed": end, "accn": f"fy-{end}", "fy": int(end[:4]), "fp": "FY"}


def _company_facts():
    rows = [
        _quarter("2025-07-01", "2025-09-30", 24),   # Q3 FY2025
        _quarter("2025-10-01", "2025-12-30", 25),   # Q4 FY2025
        _quarter("2025-12-31", "2026-03-31", 24),   # Q1 FY2026
        _quarter("2026-04-01", "2026-06-30", 25),   # Q2 FY2026
        _annual("2025-01-01", "2025-12-30", 94),    # FY2025 actual
    ]
    return {"facts": {"us-gaap": {REVENUE_CONCEPT: {"units": {"USD": rows}}}}}


def _guidance_statement(name, low, high, *, fiscal_period="FY2026",
                        target_period_type="CURRENT_FISCAL_YEAR",
                        period_type="annual", scale=None, midpoint=None,
                        evidence_id=None):
    return {
        "name": name, "low": low, "high": high, "midpoint": midpoint,
        "fiscal_year": 2026, "fiscal_period": fiscal_period,
        "target_period_type": target_period_type, "period_type": period_type,
        "forward_kind": G.ForwardInformationKind.CURRENT_FY_GUIDANCE,
        "basis": "GAAP", "bound_type": "approximately",
        "scale": scale, "evidence_id": evidence_id or f"dcf.guidance.{name}.current",
    }


def _state(statements):
    all_metrics = list(statements)
    metrics = {}
    for m in all_metrics:
        metrics.setdefault(m["name"], m)

    class _State:
        management_guidance = {"metrics": metrics, "all_metrics": all_metrics}
        historical_metrics = {}
        flows = {}
        historical_comparability = None

    return _State()


# Absolute revenue guidance ONLY (no explicit growth metric) -- the live
# JNJ shape, since the issuer's growth percentage sits inside the same
# sentence as the dollar figure, not as a separately-tagged metric today.
_ABSOLUTE_ONLY_STATE = _state([
    _guidance_statement(N.CONSOLIDATED_REVENUE, 101.0, 101.0, midpoint=101.0,
                        scale="billion"),
])

# Both an absolute revenue figure AND an explicit annual growth figure.
_EXPLICIT_AND_ABSOLUTE_STATE = _state([
    _guidance_statement(N.CONSOLIDATED_REVENUE, 101.0, 101.0, midpoint=101.0,
                        scale="billion"),
    _guidance_statement(N.CONSOLIDATED_REVENUE_GROWTH, 0.07, 0.08),
])


def test_the_mismatched_3_2_percent_style_growth_never_appears():
    """The headline assertion: dividing non-comparable annual guidance by
    an overlapping TTM must not happen. 101/98 - 1 = 3.06%-shaped
    mismatches (the JNJ-equivalent of this fixture's own bad pair) must
    never become `guidance_low`."""
    evidence = FA.collect_growth_evidence(_ABSOLUTE_ONLY_STATE, _company_facts())
    bad_ttm_growth = 101.0 / 98.0 - 1.0
    assert evidence.guidance_low is not None
    assert evidence.guidance_low != pytest.approx(bad_ttm_growth, abs=1e-4)


def test_the_overlapping_ttm_candidate_is_recorded_as_rejected():
    evidence = FA.collect_growth_evidence(_ABSOLUTE_ONLY_STATE, _company_facts())
    codes = [r["code"] for r in evidence.semantic_rejections]
    assert sem.OVERLAPPING_COMPARISON_PERIODS in codes


def test_the_fallback_to_the_prior_fiscal_year_actual_is_comparable():
    """101 / 94 - 1 = 7.45%, which must be what the absolute-guidance
    fallback produces once the overlapping TTM is correctly refused."""
    evidence = FA.collect_growth_evidence(_ABSOLUTE_ONLY_STATE, _company_facts())
    assert evidence.guidance_low == pytest.approx(101.0 / 94.0 - 1.0, rel=1e-6)
    assert evidence.guidance_forecast_compatibility == FC.ASSUMPTION_COMPARABLE


def test_explicit_annual_growth_is_eligible_as_assumption_comparable():
    evidence = FA.collect_growth_evidence(_EXPLICIT_AND_ABSOLUTE_STATE, _company_facts())
    assert evidence.guidance_low == pytest.approx(0.07)
    assert evidence.guidance_high == pytest.approx(0.08)
    assert evidence.guidance_forecast_compatibility == FC.ASSUMPTION_COMPARABLE


def test_explicit_growth_wins_over_a_derivation_from_absolute_guidance():
    """Section 3/6: explicit beats derived. When both are present, the
    derivation from the absolute figure (and ITS period-overlap questions)
    must never run at all -- `evidence.guidance_low` must be the explicit
    7%-8%, not anything computed from 101 against any base."""
    evidence = FA.collect_growth_evidence(_EXPLICIT_AND_ABSOLUTE_STATE, _company_facts())
    assert evidence.guidance_source_metric == N.CONSOLIDATED_REVENUE_GROWTH
    assert evidence.guidance_low == pytest.approx(0.07)
    for not_derived in (101.0 / 98.0 - 1.0, 101.0 / 94.0 - 1.0):
        assert evidence.guidance_low != pytest.approx(not_derived, abs=1e-4)


def test_build_growth_path_anchors_on_the_explicit_or_fallback_value_not_the_overlap():
    """End to end: the DCF year-1 path itself must carry the correct
    anchor and cite the real guidance evidence id, never a value implied
    by the overlapping TTM."""
    evidence = FA.collect_growth_evidence(_ABSOLUTE_ONLY_STATE, _company_facts())
    path = FA.build_growth_path(evidence, forecast_years=5)
    assert path.anchor_source == FA.AssumptionSourceType.MANAGEMENT_GUIDANCE
    assert path.values[0] == pytest.approx(101.0 / 94.0 - 1.0, rel=1e-4)
    derivation = path.entries[0].derivation
    assert "3.2" not in derivation and "3.06" not in derivation


# ---------------------------------------------------------------------------
# Section 9 — the additional generalized tests, A through J
# ---------------------------------------------------------------------------

# A. FY guidance vs prior FY actual -> comparable.
def test_A_fy_guidance_vs_prior_fy_actual_is_comparable():
    guided = _rev(F.ANNUAL, fiscal_year=2026, start_date="2025-12-28", end_date="2026-12-29")
    prior_actual = _rev(F.ANNUAL, fiscal_year=2025, start_date="2024-12-30",
                        end_date="2025-12-28")
    assert sem.compatible_for(O.GROWTH, guided, prior_actual)


# B. FY guidance vs current overlapping TTM -> NOT_COMPARABLE.
def test_B_fy_guidance_vs_overlapping_ttm_is_not_comparable():
    guided = _rev(F.ANNUAL, fiscal_year=2026, start_date="2025-12-28", end_date="2026-12-29")
    ttm = _rev(F.TTM, start_date="2025-06-30", end_date="2026-06-28")
    verdict = sem.compatible_for(O.GROWTH, guided, ttm)
    assert not verdict
    assert verdict.code == sem.OVERLAPPING_COMPARISON_PERIODS


# C. quarterly guidance vs same quarter prior year -> comparable.
def test_C_quarterly_guidance_vs_same_quarter_prior_year_is_comparable():
    guided = _rev(F.QUARTER, fiscal_year=2026, fiscal_quarter=3,
                 start_date="2026-07-01", end_date="2026-09-30")
    prior = _rev(F.QUARTER, fiscal_year=2025, fiscal_quarter=3,
                start_date="2025-07-01", end_date="2025-09-30")
    assert sem.compatible_for(O.GROWTH, guided, prior)


# D. quarterly guidance vs annual actual -> NOT_COMPARABLE.
def test_D_quarterly_guidance_vs_annual_actual_is_not_comparable():
    guided = _rev(F.QUARTER, fiscal_year=2026, fiscal_quarter=3)
    annual = _rev(F.ANNUAL, fiscal_year=2025)
    verdict = sem.compatible_for(O.GROWTH, guided, annual)
    assert not verdict
    assert verdict.code == sem.PERIOD_FREQUENCY_MISMATCH


# E. TTM current vs prior comparable (non-overlapping) TTM -> comparable.
def test_E_ttm_current_vs_prior_comparable_ttm_is_comparable():
    current_ttm = _rev(F.TTM, start_date="2025-06-30", end_date="2026-06-28")
    prior_ttm = _rev(F.TTM, start_date="2024-06-30", end_date="2025-06-28")
    assert sem.compatible_for(O.GROWTH, current_ttm, prior_ttm)


# F. explicit annual growth + absolute annual revenue guidance -> explicit wins.
def test_F_explicit_growth_wins_over_absolute_guidance():
    evidence = FA.collect_growth_evidence(_EXPLICIT_AND_ABSOLUTE_STATE, _company_facts())
    assert evidence.guidance_source_metric == N.CONSOLIDATED_REVENUE_GROWTH
    assert evidence.guidance_low == pytest.approx(0.07)


# G. non-comparable absolute guidance only -> no precise DCF growth derived
#    from the non-comparable pair (the fallback uses the genuinely
#    comparable prior-year actual instead).
def test_G_non_comparable_absolute_guidance_alone_derives_no_growth_from_the_overlap():
    evidence = FA.collect_growth_evidence(_ABSOLUTE_ONLY_STATE, _company_facts())
    overlap_growth = 101.0 / 98.0 - 1.0
    assert evidence.guidance_low != pytest.approx(overlap_growth, abs=1e-4)
    assert any(r["code"] == sem.OVERLAPPING_COMPARISON_PERIODS
              for r in evidence.semantic_rejections)


# H. DIRECTIONAL_CORROBORATION evidence may support narrative but cannot
#    size a numeric DCF bound.
def test_H_directional_corroboration_cannot_set_the_assumption_magnitude():
    quarterly_guidance_eligibility = sem.forecast_compatibility(
        evidence_metric=M.REVENUE, evidence_frequency=F.QUARTER,
        assumption_metric=M.REVENUE, assumption_frequency=F.ANNUAL)
    assert quarterly_guidance_eligibility.status == FC.DIRECTIONAL_CORROBORATION
    assert not quarterly_guidance_eligibility.may_set_magnitude
    assert quarterly_guidance_eligibility.may_corroborate_direction


# I. completed/stale guidance is not eligible.
def test_I_a_withdrawn_or_superseded_statement_is_filtered_before_it_reaches_evidence():
    current, superseded = G.select_current_guidance(
        [_fake_release(fiscal_year=2025, low=0.03, high=0.03, filed="2025-01-10")],
        as_of="2030-01-01")
    assert current is None
    assert superseded


def _fake_release(fiscal_year, low, high, filed):
    return G.extract_guidance_from_text(
        f"We expect fiscal year {fiscal_year} revenue growth of approximately "
        f"{low:.0%}.",
        "ZZ", f"0000000000-{filed[2:4]}-000001", "release.htm", filed,
        expected_fiscal_year=fiscal_year)


# J. scope mismatch -> not comparable.
def test_J_a_segment_figure_is_not_comparable_to_consolidated_revenue():
    consolidated = _rev(F.ANNUAL, fiscal_year=2026,
                        consolidation_scope=sem.ConsolidationScope.CONSOLIDATED)
    segment = _rev(F.ANNUAL, fiscal_year=2025,
                   consolidation_scope=sem.ConsolidationScope.SEGMENT)
    verdict = sem.compatible_for(O.GROWTH, consolidated, segment)
    assert not verdict
