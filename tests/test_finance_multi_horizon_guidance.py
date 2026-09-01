"""One release, two horizons, and everything downstream of that.

Spec §11 already says guidance identity is `(metric_id, target_period,
target_period_type, basis)`. `guidance_identity` computes exactly that and
`resolve_guidance_status` uses it. What neither could fix is the CONTAINER:
`GuidanceRelease.metrics` is a dict keyed by metric NAME, so a release
guiding both a quarter and a full year can physically hold only one of them,
and the extractor's own `if name in metrics: continue` discards the second
before the resolver ever sees it.

The consequence runs the length of the pipeline. A company guiding 27-29%
for next quarter AND $118-120B for the full year had its annual outlook
dropped; the annual growth that outlook implies was never derived; the
model-bound assessment therefore never saw a comparable annual figure past
the bound; and the valuation went on being treated as fully usable while the
company's own full-year guidance said otherwise.

These tests run that chain end to end.
"""

import pytest

from finance import forward_assumptions as FA
from finance import guidance as G
from finance import semantics as sem
from tests.fixtures.multi_horizon_release import (
    FY_ADJUSTED_EPS,
    FY_REVENUE_ABSOLUTE,
    IMPLIED_FY_GROWTH,
    MULTI_HORIZON_RELEASE,
    PRIOR_FY_REVENUE,
    Q1_ADJUSTED_EPS,
    Q1_REVENUE_GROWTH,
)

N = G.GuidanceMetricName
FC = sem.ForecastCompatibility


@pytest.fixture(scope="module")
def release():
    return G.extract_guidance_from_text(
        MULTI_HORIZON_RELEASE, "ZZ", "9999999999-27-000001", "release.htm",
        "2027-02-20", expected_fiscal_year=2027)


def _by_period(release, name):
    return {m.fiscal_period: m for m in release.all_metrics if m.name == name}


# ---------------------------------------------------------------------------
# A. both horizons survive extraction
# ---------------------------------------------------------------------------

def test_both_horizons_are_extracted_from_one_release(release):
    periods = {(m.name, m.fiscal_period) for m in release.all_metrics}
    assert (N.CONSOLIDATED_REVENUE_GROWTH, "Q1 FY2027") in periods, sorted(periods)
    assert (N.CONSOLIDATED_REVENUE, "FY2027") in periods, sorted(periods)


def test_two_statements_of_one_metric_both_survive(release):
    """The collision the name-keyed dict could not represent: adjusted EPS
    for a quarter and for the year are two statements, not two versions of
    one."""
    eps = _by_period(release, N.ADJUSTED_EPS)
    assert set(eps) == {"Q1 FY2027", "FY2027"}, sorted(eps)
    assert (eps["Q1 FY2027"].low, eps["Q1 FY2027"].high) == pytest.approx(Q1_ADJUSTED_EPS)
    assert (eps["FY2027"].low, eps["FY2027"].high) == pytest.approx(FY_ADJUSTED_EPS)


def test_each_statement_keeps_its_own_identity(release):
    identities = {G.guidance_identity(m) for m in release.all_metrics}
    assert len(identities) == len(release.all_metrics), "two statements collapsed"


def test_a_quarterly_outlook_does_not_displace_the_annual_one(release):
    """The invariant in one line: both are current, so both are kept."""
    revenue = {m.fiscal_period for m in release.all_metrics
               if m.name in (N.CONSOLIDATED_REVENUE, N.CONSOLIDATED_REVENUE_GROWTH)}
    assert "FY2027" in revenue and "Q1 FY2027" in revenue, sorted(revenue)


def test_selection_across_releases_keeps_every_current_horizon(release):
    current, _superseded = G.select_current_guidance([release], as_of="2027-03-01")
    assert current is not None
    eps = {m.fiscal_period for m in current.all_metrics if m.name == N.ADJUSTED_EPS}
    assert eps == {"Q1 FY2027", "FY2027"}, sorted(eps)


def test_the_name_keyed_view_prefers_the_horizon_a_forecast_can_use(release):
    """`metrics` stays name-keyed for every existing consumer. When two
    horizons compete for one name the ANNUAL one wins, because that view is
    what the assumption builder reads and an annual assumption is what it
    builds."""
    current, _ = G.select_current_guidance([release], as_of="2027-03-01")
    assert current.metrics[N.ADJUSTED_EPS].fiscal_period == "FY2027"


# ---------------------------------------------------------------------------
# B. annual absolute revenue guidance -> annual growth
# ---------------------------------------------------------------------------

def _state(release):
    """The minimum CurrentFinancialState surface `collect_growth_evidence`
    reads. Only the guidance matters here; the rest is what the real state
    would supply and is empty on purpose, so a failure is unambiguous."""
    current, _superseded = G.select_current_guidance([release], as_of="2027-03-01")

    class _State:
        management_guidance = current.to_dict()
        historical_metrics = {}
        flows = {}
        historical_comparability = None

    return _State()


def _company_facts(prior=PRIOR_FY_REVENUE):
    """Two reported fiscal years, so an annual outlook has a comparable."""
    return {
        "facts": {"us-gaap": {"Revenues": {"units": {"USD": [
            {"start": "2025-02-01", "end": "2026-01-31", "val": prior * 1e9,
             "form": "10-K", "fy": 2026, "fp": "FY", "accn": "a", "filed": "2026-03-01"},
            {"start": "2024-02-01", "end": "2025-01-31", "val": prior * 0.8 * 1e9,
             "form": "10-K", "fy": 2025, "fp": "FY", "accn": "b", "filed": "2025-03-01"},
        ]}}}},
    }


def test_annual_absolute_revenue_guidance_yields_an_annual_growth_signal(release):
    evidence = FA.collect_growth_evidence(_state(release), _company_facts())
    assert evidence.guidance_low is not None, (
        "annual absolute revenue guidance produced no growth signal")
    assert evidence.guidance_period_frequency == sem.PeriodFrequency.ANNUAL
    assert evidence.guidance_forecast_compatibility == FC.ASSUMPTION_COMPARABLE
    assert evidence.guidance_midpoint == pytest.approx(IMPLIED_FY_GROWTH, rel=0.02)


def test_the_annual_derivation_keeps_its_evidence(release):
    evidence = FA.collect_growth_evidence(_state(release), _company_facts())
    assert evidence.guidance_evidence_id
    assert evidence.guidance_implied_comparison_period
    assert evidence.guidance_source_metric


def test_a_quarterly_growth_guide_does_not_suppress_the_annual_derivation(release):
    """The exact bug: the annual derivation was skipped whenever ANY
    consolidated growth guidance already existed, including a quarterly one
    that cannot set an annual assumption."""
    evidence = FA.collect_growth_evidence(_state(release), _company_facts())
    assert evidence.guidance_forecast_compatibility == FC.ASSUMPTION_COMPARABLE
    # ...and the quarterly statement is not lost either.
    assert evidence.near_term_guidance_midpoint == pytest.approx(
        sum(Q1_REVENUE_GROWTH) / 2)
    assert evidence.near_term_guidance_period_label == "Q1 FY2027"


# ---------------------------------------------------------------------------
# C-D. eligibility and the model bound
# ---------------------------------------------------------------------------

def test_the_quarterly_guide_remains_directional_only(release):
    evidence = FA.collect_growth_evidence(_state(release), _company_facts())
    assert evidence.near_term_guidance_eligibility.status == FC.DIRECTIONAL_CORROBORATION
    assert not evidence.near_term_guidance_eligibility.may_set_magnitude


def test_annual_guidance_past_the_bound_creates_a_model_bound_conflict():
    """Case D, and the generic case the request states: TTM 17%, annual
    guidance implying 34%, bound 25%. The DCF must not quietly use 17% and
    call itself fully usable."""
    evidence = FA.GrowthEvidence(
        ttm_yoy=0.17, guidance_low=0.33, guidance_high=0.35,
        guidance_period_type="annual",
        guidance_source_metric=N.CONSOLIDATED_REVENUE_GROWTH)
    conflict = FA.detect_model_bound_conflict(
        evidence, raw_value=0.34, applied_value=0.25, bounds=FA.GROWTH_BOUNDS)
    assert conflict is not None
    assert conflict["corroborating_forecast_compatibility"] == FC.ASSUMPTION_COMPARABLE
    assert conflict["corroborating_value"] == pytest.approx(0.34)


def test_the_conflict_is_absent_when_no_comparable_evidence_passes_the_bound():
    evidence = FA.GrowthEvidence(
        ttm_yoy=0.17, near_term_guidance_low=0.80, near_term_guidance_high=0.88,
        near_term_guidance_period_label="Q1 FY2027")
    assert FA.detect_model_bound_conflict(
        evidence, 0.17, 0.17, FA.GROWTH_BOUNDS) is None


# ---------------------------------------------------------------------------
# E. the canonical state must actually carry the annual statement
# ---------------------------------------------------------------------------

def test_the_canonical_guidance_state_carries_both_horizons(release):
    """Case E stated as the request asks: if the annual guidance is missing
    from the canonical state, this test fails."""
    current, _ = G.select_current_guidance([release], as_of="2027-03-01")
    published = current.to_dict()
    horizons = {(m["name"], m["target_period"]) for m in published["all_metrics"]}
    assert (N.CONSOLIDATED_REVENUE, "FY2027") in horizons, sorted(horizons)
    assert (N.CONSOLIDATED_REVENUE_GROWTH, "Q1 FY2027") in horizons, sorted(horizons)


def test_the_coverage_matrix_counts_the_annual_row(release):
    current, _ = G.select_current_guidance([release], as_of="2027-03-01")
    matrix = G.build_guidance_matrix(
        {name: m.to_dict() for name, m in current.metrics.items()},
        releases_examined=1, all_metrics=[m.to_dict() for m in current.all_metrics])
    assert "revenue" in (matrix.get("current_rows") or [])
