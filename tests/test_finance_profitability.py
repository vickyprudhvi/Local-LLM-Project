"""Phase H.8 — reported vs normalized profitability, guidance completeness,
tax normalization and DCF economic suitability (sections 1-12, 16-17, 21-24,
40-44).

The property under test throughout: A CONFIGURED DEFAULT MUST NEVER BECOME
THE FORECAST. Everything else here exists to make that possible — an
operating income that can be derived when it is not tagged, an unusual item
that can be identified and adjusted for, a guidance table that is read
completely rather than in part, and a suitability verdict that says so when
none of it worked.

Nothing here is issuer-specific and nothing asserts a valuation.
"""

import pytest

from finance import guidance as G
from finance import profitability as P
from finance import suitability as SU
from finance.forward_assumptions import (
    CONFIGURED_TAX_RATE,
    UNUSUAL_TAX_EFFECT_THRESHOLD,
    build_tax_path,
    implied_margin_from_guidance,
)


# ---------------------------------------------------------------------------
# Sections 1-5 — unusual items and normalization
# ---------------------------------------------------------------------------

def _item(amount, recurrence=P.RecurrenceStatus.NON_RECURRING,
          kind=P.UnusualItemType.RESTRUCTURING, baseline=None, location="operating"):
    return P.UnusualItem(
        item_id=f"{kind}.2026-06-30", metric_type=kind, amount=amount, currency="USD",
        period="2025-07-01..2026-06-30", income_statement_location=location,
        recurrence_status=recurrence, recurring_baseline=baseline,
        source_evidence_ids=(f"financial.unusual_item.{kind}",))


def test_a_non_recurring_charge_is_added_back_in_full():
    metric = P.normalize_operating_income(1_000, [_item(200)])
    assert metric.normalized_value == pytest.approx(1_200)
    assert metric.normalization_status == P.NormalizationStatus.VALID
    assert metric.adjustments[0]["direction"] == "add_back"
    assert metric.adjustments[0]["evidence_ids"]


def test_an_uncertain_charge_is_only_partly_added_back():
    """Section 5: if recurrence is uncertain, do not add back 100%."""
    metric = P.normalize_operating_income(
        1_000, [_item(200, P.RecurrenceStatus.POTENTIALLY_RECURRING)])
    assert metric.normalized_value == pytest.approx(1_100)
    assert metric.normalization_status == P.NormalizationStatus.PARTIAL
    assert metric.warnings


def test_a_recurring_charge_is_not_added_back_at_all():
    """A charge booked every year is part of the cost base."""
    metric = P.normalize_operating_income(
        1_000, [_item(200, P.RecurrenceStatus.RECURRING)])
    assert metric.normalized_value == pytest.approx(1_000)
    assert metric.adjustments == []
    assert any("NOT added back" in w for w in metric.warnings)


def test_only_the_excess_over_a_recurring_baseline_is_added_back():
    """Section 5's hardest case: a company books an item most years at one
    scale and this year at another. The baseline stays in the cost base and
    only the excess is unusual."""
    metric = P.normalize_operating_income(
        1_000, [_item(1_400, P.RecurrenceStatus.RECURRING, baseline=350)])
    assert metric.normalized_value == pytest.approx(1_000 + 1_050)
    assert metric.normalization_status == P.NormalizationStatus.PARTIAL
    assert any("EXCESS over that recurring level" in w for w in metric.warnings)


def test_a_recurring_charge_close_to_its_baseline_is_left_alone():
    metric = P.normalize_operating_income(
        1_000, [_item(400, P.RecurrenceStatus.RECURRING, baseline=350)])
    assert metric.normalized_value == pytest.approx(1_000)


def test_an_unknown_recurrence_is_not_added_back():
    metric = P.normalize_operating_income(
        1_000, [_item(200, P.RecurrenceStatus.UNKNOWN)])
    assert metric.normalized_value == pytest.approx(1_000)
    assert metric.normalization_status == P.NormalizationStatus.PARTIAL


def test_a_non_operating_item_never_moves_operating_income():
    metric = P.normalize_operating_income(
        1_000, [_item(200, location="non_operating")])
    assert metric.normalized_value == pytest.approx(1_000)


def test_normalization_is_unavailable_when_there_is_nothing_to_normalize():
    metric = P.normalize_operating_income(None, [_item(200)])
    assert metric.normalization_status == P.NormalizationStatus.UNAVAILABLE
    assert metric.normalized_value is None


def test_reported_values_are_never_overwritten():
    """Section 2: the GAAP figure survives whatever normalization does."""
    metric = P.normalize_operating_income(1_000, [_item(200)])
    assert metric.reported_value == pytest.approx(1_000)
    assert metric.normalized_value != metric.reported_value


def test_every_adjustment_names_the_item_and_its_evidence():
    """Section 4: no hidden adjustments."""
    metric = P.normalize_operating_income(1_000, [_item(200)])
    for adjustment in metric.adjustments:
        assert adjustment["item_id"]
        assert adjustment["evidence_ids"]
        assert adjustment["direction"] in ("add_back", "subtract")
        assert adjustment["recurrence_status"] in P.RecurrenceStatus.ALL


@pytest.mark.parametrize("values,expected", [
    ([100, 100, 100, 100, 100], P.RecurrenceStatus.RECURRING),
    ([0, 0, 0, 0, 100], P.RecurrenceStatus.NON_RECURRING),
    ([0, 0, 0, 100, 100], P.RecurrenceStatus.POTENTIALLY_RECURRING),
    ([100], P.RecurrenceStatus.UNKNOWN),
    ([], P.RecurrenceStatus.UNKNOWN),
])
def test_recurrence_is_measured_from_the_issuers_own_history(values, expected):
    assert P.classify_recurrence(values) == expected


def test_the_recurring_baseline_is_a_median_not_a_mean():
    """A single outlier must not raise its own baseline and shrink the excess
    the baseline exists to expose."""
    assert P.recurring_baseline([100, 100, 100, 100, 10_000]) == pytest.approx(100)


# ---------------------------------------------------------------------------
# Section 12 — cross-metric conflicts
# ---------------------------------------------------------------------------

def test_a_margin_far_from_the_cash_flows_is_a_conflict():
    finding = P.detect_cross_metric_conflict(
        revenue=1_000, operating_margin=0.10, operating_cash_flow=400)
    assert finding is not None
    assert finding["code"] == P.PROFITABILITY_CROSS_METRIC_CONFLICT


def test_ordinary_accrual_timing_is_not_a_conflict():
    finding = P.detect_cross_metric_conflict(
        revenue=1_000, operating_margin=0.20, operating_cash_flow=280)
    assert finding is None


def test_a_configured_default_that_contradicts_the_cash_flows_is_an_error():
    """Section 21: the exact shape of the live failure -- a 10% default
    applied to a company generating 30% operating-cash-flow margins."""
    finding = P.detect_configured_margin_conflict(
        configured_margin=0.10, revenue=66_569, operating_cash_flow=19_967,
        net_margin=0.048)
    assert finding is not None
    assert finding["code"] == P.CONFIGURED_MARGIN_CONFLICTS_WITH_REPORTED_PROFITABILITY
    assert finding["severity"] == "error"


def test_a_configured_default_consistent_with_the_cash_flows_is_not_flagged():
    finding = P.detect_configured_margin_conflict(
        configured_margin=0.10, revenue=1_000, operating_cash_flow=120, net_margin=0.09)
    assert finding is None


def test_best_operating_margin_prefers_normalized_then_reported():
    state = P.ProfitabilityState(revenue=1_000, reported_operating_margin=0.10)
    state.normalized_operating_margin = 0.25
    state.normalized_operating_income = P.NormalizedMetric(
        metric="operating_income", normalization_status=P.NormalizationStatus.VALID)
    value, basis = state.best_operating_margin
    assert value == pytest.approx(0.25)
    assert basis == P.MetricBasis.SYSTEM_NORMALIZED

    plain = P.ProfitabilityState(revenue=1_000, reported_operating_margin=0.10)
    value, basis = plain.best_operating_margin
    assert value == pytest.approx(0.10)
    assert basis == P.MetricBasis.REPORTED_GAAP


def test_best_operating_margin_never_invents_a_default():
    """The configured default is the forward builder's decision to make and
    record, not something this layer hands over silently."""
    empty = P.ProfitabilityState(revenue=1_000)
    value, basis = empty.best_operating_margin
    assert value is None
    assert basis == P.MetricBasis.CONFIGURED_DEFAULT


# ---------------------------------------------------------------------------
# Sections 7-10 — guidance coverage and derived profitability
# ---------------------------------------------------------------------------

def _guided(name, low, high, basis=G.BASIS_ADJUSTED, scale=None, period="FY2026"):
    return {"name": name, "low": low, "high": high, "midpoint": (low + high) / 2,
            "basis": basis, "fiscal_period": period, "scale": scale,
            "evidence_id": f"guidance.{name}.current", "units": "currency"}


def test_full_coverage_is_recognized():
    metrics = {n: _guided(n, 1, 2) for n in
               ("revenue", "adjusted_gross_margin", "adjusted_operating_expenses",
                "adjusted_earnings_per_share", "tax_rate", "free_cash_flow")}
    record = G.assess_guidance_coverage(metrics)
    assert record["guidance_coverage_status"] == \
        G.GuidanceCoverage.COMPLETE_FOR_RELEVANT_METRICS
    assert record["missing_groups"] == []


def test_eps_and_tax_alone_is_partial_not_complete():
    """Section 8's own example: EPS + tax extracted while sales and margin
    guidance were missed is PARTIAL, and the live pipeline called it
    'guidance available' without qualification."""
    metrics = {n: _guided(n, 1, 2) for n in ("adjusted_earnings_per_share", "tax_rate")}
    record = G.assess_guidance_coverage(metrics)
    assert record["guidance_coverage_status"] in (
        G.GuidanceCoverage.PARTIAL, G.GuidanceCoverage.MINIMAL)
    assert "revenue" in record["missing_groups"]


def test_no_guidance_is_unavailable():
    assert G.assess_guidance_coverage({})["guidance_coverage_status"] == \
        G.GuidanceCoverage.UNAVAILABLE


def test_an_operating_margin_is_derived_from_compatible_guidance_components():
    """Section 10: sales x gross margin - operating expenses."""
    metrics = {
        "revenue": _guided("revenue", 66.3, 67.3, G.BASIS_GAAP, scale="billion"),
        "adjusted_gross_margin": _guided("adjusted_gross_margin", 0.81, 0.81),
        "adjusted_operating_expenses": _guided(
            "adjusted_operating_expenses", 42.0, 42.7, scale="billion"),
    }
    result = implied_margin_from_guidance(metrics)
    assert result is not None
    margin, note, ids = result
    sales = 66.8
    expected = (sales * 0.81 - 42.35) / sales
    assert margin == pytest.approx(expected)
    assert "DERIVED_FROM_GUIDANCE" in note
    assert ids


def test_an_implied_margin_is_refused_when_a_component_is_missing():
    metrics = {"revenue": _guided("revenue", 66.3, 67.3, scale="billion")}
    assert implied_margin_from_guidance(metrics) is None


def test_an_implied_margin_is_refused_across_mismatched_periods():
    metrics = {
        "revenue": _guided("revenue", 66.3, 67.3, scale="billion", period="FY2026"),
        "adjusted_gross_margin": _guided("adjusted_gross_margin", 0.81, 0.81,
                                         period="Q2 FY2026"),
        "adjusted_operating_expenses": _guided("adjusted_operating_expenses", 42.0, 42.7,
                                               scale="billion", period="FY2026"),
    }
    assert implied_margin_from_guidance(metrics) is None


def test_an_implied_margin_is_refused_across_mismatched_bases():
    """Section 9: an adjusted gross margin combined with a GAAP operating
    expense produces a number on neither basis."""
    metrics = {
        "revenue": _guided("revenue", 66.3, 67.3, scale="billion"),
        "adjusted_gross_margin": _guided("adjusted_gross_margin", 0.81, 0.81,
                                         basis=G.BASIS_ADJUSTED),
        "operating_expenses": _guided("operating_expenses", 42.0, 42.7,
                                      basis=G.BASIS_GAAP, scale="billion"),
    }
    assert implied_margin_from_guidance(metrics) is None


def test_an_implied_margin_is_refused_across_mismatched_scales():
    metrics = {
        "revenue": _guided("revenue", 66.3, 67.3, scale="billion"),
        "adjusted_gross_margin": _guided("adjusted_gross_margin", 0.81, 0.81),
        "adjusted_operating_expenses": _guided("adjusted_operating_expenses", 42_000,
                                               42_700, scale="million"),
    }
    assert implied_margin_from_guidance(metrics) is None


# ---------------------------------------------------------------------------
# Sections 16-17 — tax normalization
# ---------------------------------------------------------------------------

class _State:
    def __init__(self, reported_tax=None):
        self.profitability = {"reported": {"tax_rate": reported_tax}}
        self.management_guidance = None


def test_a_guided_rate_close_to_normal_is_simply_held():
    values, provenance = build_tax_path(
        _State(), 5, guidance={"tax_rate": _guided("tax_rate", 0.21, 0.22)},
        historical_tax_rate=0.22)
    assert values == [pytest.approx(0.215)] * 5
    assert provenance["source"] == "management_guidance"


def test_an_unusual_current_year_rate_is_not_propagated_across_the_horizon():
    """Section 16's headline rule. A live release guided a 35-36% effective
    rate against a 23.5-24.5% prior guide for the same year, moved by two
    acquisitions."""
    values, provenance = build_tax_path(
        _State(), 5, guidance={"tax_rate": _guided("tax_rate", 0.35, 0.36)},
        historical_tax_rate=0.15)
    assert values[0] == pytest.approx(0.355)
    assert values[-1] == pytest.approx(0.15)
    assert values[0] > values[-1]
    assert provenance["source"] == "management_guidance_normalized"
    codes = {f["code"] for f in provenance["findings"]}
    assert "TAX_GUIDANCE_CONFLICT" in codes


def test_the_tax_provenance_keeps_all_three_rates_apart():
    """Section 16: reported, current-guided and normalized-forward are three
    different quantities."""
    _values, provenance = build_tax_path(
        _State(reported_tax=0.47), 5,
        guidance={"tax_rate": _guided("tax_rate", 0.35, 0.36)},
        historical_tax_rate=0.15)
    assert provenance["reported_tax_rate"] == pytest.approx(0.47)
    assert provenance["current_guided_tax_rate"] == pytest.approx(0.355)
    assert provenance["normalized_forward_tax_rate"] == pytest.approx(0.15)
    assert provenance["guided_tax_basis"] == G.BASIS_ADJUSTED


def test_no_guidance_falls_back_to_the_reported_rate_then_the_default():
    values, provenance = build_tax_path(_State(reported_tax=0.24), 5)
    assert values == [pytest.approx(0.24)] * 5
    assert provenance["source"] == "reported_ttm"

    values, provenance = build_tax_path(_State(), 5)
    assert values == [pytest.approx(CONFIGURED_TAX_RATE)] * 5
    assert provenance["source"] == "configured_default"


def test_an_implausible_reported_rate_is_not_used():
    """A negative or above-60% effective rate is an artefact of a distorted
    pre-tax base, not a forecast input."""
    values, _provenance = build_tax_path(_State(reported_tax=-0.8), 5)
    assert values[0] == pytest.approx(CONFIGURED_TAX_RATE)


# ---------------------------------------------------------------------------
# Sections 21-24 — suitability responds to profitability conflicts
# ---------------------------------------------------------------------------

def test_a_configured_default_margin_makes_the_dcf_at_best_limited():
    """Section 23: a default that IS the forecast is not merely 'high
    uncertainty'."""
    assessment = SU.assess_dcf_suitability(
        revenue=66_569, operating_margin=0.10, margin_bounds=(-1.0, 0.60),
        margin_basis="configured_default")
    assert assessment.status in (SU.DcfSuitability.LIMITED,
                                 SU.DcfSuitability.NOT_SUITABLE)
    codes = {s.code for s in assessment.signals}
    assert "CONFIGURED_MARGIN_IS_THE_FORECAST" in codes


def test_a_partial_normalization_is_reported_as_unresolved():
    assessment = SU.assess_dcf_suitability(
        revenue=1_000, operating_margin=0.25, margin_bounds=(-1.0, 0.60),
        normalization_status="PARTIAL")
    codes = {s.code for s in assessment.signals}
    assert "NORMALIZATION_UNRESOLVED" in codes


def test_a_profitability_conflict_is_carried_into_suitability():
    conflict = P.detect_configured_margin_conflict(
        configured_margin=0.10, revenue=66_569, operating_cash_flow=19_967,
        net_margin=0.048)
    assessment = SU.assess_dcf_suitability(
        revenue=66_569, operating_margin=0.10, margin_bounds=(-1.0, 0.60),
        profitability_findings=[conflict])
    codes = {s.code for s in assessment.signals}
    assert P.CONFIGURED_MARGIN_CONFLICTS_WITH_REPORTED_PROFITABILITY in codes


def test_an_economic_disconnect_is_detected_without_reference_to_market_price():
    """Section 24/25: a company generating substantial recurring cash whose
    modelled equity collapses to about nothing. The market price is not an
    input to this test and must never be."""
    assessment = SU.assess_dcf_suitability(
        revenue=66_569, operating_margin=0.10, margin_bounds=(-1.0, 0.60),
        modelled_equity_value=500, recurring_free_cash_flow=16_063)
    codes = {s.code for s in assessment.signals}
    assert SU.DCF_ECONOMIC_DISCONNECT in codes
    disconnect = next(s for s in assessment.signals
                      if s.code == SU.DCF_ECONOMIC_DISCONNECT)
    assert "market price plays no part" in disconnect.detail


def test_a_healthy_equity_value_is_not_a_disconnect():
    assessment = SU.assess_dcf_suitability(
        revenue=66_569, operating_margin=0.25, margin_bounds=(-1.0, 0.60),
        modelled_equity_value=250_000, recurring_free_cash_flow=16_063)
    codes = {s.code for s in assessment.signals}
    assert SU.DCF_ECONOMIC_DISCONNECT not in codes


def test_a_normal_reported_margin_stays_suitable():
    assessment = SU.assess_dcf_suitability(
        revenue=1_000, operating_margin=0.18, free_cash_flow=120,
        operating_cash_flow=150, margin_bounds=(-1.0, 0.60),
        terminal_value_share=0.55, normalization_status="VALID",
        margin_basis="reported_GAAP", historical_comparability="COMPARABLE")
    assert assessment.status == SU.DcfSuitability.SUITABLE


def test_a_mathematically_valid_dcf_can_still_be_economically_unsuitable():
    """Section 22's central claim, as an assertion."""
    assessment = SU.assess_dcf_suitability(
        revenue=66_569, operating_margin=0.10, margin_bounds=(-1.0, 0.60),
        margin_basis="configured_default",
        modelled_equity_value=500, recurring_free_cash_flow=16_063)
    assert assessment.status == SU.DcfSuitability.NOT_SUITABLE
    assert assessment.blocks_valuation_conclusion is True


# ---------------------------------------------------------------------------
# Section 32 — a limiting factor must actually be a limitation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,is_limitation", [
    # The exact entry a live run filed under limiting factors.
    ("Strong operational fundamentals including 19.01% free cash flow margin "
     "and 34.70% ROE support business quality", False),
    ("Strong balance sheet with substantial net cash", False),
    ("High revenue growth of 65%", False),
    ("Key DCF inputs including WACC rely on configured defaults", True),
    ("Scenario spread of 186.4% indicates high model uncertainty", True),
    ("Reported profitability is distorted by unusual items", True),
    # A strength that IS framed as a limitation stays.
    ("Strong cash flow, but it depends on a single product", True),
])
def test_a_strength_is_not_a_limiting_factor(text, is_limitation):
    from finance.research_pipeline import is_genuine_limiting_factor
    assert is_genuine_limiting_factor(text) is is_limitation


def test_a_purely_favourable_limiting_factor_is_dropped():
    from finance.research_pipeline import _route_conditions_by_direction

    routed = _route_conditions_by_direction({
        "limiting_factors": [
            "Strong balance sheet with substantial net cash",
            "Key DCF inputs rely on configured defaults",
        ],
        "conditions_that_strengthen_the_view": [],
        "conditions_that_weaken_the_view": [],
        "reassessment_triggers": [],
    })
    assert routed["limiting_factors"] == ["Key DCF inputs rely on configured defaults"]


def test_a_lone_limiting_factor_is_never_dropped():
    """An empty section loses more than one loosely-worded entry."""
    from finance.research_pipeline import _route_conditions_by_direction

    routed = _route_conditions_by_direction({
        "limiting_factors": ["Strong balance sheet with substantial net cash"],
        "conditions_that_strengthen_the_view": [],
        "conditions_that_weaken_the_view": [],
        "reassessment_triggers": [],
    })
    assert len(routed["limiting_factors"]) == 1
