"""Phase H.6 — the guidance metric taxonomy (sections 6, 8, 9, 11).

tests/test_finance_guidance.py already covers the ORIGINAL precision rules
(a range is required, a forward-looking clause is required, a value with no
period is rejected). This file covers the identity rules added in H.6: which
METRIC a number belongs to, which PERIOD it covers, and which basis it is on.

The failure these exist to prevent is a number that is real, correctly
parsed, and attached to the wrong thing — which is worse than a miss,
because it looks exactly like a correct value all the way into the DCF.
"""

import pytest

from finance import guidance as G

ACCN = "0000091142-26-000096"
DOC = "exhibit991.htm"


def _extract(text, filed="2026-07-30", accession=ACCN):
    return G.extract_guidance_from_text(text, "TEST", accession, DOC, filed)


# ---------------------------------------------------------------------------
# Section 6 — never map one metric to another
# ---------------------------------------------------------------------------

def test_ebitda_growth_is_never_stored_as_revenue_growth():
    """The AT&T shape, reduced to its essentials: a revenue-growth bullet
    with no number, followed by an EBITDA-growth bullet with one."""
    release = _extract(
        "The Company's outlook for 2026 includes: "
        "Service revenue growth in the low-single-digit range annually. "
        "Adjusted EBITDA* growth in the 3% to 4% range in 2026.")
    assert "revenue_growth" not in release.metrics
    ebitda = release.metrics["adjusted_ebitda_growth"]
    assert (ebitda.low, ebitda.high) == pytest.approx((0.03, 0.04))
    assert ebitda.basis == G.BASIS_ADJUSTED


def test_service_revenue_growth_is_not_consolidated_revenue_growth():
    release = _extract(
        "For 2026 the Company expects service revenue growth of 2% to 3%.")
    assert G.GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH not in release.metrics
    service = release.metrics[G.GuidanceMetricName.SERVICE_REVENUE_GROWTH]
    assert service.scope == "service"
    assert G.is_revenue_component_metric(service.name)
    assert not G.may_anchor_revenue_growth(service.name)


def test_consolidated_revenue_growth_is_the_only_metric_that_may_anchor():
    release = _extract("For 2026 the Company expects revenue growth of 2% to 3%.")
    growth = release.metrics[G.GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH]
    assert growth.scope == "consolidated"
    assert G.may_anchor_revenue_growth(growth.name)
    for other in (G.GuidanceMetricName.ADJUSTED_EBITDA_GROWTH,
                  G.GuidanceMetricName.SERVICE_REVENUE_GROWTH,
                  G.GuidanceMetricName.SEGMENT_REVENUE_GROWTH,
                  G.GuidanceMetricName.FREE_CASH_FLOW,
                  G.GuidanceMetricName.ADJUSTED_EPS):
        assert not G.may_anchor_revenue_growth(other)


def test_longest_keyword_wins_so_a_qualifier_is_never_dropped():
    """"Advanced Connectivity service revenue growth" is a SEGMENT figure; it
    must not collapse into either service or consolidated revenue growth
    merely because those patterns also match its tail."""
    release = _extract(
        "For 2026 the Company expects Advanced Connectivity service revenue "
        "growth of 5% to 6%.")
    assert G.GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH not in release.metrics
    assert G.GuidanceMetricName.SERVICE_REVENUE_GROWTH not in release.metrics
    segment = release.metrics[G.GuidanceMetricName.SEGMENT_REVENUE_GROWTH]
    assert segment.scope == "segment"


def test_multiple_metrics_in_one_release_each_keep_their_own_identity():
    release = _extract(
        "The Company's 2026 outlook includes: revenue growth of 2% to 3% in 2026. "
        "It expects adjusted EBITDA* growth in the 4% to 5% range in 2026. "
        "It expects adjusted EPS* of $2.25 to $2.35 in 2026. "
        "It expects operating margin of 18% to 19% in 2026.")
    assert release.metrics["revenue_growth"].high == pytest.approx(0.03)
    assert release.metrics["adjusted_ebitda_growth"].high == pytest.approx(0.05)
    assert release.metrics["adjusted_earnings_per_share"].high == pytest.approx(2.35)
    assert release.metrics["operating_margin"].high == pytest.approx(0.19)


def test_a_metric_keyword_may_not_reach_past_a_clause_boundary():
    """The mechanism behind the AT&T mis-mapping, tested directly."""
    release = _extract(
        "The Company expects, for 2026: revenue growth in the low-single-digit "
        "range. It expects capital investment of $23 billion to $24 billion in 2026.")
    assert "revenue_growth" not in release.metrics
    assert release.metrics["capital_expenditure"].high == pytest.approx(24.0)


def test_gaap_and_adjusted_are_never_mixed():
    release = _extract(
        "For 2026 the Company expects adjusted EPS of $3.70 to $3.85 and "
        "diluted EPS (GAAP) of $3.60 to $3.75.")
    assert release.metrics["adjusted_earnings_per_share"].basis == G.BASIS_ADJUSTED
    if "earnings_per_share" in release.metrics:
        assert release.metrics["earnings_per_share"].basis == G.BASIS_GAAP
        assert release.metrics["earnings_per_share"].low != \
            release.metrics["adjusted_earnings_per_share"].low


def test_an_adjusted_metric_never_lands_under_its_gaap_name():
    release = _extract(
        "For 2026 the Company expects adjusted EBITDA growth of 3% to 4%.")
    assert "ebitda_growth" not in release.metrics
    assert release.metrics["adjusted_ebitda_growth"].basis == G.BASIS_ADJUSTED


# ---------------------------------------------------------------------------
# Section 8 — quarterly guidance, tolerances and non-calendar fiscal years
# ---------------------------------------------------------------------------

def test_a_point_with_a_tolerance_is_accepted_as_guidance():
    release = _extract(
        "The Company's outlook for the second quarter of fiscal 2027 is as follows: "
        "Revenue is expected to be $91.0 billion, plus or minus 2%.",
        filed="2026-05-20")
    revenue = release.metrics["revenue"]
    assert revenue.midpoint == pytest.approx(91.0)
    assert (revenue.low, revenue.high) == pytest.approx((89.18, 92.82))


def test_a_bare_point_with_no_tolerance_is_still_refused():
    """The guard that keeps reported ACTUALS out of the guidance record."""
    release = _extract(
        "For fiscal 2027 revenue is expected to be $91.0 billion.",
        filed="2026-05-20")
    assert "revenue" not in release.metrics


def test_a_basis_point_tolerance_is_converted_correctly():
    release = _extract(
        "The outlook for the second quarter of fiscal 2027 is as follows: "
        "Gross margins are expected to be 75.0%, plus or minus 50 basis points.",
        filed="2026-05-20")
    margin = release.metrics["gross_margin"]
    assert (margin.low, margin.high) == pytest.approx((0.745, 0.755))


def test_quarterly_guidance_is_labelled_as_a_quarter():
    release = _extract(
        "The Company's outlook for the second quarter of fiscal 2027 is as follows: "
        "Revenue is expected to be $91.0 billion, plus or minus 2%.",
        filed="2026-05-20")
    revenue = release.metrics["revenue"]
    assert revenue.fiscal_period == "Q2 FY2027"
    assert revenue.period_type == G.GuidancePeriodType.QUARTER
    assert revenue.fiscal_year == 2027


def test_annual_guidance_is_labelled_as_annual():
    release = _extract(
        "For the full year 2026, the Company expects revenue growth of 2% to 3%.")
    growth = release.metrics["revenue_growth"]
    assert growth.fiscal_period == "FY2026"
    assert growth.period_type == G.GuidancePeriodType.ANNUAL


def test_a_non_calendar_fiscal_year_ahead_of_the_calendar_is_accepted():
    """NVIDIA files in May 2026 and guides fiscal 2027. Requiring the value to
    name the calendar year is what discarded every NVDA figure."""
    release = _extract(
        "For the full year fiscal 2027, the Company expects revenue growth of "
        "20% to 25%.", filed="2026-05-20")
    assert release.metrics["revenue_growth"].fiscal_year == 2027


def test_a_prior_year_range_is_still_rejected_as_not_forward_looking():
    release = _extract(
        "For the full year 2024, the Company expects revenue growth of 2% to 3%.",
        filed="2026-07-30")
    assert "revenue_growth" not in release.metrics
    assert any("cannot be stating forward guidance" in w for w in release.warnings)


def test_the_year_attached_to_the_number_beats_the_span_it_sits_under():
    """"in the 3% to 4% range in 2026, improving to 5% or better in 2028"
    under a "2026-2028" heading is FY2026 guidance."""
    release = _extract(
        "The Company's long-term outlook for 2026-2028 includes: "
        "Adjusted EBITDA* growth in the 3% to 4% range in 2026, improving to "
        "5% or better in 2028.")
    ebitda = release.metrics["adjusted_ebitda_growth"]
    assert ebitda.fiscal_period == "FY2026"
    assert (ebitda.low, ebitda.high) == pytest.approx((0.03, 0.04))


def test_a_figure_inherits_the_period_from_its_outlook_header():
    """NVIDIA's gross-margin and operating-expense lines name no period at
    all; the header three sentences above is their only statement of one."""
    release = _extract(
        "The Company's outlook for the second quarter of fiscal 2027 is as follows: "
        "Revenue is expected to be $91.0 billion, plus or minus 2%. "
        "GAAP and non-GAAP gross margins are expected to be 74.9% and 75.0%, "
        "respectively, plus or minus 50 basis points.",
        filed="2026-05-20")
    assert release.metrics["adjusted_gross_margin"].fiscal_period == "Q2 FY2027"


def test_a_gaap_and_non_gaap_pair_is_never_read_as_one_range():
    """"74.9% and 75.0%" are two POINT values for two bases. Reading them as a
    74.9%-75.0% range would invent a spread and mix the two bases."""
    release = _extract(
        "The outlook for the second quarter of fiscal 2027 is as follows: "
        "GAAP and non-GAAP gross margins are expected to be 74.9% and 75.0%, "
        "respectively, plus or minus 50 basis points.",
        filed="2026-05-20")
    adjusted = release.metrics["adjusted_gross_margin"]
    assert adjusted.low == adjusted.high
    assert adjusted.basis == G.BASIS_ADJUSTED
    assert adjusted.bound_type == G.GuidanceBound.APPROXIMATELY


# ---------------------------------------------------------------------------
# Floors (AT&T states most of its plan as minimums)
# ---------------------------------------------------------------------------

def test_a_floor_is_captured_as_a_minimum_not_a_midpoint():
    release = _extract(
        "The Company's 2026 outlook includes free cash flow of $18 billion+ in 2026.")
    fcf = release.metrics["free_cash_flow"]
    assert fcf.bound_type == G.GuidanceBound.AT_LEAST
    assert fcf.low == fcf.high == pytest.approx(18.0)


def test_a_number_far_from_the_metric_name_is_not_claimed_as_its_floor():
    """"free cash flow* through 2028, its plans to return $45 billion+ to
    shareholders" is a capital-return plan, not free-cash-flow guidance."""
    release = _extract(
        "AT&T maintains its outlook for higher free cash flow* through 2028, its "
        "plans to return $45 billion+ to shareholders during 2026-2028.")
    fcf = release.metrics.get("free_cash_flow")
    assert fcf is None or fcf.low != pytest.approx(45.0)


# ---------------------------------------------------------------------------
# Section 9 — CURRENT, SUPERSEDED, WITHDRAWN, EXPIRED
# ---------------------------------------------------------------------------

def _metric(name="revenue_growth", low=0.02, high=0.03, fiscal_year=2026,
            period="FY2026", period_type=G.GuidancePeriodType.ANNUAL,
            status=G.GuidanceStatus.CURRENT, accession="a"):
    return G.GuidanceMetric(
        name=name, low=low, high=high, unit=G.GuidanceUnit.RATIO,
        basis=G.BASIS_GAAP, fiscal_year=fiscal_year,
        evidence_id=f"dcf.guidance.{name}.current", source_excerpt="x",
        guidance_id="gd_test", issued_at="2026-07-30", fiscal_period=period,
        period_type=period_type, source_accession=accession,
        source_evidence_ids=(f"dcf.guidance.{name}.current",), status=status)


def _release(filed, metrics, accession="a"):
    return G.GuidanceRelease(symbol="TEST", fiscal_year=2026, accession=accession,
                             document=DOC, filed=filed,
                             metrics={m.name: m for m in metrics})


def test_the_newest_statement_of_EACH_METRIC_wins():
    """Per-metric supersession, not per-release. AT&T's newest release
    reiterates its outlook qualitatively and quantifies nothing; the previous
    one carries the numbers, and both are relevant."""
    older = _release("2026-01-28", [_metric(low=0.02, high=0.05),
                                    _metric(name="free_cash_flow", low=18.0, high=18.0)],
                     accession="old")
    newer = _release("2026-07-22", [_metric(low=0.02, high=0.03)], accession="new")
    current, superseded = G.select_current_guidance([older, newer])
    assert current.metrics["revenue_growth"].high == pytest.approx(0.03)
    assert current.metrics["free_cash_flow"].low == pytest.approx(18.0)
    assert superseded


def test_withdrawn_guidance_is_never_current():
    withdrawn = _release("2026-07-30", [_metric(status=G.GuidanceStatus.WITHDRAWN)])
    current, superseded = G.select_current_guidance([withdrawn])
    assert current is None
    assert superseded


def test_a_withdrawal_in_the_text_is_detected():
    release = _extract(
        "The Company has withdrawn its full-year 2026 outlook. Revenue growth was "
        "previously expected to be 2% to 3% for 2026.")
    growth = release.metrics.get("revenue_growth")
    if growth is not None:
        assert growth.status == G.GuidanceStatus.WITHDRAWN
        assert growth.status_reason


def test_guidance_for_an_already_finished_year_is_expired():
    stale = _release("2025-01-28", [_metric(fiscal_year=2025, period="FY2025")])
    current, _superseded = G.select_current_guidance([stale], as_of="2026-08-17")
    assert current is None


def test_a_quarterly_outlook_is_not_expired_by_the_calendar():
    """Next-quarter guidance is the freshest forward statement there is; it is
    superseded by the next release, not by the year turning over."""
    quarterly = _release("2026-05-20", [_metric(
        name="revenue", fiscal_year=2026, period="Q2 FY2026",
        period_type=G.GuidancePeriodType.QUARTER, low=90.0, high=92.0)])
    current, _superseded = G.select_current_guidance([quarterly], as_of="2026-08-17")
    assert current is not None


# ---------------------------------------------------------------------------
# Section 11 — validation
# ---------------------------------------------------------------------------

def test_a_well_formed_metric_passes_validation():
    assert G.validate_guidance_metric(_metric()) == []


def test_a_metric_with_no_period_is_rejected():
    problems = G.validate_guidance_metric(_metric(period=None))
    assert "no fiscal period" in problems


def test_a_metric_with_no_source_filing_is_rejected():
    problems = G.validate_guidance_metric(_metric(accession=""))
    assert "no source filing" in problems


def test_a_metric_outside_the_taxonomy_is_rejected():
    problems = G.validate_guidance_metric(_metric(name="made_up_metric"))
    assert any("not in the reviewed taxonomy" in p for p in problems)


def test_units_that_disagree_with_the_taxonomy_are_rejected():
    wrong_units = G.GuidanceMetric(
        name=G.GuidanceMetricName.ADJUSTED_EPS, low=2.0, high=3.0,
        unit=G.GuidanceUnit.RATIO,           # should be currency_per_share
        basis=G.BASIS_ADJUSTED, fiscal_year=2026, evidence_id="e",
        source_excerpt="x", fiscal_period="FY2026", source_accession="a",
        source_evidence_ids=("e",))
    problems = G.validate_guidance_metric(wrong_units)
    assert any("do not match" in p for p in problems)
