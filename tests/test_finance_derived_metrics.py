"""Phase H.11 — derived metrics, growth identity, guidance coverage, bridges.

The failure that opened this phase: a report stated a financial base of the
trailing twelve months to the latest quarter, and two rows below it showed a
"Revenue Growth" that was the prior fiscal year's annual change, and a
"Current Ratio" carried over from the prior fiscal year while that quarter's
own current assets and liabilities were already selected and available.

Nothing was miscalculated. Each figure was a correct answer to a question the
report was not asking, and both arrived as bare floats with no period
attached, so no consumer could tell.

These tests assert the general rules rather than any issuer's numbers:
a current derived metric comes from the current period, growth is not one
name, and guidance coverage is per metric.
"""

import pytest

from finance import canonical as C
from finance import growth as G
from finance import growth_quality as GQ
from finance import guidance as GD


# ---------------------------------------------------------------------------
# Sections 7-9 — point-in-time ratios come from ONE balance-sheet date
# ---------------------------------------------------------------------------

class _Sel:
    def __init__(self, value, start=None, end=None, source="ttm_calculation", ttm=None):
        self.value = value
        self.period_start = start
        self.as_of_date = end
        self.source = source
        self.ttm = ttm or {"construction_method": "four_discrete_quarters",
                           "validation_status": "valid"}
        self.freshness_status = "current"
        self.provider = "sec"


class _State:
    def __init__(self, flows=None, balance=None, as_of="2026-06-27", net_debt=None):
        self.flows = flows or {}
        self.balance_sheet = balance or {}
        self.total_debt = None
        self.net_debt = net_debt
        self.net_debt_detail = {"net_debt_policy": "cash_only"}
        self.financial_as_of = as_of


def _balance(date="2026-06-27", assets=2_845_271, liabilities=595_824, equity=3_547_750):
    return {
        "current_assets": _Sel(assets, end=date),
        "current_liabilities": _Sel(liabilities, end=date),
        "stockholders_equity": _Sel(equity, end=date),
        "cash_and_cash_equivalents": _Sel(1_000_000, end=date),
    }


def test_the_current_ratio_is_derived_from_the_current_balance_sheet():
    evidence = C.build_canonical_evidence(
        _State(balance=_balance()),
        historical_metrics={"current_ratio": {"value": 5.92, "inputs": ["2025-12-27"]}})
    ratio = evidence.current["current_ratio"]
    assert ratio.value == pytest.approx(2_845_271 / 595_824)
    assert ratio.period == "2026-06-27"
    assert ratio.period_type == C.PeriodKind.DERIVED


def test_the_stale_annual_ratio_cannot_reach_the_current_namespace():
    """The live failure: 5.92 from the prior fiscal year displayed beside a
    balance-sheet date of the latest quarter, whose own components gave 4.78."""
    evidence = C.build_canonical_evidence(
        _State(balance=_balance()),
        historical_metrics={"current_ratio": {"value": 5.92, "inputs": ["2025-12-27"]}})
    assert evidence.current["current_ratio"].value != pytest.approx(5.92)
    assert evidence.historical["current_ratio"].value == pytest.approx(5.92)


def test_a_derived_ratio_carries_its_inputs_and_their_periods():
    """Section 1/37: the lightweight derivation graph."""
    evidence = C.build_canonical_evidence(_State(balance=_balance()))
    ratio = evidence.current["current_ratio"]
    assert ratio.derivation_formula == "current_assets / current_liabilities"
    assert ratio.source_metrics == ("current_assets", "current_liabilities")
    assert ratio.source_periods == ("2026-06-27", "2026-06-27")


def test_components_from_different_dates_are_refused_not_divided():
    """Section 8: a Q2 numerator over an FY denominator is not a ratio."""
    balance = _balance()
    balance["current_liabilities"] = _Sel(595_824, end="2025-12-27")
    evidence = C.build_canonical_evidence(_State(balance=balance))
    assert "current_ratio" not in evidence.current
    codes = {f["code"] for f in evidence.findings}
    assert C.DERIVED_RATIO_PERIOD_MISMATCH in codes


def test_debt_to_equity_is_derived_at_the_current_date():
    balance = _balance()
    evidence = C.build_canonical_evidence(
        _State(balance=balance),
        historical_metrics={"debt_to_equity": {"value": 0.0, "inputs": ["2025-12-27"]}})
    # total_debt is absent here, so no current ratio is derived and the
    # historical one keeps its own namespace rather than being promoted.
    assert "debt_to_equity" not in evidence.current
    assert evidence.historical["debt_to_equity"].value == 0.0


def test_a_non_positive_denominator_produces_no_ratio_at_all():
    """Negative equity does not make a leverage ratio large; it makes it
    meaningless, and a number here would give a later stage something false
    to quote."""
    balance = _balance(equity=-500_000)
    evidence = C.build_canonical_evidence(_State(balance=balance))
    assert "debt_to_equity" not in evidence.current


# ---------------------------------------------------------------------------
# Sections 4-6 — growth is not one name
# ---------------------------------------------------------------------------

def test_each_growth_kind_has_its_own_evidence_id():
    ids = set(G.EVIDENCE_IDS.values())
    assert len(ids) == len(G.GrowthKind.ALL)
    assert "current.ttm.revenue_growth_yoy" in ids
    assert "historical.fy.revenue_growth_yoy" in ids
    # Section 4: no ambiguous bare name anywhere in the vocabulary.
    assert "revenue_growth" not in ids


def test_current_precedence_prefers_trailing_twelve_months():
    assert G.CURRENT_PRECEDENCE[0] == G.GrowthKind.TTM_YOY
    assert G.CURRENT_PRECEDENCE[-1] == G.GrowthKind.FY_YOY


def test_a_fiscal_year_rate_reaching_the_current_slot_is_labelled():
    """Section 6: a fallback is allowed, unlabelled is not."""
    growth_set = G.GrowthSet(metrics={
        G.GrowthKind.FY_YOY: G.GrowthMetric(
            kind=G.GrowthKind.FY_YOY, value=0.1087,
            evidence_id=G.EVIDENCE_IDS[G.GrowthKind.FY_YOY])})
    growth_set.current_kind = G.GrowthKind.FY_YOY
    assert growth_set.current.label == "last fiscal year YoY"

    evidence = C.build_canonical_evidence(_State(balance=_balance()), growth_set=growth_set)
    assert evidence.current_growth_label == "last fiscal year YoY"
    codes = {f["code"] for f in evidence.findings}
    assert C.CURRENT_GROWTH_USED_HISTORICAL_PERIOD in codes


def test_a_trailing_rate_in_the_current_slot_raises_no_finding():
    growth_set = G.GrowthSet(metrics={
        G.GrowthKind.TTM_YOY: G.GrowthMetric(
            kind=G.GrowthKind.TTM_YOY, value=0.148,
            current_period="2025-06-29..2026-06-27",
            comparison_period="2024-06-30..2025-06-28",
            evidence_id=G.EVIDENCE_IDS[G.GrowthKind.TTM_YOY])})
    growth_set.current_kind = G.GrowthKind.TTM_YOY
    evidence = C.build_canonical_evidence(_State(balance=_balance()), growth_set=growth_set)
    assert evidence.current["revenue_growth"].value == pytest.approx(0.148)
    assert evidence.current_growth_label == "TTM YoY"
    assert C.CURRENT_GROWTH_USED_HISTORICAL_PERIOD not in {
        f["code"] for f in evidence.findings}


def test_the_fiscal_year_rate_stays_in_the_historical_namespace():
    """Section 33: a Bull Case asking for current growth cannot receive it."""
    growth_set = G.GrowthSet(metrics={
        G.GrowthKind.TTM_YOY: G.GrowthMetric(
            kind=G.GrowthKind.TTM_YOY, value=0.148,
            evidence_id=G.EVIDENCE_IDS[G.GrowthKind.TTM_YOY]),
        G.GrowthKind.FY_YOY: G.GrowthMetric(
            kind=G.GrowthKind.FY_YOY, value=0.1087,
            evidence_id=G.EVIDENCE_IDS[G.GrowthKind.FY_YOY]),
    })
    growth_set.current_kind = G.GrowthKind.TTM_YOY
    evidence = C.build_canonical_evidence(_State(balance=_balance()), growth_set=growth_set)
    assert "revenue_growth_fy_yoy_growth" in evidence.historical
    assert "revenue_growth_fy_yoy_growth" not in evidence.current
    assert evidence.current["revenue_growth"].value == pytest.approx(0.148)


def test_overlapping_trailing_windows_are_refused():
    """Section 5: a window ending after the current one began measures part
    of the period against itself."""
    assert G._windows_overlap("2025-06-29", "2025-09-30") is True   # noqa: SLF001
    assert G._windows_overlap("2025-06-29", "2025-06-28") is False  # noqa: SLF001


# ---------------------------------------------------------------------------
# Sections 11-14 — guidance coverage is per metric
# ---------------------------------------------------------------------------

def _entry(low=1.0, high=1.2, status="CURRENT"):
    return {"low": low, "high": high, "status": status}


def test_no_guidance_at_all_is_unavailable():
    matrix = GD.build_guidance_matrix({}, releases_examined=2)
    assert matrix["guidance_coverage_status"] == GD.GuidanceCoverage.UNAVAILABLE
    assert matrix["absence_reason"] == GD.GuidanceAbsence.NO_GUIDANCE_EXISTS


def test_capex_only_is_partial_not_unavailable():
    """The live failure: current capital-expenditure guidance, no revenue
    guidance, and a report that said guidance was unavailable."""
    matrix = GD.build_guidance_matrix(
        {GD.GuidanceMetricName.CAPEX: _entry()}, releases_examined=1)
    assert matrix["guidance_coverage_status"] == GD.GuidanceCoverage.PARTIAL
    assert matrix["rows"]["capex"] == GD.GuidanceMetricStatus.CURRENT
    assert matrix["rows"]["revenue"] == GD.GuidanceMetricStatus.UNAVAILABLE
    assert matrix["absence_reason"] == GD.GuidanceAbsence.NO_REVENUE_GUIDANCE
    assert "unavailable" not in GD.guidance_summary_line(matrix)


def test_revenue_and_eps_together_are_partial():
    matrix = GD.build_guidance_matrix(
        {GD.GuidanceMetricName.CONSOLIDATED_REVENUE: _entry(),
         GD.GuidanceMetricName.EPS: _entry()}, releases_examined=1)
    assert matrix["guidance_coverage_status"] == GD.GuidanceCoverage.PARTIAL
    assert matrix["rows"]["eps"] == GD.GuidanceMetricStatus.CURRENT


def test_every_dcf_relevant_row_is_complete():
    metrics = {}
    for row in GD.DCF_RELEVANT_ROWS:
        names = dict(GD.COVERAGE_ROWS)[row]
        metrics[names[0]] = _entry()
    matrix = GD.build_guidance_matrix(metrics, releases_examined=1)
    assert matrix["dcf_guidance_coverage"] == \
        GD.DcfGuidanceCoverage.COMPLETE_FOR_DCF_RELEVANT_METRICS


def test_an_extraction_failure_is_not_an_absence_of_guidance():
    matrix = GD.build_guidance_matrix({}, releases_examined=3, extraction_failed=True)
    assert matrix["absence_reason"] == GD.GuidanceAbsence.GUIDANCE_EXTRACTION_FAILED
    assert "extraction failure" in GD.guidance_summary_line(matrix)


def test_never_searching_is_not_the_same_as_finding_nothing():
    matrix = GD.build_guidance_matrix({}, releases_examined=None)
    assert all(status == GD.GuidanceMetricStatus.NOT_SEARCHED
               for status in matrix["rows"].values())
    assert matrix["absence_reason"] == GD.GuidanceAbsence.NOT_APPLICABLE


def test_capex_guidance_survives_for_the_assumption_builder():
    """Section 15: one missing row must not discard the others."""
    matrix = GD.build_guidance_matrix(
        {GD.GuidanceMetricName.CAPEX: _entry()}, releases_examined=1)
    assert "capex" in matrix["dcf_rows_current"]
    assert "revenue" in matrix["dcf_rows_missing"]


# ---------------------------------------------------------------------------
# Sections 16-27 — the growth bridge
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected_type, expected", [
    ("Acquisitions contributed 11.0% of the growth.", GQ.GrowthComponentType.ACQUISITION, 0.11),
    ("Higher pricing contributed 10.5%.", GQ.GrowthComponentType.PRICING, 0.105),
    ("Volume increased by 3.0%.", GQ.GrowthComponentType.ORGANIC_VOLUME, 0.03),
    ("Currency translation reduced net sales by 2.0%.", GQ.GrowthComponentType.FX, -0.02),
    ("Commodity price pass-through contributed 6.0%.",
     GQ.GrowthComponentType.COMMODITY_PRICE_EFFECT, 0.06),
    ("Divestitures reduced net sales by 1.5%.", GQ.GrowthComponentType.DIVESTITURE, -0.015),
])
def test_stated_numeric_contributions_are_read(text, expected_type, expected):
    components = GQ.extract_components(text)
    match = next(c for c in components if c.type == expected_type)
    assert match.contribution == pytest.approx(expected)
    assert match.qualitative_only is False


def test_a_qualitative_driver_is_recorded_without_a_number():
    """Section 44: do not invent numeric contributions from prose.

    "Driven by strong pricing" says something real and unquantified. Giving
    it a magnitude would be worse than the gap it fills.
    """
    components = GQ.extract_components("Growth was driven by strong pricing across segments.")
    assert len(components) == 1
    assert components[0].type == GQ.GrowthComponentType.PRICING
    assert components[0].contribution is None
    assert components[0].qualitative_only is True


def test_a_qualitative_only_bridge_explains_nothing_arithmetically():
    bridge = GQ.build_growth_bridge(
        0.148, source_texts=["Growth was driven by strong pricing."])
    assert bridge.coverage_status == GQ.GrowthCoverage.NONE
    assert bridge.unexplained_component == pytest.approx(0.148)
    assert bridge.quality == GQ.GrowthQuality.UNKNOWN


@pytest.mark.parametrize("text, quality", [
    ("Acquisitions contributed 11.0% of the growth.", GQ.GrowthQuality.ACQUISITION_HEAVY),
    ("Higher pricing contributed 11.0%.", GQ.GrowthQuality.PRICING_HEAVY),
    ("Volume increased by 11.0%.", GQ.GrowthQuality.PREDOMINANTLY_ORGANIC),
])
def test_a_dominant_stated_contribution_classifies_the_growth(text, quality):
    assert GQ.build_growth_bridge(0.148, source_texts=[text]).quality == quality


def test_several_contributions_of_similar_size_are_mixed():
    bridge = GQ.build_growth_bridge(
        0.148, source_texts=["Higher pricing contributed 6.0% and volume increased by 5.0%."])
    assert bridge.quality == GQ.GrowthQuality.MIXED


def test_the_unexplained_part_is_reported_not_distributed():
    bridge = GQ.build_growth_bridge(
        0.148, source_texts=["Acquisitions contributed 11.0% of the growth."])
    assert bridge.unexplained_component == pytest.approx(0.148 - 0.11)
    assert bridge.coverage_status == GQ.GrowthCoverage.PARTIAL


def test_no_decomposition_leaves_the_whole_rate_unexplained():
    bridge = GQ.build_growth_bridge(0.148, source_texts=[])
    assert bridge.components == []
    assert bridge.unexplained_component == pytest.approx(0.148)
    assert bridge.quality == GQ.GrowthQuality.UNKNOWN


def test_high_growth_without_composition_or_guidance_is_flagged():
    """Section 27: reduce confidence, do not invalidate the model."""
    bridge = GQ.build_growth_bridge(0.148, source_texts=[], revenue_guidance_present=False)
    codes = {f["code"] for f in bridge.findings}
    assert GQ.HEADLINE_GROWTH_COMPOSITION_UNCERTAIN in codes
    assert all(f["severity"] == "info" for f in bridge.findings)


def test_modest_growth_without_composition_is_not_flagged():
    assert GQ.build_growth_bridge(0.03, source_texts=[]).findings == []


def test_revenue_guidance_removes_the_composition_warning():
    bridge = GQ.build_growth_bridge(0.148, source_texts=[], revenue_guidance_present=True)
    assert bridge.findings == []


def test_the_assumption_builder_receives_parts_not_a_single_organic_rate():
    """Section 25: the deterministic layer must not invent one number."""
    bridge = GQ.build_growth_bridge(
        0.148, period="TTM",
        source_texts=["Acquisitions contributed 11.0% and higher pricing contributed 2.0%."])
    described = GQ.describe_for_assumptions(bridge)
    assert "organic_growth" not in described
    assert described["contributions"][GQ.GrowthComponentType.ACQUISITION] == pytest.approx(0.11)
    assert described["unexplained_component"] is not None
    assert "not automatically organic" in described["caution"]


def test_contribution_evidence_ids_are_stable():
    """Section 24."""
    assert GQ.COMPONENT_EVIDENCE_IDS[GQ.GrowthComponentType.PRICING] == "growth.bridge.pricing"
    assert GQ.COMPONENT_EVIDENCE_IDS[GQ.GrowthComponentType.ACQUISITION] == \
        "growth.bridge.acquisition"


# ---------------------------------------------------------------------------
# Sections 30-32 — conditions checked against canonical CURRENT evidence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("condition, satisfied", [
    ("operating margin above 22%", True),
    ("TTM revenue growth above 10%", True),
    ("current ratio above 3.0", True),
    # Persistence and further improvement are future conditions even when the
    # threshold is already met today (section 31).
    ("operating margin remains above 22% for the next two reporting periods", False),
    ("TTM revenue growth remains above 10% while FCF conversion improves", False),
    # A genuinely higher bar.
    ("operating margin above 30%", False),
])
def test_conditions_are_checked_against_current_derived_metrics(condition, satisfied):
    from finance import research_pipeline as R

    current = {"operating_margin": {"value": 0.23},
               "revenue_growth": {"value": 0.148},
               "current_ratio": {"value": 4.78}}
    assert R.is_already_satisfied(condition, current) is satisfied
