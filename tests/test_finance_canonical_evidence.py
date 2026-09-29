"""Phase H.9 — canonical current evidence, guidance periods, model bounds,
typed events and condition validation (sections 1-11, 13-19, 22-38, 46-54).

Two failure classes drive everything here, both from live analyses:

  * A report claimed one current trailing-twelve-month base and handed the
    research roles last fiscal year's operating margin and free cash flow,
    because `fundamental_metrics` and the freshness planner's TTM values
    shared names like `operating_margin`.

  * A company issued FULL-YEAR guidance alongside its second-quarter results
    and the report called it "Q2 guidance" -- naming the reporting period as
    the target.

Nothing here is issuer-specific and nothing asserts a valuation.
"""

import pytest

from finance import canonical as C
from finance import guidance as G
from finance import research_pipeline as R
from finance import structural_breaks as SB


# ---------------------------------------------------------------------------
# A minimal stand-in for CurrentFinancialState
# ---------------------------------------------------------------------------

class _Sel:
    def __init__(self, value, start, end, source="ttm_calculation", ttm=None,
                 freshness="ttm", provider="sec"):
        self.value = value
        self.period_start = start
        self.as_of_date = end
        self.source = source
        self.ttm = ttm or {"construction_method": "four_discrete_quarters",
                           "validation_status": "valid"}
        self.freshness_status = freshness
        self.provider = provider


class _State:
    def __init__(self, flows, balance=None, net_debt=None, as_of="2026-06-30"):
        self.flows = flows
        self.balance_sheet = balance or {}
        self.total_debt = None
        self.net_debt = net_debt
        self.net_debt_detail = {"net_debt_policy": "cash_only"}
        self.financial_as_of = as_of


def _ttm_flows(end="2026-06-27", start="2025-06-29"):
    return {
        "revenue": _Sel(41_305, start, end),
        "operating_income": _Sel(6_488, start, end),
        "net_income": _Sel(6_434, start, end),
        "operating_cash_flow": _Sel(10_080, start, end),
        "free_cash_flow": _Sel(8_403, start, end),
        "depreciation_and_amortization": _Sel(521, start, end),
    }


# ---------------------------------------------------------------------------
# Sections 1-3, 8 — current namespace and derived margins
# ---------------------------------------------------------------------------

def test_current_margins_are_derived_from_current_components():
    """Section 8: not read from a stale annual statement."""
    evidence = C.build_canonical_evidence(
        _State(_ttm_flows()),
        historical_metrics={"operating_margin": {"value": 0.1066,
                                                 "inputs": ["2025-12-27"]}})
    current = evidence.current["operating_margin"]
    assert current.value == pytest.approx(6_488 / 41_305)
    assert current.period_type == C.PeriodKind.DERIVED
    assert current.period == "2025-06-29..2026-06-27"


def test_the_historical_value_survives_in_its_own_namespace():
    """Section 2: history stays available, it just cannot masquerade."""
    evidence = C.build_canonical_evidence(
        _State(_ttm_flows()),
        historical_metrics={"operating_margin": {"value": 0.1066,
                                                 "inputs": ["2025-12-27"]}})
    assert evidence.historical["operating_margin"].value == pytest.approx(0.1066)
    assert evidence.current["operating_margin"].value != pytest.approx(0.1066)


def test_the_research_packet_qualifies_every_key():
    """Section 4: a role cannot reach a historical metric by asking for a
    generic name, because no generic name exists in the packet."""
    evidence = C.build_canonical_evidence(
        _State(_ttm_flows()),
        historical_metrics={"free_cash_flow": {"value": 6_735,
                                               "inputs": ["2025-12-27"]}})
    packet = evidence.research_packet()
    assert all(k.startswith("current.") for k in packet["current"])
    assert all(k.startswith("historical.") for k in packet["historical"])
    assert "free_cash_flow" not in packet["current"]
    assert "current.ttm.free_cash_flow" in packet["current"]


def test_a_margin_is_not_derived_across_mismatched_periods():
    flows = _ttm_flows()
    flows["operating_income"] = _Sel(6_488, "2024-12-29", "2025-12-27",
                                     source="annual_sec_filing")
    evidence = C.build_canonical_evidence(_State(flows))
    assert "operating_margin" not in evidence.current
    assert any("would belong to neither period" in w for w in evidence.warnings)


# ---------------------------------------------------------------------------
# Sections 6-7 — metric-specific TTM and the same-base invariant
# ---------------------------------------------------------------------------

def test_metrics_sharing_one_end_date_are_one_base():
    evidence = C.build_canonical_evidence(_State(_ttm_flows()))
    assert evidence.base_period == "2026-06-27"
    assert evidence.base_period_aligned is True
    assert evidence.findings == []


def test_one_lagging_metric_breaks_the_single_base_claim():
    """Section 7: a valid annual D&A sitting inside a TTM base means the
    snapshot is not uniformly TTM, and the whole label must not be applied."""
    flows = _ttm_flows()
    flows["depreciation_and_amortization"] = _Sel(
        521, "2024-12-29", "2025-12-27", source="annual_sec_filing", ttm={})
    evidence = C.build_canonical_evidence(_State(flows))
    assert evidence.base_period_aligned is False
    codes = {f["code"] for f in evidence.findings}
    assert C.TTM_BASE_PERIOD_MISMATCH in codes


def test_a_metric_ending_earlier_than_the_rest_is_named():
    flows = _ttm_flows()
    flows["free_cash_flow"] = _Sel(8_403, "2024-12-29", "2025-12-27")
    evidence = C.build_canonical_evidence(_State(flows))
    assert evidence.base_period_aligned is False
    finding = next(f for f in evidence.findings
                   if f["code"] == C.TTM_BASE_PERIOD_MISMATCH)
    assert "free_cash_flow" in " ".join(finding.get("lagging", []))


def test_every_current_metric_carries_its_own_period_and_definition():
    """Section 1/6: each metric validates and describes itself."""
    evidence = C.build_canonical_evidence(_State(_ttm_flows()))
    for name, metric in evidence.current.items():
        assert metric.period, name
        assert metric.period_type in C.PeriodKind.ALL, name
        assert metric.evidence_id, name


# ---------------------------------------------------------------------------
# Section 5 — cross-section consistency
# ---------------------------------------------------------------------------

def test_sections_quoting_the_canonical_value_are_consistent():
    evidence = C.build_canonical_evidence(_State(_ttm_flows()))
    findings = C.validate_section_consistency(
        {"snapshot": {"free_cash_flow": 8_403},
         "bull": {"free_cash_flow": 8_403}}, evidence)
    assert findings == []


def test_a_section_quoting_a_stale_value_is_a_conflict():
    """The live failure: Snapshot on the current figure, Bull Case on last
    fiscal year's."""
    evidence = C.build_canonical_evidence(_State(_ttm_flows()))
    findings = C.validate_section_consistency(
        {"bull": {"free_cash_flow": 6_735}}, evidence)
    assert findings
    assert findings[0]["code"] == C.CANONICAL_CURRENT_EVIDENCE_CONFLICT
    assert findings[0]["section"] == "bull"


# ---------------------------------------------------------------------------
# Sections 10-11 — guidance issue period vs target period
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Second-Quarter 2026 results were announced today.", "Q2 FY2026"),
    ("First-Quarter 2026 results", "Q1 FY2026"),
    ("Third-quarter 2026 results", "Q3 FY2026"),
    ("Fourth-Quarter 2026 earnings", "Q4 FY2026"),
])
def test_the_reporting_period_is_detected(text, expected):
    assert G.detect_reporting_period(text) == expected


def test_full_year_guidance_issued_with_quarterly_results_targets_the_year():
    """The headline fix. A company raising 2026 guidance in its second-quarter
    release is NOT issuing Q2 guidance."""
    release = G.extract_guidance_from_text(
        "Second-Quarter 2026 results. Strong performance results in the Company "
        "increasing 2026 guidance with estimated reported sales of $101.1 Billion.",
        "TEST", "0000-26-1", "ex991.htm", "2026-07-15")
    revenue = release.metrics["revenue"]
    assert revenue.fiscal_period == "FY2026"
    assert revenue.target_period_type == G.GuidanceTargetType.CURRENT_FISCAL_YEAR
    assert revenue.issued_with_reporting_period == "Q2 FY2026"
    assert revenue.midpoint == pytest.approx(101.1)


def test_next_quarter_guidance_targets_the_quarter():
    release = G.extract_guidance_from_text(
        "Second-Quarter 2026 results. The Company's outlook for the third quarter of "
        "2026 is as follows: Revenue is expected to be approximately $13.0 billion.",
        "TEST", "0000-26-1", "ex991.htm", "2026-08-04")
    revenue = release.metrics["revenue"]
    assert revenue.fiscal_period == "Q3 FY2026"
    assert revenue.target_period_type == G.GuidanceTargetType.NEXT_QUARTER
    assert revenue.issued_with_reporting_period == "Q2 FY2026"


def test_the_target_and_issued_with_periods_are_both_published():
    release = G.extract_guidance_from_text(
        "Second-Quarter 2026 results. Increasing 2026 guidance with estimated "
        "reported sales of $101.1 Billion.",
        "TEST", "0000-26-1", "ex991.htm", "2026-07-15")
    record = release.metrics["revenue"].to_dict()
    assert record["target_period"] == "FY2026"
    assert record["issued_with_reporting_period"] == "Q2 FY2026"
    assert record["target_period_type"] == G.GuidanceTargetType.CURRENT_FISCAL_YEAR


# ---------------------------------------------------------------------------
# Guidance parsing guards the period fix depends on
# ---------------------------------------------------------------------------

def test_a_range_whose_ends_disagree_about_units_is_not_a_range():
    """"sales growth of 6.6% to $25.3 Billion" is a rate and a level joined by
    the word "to". Read as a range it produced a fabricated revenue-growth
    guidance that then anchored the DCF's year-1 assumption."""
    release = G.extract_guidance_from_text(
        "For 2026 the Company expects reported sales growth of 6.6% to $25.3 Billion.",
        "TEST", "0000-26-1", "ex991.htm", "2026-07-15")
    growth = release.metrics.get("revenue_growth")
    assert growth is None or growth.high != pytest.approx(25.3)


def test_an_increment_is_not_read_as_a_range():
    """"increasing guidance BY $0.13 TO $11.68" raises guidance by $0.13,
    arriving at $11.68. It is not a $0.13-$11.68 range."""
    release = G.extract_guidance_from_text(
        "For 2026 the Company is increasing adjusted EPS guidance by $0.13 to $11.68.",
        "TEST", "0000-26-1", "ex991.htm", "2026-07-15")
    eps = release.metrics.get("adjusted_earnings_per_share")
    assert eps is None or eps.low != pytest.approx(0.13)


def test_a_reported_actual_is_not_read_as_guidance():
    release = G.extract_guidance_from_text(
        "The 2026 outlook follows. Second-Quarter 2026 reported sales growth was 6.6%.",
        "TEST", "0000-26-1", "ex991.htm", "2026-07-15")
    assert "revenue_growth" not in release.metrics


# ---------------------------------------------------------------------------
# Sections 23-28 — typed post-balance-sheet events
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("form,items,text,expected", [
    ("424B5", "", "senior notes due 2035 at a fixed rate",
     SB.PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE),
    ("424B5", "", "an offering of common stock",
     SB.PostBalanceSheetEventType.ISSUER_EQUITY_ISSUANCE),
    ("424B5", "", "convertible senior notes",
     SB.PostBalanceSheetEventType.CONVERTIBLE_ISSUANCE),
    ("144", "", "", SB.PostBalanceSheetEventType.INSIDER_SECONDARY_SALE),
    ("8-K", "2.03", "creation of a direct financial obligation",
     SB.PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE),
    ("8-K", "3.02", "unregistered sale of equity securities",
     SB.PostBalanceSheetEventType.ISSUER_EQUITY_ISSUANCE),
    ("8-K", "2.01", "completion of an acquisition",
     SB.PostBalanceSheetEventType.ACQUISITION),
    ("FWP", "", "", SB.PostBalanceSheetEventType.UNKNOWN),
])
def test_security_events_are_typed_from_form_and_language(form, items, text, expected):
    event_type, confidence = SB.classify_security_event(form, items, text)
    assert event_type == expected
    assert 0.0 <= confidence <= 1.0


def test_debt_issuance_is_not_dilution():
    """Section 26."""
    impact = SB.event_impact(SB.PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE)
    assert impact["potential_dilution"] is False
    assert impact["affects_share_count"] is False
    assert impact["affects_debt"] is True


def test_an_insider_secondary_sale_changes_nothing_about_the_issuer():
    """Section 25."""
    impact = SB.event_impact(SB.PostBalanceSheetEventType.INSIDER_SECONDARY_SALE)
    assert impact["affects_share_count"] is False
    assert impact["potential_dilution"] is False
    assert impact["affects_cash"] is False
    assert impact["requires_reassessment"] is False


def test_issuer_primary_equity_is_dilution():
    impact = SB.event_impact(SB.PostBalanceSheetEventType.ISSUER_EQUITY_ISSUANCE)
    assert impact["affects_share_count"] is True
    assert impact["potential_dilution"] is True


def test_a_convertible_is_debt_now_and_potential_dilution_later():
    """Section 27: both, not either."""
    impact = SB.event_impact(SB.PostBalanceSheetEventType.CONVERTIBLE_ISSUANCE)
    assert impact["affects_debt"] is True
    assert impact["potential_dilution"] is True
    assert impact["affects_share_count"] is False


def test_an_unknown_event_asserts_nothing():
    """Section 28."""
    impact = SB.event_impact(SB.PostBalanceSheetEventType.UNKNOWN)
    assert not any(impact[flag] for flag in impact)


# ---------------------------------------------------------------------------
# Sections 32-38 — condition validation
# ---------------------------------------------------------------------------

CURRENT = {"operating_margin": {"value": 0.16}, "revenue_growth": {"value": 0.35}}


@pytest.mark.parametrize("text,satisfied", [
    ("operating margin above 15%", True),
    ("revenue growth above 30%", True),
    ("operating margin above 25%", False),
    ("revenue growth above 40%", False),
    # Section 33: a persistence requirement makes it a future development.
    ("operating margin remains above 15% for the next four quarters", False),
    ("operating margin expands from 16% toward 20%", False),
])
def test_a_condition_current_evidence_already_meets_is_detected(text, satisfied):
    assert R.is_already_satisfied(text, CURRENT) is satisfied


def test_a_condition_built_on_a_software_limit_is_rejected():
    """Section 34: the model cap is not a company thesis."""
    assert R.uses_model_threshold("revenue growth falls below the 25% model cap") is True
    assert R.uses_model_threshold(
        "operating margin drops below the configured floor") is True
    # A threshold that names a real number rather than a model value is fine.
    assert R.uses_model_threshold("revenue growth falls below 25%") is False
    # Naming a model value WITHOUT comparing against it is a legitimate
    # reassessment trigger, not a software-limit threshold.
    assert R.uses_model_threshold(
        "a company-specific WACC replaces the configured default") is False


def test_already_satisfied_and_model_threshold_conditions_are_dropped():
    routed = R._route_conditions_by_direction(  # noqa: SLF001
        {"conditions_that_strengthen_the_view": [
            "operating margin above 15%",
            "operating margin expands from 16% toward 22%"],
         "conditions_that_weaken_the_view": [
             "revenue growth falls below the 25% model cap",
             "free cash flow deteriorates while leverage rises"],
         "reassessment_triggers": []},
        current_metrics=CURRENT)
    assert routed["conditions_that_strengthen_the_view"] == [
        "operating margin expands from 16% toward 22%"]
    assert routed["conditions_that_weaken_the_view"] == [
        "free cash flow deteriorates while leverage rises"]


def test_a_lone_condition_is_never_dropped():
    routed = R._route_conditions_by_direction(  # noqa: SLF001
        {"conditions_that_strengthen_the_view": ["operating margin above 15%"],
         "conditions_that_weaken_the_view": [],
         "reassessment_triggers": []},
        current_metrics=CURRENT)
    assert len(routed["conditions_that_strengthen_the_view"]) == 1


def test_a_guidance_availability_trigger_is_dropped_when_guidance_exists():
    """Section 38."""
    guidance = {"revenue": {"target_period_type": G.GuidanceTargetType.CURRENT_FISCAL_YEAR}}
    routed = R._route_conditions_by_direction(  # noqa: SLF001
        {"conditions_that_strengthen_the_view": [],
         "conditions_that_weaken_the_view": [],
         "reassessment_triggers": [
             "full-year guidance becomes available",
             "a company-specific WACC replaces the configured default"]},
        guidance_metrics=guidance)
    assert routed["reassessment_triggers"] == [
        "a company-specific WACC replaces the configured default"]


def test_a_guidance_availability_trigger_survives_when_none_exists():
    routed = R._route_conditions_by_direction(  # noqa: SLF001
        {"conditions_that_strengthen_the_view": [],
         "conditions_that_weaken_the_view": [],
         "reassessment_triggers": [
             "full-year guidance becomes available",
             "a company-specific WACC replaces the configured default"]},
        guidance_metrics={})
    assert len(routed["reassessment_triggers"]) == 2


# ---------------------------------------------------------------------------
# Sections 29-30 — risk consistency
# ---------------------------------------------------------------------------

def test_the_reviewers_risk_stands_when_the_final_stage_lowers_it_silently():
    risk, finding = R.reconcile_final_risk("moderate", "high", "")
    assert risk == "high"
    assert finding["code"] == R.RISK_RECONCILIATION_REQUIRED
    assert finding["severity"] == "warning"


def test_a_reasoned_reconciliation_is_accepted_and_recorded():
    risk, finding = R.reconcile_final_risk(
        "moderate", "high",
        "The reviewer weighted a post-quarter debt issuance that has since been "
        "refinanced at a materially lower coupon.")
    assert risk == "moderate"
    assert finding["severity"] == "info"


def test_matching_or_higher_final_risk_needs_no_reconciliation():
    assert R.reconcile_final_risk("high", "high", "") == ("high", None)
    assert R.reconcile_final_risk("high", "moderate", "") == ("high", None)


def test_an_unrecognized_risk_level_is_left_alone():
    risk, finding = R.reconcile_final_risk("unusual", "high", "")
    assert risk == "unusual" and finding is None


# ---------------------------------------------------------------------------
# Sections 4-5 — a present-tense claim cannot cite last fiscal year
# ---------------------------------------------------------------------------

def _index(**values):
    from finance.evidence import EvidenceItem
    return {k.replace("__", "."): EvidenceItem(k.replace("__", "."), k, v)
            for k, v in values.items()}


def test_citing_the_fiscal_year_figure_when_a_current_one_exists_is_a_conflict():
    """The live failure, reduced: one report, two free cash flows."""
    index = _index(fundamental__free_cash_flow=6_735.0, current__free_cash_flow=8_403.0)
    conflicts = C.conflicting_historical_citations(["fundamental.free_cash_flow"], index)
    assert [name for name, _, _ in conflicts] == ["free_cash_flow"]


def test_a_historical_metric_with_no_current_twin_is_never_touched():
    """A five-year CAGR has no current counterpart and citing it is correct."""
    index = _index(fundamental__revenue_cagr=0.072, current__revenue=41_305.0)
    assert C.conflicting_historical_citations(["fundamental.revenue_cagr"], index) == []


def test_two_periods_that_agree_are_not_a_conflict():
    index = _index(fundamental__operating_margin=0.1570,
                   current__operating_margin=0.1572)
    assert C.conflicting_historical_citations(["fundamental.operating_margin"], index) == []


def test_citing_the_current_id_is_always_fine():
    index = _index(fundamental__net_debt=-1_443.0, current__net_debt=-1_860.0)
    assert C.conflicting_historical_citations(["current.net_debt"], index) == []


def test_the_evidence_index_exposes_current_and_historical_under_separate_ids():
    """Section 4: a role cannot reach a stale value through a generic name,
    because both names exist and each says which period it covers."""
    from finance.evidence import build_evidence_index
    index = build_evidence_index({
        "symbol": "XYZ",
        "fundamental_metrics": {"free_cash_flow": {"value": 6_735.0,
                                                   "inputs": ["2025-12-27"]}},
        "canonical_evidence": {"current": {"free_cash_flow": {
            "value": 8_403.0, "period": "2025-06-29..2026-06-27",
            "period_type": "TTM"}}},
    })
    assert index["current.free_cash_flow"].value == 8_403.0
    assert "2025-06-29..2026-06-27" in index["current.free_cash_flow"].label
    assert "2025-12-27" in index["fundamental.free_cash_flow"].label
    assert index["fundamental.free_cash_flow"].source_type == "reported_historical"


def test_a_stale_citation_is_quarantined_rather_than_failing_the_stage():
    """Phase H.13 changed the CONSEQUENCE, not the rule.

    This asserted that `_evidence_list` raised. Rejection turned out to be
    worse than the problem: two live runs, on different issuers and
    different business models, each lost a REQUIRED researcher stage to it
    -- the model retried three times, cited the same id each time, and the
    stage died, taking the rebuttal with it by prerequisite. A report with
    no bull case is worse than one sentence resting on last year's figure.

    The rule still holds: the passage does not reach the reader. It is
    stubbed by the same quarantine every other semantic misuse uses, and
    the stage survives.
    """
    index = _index(fundamental__net_debt=-1_443.0, current__net_debt=-1_860.0)

    # The citation itself no longer raises.
    assert R._evidence_list(  # noqa: SLF001
        {"evidence_cited": ["fundamental.net_debt"]}, "evidence_cited", index) == [
        "fundamental.net_debt"]

    output = {"key_risks": [
        {"risk": "Net debt of -$1.44B is comfortable.",
         "evidence_cited": ["fundamental.net_debt"]}]}
    findings = R._stale_citation_findings(output, index)  # noqa: SLF001
    assert [f.field_path for f in findings] == ["key_risks[0].risk"]
    assert findings[0].label == "STALE_METRIC_CITATION"

    result, _records, fatal = R.apply_quarantine(output, findings)
    assert fatal == []
    assert len(result["key_risks"]) == 1
    assert "withheld" in result["key_risks"][0]["risk"]


def test_a_citation_with_no_current_twin_produces_no_finding():
    """The rule is unchanged: only a SUPERSEDED figure is caught."""
    index = _index(fundamental__revenue_cagr=0.072, current__revenue=41_305.0)
    output = {"key_risks": [{"risk": "The five-year CAGR was 7.2%.",
                             "evidence_cited": ["fundamental.revenue_cagr"]}]}
    assert R._stale_citation_findings(output, index) == []  # noqa: SLF001


# ---------------------------------------------------------------------------
# A leaked internal reference is stubbed, never fatal (section 17)
#
# THE LIVE FAILURE: `finance/report_model.py::_reject_empty` raised an
# uncaught ValueError when a stage's `limiting_factors` text named something
# internal to this program -- a field path or an enum constant it had read
# off the evidence packet -- because nothing upstream of that last boundary
# ever screened for it. `apply_quarantine` already stubs every other class of
# semantic violation instead of crashing the whole analysis; this is that
# same treatment, generalized to leaked internal references. Fictional
# fixture text, not the triggering issuer's actual sentence.
# ---------------------------------------------------------------------------

def test_a_leaked_internal_reference_is_found_and_stubbed_not_raised():
    output = {"limiting_factors": [
        "Management guidance requires significant margin expansion from "
        "CURRENT_QUARTER_GUIDANCE to meet the implied full-year target."]}
    findings = R._internal_reference_findings(output)  # noqa: SLF001
    assert [f.field_path for f in findings] == ["limiting_factors[0]"]
    assert findings[0].label == R._INTERNAL_REFERENCE_LABEL  # noqa: SLF001
    assert findings[0].severity == "semantic_misuse"

    result, _records, fatal = R.apply_quarantine(output, findings)
    assert fatal == []
    assert len(result["limiting_factors"]) == 1, "stubbed, never dropped"
    stubbed = result["limiting_factors"][0]
    assert "withheld" in stubbed
    assert "CURRENT_QUARTER_GUIDANCE" not in stubbed
    # The wording must describe the ACTUAL fault, not business-model misuse.
    assert "business model" not in stubbed


def test_a_clean_limiting_factor_produces_no_finding():
    output = {"limiting_factors": [
        "Management guidance requires meaningful margin expansion next year "
        "to meet the implied full-year target."]}
    assert R._internal_reference_findings(output) == []  # noqa: SLF001


def test_evidence_id_fields_are_never_scanned_or_quarantined():
    """`evidence_ids`/`evidence_cited` legitimately hold literal 'dcf.x.y'
    ids by design -- flagging them would destroy a real citation."""
    output = {"claims": [{"claim": "Revenue grew 12% year over year.",
                          "evidence_ids": ["dcf.guidance.revenue.current"],
                          "evidence_cited": ["dcf.guidance.revenue.current"]}]}
    assert R._internal_reference_findings(output) == []  # noqa: SLF001


def test_internal_reference_findings_are_wired_into_stage_validation():
    """The scan must run for every stage via `_validate_claim_fidelity`, not
    only be reachable in isolation."""
    import inspect
    source = inspect.getsource(R._validate_claim_fidelity)  # noqa: SLF001
    assert "_internal_reference_findings" in source


def test_the_stubbed_text_survives_the_report_models_own_last_boundary():
    """The exact site of the live crash: `report_model.py::_reject_empty`
    must no longer raise once quarantine has already run."""
    from finance import report_model as RM

    output = {"limiting_factors": [
        "Management guidance requires significant margin expansion from "
        "CURRENT_QUARTER_GUIDANCE to meet the implied full-year target."]}
    findings = R._internal_reference_findings(output)  # noqa: SLF001
    quarantined, _records, fatal = R.apply_quarantine(output, findings)
    assert fatal == []
    RM._reject_empty("the 'Limiting factors' conditions",  # noqa: SLF001
                     quarantined["limiting_factors"])


# ---------------------------------------------------------------------------
# Section 22 — a percentage the report will not print is not citable either
# ---------------------------------------------------------------------------

def _payload_with_gap(suitability):
    return {
        "symbol": "XYZ",
        "dcf_suitability": {"dcf_suitability": suitability},
        "valuation_gap": {"available": True, "direction": "above",
                          "difference_pct": 0.809,
                          "market_price_premium_pct": 4.24,
                          "modeled_return_to_value_pct": -0.809},
    }


def test_the_gap_magnitude_is_withheld_when_the_dcf_is_not_suitable():
    from finance.evidence import build_evidence_index
    index = build_evidence_index(_payload_with_gap("LIMITED"))
    assert "valuation_gap.difference_pct" not in index
    assert "valuation_gap.market_price_premium_pct" not in index
    # The direction is still supportable, and still citable.
    assert index["valuation_gap.direction"].value == "above"
    assert "do not quote" in index["valuation_gap.comparison_withheld"].value


def test_the_gap_survives_intact_when_the_dcf_is_suitable():
    from finance.evidence import build_evidence_index
    index = build_evidence_index(_payload_with_gap("SUITABLE"))
    assert index["valuation_gap.difference_pct"].value == 0.809
    assert "valuation_gap.comparison_withheld" not in index


def test_an_unreconciled_share_basis_also_withholds_the_magnitude():
    from finance.evidence import build_evidence_index
    payload = _payload_with_gap("SUITABLE")
    payload["dcf_financial_basis"] = {
        "share_reconciliation": {"status": "MATERIAL_DIFFERENCE"}}
    index = build_evidence_index(payload)
    assert "valuation_gap.difference_pct" not in index
    assert "valuation_gap.direction" in index


# ---------------------------------------------------------------------------
# Section 14 — a derivation may not deny guidance the report is showing
# ---------------------------------------------------------------------------

def test_next_quarter_guidance_is_not_described_as_no_guidance():
    """The live contradiction: the Valuation section listed the company's
    next-quarter revenue guidance while the assumption derivation said none
    was available, and a research role flagged the inconsistency."""
    from finance import forward_assumptions as FA
    evidence = FA.GrowthEvidence(ttm_yoy=0.3954,
                                 guidance_implied_next_period_growth=0.406)
    path = FA.build_growth_path(evidence, forecast_years=5)
    assert "No current guidance was available" not in path.entries[0].derivation
    assert "NEXT QUARTER" in path.entries[0].derivation


def test_component_guidance_is_named_rather_than_denied():
    from finance import forward_assumptions as FA
    evidence = FA.GrowthEvidence(ttm_yoy=0.05,
                                 component_guidance_low=0.03,
                                 component_guidance_high=0.04)
    path = FA.build_growth_path(evidence, forecast_years=5)
    assert "COMPONENT" in path.entries[0].derivation


def test_genuinely_absent_guidance_still_says_so():
    from finance import forward_assumptions as FA
    path = FA.build_growth_path(FA.GrowthEvidence(ttm_yoy=0.05), forecast_years=5)
    assert "No current guidance was available" in path.entries[0].derivation


# ---------------------------------------------------------------------------
# Item 3: a Q4 FY2026 guidance item is not FY2026 guidance merely because
# both contain the token "2026". `finance.claim_validation.
# scan_for_guidance_period_mismatch` already implemented this rule and, like
# `retire_realized_guidance`, was never called from anywhere in the pipeline.
# ---------------------------------------------------------------------------

def _guidance_index(*fiscal_periods):
    from finance.evidence import EvidenceItem
    return {
        f"dcf.guidance.revenue_growth.{i}.current": EvidenceItem(
            evidence_id=f"dcf.guidance.revenue_growth.{i}.current",
            label="x", value="93.0 to 93.0", source_periods=[period])
        for i, period in enumerate(fiscal_periods)
    }


def test_e_a_quarterly_guidance_item_cannot_be_described_as_full_year():
    index = _guidance_index("Q4 FY2026")
    findings = R._guidance_horizon_findings(  # noqa: SLF001
        {"limiting_factors": ["Management guides FY2026 revenue growth to 93%."]}, index)
    assert findings and findings[0].label.startswith("GUIDANCE_PERIOD_MISSTATED")


def test_e_the_same_figure_named_by_its_real_quarter_is_clean():
    index = _guidance_index("Q4 FY2026")
    findings = R._guidance_horizon_findings(  # noqa: SLF001
        {"limiting_factors": ["Q4 FY2026 revenue growth guidance is 93%."]}, index)
    assert findings == []


def test_f_the_same_rule_applies_to_a_margin_claim():
    index = _guidance_index("Q4 FY2026")
    findings = R._guidance_horizon_findings(  # noqa: SLF001
        {"claims": ["Management guides FY2026 operating margin to expand."]}, index)
    assert findings and findings[0].label.startswith("GUIDANCE_PERIOD_MISSTATED")


def test_annual_guidance_may_be_described_as_annual():
    """No false positive: when the guidance really IS annual, saying so is
    fine."""
    index = _guidance_index("FY2026")
    findings = R._guidance_horizon_findings(  # noqa: SLF001
        {"limiting_factors": ["Management guides FY2026 revenue growth to 93%."]}, index)
    assert findings == []


def test_a_guidance_horizon_mismatch_is_stubbed_not_fatal():
    output = {"limiting_factors": ["Management guides FY2026 revenue growth to 93%."]}
    index = _guidance_index("Q4 FY2026")
    findings = R._guidance_horizon_findings(output, index)  # noqa: SLF001
    result, _records, fatal = R.apply_quarantine(output, findings)
    assert fatal == []
    assert len(result["limiting_factors"]) == 1
    assert "withheld" in result["limiting_factors"][0]
    assert "FY2026" not in result["limiting_factors"][0]


# ---------------------------------------------------------------------------
# Item 2: a latest-quarter YoY fact must never become a TTM claim in
# research prose. Checked against AVAILABILITY (finance/canonical.py
# publishes each growth kind under its own evidence id only when it could
# actually be built), never against the claimed number.
# ---------------------------------------------------------------------------

def _growth_index(*available_evidence_ids):
    from finance.evidence import EvidenceItem
    return {eid: EvidenceItem(evidence_id=eid, label="x", value=0.86)
           for eid in available_evidence_ids}


def test_c_a_latest_quarter_fact_cannot_be_described_as_ttm():
    index = _growth_index("current.revenue_growth_latest_quarter_yoy_growth")
    findings = R._growth_frequency_findings(  # noqa: SLF001
        {"claims": ["TTM revenue growth of 86% reflects strong momentum."]}, index)
    assert findings and findings[0].label == "GROWTH_FREQUENCY_MISMATCH"


def test_d_a_ttm_fact_may_be_described_as_ttm():
    index = _growth_index("current.revenue_growth_ttm_yoy_growth")
    findings = R._growth_frequency_findings(  # noqa: SLF001
        {"claims": ["TTM revenue growth of 86% reflects strong momentum."]}, index)
    assert findings == []


def test_the_same_fact_correctly_labelled_latest_quarter_is_clean():
    index = _growth_index("current.revenue_growth_latest_quarter_yoy_growth")
    findings = R._growth_frequency_findings(  # noqa: SLF001
        {"claims": ["Latest-quarter revenue growth of 86% reflects strong momentum."]},
        index)
    assert findings == []


def test_a_full_year_growth_claim_with_no_fy_evidence_is_flagged():
    index = _growth_index("current.revenue_growth_latest_quarter_yoy_growth")
    findings = R._growth_frequency_findings(  # noqa: SLF001
        {"claims": ["Full-year revenue growth of 12% is expected."]}, index)
    assert findings and findings[0].label == "GROWTH_FREQUENCY_MISMATCH"


def test_a_growth_frequency_mismatch_is_stubbed_not_fatal():
    output = {"claims": ["TTM revenue growth of 86% reflects strong momentum.",
                         "Margins remain healthy across segments."]}
    index = _growth_index("current.revenue_growth_latest_quarter_yoy_growth")
    findings = R._growth_frequency_findings(output, index)  # noqa: SLF001
    result, _records, fatal = R.apply_quarantine(output, findings)
    assert fatal == []
    assert len(result["claims"]) == 2, "stubbed in place, never dropped from a min_items field"
    assert "withheld" in result["claims"][0]
    assert "TTM" not in result["claims"][0]


def test_text_with_no_growth_attribution_is_never_scanned():
    index = _growth_index()  # nothing available at all
    findings = R._growth_frequency_findings(  # noqa: SLF001
        {"claims": ["The company operates in a competitive market."]}, index)
    assert findings == []
