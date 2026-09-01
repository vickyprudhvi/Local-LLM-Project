"""Phases 45-51 — the finance kernel's invariants, tested as invariants.

Ticker fixtures prove a regression did not come back. They cannot prove an
architecture, because each one only exercises the path its own issuer
happens to take -- which is why every new stock kept finding a new route to
the same class of error.

These tests assert the RULES instead: that a forbidden transition cannot
happen, that identical numbers with different semantics behave differently,
and that the golden invariants in Phase 50 hold. No issuer is named in any
production path exercised here.
"""

import pytest

from finance import diagnostics as D
from finance import guidance as G
from finance import metric_policy as MP
from finance import semantics as S
from finance.business_model import CashFlowValuationProfile as P


F = S.PeriodFrequency
M = S.MetricIdentity
O = S.Operation


# ---------------------------------------------------------------------------
# Phase 48 — forbidden transitions
# ---------------------------------------------------------------------------

def _flow(metric, frequency, **kw):
    kw.setdefault("flow_or_instant", S.FlowOrInstant.FLOW)
    return S.SemanticFact(metric_id=metric, value=100.0, period_frequency=frequency, **kw)


def test_forbidden_quarter_guidance_over_ttm_cannot_yield_growth():
    verdict = S.compatible_for(
        O.GROWTH,
        _flow(M.REVENUE, F.QUARTER, fiscal_year=2026, fiscal_quarter=3),
        _flow(M.REVENUE, F.TTM, fiscal_year=2026))
    assert not verdict
    assert verdict.code == S.PERIOD_FREQUENCY_MISMATCH


def test_forbidden_current_shares_against_weighted_average_is_not_a_conflict():
    current = S.SemanticFact(metric_id=M.SHARES_CURRENT_OUTSTANDING, value=1_000.0,
                             period_frequency=F.INSTANT,
                             flow_or_instant=S.FlowOrInstant.INSTANT)
    diluted = S.SemanticFact(metric_id=M.SHARES_WEIGHTED_AVERAGE_DILUTED, value=1_400.0,
                             period_frequency=F.ANNUAL,
                             flow_or_instant=S.FlowOrInstant.FLOW)
    verdict = S.compatible_for(O.RECONCILE, current, diluted)
    assert not verdict
    # The distinction Phase 22 requires: not comparable, not a data conflict.
    assert verdict.code == S.SHARE_BASIS_NOT_COMPARABLE
    assert verdict.informational is True


def test_forbidden_component_sum_against_long_term_debt():
    def debt(metric):
        return S.SemanticFact(metric_id=metric, value=100.0, period_frequency=F.INSTANT,
                              flow_or_instant=S.FlowOrInstant.INSTANT,
                              instant_date="2026-06-30")

    verdict = S.compatible_for(O.RECONCILE, debt(M.TOTAL_DEBT), debt(M.LONG_TERM_DEBT))
    assert not verdict
    assert verdict.code == S.INCOMPATIBLE_DEBT_BASIS


def test_forbidden_specialized_simple_fcf_as_owner_cash():
    for profile in (P.INSURER, P.BANK, P.BROKER_DEALER, P.FINANCIAL_INSTITUTION):
        assert MP.evaluate("free_cash_flow", MP.MetricUse.OWNER_CASH_CLAIM,
                           profile).allowed is False


def test_forbidden_framework_becoming_a_period_forecast():
    """Phase 12: a long-term framework may not anchor a named year."""
    kind = G.classify_forward_information(
        "Our long-term model targets gross margin of 55% to 60%",
        G.GuidanceTargetType.CURRENT_FISCAL_YEAR)
    assert kind == G.ForwardInformationKind.LONG_TERM_FRAMEWORK
    assert G.may_anchor_period_forecast(kind) is False


def test_forbidden_mixed_period_ratio():
    income = _flow(M.OPERATING_INCOME, F.ANNUAL, end_date="2025-12-31")
    revenue = _flow(M.REVENUE, F.TTM, end_date="2026-06-30")
    verdict = S.compatible_for(O.RATIO, income, revenue)
    assert not verdict


def test_forbidden_historical_fact_as_a_dcf_input():
    current = S.SemanticFact(metric_id=M.REVENUE, value=1.0, period_frequency=F.TTM,
                             flow_or_instant=S.FlowOrInstant.FLOW,
                             current_or_historical=S.CurrentOrHistorical.CURRENT)
    historical = S.SemanticFact(metric_id=M.REVENUE, value=1.0, period_frequency=F.TTM,
                                flow_or_instant=S.FlowOrInstant.FLOW,
                                current_or_historical=S.CurrentOrHistorical.HISTORICAL)
    assert not S.compatible_for(O.DCF_INPUT, current, historical)


# ---------------------------------------------------------------------------
# Phase 47 — metamorphic: same numbers, different semantics
# ---------------------------------------------------------------------------

def test_metamorphic_period_changes_the_verdict():
    base = _flow(M.REVENUE, F.ANNUAL, fiscal_year=2025)
    annual = _flow(M.REVENUE, F.ANNUAL, fiscal_year=2026)
    quarter = _flow(M.REVENUE, F.QUARTER, fiscal_year=2026, fiscal_quarter=1)

    assert annual.value == quarter.value == 100.0
    assert S.compatible_for(O.GROWTH, annual, base)
    assert not S.compatible_for(O.GROWTH, quarter, base)


def test_metamorphic_business_model_changes_the_verdict():
    """One metric, one value, two issuers -- two different conclusions."""
    claim = "Positive free cash flow demonstrates strong cash generation."
    assert MP.validate_claim(claim, P.STANDARD_OPERATING_COMPANY) == []
    assert MP.validate_claim(claim, P.INSURER)


def test_metamorphic_currency_changes_the_verdict():
    left = _flow(M.REVENUE, F.ANNUAL, fiscal_year=2026, currency="USD")
    same = _flow(M.REVENUE, F.ANNUAL, fiscal_year=2025, currency="USD")
    other = _flow(M.REVENUE, F.ANNUAL, fiscal_year=2025, currency="EUR")
    assert S.compatible_for(O.GROWTH, left, same)
    assert not S.compatible_for(O.GROWTH, left, other)


def test_metamorphic_scope_changes_the_verdict():
    consolidated = _flow(M.REVENUE, F.ANNUAL, fiscal_year=2026)
    segment = S.SemanticFact(metric_id=M.REVENUE, value=100.0, period_frequency=F.ANNUAL,
                             fiscal_year=2025, flow_or_instant=S.FlowOrInstant.FLOW,
                             consolidation_scope=S.ConsolidationScope.SEGMENT)
    assert not S.compatible_for(O.GROWTH, consolidated, segment)


# ---------------------------------------------------------------------------
# Phases 43-44 — one error, one message
# ---------------------------------------------------------------------------

def test_a_symptom_is_suppressed_when_its_cause_is_present():
    """The live cascade: a bound conflict that existed only because an
    invalid guidance derivation reached the clamp."""
    codes = ["GUIDANCE_PERIOD_INCOMPATIBLE", "DCF_MODEL_BOUND_CONFLICT"]
    assert D.suppressed_symptoms(codes) == {
        "DCF_MODEL_BOUND_CONFLICT": "GUIDANCE_PERIOD_INCOMPATIBLE"}


def test_a_symptom_alone_is_still_reported():
    """A bound conflict with no bad derivation above it is a real finding."""
    assert D.suppressed_symptoms(["DCF_MODEL_BOUND_CONFLICT"]) == {}


def test_filtering_keeps_the_cause_and_drops_the_consequence():
    findings = [{"code": "GUIDANCE_PERIOD_INCOMPATIBLE"},
                {"code": "DCF_MODEL_BOUND_CONFLICT"},
                {"code": "SOMETHING_UNRECOGNISED"}]
    kept = [f["code"] for f in D.filter_to_root_causes(findings)]
    assert "GUIDANCE_PERIOD_INCOMPATIBLE" in kept
    assert "DCF_MODEL_BOUND_CONFLICT" not in kept
    # Never silences what it does not understand.
    assert "SOMETHING_UNRECOGNISED" in kept


def test_root_conditions_rank_ahead_of_consequences():
    ranked = D.rank_by_root_cause(["DCF_MODEL_BOUND_CONFLICT", "PERIOD_FREQUENCY_MISMATCH"])
    assert ranked[0] == "PERIOD_FREQUENCY_MISMATCH"


# ---------------------------------------------------------------------------
# Phase 50 — the golden invariants, stated directly
# ---------------------------------------------------------------------------

def test_invariant_compatibility_depends_on_the_operation():
    """The kernel's central claim. A pair is not 'compatible' in the
    abstract; it is compatible FOR something."""
    quarter = _flow(M.REVENUE, F.QUARTER, fiscal_year=2026, fiscal_quarter=3)
    ttm = _flow(M.REVENUE, F.TTM, fiscal_year=2026)
    assert S.compatible_for(O.COMPARE, quarter, ttm)
    assert not S.compatible_for(O.GROWTH, quarter, ttm)


def test_invariant_an_unknown_operation_is_refused():
    """A validator that permits what it does not understand guarantees
    nothing."""
    assert not S.compatible_for("SOME_NEW_OPERATION", _flow(M.REVENUE, F.ANNUAL),
                                _flow(M.REVENUE, F.ANNUAL))


def test_invariant_reporting_a_figure_is_never_restricted():
    """Phase 19/30: policy controls the CONCLUSION, never the number."""
    for profile in (P.INSURER, P.BANK, P.BROKER_DEALER):
        assert MP.evaluate("free_cash_flow", MP.MetricUse.FACTUAL_DISPLAY,
                           profile).allowed is True


def test_invariant_an_ordinary_company_is_unrestricted():
    """The kernel must not cost every normal issuer its analysis."""
    packet = MP.build_relevant_evidence(
        type("C", (), {"profile": P.STANDARD_OPERATING_COMPANY,
                       "standard_fcff_suitability": "SUITABLE",
                       "corroborating_concepts": ()})(),
        canonical_current={"revenue": {}, "free_cash_flow": {}, "current_ratio": {}})
    assert packet.prohibited_interpretations == []
    assert packet.low_information_metrics == []


def test_invariant_pipeline_complete_means_every_required_stage_completed():
    from finance.workflow import _pipeline_overall_status

    complete = {name: "COMPLETE" for name in
                ("bull_researcher", "bear_researcher", "rebuttal_round",
                 "research_manager", "risk_reviewer", "final_investment_synthesizer")}
    assert _pipeline_overall_status(complete) == "COMPLETE"
    degraded = dict(complete, rebuttal_round="FAILED")
    assert _pipeline_overall_status(degraded) == "DEGRADED"


def test_invariant_a_missing_adversarial_stage_caps_confidence():
    from finance.research_pipeline import cap_confidence_for_degraded_pipeline as cap

    assert cap(0.9, []) == 0.9
    assert cap(0.9, ["bull_researcher"]) <= 0.55
    assert cap(0.9, ["bear_researcher"]) <= 0.55
    assert cap(0.9, ["rebuttal_round"]) <= 0.55
    # Never raises a confidence that was already lower.
    assert cap(0.2, ["bull_researcher"]) == 0.2


def test_invariant_a_declined_model_is_not_an_invalid_one():
    V = MP.ValuationMethodStatus
    for status in (V.VALID_BUT_NOT_APPLICABLE, V.NOT_VALID_FOR_CURRENT_FORECAST_PATH):
        wording = MP.VALUATION_STATUS_WORDING[status]
        assert "invalid" not in wording.lower()


def test_invariant_negative_terminal_flow_is_a_forecast_problem():
    """Phase 27-28: the model working correctly is not the model failing."""
    classification = type("C", (), {"profile": P.STANDARD_OPERATING_COMPANY,
                                    "standard_fcff_suitability": "SUITABLE",
                                    "corroborating_concepts": ()})()
    assert MP.valuation_method_status(
        classification, False, True, "DCF_NEGATIVE_TERMINAL_FCFF") == \
        MP.ValuationMethodStatus.NOT_VALID_FOR_CURRENT_FORECAST_PATH


# ---------------------------------------------------------------------------
# Phase 52 — the audit trail
# ---------------------------------------------------------------------------

def test_the_audit_trail_follows_a_number_through_the_pipeline():
    trail = D.build_audit_trail({
        "current_financial_state": {
            "financial_as_of": "2026-06-30",
            "flows": {"revenue": {"value": 100.0, "period_start": "2025-07-01",
                                  "as_of_date": "2026-06-30", "source": "ttm_calculation",
                                  "ttm": {"construction_method": "four_discrete_quarters",
                                          "validation_status": "valid"}}}},
        "canonical_evidence": {"current": {
            "operating_margin": {"value": 0.15, "period": "TTM",
                                 "derivation_formula": "operating_income / revenue",
                                 "source_metrics": ["operating_income", "revenue"]}},
            "base_period": "2026-06-30", "base_period_aligned": True},
        "business_model": {"profile": P.INSURER, "sic": "6324"},
        "valuation_method_status": MP.ValuationMethodStatus.VALID_BUT_NOT_APPLICABLE,
    })
    assert trail["1_normalized_facts"]["flows"]["revenue"]["construction"] == \
        "four_discrete_quarters"
    assert trail["2_canonical_current"]["operating_margin"]["derivation"] == \
        "operating_income / revenue"
    assert trail["5_business_model"]["profile"] == P.INSURER
    assert trail["7_valuation"]["method_status"] == \
        MP.ValuationMethodStatus.VALID_BUT_NOT_APPLICABLE


def test_the_audit_trail_records_what_it_suppressed():
    trail = D.build_audit_trail({
        "semantic_rejections": [{"code": "GUIDANCE_PERIOD_INCOMPATIBLE"},
                                {"code": "DCF_MODEL_BOUND_CONFLICT"}]})
    assert "9_suppressed_symptoms" in trail
    assert "DCF_MODEL_BOUND_CONFLICT" in trail["9_suppressed_symptoms"][0]


def test_the_audit_trail_is_off_by_default():
    import tools.config as config

    assert config.finance_audit_trail_enabled() is False


def test_the_audit_trail_survives_an_empty_analysis():
    """A diagnostic that raises on thin input is useless exactly when it is
    needed."""
    assert D.build_audit_trail({}) is not None
    assert D.build_audit_trail(None) is not None


# ---------------------------------------------------------------------------
# Phase H.14 sections 19-21 / 43 — TTM alignment names the exact metric
# ---------------------------------------------------------------------------

class _Sel:
    def __init__(self, value, start, end, method="four_discrete_quarters"):
        self.value = value
        self.period_start = start
        self.as_of_date = end
        self.source = "ttm_calculation"
        self.ttm = {"construction_method": method, "validation_status": "valid"}
        self.freshness_status = "current"
        self.provider = "sec"


class _State:
    def __init__(self, flows):
        self.flows = flows
        self.balance_sheet = {}
        self.total_debt = None
        self.net_debt = None
        self.net_debt_detail = {"net_debt_policy": "cash_only"}
        self.financial_as_of = "2026-06-27"


def _flows(fcf_end="2026-06-27"):
    base = ("2025-06-29", "2026-06-27")
    return {
        "revenue": _Sel(100, *base),
        "operating_income": _Sel(15, *base),
        "net_income": _Sel(10, *base),
        "operating_cash_flow": _Sel(20, *base),
        "free_cash_flow": _Sel(18, "2025-03-29", fcf_end),
        "depreciation_and_amortization": _Sel(5, *base),
    }


def test_a_shared_base_period_is_aligned():
    from finance import canonical as C

    evidence = C.build_canonical_evidence(_State(_flows()))
    assert evidence.ttm_alignment_status == C.TtmAlignment.ALIGNED
    assert evidence.misaligned_metrics == []


def test_a_lagging_metric_is_named_with_its_date_and_construction():
    """Section 20: 'not all metrics cover the same period' names nothing.

    The scenario is section 43's: revenue and net income end Q2, free cash
    flow ends Q1.
    """
    from finance import canonical as C

    evidence = C.build_canonical_evidence(_State(_flows(fcf_end="2026-03-28")))
    assert evidence.ttm_alignment_status == C.TtmAlignment.MISALIGNED
    assert evidence.expected_end_date == "2026-06-27"

    affected = evidence.misaligned_metrics
    assert [m["metric_id"] for m in affected] == ["free_cash_flow"]
    assert affected[0]["actual_end_date"] == "2026-03-28"
    assert affected[0]["construction_method"] == "four_discrete_quarters"
    assert affected[0]["affects_valuation"] is True


def test_a_valuation_metric_lag_raises_a_period_conflict():
    """Section 21: the DCF cannot claim one clean trailing base."""
    from finance import canonical as C

    evidence = C.build_canonical_evidence(_State(_flows(fcf_end="2026-03-28")))
    codes = {f["code"] for f in evidence.findings}
    assert C.DCF_CANONICAL_PERIOD_CONFLICT in codes


def test_the_conflict_finding_carries_the_affected_metric():
    from finance import canonical as C

    evidence = C.build_canonical_evidence(_State(_flows(fcf_end="2026-03-28")))
    conflict = next(f for f in evidence.findings
                    if f["code"] == C.DCF_CANONICAL_PERIOD_CONFLICT)
    assert conflict["affected_metrics"] == ["free_cash_flow"]


# ---------------------------------------------------------------------------
# Phase H.14 sections 25/27/39 — an empty claim cannot satisfy a schema
# ---------------------------------------------------------------------------

def test_blank_bullets_cannot_satisfy_a_minimum_count():
    """The gap: `min_items` was checked BEFORE empties were filtered, so a
    three-bullet list with two blanks passed a min_items=3 schema and then
    rendered one bullet."""
    from finance.research_pipeline import _string_list

    with pytest.raises(Exception) as excinfo:
        _string_list({"k": ["A perfectly good claim here", "", "   "]}, "k", min_items=3)
    assert "actual content" in str(excinfo.value)


def test_placeholder_bullets_are_not_content():
    from finance.research_pipeline import _string_list

    with pytest.raises(Exception):
        _string_list({"k": ["A perfectly good claim here", "-", "N/A"]}, "k", min_items=3)


def test_real_bullets_still_pass():
    from finance.research_pipeline import _string_list

    kept = _string_list(
        {"k": ["Revenue grew materially this year", "Margins expanded on mix"]},
        "k", min_items=2)
    assert len(kept) == 2


def test_an_empty_claim_string_is_rejected():
    """Section 39: {"severity": "HIGH", "claim": ""} must not validate."""
    from finance.research_pipeline import _str_field

    with pytest.raises(Exception):
        _str_field({"claim": ""}, "claim")
    with pytest.raises(Exception):
        _str_field({"claim": "   "}, "claim")


# ---------------------------------------------------------------------------
# Phase H.14 sections 23-24 / 42 — compound financing events
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("form, items, description, expected", [
    ("8-K", "2.03",
     "Issued senior notes; net proceeds will be used to fund the accelerated "
     "share repurchase",
     "DEBT_FINANCED_REPURCHASE"),
    ("8-K", "2.03", "Issued senior notes for general corporate purposes",
     "ISSUER_DEBT_ISSUANCE"),
    ("8-K", "8.01", "The Board authorized an accelerated share repurchase of $2 billion",
     "ACCELERATED_SHARE_REPURCHASE"),
    ("8-K", "2.03", "Issued convertible senior notes", "CONVERTIBLE_ISSUANCE"),
])
def test_corporate_action_classification(form, items, description, expected):
    from finance.structural_breaks import classify_security_event

    event_type, _confidence = classify_security_event(form, items, description)
    assert event_type == expected


def test_a_debt_financed_buyback_preserves_both_sides():
    """Section 42: debt up AND shares down, with no dilution claim.

    Classifying it as either one alone loses half the economics -- a report
    seeing only the debt calls it deterioration, one seeing only the buyback
    calls it a return of capital.
    """
    from finance.structural_breaks import event_impact

    impact = event_impact("DEBT_FINANCED_REPURCHASE")
    assert impact["affects_debt"] is True
    assert impact["affects_share_count"] is True
    assert impact["potential_dilution"] is False
    assert impact["increases_financial_leverage"] is True


def test_an_accelerated_repurchase_is_not_dilutive():
    from finance.structural_breaks import event_impact

    impact = event_impact("ACCELERATED_SHARE_REPURCHASE")
    assert impact["affects_share_count"] is True
    assert impact["potential_dilution"] is False
    assert impact["increases_financial_leverage"] is False


def test_a_debt_filing_mentioning_a_buyback_is_not_automatically_compound():
    """The funding language is required, not merely co-occurrence."""
    from finance.structural_breaks import classify_security_event

    event_type, _ = classify_security_event(
        "8-K", "2.03",
        "Issued senior notes. The company also maintains a share repurchase program.")
    assert event_type == "ISSUER_DEBT_ISSUANCE"


# ---------------------------------------------------------------------------
# Phase H.14 sections 30-31 — a condition cannot promise to fix a phantom
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("condition, codes, is_phantom", [
    ("The share-count conflict is resolved", [], True),
    ("The share-count conflict is resolved", ["SHARE_COUNT_CONFLICT"], False),
    ("The share count discrepancy is reconciled",
     ["MARKET_CAP_RECONCILIATION_FAILURE"], False),
    ("The net debt discrepancy is resolved", [], True),
    ("The TTM period mismatch is corrected", [], True),
    ("The TTM period mismatch is corrected", ["TTM_BASE_PERIOD_MISMATCH"], False),
    # A condition naming nothing diagnostic is left alone -- this checks
    # references, it does not police vocabulary.
    ("Revenue growth accelerates above 15%", [], False),
    ("Operating margin expands beyond 20%", [], False),
])
def test_condition_issue_references(condition, codes, is_phantom):
    from finance.research_pipeline import condition_references_nonexistent_issue

    assert condition_references_nonexistent_issue(condition, codes) is is_phantom


def test_a_phantom_condition_is_dropped_from_its_bucket():
    from finance.research_pipeline import drop_conditions_referencing_absent_issues

    routed = drop_conditions_referencing_absent_issues(
        {"reassessment_triggers": ["The share-count conflict is resolved",
                                   "Full-year guidance is issued"]},
        issue_codes=[])
    assert routed["reassessment_triggers"] == ["Full-year guidance is issued"]


def test_the_last_condition_in_a_bucket_survives_a_quality_check():
    """An empty list tells a reader less than one imperfect entry.

    The rule holds for the checks it was written for -- a threshold already
    met, a bound that exists only inside this software -- because there the
    condition is merely weak. It does NOT hold for a condition that
    REFERENCES an issue this analysis never reported: see
    `test_an_unverifiable_reference_is_dropped_even_when_it_is_the_last_one`.
    A weak condition costs a reader a little attention; one promising to
    resolve a problem that does not exist sends them looking for it.
    """
    from finance.research_pipeline import validate_conditions_against_current_state

    kept = validate_conditions_against_current_state(
        {"reassessment_triggers": ["Operating margin sustains above 15% for two quarters"]},
        current_metrics={})
    assert len(kept["reassessment_triggers"]) == 1


def test_an_unverifiable_reference_is_dropped_even_when_it_is_the_last_one():
    """The narrowed case, stated where the old rule used to be.

    A live report carried "resolve the debt discrepancy" as its only
    reassessment trigger for an analysis whose debt had reconciled cleanly
    upstream. There is no reading of that condition a reader can act on.
    """
    from finance.research_pipeline import drop_conditions_referencing_absent_issues

    routed = drop_conditions_referencing_absent_issues(
        {"reassessment_triggers": ["The share-count conflict is resolved"]},
        issue_codes=[])
    assert routed["reassessment_triggers"] == []

    kept = drop_conditions_referencing_absent_issues(
        {"reassessment_triggers": ["The share-count conflict is resolved"]},
        issue_codes=["SHARE_COUNT_CONFLICT"])
    assert len(kept["reassessment_triggers"]) == 1
