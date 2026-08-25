"""Phase H.12 — business-model semantics, propagated past the DCF gate.

The previous phase decided correctly that an insurer's operating cash flow
less capital expenditure is not owner free cash flow, and declined the
standard valuation model on that basis. It then let the same number through
to everyone downstream. A live insurer produced the contradiction inside one
report: a Valuation section stating the figure is NOT owner free cash flow,
and a Bull Case two sections later citing "free cash flow of 23.6 billion"
as evidence of cash generation.

These tests assert the general rule -- a metric rejected for one use cannot
support that use anywhere -- rather than any issuer's numbers. The business
model is always supplied as a classification, never as a ticker.
"""

import pytest

from finance import metric_policy as MP
from finance.business_model import CashFlowValuationProfile as P


Use = MP.MetricUse
V = MP.ValuationMethodStatus


# ---------------------------------------------------------------------------
# Section 39 — the suitability matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile, metric, use, allowed", [
    # A standard operating company keeps every existing interpretation.
    (P.STANDARD_OPERATING_COMPANY, "free_cash_flow", Use.OWNER_CASH_CLAIM, True),
    (P.STANDARD_OPERATING_COMPANY, "free_cash_flow", Use.VALUATION_INPUT, True),
    (P.STANDARD_OPERATING_COMPANY, "current_ratio", Use.LIQUIDITY_CLAIM, True),
    # Specialized models reject the owner-cash reading of the same figure.
    (P.INSURER, "free_cash_flow", Use.OWNER_CASH_CLAIM, False),
    (P.BROKER_DEALER, "free_cash_flow", Use.OWNER_CASH_CLAIM, False),
    (P.BANK, "free_cash_flow", Use.OWNER_CASH_CLAIM, False),
    (P.FINANCIAL_INSTITUTION, "free_cash_flow", Use.VALUATION_INPUT, False),
    (P.INSURER, "net_debt_to_fcf", Use.LEVERAGE_CLAIM, False),
    # ...and the liquidity reading of a working-capital ratio.
    (P.INSURER, "current_ratio", Use.LIQUIDITY_CLAIM, False),
    (P.BANK, "quick_ratio", Use.LIQUIDITY_CLAIM, False),
    # Reporting the number is never restricted (section 8).
    (P.INSURER, "free_cash_flow", Use.FACTUAL_DISPLAY, True),
    (P.INSURER, "current_ratio", Use.FACTUAL_DISPLAY, True),
    # Metrics the model does not restrict behave normally.
    (P.INSURER, "operating_margin", Use.PROFITABILITY_CLAIM, True),
    (P.INSURER, "revenue_growth", Use.GROWTH_CLAIM, True),
    # An unknown model must not silently narrow every claim.
    (P.UNKNOWN, "free_cash_flow", Use.OWNER_CASH_CLAIM, True),
])
def test_metric_suitability_matrix(profile, metric, use, allowed):
    assert MP.evaluate(metric, use, profile).allowed is allowed


def test_a_rejected_metric_says_why_in_economic_terms():
    verdict = MP.evaluate("free_cash_flow", Use.OWNER_CASH_CLAIM, P.INSURER)
    assert verdict.suitability == MP.Suitability.NOT_ECONOMICALLY_COMPARABLE
    assert "premiums are collected" in verdict.reason
    assert "arithmetically correct" in verdict.reason


def test_a_low_information_ratio_is_not_called_incomparable():
    """The two verdicts mean different things and must not be collapsed."""
    verdict = MP.evaluate("current_ratio", Use.LIQUIDITY_CLAIM, P.INSURER)
    assert verdict.suitability == MP.Suitability.LOW_INFORMATION_VALUE
    assert "not by itself evidence of liquidity stress" in verdict.reason


# ---------------------------------------------------------------------------
# Sections 6-9, 40 — the claim validator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, profile, expected", [
    # The live Bull Case sentence.
    ("Positive free cash flow demonstrates strong cash generation.",
     P.INSURER, MP.CASH_FLOW_SEMANTIC_MISUSE),
    ("Free cash flow of 23.6 billion supports the valuation.",
     P.INSURER, MP.CASH_FLOW_SEMANTIC_MISUSE),
    ("Strong free cash flow underpins the thesis.",
     P.BROKER_DEALER, MP.CASH_FLOW_SEMANTIC_MISUSE),
    ("Net debt to free cash flow indicates manageable leverage.",
     P.INSURER, MP.CASH_FLOW_SEMANTIC_MISUSE),
    # The same sentences are fine for an ordinary operating company.
    ("Positive free cash flow demonstrates strong cash generation.",
     P.STANDARD_OPERATING_COMPANY, None),
    ("Net debt to free cash flow indicates manageable leverage.",
     P.STANDARD_OPERATING_COMPANY, None),
    # Liquidity conclusions from a working-capital ratio.
    ("Current ratio below 1.0 indicates liquidity stress.",
     P.INSURER, MP.GENERIC_RATIO_INTERPRETATION_NOT_SUPPORTED),
    ("A current ratio of 0.78 signals short-term liquidity risk.",
     P.INSURER, MP.GENERIC_RATIO_INTERPRETATION_NOT_SUPPORTED),
    ("Current ratio below 1.0 indicates liquidity stress.",
     P.STANDARD_OPERATING_COMPANY, None),
    # Stating the figure remains legal for every model (section 8/30).
    ("Operating cash flow was 27.0 billion in the trailing twelve months.",
     P.INSURER, None),
    ("Current ratio is 0.78, below 1.0.", P.INSURER, None),
])
def test_claim_validation(text, profile, expected):
    codes = [f["code"] for f in MP.validate_claim(text, profile)]
    if expected is None:
        assert codes == []
    else:
        assert expected in codes


def test_a_not_applicable_model_may_not_be_called_invalid():
    codes = [f["code"] for f in MP.validate_claim(
        "The valuation model is invalid.", P.INSURER, V.VALID_BUT_NOT_APPLICABLE)]
    assert "VALUATION_APPLICABILITY_MISSTATED" in codes


def test_an_actually_invalid_model_may_be_called_invalid():
    """The guard applies to a misdescription, not to the word itself."""
    codes = [f["code"] for f in MP.validate_claim(
        "The valuation model is invalid.", P.STANDARD_OPERATING_COMPANY, V.INVALID)]
    assert "VALUATION_APPLICABILITY_MISSTATED" not in codes


def test_a_missing_valuation_is_not_a_company_risk():
    codes = [f["code"] for f in MP.validate_claim(
        "No DCF available means company risk is high.", P.INSURER)]
    assert "VALUATION_LIMITATION_AS_COMPANY_RISK" in codes


# ---------------------------------------------------------------------------
# Section 15-16 — valid is not the same as applicable
# ---------------------------------------------------------------------------

class _Classification:
    def __init__(self, profile, fcff):
        self.profile = profile
        self.standard_fcff_suitability = fcff
        self.corroborating_concepts = ()


@pytest.mark.parametrize("profile, fcff, available, failed, expected", [
    (P.INSURER, "NOT_SUITABLE", False, False, V.VALID_BUT_NOT_APPLICABLE),
    (P.BROKER_DEALER, "NOT_SUITABLE", False, False, V.VALID_BUT_NOT_APPLICABLE),
    (P.REIT_OR_SPECIALIZED, "LIMITED", False, False, V.LIMITED),
    (P.STANDARD_OPERATING_COMPANY, "SUITABLE", True, False, V.VALID_AND_APPLICABLE),
    (P.STANDARD_OPERATING_COMPANY, "SUITABLE", True, True, V.INVALID),
    (P.STANDARD_OPERATING_COMPANY, "SUITABLE", False, False, V.LIMITED),
])
def test_valuation_method_status(profile, fcff, available, failed, expected):
    assert MP.valuation_method_status(
        _Classification(profile, fcff), available, failed) == expected


def test_not_applicable_is_worded_as_not_applicable():
    wording = MP.VALUATION_STATUS_WORDING[V.VALID_BUT_NOT_APPLICABLE]
    assert "not applicable" in wording
    assert "invalid" not in wording.lower()


def test_an_applicable_model_produces_no_caveat_line():
    assert MP.VALUATION_STATUS_WORDING[V.VALID_AND_APPLICABLE] is None


# ---------------------------------------------------------------------------
# Sections 13-14 — guidance relevance follows the business model
# ---------------------------------------------------------------------------

def test_guidance_relevance_differs_by_business_model():
    insurer = MP.relevant_guidance_rows(P.INSURER)
    standard = MP.relevant_guidance_rows(P.STANDARD_OPERATING_COMPANY)
    assert "eps" in insurer
    assert "capex" in standard
    # An insurer is not scored against a capital-expenditure checklist.
    assert "capex" not in insurer
    assert insurer != standard


def test_an_unknown_model_uses_the_standard_checklist():
    assert MP.relevant_guidance_rows(P.UNKNOWN) == \
        MP.relevant_guidance_rows(P.STANDARD_OPERATING_COMPANY)


def test_relevant_guidance_found_and_missing_are_both_reported():
    packet = MP.build_relevant_evidence(
        _Classification(P.INSURER, "NOT_SUITABLE"),
        guidance_matrix={"rows": {"eps": "CURRENT", "tax_rate": "CURRENT",
                                  "revenue": "UNAVAILABLE"}})
    assert "eps" in packet.relevant_guidance_found
    assert "tax_rate" in packet.relevant_guidance_found
    assert "revenue" in packet.relevant_guidance_missing


# ---------------------------------------------------------------------------
# Section 29 — the packet
# ---------------------------------------------------------------------------

def test_the_packet_sorts_metrics_by_what_they_are_worth_here():
    packet = MP.build_relevant_evidence(
        _Classification(P.INSURER, "NOT_SUITABLE"),
        canonical_current={"revenue": {}, "operating_margin": {},
                           "free_cash_flow": {}, "current_ratio": {}})
    assert "revenue" in packet.primary_metrics
    assert "operating_margin" in packet.primary_metrics
    assert "free_cash_flow" in packet.low_information_metrics
    assert "current_ratio" in packet.low_information_metrics


def test_a_standard_company_packet_restricts_nothing():
    packet = MP.build_relevant_evidence(
        _Classification(P.STANDARD_OPERATING_COMPANY, "SUITABLE"),
        canonical_current={"revenue": {}, "free_cash_flow": {}, "current_ratio": {}})
    assert packet.low_information_metrics == []
    assert packet.prohibited_interpretations == []
    assert packet.cash_flow_label == "FCF"


def test_the_packet_carries_the_restrictions_as_data():
    """Section 33: enforceable, not a sentence in a prompt."""
    packet = MP.build_relevant_evidence(_Classification(P.INSURER, "NOT_SUITABLE"))
    uses = {(p["metric_id"], p["use"]) for p in packet.prohibited_interpretations}
    assert ("free_cash_flow", Use.OWNER_CASH_CLAIM) in uses
    assert ("current_ratio", Use.LIQUIDITY_CLAIM) in uses
    assert all(p["reason"] for p in packet.prohibited_interpretations)


def test_the_cash_flow_label_changes_only_for_specialized_models():
    assert MP.cash_flow_label(P.INSURER) == "Cash flow after CapEx"
    assert MP.cash_flow_label(P.BANK) == "Cash flow after CapEx"
    assert MP.cash_flow_label(P.STANDARD_OPERATING_COMPANY) == "FCF"
    assert MP.cash_flow_label(P.UNKNOWN) == "FCF"


# ---------------------------------------------------------------------------
# Sections 4/29/31 — the packet reaches the evidence index
# ---------------------------------------------------------------------------

def _index_for(profile, fcff):
    from finance.evidence import build_evidence_index

    packet = MP.build_relevant_evidence(
        _Classification(profile, fcff),
        canonical_current={"free_cash_flow": {}, "current_ratio": {}},
        valuation_method_status=MP.valuation_method_status(
            _Classification(profile, fcff), False, False))
    return build_evidence_index({
        "symbol": "ZZ",
        "business_model_evidence": packet.to_dict(),
        "canonical_evidence": {"current": {
            "free_cash_flow": {"value": 23.6e9, "period": "TTM", "period_type": "TTM"},
            "current_ratio": {"value": 0.78, "period": "2026-06-30",
                              "period_type": "DERIVED"}}},
    })


def test_a_research_role_sees_the_restriction_on_the_metric_itself():
    index = _index_for(P.INSURER, "NOT_SUITABLE")
    item = index["current.free_cash_flow"]
    assert "cash flow after capex" in item.label.lower()
    assert "may NOT be used to claim cash available to the owners" in item.derivation


def test_the_prohibitions_are_citable_evidence():
    index = _index_for(P.INSURER, "NOT_SUITABLE")
    prohibited = [k for k in index if k.startswith("business_model.prohibited.")]
    assert prohibited
    assert index["business_model.classification"].value == P.INSURER
    assert index["business_model.valuation_method_status"].value == \
        V.VALID_BUT_NOT_APPLICABLE


def test_an_ordinary_company_index_is_unchanged():
    index = _index_for(P.STANDARD_OPERATING_COMPANY, "SUITABLE")
    assert index["current.free_cash_flow"].label.startswith("Current free cash flow")
    assert not [k for k in index if k.startswith("business_model.prohibited.")]


# ---------------------------------------------------------------------------
# Sections 21-23 — COMPLETE must mean the required stages completed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cascade, expected", [
    ({"bull_researcher": "COMPLETE", "bear_researcher": "COMPLETE",
      "rebuttal_round": "COMPLETE", "research_manager": "COMPLETE",
      "risk_reviewer": "COMPLETE", "final_investment_synthesizer": "COMPLETE"},
     "COMPLETE"),
    # The live contradiction: a report saying COMPLETE on the same page as
    # "rebuttal failed after 3 attempts".
    ({"bull_researcher": "COMPLETE", "bear_researcher": "COMPLETE",
      "rebuttal_round": "FAILED", "research_manager": "COMPLETE",
      "risk_reviewer": "COMPLETE", "final_investment_synthesizer": "COMPLETE"},
     "DEGRADED"),
    ({"bull_researcher": "COMPLETE", "bear_researcher": "FAILED",
      "rebuttal_round": "SKIPPED", "research_manager": "SKIPPED",
      "risk_reviewer": "COMPLETE", "final_investment_synthesizer": "FAILED"},
     "PARTIAL"),
    ({"bull_researcher": "NOT_RUN", "bear_researcher": "NOT_RUN",
      "rebuttal_round": "NOT_RUN", "research_manager": "NOT_RUN",
      "risk_reviewer": "NOT_RUN", "final_investment_synthesizer": "NOT_RUN"},
     "DISABLED"),
])
def test_pipeline_overall_status(cascade, expected):
    from finance.workflow import _pipeline_overall_status

    assert _pipeline_overall_status(cascade) == expected


def test_a_degraded_pipeline_is_not_reported_as_complete():
    """Section 23: readiness and the report must both know."""
    from finance.workflow import _pipeline_overall_status

    cascade = {"bull_researcher": "COMPLETE", "bear_researcher": "COMPLETE",
               "rebuttal_round": "FAILED", "research_manager": "COMPLETE",
               "risk_reviewer": "COMPLETE", "final_investment_synthesizer": "COMPLETE"}
    assert _pipeline_overall_status(cascade) != "COMPLETE"


# ---------------------------------------------------------------------------
# Section 33 — the claim is blocked, the stage survives
# ---------------------------------------------------------------------------

def _insurer_index():
    from finance.evidence import build_evidence_index

    return build_evidence_index({
        "symbol": "ZZ",
        "valuation_method_status": V.VALID_BUT_NOT_APPLICABLE,
        "business_model_evidence": {"business_model": P.INSURER},
    })


def test_a_business_model_claim_is_stubbed_not_dropped():
    """The claim must not reach the report, and the stage must not die.

    Both failure modes were live. Raising cost a real insurer its entire
    final synthesis -- no recommendation at all. Dropping is impossible
    here: `claims` carries min_items=2 and its quarantine policy is FAIL
    precisely because removing an element breaches that. Stubbing replaces
    the sentence and leaves the element, so neither happens.
    """
    from finance.research_pipeline import apply_quarantine, _business_model_claim_findings

    output = {"claims": [
        {"claim": "Positive free cash flow demonstrates strong cash generation.",
         "evidence_ids": ["a"]},
        {"claim": "Revenue grew 6.5% over the trailing twelve months.",
         "evidence_ids": ["b"]}]}
    findings = _business_model_claim_findings(output, _insurer_index())
    result, records, fatal = apply_quarantine(output, findings)

    assert fatal == []
    assert len(result["claims"]) == 2
    assert "withheld" in result["claims"][0]["claim"]
    assert "does not support for this business model" in result["claims"][0]["claim"]
    # The legitimate claim beside it is untouched.
    assert result["claims"][1]["claim"] == "Revenue grew 6.5% over the trailing twelve months."
    assert [r["policy"] for r in records] == ["stub"]


def test_a_fail_policy_field_still_survives_a_semantic_finding():
    """`key_risks` is FAIL-policy for the same min_items reason."""
    from finance.research_pipeline import apply_quarantine, _business_model_claim_findings

    output = {"key_risks": [{"risk": "Current ratio below 1.0 indicates liquidity stress."}]}
    findings = _business_model_claim_findings(output, _insurer_index())
    result, _records, fatal = apply_quarantine(output, findings)

    assert fatal == []
    assert len(result["key_risks"]) == 1
    assert "withheld" in result["key_risks"][0]["risk"]


def test_findings_are_path_aware():
    from finance.research_pipeline import _business_model_claim_findings

    output = {"claims": [
        {"claim": "Revenue grew.", "evidence_ids": ["a"]},
        {"claim": "Strong free cash flow underpins the thesis.", "evidence_ids": ["b"]}]}
    findings = _business_model_claim_findings(output, _insurer_index())
    assert [f.field_path for f in findings] == ["claims[1].claim"]
    assert findings[0].rule_id == "BM-001"


def test_an_ordinary_company_produces_no_semantic_findings():
    from finance.evidence import build_evidence_index
    from finance.research_pipeline import _business_model_claim_findings

    index = build_evidence_index({
        "symbol": "ZZ",
        "valuation_method_status": V.VALID_AND_APPLICABLE,
        "business_model_evidence": {"business_model": P.STANDARD_OPERATING_COMPANY},
    })
    output = {"claims": [
        {"claim": "Positive free cash flow demonstrates strong cash generation.",
         "evidence_ids": ["a"]}]}
    assert _business_model_claim_findings(output, index) == []


def test_semantic_misuse_is_never_fatal_even_beside_an_overstatement():
    """The new severity must not change how the existing ones behave."""
    from finance.content_policy import Finding, Severity
    from finance.research_pipeline import apply_quarantine

    output = {"thesis": "Strong free cash flow.", "claims": [
        {"claim": "Free cash flow demonstrates strong cash generation.",
         "evidence_ids": ["a"]},
        {"claim": "Revenue grew.", "evidence_ids": ["b"]}]}
    findings = [
        Finding("BM-001", "CASH_FLOW_SEMANTIC_MISUSE", Severity.SEMANTIC_MISUSE,
                "claims[0].claim", "free_cash_flow"),
        Finding("CV-101", "superlative", Severity.OVERSTATEMENT, "thesis", "strong"),
    ]
    result, records, fatal = apply_quarantine(output, findings)
    assert fatal == []
    assert "withheld" in result["claims"][0]["claim"]
    assert "withheld" in result["thesis"]
    assert len(records) == 2


def test_a_fabrication_finding_is_still_fatal():
    from finance.content_policy import Finding, Severity
    from finance.research_pipeline import apply_quarantine

    findings = [Finding("CP-001", "position size", Severity.FABRICATION,
                        "rationale[0].statement", "position size")]
    _result, _records, fatal = apply_quarantine({"rationale": [{"statement": "x"}]}, findings)
    assert len(fatal) == 1
