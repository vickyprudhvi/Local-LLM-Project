"""Phase H.34 — the generalized, business-model-independent single-metric
interpretation rulebook (finance/claim_validation.py CV-701..CV-717, the
guidance metric-identity check, and the recommendation directional-evidence
coherence check in finance/research_pipeline.py).

Before this phase, finance/metric_policy.py banned an unsupported liquidity/
cash-flow conclusion ONLY for specialized business models (bank/broker-dealer/
insurer/financial_institution) — an ordinary company's "strong liquidity" from
a current ratio over 1, or "no debt-servicing risk" from a low debt-to-equity,
passed completely unchecked. This file uses fictional fixtures (never a real
ticker) to prove the universal version of each rule, following the exact test
shapes given in the phase spec's section 17/18 ("TSLA-style generalized
tests" and the additional matrix) without naming TSLA or any other real
company.

See tests/test_finance_claim_validation.py for the pre-existing scanner
tests this file extends, and tests/test_finance_research_pipeline.py for the
pre-existing FinalInvestmentSynthesizer validator tests this file extends.
"""

import pytest

from finance.claim_validation import (
    GUIDANCE_METRIC_MISMATCH,
    scan_for_guidance_metric_mismatch,
    scan_for_unsupported_claims,
)
from finance.evidence import build_evidence_index
import finance.research_pipeline as rp
from tests.test_finance_research_pipeline import INVALID_DCF_INDEX, _final_raw


# ---------------------------------------------------------------------------
# Section 17, test A — net cash, low leverage, healthy ratios: still no
# "may constrain capital allocation" claim
# ---------------------------------------------------------------------------

def test_a_net_cash_low_leverage_still_rejects_capital_allocation_constraint_claim():
    hits = scan_for_unsupported_claims(
        "Although net debt is negative and the debt-to-equity ratio is a modest 0.10, high debt "
        "may constrain capital allocation going forward.")
    assert any("constrain capital allocation" in h for h in hits), hits


def test_a_allows_the_net_cash_fact_itself():
    assert scan_for_unsupported_claims(
        "Net debt is negative, which is consistent with a net cash position.") == []


# ---------------------------------------------------------------------------
# Section 17, test B — growth sustainability without decomposition evidence
# ---------------------------------------------------------------------------

def test_b_unsustainable_growth_claim_is_rejected():
    hits = scan_for_unsupported_claims(
        "Revenue grew 12% year over year, but this growth is unsustainable.")
    assert any("growth-sustainability" in h for h in hits), hits


def test_b_sustainable_growth_claim_is_also_rejected():
    hits = scan_for_unsupported_claims("The 12% revenue growth looks sustainable.")
    assert any("growth-sustainability" in h for h in hits), hits


def test_b_the_permitted_alternative_is_not_flagged():
    assert scan_for_unsupported_claims(
        "Revenue grew 12% year over year; the available evidence does not establish the "
        "drivers of current revenue growth.") == []


# ---------------------------------------------------------------------------
# Section 17, test C — guidance metric identity
# ---------------------------------------------------------------------------

_CAPEX_ONLY_GUIDANCE = {
    "dcf.guidance.capex.fy2027.current": {
        "target_period": "FY2027", "fiscal_period": "FY2027",
        "target_period_type": "CURRENT_FISCAL_YEAR",
    },
}


def test_c_capex_guidance_cannot_support_a_revenue_claim():
    findings = scan_for_guidance_metric_mismatch(
        "Management guidance points to revenue uncertainty for the coming year.",
        _CAPEX_ONLY_GUIDANCE)
    assert findings, "capex-only guidance must not license a revenue claim"
    assert all(f.startswith(GUIDANCE_METRIC_MISMATCH) for f in findings)


def test_c_capex_guidance_does_support_a_capex_claim():
    assert scan_for_guidance_metric_mismatch(
        "Management guidance points to higher capital expenditures for the coming year.",
        _CAPEX_ONLY_GUIDANCE) == []


def test_c_no_guidance_at_all_rejects_any_attribution():
    findings = scan_for_guidance_metric_mismatch(
        "Management expects operating margin improvement next year.", {})
    assert findings
    assert "no metric" in findings[0]


def test_c_disclaiming_the_absence_of_guidance_is_not_flagged():
    assert scan_for_guidance_metric_mismatch(
        "No revenue guidance is available for the coming year.", _CAPEX_ONLY_GUIDANCE) == []


# ---------------------------------------------------------------------------
# Section 17, test D — DCF-invalid overreach
# ---------------------------------------------------------------------------

def test_d_negative_terminal_fcff_overreach_claim_is_rejected():
    hits = scan_for_unsupported_claims(
        "Given the negative terminal-year cash flow, the company will experience future "
        "cash-flow deficits.")
    assert any("DCF-invalid overreach" in h for h in hits), hits


def test_d_fcff_unsuitable_for_this_company_claim_is_rejected():
    hits = scan_for_unsupported_claims("FCFF is unsuitable for this company given the forecast.")
    assert any("DCF-invalid overreach" in h for h in hits), hits


def test_d_the_permitted_alternative_is_not_flagged():
    assert scan_for_unsupported_claims(
        "This forecast path does not support a valid perpetuity terminal value.") == []


# ---------------------------------------------------------------------------
# Section 17, test E — debt decrease under a downgrade condition is
# rerouted/rejected, never left in place (pre-existing deterministic
# behavior -- classify_condition_direction / route_conditions_by_direction --
# this test pins the EXACT spec example against it).
# ---------------------------------------------------------------------------

def test_e_a_debt_decrease_is_not_classified_as_a_downgrade_direction():
    direction = rp.classify_condition_direction("Total debt decreases materially.")
    assert direction != rp.CONDITION_UNFAVOURABLE


def test_e_a_debt_decrease_filed_under_downgrade_is_moved_out():
    validated = {
        "conditions_that_strengthen_the_view": [],
        "conditions_that_weaken_the_view": ["Total debt decreases materially."],
        "reassessment_triggers": [],
    }
    routed = rp.route_conditions_by_direction(validated)
    assert "Total debt decreases materially." not in routed["conditions_that_weaken_the_view"]
    assert "Total debt decreases materially." in routed["conditions_that_strengthen_the_view"]


# ---------------------------------------------------------------------------
# Section 17, test F — an empty/placeholder claim never reaches validated
# output (pre-existing _string_list/_claims_list fail-closed behavior).
# ---------------------------------------------------------------------------

def test_f_a_blank_bull_claim_fails_the_whole_stage_rather_than_padding():
    index = build_evidence_index({
        "symbol": "ZZZ", "quote": {"price": 10.0, "price_basis": "delayed"},
        "fundamental_metrics": {"revenue_growth_yoy": {"value": 0.05, "formula": "..."}},
        "technical_metrics": {}, "dcf": {"available": False, "reason": "DCF_ASSUMPTION_REQUIRED"},
        "warnings": [], "data_provenance": {}, "plan": {"omitted_datasets": []},
    })
    raw = {
        "thesis": "Revenue is growing at a modest pace.",
        "claims": [
            {"claim_id": "bull-1", "claim": "Revenue grew approximately 5% year over year.",
             "evidence_ids": ["fundamental.revenue_growth_yoy"], "claim_type": "fact_interpretation",
             "assumptions": [], "confidence": 0.6},
            {"claim_id": "bull-2", "claim": "N/A", "evidence_ids": ["fundamental.revenue_growth_yoy"],
             "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.5},
        ],
        "confidence": "medium",
    }
    with pytest.raises(rp._Invalid):
        rp._validate_researcher_output(raw, index, "bull")


# ---------------------------------------------------------------------------
# Section 18 — additional matrix: single metrics never carry a compound claim
# ---------------------------------------------------------------------------

def test_positive_fcf_does_not_become_strong_without_a_threshold():
    assert scan_for_unsupported_claims("Free cash flow was positive at $5.8 billion.") == []
    hits = scan_for_unsupported_claims("Strong cash generation drove the quarter's results.")
    assert any("cash-generation-strength" in h for h in hits), hits


def test_current_ratio_above_one_does_not_become_strong_liquidity():
    assert scan_for_unsupported_claims(
        "The current ratio of 1.9 means current assets exceed current liabilities.") == []
    hits = scan_for_unsupported_claims(
        "A current ratio of 1.9 points to comfortable liquidity for the business.")
    assert any("liquidity-strength" in h for h in hits), hits


def test_low_debt_to_equity_does_not_prove_debt_service_safety():
    assert scan_for_unsupported_claims(
        "A debt-to-equity ratio of 0.1 reflects relatively low balance-sheet leverage.") == []
    hits = scan_for_unsupported_claims(
        "With a debt-to-equity ratio of 0.1, the company has no debt-servicing risk.")
    assert any("debt-servicing-safety" in h for h in hits), hits


def test_high_gross_debt_alone_does_not_establish_debt_stress():
    hits = scan_for_unsupported_claims(
        "Total debt exceeds trailing free cash flow by 2.4x, which signals debt stress.")
    assert any("leverage-distress" in h for h in hits), hits
    assert scan_for_unsupported_claims(
        "Total debt is approximately 2.4 times trailing free cash flow.") == []


def test_net_cash_position_still_blocks_an_unsupported_leverage_stress_claim():
    """The ban is on the CONCLUSION vocabulary, independent of which metric
    value is cited alongside it -- a net-cash position does not make 'debt
    servicing risk' a safer thing to assert about an unrelated gross-debt
    figure."""
    hits = scan_for_unsupported_claims(
        "Despite a net cash position, elevated gross debt still carries debt servicing risk.")
    assert any("leverage-distress" in h for h in hits), hits


def test_missing_guidance_is_not_itself_negative_company_evidence():
    """finance/metric_policy.py's pre-existing VALUATION_LIMITATION_AS_
    COMPANY_RISK-style guard; the regression this item covers is that
    'missing guidance' must never get classified as a COMPANY_RISK category."""
    assert rp.classify_risk_category(
        "No current revenue guidance is available, which limits forward visibility.") in (
        "ANALYSIS_LIMITATION", "DATA_QUALITY_RISK")


def test_duplicate_equivalent_claims_are_deduplicated():
    from finance.claim_validation import scan_structure_for_unsupported_claims
    found = scan_structure_for_unsupported_claims(
        ["This company has a fortress balance sheet.",
         "This company has a fortress balance sheet."])
    assert found.count("unsupported superlative (fortress)") == 1


# ---------------------------------------------------------------------------
# Section 16 — recommendation evidence coherence: missing valuation evidence
# alone is not bearish evidence.
# ---------------------------------------------------------------------------

def test_avoid_cited_only_to_valuation_unavailability_is_rejected():
    raw = _final_raw("inconclusive", evidence_id="valuation.status", recommendation="avoid")
    with pytest.raises(rp._Invalid) as excinfo:
        rp._validate_final_synthesizer_output(raw, INVALID_DCF_INDEX)
    assert rp.CONTENT_POLICY_VIOLATION_MARKER in excinfo.value.message


def test_avoid_cited_to_real_company_evidence_is_accepted():
    raw = _final_raw("inconclusive", evidence_id="fundamental.revenue_growth_yoy",
                     recommendation="avoid")
    validated = rp._validate_final_synthesizer_output(raw, INVALID_DCF_INDEX)
    assert validated["recommendation"] == "avoid"


def test_hold_is_unaffected_by_the_directional_evidence_check():
    """The check is scoped to SELL/AVOID only -- a HOLD cited entirely to
    valuation-limitation evidence is a pre-existing, legitimate shape
    (readiness is the thing being disclosed, not a bearish/bullish fact)."""
    raw = _final_raw("inconclusive", evidence_id="valuation.status", recommendation="hold")
    validated = rp._validate_final_synthesizer_output(raw, INVALID_DCF_INDEX)
    assert validated["recommendation"] == "hold"
