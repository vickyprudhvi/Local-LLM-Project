"""Phase H.32 -- a model failure is not a company risk.

Live-traced gap: the DCF correctly produced DCF_NEGATIVE_TERMINAL_FCFF and
withheld valuation evidence, but the Risk section then stated, in substance,
that negative terminal FCF implies unsustainable growth forecasts and future
cash-flow deficits -- restating a MODEL OUTPUT under the configured forecast
path as if it were a fact about the company. `classify_risk_category`'s
existing, one-directional (out of COMPANY_RISK only) reclassification
mechanism already catches "no DCF could be produced" framing; this phase
closes its gap for the DIFFERENT, indirect restatement of the same
underlying model limitation.
"""

from finance.evidence import build_evidence_index
from finance.research_pipeline import RiskCategory, classify_risk_category
from finance import research_pipeline as rp

INVALID_DCF_INDEX = build_evidence_index({
    "symbol": "ACME", "quote": {"price": 100.0, "price_basis": "delayed"},
    "fundamental_metrics": {
        "revenue_growth_yoy": {"value": 0.05, "formula": "..."},
        "free_cash_flow": {"value": -500_000_000.0, "formula": "..."},
    },
    "technical_metrics": {},
    "dcf": {"available": True, "validation_status": "DCF_NEGATIVE_TERMINAL_FCFF",
           "validation_reasons": ["DCF_NEGATIVE_TERMINAL_FCFF: negative terminal-year FCFF "
                                  "in scenario(s) base, bear."]},
    "warnings": [], "data_provenance": {}, "plan": {"omitted_datasets": []},
})


def _risk_item(text, evidence_id="quote.price", severity="medium", category=None):
    item = {"risk": text, "severity": severity, "evidence_cited": [evidence_id]}
    if category is not None:
        item["category"] = category
    return item


def _raw(*items):
    all_ids = sorted({eid for item in items for eid in item["evidence_cited"]})
    return {"key_risks": list(items), "data_quality_concerns": [],
           "evidence_cited": all_ids}


# ---------------------------------------------------------------------------
# A. DCF negative terminal FCFF only -> analysis limitation, no company
#    cash-flow-deficit risk.
# ---------------------------------------------------------------------------

def test_A_negative_terminal_fcff_alone_is_a_valuation_limitation_not_company_risk():
    raw = _raw(_risk_item(
        "Negative terminal FCF implies unsustainable growth forecasts and "
        "future cash-flow deficits.",
        evidence_id="dcf.validation_status", category="COMPANY_RISK"))
    validated = rp._validate_risk_reviewer_output(raw, INVALID_DCF_INDEX)
    assert len(validated["company_risks"]) == 0
    assert validated["key_risks"][0]["category"] == RiskCategory.VALUATION_METHOD_LIMITATION
    assert validated["company_risk_level"] is None


# ---------------------------------------------------------------------------
# B. Negative terminal FCFF + INDEPENDENTLY deteriorating reported FCF ->
#    the reported-deterioration risk may still be COMPANY_RISK, while the
#    model failure remains separately labeled.
# ---------------------------------------------------------------------------

def test_B_independent_reported_fcf_deterioration_can_still_be_company_risk():
    model_limitation = _risk_item(
        "The DCF produced a negative terminal FCFF under the base and bear "
        "scenarios, so valuation evidence is withheld for this analysis.",
        evidence_id="dcf.validation_status", category="COMPANY_RISK")
    independent_company_risk = _risk_item(
        "Reported free cash flow has been negative for the trailing twelve "
        "months and has declined in each of the last four reported "
        "quarters, indicating deteriorating cash generation.",
        evidence_id="fundamental.free_cash_flow",
        category="COMPANY_RISK")
    raw = _raw(model_limitation, independent_company_risk)
    validated = rp._validate_risk_reviewer_output(raw, INVALID_DCF_INDEX)

    categories = {r["risk"]: r["category"] for r in validated["key_risks"]}
    assert categories[model_limitation["risk"]] == RiskCategory.VALUATION_METHOD_LIMITATION
    assert categories[independent_company_risk["risk"]] == RiskCategory.COMPANY_RISK
    assert len(validated["company_risks"]) == 1
    assert validated["company_risks"][0]["risk"] == independent_company_risk["risk"]


# ---------------------------------------------------------------------------
# C. WACC default assumption -> analysis limitation, not company risk.
# ---------------------------------------------------------------------------

def test_C_default_wacc_assumption_is_a_valuation_limitation():
    raw = _raw(_risk_item(
        "The valuation relies on a default WACC assumption of 9.0% because "
        "no company-specific cost of capital could be derived from "
        "available evidence.",
        evidence_id="quote.price", category="COMPANY_RISK"))
    validated = rp._validate_risk_reviewer_output(raw, INVALID_DCF_INDEX)
    assert validated["key_risks"][0]["category"] == RiskCategory.VALUATION_METHOD_LIMITATION
    assert len(validated["company_risks"]) == 0


# ---------------------------------------------------------------------------
# D. High scenario spread -> model sensitivity/analysis limitation unless
#    independent risk evidence.
# ---------------------------------------------------------------------------

def test_D_wide_scenario_spread_is_a_valuation_limitation():
    raw = _raw(_risk_item(
        "The DCF scenarios show a wide valuation spread between the bear "
        "and bull cases, reflecting high sensitivity to growth assumptions "
        "rather than a specific business risk.",
        evidence_id="quote.price", category="COMPANY_RISK"))
    validated = rp._validate_risk_reviewer_output(raw, INVALID_DCF_INDEX)
    assert validated["key_risks"][0]["category"] == RiskCategory.VALUATION_METHOD_LIMITATION
    assert len(validated["company_risks"]) == 0


# ---------------------------------------------------------------------------
# E. Stale/missing valuation input -> limitation, not company risk.
# ---------------------------------------------------------------------------

def test_E_stale_valuation_input_is_a_data_quality_limitation():
    raw = _raw(_risk_item(
        "A key valuation input (segment-level operating margin) is stale, "
        "dated to a filing more than two years old.",
        evidence_id="quote.price", category="COMPANY_RISK"))
    validated = rp._validate_risk_reviewer_output(raw, INVALID_DCF_INDEX)
    assert validated["key_risks"][0]["category"] == RiskCategory.DATA_QUALITY_RISK
    assert len(validated["company_risks"]) == 0


# ---------------------------------------------------------------------------
# Direct marker-level checks (fast, no evidence-index plumbing) covering the
# same phrasing space, including the negative control that independent
# company-risk wording is left alone.
# ---------------------------------------------------------------------------

def test_terminal_value_failure_phrasing_is_never_company_risk():
    samples = [
        "Negative terminal FCF implies unsustainable growth forecasts and "
        "future cash-flow deficits.",
        "The DCF produced a negative terminal FCFF, implying the company "
        "faces future cash-flow deficits.",
        "The modeled forecast path does not support a valid perpetuity "
        "terminal value.",
    ]
    for text in samples:
        assert classify_risk_category(text) == RiskCategory.VALUATION_METHOD_LIMITATION


def test_independent_company_risk_wording_is_left_alone():
    text = ("The company faces future cash-flow deficits as reported free "
           "cash flow has declined for six consecutive quarters.")
    assert classify_risk_category(text) == RiskCategory.COMPANY_RISK
