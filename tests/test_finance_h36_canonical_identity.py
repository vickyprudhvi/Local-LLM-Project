"""Phase H.36 — canonical financial identity + final claim-contract
enforcement.

Traced from a REAL live CF Industries (ticker CF) run through the real
deterministic pipeline (no LLM): CF's 2026-06-30 10-Q does not tag
`cash_and_cash_equivalents` under the mapped XBRL concept, which exposed a
silent `or 0.0` default that made `dcf.net_debt` bit-identical to gross
`total_debt`. This file covers the generalized fix (never ticker-specific)
plus the companion claim-contract gaps the same live run's phase spec named:
debt-service eligibility by evidence category, upgrade/downgrade direction
for UNCLASSIFIED conditions, "intrinsic model value" terminology, net-debt/
FCF-basis evidence identity, and ambiguous freshness wording.
"""

import pytest

import finance.research_pipeline as rp
from finance.claim_validation import scan_for_unsupported_claims
from finance.evidence import build_evidence_index


# ---------------------------------------------------------------------------
# Section 2/18 — canonical net-debt identity (the root CF fix)
# ---------------------------------------------------------------------------

def test_gross_debt_cannot_masquerade_as_net_debt_in_the_dcf_bridge():
    """finance/workflow.py::_dcf_inputs_from_facts and ::_packet_values:
    missing cash must make net_debt None, never a silent total_debt copy."""
    from finance.workflow import _packet_values

    dcf_inputs = {
        "total_debt": 3_216_000_000.0,
        "cash_and_cash_equivalents": None,  # not reported, the real CF shape
        "eligible_short_term_investments": 0.0,
        "base_revenue": 1.0,
    }
    values = _packet_values("CF", {}, dcf_inputs, [{"name": "base"}], {})
    assert values["net_debt"] is None
    assert values["total_debt"] == 3_216_000_000.0  # gross debt is still visible, just not renamed


def test_net_debt_computes_normally_when_cash_is_actually_reported():
    from finance.workflow import _packet_values

    dcf_inputs = {
        "total_debt": 3_216_000_000.0,
        "cash_and_cash_equivalents": 500_000_000.0,
        "eligible_short_term_investments": 0.0,
        "base_revenue": 1.0,
    }
    values = _packet_values("CF", {}, dcf_inputs, [{"name": "base"}], {})
    assert values["net_debt"] == pytest.approx(2_716_000_000.0)


def test_required_inputs_includes_net_debt_so_the_packet_refuses_on_it():
    from finance.dcf_packet import REQUIRED_INPUTS

    assert "net_debt" in REQUIRED_INPUTS
    assert "total_debt" in REQUIRED_INPUTS


# ---------------------------------------------------------------------------
# Section 3 — evidence identity must match the displayed metric
# ---------------------------------------------------------------------------

def test_net_debt_claim_citing_only_gross_debt_evidence_is_flagged():
    findings = rp._net_debt_identity_findings(
        {"claims": [{"claim": "Net debt is approximately $3.2 billion.",
                    "evidence_ids": ["fundamental.total_debt"]}]})
    assert len(findings) == 1
    assert findings[0].rule_id == rp.NET_DEBT_IDENTITY_MISMATCH_RULE_ID


def test_net_debt_claim_citing_net_debt_evidence_is_accepted():
    assert rp._net_debt_identity_findings(
        {"claims": [{"claim": "Net debt is approximately $3.2 billion.",
                    "evidence_ids": ["fundamental.net_debt"]}]}) == []


def test_net_debt_unavailable_disclaimer_needs_no_evidence_citation():
    """Section 2's own permitted wording ('net_debt = unavailable') has
    nothing to cite -- finance/evidence.py never indexes a null metric."""
    assert rp._net_debt_identity_findings(
        {"claims": [{"claim": "Net debt could not be established for the current period.",
                    "evidence_ids": ["fundamental.total_debt"]}]}) == []


def test_long_term_debt_mention_alone_is_not_flagged():
    """The check is scoped to claims that SAY 'net debt' -- a claim correctly
    describing long-term debt by its own name is untouched."""
    assert rp._net_debt_identity_findings(
        {"claims": [{"claim": "Long-term debt is approximately $3.2 billion.",
                    "evidence_ids": ["fundamental.total_debt"]}]}) == []


# ---------------------------------------------------------------------------
# Section 4/5 — FCF basis preservation
# ---------------------------------------------------------------------------

def test_fundamental_ratio_and_current_fcf_co_citation_is_flagged():
    findings = rp._fcf_basis_mismatch_findings(
        {"claims": [{"claim": "Debt relative to FCF is elevated at 1.78x, while trailing "
                             "twelve-month free cash flow is $1.91 billion.",
                    "evidence_ids": ["fundamental.debt_to_fcf", "current.free_cash_flow"]}]})
    assert len(findings) == 1
    assert findings[0].rule_id == rp.FCF_BASIS_MISMATCH_RULE_ID


def test_fundamental_ratio_cited_alone_is_not_flagged():
    assert rp._fcf_basis_mismatch_findings(
        {"claims": [{"claim": "Debt to FCF is 1.78x.", "evidence_ids": ["fundamental.debt_to_fcf"]}]}
    ) == []


def test_current_fcf_cited_alone_is_not_flagged():
    assert rp._fcf_basis_mismatch_findings(
        {"claims": [{"claim": "Trailing twelve-month free cash flow is $1.91 billion.",
                    "evidence_ids": ["current.free_cash_flow"]}]}) == []


# ---------------------------------------------------------------------------
# Section 6/7/8 — freshness wording must say which period family
# ---------------------------------------------------------------------------

def test_ambiguous_most_recent_complete_statements_is_rejected():
    hits = scan_for_unsupported_claims(
        "The most recent complete financial statements are from 2019-12-31.")
    assert any("ambiguous freshness claim" in h for h in hits)


def test_qualified_annual_statements_wording_is_accepted():
    assert scan_for_unsupported_claims(
        "The most recent complete annual financial statements are from 2025-12-31.") == []


def test_qualified_quarterly_statements_wording_is_accepted():
    assert scan_for_unsupported_claims(
        "The latest complete quarterly financial statements are from 2026-06-30.") == []


def test_a_genuinely_stale_current_statement_warning_still_survives():
    """The ban is on AMBIGUITY, not on reporting genuine staleness -- a
    qualified claim about actual staleness is untouched."""
    assert scan_for_unsupported_claims(
        "The latest reported financial statements are from 2024-12-31, which is stale "
        "relative to the current fiscal year.") == []


# ---------------------------------------------------------------------------
# Section 9/10 — debt-service risk eligible by evidence CATEGORY
# ---------------------------------------------------------------------------

_DEBT_SERVICE_INDEX = build_evidence_index({
    "symbol": "ZZZ", "quote": {"price": 10.0, "price_basis": "delayed"},
    "fundamental_metrics": {
        "debt_to_fcf": {"value": 2.0, "formula": "..."},
        "interest_coverage": {"value": 1.1, "formula": "..."},
    },
    "technical_metrics": {}, "dcf": {"available": False, "reason": "DCF_ASSUMPTION_REQUIRED"},
    "warnings": [], "data_provenance": {}, "plan": {"omitted_datasets": []},
})


def _debt_claims(evidence_ids):
    return {
        "thesis": "Leverage is a key watch item.",
        "claims": [
            {"claim_id": "bear-1",
             "claim": "Total debt exceeds trailing free cash flow, so debt servicing "
                     "pressure is elevated.",
             "evidence_ids": evidence_ids, "claim_type": "fact_interpretation",
             "assumptions": [], "confidence": 0.6},
            {"claim_id": "bear-2", "claim": "Operating margin is thin.",
             "evidence_ids": ["fundamental.debt_to_fcf"], "claim_type": "fact_interpretation",
             "assumptions": [], "confidence": 0.5},
        ],
        "confidence": "medium",
    }


def test_debt_exceeding_fcf_alone_does_not_license_debt_service_language():
    with pytest.raises(rp._Invalid):
        rp._validate_researcher_output(
            _debt_claims(["fundamental.debt_to_fcf"]), _DEBT_SERVICE_INDEX, "bear")


def test_debt_service_language_allowed_with_interest_coverage_evidence():
    validated = rp._validate_researcher_output(
        _debt_claims(["fundamental.debt_to_fcf", "fundamental.interest_coverage"]),
        _DEBT_SERVICE_INDEX, "bear")
    assert "debt servicing" in validated["claims"][0]["claim"].lower()


def test_leverage_fact_alone_is_always_allowed():
    """Section 10: a plain factual leverage statement is never gated --
    only the DISTRESS conclusion needs qualifying evidence."""
    assert scan_for_unsupported_claims("Total debt exceeds trailing free cash flow.") == []
    assert scan_for_unsupported_claims("Debt-to-equity is 0.56.") == []


# ---------------------------------------------------------------------------
# Section 12/13 — upgrade/downgrade/reassessment direction
# ---------------------------------------------------------------------------

def test_margin_expands_is_upgrade_eligible():
    assert rp.classify_condition_direction("Operating margin expands further.") == \
        rp.CONDITION_FAVOURABLE


def test_debt_falls_materially_is_upgrade_eligible():
    assert rp.classify_condition_direction("Total debt falls materially.") == \
        rp.CONDITION_FAVOURABLE


def test_cash_flow_deteriorates_is_downgrade_eligible():
    assert rp.classify_condition_direction("Free cash flow deteriorates.") == \
        rp.CONDITION_UNFAVOURABLE


def test_uncertain_growth_drivers_classifies_as_unclassified_not_favourable():
    """Root cause: `_INCREASE_WORDS` substring-matches "grow" inside the
    NOUN "growth" regardless of what the rest of the sentence says. Without
    the uncertainty override, 'Growth drivers remain uncertain' classified
    FAVOURABLE outright -- not merely UNCLASSIFIED-and-left-in-place, which
    is a stronger bug than the routing fix alone would catch."""
    assert rp.classify_condition_direction("Growth drivers remain uncertain.") == \
        rp.CONDITION_UNCLASSIFIED


def test_uncertain_growth_drivers_is_not_an_upgrade_condition():
    """The exact live finding: an UNCLASSIFIED condition must never stay
    under Upgrade merely because the model filed it there."""
    validated = {
        "conditions_that_strengthen_the_view": ["Growth drivers remain uncertain."],
        "conditions_that_weaken_the_view": [],
        "reassessment_triggers": [],
    }
    routed = rp.route_conditions_by_direction(validated)
    assert "Growth drivers remain uncertain." not in routed["conditions_that_strengthen_the_view"]
    assert "Growth drivers remain uncertain." in routed["reassessment_triggers"]


def test_uncertain_growth_drivers_under_downgrade_also_reroutes():
    validated = {
        "conditions_that_strengthen_the_view": [],
        "conditions_that_weaken_the_view": ["Growth drivers remain uncertain."],
        "reassessment_triggers": [],
    }
    routed = rp.route_conditions_by_direction(validated)
    assert "Growth drivers remain uncertain." not in routed["conditions_that_weaken_the_view"]
    assert "Growth drivers remain uncertain." in routed["reassessment_triggers"]


def test_updated_guidance_becomes_available_is_reassessment():
    assert rp.classify_condition_direction("Updated guidance becomes available.") == \
        rp.CONDITION_BIDIRECTIONAL


def test_blank_condition_never_reaches_a_bucket():
    """Conditions are OPTIONAL (unlike `claims`, which has a real minimum),
    so a blank entry is stripped to an empty list rather than failing the
    whole stage -- but it must never survive as a populated, blank bullet."""
    index = build_evidence_index({
        "symbol": "ZZZ", "quote": {"price": 10.0, "price_basis": "delayed"},
        "fundamental_metrics": {"revenue_growth_yoy": {"value": 0.05, "formula": "..."}},
        "technical_metrics": {}, "dcf": {"available": False, "reason": "DCF_ASSUMPTION_REQUIRED"},
        "warnings": [], "data_provenance": {}, "plan": {"omitted_datasets": []},
    })
    raw = {
        "research_stance": "neutral", "valuation_view": "approximately_fair",
        "overall_risk": "moderate", "confidence": 0.5, "recommendation": "hold",
        "primary_reason": "Evidence is mixed.", "supporting_factors": [],
        "limiting_factors": ["Scenario spread is wide."],
        "rationale": [{"statement": "A plain evidence-grounded observation.",
                      "evidence_ids": ["fundamental.revenue_growth_yoy"]}],
        "conditions_that_strengthen_the_view": ["   "],
        "conditions_that_weaken_the_view": [], "key_uncertainties": [],
    }
    validated = rp._validate_final_synthesizer_output(raw, index)
    assert validated["conditions_that_strengthen_the_view"] == []


# ---------------------------------------------------------------------------
# Section 14/15 — valuation terminology + negative modeled equity
# ---------------------------------------------------------------------------

def test_intrinsic_model_value_is_rejected():
    hits = scan_for_unsupported_claims("The intrinsic model value is $370 per share.")
    assert any("intrinsic value" in h for h in hits)


def test_modeled_value_is_accepted():
    assert scan_for_unsupported_claims("The base modeled value is $370 per share.") == []


def test_negative_modeled_equity_remains_a_valid_model_output():
    assert scan_for_unsupported_claims(
        "Under this modeled forecast path, the resulting equity value is negative.") == []


def test_negative_modeled_equity_called_a_calculation_error_is_still_rejected():
    hits = scan_for_unsupported_claims(
        "The negative modeled equity reflects a calculation error in the DCF.")
    assert any("calculation error" in h for h in hits)


# ---------------------------------------------------------------------------
# Section 20 — existing valid claims are retained (no over-blocking)
# ---------------------------------------------------------------------------

def test_valid_existing_claims_pass_through_every_new_check_unchanged():
    findings = (
        rp._net_debt_identity_findings({"claims": [{"claim": "Revenue grew 9.3% year over year.",
                                                    "evidence_ids": ["fundamental.revenue_growth_yoy"]}]})
        + rp._fcf_basis_mismatch_findings({"claims": [{"claim": "Operating margin is 20%.",
                                                       "evidence_ids": ["fundamental.operating_margin"]}]})
    )
    assert findings == []
    assert scan_for_unsupported_claims("Free cash flow was positive at $5.8 billion.") == []
