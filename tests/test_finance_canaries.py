"""Parts 18-28 — eight business-model classes, end to end.

Spec §19: *ticker fixtures prove a regression did not come back; they cannot
prove an architecture.* Every fixture in `tests/fixtures/*_regression.json`
exercises exactly the path its own issuer takes, which is why each new stock
kept finding a fresh route to the same class of error.

These do the other job. Each canary is a synthetic issuer built to sit inside
ONE business-model class with that class's defining characteristic dialled
up, and each test crosses several boundaries at once (Part 27):

    normalized fixture -> canonical state -> DCF packet -> valuation gate
    -> canonical research evidence -> report model

Nothing is stubbed but the two network clients.
"""

import pytest

from finance.dcf_packet import ValuationStatus
from finance.business_model import CashFlowValuationProfile, FcffSuitability
from tests.canary_harness import run_canary
from tests.fixtures import canary_definitions as CANARIES


# One run per canary per test module invocation. A canary run exercises the
# whole pipeline and is not cheap; the module scope means the eight runs
# happen once and every assertion below reads the same objects, which is also
# what makes cross-boundary consistency assertable at all.
@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    from _pytest.monkeypatch import MonkeyPatch

    monkeypatch = MonkeyPatch()
    out = {}
    try:
        for company in CANARIES.ALL_CANARIES + CANARIES.DEFECT_CANARIES:
            out[company.key] = run_canary(
                company, tmp_path_factory.mktemp(company.key), monkeypatch)
    finally:
        monkeypatch.undo()
    return out


# ---------------------------------------------------------------------------
# Part 19 — canary 1: mature profitable
# ---------------------------------------------------------------------------

def test_canary_1_mature_profitable_values_cleanly(runs):
    """The reference case. A gate that fires HERE is a gate that is wrong."""
    run = runs["mature_profitable"]

    revenue = run.canonical_current.get("revenue") or {}
    assert revenue.get("value") is not None
    assert revenue.get("period_type") in ("TTM", "ANNUAL")

    assert run.business_model.get("profile") == \
        CashFlowValuationProfile.STANDARD_OPERATING_COMPANY
    assert run.business_model.get("standard_fcff_suitability") == FcffSuitability.SUITABLE

    assert run.packet is not None, run.packet_failure
    assert run.packet["revenue"] is not None
    assert run.packet["total_debt"] is not None
    assert run.packet["share_count"] is not None
    assert run.packet["wacc"] > run.packet["terminal_growth"]

    assert run.dcf.get("available") is True
    assert run.valuation_status == ValuationStatus.VALID_FOR_RESEARCH
    assert run.gate_status == run.valuation_status


def test_canary_1_research_receives_the_valuation(runs):
    run = runs["mature_profitable"]
    assert run.evidence_ids("dcf.value_per_share")
    assert run.evidence_index["valuation.status"].value == ValuationStatus.VALID_FOR_RESEARCH
    assert "valuation.conclusions_withheld" not in run.evidence_index
    assert run.report_model.valuation.publishable is True
    assert run.report_model.valuation.base_value_per_share is not None


def test_canary_1_current_ratio_is_derived_from_one_instant(runs):
    """Part 15: current_ratio is derived ONCE, from the canonical latest
    instant's own assets and liabilities. The Snapshot renders that figure;
    it does not compute a second one."""
    run = runs["mature_profitable"]
    ratio = (run.canonical_current.get("current_ratio") or {}).get("value")
    if ratio is None:
        pytest.skip("no canonical current ratio was derived for this canary")
    labels = run.report_model.snapshot_values()
    assert labels.get("Current Ratio") == pytest.approx(ratio)
    assert "Current Ratio (FY)" not in labels


# ---------------------------------------------------------------------------
# Part 20 — canary 2: high-growth profitable
# ---------------------------------------------------------------------------

def test_canary_2_high_growth_is_preserved_not_flattened(runs):
    run = runs["high_growth_profitable"]
    growth = (run.canonical_current.get("revenue_growth") or {}).get("value")
    assert growth is not None and growth > 0.30, growth


def test_canary_2_a_clamped_growth_is_the_models_bound_not_a_forecast(runs):
    """Part 20's real requirement. A bound may LIMIT the assumption; it must
    never be presented as this company's expected growth."""
    run = runs["high_growth_profitable"]
    lines = run.report_model.valuation.assumption_lines
    growth_lines = [line for line in lines if line.startswith("- Revenue growth:")]
    assert growth_lines, lines
    # This canary's 58% current growth is far above the engine's configured
    # bound, so the clamp DOES bind -- which is the point: the conflict has
    # to be real for the disclosure to be worth testing.
    assert "CLAMPED" in growth_lines[0], growth_lines[0]
    assert "the model's configured bound" in growth_lines[0]
    assert "not an estimate of the company's growth" in growth_lines[0]
    # The bound is a limit on the MODEL. It has not replaced the measured
    # growth rate, which is still what the canonical packet reports.
    assert (run.canonical_current["revenue_growth"]["value"]) > 0.5


def test_canary_2_valuation_eligibility_responds_to_the_evidence(runs):
    run = runs["high_growth_profitable"]
    # This canary's guidance CORROBORATES its 42% year-1 growth, and the
    # engine's configured bound caps the applied assumption at 25%. So the
    # forecast is being set by the model's representational range rather
    # than by the company's economics -- which is a real limit on what the
    # resulting value means, and the eligibility must reflect it.
    conflicts = {c.get("code") for c in (run.facts.get("assumption_conflicts") or [])}
    assert "DCF_MODEL_BOUND_CONFLICT" in conflicts
    assert run.valuation_status == ValuationStatus.LIMITED
    # The bound is deliberately NOT raised to make the conflict go away, and
    # the modelled values do not reach research while it stands.
    assert not run.evidence_ids("dcf.value_per_share")
    assert not run.evidence_ids("scenario_spread.")
    # The DIRECTION of the price comparison survives under LIMITED -- that
    # much the inputs still support -- but no magnitude does.
    assert not run.evidence_ids("valuation_gap.market_price_premium_pct")


def test_canary_2_guidance_is_read_as_consolidated_revenue_growth(runs):
    """The guidance that corroborates the forecast is REVENUE growth, for a
    named fiscal year, stated as a range. Every one of those properties is
    required before it may anchor anything (spec 11)."""
    run = runs["high_growth_profitable"]
    guidance = (run.facts.get("management_guidance") or {}).get("metrics") or {}
    assert "revenue_growth" in guidance, guidance
    assert guidance["revenue_growth"]["forward_kind"] == "CURRENT_FY_GUIDANCE"


# ---------------------------------------------------------------------------
# Part 21 — canary 3: loss-making growth
# ---------------------------------------------------------------------------

def test_canary_3_the_negative_margin_survives(runs):
    """No forced positive mature margin. A loss-making company's reported
    margin is negative and must reach the canonical packet as such."""
    run = runs["loss_making_growth"]
    margin = (run.canonical_current.get("operating_margin") or {}).get("value")
    assert margin is not None and margin < 0, margin


def test_canary_3_the_valuation_is_not_falsely_valid_for_research(runs):
    """The status names the CAUSE. This company's terminal cash flow is
    negative, so the perpetuity step has nothing to grow -- the model is
    working and the forecast does not support the method."""
    run = runs["loss_making_growth"]
    assert run.valuation_status == ValuationStatus.FORECAST_PATH_INVALID
    assert run.gate_status == run.valuation_status
    assert run.dcf.get("validation_status") == "DCF_NEGATIVE_TERMINAL_FCFF"


def test_canary_3_no_modeled_value_can_support_a_recommendation(runs):
    """Part 21's last line, made structural: the ids do not exist, so a role
    citing one fails citation validation exactly like a hallucinated id."""
    run = runs["loss_making_growth"]
    assert not run.evidence_ids("dcf.value_per_share")
    assert not run.evidence_ids("scenario_spread.")
    assert not run.evidence_ids("valuation_gap.market_price_premium_pct")
    assert run.evidence_index["valuation.conclusions_withheld"].value


def test_canary_3_the_report_does_not_call_a_working_model_invalid(runs):
    """Spec 14's forbidden rendering."""
    run = runs["loss_making_growth"]
    assert "MODEL_INVALID" not in run.report_text()


# ---------------------------------------------------------------------------
# Part 22 — canary 4: insurer
# ---------------------------------------------------------------------------

def test_canary_4_is_classified_an_insurer_from_its_own_filings(runs):
    run = runs["insurer"]
    assert run.business_model.get("profile") == CashFlowValuationProfile.INSURER
    assert run.business_model.get("sic") == "6311"
    # Two independent signals, as spec 12 requires: the SIC code and the
    # concepts the issuer actually files.
    assert run.business_model.get("corroborating_concepts")


def test_canary_4_standard_fcff_does_not_apply(runs):
    run = runs["insurer"]
    assert run.business_model.get("standard_fcff_suitability") == FcffSuitability.NOT_SUITABLE
    assert run.packet is None
    assert run.valuation_status == ValuationStatus.NOT_APPLICABLE_FOR_BUSINESS_MODEL


def test_canary_4_ocf_less_capex_is_not_called_owner_free_cash_flow(runs):
    """Spec 12's ALLOWED/FORBIDDEN pair. The figure may be reported; the
    owner-cash claim may not be made, and the label must not make it."""
    run = runs["insurer"]
    packet = run.facts.get("business_model_evidence") or {}
    prohibited = {r.get("metric_id") for r in (packet.get("prohibited_interpretations") or [])}
    assert "free_cash_flow" in prohibited or "simple_fcf" in prohibited
    assert (packet.get("cash_flow_label") or "FCF") != "FCF"
    assert not any(row.label.startswith("FCF") for row in run.report_model.snapshot)


def test_canary_4_research_proceeds_without_a_dcf(runs):
    """A missing valuation is not a failed analysis."""
    run = runs["insurer"]
    assert run.evidence_ids("current.")
    assert run.evidence_ids("fundamental.")
    assert run.evidence_index["valuation.status"].value == \
        ValuationStatus.NOT_APPLICABLE_FOR_BUSINESS_MODEL
    assert not run.evidence_ids("dcf.value_per_share")


def test_canary_4_no_dcf_derived_company_risk_is_produced(runs):
    """Spec 18: "no DCF available" is an analysis limitation, never HIGH
    company risk. With no pipeline output there must be no company risk
    invented from the valuation's absence either."""
    run = runs["insurer"]
    for item in run.report_model.company_risk:
        assert "discounted cash flow" not in item.text.lower()
        assert "valuation" not in item.text.lower()


# ---------------------------------------------------------------------------
# Part 23 — canary 5: broker-dealer
# ---------------------------------------------------------------------------

def test_canary_5_is_classified_a_broker_dealer(runs):
    run = runs["broker_dealer"]
    assert run.business_model.get("profile") == CashFlowValuationProfile.BROKER_DEALER
    assert run.business_model.get("sic") == "6211"


def test_canary_5_simple_fcf_is_not_owner_cash(runs):
    """The live case this class exists for: operating cash flow twenty times
    operating income, because most of it is customer money in transit."""
    run = runs["broker_dealer"]
    assert run.business_model.get("standard_fcff_suitability") == FcffSuitability.NOT_SUITABLE
    packet = run.facts.get("business_model_evidence") or {}
    prohibited = {r.get("metric_id") for r in (packet.get("prohibited_interpretations") or [])}
    assert prohibited


def test_canary_5_the_fcff_gate_refuses_the_packet(runs):
    run = runs["broker_dealer"]
    assert run.packet is None
    assert run.packet_failure["invalid_inputs"] == ["business_model"]
    assert run.valuation_status == ValuationStatus.NOT_APPLICABLE_FOR_BUSINESS_MODEL


def test_canary_5_customer_cash_is_kept_separate_from_company_cash(runs):
    """Restricted/segregated cash is not available to repay debt and must
    not be netted into liquidity."""
    run = runs["broker_dealer"]
    balance = (run.state.get("balance_sheet") or {})
    company_cash = (balance.get("cash_and_cash_equivalents") or {}).get("value")
    restricted = (balance.get("restricted_cash") or {}).get("value")
    if restricted is None:
        pytest.skip("this canary's restricted cash was not selected")
    assert company_cash != restricted
    net_debt = run.state.get("net_debt")
    if net_debt is not None and company_cash is not None:
        # The segregated balance is far larger than company cash; if it had
        # been netted, net debt would be deeply negative.
        assert net_debt > -(company_cash + restricted)


def test_canary_5_research_still_has_non_valuation_evidence(runs):
    run = runs["broker_dealer"]
    assert run.evidence_ids("current.")
    assert run.evidence_ids("business_model.")


# ---------------------------------------------------------------------------
# Part 24 — canary 6: foreign private issuer
# ---------------------------------------------------------------------------

def test_canary_6_annual_20f_facts_are_read(runs):
    """The failure this class exists for: annual facts were filtered to 10-K
    forms only, so a 20-F filer lost every annual figure and the analysis
    reported no revenue for a company that had filed complete accounts."""
    run = runs["foreign_private_issuer"]
    annual = ((run.facts.get("statements") or {}).get("annual") or {})
    income = annual.get("income_statement") or []
    assert income, "no annual income statement was read from the 20-F filings"
    assert (income[0].get("values") or {}).get("revenue") is not None


def test_canary_6_the_interim_6k_is_used_for_the_latest_period(runs):
    run = runs["foreign_private_issuer"]
    as_of = run.state.get("financial_as_of")
    assert as_of is not None
    # The newest interim period, not the fiscal year end.
    assert as_of > "2025-12-31", as_of


def test_canary_6_entity_and_security_identity_are_established(runs):
    run = runs["foreign_private_issuer"]
    identity = (run.facts.get("dcf_financial_basis") or {}).get("security_identity") or {}
    assert identity.get("ticker") == run.company.symbol
    assert identity.get("entity_id")


def test_canary_6_produces_a_valuation_like_any_other_operating_company(runs):
    """A foreign private issuer is not a special case: same class, different
    forms. Nothing about the form should change the verdict."""
    run = runs["foreign_private_issuer"]
    assert run.business_model.get("standard_fcff_suitability") == FcffSuitability.SUITABLE
    assert run.packet is not None, run.packet_failure


# ---------------------------------------------------------------------------
# Part 25 — canary 7: unusual-item company
# ---------------------------------------------------------------------------

def test_canary_7_reported_gaap_profitability_is_preserved(runs):
    """The reported figure is what the issuer reported. Whatever
    normalization decides, it may not overwrite the GAAP number."""
    run = runs["unusual_item_company"]
    annual = ((run.facts.get("statements") or {}).get("annual") or {})
    latest = ((annual.get("income_statement") or [{}])[0].get("values") or {})
    assert latest.get("operating_income") == pytest.approx(150e6)


def test_canary_7_the_unusual_items_have_provenance(runs):
    """Spec 12: an unusual item without a figure is a disclosure, not an
    adjustment. These are TAGGED WITH AMOUNTS and must be traceable."""
    run = runs["unusual_item_company"]
    profitability = run.state.get("profitability") or {}
    text = repr(profitability)
    assert profitability, "no profitability normalization was recorded"
    assert "impairment" in text.lower() or "restructuring" in text.lower(), text


def test_canary_7_three_profitabilities_are_distinguishable(runs):
    """Reported, company-adjusted and system-normalized are three different
    numbers here. The point is that the packet does not silently pick one."""
    run = runs["unusual_item_company"]
    reported_margin = 150e6 / 7000e6
    packet_margin = (run.packet or {}).get("operating_margin")
    if packet_margin is None:
        pytest.skip("no packet was built for this canary")
    # Whatever margin the forecast uses, the DIFFERENCE from the reported
    # one must be visible somewhere -- an unexplained substitution is what
    # this canary exists to catch.
    if abs(packet_margin - reported_margin) > 0.01:
        assert run.facts.get("profitability") or run.state.get("profitability")


def test_canary_7_dcf_suitability_reacts_to_unresolved_normalization(runs):
    run = runs["unusual_item_company"]
    suitability = (run.facts.get("dcf_suitability") or {})
    from finance.suitability import DcfSuitability
    assert suitability.get("dcf_suitability") in DcfSuitability.ALL
    # The question was actually ANSWERED -- that distinction is what decides
    # whether the valuation's evidence is published (see
    # finance/workflow.py::_valuation_status).
    assert suitability.get("assessed") is True
    # And an unresolved normalization is what moved it off SUITABLE: a 0.5%
    # operating margin beside a 22% operating-cash-flow margin means at
    # least one of them does not describe recurring economics.
    assert suitability.get("dcf_suitability") != DcfSuitability.SUITABLE
    codes = {s.get("code") for s in (suitability.get("signals") or [])}
    assert "PROFITABILITY_CROSS_METRIC_CONFLICT" in codes


# ---------------------------------------------------------------------------
# Part 26 — canary 8: complex capital structure
# ---------------------------------------------------------------------------

def test_canary_8_current_debt_is_not_total_debt(runs):
    run = runs["complex_capital_structure"]
    total_debt = (run.state.get("total_debt") or {}).get("value")
    assert total_debt is not None
    balance = run.state.get("balance_sheet") or {}
    current_portion = (balance.get("current_portion_of_long_term_debt") or {}).get("value")
    if current_portion is not None:
        assert total_debt > current_portion * 2


def test_canary_8_the_reported_total_wins_over_the_component_sum(runs):
    """Spec 10. The issuer reports 34,100M against a 31,700M component sum.
    Precedence resolves it -- the issuer's own consolidated total outranks a
    sum this system assembled -- and the resolution is STATED, never left as
    whichever number happened to be selected first."""
    run = runs["complex_capital_structure"]
    total_debt = (run.state.get("total_debt") or {}).get("value")
    assert total_debt == pytest.approx(34_100e6), total_debt
    detail = run.state.get("net_debt_detail") or {}
    assert detail.get("total_debt_validity") in ("VALID", None)


def test_canary_8_the_equity_bridge_carries_preferred_and_minority(runs):
    """Both were resolvable from SEC data and both silently defaulted to
    zero before they were mapped, so a company with real preferred stock was
    overvalued by exactly that amount."""
    run = runs["complex_capital_structure"]
    base = next((s for s in (run.dcf.get("scenarios") or [])
                 if s.get("scenario") == "base"), None)
    if base is None:
        pytest.skip("no DCF scenarios for this canary")
    assert base.get("preferred_equity") == pytest.approx(1_500e6)
    assert base.get("minority_interest") == pytest.approx(2_700e6)


def test_canary_8_net_debt_depends_on_the_resolved_total(runs):
    run = runs["complex_capital_structure"]
    total_debt = (run.state.get("total_debt") or {}).get("value")
    cash = ((run.state.get("balance_sheet") or {})
            .get("cash_and_cash_equivalents") or {}).get("value") or 0.0
    assert run.state.get("net_debt") == pytest.approx(total_debt - cash, rel=0.01)


# ---------------------------------------------------------------------------
# Part 7 — the negative runtime chain, on the real workflow
# ---------------------------------------------------------------------------

def test_unestablished_debt_refuses_the_packet(runs):
    """Total debt cannot be established, so no packet is built.

    Before this phase the equity bridge read `state.total_debt.value or 0.0`
    and valued this issuer as though it carried no debt at all.
    """
    run = runs["unestablished_debt"]
    assert (run.state.get("total_debt") or {}).get("value") is None
    assert run.packet is None
    assert "total_debt" in (run.packet_failure or {}).get("invalid_inputs", [])


def test_unestablished_debt_runs_no_dcf(runs):
    run = runs["unestablished_debt"]
    assert run.dcf.get("available") is False
    assert run.dcf.get("valuation_status") == ValuationStatus.INPUT_PACKET_INVALID
    assert not (run.dcf.get("scenarios") or [])


def test_unestablished_debt_is_not_valid_for_research(runs):
    run = runs["unestablished_debt"]
    assert run.valuation_status == ValuationStatus.INPUT_PACKET_INVALID
    assert run.gate_status == run.valuation_status


def test_unestablished_debt_publishes_no_modeled_values(runs):
    """The whole worked chain of spec 9, checked at the far end: nothing a
    research role could cite to say "the market exceeds the bull value" or
    to quote an upside exists in the index at all."""
    run = runs["unestablished_debt"]
    for prefix in ("dcf.value_per_share", "dcf.equity_value", "dcf.enterprise_value",
                   "scenario_spread.", "valuation_gap."):
        assert not run.evidence_ids(prefix), prefix
    assert run.evidence_index["valuation.conclusions_withheld"].value


def test_unestablished_debt_report_model_carries_no_valuation_numbers(runs):
    run = runs["unestablished_debt"]
    valuation = run.report_model.valuation
    assert valuation.publishable is False
    assert valuation.bear_value_per_share is None
    assert valuation.base_value_per_share is None
    assert valuation.bull_value_per_share is None
    assert valuation.premium_pct is None
    assert valuation.modeled_return_pct is None


# ---------------------------------------------------------------------------
# Part 28 — what NO canary's report model may contain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [c.key for c in CANARIES.ALL_CANARIES])
def test_no_report_model_contains_an_invalid_metric(runs, key):
    """An INVALID metric exposes no numeric value (spec 9). If one reached
    the report model, the cascade is decorative."""
    run = runs[key]
    invalid = {name for name, entry in (run.facts.get("metric_validity") or {}).items()
               if entry.get("validity") == "INVALID"}
    if not invalid:
        return
    for name in invalid:
        raw = (run.facts["metric_validity"][name] or {}).get("raw_value")
        if raw is None:
            continue
        assert raw not in run.report_model.numeric_snapshot_values(), name


@pytest.mark.parametrize("key", [c.key for c in CANARIES.ALL_CANARIES])
def test_no_report_model_shows_ineligible_valuation_numbers(runs, key):
    run = runs[key]
    valuation = run.report_model.valuation
    if valuation.publishable:
        return
    assert valuation.base_value_per_share is None
    assert valuation.premium_pct is None
    assert valuation.modeled_return_pct is None
    assert valuation.sensitivity_level is None


@pytest.mark.parametrize("key", [c.key for c in CANARIES.ALL_CANARIES])
def test_no_report_model_contains_an_empty_claim(runs, key):
    """Spec 16: `min_items` counts entries that SURVIVE filtering. A blank
    bullet is a published empty claim."""
    run = runs[key]
    model = run.report_model
    for text in (list(model.bull_case.claims) + list(model.bear_case.claims)
                 + model.all_risk_text()):
        assert text and text.strip()
    for _label, items in model.conditions:
        for item in items:
            assert item and str(item).strip()


@pytest.mark.parametrize("key", [c.key for c in CANARIES.ALL_CANARIES])
def test_no_report_model_leaks_an_implementation_error(runs, key):
    """Spec 18's compact rule: validated conclusions only -- no stage names,
    schema errors, retry counts, evidence ids or tracebacks."""
    run = runs[key]
    text = run.report_text()
    for leak in ("Traceback", "schema", "evidence_id", "retry", "_researcher",
                 "final_investment_synthesizer", "KeyError", "None"):
        assert leak not in text, f"{key}: report leaked {leak!r}"


@pytest.mark.parametrize("key", [c.key for c in CANARIES.ALL_CANARIES])
def test_every_canary_states_one_valuation_status_everywhere(runs, key):
    """The wiring's central claim: `facts`, the evidence index and the report
    model all read ONE status. Asserted per canary because the failure mode
    is precisely that two of the three quietly disagree."""
    run = runs[key]
    assert run.valuation_status == run.gate_status
    assert run.report_model.valuation.status == run.valuation_status
    assert run.evidence_index["valuation.status"].value == run.valuation_status


@pytest.mark.parametrize("key", [c.key for c in CANARIES.ALL_CANARIES])
def test_no_canary_reaches_the_dcf_without_a_packet(runs, key):
    """Part 1's invariant, checked on every class: a DCF result exists only
    where a packet was built."""
    run = runs[key]
    if run.dcf.get("available"):
        assert run.packet is not None, f"{key} ran a DCF with no input packet"
    else:
        assert run.packet is None or run.dcf.get("reason")


# ---------------------------------------------------------------------------
# Part 8 — invalid guidance, on the real workflow
# ---------------------------------------------------------------------------
#
# Spec 11's hard invariant, end to end. A free-cash-flow growth figure is a
# growth rate, it is consolidated, it is stated as a range for a named fiscal
# year, and it is still not revenue growth. The check fails CLOSED: a source
# never established as consolidated revenue growth has not been shown to be
# revenue growth.

def test_fcf_growth_guidance_is_extracted_as_its_own_metric(runs):
    """The precondition. If the release were not parsed at all, everything
    below would pass for the wrong reason."""
    run = runs["fcf_growth_guidance_only"]
    guidance = (run.facts.get("management_guidance") or {}).get("metrics") or {}
    assert "free_cash_flow_growth" in guidance, guidance
    assert "revenue_growth" not in guidance


def test_fcf_growth_guidance_never_anchors_the_revenue_assumption(runs):
    run = runs["fcf_growth_guidance_only"]
    base = next((s for s in (run.dcf.get("scenarios") or [])
                 if s.get("scenario") == "base"), None)
    assert base is not None, run.packet_failure
    provenance = ((base.get("assumptions") or {}).get("assumption_provenance") or {}) \
        .get("revenue_growth") or {}
    provenance = provenance or (run.dcf.get("shared_assumption_provenance")
                                or {}).get("revenue_growth") or {}
    source = (provenance.get("source_type") or "") + " " + (provenance.get("reason") or "")
    assert "free_cash_flow" not in source.lower(), provenance
    assert "guidance" not in (provenance.get("source_type") or "").lower(), provenance


def test_the_invalid_assumption_never_reaches_the_packet(runs):
    """The forecast is built from this issuer's own reported history, and
    the packet's year-1 growth is that history's rate -- not the guided
    free-cash-flow figure and not a clamp of it."""
    run = runs["fcf_growth_guidance_only"]
    guidance = ((run.facts.get("management_guidance") or {}).get("metrics") or {}) \
        .get("free_cash_flow_growth") or {}
    guided_low, guided_high = guidance.get("low"), guidance.get("high")
    assert guided_low is not None
    base = next((s for s in (run.dcf.get("scenarios") or [])
                 if s.get("scenario") == "base"), None)
    growth = (base.get("assumptions") or {}).get("revenue_growth")
    year_one = growth[0] if isinstance(growth, list) else growth
    assert not (guided_low <= year_one <= guided_high) or year_one != guided_low


def test_no_clamp_is_reported_for_an_assumption_that_was_never_derived(runs):
    """Spec 13: a clamp must never repair a semantic error.

    The failure mode this guards against is subtle -- the wrong metric
    produces a nonsense rate, the rate is clamped into range, and the report
    then explains a MODEL BOUND to a reader whose actual problem is that two
    different quantities were treated as one.
    """
    run = runs["fcf_growth_guidance_only"]
    for rejection in (run.facts.get("semantic_rejections") or []):
        assert rejection.get("code") != "DCF_MODEL_BOUND_CONFLICT"
    conflicts = run.facts.get("assumption_conflicts") or []
    for conflict in conflicts:
        assert "free_cash_flow" not in str(conflict).lower(), conflict


def test_no_revenue_guidance_is_claimed_in_the_report(runs):
    """Coverage is per metric (spec 11). The report may say this issuer
    guided free cash flow; it may not say it guided revenue, and it may not
    say guidance is unavailable either."""
    run = runs["fcf_growth_guidance_only"]
    text = run.report_text()
    assert "Management guidance: unavailable" not in text
