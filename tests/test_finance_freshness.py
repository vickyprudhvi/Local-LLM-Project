"""Phase H.4 — freshness planning, stale guards, and forward assumptions
(spec sections 1, 2, 9, 10, 11, 12, 13, 21).

Synthetic company_facts throughout, so each rule is pinned in isolation; the
real-data counterparts are in tests/test_finance_aos_regression.py.
"""

import pytest

from finance import forward_assumptions as fa
from finance.dcf import AssumptionSourceType
from finance.freshness import (
    DCF_CURRENT_GUIDANCE_NOT_CONSIDERED,
    DCF_STALE_DEBT_INPUT,
    DcfFreshnessPlanner,
    ValuationFreshness,
    build_current_financial_state,
)

REVENUE = "RevenueFromContractWithCustomerExcludingAssessedTax"


def _facts(rows: dict) -> dict:
    return {"facts": {"us-gaap": {
        c: {"units": {"USD": r}} for c, r in rows.items()}}}


def _instant(end, val, *, filed=None, form="10-Q"):
    return {"end": end, "val": val, "fy": 2026, "fp": "Q2", "form": form,
            "filed": filed or end, "accn": "a-1"}


def _duration(start, end, val, *, form="10-Q", filed=None, fp="Q1"):
    return {"start": start, "end": end, "val": val, "fy": 2026, "fp": fp,
            "form": form, "filed": filed or end, "accn": "a-1"}


def _company(annual_end="2025-12-31", quarter_end="2026-06-30",
             annual_debt=100.0, quarter_debt=600.0,
             annual_cash=50.0, quarter_cash=60.0):
    """A company with BOTH an annual and a newer quarterly balance sheet."""
    return _facts({
        "Assets": [_instant(annual_end, 1000, form="10-K"), _instant(quarter_end, 1200)],
        "StockholdersEquity": [_instant(annual_end, 400, form="10-K"),
                               _instant(quarter_end, 420)],
        "CashAndCashEquivalentsAtCarryingValue": [
            _instant(annual_end, annual_cash, form="10-K"),
            _instant(quarter_end, quarter_cash)],
        "LongTermDebtNoncurrent": [_instant(annual_end, annual_debt, form="10-K"),
                                   _instant(quarter_end, quarter_debt)],
        # Three fiscal years of year-to-date columns plus two annual totals.
        # Two full years of discrete quarters are needed for a TTM-over-TTM
        # growth rate, which is what the assumption ladder falls through to
        # when there is no guidance.
        REVENUE: [
            _duration("2024-01-01", "2024-12-31", 900, form="10-K", fp="FY"),
            _duration("2024-01-01", "2024-03-31", 210),
            _duration("2024-01-01", "2024-06-30", 440),
            _duration("2024-01-01", "2024-09-30", 670),
            _duration("2025-01-01", annual_end, 1000, form="10-K", fp="FY"),
            _duration("2025-01-01", "2025-03-31", 240),
            _duration("2025-01-01", "2025-06-30", 490),
            _duration("2025-01-01", "2025-09-30", 740),
            _duration("2026-01-01", "2026-03-31", 260),
            _duration("2026-01-01", "2026-06-30", 530),
        ],
    })


# ---------------------------------------------------------------------------
# 1-3: quarterly supersedes annual, per field
# ---------------------------------------------------------------------------

def test_the_latest_quarter_supersedes_the_annual_balance_sheet():
    state = build_current_financial_state(_company(), "TEST")
    assert state.financial_as_of == "2026-06-30"
    assert state.latest_annual_period == "2025-12-31"
    assert state.latest_quarterly_period == "2026-06-30"
    assert state.balance_sheet["cash_and_cash_equivalents"].source == "quarterly_sec_filing"


def test_old_annual_debt_cannot_override_newer_quarterly_debt():
    state = build_current_financial_state(_company(annual_debt=100, quarter_debt=600), "TEST")
    assert state.total_debt.value == pytest.approx(600)
    assert state.total_debt.as_of_date == "2026-06-30"


def test_old_annual_cash_cannot_override_newer_quarterly_cash():
    state = build_current_financial_state(_company(annual_cash=50, quarter_cash=60), "TEST")
    assert state.value("cash_and_cash_equivalents") == pytest.approx(60)
    assert state.net_debt == pytest.approx(600 - 60)


def test_every_selected_field_preserves_full_provenance():
    """Section 2's requirement, field by field."""
    state = build_current_financial_state(_company(), "TEST")
    selection = state.balance_sheet["long_term_debt"]
    assert selection.provider == "sec"
    assert selection.accession
    assert selection.form
    assert selection.as_of_date == "2026-06-30"
    assert selection.retrieval_timestamp
    assert selection.evidence_id == "dcf.input.long_term_debt.latest"
    assert selection.freshness_status == "current_quarter"


def test_a_field_absent_from_the_current_balance_sheet_is_not_back_filled():
    """A balance sheet is one coherent moment. Pulling a field forward from
    an older filing produces a bridge that describes no actual date."""
    facts = _company()
    facts["facts"]["us-gaap"]["ShortTermBorrowings"] = {
        "units": {"USD": [_instant("2010-09-30", 158)]}}
    state = build_current_financial_state(facts, "TEST")
    assert state.value("short_term_debt") is None
    assert "too old" in (state.balance_sheet["short_term_debt"].derivation or "")
    # ...and the stale value is therefore not in total debt.
    assert state.total_debt.value == pytest.approx(600)


# ---------------------------------------------------------------------------
# 4-5: stale guards and freshness propagation
# ---------------------------------------------------------------------------

def test_a_material_debt_change_since_the_year_end_is_reported():
    state = build_current_financial_state(
        _company(annual_debt=100, quarter_debt=600), "TEST")
    finding = next(f for f in state.findings if f["code"] == DCF_STALE_DEBT_INPUT)
    assert finding["prior_total_debt"] == pytest.approx(100)
    assert finding["current_total_debt"] == pytest.approx(600)
    assert finding["change_ratio"] == pytest.approx(5.0)


def test_an_immaterial_debt_change_is_not_reported_as_a_finding():
    state = build_current_financial_state(
        _company(annual_debt=100, quarter_debt=105), "TEST")
    assert not any(f["code"] == DCF_STALE_DEBT_INPUT for f in state.findings)


def test_guidance_retrieved_but_not_routed_is_a_finding():
    guidance = {"fiscal_year": 2026, "metrics": {"revenue_growth": {"low": 0.02, "high": 0.03}}}
    state = build_current_financial_state(
        _company(), "TEST", management_guidance=guidance, guidance_considered=False)
    assert any(f["code"] == DCF_CURRENT_GUIDANCE_NOT_CONSIDERED for f in state.findings)
    assert state.valuation_freshness == ValuationFreshness.STALE_INPUT_WARNING


def test_data_completeness_and_valuation_freshness_are_independent():
    """Section 13. The AOS failure was COMPLETE data and a stale valuation;
    a single status cannot express that."""
    guidance = {"fiscal_year": 2026, "metrics": {"revenue_growth": {"low": 0.02, "high": 0.03}}}
    state = build_current_financial_state(
        _company(), "TEST", management_guidance=guidance, guidance_considered=False,
        data_completeness="COMPLETE")
    assert state.data_completeness == "COMPLETE"
    assert state.valuation_freshness == ValuationFreshness.STALE_INPUT_WARNING


def test_a_clean_run_with_quarterly_balance_sheet_and_ttm_flows_is_current():
    state = build_current_financial_state(_company(), "TEST")
    assert state.valuation_freshness == ValuationFreshness.CURRENT
    assert state.flows["revenue"].source == "ttm_calculation"


# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------

def test_free_cash_flow_is_never_built_from_two_different_bases():
    """A trailing-twelve-month operating cash flow minus an annual capex is a
    figure belonging to no period at all."""
    planner = DcfFreshnessPlanner(_company(), "TEST")
    ocf = planner.select_flow("operating_cash_flow", reference_date="2026-06-30")
    assert ocf.value is None or ocf.source in ("ttm_calculation", "annual_sec_filing")
    state = build_current_financial_state(_company(), "TEST")
    fcf = state.flows["free_cash_flow"]
    if fcf.value is not None:
        assert fcf.source == state.flows["operating_cash_flow"].source


def test_total_debt_sums_only_the_components_actually_reported():
    """The documented policy: missing components are excluded, never
    zero-filled, and which ones fed the total stays visible."""
    state = build_current_financial_state(_company(), "TEST")
    assert state.total_debt.components == ("long_term_debt",)
    assert "long_term_debt=600" in state.total_debt.derivation


# ---------------------------------------------------------------------------
# 16-20: forward assumptions
# ---------------------------------------------------------------------------

def _state_with(guidance=None, historical_cagr=None):
    return build_current_financial_state(
        _company(), "TEST", management_guidance=guidance,
        historical_metrics={"revenue_cagr": historical_cagr} if historical_cagr else None)


def _guidance(low, high, period_type="annual"):
    """Guidance for a FULL YEAR unless a test says otherwise.

    `period_type` is explicit because forecast-horizon eligibility now reads
    it: guidance that covers a shorter span than the annual assumption may
    corroborate the direction but may not set the year-1 rate. A fixture that
    omitted the field was asserting annual behaviour without stating an
    annual horizon, and the extractor always records one.
    """
    return {"fiscal_year": 2026, "guidance_date": "2026-07-30",
            "metrics": {"revenue_growth": {
                "low": low, "high": high, "fiscal_year": 2026,
                "period_type": period_type,
                "evidence_id": "dcf.guidance.revenue_growth.current"}}}


def test_current_guidance_outranks_every_historical_signal():
    state = _state_with(guidance=_guidance(0.02, 0.03), historical_cagr=0.072)
    paths, evidence = fa.build_forward_assumptions(state, 5, company_facts=_company())
    growth = paths["revenue_growth"]
    assert growth.anchor_source == AssumptionSourceType.MANAGEMENT_GUIDANCE
    assert growth.values[0] == pytest.approx(0.025)
    assert evidence.historical_cagr is not None  # still collected, as context


def test_historical_cagr_is_context_and_says_so_when_not_used():
    state = _state_with(guidance=_guidance(0.02, 0.03), historical_cagr=0.072)
    paths, _ = fa.build_forward_assumptions(state, 5, company_facts=_company())
    notes = " ".join(paths["revenue_growth"].notes)
    assert "context" in notes.lower()
    assert "not used as the forecast anchor" in notes


def test_without_guidance_the_ladder_falls_through_to_the_ttm_trend():
    state = _state_with(historical_cagr=0.072)
    paths, evidence = fa.build_forward_assumptions(state, 5, company_facts=_company())
    assert paths["revenue_growth"].anchor_source == AssumptionSourceType.TTM_CALCULATION
    assert evidence.ttm_yoy is not None


def test_the_forecast_fades_toward_the_long_run_rate():
    state = _state_with(guidance=_guidance(0.10, 0.10))
    paths, _ = fa.build_forward_assumptions(state, 5, company_facts=_company(),
                                            terminal_growth=0.025)
    values = paths["revenue_growth"].values
    assert values[0] == pytest.approx(0.10)
    assert values[-1] == pytest.approx(0.025)
    assert values == sorted(values, reverse=True)


def test_every_assumption_entry_carries_its_evidence_ids_and_year():
    """Section 11's provenance requirements, per forecast year."""
    state = _state_with(guidance=_guidance(0.02, 0.03))
    paths, _ = fa.build_forward_assumptions(state, 5, company_facts=_company())
    entries = paths["revenue_growth"].entries
    assert len(entries) == 5
    for index, entry in enumerate(entries, start=1):
        assert entry.forecast_year == index
        assert entry.units == "ratio"
        assert entry.approval_status == "proposed"
        assert entry.derivation
        assert "dcf.guidance.revenue_growth.current" in entry.evidence_ids


def test_a_clamped_assumption_records_what_it_was_clamped_from():
    state = _state_with(guidance=_guidance(0.90, 0.90))
    paths, _ = fa.build_forward_assumptions(state, 5, company_facts=_company())
    first = paths["revenue_growth"].entries[0]
    assert first.clamped is True
    assert first.original_proposed_value == pytest.approx(0.90)
    assert first.applied_value == pytest.approx(fa.GROWTH_BOUNDS[1])
    # Phase H.6, section 19: the derivation now explains that the applied
    # value is the MODEL'S BOUND rather than an estimate, and carries the
    # derived value beside it, instead of just flagging the word CLAMPED.
    assert first.raw_value == pytest.approx(0.90)
    assert first.clamp_reason and "NOT an estimate" in first.clamp_reason
    assert "configured bound" in first.derivation


# ---------------------------------------------------------------------------
# Section 10: guidance is an input, not truth
# ---------------------------------------------------------------------------

def _evidence(low=0.02, high=0.03, period_type="annual"):
    return fa.GrowthEvidence(guidance_low=low, guidance_high=high,
                             guidance_fiscal_year=2026,
                             guidance_period_type=period_type)


def test_a_forecast_inside_guidance_is_accepted_without_special_justification():
    outcome = fa.validate_proposal(
        {"revenue_growth": {f"year_{i}": 0.025 for i in range(1, 6)},
         "operating_margin": {f"year_{i}": 0.19 for i in range(1, 6)},
         "confidence": 0.6},
        5, _evidence())
    assert outcome.accepted is True
    assert outcome.growth[0] == pytest.approx(0.025)


def test_a_forecast_far_outside_guidance_requires_an_explicit_justification():
    """Section 10's exact example: guidance 2-3%, proposal 7.2%."""
    proposal = {"revenue_growth": {f"year_{i}": 0.072 for i in range(1, 6)},
                "operating_margin": {f"year_{i}": 0.19 for i in range(1, 6)},
                "confidence": 0.6}
    rejected = fa.validate_proposal(proposal, 5, _evidence())
    assert rejected.accepted is False
    assert any("guidance" in f for f in rejected.findings)

    proposal["justification"] = (
        "Trailing-twelve-month revenue is already growing faster than the guided range "
        "and the prior two guidance updates were both raised; see dcf.input.revenue_ttm.")
    accepted = fa.validate_proposal(proposal, 5, _evidence())
    assert accepted.accepted is True


def test_a_forecast_below_guidance_is_permitted_with_justification():
    outcome = fa.validate_proposal(
        {"revenue_growth": {f"year_{i}": -0.02 for i in range(1, 6)},
         "operating_margin": {f"year_{i}": 0.19 for i in range(1, 6)},
         "justification": "Volumes declined in each of the last three reported quarters "
                          "and the guided range assumes a recovery not visible in the data.",
         "confidence": 0.4},
        5, _evidence())
    assert outcome.accepted is True
    assert outcome.growth[0] == pytest.approx(-0.02)


def test_a_malformed_proposal_falls_back_to_the_baseline_rather_than_failing():
    for bad in ({}, {"revenue_growth": "fast"},
                {"revenue_growth": {"year_1": 0.02}, "operating_margin": {}},
                {"revenue_growth": [0.02, 0.03], "operating_margin": [0.1, 0.1]}):
        outcome = fa.validate_proposal(bad, 5, fa.GrowthEvidence())
        assert outcome.accepted is False
        assert outcome.findings


def test_a_proposal_citing_an_unknown_evidence_id_is_rejected():
    outcome = fa.validate_proposal(
        {"revenue_growth": {f"year_{i}": 0.025 for i in range(1, 6)},
         "operating_margin": {f"year_{i}": 0.19 for i in range(1, 6)},
         "reasoning_evidence_ids": ["dcf.input.made_up"],
         "confidence": 0.6},
        5, _evidence(), valid_evidence_ids=["dcf.input.revenue_ttm"])
    assert outcome.accepted is False
    assert any("do not exist" in f for f in outcome.findings)


def test_an_accepted_proposal_is_labelled_llm_proposed_and_keeps_the_baseline_context():
    state = _state_with(guidance=_guidance(0.02, 0.03))
    baseline, evidence = fa.build_forward_assumptions(state, 5, company_facts=_company())
    outcome = fa.validate_proposal(
        {"revenue_growth": {f"year_{i}": 0.028 for i in range(1, 6)},
         "operating_margin": {f"year_{i}": 0.19 for i in range(1, 6)},
         "confidence": 0.55},
        5, evidence)
    applied = fa.apply_proposal(baseline, outcome)
    entry = applied["revenue_growth"].entries[0]
    assert entry.source_type == AssumptionSourceType.LLM_PROPOSED
    assert "deterministic baseline" in entry.derivation
    assert "management_guidance" in entry.derivation


def test_the_evidence_block_labels_forward_and_historical_separately():
    """Section 17: the model must be able to tell the kinds apart."""
    state = _state_with(guidance=_guidance(0.02, 0.03), historical_cagr=0.072)
    paths, evidence = fa.build_forward_assumptions(state, 5, company_facts=_company())
    block = fa.build_evidence_block(state, evidence, paths)
    assert "CURRENT MANAGEMENT GUIDANCE" in block
    assert "FORWARD-LOOKING" in block
    assert "HISTORICAL CONTEXT ONLY" in block


def test_the_evidence_block_states_guidance_is_unavailable_when_it_is():
    state = _state_with()
    paths, evidence = fa.build_forward_assumptions(state, 5, company_facts=_company())
    block = fa.build_evidence_block(state, evidence, paths)
    # Phase H.6 names WHICH guidance is unavailable: a company can publish
    # component or EBITDA guidance and still have none for consolidated
    # revenue growth, and those are different statements.
    assert "CURRENT MANAGEMENT GUIDANCE for consolidated revenue growth: unavailable" in block
