"""Phase H.2 — finance/research_pipeline.py: the staged multi-agent research
pipeline (independent bull/bear researchers, one bounded rebuttal round,
research-manager reconciliation, risk review, FinalInvestmentSynthesizer),
adapted from TauricResearch/TradingAgents' role structure at a pinned,
reviewed commit -- NOT imported as a runtime dependency.

Unit-level: a small, hand-built evidence index (not a real AnalysisResult) so
each stage's validation and the orchestrator's cascading fail-closed logic are
checked in isolation and stay fast. See
tests/test_finance_research_pipeline_integration.py for the real, end-to-end
proof through finance.workflow.synthesize_report.
"""

import inspect
import json

import pytest

import finance.research_pipeline as rp
import tools.config as config
from finance.evidence import build_evidence_index, render_evidence_index


@pytest.fixture(autouse=True)
def _two_attempts_per_stage(monkeypatch):
    """Pin stage attempts to 2 for this module: one attempt plus one repair.

    Most tests here assert the shape of the REPAIR path ("exactly one repair
    attempt, never a retry loop"), which predates attempts being
    configurable. Pinning keeps every one of those assertions meaningful and
    deterministic rather than coupling them to whatever the default happens
    to be. The default itself (3) and the per-class recovery routing are
    covered separately in tests/test_finance_stage_recovery.py.
    """
    monkeypatch.setenv("RESEARCH_STAGE_MAX_ATTEMPTS", "2")

PAYLOAD = {
    "symbol": "TEST",
    "quote": {"price": 200.0, "price_basis": "delayed"},
    "fundamental_metrics": {
        "revenue_growth_yoy": {"value": 0.12, "formula": "..."},
    },
    "technical_metrics": {
        "rsi_14": {"value": 71.2, "formula": "..."},
    },
    "dcf": {
        "available": True, "net_debt": 150.0, "net_debt_policy": "cash_only",
        "value_per_share": 210.5,
        "scenarios": [
            {"scenario": "bull", "value_per_share": 250.0, "enterprise_value": 5000.0,
             "equity_value": 4850.0, "net_debt": 150.0,
             "terminal_value_share_of_enterprise_value": 0.6},
            {"scenario": "base", "value_per_share": 210.5, "enterprise_value": 4200.0,
             "equity_value": 4050.0, "net_debt": 150.0,
             "terminal_value_share_of_enterprise_value": 0.55},
            {"scenario": "bear", "value_per_share": 170.0, "enterprise_value": 3400.0,
             "equity_value": 3250.0, "net_debt": 150.0,
             "terminal_value_share_of_enterprise_value": 0.5},
        ],
    },
    "valuation_gap": {"available": True, "market_price": 200.0, "market_price_basis": "delayed",
                      "estimated_base_modeled_value_per_share": 210.5, "difference": 10.5,
                      "difference_pct": 5.25, "direction": "undervalued"},
    "dcf_scenario_spread": {"available": True, "bull_value_per_share": 250.0,
                            "base_value_per_share": 210.5, "bear_value_per_share": 170.0,
                            "spread": 80.0, "spread_pct_of_base": 38.0},
    "warnings": [],
    "data_provenance": {"stock_quote": {"origin": "provider", "stale": False}},
    "plan": {"omitted_datasets": []},
}

INDEX = build_evidence_index(PAYLOAD)

EV1 = "quote.price"
EV2 = "dcf.net_debt"
EV3 = "valuation_gap.direction"
EV4 = "fundamental.revenue_growth_yoy"
assert EV1 in INDEX and EV2 in INDEX and EV3 in INDEX and EV4 in INDEX  # fixture sanity


def _bull_response(evidence_cited=(EV1, EV2), extra=None):
    body = {"thesis": "Growth and valuation support upside.",
            "claims": [
                {"claim_id": "bull-1", "claim": "Revenue growth is healthy.",
                 "evidence_ids": [EV4], "claim_type": "fact_interpretation",
                 "assumptions": [], "confidence": 0.6},
                {"claim_id": "bull-2", "claim": "Net debt is manageable.",
                 "evidence_ids": list(evidence_cited), "claim_type": "fact_interpretation",
                 "assumptions": [], "confidence": 0.6},
            ],
            "confidence": "medium"}
    if extra:
        body.update(extra)
    return json.dumps(body)


def _bear_response(evidence_cited=(EV3,)):
    return json.dumps({
        "thesis": "Valuation looks stretched.",
        "claims": [
            {"claim_id": "bear-1", "claim": "Price exceeds intrinsic estimate.",
             "evidence_ids": list(evidence_cited), "claim_type": "scenario_interpretation",
             "assumptions": [], "confidence": 0.6},
            {"claim_id": "bear-2", "claim": "Momentum looks elevated.",
             "evidence_ids": [EV1], "claim_type": "fact_interpretation",
             "assumptions": [], "confidence": 0.5},
        ],
        "confidence": "medium",
    })


def _rebuttal_response():
    return json.dumps({
        "bull_rebuttal": {"response": "Growth still supports the multiple.", "evidence_cited": [EV4]},
        "bear_rebuttal": {"response": "The gap remains real.", "evidence_cited": [EV3]},
    })


def _research_manager_response():
    return json.dumps({
        "evidence_balance": "mixed",
        "supported_bull_points": ["Revenue growth is healthy."],
        "supported_bear_points": ["Price exceeds intrinsic estimate."],
        "unsupported_points": [],
        "shared_findings": ["Both sides have real, evidence-backed points."],
        "key_disagreements": ["Whether growth justifies the multiple."],
        "assumption_sensitive_conclusions": [],
        "data_gaps": [],
        "balanced_assessment": "Both sides have real, evidence-backed points.",
        "evidence_cited": [EV1, EV3],
    })


def _risk_reviewer_response():
    return json.dumps({
        "key_risks": [{"risk": "Valuation could compress.", "severity": "medium", "evidence_cited": [EV3]}],
        "data_quality_concerns": [],
        "evidence_cited": [EV3],
    })


def _final_response(rationale_statement="The evidence shows a balanced case with a real valuation gap."):
    return json.dumps({
        "research_stance": "neutral", "valuation_view": "approximately_fair",
        "overall_risk": "moderate", "confidence": 0.5, "recommendation": "hold",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": rationale_statement, "evidence_ids": [EV3]}],
        "conditions_that_strengthen_the_view": ["Continued revenue growth."],
        "conditions_that_weaken_the_view": ["A pullback in the valuation gap."],
        "key_uncertainties": ["Assumption sensitivity in the DCF scenarios."],
    })


def _dispatch(system, bull=None, bear=None, rebuttal=None, research_manager=None,
             risk=None, final=None, default=None):
    """Route a fake ask_local_fn call to the right canned response by the
    stage's distinctive system-prompt prefix. Anchored with startswith (not a
    loose substring) since the research-manager prompt's own text contains
    the phrase "rebuttal round", which a loose match would misroute."""
    if system.startswith("You are the Bull Researcher"):
        content = bull if bull is not None else _bull_response()
    elif system.startswith("You are the Bear Researcher"):
        content = bear if bear is not None else _bear_response()
    elif system.startswith("You are facilitating exactly ONE bounded rebuttal round"):
        content = rebuttal if rebuttal is not None else _rebuttal_response()
    elif system.startswith("You are the Research Manager"):
        content = research_manager if research_manager is not None else _research_manager_response()
    elif system.startswith("You are the Risk Reviewer"):
        content = risk if risk is not None else _risk_reviewer_response()
    elif system.startswith("You are the FinalInvestmentSynthesizer"):
        content = final if final is not None else _final_response()
    else:
        content = default if default is not None else "unexpected stage"
    return content


def make_ask_local(record=None, **overrides):
    """A fake ask_local_fn. `record`, if given, is a list that every
    (messages, kwargs) call is appended to -- for assertions made AFTER the
    pipeline returns (never inside the fake: _run_stage's `except Exception`
    would silently swallow an AssertionError raised from inside the fake and
    turn it into an unremarkable FAILED checkpoint instead of a visible test
    failure)."""

    def fn(messages, **kwargs):
        if record is not None:
            record.append((messages, kwargs))
        system = messages[0]["content"]
        content = _dispatch(system, **overrides)
        return {"message": {"role": "assistant", "content": content},
                "metrics": {"prompt_tokens": 100, "completion_tokens": 20}, "ok": True}

    return fn


def make_ask_local_with_final_sequence(final_responses, record=None, **overrides):
    """Like make_ask_local, but the FinalInvestmentSynthesizer stage returns
    each of `final_responses` in order across successive calls -- call 1 is
    the first attempt, call 2 (if reached) is `_run_final_synthesizer_stage`'s
    one repair attempt (Phase H.3 corrective patch, Problem 1). Needed
    because that stage alone can call ask_local_fn twice within a single
    run_research_pipeline() invocation; every other stage behaves exactly
    like make_ask_local (one canned response each, from **overrides)."""
    final_iter = iter(final_responses)
    non_final_overrides = {k: v for k, v in overrides.items() if k != "final"}

    def fn(messages, **kwargs):
        if record is not None:
            record.append((messages, kwargs))
        system = messages[0]["content"]
        if system.startswith("You are the FinalInvestmentSynthesizer"):
            content = next(final_iter)
        else:
            content = _dispatch(system, **non_final_overrides)
        return {"message": {"role": "assistant", "content": content},
                "metrics": {"prompt_tokens": 100, "completion_tokens": 20}, "ok": True}

    return fn


_STAGE_SYSTEM_PREFIXES = {
    "bull": "You are the Bull Researcher",
    "bear": "You are the Bear Researcher",
    "rebuttal": "You are facilitating exactly ONE bounded rebuttal round",
    "research_manager": "You are the Research Manager",
    "risk": "You are the Risk Reviewer",
    "final": "You are the FinalInvestmentSynthesizer",
}


def make_ask_local_with_stage_sequence(sequences, record=None, **overrides):
    """Generalizes make_ask_local_with_final_sequence to ANY stage that has
    a one-repair-attempt mechanism (COR corrective patch, Phase 5-6: bull_
    researcher and bear_researcher now get one too, alongside the
    pre-existing final_investment_synthesizer). `sequences` is
    {"bull": [resp1, resp2], ...} -- each named stage returns its list in
    order across successive calls; any stage NOT in `sequences` behaves
    exactly like make_ask_local (one canned response, from **overrides)."""
    iterators = {stage: iter(responses) for stage, responses in sequences.items()}
    plain_overrides = {k: v for k, v in overrides.items() if k not in sequences}

    def fn(messages, **kwargs):
        if record is not None:
            record.append((messages, kwargs))
        system = messages[0]["content"]
        for stage, prefix in _STAGE_SYSTEM_PREFIXES.items():
            if stage in iterators and system.startswith(prefix):
                content = next(iterators[stage])
                break
        else:
            content = _dispatch(system, **plain_overrides)
        return {"message": {"role": "assistant", "content": content},
                "metrics": {"prompt_tokens": 100, "completion_tokens": 20}, "ok": True}

    return fn


# ---- happy path ----

def test_happy_path_all_six_stages_complete_in_order():
    result = rp.run_research_pipeline(INDEX, make_ask_local())

    names = [c.stage for c in result.checkpoints]
    assert names == ["bull_researcher", "bear_researcher", "rebuttal_round",
                     "research_manager", "risk_reviewer", "final_investment_synthesizer"]
    assert all(c.status == rp.StageStatus.COMPLETED for c in result.checkpoints)
    assert result.available is True


def test_happy_path_token_totals_are_summed_across_all_stages():
    result = rp.run_research_pipeline(INDEX, make_ask_local())
    assert result.total_prompt_tokens == 100 * 6
    assert result.total_completion_tokens == 20 * 6


def test_by_stage_and_output_accessors():
    result = rp.run_research_pipeline(INDEX, make_ask_local())
    assert result.by_stage("bear_researcher").status == rp.StageStatus.COMPLETED
    assert result.output("bear_researcher")["confidence"] == "medium"
    assert result.by_stage("no_such_stage") is None
    assert result.output("no_such_stage") is None


def test_to_dict_shape_is_json_serializable():
    result = rp.run_research_pipeline(INDEX, make_ask_local())
    encoded = json.dumps(result.to_dict())
    decoded = json.loads(encoded)
    assert decoded["available"] is True
    assert decoded["tradingagents_reviewed_commit"] == rp.TRADINGAGENTS_REVIEWED_COMMIT
    assert len(decoded["checkpoints"]) == 6


# ---- cascading fail-closed logic ----

def test_one_researcher_failing_does_not_kill_the_pipeline():
    """Bear alone is enough evidence to reach a verdict; the rebuttal round is
    the only thing that strictly needs BOTH sides."""
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull="not json at all"))

    assert result.by_stage("bull_researcher").status == rp.StageStatus.FAILED
    assert result.by_stage("bear_researcher").status == rp.StageStatus.COMPLETED
    assert result.by_stage("rebuttal_round").status == rp.StageStatus.SKIPPED
    assert "bull_researcher" in result.by_stage("rebuttal_round").error
    assert "bear_researcher" in result.by_stage("rebuttal_round").error
    assert result.by_stage("research_manager").status == rp.StageStatus.COMPLETED
    assert result.by_stage("risk_reviewer").status == rp.StageStatus.COMPLETED
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED
    assert result.available is True


def test_both_researchers_failing_cascades_to_total_unavailability():
    result = rp.run_research_pipeline(
        INDEX, make_ask_local(bull="not json", bear="also not json"))

    assert result.by_stage("bull_researcher").status == rp.StageStatus.FAILED
    assert result.by_stage("bear_researcher").status == rp.StageStatus.FAILED
    assert result.by_stage("rebuttal_round").status == rp.StageStatus.SKIPPED
    assert result.by_stage("research_manager").status == rp.StageStatus.SKIPPED
    assert "bull_researcher" in result.by_stage("research_manager").error
    assert "bear_researcher" in result.by_stage("research_manager").error
    assert result.by_stage("risk_reviewer").status == rp.StageStatus.SKIPPED
    assert "research_manager" in result.by_stage("risk_reviewer").error
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.SKIPPED
    assert result.available is False


def test_research_manager_failure_skips_risk_and_final_but_keeps_researchers():
    bad_rm = json.dumps({"evidence_balance": "NOT_A_VALID_ENUM_VALUE",
                         "supported_bull_points": [], "supported_bear_points": [],
                         "unsupported_points": [], "shared_findings": [], "key_disagreements": [],
                         "assumption_sensitive_conclusions": [], "data_gaps": [],
                         "balanced_assessment": "x", "evidence_cited": [EV1]})
    result = rp.run_research_pipeline(INDEX, make_ask_local(research_manager=bad_rm))

    assert result.by_stage("bull_researcher").status == rp.StageStatus.COMPLETED
    assert result.by_stage("bear_researcher").status == rp.StageStatus.COMPLETED
    assert result.by_stage("rebuttal_round").status == rp.StageStatus.COMPLETED
    assert result.by_stage("research_manager").status == rp.StageStatus.FAILED
    assert result.by_stage("risk_reviewer").status == rp.StageStatus.SKIPPED
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.SKIPPED
    assert result.available is False


def test_risk_reviewer_failure_skips_only_final_synthesizer():
    result = rp.run_research_pipeline(
        INDEX, make_ask_local(risk=json.dumps({"key_risks": []})))  # empty -> min_items violated

    assert result.by_stage("research_manager").status == rp.StageStatus.COMPLETED
    assert result.by_stage("risk_reviewer").status == rp.StageStatus.FAILED
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.SKIPPED
    assert "risk_reviewer" in result.by_stage("final_investment_synthesizer").error
    assert result.available is False


def test_ask_local_fn_raising_fails_that_stage_without_crashing_the_pipeline():
    def exploding(messages, **kwargs):
        raise RuntimeError("simulated network blip")

    result = rp.run_research_pipeline(INDEX, exploding)
    assert result.by_stage("bull_researcher").status == rp.StageStatus.FAILED
    assert "RuntimeError" in result.by_stage("bull_researcher").error
    assert result.available is False


def test_ok_false_response_fails_the_stage():
    def not_ok(messages, **kwargs):
        return {"message": {"content": ""}, "metrics": {}, "ok": False}

    result = rp.run_research_pipeline(INDEX, not_ok)
    assert result.by_stage("bull_researcher").status == rp.StageStatus.FAILED
    assert "local model call failed" in result.by_stage("bull_researcher").error


# ---- fabricated evidence citations are rejected (the "never invent facts" backstop) ----

def test_a_fabricated_evidence_id_fails_that_stage():
    bad_bull = _bull_response(evidence_cited=["fundamental.completely_made_up_metric"])
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull=bad_bull))

    assert result.by_stage("bull_researcher").status == rp.StageStatus.FAILED
    assert "fundamental.completely_made_up_metric" in result.by_stage("bull_researcher").error


def test_missing_required_field_fails_the_stage():
    incomplete = json.dumps({"key_points": [], "confidence": "medium"})  # no 'thesis'
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull=incomplete))
    assert result.by_stage("bull_researcher").status == rp.StageStatus.FAILED
    assert "thesis" in result.by_stage("bull_researcher").error


def test_invalid_confidence_enum_fails_the_stage():
    bad = json.dumps({"thesis": "x", "key_points": [{"point": "p", "evidence_cited": [EV1]}],
                      "confidence": "extremely-confident"})
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull=bad))
    assert result.by_stage("bull_researcher").status == rp.StageStatus.FAILED


def test_final_synthesizer_schema_has_no_verdict_field_at_all():
    """Phase H.3 corrective patch, Problem 1: the schema itself must not have
    a slot for a buy/sell/hold/avoid verdict -- not just reject one if
    supplied. A response using the OLD verdict-shaped schema is invalid
    under the NEW schema (missing the now-required research_stance/
    valuation_view/overall_risk/confidence/rationale fields), so it fails
    closed exactly like any other malformed response, never by accident
    succeeding with trade-advice fields smuggled through.
    """
    old_shaped_final = json.dumps({
        "verdict_not_holding": "BUY", "verdict_holding": "SELL",
        "confidence": "medium", "rationale": "x", "key_factors": ["x"],
        "would_change_mind": ["x"], "evidence_cited": [EV1],
    })
    result = rp.run_research_pipeline(INDEX, make_ask_local(final=old_shaped_final))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.FAILED


def test_invalid_research_stance_enum_fails_the_stage():
    bad_final = json.dumps({
        "research_stance": "MOON_BOUND", "valuation_view": "undervalued",  # invalid enum
        "overall_risk": "moderate", "confidence": 0.5,
        "rationale": [{"statement": "x", "evidence_ids": [EV1]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],
    })
    result = rp.run_research_pipeline(INDEX, make_ask_local(final=bad_final))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.FAILED


def test_confidence_out_of_range_fails_the_stage():
    bad_final = json.dumps({
        "research_stance": "neutral", "valuation_view": "approximately_fair",
        "overall_risk": "moderate", "confidence": 1.5,  # out of [0.0, 1.0] range
        "rationale": [{"statement": "x", "evidence_ids": [EV1]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],
    })
    result = rp.run_research_pipeline(INDEX, make_ask_local(final=bad_final))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.FAILED


def test_extra_unexpected_fields_are_dropped_not_passed_through():
    """Even if a stage's raw JSON smuggles in extra keys (e.g. something that
    looks like a position size or an order), the validator rebuilds a new
    dict from only the known schema fields -- nothing else survives into the
    checkpoint's output."""
    smuggled = _bull_response(extra={"position_size_shares": 100, "order_type": "market"})
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull=smuggled))
    output = result.output("bull_researcher")
    assert "position_size_shares" not in output
    assert "order_type" not in output
    assert set(output) == {"role", "thesis", "claims", "confidence"}


def test_final_investment_synthesizer_output_never_contains_order_or_size_fields():
    result = rp.run_research_pipeline(INDEX, make_ask_local())
    output = result.output("final_investment_synthesizer")
    assert set(output) == {"research_stance", "valuation_view", "overall_risk", "confidence",
                           "recommendation", "primary_reason", "supporting_factors",
                           "limiting_factors", "rationale",
                           "conditions_that_strengthen_the_view",
                           "conditions_that_weaken_the_view", "reassessment_triggers",
                           "risk_reconciliation_reason", "key_uncertainties"}
    for forbidden in ("verdict_not_holding", "verdict_holding", "position_size", "order",
                      "order_type", "quantity", "shares", "price_target", "entry_price",
                      "exit_price", "stop_loss", "target_allocation"):
        assert forbidden not in output


def test_final_investment_synthesizer_enum_values_are_research_language_only():
    """The schema's OWN enum values (not just a scan of free text) contain no
    trade-action words -- BUY/SELL/HOLD/AVOID never appear as valid
    research_stance/valuation_view/overall_risk choices. `_RECOMMENDATION_
    LEVELS` is DELIBERATELY excluded from this check -- that is the one
    enum that is SUPPOSED to contain them (see the dedicated positive-
    coverage test right below)."""
    action_words = {"BUY", "SELL", "HOLD", "AVOID", "ADD"}
    for level in (rp._RESEARCH_STANCE_LEVELS + rp._VALUATION_VIEW_LEVELS + rp._OVERALL_RISK_LEVELS):
        assert level.upper() not in action_words


def test_recommendation_levels_are_the_expected_five_values():
    assert set(rp._RECOMMENDATION_LEVELS) == {"buy", "hold", "sell", "avoid", "insufficient_evidence"}


# ---- recommendation reintroduction (personal-use corrective patch) ----

def test_missing_recommendation_is_retried_and_still_fails_closed():
    """An ordinary schema error (missing field), exactly like a missing
    research_stance. WM corrective patch: these are now RETRIED with the
    error quoted back (they are mechanical, and the model can fix them), but
    a stage that never supplies the field still fails closed."""
    bad_final = json.dumps({
        "research_stance": "neutral", "valuation_view": "approximately_fair",
        "overall_risk": "moderate", "confidence": 0.5,
        "rationale": [{"statement": "x", "evidence_ids": [EV1]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],
    })
    calls = []
    result = rp.run_research_pipeline(INDEX, make_ask_local(record=calls, final=bad_final))
    checkpoint = result.by_stage("final_investment_synthesizer")
    assert checkpoint.status == rp.StageStatus.FAILED
    assert checkpoint.output is None
    final_calls = [c for c in calls if c[0][0]["content"].startswith("You are the FinalInvestmentSynthesizer")]
    assert len(final_calls) == 2  # pinned to 2 attempts for this module


def test_invalid_recommendation_enum_fails_the_stage():
    bad_final = json.dumps({
        "research_stance": "neutral", "valuation_view": "approximately_fair",
        "overall_risk": "moderate", "confidence": 0.5, "recommendation": "STRONG_BUY_NOW",
        "rationale": [{"statement": "x", "evidence_ids": [EV1]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],
    })
    result = rp.run_research_pipeline(INDEX, make_ask_local(final=bad_final))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.FAILED


def test_recommendation_accepts_each_of_the_five_values():
    for value in rp._RECOMMENDATION_LEVELS:
        stance = "insufficient_data" if value == "insufficient_evidence" else "neutral"
        final = json.dumps({
            "research_stance": stance, "valuation_view": "approximately_fair",
            "overall_risk": "moderate", "confidence": 0.5, "recommendation": value,
            "primary_reason": "Stance, valuation and risk are mutually consistent.",
            "supporting_factors": [], "limiting_factors": ["Scenario spread is wide."],
            "rationale": [{"statement": "A plain evidence-grounded observation.",
                          "evidence_ids": [EV1]}],
            "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
            "key_uncertainties": [],
        })
        result = rp.run_research_pipeline(INDEX, make_ask_local(final=final))
        assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED, value
        assert result.output("final_investment_synthesizer")["recommendation"] == value


def test_recommendation_forced_consistent_when_stance_is_insufficient_data():
    """A recommendation of anything but 'insufficient_evidence' is rejected (one
    repair attempt) when research_stance is itself 'insufficient_data' --
    mirrors `_require_valid_stance_when_dcf_invalid`'s existing pattern."""
    inconsistent = json.dumps({
        "research_stance": "insufficient_data", "valuation_view": "insufficient_data",
        "overall_risk": "moderate", "confidence": 0.3, "recommendation": "buy",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "Data is too thin to characterize.", "evidence_ids": [EV1]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": ["Almost nothing is known here."],
    })
    calls = []
    result = rp.run_research_pipeline(INDEX, make_ask_local(record=calls, final=inconsistent))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.FAILED
    assert "after 2 attempts" in result.by_stage("final_investment_synthesizer").error
    final_calls = [c for c in calls if c[0][0]["content"].startswith("You are the FinalInvestmentSynthesizer")]
    assert len(final_calls) == 2  # exactly one repair attempt, never a retry loop


def test_recommendation_repairs_end_to_end_when_inconsistent_with_insufficient_data_stance():
    inconsistent = json.dumps({
        "research_stance": "insufficient_data", "valuation_view": "insufficient_data",
        "overall_risk": "moderate", "confidence": 0.3, "recommendation": "buy",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "Data is too thin to characterize.", "evidence_ids": [EV1]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": ["Almost nothing is known here."],
    })
    fixed = json.dumps({
        "research_stance": "insufficient_data", "valuation_view": "insufficient_data",
        "overall_risk": "moderate", "confidence": 0.3, "recommendation": "insufficient_evidence",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "Data is too thin to characterize.", "evidence_ids": [EV1]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": ["Almost nothing is known here."],
    })
    result = rp.run_research_pipeline(
        INDEX, make_ask_local_with_final_sequence([inconsistent, fixed]))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED
    assert result.output("final_investment_synthesizer")["recommendation"] == "insufficient_evidence"


def test_recommendation_is_excluded_from_the_content_policy_scan_but_nothing_else_is():
    """'recommendation' carries a validated verdict and is kept out of the
    dict `_validate_claim_fidelity` ever sees.

    Since the personal-use verdict patch removed the bare buy/sell/hold/avoid
    patterns from finance/content_policy.py, this exclusion is now
    defense-in-depth rather than strictly load-bearing -- the value would
    survive the scan anyway. It is deliberately kept so the field stays safe
    if any future patch reintroduces a verdict-word pattern (as the original
    Phase H.3 one did). What this test still proves for real: every OTHER
    field remains fully scanned, so a genuinely still-banned directive
    smuggled into 'rationale' fails the stage."""
    clean = json.dumps({
        "research_stance": "negative", "valuation_view": "overvalued",
        "overall_risk": "moderate", "confidence": 0.6, "recommendation": "sell",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "Margins have compressed year over year.",
                      "evidence_ids": [EV1]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],
    })
    result = rp.run_research_pipeline(INDEX, make_ask_local(final=clean))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED
    assert result.output("final_investment_synthesizer")["recommendation"] == "sell"

    smuggled_into_rationale = json.dumps({
        "research_stance": "negative", "valuation_view": "overvalued",
        "overall_risk": "moderate", "confidence": 0.6, "recommendation": "sell",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "Use a 5% position size and a stop-loss at $180.",
                      "evidence_ids": [EV1]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],
    })
    result = rp.run_research_pipeline(INDEX, make_ask_local(final=smuggled_into_rationale))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.FAILED


# ---- content-policy + claim-fidelity backstop (Phase H.3 corrective patch,
# Problems 1 & 5): the FinalInvestmentSynthesizer's one-repair-attempt path,
# and every stage's fail-closed-on-first-violation path ----

def test_final_synthesizer_repair_succeeds_when_the_rewrite_is_clean():
    violating = json.dumps({
        "research_stance": "negative", "valuation_view": "overvalued",
        "overall_risk": "high", "confidence": 0.6, "recommendation": "sell",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "If you do not currently hold a position: AVOID.",
                      "evidence_ids": [EV3]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],
    })
    clean = _final_response()
    calls = []
    result = rp.run_research_pipeline(
        INDEX, make_ask_local_with_final_sequence([violating, clean], record=calls))

    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED
    assert result.output("final_investment_synthesizer")["rationale"][0]["statement"] == (
        "The evidence shows a balanced case with a real valuation gap.")
    final_calls = [c for c in calls if c[0][0]["content"].startswith("You are the FinalInvestmentSynthesizer")]
    assert len(final_calls) == 2  # the original attempt, plus exactly one repair


def test_final_synthesizer_repair_prompt_redacts_rather_than_echoing_the_violation():
    violating = json.dumps({
        "research_stance": "negative", "valuation_view": "overvalued",
        "overall_risk": "high", "confidence": 0.6, "recommendation": "sell",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "Use a 5% position size.", "evidence_ids": [EV3]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],
    })
    calls = []
    rp.run_research_pipeline(
        INDEX, make_ask_local_with_final_sequence([violating, violating], record=calls))
    final_calls = [c for c in calls if c[0][0]["content"].startswith("You are the FinalInvestmentSynthesizer")]
    repair_user_prompt = final_calls[1][0][1]["content"]
    # Phase H.5, Phase 3: this assertion is INVERTED on purpose. The prompt
    # used to echo the violation verbatim, which taught the model the exact
    # vocabulary it then reused -- measured live on WM as 1 violation, then
    # 4, then 3, then a dead stage.
    #
    # This fixture actually trips the RISK-RECONCILIATION guard before it
    # ever reaches claim fidelity (overall_risk 'high' against the risk
    # reviewer's own aggregate), so it exercises the STRUCTURAL branch: a
    # coherence guard names nothing forbidden and its message IS the fix, so
    # it is sent verbatim. Redaction of the vocabulary-bearing branch is
    # covered by test_finance_repair_prompt.py.
    assert "position size" not in repair_user_prompt.lower()
    assert "not internally consistent" in repair_user_prompt
    assert "corrected JSON object" in repair_user_prompt


def test_final_synthesizer_repair_fails_closed_when_the_rewrite_still_violates():
    violating = json.dumps({
        "research_stance": "negative", "valuation_view": "overvalued",
        "overall_risk": "high", "confidence": 0.6, "recommendation": "sell",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "Use a 5% position size.", "evidence_ids": [EV3]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],
    })
    still_violating = json.dumps({
        "research_stance": "negative", "valuation_view": "overvalued",
        "overall_risk": "high", "confidence": 0.6, "recommendation": "sell",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "This has a fortress balance sheet.", "evidence_ids": [EV3]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],
    })
    calls = []
    result = rp.run_research_pipeline(
        INDEX, make_ask_local_with_final_sequence([violating, still_violating], record=calls))

    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.FAILED
    assert "after 2 attempts" in result.by_stage("final_investment_synthesizer").error
    assert result.output("final_investment_synthesizer") is None  # never exposed, from either attempt
    final_calls = [c for c in calls if c[0][0]["content"].startswith("You are the FinalInvestmentSynthesizer")]
    assert len(final_calls) == 2  # exactly one repair attempt, never a retry loop


def test_a_schema_error_is_retried_but_still_fails_closed_when_never_fixed():
    """WM corrective patch: this contract CHANGED, deliberately.

    A schema error used to fail closed on the first attempt, on the reasoning
    that repair was reserved for content-policy violations. But a schema
    error is exactly as mechanical as a policy violation — the model can fix
    an invalid enum given the error text, without being told anything new
    about the company — and refusing to ask cost the entire pipeline over one
    bad field. It is now retried with the error quoted back.

    What has NOT changed: a stage that never produces valid output still
    fails closed. Nothing invalid is ever placed in a COMPLETED checkpoint.
    """
    bad_schema_final = json.dumps({"research_stance": "NOT_A_VALID_ENUM"})
    calls = []
    result = rp.run_research_pipeline(INDEX, make_ask_local(record=calls, final=bad_schema_final))

    checkpoint = result.by_stage("final_investment_synthesizer")
    assert checkpoint.status == rp.StageStatus.FAILED
    assert checkpoint.output is None
    final_calls = [c for c in calls
                   if c[0][0]["content"].startswith("You are the FinalInvestmentSynthesizer")]
    assert len(final_calls) == 2  # pinned to 2 attempts for this module
    # The retry quoted the actual problem back to the model.
    assert "did not satisfy the schema" in final_calls[1][0][1]["content"]


def test_final_synthesizer_repair_also_triggers_for_a_claim_fidelity_violation():
    """Problem 5's unsupported-claim scan uses the SAME
    CONTENT_POLICY_VIOLATION_MARKER as Problem 1's trade-advice scan, so the
    repair path covers both classes of violation without any change to
    _run_final_synthesizer_stage itself."""
    violating = json.dumps({
        "research_stance": "positive", "valuation_view": "undervalued",
        "overall_risk": "low", "confidence": 0.6, "recommendation": "buy",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "Industry-leading ROE confirms this is a fortress balance sheet.",
                      "evidence_ids": [EV3]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],
    })
    clean = _final_response()
    result = rp.run_research_pipeline(INDEX, make_ask_local_with_final_sequence([violating, clean]))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED


def test_final_synthesizer_provenance_mismatch_also_triggers_repair():
    """This fixture's evidence index has no 'provenance.*.provider' entries
    at all (PAYLOAD's data_provenance never sets one) -- so ANY named
    provider is, by definition, one that supplied nothing in this analysis."""
    violating = json.dumps({
        "research_stance": "neutral", "valuation_view": "approximately_fair",
        "overall_risk": "moderate", "confidence": 0.5, "recommendation": "hold",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "Per SEC filings, the valuation gap is real.",
                      "evidence_ids": [EV3]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],
    })
    clean = _final_response()
    result = rp.run_research_pipeline(INDEX, make_ask_local_with_final_sequence([violating, clean]))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED


def test_bull_researcher_claim_fidelity_violation_triggers_one_repair_attempt():
    """COR corrective patch (Phase 5): bull_researcher now gets the SAME
    one-repair-attempt treatment as the final synthesizer -- a claim-
    fidelity violation is not an automatic total loss of this stage's work
    anymore. Using the SAME (still-violating) text for both attempts here
    proves the repair genuinely HAPPENS (two calls) and, since this
    particular text never gets fixed, that it still fails closed afterward."""
    bad_bull = _bull_response(extra={"thesis": "This is a fortress balance sheet."})
    calls = []
    result = rp.run_research_pipeline(INDEX, make_ask_local(record=calls, bull=bad_bull))

    # Phase H.5, Phase 2: an OVERSTATEMENT match no longer destroys the stage.
    # 'fortress' is a superlative -- a real claim phrased too strongly, not a
    # sentence the system had no information to write. The field is
    # quarantined and the pipeline continues.
    checkpoint = result.by_stage("bull_researcher")
    assert checkpoint.status == rp.StageStatus.COMPLETED
    assert checkpoint.output["thesis"] == rp._QUARANTINE_STUB_TEXT
    assert "fortress" not in json.dumps(checkpoint.output).lower()
    assert [q["field_path"] for q in checkpoint.quarantines] == ["thesis"]
    assert checkpoint.quarantines[0]["severity"] == "overstatement"
    assert checkpoint.quarantines[0]["matched_span"].lower() == "fortress"
    # ...and it took ONE call, because there is nothing to repair.
    bull_calls = [c for c in calls if c[0][0]["content"].startswith("You are the Bull Researcher")]
    assert len(bull_calls) == 1


def test_a_quarantined_stage_does_not_block_the_stages_downstream():
    """The point of the phase: one flagged phrase in one field must not cost
    six stages of work."""
    bad_bull = _bull_response(extra={"thesis": "This is a fortress balance sheet."})
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull=bad_bull))
    for stage in ("bull_researcher", "bear_researcher", "rebuttal_round",
                  "research_manager", "risk_reviewer", "final_investment_synthesizer"):
        assert result.by_stage(stage).status == rp.StageStatus.COMPLETED, stage
    assert result.available is True


def test_bear_researcher_causal_overreach_violation_triggers_one_repair_attempt():
    body = json.loads(_bear_response())
    body["claims"][0]["claim"] = "Net cash offers significant downside protection."
    calls = []
    result = rp.run_research_pipeline(
        INDEX, make_ask_local(record=calls, bear=json.dumps(body)))
    assert result.by_stage("bear_researcher").status == rp.StageStatus.FAILED
    assert "after 2 attempts" in result.by_stage("bear_researcher").error
    assert "downside protection" in result.by_stage("bear_researcher").error.lower()
    bear_calls = [c for c in calls if c[0][0]["content"].startswith("You are the Bear Researcher")]
    assert len(bear_calls) == 2


def test_bear_researcher_repair_succeeds_when_the_rewrite_is_clean():
    body = json.loads(_bear_response())
    body["claims"][0]["claim"] = "Net cash offers significant downside protection."
    violating = json.dumps(body)
    clean = _bear_response()
    result = rp.run_research_pipeline(
        INDEX, make_ask_local_with_stage_sequence({"bear": [violating, clean]}))
    assert result.by_stage("bear_researcher").status == rp.StageStatus.COMPLETED


def test_rebuttal_round_causal_overreach_violation_triggers_one_repair_attempt():
    """GE corrective patch: rebuttal_round now gets the SAME one-repair-
    attempt treatment as every other free-text stage -- a live GE run showed
    bull_rebuttal text tripping the causal-overreach scanner in 2 of 5 runs,
    and with no repair path at the time (rebuttal_round was believed exempt
    -- see finance/research_pipeline.py's own corrected docstring), each one
    was a hard, unrecoverable stage failure over a single flagged phrase.
    Using the SAME (still-violating) text for both attempts here proves the
    repair genuinely HAPPENS (two calls), matching the bear_researcher/
    research_manager tests above. (The original live trip was on 'confirms',
    which the later severity-scoping patch removed from the scanner
    entirely; this fixture uses 'downside protection', a causal-overreach
    phrase that is still banned, so the test still exercises the same repair
    path.)"""
    body = json.loads(_rebuttal_response())
    body["bull_rebuttal"]["response"] = "Net cash offers significant downside protection."
    calls = []
    result = rp.run_research_pipeline(
        INDEX, make_ask_local(record=calls, rebuttal=json.dumps(body)))
    # Phase H.5, Phase 2: quarantined, not failed. `bull_rebuttal` is STUBBED
    # rather than dropped -- deleting `response` from inside it would leave a
    # dict the rebuttal schema says must have one, i.e. a structurally invalid
    # output rather than a redacted one.
    checkpoint = result.by_stage("rebuttal_round")
    assert checkpoint.status == rp.StageStatus.COMPLETED
    assert checkpoint.output["bull_rebuttal"]["response"] == rp._QUARANTINE_STUB_TEXT
    assert "downside protection" not in json.dumps(checkpoint.output).lower()
    assert checkpoint.output["bear_rebuttal"]["response"], "the clean side must survive intact"
    assert [q["field_path"] for q in checkpoint.quarantines] == ["bull_rebuttal.response"]
    rb_calls = [c for c in calls if c[0][0]["content"].startswith(
        "You are facilitating exactly ONE bounded rebuttal round")]
    assert len(rb_calls) == 1


def test_rebuttal_round_repair_succeeds_when_the_rewrite_is_clean():
    body = json.loads(_rebuttal_response())
    body["bull_rebuttal"]["response"] = "Net cash offers significant downside protection."
    violating = json.dumps(body)
    clean = _rebuttal_response()
    result = rp.run_research_pipeline(
        INDEX, make_ask_local_with_stage_sequence({"rebuttal": [violating, clean]}))
    assert result.by_stage("rebuttal_round").status == rp.StageStatus.COMPLETED
    # rebuttal_round recovering must not, by itself, block anything downstream.
    assert result.by_stage("research_manager").status == rp.StageStatus.COMPLETED
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED


def test_research_manager_consensus_language_violation_triggers_one_repair_attempt():
    """COR corrective patch (live verification): research_manager now gets
    the SAME one-repair-attempt treatment as bull/bear/final -- a live COR
    run showed research_manager's own 'balanced_assessment' tripping the
    consensus-language scanner with no way to recover, cascade-failing
    risk_reviewer and final_investment_synthesizer even though both
    researchers had completed cleanly. Using the SAME (still-violating) text
    for both attempts here proves the repair genuinely HAPPENS (two calls)."""
    body = json.loads(_research_manager_response())
    body["balanced_assessment"] = "The consensus value points toward more upside."
    calls = []
    result = rp.run_research_pipeline(
        INDEX, make_ask_local(record=calls, research_manager=json.dumps(body)))
    # Phase H.5, Phase 2: quarantined, not failed. This is the WM failure
    # shape -- research_manager is where these trips concentrated, and where
    # a stage failure cost the whole analysis.
    checkpoint = result.by_stage("research_manager")
    assert checkpoint.status == rp.StageStatus.COMPLETED
    assert checkpoint.output["balanced_assessment"] == rp._QUARANTINE_STUB_TEXT
    assert "consensus" not in json.dumps(checkpoint.output).lower()
    assert [q["field_path"] for q in checkpoint.quarantines] == ["balanced_assessment"]
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED
    rm_calls = [c for c in calls if c[0][0]["content"].startswith("You are the Research Manager")]
    assert len(rm_calls) == 1


def test_research_manager_repair_succeeds_when_the_rewrite_is_clean():
    violating = json.loads(_research_manager_response())
    violating["balanced_assessment"] = "The consensus value points toward more upside."
    clean = _research_manager_response()
    result = rp.run_research_pipeline(
        INDEX, make_ask_local_with_stage_sequence({"research_manager": [json.dumps(violating), clean]}))
    assert result.by_stage("research_manager").status == rp.StageStatus.COMPLETED
    assert result.by_stage("risk_reviewer").status == rp.StageStatus.COMPLETED
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED


def test_risk_reviewer_trade_advice_violation_triggers_one_repair_attempt():
    body = json.loads(_risk_reviewer_response())
    body["key_risks"][0]["risk"] = "Set a stop-loss at $180 if the gap widens further."
    calls = []
    result = rp.run_research_pipeline(INDEX, make_ask_local(record=calls, risk=json.dumps(body)))
    assert result.by_stage("risk_reviewer").status == rp.StageStatus.FAILED
    assert "after 2 attempts" in result.by_stage("risk_reviewer").error
    risk_calls = [c for c in calls if c[0][0]["content"].startswith("You are the Risk Reviewer")]
    assert len(risk_calls) == 2


def test_risk_reviewer_repair_succeeds_when_the_rewrite_is_clean():
    body = json.loads(_risk_reviewer_response())
    body["key_risks"][0]["risk"] = "Set a stop-loss at $180 if the gap widens further."
    violating = json.dumps(body)
    clean = _risk_reviewer_response()
    result = rp.run_research_pipeline(
        INDEX, make_ask_local_with_stage_sequence({"risk": [violating, clean]}))
    assert result.by_stage("risk_reviewer").status == rp.StageStatus.COMPLETED
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED


def test_risk_reviewer_prompt_instructs_against_debt_to_equity_in_isolation():
    """COR corrective patch (Phase 9): COR's own live report showed a very
    high debt_to_equity ratio (~5.08) driven mainly by a small equity base,
    not necessarily severe balance-sheet distress. The RiskReviewer prompt
    must say so explicitly and point it at the broader leverage/liquidity
    context finance/metrics.py now computes (net_debt, net_debt_to_fcf,
    debt_to_fcf, operating_cash_flow, interest_coverage) -- there is no
    schema field this instruction could be enforced through mechanically
    (unlike a prohibited word), so the prompt text itself is what this test
    verifies."""
    research_manager_output = json.loads(_research_manager_response())
    system, _user = rp._risk_reviewer_prompt("EVIDENCE", research_manager_output)
    assert "debt_to_equity" in system
    assert "ALONE" in system or "alone" in system
    for term in ("net_debt", "current_ratio", "operating cash flow"):
        assert term in system
    assert "never invent" in system.lower() or "not present in the evidence" in system.lower()


# ---- H.4 corrective patch: negative equity (goal 1, test 5) ----

def test_risk_reviewer_prompt_addresses_not_meaningful_roe_and_debt_to_equity():
    """Test 5: the RiskReviewer's prompt must say explicitly that an absent
    (not_meaningful) ROE/debt-to-equity is not itself evidence of severity,
    and must point it at the same broader leverage metrics as the
    debt_to_equity-alone guidance above."""
    research_manager_output = json.loads(_research_manager_response())
    system, _user = rp._risk_reviewer_prompt("EVIDENCE", research_manager_output)
    assert "not_meaningful" in system
    assert "not itself evidence" in system.lower() or "not evidence of" in system.lower()
    for term in ("total_debt", "net_debt", "debt_to_fcf", "current_ratio", "operating_cash_flow"):
        assert term in system


def test_shared_guardrails_forbid_dcf_terminology_and_not_meaningful_misuse():
    """Every stage shares one guardrail block (goal 2 + goal 1) -- checked
    once here rather than once per stage, since _stage_prompt embeds
    _SHARED_GUARDRAILS identically into all six."""
    system, _user = rp._researcher_prompt("bull", "EVIDENCE")
    for phrase in ("intrinsic value", "authoritative value", "price target",
                  "fair-value target", "consensus value", "not_meaningful"):
        assert phrase in system


# ---- H.4 corrective patch: DCF assumptions are evidence (goal 3, tests 17-18) ----

def _dcf_payload_with_assumptions():
    payload = dict(PAYLOAD)
    payload["dcf"] = dict(PAYLOAD["dcf"])
    payload["dcf"]["calculation_version"] = "fcff_enterprise_v1"
    payload["dcf"]["primary_scenario"] = "base"
    growth_by_scenario = {"bull": 0.13, "base": 0.09, "bear": 0.05}
    payload["dcf"]["scenarios"] = [
        dict(s, assumptions={
            "revenue_growth": [growth_by_scenario[s["scenario"]]] * 5,
            "operating_margin": [0.20] * 5, "wacc": 0.09, "terminal_growth": 0.025,
            "capex_pct_revenue": [0.02] * 5, "depreciation_pct_revenue": [0.01] * 5,
            "working_capital_pct_revenue": [0.0] * 5, "tax_rate": 0.21,
            "assumption_provenance": {
                "revenue_growth": {
                    "value": growth_by_scenario[s["scenario"]],
                    "source_type": "deterministic_calculation", "source_periods": [],
                    "source_evidence_ids": [], "approval_status": "proposed", "units": "ratio",
                    "derivation": f"Derived from reported history for the {s['scenario']!r} scenario.",
                },
            },
        })
        for s in PAYLOAD["dcf"]["scenarios"]
    ]
    return payload


def test_researcher_prompt_evidence_text_includes_dcf_assumption_items():
    """Test 17: dcf.assumption.* items are part of the SAME evidence_text
    block _stage_prompt embeds for every stage -- checked directly against
    the Bull Researcher's rendered prompt (bear/research_manager/risk/final
    all receive the byte-identical block per
    test_every_stage_receives_the_byte_identical_evidence_index_block)."""
    index = build_evidence_index(_dcf_payload_with_assumptions())
    assert "dcf.assumption.revenue_growth.base" in index
    evidence_text = render_evidence_index(index)
    _system, user = rp._researcher_prompt("bull", evidence_text)
    assert "dcf.assumption.revenue_growth.base" in user
    assert "dcf.assumption.revenue_growth.bull" in user


def test_bull_researcher_can_validly_cite_a_dcf_assumption_evidence_id():
    """Companion to the above: the new evidence IDs are not just PRESENT in
    the prompt text, they also pass real citation validation."""
    index = build_evidence_index(_dcf_payload_with_assumptions())
    raw = json.loads(_bull_response())
    raw["claims"][0]["evidence_ids"] = ["dcf.assumption.revenue_growth.base"]
    validated = rp._validate_researcher_output(raw, index, "bull")
    assert validated["claims"][0]["evidence_ids"] == ["dcf.assumption.revenue_growth.base"]


def test_research_manager_prompt_shows_differing_assumptions_across_scenarios():
    """Test 18: the Research Manager can explain WHY bull/base/bear differ
    because the rendered evidence text shows each scenario's OWN
    revenue_growth value and derivation, not merely the resulting modeled
    value."""
    index = build_evidence_index(_dcf_payload_with_assumptions())
    evidence_text = render_evidence_index(index)
    bull_out, bear_out = json.loads(_bull_response()), json.loads(_bear_response())
    _system, user = rp._research_manager_prompt(evidence_text, bull_out, bear_out, None)
    assert "dcf.assumption.revenue_growth.bull: " in user and "0.13" in user
    assert "dcf.assumption.revenue_growth.bear: " in user and "0.05" in user
    assert "scenario=bull" in user and "scenario=bear" in user


def test_clean_happy_path_is_unaffected_by_the_claim_fidelity_backstop():
    """The default canned fixtures (used throughout this file's happy-path
    tests) must never trip the new scan -- a sanity check that the backstop
    is not so aggressive it breaks ordinary, clean research language."""
    result = rp.run_research_pipeline(INDEX, make_ask_local())
    assert result.available is True
    assert all(c.status in (rp.StageStatus.COMPLETED, rp.StageStatus.SKIPPED) for c in result.checkpoints)


# ---- reduced-mode confidence capping and omission disclosure (Phase H.3
# corrective patch, Problem 10) -- PAYLOAD's own "plan" is always empty
# (no omitted datasets), so a SEPARATE reduced-mode fixture is needed to
# exercise any of this. ----

REDUCED_PAYLOAD = dict(PAYLOAD)
REDUCED_PAYLOAD["plan"] = {"omitted_datasets": ["earnings"],
                           "omission_effects": {"earnings": "EPS surprise history is unavailable this run."}}
REDUCED_INDEX = build_evidence_index(REDUCED_PAYLOAD)
EV_OMITTED = "plan.omitted.earnings"
assert EV_OMITTED in REDUCED_INDEX and EV1 in REDUCED_INDEX  # fixture sanity


def test_has_material_omissions_true_only_when_plan_omitted_entries_exist():
    assert rp._has_material_omissions(REDUCED_INDEX) is True
    assert rp._has_material_omissions(INDEX) is False


def test_bull_researcher_high_confidence_is_capped_when_datasets_are_omitted():
    high_conf_bull = _bull_response(extra={"confidence": "high"})
    result = rp.run_research_pipeline(REDUCED_INDEX, make_ask_local(bull=high_conf_bull))
    assert result.by_stage("bull_researcher").status == rp.StageStatus.COMPLETED
    assert result.output("bull_researcher")["confidence"] == "medium"


def test_bull_researcher_confidence_is_not_capped_when_nothing_is_omitted():
    high_conf_bull = _bull_response(extra={"confidence": "high"})
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull=high_conf_bull))
    assert result.by_stage("bull_researcher").status == rp.StageStatus.COMPLETED
    assert result.output("bull_researcher")["confidence"] == "high"


def test_bear_researcher_medium_confidence_is_not_further_capped():
    """'medium' is already at (not above) the default cap -- clamping must
    not touch a value that is already compliant."""
    result = rp.run_research_pipeline(REDUCED_INDEX, make_ask_local())
    assert result.output("bear_researcher")["confidence"] == "medium"


def test_final_synthesizer_confidence_is_capped_when_datasets_are_omitted():
    high_conf_final = json.dumps({
        "research_stance": "positive", "valuation_view": "undervalued",
        "overall_risk": "moderate", "confidence": 0.95, "recommendation": "buy",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "Growth remains strong despite missing earnings data.",
                      "evidence_ids": [EV_OMITTED]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": ["Earnings history is unavailable this run."],
    })
    result = rp.run_research_pipeline(REDUCED_INDEX, make_ask_local(final=high_conf_final))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED
    assert result.output("final_investment_synthesizer")["confidence"] == 0.6


def test_final_synthesizer_confidence_is_not_capped_when_nothing_is_omitted():
    high_conf_final = json.dumps({
        "research_stance": "positive", "valuation_view": "undervalued",
        "overall_risk": "moderate", "confidence": 0.95, "recommendation": "buy",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "Growth remains strong.", "evidence_ids": [EV1]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],
    })
    result = rp.run_research_pipeline(INDEX, make_ask_local(final=high_conf_final))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED
    assert result.output("final_investment_synthesizer")["confidence"] == 0.95


def test_final_synthesizer_requires_omission_disclosure_and_repairs_once():
    no_disclosure_final = json.dumps({
        "research_stance": "neutral", "valuation_view": "approximately_fair",
        "overall_risk": "moderate", "confidence": 0.5, "recommendation": "hold",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "The valuation gap looks modest.", "evidence_ids": [EV3]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],  # no acknowledgement of the omission at all
    })
    disclosed_final = json.dumps({
        "research_stance": "neutral", "valuation_view": "approximately_fair",
        "overall_risk": "moderate", "confidence": 0.5, "recommendation": "hold",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "Earnings history is unavailable, which limits this view.",
                      "evidence_ids": [EV_OMITTED]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": ["Earnings history is unavailable this run."],
    })
    calls = []
    result = rp.run_research_pipeline(
        REDUCED_INDEX,
        make_ask_local_with_final_sequence([no_disclosure_final, disclosed_final], record=calls))

    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED
    final_calls = [c for c in calls if c[0][0]["content"].startswith("You are the FinalInvestmentSynthesizer")]
    assert len(final_calls) == 2  # the missing-disclosure failure triggered exactly one repair


def test_final_synthesizer_omission_disclosure_failure_fails_closed_if_never_fixed():
    no_disclosure_final = json.dumps({
        "research_stance": "neutral", "valuation_view": "approximately_fair",
        "overall_risk": "moderate", "confidence": 0.5, "recommendation": "hold",
        "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "The valuation gap looks modest.", "evidence_ids": [EV3]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],
    })
    calls = []
    result = rp.run_research_pipeline(
        REDUCED_INDEX,
        make_ask_local_with_final_sequence([no_disclosure_final, no_disclosure_final], record=calls))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.FAILED
    assert result.output("final_investment_synthesizer") is None
    final_calls = [c for c in calls if c[0][0]["content"].startswith("You are the FinalInvestmentSynthesizer")]
    assert len(final_calls) == 2  # the omission-disclosure failure still gets exactly one repair attempt


def test_default_fixtures_trigger_the_disclosure_requirement_against_a_reduced_index():
    """The canned default final-synthesizer fixture used throughout this
    file's happy-path tests does not cite plan.omitted.* -- proving the
    requirement is real (not just a comment) rather than incidentally
    satisfied by every response shape."""
    calls = []
    result = rp.run_research_pipeline(
        REDUCED_INDEX,
        make_ask_local_with_final_sequence([_final_response(), _final_response()], record=calls))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.FAILED
    final_calls = [c for c in calls if c[0][0]["content"].startswith("You are the FinalInvestmentSynthesizer")]
    assert len(final_calls) == 2


# ---- malformed-JSON extraction fallbacks ----

def test_json_wrapped_in_a_markdown_fence_still_parses():
    fenced = "```json\n" + _bull_response() + "\n```"
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull=fenced))
    assert result.by_stage("bull_researcher").status == rp.StageStatus.COMPLETED


def test_json_surrounded_by_prose_still_parses_via_brace_fallback():
    wrapped = "Sure, here is my analysis:\n" + _bull_response() + "\nHope that helps!"
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull=wrapped))
    assert result.by_stage("bull_researcher").status == rp.StageStatus.COMPLETED


def test_pure_prose_with_no_json_object_fails_cleanly():
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull="I think this looks bullish overall."))
    assert result.by_stage("bull_researcher").status == rp.StageStatus.FAILED
    assert "JSON" in result.by_stage("bull_researcher").error


# ---- structural guarantees: no tools, budgets threaded through, shared immutable snapshot ----

def test_no_stage_call_ever_receives_a_tools_argument():
    calls = []
    rp.run_research_pipeline(INDEX, make_ask_local(record=calls))
    assert len(calls) == 6
    for _messages, kwargs in calls:
        assert "tools" not in kwargs


_SYSTEM_PREFIX_TO_STAGE = {
    "You are the Bull Researcher": "bull_researcher",
    "You are the Bear Researcher": "bear_researcher",
    "You are facilitating exactly ONE bounded rebuttal round": "rebuttal_round",
    "You are the Research Manager": "research_manager",
    "You are the Risk Reviewer": "risk_reviewer",
    "You are the FinalInvestmentSynthesizer": "final_investment_synthesizer",
}


def _stage_of(messages):
    system = messages[0]["content"]
    for prefix, stage in _SYSTEM_PREFIX_TO_STAGE.items():
        if system.startswith(prefix):
            return stage
    raise AssertionError(f"unrecognized stage system prompt: {system[:60]!r}")


def test_every_stage_call_requests_json_format_and_its_own_bounded_budget():
    """WM corrective patch: the output budget is now PER STAGE.

    research_manager reconciles two full researcher outputs into a ten-field
    schema and measurably needs the most room; rebuttal_round needs a
    fraction of it. A single global ceiling meant the heaviest stage ran
    permanently at the edge of truncation.
    """
    calls = []
    rp.run_research_pipeline(INDEX, make_ask_local(record=calls))
    for messages, kwargs in calls:
        stage = _stage_of(messages)
        assert kwargs.get("response_format") == "json"
        assert kwargs.get("options") == {
            "num_predict": config.research_stage_max_output_tokens(stage)}
        assert kwargs.get("timeout") == config.research_stage_timeout_seconds()


def test_research_manager_gets_the_largest_output_budget():
    """The stage that measurably needed it. Pinning the ORDER rather than the
    numbers, so tuning the base budget does not break this."""
    manager = config.research_stage_max_output_tokens("research_manager")
    assert manager > config.research_stage_max_output_tokens("rebuttal_round")
    assert manager >= config.research_stage_max_output_tokens("bull_researcher")


def test_configured_token_and_timeout_budgets_actually_propagate(monkeypatch):
    monkeypatch.setenv("RESEARCH_STAGE_MAX_OUTPUT_TOKENS", "1000")
    monkeypatch.setenv("RESEARCH_STAGE_TIMEOUT_SECONDS", "17")
    calls = []
    rp.run_research_pipeline(INDEX, make_ask_local(record=calls))
    for messages, kwargs in calls:
        stage = _stage_of(messages)
        assert kwargs["options"] == {
            "num_predict": config.research_stage_max_output_tokens(stage)}
        assert kwargs["timeout"] == 17
    # The base setting genuinely drives the per-stage values.
    assert config.research_stage_max_output_tokens("research_manager") == 1250
    assert config.research_stage_max_output_tokens("rebuttal_round") == 600


def test_a_single_stages_budget_can_be_overridden_directly(monkeypatch):
    monkeypatch.setenv("RESEARCH_STAGE_MAX_OUTPUT_TOKENS", "1000")
    monkeypatch.setenv("RESEARCH_STAGE_MAX_OUTPUT_TOKENS_RESEARCH_MANAGER", "9999")
    assert config.research_stage_max_output_tokens("research_manager") == 9999
    assert config.research_stage_max_output_tokens("bull_researcher") == 1000


def test_every_stage_receives_the_byte_identical_evidence_index_block():
    calls = []
    rp.run_research_pipeline(INDEX, make_ask_local(record=calls))
    # Every call's user message must contain the SAME rendered evidence text
    # as a substring -- proving the identical immutable snapshot claim.
    rendered = render_evidence_index(INDEX)
    assert len(calls) == 6
    for messages, _kwargs in calls:
        assert rendered in messages[1]["content"]


def test_rebuttal_round_context_actually_contains_both_thesis_texts():
    calls = []
    rp.run_research_pipeline(INDEX, make_ask_local(record=calls))
    rebuttal_call = next(m for m, kw in calls if m[0]["content"].startswith(
        "You are facilitating exactly ONE bounded rebuttal round"))
    user_text = rebuttal_call[1]["content"]
    assert "Growth and valuation support upside." in user_text  # bull thesis
    assert "Valuation looks stretched." in user_text  # bear thesis


def test_only_bull_and_bear_run_when_evidence_index_is_empty():
    """No scenarios/evidence available at all -- researchers still run (they
    can note the absence) but nothing downstream is fabricated when there is
    truly nothing to reconcile.

    Both researchers are FAILED here because they are required to cite >=1
    evidence ID and the index is empty -- proving they cannot cite something
    that isn't there even under a degenerate input.
    """
    empty_index = {}
    result = rp.run_research_pipeline(empty_index, make_ask_local())
    assert result.by_stage("bull_researcher").status == rp.StageStatus.FAILED
    assert result.by_stage("bear_researcher").status == rp.StageStatus.FAILED
    assert result.available is False


# ---- architectural guarantees: no network, no deterministic-calculation coupling ----

def test_module_has_no_direct_network_or_process_capability():
    src = inspect.getsource(rp)
    for forbidden in ("import requests", "import urllib", "import socket", "import subprocess"):
        assert forbidden not in src, \
            f"finance/research_pipeline.py must only reach the model through the injected ask_local_fn"


def test_module_never_imports_the_deterministic_calculation_layer():
    src = inspect.getsource(rp)
    assert "finance.dcf" not in src
    assert "finance.metrics" not in src
    assert "finance.normalization" not in src


def test_final_stage_is_named_final_investment_synthesizer_not_portfolio_manager():
    result = rp.run_research_pipeline(INDEX, make_ask_local())
    names = [c.stage for c in result.checkpoints]
    assert "final_investment_synthesizer" in names
    assert not any("portfolio" in n.lower() for n in names)


# ===========================================================================
# TSLA DCF validation patch
# ===========================================================================

INVALID_DCF_PAYLOAD = {
    "symbol": "TSLA",
    "quote": {"price": 328.58, "price_basis": "delayed"},
    "fundamental_metrics": {"revenue_growth_yoy": {"value": 0.15, "formula": "..."}},
    "technical_metrics": {"rsi_14": {"value": 55.0, "formula": "..."}},
    "dcf": {"available": True, "validation_status": "DCF_NEGATIVE_TERMINAL_FCFF",
           "validation_reasons": ["DCF_NEGATIVE_TERMINAL_FCFF: negative terminal-year FCFF in "
                                  "scenario(s) base, bear."]},
    "warnings": [], "data_provenance": {}, "plan": {"omitted_datasets": []},
}
INVALID_DCF_INDEX = build_evidence_index(INVALID_DCF_PAYLOAD)

# Same shape, but with NO fundamentals/technicals at all -- for the "guard
# only fires when other data IS available" test.
NO_OTHER_DATA_INVALID_DCF_PAYLOAD = {
    "symbol": "TSLA", "quote": {"price": 328.58, "price_basis": "delayed"},
    "fundamental_metrics": {}, "technical_metrics": {},
    "dcf": INVALID_DCF_PAYLOAD["dcf"], "warnings": [], "data_provenance": {},
    "plan": {"omitted_datasets": []},
}
NO_OTHER_DATA_INVALID_DCF_INDEX = build_evidence_index(NO_OTHER_DATA_INVALID_DCF_PAYLOAD)

VALID_DCF_INDEX = build_evidence_index({
    "symbol": "TSLA", "quote": {"price": 200.0, "price_basis": "delayed"},
    "fundamental_metrics": {"revenue_growth_yoy": {"value": 0.15, "formula": "..."}},
    "technical_metrics": {}, "dcf": {"available": True, "validation_status": "DCF_VALID"},
    "warnings": [], "data_provenance": {}, "plan": {"omitted_datasets": []},
})

assert "dcf.validation_status" in INVALID_DCF_INDEX  # fixture sanity
assert "dcf.value_per_share.bull" not in INVALID_DCF_INDEX  # fixture sanity


def _risk_raw(evidence_id="quote.price"):
    return {"key_risks": [{"risk": "Some risk.", "severity": "medium",
                          "evidence_cited": [evidence_id]}],
           "data_quality_concerns": [], "evidence_cited": [evidence_id]}


def _final_raw(research_stance, valuation_view="approximately_fair", evidence_id="quote.price",
               recommendation=None):
    if recommendation is None:
        recommendation = "insufficient_evidence" if research_stance == "insufficient_data" else "hold"
    return {"research_stance": research_stance, "valuation_view": valuation_view,
           "overall_risk": "moderate", "confidence": 0.5, "recommendation": recommendation,
           "primary_reason": "Stance, valuation and risk are mutually consistent.",
           "supporting_factors": [], "limiting_factors": ["Scenario spread is wide."],
           "rationale": [{"statement": "A plain evidence-grounded observation.",
                         "evidence_ids": [evidence_id]}],
           "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
           "key_uncertainties": []}


# ---- enum vocabulary (section 10) ----

def test_inconclusive_is_a_valid_research_stance():
    assert "inconclusive" in rp._RESEARCH_STANCE_LEVELS


def test_model_invalid_is_a_valid_valuation_view():
    assert "model_invalid" in rp._VALUATION_VIEW_LEVELS


def test_new_enum_values_are_still_research_language_only():
    action_words = {"BUY", "SELL", "HOLD", "AVOID", "ADD"}
    assert "INCONCLUSIVE" not in action_words
    assert "MODEL_INVALID" not in action_words


def test_dcf_invalid_statuses_match_the_deterministic_dcf_module():
    """Keeps finance/research_pipeline.py's own duplicated string-literal set
    (this module never imports finance.dcf -- see
    test_module_never_imports_the_deterministic_calculation_layer) in
    lockstep with the authoritative one."""
    from finance.dcf import DcfValidationStatus
    assert set(rp._DCF_INVALID_VALIDATION_STATUSES) == set(DcfValidationStatus.INVALID)


# ---- evidence-index helpers (section 10) ----

def test_dcf_validation_status_read_from_index():
    assert rp._dcf_validation_status_from_index(INVALID_DCF_INDEX) == "DCF_NEGATIVE_TERMINAL_FCFF"
    assert rp._dcf_validation_status_from_index(VALID_DCF_INDEX) == "DCF_VALID"
    assert rp._dcf_validation_status_from_index(INDEX) is None  # module fixture sets none at all


def test_dcf_validation_failed_helper():
    assert rp._dcf_validation_failed(INVALID_DCF_INDEX) is True
    assert rp._dcf_validation_failed(VALID_DCF_INDEX) is False
    assert rp._dcf_validation_failed(INDEX) is False  # absent -> treated as not-failed


# ---- Risk Reviewer: deterministic model_risk (section 10) ----

def test_risk_reviewer_sets_model_risk_high_when_dcf_invalid():
    validated = rp._validate_risk_reviewer_output(_risk_raw(), INVALID_DCF_INDEX)
    assert validated["model_risk"] == "high"


def test_risk_reviewer_sets_model_risk_not_applicable_when_dcf_valid():
    validated = rp._validate_risk_reviewer_output(_risk_raw(), VALID_DCF_INDEX)
    assert validated["model_risk"] == "not_applicable"


def test_risk_reviewer_model_risk_is_deterministic_regardless_of_model_text():
    """The model's own free text never controls this field -- set purely
    from the evidence index, exactly like `_cap_confidence_for_omissions`."""
    raw_saying_nothing_about_risk = _risk_raw()
    raw_saying_nothing_about_risk["key_risks"][0]["risk"] = "Unrelated commentary about margins."
    validated = rp._validate_risk_reviewer_output(raw_saying_nothing_about_risk, INVALID_DCF_INDEX)
    assert validated["model_risk"] == "high"


def test_risk_reviewer_end_to_end_reports_high_model_risk_when_dcf_invalid():
    result = rp.run_research_pipeline(
        INVALID_DCF_INDEX,
        make_ask_local(risk=json.dumps(_risk_raw()),
                      bull=_bull_response(evidence_cited=("fundamental.revenue_growth_yoy",)),
                      bear=_bear_response(evidence_cited=("quote.price",)),
                      research_manager=json.dumps({
                          "evidence_balance": "mixed", "supported_bull_points": [],
                          "supported_bear_points": [], "unsupported_points": [],
                          "shared_findings": [], "key_disagreements": [],
                          "assumption_sensitive_conclusions": [], "data_gaps": [],
                          "balanced_assessment": "Mixed picture.",
                          "evidence_cited": ["quote.price"]}),
                      final=json.dumps(_final_raw("inconclusive", "model_invalid"))))
    risk_output = result.output("risk_reviewer")
    assert risk_output is not None
    assert risk_output["model_risk"] == "high"


# ---- research_manager: DCF-terminology exemption when DCF invalid (HOOD corrective patch) ----

def test_research_manager_dcf_terminology_exempt_when_dcf_invalid():
    """HOOD corrective patch: `_validate_claim_fidelity` now passes
    `dcf_invalid=_dcf_validation_failed(index)` through to finance/
    claim_validation.py's scanner -- a non-disclaimed 'intrinsic value' claim
    (which would otherwise be a genuine, correctly-rejected violation) must
    validate CLEANLY when the DCF failed (no 'dcf.value_per_share.*'
    evidence exists in that case, so no cited claim could be genuinely
    DCF-based either way), and must still be REJECTED when the DCF is valid
    (a genuine DCF-grounded claim IS possible there, so the ban stays in
    force)."""
    raw = {
        "evidence_balance": "mixed", "supported_bull_points": [], "supported_bear_points": [],
        "unsupported_points": [], "shared_findings": [], "key_disagreements": [],
        "assumption_sensitive_conclusions": [], "data_gaps": [],
        "balanced_assessment": "The stock trades below its intrinsic value based on the model.",
        "evidence_cited": ["quote.price"],
    }
    validated = rp._validate_research_manager_output(raw, INVALID_DCF_INDEX)
    assert validated["balanced_assessment"] == raw["balanced_assessment"]

    # Phase H.5, Phase 2: when the DCF IS valid the ban stays in force, but
    # the consequence changed -- the field is QUARANTINED rather than the
    # stage destroyed. `balanced_assessment` is load-bearing downstream (it
    # feeds the risk_reviewer and final_investment_synthesizer prompts), so
    # its policy is STUB rather than DROP.
    quarantined = rp._validate_research_manager_output(raw, VALID_DCF_INDEX)
    records = quarantined.pop(rp._QUARANTINE_KEY)
    assert quarantined["balanced_assessment"] == rp._QUARANTINE_STUB_TEXT
    assert "intrinsic value" not in quarantined["balanced_assessment"].lower()
    assert [r["field_path"] for r in records] == ["balanced_assessment"]
    assert records[0]["severity"] == "overstatement"
    assert records[0]["policy"] == rp.QUARANTINE_STUB
    assert records[0]["rule_id"].startswith("CV-5")  # DCF-terminology group


# ---- FinalInvestmentSynthesizer: deterministic valuation_view + stance guard (section 10) ----

def test_final_synthesizer_forces_valuation_view_to_model_invalid():
    validated = rp._validate_final_synthesizer_output(
        _final_raw("cautious", "overvalued"), INVALID_DCF_INDEX)
    assert validated["valuation_view"] == "model_invalid"


def test_final_synthesizer_valuation_view_unaffected_when_dcf_valid():
    validated = rp._validate_final_synthesizer_output(
        _final_raw("neutral", "approximately_fair"), VALID_DCF_INDEX)
    assert validated["valuation_view"] == "approximately_fair"


def test_final_synthesizer_rejects_insufficient_data_when_other_evidence_exists():
    """'insufficient_data' is reserved for genuinely thin underlying data --
    when fundamentals/technicals ARE in the evidence index, a failed
    valuation MODEL is a different problem and must be rejected (one repair
    attempt), never silently accepted."""
    with pytest.raises(rp._Invalid) as excinfo:
        rp._validate_final_synthesizer_output(_final_raw("insufficient_data"), INVALID_DCF_INDEX)
    assert rp.CONTENT_POLICY_VIOLATION_MARKER in excinfo.value.message


def test_final_synthesizer_accepts_insufficient_data_when_truly_no_other_data():
    """The guard is conditional -- when NEITHER fundamentals NOR technicals
    are present, 'insufficient_data' is legitimate and must NOT be rejected."""
    validated = rp._validate_final_synthesizer_output(
        _final_raw("insufficient_data"), NO_OTHER_DATA_INVALID_DCF_INDEX)
    assert validated["research_stance"] == "insufficient_data"
    assert validated["valuation_view"] == "model_invalid"  # still forced -- DCF is still invalid


def test_final_synthesizer_accepts_inconclusive_when_dcf_invalid_and_other_data_present():
    validated = rp._validate_final_synthesizer_output(
        _final_raw("inconclusive"), INVALID_DCF_INDEX)
    assert validated["research_stance"] == "inconclusive"
    assert validated["valuation_view"] == "model_invalid"


def test_final_synthesizer_stance_guard_does_not_fire_when_dcf_is_valid():
    """No guard, no override at all when the DCF passed validation --
    'insufficient_data' is the model's own free choice in that case."""
    validated = rp._validate_final_synthesizer_output(
        _final_raw("insufficient_data"), VALID_DCF_INDEX)
    assert validated["research_stance"] == "insufficient_data"
    assert validated["valuation_view"] == "approximately_fair"


def test_final_synthesizer_repairs_an_insufficient_data_stance_end_to_end():
    """The one-repair-attempt wrapper applies to this guard exactly like
    every other content-policy/claim-fidelity violation: reject once, retry
    once, then fail closed if still wrong."""
    bad_then_good = [
        json.dumps(_final_raw("insufficient_data")),
        json.dumps(_final_raw("inconclusive")),
    ]
    result = rp.run_research_pipeline(
        INVALID_DCF_INDEX,
        make_ask_local_with_final_sequence(
            bad_then_good,
            bull=_bull_response(evidence_cited=("fundamental.revenue_growth_yoy",)),
            bear=_bear_response(evidence_cited=("quote.price",)),
            research_manager=json.dumps({
                "evidence_balance": "mixed", "supported_bull_points": [],
                "supported_bear_points": [], "unsupported_points": [], "shared_findings": [],
                "key_disagreements": [], "assumption_sensitive_conclusions": [], "data_gaps": [],
                "balanced_assessment": "Mixed picture.", "evidence_cited": ["quote.price"]}),
            risk=json.dumps(_risk_raw())))
    final_checkpoint = result.by_stage("final_investment_synthesizer")
    assert final_checkpoint.status == rp.StageStatus.COMPLETED
    assert final_checkpoint.output["research_stance"] == "inconclusive"
    assert final_checkpoint.output["valuation_view"] == "model_invalid"


# ---- schema text mentions the new vocabulary (documentation-as-contract) ----

def test_final_synthesizer_schema_text_mentions_model_invalid_and_inconclusive():
    system, _user = rp._final_synthesizer_prompt("EVIDENCE", {
        "evidence_balance": "mixed", "balanced_assessment": "x"}, {"key_risks": [],
                                                                    "data_quality_concerns": []})
    assert "model_invalid" in system
    assert "inconclusive" in system
    assert "dcf.validation_status" in system


def test_shared_guardrails_explain_the_dcf_invalid_evidence_gap():
    assert "dcf.validation_status" in rp._SHARED_GUARDRAILS
    assert "DCF_VALID" in rp._SHARED_GUARDRAILS


# ---- MLI corrective patch: recommendation consistency (spec 3/6/7/14) ----

def _final_with(**overrides):
    body = {
        "research_stance": "cautiously_positive", "valuation_view": "undervalued",
        "overall_risk": "moderate", "confidence": 0.45, "recommendation": "hold",
        "primary_reason": "Modeled upside exists but confidence is limited.",
        "supporting_factors": [], "limiting_factors": ["Scenario spread is wide."],
        "rationale": [{"statement": "A plain evidence-grounded observation.",
                      "evidence_ids": [EV1]}],
        "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],
    }
    body.update(overrides)
    return json.dumps(body)


def test_hold_on_a_favourable_valuation_requires_a_limiting_factor():
    """Spec 3: the live MLI report paired 'undervalued' with HOLD and never
    said why. Either a hidden limitation the reader deserved, or an
    unsupported downgrade -- both fixed by requiring it be named."""
    result = rp.run_research_pipeline(
        INDEX, make_ask_local(final=_final_with(limiting_factors=[])))
    checkpoint = result.by_stage("final_investment_synthesizer")
    assert checkpoint.status == rp.StageStatus.FAILED
    assert "limiting_factors" in checkpoint.error


def test_hold_on_a_favourable_valuation_is_accepted_with_a_limiting_factor():
    result = rp.run_research_pipeline(INDEX, make_ask_local(final=_final_with()))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED
    assert result.output("final_investment_synthesizer")["recommendation"] == "hold"


def test_buy_on_a_favourable_valuation_needs_no_limiting_factor():
    """The guard asks 'why NOT buy' -- it must not fire on buy itself."""
    result = rp.run_research_pipeline(
        INDEX, make_ask_local(final=_final_with(recommendation="buy", limiting_factors=[])))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED


def test_buy_is_rejected_when_research_readiness_is_not_ready():
    """Spec 14: a NOT_READY analysis has a failed DCF or an unresolved core
    reconciliation conflict -- its valuation cannot carry an action call."""
    result = rp.run_research_pipeline(
        INDEX, make_ask_local(final=_final_with(recommendation="buy", limiting_factors=[])),
        readiness_status="NOT_READY")
    checkpoint = result.by_stage("final_investment_synthesizer")
    assert checkpoint.status == rp.StageStatus.FAILED
    assert "NOT_READY" in checkpoint.error


def test_non_insufficient_recommendation_under_not_ready_needs_reason_and_low_confidence():
    """Spec 12: readiness constrains how CONFIDENTLY a call is held and
    demands the limitation be named -- it never picks the recommendation.
    'insufficient_evidence' is the normal answer and needs no justification;
    anything else is permitted with a limiting factor at low confidence."""
    ok = _final_with(recommendation="hold", confidence=0.30,
                     limiting_factors=["DCF failed validation."])
    result = rp.run_research_pipeline(
        INDEX, make_ask_local(final=ok), readiness_status="NOT_READY")
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED

    too_confident = _final_with(recommendation="hold", confidence=0.80,
                                limiting_factors=["DCF failed validation."])
    result = rp.run_research_pipeline(
        INDEX, make_ask_local(final=too_confident), readiness_status="NOT_READY")
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.FAILED

    unexplained = _final_with(recommendation="hold", confidence=0.30, limiting_factors=[])
    result = rp.run_research_pipeline(
        INDEX, make_ask_local(final=unexplained), readiness_status="NOT_READY")
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.FAILED


def test_insufficient_evidence_under_not_ready_needs_no_justification():
    result = rp.run_research_pipeline(
        INDEX, make_ask_local(final=_final_with(
            research_stance="insufficient_data", valuation_view="insufficient_data",
            overall_risk="moderate", recommendation="insufficient_evidence",
            confidence=0.2, limiting_factors=[])),
        readiness_status="NOT_READY")
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED


def test_action_recommendation_is_unaffected_when_readiness_is_limited():
    result = rp.run_research_pipeline(
        INDEX, make_ask_local(final=_final_with(recommendation="buy", limiting_factors=[])),
        readiness_status="LIMITED")
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED


# ---- risk aggregation (spec 6) ----

def test_risk_aggregation_never_averages_high_away():
    """The live MLI Risk section listed HIGH/HIGH/MEDIUM/MEDIUM and the
    synthesis reported 'moderate'."""
    assert rp.aggregate_risk_level(["high", "high", "medium", "medium"]) == "high"
    assert rp.aggregate_risk_level(["high", "medium", "medium"]) == "high"
    assert rp.aggregate_risk_level(["high"]) == "high"


def test_risk_aggregation_of_lower_severities():
    assert rp.aggregate_risk_level(["medium", "medium", "low"]) == "moderate"
    assert rp.aggregate_risk_level(["low", "low"]) == "low"
    assert rp.aggregate_risk_level(["very_high", "low"]) == "very_high"
    assert rp.aggregate_risk_level([]) is None


def test_risk_reviewer_publishes_its_aggregated_risk():
    raw = {"key_risks": [
        {"risk": "A.", "severity": "high", "evidence_cited": [EV1]},
        {"risk": "B.", "severity": "high", "evidence_cited": [EV1]},
        {"risk": "C.", "severity": "medium", "evidence_cited": [EV1]},
    ], "data_quality_concerns": [], "evidence_cited": [EV1]}
    assert rp._validate_risk_reviewer_output(raw, INDEX)["aggregated_risk"] == "high"


_TWO_HIGH_RISKS = json.dumps({"key_risks": [
    {"risk": "A.", "severity": "high", "evidence_cited": [EV3]},
    {"risk": "B.", "severity": "high", "evidence_cited": [EV3]},
], "data_quality_concerns": [], "evidence_cited": [EV3]})


def test_final_risk_cannot_silently_contradict_the_risk_reviewer():
    """Spec 13: differing is allowed, silently differing is not."""
    result = rp.run_research_pipeline(
        INDEX, make_ask_local(risk=_TWO_HIGH_RISKS, final=_final_with(overall_risk="low")))
    checkpoint = result.by_stage("final_investment_synthesizer")
    assert checkpoint.status == rp.StageStatus.FAILED
    assert "risk_reconciliation_reason" in checkpoint.error


def test_final_risk_may_differ_with_an_explicit_reconciliation_reason():
    """The legitimate case the previous hard override made impossible: two
    HIGH flags tracing to ONE underlying issue rather than two independent
    risks."""
    reconciled = _final_with(
        overall_risk="moderate",
        risk_reconciliation_reason=(
            "Both HIGH flags describe the same DCF sensitivity issue rather than two "
            "independent risks."))
    result = rp.run_research_pipeline(
        INDEX, make_ask_local(risk=_TWO_HIGH_RISKS, final=reconciled))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED
    assert result.output("final_investment_synthesizer")["overall_risk"] == "moderate"


def test_matching_the_risk_reviewer_needs_no_reconciliation_reason():
    result = rp.run_research_pipeline(
        INDEX, make_ask_local(risk=_TWO_HIGH_RISKS, final=_final_with(overall_risk="high")))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED


# ---- LLM-owns-the-recommendation patch: condition direction (spec 8/9) ----

def test_condition_direction_classification():
    c = rp.classify_condition_direction
    assert c("Operating margins remain above 15% while revenue growth accelerates") == "FAVORABLE"
    assert c("Free cash flow materially deteriorates while debt increases") == "UNFAVORABLE"
    # Information changes have NO known direction, however positive they sound.
    assert c("A company-specific WACC becomes available") == "BIDIRECTIONAL"
    assert c("The share-count conflict is reconciled") == "BIDIRECTIONAL"
    assert c("A restated filing changes historical figures") == "BIDIRECTIONAL"
    assert c("The DCF is updated with better assumptions") == "BIDIRECTIONAL"
    # No clear signal -> accepted rather than rejected.
    assert c("The company holds its annual meeting") == "UNCLASSIFIED"


def test_bidirectional_wins_over_favourable_sounding_wording():
    """Spec 9 rules out bare keyword matching: "improves" inside an
    information-change clause must NOT make it an upgrade condition."""
    assert rp.classify_condition_direction(
        "A validated company-specific WACC becomes available and improves the model") == "BIDIRECTIONAL"


def test_misdirected_conditions_are_moved_not_rejected():
    """Direction is a HEURISTIC judgment, so it corrects rather than fails.
    A live AMZN run filed 'reported CapEx history becomes available' as an
    upgrade condition -- defensible, but BIDIRECTIONAL by this spec -- and
    the reject-based first design killed the whole stage over it. Routing
    reaches the same required outcome (each list holds conditions of the
    right direction) with no failure mode."""
    misfiled = _final_with(
        valuation_view="overvalued", recommendation="avoid",
        conditions_that_weaken_the_view=["A company-specific WACC becomes available"],
        conditions_that_strengthen_the_view=["Reported CapEx history becomes available"],
        reassessment_triggers=[])
    result = rp.run_research_pipeline(INDEX, make_ask_local(final=misfiled))
    checkpoint = result.by_stage("final_investment_synthesizer")
    assert checkpoint.status == rp.StageStatus.COMPLETED
    output = result.output("final_investment_synthesizer")
    # Both were information changes -> both land in reassessment_triggers,
    # and neither remains in a directional bucket.
    assert output["conditions_that_weaken_the_view"] == []
    assert output["conditions_that_strengthen_the_view"] == []
    assert len(output["reassessment_triggers"]) == 2


def test_routing_respects_the_two_item_cap():
    crowded = _final_with(
        reassessment_triggers=["A company-specific WACC becomes available",
                               "The share-count conflict is reconciled"],
        conditions_that_strengthen_the_view=["A restated filing is disclosed"])
    result = rp.run_research_pipeline(INDEX, make_ask_local(final=crowded))
    output = result.output("final_investment_synthesizer")
    assert len(output["reassessment_triggers"]) == 2  # capped, not overflowed


def test_correctly_filed_conditions_are_left_alone():
    body = _final_with(
        conditions_that_strengthen_the_view=["Operating margins expand further"],
        conditions_that_weaken_the_view=["Free cash flow deteriorates"],
        reassessment_triggers=["A company-specific WACC becomes available"])
    result = rp.run_research_pipeline(INDEX, make_ask_local(final=body))
    output = result.output("final_investment_synthesizer")
    assert output["conditions_that_strengthen_the_view"] == ["Operating margins expand further"]
    assert output["conditions_that_weaken_the_view"] == ["Free cash flow deteriorates"]
    assert output["reassessment_triggers"] == ["A company-specific WACC becomes available"]


def test_correctly_directed_conditions_are_accepted():
    good = _final_with(
        conditions_that_strengthen_the_view=["Operating margins expand while growth accelerates"],
        conditions_that_weaken_the_view=["Free cash flow deteriorates while debt increases"],
        reassessment_triggers=["A company-specific WACC becomes available"])
    result = rp.run_research_pipeline(INDEX, make_ask_local(final=good))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED


def test_unclassifiable_conditions_are_not_rejected():
    """The checker guards against clear contradictions, not style."""
    result = rp.run_research_pipeline(INDEX, make_ask_local(final=_final_with(
        conditions_that_strengthen_the_view=["The Q3 filing lands on schedule"])))
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED


# ---- recommendation freedom (spec 17 items 1-8) ----

def test_no_deterministic_valuation_to_recommendation_mapping_exists():
    """Spec 3/7: every recommendation must be reachable from BOTH an
    undervalued and an overvalued view -- proving no hard-coded matrix."""
    for valuation, recommendation in [
        ("undervalued", "buy"), ("undervalued", "hold"), ("undervalued", "avoid"),
        ("overvalued", "hold"), ("overvalued", "sell"), ("overvalued", "buy"),
    ]:
        body = _final_with(
            valuation_view=valuation, recommendation=recommendation,
            limiting_factors=["Scenario spread is wide and confidence is limited."])
        result = rp.run_research_pipeline(INDEX, make_ask_local(final=body))
        assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED, \
            f"{valuation} + {recommendation} should be permitted"
        assert result.output("final_investment_synthesizer")["recommendation"] == recommendation
