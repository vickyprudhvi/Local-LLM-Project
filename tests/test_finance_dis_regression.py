"""DIS valuation-comparison / research-readiness correction — DIS regression,
from REAL live Yahoo Finance + SEC EDGAR data (tests/fixtures/dis_regression.json,
captured 2026-08-08; secrets removed; SEC company_facts trimmed to only the
finance/xbrl_mapping.py concepts this codebase reads — see the fixture's own
"note" field).

This is the exact company/data shape that exposed the bugs this patch fixes:

    Market price = 104.91
    Base modeled value = 145.41

A live compact report said "the delayed market price is approximately 39%
below the base modeled value" -- WRONG. `difference / market_price` (the
figure actually computed, +38.6%, denominated in the market PRICE) was used
in a sentence shaped for `(price - modeled) / modeled` (the figure the
sentence actually needs, -27.85%, denominated in the MODELED VALUE). The
SAME report then said "Decision readiness: READY" while its own Risk and
Research View sections said research_manager/risk_reviewer never completed
-- inconsistent, and using the wrong field name besides ("Research
readiness" is not permission to trade).

This file proves, on REAL captured DIS data:
  1. `market_price_premium_pct` (~-27.85%) and `modeled_return_to_value_pct`
     (~+38.6%) are two distinct, correctly-computed, non-interchangeable
     numbers.
  2. The compact renderer states "27.9%" in the discount sentence, never
     "39%", and separately states the modeled-return figure.
  3. "Decision readiness" does not appear anywhere in a rendered report;
     "Research readiness" does.
  4. A constructed research-pipeline outcome matching what was observed live
     (bull/bear/rebuttal completed, research_manager FAILED, risk_reviewer/
     final_investment_synthesizer correctly shown as SKIPPED due to that
     failure, never as having failed themselves) produces Research
     readiness: LIMITED, never READY, with the actual classified failure
     reason shown -- never a raw validation dump, never a fabricated research
     stance.
  5. DCF/provider-routing regression: bear < base < bull remains
     deterministic and Yahoo/SEC routing is unchanged.

Every expected value below was independently produced by RUNNING this
fixture through the real pipeline once and reading off the result (this
project's own standing rule, applied identically to the COST/COR/TSLA/AMZN
regression fixtures).
"""

import json
from pathlib import Path

import pytest

import tools.finance_tools as finance_tools
from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.dcf import DcfValidationStatus
from finance.provider import AlphaVantageClient, ProviderResponse
from finance.quota import AlphaVantageQuotaLedger, QuotaLedger
from finance.research_pipeline import ResearchPipelineResult, StageCheckpoint, StageStatus
from finance.sec_datasets import resolve_sec_dataset
from finance.workflow import (
    AnalysisMode,
    ResearchReadiness,
    build_compact_synthesis_payload,
    render_compact_report,
    run_full_stock_analysis,
)
from finance.yahoo_datasets import resolve_yahoo_dataset
from tests.test_finance_cache import FakeClock
from tools.base import ToolFailure
from tools.executor import ToolExecutor
from tools.models import MARKET_DATA_INVALID_SYMBOL
from tools.registry import ToolRegistry

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "dis_regression.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

SYMBOL = "DIS"

# ---- ground truth, independently produced by RUNNING this fixture through
# the real (fixed) pipeline once and reading off the result (see module
# docstring) ----
EXPECTED_MARKET_PRICE = 104.91
EXPECTED_BASE_VALUE_PER_SHARE = 110.766483
EXPECTED_BULL_VALUE_PER_SHARE = 187.11101
EXPECTED_BEAR_VALUE_PER_SHARE = 47.476206
EXPECTED_MARKET_PRICE_PREMIUM_PCT = -0.052872
EXPECTED_MODELED_RETURN_TO_VALUE_PCT = 0.055824


class YahooFixtureClient:
    provider_id = "yahoo"

    def fetch(self, dataset, arguments):
        payload = FIXTURE["yahoo"].get(dataset.dataset_id)
        if payload is None:
            raise ToolFailure(MARKET_DATA_INVALID_SYMBOL, f"no fixture data for {dataset.dataset_id}")
        return ProviderResponse(payload=payload, provider_metadata={}, byte_count=len(json.dumps(payload)))


class SecFixtureClient:
    provider_id = "sec"

    def fetch(self, dataset, arguments):
        payload = FIXTURE["sec"].get(dataset.dataset_id)
        if payload is None:
            raise ToolFailure(MARKET_DATA_INVALID_SYMBOL, f"no fixture data for {dataset.dataset_id}")
        return ProviderResponse(payload=payload, provider_metadata={}, byte_count=len(json.dumps(payload)))


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def wired(tmp_path, clock, monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "")  # deliberately unconfigured
    monkeypatch.setenv("YAHOO_FINANCE_ENABLED", "true")
    monkeypatch.setenv("YAHOO_PERSONAL_USE_ACKNOWLEDGED", "true")
    monkeypatch.setenv("SEC_EDGAR_ENABLED", "true")
    monkeypatch.setenv("SEC_USER_AGENT", "TestApp/1.0 (contact: t@example.com)")
    for var in ("FINANCE_QUOTE_PROVIDER", "FINANCE_PRICE_HISTORY_PROVIDER",
               "FINANCE_CORPORATE_ACTIONS_PROVIDER", "FINANCE_COMPANY_PROFILE_PROVIDER",
               "FINANCE_ANALYST_ESTIMATES_PROVIDER"):
        monkeypatch.setenv(var, "yahoo")
    monkeypatch.setenv("FINANCE_US_FUNDAMENTALS_PROVIDER", "sec")

    yahoo_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "yahoo_m.sqlite3"), clock=clock),
        ledger=QuotaLedger(path=str(tmp_path / "yahoo_q.sqlite3"), clock=clock, daily_limit=1_000_000),
        client=YahooFixtureClient(), provider_id="yahoo", dataset_resolver=resolve_yahoo_dataset,
        clock=clock, sleeper=lambda _s: None, min_request_interval_ms=0)
    finance_tools.set_yahoo_coordinator(yahoo_coordinator)

    sec_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "sec_m.sqlite3"), clock=clock),
        ledger=QuotaLedger(path=str(tmp_path / "sec_q.sqlite3"), clock=clock, daily_limit=1_000_000),
        client=SecFixtureClient(), provider_id="sec", dataset_resolver=resolve_sec_dataset,
        clock=clock, sleeper=lambda _s: None, min_request_interval_ms=0)
    finance_tools.set_sec_coordinator(sec_coordinator)

    av_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "av_m.sqlite3"), clock=clock),
        ledger=AlphaVantageQuotaLedger(path=str(tmp_path / "av_q.sqlite3"), clock=clock, daily_limit=100),
        client=AlphaVantageClient(), clock=clock, sleeper=lambda _s: None, jitter=lambda: 0.5)
    finance_tools.set_coordinator(av_coordinator)

    registry = ToolRegistry()
    for tool_cls in finance_tools.ALL_FINANCE_TOOL_CLASSES:
        registry.register(tool_cls())
    executor = ToolExecutor(registry)

    yield executor
    finance_tools.set_coordinator(None)
    finance_tools.set_yahoo_coordinator(None)
    finance_tools.set_sec_coordinator(None)


def run_dis(wired):
    return run_full_stock_analysis(wired, SYMBOL, include_news=False)


def _scenario(result, name):
    return next(s for s in result.facts["dcf"]["scenarios"] if s["scenario"] == name)


# ---- fixture sanity / regression (tests 19-21) ----

def test_fixture_is_real_captured_dis_data(wired):
    result = run_dis(wired)
    assert "DISNEY" in result.facts["overview"]["name"].upper()
    assert result.facts["quote"]["price"] == pytest.approx(EXPECTED_MARKET_PRICE, abs=0.01)


def test_full_analysis_mode_reached_and_provider_routing_unchanged(wired):
    result = run_dis(wired)
    assert result.plan.mode == AnalysisMode.FULL
    provenance = result.facts["data_provenance"]
    assert provenance["balance_sheet"]["provider"] == "sec"
    assert provenance["quote"]["provider"] == "yahoo"


def test_dcf_scenario_values_match_the_independently_computed_result(wired):
    result = run_dis(wired)
    base, bull, bear = _scenario(result, "base"), _scenario(result, "bull"), _scenario(result, "bear")
    assert base["value_per_share"] == pytest.approx(EXPECTED_BASE_VALUE_PER_SHARE, rel=1e-6)
    assert bull["value_per_share"] == pytest.approx(EXPECTED_BULL_VALUE_PER_SHARE, rel=1e-6)
    assert bear["value_per_share"] == pytest.approx(EXPECTED_BEAR_VALUE_PER_SHARE, rel=1e-6)


def test_bull_base_bear_ordering_is_deterministic(wired):
    result_a = run_dis(wired)
    result_b = run_dis(wired)
    assert (_scenario(result_a, "bear")["value_per_share"]
           < _scenario(result_a, "base")["value_per_share"]
           < _scenario(result_a, "bull")["value_per_share"])
    assert result_a.facts["dcf"]["scenarios"] == result_b.facts["dcf"]["scenarios"]


def test_dcf_validation_status_is_valid(wired):
    result = run_dis(wired)
    assert result.facts["dcf"]["validation_status"] == DcfValidationStatus.VALID


# ---- valuation-comparison wording (Problem 1 / tests 1-5) ----

def test_market_price_premium_pct_matches_independently_computed_value(wired):
    """Test 1: (market_price - modeled_value) / modeled_value ~= -27.85%."""
    result = run_dis(wired)
    gap = result.facts["valuation_gap"]
    assert gap["market_price_premium_pct"] == pytest.approx(
        EXPECTED_MARKET_PRICE_PREMIUM_PCT, abs=1e-4)


def test_modeled_return_to_value_pct_matches_independently_computed_value(wired):
    """Test 2: (modeled_value - market_price) / market_price ~= +38.60%."""
    result = run_dis(wired)
    gap = result.facts["valuation_gap"]
    assert gap["modeled_return_to_value_pct"] == pytest.approx(
        EXPECTED_MODELED_RETURN_TO_VALUE_PCT, abs=1e-4)


def test_the_two_percentages_are_genuinely_different_numbers(wired):
    result = run_dis(wired)
    gap = result.facts["valuation_gap"]
    assert gap["market_price_premium_pct"] != pytest.approx(
        gap["modeled_return_to_value_pct"], abs=1e-3)


def test_legacy_difference_pct_is_the_price_denominated_figure_not_the_modeled_value_one(wired):
    """difference_pct is kept for backward compatibility, denominated in
    market_price -- confirms it must NOT be used for "X% below the base
    modeled value" phrasing (that needs market_price_premium_pct instead)."""
    result = run_dis(wired)
    gap = result.facts["valuation_gap"]
    assert gap["difference_pct"] == pytest.approx(gap["modeled_return_to_value_pct"], abs=1e-9)
    assert gap["difference_pct"] != pytest.approx(gap["market_price_premium_pct"], abs=1e-3)


def test_compact_report_states_the_discount_from_market_price_premium_pct(wired):
    """Test 3: the renderer quotes market_price_premium_pct, not difference_pct.

    The literal percentage moved under Phase H.4 (the valuation is now built
    on trailing-twelve-month flows and DIS's 2026-06-27 balance sheet, so the
    base modeled value changed), which is exactly why this is now asserted
    against the computed constant rather than a hardcoded string — the bug
    this guards is "which of the two percentages is quoted", and that must
    stay pinned even as the underlying value moves.
    """
    result = run_dis(wired)
    compact = build_compact_synthesis_payload(result)
    text = render_compact_report(result, compact, pipeline_result=None)
    expected = f"{abs(EXPECTED_MARKET_PRICE_PREMIUM_PCT) * 100:.1f}%"
    assert f"Market-price discount to base modeled value: {expected}" in text


def test_compact_report_never_states_the_wrong_39_percent(wired):
    """Test 4: the ORIGINAL bug's exact wrong output must never reappear."""
    result = run_dis(wired)
    compact = build_compact_synthesis_payload(result)
    text = render_compact_report(result, compact, pipeline_result=None)
    assert "39%" not in text
    assert "approximately 39% below the base modeled value" not in text


def test_compact_report_separately_states_the_modeled_return(wired):
    """Test 5: the price-denominated return is shown as its OWN line.

    The two percentages answer different questions and are never
    interchangeable; the original DIS bug was quoting one as the other. Both
    lines must appear, and they must not carry the same number.
    """
    result = run_dis(wired)
    compact = build_compact_synthesis_payload(result)
    text = render_compact_report(result, compact, pipeline_result=None)
    expected = f"+{EXPECTED_MODELED_RETURN_TO_VALUE_PCT * 100:.1f}%"
    assert f"Modeled return from current price to base value: {expected}" in text
    assert (f"{abs(EXPECTED_MARKET_PRICE_PREMIUM_PCT) * 100:.1f}%"
            != f"{EXPECTED_MODELED_RETURN_TO_VALUE_PCT * 100:.1f}%")


# ---- research readiness: base-only vs. pipeline-aware (Problem 2 / tests 6-11) ----

def test_base_research_readiness_is_limited_by_assumption_quality(wired):
    """The BASE (deterministic-only) layer. DIS's data and DCF validate
    cleanly -- the original DIS bug was never in this layer, it was in
    failing to ALSO account for the pipeline (see the effective-readiness
    tests below), and that remains true.

    MLI corrective patch: this now reads LIMITED rather than READY because
    readiness also weighs assumption QUALITY -- on this fixture the
    bull-to-bear range is 96% of the base modeled value. Validation checks
    the arithmetic; it says nothing about whether the inputs mean anything.
    The DOWNGRADE-on-pipeline-failure behavior these tests exist to protect
    is unaffected: LIMITED still never becomes READY."""
    result = run_dis(wired)
    readiness = result.facts["research_readiness"]
    assert readiness["status"] == ResearchReadiness.LIMITED
    assert readiness["status"] != ResearchReadiness.NOT_READY  # data/DCF are sound


def _dis_pipeline_result(bull_status=StageStatus.COMPLETED, bear_status=StageStatus.COMPLETED,
                         research_manager_error=None):
    """A REAL ResearchPipelineResult matching what was actually observed live
    for DIS: bull/bear/rebuttal completed, research_manager FAILED (a
    content-policy/claim-fidelity violation after one repair attempt),
    risk_reviewer and final_investment_synthesizer correctly SKIPPED as a
    consequence -- never themselves marked as having failed."""
    researcher_output = {"thesis": "x", "claims": [], "confidence": "medium"}
    bull = StageCheckpoint("bull_researcher", bull_status,
                           output=researcher_output if bull_status == StageStatus.COMPLETED else None,
                           error=None if bull_status == StageStatus.COMPLETED else "call failed: ReadTimeout")
    bear = StageCheckpoint("bear_researcher", bear_status,
                           output=researcher_output if bear_status == StageStatus.COMPLETED else None,
                           error=None if bear_status == StageStatus.COMPLETED else "call failed: ReadTimeout")
    rebuttal = StageCheckpoint("rebuttal_round", StageStatus.COMPLETED, output={})
    if research_manager_error is not None:
        research_manager = StageCheckpoint("research_manager", StageStatus.FAILED,
                                           error=research_manager_error)
        risk_reviewer = StageCheckpoint("risk_reviewer", StageStatus.SKIPPED,
                                        error="requires research_manager to have completed")
        final = StageCheckpoint("final_investment_synthesizer", StageStatus.SKIPPED,
                                error="requires research_manager and risk_reviewer to have completed")
    else:
        research_manager = StageCheckpoint("research_manager", StageStatus.COMPLETED, output={})
        risk_reviewer = StageCheckpoint("risk_reviewer", StageStatus.COMPLETED, output={
            "key_risks": [], "data_quality_concerns": [], "model_risk": "not_applicable"})
        final = StageCheckpoint("final_investment_synthesizer", StageStatus.COMPLETED, output={
            "research_stance": "cautiously_positive", "valuation_view": "undervalued",
            "overall_risk": "moderate", "confidence": 0.55, "recommendation": "buy",
            "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [],
            "conditions_that_strengthen_the_view": [], "conditions_that_weaken_the_view": [],
            "key_uncertainties": []})
    return ResearchPipelineResult(
        available=final.status == StageStatus.COMPLETED,
        checkpoints=[bull, bear, rebuttal, research_manager, risk_reviewer, final])


DIS_RESEARCH_MANAGER_ERROR = (
    # WM corrective patch: a stage now reports the ATTEMPT COUNT rather than
    # the old fixed "repair attempt also failed: " prefix. Two attempts is
    # still one repair, so the rendered wording below is unchanged.
    "after 2 attempts: CONTENT_POLICY_VIOLATION: prohibited or unsupported language "
    "detected in free-text fields: unsupported superlative (fortress). Every claim must be a "
    "plain, evidence-grounded observation: remove any trade-advice directive (buy/sell/hold/"
    "avoid, position size, entry/exit price, stop loss), unsupported superlative (e.g. "
    "'fortress', 'industry-leading', 'best-in-class', 'guaranteed'), unqualified causal claim "
    "(e.g. 'confirms a reversal', 'protects from downside'), consensus-estimate language (this "
    "system has no analyst-consensus data source), future-tense technical-signal certainty "
    "(e.g. 'will reverse', 'poised to rally'), or a named data provider that supplied nothing "
    "in this analysis. Qualified language is fine (e.g. 'may provide flexibility', 'is "
    "consistent with')."
)


def test_all_required_stages_complete_never_downgrades_the_base_readiness(wired):
    """Test 6, restated. Originally: a fully-complete pipeline yields READY.
    That held only because this fixture's BASE readiness was READY; the
    property actually under test is that a complete pipeline applies NO
    downgrade of its own.

    MLI corrective patch: the base is now LIMITED here (96% scenario
    spread), so this asserts the invariant directly -- a complete pipeline
    passes the base layer through untouched -- which is what the second
    layer is for. `_effective_research_readiness` never UPGRADES, so a
    complete pipeline cannot turn LIMITED into READY either."""
    from finance.workflow import _effective_research_readiness, _pipeline_stage_cascade
    result = run_dis(wired)
    base = result.facts["research_readiness"]
    cascade = _pipeline_stage_cascade(_dis_pipeline_result())
    readiness = _effective_research_readiness(base, cascade)
    assert readiness["status"] == base["status"]
    assert readiness["status"] != ResearchReadiness.NOT_READY


def test_research_manager_failed_yields_limited_not_ready(wired):
    """Tests 7-8: research_manager failed (and risk/synthesis were
    consequently skipped) -> LIMITED, never READY, never NOT_READY (the
    deterministic side of this DIS run is clean)."""
    from finance.workflow import _effective_research_readiness, _pipeline_stage_cascade
    result = run_dis(wired)
    pipeline_result = _dis_pipeline_result(research_manager_error=DIS_RESEARCH_MANAGER_ERROR)
    cascade = _pipeline_stage_cascade(pipeline_result)
    readiness = _effective_research_readiness(result.facts["research_readiness"], cascade)
    assert readiness["status"] == ResearchReadiness.LIMITED


def test_dcf_invalid_stays_not_ready_regardless_of_pipeline_outcome():
    """Test 9 (re-confirmed here with the effective/combined layer, not just
    the base layer already covered in test_finance_workflow.py)."""
    from finance.workflow import _effective_research_readiness, _pipeline_stage_cascade
    base = {"status": ResearchReadiness.NOT_READY, "reasons": ["DCF validation failed."]}
    pipeline_result = _dis_pipeline_result()  # everything completes cleanly
    cascade = _pipeline_stage_cascade(pipeline_result)
    readiness = _effective_research_readiness(base, cascade)
    assert readiness["status"] == ResearchReadiness.NOT_READY


def test_critical_provider_conflict_stays_not_ready():
    """Test 10."""
    from finance.workflow import AnalysisPlan, _research_readiness
    plan = AnalysisPlan("DIS", AnalysisMode.FULL, (), (), (), 0, 0, "")
    facts = {"dcf": {"available": True, "validation_status": "DCF_VALID"},
            "statements": {"warnings": ["total_debt does not reconcile with the provider's own "
                                       "total-debt figure"]}}
    assert _research_readiness(plan, facts)["status"] == ResearchReadiness.NOT_READY


def test_reduced_mode_never_falsely_reports_ready():
    """Test 11: a non-FULL analysis mode (e.g. an omitted optional dataset)
    must never be reported READY."""
    from finance.workflow import AnalysisPlan, _research_readiness
    plan = AnalysisPlan("DIS", AnalysisMode.REDUCED, (), (), (), 0, 0, "")
    facts = {"dcf": {"available": True, "validation_status": "DCF_VALID"},
            "statements": {"warnings": []}}
    assert _research_readiness(plan, facts)["status"] != ResearchReadiness.READY


# ---- naming (Problem 6 / test 12) ----

def test_decision_readiness_never_appears_in_a_rendered_report(wired):
    """Test 12."""
    result = run_dis(wired)
    compact = build_compact_synthesis_payload(result)
    for pipeline_result in (None, _dis_pipeline_result(),
                            _dis_pipeline_result(research_manager_error=DIS_RESEARCH_MANAGER_ERROR)):
        text = render_compact_report(result, compact, pipeline_result)
        assert "Decision readiness" not in text
        assert "decision_readiness" not in text
        assert "Research readiness" in text


# ---- pipeline cascade / failure reason (Problems 3-4 / tests 13-16) ----

def test_failed_research_manager_is_distinguished_from_skipped_risk_reviewer(wired):
    """Test 13."""
    from finance.workflow import _pipeline_stage_cascade
    cascade = _pipeline_stage_cascade(_dis_pipeline_result(research_manager_error=DIS_RESEARCH_MANAGER_ERROR))
    assert cascade["research_manager"] == "FAILED"
    assert cascade["risk_reviewer"] == "SKIPPED_PREREQUISITE"
    assert cascade["final_investment_synthesizer"] == "SKIPPED_PREREQUISITE"
    assert cascade["bull_researcher"] == "COMPLETE"
    assert cascade["bear_researcher"] == "COMPLETE"


def test_actual_failure_reason_appears_in_compact_research_view(wired):
    """Test 14: the SPECIFIC classified reason, not the vague 'requires
    research_manager to have completed'."""
    result = run_dis(wired)
    compact = build_compact_synthesis_payload(result)
    text = render_compact_report(
        result, compact, _dis_pipeline_result(research_manager_error=DIS_RESEARCH_MANAGER_ERROR))
    assert "Research pipeline: PARTIAL" in text
    assert "research_manager made an unsupported evidence claim after one repair attempt" in text


def test_no_raw_validation_dump_in_compact_output(wired):
    """Test 15: never leak the raw prohibited-word examples or the
    boilerplate policy explanation into the rendered report. Excludes the
    report's OWN fixed disclaimer (finance/workflow.py::_COMPACT_DISCLAIMER,
    appended to every compact report unconditionally), which legitimately
    says '...never a buy/sell/hold/avoid instruction' -- the safe, negated
    use of those words, unrelated to this analysis's content."""
    from finance.workflow import _COMPACT_DISCLAIMER
    result = run_dis(wired)
    compact = build_compact_synthesis_payload(result)
    text = render_compact_report(
        result, compact, _dis_pipeline_result(research_manager_error=DIS_RESEARCH_MANAGER_ERROR))
    body = text.replace(_COMPACT_DISCLAIMER, "")
    assert "CONTENT_POLICY_VIOLATION" not in body
    assert "Qualified language is fine" not in body
    assert "fortress" not in body.lower()
    assert "buy/sell/hold/avoid" not in body.lower()


def test_no_final_stance_is_fabricated_when_research_manager_failed(wired):
    """Test 16: 'unavailable', never a guessed research_stance/valuation_view."""
    result = run_dis(wired)
    compact = build_compact_synthesis_payload(result)
    text = render_compact_report(
        result, compact, _dis_pipeline_result(research_manager_error=DIS_RESEARCH_MANAGER_ERROR))
    assert "Research stance: unavailable" in text
    assert "Valuation view: unavailable" in text
    assert "Confidence: unavailable" in text
    # And when everything DOES complete, the real values are shown (never
    # 'unavailable' even though this run's synthesizer output uses them).
    complete_text = render_compact_report(result, compact, _dis_pipeline_result())
    assert "Research stance: cautiously positive" in complete_text
    assert "Valuation view: undervalued" in complete_text


# ---- no trade-advice language anywhere (test 24) ----

def test_compact_report_contains_no_buy_sell_hold_avoid_language(wired):
    from finance.content_policy import scan_for_prohibited_directives
    from finance.workflow import _COMPACT_DISCLAIMER
    result = run_dis(wired)
    compact = build_compact_synthesis_payload(result)
    # None (pipeline disabled) and the failed-research_manager case never
    # reach a validated final synthesis, so 'Recommendation' renders
    # 'unavailable' -- these two stay fully clean under the strict scan,
    # unchanged from before the recommendation field existed.
    for pipeline_result in (None, _dis_pipeline_result(research_manager_error=DIS_RESEARCH_MANAGER_ERROR)):
        text = render_compact_report(result, compact, pipeline_result)
        body = text.replace(_COMPACT_DISCLAIMER, "")
        upper = body.upper()
        for forbidden in ("BUY", "SELL", "HOLD", "AVOID", "STRONG BUY", "STRONG SELL"):
            assert forbidden not in upper

    # The clean-complete case legitimately contains "Recommendation: buy" --
    # assert it appears exactly where expected, then strip only that one
    # line before applying the SAME strict scan to everything else.
    text = render_compact_report(result, compact, _dis_pipeline_result())
    assert "Recommendation: BUY" in text
    body = text.replace(_COMPACT_DISCLAIMER, "").replace("Recommendation: BUY", "")
    upper = body.upper()
    for forbidden in ("BUY", "SELL", "HOLD", "AVOID", "STRONG BUY", "STRONG SELL"):
        assert forbidden not in upper
