"""H.4 corrective patch — finance/workflow.py: report_detail="compact" (the
new default) vs. "full" (today's original, unchanged behavior).

Unit-level rendering pieces (table formatting, N/M handling, section
builders) are exercised directly; this file additionally proves the WIRING
through synthesize_report end to end via a real (small, synthetic)
AnalysisResult, mirroring test_finance_research_pipeline_integration.py.
See tests/test_finance_mo_regression.py for the negative-equity-specific,
end-to-end MO scenario.
"""

import json

import pytest

import tools.config as config
import tools.finance_tools as finance_tools
from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.quota import AlphaVantageQuotaLedger
from finance.workflow import (
    ReportDetail,
    detect_report_detail,
    run_full_stock_analysis,
    synthesize_report,
)
from tests.test_finance_cache import FakeClock
from tests.test_finance_workflow import DatasetClient
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

# ---- canned, per-stage-aware fake ask_local (same shape as
# test_finance_research_pipeline_integration.py's make_happy_ask_local) ----

_BULL = json.dumps({"thesis": "Solid growth supports upside.",
                    "claims": [
                        {"claim_id": "bull-1", "claim": "Revenue growth is healthy.",
                         "evidence_ids": ["fundamental.revenue_growth_yoy"],
                         "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.6},
                        {"claim_id": "bull-2", "claim": "Free cash flow is positive.",
                         "evidence_ids": ["quote.price"],
                         "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.5},
                        {"claim_id": "bull-3", "claim": "The bull scenario uses a higher revenue "
                        "growth and a lower WACC than base.",
                         "evidence_ids": ["quote.price"],
                         "claim_type": "scenario_interpretation", "assumptions": [], "confidence": 0.5},
                    ],
                    "confidence": "medium"})
_BEAR = json.dumps({"thesis": "Valuation looks stretched.",
                    "claims": [
                        {"claim_id": "bear-1", "claim": "Price exceeds the base-case modeled value.",
                         "evidence_ids": ["valuation_gap.direction"],
                         "claim_type": "scenario_interpretation", "assumptions": [], "confidence": 0.6},
                        {"claim_id": "bear-2", "claim": "The scenario spread is wide.",
                         "evidence_ids": ["quote.price"],
                         "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.5},
                    ],
                    "confidence": "medium"})
_REBUTTAL = json.dumps({
    "bull_rebuttal": {"response": "Growth still supports it.",
                      "evidence_cited": ["fundamental.revenue_growth_yoy"]},
    "bear_rebuttal": {"response": "The gap is still real.", "evidence_cited": ["valuation_gap.direction"]},
})
_RESEARCH_MANAGER = json.dumps({
    "evidence_balance": "mixed", "supported_bull_points": ["Revenue growth is healthy."],
    "supported_bear_points": ["Price exceeds the base-case modeled value."],
    "unsupported_points": [], "shared_findings": ["Both sides have real evidence."],
    "key_disagreements": [], "assumption_sensitive_conclusions": [], "data_gaps": [],
    "balanced_assessment": "Both sides have real evidence.",
    "evidence_cited": ["valuation_gap.direction"],
})
_RISK = json.dumps({
    "key_risks": [
        {"risk": "Valuation could compress if growth slows.", "severity": "medium",
         "evidence_cited": ["valuation_gap.direction"]},
        {"risk": "The scenario spread is wide, so the model is assumption-sensitive.",
         "severity": "low", "evidence_cited": ["quote.price"]},
    ],
    "data_quality_concerns": [], "evidence_cited": ["valuation_gap.direction"],
})
_FINAL = json.dumps({
    "research_stance": "cautious", "valuation_view": "overvalued", "overall_risk": "moderate",
    "confidence": 0.5, "recommendation": "avoid",
    "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "The market price appears above the base modeled value.",
                  "evidence_ids": ["valuation_gap.direction"]}],
    "conditions_that_strengthen_the_view": ["Continued revenue growth."],
    "conditions_that_weaken_the_view": ["A pullback toward the base modeled value."],
    "key_uncertainties": ["Assumption sensitivity in the DCF scenarios."],
})


def make_happy_ask_local(record=None):
    def fn(messages, tools=None, timeout=120, options=None, response_format=None):
        if record is not None:
            record.append(messages)
        system = messages[0]["content"]
        if system.startswith("You are the Bull Researcher"):
            content = _BULL
        elif system.startswith("You are the Bear Researcher"):
            content = _BEAR
        elif system.startswith("You are facilitating exactly ONE bounded rebuttal round"):
            content = _REBUTTAL
        elif system.startswith("You are the Research Manager"):
            content = _RESEARCH_MANAGER
        elif system.startswith("You are the Risk Reviewer"):
            content = _RISK
        elif system.startswith("You are the FinalInvestmentSynthesizer"):
            content = _FINAL
        else:
            content = "FULL SINGLE-SHOT / FACTS-ONLY REPORT TEXT"
        return {"message": {"role": "assistant", "content": content},
                "metrics": {"prompt_tokens": 100, "completion_tokens": 20}, "ok": True}

    return fn


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def wired(tmp_path, clock, monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "TEST_KEY_NOT_REAL")
    monkeypatch.setenv("STOCK_ANALYSIS_RESEARCH_PIPELINE_ENABLED", "true")
    coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "m.sqlite3"), clock=clock),
        ledger=AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock, daily_limit=100),
        client=DatasetClient(), clock=clock, sleeper=lambda _s: None, jitter=lambda: 0.5)
    finance_tools.set_coordinator(coordinator)
    registry = ToolRegistry()
    for tool_cls in finance_tools.ALL_FINANCE_TOOL_CLASSES:
        registry.register(tool_cls())
    executor = ToolExecutor(registry)
    yield executor, coordinator
    finance_tools.set_coordinator(None)


def run_analysis(wired):
    executor, _coordinator = wired
    return run_full_stock_analysis(executor, "AAPL", include_news=False)


_COMPACT_SECTION_HEADERS = (
    "## Status", "## Snapshot", "## Valuation", "## Technical", "## Bull Case",
    # Spec 15: "## What Could Change the View" was REMOVED from compact mode --
    # it duplicated the upgrade/downgrade lists the Research View section
    # already renders. Full mode keeps its own expanded detail.
    "## Bear Case", "## Risk", "## Research View",
    "## Sources",
)


# ---- test 21: compact is the default ----

def test_compact_is_the_configured_default():
    assert config.stock_analysis_report_detail_default() == ReportDetail.COMPACT
    assert detect_report_detail(None) == ReportDetail.COMPACT
    assert detect_report_detail("") == ReportDetail.COMPACT
    assert detect_report_detail("What is AAPL trading at?") == ReportDetail.COMPACT
    # The workflow's own colloquial name must NOT force full detail -- see
    # finance/workflow.py::_FULL_DETAIL_TRIGGER_RE's docstring.
    assert detect_report_detail("Give me a full stock analysis of MO") == ReportDetail.COMPACT


def test_report_detail_full_requires_an_explicit_detail_word():
    assert detect_report_detail("Give me a detailed stock research report for MO.") == ReportDetail.FULL
    assert detect_report_detail("I want a full report on MO") == ReportDetail.FULL
    assert detect_report_detail("Give me a comprehensive research report on MO") == ReportDetail.FULL


def test_synthesize_report_defaults_to_compact_with_no_report_detail_argument(wired):
    result = run_analysis(wired)
    text, _metrics = synthesize_report(result, make_happy_ask_local())
    # No redundant bracketed status banner in compact mode -- "## Status"
    # inside the report itself is the one place completeness is stated.
    assert text.startswith("# Stock Analysis")
    assert "## Independent Research" not in text


# ---- test 22: required sections ----

def test_compact_report_contains_every_required_section(wired):
    result = run_analysis(wired)
    text, _metrics = synthesize_report(result, make_happy_ask_local())
    for header in _COMPACT_SECTION_HEADERS:
        assert header in text, header


def test_compact_report_never_calls_the_facts_narrative_llm(wired):
    """Compact mode is rendered ENTIRELY from deterministic facts and the
    pipeline's own validated stage outputs -- exactly 6 calls (one per
    pipeline stage), never a 7th call for narrative prose."""
    calls = []
    result = run_analysis(wired)
    synthesize_report(result, make_happy_ask_local(record=calls))
    assert len(calls) == 6


# ---- test 23-26: excluded internal noise ----

def test_compact_output_excludes_full_evidence_ids(wired):
    """Test 23: bullets are plain claim text, never raw evidence IDs."""
    result = run_analysis(wired)
    text, _metrics = synthesize_report(result, make_happy_ask_local())
    assert "fundamental.revenue_growth_yoy" not in text
    assert "evidence_ids" not in text
    assert "evidence:" not in text.lower()


def test_compact_output_excludes_full_rebuttal_transcript(wired):
    """Test 24."""
    result = run_analysis(wired)
    text, _metrics = synthesize_report(result, make_happy_ask_local())
    assert "Rebuttal Round" not in text
    assert "rebuts" not in text.lower()


def test_compact_output_excludes_implementation_notes(wired):
    """Test 25."""
    result = run_analysis(wired)
    text, _metrics = synthesize_report(result, make_happy_ask_local())
    assert "TradingAgents" not in text
    assert "checkpoint" not in text.lower()
    assert "calculation_version" not in text


def test_compact_output_excludes_repeated_provider_disclaimers(wired):
    """Test 26: the Yahoo-unofficial-source note (when applicable) appears
    at most once, in Status -- never repeated per section."""
    result = run_analysis(wired)
    text, _metrics = synthesize_report(result, make_happy_ask_local())
    assert text.lower().count("unofficial") <= 1


# ---- test 27: size target ----

def test_compact_report_remains_below_configured_size_target(wired):
    result = run_analysis(wired)
    text, _metrics = synthesize_report(result, make_happy_ask_local())
    word_count = len(text.split())
    target = config.stock_analysis_compact_report_target_words()
    # A generous safety margin over the soft target -- the fixed section/
    # bullet caps are what actually bound length; this guards against a
    # structural regression (e.g. a whole extra section rendered twice),
    # not against natural variation in claim-text length.
    assert word_count < target * 1.5, f"{word_count} words, target {target}"
    # The instrumentation figure measures the rendered report alone (before
    # the short status banner is prepended to `text`), so it is a few words
    # smaller than `text`'s own count, not necessarily identical.
    assert 0 < result.instrumentation["compact_report_word_count"] <= word_count


# ---- test 28: full mode still contains detailed analysis ----

def test_full_mode_still_contains_detailed_analysis(wired):
    result = run_analysis(wired)
    text, _metrics = synthesize_report(result, make_happy_ask_local(), report_detail=ReportDetail.FULL)
    assert "## Independent Research (Bull / Bear / Risk / Synthesis)" in text
    assert "### Bull Case" in text and "### Bear Case" in text
    assert "### Research Manager — Evidence Reconciliation" in text
    assert "### Risk Review" in text
    assert "### Research Synthesis — FinalInvestmentSynthesizer" in text
    assert "Rebuttal Round" in text
    assert "FULL SINGLE-SHOT / FACTS-ONLY REPORT TEXT" in text


def test_full_mode_makes_the_extra_facts_narrative_call(wired):
    calls = []
    result = run_analysis(wired)
    synthesize_report(result, make_happy_ask_local(record=calls), report_detail=ReportDetail.FULL)
    assert len(calls) == 7  # 1 facts/narrative call + 6 pipeline stages


def test_compact_report_is_shorter_than_the_full_report(wired):
    result = run_analysis(wired)
    compact_text, _m1 = synthesize_report(result, make_happy_ask_local(), report_detail=ReportDetail.COMPACT)
    full_text, _m2 = synthesize_report(result, make_happy_ask_local(), report_detail=ReportDetail.FULL)
    assert len(compact_text.split()) < len(full_text.split())


# ===========================================================================
# TSLA DCF validation patch: MODEL_INVALID rendering (section 14)
# ===========================================================================

def _invalid_dcf_compact(quote_price=328.58):
    """A hand-built compact payload with an INVALID DCF -- the exact shape
    `finance.dcf_model` produces via `finance/dcf.py::run_dcf` when
    validation fails, but built directly here so the rendering can be
    tested in isolation from the full analysis pipeline."""
    return {
        "symbol": "TSLA",
        "financial_history_currency": "USD",
        "quote": {"price": quote_price, "price_basis": "delayed"},
        "dcf": {
            "available": True,
            "validation_status": "DCF_NEGATIVE_TERMINAL_FCFF",
            "validation_reasons": [
                "DCF_NEGATIVE_TERMINAL_FCFF: negative terminal-year FCFF in scenario(s) base, "
                "bear; the perpetuity terminal value for those scenarios is not treated as "
                "valid valuation evidence."],
            "primary_scenario": "base",
            "scenarios": [
                {"scenario": "base", "value_per_share": -8.1455, "warnings": []},
                {"scenario": "bull", "value_per_share": 6.3525, "warnings": []},
                {"scenario": "bear", "value_per_share": -15.0622, "warnings": []},
            ],
        },
        # These mirror what finance/workflow.py::_valuation_gap / _scenario_spread
        # ACTUALLY return once the DCF is invalid (see their own tests in
        # tests/test_finance_workflow.py) -- both unavailable.
        "valuation_gap": {"available": False,
                          "reason": "The DCF failed deterministic validation "
                                   "(DCF_NEGATIVE_TERMINAL_FCFF); a market-price comparison "
                                   "is withheld."},
        "dcf_scenario_spread": {"available": False,
                                "reason": "The DCF failed deterministic validation "
                                         "(DCF_NEGATIVE_TERMINAL_FCFF); the scenario spread "
                                         "is not meaningful."},
        "decision_readiness": {"status": "NOT_READY",
                               "reasons": ["DCF validation failed (DCF_NEGATIVE_TERMINAL_FCFF); "
                                          "valuation-based conclusions are withheld."]},
    }


class _StubPlan:
    """The minimum an AnalysisResult needs for the report model.

    These tests are about ONE section each, so the surrounding analysis is a
    stub -- but the section itself is built by the real model builder from a
    real compact payload, which is the behaviour under test.
    """

    mode = "full"
    reason = ""
    omitted_datasets = ()
    omission_effects = {}
    requested_datasets = ()
    datasets = ()
    stale_datasets = ()
    omission_reasons = {}


class _StubResult:
    def __init__(self, compact):
        self.symbol = compact.get("symbol") or "TEST"
        self.plan = _StubPlan()
        self.facts = {}
        self.warnings = []
        self.errors = []


def _model_for(compact, pipeline_result=None):
    """The report model for one compact payload.

    Parts 11-13 moved every decision these tests assert out of the renderer
    and into `finance/report_model.py`. The assertions are unchanged -- they
    still read the rendered text -- but the object under test is now the
    model the renderer consumes, which is where the decision actually lives.
    """
    from finance.report_model import build_stock_analysis_report_model
    return build_stock_analysis_report_model(_StubResult(compact), compact, pipeline_result)


def test_valuation_section_states_the_valuation_status_by_cause():
    """Spec 14. The status names WHICH of four things happened.

    This used to read "Status: MODEL_INVALID (DCF_NEGATIVE_TERMINAL_FCFF)"
    for every unusable valuation, including this one -- a model that
    correctly declined to grow a negative terminal cash flow into a
    perpetuity. That is the exact rendering the spec forbids: the model is
    working, the forecast does not support the method, and telling a reader
    the model failed sends them to debug something with nothing wrong
    with it.
    """
    from finance.workflow import _valuation_section
    text = "\n".join(_valuation_section(_model_for(_invalid_dcf_compact())))
    assert "Status: FORECAST_PATH_INVALID" in text
    assert "The model is working; the forecast does not support this method." in text
    assert "MODEL_INVALID" not in text
    # The model's own reason survives -- it is what makes the status checkable.
    assert "DCF_NEGATIVE_TERMINAL_FCFF" in text


def test_valuation_section_never_prints_invalid_bear_base_bull_numbers():
    from finance.workflow import _valuation_section
    text = "\n".join(_valuation_section(_model_for(_invalid_dcf_compact())))
    assert "Bear modeled value" not in text
    assert "Base modeled value" not in text
    assert "Bull modeled value" not in text
    # And the actual invalid figures never leak through in any form.
    for figure in ("-8.15", "6.35", "-15.06", "8.1455", "6.3525", "15.0622"):
        assert figure not in text


def test_valuation_section_never_prints_a_percentage_gap_for_invalid_dcf():
    from finance.workflow import _valuation_section
    text = "\n".join(_valuation_section(_model_for(_invalid_dcf_compact())))
    assert "% above" not in text and "% below" not in text
    assert "Base comparison" not in text


def test_valuation_section_states_comparison_is_withheld():
    from finance.workflow import _valuation_section
    text = "\n".join(_valuation_section(_model_for(_invalid_dcf_compact())))
    assert "withheld" in text.lower()
    # Market price itself is still a plain fact and IS shown.
    assert "328.58" in text


def test_valuation_section_normal_rendering_is_unaffected_when_dcf_valid():
    """The MODEL_INVALID branch must not affect the ordinary (valid) render
    path at all -- same section, same function, different input."""
    from finance.workflow import _valuation_section
    valid_compact = {
        "symbol": "TEST", "financial_history_currency": "USD",
        "quote": {"price": 200.0},
        "dcf": {"available": True, "validation_status": "DCF_VALID", "primary_scenario": "base",
               "scenarios": [{"scenario": "base", "assumptions": {
                   "revenue_growth": [0.1], "wacc": 0.09, "terminal_growth": 0.025},
                   "terminal_value_share_of_enterprise_value": 0.5}]},
        "valuation_gap": {"available": True, "direction": "below", "difference_pct": 0.10},
        "dcf_scenario_spread": {"available": True, "bull_value_per_share": 220.0,
                                "base_value_per_share": 200.0, "bear_value_per_share": 180.0,
                                "spread_pct_of_base": 0.20},
    }
    text = "\n".join(_valuation_section(_model_for(valid_compact)))
    assert "Status: MODEL_INVALID" not in text
    assert "Bear modeled value" in text
    assert "Base modeled value" in text
    assert "Bull modeled value" in text


def _all_complete_pipeline_result(final_output):
    """A REAL ResearchPipelineResult with all six stages COMPLETED --
    exercises `_compact_research_view_section`'s "pipeline completed, but
    base readiness is still not READY for an unrelated deterministic
    reason" branch (the exact TSLA DCF-invalid scenario: fundamentals/
    technicals analysis completes fine, but readiness is NOT_READY because
    of the DCF, not the pipeline)."""
    from finance.research_pipeline import ResearchPipelineResult, StageCheckpoint, StageStatus
    researcher_output = {"thesis": "x", "claims": [], "confidence": "medium"}
    checkpoints = [
        StageCheckpoint("bull_researcher", StageStatus.COMPLETED, output=researcher_output),
        StageCheckpoint("bear_researcher", StageStatus.COMPLETED, output=researcher_output),
        StageCheckpoint("rebuttal_round", StageStatus.COMPLETED, output={}),
        StageCheckpoint("research_manager", StageStatus.COMPLETED, output={}),
        StageCheckpoint("risk_reviewer", StageStatus.COMPLETED, output={}),
        StageCheckpoint("final_investment_synthesizer", StageStatus.COMPLETED, output=final_output),
    ]
    return ResearchPipelineResult(available=True, checkpoints=checkpoints)


def test_research_readiness_line_renders_in_research_view_section():
    from finance.workflow import _compact_research_view_section
    compact = {"research_readiness": {"status": "NOT_READY",
                                      "reasons": ["DCF validation failed."]}}
    lines = _compact_research_view_section(_model_for(compact, None))
    text = "\n".join(lines)
    assert "Research readiness: NOT_READY" in text
    assert "Reason: DCF validation failed." in text
    # No validated final synthesis (pipeline is None/disabled) -- nothing fabricated.
    assert "Research stance: unavailable" in text
    assert "Decision readiness" not in text


def test_research_readiness_line_renders_even_when_pipeline_completed():
    from finance.workflow import _compact_research_view_section

    final_output = {"research_stance": "inconclusive", "valuation_view": "model_invalid",
                    "overall_risk": "high", "confidence": 0.2, "recommendation": "avoid",
                    "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": []}
    compact = {"research_readiness": {"status": "NOT_READY", "reasons": ["DCF invalid."]}}
    lines = _compact_research_view_section(
        _model_for(compact, _all_complete_pipeline_result(final_output)))
    text = "\n".join(lines)
    assert "Research pipeline: COMPLETE" in text
    assert "Research readiness: NOT_READY" in text
    assert "Reason: DCF invalid." in text
    assert "Valuation view: model invalid" in text
    assert "Recommendation: AVOID" in text


def test_recommendation_line_renders_in_research_view_section():
    from finance.workflow import _compact_research_view_section

    final_output = {"research_stance": "positive", "valuation_view": "undervalued",
                    "overall_risk": "low", "confidence": 0.7, "recommendation": "buy",
                    "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": []}
    compact = {"research_readiness": {"status": "READY", "reasons": []}}
    lines = _compact_research_view_section(
        _model_for(compact, _all_complete_pipeline_result(final_output)))
    text = "\n".join(lines)
    assert "Recommendation: BUY" in text


def test_recommendation_is_unavailable_when_pipeline_is_none():
    from finance.workflow import _compact_research_view_section

    compact = {"research_readiness": {"status": "NOT_READY", "reasons": ["DCF validation failed."]}}
    lines = _compact_research_view_section(_model_for(compact, None))
    text = "\n".join(lines)
    assert "Recommendation: unavailable" in text
