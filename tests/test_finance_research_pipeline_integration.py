"""Phase H.2 — integration: finance/research_pipeline.py wired into
finance/workflow.py::synthesize_report.

Unit-level stage/validator behavior lives in test_finance_research_pipeline.py
and test_finance_evidence.py; this file proves the WIRING is correct end to
end through a real (small, synthetic) AnalysisResult produced by
run_full_stock_analysis, mirroring how assistant.py actually calls this code.
"""

import json

import tools.config as config

import pytest

import tools.finance_tools as finance_tools
from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.quota import AlphaVantageQuotaLedger
from finance.research_pipeline import StageStatus
from finance.workflow import (
    FINANCE_FACTS_REPORT_SYSTEM_INSTRUCTIONS,
    FINANCE_REPORT_SYSTEM_INSTRUCTIONS,
    AnalysisMode,
    run_full_stock_analysis,
    synthesize_report,
)
from tests.test_finance_cache import FakeClock
from tests.test_finance_workflow import DatasetClient
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

# ---- canned, per-stage-aware fake ask_local, reused across this file ----

_BULL = json.dumps({"thesis": "Solid growth supports upside.",
                    "claims": [
                        {"claim_id": "bull-1", "claim": "Revenue growth is healthy.",
                         "evidence_ids": ["fundamental.revenue_growth_yoy"],
                         "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.6},
                        {"claim_id": "bull-2", "claim": "Quote reflects a reasonable entry point.",
                         "evidence_ids": ["quote.price"],
                         "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.5},
                    ],
                    "confidence": "medium"})
_BEAR = json.dumps({"thesis": "Valuation looks stretched.",
                    "claims": [
                        {"claim_id": "bear-1", "claim": "Price exceeds the base-case estimate.",
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
    "supported_bear_points": ["Price exceeds the base-case estimate."],
    "unsupported_points": [], "shared_findings": ["Both sides have real evidence."],
    "key_disagreements": [], "assumption_sensitive_conclusions": [], "data_gaps": [],
    "balanced_assessment": "Both sides have real evidence.",
    "evidence_cited": ["valuation_gap.direction"],
})
_RISK = json.dumps({
    "key_risks": [{"risk": "Valuation could compress.", "severity": "medium",
                  "evidence_cited": ["valuation_gap.direction"]}],
    "data_quality_concerns": [], "evidence_cited": ["valuation_gap.direction"],
})
_FINAL = json.dumps({
    "research_stance": "cautious", "valuation_view": "overvalued", "overall_risk": "moderate",
    "confidence": 0.5, "recommendation": "avoid",
    "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": [], "rationale": [{"statement": "The market price appears above the base modeled value.",
                  "evidence_ids": ["valuation_gap.direction"]}],
    "conditions_that_strengthen_the_view": [],
    "conditions_that_weaken_the_view": ["A pullback toward the base modeled value."],
    "key_uncertainties": [],
})


def make_happy_ask_local(record=None):
    def fn(messages, tools=None, timeout=120, options=None, response_format=None):
        if record is not None:
            record.append((messages, {"tools": tools, "timeout": timeout, "options": options,
                                      "response_format": response_format}))
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
            content = "FACTS-ONLY REPORT TEXT"
        return {"message": {"role": "assistant", "content": content},
                "metrics": {"prompt_tokens": 100, "completion_tokens": 20}, "ok": True}

    return fn


def make_total_failure_ask_local(record=None):
    def fn(messages, tools=None, timeout=120, options=None, response_format=None):
        if record is not None:
            record.append((messages, {"tools": tools}))
        system = messages[0]["content"]
        if system.startswith("You are the Bull Researcher") or system.startswith("You are the Bear Researcher"):
            return {"message": {"role": "assistant", "content": "not json"},
                    "metrics": {"prompt_tokens": 10, "completion_tokens": 2}, "ok": True}
        # The FALLBACK single-shot instructions (FINANCE_REPORT_SYSTEM_INSTRUCTIONS)
        # include their own embedded Research Stance section -- the facts-only
        # variant (FINANCE_FACTS_REPORT_SYSTEM_INSTRUCTIONS) does not.
        assert system == FINANCE_REPORT_SYSTEM_INSTRUCTIONS
        return {"message": {"role": "assistant", "content": "FULL SINGLE-SHOT REPORT WITH JUDGMENT"},
                "metrics": {"prompt_tokens": 900, "completion_tokens": 250}, "ok": True}

    return fn


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def wired(tmp_path, clock, monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "TEST_KEY_NOT_REAL")
    # H.4 corrective patch: report_detail now defaults to "compact". This
    # file's assertions are specifically about full-mode's
    # _render_research_pipeline_section text/fallback-note shape (Phase H.2
    # integration), so it is pinned to "full" here — compact-mode's own
    # rendering is covered by tests/test_finance_report_compaction.py.
    monkeypatch.setenv("STOCK_ANALYSIS_REPORT_DETAIL", "full")
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


# ---- pipeline enabled, happy path ----

def test_enabled_happy_path_appends_rendered_research_section(wired, monkeypatch):
    monkeypatch.setenv("STOCK_ANALYSIS_RESEARCH_PIPELINE_ENABLED", "true")
    result = run_analysis(wired)
    calls = []

    text, metrics = synthesize_report(result, make_happy_ask_local(record=calls))

    assert "FACTS-ONLY REPORT TEXT" in text
    assert "## Independent Research (Bull / Bear / Risk / Synthesis)" in text
    assert "### Bull Case" in text and "### Bear Case" in text
    assert "### Research Synthesis — FinalInvestmentSynthesizer" in text
    assert "cautious" in text.lower() and "overvalued" in text.lower()
    assert "**Recommendation:** AVOID" in text
    assert "a research-based recommendation derived from them" in text
    for forbidden in ("BUY.", "SELL.", " HOLD_OFF", "AVOID.", "if you hold", "if you do not hold"):
        assert forbidden not in text

    assert len(calls) == 7  # 1 facts call + 6 pipeline stages
    assert result.research_pipeline["available"] is True
    assert result.research_pipeline["tradingagents_reviewed_commit"]

    # metrics sum the facts call AND every pipeline stage (7 * 100 / 7 * 20)
    assert metrics == {"prompt_tokens": 700, "completion_tokens": 140}


def test_enabled_happy_path_uses_the_facts_only_system_prompt(wired, monkeypatch):
    monkeypatch.setenv("STOCK_ANALYSIS_RESEARCH_PIPELINE_ENABLED", "true")
    result = run_analysis(wired)
    calls = []
    synthesize_report(result, make_happy_ask_local(record=calls))

    assert any(m[0]["content"] == FINANCE_FACTS_REPORT_SYSTEM_INSTRUCTIONS for m, _kw in calls)
    assert not any(m[0]["content"] == FINANCE_REPORT_SYSTEM_INSTRUCTIONS for m, _kw in calls), \
        "the full single-shot fallback prompt must not be used when the pipeline succeeds"


def test_enabled_happy_path_never_passes_tools_to_any_call(wired, monkeypatch):
    monkeypatch.setenv("STOCK_ANALYSIS_RESEARCH_PIPELINE_ENABLED", "true")
    result = run_analysis(wired)
    calls = []
    synthesize_report(result, make_happy_ask_local(record=calls))
    for _messages, kwargs in calls:
        assert not kwargs.get("tools")


def test_enabled_happy_path_never_mutates_deterministic_facts(wired, monkeypatch):
    monkeypatch.setenv("STOCK_ANALYSIS_RESEARCH_PIPELINE_ENABLED", "true")
    result = run_analysis(wired)
    dcf_before = json.dumps(result.facts["dcf"], default=str, sort_keys=True)
    fundamentals_before = json.dumps(result.facts["fundamental_metrics"], default=str, sort_keys=True)

    synthesize_report(result, make_happy_ask_local())

    assert json.dumps(result.facts["dcf"], default=str, sort_keys=True) == dcf_before
    assert json.dumps(result.facts["fundamental_metrics"], default=str, sort_keys=True) == fundamentals_before


# ---- pipeline enabled, total failure -> fallback ----

def test_enabled_total_failure_falls_back_to_single_shot_with_a_note(wired, monkeypatch):
    monkeypatch.setenv("STOCK_ANALYSIS_RESEARCH_PIPELINE_ENABLED", "true")
    result = run_analysis(wired)

    text, metrics = synthesize_report(result, make_total_failure_ask_local())

    assert "FULL SINGLE-SHOT REPORT WITH JUDGMENT" in text
    assert "## Independent Research" not in text, \
        "no rendered pipeline section may appear when the pipeline never reached a verdict"
    assert "could not complete" in text
    assert "bull_researcher" in text  # names the root-cause stage
    assert result.research_pipeline["available"] is False
    # Recommendation reintroduction: the single-shot fallback path never
    # produces a recommendation -- that stays a staged-pipeline-only
    # capability (see finance/workflow.py's FINANCE_REPORT_SYSTEM_
    # INSTRUCTIONS, which is intentionally unchanged by that feature).
    assert "Recommendation:" not in text and "**Recommendation:**" not in text
    # Token accounting sums EVERY attempt. bull_researcher and
    # bear_researcher each return unparseable output, so each is attempted
    # `research_stage_max_attempts()` times at 10 prompt tokens per call,
    # then the single-shot fallback costs 900.
    attempts = config.research_stage_max_attempts()
    assert metrics["prompt_tokens"] == (10 * attempts) + (10 * attempts) + 900
    assert metrics["completion_tokens"] == (2 * attempts) + (2 * attempts) + 250


def test_fallback_note_never_leaks_a_failed_stages_prohibited_word_examples(wired, monkeypatch):
    """Found via live verification against real COST data: a bull_researcher
    claim-fidelity failure's OWN error message explains what was rejected by
    NAMING example prohibited words ("...buy/sell/hold/avoid...", "...
    fortress..."). The user-facing 'could not complete' note must not quote
    that raw message verbatim -- see finance/workflow.py::
    _sanitize_stage_error_for_display (used elsewhere) and, for THIS specific
    fallback note (DIS valuation/readiness correction, Problem 3),
    `_classify_stage_error`/`_pipeline_failure_summary`, which produce a
    SPECIFIC classified reason ("made an unsupported evidence claim") rather
    than the older generic "content-policy/claim-fidelity violation" label —
    still never the raw prohibited-word examples themselves.
    """
    monkeypatch.setenv("STOCK_ANALYSIS_RESEARCH_PIPELINE_ENABLED", "true")
    # Pinned to 2 attempts (one repair) so the rendered wording asserted at
    # the end stays exact. The retry COUNT is not what this test is about —
    # the leak-proofing of the note is.
    monkeypatch.setenv("RESEARCH_STAGE_MAX_ATTEMPTS", "2")
    result = run_analysis(wired)

    # Phase H.5, Phase 2: the superlative sits in `claims`, not `thesis`.
    # `claims` carries min_items=2, so its quarantine policy is FAIL -- the
    # stage genuinely fails, which is what this test needs in order to
    # exercise the fallback note at all. In `thesis` it would now be
    # quarantined and the pipeline would complete, testing nothing here.
    violating_bull = json.dumps({
        "thesis": "Growth and valuation support upside.",
        "claims": [
            {"claim_id": "bull-1", "claim": "A fortress balance sheet supports growth.",
             "evidence_ids": ["fundamental.revenue_growth_yoy"],
             "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.6},
            {"claim_id": "bull-2", "claim": "Quote reflects a reasonable entry point.",
             "evidence_ids": ["quote.price"],
             "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.5},
        ],
        "confidence": "medium",
    })

    def fn(messages, tools=None, timeout=120, options=None, response_format=None):
        system = messages[0]["content"]
        # bull_researcher now gets one repair attempt (COR corrective patch,
        # Phase 5) -- return the SAME violating text both times so the repair
        # also fails, and the stage still ends up FAILED like this test expects.
        content = violating_bull if system.startswith("You are the Bull Researcher") else "not json"
        return {"message": {"role": "assistant", "content": content},
                "metrics": {"prompt_tokens": 10, "completion_tokens": 2}, "ok": True}

    text, _metrics = synthesize_report(result, fn)

    assert "could not complete" in text
    assert "bull_researcher" in text
    for forbidden in ("buy", "sell", "hold", "avoid", "fortress"):
        assert forbidden not in text.lower(), f"leaked prohibited-word example {forbidden!r} into the report"
    assert "bull_researcher made an unsupported evidence claim after one repair attempt" in text
    assert "Research synthesis: FALLBACK" in text


# ---- pipeline disabled: exactly today's original behavior ----

def test_disabled_matches_original_single_call_behavior(wired, monkeypatch):
    monkeypatch.setenv("STOCK_ANALYSIS_RESEARCH_PIPELINE_ENABLED", "false")
    result = run_analysis(wired)
    calls = []

    def fake_ask_local(messages, tools=None, timeout=120, options=None, response_format=None):
        calls.append(messages)
        return {"message": {"content": "SINGLE-SHOT REPORT"},
                "metrics": {"prompt_tokens": 42, "completion_tokens": 7}, "ok": True}

    text, metrics = synthesize_report(result, fake_ask_local)

    assert len(calls) == 1
    assert calls[0][0]["content"] == FINANCE_REPORT_SYSTEM_INSTRUCTIONS
    assert "SINGLE-SHOT REPORT" in text
    assert "## Independent Research" not in text
    assert result.research_pipeline is None
    assert metrics == {"prompt_tokens": 42, "completion_tokens": 7}


# ---- a STOPPED plan never reaches the pipeline (pre-existing guarantee, re-checked) ----

def test_stopped_plan_never_invokes_the_pipeline(wired, monkeypatch):
    monkeypatch.setenv("STOCK_ANALYSIS_RESEARCH_PIPELINE_ENABLED", "true")
    _executor, coordinator = wired
    coordinator.ledger.record_rate_limit(exhausted=True)
    result = run_analysis(wired)
    assert result.plan.mode == AnalysisMode.STOPPED

    calls = []
    text, metrics = synthesize_report(result, make_happy_ask_local(record=calls))

    assert calls == []
    assert result.research_pipeline is None
    assert metrics == {"prompt_tokens": 0, "completion_tokens": 0}
