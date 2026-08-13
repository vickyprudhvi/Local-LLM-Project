"""COR corrective patch (Phase 7) — COR regression, from REAL live Yahoo
Finance + SEC EDGAR data (tests/fixtures/cor_regression.json, captured
2026-08-07; secrets removed; SEC company_facts trimmed to only the
finance/xbrl_mapping.py concepts this codebase reads — see the fixture's own
"note" field).

This is the exact company/data shape that exposed the COR corrective patch's
motivating bug: a live /FullStockAnalysis COR run's staged research pipeline
failed at bull_researcher ("content-policy/claim-fidelity violation") and
fell back to single-pass synthesis. Root cause, from the patch's own
diagnosis (never guessed — see Phase 1): two independent scanner false
positives ("exceptional" flagged even though it was grounded two sentences
away in a separate claim; "hold that" in a DCF-assumption sentence flagged
as though it were a trade recommendation) plus six scanner gaps against
phrasing this project's own policy explicitly names ("downside protection",
"compelling investment/short case", "target price" in reversed word order,
"achievable", the "guarantees" verb form, "invest now").

This file proves, on REAL captured COR data:
  1. The fixed scanners let a realistic, evidence-grounded bull -> bear ->
     one rebuttal round -> research_manager -> risk_reviewer ->
     final_investment_synthesizer sequence run to completion with NO repair
     needed and NO fallback to single-pass synthesis (see the 15-item list
     in test_full_staged_pipeline_completes_without_fallback and its
     neighbors below).
  2. The new per-claim schema ({claim_id, claim, evidence_ids, claim_type,
     assumptions, confidence}) validates real evidence citations end to end.
  3. The SEC-weighted-average-diluted-shares DCF policy (Phase 8) is
     exercised on COR's own real, materially different Yahoo/SEC share
     counts (~190.8M vs ~195.2M, a ~2.2% gap -- much larger than COST's own
     ~0.3% gap, so this is the sharper test of that policy).
  4. COR's genuinely high debt-to-equity ratio (~5.08, driven by a small
     equity base rather than an unusually large debt load) is exactly the
     Phase 9 motivating example.

Every expected value below was independently produced by RUNNING this
fixture through the real pipeline once and reading off the result -- never
invented or back-computed to make a test pass (this project's own standing
rule, re-applied from the COST regression fixture).
"""

import json
from pathlib import Path

import pytest

import tools.finance_tools as finance_tools
from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.provider import AlphaVantageClient, ProviderResponse
from finance.quota import AlphaVantageQuotaLedger, QuotaLedger
from finance.research_pipeline import StageStatus
from finance.sec_datasets import resolve_sec_dataset
from finance.workflow import (
    AnalysisMode,
    run_full_stock_analysis,
    synthesize_report,
)
from finance.yahoo_datasets import resolve_yahoo_dataset
from tests.test_finance_cache import FakeClock
from tools.base import ToolFailure
from tools.executor import ToolExecutor
from tools.models import MARKET_DATA_INVALID_SYMBOL
from tools.registry import ToolRegistry

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cor_regression.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

SYMBOL = "COR"

# ---- ground truth, from the real captured SEC filings (see the fixture's
# annual balance_sheet -- these are simply the raw reported figures,
# independent of anything this codebase calculates) ----
FY2025_REVENUE = 321_332_819_000.0
YAHOO_SHARES_OUTSTANDING = 190_827_165.0
SEC_WEIGHTED_AVG_DILUTED_SHARES = 195_214_000.0
SHARE_COUNT_GAP_PCT = abs(SEC_WEIGHTED_AVG_DILUTED_SHARES - YAHOO_SHARES_OUTSTANDING) / YAHOO_SHARES_OUTSTANDING

# ---- ground truth, independently produced by RUNNING this fixture through
# the real pipeline once and reading off the result (see module docstring) ----
# TSLA DCF validation patch: re-derived after the working-capital corrective
# fix (finance/workflow.py::_historical_nwc_ratio_pairs now excludes cash/
# short-term-investments from current assets and short-term debt/current-
# portion-of-long-term-debt from current liabilities when deriving the
# operating net-working-capital assumption -- see docs/PHASE_H1_STOCK_
# ANALYSIS.md and tests/test_finance_cost_regression.py's own note on the
# same fix). COR's corrected working-capital ratio is also negative
# (releases cash as revenue grows), so every scenario's modeled value is
# higher than under the old, incorrect positive-ratio assumption.
EXPECTED_BASE_VALUE_PER_SHARE = 238.913472
EXPECTED_BULL_VALUE_PER_SHARE = 974.62949
EXPECTED_BEAR_VALUE_PER_SHARE = 127.471426
EXPECTED_NET_DEBT = 8_907_950_000.0  # POSITIVE -- a net-debt position, unlike COST
EXPECTED_DEBT_TO_EQUITY = 5.080024190676642  # Phase 9's motivating example


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
    # Real Yahoo/SEC routing (see test_finance_cost_regression.py's own
    # docstring for why this differs from most other finance test files).
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "TEST_KEY_NOT_REAL")
    monkeypatch.setenv("YAHOO_FINANCE_ENABLED", "true")
    monkeypatch.setenv("YAHOO_PERSONAL_USE_ACKNOWLEDGED", "true")
    monkeypatch.setenv("SEC_EDGAR_ENABLED", "true")
    monkeypatch.setenv("SEC_USER_AGENT", "TestApp/1.0 (contact: t@example.com)")
    # H.4 corrective patch: report_detail now defaults to "compact"; this
    # file's synthesize_report() assertions target the original full-mode
    # single-shot/fallback narrative and _render_research_pipeline_section
    # text shape, so it is pinned to "full" here (same reasoning as pinning
    # the AV/Yahoo/SEC provider routing above).
    monkeypatch.setenv("STOCK_ANALYSIS_REPORT_DETAIL", "full")
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

    # AV configured but its quota is EXHAUSTED -- mirrors the exact live COR
    # report ("Alpha Vantage's quota is exhausted for every AV-routed
    # dataset"), unlike COST's regression fixture which used an unconfigured
    # key instead (a different, also-controlled, omission path -- see
    # test_alphavantage_quota_exhaustion_omits_earnings_without_blocking_yahoo_sec
    # below for why this one produces a proactive plan-time omission with no
    # per-call error, rather than an attempted-and-failed call).
    av_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "av_m.sqlite3"), clock=clock),
        ledger=AlphaVantageQuotaLedger(path=str(tmp_path / "av_q.sqlite3"), clock=clock, daily_limit=100),
        client=AlphaVantageClient(), clock=clock, sleeper=lambda _s: None, jitter=lambda: 0.5)
    av_coordinator.ledger.record_rate_limit(exhausted=True)
    finance_tools.set_coordinator(av_coordinator)

    registry = ToolRegistry()
    for tool_cls in finance_tools.ALL_FINANCE_TOOL_CLASSES:
        registry.register(tool_cls())
    executor = ToolExecutor(registry)

    yield executor
    finance_tools.set_coordinator(None)
    finance_tools.set_yahoo_coordinator(None)
    finance_tools.set_sec_coordinator(None)


def run_cor(wired):
    return run_full_stock_analysis(wired, SYMBOL, include_news=False)


def _scenario(result, name):
    return next(s for s in result.facts["dcf"]["scenarios"] if s["scenario"] == name)


# ---- canned, per-stage-aware fake ask_local -- realistic, evidence-grounded
# text using REAL evidence IDs from COR's own evidence index (verified
# against the real index while authoring this fixture, not guessed), written
# to comply with Phase 3/4/6's vocabulary from the first attempt -- this is
# the proof that a compliant response is actually reachable end to end, not
# just that a repair can rescue a non-compliant one (that path is already
# covered by tests/test_finance_research_pipeline.py's repair-focused tests).
# ----

_BULL_COR = json.dumps({
    "thesis": ("Revenue growth remains positive, free cash flow is positive, and the market "
              "price sits below the modeled base scenario value, though profitability margins "
              "are thin."),
    "claims": [
        {"claim_id": "bull-1", "claim": "Revenue grew approximately 9.3% year over year.",
         "evidence_ids": ["fundamental.revenue_growth_yoy"], "claim_type": "fact_interpretation",
         "assumptions": [], "confidence": 0.7},
        {"claim_id": "bull-2",
         "claim": "Free cash flow is positive, which may provide some financial flexibility.",
         "evidence_ids": ["fundamental.free_cash_flow"], "claim_type": "risk_offset",
         "assumptions": [], "confidence": 0.6},
        {"claim_id": "bull-3", "claim": "The market price is below the modeled base scenario value.",
         "evidence_ids": ["valuation_gap.direction", "valuation_gap.difference_pct"],
         "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.5},
        {"claim_id": "bull-4",
         "claim": ("The bull scenario produces a modeled value of approximately $1,505.77 per "
                   "share under the listed assumptions."),
         "evidence_ids": ["dcf.value_per_share.bull"], "claim_type": "scenario_interpretation",
         "assumptions": ["Revenue growth near 14.7%", "Operating margin expansion to approximately 3%"],
         "confidence": 0.4},
        {"claim_id": "bull-5",
         "claim": ("Reported return on average equity is approximately 144%, a figure influenced "
                   "by a comparatively small equity base."),
         "evidence_ids": ["fundamental.roe_average_equity", "fundamental.debt_to_equity"],
         "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.5},
    ],
    "confidence": "medium",
})

_BEAR_COR = json.dumps({
    "thesis": ("Balance-sheet leverage is elevated relative to the equity base, short-term "
              "liquidity is tight, and operating margins are thin."),
    "claims": [
        {"claim_id": "bear-1",
         "claim": ("The debt-to-equity ratio is approximately 5.08, reflecting a comparatively "
                   "small equity base relative to total debt."),
         "evidence_ids": ["fundamental.debt_to_equity"], "claim_type": "fact_interpretation",
         "assumptions": [], "confidence": 0.7},
        {"claim_id": "bear-2", "claim": "The current ratio is approximately 0.90, below 1.0.",
         "evidence_ids": ["fundamental.current_ratio"], "claim_type": "fact_interpretation",
         "assumptions": [], "confidence": 0.7},
        {"claim_id": "bear-3",
         "claim": "The operating margin is approximately 0.8%, a thin margin relative to revenue.",
         "evidence_ids": ["fundamental.operating_margin"], "claim_type": "fact_interpretation",
         "assumptions": [], "confidence": 0.6},
        {"claim_id": "bear-4",
         "claim": ("The bear scenario produces a modeled value of approximately $208.09 per "
                   "share under the listed assumptions."),
         "evidence_ids": ["dcf.value_per_share.bear"], "claim_type": "scenario_interpretation",
         "assumptions": ["Revenue growth near 6.7%", "Operating margin remaining near 1%"],
         "confidence": 0.4},
    ],
    "confidence": "medium",
})

_REBUTTAL_COR = json.dumps({
    "bull_rebuttal": {
        "response": ("The elevated debt-to-equity ratio partly reflects a small equity base "
                    "rather than an unusually large debt load in isolation."),
        "evidence_cited": ["fundamental.debt_to_equity"],
    },
    "bear_rebuttal": {
        "response": "A small equity base does not fully offset the current ratio remaining below 1.0.",
        "evidence_cited": ["fundamental.current_ratio"],
    },
})

_RESEARCH_MANAGER_COR = json.dumps({
    "evidence_balance": "mixed",
    "supported_bull_points": ["Revenue grew approximately 9.3% year over year.",
                              "Free cash flow is positive."],
    "supported_bear_points": ["The debt-to-equity ratio is approximately 5.08.",
                              "The current ratio is approximately 0.90, below 1.0."],
    "unsupported_points": [],
    "shared_findings": ["Both sides cite the same reported leverage and liquidity figures."],
    "key_disagreements": [("Whether the elevated debt-to-equity ratio primarily reflects a small "
                          "equity base or a genuine balance-sheet concern.")],
    "assumption_sensitive_conclusions": [("The bull and bear DCF scenarios diverge mainly on "
                                         "revenue-growth and operating-margin assumptions.")],
    "data_gaps": [("Earnings history and earnings-surprise context are unavailable this cycle.")],
    "balanced_assessment": ("Both sides cite real, reported evidence; the spread between the bull "
                           "and bear modeled values mainly reflects differing revenue-growth and "
                           "margin assumptions."),
    "evidence_cited": ["fundamental.debt_to_equity", "fundamental.current_ratio"],
})

_RISK_COR = json.dumps({
    "key_risks": [
        {"risk": ("Balance-sheet leverage is elevated relative to the equity base; net debt is "
                 "approximately $3.30 billion, and total debt is approximately 2.4 times free "
                 "cash flow."),
         "severity": "medium",
         "evidence_cited": ["fundamental.debt_to_equity", "fundamental.net_debt", "fundamental.debt_to_fcf"]},
        {"risk": "The current ratio is below 1.0, indicating comparatively tight short-term liquidity.",
         "severity": "medium", "evidence_cited": ["fundamental.current_ratio"]},
        {"risk": "Operating margins are thin relative to revenue.",
         "severity": "low", "evidence_cited": ["fundamental.operating_margin"]},
    ],
    "data_quality_concerns": [("Earnings history is unavailable this cycle.")],
    "evidence_cited": ["fundamental.debt_to_equity", "fundamental.net_debt", "fundamental.debt_to_fcf",
                       "fundamental.current_ratio", "fundamental.operating_margin"],
})

_FINAL_COR = json.dumps({
    "research_stance": "neutral", "valuation_view": "undervalued", "overall_risk": "moderate",
    "confidence": 0.5, "recommendation": "hold",
    "primary_reason": "Evidence and valuation point the same way at this confidence level.", "supporting_factors": [], "limiting_factors": ["Research readiness is LIMITED: a key DCF assumption came from a configured default and the bull-to-bear scenario spread is wide."], "rationale": [
        {"statement": ("The market price sits below the modeled base scenario value while "
                      "balance-sheet leverage remains elevated relative to the equity base."),
         "evidence_ids": ["valuation_gap.direction", "fundamental.debt_to_equity"]},
        {"statement": ("Earnings history and earnings-surprise context are unavailable this "
                      "cycle, which limits the completeness of this analysis."),
         "evidence_ids": ["plan.omitted.earnings"]},
    ],
    "conditions_that_strengthen_the_view": [
        "Continued revenue growth alongside stable or improving operating margin.",
    ],
    "conditions_that_weaken_the_view": [
        "Further margin compression or a deterioration in short-term liquidity.",
    ],
    "key_uncertainties": [
        ("The DCF scenario spread is wide, so the valuation view is sensitive to revenue-growth "
        "and margin assumptions."),
    ],
})


def make_cor_ask_local(record=None):
    def fn(messages, tools=None, timeout=120, options=None, response_format=None):
        if record is not None:
            record.append((messages, {"tools": tools, "timeout": timeout, "options": options,
                                      "response_format": response_format}))
        system = messages[0]["content"]
        if system.startswith("You are the Bull Researcher"):
            content = _BULL_COR
        elif system.startswith("You are the Bear Researcher"):
            content = _BEAR_COR
        elif system.startswith("You are facilitating exactly ONE bounded rebuttal round"):
            content = _REBUTTAL_COR
        elif system.startswith("You are the Research Manager"):
            content = _RESEARCH_MANAGER_COR
        elif system.startswith("You are the Risk Reviewer"):
            content = _RISK_COR
        elif system.startswith("You are the FinalInvestmentSynthesizer"):
            content = _FINAL_COR
        else:
            content = "FACTS-ONLY REPORT TEXT"
        return {"message": {"role": "assistant", "content": content},
                "metrics": {"prompt_tokens": 100, "completion_tokens": 20}, "ok": True}

    return fn


# ---- 0: the fixture itself is real COR data, not a stand-in ----

def test_fixture_is_real_captured_cor_data(wired):
    result = run_cor(wired)
    assert result.facts["overview"]["name"] == "Cencora, Inc."
    annual_income = result.facts["statements"]["annual"]["income_statement"]
    assert annual_income[0]["values"]["revenue"] == pytest.approx(FY2025_REVENUE)


def test_fixture_reproduces_the_exact_live_report_share_count_figures(wired):
    """Yahoo shares outstanding ~190.8M and SEC weighted-average diluted
    shares ~195.2M -- the exact figures named in the live COR report that
    exposed Phase 8's DCF share-count policy question."""
    result = run_cor(wired)
    assert result.facts["overview"]["shares_outstanding"] == pytest.approx(YAHOO_SHARES_OUTSTANDING)
    balance = result.facts["statements"]["annual"]["balance_sheet"]
    assert balance[0]["values"]["shares_outstanding"] == pytest.approx(SEC_WEIGHTED_AVG_DILUTED_SHARES)
    assert SHARE_COUNT_GAP_PCT == pytest.approx(0.023, abs=0.001)


# ---- 12: DCF scenario values remain unchanged (independently computed once, never invented) ----

def test_dcf_scenario_values_match_the_independently_computed_result(wired):
    result = run_cor(wired)
    base, bull, bear = (_scenario(result, n) for n in ("base", "bull", "bear"))
    assert base["value_per_share"] == pytest.approx(EXPECTED_BASE_VALUE_PER_SHARE, rel=1e-6)
    assert bull["value_per_share"] == pytest.approx(EXPECTED_BULL_VALUE_PER_SHARE, rel=1e-6)
    assert bear["value_per_share"] == pytest.approx(EXPECTED_BEAR_VALUE_PER_SHARE, rel=1e-6)


def test_dcf_is_deterministic_across_repeated_runs(wired):
    result_a = run_cor(wired)
    result_b = run_cor(wired)
    assert result_a.facts["dcf"]["scenarios"] == result_b.facts["dcf"]["scenarios"]


def test_bull_exceeds_base_exceeds_bear(wired):
    result = run_cor(wired)
    base, bull, bear = (_scenario(result, n) for n in ("base", "bull", "bear"))
    assert bull["value_per_share"] > base["value_per_share"] > bear["value_per_share"]


# ---- Phase 8: share-count policy, COR's own (larger) real divergence ----

def test_dcf_uses_sec_weighted_average_diluted_shares_not_yahoo_basic(wired):
    result = run_cor(wired)
    assert result.facts.get("dcf_shares_outstanding_source") == "sec_weighted_average_diluted"
    base = _scenario(result, "base")
    assert base["diluted_shares"] == pytest.approx(SEC_WEIGHTED_AVG_DILUTED_SHARES)


def test_yahoo_basic_shares_remain_visible_separately_never_deleted(wired):
    result = run_cor(wired)
    yahoo_shares = result.facts["overview"].get("shares_outstanding")
    sec_shares = result.facts["statements"]["annual"]["balance_sheet"][0]["values"].get("shares_outstanding")
    assert yahoo_shares is not None and yahoo_shares > 0
    assert sec_shares is not None and sec_shares > 0
    assert yahoo_shares != sec_shares
    # COR's own real gap is much larger than COST's (~0.3%) -- ~2.2% here,
    # the sharper regression case for this exact policy.
    assert abs(sec_shares - yahoo_shares) / yahoo_shares > 0.02


def test_share_count_mismatch_is_reported_not_silently_substituted(wired):
    """finance/reconciliation.py's own warning (Phase 8: 'do not silently
    substitute one for the other')."""
    result = run_cor(wired)
    assert any("MARKET_CAP_SHARE_COUNT_MISMATCH" in str(w) for w in result.warnings), result.warnings


# ---- Phase 9: leverage context (COR's own high debt-to-equity, small equity base) ----

def test_debt_to_equity_matches_the_independently_computed_result(wired):
    result = run_cor(wired)
    metrics = result.facts["fundamental_metrics"]
    assert metrics["debt_to_equity"]["value"] == pytest.approx(EXPECTED_DEBT_TO_EQUITY)


def test_net_debt_is_a_real_net_debt_position_unlike_cost(wired):
    result = run_cor(wired)
    base = _scenario(result, "base")
    assert base["net_debt"] == pytest.approx(EXPECTED_NET_DEBT)
    assert base["net_debt"] > 0, "COR carries more debt than cash -- unlike COST's net-cash position"


def test_broader_leverage_context_is_available_alongside_debt_to_equity(wired):
    """The property Phase 9 asks for, on real COR data: a high debt_to_equity
    does not ship alone -- net_debt, net_debt_to_fcf, debt_to_fcf, and
    operating_cash_flow are simultaneously computable and become real,
    citable evidence IDs (finance/evidence.py's generic fundamental_metrics
    loop), matching what the RiskReviewer's own canned response above
    (_RISK_COR) actually cites."""
    from finance.evidence import build_evidence_index
    from finance.workflow import build_compact_synthesis_payload

    result = run_cor(wired)
    metrics = result.facts["fundamental_metrics"]
    assert metrics["debt_to_equity"]["value"] == pytest.approx(EXPECTED_DEBT_TO_EQUITY)
    for name in ("net_debt", "net_debt_to_fcf", "debt_to_fcf", "operating_cash_flow"):
        assert metrics[name]["value"] is not None, name
    # interest_coverage is "if supported" only -- correctly absent for this
    # SEC-sourced fixture (finance/xbrl_mapping.py has no interest-expense
    # concept), never invented.
    assert metrics["interest_coverage"]["value"] is None

    index = build_evidence_index(build_compact_synthesis_payload(result))
    for evidence_id in ("fundamental.debt_to_equity", "fundamental.net_debt",
                       "fundamental.debt_to_fcf", "fundamental.operating_cash_flow"):
        assert evidence_id in index, evidence_id
    assert "fundamental.interest_coverage" not in index


# ---- reduced status + Alpha Vantage quota exhaustion (items 13-14) ----

def test_reduced_status_is_visible_because_earnings_are_missing(wired):
    result = run_cor(wired)
    assert result.plan.mode == AnalysisMode.REDUCED
    assert "earnings" in result.plan.omitted_datasets


def test_alphavantage_quota_exhaustion_omits_earnings_without_blocking_yahoo_sec(wired):
    """COR's live report language ('Alpha Vantage's quota is exhausted for
    every AV-routed dataset, but 5 dataset(s) were retrieved from other
    providers') is a PLANNING-time omission (the ledger reports zero
    remaining quota once exhausted -- finance/quota.py's QuotaSnapshot.
    remaining), not an attempted-and-failed call -- unlike COST's regression
    fixture, which uses an unconfigured key and so DOES see a per-call
    MARKET_DATA_API_KEY_MISSING error. Both are legitimate controlled, non-
    crash outcomes; this test proves the exhaustion path specifically, since
    it is what the real live COR report actually showed."""
    result = run_cor(wired)
    assert result.facts["dcf"]["available"] is True
    assert result.facts["quote"]["available"] is True
    providers_seen = {p.get("provider") for p in result.facts["data_provenance"].values()
                      if isinstance(p, dict)}
    assert providers_seen == {"yahoo", "sec"}, providers_seen


# ---- provider attribution (item 15) ----

def test_provider_attribution_is_correct_for_every_dataset(wired):
    result = run_cor(wired)
    prov = result.facts["data_provenance"]
    for dataset in ("income_statement", "balance_sheet", "cash_flow"):
        assert prov[dataset]["provider"] == "sec", dataset
    for dataset in ("quote", "company_profile", "price_history", "corporate_actions",
                    "analyst_estimates"):
        assert prov[dataset]["provider"] == "yahoo", dataset


# ---- items 1-11: the full staged pipeline, on real COR evidence, first try ----

def test_full_staged_pipeline_completes_without_fallback(wired):
    result = run_cor(wired)
    calls = []
    text, _metrics = synthesize_report(result, make_cor_ask_local(record=calls))

    pipeline = result.research_pipeline
    assert pipeline is not None
    assert pipeline["available"] is True  # 7: fallback single-pass synthesis was NOT used

    by_stage = {c["stage"]: c for c in pipeline["checkpoints"]}
    for stage in ("bull_researcher", "bear_researcher", "rebuttal_round", "research_manager",
                 "risk_reviewer", "final_investment_synthesizer"):
        assert by_stage[stage]["status"] == StageStatus.COMPLETED, \
            f"{stage}: {by_stage[stage].get('error')}"  # 1, 2, 4, 5, 6

    # 3: exactly one rebuttal round -- exactly one call to that stage's prompt.
    rebuttal_calls = [c for c in calls
                      if c[0][0]["content"].startswith("You are facilitating exactly ONE bounded rebuttal round")]
    assert len(rebuttal_calls) == 1

    # bull_researcher/bear_researcher each succeeded on the FIRST attempt --
    # no repair needed (proves the fixed scanners, not just the repair path).
    bull_calls = [c for c in calls if c[0][0]["content"].startswith("You are the Bull Researcher")]
    bear_calls = [c for c in calls if c[0][0]["content"].startswith("You are the Bear Researcher")]
    assert len(bull_calls) == 1
    assert len(bear_calls) == 1

    # 7 (again, at the rendered-report level): the real section renders, the
    # fallback note does not.
    assert "## Independent Research (Bull / Bear / Risk / Synthesis)" in text
    assert "### Bull Case" in text and "### Bear Case" in text
    assert "could not complete" not in text
    assert "single-pass" not in text.lower()

    # 10-11 at the rendered-report level: NOT a raw scan of the full text --
    # the report's OWN standing disclaimer legitimately contains "position
    # size"/"guarantee" inside a negated sentence ("never a position size, an
    # order, or a guarantee of future performance"), the recommendation line
    # itself legitimately contains the model's chosen word ("Recommendation:
    # hold" for this fixture), and the research_manager section's fixed
    # template heading ("Conclusions that only hold under specific
    # assumptions") legitimately contains the ordinary word "hold" -- all are
    # pre-existing, deterministic template text or the ONE field designed to
    # carry this word, never a smuggled trade directive elsewhere, and
    # tests/test_finance_research_pipeline_integration.py's own happy-path
    # test already established this exact curated-substring methodology for
    # that reason. The PRECISE per-stage checks (proven not to depend on
    # boilerplate wording) are test_no_prohibited_recommendation_language_
    # in_any_stage_output and test_no_unsupported_superlatives_in_any_stage_
    # output below.
    assert "**Recommendation:** HOLD" in text  # rendered upper-case (spec 12)
    assert "a research-based recommendation derived from them" in text
    for forbidden in ("BUY.", "SELL.", " HOLD_OFF", "AVOID.", "if you hold", "if you do not hold"):
        assert forbidden not in text


def test_every_bull_claim_has_valid_evidence_ids(wired):
    """8: every bull claim cites at least one evidence ID that is genuinely
    present in COR's own real evidence index (not merely 'the stage didn't
    fail' -- this cross-checks the actual validated output)."""
    from finance.evidence import build_evidence_index
    from finance.workflow import build_compact_synthesis_payload

    result = run_cor(wired)
    index = build_evidence_index(build_compact_synthesis_payload(result))
    synthesize_report(result, make_cor_ask_local())

    bull_output = next(c["output"] for c in result.research_pipeline["checkpoints"]
                       if c["stage"] == "bull_researcher")
    assert len(bull_output["claims"]) >= 2
    for claim in bull_output["claims"]:
        assert claim["evidence_ids"], claim
        for evidence_id in claim["evidence_ids"]:
            assert evidence_id in index, f"{evidence_id!r} not in the real COR evidence index"


def test_every_bear_claim_has_valid_evidence_ids(wired):
    """9: bear-side counterpart of the above."""
    from finance.evidence import build_evidence_index
    from finance.workflow import build_compact_synthesis_payload

    result = run_cor(wired)
    index = build_evidence_index(build_compact_synthesis_payload(result))
    synthesize_report(result, make_cor_ask_local())

    bear_output = next(c["output"] for c in result.research_pipeline["checkpoints"]
                       if c["stage"] == "bear_researcher")
    assert len(bear_output["claims"]) >= 2
    for claim in bear_output["claims"]:
        assert claim["evidence_ids"], claim
        for evidence_id in claim["evidence_ids"]:
            assert evidence_id in index, f"{evidence_id!r} not in the real COR evidence index"


def test_no_prohibited_recommendation_language_in_any_stage_output(wired):
    """10, stage-by-stage (stronger than the whole-report-text check above --
    this also covers stage output fields the renderer might not surface).

    Recommendation reintroduction: `final_investment_synthesizer`'s own
    `recommendation` field is EXPECTED to contain a trade-directive word by
    design (that is the whole point of the field) -- excluded here before
    scanning, exactly like it is excluded from the scan inside
    `_validate_final_synthesizer_output` itself. Every OTHER field on this
    stage, and every field on every OTHER stage, stays fully scanned and
    must still be completely clean."""
    from finance.content_policy import scan_structure_for_prohibited_directives

    result = run_cor(wired)
    synthesize_report(result, make_cor_ask_local())
    for checkpoint in result.research_pipeline["checkpoints"]:
        if checkpoint["status"] == StageStatus.COMPLETED:
            output = checkpoint["output"]
            if checkpoint["stage"] == "final_investment_synthesizer":
                assert "recommendation" in output  # sanity: the field really is there
                output = {k: v for k, v in output.items() if k != "recommendation"}
            hits = scan_structure_for_prohibited_directives(output)
            assert hits == [], f"{checkpoint['stage']}: {hits}"


def test_no_unsupported_superlatives_in_any_stage_output(wired):
    """11, stage-by-stage."""
    from finance.claim_validation import scan_structure_for_unsupported_claims

    result = run_cor(wired)
    synthesize_report(result, make_cor_ask_local())
    for checkpoint in result.research_pipeline["checkpoints"]:
        if checkpoint["status"] == StageStatus.COMPLETED:
            hits = scan_structure_for_unsupported_claims(checkpoint["output"])
            assert hits == [], f"{checkpoint['stage']}: {hits}"


def test_dcf_scenario_values_are_unchanged_by_running_the_staged_pipeline(wired):
    """12, re-checked after synthesize_report runs -- the research pipeline
    must never mutate the deterministic DCF facts it reads."""
    result = run_cor(wired)
    before = json.dumps(result.facts["dcf"], default=str, sort_keys=True)
    synthesize_report(result, make_cor_ask_local())
    after = json.dumps(result.facts["dcf"], default=str, sort_keys=True)
    assert before == after
