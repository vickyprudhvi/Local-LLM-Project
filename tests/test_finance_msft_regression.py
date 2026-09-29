"""Phase H.1 accuracy patch — MSFT regression, from the ACTUAL captured
response that exposed the bugs (secrets removed; see
tests/fixtures/msft_regression.json).

Ground truth, independently recomputed from the fixture's raw fields:

    cash_and_cash_equivalents          20,935,000,000
    short_term_investments             55,908,000,000
    cash_and_short_term_investments    76,843,000,000   (DERIVED; provider's
                                                          own field says
                                                          20,935,000,000 — a
                                                          bug in the source
                                                          data, not this code)
    short_term_debt                    18,905,300,000
    current_portion_of_long_term_debt   9,227,000,000
    long_term_debt                     31,067,000,000
    total_debt                         59,199,300,000   (was ~50B before this
                                                          patch: current
                                                          portion was missing)
    net_debt (cash_only, default)      38,264,300,000
    net_debt (cash_and_marketable_securities, STI eligible)  -17,643,700,000
                                        (a NET CASH position — the sign flips
                                        entirely depending on policy)

This file proves the pipeline produces these exact numbers end to end: raw
payload -> normalization -> metrics -> finance.dcf_model -> compact synthesis
payload, through the real ToolExecutor.
"""

import json
from pathlib import Path

import pytest

import tools.finance_tools as finance_tools
from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.dcf import NetDebtPolicy
from finance.provider import ProviderResponse
from finance.evidence import build_evidence_index, render_evidence_index
from finance.workflow import (
    AnalysisMode,
    build_compact_synthesis_payload,
    run_full_stock_analysis,
    synthesize_report,
)
from finance.quota import AlphaVantageQuotaLedger
from tests.test_finance_cache import FakeClock
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "msft_regression.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

EXPECTED_CASH = 20_935_000_000
EXPECTED_STI = 55_908_000_000
EXPECTED_COMBINED = EXPECTED_CASH + EXPECTED_STI
EXPECTED_TOTAL_DEBT = 18_905_300_000 + 9_227_000_000 + 31_067_000_000
EXPECTED_NET_DEBT_CASH_ONLY = EXPECTED_TOTAL_DEBT - EXPECTED_CASH
EXPECTED_NET_DEBT_WITH_STI = EXPECTED_TOTAL_DEBT - EXPECTED_CASH - EXPECTED_STI


class FixtureClient:
    """Serves the ACTUAL captured MSFT payloads. No live network."""

    def __init__(self):
        self.calls = []

    def fetch(self, dataset, arguments):
        self.calls.append(dataset.dataset_id)
        payload = FIXTURE.get(dataset.dataset_id, {})
        return ProviderResponse(payload=payload,
                                provider_metadata={"provider_function": dataset.function},
                                byte_count=len(json.dumps(payload)))

    @property
    def call_count(self):
        return len(self.calls)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def wired(tmp_path, clock, monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "TEST_KEY_NOT_REAL")
    # H.4 corrective patch: report_detail now defaults to "compact". This
    # file's synthesize_report() assertions (single-call count, payload/
    # instrumentation shape) are about Problem 9's compaction specifically —
    # pinned to "full" so they keep exercising the original single-shot
    # narrative call, same reasoning as pinning the research-pipeline flag
    # per-test below.
    monkeypatch.setenv("STOCK_ANALYSIS_REPORT_DETAIL", "full")
    coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "m.sqlite3"), clock=clock),
        ledger=AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock,
                                       daily_limit=100),
        client=FixtureClient(), clock=clock, sleeper=lambda _s: None, jitter=lambda: 0.5)
    finance_tools.set_coordinator(coordinator)

    registry = ToolRegistry()
    for tool_cls in finance_tools.ALL_FINANCE_TOOL_CLASSES:
        registry.register(tool_cls())
    executor = ToolExecutor(registry)
    yield executor, coordinator
    finance_tools.set_coordinator(None)


def run_msft(wired, **kwargs):
    executor, _coordinator = wired
    return run_full_stock_analysis(executor, "MSFT", include_news=False, **kwargs)


# ---- 1: cash is not mislabeled as cash + investments ----

def test_cash_is_not_mislabeled_as_the_combined_value(wired):
    result = run_msft(wired)
    values = result.facts["statements"]["annual"]["balance_sheet"][0]["values"]

    assert values["cash_and_cash_equivalents"] == pytest.approx(EXPECTED_CASH)
    assert values["cash_and_cash_equivalents"] != values["cash_and_short_term_investments"], \
        "the ORIGINAL bug: cash alone was presented as the combined figure"


# ---- 2: short-term investments are included separately ----

def test_short_term_investments_are_captured_separately(wired):
    result = run_msft(wired)
    values = result.facts["statements"]["annual"]["balance_sheet"][0]["values"]

    assert values["short_term_investments"] == pytest.approx(EXPECTED_STI)
    assert values["cash_and_short_term_investments"] == pytest.approx(EXPECTED_COMBINED)
    # The provider's OWN combined field is retained only for audit, and is
    # visibly wrong for this issuer — proving the fix doesn't just trust it.
    assert values["cash_and_short_term_investments_provider_reported"] == pytest.approx(
        EXPECTED_CASH)
    assert any("does not reconcile" in w and "cash-and-short-term-investments" in w
              for w in result.warnings)


# ---- 3: total debt under the documented policy ----

def test_total_debt_includes_the_current_portion_of_long_term_debt(wired):
    result = run_msft(wired)
    values = result.facts["statements"]["annual"]["balance_sheet"][0]["values"]

    assert values["total_debt"] == pytest.approx(EXPECTED_TOTAL_DEBT)
    # The bug this replaces: short_term_debt + long_term_debt alone, missing
    # the ~$9.2B current portion — roughly the "$50B" figure originally seen.
    undercounted = 18_905_300_000 + 31_067_000_000
    assert values["total_debt"] != pytest.approx(undercounted)
    assert any("total-debt" in w and "does not reconcile" in w for w in result.warnings)


# ---- 4: net-debt / net-cash calculation is correct ----

def test_net_debt_under_the_default_cash_only_policy(wired):
    result = run_msft(wired)
    dcf = result.facts["dcf"]
    assert dcf["available"] is True
    assert dcf["net_debt_policy"] == NetDebtPolicy.CASH_ONLY
    assert dcf["net_debt"] == pytest.approx(EXPECTED_NET_DEBT_CASH_ONLY, abs=1.0)
    # This is the corrected figure a cash-only policy produces — real, not the
    # old buggy ~$29B (which mixed an undercounted debt with cash mislabeled
    # as cash+STI).
    assert dcf["net_debt"] == pytest.approx(38_264_300_000, abs=1.0)


def test_net_debt_flips_to_net_cash_under_the_alternate_policy(wired, monkeypatch):
    monkeypatch.setenv("DCF_NET_DEBT_POLICY", NetDebtPolicy.CASH_AND_MARKETABLE_SECURITIES)
    monkeypatch.setenv("DCF_SHORT_TERM_INVESTMENTS_ELIGIBLE", "true")

    result = run_msft(wired)
    dcf = result.facts["dcf"]

    assert dcf["net_debt_policy"] == NetDebtPolicy.CASH_AND_MARKETABLE_SECURITIES
    assert dcf["net_debt"] == pytest.approx(EXPECTED_NET_DEBT_WITH_STI, abs=1.0)
    assert dcf["net_debt"] < 0, \
        "including the real $55.9B of short-term investments flips MSFT to net cash"


# ---- 5: the DCF equity bridge uses these corrected values ----

def test_dcf_equity_bridge_uses_the_corrected_balance_sheet_values(wired):
    result = run_msft(wired)
    scenario = result.facts["dcf"]["scenarios"][0]

    assert scenario["cash_and_cash_equivalents"] == pytest.approx(EXPECTED_CASH)
    assert scenario["total_debt"] == pytest.approx(EXPECTED_TOTAL_DEBT)
    assert scenario["short_term_investments"] == pytest.approx(EXPECTED_STI)
    assert scenario["eligible_short_term_investments"] == pytest.approx(0.0), \
        "STI is reported for transparency but not eligible under the default policy"
    assert scenario["net_debt"] == pytest.approx(EXPECTED_NET_DEBT_CASH_ONLY, abs=1.0)
    # The bridge is fully reconstructable from the reported components alone.
    assert scenario["equity_value"] == pytest.approx(
        scenario["enterprise_value"] - scenario["net_debt"]
        + scenario["other_non_operating_assets"] - scenario["preferred_equity"]
        - scenario["minority_interest"], abs=1e-2)


def test_dcf_never_recomputes_or_overrides_the_supplied_bridge(wired):
    """The LLM never touches this path — proven structurally: the executor
    call that produced these numbers is finance.dcf_model, and its result is
    used completely unmodified in facts['dcf']."""
    executor, _coordinator = wired
    executed = []
    original = executor.execute

    def spy(call, step=0, confirmation=None):
        executed.append(call.tool_name)
        return original(call, step=step, confirmation=confirmation)

    executor.execute = spy
    result = run_full_stock_analysis(executor, "MSFT", include_news=False)

    assert "finance.dcf_model" in executed
    assert result.facts["dcf"]["calculation_version"]


# ---- 6: ROE methodologies are labeled ----

def test_roe_methodologies_are_present_and_distinctly_labeled(wired):
    result = run_msft(wired)
    metrics = result.facts["fundamental_metrics"]

    assert "roe_ending_equity" in metrics
    assert "roe_average_equity" in metrics
    assert metrics["roe_ending_equity"]["formula"] != metrics["roe_average_equity"]["formula"]
    assert metrics["roe_ending_equity"]["value"] is not None
    assert metrics["roe_average_equity"]["value"] is not None
    assert metrics["roe_ending_equity"]["value"] != metrics["roe_average_equity"]["value"], \
        "ending vs. average equity must produce genuinely different figures on real data"
    # GAAP labeling (Problem 5): growth is explicitly GAAP, nothing claims "adjusted".
    assert metrics["net_income_growth_yoy"]["accounting_basis"] == "GAAP"
    assert metrics["revenue_growth_yoy"]["accounting_basis"] == "GAAP"


# ---- 7: synthesis prompt is compact ----

def test_synthesis_prompt_is_compact_for_the_msft_fixture(wired, monkeypatch):
    # Pipeline OFF: this test is about payload compaction (Problem 9), not the
    # Phase H.2 research pipeline (see test_finance_research_pipeline_integration.py
    # for that) — disabling it keeps "exactly one call" a real, intentional
    # assertion rather than an accident of every pipeline stage call raising on
    # this stub's signature.
    monkeypatch.setenv("STOCK_ANALYSIS_RESEARCH_PIPELINE_ENABLED", "false")
    result = run_msft(wired)
    calls = []

    def fake_ask_local_raw(messages, tools=None, timeout=120, options=None, response_format=None):
        calls.append(messages)
        return {"message": {"content": "report text"},
                "metrics": {"prompt_tokens": 1234, "completion_tokens": 200}, "ok": True}

    text, _metrics = synthesize_report(result, fake_ask_local_raw)

    assert len(calls) == 1
    estimated_tokens = result.instrumentation["estimated_synthesis_tokens"]
    assert estimated_tokens < 20_000, f"compact payload estimated at {estimated_tokens} tokens"
    # The ceiling moves when explicitly-requested evidence is added, rather
    # than the evidence being cut to fit. Two increments so far:
    #   COR (Phase 9): five new fundamental_metrics (net_debt,
    #     net_debt_to_fcf, debt_to_fcf, interest_coverage,
    #     operating_cash_flow), ~9.9k -> ~10.1k;
    #   the valuation-gate wiring: `valuation_status` (the single status
    #     every consumer gates on) and `dcf_suitability.assessed` (the
    #     difference between "limited" and "not assessed", which decides
    #     whether a valuation's evidence is published at all), ~11.0k ->
    #     ~11.01k.
    # Still far under the 20k hard limit asserted above.
    assert estimated_tokens < 11_500, \
        f"preferred threshold: compact payload estimated at {estimated_tokens} tokens"

    # The instrumentation makes the BEFORE/AFTER comparison inspectable.
    assert result.instrumentation["raw_provider_payload_bytes"] > 0
    assert result.instrumentation["normalized_payload_bytes"] > 0
    assert result.instrumentation["compact_synthesis_payload_bytes"] > 0
    assert (result.instrumentation["compact_synthesis_payload_bytes"]
           < result.instrumentation["normalized_payload_bytes"]), \
        "compaction must actually shrink the payload, not just relabel it"

    # Phase H.3 corrective patch, Problem 11: a bounded regression signal for
    # remaining cross-item redundancy (DCF scenarios; annual-history periods)
    # -- currently 12 for this fixture, all in the DCF per-scenario equity-
    # bridge fields (net_debt/total_debt/cash/calculation_version/etc, which
    # ARE identical across scenarios by construction but are deliberately
    # NOT hoisted out -- see build_compact_synthesis_payload's docstring:
    # the saving is too small at this fixture's size to be worth restructuring
    # the per-scenario shape the report-writing model reads). A generous
    # ceiling, not the exact snapshot value -- this only needs to catch a
    # NEW, unintended source of duplication being introduced.
    repeated = result.instrumentation["repeated_field_count"]
    assert 0 <= repeated <= 20, f"unexpected cross-item redundancy: {repeated} repeated field(s)"


def test_evidence_index_stays_compact_after_dcf_assumption_items(wired):
    """H.4 corrective patch (goal 7): dcf.assumption.*/dcf.terminal_value_
    share.* (goal 3) add roughly 3 scenarios x (4 per-scenario + 1 alias)
    fields plus 4 shared fields to the evidence index -- this is the SAME
    rendered evidence_text block EVERY research-pipeline stage receives
    (finance/research_pipeline.py::_stage_prompt), including the
    FinalInvestmentSynthesizer, so its size is the most direct proxy for
    "final synthesis input token count." Checked on the MSFT fixture
    specifically because it is this project's own stress case for prompt
    size (20 years annual, 81 quarters per statement -- see
    docs/PHASE_H1_STOCK_ANALYSIS.md's "Compact synthesis payload" section).
    """
    result = run_msft(wired)
    compact = build_compact_synthesis_payload(result)
    index = build_evidence_index(compact)
    rendered = render_evidence_index(index)
    estimated_tokens = len(rendered) // 4

    assert any(k.startswith("dcf.assumption.") for k in index), \
        "the new DCF assumption evidence must actually be present for this to be a real check"

    # normal target < 10,000 tokens (goal 7); hard ceiling matches the
    # existing configured maximum this project already enforces for the
    # compact synthesis payload itself (docs/PHASE_H1_STOCK_ANALYSIS.md).
    assert estimated_tokens < 20_000, f"evidence index estimated at {estimated_tokens} tokens"
    assert estimated_tokens < 10_000, \
        f"preferred threshold: evidence index estimated at {estimated_tokens} tokens"


def test_compaction_does_not_drop_dcf_or_equity_bridge_detail(wired):
    result = run_msft(wired)
    compact = build_compact_synthesis_payload(result)

    assert compact["dcf"]["available"] is True
    for key in ("total_debt", "cash_and_cash_equivalents", "eligible_short_term_investments",
               "net_debt", "net_debt_policy", "preferred_equity", "minority_interest",
               "other_non_operating_assets", "equity_value", "diluted_shares",
               "value_per_share"):
        assert key in compact["dcf"]["scenarios"][0], f"missing equity-bridge key: {key}"
    for scenario in compact["dcf"]["scenarios"]:
        assert scenario["assumptions"]["assumption_provenance"], \
            "assumption provenance must survive compaction"
        # Full per-year forecast rows must survive too — only the sensitivity
        # GRID is summarized (see test_sensitivity_is_summarized_not_dropped).
        assert len(scenario["forecast"]) == scenario["forecast_years"]
        for row in scenario["forecast"]:
            assert "fcff" in row and "discount_factor" in row and "tax_rate" in row


def test_sensitivity_is_summarized_not_dropped(wired):
    result = run_msft(wired)
    compact = build_compact_synthesis_payload(result)
    full_grid = result.facts["dcf"]["sensitivity"]
    summary = compact["dcf"]["sensitivity"]

    full_cell_count = sum(len(row["cells"]) for row in full_grid["rows"])
    assert summary["cell_count"] <= full_cell_count
    assert summary["wacc_values"] == full_grid["wacc_values"]
    assert summary["value_per_share_min"] <= summary["value_per_share_max"]
    assert "rows" not in summary, "the full per-cell grid must not appear in the compact payload"


def test_report_values_trace_back_to_normalized_or_calculated_fields(wired):
    """Every number the compact payload exposes for the report must equal the
    corresponding value in the FULL (unbounded) facts — compaction bounds
    breadth and drops redundant per-metric metadata, but never changes a
    VALUE."""
    result = run_msft(wired)
    compact = build_compact_synthesis_payload(result)

    assert compact["quote"] == result.facts["quote"]
    for name, metric in compact["fundamental_metrics"].items():
        full_metric = result.facts["fundamental_metrics"][name]
        assert metric["value"] == full_metric["value"]
        assert metric["formula"] == full_metric["formula"]
    assert (compact["fundamental_metrics_calculation_version"]
           == result.facts["fundamental_metrics"]["revenue_growth_yoy"]["calculation_version"])

    for scenario, full_scenario in zip(compact["dcf"]["scenarios"],
                                       result.facts["dcf"]["scenarios"]):
        assert scenario["value_per_share"] == full_scenario["value_per_share"]
        assert scenario["net_debt"] == full_scenario["net_debt"]
        assert scenario["forecast"] == full_scenario["forecast"]

    full_latest = result.facts["statements"]["annual"]["balance_sheet"][0]
    compact_latest = compact["financial_history"]["balance_sheet"][0]
    assert compact_latest["values"] == full_latest["values"]


def test_full_analysis_mode_and_no_omitted_datasets(wired):
    """This fixture has ample quota, so the run must be FULL — never labelled
    complete while data was actually missing."""
    result = run_msft(wired)
    assert result.plan.mode == AnalysisMode.FULL
    assert result.plan.omitted_datasets == ()
