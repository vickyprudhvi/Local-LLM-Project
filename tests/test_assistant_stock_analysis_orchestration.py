"""Phase H.1 orchestration — a natural-language full-stock-analysis request must
bypass Phase B tool selection entirely and run the existing FullStockAnalysis
workflow directly, through the SAME ToolRegistry/ToolExecutor Phase B uses.

Regression target: "Give me a full stock analysis of TSLA" previously fell
through to the generic tool loop, which called finance.stock_quote plus browser
tools ad hoc and never touched finance.dcf_model. These tests prove that no
longer happens, using a scripted market-data client (no real network) and a
stubbed local-LLM call (no real Ollama) — "live-style" in that they exercise the
real detection, the real workflow, and the real ToolExecutor end to end.
"""

import json

import pytest

import assistant
import tool_loop
import tools.finance_tools as finance_tools
from finance.cache import MarketDataCache, Origin
from finance.coordinator import MarketDataRequestCoordinator
from finance.quota import AlphaVantageQuotaLedger
from finance.workflow import DCF_TOOL_NAME, AnalysisMode
from tests.test_finance_cache import FakeClock
from tests.test_finance_workflow import PAYLOAD_BY_DATASET, DatasetClient
from tools.base import BaseTool
from tools.executor import ToolExecutor
from tools.models import ToolPermission
from tools.registry import ToolRegistry

SECRET = "TEST_KEY_NOT_REAL"


class SpyTool(BaseTool):
    """A stand-in for an unrelated tool (browser.search, github.clone_repository,
    ...) that records every call it receives, so a test can assert it was never
    invoked during a stock-analysis turn."""

    permission = ToolPermission.READ

    def __init__(self, name):
        self.name = name
        self.description = f"spy stand-in for {name}"
        self.input_schema = {"type": "object", "properties": {}}
        self.calls = []

    def execute(self, arguments):
        self.calls.append(arguments)
        return {"spy": True}


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def orchestrated(monkeypatch, clock, tmp_path):
    """Wire tool_loop.REGISTRY/EXECUTOR to an isolated registry: real finance
    tools backed by a scripted provider client, plus spy stand-ins for the
    tools a generic Phase B run picked in the reported regression. Also stubs
    the local-LLM call and makes tool_loop.run_local_tool_loop (Phase B) raise
    if it is ever entered, so "Phase B never ran" is enforced, not assumed.

    Uses real tmp_path SQLite files, not ":memory:" — MarketDataCache opens a
    fresh connection per call, and an in-memory SQLite DB is private to the
    connection that created it, so ":memory:" here would silently lose its
    schema between calls (see tests/test_finance_workflow.py's own fixture).
    """
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", SECRET)
    # This file tests ORCHESTRATION (routing, Phase B bypass, tool selection) —
    # the Phase H.2 research pipeline is separately covered in
    # test_finance_research_pipeline_integration.py. Disabling it here keeps
    # llm_calls/metrics assertions below about the ONE report-synthesis call,
    # not entangled with a feature this file isn't about.
    monkeypatch.setenv("STOCK_ANALYSIS_RESEARCH_PIPELINE_ENABLED", "false")
    # H.4 corrective patch: report_detail now defaults to "compact", which
    # renders without ever calling ask_local_raw for the narrative — this
    # file's llm_calls/metrics/"STUBBED REPORT TEXT" assertions are about
    # ORCHESTRATION (Phase B bypass, tool call ordering), not report
    # rendering, so it is pinned to "full" here, same reasoning as the
    # pipeline flag directly above. Compact rendering itself is covered by
    # tests/test_finance_report_compaction.py.
    monkeypatch.setenv("STOCK_ANALYSIS_REPORT_DETAIL", "full")

    cache = MarketDataCache(path=str(tmp_path / "market.sqlite3"), clock=clock)
    ledger = AlphaVantageQuotaLedger(path=str(tmp_path / "quota.sqlite3"), clock=clock,
                                     daily_limit=100)
    client = DatasetClient()
    coordinator = MarketDataRequestCoordinator(
        cache=cache, ledger=ledger, client=client, clock=clock,
        sleeper=lambda _s: None, jitter=lambda: 0.5)
    finance_tools.set_coordinator(coordinator)

    registry = ToolRegistry()
    for tool_cls in finance_tools.ALL_FINANCE_TOOL_CLASSES:
        registry.register(tool_cls())
    spies = {}
    for name in ("browser.search", "browser.fetch_page", "github.clone_repository"):
        spy = SpyTool(name)
        registry.register(spy)
        spies[name] = spy
    executor = ToolExecutor(registry)

    monkeypatch.setattr(tool_loop, "REGISTRY", registry)
    monkeypatch.setattr(tool_loop, "EXECUTOR", executor)

    def phase_b_must_not_run(*args, **kwargs):
        raise AssertionError(
            "tool_loop.run_local_tool_loop (Phase B) must not run for a "
            "detected full-stock-analysis request")

    monkeypatch.setattr(tool_loop, "run_local_tool_loop", phase_b_must_not_run)

    llm_calls = []

    def fake_ask_local_raw(messages, tools=None, timeout=120, options=None, response_format=None):
        llm_calls.append(messages)
        return {"message": {"content": "STUBBED REPORT TEXT"},
                "metrics": {"prompt_tokens": 11, "completion_tokens": 22}, "ok": True}

    monkeypatch.setattr(assistant, "ask_local_raw", fake_ask_local_raw)

    yield {"registry": registry, "executor": executor, "coordinator": coordinator,
           "spies": spies, "llm_calls": llm_calls, "client": client}
    finance_tools.set_coordinator(None)


def _executed_tool_names(executor):
    """Spy on ToolExecutor.execute for the duration of one call, returning the
    ordered list of tool names it actually ran."""
    executed = []
    original = executor.execute

    def spy(call, step=0, confirmation=None):
        executed.append(call.tool_name)
        return original(call, step=step, confirmation=confirmation)

    executor.execute = spy
    return executed


# ---- 1: natural-language detection selects the H1 workflow ----

def test_full_stock_analysis_request_selects_the_h1_workflow(orchestrated):
    executor = orchestrated["executor"]
    executed = _executed_tool_names(executor)

    reply, metrics, pending = assistant._process_local_request_with_capability_selection(
        None, None, "Give me a full stock analysis of TSLA", "prompt", [], "sys", set())

    assert pending is None
    assert "STUBBED REPORT TEXT" in reply
    assert metrics == {"prompt_tokens": 11, "completion_tokens": 22}
    assert any(name.startswith("finance.") for name in executed)


# ---- 2: generic Phase B tool selection is never entered ----

def test_generic_phase_b_selection_is_not_entered_first(orchestrated):
    # tool_loop.run_local_tool_loop is monkeypatched to raise if called at all;
    # simply not raising here IS the proof.
    assistant._process_local_request_with_capability_selection(
        None, None, "Give me a full stock analysis of TSLA", "prompt", [], "sys", set())


# ---- 3: browser tools are never called ----

def test_browser_tools_are_never_called(orchestrated):
    assistant._process_local_request_with_capability_selection(
        None, None, "Give me a full stock analysis of TSLA", "prompt", [], "sys", set())

    assert orchestrated["spies"]["browser.search"].calls == []
    assert orchestrated["spies"]["browser.fetch_page"].calls == []


# ---- 4: github.clone_repository is never shortlisted / called ----

def test_github_clone_is_never_called(orchestrated):
    assistant._process_local_request_with_capability_selection(
        None, None, "Give me a full stock analysis of TSLA", "prompt", [], "sys", set())

    assert orchestrated["spies"]["github.clone_repository"].calls == []


# ---- 5: finance.dcf_model is called on a successful full run ----

def test_dcf_model_is_called_on_a_successful_full_run(orchestrated):
    executor = orchestrated["executor"]
    executed = _executed_tool_names(executor)

    reply, _metrics, _pending = assistant._process_local_request_with_capability_selection(
        None, None, "Give me a full stock analysis of TSLA", "prompt", [], "sys", set())

    assert DCF_TOOL_NAME in executed
    assert "[Full stock analysis" in reply


def test_the_llm_receives_structured_facts_not_a_free_form_prompt(orchestrated):
    assistant._process_local_request_with_capability_selection(
        None, None, "Give me a full stock analysis of TSLA", "prompt", [], "sys", set())

    llm_calls = orchestrated["llm_calls"]
    assert len(llm_calls) == 1
    user_message = llm_calls[0][-1]["content"]
    # Phase H.1 accuracy patch: the LLM receives the BOUNDED compact payload
    # (finance.workflow.build_compact_synthesis_payload), not the full,
    # unbounded AnalysisResult.to_dict() — so fields sit at the top level, not
    # nested under "facts".
    payload = json.loads(user_message.split("structured data:\n\n", 1)[1])
    assert payload["symbol"] == "TSLA"
    assert payload["dcf"]["available"] is True
    assert "data_provenance" in payload


# ---- 6: provider/quota failures produce controlled partial/error results ----

def test_quota_exhaustion_produces_a_controlled_result_not_a_crash(orchestrated):
    orchestrated["coordinator"].ledger.record_rate_limit(exhausted=True)
    executor = orchestrated["executor"]
    executed = _executed_tool_names(executor)

    reply, metrics, pending = assistant._process_local_request_with_capability_selection(
        None, None, "Give me a full stock analysis of TSLA", "prompt", [], "sys", set())

    assert pending is None
    assert DCF_TOOL_NAME not in executed, \
        "no valuation may run when the workflow could not gather any data"
    assert metrics == {"prompt_tokens": 0, "completion_tokens": 0}
    assert not orchestrated["llm_calls"], "a STOPPED plan must not reach the LLM"
    assert "quota" in reply.lower() or "no external" in reply.lower()


def test_a_provider_failure_still_yields_a_labelled_partial_report(orchestrated):
    """One dataset failing (e.g. a premium-entitlement rejection) must not
    abort the whole analysis — the report is still produced, with the gap
    visible in errors/warnings, and the DCF still runs from what IS available."""
    from finance.provider import ProviderResponse
    from tools.base import ToolFailure
    from tools.models import MARKET_DATA_ENTITLEMENT_REQUIRED

    client = orchestrated["client"]
    original_fetch = client.fetch

    def flaky_fetch(dataset, arguments):
        if dataset.dataset_id == "earnings":
            raise ToolFailure(MARKET_DATA_ENTITLEMENT_REQUIRED, "premium only")
        return original_fetch(dataset, arguments)

    client.fetch = flaky_fetch
    executor = orchestrated["executor"]
    executed = _executed_tool_names(executor)

    reply, _metrics, _pending = assistant._process_local_request_with_capability_selection(
        None, None, "Give me a full stock analysis of TSLA", "prompt", [], "sys", set())

    assert DCF_TOOL_NAME in executed, "a partial dataset failure must not block the DCF"
    assert "STUBBED REPORT TEXT" in reply


# ---- 7: unrelated finance questions still use the ordinary tool loop ----

def test_unrelated_finance_question_is_not_detected(orchestrated):
    from finance.workflow import detect_full_stock_analysis_request

    assert detect_full_stock_analysis_request("What is AAPL trading at") is None
    assert detect_full_stock_analysis_request("What's the P/E ratio mean?") is None


def test_plain_quote_request_falls_through_to_phase_b(orchestrated, monkeypatch):
    """With detection returning None, the early-return branch must not fire —
    proven by observing the (still-stubbed-to-raise) Phase B entry point is
    reached, i.e. THIS call raises the sentinel AssertionError from the fixture
    rather than silently short-circuiting like the stock-analysis path does."""
    with pytest.raises(AssertionError, match="Phase B"):
        assistant._process_local_request_with_capability_selection(
            None, None, "What is AAPL trading at", "prompt", [], "sys", set())


# ---- symbol validation stays intact ----

def test_a_request_naming_no_ticker_is_not_detected():
    from finance.workflow import detect_full_stock_analysis_request

    assert detect_full_stock_analysis_request("Full stock analysis of Microsoft") is None
    assert detect_full_stock_analysis_request("give me a full stock analysis") is None


# ---- recommendation reintroduction: advice-shaped phrasing also detected ----

def test_should_i_buy_ticker_phrasing_is_detected():
    from finance.workflow import detect_full_stock_analysis_request

    assert detect_full_stock_analysis_request("Should I buy AAPL?") == "AAPL"
    assert detect_full_stock_analysis_request("should I sell TSLA") == "TSLA"
    assert detect_full_stock_analysis_request("Is NVDA a good buy right now?") == "NVDA"
    assert detect_full_stock_analysis_request("buy or sell GOOGL") == "GOOGL"


def test_should_i_buy_phrasing_with_no_ticker_is_not_detected():
    from finance.workflow import detect_full_stock_analysis_request

    assert detect_full_stock_analysis_request("Should I buy a new laptop?") is None
    assert detect_full_stock_analysis_request("should I buy milk") is None


def test_slash_command_is_detected(orchestrated):
    executor = orchestrated["executor"]
    executed = _executed_tool_names(executor)

    reply, _metrics, _pending = assistant._process_local_request_with_capability_selection(
        None, None, "/FullStockAnalysis AAPL", "prompt", [], "sys", set())

    assert DCF_TOOL_NAME in executed
    assert "[Full stock analysis — AAPL]" in reply
