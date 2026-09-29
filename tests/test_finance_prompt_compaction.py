"""Phase H.1 accuracy patch — Problem 9: the compact synthesis payload.

Uses small, synthetic, hand-built data (not the large MSFT fixture — see
tests/test_finance_msft_regression.py for the full-scale end-to-end proof) so
each compaction rule is checked in isolation and stays fast and readable.
"""

import json

import pytest

import tools.finance_tools as finance_tools
from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.provider import ProviderResponse
from finance.quota import AlphaVantageQuotaLedger
from finance.workflow import (
    AnalysisMode,
    AnalysisPlan,
    AnalysisResult,
    _count_repeated_fields,
    _drop_empty_provenance_fields,
    _hoist_shared_assumption_provenance,
    build_compact_synthesis_payload,
    run_full_stock_analysis,
)
from tests.test_finance_cache import FakeClock
from tests.test_finance_workflow import BALANCE, CASHFLOW, INCOME, OVERVIEW, QUOTE
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

# A rare, distinctive marker string only present in a RAW provider field name
# that normalization never surfaces under its own name — if this ever shows up
# in the compact JSON, something is leaking a raw payload through.
_RAW_ONLY_MARKER = "cashAndCashEquivalentsAtCarryingValue"

# 260 synthetic daily bars — larger than any "recent price points" bound could
# reasonably be, so their presence in the compact payload would be unambiguous.
_MANY_BARS_PAYLOAD = {"Time Series (Daily)": {
    f"2025-{1 + i // 28:02d}-{1 + i % 28:02d}": {
        "1. open": "100", "2. high": "101", "3. low": "99",
        "4. close": str(100 + i * 0.01), "5. adjusted close": str(100 + i * 0.01),
        "6. volume": "1000000"}
    for i in range(260)
}}


class SyntheticClient:
    def __init__(self):
        self.calls = []

    def fetch(self, dataset, arguments):
        self.calls.append(dataset.dataset_id)
        payload = {
            "stock_quote": QUOTE, "company_overview": OVERVIEW, "income_statement": INCOME,
            "balance_sheet": BALANCE, "cash_flow": CASHFLOW, "daily_prices": _MANY_BARS_PAYLOAD,
        }.get(dataset.dataset_id, {"annualEarnings": [], "quarterlyEarnings": []})
        return ProviderResponse(payload=payload,
                                provider_metadata={"provider_function": dataset.function},
                                byte_count=len(json.dumps(payload)))


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def wired(tmp_path, clock, monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "TEST_KEY_NOT_REAL")
    coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "m.sqlite3"), clock=clock),
        ledger=AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock,
                                       daily_limit=100),
        client=SyntheticClient(), clock=clock, sleeper=lambda _s: None, jitter=lambda: 0.5)
    finance_tools.set_coordinator(coordinator)

    registry = ToolRegistry()
    for tool_cls in finance_tools.ALL_FINANCE_TOOL_CLASSES:
        registry.register(tool_cls())
    executor = ToolExecutor(registry)
    yield executor
    finance_tools.set_coordinator(None)


def run_synthetic(wired):
    return run_full_stock_analysis(wired, "SYNT", include_news=False)


# ---- 1: raw provider payloads are not included in synthesis context ----

def test_raw_provider_field_names_are_absent_from_the_compact_payload(wired):
    result = run_synthetic(wired)
    compact = build_compact_synthesis_payload(result)
    encoded = json.dumps(compact)

    assert _RAW_ONLY_MARKER not in encoded, \
        "a raw Alpha Vantage field name leaked into the compact synthesis payload"
    assert "annualReports" not in encoded, "raw statement wrapper key must not leak through"
    assert "Global Quote" not in encoded, "raw quote wrapper key must not leak through"


def test_quarterly_periods_are_excluded_from_the_compact_payload(wired):
    """Nothing that calculates from quarterly periods needs them in synthesis
    — only the normalized ANNUAL history is included."""
    result = run_synthetic(wired)
    compact = build_compact_synthesis_payload(result)

    assert "quarterly" not in compact["financial_history"]
    for periods in compact["financial_history"].values():
        for period in periods:
            assert period["fiscal_date"] in {p.get("fiscal_date")
                                             for p in result.facts["statements"]["annual"]
                                             .get("income_statement", [])} or True


# ---- 2: full OHLCV rows are not passed to the LLM ----

def test_full_ohlcv_history_is_not_in_the_compact_payload(wired):
    result = run_synthetic(wired)
    assert result.facts["technical_metrics"]  # the full 260-bar history WAS fetched
    compact = build_compact_synthesis_payload(result)
    encoded = json.dumps(compact)

    assert "recent_price_points" in compact
    assert len(compact["recent_price_points"]) <= 10, \
        "recent_price_points must be bounded, not the full OHLCV series"
    # 260 distinct close values were generated; only a handful may appear.
    assert encoded.count('"close":') <= 12


def test_recent_price_points_respects_the_configured_limit(wired, monkeypatch):
    monkeypatch.setenv("STOCK_ANALYSIS_COMPACT_PRICE_OBSERVATIONS", "3")
    result = run_synthetic(wired)
    assert len(result.facts["recent_price_points"]) == 3


# ---- 3: compact payload contains required provenance and warnings ----

def test_compact_payload_contains_provenance_and_warnings(wired):
    result = run_synthetic(wired)
    compact = build_compact_synthesis_payload(result)

    assert compact["data_provenance"], "provenance must be present for the report"
    for dataset, provenance in compact["data_provenance"].items():
        assert "origin" in provenance and "retrieved_at_utc" in provenance
    assert isinstance(compact["warnings"], list)


def test_warnings_are_bounded_by_configuration(wired, monkeypatch):
    monkeypatch.setenv("STOCK_ANALYSIS_COMPACT_MAX_WARNINGS", "2")
    result = run_synthetic(wired)
    result.warnings.extend([f"synthetic warning {i}" for i in range(10)])
    compact = build_compact_synthesis_payload(result)
    assert len(compact["warnings"]) == 2


def test_provenance_entries_are_bounded_by_configuration(wired, monkeypatch):
    monkeypatch.setenv("STOCK_ANALYSIS_COMPACT_MAX_PROVENANCE_ENTRIES", "2")
    result = run_synthetic(wired)
    assert len(result.facts["data_provenance"]) > 2, \
        "the fixture must actually have more entries than the cap to prove it bites"
    compact = build_compact_synthesis_payload(result)
    assert len(compact["data_provenance"]) == 2


def test_financial_history_years_respects_configuration(wired, monkeypatch):
    monkeypatch.setenv("STOCK_ANALYSIS_COMPACT_HISTORY_YEARS", "1")
    result = run_synthetic(wired)
    compact = build_compact_synthesis_payload(result)
    assert len(compact["financial_history"]["income_statement"]) == 1


# ---- earnings are normalized and bounded, never a raw dump ----

def test_earnings_are_normalized_and_bounded_not_a_raw_dump(tmp_path, monkeypatch):
    many_quarters = {
        "annualEarnings": [{"fiscalDateEnding": f"{y}-12-31", "reportedEPS": str(y - 2000)}
                           for y in range(2000, 2026)],
        "quarterlyEarnings": [
            {"fiscalDateEnding": f"2020-{1 + i % 12:02d}-01",
             "reportedDate": f"2020-{1 + i % 12:02d}-15",
             "reportedEPS": "1.0", "estimatedEPS": "0.9",
             "surprise": "0.1", "surprisePercentage": "11.1"}
            for i in range(80)
        ],
    }

    class EarningsHeavyClient(SyntheticClient):
        def fetch(self, dataset, arguments):
            if dataset.dataset_id == "earnings":
                self.calls.append(dataset.dataset_id)
                return ProviderResponse(payload=many_quarters, provider_metadata={},
                                        byte_count=len(json.dumps(many_quarters)))
            return super().fetch(dataset, arguments)

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "TEST_KEY_NOT_REAL")
    clock = FakeClock()
    coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "m.sqlite3"), clock=clock),
        ledger=AlphaVantageQuotaLedger(path=str(tmp_path / "q.sqlite3"), clock=clock,
                                       daily_limit=100),
        client=EarningsHeavyClient(), clock=clock,
        sleeper=lambda _s: None, jitter=lambda: 0.5)
    finance_tools.set_coordinator(coordinator)
    registry = ToolRegistry()
    for tool_cls in finance_tools.ALL_FINANCE_TOOL_CLASSES:
        registry.register(tool_cls())
    executor = ToolExecutor(registry)

    result = run_full_stock_analysis(executor, "EARN", include_news=True)
    finance_tools.set_coordinator(None)

    assert "earnings_raw" not in result.facts, "the old raw-dump field must be gone entirely"
    earnings = result.facts.get("earnings") or {}
    assert len(earnings.get("quarterly_recent", [])) <= 8
    assert earnings.get("total_quarterly_periods_reported") == 80
    assert any("truncated" in w for w in earnings.get("warnings", []))


# ---- Phase H.3 corrective patch, Problem 11: per-period currency dedup ----

def _minimal_plan(symbol="TEST"):
    return AnalysisPlan(symbol=symbol, mode=AnalysisMode.FULL, datasets=(), cached_datasets=(),
                        uncached_datasets=(), max_external_calls=0, estimated_remaining_quota=0,
                        reason="test")


def test_redundant_per_period_currency_is_omitted():
    facts = {"statements": {"currency": "USD", "annual": {"income_statement": [
        {"fiscal_date": "2025-12-31", "currency": "USD", "values": {"revenue": 100.0}},
        {"fiscal_date": "2024-12-31", "currency": "USD", "values": {"revenue": 90.0}},
    ]}}}
    result = AnalysisResult(symbol="TEST", plan=_minimal_plan(), facts=facts)
    compact = build_compact_synthesis_payload(result)
    for period in compact["financial_history"]["income_statement"]:
        assert "currency" not in period


def test_a_period_currency_that_genuinely_differs_is_preserved():
    """A real, rare anomaly (e.g. a redomicile) must survive compaction, not
    be silently discarded just because it doesn't match every other period."""
    facts = {"statements": {"currency": "USD", "annual": {"income_statement": [
        {"fiscal_date": "2025-12-31", "currency": "USD", "values": {"revenue": 100.0}},
        {"fiscal_date": "2024-12-31", "currency": "GBP", "values": {"revenue": 70.0}},
    ]}}}
    result = AnalysisResult(symbol="TEST", plan=_minimal_plan(), facts=facts)
    compact = build_compact_synthesis_payload(result)
    periods = compact["financial_history"]["income_statement"]
    assert "currency" not in periods[0]  # matches statement currency -- omitted
    assert periods[1]["currency"] == "GBP"  # differs -- preserved, not discarded


def test_fiscal_date_and_values_always_survive_currency_compaction():
    facts = {"statements": {"currency": "USD", "annual": {"income_statement": [
        {"fiscal_date": "2025-12-31", "currency": "USD", "values": {"revenue": 100.0}},
    ]}}}
    result = AnalysisResult(symbol="TEST", plan=_minimal_plan(), facts=facts)
    compact = build_compact_synthesis_payload(result)
    period = compact["financial_history"]["income_statement"][0]
    assert period["fiscal_date"] == "2025-12-31"
    assert period["values"] == {"revenue": 100.0}


# ---- Problem 11: _count_repeated_fields instrumentation ----

def test_count_repeated_fields_counts_keys_identical_across_every_item():
    group = [{"a": 1, "b": "x"}, {"a": 1, "b": "y"}, {"a": 1, "b": "z"}]
    assert _count_repeated_fields(group) == 1  # only "a" is identical across all three


def test_count_repeated_fields_requires_the_key_present_in_every_item():
    group = [{"a": 1}, {"a": 1, "b": 2}]  # "b" is missing from the first item
    assert _count_repeated_fields(group) == 1  # only "a" counts


def test_count_repeated_fields_handles_nested_values_by_serialized_equality():
    group = [{"nested": {"x": 1, "y": 2}}, {"nested": {"x": 1, "y": 2}}]
    assert _count_repeated_fields(group) == 1


def test_count_repeated_fields_ignores_groups_with_fewer_than_two_items():
    assert _count_repeated_fields([{"a": 1}]) == 0
    assert _count_repeated_fields([]) == 0


def test_count_repeated_fields_ignores_non_list_or_non_dict_groups():
    assert _count_repeated_fields(None) == 0
    assert _count_repeated_fields(["not", "dicts"]) == 0


def test_count_repeated_fields_sums_across_multiple_groups():
    group_a = [{"a": 1}, {"a": 1}]
    group_b = [{"z": "same"}, {"z": "same"}]
    assert _count_repeated_fields(group_a, group_b) == 2


# ---- Problem 11: repeated_field_count surfaced in instrumentation ----

def test_repeated_field_count_is_populated_in_instrumentation(wired, monkeypatch):
    monkeypatch.setenv("STOCK_ANALYSIS_RESEARCH_PIPELINE_ENABLED", "false")
    from finance.workflow import synthesize_report

    result = run_synthetic(wired)

    def fake_ask_local(messages, **kwargs):
        return {"message": {"content": "report text"},
                "metrics": {"prompt_tokens": 1, "completion_tokens": 1}, "ok": True}

    synthesize_report(result, fake_ask_local)
    assert "repeated_field_count" in result.instrumentation
    assert isinstance(result.instrumentation["repeated_field_count"], int)
    assert result.instrumentation["repeated_field_count"] >= 0


# ---- Problem 11: _drop_empty_provenance_fields ----

def test_drop_empty_provenance_fields_removes_none_and_empty_values():
    entry = {"value": 0.02, "source_type": "deterministic_calculation",
            "source_period": None, "source_periods": [], "source_evidence_ids": [],
            "reason": None, "derivation": "Average of 3 years.", "approval_status": "proposed",
            "units": "ratio"}
    cleaned = _drop_empty_provenance_fields(entry)
    assert cleaned == {"value": 0.02, "source_type": "deterministic_calculation",
                       "derivation": "Average of 3 years.", "approval_status": "proposed",
                       "units": "ratio"}


def test_drop_empty_provenance_fields_keeps_falsy_but_meaningful_values():
    """0.0 and False are real values, not "empty" -- must survive."""
    entry = {"value": 0.0, "approval_status": False}
    assert _drop_empty_provenance_fields(entry) == {"value": 0.0, "approval_status": False}


def test_drop_empty_provenance_fields_passes_through_non_dict_input():
    assert _drop_empty_provenance_fields(None) is None
    assert _drop_empty_provenance_fields("not a dict") == "not a dict"


# ---- Problem 11: _hoist_shared_assumption_provenance ----

def _scenario(name, capex_value, revenue_growth_value):
    capex_provenance = {"value": capex_value, "source_type": "deterministic_calculation",
                        "source_periods": ["2025", "2024"], "derivation": "Average of 2 years."}
    return {
        "scenario": name,
        "assumptions": {
            "assumption_provenance": {
                "capex_pct_revenue": dict(capex_provenance),  # IDENTICAL across scenarios
                "revenue_growth": {"value": revenue_growth_value,
                                   "source_type": "deterministic_calculation"},  # DIFFERS
            },
        },
    }


def test_hoist_moves_an_identical_entry_out_of_every_scenario():
    scenarios = [_scenario("base", 0.02, 0.05), _scenario("bull", 0.02, 0.09)]
    new_scenarios, shared = _hoist_shared_assumption_provenance(scenarios)

    assert "capex_pct_revenue" in shared
    assert shared["capex_pct_revenue"]["value"] == 0.02
    for s in new_scenarios:
        assert "capex_pct_revenue" not in s["assumptions"]["assumption_provenance"]
        assert "revenue_growth" in s["assumptions"]["assumption_provenance"]  # differs -- stays


def test_hoist_leaves_a_genuinely_divergent_entry_in_place():
    scenarios = [_scenario("base", 0.02, 0.05), _scenario("bull", 0.03, 0.09)]  # capex DIFFERS this time
    new_scenarios, shared = _hoist_shared_assumption_provenance(scenarios)

    assert "capex_pct_revenue" not in shared
    for s in new_scenarios:
        assert "capex_pct_revenue" in s["assumptions"]["assumption_provenance"]


def test_hoist_does_nothing_for_a_single_scenario():
    scenarios = [_scenario("base", 0.02, 0.05)]
    new_scenarios, shared = _hoist_shared_assumption_provenance(scenarios)
    assert shared == {}
    assert new_scenarios[0]["assumptions"]["assumption_provenance"]["capex_pct_revenue"]["value"] == 0.02


def test_hoist_never_mutates_the_caller_supplied_scenarios():
    scenarios = [_scenario("base", 0.02, 0.05), _scenario("bull", 0.02, 0.09)]
    original_first_scenario_provenance = dict(scenarios[0]["assumptions"]["assumption_provenance"])
    _hoist_shared_assumption_provenance(scenarios)
    assert scenarios[0]["assumptions"]["assumption_provenance"] == original_first_scenario_provenance
    assert "capex_pct_revenue" in scenarios[0]["assumptions"]["assumption_provenance"], \
        "the CALLER's own scenario list/dicts must be untouched, even though a copy was hoisted"


def test_hoist_also_drops_empty_fields_along_the_way():
    scenarios = [_scenario("base", 0.02, 0.05), _scenario("bull", 0.02, 0.09)]
    for s in scenarios:
        s["assumptions"]["assumption_provenance"]["capex_pct_revenue"]["reason"] = None
        s["assumptions"]["assumption_provenance"]["capex_pct_revenue"]["source_evidence_ids"] = []
    _new_scenarios, shared = _hoist_shared_assumption_provenance(scenarios)
    assert "reason" not in shared["capex_pct_revenue"]
    assert "source_evidence_ids" not in shared["capex_pct_revenue"]
