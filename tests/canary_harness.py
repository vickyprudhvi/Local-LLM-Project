"""Part 27 — one canary run, crossing every boundary.

A canary is worth nothing as a unit test. The failures this suite exists to
catch all happened BETWEEN components: a value that was correct where it was
computed and wrong where it was consumed. So the harness runs the REAL
`run_full_stock_analysis` against fixture provider clients and hands back
every stage's output at once:

    normalized fixture -> canonical state -> DCF packet -> valuation gate
    -> canonical research evidence -> report model

Only the two network clients are substituted. Normalization, the freshness
planner, TTM construction, business-model classification, the validity
cascade, packet construction, the evidence gate and the report model are all
the production code.
"""

import json
from dataclasses import dataclass
from typing import Dict, Optional

import pytest

import tools.finance_tools as finance_tools
from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.evidence import build_evidence_index, valuation_evidence_status
from finance.provider import AlphaVantageClient, ProviderResponse
from finance.quota import AlphaVantageQuotaLedger, QuotaLedger
from finance.report_model import build_stock_analysis_report_model
from finance.sec_datasets import resolve_sec_dataset
from finance.workflow import (
    AnalysisResult,
    build_compact_synthesis_payload,
    render_compact_report,
    run_full_stock_analysis,
)
from finance.yahoo_datasets import resolve_yahoo_dataset
from tests.fixtures.canary_company import build_fixture
from tests.test_finance_cache import FakeClock
from tools.base import ToolFailure
from tools.executor import ToolExecutor
from tools.models import MARKET_DATA_INVALID_SYMBOL
from tools.registry import ToolRegistry


class _FixtureClient:
    def __init__(self, provider_id, payloads, documents=None):
        self.provider_id = provider_id
        self._payloads = payloads
        self._documents = documents or {}
        self.calls = []

    def fetch(self, dataset, arguments):
        self.calls.append(dataset.dataset_id)
        if dataset.dataset_id == "filing_document":
            key = f"{arguments.get('accession')}/{arguments.get('document')}"
            text = self._documents.get(key)
            if text is None:
                raise ToolFailure(MARKET_DATA_INVALID_SYMBOL,
                                  f"no fixture document for {key}")
            return ProviderResponse(
                payload={"document_text": text, "byte_count": len(text),
                         "accession": arguments.get("accession"),
                         "document": arguments.get("document"), "truncated": False},
                provider_metadata={}, byte_count=len(text))
        payload = self._payloads.get(dataset.dataset_id)
        if payload is None:
            raise ToolFailure(MARKET_DATA_INVALID_SYMBOL,
                              f"no fixture data for {dataset.dataset_id}")
        return ProviderResponse(payload=payload, provider_metadata={},
                                byte_count=len(json.dumps(payload, default=str)))


@dataclass
class CanaryRun:
    """Every boundary's output for one canary, in one object.

    Deliberately not lazily recomputed: an assertion in one test must see the
    same objects an assertion in another test saw, or the suite is checking
    consistency it did not actually establish.
    """

    company: object
    result: AnalysisResult
    compact: dict
    evidence_index: dict
    report_model: object

    @property
    def facts(self) -> dict:
        return self.result.facts

    @property
    def state(self) -> dict:
        return self.facts.get("current_financial_state") or {}

    @property
    def canonical_current(self) -> dict:
        return (self.facts.get("canonical_evidence") or {}).get("current") or {}

    @property
    def business_model(self) -> dict:
        return self.facts.get("business_model") or {}

    @property
    def dcf(self) -> dict:
        return self.facts.get("dcf") or {}

    @property
    def packet(self) -> Optional[dict]:
        return self.facts.get("dcf_input_packet")

    @property
    def packet_failure(self) -> Optional[dict]:
        return self.facts.get("dcf_packet_failure")

    @property
    def valuation_status(self) -> str:
        return self.facts.get("valuation_status")

    @property
    def gate_status(self) -> str:
        """What the EVIDENCE INDEX decided, independently of `facts`.

        Asserted separately from `valuation_status` on purpose: the point of
        the wiring is that the two agree, and a helper that returned one for
        both could not tell you when they stopped agreeing.
        """
        return valuation_evidence_status(self.compact)

    def evidence_ids(self, prefix: str):
        return sorted(k for k in self.evidence_index if k.startswith(prefix))

    def report_text(self) -> str:
        return render_compact_report(self.result, self.compact, None)


def wire_canary(company, tmp_path, monkeypatch, clock=None) -> ToolExecutor:
    """A real ToolExecutor over fixture clients for one synthetic issuer."""
    clock = clock or FakeClock()
    fixture = build_fixture(company)

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "")
    monkeypatch.setenv("YAHOO_FINANCE_ENABLED", "true")
    monkeypatch.setenv("YAHOO_PERSONAL_USE_ACKNOWLEDGED", "true")
    monkeypatch.setenv("SEC_EDGAR_ENABLED", "true")
    monkeypatch.setenv("SEC_USER_AGENT", "TestApp/1.0 (contact: t@example.com)")
    for var in ("FINANCE_QUOTE_PROVIDER", "FINANCE_PRICE_HISTORY_PROVIDER",
                "FINANCE_CORPORATE_ACTIONS_PROVIDER", "FINANCE_COMPANY_PROFILE_PROVIDER",
                "FINANCE_ANALYST_ESTIMATES_PROVIDER"):
        monkeypatch.setenv(var, "yahoo")
    monkeypatch.setenv("FINANCE_US_FUNDAMENTALS_PROVIDER", "sec")
    monkeypatch.setenv("GUIDANCE_INGESTION_ENABLED", "true")

    yahoo = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "y_m.sqlite3"), clock=clock),
        ledger=QuotaLedger(path=str(tmp_path / "y_q.sqlite3"), clock=clock,
                           daily_limit=1_000_000),
        client=_FixtureClient("yahoo", fixture["yahoo"]), provider_id="yahoo",
        dataset_resolver=resolve_yahoo_dataset, clock=clock,
        sleeper=lambda _s: None, min_request_interval_ms=0)
    finance_tools.set_yahoo_coordinator(yahoo)

    sec = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "s_m.sqlite3"), clock=clock),
        ledger=QuotaLedger(path=str(tmp_path / "s_q.sqlite3"), clock=clock,
                           daily_limit=1_000_000),
        client=_FixtureClient("sec", fixture["sec"], fixture["sec"]["filing_documents"]),
        provider_id="sec", dataset_resolver=resolve_sec_dataset, clock=clock,
        sleeper=lambda _s: None, min_request_interval_ms=0)
    finance_tools.set_sec_coordinator(sec)

    finance_tools.set_coordinator(MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "a_m.sqlite3"), clock=clock),
        ledger=AlphaVantageQuotaLedger(path=str(tmp_path / "a_q.sqlite3"), clock=clock,
                                       daily_limit=100),
        client=AlphaVantageClient(), clock=clock, sleeper=lambda _s: None,
        jitter=lambda: 0.5))

    registry = ToolRegistry()
    for tool_cls in finance_tools.ALL_FINANCE_TOOL_CLASSES:
        registry.register(tool_cls())
    return ToolExecutor(registry)


def run_canary(company, tmp_path, monkeypatch) -> CanaryRun:
    """Part 27's flow, end to end, for one canary."""
    executor = wire_canary(company, tmp_path, monkeypatch)
    try:
        result = run_full_stock_analysis(executor, company.symbol)
        compact = build_compact_synthesis_payload(result)
        return CanaryRun(
            company=company,
            result=result,
            compact=compact,
            evidence_index=build_evidence_index(compact),
            report_model=build_stock_analysis_report_model(result, compact, None),
        )
    finally:
        finance_tools.set_coordinator(None)
        finance_tools.set_yahoo_coordinator(None)
        finance_tools.set_sec_coordinator(None)
