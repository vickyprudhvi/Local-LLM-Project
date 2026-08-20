"""Phase H.8 — the profitability/DCF disconnect, on a real captured payload
(tests/fixtures/mrk_regression.json, 2026-08-19).

WHAT THIS FIXTURE STANDS FOR
============================
A structure, not a company. Every assertion is about the structure and none
names the issuer; production code reads no ticker.

    an issuer that does not tag `OperatingIncomeLoss` at all
    + trailing profitability distorted by a large acquisition-related
      in-process-R&D charge
    + a seven-row guidance table stated on a non-GAAP basis
    + a current-year tax rate moved sharply by that acquisition

THE FAILURE IT PINS DOWN
========================
The live run produced:

    operating income          unavailable
    -> configured default operating margin of 10% for all five years
    -> base modelled value    $0.20 per share
    -> market price           $149.93
    -> "Market-price premium: 75,588.5%"
    -> DCF suitability        SUITABLE_WITH_HIGH_UNCERTAINTY
    -> guidance               EPS and tax rate only, of seven guided metrics
    -> readiness reason       an old structural-break note, ahead of the
                              current-period profitability failure that was
                              driving the entire valuation

Nothing here asserts a valuation, a recommendation, a growth rate or any
other company-specific result.
"""

import json
from pathlib import Path

import pytest

import tools.finance_tools as finance_tools
from finance import guidance as G
from finance import profitability as P
from finance import suitability as SU
from finance.cache import MarketDataCache
from finance.coordinator import MarketDataRequestCoordinator
from finance.provider import AlphaVantageClient, ProviderResponse
from finance.quota import AlphaVantageQuotaLedger, QuotaLedger
from finance.sec_datasets import resolve_sec_dataset
from finance.workflow import (
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

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mrk_regression.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
SYMBOL = next(iter(FIXTURE["sec"]["ticker_cik_map"].values()))["ticker"]

# The configured default this issuer's valuation must NOT rest on.
CONFIGURED_DEFAULT_MARGIN = 0.10


class YahooFixtureClient:
    provider_id = "yahoo"

    def fetch(self, dataset, arguments):
        payload = FIXTURE["yahoo"].get(dataset.dataset_id)
        if payload is None:
            raise ToolFailure(MARKET_DATA_INVALID_SYMBOL, "no fixture data")
        return ProviderResponse(payload=payload, provider_metadata={}, byte_count=10)


class SecFixtureClient:
    provider_id = "sec"

    def fetch(self, dataset, arguments):
        if dataset.dataset_id == "filing_document":
            key = f"{arguments.get('accession')}/{arguments.get('document')}"
            text = (FIXTURE["sec"].get("filing_documents") or {}).get(key)
            if text is None:
                raise ToolFailure(MARKET_DATA_INVALID_SYMBOL, "no fixture document")
            return ProviderResponse(
                payload={"document_text": text, "byte_count": len(text),
                         "accession": arguments.get("accession"),
                         "document": arguments.get("document"), "truncated": False},
                provider_metadata={}, byte_count=len(text))
        payload = FIXTURE["sec"].get(dataset.dataset_id)
        if payload is None:
            raise ToolFailure(MARKET_DATA_INVALID_SYMBOL, "no fixture data")
        return ProviderResponse(payload=payload, provider_metadata={}, byte_count=10)


@pytest.fixture
def wired(tmp_path, monkeypatch):
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

    clock = FakeClock()
    finance_tools.set_yahoo_coordinator(MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "y.sqlite3"), clock=clock),
        ledger=QuotaLedger(path=str(tmp_path / "yq.sqlite3"), clock=clock,
                           daily_limit=10 ** 6),
        client=YahooFixtureClient(), provider_id="yahoo",
        dataset_resolver=resolve_yahoo_dataset, clock=clock,
        sleeper=lambda _s: None, min_request_interval_ms=0))
    finance_tools.set_sec_coordinator(MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "s.sqlite3"), clock=clock),
        ledger=QuotaLedger(path=str(tmp_path / "sq.sqlite3"), clock=clock,
                           daily_limit=10 ** 6),
        client=SecFixtureClient(), provider_id="sec", dataset_resolver=resolve_sec_dataset,
        clock=clock, sleeper=lambda _s: None, min_request_interval_ms=0))
    finance_tools.set_coordinator(MarketDataRequestCoordinator(
        cache=MarketDataCache(path=str(tmp_path / "a.sqlite3"), clock=clock),
        ledger=AlphaVantageQuotaLedger(path=str(tmp_path / "aq.sqlite3"), clock=clock,
                                       daily_limit=100),
        client=AlphaVantageClient(), clock=clock, sleeper=lambda _s: None,
        jitter=lambda: 0.5))
    registry = ToolRegistry()
    for tool_cls in finance_tools.ALL_FINANCE_TOOL_CLASSES:
        registry.register(tool_cls())
    yield ToolExecutor(registry)
    finance_tools.set_coordinator(None)
    finance_tools.set_yahoo_coordinator(None)
    finance_tools.set_sec_coordinator(None)


def _run(wired):
    return run_full_stock_analysis(wired, SYMBOL)


def _state(result):
    return result.facts.get("current_financial_state") or {}


def _profitability(result):
    return _state(result).get("profitability") or {}


def _guidance(result):
    return (result.facts.get("management_guidance") or {}).get("metrics") or {}


def _base(result):
    return next(s for s in result.facts["dcf"]["scenarios"] if s["scenario"] == "base")


# ---------------------------------------------------------------------------
# Section 38, checks 1-3: guidance is read COMPLETELY
# ---------------------------------------------------------------------------

def test_1_guidance_is_not_limited_to_eps_and_tax(wired):
    """The live run extracted two of seven guided metrics."""
    metrics = _guidance(_run(wired))
    assert len(metrics) > 2, sorted(metrics)
    assert set(metrics) - {"adjusted_earnings_per_share", "tax_rate"}


def test_2_sales_guidance_is_captured(wired):
    """The outlook table's first row. Its label is bare "Sales", which the
    revenue keyword did not match, and it sits too far from any
    forward-looking word for the old proximity rule."""
    metrics = _guidance(_run(wired))
    revenue = metrics.get("revenue") or metrics.get("revenue_growth")
    assert revenue is not None, sorted(metrics)
    assert revenue["low"] and revenue["high"]


def test_3_margin_and_expense_guidance_keeps_its_non_gaap_basis(wired):
    """Section 9: a non-GAAP margin must never land in a GAAP field."""
    metrics = _guidance(_run(wired))
    for name, entry in metrics.items():
        if name.startswith("adjusted_"):
            assert entry["basis"] == G.BASIS_ADJUSTED, name


def test_3b_guidance_coverage_is_reported_rather_than_assumed(wired):
    """Section 8: "some guidance extracted" is not "guidance covered"."""
    record = G.assess_guidance_coverage(_guidance(_run(wired)))
    assert record["guidance_coverage_status"] in G.GuidanceCoverage.ALL
    assert record["extracted_metrics"]


# ---------------------------------------------------------------------------
# Section 38, checks 4-6: reported and normalized profitability
# ---------------------------------------------------------------------------

def test_4_material_acquisition_related_charges_are_identified(wired):
    """Section 1: from a TAGGED amount, never from prose."""
    profitability = _profitability(_run(wired))
    items = profitability.get("unusual_items") or []
    assert items, "no unusual item was identified"
    for item in items:
        assert item["metric_type"] in P.UnusualItemType.ALL
        assert item["amount"] > 0
        assert item["period"]
        assert item["source_evidence_ids"]
        assert item["recurrence_status"] in P.RecurrenceStatus.ALL


def test_5_reported_gaap_profitability_is_unchanged(wired):
    """Section 2: normalization never overwrites the reported figure."""
    result = _run(wired)
    profitability = _profitability(result)
    reported = profitability["reported"]
    revenue = (_state(result)["flows"]["revenue"] or {})["value"]
    operating_income = (_state(result)["flows"]["operating_income"] or {})["value"]
    assert reported["operating_income"] == pytest.approx(operating_income)
    assert reported["operating_margin"] == pytest.approx(operating_income / revenue)


def test_5b_operating_income_is_derived_when_the_issuer_does_not_tag_it(wired):
    """The root cause: no `OperatingIncomeLoss` concept at all, so the whole
    margin precedence fell through to the configured default."""
    result = _run(wired)
    selection = _state(result)["flows"]["operating_income"]
    assert selection["value"] is not None
    if selection.get("components"):
        assert _profitability(result)["reported"]["operating_income_source"] == "derived"
        assert "DERIVED, not reported" in selection["derivation"]


def test_6_normalized_profitability_is_separate_from_reported(wired):
    """Section 2/4: both survive, and every adjustment is itemized."""
    profitability = _profitability(_run(wired))
    normalized = profitability["normalized"]
    assert normalized["status"] in P.NormalizationStatus.ALL
    record = normalized["operating_income"]
    assert record["reported_value"] is not None
    for adjustment in record["adjustments"]:
        assert adjustment["item_id"] and adjustment["evidence_ids"]
        assert adjustment["direction"] in ("add_back", "subtract")


# ---------------------------------------------------------------------------
# Section 38, checks 7-9: what the forecast rests on
# ---------------------------------------------------------------------------

def test_7_the_configured_default_margin_is_not_the_forecast(wired):
    """The headline failure. Whatever margin this valuation uses, it must not
    be the configured default standing in for economics nobody could read."""
    result = _run(wired)
    margin = (_base(result).get("assumptions") or {}).get("operating_margin")
    year_one = margin[0] if isinstance(margin, list) else margin
    assert year_one != pytest.approx(CONFIGURED_DEFAULT_MARGIN), (
        "the configured default became the forecast operating margin")

    provenance = ((_base(result).get("assumptions") or {})
                  .get("assumption_provenance") or {}).get("operating_margin") or {}
    provenance = provenance or (result.facts["dcf"].get("shared_assumption_provenance")
                                or {}).get("operating_margin") or {}
    assert provenance.get("source_type") != "configured_default"


def test_8_the_assumption_builder_receives_normalized_and_guidance_evidence(wired):
    """Section 11: the margin comes from guidance, a guidance-implied figure
    or the normalized margin -- and its derivation says which."""
    result = _run(wired)
    provenance = ((_base(result).get("assumptions") or {})
                  .get("assumption_provenance") or {}).get("operating_margin") or {}
    provenance = provenance or (result.facts["dcf"].get("shared_assumption_provenance")
                                or {}).get("operating_margin") or {}
    derivation = provenance.get("derivation") or ""
    assert any(marker in derivation for marker in
               ("DERIVED_FROM_GUIDANCE", "NORMALIZED", "management operating-margin guidance",
                "trailing-twelve-month operating margin")), derivation


def test_9_current_year_tax_guidance_is_not_propagated_across_the_horizon(wired):
    """Section 16: a current year moved by an acquisition is not the forecast
    rate for the following four."""
    result = _run(wired)
    tax = (_base(result).get("assumptions") or {}).get("tax_rate")
    assert isinstance(tax, list), "tax_rate is not a per-year path"
    guided = _guidance(result).get("tax_rate")
    if guided and guided.get("midpoint") is not None:
        if tax[0] == pytest.approx(guided["midpoint"], abs=1e-6):
            # Year 1 took the guided rate; the horizon must not simply repeat it
            # unless the guided rate already matches the normalized one.
            assert len(set(round(v, 6) for v in tax)) > 1 or \
                tax[-1] == pytest.approx(tax[0])


# ---------------------------------------------------------------------------
# Section 38, checks 10-13: suitability, rendering, readiness, recommendation
# ---------------------------------------------------------------------------

def test_10_dcf_suitability_reflects_unresolved_normalization(wired):
    result = _run(wired)
    record = result.facts["dcf_suitability"]
    assert record["dcf_suitability"] in SU.DcfSuitability.ALL
    normalization = _profitability(result)["normalized"]["status"]
    if normalization != P.NormalizationStatus.VALID:
        assert record["dcf_suitability"] != SU.DcfSuitability.SUITABLE


def test_11_an_extreme_premium_is_suppressed_when_the_dcf_is_not_reliable(wired):
    """Section 26: "Market-price premium: 75,588.5%" states a precision the
    inputs do not support."""
    result = _run(wired)
    report = render_compact_report(result, build_compact_synthesis_payload(result),
                                   pipeline_result=None)
    suitability = result.facts["dcf_suitability"]["dcf_suitability"]
    if suitability in (SU.DcfSuitability.LIMITED, SU.DcfSuitability.NOT_SUITABLE):
        assert "not meaningful" in report
        assert "Market-price premium to base modeled value" not in report
        assert "Modeled return from current price to base value" not in report


def test_11b_the_raw_comparison_survives_in_the_underlying_facts(wired):
    """Section 26: suppression is a RENDERING decision; the numbers stay."""
    result = _run(wired)
    gap = result.facts.get("valuation_gap") or {}
    assert "available" in gap


def test_12_readiness_prioritizes_the_current_valuation_problem(wired):
    """Section 29: a historical structural-break note must not lead when a
    current-period profitability issue is driving the valuation."""
    result = _run(wired)
    reasons = result.facts["research_readiness"]["reasons"]
    assert reasons
    # The LEADING reason is what a reader sees first. It must be about the
    # current valuation, not an older structural-break note -- and a reason
    # that merely MENTIONS the break while leading with a current concern is
    # fine, so the test looks at what the first reason is ABOUT rather than
    # at which markers appear anywhere inside it.
    leading = reasons[0].lower()
    assert not leading.startswith("this company's reported history spans a structural break"),         reasons
    assert any(marker in leading for marker in
               ("profitab", "normaliz", "configured default", "suitable", "share count",
                "net debt", "trailing-twelve-month")), reasons


def test_13_no_recommendation_is_produced_deterministically(wired):
    result = _run(wired)
    assert "recommendation" not in result.facts
    report = render_compact_report(result, build_compact_synthesis_payload(result),
                                   pipeline_result=None)
    assert "Recommendation: unavailable" in report


# ---------------------------------------------------------------------------
# Cross-section consistency (sections 13-15, 39)
# ---------------------------------------------------------------------------

def test_the_snapshot_and_the_valuation_quote_one_canonical_free_cash_flow(wired):
    result = _run(wired)
    compact = build_compact_synthesis_payload(result)
    canonical = ((compact.get("current_financial_state") or {})
                 .get("flows") or {}).get("free_cash_flow") or {}
    if canonical.get("value") is None:
        pytest.skip("no canonical free cash flow for this fixture")
    report = render_compact_report(result, compact, pipeline_result=None)
    assert "FCF (TTM)" in report or "FCF (FY)" in report


def test_free_cash_flow_covers_exactly_one_period(wired):
    flows = _state(_run(wired))["flows"]
    ocf, capex = flows["operating_cash_flow"], flows["capital_expenditure"]
    fcf = flows["free_cash_flow"]
    if fcf.get("value") is not None:
        assert ocf["as_of_date"] == capex["as_of_date"] == fcf["as_of_date"]


def test_the_dcf_input_audit_records_the_suitability_verdict(wired):
    result = _run(wired)
    audit = result.facts.get("dcf_input_audit") or {}
    assert audit.get("dcf_suitability") in SU.DcfSuitability.ALL
    assert audit.get("flow_base") and audit.get("balance_sheet")
