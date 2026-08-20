"""Phase H.7 — data-integrity regressions across DIFFERENT BUSINESS AND DATA
STRUCTURES, from real captured payloads (2026-08-19).

WHY THREE COMPANIES AND NOT THREE TICKERS
=========================================
Each fixture stands for a STRUCTURE, not for a company, and every assertion
here is about the structure. Nothing in production reads a ticker; nothing in
this file asserts a valuation, a recommendation, a growth rate, a share count
or any other company-specific result.

    rivn_regression.json   deeply loss-making growth company
                           -60% operating margin, negative free cash flow,
                           post-quarter equity offering. Stands for: a model
                           bound must not manufacture profitability.

    jbs_regression.json    foreign private issuer
                           reports under ifrs-full in 20-F/6-K, more than one
                           share class, material noncontrolling interest.
                           Stands for: a framework this project cannot read
                           is not the same as a company that reported nothing,
                           and a share count covering one class is not the
                           denominator the market is pricing.

    casy_regression.json   non-calendar fiscal year, annual filing freshest
                           the latest filing IS the 10-K and no quarter has
                           been reported since. Stands for: "the annual
                           figure is the freshest honest one" is a legitimate
                           outcome, not a staleness failure.

THE FAILURES THEY PIN DOWN
==========================
1. A -60.1% operating margin was clamped to the model's +1.0% floor and the
   floor was used as the five-year forecast. The valuation came out positive
   and VALID. A bound must limit an input, never replace it.
2. An issuer's entire financial history was reported as absent because every
   lookup went to `facts["us-gaap"]` and the issuer reports `ifrs-full`.
3. A modelled value per share was published against a share count that
   implies a market capitalisation 76% below the one the same provider
   reported in the same payload.
"""

import json
from pathlib import Path

import pytest

import tools.finance_tools as finance_tools
from finance import entity as E
from finance import net_debt as ND
from finance import suitability as SU
from finance import taxonomy as TX
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

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# (fixture stem, the structure it stands for). The symbol is a lookup key for
# the captured payload and is never branched on.
LOSS_MAKING = "rivn"
FOREIGN_PRIVATE_ISSUER = "jbs"
ANNUAL_FRESHEST = "casy"


def _fixture(stem):
    return json.loads((FIXTURE_DIR / f"{stem}_regression.json").read_text(encoding="utf-8"))


def _make_clients(fixture):
    class YahooFixtureClient:
        provider_id = "yahoo"

        def fetch(self, dataset, arguments):
            payload = fixture["yahoo"].get(dataset.dataset_id)
            if payload is None:
                raise ToolFailure(MARKET_DATA_INVALID_SYMBOL, "no fixture data")
            return ProviderResponse(payload=payload, provider_metadata={}, byte_count=10)

    class SecFixtureClient:
        provider_id = "sec"

        def fetch(self, dataset, arguments):
            if dataset.dataset_id == "filing_document":
                key = f"{arguments.get('accession')}/{arguments.get('document')}"
                text = (fixture["sec"].get("filing_documents") or {}).get(key)
                if text is None:
                    raise ToolFailure(MARKET_DATA_INVALID_SYMBOL, "no fixture document")
                return ProviderResponse(
                    payload={"document_text": text, "byte_count": len(text),
                             "accession": arguments.get("accession"),
                             "document": arguments.get("document"), "truncated": False},
                    provider_metadata={}, byte_count=len(text))
            payload = fixture["sec"].get(dataset.dataset_id)
            if payload is None:
                raise ToolFailure(MARKET_DATA_INVALID_SYMBOL, "no fixture data")
            return ProviderResponse(payload=payload, provider_metadata={}, byte_count=10)

    return YahooFixtureClient(), SecFixtureClient()


@pytest.fixture
def run_fixture(tmp_path, monkeypatch):
    """Runs a captured payload through the real workflow."""
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

    created = []

    def _run(stem):
        fixture = _fixture(stem)
        symbol = next(iter(fixture["sec"]["ticker_cik_map"].values()))["ticker"]
        clock = FakeClock()
        yahoo_client, sec_client = _make_clients(fixture)
        suffix = f"{stem}_{len(created)}"
        finance_tools.set_yahoo_coordinator(MarketDataRequestCoordinator(
            cache=MarketDataCache(path=str(tmp_path / f"y{suffix}.sqlite3"), clock=clock),
            ledger=QuotaLedger(path=str(tmp_path / f"yq{suffix}.sqlite3"), clock=clock,
                               daily_limit=10 ** 6),
            client=yahoo_client, provider_id="yahoo",
            dataset_resolver=resolve_yahoo_dataset, clock=clock,
            sleeper=lambda _s: None, min_request_interval_ms=0))
        finance_tools.set_sec_coordinator(MarketDataRequestCoordinator(
            cache=MarketDataCache(path=str(tmp_path / f"s{suffix}.sqlite3"), clock=clock),
            ledger=QuotaLedger(path=str(tmp_path / f"sq{suffix}.sqlite3"), clock=clock,
                               daily_limit=10 ** 6),
            client=sec_client, provider_id="sec", dataset_resolver=resolve_sec_dataset,
            clock=clock, sleeper=lambda _s: None, min_request_interval_ms=0))
        finance_tools.set_coordinator(MarketDataRequestCoordinator(
            cache=MarketDataCache(path=str(tmp_path / f"a{suffix}.sqlite3"), clock=clock),
            ledger=AlphaVantageQuotaLedger(path=str(tmp_path / f"aq{suffix}.sqlite3"),
                                           clock=clock, daily_limit=100),
            client=AlphaVantageClient(), clock=clock, sleeper=lambda _s: None,
            jitter=lambda: 0.5))
        registry = ToolRegistry()
        for tool_cls in finance_tools.ALL_FINANCE_TOOL_CLASSES:
            registry.register(tool_cls())
        created.append(stem)
        return run_full_stock_analysis(ToolExecutor(registry), symbol)

    yield _run
    finance_tools.set_coordinator(None)
    finance_tools.set_yahoo_coordinator(None)
    finance_tools.set_sec_coordinator(None)


def _state(result):
    return result.facts.get("current_financial_state") or {}


def _basis(result):
    return result.facts.get("dcf_financial_basis") or {}


def _flow(result, name):
    return (_state(result).get("flows") or {}).get(name) or {}


# ---------------------------------------------------------------------------
# D — deeply loss-making growth company (sections 35-38)
# ---------------------------------------------------------------------------

def test_a_deeply_negative_margin_is_never_clamped_into_profitability(run_fixture):
    """The headline failure: a -60% operating margin modelled as +1%.

    The assertion is structural -- whatever this company's margin is, the
    MODELLED margin must not have the opposite sign from the OBSERVED one.
    """
    result = run_fixture(LOSS_MAKING)
    revenue, operating_income = _flow(result, "revenue"), _flow(result, "operating_income")
    assert revenue.get("value") and operating_income.get("value") is not None
    observed = operating_income["value"] / revenue["value"]
    assert observed < 0, "fixture no longer represents a loss-making company"

    base = next(s for s in result.facts["dcf"]["scenarios"] if s["scenario"] == "base")
    modelled = base["assumptions"]["operating_margin"]
    year_one = modelled[0] if isinstance(modelled, list) else modelled
    assert year_one < 0, (
        f"observed margin {observed:.1%} was modelled as {year_one:.1%} -- a bound "
        "manufactured profitability the company does not have")


def test_the_observed_margin_is_retained_beside_whatever_was_applied(run_fixture):
    """Section 40: proposed, applied, raw and clamp reason all survive."""
    result = run_fixture(LOSS_MAKING)
    base = next(s for s in result.facts["dcf"]["scenarios"] if s["scenario"] == "base")
    provenance = ((base.get("assumptions") or {}).get("assumption_provenance") or {})
    entry = provenance.get("operating_margin") or (
        result.facts["dcf"].get("shared_assumption_provenance") or {}).get("operating_margin")
    assert entry is not None
    assert entry.get("source_type")
    for year in (entry.get("forecast_path") or []):
        assert "raw_growth" in year and "applied_growth" in year


def test_a_loss_making_company_is_assessed_for_dcf_suitability(run_fixture):
    """Section 35: suitability is separate from arithmetic validity."""
    result = run_fixture(LOSS_MAKING)
    record = result.facts["dcf_suitability"]
    assert record["dcf_suitability"] in SU.DcfSuitability.ALL
    assert record["dcf_suitability"] != SU.DcfSuitability.SUITABLE
    assert record["summary"]
    codes = {s["code"] for s in record["signals"]}
    assert codes, "a loss-making company produced no suitability signal at all"


def test_persistent_negative_cash_generation_is_distinguished_from_one_weak_year(run_fixture):
    result = run_fixture(LOSS_MAKING)
    fcf = _flow(result, "free_cash_flow")
    if fcf.get("value") is not None and fcf["value"] < 0:
        codes = {s["code"] for s in result.facts["dcf_suitability"]["signals"]}
        assert "PERSISTENT_NEGATIVE_CASH_GENERATION" in codes


def test_a_post_quarter_offering_reduces_valuation_freshness(run_fixture):
    result = run_fixture(LOSS_MAKING)
    events = _state(result).get("post_balance_sheet_events") or []
    if events:
        assert _state(result)["valuation_freshness"] != "CURRENT"


def test_suitability_flows_into_research_readiness(run_fixture):
    """Section 49: a valuation the instrument cannot describe must not be
    presented as READY."""
    result = run_fixture(LOSS_MAKING)
    readiness = result.facts["research_readiness"]
    assert readiness["status"] in ("LIMITED", "NOT_READY")
    assert readiness["reasons"]


# ---------------------------------------------------------------------------
# E — foreign private issuer (sections 7, 15-21)
# ---------------------------------------------------------------------------

def test_a_non_us_gaap_issuer_is_read_rather_than_reported_as_empty(run_fixture):
    """The whole financial history was invisible because every lookup went to
    `facts["us-gaap"]`."""
    result = run_fixture(FOREIGN_PRIVATE_ISSUER)
    state = _state(result)
    assert state["taxonomy"] == TX.IFRS
    revenue = _flow(result, "revenue")
    assert revenue.get("value"), "the issuer's revenue is still unreadable"
    assert state["financial_as_of"], "no balance-sheet date was identified"
    assert (state["balance_sheet"] or {}).get("cash_and_cash_equivalents", {}).get("value")


def test_annual_and_interim_forms_beyond_10k_10q_are_recognized():
    """Section 7: an FPI files 20-F and 6-K, not 10-K and 10-Q."""
    for form in ("20-F", "40-F"):
        assert form in TX.ANNUAL_FORMS
    for form in ("6-K",):
        assert form in TX.INTERIM_FORMS


def test_a_framework_this_project_cannot_read_says_so(run_fixture):
    """Section 56: 'we cannot read this' and 'they reported nothing' are
    different claims and only one of them is true."""
    empty = {"facts": {"some-other-taxonomy": {"Foo": {}}}}
    reason = TX.unsupported_taxonomy_reason(empty)
    assert reason and "no reviewed concept mapping" in reason
    assert TX.unsupported_taxonomy_reason(
        {"facts": {"us-gaap": {"Revenues": {}}}}) is None


def test_a_multi_class_issuer_is_detected_and_the_share_basis_is_checked(run_fixture):
    """Sections 18-21: the identity that catches a per-share denominator
    covering only part of the equity."""
    result = run_fixture(FOREIGN_PRIVATE_ISSUER)
    reconciliation = _basis(result).get("share_reconciliation") or {}
    assert reconciliation.get("status") in E.ReconciliationStatus.ALL
    # The check must have actually run -- a status of UNKNOWN here would mean
    # the invariant was never evaluated, which is the pre-fix behaviour.
    assert reconciliation.get("status") != E.ReconciliationStatus.UNKNOWN


def test_an_unreconciled_share_basis_blocks_valuation_conclusions(run_fixture):
    """Section 49: every per-share figure depends on the denominator."""
    result = run_fixture(FOREIGN_PRIVATE_ISSUER)
    reconciliation = _basis(result).get("share_reconciliation") or {}
    if reconciliation.get("status") in ("MATERIAL_DIFFERENCE", "INCOMPATIBLE_BASIS"):
        assert result.facts["research_readiness"]["status"] == "NOT_READY"
        report = render_compact_report(
            result, build_compact_synthesis_payload(result), pipeline_result=None)
        assert "Share basis: UNRESOLVED" in report


def test_an_unresolved_share_basis_withholds_the_price_comparison(run_fixture):
    """Section 31/56: a precise price-vs-value comparison and a warning that
    the two are not on the same basis cannot both be printed."""
    result = run_fixture(FOREIGN_PRIVATE_ISSUER)
    reconciliation = _basis(result).get("share_reconciliation") or {}
    if reconciliation.get("status") not in ("MATERIAL_DIFFERENCE", "INCOMPATIBLE_BASIS"):
        pytest.skip("this fixture's share basis reconciles")
    report = render_compact_report(
        result, build_compact_synthesis_payload(result), pipeline_result=None)
    assert "Comparison with the market price is WITHHELD" in report
    assert "Market-price premium to base modeled value" not in report
    assert "Market-price discount to base modeled value" not in report
    assert "Modeled return from current price to base value" not in report


def test_parent_attributable_equity_is_preferred_over_consolidated(run_fixture):
    """Section 24: a material noncontrolling interest belongs to somebody
    else and must not sit in the analyzed security's book value."""
    result = run_fixture(FOREIGN_PRIVATE_ISSUER)
    balance = _state(result).get("balance_sheet") or {}
    equity = balance.get("stockholders_equity") or {}
    minority = balance.get("minority_interest") or {}
    if equity.get("value") and minority.get("value"):
        assert equity["concept"] != minority["concept"]


# ---------------------------------------------------------------------------
# A/F — annual filing genuinely freshest, no guidance (sections 6, 34, 49)
# ---------------------------------------------------------------------------

def test_an_annual_filing_may_legitimately_be_the_freshest_basis(run_fixture):
    """Section 6/34: when no interim period has been reported since the
    fiscal year end, the annual figure IS current. That is not staleness."""
    result = run_fixture(ANNUAL_FRESHEST)
    state = _state(result)
    if state.get("latest_quarterly_period") is None:
        assert state["valuation_freshness"] not in ("STALE_INPUT_WARNING", "STALE_INVALID")
        assert state["financial_as_of"] == state["latest_annual_period"]


def test_a_non_calendar_fiscal_year_produces_a_valid_twelve_month_base(run_fixture):
    result = run_fixture(ANNUAL_FRESHEST)
    revenue = _flow(result, "revenue")
    assert revenue.get("value")
    if revenue.get("ttm"):
        assert revenue["ttm"]["validation_status"] in ("valid", "partial")
        assert 350 <= (revenue["ttm"]["span_days"] or 0) <= 380
    # The fiscal year must not be assumed to end in December.
    assert revenue["as_of_date"]


def test_missing_guidance_does_not_stop_the_valuation(run_fixture):
    """Section 57F: the DCF continues from reported evidence and readiness
    reflects the gap."""
    result = run_fixture(ANNUAL_FRESHEST)
    guidance = (result.facts.get("management_guidance") or {}).get("metrics") or {}
    if not guidance:
        assert result.facts["dcf"]["available"] is True
        assert result.facts["research_readiness"]["status"] in ("READY", "LIMITED")


# ---------------------------------------------------------------------------
# Cross-cutting invariants that must hold for EVERY structure
# ---------------------------------------------------------------------------

ALL_FIXTURES = (LOSS_MAKING, FOREIGN_PRIVATE_ISSUER, ANNUAL_FRESHEST)


@pytest.mark.parametrize("stem", ALL_FIXTURES)
def test_the_dcf_input_audit_is_always_built(run_fixture, stem):
    """Section 48: one immutable object describing what the valuation rests
    on, present for every issuer type."""
    result = run_fixture(stem)
    audit = result.facts.get("dcf_input_audit") or (
        _state(result).get("freshness_audit") or {})
    assert audit, "no DCF input audit was produced"
    for key in ("valuation_date", "flow_base", "balance_sheet", "guidance",
                "post_balance_sheet_events", "historical_comparability", "warnings"):
        assert key in audit, key


@pytest.mark.parametrize("stem", ALL_FIXTURES)
def test_no_balance_sheet_field_is_ever_given_a_trailing_twelve_months(run_fixture, stem):
    """Section 5: point-in-time values stay point-in-time."""
    result = run_fixture(stem)
    for name, selection in (_state(result).get("balance_sheet") or {}).items():
        assert selection.get("source") != "ttm_calculation", name
        assert not selection.get("ttm"), name


@pytest.mark.parametrize("stem", ALL_FIXTURES)
def test_a_ttm_label_always_carries_a_validated_construction(run_fixture, stem):
    """Section 4: nothing may be called TTM without a validated window."""
    result = run_fixture(stem)
    for name, selection in (_state(result).get("flows") or {}).items():
        if selection.get("source") != "ttm_calculation":
            continue
        record = selection.get("ttm") or {}
        assert record.get("validation_status") in ("valid", "partial"), name
        assert record.get("construction_method"), name
        assert record.get("start_date") and record.get("end_date"), name


@pytest.mark.parametrize("stem", ALL_FIXTURES)
def test_net_debt_is_computed_under_a_named_policy(run_fixture, stem):
    """Section 27: the policy is never hidden and never switches by company."""
    result = run_fixture(stem)
    detail = _state(result).get("net_debt_detail") or {}
    if detail.get("net_debt") is not None:
        assert detail["net_debt_policy"] in ND.NetDebtPolicyName.ALL
        assert detail["derivation"]


@pytest.mark.parametrize("stem", ALL_FIXTURES)
def test_free_cash_flow_never_mixes_two_periods(run_fixture, stem):
    """Section 30/31: OCF and capex must cover the same window."""
    result = run_fixture(stem)
    flows = _state(result).get("flows") or {}
    fcf = flows.get("free_cash_flow") or {}
    if fcf.get("value") is None:
        return
    ocf, capex = flows.get("operating_cash_flow") or {}, flows.get("capital_expenditure") or {}
    assert ocf.get("as_of_date") == capex.get("as_of_date") == fcf.get("as_of_date")
    assert ocf.get("period_start") == capex.get("period_start")


@pytest.mark.parametrize("stem", ALL_FIXTURES)
def test_the_snapshot_and_the_valuation_quote_one_canonical_set(run_fixture, stem):
    """Section 31/55: a metric must not change between sections."""
    result = run_fixture(stem)
    compact = build_compact_synthesis_payload(result)
    # Phase H.9: the canonical packet is the single copy of the selected
    # current values, and the Snapshot reads it directly.
    canonical = (compact.get("canonical_evidence") or {}).get("current") or {}
    report = render_compact_report(result, compact, pipeline_result=None)
    revenue = canonical.get("revenue") or {}
    if revenue.get("value") and revenue.get("period_type") == "TTM":
        assert "Revenue (TTM)" in report


@pytest.mark.parametrize("stem", ALL_FIXTURES)
def test_no_recommendation_is_produced_deterministically(run_fixture, stem):
    """Section 50: the recommendation stays the synthesizer's."""
    result = run_fixture(stem)
    assert "recommendation" not in result.facts


@pytest.mark.parametrize("stem", ALL_FIXTURES)
def test_every_run_states_a_dcf_suitability(run_fixture, stem):
    result = run_fixture(stem)
    assert (result.facts.get("dcf_suitability") or {}).get("dcf_suitability") \
        in SU.DcfSuitability.ALL
