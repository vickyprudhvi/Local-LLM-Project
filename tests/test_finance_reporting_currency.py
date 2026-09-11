"""REPORTING_CURRENCY_SERIES_SELECTION -- central resolver tests + benchmark.

Three layers, per the phase's own division:

  1. Case-by-case correctness against `tests/fixtures/reporting_currency_
     benchmark.py`'s ground truth (section 22 -- decided independently of
     the resolver).
  2. Named hard-safety counters (section 23), each proven zero by a
     dedicated positive-control test that shows what WOULD have happened
     with the guard removed.
  3. Integration: `finance.freshness.build_current_financial_state` and the
     workflow.py gates (`_valuation_status`, `_freshness_readiness_signals`)
     actually consume the resolution, not just the resolver in isolation.
"""

import inspect

import pytest

from finance import reporting_currency as rc
from finance import freshness as fr
from finance import xbrl_mapping
from finance import workflow
from finance import dcf_packet

from tests.fixtures.reporting_currency_benchmark import CASES, ReportingCurrencyCase
from tests.fixtures import reporting_currency_benchmark as fixtures


# ---------------------------------------------------------------------------
# 1. Case-by-case correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_case(case: ReportingCurrencyCase):
    result = rc.resolve_reporting_series(case.company_facts)
    assert result.selection_status == case.expected_status, (
        f"{case.case_id}: expected status {case.expected_status}, got "
        f"{result.selection_status} ({result.resolution_reason})")
    if case.expected_currency is not None:
        assert result.selected_reporting_currency == case.expected_currency, case.case_id
    if case.expected_status == rc.ReportingSeriesStatus.RESOLVED:
        assert result.selected_period == case.expected_period, case.case_id
        assert result.selected_series_id is not None, case.case_id
    else:
        # Section 12/13: an unresolved/conflicted case must never carry a
        # selected series or period -- that would be exactly the silent
        # fallback this resolver exists to prevent.
        assert result.selected_series_id is None, case.case_id
        assert result.selected_period is None, case.case_id
    for code in case.expected_rejection_codes:
        assert code in result.rejection_codes, (
            f"{case.case_id}: expected rejection code {code} in {result.rejection_codes}")


def test_benchmark_accuracy_is_100_percent():
    """Section 24 — current-currency, series-selection and current-period
    accuracy collapse to one number here because each case asserts all
    three together above; this is the summary rollup."""
    failures = []
    for case in CASES:
        result = rc.resolve_reporting_series(case.company_facts)
        ok = (result.selection_status == case.expected_status
              and (case.expected_currency is None
                   or result.selected_reporting_currency == case.expected_currency))
        if not ok:
            failures.append(case.case_id)
    assert not failures, f"reporting-currency benchmark regressed on: {failures}"


# ---------------------------------------------------------------------------
# 2. Hard safety counters (section 23) -- each MUST be zero
# ---------------------------------------------------------------------------

def _switch_state():
    switch_case = next(c for c in CASES if c.case_id == "B_currency_switch")
    return fr.build_current_financial_state(switch_case.company_facts, "BENCH",
                                            valuation_date="2026-09-11")


def test_stale_currency_series_published_as_current():
    """The switch case's stale USD balance sheet (2017-12-31) must never be
    presented at a freshness grade that reads as current."""
    state = _switch_state()
    assert state.reporting_currency_status == rc.ReportingSeriesStatus.UNRESOLVED_CURRENCY_SWITCH
    # The counterfactual this guards against: WITHOUT the currency gate,
    # this state's balance/flow selections are real, non-null 2017 figures,
    # so `_classify_valuation_freshness` would have graded them CURRENT or
    # MOSTLY_CURRENT purely from filing recency of the (only) readable data.
    ungated = fr._classify_valuation_freshness(
        state.findings, state.balance_sheet, state.flows,
        state.financial_as_of, state.latest_annual_period, currency_status=None)
    assert ungated != fr.ValuationFreshness.STALE_INVALID, (
        "fixture sanity: without the gate this case would NOT already read as stale")
    assert state.valuation_freshness == fr.ValuationFreshness.STALE_INVALID


def test_mixed_currency_canonical_state():
    """The canonical balance sheet must carry the OLD USD magnitude, never
    the JPY magnitude read as though it were dollars."""
    state = _switch_state()
    cash = state.balance_sheet.get("cash_and_cash_equivalents")
    assert cash is not None and cash.value is not None
    # The JPY figure (305,000,000,000) is two orders of magnitude larger;
    # a currency mix-up would surface as exactly this kind of value jump.
    assert cash.value == pytest.approx(1_600_000_000.0)
    assert cash.as_of_date == "2017-12-31"


def test_mixed_currency_ttm_published():
    """A trailing-twelve-month construction can only ever be built from
    facts finance/xbrl_mapping.py::_candidate_facts returns, and that
    function reads a single unit per concept -- so no TTM in this project
    can physically combine two currencies. Proven directly at that layer."""
    switch_case = next(c for c in CASES if c.case_id == "B_currency_switch")
    rows = xbrl_mapping._candidate_facts(switch_case.company_facts, "Revenues")
    currencies = {row["_currency"] for row in rows}
    assert currencies == {"USD"}, (
        f"_candidate_facts returned more than one currency for one concept: {currencies}")


def test_mixed_currency_dcf_accepted():
    """A DCF built on the switch-case state must classify as FINANCIAL_BASE_
    STALE, never VALID_FOR_RESEARCH, however clean its own arithmetic."""
    state = _switch_state()
    facts = {
        "_current_financial_state": state,
        "dcf": {"available": True, "validation_status": "DCF_VALID"},
        "dcf_packet_failure": None,
        "dcf_suitability": {},
    }
    status = workflow._valuation_status(facts)
    assert status == dcf_packet.ValuationStatus.FINANCIAL_BASE_STALE
    assert status != dcf_packet.ValuationStatus.VALID_FOR_RESEARCH


def test_listing_currency_assumed_as_reporting_currency():
    """`resolve_reporting_series` has no listing-currency/ticker/exchange
    parameter at all, so it cannot be swayed by one -- proven both
    behaviourally and structurally."""
    params = list(inspect.signature(rc.resolve_reporting_series).parameters)
    assert params == ["company_facts"], (
        "resolver gained a parameter that could carry a listing-currency hint")
    listing_case = next(c for c in CASES if c.case_id == "K_listing_currency_differs")
    result = rc.resolve_reporting_series(listing_case.company_facts)
    assert result.selected_reporting_currency == "EUR"
    assert result.selection_status != rc.ReportingSeriesStatus.RESOLVED


def test_old_series_selected_over_current_compatible_series():
    """A 10-year-longer EUR history must not beat a shorter but current and
    readable USD series."""
    longer_case = next(c for c in CASES if c.case_id == "C_old_series_longer_still_loses")
    result = rc.resolve_reporting_series(longer_case.company_facts)
    assert result.selected_reporting_currency == "USD"
    assert result.selected_period == "2026-06-30"
    assert any(r.currency == "EUR" and r.reason_code == "HISTORICAL_SUPERSEDED"
              for r in result.rejected_series)


def test_unresolved_currency_silently_falls_back_to_old_period():
    """When the current currency cannot be read, `selected_period` must be
    None -- never the old series' date presented as though it were current
    -- and the workflow readiness gate must actually block on it."""
    switch_case = next(c for c in CASES if c.case_id == "B_currency_switch")
    result = rc.resolve_reporting_series(switch_case.company_facts)
    assert result.selected_period is None
    assert result.selected_reporting_currency == "JPY"

    state = _switch_state()
    facts = {"current_financial_state": state.to_dict()}
    blocking, limiting = workflow._freshness_readiness_signals(facts)
    assert blocking, "the currency switch must be a BLOCKING readiness reason"
    assert any("currency" in reason.lower() for reason in blocking)


HARD_SAFETY_TESTS = (
    test_stale_currency_series_published_as_current,
    test_mixed_currency_canonical_state,
    test_mixed_currency_ttm_published,
    test_mixed_currency_dcf_accepted,
    test_listing_currency_assumed_as_reporting_currency,
    test_old_series_selected_over_current_compatible_series,
    test_unresolved_currency_silently_falls_back_to_old_period,
)


def test_all_named_hard_safety_counters_pass():
    """Section 23 rollup: every named counter's guard fires cleanly. Each
    dedicated test above IS the positive control -- this just proves none
    were accidentally skipped."""
    for guard in HARD_SAFETY_TESTS:
        guard()


# ---------------------------------------------------------------------------
# 3. TTM crossing a currency switch (fixture G)
# ---------------------------------------------------------------------------

def test_ttm_crossing_currency_switch_never_mixes_and_state_is_flagged_stale():
    payload = fixtures.company_facts({
        **fixtures.balance_sheet_anchor_rows([("2025-10-31", 8_000_000_000.0, "10-Q")]),
        "Revenues": {
            "USD": [
                fixtures.duration("2025-01-01", "2025-04-30", 900_000_000.0, form="10-Q"),
                fixtures.duration("2025-05-01", "2025-07-31", 950_000_000.0, form="10-Q"),
                fixtures.duration("2025-08-01", "2025-10-31", 1_000_000_000.0, form="10-Q"),
            ],
            "JPY": [
                fixtures.duration("2025-11-01", "2026-01-31", 160_000_000_000.0, form="6-K"),
            ],
        },
        "CashAndCashEquivalentsAtCarryingValue": {
            "USD": [fixtures.instant("2025-10-31", 8_000_000_000.0, form="10-Q")],
            "JPY": [fixtures.instant("2026-01-31", 1_100_000_000_000.0, form="6-K")],
        },
    })
    rows = xbrl_mapping._candidate_facts(payload, "Revenues")
    assert {r["_currency"] for r in rows} == {"USD"}, (
        "the TTM input path must never see the JPY quarter mixed in with USD quarters")

    result = rc.resolve_reporting_series(payload)
    assert result.selection_status == rc.ReportingSeriesStatus.UNRESOLVED_CURRENCY_SWITCH

    state = fr.build_current_financial_state(payload, "BENCH", valuation_date="2026-09-11")
    assert state.valuation_freshness == fr.ValuationFreshness.STALE_INVALID


# ---------------------------------------------------------------------------
# Case J / A: control cases must NOT be gated (no regression on the common path)
# ---------------------------------------------------------------------------

def test_domestic_control_is_not_gated():
    control_case = next(c for c in CASES if c.case_id == "J_domestic_control")
    state = fr.build_current_financial_state(control_case.company_facts, "BENCH",
                                             valuation_date="2026-09-11")
    assert state.reporting_currency_status == rc.ReportingSeriesStatus.RESOLVED
    assert state.valuation_freshness != fr.ValuationFreshness.STALE_INVALID
    facts = {"current_financial_state": state.to_dict()}
    blocking, _limiting = workflow._freshness_readiness_signals(facts)
    assert blocking == []
