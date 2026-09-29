"""Phase H.1 — normalization and locally calculated metrics.

The central guarantee under test: a missing input stays missing. Nothing is
interpolated, carried forward, or defaulted to zero, and annual/quarterly
periods are never merged.
"""

import math

import pytest

from finance.metrics import (
    annualized_volatility,
    exponential_moving_average,
    fundamental_metrics,
    maximum_drawdown,
    metrics_to_dict,
    relative_strength_index,
    simple_moving_average,
    technical_metrics,
    total_return,
)
from finance.normalization import (
    PeriodType,
    ValueBasis,
    normalize_all,
    normalize_price_history,
    normalize_quote,
    normalize_statement,
    parse_fiscal_date,
    parse_number,
)


# ---- primitives ----

@pytest.mark.parametrize("raw", ["None", "", "-", "n/a", "null", None, "abc", "NaN"])
def test_missing_tokens_become_none_not_zero(raw):
    assert parse_number(raw) is None, "a missing value must never become 0"


@pytest.mark.parametrize("raw,expected", [
    ("1234", 1234.0), ("1,234.5", 1234.5), ("-42", -42.0), ("0", 0.0),
    (17, 17.0), (3.5, 3.5), ("$1,000", 1000.0),
])
def test_numeric_parsing(raw, expected):
    assert parse_number(raw) == pytest.approx(expected)


def test_reported_zero_is_preserved_and_distinct_from_missing():
    assert parse_number("0") == 0.0
    assert parse_number("None") is None


@pytest.mark.parametrize("raw", ["2026-08-05", "1999-12-31"])
def test_valid_fiscal_dates(raw):
    assert parse_fiscal_date(raw) == raw


@pytest.mark.parametrize("raw", ["2026/08/05", "05-08-2026", "2026-13-01", "", None, "x"])
def test_invalid_fiscal_dates_are_rejected(raw):
    assert parse_fiscal_date(raw) is None


# ---- statements ----

INCOME_PAYLOAD = {
    "symbol": "TEST",
    "annualReports": [
        {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD",
         "totalRevenue": "1000", "grossProfit": "400", "operatingIncome": "200",
         "netIncome": "150", "incomeBeforeTax": "190", "incomeTaxExpense": "40"},
        {"fiscalDateEnding": "2024-12-31", "reportedCurrency": "USD",
         "totalRevenue": "800", "grossProfit": "300", "operatingIncome": "150",
         "netIncome": "100", "incomeBeforeTax": "140", "incomeTaxExpense": "40"},
    ],
    "quarterlyReports": [
        {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD",
         "totalRevenue": "260", "netIncome": "40"},
    ],
}
BALANCE_PAYLOAD = {
    "annualReports": [
        {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD",
         "totalCurrentAssets": "500", "totalCurrentLiabilities": "250",
         "totalShareholderEquity": "600", "shortTermDebt": "100",
         "longTermDebt": "200", "commonStockSharesOutstanding": "100",
         "cashAndCashEquivalentsAtCarryingValue": "150"},
        {"fiscalDateEnding": "2024-12-31", "reportedCurrency": "USD",
         "totalShareholderEquity": "500", "commonStockSharesOutstanding": "105"},
    ],
}
CASHFLOW_PAYLOAD = {
    "annualReports": [
        {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD",
         "operatingCashflow": "250", "capitalExpenditures": "70",
         "depreciationDepletionAndAmortization": "50"},
    ],
}


def build_statements():
    return normalize_all({
        "income_statement": (INCOME_PAYLOAD, {"origin": "provider"}),
        "balance_sheet": (BALANCE_PAYLOAD, {"origin": "cache"}),
        "cash_flow": (CASHFLOW_PAYLOAD, {"origin": "provider"}),
    }, "TEST")


def test_annual_and_quarterly_are_kept_separate():
    annual, quarterly, _warnings = normalize_statement(
        INCOME_PAYLOAD, "income_statement", "TEST")

    assert len(annual) == 2 and len(quarterly) == 1
    assert all(p.period_type == PeriodType.ANNUAL for p in annual)
    assert all(p.period_type == PeriodType.QUARTERLY for p in quarterly)
    # The 260 quarterly revenue must not appear in the annual series.
    assert [p.get("revenue") for p in annual] == [1000.0, 800.0]


def test_periods_are_newest_first_and_carry_provenance():
    statements = build_statements()
    income = statements.annual["income_statement"]
    assert [p.fiscal_date for p in income] == ["2025-12-31", "2024-12-31"]
    assert all(p.basis == ValueBasis.REPORTED for p in income)
    assert all(p.dataset_id == "income_statement" for p in income)
    assert statements.provenance["balance_sheet"] == {"origin": "cache"}


def test_reported_currency_is_preserved():
    statements = build_statements()
    assert statements.currency == "USD"
    assert statements.annual["income_statement"][0].currency == "USD"


# ---- cash / short-term-investments (Problem 1 regression) ----
#
# Fixture mirrors the ACTUAL captured MSFT defect: the provider's own combined
# "cashAndShortTermInvestments" field silently equals cash-and-equivalents
# alone, while a genuine, nonzero "shortTermInvestments" field sits unused
# right next to it. See tests/fixtures/msft_balance_sheet_regression.json for
# the full captured payload.

_BUGGY_COMBINED_FIELD_PAYLOAD = {"annualReports": [
    {"fiscalDateEnding": "2026-06-30", "reportedCurrency": "USD",
     "cashAndCashEquivalentsAtCarryingValue": "20935000000",
     "shortTermInvestments": "55908000000",
     "cashAndShortTermInvestments": "20935000000",  # BUGGY: equals cash alone
     "totalShareholderEquity": "442387000000"}]}


def test_cash_and_short_term_investments_are_normalized_separately():
    annual, _q, _w = normalize_statement(_BUGGY_COMBINED_FIELD_PAYLOAD, "balance_sheet", "TEST")
    v = annual[0].values
    assert v["cash_and_cash_equivalents"] == pytest.approx(20_935_000_000)
    assert v["short_term_investments"] == pytest.approx(55_908_000_000)


def test_combined_cash_and_investments_is_derived_correctly():
    annual, _q, _w = normalize_statement(_BUGGY_COMBINED_FIELD_PAYLOAD, "balance_sheet", "TEST")
    v = annual[0].values
    assert v["cash_and_short_term_investments"] == pytest.approx(
        20_935_000_000 + 55_908_000_000)


def test_cash_alone_is_never_mislabeled_as_the_combined_value():
    annual, _q, _w = normalize_statement(_BUGGY_COMBINED_FIELD_PAYLOAD, "balance_sheet", "TEST")
    v = annual[0].values
    assert v["cash_and_short_term_investments"] != v["cash_and_cash_equivalents"]
    # The provider's own (buggy, for this issuer) figure is retained ONLY for
    # audit/comparison — never presented as the authoritative combined value.
    assert v["cash_and_short_term_investments_provider_reported"] == pytest.approx(
        20_935_000_000)


def test_combined_value_is_not_derived_when_only_one_component_is_known():
    payload = {"annualReports": [
        {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD",
         "cashAndCashEquivalentsAtCarryingValue": "100"}]}  # no shortTermInvestments at all
    annual, _q, _w = normalize_statement(payload, "balance_sheet", "TEST")
    assert annual[0].values["cash_and_short_term_investments"] is None, \
        "the combined figure must never be guessed from cash alone"


def test_cash_vs_investments_reconciliation_warning_fires_on_conflict():
    _annual, _q, warnings = normalize_statement(
        _BUGGY_COMBINED_FIELD_PAYLOAD, "balance_sheet", "TEST")
    assert any("cash-and-short-term-investments" in w and "does not reconcile" in w
              for w in warnings)


def test_no_reconciliation_warning_when_figures_genuinely_agree():
    payload = {"annualReports": [
        {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD",
         "cashAndCashEquivalentsAtCarryingValue": "100", "shortTermInvestments": "50",
         "cashAndShortTermInvestments": "150"}]}
    _annual, _q, warnings = normalize_statement(payload, "balance_sheet", "TEST")
    assert not any("does not reconcile" in w for w in warnings)


# ---- debt aggregation (Problem 2 regression) ----

_CURRENT_PORTION_PAYLOAD = {"annualReports": [
    {"fiscalDateEnding": "2026-06-30", "reportedCurrency": "USD",
     "shortTermDebt": "18905300000", "currentLongTermDebt": "9227000000",
     "longTermDebt": "31067000000",
     "shortLongTermDebtTotal": "128808300000"}]}  # provider figure does NOT reconcile


def test_debt_components_are_normalized_separately():
    annual, _q, _w = normalize_statement(_CURRENT_PORTION_PAYLOAD, "balance_sheet", "TEST")
    v = annual[0].values
    assert v["short_term_debt"] == pytest.approx(18_905_300_000)
    assert v["current_portion_of_long_term_debt"] == pytest.approx(9_227_000_000)
    assert v["long_term_debt"] == pytest.approx(31_067_000_000)


def test_total_debt_includes_the_current_portion_without_double_counting():
    annual, _q, _w = normalize_statement(_CURRENT_PORTION_PAYLOAD, "balance_sheet", "TEST")
    # 18,905,300,000 + 9,227,000,000 + 31,067,000,000 = 59,199,300,000 —
    # NOT short_term_debt + long_term_debt alone (49,972,300,000), and NOT
    # double-counted against shortTermDebt.
    assert annual[0].values["total_debt"] == pytest.approx(59_199_300_000)


def test_total_debt_uses_only_known_components_not_zero_filled_silently():
    payload = {"annualReports": [
        {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD",
         "shortTermDebt": "100", "longTermDebt": "200"}]}  # no currentLongTermDebt at all
    annual, _q, _w = normalize_statement(payload, "balance_sheet", "TEST")
    assert annual[0].values["total_debt"] == pytest.approx(300.0)

    no_debt_at_all = {"annualReports": [
        {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD"}]}
    annual2, _q2, _w2 = normalize_statement(no_debt_at_all, "balance_sheet", "TEST")
    assert annual2[0].values["total_debt"] is None


def test_conflicting_provider_debt_field_produces_a_warning():
    _annual, _q, warnings = normalize_statement(
        _CURRENT_PORTION_PAYLOAD, "balance_sheet", "TEST")
    assert any("total-debt" in w and "does not reconcile" in w for w in warnings)


def test_no_debt_warning_when_provider_total_genuinely_reconciles():
    payload = {"annualReports": [
        {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD",
         "shortTermDebt": "100", "currentLongTermDebt": "50", "longTermDebt": "200",
         "shortLongTermDebtTotal": "350"}]}
    _annual, _q, warnings = normalize_statement(payload, "balance_sheet", "TEST")
    assert not any("total-debt" in w and "does not reconcile" in w for w in warnings)


def test_reconciliation_warnings_are_scoped_to_the_latest_period_only():
    """A systemic provider data-quality issue across many historical years must
    produce ONE clear warning, not one per fiscal year."""
    many_years = {"annualReports": [
        {"fiscalDateEnding": f"{year}-12-31", "reportedCurrency": "USD",
         "cashAndCashEquivalentsAtCarryingValue": "100", "shortTermInvestments": "50",
         "cashAndShortTermInvestments": "100"}  # buggy every year
        for year in range(2010, 2026)]}
    _annual, _q, warnings = normalize_statement(many_years, "balance_sheet", "TEST")
    reconciliation_warnings = [w for w in warnings if "does not reconcile" in w]
    assert len(reconciliation_warnings) == 1


# ---- provenance (Problem 1/2 requirement: exact source fields + period) ----

def test_provenance_retains_source_fields_and_fiscal_period():
    statements = build_statements()
    # Per-dataset provenance (origin, e.g. cache/provider) is preserved...
    assert statements.provenance["balance_sheet"] == {"origin": "cache"}
    # ...and every period carries its own fiscal date and dataset origin, so a
    # normalized VALUE can always be traced back to WHEN and FROM WHICH
    # statement it came.
    latest = statements.annual["balance_sheet"][0]
    assert latest.fiscal_date == "2025-12-31"
    assert latest.dataset_id == "balance_sheet"
    assert latest.basis == ValueBasis.REPORTED


def test_mixed_currency_raises_a_warning_and_is_not_converted():
    payload = {
        "annualReports": [
            {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD", "totalRevenue": "100"},
            {"fiscalDateEnding": "2024-12-31", "reportedCurrency": "EUR", "totalRevenue": "90"},
        ]
    }
    annual, _q, warnings = normalize_statement(payload, "income_statement", "TEST")
    assert any("more than one currency" in w for w in warnings)
    assert [p.get("revenue") for p in annual] == [100.0, 90.0], "values must not be converted"


def test_misaligned_periods_across_statements_are_flagged():
    shifted = {"annualReports": [dict(CASHFLOW_PAYLOAD["annualReports"][0],
                                      fiscalDateEnding="2024-12-31")]}
    statements = normalize_all({
        "income_statement": (INCOME_PAYLOAD, {}),
        "cash_flow": (shifted, {}),
    }, "TEST")
    assert any("do not align" in w for w in statements.warnings)


def test_unusable_fiscal_date_is_skipped_with_a_warning():
    payload = {"annualReports": [
        {"fiscalDateEnding": "not-a-date", "totalRevenue": "100"},
        {"fiscalDateEnding": "2025-12-31", "totalRevenue": "200"},
    ]}
    annual, _q, warnings = normalize_statement(payload, "income_statement", "TEST")
    assert len(annual) == 1
    assert any("fiscal date" in w for w in warnings)


# ---- quote and price history ----

def test_quote_is_never_labelled_realtime():
    quote = normalize_quote({"Global Quote": {
        "01. symbol": "AAPL", "05. price": "195.50", "08. previous close": "193.00",
        "07. latest trading day": "2026-08-04", "10. change percent": "1.2953%"}})
    assert quote["available"] is True
    assert quote["price"] == pytest.approx(195.50)
    assert quote["change_percent"] == pytest.approx(1.2953)
    assert quote["price_basis"] == "delayed", "free-tier data must never claim realtime"


def test_empty_quote_reports_unavailable():
    assert normalize_quote({})["available"] is False


def test_price_history_is_oldest_first_and_drops_unusable_bars():
    payload = {"Time Series (Daily)": {
        "2026-08-04": {"1. open": "10", "2. high": "11", "3. low": "9",
                       "4. close": "10.5", "5. adjusted close": "10.5", "6. volume": "100"},
        "2026-08-01": {"1. open": "9", "2. high": "10", "3. low": "8",
                       "4. close": "9.5", "5. adjusted close": "9.5", "6. volume": "90"},
        "bad-date": {"4. close": "1"},
        "2026-08-03": {"4. close": "None", "5. adjusted close": "None"},
    }}
    result = normalize_price_history(payload)

    assert result["available"] is True
    assert [b.date for b in result["bars"]] == ["2026-08-01", "2026-08-04"]
    assert result["dropped"] == 2
    assert any("unusable" in w for w in result["warnings"])


# ---- fundamental metrics ----

def test_fundamental_metrics_match_hand_calculation():
    metrics = metrics_to_dict(fundamental_metrics(build_statements()))

    assert metrics["revenue_growth_yoy"]["value"] == pytest.approx((1000 - 800) / 800)
    assert metrics["gross_margin"]["value"] == pytest.approx(400 / 1000)
    assert metrics["operating_margin"]["value"] == pytest.approx(200 / 1000)
    assert metrics["net_margin"]["value"] == pytest.approx(150 / 1000)
    assert metrics["current_ratio"]["value"] == pytest.approx(500 / 250)
    assert metrics["debt_to_equity"]["value"] == pytest.approx(300 / 600)
    assert metrics["roe_ending_equity"]["value"] == pytest.approx(150 / 600)
    # FCF = operating cash flow - |capex| = 250 - 70
    assert metrics["free_cash_flow"]["value"] == pytest.approx(180.0)
    assert metrics["free_cash_flow_margin"]["value"] == pytest.approx(180 / 1000)
    # ROIC = EBIT x (1 - 40/190) / (100 + 200 + 600)
    expected_roic = 200 * (1 - 40 / 190) / 900
    assert metrics["return_on_invested_capital"]["value"] == pytest.approx(expected_roic)


# ---- ROE methodology (Problem 4) ----

def test_roe_ending_equity_uses_only_the_latest_period():
    metrics = metrics_to_dict(fundamental_metrics(build_statements()))
    # net_income=150 (latest), ending equity=600 (latest annual period).
    assert metrics["roe_ending_equity"]["value"] == pytest.approx(150 / 600)
    assert metrics["roe_ending_equity"]["formula"] == "net_income / ending_shareholder_equity"


def test_roe_average_equity_uses_beginning_and_ending_equity():
    metrics = metrics_to_dict(fundamental_metrics(build_statements()))
    # net_income=150, beginning equity=500 (prior period), ending equity=600.
    expected = 150 / ((600 + 500) / 2)
    assert metrics["roe_average_equity"]["value"] == pytest.approx(expected)
    assert metrics["roe_average_equity"]["value"] != metrics["roe_ending_equity"]["value"], \
        "the two methodologies must diverge on this fixture, proving neither silently aliases the other"


def test_roe_average_equity_is_null_with_a_warning_when_prior_equity_is_missing():
    payload = {"annualReports": [
        {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD",
         "totalShareholderEquity": "600"}]}
    statements = normalize_all({
        "income_statement": (INCOME_PAYLOAD, {}), "balance_sheet": (payload, {})}, "TEST")
    metrics = metrics_to_dict(fundamental_metrics(statements))

    assert metrics["roe_average_equity"]["value"] is None
    assert "prior period" in metrics["roe_average_equity"]["reason"].lower()
    # roe_ending_equity must still be computable from what IS available.
    assert metrics["roe_ending_equity"]["value"] == pytest.approx(150 / 600)


def test_provider_reported_roe_stays_separate_from_locally_calculated_roe():
    """Problem 4 requirement 4: the OVERVIEW endpoint's ReturnOnEquityTTM must
    never be conflated with, or overwritten by, the locally-calculated
    roe_ending_equity / roe_average_equity metrics — they live in different
    parts of the facts structure entirely."""
    from finance.normalization import normalize_overview

    overview = normalize_overview({"ReturnOnEquityTTM": "0.34", "SharesOutstanding": "100",
                                   "MarketCapitalization": "1000"})
    fundamentals = metrics_to_dict(fundamental_metrics(build_statements()))

    assert overview["return_on_equity_ttm"] == pytest.approx(0.34)
    assert "roe_ending_equity" not in overview
    assert "return_on_equity_ttm" not in fundamentals
    assert overview["return_on_equity_ttm"] != fundamentals["roe_ending_equity"]["value"]


# ---- GAAP labeling (Problem 5) ----

def test_growth_metrics_are_labeled_gaap_and_nothing_else_is():
    metrics = metrics_to_dict(fundamental_metrics(build_statements()))
    assert metrics["revenue_growth_yoy"]["accounting_basis"] == "GAAP"
    assert metrics["net_income_growth_yoy"]["accounting_basis"] == "GAAP"
    # Metrics with no accounting-basis dimension (ratios, not growth figures)
    # must not have a stray label invented for them.
    assert "accounting_basis" not in metrics["gross_margin"]
    assert "accounting_basis" not in metrics["debt_to_equity"]


# ---- debt aggregation reuse (Problem 2 <-> ROIC/debt-to-equity) ----

def test_debt_to_equity_and_roic_use_the_reconciled_total_debt_field():
    """debt_to_equity and ROIC must consume the SAME normalized total_debt the
    balance sheet derives (short_term_debt + current_portion_of_long_term_debt
    + long_term_debt) — not re-sum short+long term debt independently, which
    would silently reintroduce the current-portion undercount Problem 2 fixed."""
    payload = {"annualReports": [
        {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD",
         "totalShareholderEquity": "600", "shortTermDebt": "50",
         "currentLongTermDebt": "40", "longTermDebt": "200"}]}
    statements = normalize_all({
        "income_statement": (INCOME_PAYLOAD, {}), "balance_sheet": (payload, {})}, "TEST")
    metrics = metrics_to_dict(fundamental_metrics(statements))

    # total_debt = 50 + 40 + 200 = 290, NOT 250 (short+long only).
    assert metrics["debt_to_equity"]["value"] == pytest.approx(290 / 600)


def test_share_count_trend_detects_buybacks():
    metrics = metrics_to_dict(fundamental_metrics(build_statements()))
    assert metrics["share_count_change"]["value"] == pytest.approx((100 - 105) / 105)


def test_every_metric_carries_formula_and_version():
    for metric in fundamental_metrics(build_statements()):
        assert metric["formula"]
        assert metric["calculation_version"]
        assert "inputs" in metric
        assert metric["basis"] == ValueBasis.CALCULATED


def test_missing_inputs_yield_none_with_a_reason_never_zero():
    empty = normalize_all({"income_statement": ({"annualReports": []}, {})}, "TEST")
    metrics = metrics_to_dict(fundamental_metrics(empty))

    for name in ("gross_margin", "current_ratio", "free_cash_flow",
                 "return_on_invested_capital"):
        assert metrics[name]["value"] is None, f"{name} must not be fabricated"
        assert metrics[name].get("reason")


def test_growth_from_a_negative_base_is_undefined_not_misleading():
    payload = {"annualReports": [
        {"fiscalDateEnding": "2025-12-31", "totalRevenue": "100", "netIncome": "50"},
        {"fiscalDateEnding": "2024-12-31", "totalRevenue": "80", "netIncome": "-20"},
    ]}
    statements = normalize_all({"income_statement": (payload, {})}, "TEST")
    metrics = metrics_to_dict(fundamental_metrics(statements))
    assert metrics["net_income_growth_yoy"]["value"] is None


def test_division_by_zero_yields_none_not_infinity():
    payload = {"annualReports": [
        {"fiscalDateEnding": "2025-12-31", "totalRevenue": "0", "grossProfit": "100"}]}
    statements = normalize_all({"income_statement": (payload, {})}, "TEST")
    metrics = metrics_to_dict(fundamental_metrics(statements))
    assert metrics["gross_margin"]["value"] is None


# ---- technical indicators ----

def test_simple_moving_average():
    assert simple_moving_average([1, 2, 3, 4, 5], 5) == pytest.approx(3.0)
    assert simple_moving_average([1, 2, 3, 4, 5], 2) == pytest.approx(4.5)
    assert simple_moving_average([1, 2], 5) is None, "too little history must be None"


def test_exponential_moving_average_seeds_with_sma():
    # Seed = SMA of first 3 = 2.0; then 4 -> 3.0, 5 -> 4.0 (multiplier 0.5).
    assert exponential_moving_average([1, 2, 3, 4, 5], 3) == pytest.approx(4.0)
    assert exponential_moving_average([1, 2], 3) is None


def test_rsi_of_a_monotonic_rise_is_one_hundred():
    assert relative_strength_index(list(range(1, 40)), 14) == pytest.approx(100.0)


def test_rsi_of_a_monotonic_fall_is_zero():
    assert relative_strength_index(list(range(40, 1, -1)), 14) == pytest.approx(0.0)


def test_rsi_needs_enough_history():
    assert relative_strength_index([1, 2, 3], 14) is None


def test_maximum_drawdown_is_a_negative_ratio():
    # Peak 100, trough 60 -> -40%.
    assert maximum_drawdown([50, 100, 60, 80]) == pytest.approx(-0.40)
    assert maximum_drawdown([1, 2, 3]) == pytest.approx(0.0)


def test_total_return():
    assert total_return([100, 150]) == pytest.approx(0.5)
    assert total_return([100]) is None


def test_volatility_of_a_flat_series_is_zero():
    assert annualized_volatility([100.0] * 30) == pytest.approx(0.0)


def test_volatility_is_positive_for_a_varying_series():
    series = [100 + (5 if i % 2 else -5) for i in range(40)]
    assert annualized_volatility(series) > 0


def test_technical_metrics_report_missing_windows_rather_than_guessing():
    from finance.normalization import PriceBar

    bars = [PriceBar(date=f"2026-01-{i:02d}", open=10, high=11, low=9,
                     close=10 + i * 0.1, adjusted_close=10 + i * 0.1, volume=100)
            for i in range(1, 26)]
    metrics = metrics_to_dict(technical_metrics(bars))

    assert metrics["sma_20"]["value"] is not None
    assert metrics["sma_200"]["value"] is None, "25 bars cannot produce a 200-day average"
    assert metrics["sma_200"]["reason"]
    assert metrics["latest_close"]["value"] == pytest.approx(12.5)
    assert metrics["price_vs_moving_averages"]["value"] is None


def test_no_technical_metric_is_nan_or_infinite():
    from finance.normalization import PriceBar

    bars = [PriceBar(date=f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", open=10, high=11,
                     low=9, close=100 + i, adjusted_close=100 + i, volume=100)
            for i in range(220)]
    for metric in technical_metrics(bars):
        value = metric["value"]
        if isinstance(value, float):
            assert not math.isnan(value) and not math.isinf(value)
