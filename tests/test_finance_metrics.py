"""COR corrective patch (Phase 9) — finance/metrics.py::fundamental_metrics.

COR's own live report showed a very high debt_to_equity ratio (~5.08) driven
mainly by a small equity base (~$1.5B) against a comparatively ordinary debt
load (~$7.7B) -- not evidence of severe balance-sheet distress by itself.
This file tests the five metrics added so the RiskReviewer has broader
leverage/liquidity context available (finance/research_pipeline.py's
_risk_reviewer_prompt now instructs it not to treat debt_to_equity alone as
sufficient evidence of severe risk): net_debt, net_debt_to_fcf, debt_to_fcf,
operating_cash_flow, and interest_coverage ("if supported" -- None, never
invented, when the input this project's data model does not capture).

Pre-existing metrics (debt_to_equity, current_ratio, free_cash_flow, ...) are
already covered end-to-end by the real-data regression fixtures
(test_finance_cost_regression.py, test_finance_cor_regression.py,
test_finance_msft_regression.py); this file is scoped to what Phase 9 added.
"""

import pytest

from finance.metrics import fundamental_metrics, metrics_to_dict
from finance.normalization import FinancialPeriod, NormalizedStatements


def _period(dataset_id, fiscal_date, **values):
    return FinancialPeriod(fiscal_date=fiscal_date, period_type="annual", currency="USD",
                           values=values, dataset_id=dataset_id)


def _statements(income=None, balance=None, cash=None):
    annual = {}
    if income is not None:
        annual["income_statement"] = income
    if balance is not None:
        annual["balance_sheet"] = balance
    if cash is not None:
        annual["cash_flow"] = cash
    return NormalizedStatements(symbol="TEST", annual=annual)


def _metrics(statements):
    return metrics_to_dict(fundamental_metrics(statements))


# ---- net_debt: reuses finance.dcf.compute_net_debt's CASH_ONLY policy ----

def test_net_debt_is_total_debt_minus_cash():
    statements = _statements(
        balance=[_period("sec_company_facts", "2025-12-31",
                         total_debt=7_660_773_000.0, cash_and_cash_equivalents=4_356_138_000.0,
                         shareholder_equity=1_508_019_000.0)])
    metrics = _metrics(statements)
    assert metrics["net_debt"]["value"] == 7_660_773_000.0 - 4_356_138_000.0


def test_net_debt_can_be_negative_for_a_net_cash_position():
    """COST's own real shape (Arc A's fixture): more cash than debt. Never
    clamped to zero -- a genuine net-cash position must read as negative,
    exactly like finance/dcf.py::compute_net_debt's own contract."""
    statements = _statements(
        balance=[_period("sec_company_facts", "2025-08-31",
                         total_debt=5_788_000_000.0, cash_and_cash_equivalents=14_161_000_000.0,
                         shareholder_equity=10_000_000_000.0)])
    metrics = _metrics(statements)
    assert metrics["net_debt"]["value"] == 5_788_000_000.0 - 14_161_000_000.0
    assert metrics["net_debt"]["value"] < 0


def test_net_debt_is_none_when_total_debt_is_missing():
    statements = _statements(
        balance=[_period("sec_company_facts", "2025-12-31",
                         cash_and_cash_equivalents=4_356_138_000.0, shareholder_equity=1.0)])
    metrics = _metrics(statements)
    assert metrics["net_debt"]["value"] is None
    assert metrics["net_debt"]["reason"]


def test_net_debt_is_none_when_cash_is_missing():
    statements = _statements(
        balance=[_period("sec_company_facts", "2025-12-31",
                         total_debt=7_660_773_000.0, shareholder_equity=1.0)])
    metrics = _metrics(statements)
    assert metrics["net_debt"]["value"] is None


def test_net_debt_is_none_when_balance_sheet_is_entirely_absent():
    statements = _statements()
    metrics = _metrics(statements)
    assert metrics["net_debt"]["value"] is None


# ---- net_debt_to_fcf / debt_to_fcf: leverage relative to cash-generating capacity ----

def _cor_shaped_statements(operating_cash_flow=3_875_120_000.0, capital_expenditure=667_981_000.0):
    return _statements(
        balance=[_period("sec_company_facts", "2025-09-30",
                         total_debt=7_660_773_000.0, cash_and_cash_equivalents=4_356_138_000.0,
                         shareholder_equity=1_508_019_000.0, current_assets=52_239_073_000.0,
                         current_liabilities=57_820_539_000.0)],
        cash=[_period("sec_company_facts", "2025-09-30",
                      operating_cash_flow=operating_cash_flow,
                      capital_expenditure=capital_expenditure)])


def test_net_debt_to_fcf_and_debt_to_fcf_match_hand_computed_values():
    statements = _cor_shaped_statements()
    metrics = _metrics(statements)
    free_cash_flow = 3_875_120_000.0 - 667_981_000.0
    net_debt = 7_660_773_000.0 - 4_356_138_000.0
    assert metrics["free_cash_flow"]["value"] == free_cash_flow
    assert metrics["net_debt_to_fcf"]["value"] == net_debt / free_cash_flow
    assert metrics["debt_to_fcf"]["value"] == 7_660_773_000.0 / free_cash_flow


def test_net_debt_to_fcf_is_none_when_free_cash_flow_is_unavailable():
    """net_debt itself IS known here (balance-sheet inputs present) -- only
    free_cash_flow is missing (no cash-flow statement at all) -- proving the
    two failure modes are independent, never silently defaulted together."""
    statements = _statements(
        balance=[_period("sec_company_facts", "2025-09-30",
                         total_debt=7_660_773_000.0, cash_and_cash_equivalents=4_356_138_000.0,
                         shareholder_equity=1_508_019_000.0)])
    metrics = _metrics(statements)
    assert metrics["net_debt"]["value"] is not None
    assert metrics["net_debt_to_fcf"]["value"] is None
    assert metrics["debt_to_fcf"]["value"] is None


# ---- operating_cash_flow: exposed as its own citable metric (Phase 9) ----

def test_operating_cash_flow_is_exposed_as_a_directly_reported_metric():
    statements = _cor_shaped_statements()
    metrics = _metrics(statements)
    assert metrics["operating_cash_flow"]["value"] == 3_875_120_000.0
    assert metrics["operating_cash_flow"]["basis"] == "calculated"  # ValueBasis envelope, not a claim of derivation


def test_operating_cash_flow_is_none_when_not_reported():
    statements = _statements(cash=[_period("sec_company_facts", "2025-09-30",
                                          capital_expenditure=667_981_000.0)])
    metrics = _metrics(statements)
    assert metrics["operating_cash_flow"]["value"] is None


# ---- interest_coverage: "if supported" -- never invented when absent ----

def test_interest_coverage_is_none_when_interest_expense_is_not_reported():
    """This project's SEC concept map (finance/xbrl_mapping.py) does not
    currently capture an interest-expense XBRL concept -- the real, honest
    shape for every SEC-sourced company this codebase has fixtures for
    (COST, COR). Must be None with a reason, never invented or defaulted."""
    statements = _statements(
        income=[_period("sec_company_facts", "2025-09-30",
                        revenue=100.0, operating_income=10.0)])
    metrics = _metrics(statements)
    assert metrics["interest_coverage"]["value"] is None
    assert metrics["interest_coverage"]["reason"]


def test_interest_coverage_is_computed_when_interest_expense_is_reported():
    """The Alpha-Vantage-shaped case: interestExpense IS mapped
    (finance/normalization.py's _INCOME_FIELDS), so this must genuinely
    compute rather than staying None just because SOME providers lack it."""
    statements = _statements(
        income=[_period("income_statement", "2025-09-30",
                        revenue=100.0, operating_income=20.0, interest_expense=4.0)])
    metrics = _metrics(statements)
    assert metrics["interest_coverage"]["value"] == 5.0


def test_interest_coverage_prefers_ebit_over_operating_income_when_both_are_reported():
    statements = _statements(
        income=[_period("income_statement", "2025-09-30",
                        revenue=100.0, operating_income=20.0, ebit=25.0, interest_expense=5.0)])
    metrics = _metrics(statements)
    assert metrics["interest_coverage"]["value"] == 5.0  # 25 / 5, not 20 / 5


# ---- the COR scenario itself: high debt_to_equity alongside full broader context ----

def test_high_debt_to_equity_still_ships_alongside_broader_leverage_context():
    """The property Phase 9 actually asks for: when a company's debt_to_equity
    is high mainly because of a small equity base (COR's real shape), the
    OTHER context a RiskReviewer needs is simultaneously available, not
    absent -- debt_to_equity is not the only number on the table."""
    statements = _cor_shaped_statements()
    metrics = _metrics(statements)

    debt_to_equity = metrics["debt_to_equity"]["value"]
    assert debt_to_equity > 4.0, "this scenario is deliberately COR-shaped: small equity base, real debt"

    for name in ("net_debt", "net_debt_to_fcf", "debt_to_fcf", "current_ratio", "operating_cash_flow"):
        assert metrics[name]["value"] is not None, f"{name} should be available given these inputs"

    # interest_coverage is the one exception -- "if supported" -- absent here
    # because interest_expense was never in this fixture's income statement,
    # matching COR's/COST's own real, SEC-sourced shape.
    assert metrics["interest_coverage"]["value"] is None


# ---- H.4 corrective patch: negative-equity applicability (goal 1) ----
#
# Live MO (Altria) shape: a profitable company whose shareholder equity is
# NEGATIVE (large, sustained buybacks exceeding retained earnings) -- ROE and
# debt_to_equity are mathematically computable but economically misleading
# (e.g. ROE = net_income / a small negative number reads like a catastrophic
# loss it does not represent). These must report as not_meaningful, never as
# an ordinary signed ratio.

from finance.metrics import (
    REASON_NEGATIVE_AVERAGE_SHAREHOLDER_EQUITY,
    REASON_NEGATIVE_SHAREHOLDER_EQUITY,
    STATUS_NOT_MEANINGFUL,
)


def _mo_shaped_statements(equity=-800.0, prior_equity=-650.0, net_income=400.0):
    return _statements(
        income=[_period("sec_company_facts", "2025-12-31", revenue=1000.0,
                        net_income=net_income, operating_income=500.0)],
        balance=[
            _period("sec_company_facts", "2025-12-31",
                    total_debt=4500.0, cash_and_cash_equivalents=600.0,
                    shareholder_equity=equity, current_assets=1200.0, current_liabilities=900.0),
            _period("sec_company_facts", "2024-12-31",
                    total_debt=4380.0, cash_and_cash_equivalents=550.0,
                    shareholder_equity=prior_equity, current_assets=1100.0, current_liabilities=850.0),
        ],
        cash=[_period("sec_company_facts", "2025-12-31",
                      operating_cash_flow=550.0, capital_expenditure=60.0)])


def test_roe_ending_equity_is_not_meaningful_when_shareholder_equity_is_negative():
    """Test 1."""
    metrics = _metrics(_mo_shaped_statements())
    roe = metrics["roe_ending_equity"]
    assert roe["value"] is None
    assert roe["status"] == STATUS_NOT_MEANINGFUL
    assert roe["reason"] == REASON_NEGATIVE_SHAREHOLDER_EQUITY


def test_roe_ending_equity_is_not_meaningful_when_shareholder_equity_is_exactly_zero():
    metrics = _metrics(_mo_shaped_statements(equity=0.0))
    roe = metrics["roe_ending_equity"]
    assert roe["value"] is None
    assert roe["status"] == STATUS_NOT_MEANINGFUL


def test_roe_average_equity_is_not_meaningful_when_the_average_is_negative():
    """Test 2: both periods negative -> the average is negative too."""
    metrics = _metrics(_mo_shaped_statements())
    roe_avg = metrics["roe_average_equity"]
    assert roe_avg["value"] is None
    assert roe_avg["status"] == STATUS_NOT_MEANINGFUL
    assert roe_avg["reason"] == REASON_NEGATIVE_AVERAGE_SHAREHOLDER_EQUITY


def test_roe_average_equity_is_not_meaningful_when_the_average_straddles_zero_negative():
    """Ending equity is barely POSITIVE but the prior period was so deeply
    negative that the two-period AVERAGE is still <= 0 -- the average, not
    just the ending figure, is what must be checked."""
    metrics = _metrics(_mo_shaped_statements(equity=50.0, prior_equity=-500.0))
    assert metrics["roe_ending_equity"].get("status") != STATUS_NOT_MEANINGFUL
    assert metrics["roe_ending_equity"]["value"] is not None
    roe_avg = metrics["roe_average_equity"]
    assert roe_avg["value"] is None
    assert roe_avg["status"] == STATUS_NOT_MEANINGFUL
    assert roe_avg["reason"] == REASON_NEGATIVE_AVERAGE_SHAREHOLDER_EQUITY


def test_roe_average_equity_remains_ordinary_when_the_average_is_positive():
    """Sanity check: a genuinely positive average is NOT flagged -- this
    patch must not regress the ordinary case."""
    metrics = _metrics(_mo_shaped_statements(equity=600.0, prior_equity=500.0, net_income=150.0))
    roe_avg = metrics["roe_average_equity"]
    assert roe_avg.get("status") != STATUS_NOT_MEANINGFUL
    assert roe_avg["value"] == pytest.approx(150.0 / 550.0)


def test_debt_to_equity_is_not_meaningful_when_shareholder_equity_is_negative():
    """Test 3."""
    metrics = _metrics(_mo_shaped_statements())
    dte = metrics["debt_to_equity"]
    assert dte["value"] is None
    assert dte["status"] == STATUS_NOT_MEANINGFUL
    assert dte["reason"] == REASON_NEGATIVE_SHAREHOLDER_EQUITY


def test_debt_to_equity_remains_an_ordinary_ratio_when_equity_is_positive():
    """Sanity check against regressing the COR-shaped high-but-positive-equity
    case covered above."""
    metrics = _metrics(_cor_shaped_statements())
    dte = metrics["debt_to_equity"]
    assert dte.get("status") != STATUS_NOT_MEANINGFUL
    assert dte["value"] is not None


def test_negative_shareholder_equity_remains_available_as_its_own_metric():
    """Test 4: the underlying (negative) equity value is NEVER hidden, only
    the misleading ratios computed FROM it are suppressed."""
    metrics = _metrics(_mo_shaped_statements(equity=-800.0))
    equity_metric = metrics["shareholder_equity"]
    assert equity_metric["value"] == -800.0
    assert equity_metric.get("status") != STATUS_NOT_MEANINGFUL


def test_total_debt_remains_available_as_its_own_metric_for_leverage_context():
    """Companion to the above: total_debt must be independently citable so a
    RiskReviewer can discuss leverage without debt_to_equity."""
    metrics = _metrics(_mo_shaped_statements())
    assert metrics["total_debt"]["value"] == 4500.0


def test_no_negative_roe_value_is_ever_reported_for_a_negative_equity_company():
    """Test 6: structurally, a not_meaningful metric's `value` is None, never
    a large negative number -- there is nothing for a downstream reader to
    misinterpret as 'profitability collapsed by X%', because no percentage
    is ever produced in this case."""
    metrics = _metrics(_mo_shaped_statements(equity=-800.0, net_income=400.0))
    assert metrics["roe_ending_equity"]["value"] is None
    assert metrics["roe_average_equity"]["value"] is None
    assert metrics["debt_to_equity"]["value"] is None
    # The other leverage/liquidity metrics the RiskReviewer is instructed to
    # prefer instead all remain genuinely computed, not also suppressed.
    for name in ("net_debt", "net_debt_to_fcf", "debt_to_fcf", "current_ratio",
                "operating_cash_flow", "total_debt", "shareholder_equity"):
        assert metrics[name]["value"] is not None, name
