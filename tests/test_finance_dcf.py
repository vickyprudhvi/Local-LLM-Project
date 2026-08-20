"""Phase H.1 — the deterministic DCF engine.

Expected values are calculated INDEPENDENTLY in the test (see
`_expected_fixture_valuation`) from the same published formulas, not by calling
the engine and recording whatever it produced. A regression in `finance/dcf.py`
therefore fails these tests rather than silently rewriting the baseline.

Fixture (deliberately round numbers so the arithmetic is checkable by hand):

    base revenue        1,000.0
    horizon             3 years
    revenue growth      10% every year
    operating margin    20% every year
    tax rate            25%
    D&A                 5% of revenue
    capex               7% of revenue
    working capital     10% of revenue   (base working capital = 100.0)
    WACC                10%
    terminal growth     2%
    diluted shares      100
    total debt          250   (net debt 200 under the default cash_only policy)
    cash                50
"""

import json

import pytest

import tools.config as config
from finance.dcf import (
    CALCULATION_VERSION,
    MODEL_TYPE,
    AssumptionSourceType,
    DcfInputs,
    DcfValidationStatus,
    NetDebtPolicy,
    ScenarioResultStatus,
    _scenario_monotonicity_check,
    build_scenario,
    compute_net_debt,
    run_dcf,
    value_scenario,
)
from tools.base import ToolFailure
from tools.models import (
    DCF_ASSUMPTION_INVALID,
    DCF_ASSUMPTION_REQUIRED,
    DCF_CURRENCY_MISMATCH,
    DCF_DISCOUNT_RATE_OUT_OF_BOUNDS,
    DCF_FORECAST_HORIZON_INVALID,
    DCF_SCENARIO_INVALID,
    DCF_SHARES_INVALID,
    DCF_TERMINAL_GROWTH_TOO_HIGH,
)

BASE_REVENUE = 1000.0
HORIZON = 3
GROWTH = 0.10
MARGIN = 0.20
TAX = 0.25
DA_PCT = 0.05
CAPEX_PCT = 0.07
WC_PCT = 0.10
WACC = 0.10
TERMINAL_GROWTH = 0.02
SHARES = 100.0
TOTAL_DEBT = 250.0
CASH = 50.0
NET_DEBT = TOTAL_DEBT - CASH  # 200, under the default cash_only policy
TOLERANCE = 1e-6


def base_inputs(**overrides):
    kwargs = dict(
        ticker="TEST", valuation_date="2026-08-05", currency="USD",
        base_revenue=BASE_REVENUE, forecast_years=HORIZON, diluted_shares=SHARES,
        total_debt=TOTAL_DEBT, cash_and_cash_equivalents=CASH,
        base_working_capital=BASE_REVENUE * WC_PCT,
        source_periods=("FY2025", "FY2024", "FY2023"),
    )
    kwargs.update(overrides)
    return DcfInputs(**kwargs)


def base_scenario(name="base", **overrides):
    raw = {
        "name": name,
        "revenue_growth": GROWTH,
        "operating_margin": MARGIN,
        "tax_rate": TAX,
        "depreciation_pct_revenue": DA_PCT,
        "capex_pct_revenue": CAPEX_PCT,
        "working_capital_pct_revenue": WC_PCT,
        "wacc": WACC,
        "terminal_growth": TERMINAL_GROWTH,
    }
    raw.update(overrides)
    return raw


def _expected_fixture_valuation():
    """Recompute the fixture from first principles, independently of the engine."""
    revenue = BASE_REVENUE
    prior_wc = BASE_REVENUE * WC_PCT
    rows = []
    for year in range(1, HORIZON + 1):
        revenue = revenue * (1 + GROWTH)
        ebit = revenue * MARGIN
        nopat = ebit * (1 - TAX)
        da = revenue * DA_PCT
        capex = revenue * CAPEX_PCT
        wc = revenue * WC_PCT
        change_wc = wc - prior_wc
        prior_wc = wc
        fcff = nopat + da - capex - change_wc
        discount = 1 / ((1 + WACC) ** year)
        rows.append({"year": year, "revenue": revenue, "ebit": ebit, "fcff": fcff,
                     "discount_factor": discount, "pv": fcff * discount})

    pv_forecast = sum(r["pv"] for r in rows)
    terminal = rows[-1]["fcff"] * (1 + TERMINAL_GROWTH) / (WACC - TERMINAL_GROWTH)
    pv_terminal = terminal * rows[-1]["discount_factor"]
    enterprise = pv_forecast + pv_terminal
    equity = enterprise - NET_DEBT
    return {
        "rows": rows, "pv_forecast": pv_forecast, "terminal_value": terminal,
        "pv_terminal": pv_terminal, "enterprise_value": enterprise,
        "equity_value": equity, "value_per_share": equity / SHARES,
    }


EXPECTED = _expected_fixture_valuation()


@pytest.fixture
def result():
    return value_scenario(base_inputs(), build_scenario(base_scenario(), HORIZON))


# ---- 1: FCFF ----

def test_fcff_matches_independent_calculation(result):
    for index, expected_row in enumerate(EXPECTED["rows"]):
        actual = result["forecast"][index]
        assert actual["year"] == expected_row["year"]
        assert actual["revenue"] == pytest.approx(expected_row["revenue"], abs=TOLERANCE)
        assert actual["ebit"] == pytest.approx(expected_row["ebit"], abs=TOLERANCE)
        assert actual["fcff"] == pytest.approx(expected_row["fcff"], abs=TOLERANCE)


def test_fcff_components_follow_the_documented_formula(result):
    """FCFF = EBIT x (1-t) + D&A - capex - change in NWC, term by term."""
    row = result["forecast"][0]
    revenue_y1 = BASE_REVENUE * (1 + GROWTH)  # 1100
    assert row["revenue"] == pytest.approx(1100.0, abs=TOLERANCE)
    assert row["ebit"] == pytest.approx(220.0, abs=TOLERANCE)          # 1100 x 0.20
    assert row["nopat"] == pytest.approx(165.0, abs=TOLERANCE)         # 220 x 0.75
    assert row["depreciation_amortization"] == pytest.approx(55.0, abs=TOLERANCE)
    assert row["capital_expenditure"] == pytest.approx(77.0, abs=TOLERANCE)
    assert row["change_in_net_working_capital"] == pytest.approx(10.0, abs=TOLERANCE)
    assert row["fcff"] == pytest.approx(165.0 + 55.0 - 77.0 - 10.0, abs=TOLERANCE)


# ---- 2: discount factors ----

def test_discount_factors_are_correct(result):
    for year in (1, 2, 3):
        expected = 1 / (1.10 ** year)
        assert result["forecast"][year - 1]["discount_factor"] == pytest.approx(
            expected, abs=TOLERANCE)


def test_present_value_is_fcff_times_discount_factor(result):
    for row in result["forecast"]:
        assert row["present_value_fcff"] == pytest.approx(
            row["fcff"] * row["discount_factor"], abs=1e-4)


# ---- 3-6: valuation bridge ----

def test_terminal_value(result):
    assert result["terminal_value"] == pytest.approx(EXPECTED["terminal_value"], abs=1e-4)


def test_enterprise_value(result):
    assert result["enterprise_value"] == pytest.approx(
        EXPECTED["enterprise_value"], abs=1e-4)
    assert result["enterprise_value"] == pytest.approx(
        result["present_value_of_forecast_fcff"]
        + result["present_value_of_terminal_value"], abs=1e-4)


def test_equity_value_applies_the_net_debt_bridge(result):
    assert result["equity_value"] == pytest.approx(EXPECTED["equity_value"], abs=1e-4)
    assert result["equity_value"] == pytest.approx(
        result["enterprise_value"] - NET_DEBT, abs=1e-4)


def test_equity_bridge_includes_every_declared_component():
    inputs = base_inputs(preferred_equity=50.0, minority_interest=25.0,
                         other_non_operating_assets=75.0)
    out = value_scenario(inputs, build_scenario(base_scenario(), HORIZON))
    assert out["equity_value"] == pytest.approx(
        out["enterprise_value"] - NET_DEBT + 75.0 - 50.0 - 25.0, abs=1e-4)


def test_value_per_share(result):
    assert result["value_per_share"] == pytest.approx(
        EXPECTED["value_per_share"], abs=1e-4)
    assert result["value_per_share"] == pytest.approx(
        result["equity_value"] / SHARES, abs=1e-4)


# ---- 7: scenarios ----

def test_base_bull_and_bear_scenarios_are_ordered_sensibly():
    out = run_dcf(base_inputs(), [
        base_scenario("base"),
        base_scenario("bull", revenue_growth=0.15, operating_margin=0.24),
        base_scenario("bear", revenue_growth=0.04, operating_margin=0.16),
    ])
    by_name = {s["scenario"]: s["value_per_share"] for s in out["scenarios"]}

    assert set(by_name) == {"base", "bull", "bear"}
    assert by_name["bull"] > by_name["base"] > by_name["bear"]
    assert out["primary_scenario"] == "base"
    assert out["value_per_share"] == by_name["base"]


def test_duplicate_scenario_names_are_rejected():
    with pytest.raises(ToolFailure) as excinfo:
        run_dcf(base_inputs(), [base_scenario("base"), base_scenario("base")])
    assert excinfo.value.code == DCF_SCENARIO_INVALID


def test_per_year_assumption_arrays_are_honoured():
    out = value_scenario(
        base_inputs(),
        build_scenario(base_scenario(revenue_growth=[0.20, 0.10, 0.05]), HORIZON))
    revenues = [r["revenue"] for r in out["forecast"]]
    assert revenues[0] == pytest.approx(1200.0, abs=TOLERANCE)
    assert revenues[1] == pytest.approx(1320.0, abs=TOLERANCE)
    assert revenues[2] == pytest.approx(1386.0, abs=TOLERANCE)


def test_forecast_array_length_must_match_the_horizon():
    with pytest.raises(ToolFailure) as excinfo:
        build_scenario(base_scenario(revenue_growth=[0.1, 0.1]), HORIZON)
    assert excinfo.value.code == DCF_ASSUMPTION_INVALID


# ---- 8: sensitivity ----

def test_sensitivity_matrix_shape_and_values():
    out = run_dcf(base_inputs(), [base_scenario()], sensitivity={
        "wacc_values": [0.08, 0.10, 0.12],
        "terminal_growth_values": [0.01, 0.02, 0.03],
    })
    grid = out["sensitivity"]

    assert len(grid["rows"]) == 3
    assert all(len(row["cells"]) == 3 for row in grid["rows"])

    # The cell matching the base scenario must reproduce the base valuation.
    centre = grid["rows"][1]["cells"][1]
    assert centre["wacc"] == pytest.approx(0.10)
    assert centre["terminal_growth"] == pytest.approx(0.02)
    assert centre["value_per_share"] == pytest.approx(
        EXPECTED["value_per_share"], abs=1e-3)

    # Value falls as WACC rises and rises as terminal growth rises.
    low_wacc = grid["rows"][0]["cells"][1]["value_per_share"]
    high_wacc = grid["rows"][2]["cells"][1]["value_per_share"]
    assert low_wacc > centre["value_per_share"] > high_wacc


# ---- 9: terminal growth >= WACC ----

def test_terminal_growth_equal_to_wacc_is_rejected():
    with pytest.raises(ToolFailure) as excinfo:
        build_scenario(base_scenario(terminal_growth=0.10, wacc=0.10), HORIZON)
    assert excinfo.value.code == DCF_TERMINAL_GROWTH_TOO_HIGH


def test_terminal_growth_above_wacc_is_rejected():
    with pytest.raises(ToolFailure) as excinfo:
        build_scenario(base_scenario(terminal_growth=0.15, wacc=0.10), HORIZON)
    assert excinfo.value.code == DCF_TERMINAL_GROWTH_TOO_HIGH


def test_sensitivity_rejects_invalid_combinations_rather_than_clamping():
    out = run_dcf(base_inputs(), [base_scenario()], sensitivity={
        "wacc_values": [0.04],
        "terminal_growth_values": [0.02, 0.04, 0.06],
    })
    cells = out["sensitivity"]["rows"][0]["cells"]

    assert cells[0]["rejected"] is False and cells[0]["value_per_share"] is not None
    for cell in cells[1:]:
        assert cell["rejected"] is True
        assert cell["value_per_share"] is None, "an undefined perpetuity must carry no number"
        assert cell["reason"] == DCF_TERMINAL_GROWTH_TOO_HIGH


# ---- 10: missing assumptions ----

@pytest.mark.parametrize("field", [
    "revenue_growth", "operating_margin", "tax_rate", "depreciation_pct_revenue",
    "capex_pct_revenue", "working_capital_pct_revenue", "wacc", "terminal_growth",
])
def test_missing_assumption_is_reported_not_defaulted(field):
    raw = base_scenario()
    del raw[field]
    with pytest.raises(ToolFailure) as excinfo:
        build_scenario(raw, HORIZON)
    assert excinfo.value.code == DCF_ASSUMPTION_REQUIRED
    assert field in excinfo.value.message


def test_no_scenarios_at_all_is_reported():
    with pytest.raises(ToolFailure) as excinfo:
        run_dcf(base_inputs(), [])
    assert excinfo.value.code == DCF_ASSUMPTION_REQUIRED


# ---- 11-13: validation ----

@pytest.mark.parametrize("shares", [0, -100, -0.5])
def test_invalid_shares_are_rejected(shares):
    with pytest.raises(ToolFailure) as excinfo:
        run_dcf(base_inputs(diluted_shares=shares), [base_scenario()])
    assert excinfo.value.code == DCF_SHARES_INVALID


def test_currency_mismatch_is_rejected():
    with pytest.raises(ToolFailure) as excinfo:
        run_dcf(base_inputs(currency="USD", statement_currency="EUR"), [base_scenario()])
    assert excinfo.value.code == DCF_CURRENCY_MISMATCH


@pytest.mark.parametrize("years", [0, -1, 99])
def test_forecast_horizon_bounds_are_enforced(years):
    with pytest.raises(ToolFailure) as excinfo:
        run_dcf(base_inputs(forecast_years=years), [base_scenario()])
    assert excinfo.value.code == DCF_FORECAST_HORIZON_INVALID


@pytest.mark.parametrize("wacc", [0.001, 0.95])
def test_discount_rate_bounds_are_enforced(wacc):
    with pytest.raises(ToolFailure) as excinfo:
        build_scenario(base_scenario(wacc=wacc, terminal_growth=0.0), HORIZON)
    assert excinfo.value.code == DCF_DISCOUNT_RATE_OUT_OF_BOUNDS


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinity_assumptions_are_rejected(bad):
    with pytest.raises(ToolFailure) as excinfo:
        build_scenario(base_scenario(revenue_growth=bad), HORIZON)
    assert excinfo.value.code == DCF_ASSUMPTION_INVALID


def test_no_non_finite_value_appears_anywhere_in_the_output():
    out = run_dcf(base_inputs(), [base_scenario()], sensitivity={
        "wacc_values": [0.08, 0.12], "terminal_growth_values": [0.01, 0.03]})

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, float):
            assert node == node, "NaN in output"          # noqa: PLR0124
            assert node not in (float("inf"), float("-inf")), "infinity in output"

    walk(out)


def test_non_numeric_assumption_is_rejected():
    with pytest.raises(ToolFailure) as excinfo:
        build_scenario(base_scenario(wacc="ten percent"), HORIZON)
    assert excinfo.value.code == DCF_ASSUMPTION_INVALID


# ---- 14-15: stability ----

def test_output_is_json_serializable_and_stable():
    out = run_dcf(base_inputs(), [base_scenario()], sensitivity={
        "wacc_values": [0.09, 0.11], "terminal_growth_values": [0.015, 0.025]})
    encoded = json.dumps(out, sort_keys=True)
    assert isinstance(encoded, str)
    assert out["model_type"] == MODEL_TYPE
    assert out["calculation_version"] == CALCULATION_VERSION


def test_repeat_runs_with_identical_input_produce_identical_output():
    sensitivity = {"wacc_values": [0.09, 0.10, 0.11],
                   "terminal_growth_values": [0.015, 0.02, 0.025]}
    scenarios = [base_scenario("base"), base_scenario("bull", revenue_growth=0.15)]

    first = run_dcf(base_inputs(), scenarios, sensitivity=sensitivity)
    second = run_dcf(base_inputs(), scenarios, sensitivity=sensitivity)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_assumptions_are_preserved_exactly_in_the_output(result):
    used = result["assumptions"]
    assert used["wacc"] == pytest.approx(WACC)
    assert used["terminal_growth"] == pytest.approx(TERMINAL_GROWTH)
    # Phase H.8: `tax_rate` became a per-year SERIES, like revenue_growth
    # and operating_margin already were, so a current year distorted by an
    # acquisition or a settlement is not repeated across the whole horizon.
    # A scalar input is still accepted and held flat, which is what this
    # scenario supplies -- so the round-trip is a flat series.
    assert used["tax_rate"] == [pytest.approx(TAX)] * HORIZON
    assert used["revenue_growth"] == [pytest.approx(GROWTH)] * HORIZON


def test_high_terminal_value_share_produces_a_warning():
    out = run_dcf(base_inputs(), [base_scenario(terminal_growth=0.06, wacc=0.08)])
    assert any("terminal value" in w for w in out["warnings"])


# ---- net-debt policy ----

def test_compute_net_debt_cash_only_policy():
    assert compute_net_debt(total_debt=250.0, cash_and_cash_equivalents=50.0,
                            eligible_short_term_investments=80.0,
                            policy=NetDebtPolicy.CASH_ONLY) == pytest.approx(200.0), \
        "cash_only must ignore eligible_short_term_investments entirely"


def test_compute_net_debt_cash_and_marketable_securities_policy():
    assert compute_net_debt(
        total_debt=250.0, cash_and_cash_equivalents=50.0,
        eligible_short_term_investments=80.0,
        policy=NetDebtPolicy.CASH_AND_MARKETABLE_SECURITIES) == pytest.approx(120.0)


def test_negative_net_debt_is_preserved_as_net_cash_never_clamped():
    net_cash = compute_net_debt(total_debt=50.0, cash_and_cash_equivalents=200.0,
                                eligible_short_term_investments=0.0,
                                policy=NetDebtPolicy.CASH_ONLY)
    assert net_cash == pytest.approx(-150.0)

    inputs = base_inputs(total_debt=50.0, cash_and_cash_equivalents=200.0)
    out = value_scenario(inputs, build_scenario(base_scenario(), HORIZON))
    assert out["net_debt"] == pytest.approx(-150.0)
    assert out["equity_value"] == pytest.approx(out["enterprise_value"] + 150.0, abs=1e-4)


def test_unknown_net_debt_policy_is_rejected():
    with pytest.raises(ToolFailure) as excinfo:
        compute_net_debt(100.0, 50.0, 0.0, policy="unlimited_leverage")
    assert excinfo.value.code == DCF_ASSUMPTION_INVALID

    with pytest.raises(ToolFailure) as excinfo:
        run_dcf(base_inputs(net_debt_policy="unlimited_leverage"), [base_scenario()])
    assert excinfo.value.code == DCF_ASSUMPTION_INVALID


def test_equity_bridge_uses_the_selected_policy():
    inputs_cash_only = base_inputs(
        total_debt=250.0, cash_and_cash_equivalents=50.0,
        short_term_investments=80.0, eligible_short_term_investments=80.0,
        net_debt_policy=NetDebtPolicy.CASH_ONLY)
    inputs_with_sti = base_inputs(
        total_debt=250.0, cash_and_cash_equivalents=50.0,
        short_term_investments=80.0, eligible_short_term_investments=80.0,
        net_debt_policy=NetDebtPolicy.CASH_AND_MARKETABLE_SECURITIES)

    out_cash_only = value_scenario(inputs_cash_only, build_scenario(base_scenario(), HORIZON))
    out_with_sti = value_scenario(inputs_with_sti, build_scenario(base_scenario(), HORIZON))

    assert out_cash_only["net_debt_policy"] == NetDebtPolicy.CASH_ONLY
    assert out_with_sti["net_debt_policy"] == NetDebtPolicy.CASH_AND_MARKETABLE_SECURITIES
    assert out_cash_only["net_debt"] == pytest.approx(200.0)
    assert out_with_sti["net_debt"] == pytest.approx(120.0)


def test_per_share_value_changes_correctly_when_short_term_investments_included():
    """Requirement 5: including eligible STI must raise equity value by EXACTLY
    the STI amount, and value per share by EXACTLY that amount / shares."""
    inputs_cash_only = base_inputs(
        total_debt=250.0, cash_and_cash_equivalents=50.0,
        short_term_investments=80.0, eligible_short_term_investments=80.0,
        net_debt_policy=NetDebtPolicy.CASH_ONLY)
    inputs_with_sti = base_inputs(
        total_debt=250.0, cash_and_cash_equivalents=50.0,
        short_term_investments=80.0, eligible_short_term_investments=80.0,
        net_debt_policy=NetDebtPolicy.CASH_AND_MARKETABLE_SECURITIES)

    out_cash_only = value_scenario(inputs_cash_only, build_scenario(base_scenario(), HORIZON))
    out_with_sti = value_scenario(inputs_with_sti, build_scenario(base_scenario(), HORIZON))

    assert out_with_sti["equity_value"] == pytest.approx(
        out_cash_only["equity_value"] + 80.0, abs=1e-4)
    assert out_with_sti["value_per_share"] == pytest.approx(
        out_cash_only["value_per_share"] + 80.0 / SHARES, abs=1e-6)


def test_short_term_investments_are_always_reported_even_when_not_eligible():
    """The raw STI figure is visible for audit even under cash_only, where it
    contributes nothing to the arithmetic."""
    inputs = base_inputs(short_term_investments=80.0, eligible_short_term_investments=0.0,
                         net_debt_policy=NetDebtPolicy.CASH_ONLY)
    out = value_scenario(inputs, build_scenario(base_scenario(), HORIZON))
    assert out["short_term_investments"] == pytest.approx(80.0)
    assert out["eligible_short_term_investments"] == pytest.approx(0.0)
    assert out["net_debt"] == pytest.approx(NET_DEBT), \
        "STI must not silently affect net debt when not eligible"


def test_run_dcf_top_level_net_debt_and_policy_match_the_scenario():
    out = run_dcf(base_inputs(), [base_scenario()])
    assert out["net_debt"] == pytest.approx(out["scenarios"][0]["net_debt"])
    assert out["net_debt_policy"] == out["scenarios"][0]["net_debt_policy"]


# ---- full yearly forecast detail (auditability) ----

def test_every_forecast_row_carries_full_audit_detail(result):
    expected_keys = {
        "year", "revenue", "revenue_growth", "operating_margin", "ebit", "tax_rate",
        "nopat", "depreciation_amortization", "capital_expenditure",
        "net_working_capital", "change_in_net_working_capital", "fcff",
        "discount_factor", "present_value_fcff",
    }
    for row in result["forecast"]:
        assert expected_keys <= set(row.keys())
        assert row["tax_rate"] == pytest.approx(TAX)


def test_terminal_year_fcff_matches_the_final_forecast_row(result):
    assert result["terminal_year_fcff"] == pytest.approx(
        result["forecast"][-1]["fcff"], abs=1e-4)


def test_scenario_result_carries_its_own_calculation_version(result):
    assert result["calculation_version"] == CALCULATION_VERSION


def test_full_detail_is_present_for_bear_base_and_bull():
    out = run_dcf(base_inputs(), [
        base_scenario("base"),
        base_scenario("bull", revenue_growth=0.15),
        base_scenario("bear", revenue_growth=0.02),
    ])
    for scenario_result in out["scenarios"]:
        assert len(scenario_result["forecast"]) == HORIZON
        assert scenario_result["terminal_year_fcff"] is not None
        assert scenario_result["net_debt_policy"]
        assert scenario_result["cash_and_cash_equivalents"] is not None
        assert scenario_result["total_debt"] is not None


# ---- assumption provenance ----

def test_assumption_provenance_is_echoed_back_exactly():
    provenance = {
        "revenue_growth": {
            "value": GROWTH, "source_type": AssumptionSourceType.DETERMINISTIC_CALCULATION,
            "source_period": "revenue_cagr (5 periods)",
            "reason": "Derived from reported revenue CAGR.",
            "approval_status": "proposed", "units": "ratio",
        },
        "wacc": {
            "value": WACC, "source_type": AssumptionSourceType.CONFIGURED_DEFAULT,
            "source_period": None, "reason": "Configured default WACC.",
            "approval_status": "proposed", "units": "ratio",
        },
    }
    scenario_raw = base_scenario(assumption_provenance=provenance)
    out = value_scenario(base_inputs(), build_scenario(scenario_raw, HORIZON))

    got = out["assumptions"]["assumption_provenance"]
    assert got["revenue_growth"]["source_type"] == AssumptionSourceType.DETERMINISTIC_CALCULATION
    assert got["revenue_growth"]["source_period"] == "revenue_cagr (5 periods)"
    assert got["wacc"]["source_type"] == AssumptionSourceType.CONFIGURED_DEFAULT


def test_assumption_provenance_rejects_an_unknown_source_type():
    scenario_raw = base_scenario(assumption_provenance={
        "wacc": {"value": WACC, "source_type": "guessed_by_vibes"}})
    with pytest.raises(ToolFailure) as excinfo:
        build_scenario(scenario_raw, HORIZON)
    assert excinfo.value.code == DCF_ASSUMPTION_INVALID


def test_assumption_provenance_rejects_an_unknown_field_name():
    scenario_raw = base_scenario(assumption_provenance={
        "not_a_real_assumption": {"value": 1, "source_type": "configured_default"}})
    with pytest.raises(ToolFailure) as excinfo:
        build_scenario(scenario_raw, HORIZON)
    assert excinfo.value.code == DCF_ASSUMPTION_INVALID


def test_assumption_provenance_is_optional():
    out = value_scenario(base_inputs(), build_scenario(base_scenario(), HORIZON))
    assert out["assumptions"]["assumption_provenance"] == {}


def test_assumption_provenance_preserves_the_current_reported_history_fields():
    """Phase H.3 corrective patch (Problem 2): 'source_periods' (plural, every
    period a multi-year average actually used), 'derivation' (the human-
    readable audit-trail sentence), and 'source_evidence_ids' are the
    CURRENT fields finance/workflow.py::propose_assumptions's
    provenance_entry() emits. A real, live-data COST regression fixture
    (tests/test_finance_cost_regression.py) caught these being silently
    dropped here -- this validator used to rebuild every entry from a fixed
    allowlist of only the ORIGINAL singular-period/free-text fields, which
    meant CapEx/D&A/NWC's whole multi-year audit trail never survived past
    this function despite being computed correctly upstream."""
    provenance = {
        "capex_pct_revenue": {
            "value": 0.0184, "source_type": AssumptionSourceType.DETERMINISTIC_CALCULATION,
            "source_periods": ["2025-08-31", "2024-09-01", "2023-09-03"],
            "source_evidence_ids": ["fundamental.capex_pct_revenue"],
            "derivation": "Average of 3 reported CapEx/revenue ratios (0.0200, 0.0185, 0.0178, "
                          "in the same order as source_periods) = 0.0184. Method: simple average.",
            "approval_status": "proposed", "units": "ratio",
        },
    }
    scenario_raw = base_scenario(assumption_provenance=provenance)
    out = value_scenario(base_inputs(), build_scenario(scenario_raw, HORIZON))

    got = out["assumptions"]["assumption_provenance"]["capex_pct_revenue"]
    assert got["source_periods"] == ["2025-08-31", "2024-09-01", "2023-09-03"]
    assert got["source_evidence_ids"] == ["fundamental.capex_pct_revenue"]
    assert got["derivation"] == ("Average of 3 reported CapEx/revenue ratios (0.0200, 0.0185, "
                                 "0.0178, in the same order as source_periods) = 0.0184. Method: "
                                 "simple average.")
    # backward-compat singular/legacy fields still present alongside the new ones
    assert got["source_period"] is None  # this entry never set the singular alias
    assert got["reason"] is None  # this entry never set the legacy free-text alias


def test_assumption_provenance_defaults_new_fields_when_absent():
    """An OLD-shaped provenance entry (no source_periods/derivation/
    source_evidence_ids at all) must still validate cleanly -- the new
    fields default to empty, never a KeyError or a validation failure."""
    provenance = {
        "wacc": {"value": WACC, "source_type": AssumptionSourceType.CONFIGURED_DEFAULT,
                 "source_period": None, "reason": "Configured default WACC.",
                 "approval_status": "proposed", "units": "ratio"},
    }
    scenario_raw = base_scenario(assumption_provenance=provenance)
    out = value_scenario(base_inputs(), build_scenario(scenario_raw, HORIZON))

    got = out["assumptions"]["assumption_provenance"]["wacc"]
    assert got["source_periods"] == []
    assert got["source_evidence_ids"] == []
    assert got["derivation"] is None
    assert got["reason"] == "Configured default WACC."


# ---- the LLM may propose, never alter, the authoritative arithmetic ----

def test_provenance_metadata_cannot_influence_the_arithmetic():
    """Two scenarios with IDENTICAL numeric assumptions but DIFFERENT
    (even self-contradictory) provenance metadata must value identically —
    the engine never reads assumption_provenance for arithmetic, only echoes it."""
    plain = value_scenario(base_inputs(), build_scenario(base_scenario(), HORIZON))
    with_bogus_provenance = value_scenario(
        base_inputs(),
        build_scenario(base_scenario(assumption_provenance={
            "wacc": {"value": 0.99, "source_type": AssumptionSourceType.LLM_PROPOSED,
                     "reason": "this text does not change WACC"},
        }), HORIZON))

    assert plain["value_per_share"] == with_bogus_provenance["value_per_share"]
    assert plain["enterprise_value"] == with_bogus_provenance["enterprise_value"]
    # The scenario's actual WACC (used in arithmetic) is still the real one.
    assert with_bogus_provenance["assumptions"]["wacc"] == pytest.approx(WACC)


def test_mutating_a_returned_result_never_touches_the_engines_internals():
    """The LLM (or any caller) may hold and even mutate its own copy of a
    result dict — that must never retroactively change a later call."""
    first = run_dcf(base_inputs(), [base_scenario()])
    first["scenarios"][0]["value_per_share"] = -99999.0
    first["scenarios"][0]["net_debt"] = 0.0

    second = run_dcf(base_inputs(), [base_scenario()])
    assert second["scenarios"][0]["value_per_share"] != -99999.0
    assert second["scenarios"][0]["value_per_share"] == pytest.approx(
        EXPECTED["value_per_share"], abs=1e-4)


# ===========================================================================
# TSLA DCF validation patch
# ===========================================================================

# ---- FCFF sign convention (section 2) ----

def test_capex_pct_revenue_is_subtracted_exactly_once(result):
    """The documented formula: FCFF = NOPAT + D&A - capex - delta(NWC).
    capex_pct_revenue is a POSITIVE magnitude by contract; confirms it is
    SUBTRACTED, never added (the double-negative bug the patch guards
    against would instead increase FCFF when capex increases)."""
    higher_capex = value_scenario(
        base_inputs(), build_scenario(base_scenario(capex_pct_revenue=CAPEX_PCT * 2), HORIZON))
    assert higher_capex["forecast"][0]["fcff"] < result["forecast"][0]["fcff"]


def test_negative_capex_pct_revenue_is_rejected_not_silently_flipped():
    """A still-negative-signed CapEx (e.g. an un-normalized provider figure,
    or an LLM-proposed assumption that forgot to take the magnitude) must be
    REJECTED at input validation, never silently subtracted-a-negative
    (which would ADD capex to FCFF instead of subtracting it)."""
    with pytest.raises(ToolFailure) as excinfo:
        build_scenario(base_scenario(capex_pct_revenue=-0.07), HORIZON)
    assert excinfo.value.code == DCF_ASSUMPTION_INVALID


def test_negative_depreciation_pct_revenue_is_rejected():
    with pytest.raises(ToolFailure) as excinfo:
        build_scenario(base_scenario(depreciation_pct_revenue=-0.01), HORIZON)
    assert excinfo.value.code == DCF_ASSUMPTION_INVALID


def test_negative_working_capital_pct_revenue_is_allowed_and_releases_cash():
    """UNLIKE CapEx/D&A, working_capital_pct_revenue may legitimately be
    negative (a real, negative operating-working-capital business) and must
    NOT be rejected -- a growing (more negative) working-capital balance
    RELEASES cash into FCFF as revenue grows, never rejected as invalid."""
    out = value_scenario(
        base_inputs(base_working_capital=BASE_REVENUE * -0.05),
        build_scenario(base_scenario(working_capital_pct_revenue=-0.05), HORIZON))
    assert out["forecast"][0]["change_in_net_working_capital"] < 0
    assert out["forecast"][0]["fcff"] > (out["forecast"][0]["nopat"]
                                         + out["forecast"][0]["depreciation_amortization"]
                                         - out["forecast"][0]["capital_expenditure"])


def test_d_and_a_and_capex_are_independent_never_cross_derived(result):
    """Increasing D&A alone must not change CapEx, and vice versa -- the two
    inputs are read independently, never inferred from one another or from
    an FCF-margin difference (the ORIGINAL CapEx bug this project already
    fixed once -- see docs/PHASE_H1_STOCK_ANALYSIS.md's Phase H.3 corrective
    patch)."""
    higher_da = value_scenario(
        base_inputs(), build_scenario(base_scenario(depreciation_pct_revenue=DA_PCT * 2), HORIZON))
    assert higher_da["forecast"][0]["capital_expenditure"] == pytest.approx(
        result["forecast"][0]["capital_expenditure"])
    assert higher_da["forecast"][0]["depreciation_amortization"] != pytest.approx(
        result["forecast"][0]["depreciation_amortization"])


# ---- working-capital policy: LEVEL vs CHANGE (section 3) ----

def test_fcff_subtracts_the_change_in_nwc_not_the_whole_nwc_balance(result):
    """FCFF must subtract Change_NWC_t = NWC_t - NWC_(t-1), never the entire
    NWC_t balance every year -- the level and the change are different
    quantities with very different magnitudes."""
    row = result["forecast"][0]
    assert row["change_in_net_working_capital"] != pytest.approx(row["net_working_capital"])
    assert abs(row["change_in_net_working_capital"]) < abs(row["net_working_capital"])


def test_nwc_level_is_computed_fresh_each_year_from_that_years_revenue():
    out = value_scenario(base_inputs(), build_scenario(base_scenario(), HORIZON))
    for row in out["forecast"]:
        assert row["net_working_capital"] == pytest.approx(row["revenue"] * WC_PCT, abs=1e-4)


# ---- terminal-value validation (section 5) ----

def test_negative_terminal_fcff_marks_the_scenario_invalid():
    """A scenario whose terminal-year FCFF is negative must be classified
    'invalid' (ScenarioResultStatus), not silently perpetuity-valued as if
    it were a normal going concern."""
    # margin far below capex -- guarantees negative FCFF every year.
    starved = build_scenario(
        base_scenario(operating_margin=0.02, capex_pct_revenue=0.30), HORIZON)
    out = value_scenario(base_inputs(), starved)
    assert out["terminal_year_fcff"] < 0
    assert out["status"] == ScenarioResultStatus.INVALID
    assert any("negative" in w.lower() for w in out["warnings"])


def test_negative_terminal_fcff_propagates_to_overall_validation_status():
    starved = base_scenario("base", operating_margin=0.02, capex_pct_revenue=0.30)
    out = run_dcf(base_inputs(), [starved])
    assert out["validation_status"] == DcfValidationStatus.NEGATIVE_TERMINAL_FCFF
    assert any("DCF_NEGATIVE_TERMINAL_FCFF" in r for r in out["validation_reasons"])


def test_allow_negative_terminal_fcff_config_flag(monkeypatch):
    """Fail-closed by default; a reviewed policy decision can opt back in."""
    starved = base_scenario("base", operating_margin=0.02, capex_pct_revenue=0.30)
    monkeypatch.setenv("DCF_ALLOW_NEGATIVE_TERMINAL_FCFF", "true")
    out = run_dcf(base_inputs(), [starved])
    assert out["scenarios"][0]["status"] != ScenarioResultStatus.INVALID
    assert out["validation_status"] != DcfValidationStatus.NEGATIVE_TERMINAL_FCFF


def test_wacc_less_than_or_equal_to_terminal_growth_still_rejected():
    """Pre-existing hard guard, re-asserted here alongside the new
    validation layer so the full set of terminal-value protections is
    documented in one place."""
    with pytest.raises(ToolFailure) as excinfo:
        build_scenario(base_scenario(terminal_growth=WACC), HORIZON)
    assert excinfo.value.code == DCF_TERMINAL_GROWTH_TOO_HIGH


def test_nan_and_inf_are_rejected_in_terminal_calculation_inputs():
    with pytest.raises(ToolFailure):
        build_scenario(base_scenario(wacc=float("nan")), HORIZON)
    with pytest.raises(ToolFailure):
        build_scenario(base_scenario(wacc=float("inf")), HORIZON)


# ---- scenario monotonicity (section 6) ----

def test_monotonicity_check_passes_on_the_normal_fixture():
    out = run_dcf(base_inputs(), [
        base_scenario("base"),
        base_scenario("bull", revenue_growth=0.15, operating_margin=0.24,
                      wacc=WACC - 0.01, terminal_growth=TERMINAL_GROWTH + 0.005),
        base_scenario("bear", revenue_growth=0.04, operating_margin=0.16,
                      wacc=WACC + 0.01, terminal_growth=TERMINAL_GROWTH - 0.005),
    ])
    assert out["scenario_monotonicity"]["checked"] is True
    assert out["scenario_monotonicity"]["passed"] is True
    assert out["validation_status"] == DcfValidationStatus.VALID


def test_monotonicity_check_detects_an_inverted_ordering_directly():
    """Unit-tests the detector in isolation with CONTRIVED result values
    (never relying on finding a real economic fixture that happens to
    invert) -- assumptions ARE properly ordered (bull uniformly more
    favorable than base, bear uniformly less), but the given calculated
    values are deliberately inverted, exactly the TSLA symptom (bull MORE
    negative than base, bear LESS negative than both)."""
    base = build_scenario(base_scenario("base"), HORIZON)
    bull = build_scenario(base_scenario("bull", revenue_growth=0.20, operating_margin=0.25,
                                        wacc=0.08, terminal_growth=0.03), HORIZON)
    bear = build_scenario(base_scenario("bear", revenue_growth=0.05, operating_margin=0.15,
                                        wacc=0.12, terminal_growth=0.01), HORIZON)
    results_by_name = {
        "base": {"value_per_share": -10.0},
        "bull": {"value_per_share": -20.0},   # should be >= base, is NOT
        "bear": {"value_per_share": -5.0},    # should be <= base, is NOT
    }
    outcome = _scenario_monotonicity_check([base, bull, bear], results_by_name)
    assert outcome["checked"] is True
    assert outcome["passed"] is False
    assert len(outcome["violated_relations"]) == 2
    assert outcome["values"] == {"bull": -20.0, "base": -10.0, "bear": -5.0}
    assert outcome["assumptions"]["bull"]["revenue_growth"] == pytest.approx(0.20)


def test_monotonicity_violation_never_reorders_or_clamps_the_values():
    """The detector must be pure DETECTION -- the actual scenario results in
    run_dcf()'s output are never reordered, swapped, or clamped to force
    the expected ordering, even when a violation is found."""
    base = build_scenario(base_scenario("base"), HORIZON)
    bull = build_scenario(base_scenario("bull", revenue_growth=0.20, operating_margin=0.25,
                                        wacc=0.08, terminal_growth=0.03), HORIZON)
    bear = build_scenario(base_scenario("bear", revenue_growth=0.05, operating_margin=0.15,
                                        wacc=0.12, terminal_growth=0.01), HORIZON)
    results_by_name = {"base": {"value_per_share": -10.0}, "bull": {"value_per_share": -20.0},
                       "bear": {"value_per_share": -5.0}}
    _scenario_monotonicity_check([base, bull, bear], results_by_name)
    # The dicts passed in are untouched -- the checker never mutates them.
    assert results_by_name["bull"]["value_per_share"] == -20.0
    assert results_by_name["bear"]["value_per_share"] == -5.0


def test_monotonicity_check_does_not_apply_when_scenario_names_are_not_the_standard_three():
    out = run_dcf(base_inputs(), [base_scenario("base"), base_scenario("custom_scenario",
                                                                       revenue_growth=0.5)])
    assert out["scenario_monotonicity"] is None


def test_monotonicity_check_does_not_apply_when_assumption_construction_does_not_imply_an_ordering():
    """A caller-supplied 'bull' whose assumptions are NOT uniformly more
    favorable than base's does not imply an ordering at all -- nothing to
    enforce, and this must NOT be reported as a violation."""
    out = run_dcf(base_inputs(), [
        base_scenario("base"),
        base_scenario("bull", operating_margin=0.05),  # WORSE margin than base -- not a "bull"
        base_scenario("bear", operating_margin=0.16),
    ])
    assert out["scenario_monotonicity"] is None
    assert out["validation_status"] in (DcfValidationStatus.VALID, DcfValidationStatus.VALID_WITH_WARNINGS)


def test_monotonicity_check_disabled_via_config(monkeypatch):
    monkeypatch.setenv("DCF_SCENARIO_MONOTONICITY_CHECK_ENABLED", "false")
    out = run_dcf(base_inputs(), [
        base_scenario("base"),
        base_scenario("bull", revenue_growth=0.15, operating_margin=0.24),
        base_scenario("bear", revenue_growth=0.04, operating_margin=0.16),
    ])
    assert out["scenario_monotonicity"] is None


# ---- negative equity value classification (section 7) ----

def test_negative_equity_value_is_preserved_never_clamped_to_zero():
    inputs = base_inputs(total_debt=1_000_000.0, cash_and_cash_equivalents=0.0)
    out = value_scenario(inputs, build_scenario(base_scenario(), HORIZON))
    assert out["equity_value"] < 0
    assert out["value_per_share"] < 0
    assert out["status"] == ScenarioResultStatus.NEGATIVE_EQUITY_VALUE


def test_negative_equity_value_produces_valid_with_warnings_not_invalid():
    """A structurally sound negative equity value is mathematically possible
    (section 7) and must NOT be treated as a hard failure -- only warned."""
    inputs = base_inputs(total_debt=1_000_000.0, cash_and_cash_equivalents=0.0)
    out = run_dcf(inputs, [base_scenario()])
    assert out["validation_status"] == DcfValidationStatus.VALID_WITH_WARNINGS
    assert out["scenarios"][0]["status"] == ScenarioResultStatus.NEGATIVE_EQUITY_VALUE


def test_all_negative_scenarios_still_yield_a_full_auditable_result():
    inputs = base_inputs(total_debt=1_000_000.0, cash_and_cash_equivalents=0.0)
    out = run_dcf(inputs, [
        base_scenario("base"),
        base_scenario("bull", revenue_growth=0.15, operating_margin=0.24,
                      wacc=WACC - 0.01, terminal_growth=TERMINAL_GROWTH + 0.005),
        base_scenario("bear", revenue_growth=0.04, operating_margin=0.16,
                      wacc=WACC + 0.01, terminal_growth=TERMINAL_GROWTH - 0.005),
    ])
    for s in out["scenarios"]:
        assert s["status"] in ScenarioResultStatus.ALL
        assert s["equity_value"] is not None and s["value_per_share"] is not None


def test_scenario_status_valid_on_the_ordinary_fixture(result):
    assert result["status"] == ScenarioResultStatus.VALID
    assert result["warnings"] == []


# ---- overall DcfValidationStatus (section 9) ----

def test_dcf_validation_status_valid_on_the_ordinary_fixture():
    out = run_dcf(base_inputs(), [base_scenario()])
    assert out["validation_status"] == DcfValidationStatus.VALID
    assert out["validation_reasons"] == []


def test_dcf_validation_status_is_always_present_and_recognized():
    out = run_dcf(base_inputs(), [base_scenario()])
    assert out["validation_status"] in DcfValidationStatus.ALL


def test_dcf_validation_status_survives_repeated_runs_identically():
    first = run_dcf(base_inputs(), [base_scenario()])
    second = run_dcf(base_inputs(), [base_scenario()])
    assert first["validation_status"] == second["validation_status"]
    assert first["validation_reasons"] == second["validation_reasons"]
