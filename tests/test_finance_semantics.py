"""Phase H.10 — the semantic compatibility contract.

These are not tests that a particular company produces a particular number.
They are tests that certain COMBINATIONS OF FACTS are impossible, which is
the difference this phase is trying to make: every previous phase asserted
outputs, so each new stock found a new input that reached the same broken
arithmetic by a different route.

Three kinds of test here, and the second and third are the point:

  * matrix tests, walking the period/share/debt/metric combinations and
    asserting which operations each one permits (sections 39);
  * metamorphic tests, where the NUMBERS are identical and only the semantic
    metadata differs -- if the system treats those two cases the same, it is
    reasoning about floats and not about finance (section 40);
  * negative tests, asserting that a specific bad calculation cannot be
    performed at all (section 41).

No ticker appears in any production path exercised here.
"""

import pytest

from finance import semantics as S


F = S.PeriodFrequency
M = S.MetricIdentity
O = S.Operation


def _rev(frequency, **kw):
    kw.setdefault("flow_or_instant", S.FlowOrInstant.FLOW)
    return S.SemanticFact(metric_id=M.REVENUE, value=100.0,
                          period_frequency=frequency, **kw)


# ---------------------------------------------------------------------------
# Section 39 — the period compatibility matrix for GROWTH
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("left, right, allowed", [
    (F.QUARTER, F.QUARTER, True),
    (F.ANNUAL, F.ANNUAL, True),
    (F.TTM, F.TTM, True),
    (F.ANNUAL, F.TTM, True),        # both span twelve months
    (F.TTM, F.ANNUAL, True),
    (F.YTD_6M, F.YTD_6M, True),
    (F.HALF_YEAR, F.YTD_6M, True),  # same six months, different name
    (F.QUARTER, F.ANNUAL, False),
    (F.QUARTER, F.TTM, False),
    (F.ANNUAL, F.QUARTER, False),
    (F.TTM, F.QUARTER, False),
    (F.YTD_6M, F.ANNUAL, False),
    (F.YTD_9M, F.TTM, False),
    (F.INSTANT, F.ANNUAL, False),
    (F.UNKNOWN, F.ANNUAL, False),
    (F.ANNUAL, F.UNKNOWN, False),
    (F.MULTI_YEAR, F.ANNUAL, False),
])
def test_growth_period_matrix(left, right, allowed):
    verdict = S.compatible_for(
        O.GROWTH,
        _rev(left, fiscal_year=2026, fiscal_quarter=3 if left == F.QUARTER else None),
        _rev(right, fiscal_year=2025, fiscal_quarter=3 if right == F.QUARTER else None))
    assert bool(verdict) is allowed, verdict.reason


def test_a_rejected_growth_names_the_frequency_problem():
    verdict = S.compatible_for(O.GROWTH, _rev(F.QUARTER, fiscal_year=2026, fiscal_quarter=3),
                               _rev(F.TTM, fiscal_year=2026))
    assert verdict.code == S.PERIOD_FREQUENCY_MISMATCH
    assert "3 months" in verdict.reason and "12" in verdict.reason


# ---------------------------------------------------------------------------
# Section 41 — the forbidden calculation, stated as such
# ---------------------------------------------------------------------------

def test_quarterly_guidance_over_ttm_revenue_is_not_a_growth_rate():
    """The ADBE failure, reduced to its semantics.

    A guided quarterly revenue level over a trailing-twelve-month actual
    produced -73%, which was then clamped to the model's -20% bound and
    reported as a growth assumption. Nothing about the arithmetic was
    wrong; the operation was.
    """
    guided = S.SemanticFact(
        metric_id=M.REVENUE, value=6.695e9, period_frequency=F.QUARTER,
        fiscal_year=2026, fiscal_quarter=3, flow_or_instant=S.FlowOrInstant.FLOW,
        current_or_historical=S.CurrentOrHistorical.FORWARD)
    actual = S.SemanticFact(
        metric_id=M.REVENUE, value=25.2e9, period_frequency=F.TTM,
        flow_or_instant=S.FlowOrInstant.FLOW,
        current_or_historical=S.CurrentOrHistorical.CURRENT)
    verdict = S.compatible_for(O.GROWTH, guided, actual)
    assert not verdict
    assert verdict.code == S.PERIOD_FREQUENCY_MISMATCH


def test_full_year_guidance_over_ttm_revenue_is_a_growth_rate():
    """The companion case, which must stay ALLOWED.

    A contract that rejects everything is not a contract, it is an outage.
    """
    guided = S.SemanticFact(
        metric_id=M.REVENUE, value=101.1e9, period_frequency=F.ANNUAL,
        fiscal_year=2026, flow_or_instant=S.FlowOrInstant.FLOW,
        current_or_historical=S.CurrentOrHistorical.FORWARD)
    actual = S.SemanticFact(
        metric_id=M.REVENUE, value=97.9e9, period_frequency=F.TTM,
        fiscal_year=2025, flow_or_instant=S.FlowOrInstant.FLOW,
        current_or_historical=S.CurrentOrHistorical.CURRENT)
    assert S.compatible_for(O.GROWTH, guided, actual)


def test_a_quarter_against_a_different_quarter_is_not_year_over_year():
    q3 = _rev(F.QUARTER, fiscal_year=2026, fiscal_quarter=3)
    q2 = _rev(F.QUARTER, fiscal_year=2026, fiscal_quarter=2)
    verdict = S.compatible_for(O.GROWTH, q3, q2)
    assert not verdict
    assert verdict.code == S.INCOMPATIBLE_PERIODS


def test_the_same_period_against_itself_is_not_growth():
    verdict = S.compatible_for(O.GROWTH, _rev(F.ANNUAL, fiscal_year=2026),
                               _rev(F.ANNUAL, fiscal_year=2026))
    assert not verdict
    assert verdict.code == S.INCOMPATIBLE_PERIODS


def test_growth_needs_the_same_metric_on_both_sides():
    revenue = _rev(F.ANNUAL, fiscal_year=2026)
    ebitda = S.SemanticFact(metric_id=M.EBITDA, value=20.0, period_frequency=F.ANNUAL,
                            fiscal_year=2025, flow_or_instant=S.FlowOrInstant.FLOW)
    verdict = S.compatible_for(O.GROWTH, revenue, ebitda)
    assert not verdict
    assert verdict.code == S.INCOMPATIBLE_METRICS


# ---------------------------------------------------------------------------
# Section 40 — metamorphic: same numbers, different meaning
# ---------------------------------------------------------------------------

def test_identical_values_with_different_periods_behave_differently():
    """The clearest statement of what this phase changed.

    Both facts hold the value 100. Nothing about the arithmetic can
    distinguish them. Only the metadata can, and the system must.
    """
    annual = _rev(F.ANNUAL, fiscal_year=2026)
    quarter = _rev(F.QUARTER, fiscal_year=2026, fiscal_quarter=1)
    base = _rev(F.ANNUAL, fiscal_year=2025)

    assert annual.value == quarter.value == 100.0
    assert S.compatible_for(O.GROWTH, annual, base)
    assert not S.compatible_for(O.GROWTH, quarter, base)


def test_identical_share_values_with_different_bases_behave_differently():
    current = S.SemanticFact(metric_id=M.SHARES_CURRENT_OUTSTANDING, value=1_000.0,
                            period_frequency=F.INSTANT,
                            flow_or_instant=S.FlowOrInstant.INSTANT)
    diluted = S.SemanticFact(metric_id=M.SHARES_WEIGHTED_AVERAGE_DILUTED, value=1_000.0,
                             period_frequency=F.ANNUAL,
                             flow_or_instant=S.FlowOrInstant.FLOW)
    other_current = S.SemanticFact(metric_id=M.SHARES_CURRENT_OUTSTANDING, value=1_000.0,
                                   period_frequency=F.INSTANT,
                                   flow_or_instant=S.FlowOrInstant.INSTANT)
    assert current.value == diluted.value
    assert S.compatible_for(O.RECONCILE, current, other_current)
    assert not S.compatible_for(O.RECONCILE, current, diluted)


def test_the_same_pair_is_allowed_for_one_operation_and_refused_for_another():
    """Section 5's whole argument in one assertion.

    Quarterly guidance and a trailing-twelve-month actual belong side by
    side in a report; dividing one by the other does not mean anything.
    Compatibility is a property of the pair AND the operation.
    """
    quarter = _rev(F.QUARTER, fiscal_year=2026, fiscal_quarter=3)
    ttm = _rev(F.TTM, fiscal_year=2026)
    assert S.compatible_for(O.COMPARE, quarter, ttm)
    assert not S.compatible_for(O.GROWTH, quarter, ttm)


# ---------------------------------------------------------------------------
# Sections 12-13 — the share basis matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("left, right, allowed, informational", [
    (M.SHARES_CURRENT_OUTSTANDING, M.SHARES_CURRENT_OUTSTANDING, True, False),
    (M.SHARES_ECONOMIC_CURRENT, M.SHARES_CURRENT_OUTSTANDING, True, False),
    (M.SHARES_WEIGHTED_AVERAGE_DILUTED, M.SHARES_WEIGHTED_AVERAGE_DILUTED, True, False),
    (M.SHARES_CURRENT_OUTSTANDING, M.SHARES_WEIGHTED_AVERAGE_DILUTED, False, True),
    (M.SHARES_CURRENT_OUTSTANDING, M.SHARES_WEIGHTED_AVERAGE_BASIC, False, True),
    (M.SHARES_WEIGHTED_AVERAGE_BASIC, M.SHARES_WEIGHTED_AVERAGE_DILUTED, False, True),
    (M.SHARES_ADR_EQUIVALENT, M.SHARES_WEIGHTED_AVERAGE_DILUTED, False, True),
])
def test_share_basis_reconciliation_matrix(left, right, allowed, informational):
    def fact(identity):
        weighted = identity in S.WEIGHTED_AVERAGE_SHARE_IDENTITIES
        return S.SemanticFact(
            metric_id=identity, value=1_000.0,
            period_frequency=F.ANNUAL if weighted else F.INSTANT,
            flow_or_instant=S.FlowOrInstant.FLOW if weighted else S.FlowOrInstant.INSTANT)

    verdict = S.compatible_for(O.RECONCILE, fact(left), fact(right))
    assert bool(verdict) is allowed, verdict.reason
    assert verdict.informational is informational


def test_an_incompatible_share_basis_is_informational_not_a_fault():
    """Section 13: a difference by construction is not a data-integrity failure.

    This is the assertion that keeps the fix honest -- the point was never
    to suppress the comparison, it was to stop calling an expected
    difference a conflict.
    """
    current = S.SemanticFact(metric_id=M.SHARES_CURRENT_OUTSTANDING, value=1_000.0,
                             period_frequency=F.INSTANT,
                             flow_or_instant=S.FlowOrInstant.INSTANT)
    diluted = S.SemanticFact(metric_id=M.SHARES_WEIGHTED_AVERAGE_DILUTED, value=1_400.0,
                             period_frequency=F.ANNUAL,
                             flow_or_instant=S.FlowOrInstant.FLOW)
    verdict = S.compatible_for(O.RECONCILE, current, diluted)
    assert verdict.code == S.SHARE_BASIS_NOT_COMPARABLE
    assert verdict.informational is True

    audit = S.SemanticAudit()
    audit.record(verdict, current, diluted)
    assert audit.rejections and audit.blocking() == []


# ---------------------------------------------------------------------------
# Section 14 — the debt basis matrix
# ---------------------------------------------------------------------------

def _debt(identity, value=100.0):
    return S.SemanticFact(metric_id=identity, value=value, period_frequency=F.INSTANT,
                          flow_or_instant=S.FlowOrInstant.INSTANT,
                          instant_date="2026-06-30")


# A component SUM carries the identity TOTAL_DEBT -- it is total debt, derived
# a second way -- so the valid reconciliation is total against total. A single
# component against the total is not a reconciliation at all: the gap between
# them is simply the rest of the debt.
@pytest.mark.parametrize("left, right, allowed", [
    (M.TOTAL_DEBT, M.TOTAL_DEBT, True),            # component sum vs reported total
    (M.LONG_TERM_DEBT, M.LONG_TERM_DEBT, True),    # one component, two sources
    (M.SHORT_TERM_DEBT, M.SHORT_TERM_DEBT, True),
    (M.TOTAL_DEBT, M.LONG_TERM_DEBT, False),       # the live failure
    (M.LONG_TERM_DEBT, M.TOTAL_DEBT, False),
    (M.SHORT_TERM_DEBT, M.TOTAL_DEBT, False),
    (M.LONG_TERM_DEBT, M.SHORT_TERM_DEBT, False),  # part against a different part
    (M.SHORT_TERM_DEBT, M.LONG_TERM_DEBT, False),
])
def test_debt_reconciliation_matrix(left, right, allowed):
    verdict = S.compatible_for(O.RECONCILE, _debt(left), _debt(right))
    assert bool(verdict) is allowed, verdict.reason


def test_total_debt_cannot_be_added_to_its_own_component():
    verdict = S.compatible_for(O.SUM, _debt(M.TOTAL_DEBT), _debt(M.LONG_TERM_DEBT))
    assert not verdict
    assert verdict.code == S.INCOMPATIBLE_DEBT_BASIS


def test_debt_components_at_different_dates_do_not_sum():
    left = _debt(M.SHORT_TERM_DEBT)
    right = S.SemanticFact(metric_id=M.LONG_TERM_DEBT, value=50.0,
                           period_frequency=F.INSTANT,
                           flow_or_instant=S.FlowOrInstant.INSTANT,
                           instant_date="2025-12-31")
    verdict = S.compatible_for(O.SUM, left, right)
    assert not verdict
    assert verdict.code == S.INCOMPATIBLE_PERIODS


# ---------------------------------------------------------------------------
# Free cash flow definitions
# ---------------------------------------------------------------------------

def test_three_things_called_free_cash_flow_do_not_reconcile_with_each_other():
    def fcf(identity):
        return S.SemanticFact(metric_id=identity, value=8_403.0, period_frequency=F.TTM,
                              flow_or_instant=S.FlowOrInstant.FLOW)

    assert S.compatible_for(O.RECONCILE, fcf(M.SIMPLE_FCF), fcf(M.SIMPLE_FCF))
    verdict = S.compatible_for(O.RECONCILE, fcf(M.SIMPLE_FCF), fcf(M.COMPANY_DEFINED_FCF))
    assert not verdict
    assert verdict.code == S.INCOMPATIBLE_FCF_DEFINITION


def test_one_name_two_definitions_is_still_a_mismatch():
    left = S.SemanticFact(metric_id=M.SIMPLE_FCF, value=1.0, period_frequency=F.TTM,
                          flow_or_instant=S.FlowOrInstant.FLOW,
                          definition_id="ocf_less_capex")
    right = S.SemanticFact(metric_id=M.SIMPLE_FCF, value=1.0, period_frequency=F.TTM,
                           flow_or_instant=S.FlowOrInstant.FLOW,
                           definition_id="ocf_less_capex_less_leases")
    assert not S.compatible_for(O.RECONCILE, left, right)


# ---------------------------------------------------------------------------
# Ratios, per-share conversion, entity scope
# ---------------------------------------------------------------------------

def test_a_margin_across_two_periods_is_refused():
    income = S.SemanticFact(metric_id=M.OPERATING_INCOME, value=6_488.0,
                            period_frequency=F.ANNUAL, end_date="2025-12-27",
                            flow_or_instant=S.FlowOrInstant.FLOW)
    revenue = S.SemanticFact(metric_id=M.REVENUE, value=41_305.0,
                             period_frequency=F.TTM, end_date="2026-06-27",
                             flow_or_instant=S.FlowOrInstant.FLOW)
    verdict = S.compatible_for(O.RATIO, income, revenue)
    assert not verdict
    assert verdict.code == S.INCOMPATIBLE_PERIODS


def test_a_margin_within_one_period_is_allowed():
    income = S.SemanticFact(metric_id=M.OPERATING_INCOME, value=6_488.0,
                            period_frequency=F.TTM, end_date="2026-06-27",
                            flow_or_instant=S.FlowOrInstant.FLOW)
    revenue = S.SemanticFact(metric_id=M.REVENUE, value=41_305.0,
                             period_frequency=F.TTM, end_date="2026-06-27",
                             flow_or_instant=S.FlowOrInstant.FLOW)
    assert S.compatible_for(O.RATIO, income, revenue)


def test_a_current_valuation_may_not_be_divided_by_a_weighted_average_count():
    equity = S.SemanticFact(metric_id=M.STOCKHOLDERS_EQUITY, value=1e9,
                            period_frequency=F.INSTANT,
                            flow_or_instant=S.FlowOrInstant.INSTANT,
                            current_or_historical=S.CurrentOrHistorical.CURRENT)
    diluted = S.SemanticFact(metric_id=M.SHARES_WEIGHTED_AVERAGE_DILUTED, value=1e6,
                             period_frequency=F.ANNUAL,
                             flow_or_instant=S.FlowOrInstant.FLOW)
    verdict = S.compatible_for(O.PER_SHARE_CONVERSION, equity, diluted)
    assert not verdict
    assert verdict.code == S.INCOMPATIBLE_SHARE_BASIS


def test_a_segment_figure_is_not_the_consolidated_one():
    consolidated = _rev(F.ANNUAL, fiscal_year=2026)
    segment = S.SemanticFact(metric_id=M.REVENUE, value=40.0, period_frequency=F.ANNUAL,
                             fiscal_year=2025, flow_or_instant=S.FlowOrInstant.FLOW,
                             consolidation_scope=S.ConsolidationScope.SEGMENT)
    verdict = S.compatible_for(O.GROWTH, consolidated, segment)
    assert not verdict
    assert verdict.code == S.INCOMPATIBLE_ENTITY_SCOPE


def test_two_currencies_never_combine():
    left = _rev(F.ANNUAL, fiscal_year=2026, currency="USD")
    right = _rev(F.ANNUAL, fiscal_year=2025, currency="BRL")
    verdict = S.compatible_for(O.GROWTH, left, right)
    assert not verdict
    assert verdict.code == S.INCOMPATIBLE_CURRENCY


def test_gaap_and_adjusted_are_not_mixed_into_one_growth_rate():
    left = _rev(F.ANNUAL, fiscal_year=2026, accounting_basis=S.AccountingBasis.ADJUSTED)
    right = _rev(F.ANNUAL, fiscal_year=2025, accounting_basis=S.AccountingBasis.GAAP)
    verdict = S.compatible_for(O.GROWTH, left, right)
    assert not verdict
    assert verdict.code == S.INCOMPATIBLE_ACCOUNTING_BASIS


# ---------------------------------------------------------------------------
# The validator's own defaults
# ---------------------------------------------------------------------------

def test_an_unrecognised_operation_is_refused_rather_than_permitted():
    """A validator that allows what it does not understand guarantees nothing."""
    verdict = S.compatible_for("MULTIPLY_SOMEHOW", _rev(F.ANNUAL), _rev(F.ANNUAL))
    assert not verdict


def test_a_dcf_input_may_not_be_historical():
    current = S.SemanticFact(metric_id=M.REVENUE, value=1.0, period_frequency=F.TTM,
                             flow_or_instant=S.FlowOrInstant.FLOW,
                             current_or_historical=S.CurrentOrHistorical.CURRENT)
    historical = S.SemanticFact(metric_id=M.REVENUE, value=1.0, period_frequency=F.TTM,
                                flow_or_instant=S.FlowOrInstant.FLOW,
                                current_or_historical=S.CurrentOrHistorical.HISTORICAL)
    assert S.compatible_for(O.DCF_INPUT, current, current)
    assert not S.compatible_for(O.DCF_INPUT, current, historical)


def test_the_audit_records_identities_and_never_values():
    left = _rev(F.QUARTER, fiscal_year=2026, fiscal_quarter=3)
    right = _rev(F.TTM, fiscal_year=2026)
    audit = S.SemanticAudit()
    audit.record(S.compatible_for(O.GROWTH, left, right), left, right, context="unit test")
    record = audit.rejections[0]
    assert record["code"] == S.PERIOD_FREQUENCY_MISMATCH
    assert "100" not in record["left"] and "100" not in record["right"]
    assert audit.blocking()
