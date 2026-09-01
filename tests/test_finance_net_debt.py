"""Phase H.6 — net-debt components, policy and reconciliation (sections 14-16).

The live case is tests/test_finance_nvda_regression.py. This file isolates
the rules: which components a policy consumes, when two concepts overlap, and
what happens when the figure does not reconcile against its own inputs.

Every test here is about a number that would otherwise be WRONG BY A REAL
AMOUNT while looking completely ordinary — a $1.0B double-count, a $37.1B
omission, a policy that silently changed between two runs of the same
company.
"""

import pytest

from finance import net_debt as ND
from finance.freshness import FreshnessStatus, SelectedValue


def _sel(field, value, concept, as_of="2026-04-26"):
    return SelectedValue(
        field=field, value=value, unit="USD", source="quarterly_sec_filing",
        provider="sec", accession="0000-26-1", form="10-Q", fiscal_period="Q1",
        as_of_date=as_of, retrieval_timestamp="2026-08-17T00:00:00Z",
        evidence_id=f"dcf.input.{field}.latest",
        freshness_status=FreshnessStatus.CURRENT_QUARTER, concept=concept)


def _components(short_term_debt_concept="ShortTermBorrowings",
                current_portion=1_000.0, short_term_investments=37_098.0,
                reported_total=None, reported_total_concept=None):
    selections = {
        "cash_and_cash_equivalents": _sel("cash_and_cash_equivalents", 13_237.0,
                                          "CashAndCashEquivalentsAtCarryingValue"),
        "short_term_investments": _sel("short_term_investments", short_term_investments,
                                       "DebtSecuritiesCurrent"),
        "equity_securities_at_fair_value": _sel("equity_securities_at_fair_value",
                                                30_237.0, "EquitySecuritiesFvNi"),
        "short_term_debt": _sel("short_term_debt", 1_000.0, short_term_debt_concept),
        "long_term_debt": _sel("long_term_debt", 7_470.0, "LongTermDebtNoncurrent"),
    }
    if current_portion is not None:
        selections["current_portion_of_long_term_debt"] = _sel(
            "current_portion_of_long_term_debt", current_portion, "LongTermDebtCurrent")
    return ND.collect_components(
        selections, reported_total_debt=reported_total,
        reported_total_debt_concept=reported_total_concept, as_of_date="2026-04-26")


# ---------------------------------------------------------------------------
# Section 14 — the components stay separate and individually citable
# ---------------------------------------------------------------------------

def test_every_component_keeps_its_own_field_and_evidence_id():
    components = _components()
    for name in ("cash_and_cash_equivalents", "short_term_investments",
                 "equity_securities_at_fair_value", "short_term_debt",
                 "long_term_debt"):
        assert getattr(components, name) is not None, name
        assert components.evidence_ids[name].startswith("dcf.input."), name
        assert components.concepts[name], name


def test_marketable_securities_are_never_folded_into_cash():
    components = _components()
    assert components.cash_and_cash_equivalents == pytest.approx(13_237.0)
    assert components.short_term_investments == pytest.approx(37_098.0)
    assert components.cash_and_cash_equivalents != components.short_term_investments


# ---------------------------------------------------------------------------
# Section 15 — overlapping concepts are detected, never summed
# ---------------------------------------------------------------------------

def test_debt_current_already_contains_the_current_portion_of_ltd():
    """`DebtCurrent` is defined in us-gaap as short-term AND current-portion
    debt. NVDA reports it at exactly the same $1.0B as `LongTermDebtCurrent`,
    and summing both overstated total debt by $1.0B."""
    components = _components(short_term_debt_concept="DebtCurrent")
    assert components.total_debt == pytest.approx(1_000.0 + 7_470.0)
    excluded = {e["field"] for e in components.excluded}
    assert "current_portion_of_long_term_debt" in excluded
    assert ND.DCF_NET_DEBT_COMPONENT_OVERLAP in {f["code"] for f in components.findings}


def test_a_non_overlapping_short_term_debt_concept_keeps_all_three_components():
    """An issuer that separates the two genuinely has three components, and
    the overlap rule must not remove one."""
    components = _components(short_term_debt_concept="ShortTermBorrowings")
    assert components.total_debt == pytest.approx(1_000.0 + 1_000.0 + 7_470.0)
    assert components.excluded == []


def test_a_missing_component_is_excluded_not_zero_filled():
    components = _components(short_term_debt_concept="ShortTermBorrowings",
                             current_portion=None)
    assert components.total_debt == pytest.approx(1_000.0 + 7_470.0)
    assert components.current_portion_of_long_term_debt is None


def test_a_reported_total_that_disagrees_with_the_components_wins():
    """Phase H.15 changed the RESOLUTION, and this test's own premise is why.

    The docstring here already said the issuer's reported $8,470M "is what
    proves the $1.0B pair is one obligation" -- that is, the component sum of
    $9,470M double-counts and the issuer's total is the correct figure. The
    old behaviour nonetheless kept the $9,470M and filed a warning beside
    it, which meant every downstream consumer used the number the test
    itself described as wrong.

    Sections 5-7: identity is established first (the concept is on the
    inclusive list and the issuer does not separately report an equal
    noncurrent portion), and only then does precedence apply -- an issuer's
    own consolidated total outranks a sum assembled here, which can
    double-count or omit.
    """
    components = _components(short_term_debt_concept="ShortTermBorrowings",
                             reported_total=8_470.0,
                             reported_total_concept="LongTermDebt")
    assert components.total_debt == pytest.approx(8_470.0)
    assert components.total_debt_validity == "VALID"
    # Recorded as a resolution, not as an unexplained discrepancy.
    resolution = next(f for f in components.findings
                      if f["code"] == ND.DCF_NET_DEBT_COMPONENT_OVERLAP)
    assert resolution["severity"] == "info"
    assert resolution.get("resolution") == "issuer_reported_total"


def test_a_reported_total_that_agrees_produces_no_finding():
    components = _components(short_term_debt_concept="DebtCurrent",
                             reported_total=8_470.0,
                             reported_total_concept="LongTermDebt")
    assert components.total_debt == pytest.approx(8_470.0)
    warnings = [f for f in components.findings if f["severity"] == "warning"]
    assert warnings == []


# ---------------------------------------------------------------------------
# Section 14 — the two policies, and the fact that they differ
# ---------------------------------------------------------------------------

def test_cash_only_net_debt_is_total_debt_less_cash():
    components = _components(short_term_debt_concept="DebtCurrent")
    result = ND.compute_net_debt(components, ND.NetDebtPolicyName.CASH_ONLY)
    assert result.value == pytest.approx(8_470.0 - 13_237.0)
    assert result.policy == ND.NetDebtPolicyName.CASH_ONLY
    assert result.eligible_marketable_securities == 0.0


def test_cash_and_securities_net_debt_also_deducts_the_securities():
    components = _components(short_term_debt_concept="DebtCurrent")
    result = ND.compute_net_debt(
        components, ND.NetDebtPolicyName.CASH_AND_MARKETABLE_SECURITIES)
    assert result.value == pytest.approx(8_470.0 - 13_237.0 - 37_098.0)
    assert result.eligible_marketable_securities == pytest.approx(37_098.0)


def test_the_two_policies_produce_different_numbers():
    """If they did not, the policy field would be decoration."""
    components = _components(short_term_debt_concept="DebtCurrent")
    cash_only = ND.compute_net_debt(components, ND.NetDebtPolicyName.CASH_ONLY)
    with_securities = ND.compute_net_debt(
        components, ND.NetDebtPolicyName.CASH_AND_MARKETABLE_SECURITIES)
    assert cash_only.value != pytest.approx(with_securities.value)


def test_the_policy_is_recorded_on_every_result():
    components = _components()
    for policy in ND.NetDebtPolicyName.ALL:
        result = ND.compute_net_debt(components, policy)
        assert result.policy == policy
        assert policy in result.derivation


def test_ineligible_securities_are_reported_but_not_netted():
    """The configured switch must change the NUMBER visibly, not silently."""
    components = _components(short_term_debt_concept="DebtCurrent")
    result = ND.compute_net_debt(
        components, ND.NetDebtPolicyName.CASH_AND_MARKETABLE_SECURITIES,
        marketable_securities_eligible=False)
    assert result.value == pytest.approx(8_470.0 - 13_237.0)
    assert result.eligible_marketable_securities == 0.0
    assert "does not treat them as eligible" in result.derivation
    assert result.components.short_term_investments == pytest.approx(37_098.0)


def test_equity_securities_are_never_netted_by_either_policy():
    components = _components(short_term_debt_concept="DebtCurrent")
    for policy in ND.NetDebtPolicyName.ALL:
        result = ND.compute_net_debt(components, policy)
        assert result.value >= 8_470.0 - 13_237.0 - 37_098.0
    assert components.equity_securities_at_fair_value == pytest.approx(30_237.0)


def test_an_unsupported_policy_is_rejected_rather_than_defaulted():
    with pytest.raises(ValueError):
        ND.compute_net_debt(_components(), "cash_and_vibes")


def test_a_negative_net_debt_is_a_normal_result():
    """A net-cash company is not an error."""
    result = ND.compute_net_debt(_components(short_term_debt_concept="DebtCurrent"),
                                 ND.NetDebtPolicyName.CASH_ONLY)
    assert result.value < 0


def test_a_positive_net_debt_is_a_normal_result():
    selections = {
        "cash_and_cash_equivalents": _sel("cash_and_cash_equivalents", 17_570.0,
                                          "CashAndCashEquivalentsAtCarryingValue"),
        "short_term_debt": _sel("short_term_debt", 9_323.0, "ShortTermBorrowings"),
        "long_term_debt": _sel("long_term_debt", 134_631.0,
                               "LongTermDebtAndCapitalLeaseObligations"),
    }
    components = ND.collect_components(selections, as_of_date="2026-06-30")
    result = ND.compute_net_debt(components, ND.NetDebtPolicyName.CASH_ONLY)
    assert result.value == pytest.approx(9_323.0 + 134_631.0 - 17_570.0)


def test_missing_cash_or_debt_produces_no_number_rather_than_a_guess():
    components = ND.collect_components({}, as_of_date="2026-06-30")
    result = ND.compute_net_debt(components, ND.NetDebtPolicyName.CASH_ONLY)
    assert result.value is None
    assert result.reconciled is False
    assert "could not be computed" in result.derivation


# ---------------------------------------------------------------------------
# Section 16 — recalculation before the DCF
# ---------------------------------------------------------------------------

def test_a_matching_figure_reconciles():
    components = _components(short_term_debt_concept="DebtCurrent")
    result = ND.compute_net_debt(components, ND.NetDebtPolicyName.CASH_ONLY)
    verified = ND.verify_net_debt(result, 8_470.0 - 13_237.0)
    assert verified.reconciled is True
    assert not [f for f in verified.findings if f["severity"] == "error"]


def test_a_mismatched_figure_produces_a_structured_failure():
    """The $1.0B double-count, caught before the valuation is used."""
    components = _components(short_term_debt_concept="DebtCurrent")
    result = ND.compute_net_debt(components, ND.NetDebtPolicyName.CASH_ONLY)
    verified = ND.verify_net_debt(result, 9_470.0 - 13_237.0)   # the wrong figure
    assert verified.reconciled is False
    codes = {f["code"] for f in verified.findings}
    assert ND.DCF_NET_DEBT_RECONCILIATION_FAILURE in codes
    failure = next(f for f in verified.findings
                   if f["code"] == ND.DCF_NET_DEBT_RECONCILIATION_FAILURE)
    # reported - recalculated: the reported figure is the DOUBLE-COUNTED one,
    # so it is $1.0B higher than the components support.
    assert failure["difference"] == pytest.approx(1_000.0)
    assert failure["severity"] == "error"


def test_reconciliation_is_skipped_when_there_is_nothing_to_compare():
    components = _components()
    result = ND.compute_net_debt(components, ND.NetDebtPolicyName.CASH_ONLY)
    assert ND.verify_net_debt(result, None).reconciled is True
