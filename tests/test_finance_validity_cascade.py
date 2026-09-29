"""Phase H.15 — an invalid figure cannot be consumed as a number.

The architecture could already detect a bad canonical fact and then carried
on computing with it. Measured on a synthetic issuer whose component debt
sum disagreed with its own reported total by 16%:

    total_debt = 80_001_000_000
    findings   = [DCF_NET_DEBT_COMPONENT_OVERLAP]

A warning beside a usable number. `validation_status` existed on every
metric, was copied everywhere, and was checked nowhere.

These tests assert the two halves of the fix: invalidity PROPAGATES along
declared dependencies (section 1), and it propagates ONLY there (section
21). A debt conflict must take out leverage and the equity bridge, and must
leave revenue, margin and the technicals alone.
"""

import pytest

from finance import validity as V
from finance.validity import ValidatedMetric, Validity


ALL_METRICS = (
    "total_debt", "eligible_liquidity", "stockholders_equity",
    "cash_and_cash_equivalents", "current_assets", "current_liabilities",
    "revenue", "operating_income", "net_income", "operating_cash_flow",
    "capital_expenditure", "free_cash_flow", "operating_margin", "net_margin",
    "net_debt", "debt_to_equity", "net_debt_to_equity", "current_ratio",
    "free_cash_flow_margin", "debt_to_fcf", "net_debt_to_fcf",
    "dcf_enterprise_value", "dcf_equity_bridge", "dcf_equity_value",
    "share_basis", "modeled_value_per_share", "market_price_comparison",
    "valuation_derived_risk", "valuation_recommendation_reason",
)


def _graph(broken=None, code="TEST_CONFLICT"):
    metrics = {name: ValidatedMetric(name, 100.0) for name in ALL_METRICS}
    if broken:
        metrics[broken].invalidate(code, f"{broken} failed its own validation")
    return V.propagate(metrics)


def _invalid(metrics):
    return {n for n, m in metrics.items() if m.validity == Validity.INVALID}


# ---------------------------------------------------------------------------
# Section 20 — the cascade table
# ---------------------------------------------------------------------------

def test_a_invalid_total_debt_takes_out_the_whole_valuation_chain():
    """Section 1's worked example, end to end."""
    metrics = _graph("total_debt")
    invalid = _invalid(metrics)
    for name in ("net_debt", "debt_to_equity", "dcf_equity_bridge",
                 "dcf_equity_value", "modeled_value_per_share",
                 "market_price_comparison", "valuation_derived_risk",
                 "valuation_recommendation_reason"):
        assert name in invalid, name


def test_a_invalid_total_debt_leaves_unrelated_metrics_alone():
    """Section 21: invalidation follows dependencies, not panic."""
    metrics = _graph("total_debt")
    for name in ("revenue", "operating_margin", "net_margin",
                 "operating_cash_flow", "free_cash_flow", "current_ratio"):
        assert metrics[name].usable, name


def test_b_invalid_shares_stop_per_share_but_not_enterprise_value():
    metrics = _graph("share_basis")
    assert metrics["modeled_value_per_share"].validity == Validity.INVALID
    assert metrics["market_price_comparison"].validity == Validity.INVALID
    # Enterprise value does not depend on the share basis and remains
    # available internally (section 20B).
    assert metrics["dcf_enterprise_value"].usable
    assert metrics["dcf_equity_value"].usable


def test_c_invalid_current_assets_stop_the_ratio_and_nothing_else():
    metrics = _graph("current_assets")
    assert metrics["current_ratio"].validity == Validity.INVALID
    assert metrics["dcf_equity_value"].usable
    assert metrics["revenue"].usable
    assert metrics["net_debt"].usable


def test_d_invalid_fcf_stops_fcf_metrics_only():
    metrics = _graph("free_cash_flow")
    assert metrics["free_cash_flow_margin"].validity == Validity.INVALID
    assert metrics["net_debt_to_fcf"].validity == Validity.INVALID
    assert metrics["debt_to_fcf"].validity == Validity.INVALID
    assert metrics["operating_margin"].usable
    assert metrics["revenue"].usable


def test_the_cascade_reaches_a_fixed_point():
    """total_debt -> net_debt -> net_debt_to_equity is three levels deep."""
    metrics = _graph("total_debt")
    assert metrics["net_debt_to_equity"].validity == Validity.INVALID


# ---------------------------------------------------------------------------
# Section 13 — the cause, distinguished from its consequences
# ---------------------------------------------------------------------------

def test_only_the_originating_metric_is_a_root_cause():
    metrics = _graph("total_debt")
    assert V.root_causes(metrics) == ["total_debt"]
    cascade = V.invalidated_by_propagation(metrics)
    assert "net_debt" in cascade
    assert "total_debt" not in cascade


def test_a_propagated_invalidation_names_what_it_depended_on():
    metrics = _graph("total_debt")
    assert "total_debt" in metrics["net_debt"].reasons[0]


# ---------------------------------------------------------------------------
# Sections 4 / 22 — the number is not readable, and the metadata is enforced
# ---------------------------------------------------------------------------

def test_an_invalid_metric_exposes_no_value():
    metric = ValidatedMetric("total_debt", 7_200_000_000.0)
    assert metric.value == 7_200_000_000.0
    metric.invalidate("TOTAL_DEBT_CONFLICT", "components disagree")
    assert metric.value is None
    assert metric.usable is False


def test_the_figure_survives_for_the_audit_view_only():
    """A reader investigating a withheld valuation needs the number that
    failed; nothing on the calculation path reaches for it."""
    metric = ValidatedMetric("total_debt", 7_200_000_000.0)
    metric.invalidate("TOTAL_DEBT_CONFLICT", "components disagree")
    assert metric.raw_value == 7_200_000_000.0


def test_identical_numbers_with_different_validity_behave_differently():
    """Section 22: proves the metadata is enforced rather than carried."""
    valid = ValidatedMetric("total_debt", 100.0)
    invalid = ValidatedMetric("total_debt", 100.0)
    invalid.invalidate("TOTAL_DEBT_CONFLICT", "components disagree")

    assert valid.raw_value == invalid.raw_value
    assert valid.value == 100.0
    assert invalid.value is None

    downstream_valid = V.propagate({"total_debt": valid,
                                    "net_debt": ValidatedMetric("net_debt", 50.0)})
    downstream_invalid = V.propagate({"total_debt": invalid,
                                      "net_debt": ValidatedMetric("net_debt", 50.0)})
    assert downstream_valid["net_debt"].value == 50.0
    assert downstream_invalid["net_debt"].value is None


@pytest.mark.parametrize("state, usable", [
    (Validity.VALID, True),
    (Validity.LIMITED, True),      # a caveat, not a disqualification
    (Validity.INVALID, False),
    (Validity.UNAVAILABLE, False),
    (Validity.NOT_APPLICABLE, False),
])
def test_which_states_may_be_read(state, usable):
    metric = ValidatedMetric("x", 1.0)
    metric.validity = state
    assert metric.usable is usable


def test_the_weakest_input_decides_a_derivation():
    assert Validity.worst([Validity.VALID, Validity.LIMITED]) == Validity.LIMITED
    assert Validity.worst([Validity.VALID, Validity.INVALID]) == Validity.INVALID
    assert Validity.worst([]) == Validity.UNAVAILABLE


# ---------------------------------------------------------------------------
# Sections 5-7 — resolve where a rule proves it, invalidate otherwise
# ---------------------------------------------------------------------------

def test_an_authoritative_total_resolves_the_conflict():
    """Section 6: an issuer's own consolidated figure outranks a sum
    assembled here, which can omit or double-count a component."""
    metric = V.resolve_or_invalidate(
        "total_debt", component_sum=80_001_000_000, reported_total=95_000_000_000,
        reported_is_authoritative=True, reported_concept="DebtLongtermAndShorttermCombinedAmount")
    assert metric.validity == Validity.VALID
    assert metric.value == 95_000_000_000
    assert "issuer" in metric.reasons[0].lower()


def test_a_non_authoritative_conflict_invalidates():
    """Section 5: promoting one side of an unexplained 16% disagreement is a
    guess, and a guess that reaches a valuation looks like a measurement."""
    metric = V.resolve_or_invalidate(
        "total_debt", component_sum=80_001_000_000, reported_total=95_000_000_000,
        reported_is_authoritative=False, reported_concept="LongTermDebt")
    assert metric.validity == Validity.INVALID
    assert metric.value is None
    assert "TOTAL_DEBT_CONFLICT" in metric.reason_codes


def test_agreement_within_tolerance_is_not_a_conflict():
    metric = V.resolve_or_invalidate(
        "total_debt", component_sum=80_000_000_000, reported_total=80_100_000_000,
        reported_is_authoritative=False, reported_concept="LongTermDebt")
    assert metric.validity == Validity.VALID
    assert metric.value == 80_000_000_000


def test_nothing_reported_at_all_is_unavailable_not_invalid():
    """Absence and contradiction are different states."""
    metric = V.resolve_or_invalidate("total_debt", None, None, False)
    assert metric.validity == Validity.UNAVAILABLE


# ---------------------------------------------------------------------------
# Section 23 — forbidden consumption, stated as impossibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("broken, forbidden", [
    ("total_debt", "net_debt"),
    ("net_debt", "dcf_equity_bridge"),
    ("share_basis", "modeled_value_per_share"),
    ("modeled_value_per_share", "market_price_comparison"),
    ("market_price_comparison", "valuation_derived_risk"),
    ("valuation_derived_risk", "valuation_recommendation_reason"),
])
def test_forbidden_consumption(broken, forbidden):
    metrics = _graph(broken)
    assert metrics[forbidden].value is None
    assert metrics[forbidden].validity == Validity.INVALID


def test_an_invalid_valuation_cannot_reach_the_recommendation_rationale():
    """The full section 1 chain: a debt conflict must not end in a
    valuation-based reason to buy or avoid."""
    metrics = _graph("total_debt")
    assert metrics["valuation_recommendation_reason"].value is None


def test_the_debt_conflict_does_not_reach_the_live_path_as_a_number():
    """The measured failure, at the production entry point."""
    from finance import net_debt as ND

    class _Sel:
        def __init__(self, value, concept):
            self.value, self.concept = value, concept
            self.as_of_date = "2026-06-30"

    components = ND.collect_components(
        {"short_term_debt": _Sel(1_000_000, "DebtCurrent"),
         "long_term_debt": _Sel(80_000_000_000, "LongTermDebtNoncurrent")},
        reported_total_debt=95_000_000_000,
        reported_total_debt_concept="DebtLongtermAndShorttermCombinedAmount")
    # Resolvable here, and resolved to the issuer's own total rather than
    # the component sum that disagreed with it.
    assert components.total_debt == 95_000_000_000
    assert components.total_debt_validity == Validity.VALID


# ---------------------------------------------------------------------------
# Spec section 9 — the cascade is wired into the pipeline, not just tested
# ---------------------------------------------------------------------------

class _StateWithDebtConflict:
    net_debt_detail = {"total_debt_validity": "INVALID",
                       "total_debt_reasons": ["components disagree by 16%"]}


class _StateClean:
    net_debt_detail = {"total_debt_validity": "VALID", "total_debt_reasons": []}


def _facts():
    return {"canonical_evidence": {"current": {
        "total_debt": {"value": 80e9}, "net_debt": {"value": 70e9},
        "stockholders_equity": {"value": 50e9}, "debt_to_equity": {"value": 1.6},
        "revenue": {"value": 100e9}, "operating_margin": {"value": 0.2},
        "free_cash_flow": {"value": 8e9}}}}


def test_the_graph_is_built_from_canonical_state():
    """The wiring that was missing: `propagate()` existed and nothing called
    it, so the global invariant held in tests and not in the pipeline."""
    from finance.workflow import _build_validity_graph

    graph = _build_validity_graph(_facts(), _StateClean())
    assert graph["total_debt"].usable
    assert graph["net_debt"].usable


def test_an_unresolved_debt_conflict_cascades_through_the_pipeline_graph():
    from finance.workflow import _build_validity_graph

    graph = _build_validity_graph(_facts(), _StateWithDebtConflict())
    assert graph["total_debt"].validity == Validity.INVALID
    assert graph["net_debt"].validity == Validity.INVALID
    assert graph["net_debt"].value is None
    assert graph["debt_to_equity"].validity == Validity.INVALID


def test_the_cascade_leaves_unrelated_pipeline_metrics_alone():
    """Section 13/21 at the production entry point."""
    from finance.workflow import _build_validity_graph

    graph = _build_validity_graph(_facts(), _StateWithDebtConflict())
    for name in ("revenue", "operating_margin", "stockholders_equity", "free_cash_flow"):
        assert graph[name].usable, name


def test_an_invalid_equity_bridge_input_blocks_the_dcf():
    """Spec section 13: the bridge is refused rather than built on a number
    the system already knows is wrong."""
    from finance import workflow as W

    facts = _facts()
    facts["_validity_graph"] = W._build_validity_graph(facts, _StateWithDebtConflict())
    inputs, reason = W._dcf_inputs_from_facts("ZZ", facts, forecast_years=5)
    assert inputs is None
    assert facts["dcf_unavailable_code"] == W.DCF_EQUITY_BRIDGE_INPUT_INVALID
    assert "not valid" in reason


def test_a_clean_graph_does_not_block_the_dcf():
    """The gate must not cost every ordinary issuer its valuation."""
    from finance import workflow as W

    facts = _facts()
    facts["_validity_graph"] = W._build_validity_graph(facts, _StateClean())
    _inputs, reason = W._dcf_inputs_from_facts("ZZ", facts, forecast_years=5)
    # It may still decline for unrelated reasons (no statements here), but
    # never for the equity-bridge validity reason.
    assert facts.get("dcf_unavailable_code") != W.DCF_EQUITY_BRIDGE_INPUT_INVALID
