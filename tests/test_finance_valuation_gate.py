"""Parts 19/21 — the valuation boundary and guidance supersession.

Two boundaries that had the same shape of gap: a rule that existed as
metadata and was never the thing that decided. Valuation eligibility was
spread across several consumers each of which had to remember to check it;
guidance supersession existed as an enum with no resolver.

These tests assert the boundaries refuse, and refuse for the right reason.
No issuer is named in any production path exercised here.
"""

import pytest

from finance import guidance as G
from finance.dcf_packet import (
    PacketFailure, ValuationStatus, build_dcf_input_packet,
    build_valuation_research_evidence, classify_valuation,
    may_publish_valuation_conclusion,
)
from finance.validity import ValidatedMetric, Validity


def _inputs(**overrides):
    base = dict(revenue=100e9, operating_margin=0.2, tax_rate=0.21,
                total_debt=10e9, net_debt=5e9, share_count=1e9,
                share_basis="CURRENT_OUTSTANDING", wacc=0.09, terminal_growth=0.025)
    base.update(overrides)
    return base


def _broken(name):
    metric = ValidatedMetric(name, 1.0)
    metric.invalidate("TEST_CONFLICT", f"{name} failed validation")
    return {name: metric}


class _Model:
    def __init__(self, suitability="SUITABLE", profile="STANDARD_OPERATING_COMPANY"):
        self.standard_fcff_suitability = suitability
        self.profile = profile


# ---------------------------------------------------------------------------
# Parts 2-3 — construction fails closed
# ---------------------------------------------------------------------------

def test_a_complete_set_of_valid_inputs_builds_a_packet():
    packet, failure = build_dcf_input_packet(values=_inputs())
    assert failure is None
    assert packet is not None and packet.ok


@pytest.mark.parametrize("broken", ["total_debt", "net_debt", "share_count",
                                    "share_basis", "revenue", "operating_margin"])
def test_an_invalid_required_input_refuses_construction(broken):
    """Part 3: `value + warning` must never reach the engine."""
    packet, failure = build_dcf_input_packet(
        values=_inputs(), validity_graph=_broken(broken))
    assert packet is None
    assert broken in failure.invalid_inputs


@pytest.mark.parametrize("missing", ["revenue", "net_debt", "share_count", "wacc"])
def test_a_missing_required_input_refuses_construction(missing):
    packet, failure = build_dcf_input_packet(values=_inputs(**{missing: None}))
    assert packet is None
    assert missing in failure.invalid_inputs


def test_a_discount_rate_below_the_growth_rate_refuses_construction():
    """A perpetuity needs WACC > g; otherwise the terminal value is not
    merely uncertain, it is meaningless."""
    packet, failure = build_dcf_input_packet(
        values=_inputs(wacc=0.02, terminal_growth=0.025))
    assert packet is None
    assert "terminal_growth" in failure.invalid_inputs


def test_an_inapplicable_business_model_refuses_construction():
    packet, failure = build_dcf_input_packet(
        values=_inputs(), business_model=_Model(suitability="NOT_SUITABLE"))
    assert packet is None
    assert "business_model" in failure.invalid_inputs


def test_a_rejected_assumption_refuses_construction():
    """Spec §13: a clamp must never repair a semantic error, and neither may
    a default quietly standing in for a rejected assumption."""
    packet, failure = build_dcf_input_packet(
        values=_inputs(),
        assumption_rejections=[{"code": "GUIDANCE_METRIC_MISMATCH",
                                "reason": "FCF growth is not revenue growth"}])
    assert packet is None
    assert "forecast_assumptions" in failure.invalid_inputs


def test_a_built_packet_cannot_be_mutated():
    """Validating at construction is pointless if a caller can adjust a
    field on the way to the model."""
    packet, _ = build_dcf_input_packet(values=_inputs())
    with pytest.raises(Exception):
        packet.net_debt = 999.0


# ---------------------------------------------------------------------------
# Parts 4-6 / 19 — the output gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, expected", [
    ({"business_model": _Model(suitability="NOT_SUITABLE")},
     ValuationStatus.NOT_APPLICABLE_FOR_BUSINESS_MODEL),
    ({"packet_failure": PacketFailure()}, ValuationStatus.INPUT_PACKET_INVALID),
    ({"dcf_available": True, "dcf_validation_status": "DCF_NEGATIVE_TERMINAL_FCFF"},
     ValuationStatus.FORECAST_PATH_INVALID),
    ({"dcf_available": True, "dcf_validation_status": "DCF_NONFINITE_OUTPUT"},
     ValuationStatus.MODEL_ARITHMETIC_INVALID),
    ({"dcf_available": True, "dcf_validation_status": "DCF_VALID",
      "suitability_status": "LIMITED"}, ValuationStatus.LIMITED),
    ({"dcf_available": True, "dcf_validation_status": "DCF_VALID"},
     ValuationStatus.VALID_FOR_RESEARCH),
])
def test_valuation_status_is_decided_by_cause(kwargs, expected):
    assert classify_valuation(**kwargs) == expected


@pytest.mark.parametrize("status", [
    ValuationStatus.INPUT_PACKET_INVALID,
    ValuationStatus.MODEL_ARITHMETIC_INVALID,
    ValuationStatus.FORECAST_PATH_INVALID,
    ValuationStatus.NOT_APPLICABLE_FOR_BUSINESS_MODEL,
    ValuationStatus.LIMITED,
])
def test_only_valid_for_research_exposes_numbers(status):
    """A role cannot misuse a figure it was never given -- a stronger
    guarantee than asking it not to."""
    evidence = build_valuation_research_evidence(
        status, {"modeled_value_per_share": 120.0,
                 "market_price_premium_pct": 0.4,
                 "modeled_return_to_value_pct": -0.3})
    assert "modeled_value_per_share" not in evidence
    assert "market_price_premium_pct" not in evidence
    assert evidence["valuation_status"] == status
    assert may_publish_valuation_conclusion(status) is False


def test_a_valid_valuation_is_passed_through():
    evidence = build_valuation_research_evidence(
        ValuationStatus.VALID_FOR_RESEARCH, {"modeled_value_per_share": 120.0})
    assert evidence["modeled_value_per_share"] == 120.0
    assert may_publish_valuation_conclusion(ValuationStatus.VALID_FOR_RESEARCH)


def test_the_withheld_conclusions_are_named():
    """A role must not read the absence of a figure as the figure being
    zero or unremarkable."""
    evidence = build_valuation_research_evidence(ValuationStatus.INPUT_PACKET_INVALID)
    assert "valuation_conclusions_withheld" in evidence
    assert "valuation_derived_risk" in evidence["valuation_conclusions_withheld"]


def test_no_status_is_described_as_a_broken_model_unless_it_is():
    from finance.dcf_packet import STATUS_EXPLANATION

    for status in (ValuationStatus.NOT_APPLICABLE_FOR_BUSINESS_MODEL,
                   ValuationStatus.FORECAST_PATH_INVALID,
                   ValuationStatus.INPUT_PACKET_INVALID):
        assert "failed its own" not in (STATUS_EXPLANATION[status] or "")


def test_invalid_debt_cascades_to_a_withheld_valuation():
    """Part 19's chain, end to end: invalid total debt → invalid net debt →
    no packet → no valuation conclusions."""
    packet, failure = build_dcf_input_packet(
        values=_inputs(), validity_graph=_broken("net_debt"))
    assert packet is None
    status = classify_valuation(packet_failure=failure)
    evidence = build_valuation_research_evidence(status, {"modeled_value_per_share": 120.0})
    assert "modeled_value_per_share" not in evidence


# ---------------------------------------------------------------------------
# Parts 9-10 / 21 — supersession
# ---------------------------------------------------------------------------

def _item(name, period, ptype, issued, basis="GAAP", status=None):
    return {"name": name, "fiscal_period": period, "target_period_type": ptype,
            "issued_at": issued, "basis": basis, "status": status}


def _statuses(entries):
    return [e["status"] for e in G.resolve_guidance_status(entries)]


def test_revised_guidance_supersedes_the_figure_it_revised():
    assert _statuses([
        _item("revenue", "FY2027", "CURRENT_FISCAL_YEAR", "2026-06-01"),
        _item("revenue", "FY2027", "CURRENT_FISCAL_YEAR", "2026-09-01"),
    ]) == [G.GuidanceStatus.SUPERSEDED, G.GuidanceStatus.CURRENT]


def test_guidance_for_a_different_quarter_supersedes_nothing():
    """"Latest filing wins" would drop the Q2 figure here."""
    assert _statuses([
        _item("revenue", "Q2 FY2027", "NEXT_QUARTER", "2026-06-01"),
        _item("revenue", "Q3 FY2027", "NEXT_QUARTER", "2026-09-01"),
    ]) == [G.GuidanceStatus.CURRENT, G.GuidanceStatus.CURRENT]


def test_a_long_term_framework_is_not_superseded_by_this_years_outlook():
    """Different forward-information types are different statements."""
    assert _statuses([
        _item("operating_margin", "FY2027", "CURRENT_FISCAL_YEAR", "2026-09-01"),
        _item("operating_margin", "FY2030", "MULTI_YEAR", "2026-03-01"),
    ]) == [G.GuidanceStatus.CURRENT, G.GuidanceStatus.CURRENT]


def test_withdrawn_guidance_never_becomes_current_again():
    """A company that pulled its guidance has said something, and the newest
    remaining figure is not a replacement for it."""
    statuses = _statuses([
        _item("revenue", "FY2027", "CURRENT_FISCAL_YEAR", "2026-06-01",
              status=G.GuidanceStatus.WITHDRAWN),
        _item("revenue", "FY2027", "CURRENT_FISCAL_YEAR", "2026-09-01"),
    ])
    assert statuses == [G.GuidanceStatus.WITHDRAWN, G.GuidanceStatus.CURRENT]


def test_gaap_and_adjusted_are_separate_statements():
    assert _statuses([
        _item("operating_margin", "FY2027", "CURRENT_FISCAL_YEAR", "2026-09-01", "GAAP"),
        _item("operating_margin", "FY2027", "CURRENT_FISCAL_YEAR", "2026-09-01", "adjusted"),
    ]) == [G.GuidanceStatus.CURRENT, G.GuidanceStatus.CURRENT]


def test_an_item_with_no_issue_date_does_not_claim_to_be_newest():
    statuses = _statuses([
        _item("revenue", "FY2027", "CURRENT_FISCAL_YEAR", None),
        _item("revenue", "FY2027", "CURRENT_FISCAL_YEAR", "2026-09-01"),
    ])
    assert statuses == [G.GuidanceStatus.SUPERSEDED, G.GuidanceStatus.CURRENT]


def test_current_guidance_only_filters_to_the_active_set():
    active = G.current_guidance_only([
        _item("revenue", "FY2027", "CURRENT_FISCAL_YEAR", "2026-06-01"),
        _item("revenue", "FY2027", "CURRENT_FISCAL_YEAR", "2026-09-01"),
    ])
    assert len(active) == 1
    assert active[0]["issued_at"] == "2026-09-01"


def test_a_superseded_item_says_why():
    entries = G.resolve_guidance_status([
        _item("revenue", "FY2027", "CURRENT_FISCAL_YEAR", "2026-06-01"),
        _item("revenue", "FY2027", "CURRENT_FISCAL_YEAR", "2026-09-01"),
    ])
    assert "2026-09-01" in entries[0]["status_reason"]
