"""Part 17 — the boundaries, proven where they are actually crossed.

Spec §19 warns against brittle source-code-string tests, and most of what
this phase built is better checked by behaviour (see
`tests/test_finance_canaries.py`, which runs the whole pipeline eight times).
What behaviour cannot check is the SHAPE of the code: that there is exactly
one route to the valuation engine, and that a future edit adding a second one
is caught the day it is written rather than the day it produces a wrong
number.

So the static checks here are deliberately few, and each one guards a
property that has no behavioural equivalent:

  * `finance.dcf_model` is invoked from exactly one place
  * that place requires a ValidatedDCFInputPacket to invoke it
  * the compact renderer reads the report model, not the payload

Everything else in this file is a behaviour test.
"""

import inspect
import re

import pytest

from finance import dcf_packet, evidence, report_model, workflow


# ---------------------------------------------------------------------------
# One route to the model
# ---------------------------------------------------------------------------

def test_the_dcf_tool_is_invoked_from_exactly_one_function():
    """A gate you must remember to call is a gate that eventually is not
    called. There is one call site, and it is the validated adapter."""
    source = inspect.getsource(workflow)
    callers = re.findall(r"_call_tool\(\s*executor,\s*DCF_TOOL_NAME", source)
    assert len(callers) == 1, f"{len(callers)} call sites invoke the DCF tool"
    assert "_call_tool(executor, DCF_TOOL_NAME, packet.arguments(), step=step)" in \
        inspect.getsource(workflow._run_validated_dcf)


def test_the_adapter_refuses_anything_that_is_not_a_packet():
    with pytest.raises(TypeError):
        workflow._run_validated_dcf(object(), {"base_revenue": 1.0})
    with pytest.raises(TypeError):
        workflow._run_validated_dcf(object(), None)


def test_the_adapter_refuses_a_packet_that_did_not_pass_validation():
    """A packet that exists is a packet that passed -- but the frozen
    dataclass can still be constructed directly, so the adapter checks."""
    from finance.validity import Validity

    packet = dcf_packet.ValidatedDCFInputPacket(
        revenue=1.0, validation_status=Validity.INVALID,
        validation_reasons=("total_debt is invalid",))
    with pytest.raises(ValueError, match="did not pass validation"):
        workflow._run_validated_dcf(object(), packet)


def test_the_engine_arguments_live_inside_the_packet():
    """Part 2's structural point. If the arguments travelled beside the
    packet, a caller could hold them and skip the packet; because they are
    reachable only through it, skipping validation means constructing the
    very object validation produces."""
    packet, failure = dcf_packet.build_dcf_input_packet(values={
        "revenue": 1000.0, "operating_margin": 0.15, "tax_rate": 0.21,
        "total_debt": 100.0, "net_debt": 50.0, "share_count": 10.0,
        "share_basis": "sec_weighted_average_diluted",
        "wacc": 0.09, "terminal_growth": 0.025,
        "model_arguments": {"base_revenue": 1000.0, "ticker": "TEST"},
    })
    assert failure is None
    assert packet.arguments() == {"base_revenue": 1000.0, "ticker": "TEST"}
    # Frozen: an input that passed validation cannot be adjusted afterwards.
    with pytest.raises(Exception):
        packet.revenue = 2000.0


# ---------------------------------------------------------------------------
# The evidence index gates on the valuation status
# ---------------------------------------------------------------------------

def _payload(status, **extra):
    payload = {
        "symbol": "TEST",
        "valuation_status": status,
        "dcf": {"available": True, "validation_status": "DCF_VALID",
                "primary_scenario": "base", "value_per_share": 100.0,
                "scenarios": [{"scenario": "base", "value_per_share": 100.0,
                               "enterprise_value": 1000.0, "equity_value": 900.0}]},
        "dcf_scenario_spread": {"available": True, "bull_value_per_share": 120.0,
                                "base_value_per_share": 100.0,
                                "bear_value_per_share": 80.0,
                                "spread_pct_of_base": 0.40},
        "valuation_gap": {"available": True, "direction": "above",
                          "market_price_premium_pct": 0.25,
                          "modeled_return_to_value_pct": -0.20,
                          "difference_pct": 0.25},
    }
    payload.update(extra)
    return payload


@pytest.mark.parametrize("status", [
    dcf_packet.ValuationStatus.INPUT_PACKET_INVALID,
    dcf_packet.ValuationStatus.MODEL_ARITHMETIC_INVALID,
    dcf_packet.ValuationStatus.FORECAST_PATH_INVALID,
    dcf_packet.ValuationStatus.NOT_APPLICABLE_FOR_BUSINESS_MODEL,
])
def test_no_valuation_number_is_indexed_for_a_refused_status(status):
    """Part 4: only VALID_FOR_RESEARCH may expose a modelled value, a
    premium, a modelled return or a direction."""
    index = evidence.build_evidence_index(_payload(status))
    for prefix in ("dcf.value_per_share", "dcf.enterprise_value", "dcf.equity_value",
                   "scenario_spread.", "valuation_gap."):
        assert not [k for k in index if k.startswith(prefix)], prefix
    assert index["valuation.status"].value == status
    assert index["valuation.conclusions_withheld"].value


def test_limited_keeps_the_direction_and_drops_every_magnitude():
    """LIMITED is usable with substantial caveats: "the price is above the
    modelled value" is a statement those inputs still support; a percentage
    rendered to one decimal place is not."""
    index = evidence.build_evidence_index(
        _payload(dcf_packet.ValuationStatus.LIMITED))
    assert index["valuation_gap.direction"].value == "above"
    assert "valuation_gap.market_price_premium_pct" not in index
    assert "valuation_gap.modeled_return_to_value_pct" not in index
    assert not [k for k in index if k.startswith("scenario_spread.")]


def test_valid_for_research_publishes_the_full_valuation():
    index = evidence.build_evidence_index(
        _payload(dcf_packet.ValuationStatus.VALID_FOR_RESEARCH))
    assert index["dcf.value_per_share.base"].value == 100.0
    assert index["scenario_spread.spread_pct_of_base"].value == 0.40
    assert index["valuation_gap.market_price_premium_pct"].value == 0.25
    assert "valuation.conclusions_withheld" not in index


def test_the_gate_is_the_same_function_the_report_model_uses():
    """One status, read by both. The failure mode this guards against is the
    evidence index and the report disagreeing about whether a valuation may
    be stated -- which is how a report came to carry "comparison: not
    meaningful" in one section and a precise premium in the next."""
    payload = _payload(dcf_packet.ValuationStatus.LIMITED)
    assert report_model.valuation_from(payload).status == \
        evidence.valuation_evidence_status(payload)


def test_an_unassessed_suitability_is_not_a_limited_valuation():
    """"A discounted-cash-flow valuation suits this company only with
    caveats" and "nothing was available to decide that" are different
    statements. Both had to be reported as LIMITED because there was
    nowhere else to put the second, and the gate then read the second as
    though it were the first."""
    payload = _payload(None)
    payload["dcf_suitability"] = {"dcf_suitability": "LIMITED", "assessed": False}
    assert evidence.valuation_evidence_status(payload) == \
        dcf_packet.ValuationStatus.VALID_FOR_RESEARCH

    payload["dcf_suitability"] = {"dcf_suitability": "LIMITED", "assessed": True}
    assert evidence.valuation_evidence_status(payload) == \
        dcf_packet.ValuationStatus.LIMITED


# ---------------------------------------------------------------------------
# The renderer consumes the report model
# ---------------------------------------------------------------------------

_RENDERER_FUNCTIONS = (
    "_snapshot_rows", "_compact_status_section", "_valuation_section",
    "_technical_section", "_compact_claims_section", "_compact_risk_section",
    "_compact_research_view_section", "_compact_sources_section",
)


def test_no_compact_renderer_function_reads_the_payload_directly():
    """Part 13. Not "zero dictionary reads" -- `compact.get(...)` reads used
    to INFER FINANCE SEMANTICS. There are none left in these functions,
    because none of them receives the payload at all."""
    for name in _RENDERER_FUNCTIONS:
        source = inspect.getsource(getattr(workflow, name))
        assert "compact.get(" not in source, f"{name} still reads the compact payload"
        assert "compact_payload" not in source, name


def test_the_compact_report_builds_the_model_once():
    source = inspect.getsource(workflow.render_compact_report)
    assert "build_stock_analysis_report_model" in source
    assert source.count("build_stock_analysis_report_model") == 1


@pytest.mark.parametrize("name", _RENDERER_FUNCTIONS)
def test_no_renderer_function_decides_valuation_validity(name):
    """Part 12's list, checked where it can be: the renderer may not consult
    validation statuses, suitability, share reconciliation or business-model
    applicability. Every one of those was a branch in here."""
    source = inspect.getsource(getattr(workflow, name))
    for forbidden in ("validation_status", "dcf_suitability", "share_reconciliation",
                      "standard_fcff_suitability", "valuation_method_status",
                      "MATERIAL_DIFFERENCE", "NOT_SUITABLE"):
        assert forbidden not in source, f"{name} still decides {forbidden}"


def test_the_report_model_carries_no_raw_provider_objects():
    """Part 9: display-ready semantic facts, not raw provider data. Every
    snapshot row is a settled label with a settled value and a settled kind,
    so there is nothing for a renderer to reach into."""
    from finance.report_model import SnapshotRow, ValueKind

    row = SnapshotRow(label="Revenue (TTM)", value=1.0, kind=ValueKind.CURRENCY)
    assert not hasattr(row, "raw")
    assert set(row.__dataclass_fields__) == {"label", "kind", "value", "text"}


# ---------------------------------------------------------------------------
# Packet failure propagation (Part 3)
# ---------------------------------------------------------------------------

def test_a_refused_packet_produces_no_partial_inputs():
    packet, failure = dcf_packet.build_dcf_input_packet(values={
        "revenue": 1000.0, "operating_margin": 0.15, "tax_rate": 0.21,
        "total_debt": None, "net_debt": None, "share_count": 10.0,
        "share_basis": "sec", "wacc": 0.09, "terminal_growth": 0.025,
    })
    assert packet is None
    assert "total_debt" in failure.invalid_inputs
    assert "net_debt" in failure.invalid_inputs


def test_a_refusal_record_carries_the_cause_and_no_numbers():
    failure = dcf_packet.PacketFailure(
        code="DCF_INPUT_PACKET_INVALID",
        reasons=["total_debt is required for an equity valuation and is not available."],
        invalid_inputs=["total_debt"])
    record = workflow._valuation_unavailable_record({}, failure)
    assert record["available"] is False
    assert record["valuation_status"] == dcf_packet.ValuationStatus.INPUT_PACKET_INVALID
    assert record["packet_failure"]["invalid_inputs"] == ["total_debt"]
    # No previously-computed value, no partial input set, no figure with a
    # warning attached.
    assert "scenarios" not in record
    assert "value_per_share" not in record
    assert "inputs_ready" not in record


def test_every_pre_packet_refusal_names_the_input_it_is_about():
    """A bare code sends a reader looking for the failure; the named input
    tells them where it is."""
    for code, inputs in workflow._INPUT_BLAMED_BY_CODE.items():
        assert inputs, code
        for name in inputs:
            assert name in dcf_packet.REQUIRED_INPUTS or name in (
                "business_model", "financial_base", "forecast_assumptions"), name
