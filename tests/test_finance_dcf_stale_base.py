"""A valuation built on a superseded period is not a current valuation.

THE FAILURE

Every figure in the model is arithmetically perfect and the answer describes a
quarter that is over. Nothing in the existing status vocabulary said so: the
model ran, its validation passed, and it was published as VALID_FOR_RESEARCH.

WHY IT IS ITS OWN CAUSE

`ValuationStatus` already distinguishes four different reasons a valuation may
not be used, on the principle that "model invalid" told a reader the wrong
thing four ways. A stale base is a fifth and genuinely different fact -- and
notably NOT the model's fault, which is exactly why folding it into
MODEL_ARITHMETIC_INVALID would misdescribe it.

WHAT DID NOT NEED CHANGING

`VALUATION_DERIVED_CONCLUSIONS` and `build_valuation_research_evidence`
already withhold every numeric conclusion on any non-VALID status. Adding the
cause where the other causes live meant the withholding came for free -- which
is what §7 means by not building a second valuation-validity system.
"""

import pytest

from finance.actualization import (
    DCF_BASE_TOLERANCE_DAYS,
    DCF_FINANCIAL_BASE_STALE,
    DcfBaseFreshness,
    assess_dcf_base_freshness,
)
from finance.dcf_packet import (
    VALUATION_DERIVED_CONCLUSIONS,
    ValuationStatus,
    build_valuation_research_evidence,
    classify_valuation,
)

CURRENT = "2026-07-31"
PRIOR = "2026-04-30"


# ---------------------------------------------------------------------------
# §8: canonical period comparison, not string comparison
# ---------------------------------------------------------------------------

def test_the_same_period_is_compatible():
    assessment = assess_dcf_base_freshness(CURRENT, CURRENT)
    assert assessment.freshness == DcfBaseFreshness.CURRENT
    assert assessment.may_be_research_valid is True
    assert assessment.is_stale is False


def test_a_newer_reported_period_makes_the_base_stale():
    assessment = assess_dcf_base_freshness(CURRENT, PRIOR)
    assert assessment.freshness == DcfBaseFreshness.STALE
    assert assessment.is_stale is True
    assert assessment.may_be_research_valid is False
    assert assessment.code == DCF_FINANCIAL_BASE_STALE


def test_an_older_reported_period_is_not_a_false_stale_flag():
    """§8 names this specifically. A base NEWER than any resolved actual is
    not stale, and flagging it would withhold a perfectly current valuation."""
    assessment = assess_dcf_base_freshness(PRIOR, CURRENT)
    assert assessment.freshness == DcfBaseFreshness.AHEAD
    assert assessment.is_stale is False
    assert assessment.may_be_research_valid is True


@pytest.mark.parametrize("current,base", [
    (None, CURRENT), (CURRENT, None), (None, None), ("not-a-date", CURRENT),
])
def test_an_unresolvable_period_fails_closed(current, base):
    """A base whose period cannot be established cannot be SHOWN to be
    current, and an unverifiable valuation is not a verified one."""
    assessment = assess_dcf_base_freshness(current, base)
    assert assessment.freshness == DcfBaseFreshness.UNKNOWN
    assert assessment.may_be_research_valid is False


def test_a_base_within_the_same_reporting_period_is_current():
    """Period ends move by days between issuers and years; a whole period
    does not, which is what the tolerance separates."""
    near = "2026-07-10"
    assessment = assess_dcf_base_freshness(CURRENT, near)
    assert assessment.lag_days is not None
    assert assessment.lag_days <= DCF_BASE_TOLERANCE_DAYS
    assert assessment.freshness == DcfBaseFreshness.CURRENT


def test_the_assessment_reports_both_periods_so_it_can_be_checked():
    assessment = assess_dcf_base_freshness(CURRENT, PRIOR)
    assert assessment.current_period_end == CURRENT
    assert assessment.base_period_end == PRIOR
    assert PRIOR in assessment.reason and CURRENT in assessment.reason


def test_periods_are_compared_as_dates_not_as_labels():
    """The failure §8 forbids: "Q3 FY2026" against "2026-04-25" as strings.

    Two labels for one period must not read as a difference, and this layer
    only ever sees dates -- a label that reached it would be UNKNOWN rather
    than silently mis-compared.
    """
    assessment = assess_dcf_base_freshness("2026-07-31", "Q3 FY2026")
    assert assessment.freshness == DcfBaseFreshness.UNKNOWN
    assert assessment.may_be_research_valid is False


# ---------------------------------------------------------------------------
# §7: the status, through the existing gate
# ---------------------------------------------------------------------------

def test_a_stale_base_is_not_research_valid():
    status = classify_valuation(dcf_available=True,
                                dcf_validation_status="DCF_VALID",
                                financial_base_stale=True)
    assert status == ValuationStatus.FINANCIAL_BASE_STALE
    assert status != ValuationStatus.VALID_FOR_RESEARCH


def test_a_current_base_leaves_the_status_alone():
    status = classify_valuation(dcf_available=True,
                                dcf_validation_status="DCF_VALID",
                                financial_base_stale=False)
    assert status == ValuationStatus.VALID_FOR_RESEARCH


def test_a_stale_base_outranks_limited():
    """A model running on old data is a harder stop than one with caveats."""
    status = classify_valuation(dcf_available=True,
                                dcf_validation_status="DCF_VALID",
                                suitability_status="LIMITED",
                                financial_base_stale=True)
    assert status == ValuationStatus.FINANCIAL_BASE_STALE


@pytest.mark.parametrize("kwargs,expected", [
    ({"packet_failure": object()}, ValuationStatus.INPUT_PACKET_INVALID),
    ({"dcf_available": True, "dcf_validation_status": "DCF_NEGATIVE_TERMINAL_FCFF"},
     ValuationStatus.FORECAST_PATH_INVALID),
    ({"dcf_available": False}, ValuationStatus.INPUT_PACKET_INVALID),
])
def test_more_fundamental_causes_still_win(kwargs, expected):
    """A model that never ran is not "stale" -- it is absent.

    Reporting staleness there would describe the wrong problem.
    """
    assert classify_valuation(financial_base_stale=True, **kwargs) == expected


def test_a_business_model_the_dcf_never_applied_to_is_not_stale():
    class _Model:
        standard_fcff_suitability = "NOT_SUITABLE"

    status = classify_valuation(business_model=_Model(),
                                dcf_available=True,
                                financial_base_stale=True)
    assert status == ValuationStatus.NOT_APPLICABLE_FOR_BUSINESS_MODEL


# ---------------------------------------------------------------------------
# §7: nothing valuation-derived may be published
# ---------------------------------------------------------------------------

def test_no_numeric_conclusion_survives_a_stale_base():
    valuation = {name: 42.0 for name in VALUATION_DERIVED_CONCLUSIONS}
    valuation["modeled_value_per_share"] = 123.45
    evidence = build_valuation_research_evidence(
        ValuationStatus.FINANCIAL_BASE_STALE, valuation)

    assert 123.45 not in evidence.values()
    for name in VALUATION_DERIVED_CONCLUSIONS:
        assert evidence.get(name) is None, f"{name} was published"
    assert evidence["valuation_status"] == ValuationStatus.FINANCIAL_BASE_STALE


def test_the_withheld_conclusions_are_named_so_a_reader_knows_what_is_missing():
    evidence = build_valuation_research_evidence(
        ValuationStatus.FINANCIAL_BASE_STALE, {"modeled_value_per_share": 1.0})
    withheld = evidence.get("valuation_conclusions_withheld") or []
    assert "modeled_value_per_share" in withheld
    assert "valuation_derived_risk" in withheld


def test_the_status_is_explained_without_blaming_the_model():
    from finance.dcf_packet import STATUS_EXPLANATION

    explanation = STATUS_EXPLANATION[ValuationStatus.FINANCIAL_BASE_STALE]
    assert explanation
    lowered = explanation.lower()
    assert "invalid" not in lowered
    assert "superseded" in lowered or "over" in lowered


def test_the_new_status_is_in_the_vocabulary():
    assert ValuationStatus.FINANCIAL_BASE_STALE in ValuationStatus.ALL


# ---------------------------------------------------------------------------
# §27: the gate does not fire under v1 or compare
# ---------------------------------------------------------------------------

def test_the_gate_is_inert_unless_the_v2_layer_answered():
    """§27: compare must never change production behaviour.

    Under v1 there is no resolved period to compare against; under compare,
    V1's answer is the answer by definition. A gate firing in either would
    make the diagnostic mode a silent v2.
    """
    from finance.workflow import _dcf_base_staleness

    for layer in ("v1", None):
        stale, assessment = _dcf_base_staleness({"actualization": {"layer_used": layer}})
        assert stale is False and assessment is None

    stale, assessment = _dcf_base_staleness({})
    assert stale is False and assessment is None


def test_the_gate_engages_when_v2_answered():
    from finance.workflow import _dcf_base_staleness

    class _State:
        latest_quarterly_period = PRIOR
        latest_annual_period = None

    class _Resolution:
        period_end = CURRENT

    stale, assessment = _dcf_base_staleness({
        "actualization": {"layer_used": "v2"},
        "_actual_state_resolution": _Resolution(),
        "_current_financial_state": _State()})
    assert stale is True
    assert assessment.freshness == DcfBaseFreshness.STALE
