"""Phase H.7 — reporting frameworks (sections 7, 58) and research-condition
semantics (sections 52-53, 63).

Nothing here is issuer-specific.
"""

import pytest

from finance import taxonomy as TX
from finance.research_pipeline import (
    CONDITION_BIDIRECTIONAL,
    CONDITION_FAVOURABLE,
    CONDITION_UNFAVOURABLE,
    _route_conditions_by_direction,
    classify_condition_direction,
    is_circular_price_condition,
)


# ---------------------------------------------------------------------------
# Section 7 — reporting frameworks
# ---------------------------------------------------------------------------

def _facts(taxonomies):
    return {"facts": {name: {f"C{i}": {} for i in range(count)}
                      for name, count in taxonomies.items()}}


def test_a_us_gaap_filer_is_detected():
    assert TX.detect_taxonomy(_facts({"us-gaap": 300, "dei": 2})) == TX.US_GAAP


def test_an_ifrs_filer_is_detected():
    assert TX.detect_taxonomy(_facts({"ifrs-full": 287, "dei": 2})) == TX.IFRS


def test_the_larger_framework_wins_during_a_transition():
    """An issuer changing filing status genuinely carries both; preferring
    one by default would pick the stub side."""
    assert TX.detect_taxonomy(_facts({"us-gaap": 12, "ifrs-full": 250})) == TX.IFRS
    assert TX.detect_taxonomy(_facts({"us-gaap": 250, "ifrs-full": 12})) == TX.US_GAAP


def test_an_unreadable_framework_is_named_rather_than_reported_as_empty():
    reason = TX.unsupported_taxonomy_reason(_facts({"jp-gaap": 100}))
    assert reason and "jp-gaap" in reason and "no reviewed concept mapping" in reason


def test_document_metadata_alone_is_not_financial_statements():
    reason = TX.unsupported_taxonomy_reason(_facts({"dei": 3, "ecd": 2}))
    assert reason and "only document metadata" in reason


def test_an_empty_payload_says_so():
    assert TX.unsupported_taxonomy_reason({}) is not None


def test_a_readable_framework_produces_no_complaint():
    assert TX.unsupported_taxonomy_reason(_facts({"us-gaap": 300})) is None


def test_each_framework_gets_its_own_concept_map():
    us = TX.concept_map_for(TX.US_GAAP)
    ifrs = TX.concept_map_for(TX.IFRS)
    assert "revenue" in us and "revenue" in ifrs
    assert us["revenue"][1] != ifrs["revenue"][1]
    # The normalized field NAMES are shared, which is what keeps every
    # consumer downstream framework-agnostic.
    assert {"revenue", "cash_and_cash_equivalents", "long_term_debt"} <= set(ifrs)


def test_instant_and_duration_flags_agree_across_frameworks():
    """A field that is a point-in-time balance in one framework must not be a
    flow in the other, or the same normalized name would mean two things."""
    us = TX.concept_map_for(TX.US_GAAP)
    ifrs = TX.concept_map_for(TX.IFRS)
    for field in set(us) & set(ifrs):
        assert us[field][0] == ifrs[field][0], field


def test_foreign_private_issuer_forms_are_recognized():
    for form in ("20-F", "20-F/A", "40-F"):
        assert form in TX.ANNUAL_FORMS
    for form in ("6-K", "10-Q"):
        assert form in TX.INTERIM_FORMS
    assert "8-K" in TX.EARNINGS_MATERIAL_FORMS
    assert "6-K" in TX.EARNINGS_MATERIAL_FORMS


def test_a_non_usd_reporting_currency_is_reported_rather_than_silently_dropped():
    facts = {"facts": {"ifrs-full": {"Revenue": {"units": {"EUR": [
        {"start": "2025-01-01", "end": "2025-12-31", "val": 100}]}}}}}
    note = TX.reporting_currency_note(facts)
    assert note and "EUR" in note and "no currency conversion" in note


def test_a_usd_reporting_currency_produces_no_note():
    facts = {"facts": {"ifrs-full": {"Revenue": {"units": {"USD": [
        {"start": "2025-01-01", "end": "2025-12-31", "val": 100}]}}}}}
    assert TX.reporting_currency_note(facts) is None


# ---------------------------------------------------------------------------
# Section 53 — no circular price conditions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "the bear thesis becomes more likely if the price falls toward the bear value",
    "the share price re-rates toward the modeled value",
    "the market cap discount to the base modeled value widens",
    "the stock trades at a premium to fair value",
])
def test_a_price_only_condition_is_circular(text):
    assert is_circular_price_condition(text) is True


@pytest.mark.parametrize("text", [
    "the price falls below $20 while free cash flow stays negative",
    "revenue growth falls below 5%",
    "operating margin compresses below 18%",
    "the valuation gap widens as leverage increases",
])
def test_a_condition_naming_an_operating_driver_is_not_circular(text):
    assert is_circular_price_condition(text) is False


def test_a_circular_condition_is_dropped_from_every_bucket():
    routed = _route_conditions_by_direction({
        "conditions_that_strengthen_the_view": [
            "the share price re-rates toward the modeled value"],
        "conditions_that_weaken_the_view": [
            "the price falls toward the bear modeled value"],
        "reassessment_triggers": [],
    })
    assert routed["conditions_that_strengthen_the_view"] == []
    assert routed["conditions_that_weaken_the_view"] == []
    assert routed["reassessment_triggers"] == []


def test_a_legitimate_condition_survives_routing_alongside_a_circular_one():
    routed = _route_conditions_by_direction({
        "conditions_that_strengthen_the_view": [
            "the share price re-rates toward the modeled value",
            "operating margin expands above 20%"],
        "conditions_that_weaken_the_view": [],
        "reassessment_triggers": [],
    })
    assert routed["conditions_that_strengthen_the_view"] == [
        "operating margin expands above 20%"]


# ---------------------------------------------------------------------------
# Section 52 — direction semantics
# ---------------------------------------------------------------------------

def test_lower_leverage_with_strong_margins_is_favorable():
    assert classify_condition_direction(
        "lower leverage while margins remain strong") == CONDITION_FAVOURABLE


def test_worsening_cash_burn_is_unfavorable():
    assert classify_condition_direction(
        "cash burn worsens through the year") == CONDITION_UNFAVOURABLE


def test_a_share_count_reconciliation_is_bidirectional():
    assert classify_condition_direction(
        "the share-count conflict is reconciled") == CONDITION_BIDIRECTIONAL


def test_a_reversed_condition_is_moved_to_the_bucket_its_direction_implies():
    routed = _route_conditions_by_direction({
        "conditions_that_strengthen_the_view": [],
        "conditions_that_weaken_the_view": ["net debt to EBITDA falls below 2.5x"],
        "reassessment_triggers": [],
    })
    assert routed["conditions_that_strengthen_the_view"] == [
        "net debt to EBITDA falls below 2.5x"]
    assert routed["conditions_that_weaken_the_view"] == []


# ---------------------------------------------------------------------------
# Section 52 — vocabulary gaps found by a live loss-making run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    # Filed as an UPGRADE condition by a live run: "accelerates" was scored
    # as unconditionally favourable and neither "burn" nor "equity" was a
    # known subject.
    ("cash burn accelerates while shareholder equity declines", CONDITION_UNFAVOURABLE),
    # Filed as a DOWNGRADE condition by the same run.
    ("operating margins improve toward positivity while free cash flow burn decreases",
     CONDITION_FAVOURABLE),
    ("shareholder equity declines", CONDITION_UNFAVOURABLE),
    ("cash runway extends beyond three years", CONDITION_FAVOURABLE),
    ("the net loss widens", CONDITION_UNFAVOURABLE),
    ("share count rises through further issuance", CONDITION_UNFAVOURABLE),
])
def test_burn_equity_and_loss_conditions_are_scored_by_subject(text, expected):
    assert classify_condition_direction(text) == expected


def test_acceleration_is_a_direction_not_a_verdict():
    """Accelerating revenue is good; accelerating cash burn is not."""
    assert classify_condition_direction("revenue growth accelerates") ==         CONDITION_FAVOURABLE
    assert classify_condition_direction("cash burn accelerates") ==         CONDITION_UNFAVOURABLE
