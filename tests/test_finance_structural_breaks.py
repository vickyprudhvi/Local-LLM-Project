"""Phase H.6 — historical comparability and post-balance-sheet events
(sections 13, 18, 22, 25).

The rule under test is that a STRUCTURAL BREAK is something the ISSUER said,
not something a growth rate implies. AT&T's history is not comparable because
AT&T tagged discontinued operations through the DirecTV and WarnerMedia
separations; NVIDIA's 114% revenue year is not a break because NVIDIA sold
more of the same product and tagged nothing. Getting that backwards would
discard the only real information there is about a fast-growing company.
"""

import pytest

from finance import structural_breaks as SB
from finance.research_pipeline import (
    CONDITION_BIDIRECTIONAL,
    CONDITION_FAVOURABLE,
    CONDITION_UNCLASSIFIED,
    CONDITION_UNFAVOURABLE,
    classify_condition_direction,
)

REVENUE = "RevenueFromContractWithCustomerExcludingAssessedTax"
DISCONTINUED = "IncomeLossFromDiscontinuedOperationsNetOfTax"


def _annual(year, value):
    return {"start": f"{year}-01-01", "end": f"{year}-12-31", "val": value,
            "form": "10-K", "filed": f"{year + 1}-02-15", "accn": f"0000-{year}-1",
            "fy": year, "fp": "FY"}


def _facts(revenues, discontinued_ends=()):
    concepts = {REVENUE: {"units": {"USD": [_annual(y, v) for y, v in revenues]}}}
    if discontinued_ends:
        concepts[DISCONTINUED] = {"units": {"USD": [
            {"start": f"{end[:4]}-01-01", "end": end, "val": 1_000, "form": "10-K",
             "filed": end, "accn": "0000-1", "fy": int(end[:4]), "fp": "FY"}
            for end in discontinued_ends]}}
    return {"facts": {"us-gaap": concepts}}


def _submissions(rows):
    return {"filings": {"recent": {
        "form": [r[0] for r in rows],
        "items": [r[1] for r in rows],
        "filingDate": [r[2] for r in rows],
        "accessionNumber": [f"0000-26-{i}" for i, _ in enumerate(rows)],
    }}}


STEADY = [(2021, 100), (2022, 105), (2023, 110), (2024, 116), (2025, 122), (2026, 128)]
FAST = [(2021, 100), (2022, 160), (2023, 170), (2024, 380), (2025, 810), (2026, 1340)]
SEPARATED = [(2021, 180), (2022, 130), (2023, 122), (2024, 122), (2025, 125),
             (2026, 128)]


# ---------------------------------------------------------------------------
# Section 13 — classification
# ---------------------------------------------------------------------------

def test_a_steady_history_with_no_filed_change_is_comparable():
    assessment = SB.assess_historical_comparability(_facts(STEADY))
    assert assessment.status == SB.HistoricalComparability.COMPARABLE
    assert assessment.summary


def test_fast_growth_alone_is_not_a_structural_break():
    """The NVIDIA rule. Revenue more than doubling twice is this company's own
    reported growth; downweighting it would discard the real information."""
    assessment = SB.assess_historical_comparability(_facts(FAST))
    assert assessment.status == SB.HistoricalComparability.COMPARABLE
    # The discontinuity is still REPORTED as context.
    assert any(e.kind == SB.BreakKind.REVENUE_DISCONTINUITY
               for e in assessment.evidence)


def test_discontinued_operations_inside_the_window_is_a_structural_break():
    """The AT&T rule: the ISSUER said the business changed, in the years a
    five-year CAGR is computed over."""
    assessment = SB.assess_historical_comparability(
        _facts(SEPARATED, discontinued_ends=("2022-12-31", "2023-12-31")))
    assert assessment.status == SB.HistoricalComparability.STRUCTURAL_BREAK
    assert assessment.affected_periods
    assert "structural" in assessment.summary.lower() or "reduced weight" in \
        assessment.summary


def test_discontinued_operations_before_the_window_is_only_partial():
    """A separation completed a decade ago does not make the last five years
    non-comparable."""
    long_history = [(y, 100 + i * 5) for i, y in enumerate(range(2012, 2027))]
    assessment = SB.assess_historical_comparability(
        _facts(long_history, discontinued_ends=("2013-12-31",)))
    assert assessment.status == SB.HistoricalComparability.PARTIALLY_COMPARABLE


def test_too_little_history_is_unknown_not_comparable():
    """"We did not look far enough back" and "we looked and it is clean" are
    different claims."""
    assessment = SB.assess_historical_comparability(_facts([(2025, 100), (2026, 110)]))
    assert assessment.status == SB.HistoricalComparability.UNKNOWN


def test_a_completed_acquisition_on_an_8k_is_partial_comparability():
    assessment = SB.assess_historical_comparability(
        _facts(STEADY),
        _submissions([("8-K", "2.01", "2024-06-01")]))
    assert assessment.status == SB.HistoricalComparability.PARTIALLY_COMPARABLE


def test_the_break_names_the_first_comparable_period_after_it():
    assessment = SB.assess_historical_comparability(
        _facts(SEPARATED, discontinued_ends=("2022-12-31",)))
    assert assessment.comparable_from is not None
    assert assessment.comparable_from > "2022-12-31"


def test_the_model_is_never_asked_whether_a_break_happened():
    """Every evidence entry names a FILING or a reported figure — there is no
    path by which a model-authored claim becomes a structural break."""
    assessment = SB.assess_historical_comparability(
        _facts(SEPARATED, discontinued_ends=("2022-12-31",)))
    for evidence in assessment.evidence:
        assert evidence.source in ("sec_company_facts", "sec_submissions")


# ---------------------------------------------------------------------------
# Section 18 — material events after the balance-sheet date
# ---------------------------------------------------------------------------

def test_a_debt_obligation_filed_after_the_balance_sheet_is_reported():
    events = SB.find_post_balance_sheet_events(
        _submissions([("8-K", "1.01,2.03", "2026-08-01")]),
        balance_sheet_date="2026-06-30", valuation_date="2026-08-17")
    assert len(events) == 1
    assert "direct financial obligation" in events[0].description


def test_a_filing_BEFORE_the_balance_sheet_date_is_not_an_event():
    events = SB.find_post_balance_sheet_events(
        _submissions([("8-K", "2.03", "2026-05-01")]),
        balance_sheet_date="2026-06-30", valuation_date="2026-08-17")
    assert events == []


def test_a_filing_after_the_valuation_date_is_not_an_event():
    events = SB.find_post_balance_sheet_events(
        _submissions([("8-K", "2.03", "2026-09-01")]),
        balance_sheet_date="2026-06-30", valuation_date="2026-08-17")
    assert events == []


def test_a_routine_8k_item_is_not_a_material_event():
    """Item 5.02 is an officer appointment; it does not change the bridge."""
    events = SB.find_post_balance_sheet_events(
        _submissions([("8-K", "5.02", "2026-08-01")]),
        balance_sheet_date="2026-06-30", valuation_date="2026-08-17")
    assert events == []


def test_an_unidentifiable_securities_filing_claims_nothing():
    """Phase H.9, section 28 reversed this deliberately.

    A bare prospectus supplement with no describable security was previously
    reported as "a securities offering" and treated as potential equity
    dilution. On a live issuer that produced a dilution warning built from an
    investment-grade DEBT offering and an insider's Form 144. An event whose
    type cannot be determined now asserts nothing -- in particular, not
    dilution.
    """
    events = SB.find_post_balance_sheet_events(
        _submissions([("424B2", "", "2026-08-01")]),
        balance_sheet_date="2026-06-30", valuation_date="2026-08-17")
    assert len(events) == 1
    event = events[0]
    assert event.event_type == SB.PostBalanceSheetEventType.UNKNOWN
    assert event.impact["potential_dilution"] is False
    assert event.impact["affects_share_count"] is False


def test_a_debt_offering_is_identified_as_debt_not_dilution():
    events = SB.find_post_balance_sheet_events(
        _submissions([("8-K", "2.03", "2026-08-01")]),
        balance_sheet_date="2026-06-30", valuation_date="2026-08-17")
    assert len(events) == 1
    event = events[0]
    assert event.event_type == SB.PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE
    assert event.impact["affects_debt"] is True
    assert event.impact["potential_dilution"] is False
    assert "debt" in event.description


def test_an_insider_sale_changes_nothing_about_the_issuer():
    """Section 25: existing shares changing hands."""
    events = SB.find_post_balance_sheet_events(
        _submissions([("144", "", "2026-08-01")]),
        balance_sheet_date="2026-06-30", valuation_date="2026-08-17")
    assert len(events) == 1
    event = events[0]
    assert event.event_type == SB.PostBalanceSheetEventType.INSIDER_SECONDARY_SALE
    assert event.impact["affects_share_count"] is False
    assert event.impact["potential_dilution"] is False
    assert event.impact["affects_cash"] is False
    assert event.impact["requires_reassessment"] is False


def test_no_submissions_index_means_no_events_not_a_crash():
    assert SB.find_post_balance_sheet_events(None, "2026-06-30") == []
    assert SB.find_post_balance_sheet_events({}, None) == []


# ---------------------------------------------------------------------------
# Section 22 — condition direction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "net debt / FCF declines below 5x while margins remain stable",
    "net debt to EBITDA falls below 2.5x",
    "leverage declines toward the target range",
    "churn declines below 0.8%",
    "interest expense declines as debt is refinanced",
    "free cash flow grows above $20 billion",
    "operating margin expands above 20%",
])
def test_favorable_conditions_are_classified_favorable(text):
    assert classify_condition_direction(text) == CONDITION_FAVOURABLE


@pytest.mark.parametrize("text", [
    "net debt to EBITDA rises above 3.5x",
    "leverage increases while free cash flow declines",
    "revenue declines for two consecutive quarters",
    "free cash flow falls below $15 billion",
    "operating margin declines below 18%",
    "churn rises above 1%",
    "costs increase faster than revenue",
])
def test_unfavorable_conditions_are_classified_unfavorable(text):
    assert classify_condition_direction(text) == CONDITION_UNFAVOURABLE


@pytest.mark.parametrize("text", [
    "a company-specific WACC becomes available",
    "reported CapEx history becomes available",
    "the share-count conflict is reconciled",
    "the segment disclosure is restated",
])
def test_information_changes_are_bidirectional(text):
    assert classify_condition_direction(text) == CONDITION_BIDIRECTIONAL


def test_an_unclassifiable_condition_is_accepted_rather_than_forced():
    assert classify_condition_direction("management changes its capital plan") == \
        CONDITION_UNCLASSIFIED
    assert classify_condition_direction("") == CONDITION_UNCLASSIFIED


def test_a_leverage_ratio_is_not_read_as_the_metric_in_its_denominator():
    """"net debt / FCF" contains "FCF", which is a higher-is-better metric.
    Longest-match on the LOWER-is-better list is what keeps the ratio a
    leverage measure."""
    assert classify_condition_direction("net debt / FCF declines") == \
        CONDITION_FAVOURABLE
    assert classify_condition_direction("FCF declines") == CONDITION_UNFAVOURABLE


def test_conditions_are_routed_into_the_bucket_their_direction_implies():
    """Section 22: upgrade -> FAVORABLE, downgrade -> UNFAVORABLE,
    reassessment -> BIDIRECTIONAL. A misfiled condition is MOVED, never
    rejected — rejecting took whole pipeline stages down over one phrase."""
    from finance.research_pipeline import _route_conditions_by_direction

    routed = _route_conditions_by_direction({
        "conditions_that_strengthen_the_view": [],
        # The exact misfiling the live T report produced.
        "conditions_that_weaken_the_view": [
            "net debt / FCF declines below 5x while margins remain stable"],
        "reassessment_triggers": [],
    })
    assert routed["conditions_that_strengthen_the_view"] == [
        "net debt / FCF declines below 5x while margins remain stable"]
    assert routed["conditions_that_weaken_the_view"] == []


def test_a_genuine_downgrade_condition_stays_where_it_was_filed():
    from finance.research_pipeline import _route_conditions_by_direction

    routed = _route_conditions_by_direction({
        "conditions_that_strengthen_the_view": [],
        "conditions_that_weaken_the_view": ["revenue declines for two quarters"],
        "reassessment_triggers": [],
    })
    assert routed["conditions_that_weaken_the_view"] == [
        "revenue declines for two quarters"]
