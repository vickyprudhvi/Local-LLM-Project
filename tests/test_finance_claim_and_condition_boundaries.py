"""Cases F-J: what a research stage may say, and what may reach a report.

Four boundaries, one shape of failure in each: a rule that existed and was
applied to some of the places that needed it.

F. A claim may not restate a quarter's guidance as the year's. The guidance
   record has always carried its target period; nothing checked the claim
   against it.

G. An unfavourable outcome may not sit under Upgrade Conditions. The polarity
   classifier existed and read direction words without the negation that
   reverses them, so "revenue fails to meet guidance" scored UNCLASSIFIED and
   "margins fail to expand" scored FAVORABLE.

H. A condition may not promise to resolve an issue this analysis did not
   report. The check existed and refused to act on a bucket holding a single
   condition, so the one case where the whole bucket is wrong was the one it
   skipped.

I. The minimum-content rule applied to claims and risks but not to every
   claim-bearing field, so a placeholder could still reach the report through
   `thesis`, `primary_reason` or a rationale statement.

J. Internal field names are not user-facing text. A readiness reason read
   "see dcf.warnings / dcf.validation_reasons" and went straight into the
   compact report.
"""

import pytest

from finance import claim_validation as CV
from finance import research_pipeline as R
from finance.report_model import (ClaimSet, ReportStatus, RiskItem,
                                  StockAnalysisReportModel)


# ---------------------------------------------------------------------------
# F. a claim must preserve the guidance target period
# ---------------------------------------------------------------------------

_Q1_GUIDANCE = {
    "revenue_growth": {
        "name": "revenue_growth", "low": 0.27, "high": 0.29,
        "target_period": "Q1 FY2027", "fiscal_period": "Q1 FY2027",
        "target_period_type": "NEXT_QUARTER", "units": "ratio",
        "evidence_id": "dcf.guidance.revenue_growth.current",
    },
}


@pytest.mark.parametrize("claim", [
    "Management guides 27-29% growth for fiscal year 2027.",
    "Management guides 27% to 29% revenue growth for FY2027.",
    "The company expects full-year 2027 revenue growth of 27-29%.",
])
def test_a_quarter_guide_may_not_be_restated_as_a_full_year(claim):
    findings = CV.scan_for_guidance_period_mismatch(claim, _Q1_GUIDANCE)
    assert findings, f"no finding for {claim!r}"
    assert any("Q1 FY2027" in f for f in findings), findings


@pytest.mark.parametrize("claim", [
    "Management guides Q1 FY2027 revenue growth of 27-29%.",
    "The company expects first-quarter fiscal 2027 revenue growth of 27% to 29%.",
    "Next-quarter revenue growth is guided at 27-29%.",
])
def test_a_claim_that_names_the_right_period_is_accepted(claim):
    assert CV.scan_for_guidance_period_mismatch(claim, _Q1_GUIDANCE) == []


def test_a_claim_naming_no_period_is_left_alone():
    """This checks references, not vocabulary. A claim that states no period
    makes no period error."""
    assert CV.scan_for_guidance_period_mismatch(
        "Management guidance points to continued strong growth.", _Q1_GUIDANCE) == []


def test_an_annual_claim_is_accepted_when_annual_guidance_exists():
    """The whole point of keeping both horizons: an FY claim is correct when
    the company published FY guidance, and wrong when it did not."""
    guidance = dict(_Q1_GUIDANCE)
    guidance["revenue"] = {
        "name": "revenue", "low": 118.0, "high": 120.0,
        "target_period": "FY2027", "fiscal_period": "FY2027",
        "target_period_type": "CURRENT_FISCAL_YEAR", "units": "currency",
    }
    assert CV.scan_for_guidance_period_mismatch(
        "Management guides FY2027 revenue of $118-120 billion.", guidance) == []


# ---------------------------------------------------------------------------
# G. condition polarity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("condition", [
    "revenue fails to meet guidance",
    "Revenue growth fails to meet management guidance",
    "the company misses its revenue guidance",
    "margins fail to expand as expected",
    "free cash flow does not improve",
    "revenue growth is unable to sustain current levels",
])
def test_an_unfavourable_outcome_is_classified_unfavourable(condition):
    assert R.classify_condition_direction(condition) == R.CONDITION_UNFAVOURABLE, condition


@pytest.mark.parametrize("condition", [
    "margins expand beyond the guided range",
    "revenue growth accelerates above guidance",
    "free cash flow improves materially",
])
def test_a_favourable_outcome_is_still_favourable(condition):
    assert R.classify_condition_direction(condition) == R.CONDITION_FAVOURABLE, condition


def test_an_unfavourable_condition_is_moved_out_of_the_upgrade_bucket():
    routed = R.route_conditions_by_direction({
        "conditions_that_strengthen_the_view": ["revenue fails to meet guidance"],
        "conditions_that_weaken_the_view": [],
        "reassessment_triggers": [],
    })
    assert routed["conditions_that_strengthen_the_view"] == []
    assert "revenue fails to meet guidance" in routed["conditions_that_weaken_the_view"]


def test_a_sole_wrong_polarity_condition_is_still_moved():
    """The "never empty a bucket" rule protected the exact case where the
    bucket is entirely wrong. An empty Upgrade Conditions list is honest; a
    downgrade filed as an upgrade is not."""
    routed = R.route_conditions_by_direction({
        "conditions_that_strengthen_the_view": ["margins fail to expand as expected"],
        "conditions_that_weaken_the_view": [],
        "reassessment_triggers": [],
    })
    assert routed["conditions_that_strengthen_the_view"] == []


# ---------------------------------------------------------------------------
# H. a condition may not reference an issue that no longer exists
# ---------------------------------------------------------------------------

def test_a_condition_referencing_a_resolved_issue_is_dropped():
    validated = R.drop_conditions_referencing_absent_issues(
        {"reassessment_triggers": ["the debt discrepancy is resolved"],
         "conditions_that_strengthen_the_view": [],
         "conditions_that_weaken_the_view": []},
        issue_codes=[])
    assert validated["reassessment_triggers"] == []


def test_a_condition_referencing_a_present_issue_is_kept():
    validated = R.drop_conditions_referencing_absent_issues(
        {"reassessment_triggers": ["the debt discrepancy is resolved"],
         "conditions_that_strengthen_the_view": [],
         "conditions_that_weaken_the_view": []},
        issue_codes=["TOTAL_DEBT_CONFLICT"])
    assert validated["reassessment_triggers"] == ["the debt discrepancy is resolved"]


def test_the_last_condition_in_a_bucket_is_not_protected_from_this_check():
    """A reference that cannot be verified is wrong however few siblings it
    has, and a reader acting on it goes looking for a problem this analysis
    never found."""
    validated = R.drop_conditions_referencing_absent_issues(
        {"conditions_that_strengthen_the_view": ["the share count conflict is reconciled"],
         "conditions_that_weaken_the_view": [], "reassessment_triggers": []},
        issue_codes=[])
    assert validated["conditions_that_strengthen_the_view"] == []


# ---------------------------------------------------------------------------
# I. one content contract, every claim-bearing field
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blank", ["", "   ", ".", "N/A", "TBD", "-"])
@pytest.mark.parametrize("field", ["thesis", "primary_reason"])
def test_a_placeholder_is_refused_in_every_claim_bearing_field(blank, field):
    with pytest.raises(R._Invalid):
        R._claim_text({field: blank}, field, max_len=500)


@pytest.mark.parametrize("blank", ["", "  ", ".", "N/A"])
def test_the_report_model_refuses_a_placeholder_analysis_limitation(blank):
    with pytest.raises(ValueError):
        StockAnalysisReportModel(
            symbol="TEST", status=ReportStatus(headline="**COMPLETE**."),
            analysis_limitations=(RiskItem(text="A real limitation of the analysis.",),
                                  RiskItem(text=blank)))


@pytest.mark.parametrize("blank", [".", "N/A", "TBD"])
def test_the_report_model_refuses_a_placeholder_claim(blank):
    with pytest.raises(ValueError):
        ClaimSet(claims=("A real claim about revenue.", blank))


def test_real_content_still_constructs():
    model = StockAnalysisReportModel(
        symbol="TEST", status=ReportStatus(headline="**COMPLETE**."),
        analysis_limitations=(RiskItem(text="No valuation could be produced."),),
        bull_case=ClaimSet(claims=("Revenue grew materially year over year.",)))
    assert model.analysis_limitations and model.bull_case.claims


# ---------------------------------------------------------------------------
# J. internal diagnostic text is not user-facing text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "The DCF passed validation but carries a warning (see dcf.warnings / "
    "dcf.validation_reasons).",
    "See facts['dcf_suitability'] for detail.",
    "Blocked by DCF_INPUT_PACKET_INVALID.",
    "bull_researcher did not complete.",
])
def test_internal_references_are_detected(text):
    assert R.contains_internal_reference(text), text


@pytest.mark.parametrize("text", [
    "The valuation is usable only with substantial caveats.",
    "No discounted-cash-flow valuation was produced for this company.",
    "Management guidance targets the next quarter rather than the full year.",
    "Revenue growth of 27% to 29% was guided for the coming quarter.",
])
def test_ordinary_prose_is_not_flagged(text):
    assert not R.contains_internal_reference(text), text


def test_the_report_model_refuses_internal_text_in_a_readiness_reason():
    from finance.report_model import ResearchView
    with pytest.raises(ValueError, match="internal"):
        StockAnalysisReportModel(
            symbol="TEST", status=ReportStatus(headline="**COMPLETE**."),
            research_view=ResearchView(
                readiness_reason="The DCF passed validation but carries a warning "
                                 "(see dcf.warnings / dcf.validation_reasons)."))
