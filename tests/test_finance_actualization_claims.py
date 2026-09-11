"""Fixture H: a "latest quarter" label must be backed by the latest quarter.

THE FAILURE

"Latest-quarter revenue growth" whose value belongs to the prior quarter. Not
a labelling slip -- a wrong number under a right name, which is the hardest
kind to notice. Every other check passes: the figure is real, the metric is
real, only the period is a quarter behind.

THE RULE

A label may not assert currency. Currency is CHECKED, against the period the
resolver selected, and the answer is one of five explicit states. There is no
outcome in which a reader cannot tell.
"""

import pytest

from finance.actualization import (
    ActualStateStatus,
    FreshnessEligibility,
    LatestQuarterCode,
    assess_claim_freshness,
    assess_latest_quarter_claims,
    claims_safe_to_publish_as_current,
    reconstruct_ttm,
    resolve_current_actual_state,
)
from finance.extraction.schema import (
    ReportedActualCandidate,
    SourceType,
    StatementCompleteness,
)

TODAY = "2026-11-01"
NEW_Q_END = "2026-07-31"
PRIOR_Q_END = "2026-04-30"

FULL = ("revenue", "operating_income", "net_income", "cash_and_cash_equivalents",
        "total_debt", "stockholders_equity", "operating_cash_flow",
        "capital_expenditure", "shares_diluted")


def _actual(period_end, form, issued_at, fiscal_period, present=FULL,
            completeness=StatementCompleteness.COMPLETE):
    return ReportedActualCandidate(
        period_end=period_end, form=form, issued_at=issued_at,
        fiscal_period=fiscal_period, statement_completeness=completeness,
        present_metrics=tuple(present),
        source_type=(SourceType.FORMAL_PERIODIC_FILING if form != "8-K"
                     else SourceType.PRELIMINARY_EARNINGS_RELEASE))


def _resolution_with_current_quarter():
    return resolve_current_actual_state(
        [_actual(PRIOR_Q_END, "10-Q", "2026-06-05", "Q3"),
         _actual(NEW_Q_END, "8-K", "2026-09-02", "FY")], as_of=TODAY)


# ---------------------------------------------------------------------------
# H: the label matches the resolved quarter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["revenue", "net_income", "operating_income"])
def test_h_a_metric_from_the_current_quarter_may_be_called_latest(metric):
    resolution = _resolution_with_current_quarter()
    assessment = assess_claim_freshness(metric, "latest quarter", resolution)
    assert assessment.eligibility == FreshnessEligibility.CURRENT
    assert assessment.may_be_called_current is True
    assert assessment.fact_period_end == NEW_Q_END
    assert assessment.code == LatestQuarterCode.LABEL_MATCHES_PERIOD


def test_h_the_assessment_names_both_periods_so_it_can_be_checked():
    resolution = _resolution_with_current_quarter()
    assessment = assess_claim_freshness("revenue", "latest quarter", resolution)
    assert assessment.fact_period_end == resolution.period_end
    assert assessment.current_period_end == resolution.period_end


# ---------------------------------------------------------------------------
# H: a prior-quarter value may NEVER wear the label
# ---------------------------------------------------------------------------

def test_h_a_prior_quarter_value_is_refused_the_latest_label():
    """The failure this fixture exists for."""
    resolution = resolve_current_actual_state(
        [_actual(PRIOR_Q_END, "10-Q", "2026-06-05", "Q3"),
         _actual(NEW_Q_END, "8-K", "2026-09-02", "FY",
                 present=("revenue", "net_income"))],
        as_of=TODAY,
        prior_state_metrics={"operating_income": {"period_end": PRIOR_Q_END,
                                                  "form": "10-Q"}})
    assessment = assess_claim_freshness("operating_income", "latest quarter",
                                        resolution)
    assert assessment.may_be_called_current is False
    assert assessment.fact_period_end == PRIOR_Q_END
    assert assessment.code == LatestQuarterCode.LABEL_PERIOD_MISMATCH
    assert "never as" in assessment.reason


def test_h_a_fallback_is_distinguished_from_plain_staleness():
    """Both are refused the label; they are not the same fact.

    FALLBACK means an older value was deliberately retained and labelled;
    STALE means the value simply is not current. A reader deciding whether to
    trust a figure needs to know which.
    """
    resolution = resolve_current_actual_state(
        [_actual(NEW_Q_END, "8-K", "2026-09-02", "FY",
                 present=("revenue",))],
        as_of=TODAY,
        prior_state_metrics={"total_debt": {"period_end": PRIOR_Q_END,
                                            "form": "10-Q"}})
    assessment = assess_claim_freshness("total_debt", "latest quarter", resolution)
    assert assessment.eligibility == FreshnessEligibility.FALLBACK
    assert assessment.may_be_called_current is False


def test_h_a_metric_with_no_provenance_is_unknown_not_current():
    """Fail closed. Absence of evidence is not evidence of currency."""
    resolution = _resolution_with_current_quarter()
    assessment = assess_claim_freshness("free_cash_flow", "latest quarter",
                                        resolution)
    assert assessment.eligibility == FreshnessEligibility.UNKNOWN
    assert assessment.may_be_called_current is False
    assert assessment.code == LatestQuarterCode.METRIC_ABSENT


def test_h_every_metric_lands_in_exactly_one_bucket():
    """§19: no silent third outcome."""
    resolution = _resolution_with_current_quarter()
    assessments = assess_latest_quarter_claims(resolution)
    allowed, refused = claims_safe_to_publish_as_current(assessments)
    assert set(allowed) | set(refused) == set(assessments)
    assert not (set(allowed) & set(refused))
    for assessment in assessments.values():
        assert assessment.eligibility in FreshnessEligibility.ALL


def test_h_only_current_may_claim_currency():
    for state in FreshnessEligibility.ALL:
        may = state in FreshnessEligibility.MAY_CLAIM_CURRENT
        assert may is (state == FreshnessEligibility.CURRENT)


# ---------------------------------------------------------------------------
# TTM claims are judged against the WINDOW, not the metric record
# ---------------------------------------------------------------------------

REVENUE = "Revenues"
Q_ENDS = [("2025-04-01", "2025-06-30"), ("2025-07-01", "2025-09-30"),
          ("2025-10-01", "2025-12-31"), ("2026-01-01", "2026-03-31"),
          ("2026-04-01", "2026-06-30")]


def _facts(rows):
    return {"facts": {"us-gaap": {REVENUE: {"units": {"USD": rows}}}}}


def _quarters(ends):
    return [{"start": s, "end": e, "val": 100 + i * 10, "form": "10-Q",
             "accn": f"a-{e}", "fy": int(e[:4]), "fp": "Q1", "filed": e}
            for i, (s, e) in enumerate(ends)]


def test_a_ttm_claim_is_refused_when_its_window_ends_early():
    """§9's second example: "TTM revenue is Y" when the window ends a quarter
    before the resolved current period."""
    ttm = reconstruct_ttm(_facts(_quarters(Q_ENDS)), "2026-09-30",
                          metrics=("revenue",))
    resolution = _resolution_with_current_quarter()
    assessment = assess_claim_freshness("revenue", "TTM revenue", resolution,
                                        ttm=ttm)
    assert assessment.may_be_called_current is False
    assert assessment.eligibility == FreshnessEligibility.STALE
    assert "ends" in assessment.reason


def test_a_ttm_claim_is_allowed_when_the_window_reaches_the_period():
    ttm = reconstruct_ttm(_facts(_quarters(Q_ENDS)), "2026-06-30",
                          metrics=("revenue",))
    resolution = _resolution_with_current_quarter()
    assessment = assess_claim_freshness("revenue", "TTM revenue", resolution,
                                        ttm=ttm)
    assert assessment.eligibility == FreshnessEligibility.CURRENT


def test_a_ttm_claim_for_a_metric_with_no_window_is_unknown():
    ttm = reconstruct_ttm(_facts(_quarters(Q_ENDS)), "2026-06-30",
                          metrics=("revenue",))
    resolution = _resolution_with_current_quarter()
    assessment = assess_claim_freshness("net_income", "TTM net income",
                                        resolution, ttm=ttm)
    assert assessment.eligibility == FreshnessEligibility.UNKNOWN


def test_the_claim_wording_does_not_decide_anything():
    """Provenance decides. The label is what is being CHECKED, not the input.

    The same metric and the same resolution give the same eligibility whether
    the claim is worded "latest quarter" or "current" or anything else.
    """
    resolution = _resolution_with_current_quarter()
    first = assess_claim_freshness("revenue", "latest quarter", resolution)
    second = assess_claim_freshness("revenue", "current revenue growth", resolution)
    assert first.eligibility == second.eligibility
