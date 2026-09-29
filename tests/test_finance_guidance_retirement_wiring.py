"""`finance.actualization.retire_realized_guidance` (Fixture D) was correct
in isolation and was never called from anywhere in the production pipeline,
so completed-period guidance survived indefinitely in a live full-stack V2
run. This tests the WIRING itself -- `finance.workflow._retire_completed_
guidance` -- not the retirement rule, which `tests/test_finance_
actualization_guidance.py` already covers exhaustively.
"""

from finance.actualization import resolve_current_actual_state
from finance.extraction.schema import (
    ReportedActualCandidate,
    SourceType,
    StatementCompleteness,
)
from finance.guidance import GuidanceStatus
from finance.workflow import _retire_completed_guidance

FULL = ("revenue", "operating_income", "net_income", "cash_and_cash_equivalents",
        "total_debt", "stockholders_equity", "operating_cash_flow",
        "capital_expenditure", "shares_diluted")
TODAY = "2026-11-01"


def _q3_actual_resolution():
    q3 = ReportedActualCandidate(
        period_end="2026-04-30", form="10-Q", issued_at="2026-06-05",
        fiscal_period="Q3", fiscal_year=2026,
        statement_completeness=StatementCompleteness.COMPLETE,
        present_metrics=FULL, source_type=SourceType.FORMAL_PERIODIC_FILING)
    return resolve_current_actual_state([q3], as_of=TODAY)


def _guidance_payload():
    """The real production SHAPE: `finance.sec.current_guidance`'s dict
    output, already past the point of being live GuidanceMetric objects."""
    return {
        "fiscal_year": 2026,
        "metrics": {
            "revenue": {"name": "revenue", "fiscal_period": "Q3 FY2026",
                       "status": GuidanceStatus.CURRENT},
        },
        "all_metrics": [
            {"name": "revenue", "fiscal_period": "Q3 FY2026",
             "status": GuidanceStatus.CURRENT},
            {"name": "revenue_growth", "fiscal_period": "Q4 FY2026",
             "status": GuidanceStatus.CURRENT},
        ],
    }


def test_v1_produces_no_change_at_all():
    """No resolution to check against -- byte-for-byte untouched."""
    updated, reasons = _retire_completed_guidance(_guidance_payload(), None)
    assert updated is None
    assert reasons == []


def test_no_guidance_produces_no_change():
    updated, reasons = _retire_completed_guidance(None, _q3_actual_resolution())
    assert updated is None
    assert reasons == []


def test_nothing_to_retire_produces_no_change():
    """Actualization resolved a period, but nothing guided targets it."""
    payload = {"metrics": {}, "all_metrics": [
        {"name": "revenue", "fiscal_period": "Q4 FY2026",
         "status": GuidanceStatus.CURRENT}]}
    updated, reasons = _retire_completed_guidance(payload, _q3_actual_resolution())
    assert updated is None
    assert reasons == []


def test_q3_guidance_retires_leaving_q4_as_current_guidance():
    """7A, at the wiring layer: Q3 guidance + Q3 actual + Q4 guidance already
    present -> the rebuilt payload's current coverage is Q4 only."""
    updated, reasons = _retire_completed_guidance(
        _guidance_payload(), _q3_actual_resolution())

    assert updated is not None
    assert reasons and "revenue" in reasons[0]

    current_periods = {m["fiscal_period"] for m in updated["all_metrics"]}
    assert current_periods == {"Q4 FY2026"}
    assert "revenue" not in updated["metrics"], (
        "the retired metric must not remain in the name-keyed current view")

    realized_periods = {m["fiscal_period"] for m in updated["realized_guidance"]}
    assert realized_periods == {"Q3 FY2026"}
    assert updated["realized_guidance"][0]["status"] == GuidanceStatus.REALIZED


def test_the_name_keyed_view_is_rebuilt_from_the_surviving_statements():
    """7B: a metric guided ONLY for the completed quarter disappears from
    current-guided-metric coverage entirely -- it is never backfilled from a
    stale entry that happens to share its name."""
    payload = {
        "metrics": {"revenue": {"name": "revenue", "fiscal_period": "Q3 FY2026",
                                "status": GuidanceStatus.CURRENT}},
        "all_metrics": [
            {"name": "revenue", "fiscal_period": "Q3 FY2026",
             "status": GuidanceStatus.CURRENT},
        ],
    }
    updated, _reasons = _retire_completed_guidance(payload, _q3_actual_resolution())
    assert updated is not None
    assert updated["metrics"] == {}
    assert updated["all_metrics"] == []
