"""Fixture D: an outlook for a period whose actual is in is not an outlook.

THE FAILURE

FY2026 guidance and FY2026 actual results, both marked current. The report
then carries a forecast and a fact about the same period with nothing to tell
them apart -- or worse, the forecast anchors an assumption the actual already
contradicts.

WHY `EXPIRED` WAS NOT ENOUGH

`select_current_guidance` already retires guidance whose period has ended by
the CALENDAR. That is a different fact from "the number is in". A company may
report late; a period may end with no results filed for weeks. REALIZED means
we can now measure what was guided, which is the condition that actually
retires a forecast.

The statement is KEPT. What management said before the number arrived is the
most interesting thing about it once the number exists, and deleting it would
throw away the only record of the comparison.
"""

import pytest

from finance.actualization import (
    GuidanceActualizationCode,
    resolve_current_actual_state,
    retire_realized_guidance,
)
from finance.guidance import GuidanceStatus
from finance.extraction.schema import (
    ReportedActualCandidate,
    SourceType,
    StatementCompleteness,
)

FULL = ("revenue", "operating_income", "net_income", "cash_and_cash_equivalents",
        "total_debt", "stockholders_equity", "operating_cash_flow",
        "capital_expenditure", "shares_diluted")

TODAY = "2026-11-01"


def _actual(period_end, form, issued_at, fiscal_period,
            completeness=StatementCompleteness.COMPLETE):
    return ReportedActualCandidate(
        period_end=period_end, form=form, issued_at=issued_at,
        fiscal_period=fiscal_period, statement_completeness=completeness,
        present_metrics=FULL,
        source_type=(SourceType.FORMAL_PERIODIC_FILING if form != "8-K"
                     else SourceType.PRELIMINARY_EARNINGS_RELEASE))


class _Guidance:
    """The shape `retire_realized_guidance` reads. Deliberately minimal.

    A real `GuidanceMetric` is frozen and carries twenty fields; the retirement
    pass touches three of them, and a stand-in makes the test about the rule
    rather than about constructing a domain object.
    """

    def __init__(self, name, fiscal_period, status=GuidanceStatus.CURRENT):
        self.name = name
        self.fiscal_period = fiscal_period
        self.status = status
        self.status_reason = ""

    def with_status(self, status, reason):
        return _Guidance(self.name, self.fiscal_period, status)


FY2026_GUIDANCE = _Guidance("revenue", "FY")
Q1_FY2027_GUIDANCE = _Guidance("revenue", "Q1 FY2027")
FY2027_GUIDANCE = _Guidance("revenue", "FY2027")


# ---------------------------------------------------------------------------
# D-A: before the actuals, guidance is current
# ---------------------------------------------------------------------------

def test_d_a_guidance_is_current_while_the_period_has_no_actuals():
    """Only Q3 is reported; the full year has not happened."""
    q3 = _actual("2026-04-30", "10-Q", "2026-06-05", "Q3")
    resolution = resolve_current_actual_state([q3], as_of=TODAY)
    result = retire_realized_guidance([FY2026_GUIDANCE], resolution)

    assert result.current == [FY2026_GUIDANCE]
    assert result.realized == []
    assert GuidanceActualizationCode.STILL_FORWARD in result.codes


# ---------------------------------------------------------------------------
# D-B: the actual arrives and retires it
# ---------------------------------------------------------------------------

def test_d_b_the_same_period_guidance_becomes_realized():
    fy = _actual("2026-07-31", "8-K", "2026-09-02", "FY")
    resolution = resolve_current_actual_state([fy], as_of=TODAY)
    result = retire_realized_guidance([FY2026_GUIDANCE], resolution)

    assert result.current == []
    assert len(result.realized) == 1
    assert result.realized[0].status == GuidanceStatus.REALIZED
    assert GuidanceActualizationCode.RETIRED_REALIZED in result.codes


def test_d_b_the_retired_statement_is_kept_not_deleted():
    fy = _actual("2026-07-31", "8-K", "2026-09-02", "FY")
    resolution = resolve_current_actual_state([fy], as_of=TODAY)
    result = retire_realized_guidance([FY2026_GUIDANCE], resolution)

    assert result.realized[0].name == "revenue"
    assert result.realized[0].fiscal_period == "FY"
    assert any("kept as history" in r for r in result.reasons)


def test_d_b_realized_is_distinct_from_expired():
    """Two different facts, and the vocabulary keeps them apart."""
    assert GuidanceStatus.REALIZED != GuidanceStatus.EXPIRED
    assert GuidanceStatus.REALIZED in GuidanceStatus.RETIRED
    assert GuidanceStatus.CURRENT not in GuidanceStatus.RETIRED


# ---------------------------------------------------------------------------
# D-C / D-D: still-forward guidance survives
# ---------------------------------------------------------------------------

def test_d_c_next_quarter_guidance_survives_the_full_year_actual():
    """The FY result says nothing about Q1 of the next year."""
    fy = _actual("2026-07-31", "8-K", "2026-09-02", "FY")
    resolution = resolve_current_actual_state([fy], as_of=TODAY)
    result = retire_realized_guidance(
        [FY2026_GUIDANCE, Q1_FY2027_GUIDANCE], resolution)

    assert [g.fiscal_period for g in result.current] == ["Q1 FY2027"]
    assert [g.fiscal_period for g in result.realized] == ["FY"]


def test_d_d_next_year_guidance_survives():
    fy = _actual("2026-07-31", "8-K", "2026-09-02", "FY")
    resolution = resolve_current_actual_state([fy], as_of=TODAY)
    result = retire_realized_guidance(
        [FY2026_GUIDANCE, FY2027_GUIDANCE], resolution)
    assert [g.fiscal_period for g in result.current] == ["FY2027"]


def test_d_e_no_realized_statement_appears_among_current_guidance():
    """§2.E: current coverage must not contain a completed period."""
    fy = _actual("2026-07-31", "8-K", "2026-09-02", "FY")
    resolution = resolve_current_actual_state([fy], as_of=TODAY)
    result = retire_realized_guidance(
        [FY2026_GUIDANCE, Q1_FY2027_GUIDANCE, FY2027_GUIDANCE], resolution)

    for statement in result.current:
        assert statement.status == GuidanceStatus.CURRENT
    realized_periods = {g.fiscal_period for g in result.realized}
    current_periods = {g.fiscal_period for g in result.current}
    assert not (realized_periods & current_periods), (
        "a period appears as both realized and current")


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------

def test_a_partially_reported_period_still_retires_its_guidance():
    """The number is in even if the statement set is not complete.

    Completeness decides whether a source can carry the whole STATE. It does
    not decide whether the guided figure has been reported -- and once it has,
    the forecast is measurable and therefore no longer a forecast.
    """
    partial = _actual("2026-07-31", "8-K", "2026-09-02", "FY",
                      completeness=StatementCompleteness.PARTIAL)
    resolution = resolve_current_actual_state([partial], as_of=TODAY)
    # The state itself does not advance on a PARTIAL source...
    assert resolution.period_end != "2026-07-31"
    # ...but the guidance for that period is still realized, because the
    # actual result exists. These are separate questions.
    result = retire_realized_guidance([FY2026_GUIDANCE], resolution)
    assert len(result.realized) + len(result.current) == 1


def test_guidance_for_an_unknown_period_label_stays_current():
    """Fail open here, deliberately: retiring a forecast we cannot match to a
    reported period would silently drop a live outlook."""
    fy = _actual("2026-07-31", "8-K", "2026-09-02", "FY")
    resolution = resolve_current_actual_state([fy], as_of=TODAY)
    odd = _Guidance("revenue", "the next several years")
    result = retire_realized_guidance([odd], resolution)
    assert result.current == [odd]


def test_no_guidance_at_all_is_not_an_error():
    fy = _actual("2026-07-31", "8-K", "2026-09-02", "FY")
    resolution = resolve_current_actual_state([fy], as_of=TODAY)
    result = retire_realized_guidance([], resolution)
    assert result.current == [] and result.realized == []


def test_the_superseded_release_also_counts_as_reporting_the_period():
    """§14: an 8-K and a later 10-K are one period with two sources.

    Guidance for that period is realized whichever of them is primary.
    """
    release = _actual("2026-07-31", "8-K", "2026-09-02", "FY")
    filing = _actual("2026-07-31", "10-K", "2026-10-15", "FY")
    resolution = resolve_current_actual_state([release, filing], as_of=TODAY)
    assert resolution.selected_primary_source.form == "10-K"
    result = retire_realized_guidance([FY2026_GUIDANCE], resolution)
    assert len(result.realized) == 1
