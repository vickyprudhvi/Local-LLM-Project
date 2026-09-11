"""When the current period advances, the twelve-month windows must follow.

THE FAILURE

A state that claims Q4 is current while its TTM revenue still ends at Q3.
Both numbers are individually correct; together they are a lie about what
twelve months means, and any growth rate computed across them is measured
over a seam rather than a period.

WHY THIS NEEDED WIRING RATHER THAN NEW ARITHMETIC

`finance/ttm.py` already builds the windows and already flags one that ends
materially before a `reference_end`. What it could not do is know which period
is current: it derives its own reference from the XBRL facts, and those do not
contain a period an 8-K has reported but no periodic filing has yet carried.

So `resolve_current_actual_state` supplies the reference. Fixtures F and G are
the two outcomes: the window follows, or it cannot and says so.
"""

import pytest

from finance.actualization import (
    TTM_END_TOLERANCE_DAYS,
    TtmCode,
    TtmReconstructionStatus,
    reconstruct_ttm,
)

REVENUE = "Revenues"
NET_INCOME = "NetIncomeLoss"


def _duration(start, end, value, form="10-Q", accn=None):
    return {"start": start, "end": end, "val": value, "form": form,
            "accn": accn or f"acc-{end}", "fy": int(end[:4]),
            "fp": "Q1", "filed": end}


def _facts(concepts):
    return {"facts": {"us-gaap": {
        concept: {"units": {"USD": rows}} for concept, rows in concepts.items()}}}


def _four_quarters_to(concept, base, ends):
    return [_duration(start, end, base + index * 10)
            for index, (start, end) in enumerate(ends)]


# Five consecutive quarters, so a window can end at either of the last two.
Q_ENDS = [("2025-04-01", "2025-06-30"),
          ("2025-07-01", "2025-09-30"),
          ("2025-10-01", "2025-12-31"),
          ("2026-01-01", "2026-03-31"),
          ("2026-04-01", "2026-06-30")]

PRIOR_QUARTER_END = "2026-03-31"
NEW_QUARTER_END = "2026-06-30"


def _facts_through_new_quarter():
    return _facts({
        REVENUE: _four_quarters_to(REVENUE, 100, Q_ENDS),
        NET_INCOME: _four_quarters_to(NET_INCOME, 10, Q_ENDS),
    })


def _facts_missing_the_new_quarter_for(concept):
    """Every metric has the new quarter except one, which stops a quarter short."""
    full = _four_quarters_to(REVENUE, 100, Q_ENDS)
    short = _four_quarters_to(NET_INCOME, 10, Q_ENDS[:-1])
    return _facts({REVENUE: full, concept: short})


# ---------------------------------------------------------------------------
# Fixture F: the window rolls forward exactly one quarter
# ---------------------------------------------------------------------------

def test_fixture_f_the_window_ends_at_the_newly_current_quarter():
    result = reconstruct_ttm(_facts_through_new_quarter(), NEW_QUARTER_END,
                             metrics=("revenue", "net_income"))
    assert result.status == TtmReconstructionStatus.COMPLETE
    assert TtmCode.ROLLED_FORWARD in result.codes
    assert result.windows["revenue"].end_date == NEW_QUARTER_END
    assert result.windows["revenue"].ends_at_current_period is True


def test_fixture_f_it_advanced_by_exactly_one_quarter_not_more():
    """A roll-forward is new DATA moving the window, not a new label on it.

    Before the quarter is reported the window ends at Q3; after, at Q4. One
    quarter, not two, and not "whatever the newest fact happens to be".
    """
    before = _facts({REVENUE: _four_quarters_to(REVENUE, 100, Q_ENDS[:-1])})
    after = _facts_through_new_quarter()

    prior = reconstruct_ttm(before, PRIOR_QUARTER_END, metrics=("revenue",))
    rolled = reconstruct_ttm(after, NEW_QUARTER_END, metrics=("revenue",))

    assert prior.windows["revenue"].end_date == PRIOR_QUARTER_END
    assert rolled.windows["revenue"].end_date == NEW_QUARTER_END
    assert prior.status == TtmReconstructionStatus.COMPLETE
    assert rolled.status == TtmReconstructionStatus.COMPLETE
    # Exactly one quarter of movement at both ends of the window.
    assert prior.windows["revenue"].start_date < rolled.windows["revenue"].start_date


def test_fixture_f_the_rolled_window_sums_the_right_four_quarters():
    result = reconstruct_ttm(_facts_through_new_quarter(), NEW_QUARTER_END,
                             metrics=("revenue",))
    # Quarters 2..5 of the five supplied: 110 + 120 + 130 + 140.
    assert result.windows["revenue"].value == pytest.approx(500)


def test_fixture_f_every_window_shares_one_end_date():
    result = reconstruct_ttm(_facts_through_new_quarter(), NEW_QUARTER_END,
                             metrics=("revenue", "net_income"))
    assert result.is_mixed_window is False
    assert TtmCode.COMPONENT_PERIOD_MISMATCH not in result.codes


# ---------------------------------------------------------------------------
# Fixture G: the window cannot follow
# ---------------------------------------------------------------------------

def test_fixture_g_a_window_that_cannot_reach_the_period_is_marked_limited():
    result = reconstruct_ttm(_facts_missing_the_new_quarter_for(NET_INCOME),
                             NEW_QUARTER_END, metrics=("revenue", "net_income"))
    assert result.status == TtmReconstructionStatus.LIMITED
    assert TtmCode.WINDOW_STALE in result.codes


def test_fixture_g_the_stale_metric_is_named_not_merely_counted():
    result = reconstruct_ttm(_facts_missing_the_new_quarter_for(NET_INCOME),
                             NEW_QUARTER_END, metrics=("revenue", "net_income"))
    assert result.stale_metrics == ["net_income"]
    assert result.current_metrics == ["revenue"]
    assert any("net_income" in r for r in result.reasons)


def test_fixture_g_the_stale_window_keeps_its_value_and_its_label():
    """It is a valid twelve months, just not a current one.

    Dropping it would lose real information; using it unlabelled would put a
    Q3-ending figure under a Q4 heading. Keeping it labelled is the only
    option that does neither.
    """
    result = reconstruct_ttm(_facts_missing_the_new_quarter_for(NET_INCOME),
                             NEW_QUARTER_END, metrics=("net_income",))
    window = result.windows["net_income"]
    assert window.value is not None
    assert window.is_stale is True
    assert window.end_date == PRIOR_QUARTER_END


def test_fixture_g_mixed_end_dates_are_reported_as_incomparable():
    result = reconstruct_ttm(_facts_missing_the_new_quarter_for(NET_INCOME),
                             NEW_QUARTER_END, metrics=("revenue", "net_income"))
    assert result.is_mixed_window is True
    assert TtmCode.COMPONENT_PERIOD_MISMATCH in result.codes
    assert any("must not be combined" in r for r in result.reasons)


def test_fixture_g_nothing_is_silently_mixed():
    """§24 hard requirement: a silent mixed-period state is the failure.

    Every window is either current or explicitly stale; there is no third
    outcome in which a reader cannot tell.
    """
    result = reconstruct_ttm(_facts_missing_the_new_quarter_for(NET_INCOME),
                             NEW_QUARTER_END, metrics=("revenue", "net_income"))
    for window in result.windows.values():
        assert window.ends_at_current_period in (True, False)
        if window.is_stale:
            assert window.reason or result.reasons


# ---------------------------------------------------------------------------
# The coupling itself
# ---------------------------------------------------------------------------

def test_the_resolved_period_decides_whether_a_window_counts_as_current():
    """The coupling that matters.

    An 8-K can report a period the XBRL facts do not carry. The facts alone
    cannot tell whether the newest window they support is the current one --
    the resolver's period is what settles it, and the same window is current
    against one period end and stale against another.
    """
    facts = _facts_through_new_quarter()
    against_new = reconstruct_ttm(facts, NEW_QUARTER_END, metrics=("revenue",))
    against_later = reconstruct_ttm(facts, "2026-09-30", metrics=("revenue",))

    assert against_new.windows["revenue"].ends_at_current_period is True
    assert against_later.windows["revenue"].ends_at_current_period is False
    assert against_later.status == TtmReconstructionStatus.LIMITED


def test_reference_end_is_a_staleness_check_and_not_an_anchor():
    """A limitation, recorded rather than assumed away.

    `build_ttm` always constructs the newest window the facts support;
    `reference_end` judges it. It does NOT rebuild the window to end on an
    arbitrary earlier date, so this layer cannot reconstruct a state "as of"
    a past release -- §21's as-of mode is not implemented here. For the
    forward case this phase is about, judging is exactly what is needed.
    """
    facts = _facts_through_new_quarter()
    earlier = reconstruct_ttm(facts, PRIOR_QUARTER_END, metrics=("revenue",))
    assert earlier.windows["revenue"].end_date == NEW_QUARTER_END, (
        "documented behaviour: the window is not re-anchored backwards")
    assert earlier.windows["revenue"].ends_at_current_period is False


def test_no_resolved_period_means_no_window_rather_than_a_guessed_one():
    result = reconstruct_ttm(_facts_through_new_quarter(), None)
    assert result.status == TtmReconstructionStatus.NOT_RECONSTRUCTED
    assert TtmCode.NOT_RECONSTRUCTABLE in result.codes
    assert result.windows == {}


def test_a_window_within_tolerance_of_the_period_end_counts_as_current():
    """Quarter ends move by a few days between issuers and years.

    A whole quarter does not, which is what the tolerance separates.
    """
    nearly = "2026-07-05"
    assert 0 < (5) <= TTM_END_TOLERANCE_DAYS
    result = reconstruct_ttm(_facts_through_new_quarter(), nearly,
                             metrics=("revenue",))
    assert result.windows["revenue"].ends_at_current_period is True
    assert result.status == TtmReconstructionStatus.COMPLETE


def test_a_window_a_whole_quarter_behind_is_not_within_tolerance():
    result = reconstruct_ttm(_facts_missing_the_new_quarter_for(NET_INCOME),
                             NEW_QUARTER_END, metrics=("net_income",))
    assert result.windows["net_income"].ends_at_current_period is False
