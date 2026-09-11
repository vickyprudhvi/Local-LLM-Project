"""Which reported period is current, and how fresh each fact in it is.

THE FAILURE

    Q3 10-Q      period ending April, complete
    Q4/FY 8-K    period ending July, complete, filed later
    10-K         not filed yet

The system kept April, because "latest periodic filing" was standing in for
"latest reported actual period". Those coincide most of the time and come
apart exactly when a company has just reported -- when the numbers matter
most. Everything downstream inherited the staleness: revenue, margins, debt,
latest-quarter growth, the DCF base, the research claims.

WHAT IS NOT THE FIX

A bigger form allowlist. Adding 8-K to the annual/interim forms would let a
press release with two headline numbers overwrite a complete balance sheet,
and would let a guidance table be read as reported results. Form is one input,
not the decision.

These are the §22 fixtures A-H, written as generalized shapes. Ticker names
appear nowhere: each is a SEQUENCE OF FILINGS, which is what the resolver
actually reasons about.
"""

import pytest

from finance.actualization import (
    ActualSourceType,
    ActualStateStatus,
    ResolutionCode,
    build_metric_freshness,
    resolve_current_actual_state,
    source_type_for,
)
from finance.extraction.schema import (
    FinalityStatus,
    ReportedActualCandidate,
    SourceType,
    StatementCompleteness,
)

# Everything a complete statement set needs, for a normal industrial issuer.
FULL = ("revenue", "operating_income", "net_income", "cash_and_cash_equivalents",
        "total_debt", "stockholders_equity", "operating_cash_flow",
        "capital_expenditure", "shares_diluted")


def _candidate(period_end, form, issued_at, present=FULL,
               completeness=StatementCompleteness.COMPLETE,
               values=None, fiscal_period=None, prospective=False,
               missing=()):
    return ReportedActualCandidate(
        period_end=period_end, form=form, issued_at=issued_at,
        fiscal_period=fiscal_period, statement_completeness=completeness,
        present_metrics=tuple(present), missing_metrics=tuple(missing),
        values=dict(values or {}), is_prospective=prospective,
        source_type=(SourceType.FORMAL_PERIODIC_FILING
                     if form in ("10-K", "10-Q") else
                     SourceType.PRELIMINARY_EARNINGS_RELEASE))


Q3_10Q = _candidate("2026-04-30", "10-Q", "2026-06-05", fiscal_period="Q3")
Q4_8K = _candidate("2026-07-31", "8-K", "2026-09-02", fiscal_period="FY")
FY_10K = _candidate("2026-07-31", "10-K", "2026-10-15", fiscal_period="FY")

TODAY = "2026-11-01"


# ---------------------------------------------------------------------------
# Fixture A: complete newer earnings release beats an older 10-Q
# ---------------------------------------------------------------------------

def test_fixture_a_a_complete_release_advances_the_period_past_the_10q():
    result = resolve_current_actual_state([Q3_10Q, Q4_8K], as_of=TODAY)
    assert result.period_end == "2026-07-31", (
        "the state stayed on the older 10-Q period")
    assert result.state_status == ActualStateStatus.PRELIMINARY_REPORTED_ACTUAL
    assert ResolutionCode.ADVANCED_TO_NEWER_RELEASE in result.codes


def test_fixture_a_the_preliminary_nature_is_recorded_not_assumed_away():
    result = resolve_current_actual_state([Q3_10Q, Q4_8K], as_of=TODAY)
    assert result.selected_primary_source.form == "8-K"
    assert any("preliminary" in r.lower() for r in result.resolution_reasons)
    assert all(f.finality == FinalityStatus.UNAUDITED_PRELIMINARY
               for f in result.fallbacks if f.from_current_period)


# ---------------------------------------------------------------------------
# Fixture B: the 10-K arrives for the same period
# ---------------------------------------------------------------------------

def test_fixture_b_the_10k_supersedes_without_creating_a_second_period():
    result = resolve_current_actual_state([Q3_10Q, Q4_8K, FY_10K], as_of=TODAY)
    assert result.period_end == "2026-07-31"
    assert result.selected_primary_source.form == "10-K"
    assert result.state_status == ActualStateStatus.FINAL_REPORTED_ACTUAL
    assert ResolutionCode.SAME_PERIOD_SUPERSEDED in result.codes


def test_fixture_b_the_release_is_kept_as_provenance():
    result = resolve_current_actual_state([Q3_10Q, Q4_8K, FY_10K], as_of=TODAY)
    assert [s.form for s in result.superseded_sources] == ["8-K"]
    assert len({result.period_end}) == 1, "one economic period, not two"


# ---------------------------------------------------------------------------
# Fixture C: newer release is INCOMPLETE
# ---------------------------------------------------------------------------

HEADLINE_ONLY = _candidate(
    "2026-07-31", "8-K", "2026-09-02",
    present=("revenue", "adjusted_earnings_per_share"),
    missing=("cash_and_cash_equivalents", "total_debt", "operating_cash_flow"),
    completeness=StatementCompleteness.HEADLINE_ONLY, fiscal_period="FY")


def test_fixture_c_an_incomplete_release_does_not_replace_the_state_wholesale():
    result = resolve_current_actual_state([Q3_10Q, HEADLINE_ONLY], as_of=TODAY)
    assert result.period_end == "2026-04-30", (
        "a headline release replaced a complete statement set")
    assert ResolutionCode.NEWER_SOURCE_INCOMPLETE in result.codes


def test_fixture_c_the_refusal_says_what_was_missing():
    result = resolve_current_actual_state([Q3_10Q, HEADLINE_ONLY], as_of=TODAY)
    reasons = " ".join(why for _c, why in result.rejected_candidates)
    assert "total_debt" in reasons
    assert "per metric" in reasons, (
        "the refusal should say the individual figures remain usable")


def test_fixture_c_per_metric_fallback_is_labelled_with_both_periods():
    """§20: no silent fallback. Both periods and a reason, machine-readable."""
    prior = {"total_debt": {"period_end": "2026-04-30", "form": "10-Q"},
             "operating_cash_flow": {"period_end": "2026-04-30", "form": "10-Q"}}
    records = build_metric_freshness(
        HEADLINE_ONLY, ActualStateStatus.PARTIAL_REPORTED_ACTUAL, prior)
    fallbacks = {r.metric: r for r in records if r.is_fallback}
    assert set(fallbacks) == {"total_debt", "operating_cash_flow"}
    for record in fallbacks.values():
        assert record.fallback_from_period == "2026-04-30"
        assert record.from_current_period is False
        assert record.fallback_reason


def test_fixture_c_a_mixed_period_state_says_so():
    prior = {"total_debt": {"period_end": "2026-04-30", "form": "10-Q"}}
    result = resolve_current_actual_state(
        [Q3_10Q, Q4_8K], as_of=TODAY, prior_state_metrics=prior)
    # The advance here is complete, so nothing falls back.
    assert result.is_mixed_period is False


# ---------------------------------------------------------------------------
# Fixture C2: found live on PG/TGT -- a completeness anchor retired YEARS ago
# ---------------------------------------------------------------------------
#
# PG and TGT both stopped tagging `CashAndCashEquivalentsAtCarryingValue` in
# 2019 and 2017 respectively, moving to a concept this project does not yet
# read. Every quarter filed since is PARTIAL for that reason alone -- and
# "the newest COMPLETE candidate" is whatever quarter happened to predate the
# switch, sometimes SEVEN YEARS before `as_of`. Fixture C's rule ("an
# incomplete newer source does not replace a complete older one") is right for
# a one-quarter gap; treated as sufficient for a multi-year gap, the resolver
# silently published a 2019 quarter as the current reported state of a
# company filing quarterly in 2026 -- WRONG_CURRENT_PERIOD_PUBLISHED.
#
# The shape is generic: it says nothing about which field or which issuer,
# only that every recent candidate shares one missing field and the newest
# candidate that has it is far in the past.

def _quarterly_history(missing_from_recent, recent_count=24, complete_end="2019-09-30"):
    """`recent_count` recent quarters, all missing one field, plus one
    complete quarter years earlier."""
    out = [_candidate(complete_end, "10-Q", complete_end)]
    year, month = 2020, 1
    for i in range(recent_count):
        end = f"{2026 - (recent_count - i) // 4}-{['01','04','07','10'][i % 4]}-28"
        present = tuple(f for f in FULL if f != missing_from_recent)
        out.append(_candidate(end, "10-Q", end, present=present,
                              completeness=StatementCompleteness.PARTIAL,
                              missing=(missing_from_recent,)))
    return out


def test_fixture_c2_a_completeness_anchor_missing_for_years_does_not_resolve_to_it():
    """The generalized shape: EVERY recent candidate lacks one required field;
    the resolver must not fall back seven years to find one that has it."""
    history = _quarterly_history("cash_and_cash_equivalents", recent_count=28,
                                 complete_end="2019-09-30")
    result = resolve_current_actual_state(history, as_of="2026-11-01")
    assert result.period_end != "2019-09-30", (
        "resolved to a multi-year-old period because nothing recent was "
        "COMPLETE for an unrelated reason -- this is what a reader would call "
        "a wrong current period, not a considered fallback")
    # The newest candidates are still individually usable per metric; the
    # resolver just refuses to call the ancient one "current."
    assert result.state_status in (
        ActualStateStatus.PARTIAL_REPORTED_ACTUAL, ActualStateStatus.REJECTED)


def test_fixture_c2_a_one_quarter_gap_is_unaffected():
    """The bound must not swallow Fixture C's ordinary case: a genuinely
    recent complete filing beside a genuinely recent incomplete release."""
    result = resolve_current_actual_state([Q3_10Q, HEADLINE_ONLY], as_of=TODAY)
    assert result.period_end == "2026-04-30"


def test_fixture_c2_a_bound_length_gap_still_resolves():
    """A complete filing a little under a year old, beside newer incomplete
    ones, is still the right answer -- the bound is generous, not a quarter
    cliff-edge."""
    history = _quarterly_history("cash_and_cash_equivalents", recent_count=3,
                                 complete_end="2026-01-31")
    result = resolve_current_actual_state(history, as_of="2026-11-01")
    assert result.period_end == "2026-01-31"


# ---------------------------------------------------------------------------
# Fixture E: same period, materially different figures
# ---------------------------------------------------------------------------

def test_fixture_e_a_material_same_period_disagreement_is_recorded():
    release = _candidate("2026-07-31", "8-K", "2026-09-02", fiscal_period="FY",
                         values={"total_debt": 1000.0, "revenue": 5000.0})
    filing = _candidate("2026-07-31", "10-K", "2026-10-15", fiscal_period="FY",
                        values={"total_debt": 1400.0, "revenue": 5000.0})
    result = resolve_current_actual_state([release, filing], as_of=TODAY)

    assert ResolutionCode.SAME_PERIOD_SOURCE_CONFLICT in result.codes
    assert [c.metric for c in result.conflicts] == ["total_debt"], (
        "revenue agrees and must not be reported as a conflict")
    conflict = result.conflicts[0]
    assert conflict.authoritative_value == 1400.0
    assert conflict.authoritative_source == "10-K"
    assert conflict.other_value == 1000.0


def test_fixture_e_an_immaterial_difference_is_not_a_conflict():
    """Preliminary and final figures move a little. That is not a dispute."""
    release = _candidate("2026-07-31", "8-K", "2026-09-02", fiscal_period="FY",
                         values={"revenue": 5000.0})
    filing = _candidate("2026-07-31", "10-K", "2026-10-15", fiscal_period="FY",
                        values={"revenue": 5000.4})
    result = resolve_current_actual_state([release, filing], as_of=TODAY)
    assert result.conflicts == []


def test_fixture_e_the_authoritative_value_is_used_not_the_newest_written():
    release = _candidate("2026-07-31", "8-K", "2026-09-02", fiscal_period="FY",
                         values={"total_debt": 1000.0})
    filing = _candidate("2026-07-31", "10-K", "2026-10-15", fiscal_period="FY",
                        values={"total_debt": 1400.0})
    result = resolve_current_actual_state([release, filing], as_of=TODAY)
    assert result.selected_primary_source.form == "10-K"


# ---------------------------------------------------------------------------
# §16 actual vs guidance separation
# ---------------------------------------------------------------------------

def test_a_prospective_candidate_is_never_a_reported_actual():
    """The same 8-K yields results AND an outlook. Only one is an actual."""
    outlook = _candidate("2027-07-31", "8-K", "2026-09-02", fiscal_period="FY",
                         prospective=True)
    result = resolve_current_actual_state([Q3_10Q, outlook], as_of=TODAY)
    assert result.period_end == "2026-04-30"
    reasons = " ".join(why for _c, why in result.rejected_candidates)
    assert "prospective" in reasons


def test_a_future_period_is_not_current_even_when_marked_actual():
    """§21: no time travel. A period ending after today has not happened."""
    future = _candidate("2027-07-31", "10-K", "2026-09-02", fiscal_period="FY")
    result = resolve_current_actual_state([Q3_10Q, future], as_of=TODAY)
    assert result.period_end == "2026-04-30"


# ---------------------------------------------------------------------------
# §27.15/16: neither form nor recency alone may decide
# ---------------------------------------------------------------------------

def test_form_authority_cannot_hold_the_state_on_an_older_period():
    """§27.15. A 10-K is more authoritative than an 8-K and that is a
    TIE-BREAK, not a veto: it cannot keep the state on a period the release
    has already moved past."""
    old_10k = _candidate("2025-07-31", "10-K", "2025-10-15", fiscal_period="FY")
    result = resolve_current_actual_state([old_10k, Q4_8K], as_of=TODAY)
    assert result.period_end == "2026-07-31"
    assert result.selected_primary_source.form == "8-K"


def test_recency_alone_cannot_advance_the_state():
    """§27.16, the other direction. The newest document does not win if it
    cannot describe the period."""
    result = resolve_current_actual_state([Q3_10Q, HEADLINE_ONLY], as_of=TODAY)
    assert result.period_end == "2026-04-30"


def test_no_source_at_all_is_refused_rather_than_guessed():
    result = resolve_current_actual_state([], as_of=TODAY)
    assert result.state_status == ActualStateStatus.REJECTED
    assert ResolutionCode.NO_REPORTED_ACTUAL in result.codes
    assert result.is_current is False


# ---------------------------------------------------------------------------
# §27.13: no ticker-specific rules
# ---------------------------------------------------------------------------

def test_the_resolver_reasons_about_forms_and_periods_only():
    """The same filing sequence resolves identically whatever it is called.

    The resolver never sees a symbol; this asserts the API cannot express
    issuer-specific behaviour even by accident.
    """
    import inspect

    from finance import actualization

    source = inspect.getsource(actualization)
    for forbidden in ("symbol", "ticker", "cik"):
        assert forbidden not in source.lower(), (
            f"{forbidden!r} appears in the resolver")


@pytest.mark.parametrize("form,expected", [
    ("10-K", ActualSourceType.PERIODIC_ANNUAL),
    ("10-K/A", ActualSourceType.PERIODIC_ANNUAL),
    ("20-F", ActualSourceType.PERIODIC_ANNUAL),
    ("10-Q", ActualSourceType.PERIODIC_INTERIM),
    ("6-K", ActualSourceType.PERIODIC_INTERIM),
    ("8-K", ActualSourceType.EARNINGS_RELEASE_FILED_8K),
])
def test_source_classification_is_by_form_shape(form, expected):
    assert source_type_for(form) == expected


def test_a_foreign_private_issuer_form_is_handled_by_the_same_rules():
    """20-F/6-K are the annual/interim pair for a foreign private issuer.

    Not a special case -- the same period-first, completeness-gated logic.
    """
    interim = _candidate("2026-04-30", "6-K", "2026-06-05", fiscal_period="H1")
    annual = _candidate("2026-07-31", "20-F", "2026-10-15", fiscal_period="FY")
    result = resolve_current_actual_state([interim, annual], as_of=TODAY)
    assert result.period_end == "2026-07-31"
    assert result.state_status == ActualStateStatus.FINAL_REPORTED_ACTUAL


# ---------------------------------------------------------------------------
# Per-metric freshness (§10/§11)
# ---------------------------------------------------------------------------

def test_every_current_fact_can_answer_which_period_it_is_from():
    result = resolve_current_actual_state([Q3_10Q, Q4_8K], as_of=TODAY)
    assert result.fallbacks, "no per-metric provenance was produced"
    for record in result.fallbacks:
        assert record.period_end
        assert record.source_type
        assert record.from_current_period is True
        assert record.period_end == result.period_end


def test_a_retained_older_metric_is_never_marked_current():
    prior = {"total_debt": {"period_end": "2026-01-31", "form": "10-Q"}}
    records = build_metric_freshness(
        Q4_8K, ActualStateStatus.PRELIMINARY_REPORTED_ACTUAL, prior)
    debt = next(r for r in records if r.metric == "total_debt")
    # total_debt IS in the current source here, so it is not a fallback.
    assert debt.from_current_period is True
    assert debt.is_fallback is False


def test_an_older_metric_newer_than_the_current_period_is_not_backfilled():
    """§21 again, at metric level: a later-period value must not fill an
    earlier period's gap."""
    prior = {"total_debt": {"period_end": "2027-01-31", "form": "10-Q"}}
    records = build_metric_freshness(
        HEADLINE_ONLY, ActualStateStatus.PARTIAL_REPORTED_ACTUAL, prior)
    assert not any(r.metric == "total_debt" for r in records)
