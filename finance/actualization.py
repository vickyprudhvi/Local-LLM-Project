"""Which reported period is CURRENT, from which source, and how fresh each fact is.

THE PROBLEM

    Q3 10-Q          period ending April, complete
    Q4/FY 8-K        period ending July, complete, filed later
    10-K             not filed yet

The system kept April. Everything downstream inherited it: revenue, margins,
debt, latest-quarter growth, the DCF base, the research claims. Each was
individually correct and collectively three months stale.

The mistake is treating "the latest periodic filing" as a synonym for "the
latest reported actual period". They coincide most of the time, which is why
this survives so long, and they come apart exactly when a company has just
reported -- the moment the numbers matter most.

WHAT THIS IS NOT

It is not a bigger form allowlist. Adding "8-K" to `ANNUAL_FORMS` would let a
press release carrying two headline numbers overwrite a complete balance
sheet, and would let a guidance table be read as reported results. Form is one
input to the decision and not the decision.

THE DECISION

    period end        the question is about a PERIOD, so this leads
    actuality         reported results, never an outlook
    completeness      a source that cannot describe the whole period does not
                      carry the whole state
    authority         on the SAME period, a periodic filing outranks a release
    provenance        every accepted fact keeps the period and source it came
                      from, so "is this from the current period?" is answerable

ONE PERIOD, MANY SOURCES

An 8-K for FY2026 followed by a 10-K for FY2026 is ONE economic period with
two sources, not two current periods. The 10-K supersedes as the authority and
the release stays in the provenance history. Where they disagree materially
the disagreement is recorded rather than resolved by overwrite.

NO SILENT FALLBACK

Where the current period cannot supply a metric and an older value is
retained, the substitution is recorded with both periods and a reason. A state
assembled from two periods says so.
"""

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from finance.extraction.document_resolver import (
    REQUIRED_FOR_COMPLETE,
    classify_completeness,
)
from finance.guidance import GuidanceStatus
from finance.extraction.schema import (
    FinalityStatus,
    ReportedActualCandidate,
    SourceType,
    StatementCompleteness,
)


# ---------------------------------------------------------------------------
# Vocabulary (§3/§4)
# ---------------------------------------------------------------------------

class ActualSourceType:
    """WHERE a reported actual came from.

    A refinement of `schema.SourceType`, which answers the coarser question
    "periodic filing or press release?". Both are kept: the coarse one already
    drives guidance authority, and re-labelling it would break that.
    """

    PERIODIC_ANNUAL = "PERIODIC_ANNUAL"
    PERIODIC_INTERIM = "PERIODIC_INTERIM"
    EARNINGS_RELEASE_FILED_8K = "EARNINGS_RELEASE_FILED_8K"
    EARNINGS_RELEASE_OTHER = "EARNINGS_RELEASE_OTHER"
    STRUCTURED_XBRL = "STRUCTURED_XBRL"
    OTHER_REPORTED_ACTUAL = "OTHER_REPORTED_ACTUAL"

    ALL = (PERIODIC_ANNUAL, PERIODIC_INTERIM, EARNINGS_RELEASE_FILED_8K,
           EARNINGS_RELEASE_OTHER, STRUCTURED_XBRL, OTHER_REPORTED_ACTUAL)

    # Authority for the SAME period. Lower is stronger. Deliberately used only
    # as a tie-break: a newer period beats a stronger source, because the
    # question being asked is which period is current.
    AUTHORITY = {
        PERIODIC_ANNUAL: 0,
        PERIODIC_INTERIM: 1,
        STRUCTURED_XBRL: 2,
        EARNINGS_RELEASE_FILED_8K: 3,
        EARNINGS_RELEASE_OTHER: 4,
        OTHER_REPORTED_ACTUAL: 5,
    }


class ActualStateStatus:
    """How settled the current reported state is."""

    FINAL_REPORTED_ACTUAL = "FINAL_REPORTED_ACTUAL"
    PRELIMINARY_REPORTED_ACTUAL = "PRELIMINARY_REPORTED_ACTUAL"
    PARTIAL_REPORTED_ACTUAL = "PARTIAL_REPORTED_ACTUAL"
    HISTORICAL = "HISTORICAL"
    REJECTED = "REJECTED"

    ALL = (FINAL_REPORTED_ACTUAL, PRELIMINARY_REPORTED_ACTUAL,
           PARTIAL_REPORTED_ACTUAL, HISTORICAL, REJECTED)

    # States in which the period is CURRENT. `PARTIAL` is current-with-caveats:
    # the period advanced but not every metric came with it.
    CURRENT = (FINAL_REPORTED_ACTUAL, PRELIMINARY_REPORTED_ACTUAL,
               PARTIAL_REPORTED_ACTUAL)


class ResolutionCode:
    """Named outcomes, so a benchmark counts classes and not sentences."""

    ADVANCED_TO_NEWER_RELEASE = "ADVANCED_TO_NEWER_RELEASE"
    NEWEST_PERIODIC_FILING = "NEWEST_PERIODIC_FILING"
    NEWER_SOURCE_INCOMPLETE = "NEWER_SOURCE_INCOMPLETE"
    SAME_PERIOD_SUPERSEDED = "SAME_PERIOD_SUPERSEDED"
    SAME_PERIOD_SOURCE_CONFLICT = "SAME_PERIOD_SOURCE_CONFLICT"
    NO_REPORTED_ACTUAL = "NO_REPORTED_ACTUAL"
    PARTIAL_ADVANCE_WITH_FALLBACK = "PARTIAL_ADVANCE_WITH_FALLBACK"
    COMPLETE_CANDIDATE_TOO_STALE = "COMPLETE_CANDIDATE_TOO_STALE"


def source_type_for(form: Optional[str], has_xbrl: bool = False) -> str:
    """Classify a form WITHOUT branching on issuer.

    Form shape only: a 10-K is annual, a 10-Q is interim, an 8-K carrying
    statements is a filed earnings release.
    """
    text = (form or "").strip().upper().rstrip("/A").strip()
    if text in ("10-K", "20-F", "40-F", "11-K"):
        return ActualSourceType.PERIODIC_ANNUAL
    if text in ("10-Q", "6-K"):
        return ActualSourceType.PERIODIC_INTERIM
    if text.startswith("8-K"):
        return ActualSourceType.EARNINGS_RELEASE_FILED_8K
    if has_xbrl:
        return ActualSourceType.STRUCTURED_XBRL
    if text:
        return ActualSourceType.OTHER_REPORTED_ACTUAL
    return ActualSourceType.OTHER_REPORTED_ACTUAL


def finality_for(source_type: str) -> str:
    if source_type in (ActualSourceType.PERIODIC_ANNUAL,
                       ActualSourceType.PERIODIC_INTERIM):
        return FinalityStatus.AUDITED
    if source_type in (ActualSourceType.EARNINGS_RELEASE_FILED_8K,
                       ActualSourceType.EARNINGS_RELEASE_OTHER):
        return FinalityStatus.UNAUDITED_PRELIMINARY
    return FinalityStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricFreshness:
    """Where ONE accepted fact came from, so staleness is answerable per metric."""

    metric: str
    period_end: Optional[str]
    issued_at: Optional[str]
    source_form: Optional[str]
    source_type: str
    actual_status: str
    finality: str
    from_current_period: bool
    fallback_from_period: Optional[str] = None
    fallback_reason: str = ""

    @property
    def is_fallback(self) -> bool:
        return self.fallback_from_period is not None

    def to_dict(self) -> dict:
        payload = {
            "metric": self.metric, "period_end": self.period_end,
            "issued_at": self.issued_at, "source_form": self.source_form,
            "source_type": self.source_type, "actual_status": self.actual_status,
            "finality": self.finality,
            "from_current_period": self.from_current_period,
        }
        if self.is_fallback:
            payload["fallback_from_period"] = self.fallback_from_period
            payload["fallback_to_period"] = self.period_end
            payload["fallback_reason"] = self.fallback_reason
        return payload


@dataclass(frozen=True)
class SourceConflict:
    """Two sources for one period disagreeing materially about one metric."""

    metric: str
    period_end: str
    authoritative_value: float
    authoritative_source: Optional[str]
    other_value: float
    other_source: Optional[str]
    relative_difference: float

    def to_dict(self) -> dict:
        return {"metric": self.metric, "period_end": self.period_end,
                "authoritative_value": self.authoritative_value,
                "authoritative_source": self.authoritative_source,
                "other_value": self.other_value,
                "other_source": self.other_source,
                "relative_difference": round(self.relative_difference, 6),
                "code": ResolutionCode.SAME_PERIOD_SOURCE_CONFLICT}


@dataclass
class ActualStateResolution:
    """The answer, with everything needed to check it."""

    selected_period: Optional[str] = None
    selected_primary_source: Optional[ReportedActualCandidate] = None
    state_status: str = ActualStateStatus.REJECTED
    statement_completeness: str = StatementCompleteness.NONE
    source_authority: Optional[int] = None
    issued_at: Optional[str] = None
    period_end: Optional[str] = None
    superseded_sources: List[ReportedActualCandidate] = field(default_factory=list)
    fallbacks: List[MetricFreshness] = field(default_factory=list)
    rejected_candidates: List[Tuple[ReportedActualCandidate, str]] = field(
        default_factory=list)
    conflicts: List[SourceConflict] = field(default_factory=list)
    resolution_reasons: List[str] = field(default_factory=list)
    codes: List[str] = field(default_factory=list)

    @property
    def is_current(self) -> bool:
        return self.state_status in ActualStateStatus.CURRENT

    @property
    def is_mixed_period(self) -> bool:
        """Does the state draw on more than one period? Never silently."""
        return any(f.is_fallback for f in self.fallbacks)

    def to_dict(self) -> dict:
        return {
            "selected_period": self.selected_period,
            "selected_primary_source": (self.selected_primary_source.to_dict()
                                        if self.selected_primary_source else None),
            "state_status": self.state_status,
            "statement_completeness": self.statement_completeness,
            "source_authority": self.source_authority,
            "issued_at": self.issued_at,
            "period_end": self.period_end,
            "superseded_sources": [s.to_dict() for s in self.superseded_sources],
            "fallbacks": [f.to_dict() for f in self.fallbacks],
            "rejected_candidates": [{"candidate": c.to_dict(), "reason": why}
                                    for c, why in self.rejected_candidates],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "resolution_reasons": list(self.resolution_reasons),
            "codes": list(self.codes),
            "is_mixed_period": self.is_mixed_period,
        }


# ---------------------------------------------------------------------------
# The resolver (§5)
# ---------------------------------------------------------------------------

# How far two sources for one period may differ before it is a conflict rather
# than a rounding difference. Preliminary and final figures routinely move a
# little; a material move is a different claim about the same period.
MATERIAL_DIFFERENCE = 0.01

# How far "the newest COMPLETE candidate" may lag the newest USABLE one
# before it is treated as too old to BE the current period, rather than as
# a considered one-quarter fallback.
#
# Found live: PG and TGT both retired `CashAndCashEquivalentsAtCarryingValue`
# for a combined cash-and-restricted-cash concept this project does not read
# (2019 and 2017 respectively). Every quarter filed since is PARTIAL for that
# one reason, so "the newest complete candidate" was a quarter seven to nine
# YEARS old -- and the resolver published it as the current reported state of
# a company that had filed quarterly ever since. That is not what §13's rule
# ("an incomplete newer source does not replace a complete older one") means:
# it protects a one-quarter gap, where the older filing is still informative
# about where the company stands today. A multi-year gap says the opposite --
# this project cannot currently establish completeness for ANY recent period
# of this issuer, which is a different, honest answer than naming an ancient
# one "current."
#
# 400 days (a year plus slack) matches the annual lookback already used
# elsewhere in this codebase (`period_facts.ANNUAL_DAYS`, `ttm.TTM_SPAN_DAYS`)
# rather than inventing a new number: a complete annual filing is still
# informative about a company's current state; nothing older is.
MAX_COMPLETE_CANDIDATE_LAG_DAYS = 400


def _authority(candidate: ReportedActualCandidate) -> int:
    return ActualSourceType.AUTHORITY.get(
        source_type_for(candidate.form), 9)


def _status_for(candidate: ReportedActualCandidate, completeness: str) -> str:
    if completeness == StatementCompleteness.COMPLETE:
        source_type = source_type_for(candidate.form)
        if source_type in (ActualSourceType.PERIODIC_ANNUAL,
                           ActualSourceType.PERIODIC_INTERIM):
            return ActualStateStatus.FINAL_REPORTED_ACTUAL
        return ActualStateStatus.PRELIMINARY_REPORTED_ACTUAL
    return ActualStateStatus.PARTIAL_REPORTED_ACTUAL


def resolve_current_actual_state(
        candidates: List[ReportedActualCandidate],
        as_of: Optional[str] = None,
        prior_state_metrics: Optional[Dict[str, dict]] = None
) -> ActualStateResolution:
    """The newest reported actual period, its source, and per-metric freshness.

    `prior_state_metrics` is the state being replaced, keyed by metric, each
    carrying at least `period_end`. It exists so a partial advance can record
    what it fell back to instead of silently keeping it.
    """
    as_of = as_of or datetime.date.today().isoformat()
    resolution = ActualStateResolution()

    usable = [c for c in candidates
              if c.period_end and c.period_end <= as_of and not c.is_prospective]
    for candidate in candidates:
        if candidate.is_prospective:
            # §16: an earnings release carries an outlook as well as results.
            # A forward figure is never a reported actual, whatever else is
            # true of the document it arrived in.
            resolution.rejected_candidates.append((
                candidate, "the figures are prospective, not reported results"))
        elif not candidate.period_end:
            resolution.rejected_candidates.append((
                candidate, "the source names no period end"))
        elif candidate.period_end > as_of:
            resolution.rejected_candidates.append((
                candidate, f"the period ends {candidate.period_end}, after {as_of}"))

    if not usable:
        resolution.state_status = ActualStateStatus.REJECTED
        resolution.codes.append(ResolutionCode.NO_REPORTED_ACTUAL)
        resolution.resolution_reasons.append(
            "No source reported a completed period on or before this date.")
        return resolution

    # -- the newest period any source can fully describe --------------------
    complete = [c for c in usable
                if c.statement_completeness == StatementCompleteness.COMPLETE]
    newest_period = max(c.period_end for c in usable)

    for candidate in usable:
        if candidate.period_end == newest_period \
                and candidate.statement_completeness != StatementCompleteness.COMPLETE:
            resolution.rejected_candidates.append((
                candidate,
                f"{candidate.form or 'this source'} reports {candidate.period_end} but is "
                f"{candidate.statement_completeness}: missing "
                f"{', '.join(candidate.missing_metrics) or 'required statements'}. It "
                "cannot carry the whole state; its individual figures remain "
                "available per metric."))

    if not complete:
        # §8: nothing can carry the whole state. The period does not advance
        # wholesale; per-metric fallback is the caller's controlled path.
        resolution.state_status = ActualStateStatus.PARTIAL_REPORTED_ACTUAL
        resolution.codes.append(ResolutionCode.NEWER_SOURCE_INCOMPLETE)
        resolution.resolution_reasons.append(
            "No source carried a complete statement set, so the state was not "
            "replaced wholesale.")
        return resolution

    newest_complete = max(c.period_end for c in complete)
    complete_lag = _days_between(newest_complete, newest_period)
    if complete_lag is not None and complete_lag > MAX_COMPLETE_CANDIDATE_LAG_DAYS:
        # The newest source that ever carried a complete statement set is not
        # a considered one-quarter fallback -- it is evidence that a required
        # field has been missing from EVERY recent source for a reason
        # unrelated to how current this issuer's filings are. Naming a
        # multi-year-old quarter "current" would be a wrong current period,
        # not a fallback; treated the same as no complete source at all.
        resolution.state_status = ActualStateStatus.PARTIAL_REPORTED_ACTUAL
        resolution.codes.append(ResolutionCode.COMPLETE_CANDIDATE_TOO_STALE)
        resolution.resolution_reasons.append(
            f"The newest source with a complete statement set reports "
            f"{newest_complete}, {complete_lag} days before the newest reported "
            f"period {newest_period}. That gap is too wide for the older "
            "period to represent this issuer's current state, so the state "
            "was not replaced wholesale; per-metric fallback is the caller's "
            "controlled path.")
        return resolution

    selected_period = max(c.period_end for c in complete)
    # §14: everything describing the SAME period is one economic period.
    for_period = [c for c in complete if c.period_end == selected_period]
    for_period.sort(key=lambda c: (_authority(c), -(_ordinal(c.issued_at))))
    primary = for_period[0]
    superseded = for_period[1:]

    resolution.selected_primary_source = primary
    resolution.selected_period = primary.fiscal_period or selected_period
    resolution.period_end = selected_period
    resolution.issued_at = primary.issued_at
    resolution.statement_completeness = primary.statement_completeness
    resolution.source_authority = _authority(primary)
    resolution.state_status = _status_for(primary, primary.statement_completeness)
    resolution.superseded_sources = superseded

    source_type = source_type_for(primary.form)
    if superseded:
        resolution.codes.append(ResolutionCode.SAME_PERIOD_SUPERSEDED)
        resolution.resolution_reasons.append(
            f"{primary.form or 'the periodic filing'} and "
            f"{', '.join(s.form or '?' for s in superseded)} describe the same period "
            f"({selected_period}). The higher-authority source is primary; the others "
            "are kept as provenance, not as a second current period.")

    beaten = [c for c in usable if (c.period_end or "") > selected_period]
    if beaten:
        resolution.codes.append(ResolutionCode.NEWER_SOURCE_INCOMPLETE)
        resolution.resolution_reasons.append(
            f"A newer source reports {beaten[0].period_end} but cannot describe it "
            f"completely, so {selected_period} remains the current period.")
    elif source_type == ActualSourceType.EARNINGS_RELEASE_FILED_8K:
        resolution.codes.append(ResolutionCode.ADVANCED_TO_NEWER_RELEASE)
        resolution.resolution_reasons.append(
            f"An earnings release dated {primary.issued_at} carries a complete "
            f"statement set for {selected_period}, newer than any periodic filing. "
            "The figures are preliminary and unaudited until the periodic filing "
            "arrives, which is recorded rather than assumed away.")
    else:
        resolution.codes.append(ResolutionCode.NEWEST_PERIODIC_FILING)
        resolution.resolution_reasons.append(
            f"{primary.form or 'A periodic filing'} is the newest complete statement "
            f"set, covering {selected_period}.")

    resolution.conflicts = detect_same_period_conflicts(primary, superseded)
    if resolution.conflicts:
        resolution.codes.append(ResolutionCode.SAME_PERIOD_SOURCE_CONFLICT)
        resolution.resolution_reasons.append(
            f"{len(resolution.conflicts)} metric(s) differ materially between sources "
            f"for {selected_period}. The authoritative source is used and the "
            "disagreement is recorded rather than overwritten.")

    resolution.fallbacks = build_metric_freshness(
        primary, resolution.state_status, prior_state_metrics)
    if any(f.is_fallback for f in resolution.fallbacks):
        resolution.codes.append(ResolutionCode.PARTIAL_ADVANCE_WITH_FALLBACK)
        resolution.resolution_reasons.append(
            "Some metrics could not be sourced from the current period and retain "
            "an older value; each is labelled with the period it came from.")
    return resolution


def _ordinal(value: Optional[str]) -> int:
    """A sortable issue date. Missing sorts oldest rather than crashing."""
    try:
        return int((value or "").replace("-", "")[:8] or 0)
    except ValueError:
        return 0


def detect_same_period_conflicts(primary: ReportedActualCandidate,
                                 others: List[ReportedActualCandidate]
                                 ) -> List[SourceConflict]:
    """§14: where two sources for one period disagree materially, say so.

    Only where the two are MEASURING THE SAME THING. One period end carries a
    quarter and a year-to-date figure and, at a fiscal year end, a quarter and
    a full year; a source that reported the quarter and one that reported the
    year disagree about nothing, and reporting that as a conflict would fill
    the state with contradictions that are arithmetic rather than factual.
    `period_type` says which duration a candidate's values are, and an unset
    one is compared -- the field is optional and an older caller that fills
    `values` without it should not lose the check.
    """
    conflicts = []
    primary_metrics = primary.values or {}
    for other in others:
        if primary.period_type and other.period_type                 and primary.period_type != other.period_type:
            continue
        for metric, value in (other.values or {}).items():
            mine = primary_metrics.get(metric)
            if mine is None or value is None:
                continue
            try:
                scale = max(abs(float(mine)), 1e-9)
                difference = abs(float(mine) - float(value)) / scale
            except (TypeError, ValueError):
                continue
            if difference > MATERIAL_DIFFERENCE:
                conflicts.append(SourceConflict(
                    metric=metric, period_end=primary.period_end or "",
                    authoritative_value=float(mine),
                    authoritative_source=primary.form,
                    other_value=float(value), other_source=other.form,
                    relative_difference=difference))
    return conflicts


def build_metric_freshness(primary: ReportedActualCandidate,
                           state_status: str,
                           prior_state_metrics: Optional[Dict[str, dict]] = None
                           ) -> List[MetricFreshness]:
    """One record per metric: which period it is from, and whether that is current.

    §20: a metric the current period cannot supply, retained from an older
    one, is recorded with both periods and a reason. There is no path here
    that quietly keeps an old number.
    """
    prior = prior_state_metrics or {}
    source_type = source_type_for(primary.form)
    finality = finality_for(source_type)
    records: List[MetricFreshness] = []
    present = set(primary.present_metrics or ())

    for metric in sorted(present):
        records.append(MetricFreshness(
            metric=metric, period_end=primary.period_end,
            issued_at=primary.issued_at, source_form=primary.form,
            source_type=source_type, actual_status=state_status,
            finality=finality, from_current_period=True))

    for metric, older in sorted(prior.items()):
        if metric in present:
            continue
        older_period = (older or {}).get("period_end")
        if not older_period or older_period >= (primary.period_end or ""):
            continue
        records.append(MetricFreshness(
            metric=metric, period_end=older_period,
            issued_at=(older or {}).get("issued_at"),
            source_form=(older or {}).get("form"),
            source_type=(older or {}).get("source_type")
            or ActualSourceType.OTHER_REPORTED_ACTUAL,
            actual_status=ActualStateStatus.HISTORICAL,
            finality=(older or {}).get("finality", FinalityStatus.UNKNOWN),
            from_current_period=False,
            fallback_from_period=older_period,
            fallback_reason=(
                f"{primary.form or 'the current source'} does not report {metric} "
                f"for {primary.period_end}; the {older_period} value is retained")))
    return records


# ---------------------------------------------------------------------------
# TTM reconstruction against the RESOLVED period (§12)
# ---------------------------------------------------------------------------
#
# `finance/ttm.py` already builds a twelve-month window and already flags one
# that ends materially before a `reference_end`. What it cannot do is know
# which period is current -- it derives its own reference from the XBRL facts,
# and those do not contain a period an 8-K reported but no periodic filing has
# yet carried.
#
# So the resolver supplies the reference. That is the whole coupling: when the
# current period advances, every TTM window is rebuilt against the new period
# end, and a window that cannot follow says so.
#
# The failure this prevents is a state claiming the new quarter is current
# while its TTM revenue still ends at the previous one -- two windows, one
# label, and a growth rate computed across the seam.

DEFAULT_TTM_METRICS = ("revenue", "net_income", "operating_income",
                       "operating_cash_flow", "capital_expenditure")


class TtmReconstructionStatus:
    """How well the twelve-month windows followed the current period."""

    COMPLETE = "COMPLETE"                # every window ends at the current period
    LIMITED = "LIMITED"                  # some did not, and are labelled
    NOT_RECONSTRUCTED = "NOT_RECONSTRUCTED"

    ALL = (COMPLETE, LIMITED, NOT_RECONSTRUCTED)


class TtmCode:
    ROLLED_FORWARD = "TTM_ROLLED_FORWARD"
    WINDOW_STALE = "TTM_WINDOW_STALE"
    NOT_RECONSTRUCTABLE = "TTM_NOT_RECONSTRUCTABLE"
    COMPONENT_PERIOD_MISMATCH = "TTM_COMPONENT_PERIOD_MISMATCH"


@dataclass(frozen=True)
class TtmWindow:
    """One twelve-month window and whether it kept up with the period."""

    metric: str
    value: Optional[float]
    start_date: Optional[str]
    end_date: Optional[str]
    construction_method: Optional[str]
    validation_status: str
    ends_at_current_period: bool
    reason: str = ""

    @property
    def is_stale(self) -> bool:
        return not self.ends_at_current_period

    def to_dict(self) -> dict:
        return {"metric": self.metric, "value": self.value,
                "start_date": self.start_date, "end_date": self.end_date,
                "construction_method": self.construction_method,
                "validation_status": self.validation_status,
                "ends_at_current_period": self.ends_at_current_period,
                "is_stale": self.is_stale, "reason": self.reason}


@dataclass
class TtmReconstruction:
    """Every window, against one reference period."""

    reference_period_end: Optional[str] = None
    status: str = TtmReconstructionStatus.NOT_RECONSTRUCTED
    windows: Dict[str, TtmWindow] = field(default_factory=dict)
    codes: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    @property
    def stale_metrics(self) -> List[str]:
        return sorted(m for m, w in self.windows.items() if w.is_stale)

    @property
    def current_metrics(self) -> List[str]:
        return sorted(m for m, w in self.windows.items() if not w.is_stale)

    @property
    def is_mixed_window(self) -> bool:
        """Do the windows end on different dates? Never silently."""
        ends = {w.end_date for w in self.windows.values() if w.end_date}
        return len(ends) > 1

    def to_dict(self) -> dict:
        return {"reference_period_end": self.reference_period_end,
                "status": self.status,
                "windows": {m: w.to_dict() for m, w in self.windows.items()},
                "stale_metrics": self.stale_metrics,
                "current_metrics": self.current_metrics,
                "is_mixed_window": self.is_mixed_window,
                "codes": list(self.codes), "reasons": list(self.reasons)}


# A window ending within this many days of the current period end is that
# period's window. Quarter ends move by a few days between issuers and years;
# a whole quarter does not.
TTM_END_TOLERANCE_DAYS = 45


def _days_between(earlier: Optional[str], later: Optional[str]) -> Optional[int]:
    if not earlier or not later:
        return None
    try:
        return (datetime.date.fromisoformat(later)
                - datetime.date.fromisoformat(earlier)).days
    except ValueError:
        return None


def reconstruct_ttm(company_facts: dict,
                    reference_period_end: Optional[str],
                    metrics: Optional[Tuple[str, ...]] = None
                    ) -> TtmReconstruction:
    """Rebuild every twelve-month window against the CURRENT period.

    `reference_period_end` comes from `resolve_current_actual_state`, not from
    the XBRL facts. That is the point: when an earnings release advances the
    period past the last periodic filing, the facts alone cannot know it.

    A window that cannot reach the current period is kept and LABELLED rather
    than dropped or silently used -- it is a valid twelve months, just not a
    current one, and the distinction is what stops a growth rate being
    computed across the seam.
    """
    from finance import ttm as ttm_module

    metrics = metrics or DEFAULT_TTM_METRICS
    result = TtmReconstruction(reference_period_end=reference_period_end)
    if not reference_period_end:
        result.reasons.append(
            "No current period was resolved, so no twelve-month window could be "
            "anchored to one.")
        result.codes.append(TtmCode.NOT_RECONSTRUCTABLE)
        return result

    for metric in metrics:
        built = ttm_module.build_ttm(company_facts, metric,
                                     reference_end=reference_period_end)
        lag = _days_between(built.end_date, reference_period_end)
        ends_current = bool(built.end_date) and lag is not None \
            and abs(lag) <= TTM_END_TOLERANCE_DAYS
        result.windows[metric] = TtmWindow(
            metric=metric, value=built.value, start_date=built.start_date,
            end_date=built.end_date,
            construction_method=built.construction_method,
            validation_status=built.validation_status,
            ends_at_current_period=ends_current,
            reason=built.reason or "")

    stale = [m for m, w in result.windows.items() if w.is_stale]
    if not result.windows:
        result.status = TtmReconstructionStatus.NOT_RECONSTRUCTED
        result.codes.append(TtmCode.NOT_RECONSTRUCTABLE)
    elif stale:
        result.status = TtmReconstructionStatus.LIMITED
        result.codes.append(TtmCode.WINDOW_STALE)
        result.reasons.append(
            f"{len(stale)} twelve-month window(s) could not be rebuilt to "
            f"{reference_period_end} and end earlier: {', '.join(sorted(stale))}. "
            "They remain valid twelve-month figures and are marked not current "
            "rather than combined with current-period values.")
    else:
        result.status = TtmReconstructionStatus.COMPLETE
        result.codes.append(TtmCode.ROLLED_FORWARD)
        result.reasons.append(
            f"Every twelve-month window ends at {reference_period_end}, the "
            "current reported period.")

    if result.is_mixed_window:
        result.codes.append(TtmCode.COMPONENT_PERIOD_MISMATCH)
        result.reasons.append(
            "The windows do not all end on the same date, so they are not "
            "components of one twelve-month period and must not be combined.")
    return result


# ---------------------------------------------------------------------------
# Completed-period guidance retirement (Fixture D, §2)
# ---------------------------------------------------------------------------
#
# An outlook for a period whose actual result is now reported is not an
# outlook. Leaving it CURRENT means the report carries a forecast and a fact
# about the same period, and a reader has no way to tell which is which --
# or worse, the forecast anchors an assumption the actual already contradicts.
#
# The statement is RETAINED, with its status changed. Deleting it would lose
# the record of what management said before the number arrived, which is the
# most interesting thing about it once the number exists.


class GuidanceActualizationCode:
    RETIRED_REALIZED = "GUIDANCE_RETIRED_REALIZED"
    STILL_FORWARD = "GUIDANCE_STILL_FORWARD"


@dataclass
class GuidanceActualization:
    """Which guidance survived the arrival of actual results, and which did not."""

    current: List[object] = field(default_factory=list)
    realized: List[object] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    codes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "current": [_metric_summary(m) for m in self.current],
            "realized": [_metric_summary(m) for m in self.realized],
            "reasons": list(self.reasons), "codes": list(self.codes),
        }


def _metric_field(metric, name: str, default=None):
    """One accessor for a live `GuidanceMetric` and its serialized dict form.

    By the time `workflow.py` has both `management_guidance` and an
    actualization resolution, guidance has already been serialized to a plain
    dict by `finance.sec.current_guidance` -- `finance.guidance` never hands
    back the live object past that tool boundary. A helper that only tried
    `getattr` would silently read `None` off every field of a dict metric and
    retire nothing, which is the live failure this function exists to close.
    """
    if isinstance(metric, dict):
        return metric.get(name, default)
    return getattr(metric, name, default)


def _metric_summary(metric) -> dict:
    return {"name": _metric_field(metric, "name"),
            "fiscal_period": _metric_field(metric, "fiscal_period"),
            "status": _metric_field(metric, "status")}


# SEC XBRL's own fiscal-period vocabulary -- what a `ReportedActualCandidate`
# actually carries in `fiscal_period` (see finance/extraction/document_
# resolver.py, which reads it straight off `entry["fp"]`). Never a label a
# guidance release wrote in its own prose.
_XBRL_FISCAL_PERIODS = frozenset({"Q1", "Q2", "Q3", "Q4", "FY"})


def _candidate_period_key(candidate) -> Optional[Tuple[int, str]]:
    """(fiscal year, "Q1".."Q4"/"FY") for one reported-actual candidate."""
    fiscal_year = getattr(candidate, "fiscal_year", None)
    fiscal_period = (getattr(candidate, "fiscal_period", None) or "").strip().upper()
    if not fiscal_year or fiscal_period not in _XBRL_FISCAL_PERIODS:
        return None
    return (fiscal_year, fiscal_period)


def _metric_period_key(metric) -> Optional[Tuple[int, str]]:
    """(fiscal year, "Q1".."Q4"/"FY") for one guidance metric's TARGET period.

    From `finance.guidance.parse_guidance_period` -- the SAME canonical period
    resolver `finance/extraction/validator.py::parse_target_period` already
    delegates to, never a second, independent label parser (that duplication
    is exactly what silently dropped a real AMBIGUOUS_TARGET_PERIOD rejection
    in the prior reader-recall phase). A label this cannot parse -- a
    multi-year framework, a bare "several years" -- fails open: retiring a
    forecast we cannot confidently match to a reported period would silently
    drop a live outlook, which is a worse failure than leaving it current one
    cycle too long.
    """
    from finance.guidance import GuidancePeriodType, parse_guidance_period

    label = _metric_field(metric, "fiscal_period")
    if not label:
        return None
    period = parse_guidance_period(label, "")
    if period is None:
        return None
    if period.period_type == GuidancePeriodType.QUARTER and period.quarter:
        return (period.fiscal_year, f"Q{period.quarter}")
    if period.period_type == GuidancePeriodType.ANNUAL:
        return (period.fiscal_year, "FY")
    return None


def _period_end_for(metric, period_ends: Dict[Tuple[int, str], str]) -> Optional[str]:
    """The calendar end of a guided period, from a reported candidate that
    covers the SAME (fiscal year, quarter/FY) -- never computed. The
    issuer's own fiscal calendar is what decides when "Q3 FY2026" ends, and
    inventing a second fiscal-calendar parser is exactly what §6 of the
    actualization brief forbids.
    """
    key = _metric_period_key(metric)
    return period_ends.get(key) if key else None


def retire_realized_guidance(metrics, resolution) -> GuidanceActualization:
    """Split guidance into still-forward and realized, against reported actuals.

    `resolution` is an `ActualStateResolution`. Its candidates carry the
    period ends actual results have been reported for, and
    `document_resolver.period_has_actuals` answers whether a given guided
    period is among them.
    """
    from finance.extraction.document_resolver import period_has_actuals

    result = GuidanceActualization()
    candidates = _resolution_candidates(resolution)
    period_ends: Dict[Tuple[int, str], str] = {}
    for c in candidates:
        key = _candidate_period_key(c)
        if key and c.period_end:
            period_ends[key] = c.period_end

    for metric in metrics or ():
        label = _metric_field(metric, "fiscal_period")
        target_end = _period_end_for(metric, period_ends)
        if target_end and period_has_actuals(label, target_end, candidates):
            result.realized.append(_with_status(metric, GuidanceStatus.REALIZED))
            result.reasons.append(
                f"{_metric_field(metric, 'name', 'a metric')} guidance for {label} is no "
                f"longer an outlook: actual results for {target_end} have been "
                "reported. The statement is kept as history.")
        else:
            result.current.append(metric)

    result.codes.append(GuidanceActualizationCode.RETIRED_REALIZED
                        if result.realized
                        else GuidanceActualizationCode.STILL_FORWARD)
    return result


def _resolution_candidates(resolution) -> list:
    """Every source the resolution considered, primary and superseded."""
    found = []
    primary = getattr(resolution, "selected_primary_source", None)
    if primary is not None:
        found.append(primary)
    found.extend(getattr(resolution, "superseded_sources", None) or [])
    return found


def _with_status(metric, status):
    """A copy carrying the new status. `GuidanceMetric` is frozen by design."""
    reason = "actual results for this period are in"
    if hasattr(metric, "with_status"):
        return metric.with_status(status, reason)
    if isinstance(metric, dict):
        return {**metric, "status": status, "status_reason": reason}
    return metric


# ---------------------------------------------------------------------------
# Current-state claim gate (§10) and latest-quarter provenance (§3)
# ---------------------------------------------------------------------------
#
# The question every "current", "latest" or "TTM" claim has to answer before
# it may be phrased that way: is the fact behind it from the period we say is
# current? A label cannot assert that. It is checked.


class FreshnessEligibility:
    """Whether a fact may be presented as current."""

    CURRENT = "CURRENT"            # from the resolved current period
    FALLBACK = "FALLBACK"          # older, explicitly labelled as such
    STALE = "STALE"                # older, and NOT eligible for a current label
    MIXED_PERIOD = "MIXED_PERIOD"  # assembled from more than one period
    UNKNOWN = "UNKNOWN"            # provenance insufficient to judge

    ALL = (CURRENT, FALLBACK, STALE, MIXED_PERIOD, UNKNOWN)

    # The only state in which a claim may be called current/latest/TTM.
    MAY_CLAIM_CURRENT = (CURRENT,)


class LatestQuarterCode:
    LABEL_MATCHES_PERIOD = "LATEST_QUARTER_LABEL_MATCHES_PERIOD"
    LABEL_PERIOD_MISMATCH = "LATEST_QUARTER_LABEL_PERIOD_MISMATCH"
    METRIC_ABSENT = "LATEST_QUARTER_METRIC_ABSENT"


@dataclass(frozen=True)
class ClaimFreshness:
    """One claim, and whether its provenance supports how it is labelled."""

    claim: str
    metric: str
    eligibility: str
    fact_period_end: Optional[str]
    current_period_end: Optional[str]
    reason: str = ""
    code: Optional[str] = None

    @property
    def may_be_called_current(self) -> bool:
        return self.eligibility in FreshnessEligibility.MAY_CLAIM_CURRENT

    def to_dict(self) -> dict:
        return {"claim": self.claim, "metric": self.metric,
                "eligibility": self.eligibility,
                "fact_period_end": self.fact_period_end,
                "current_period_end": self.current_period_end,
                "may_be_called_current": self.may_be_called_current,
                "reason": self.reason, "code": self.code}


# Point-in-time balance-sheet fields, read from the ONE reviewed map so the
# distinction lives in a single place. A point-in-time figure is CURRENT when
# it is from the resolved period's balance-sheet date; it never has a
# twelve-month window and must not be judged as though it should
# (finance_extraction_v2 §17). Anything not listed here is treated as a flow.
def _point_in_time_metrics() -> frozenset:
    try:
        from finance.xbrl_mapping import CONCEPT_MAP
        return frozenset(name for name, (is_instant, _c) in CONCEPT_MAP.items()
                         if is_instant)
    except Exception:                                        # noqa: BLE001
        return frozenset({
            "cash_and_cash_equivalents", "short_term_investments", "assets",
            "current_assets", "liabilities", "current_liabilities",
            "stockholders_equity", "short_term_debt", "long_term_debt",
            "current_portion_of_long_term_debt", "total_debt"})


POINT_IN_TIME_METRICS = _point_in_time_metrics() | {"total_debt", "net_debt"}


# Metrics a "latest quarter" claim is made about. §3's minimum set.
LATEST_QUARTER_METRICS = ("revenue", "net_income", "operating_income",
                          "operating_margin", "earnings_per_share")


def assess_claim_freshness(metric: str, claim_label: str,
                           resolution: "ActualStateResolution",
                           ttm: Optional["TtmReconstruction"] = None
                           ) -> ClaimFreshness:
    """May this metric be presented under this label?

    Answers from PROVENANCE, never from the wording. A claim about a TTM
    window is judged against the window's end date; a claim about the latest
    quarter against the resolved current period; and a metric with no
    freshness record at all is UNKNOWN rather than assumed current.
    """
    current_end = resolution.period_end

    # A POINT-IN-TIME metric is judged against the resolved BALANCE-SHEET
    # date, never against a twelve-month window it will never have. Passing
    # `ttm` used to force every metric down the window path, so cash and debt
    # -- present, current, from the right balance sheet -- came back UNKNOWN
    # because no window was built for them. That was the RESEARCH_FRESHNESS
    # over-conservatism the Actualization benchmark recorded. The fix is
    # narrow: route balances to the provenance record (which `build_metric_
    # freshness` already anchors on the balance-sheet date) and leave the
    # window path to the flows.
    if metric in POINT_IN_TIME_METRICS:
        ttm = None

    # The presence of a WINDOW decides that this is a twelve-month claim, not
    # the wording of the label. Keying on the word "ttm" was prose inspection
    # inside the function that exists to replace prose inspection, and it let
    # a caller asking about "current revenue" past a stale window.
    if ttm is not None:
        window = ttm.windows.get(metric)
        if window is None:
            return ClaimFreshness(
                claim_label, metric, FreshnessEligibility.UNKNOWN,
                None, current_end,
                f"no twelve-month window was built for {metric}",
                LatestQuarterCode.METRIC_ABSENT)
        if window.is_stale:
            return ClaimFreshness(
                claim_label, metric, FreshnessEligibility.STALE,
                window.end_date, current_end,
                f"the twelve-month window ends {window.end_date}, before the "
                f"current period {current_end}; it may not be called current",
                LatestQuarterCode.LABEL_PERIOD_MISMATCH)
        return ClaimFreshness(
            claim_label, metric, FreshnessEligibility.CURRENT,
            window.end_date, current_end,
            "the twelve-month window ends at the current period",
            LatestQuarterCode.LABEL_MATCHES_PERIOD)

    record = next((f for f in resolution.fallbacks if f.metric == metric), None)
    if record is None:
        return ClaimFreshness(
            claim_label, metric, FreshnessEligibility.UNKNOWN,
            None, current_end,
            f"{metric} has no freshness record, so its period cannot be checked",
            LatestQuarterCode.METRIC_ABSENT)

    if record.from_current_period:
        return ClaimFreshness(
            claim_label, metric, FreshnessEligibility.CURRENT,
            record.period_end, current_end,
            "the value is from the current reported period",
            LatestQuarterCode.LABEL_MATCHES_PERIOD)

    eligibility = (FreshnessEligibility.FALLBACK if record.is_fallback
                   else FreshnessEligibility.STALE)
    return ClaimFreshness(
        claim_label, metric, eligibility, record.period_end, current_end,
        f"the value is from {record.period_end}, not the current period "
        f"{current_end}; it may be reported with that period named, never as "
        "the latest",
        LatestQuarterCode.LABEL_PERIOD_MISMATCH)


def assess_latest_quarter_claims(resolution: "ActualStateResolution",
                                 metrics: Optional[Tuple[str, ...]] = None
                                 ) -> Dict[str, ClaimFreshness]:
    """Every latest-quarter claim, judged against the resolved quarter."""
    metrics = metrics or LATEST_QUARTER_METRICS
    return {metric: assess_claim_freshness(metric, "latest quarter", resolution)
            for metric in metrics}


def claims_safe_to_publish_as_current(
        assessments: Dict[str, ClaimFreshness]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(may be called current, must not be). No third outcome."""
    allowed = tuple(sorted(m for m, a in assessments.items()
                           if a.may_be_called_current))
    refused = tuple(sorted(m for m, a in assessments.items()
                           if not a.may_be_called_current))
    return allowed, refused


# ---------------------------------------------------------------------------
# DCF financial-base freshness (§7/§8)
# ---------------------------------------------------------------------------
#
# A discounted cash flow built on a period the company has since superseded is
# not a valuation of the company as it stands. Every figure in it may be
# arithmetically perfect and the answer still describes a quarter that is over.
#
# The comparison is on PERIOD ENDS -- dates -- never on labels. "Q3 FY2026"
# and "2026-04-25" are the same period expressed two ways, and comparing them
# as strings either misses a stale base or invents one.


class DcfBaseFreshness:
    """Whether a DCF's financial base keeps up with reported results."""

    CURRENT = "CURRENT"        # built on the current reported period
    STALE = "STALE"            # a newer actual period has been reported
    AHEAD = "AHEAD"            # base is newer than any resolved actual
    UNKNOWN = "UNKNOWN"        # a period could not be established

    ALL = (CURRENT, STALE, AHEAD, UNKNOWN)


DCF_FINANCIAL_BASE_STALE = "DCF_FINANCIAL_BASE_STALE"

# How far a base may trail the current period and still be that period's base.
# A base within the same reporting period is current; a whole period behind is
# not. Matches `TTM_END_TOLERANCE_DAYS` deliberately -- the two answer the
# same question about the same calendar.
DCF_BASE_TOLERANCE_DAYS = TTM_END_TOLERANCE_DAYS


@dataclass(frozen=True)
class DcfBaseAssessment:
    """The verdict, with both periods so a reader can check it."""

    freshness: str
    current_period_end: Optional[str]
    base_period_end: Optional[str]
    lag_days: Optional[int]
    reason: str = ""
    code: Optional[str] = None

    @property
    def is_stale(self) -> bool:
        return self.freshness == DcfBaseFreshness.STALE

    @property
    def may_be_research_valid(self) -> bool:
        """UNKNOWN fails closed: a base whose period we cannot establish
        cannot be shown to be current, and an unverifiable valuation is not a
        verified one."""
        return self.freshness in (DcfBaseFreshness.CURRENT, DcfBaseFreshness.AHEAD)

    def to_dict(self) -> dict:
        return {"freshness": self.freshness,
                "current_period_end": self.current_period_end,
                "base_period_end": self.base_period_end,
                "lag_days": self.lag_days, "reason": self.reason,
                "code": self.code}


def assess_dcf_base_freshness(current_period_end: Optional[str],
                              base_period_end: Optional[str]
                              ) -> DcfBaseAssessment:
    """Is this DCF built on the period the company has actually reported?"""
    if not current_period_end or not base_period_end:
        return DcfBaseAssessment(
            DcfBaseFreshness.UNKNOWN, current_period_end, base_period_end, None,
            "the DCF base period or the current reported period could not be "
            "established, so the valuation cannot be shown to rest on current "
            "results", DCF_FINANCIAL_BASE_STALE)

    lag = _days_between(base_period_end, current_period_end)
    if lag is None:
        return DcfBaseAssessment(
            DcfBaseFreshness.UNKNOWN, current_period_end, base_period_end, None,
            "the periods could not be compared as dates",
            DCF_FINANCIAL_BASE_STALE)

    if lag < 0:
        # The base is NEWER than any resolved actual. Not stale -- flagging it
        # would be a false positive, which §8 calls out specifically.
        return DcfBaseAssessment(
            DcfBaseFreshness.AHEAD, current_period_end, base_period_end, lag,
            f"the valuation base ({base_period_end}) is not older than the "
            f"current reported period ({current_period_end})")

    if lag <= DCF_BASE_TOLERANCE_DAYS:
        return DcfBaseAssessment(
            DcfBaseFreshness.CURRENT, current_period_end, base_period_end, lag,
            f"the valuation is built on {base_period_end}, the current "
            "reported period")

    return DcfBaseAssessment(
        DcfBaseFreshness.STALE, current_period_end, base_period_end, lag,
        f"the valuation is built on {base_period_end} but the company has "
        f"since reported {current_period_end}, {lag} days later. The model "
        "describes a period that is over.",
        DCF_FINANCIAL_BASE_STALE)


# ---------------------------------------------------------------------------
# Research-evidence freshness (§9)
# ---------------------------------------------------------------------------
#
# The claim gate above decides whether ONE metric may be called current. This
# turns that into something the evidence index can apply to every item in the
# current namespace at once.
#
# Rewriting the LABEL rather than dropping the item is deliberate. A
# prior-quarter figure is still real and still useful; what it may not do is
# wear the word "current". Dropping it would lose information a role
# legitimately needs, and leaving it unqualified is the failure itself.

# The evidence namespace whose items assert currency by being in it.
CURRENT_EVIDENCE_PREFIX = "current."


@dataclass
class ResearchFreshnessView:
    """Which current-namespace metrics may keep their label, and which may not."""

    current_period_end: Optional[str] = None
    eligibility: Dict[str, ClaimFreshness] = field(default_factory=dict)
    requalified: Dict[str, str] = field(default_factory=dict)
    codes: List[str] = field(default_factory=list)

    @property
    def may_claim_current(self) -> Tuple[str, ...]:
        return tuple(sorted(m for m, a in self.eligibility.items()
                            if a.may_be_called_current))

    @property
    def must_be_qualified(self) -> Tuple[str, ...]:
        return tuple(sorted(m for m, a in self.eligibility.items()
                            if not a.may_be_called_current))

    def to_dict(self) -> dict:
        return {"current_period_end": self.current_period_end,
                "may_claim_current": list(self.may_claim_current),
                "must_be_qualified": list(self.must_be_qualified),
                "requalified": dict(self.requalified),
                "eligibility": {m: a.to_dict() for m, a in self.eligibility.items()},
                "codes": list(self.codes)}


def build_research_freshness_view(resolution: "ActualStateResolution",
                                  ttm: Optional["TtmReconstruction"] = None,
                                  metrics: Optional[Tuple[str, ...]] = None
                                  ) -> ResearchFreshnessView:
    """Judge every current-namespace metric against the resolved period."""
    view = ResearchFreshnessView(current_period_end=resolution.period_end)
    names = metrics or tuple(sorted(
        {f.metric for f in resolution.fallbacks}
        | set((ttm.windows if ttm else {}).keys())))

    for metric in names:
        assessment = assess_claim_freshness(metric, "current", resolution, ttm=ttm)
        view.eligibility[metric] = assessment
        if not assessment.may_be_called_current:
            period = assessment.fact_period_end or "an earlier period"
            view.requalified[metric] = period
    if view.requalified:
        view.codes.append("RESEARCH_EVIDENCE_REQUALIFIED")
    return view


def requalify_evidence_label(label: str, period_end: str) -> str:
    """Replace an assertion of currency with the period the figure is from.

    The word is removed, not decorated: "Current revenue (as of 2026-04-30)"
    still reads as current at a glance, and a role skimming for the latest
    figure would take it.
    """
    text = label or ""
    for word in ("Current ", "current ", "Latest ", "latest "):
        if text.startswith(word):
            text = text[len(word):]
            break
        text = text.replace(f" {word.strip()} ", " ")
    return f"{text.strip()} for the period ended {period_end}".strip()
