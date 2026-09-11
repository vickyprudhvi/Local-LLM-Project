"""Running the Actualization benchmark, and scoring what came back.

TWO STAGES, DELIBERATELY SEPARATE

    run_case(case, layer)  ->  Measurement     RAW. Pure data, no verdicts.
    score(measurements)    ->  Score           Verdicts only, no execution.

The split is what section 9 asks for. A `Measurement` is a record of what a
layer DID; nothing in it consults the expected data, so a scoring rule that
turns out to be wrong can be corrected and the stored measurements re-scored
without running production code again -- and without the temptation to adjust
a rule while watching a number move.

It is also what makes "checkpoint failure must never destroy a completed
measurement" true rather than aspirational: reporting reads measurements, and
a reporter that raises cannot reach back into one.

WHAT IS MEASURED, AND WHAT IS NOT

Both layers are run over the SAME company-facts payload:

    V1  finance/freshness.py::build_current_financial_state
    V2  finance/extraction/document_resolver.py::discover_reported_actuals
        -> finance/actualization.py::resolve_current_actual_state
        -> reconstruct_ttm / retire_realized_guidance /
           assess_dcf_base_freshness / build_research_freshness_view

Neither is scored through the other. Compare mode is NOT used: it returns V1's
answer by design, and reading that as V2's behaviour is the single mistake
section 5 names.

V1 HAS NO VOCABULARY FOR MOST OF THIS, AND THAT IS THE MEASUREMENT

V1 reports no actual-status, no statement completeness, no fallback label, no
TTM reconstruction status, no guidance retirement and no DCF base freshness.
Those are recorded as absent, not as passes. Its per-metric periods ARE
recorded, and every one of them is recorded as PRESENTED AS CURRENT -- because
V1 has no other way to present a figure, which is exactly why a state whose
balance sheet ends in April and whose revenue window ends in July is a silent
mixed-period state rather than a labelled one.
"""

import datetime
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from tests.fixtures import actualization_benchmark as bench
from tests.fixtures.actualization_benchmark import (
    ActualizationCase,
    ConflictStatus,
    Currentness,
    DcfBase,
    FailureClass,
    Hard,
    Truth,
)


V1 = "V1"
V2 = "V2"
LAYERS = (V1, V2)

# The metrics every case states per-metric expectations about.
SCORED_METRICS = bench.SCORED_METRICS
FLOW_METRICS = ("revenue", "net_income", "operating_income",
                "operating_cash_flow", "capital_expenditure")

# How far a window end may sit from a period end and still be that period's.
# The production constant, imported rather than restated -- a benchmark with
# its own tolerance measures its own tolerance.
from finance.actualization import TTM_END_TOLERANCE_DAYS  # noqa: E402


def _days(earlier: Optional[str], later: Optional[str]) -> Optional[int]:
    if not earlier or not later:
        return None
    try:
        return (datetime.date.fromisoformat(later)
                - datetime.date.fromisoformat(earlier)).days
    except ValueError:
        return None


def _same_period(left: Optional[str], right: Optional[str]) -> bool:
    lag = _days(left, right)
    return lag is not None and abs(lag) <= TTM_END_TOLERANCE_DAYS


# ---------------------------------------------------------------------------
# The raw record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Measurement:
    """What ONE layer did on ONE case. No verdicts, no expected data."""

    case_id: str
    layer: str

    selected_period: Optional[str] = None
    selected_primary_source: Optional[str] = None
    state_status: Optional[str] = None
    statement_completeness: Optional[str] = None

    # metric -> the period end of the fact this layer used
    metric_periods: Dict[str, Optional[str]] = field(default_factory=dict)
    metric_sources: Dict[str, Optional[str]] = field(default_factory=dict)
    # metrics this layer PRESENTS as belonging to the current period
    presented_as_current: Tuple[str, ...] = ()
    # metric -> the older period it is LABELLED as retained from
    fallback_labels: Dict[str, str] = field(default_factory=dict)

    ttm_reference_end: Optional[str] = None
    ttm_status: Optional[str] = None
    # metric -> {"end": ..., "start": ..., "quarters": [...] , "current": bool}
    ttm_windows: Dict[str, dict] = field(default_factory=dict)

    guidance_realized: Tuple[str, ...] = ()
    guidance_current: Tuple[str, ...] = ()

    conflict_metrics: Tuple[str, ...] = ()
    superseded_forms: Tuple[str, ...] = ()
    # every period this layer treats as A current reported state
    current_state_periods: Tuple[str, ...] = ()
    # (period_end, reason) for every candidate refused
    rejected: Tuple[Tuple[str, str], ...] = ()
    # periods of sources this layer ACCEPTED as reported actuals
    accepted_periods: Tuple[str, ...] = ()

    dcf_base_period: Optional[str] = None
    dcf_freshness: Optional[str] = None
    dcf_research_valid: Optional[bool] = None

    research_current: Tuple[str, ...] = ()
    research_qualified: Tuple[str, ...] = ()

    error: Optional[str] = None

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))


# ---------------------------------------------------------------------------
# V2
# ---------------------------------------------------------------------------

def _v2_candidates(case: ActualizationCase):
    from finance.extraction.document_resolver import discover_reported_actuals

    return list(discover_reported_actuals(case.company_facts())) \
        + list(case.extra_candidates)


def run_v2(case: ActualizationCase) -> Measurement:
    """The V2 layer, called exactly as `actualization_runtime` calls it."""
    from finance import actualization as act

    facts = case.company_facts()
    candidates = _v2_candidates(case)
    resolution = act.resolve_current_actual_state(
        candidates, as_of=case.as_of,
        prior_state_metrics=dict(case.prior_state_metrics) or None)

    metric_periods: Dict[str, Optional[str]] = {}
    metric_sources: Dict[str, Optional[str]] = {}
    fallback_labels: Dict[str, str] = {}
    presented: List[str] = []
    for record in resolution.fallbacks:
        metric_periods[record.metric] = record.period_end
        metric_sources[record.metric] = record.source_form
        if record.is_fallback:
            fallback_labels[record.metric] = record.fallback_from_period or ""
        if record.from_current_period:
            presented.append(record.metric)

    ttm = act.reconstruct_ttm(facts, resolution.period_end) \
        if resolution.period_end else None
    windows: Dict[str, dict] = {}
    if ttm is not None:
        for metric, window in ttm.windows.items():
            windows[metric] = {
                "end": window.end_date, "start": window.start_date,
                "current": bool(window.ends_at_current_period),
                "method": window.construction_method,
                "validation": window.validation_status,
                "quarters": list(_quarter_ends(facts, metric, window)),
            }

    guidance = act.retire_realized_guidance(list(case.guidance), resolution) \
        if case.guidance else None

    assessment = act.assess_dcf_base_freshness(
        resolution.period_end, case.dcf_base_period_end)

    view = act.build_research_freshness_view(
        resolution, ttm, metrics=tuple(sorted(
            set(case.expected_research_currentness) or set(FLOW_METRICS))))

    for_period = [c for c in candidates
                  if c.period_end == resolution.period_end and not c.is_prospective]

    return Measurement(
        case_id=case.case_id, layer=V2,
        selected_period=resolution.period_end,
        selected_primary_source=(resolution.selected_primary_source.form
                                 if resolution.selected_primary_source else None),
        state_status=resolution.state_status,
        statement_completeness=resolution.statement_completeness,
        metric_periods=metric_periods,
        metric_sources=metric_sources,
        presented_as_current=tuple(sorted(presented)),
        fallback_labels=fallback_labels,
        ttm_reference_end=(ttm.reference_period_end if ttm else None),
        ttm_status=(ttm.status if ttm else None),
        ttm_windows=windows,
        guidance_realized=tuple(sorted(
            str(getattr(m, "fiscal_period", "")) for m in (guidance.realized if guidance else ()))),
        guidance_current=tuple(sorted(
            str(getattr(m, "fiscal_period", "")) for m in (guidance.current if guidance else ()))),
        conflict_metrics=tuple(sorted({c.metric for c in resolution.conflicts})),
        superseded_forms=tuple(s.form or "?" for s in resolution.superseded_sources),
        current_state_periods=((resolution.period_end,)
                               if resolution.period_end and resolution.is_current else ()),
        rejected=tuple((c.period_end or "", why) for c, why in resolution.rejected_candidates),
        accepted_periods=tuple(sorted({c.period_end or "" for c in for_period})),
        dcf_base_period=case.dcf_base_period_end,
        dcf_freshness=assessment.freshness,
        dcf_research_valid=assessment.may_be_research_valid,
        research_current=view.may_claim_current,
        research_qualified=view.must_be_qualified,
    )


def _quarter_ends(facts: dict, metric: str, window) -> Tuple[str, ...]:
    """The period ends of the quarters a window was built from.

    Read back out of `finance/ttm.py`'s own record rather than recomputed, so
    a window that says it summed four quarters can be checked against which
    four.
    """
    from finance import ttm as ttm_module

    if not window.end_date:
        return ()
    built = ttm_module.build_ttm(facts, metric, reference_end=window.end_date)
    ends = []
    for span in built.quarters_included:
        _start, _sep, end = span.partition("..")
        if end:
            ends.append(end)
    return tuple(ends)


# ---------------------------------------------------------------------------
# V1
# ---------------------------------------------------------------------------

# Which compartment of `CurrentFinancialState` each scored metric lives in.
_V1_BALANCE_METRICS = ("cash_and_cash_equivalents", "total_debt")


def run_v1(case: ActualizationCase) -> Measurement:
    """The existing freshness planner, run over the same payload."""
    from finance import ttm as ttm_module
    from finance.freshness import build_current_financial_state

    facts = case.company_facts()
    state = build_current_financial_state(facts, "BENCH",
                                          valuation_date=case.as_of)

    metric_periods: Dict[str, Optional[str]] = {}
    metric_sources: Dict[str, Optional[str]] = {}
    presented: List[str] = []
    for metric in SCORED_METRICS:
        selection = state.selection(metric)
        period = getattr(selection, "as_of_date", None)
        metric_periods[metric] = period
        metric_sources[metric] = getattr(selection, "form", None)
        if getattr(selection, "value", None) is not None and period:
            # V1 has no label but "current". Every figure it publishes is
            # published as the company's present state.
            presented.append(metric)

    windows: Dict[str, dict] = {}
    for metric in FLOW_METRICS:
        selection = state.flows.get(metric)
        record = getattr(selection, "ttm", None) or {}
        end = record.get("end_date") or getattr(selection, "as_of_date", None)
        quarters = []
        for span in record.get("quarters_included") or ():
            _s, _sep, quarter_end = str(span).partition("..")
            if quarter_end:
                quarters.append(quarter_end)
        windows[metric] = {
            "end": end, "start": record.get("start_date"),
            # V1 draws no distinction: a window it used is a window it
            # presents as the company's trailing twelve months.
            "current": end is not None,
            "method": record.get("construction_method"),
            "validation": record.get("validation_status"),
            "quarters": quarters,
        }

    return Measurement(
        case_id=case.case_id, layer=V1,
        selected_period=state.financial_as_of,
        selected_primary_source=_v1_primary_form(state),
        state_status=None,               # V1 has no actual-status vocabulary
        statement_completeness=None,     # nor a completeness one
        metric_periods=metric_periods,
        metric_sources=metric_sources,
        presented_as_current=tuple(sorted(presented)),
        fallback_labels={},              # nor a fallback label
        ttm_reference_end=ttm_module.latest_reported_period_end(facts),
        ttm_status=None,                 # nor a reconstruction status
        ttm_windows=windows,
        guidance_realized=(),            # V1 retires no completed-period guidance
        guidance_current=tuple(sorted(
            str(getattr(m, "fiscal_period", "")) for m in case.guidance)),
        conflict_metrics=(),
        superseded_forms=(),
        current_state_periods=((state.financial_as_of,)
                               if state.financial_as_of else ()),
        rejected=(),
        accepted_periods=(),
        dcf_base_period=case.dcf_base_period_end,
        dcf_freshness=None,
        # `workflow._dcf_base_staleness` returns (False, None) unless the V2
        # layer was used. Under V1 a stale base is never withheld.
        dcf_research_valid=True,
        research_current=tuple(sorted(
            set(case.expected_research_currentness) & set(presented))),
        research_qualified=(),
    )


def _v1_primary_form(state) -> Optional[str]:
    """The form of the filing V1's balance-sheet date came from."""
    for name in ("cash_and_cash_equivalents", "stockholders_equity", "assets"):
        selection = state.balance_sheet.get(name)
        if selection is not None and selection.as_of_date == state.financial_as_of:
            return selection.form
    return None


RUNNERS = {V1: run_v1, V2: run_v2}


def run_case(case: ActualizationCase, layer: str) -> Measurement:
    try:
        return RUNNERS[layer](case)
    except Exception as failure:                             # noqa: BLE001
        # A layer that raises has still been measured: it failed to produce a
        # state. Recorded rather than crashing the run, so one broken case
        # cannot destroy seventeen completed measurements.
        return Measurement(case_id=case.case_id, layer=layer,
                           error=f"{type(failure).__name__}: {failure}")


def run_all(cases: Optional[Sequence[ActualizationCase]] = None
            ) -> List[Measurement]:
    """Every case, both layers. The measurement, once."""
    cases = list(cases if cases is not None else bench.cases())
    return [run_case(case, layer) for case in cases for layer in LAYERS]


# ---------------------------------------------------------------------------
# Scoring (section 6)
# ---------------------------------------------------------------------------

@dataclass
class Tally:
    """Correct out of scored, with what was excluded kept visible."""

    correct: int = 0
    scored: int = 0
    excluded: int = 0
    misses: List[str] = field(default_factory=list)

    def hit(self, ok: bool, label: str = "") -> None:
        self.scored += 1
        if ok:
            self.correct += 1
        elif label:
            self.misses.append(label)

    def skip(self) -> None:
        self.excluded += 1

    @property
    def rate(self) -> float:
        return self.correct / self.scored if self.scored else 1.0

    def text(self) -> str:
        if not self.scored:
            return "n/a"
        return f"{self.rate:.0%} ({self.correct}/{self.scored})"


@dataclass
class Score:
    layer: str
    cases: int = 0
    resolved_period: Tally = field(default_factory=Tally)
    latest_period: Tally = field(default_factory=Tally)
    primary_source: Tally = field(default_factory=Tally)
    state_status: Tally = field(default_factory=Tally)
    per_metric_freshness: Tally = field(default_factory=Tally)
    per_metric_not_covered: int = 0
    fallback_label: Tally = field(default_factory=Tally)
    ttm_window: Tally = field(default_factory=Tally)
    ttm_end_period: Tally = field(default_factory=Tally)
    ttm_status: Tally = field(default_factory=Tally)
    guidance_retirement: Tally = field(default_factory=Tally)
    actual_guidance_separation: Tally = field(default_factory=Tally)
    same_period_supersession: Tally = field(default_factory=Tally)
    conflict_detection: Tally = field(default_factory=Tally)
    dcf_stale_base: Tally = field(default_factory=Tally)
    research_currentness: Tally = field(default_factory=Tally)
    research_over_qualified: int = 0       # safe direction: too cautious
    research_under_qualified: int = 0      # unsafe direction
    hard: Dict[str, int] = field(default_factory=dict)
    hard_detail: Dict[str, List[str]] = field(default_factory=dict)
    failure_classes: Dict[str, List[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def hard_total(self) -> int:
        return sum(self.hard.values())

    def note(self, failure_class: str, label: str) -> None:
        self.failure_classes.setdefault(failure_class, []).append(label)

    def flag(self, name: str, label: str) -> None:
        self.hard[name] = self.hard.get(name, 0) + 1
        self.hard_detail.setdefault(name, []).append(label)


def _blank_score(layer: str) -> Score:
    return Score(layer=layer, hard={name: 0 for name in Hard.ALL})


def score(measurements: Sequence[Measurement],
          cases: Optional[Sequence[ActualizationCase]] = None
          ) -> Dict[str, Score]:
    """Verdicts from stored measurements. Runs no production code.

    Every layer is scored independently against the case's expected data.
    Nothing here reads one layer's output to judge another's, and nothing
    reads a layer's output to decide what the truth is.
    """
    by_id = {case.case_id: case for case in (cases if cases is not None
                                             else bench.cases())}
    scores = {layer: _blank_score(layer) for layer in LAYERS}

    for record in measurements:
        case = by_id.get(record.case_id)
        if case is None:
            continue
        result = scores.setdefault(record.layer, _blank_score(record.layer))
        result.cases += 1
        if record.error:
            result.errors.append(f"{record.case_id}: {record.error}")
            continue
        _score_one(case, record, result)
    return scores


def _score_one(case: ActualizationCase, m: Measurement, s: Score) -> None:
    tag = f"{case.case_id}"

    # -- section 10: the resolved period ------------------------------------
    period_ok = None
    if case.expected_current_period != Truth.UNKNOWN:
        period_ok = m.selected_period == case.expected_current_period
        s.resolved_period.hit(period_ok, f"{tag}: {m.selected_period!r}")
        if not period_ok:
            s.note(FailureClass.PERIOD_RESOLUTION,
                   f"{tag}: {m.selected_period!r} for "
                   f"{case.expected_current_period!r}")
    else:
        s.resolved_period.skip()

    source_ok = None
    if case.expected_primary_source != Truth.UNKNOWN:
        source_ok = m.selected_primary_source == case.expected_primary_source
        s.primary_source.hit(source_ok, f"{tag}: {m.selected_primary_source!r}")
        if not source_ok:
            s.note(FailureClass.ACTUAL_SOURCE_PRECEDENCE,
                   f"{tag}: {m.selected_primary_source!r} for "
                   f"{case.expected_primary_source!r}")
    else:
        s.primary_source.skip()

    status_ok = None
    if case.expected_actual_status != Truth.UNKNOWN:
        status_ok = m.state_status == case.expected_actual_status
        s.state_status.hit(status_ok, f"{tag}: {m.state_status!r}")
    else:
        s.state_status.skip()

    if None in (period_ok, source_ok, status_ok):
        s.latest_period.skip()
    else:
        s.latest_period.hit(bool(period_ok and source_ok and status_ok), tag)

    if case.expected_statement_completeness != Truth.UNKNOWN \
            and m.statement_completeness is not None \
            and m.statement_completeness != case.expected_statement_completeness:
        s.note(FailureClass.STATEMENT_COMPLETENESS,
               f"{tag}: {m.statement_completeness}")

    # -- section 12: per-metric freshness -----------------------------------
    for metric, expected in sorted(case.expected_current_metrics.items()):
        if expected in (Truth.UNKNOWN,):
            s.per_metric_freshness.skip()
            continue
        if expected == Truth.NOT_TRACKED:
            s.per_metric_freshness.skip()
            s.per_metric_not_covered += 1
            continue
        actual = m.metric_periods.get(metric)
        ok = actual == expected
        s.per_metric_freshness.hit(ok, f"{tag}/{metric}: {actual!r} for {expected!r}")
        if not ok:
            s.note(FailureClass.PER_METRIC_FRESHNESS, f"{tag}/{metric}: {actual!r}")

    # -- fallback labelling --------------------------------------------------
    for metric, older in sorted(case.expected_fallback_metrics.items()):
        labelled = m.fallback_labels.get(metric)
        s.fallback_label.hit(labelled == older,
                             f"{tag}/{metric}: labelled {labelled!r}")
    for metric in sorted(m.fallback_labels):
        if metric not in case.expected_fallback_metrics:
            s.fallback_label.hit(False, f"{tag}/{metric}: labelled a fallback and is not one")

    # -- section 13: TTM -----------------------------------------------------
    _score_ttm(case, m, s, tag)

    # -- section 14: completed-period guidance -------------------------------
    if case.guidance:
        expected_realized = () if case.expected_completed_guidance_status == "CURRENT" \
            else tuple(sorted(set(g.fiscal_period for g in case.guidance)
                              - set(case.expected_current_guidance_periods)))
        ok = (tuple(sorted(m.guidance_realized)) == expected_realized
              and tuple(sorted(m.guidance_current))
              == tuple(sorted(case.expected_current_guidance_periods)))
        s.guidance_retirement.hit(
            ok, f"{tag}: realized={m.guidance_realized} current={m.guidance_current}")
        if not ok:
            s.note(FailureClass.GUIDANCE_ACTUALIZATION,
                   f"{tag}: realized={m.guidance_realized}")
    else:
        s.guidance_retirement.skip()

    # -- section 15: actual vs guidance --------------------------------------
    leaked = [p for p in case.forbidden_actual_periods
              if p == m.selected_period or p in m.accepted_periods
              or p in m.current_state_periods]
    s.actual_guidance_separation.hit(not leaked, f"{tag}: {leaked}")
    if leaked:
        s.note(FailureClass.ACTUAL_GUIDANCE_SEPARATION, f"{tag}: {leaked}")
        s.flag(Hard.GUIDANCE_CLASSIFIED_AS_ACTUAL, f"{tag}: {leaked}")

    # -- section 16: same-period supersession --------------------------------
    _score_supersession(case, m, s, tag)

    # -- section 17: DCF base ------------------------------------------------
    if case.expected_dcf_base_status == Truth.UNKNOWN:
        s.dcf_stale_base.skip()
    else:
        seen = {"AHEAD": DcfBase.CURRENT}.get(m.dcf_freshness, m.dcf_freshness)
        ok = seen == case.expected_dcf_base_status
        s.dcf_stale_base.hit(ok, f"{tag}: {m.dcf_freshness!r} for "
                                 f"{case.expected_dcf_base_status!r}")
        if not ok:
            s.note(FailureClass.DCF_BASE_STALE, f"{tag}: {m.dcf_freshness!r}")
        if case.expected_dcf_base_status == DcfBase.STALE and m.dcf_research_valid:
            s.flag(Hard.STALE_DCF_MARKED_VALID_FOR_RESEARCH,
                   f"{tag}: base {case.dcf_base_period_end} against "
                   f"{case.expected_current_period}")

    # -- section 18: research currentness ------------------------------------
    for metric, expected in sorted(case.expected_research_currentness.items()):
        if expected == Truth.UNKNOWN:
            s.research_currentness.skip()
            continue
        called_current = metric in m.research_current
        ok = called_current == (expected == Currentness.CURRENT)
        s.research_currentness.hit(ok, f"{tag}/{metric}: current={called_current}")
        if ok:
            continue
        s.note(FailureClass.RESEARCH_FRESHNESS, f"{tag}/{metric}")
        if called_current:
            s.research_under_qualified += 1
        else:
            s.research_over_qualified += 1

    # -- hard safety ---------------------------------------------------------
    _score_hard(case, m, s, tag)


def _score_ttm(case: ActualizationCase, m: Measurement, s: Score, tag: str) -> None:
    if case.expected_ttm_end_period == Truth.UNKNOWN:
        s.ttm_end_period.skip()
    else:
        ok = m.ttm_reference_end == case.expected_ttm_end_period
        s.ttm_end_period.hit(ok, f"{tag}: {m.ttm_reference_end!r}")
        if not ok:
            s.note(FailureClass.TTM_RECONSTRUCTION, f"{tag}: anchor {m.ttm_reference_end!r}")

    if case.expected_ttm_status == Truth.UNKNOWN:
        s.ttm_status.skip()
    else:
        ok = m.ttm_status == case.expected_ttm_status
        s.ttm_status.hit(ok, f"{tag}: {m.ttm_status!r} for {case.expected_ttm_status!r}")

    if case.expected_ttm_end_period == Truth.UNKNOWN:
        for _metric in FLOW_METRICS:
            s.ttm_window.skip()
        return

    stale = set(case.expected_ttm_stale_metrics)
    for metric in FLOW_METRICS:
        window = m.ttm_windows.get(metric) or {}
        presented_current = bool(window.get("current"))
        end = window.get("end")
        if metric in stale:
            ok = not presented_current
        else:
            ok = presented_current and _same_period(end, case.expected_ttm_end_period)
        s.ttm_window.hit(ok, f"{tag}/{metric}: end={end!r} current={presented_current}")
        if not ok:
            s.note(FailureClass.TTM_RECONSTRUCTION, f"{tag}/{metric}: end={end!r}")

    if case.expected_ttm_window:
        window = m.ttm_windows.get("revenue") or {}
        quarters = tuple(window.get("quarters") or ())
        ok = quarters == tuple(case.expected_ttm_window)
        s.ttm_window.hit(ok, f"{tag}/revenue quarters: {quarters}")
        if not ok:
            s.note(FailureClass.TTM_RECONSTRUCTION,
                   f"{tag}: revenue window {quarters} for {case.expected_ttm_window}")


def _score_supersession(case: ActualizationCase, m: Measurement, s: Score,
                        tag: str) -> None:
    if case.expected_same_period_conflict_status == ConflictStatus.NOT_APPLICABLE:
        s.same_period_supersession.skip()
        s.conflict_detection.skip()
        return

    one_period = len(set(m.current_state_periods)) == 1
    primary_ok = m.selected_primary_source == case.expected_primary_source
    kept_provenance = len(m.superseded_forms) >= 1
    ok = one_period and primary_ok and kept_provenance
    s.same_period_supersession.hit(
        ok, f"{tag}: primary={m.selected_primary_source!r} "
            f"superseded={m.superseded_forms} periods={m.current_state_periods}")
    if not ok:
        s.note(FailureClass.SAME_PERIOD_RECONCILIATION,
               f"{tag}: superseded={m.superseded_forms}")

    expected_conflict = (case.expected_same_period_conflict_status
                         == ConflictStatus.CONFLICT_RECORDED)
    seen_conflict = bool(m.conflict_metrics)
    conflict_ok = seen_conflict == expected_conflict
    if conflict_ok and expected_conflict:
        # A conflict must name the metric that actually differs, not merely
        # exist. Every case that expects one differs on revenue alone.
        conflict_ok = tuple(m.conflict_metrics) == ("revenue",)
    s.conflict_detection.hit(
        conflict_ok, f"{tag}: conflicts={m.conflict_metrics} "
                     f"expected={case.expected_same_period_conflict_status}")
    if not conflict_ok:
        s.note(FailureClass.SAME_PERIOD_RECONCILIATION,
               f"{tag}: conflicts={m.conflict_metrics}")


def _score_hard(case: ActualizationCase, m: Measurement, s: Score,
                tag: str) -> None:
    """The seven counts section 7 requires to be zero for V2."""

    # 1. a period published that is not the one that is current
    if case.expected_current_period != Truth.UNKNOWN and m.current_state_periods:
        for published in set(m.current_state_periods):
            if published != case.expected_current_period:
                s.flag(Hard.WRONG_CURRENT_PERIOD_PUBLISHED,
                       f"{tag}: published {published} for "
                       f"{case.expected_current_period}")

    # 2. GUIDANCE_CLASSIFIED_AS_ACTUAL is flagged in `_score_one` from the
    #    case's forbidden periods, which is where the outlook lives.

    # 3. metrics from more than one period, with nothing saying so
    periods = {m.metric_periods.get(metric) for metric in m.presented_as_current}
    periods.discard(None)
    if len(periods) > 1:
        s.flag(Hard.SILENT_MIXED_PERIOD_STATE,
               f"{tag}: presented as current from {sorted(periods)}")
    # a window presented as the current twelve months while ending elsewhere
    for metric, window in sorted(m.ttm_windows.items()):
        if not window.get("current") or not m.selected_period:
            continue
        if not _same_period(window.get("end"), m.selected_period):
            s.flag(Hard.SILENT_MIXED_PERIOD_STATE,
                   f"{tag}/{metric}: window ends {window.get('end')} while the "
                   f"state is {m.selected_period}")

    # 4. STALE_DCF_MARKED_VALID_FOR_RESEARCH is flagged in `_score_one`.

    # 5. two current states for one economic period
    if len(m.current_state_periods) != len(set(m.current_state_periods)):
        s.flag(Hard.DUPLICATE_CURRENT_STATE_SAME_PERIOD,
               f"{tag}: {m.current_state_periods}")
    overlap = set(m.superseded_forms) and m.selected_primary_source in m.superseded_forms
    if overlap:
        s.flag(Hard.DUPLICATE_CURRENT_STATE_SAME_PERIOD,
               f"{tag}: {m.selected_primary_source} is both primary and superseded")

    # 6. a metric from an older quarter presented as the latest quarter's
    for metric, older in sorted(case.expected_fallback_metrics.items()):
        if metric in m.presented_as_current:
            s.flag(Hard.LATEST_QUARTER_METRIC_FROM_OLDER_QUARTER,
                   f"{tag}/{metric}: {older} presented as current")
    for metric in sorted(m.presented_as_current):
        expected = case.expected_current_metrics.get(metric)
        if expected in (None, Truth.UNKNOWN, Truth.NOT_TRACKED):
            continue
        if expected != case.expected_current_period \
                and m.metric_periods.get(metric) == expected:
            s.flag(Hard.LATEST_QUARTER_METRIC_FROM_OLDER_QUARTER,
                   f"{tag}/{metric}: from {expected}, presented as current")

    # 7. a twelve-month window called current that stops short of the period
    for metric, window in sorted(m.ttm_windows.items()):
        if not window.get("current"):
            continue
        reference = m.selected_period or case.expected_current_period
        lag = _days(window.get("end"), reference)
        if lag is not None and lag > TTM_END_TOLERANCE_DAYS:
            s.flag(Hard.CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD,
                   f"{tag}/{metric}: window ends {window.get('end')}, "
                   f"before {reference}")


# ---------------------------------------------------------------------------
# Reporting (never able to change a measurement)
# ---------------------------------------------------------------------------

ROWS = (
    ("Resolved-period accuracy (period only)", "resolved_period"),
    ("Latest-period accuracy (period+status+source)", "latest_period"),
    ("Primary-source accuracy", "primary_source"),
    ("State-status accuracy", "state_status"),
    ("Per-metric freshness accuracy", "per_metric_freshness"),
    ("Fallback-label accuracy", "fallback_label"),
    ("TTM-anchor accuracy", "ttm_end_period"),
    ("TTM-window accuracy", "ttm_window"),
    ("TTM-status accuracy", "ttm_status"),
    ("Guidance-retirement accuracy", "guidance_retirement"),
    ("Actual-vs-guidance separation", "actual_guidance_separation"),
    ("Same-period supersession", "same_period_supersession"),
    ("Same-period conflict detection", "conflict_detection"),
    ("DCF stale-base detection", "dcf_stale_base"),
    ("Research-currentness accuracy", "research_currentness"),
)


def render_table(scores: Dict[str, Score]) -> str:
    layers = [layer for layer in LAYERS if layer in scores]
    rows = [["Metric"] + layers]
    for label, attribute in ROWS:
        rows.append([label] + [getattr(scores[layer], attribute).text()
                               for layer in layers])
    rows.append(["Hard safety failures"]
                + [str(scores[layer].hard_total) for layer in layers])
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    lines = [" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(rows[0])),
             "-|-".join("-" * width for width in widths)]
    for row in rows[1:]:
        lines.append(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def render_hard(scores: Dict[str, Score]) -> str:
    lines = []
    for name in Hard.ALL:
        cells = " ".join(f"{layer}={scores[layer].hard.get(name, 0)}"
                         for layer in LAYERS if layer in scores)
        lines.append(f"  {name}: {cells}")
    return "\n".join(lines)
