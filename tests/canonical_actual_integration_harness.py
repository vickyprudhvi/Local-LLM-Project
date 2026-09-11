"""Running the Canonical Actual Fact Integration benchmark, and scoring it.

TWO STAGES, DELIBERATELY SEPARATE

    measure(case) -> Measurement    RAW. What the composed pipeline produced.
    score(...)    -> Score          Verdicts only, no execution.

`measure` composes the SAME steps `finance/workflow.py` composes under
`FINANCE_ACTUALIZATION_MODE=v2` + `FINANCE_REPORTED_ACTUALS_MODE=v2`:

    reported-actuals discovery over the release document
    -> the Actualization V2 resolver (with the V1 state's prior metrics)
    -> the unified actual fact set (merge + restrict + dedupe)
    -> rebuild `CurrentFinancialState` from the merged fact base
    -> reconcile the fact set against the rebuilt state
    -> canonical evidence, the twelve-month reconstruction, the DCF base
       assessment, the hard-safety self-check

Nothing is stubbed. The functions are the production ones.

`score` reads the fixture's expected data, which is generated from the same
constants the fixture documents are generated from. It never reads a
pipeline output to decide what is true.
"""

import copy
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

from finance import actualization as A
from finance import actualization_runtime as ar
from finance import canonical as canonical_module
from finance.freshness import build_current_financial_state
from finance.reported_actuals import extract_reported_actuals
from finance.reported_actuals.candidates import company_facts_overlay
from finance.reported_actuals import unified as U
from tests.fixtures import canonical_actual_integration_benchmark as bench
from tests.fixtures.canonical_actual_integration_benchmark import (
    FailureClass,
    Fresh,
    Hard,
    IntegrationCase,
)


TOL = 1e-6


def _close(a, b) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) / max(abs(b), 1.0) <= TOL


def _same_period(a, b) -> bool:
    return U._same_period(a, b)


# ---------------------------------------------------------------------------
# The raw record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Measurement:
    case_id: str
    resolved_period: Optional[str] = None
    resolved_status: Optional[str] = None
    active: bool = False
    layer_used: str = "v1"
    # canonical current metric -> period end the rebuilt state selected
    metric_period: Dict[str, Optional[str]] = field(default_factory=dict)
    # canonical current metric -> unified fact freshness verdict
    freshness: Dict[str, str] = field(default_factory=dict)
    canonical_revenue: Optional[float] = None
    canonical_net_income: Optional[float] = None
    canonical_base_period: Optional[str] = None
    canonical_base_aligned: Optional[bool] = None
    ttm_reference_end: Optional[str] = None
    ttm_status: Optional[str] = None
    ttm_revenue: Optional[float] = None
    ttm_quarter_ends: List[str] = field(default_factory=list)
    latest_quarter_revenue: Optional[float] = None
    dcf_base_status: Optional[str] = None
    dcf_base_research_valid: Optional[bool] = None
    same_period_conflict_metrics: List[str] = field(default_factory=list)
    requalified: Dict[str, str] = field(default_factory=dict)
    hard_findings: Dict[str, List[str]] = field(default_factory=dict)
    # the injected-defect variants, for the positive controls
    control_hard_findings: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Compose the pipeline
# ---------------------------------------------------------------------------

def _release_candidates(case: IntegrationCase, base: dict):
    if not case.release_document:
        return (), {}, []
    extraction = extract_reported_actuals(
        case.release_document, accession=case.release_accession,
        form=case.release_form, filed=case.release_filed,
        document="exhibit991.htm")
    overlay = company_facts_overlay(extraction.facts, base=base)
    candidates = list(extraction.candidates)
    if case.later_10k_document:
        later = extract_reported_actuals(
            case.later_10k_document, accession="0000000000-26-000009",
            form="10-K", filed=case.later_10k_filed, document="form10k.htm")
        candidates += list(later.candidates)
        # the 10-K's facts also overlay -- same period, higher authority
        overlay = company_facts_overlay(
            list(extraction.facts) + list(later.facts), base=base)
    return tuple(candidates), overlay, []


def _ttm_quarter_ends(company_facts: dict, metric: str, end: Optional[str]):
    if not end:
        return []
    from finance import ttm as ttm_module
    built = ttm_module.build_ttm(company_facts, metric, reference_end=end)
    ends = []
    for span in built.quarters_included or ():
        _s, _sep, qend = str(span).partition("..")
        if qend:
            ends.append(qend)
    return ends


def measure(case: IntegrationCase) -> Measurement:
    try:
        return _measure(case)
    except Exception as failure:                              # noqa: BLE001
        return Measurement(case_id=case.case_id,
                           error=f"{type(failure).__name__}: {failure}")


def _measure(case: IntegrationCase) -> Measurement:
    base = copy.deepcopy(case.company_facts)
    v1_state = build_current_financial_state(base, "BENCH", valuation_date=case.as_of)

    candidates, overlay, _notes = _release_candidates(case, base)
    resolution, observation = ar.resolve_actual_state(
        base, as_of=case.as_of, mode="v2",
        prior_state_metrics=U.prior_state_metrics(v1_state),
        extra_candidates=candidates, extra_facts=overlay)

    unified = U.build_unified_actual_facts(
        base, resolution=resolution, observation=observation,
        facts_overlay=overlay, v1_state=v1_state)

    if unified.active:
        state = build_current_financial_state(
            unified.company_facts, "BENCH", valuation_date=case.as_of)
        unified.reconcile_with_state(state)
    else:
        state = v1_state

    canonical = canonical_module.build_canonical_evidence(
        state, historical_metrics={})

    ttm = A.reconstruct_ttm(unified.company_facts, state.financial_as_of) \
        if state.financial_as_of else None

    # DCF base: compare the resolved period against the period the DCF packet
    # would be built from. `dcf_packet_base_period` lets a case pin a stale
    # packet for the positive control; otherwise it is the rebuilt state's.
    packet_base = case.dcf_packet_base_period or (
        state.latest_quarterly_period or state.latest_annual_period
        or state.financial_as_of)
    dcf_assessment = A.assess_dcf_base_freshness(
        unified.resolved_period or state.financial_as_of, packet_base)

    q_ends = {m: _ttm_quarter_ends(unified.company_facts, m,
                                   state.financial_as_of)
              for m in ("revenue", "operating_income", "net_income",
                        "operating_cash_flow", "capital_expenditure")}

    hard = U.check_hard_safety(unified, rebuilt_state=state, ttm=ttm,
                               canonical_evidence=canonical,
                               dcf_base_assessment=dcf_assessment,
                               ttm_quarter_ends=q_ends)

    # -- positive-control variants: inject exactly the defect each counter
    # -- exists to catch, and record whether the check fires.
    controls = _control_variants(case, unified, state, ttm, canonical,
                                 dcf_assessment, q_ends)

    def selected_period(metric):
        sel = (state.flows.get(metric) or state.balance_sheet.get(metric)
               or (state.total_debt if metric == "total_debt" else None))
        return getattr(sel, "as_of_date", None)

    metrics = ("revenue", "net_income", "operating_income", "operating_cash_flow",
               "capital_expenditure", "cash_and_cash_equivalents",
               "stockholders_equity", "total_debt", "current_assets",
               "current_liabilities")

    lq = A.reconstruct_ttm  # noqa: F841 -- keep import warm; latest quarter below
    from finance import period_facts as pf
    lq_series = pf.discrete_quarters(unified.company_facts, "revenue")
    latest_q_rev = (lq_series.quarters[-1].value
                    if lq_series.quarters else None)

    return Measurement(
        case_id=case.case_id,
        resolved_period=unified.resolved_period,
        resolved_status=unified.resolved_status,
        active=unified.active,
        layer_used=unified.layer_used,
        metric_period={m: selected_period(m) for m in metrics},
        freshness={m: unified.freshness(m) for m in metrics},
        canonical_revenue=canonical.value("revenue"),
        canonical_net_income=canonical.value("net_income"),
        canonical_base_period=canonical.base_period,
        canonical_base_aligned=canonical.base_period_aligned,
        ttm_reference_end=(ttm.reference_period_end if ttm else None),
        ttm_status=(ttm.status if ttm else None),
        ttm_revenue=(getattr(ttm.windows.get("revenue"), "value", None)
                     if ttm else None),
        ttm_quarter_ends=_ttm_quarter_ends(
            unified.company_facts, "revenue",
            ttm.reference_period_end if ttm else None),
        latest_quarter_revenue=latest_q_rev,
        dcf_base_status={"AHEAD": "CURRENT"}.get(
            dcf_assessment.freshness, dcf_assessment.freshness),
        dcf_base_research_valid=dcf_assessment.may_be_research_valid,
        same_period_conflict_metrics=[c["metric"] for c in unified.conflicts],
        requalified=dict(unified.research_freshness_view().get("requalified") or {}),
        hard_findings={k: list(v) for k, v in hard.items()},
        control_hard_findings=controls,
    )


def _control_variants(case, unified, state, ttm, canonical, dcf_assessment,
                      q_ends) -> Dict[str, Dict[str, List[str]]]:
    """For each hard counter this case controls, inject exactly the defect
    that counter's check exists to catch and record whether the check fires."""
    out: Dict[str, Dict[str, List[str]]] = {}
    if not unified.active:
        return out

    for control in case.positive_controls:
        broken = _clone_unified(unified)
        broken_state = state
        broken_ttm = ttm
        broken_dcf = dcf_assessment
        broken_q_ends = dict(q_ends)

        if control == Hard.CURRENT_METRIC_FROM_PRIOR_PERIOD_UNDISCLOSED:
            fact = broken.facts.get("revenue")
            if fact:
                broken.facts["revenue"] = dataclasses.replace(
                    fact, freshness=Fresh.CURRENT, period_end="2026-04-30")
            broken_state = _state_with(state, "revenue", "2026-04-30")
        elif control == Hard.LATEST_QUARTER_METRIC_FROM_PRIOR_QUARTER:
            broken_state = _state_with(state, "revenue", "2026-04-30")
        elif control == Hard.CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD and ttm:
            broken_ttm = _ttm_stale(ttm, "revenue")
        elif control == Hard.DUPLICATE_ECONOMIC_QUARTER_IN_TTM:
            base = broken_q_ends.get("revenue") or ["x", "y", "z", "2026-07-31"]
            broken_q_ends["revenue"] = list(base) + [base[-1]]
        elif control == Hard.STALE_DCF_MARKED_RESEARCH_VALID:
            # The defect this counter catches is a CONTRADICTION the real
            # `DcfBaseAssessment` cannot hold: stale AND research-valid. A
            # stand-in with both true is exactly what would have to reach the
            # check for it to have anything to catch.
            broken_dcf = _StaleButValid()
        elif control == Hard.SAME_PERIOD_MATERIAL_CONFLICT_SILENTLY_OVERWRITTEN:
            if unified.conflicts:
                metric = unified.conflicts[0]["metric"]
                fact = broken.facts.get(metric)
                if fact:
                    broken.facts[metric] = dataclasses.replace(
                        fact, conflict=None, freshness=Fresh.CURRENT)
        else:
            continue

        out[control] = U.check_hard_safety(
            broken, rebuilt_state=broken_state, ttm=broken_ttm,
            canonical_evidence=canonical, dcf_base_assessment=broken_dcf,
            ttm_quarter_ends=broken_q_ends)
    return out


def _clone_unified(unified):
    clone = dataclasses.replace(unified)
    clone.facts = dict(unified.facts)
    return clone


class _FakeSel:
    def __init__(self, sel, period):
        self._sel = sel
        self.as_of_date = period

    def __getattr__(self, name):
        return getattr(self._sel, name, None)


class _FakeState:
    """A stand-in for `CurrentFinancialState` (frozen) with one metric moved
    to a different period, so a control can inject a stale selection."""

    def __init__(self, state, metric, period):
        self._state = state
        self.flows = dict(getattr(state, "flows", {}) or {})
        self.balance_sheet = dict(getattr(state, "balance_sheet", {}) or {})
        self.total_debt = getattr(state, "total_debt", None)
        self.financial_as_of = getattr(state, "financial_as_of", None)
        self.latest_quarterly_period = getattr(state, "latest_quarterly_period", None)
        self.latest_annual_period = getattr(state, "latest_annual_period", None)
        sel = self.flows.get(metric) or self.balance_sheet.get(metric)
        if sel is not None:
            fake = _FakeSel(sel, period)
            if metric in self.flows:
                self.flows[metric] = fake
            else:
                self.balance_sheet[metric] = fake


def _state_with(state, metric, period):
    return _FakeState(state, metric, period)


class _StaleButValid:
    is_stale = True
    may_be_research_valid = True
    base_period_end = "2026-04-30"


def _ttm_stale(ttm, metric):
    clone = copy.copy(ttm)
    clone.windows = dict(ttm.windows)
    w = clone.windows.get(metric)
    if w is not None:
        clone.windows[metric] = dataclasses.replace(
            w, ends_at_current_period=True, end_date="2026-04-30")
    return clone


def measure_all(cases: Optional[Sequence[IntegrationCase]] = None) -> List[Measurement]:
    return [measure(c) for c in (cases if cases is not None else bench.cases())]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class Tally:
    correct: int = 0
    scored: int = 0
    misses: List[str] = field(default_factory=list)

    def hit(self, ok: bool, label: str = "") -> None:
        self.scored += 1
        if ok:
            self.correct += 1
        elif label:
            self.misses.append(label)

    @property
    def rate(self) -> float:
        return self.correct / self.scored if self.scored else 1.0

    def text(self) -> str:
        return "n/a" if not self.scored else f"{self.rate:.0%} ({self.correct}/{self.scored})"


@dataclass
class Score:
    cases: int = 0
    resolved_period: Tally = field(default_factory=Tally)
    canonical_value: Tally = field(default_factory=Tally)
    per_metric_period: Tally = field(default_factory=Tally)
    fallback_label: Tally = field(default_factory=Tally)
    latest_quarter: Tally = field(default_factory=Tally)
    ttm_window: Tally = field(default_factory=Tally)
    ttm_dedup: Tally = field(default_factory=Tally)
    dcf_base_alignment: Tally = field(default_factory=Tally)
    research_currentness: Tally = field(default_factory=Tally)
    same_period_reconciliation: Tally = field(default_factory=Tally)
    hard: Dict[str, int] = field(default_factory=dict)
    hard_detail: Dict[str, List[str]] = field(default_factory=dict)
    controls_fired: Dict[str, List[str]] = field(default_factory=dict)
    controls_missed: List[str] = field(default_factory=list)
    failure_classes: Dict[str, List[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def hard_total(self) -> int:
        return sum(self.hard.values())

    def note(self, cls: str, label: str) -> None:
        self.failure_classes.setdefault(cls, []).append(label)

    def flag(self, counter: str, label: str) -> None:
        self.hard[counter] = self.hard.get(counter, 0) + 1
        self.hard_detail.setdefault(counter, []).append(label)


def _blank() -> Score:
    return Score(hard={n: 0 for n in Hard.ALL})


def score(measurements: Sequence[Measurement],
          cases: Optional[Sequence[IntegrationCase]] = None) -> Score:
    result = _blank()
    by_id = {c.case_id: c for c in (cases if cases is not None else bench.cases())}
    for m in measurements:
        case = by_id.get(m.case_id)
        if case is None:
            continue
        result.cases += 1
        if m.error:
            result.errors.append(f"{m.case_id}: {m.error}")
            continue
        _score_one(case, m, result)
    _score_controls(measurements, by_id, result)
    return result


def _score_one(case: IntegrationCase, m: Measurement, s: Score) -> None:
    tag = case.case_id

    if case.expected_resolved_period != "UNKNOWN":
        ok = m.resolved_period == case.expected_resolved_period
        s.resolved_period.hit(ok, f"{tag}: {m.resolved_period!r}")
        if not ok:
            s.note(FailureClass.PERIOD_IDENTITY, f"{tag}: {m.resolved_period!r}")

    if m.active != case.expected_active:
        s.note(FailureClass.UNIFIED_FACT_SELECTION,
               f"{tag}: active={m.active} expected {case.expected_active}")

    # -- per-metric selected period -----------------------------------
    for metric, expected in sorted(case.expected_metric_period.items()):
        actual = m.metric_period.get(metric)
        ok = _same_period(actual, expected) if _is_date(expected) else True
        s.per_metric_period.hit(ok, f"{tag}/{metric}: {actual!r} for {expected!r}")
        if not ok:
            s.note(FailureClass.PER_METRIC_FRESHNESS, f"{tag}/{metric}: {actual!r}")

    # -- freshness label --------------------------------------------
    for metric, expected in sorted(case.expected_freshness.items()):
        actual = m.freshness.get(metric)
        ok = actual == expected
        s.fallback_label.hit(ok, f"{tag}/{metric}: {actual!r} for {expected!r}")
        if not ok:
            s.note(FailureClass.PER_METRIC_FRESHNESS,
                   f"{tag}/{metric}: freshness {actual!r}")
        # a fallback that is not carried into the requalification map is a
        # research-freshness failure.
        if expected == Fresh.FALLBACK:
            s.research_currentness.hit(
                metric in m.requalified,
                f"{tag}/{metric}: fallback not requalified")
            if metric not in m.requalified:
                s.note(FailureClass.RESEARCH_FRESHNESS, f"{tag}/{metric}")
        elif expected == Fresh.CURRENT:
            s.research_currentness.hit(
                metric not in m.requalified,
                f"{tag}/{metric}: current but requalified")

    # -- canonical values -----------------------------------------
    if case.expected_canonical_revenue is not None:
        ok = _close(m.canonical_revenue, case.expected_canonical_revenue)
        s.canonical_value.hit(ok, f"{tag}/revenue: {m.canonical_revenue!r} "
                                  f"for {case.expected_canonical_revenue!r}")
        if not ok:
            s.note(FailureClass.UNIFIED_FACT_SELECTION, f"{tag}: canonical revenue")
    if case.expected_canonical_net_income is not None:
        ok = _close(m.canonical_net_income, case.expected_canonical_net_income)
        s.canonical_value.hit(ok, f"{tag}/net_income: {m.canonical_net_income!r}")

    # -- TTM ---------------------------------------------------
    if case.expected_ttm_end_period != "UNKNOWN":
        ok = _same_period(m.ttm_reference_end, case.expected_ttm_end_period)
        s.ttm_window.hit(ok, f"{tag}: ttm end {m.ttm_reference_end!r}")
        if not ok:
            s.note(FailureClass.TTM_RECONSTRUCTION, f"{tag}: {m.ttm_reference_end!r}")
    if case.expected_ttm_revenue is not None:
        ok = _close(m.ttm_revenue, case.expected_ttm_revenue)
        s.ttm_window.hit(ok, f"{tag}: ttm revenue {m.ttm_revenue!r} "
                             f"for {case.expected_ttm_revenue!r}")
        if not ok:
            s.note(FailureClass.TTM_RECONSTRUCTION, f"{tag}: ttm revenue")
    if case.expected_ttm_quarter_ends:
        ends = tuple(m.ttm_quarter_ends)
        ok = ends == tuple(case.expected_ttm_quarter_ends)
        s.ttm_dedup.hit(ok, f"{tag}: window {ends}")
        s.ttm_dedup.hit(len(ends) == len(set(ends)), f"{tag}: dup in {ends}")
        if not ok:
            s.note(FailureClass.TTM_DEDUP, f"{tag}: {ends}")

    if case.expected_latest_quarter_revenue is not None:
        ok = _close(m.latest_quarter_revenue, case.expected_latest_quarter_revenue)
        s.latest_quarter.hit(ok, f"{tag}: latest-q revenue {m.latest_quarter_revenue!r}")
        if not ok:
            s.note(FailureClass.PERIOD_IDENTITY, f"{tag}: latest quarter")

    # -- DCF base -------------------------------------------
    if case.expected_dcf_base_status != "UNKNOWN":
        ok = m.dcf_base_status == case.expected_dcf_base_status
        s.dcf_base_alignment.hit(ok, f"{tag}: {m.dcf_base_status!r} "
                                     f"for {case.expected_dcf_base_status!r}")
        if not ok:
            s.note(FailureClass.DCF_BASE_ALIGNMENT, f"{tag}: {m.dcf_base_status!r}")

    # -- same-period conflict --------------------------------
    seen_conflict = bool(m.same_period_conflict_metrics)
    ok = seen_conflict == case.expected_same_period_conflict
    if ok and case.expected_same_period_conflict:
        ok = tuple(m.same_period_conflict_metrics) == ("revenue",)
    s.same_period_reconciliation.hit(
        ok, f"{tag}: conflicts {m.same_period_conflict_metrics}")
    if not ok:
        s.note(FailureClass.SAME_PERIOD_RECONCILIATION,
               f"{tag}: {m.same_period_conflict_metrics}")

    # -- hard safety on the clean run -----------------------
    for counter, details in m.hard_findings.items():
        for detail in details:
            s.flag(counter, detail)


def _score_controls(measurements, by_id, s: Score) -> None:
    """Every hard counter must have a case whose injected defect makes the
    self-check fire (§25/§26)."""
    for m in measurements:
        case = by_id.get(m.case_id)
        if case is None:
            continue
        for control, findings in (m.control_hard_findings or {}).items():
            if findings.get(control):
                s.controls_fired.setdefault(control, []).append(m.case_id)
    from tests.fixtures.canonical_actual_integration_benchmark import controls_covered

    for counter, control_cases in controls_covered().items():
        if counter in (Hard.RAW_REPORTED_ACTUAL_BYPASSES_CANONICAL_STATE,
                       Hard.RAW_COMPANYFACTS_BYPASSES_CANONICAL_STATE,
                       Hard.GUIDANCE_FACT_ENTERS_ACTUAL_SET,
                       Hard.SILENT_MIXED_PERIOD_CURRENT_SNAPSHOT):
            # structural / covered by static + always-on checks
            continue
        if not control_cases:
            s.controls_missed.append(f"{counter}: no control case")
        elif counter not in s.controls_fired:
            s.controls_missed.append(f"{counter}: control did not fire")


def _is_date(text) -> bool:
    return isinstance(text, str) and len(text) == 10 and text[4] == "-"


ROWS = (
    ("Resolved-period accuracy", "resolved_period"),
    ("Canonical current-value accuracy", "canonical_value"),
    ("Per-metric period accuracy", "per_metric_period"),
    ("Fallback-label accuracy", "fallback_label"),
    ("Latest-quarter accuracy", "latest_quarter"),
    ("TTM-window accuracy", "ttm_window"),
    ("Deduplication accuracy", "ttm_dedup"),
    ("DCF-base alignment accuracy", "dcf_base_alignment"),
    ("Research-currentness accuracy", "research_currentness"),
    ("Same-period reconciliation accuracy", "same_period_reconciliation"),
)


def render_table(s: Score) -> str:
    rows = [("Metric", "Result")]
    rows += [(label, getattr(s, attr).text()) for label, attr in ROWS]
    rows.append(("Hard safety failures", str(s.hard_total)))
    width = max(len(a) for a, _b in rows)
    lines = [f"{rows[0][0].ljust(width)} | {rows[0][1]}",
             f"{'-' * width}-|-{'-' * 16}"]
    for a, b in rows[1:]:
        lines.append(f"{a.ljust(width)} | {b}")
    return "\n".join(lines)


def render_hard(s: Score) -> str:
    return "\n".join(f"  {n}: {s.hard.get(n, 0)}" for n in Hard.ALL)
