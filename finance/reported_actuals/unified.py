"""The unified actual fact set: one merged fact base, anchored on the resolved period.

THE GAP THIS CLOSES

Actualization V2 can resolve period P as current. But the canonical
fact/freshness path -- `finance/freshness.py::build_current_financial_state`
and everything it feeds -- reads the SEC CompanyFacts payload, which does not
contain an earnings release's quarter. So the system could know

    resolved current period = P

while still selecting every downstream number from P-1, and labelling it
current.

WHAT THIS LAYER DOES

It merges the accepted reported-actual facts into the CompanyFacts payload,
restricted to the period the resolver selected and de-duplicated by economic
quarter, and hands the result to the ONE existing freshness engine. The
engine then anchors on P: the balance sheet, the flows, the twelve-month
windows and the DCF base all advance together.

It also records, per canonical metric, WHERE the current figure came from and
whether it is the resolved period's own or an explicitly labelled fallback --
the structured freshness §6 asks for, built from the resolver's own
`MetricFreshness` records rather than re-derived.

WHAT THIS LAYER IS NOT

  * Not a resolver. It never decides which period is current. It is handed
    the resolver's answer and merges facts to match it.
  * Not a second `CanonicalFinancialState`. The caller rebuilds the existing
    one from `.company_facts`; nothing here holds selected values.
  * Not a second freshness engine. `assess_claim_freshness` and
    `build_current_financial_state` do the judging; this assembles their
    input and cross-checks their output.

FAIL CLOSED

Under the default mode nothing merges: `.company_facts` is the payload
unchanged and `.layer_used` is `v1`. A non-USD release fact never enters the
numeric path. A quarter already in CompanyFacts is never added a second time.
"""

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from finance import semantics as sem
from finance.actualization import (
    POINT_IN_TIME_METRICS,
    TTM_END_TOLERANCE_DAYS,
    ActualStateStatus,
)
from finance.reported_actuals.candidates import (
    merge_company_facts,
    restrict_to_period,
)


class FactFreshness:
    """§6's per-metric currentness verdict."""

    CURRENT_PERIOD = "CURRENT_PERIOD"
    FALLBACK_PRIOR_PERIOD = "FALLBACK_PRIOR_PERIOD"
    MISSING = "MISSING"
    CONFLICTED = "CONFLICTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"

    ALL = (CURRENT_PERIOD, FALLBACK_PRIOR_PERIOD, MISSING, CONFLICTED,
           NOT_APPLICABLE)


class HardSafety:
    """§25's counters. Every one must be zero, each with a positive control."""

    CURRENT_METRIC_FROM_PRIOR_PERIOD_UNDISCLOSED = (
        "CURRENT_METRIC_FROM_PRIOR_PERIOD_UNDISCLOSED")
    LATEST_QUARTER_METRIC_FROM_PRIOR_QUARTER = (
        "LATEST_QUARTER_METRIC_FROM_PRIOR_QUARTER")
    CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD = (
        "CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD")
    DUPLICATE_ECONOMIC_QUARTER_IN_TTM = "DUPLICATE_ECONOMIC_QUARTER_IN_TTM"
    STALE_DCF_MARKED_RESEARCH_VALID = "STALE_DCF_MARKED_RESEARCH_VALID"
    RAW_REPORTED_ACTUAL_BYPASSES_CANONICAL_STATE = (
        "RAW_REPORTED_ACTUAL_BYPASSES_CANONICAL_STATE")
    RAW_COMPANYFACTS_BYPASSES_CANONICAL_STATE = (
        "RAW_COMPANYFACTS_BYPASSES_CANONICAL_STATE")
    GUIDANCE_FACT_ENTERS_ACTUAL_SET = "GUIDANCE_FACT_ENTERS_ACTUAL_SET"
    SILENT_MIXED_PERIOD_CURRENT_SNAPSHOT = "SILENT_MIXED_PERIOD_CURRENT_SNAPSHOT"
    SAME_PERIOD_MATERIAL_CONFLICT_SILENTLY_OVERWRITTEN = (
        "SAME_PERIOD_MATERIAL_CONFLICT_SILENTLY_OVERWRITTEN")

    ALL = (CURRENT_METRIC_FROM_PRIOR_PERIOD_UNDISCLOSED,
           LATEST_QUARTER_METRIC_FROM_PRIOR_QUARTER,
           CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD,
           DUPLICATE_ECONOMIC_QUARTER_IN_TTM,
           STALE_DCF_MARKED_RESEARCH_VALID,
           RAW_REPORTED_ACTUAL_BYPASSES_CANONICAL_STATE,
           RAW_COMPANYFACTS_BYPASSES_CANONICAL_STATE,
           GUIDANCE_FACT_ENTERS_ACTUAL_SET,
           SILENT_MIXED_PERIOD_CURRENT_SNAPSHOT,
           SAME_PERIOD_MATERIAL_CONFLICT_SILENTLY_OVERWRITTEN)


# The canonical current metrics this layer tracks. `total_debt` is an
# aggregate the freshness planner derives from components; it is tracked for
# provenance but its currentness follows its components' balance-sheet date.
TRACKED_METRICS = ("revenue", "net_income", "operating_income",
                   "operating_cash_flow", "capital_expenditure",
                   "cash_and_cash_equivalents", "short_term_investments",
                   "current_assets", "current_liabilities",
                   "stockholders_equity", "short_term_debt", "long_term_debt",
                   "total_debt")

# Flow metrics whose "latest quarter" figure must come from the resolved
# quarter (§8). Kept in this layer's own tuple rather than imported, because
# `actualization.LATEST_QUARTER_METRICS` includes ratio names this fact set
# does not carry.
LATEST_QUARTER_FLOW_METRICS = ("revenue", "net_income", "operating_income",
                               "operating_cash_flow", "capital_expenditure")


def _days_between(earlier: Optional[str], later: Optional[str]) -> Optional[int]:
    if not earlier or not later:
        return None
    try:
        return (datetime.date.fromisoformat(later)
                - datetime.date.fromisoformat(earlier)).days
    except ValueError:
        return None


def _same_period(left: Optional[str], right: Optional[str]) -> bool:
    lag = _days_between(left, right)
    return lag is not None and abs(lag) <= TTM_END_TOLERANCE_DAYS


# ---------------------------------------------------------------------------
# The records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UnifiedActualFact:
    """One canonical current metric, with everything §3 asks a fact to carry."""

    metric_id: str
    value: Optional[float]
    period_start: Optional[str]
    period_end: Optional[str]
    period_frequency: str
    flow_or_instant: str
    currency: Optional[str]
    scale: Optional[str]
    basis: str
    source_type: str
    source_document: Optional[str]
    filing_date: Optional[str]
    actual_status: str
    authority: Optional[int]
    freshness: str
    fallback_from_period: Optional[str] = None
    conflict: Optional[dict] = None

    @property
    def is_current(self) -> bool:
        return self.freshness == FactFreshness.CURRENT_PERIOD

    def to_dict(self) -> dict:
        payload = {
            "metric_id": self.metric_id, "value": self.value,
            "period_start": self.period_start, "period_end": self.period_end,
            "period_frequency": self.period_frequency,
            "flow_or_instant": self.flow_or_instant,
            "currency": self.currency, "scale": self.scale, "basis": self.basis,
            "source_type": self.source_type,
            "source_document": self.source_document,
            "filing_date": self.filing_date,
            "actual_status": self.actual_status, "authority": self.authority,
            "freshness": self.freshness,
        }
        if self.fallback_from_period:
            payload["fallback_from_period"] = self.fallback_from_period
        if self.conflict:
            payload["conflict"] = self.conflict
        return payload


@dataclass
class UnifiedActualFactSet:
    """The merged fact base, plus per-metric provenance and self-checks."""

    layer_used: str = "v1"
    resolved_period: Optional[str] = None
    resolved_status: str = ActualStateStatus.REJECTED
    company_facts: dict = field(default_factory=dict)
    facts: Dict[str, UnifiedActualFact] = field(default_factory=dict)
    conflicts: List[dict] = field(default_factory=list)
    merged_periods: Tuple[str, ...] = ()
    dropped_non_usd: Tuple[str, ...] = ()
    notes: List[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        """Did the reported-actual overlay actually change the fact base?"""
        return self.layer_used == "v2" and bool(self.merged_periods)

    def freshness(self, metric: str) -> str:
        fact = self.facts.get(metric)
        return fact.freshness if fact else FactFreshness.MISSING

    def fallback_metrics(self) -> Tuple[str, ...]:
        return tuple(sorted(m for m, f in self.facts.items()
                            if f.freshness == FactFreshness.FALLBACK_PRIOR_PERIOD))

    def research_freshness_view(self) -> dict:
        """`{"requalified": {metric: period}}` for `evidence.apply_research_freshness`.

        A metric that is a fallback or is stale may not wear the word
        "current"; its label is requalified with the period it is actually
        from. A CURRENT_PERIOD or NOT_APPLICABLE metric is left alone.
        """
        requalified = {
            metric: (fact.fallback_from_period or fact.period_end
                     or "an earlier period")
            for metric, fact in self.facts.items()
            if fact.freshness in (FactFreshness.FALLBACK_PRIOR_PERIOD,
                                  FactFreshness.CONFLICTED)
        }
        return {"current_period_end": self.resolved_period,
                "requalified": requalified}

    def reconcile_with_state(self, state) -> None:
        """Make the REBUILT `CurrentFinancialState` the source of truth for
        freshness. §13: one chosen fact per canonical current metric.

        The resolver's records say what it EXPECTED; the freshness engine
        (`build_current_financial_state`) says what was actually SELECTED,
        from the merged fact base. Where they differ, the engine wins -- it is
        the one selection policy, and this layer must not become a second one.
        Aggregate metrics (`total_debt`) have no resolver record of their own
        and take the engine's balance-sheet date directly.
        """
        if state is None or not self.active:
            return
        flows = getattr(state, "flows", {}) or {}
        balance = getattr(state, "balance_sheet", {}) or {}
        for metric in TRACKED_METRICS:
            if metric == "total_debt":
                selection = getattr(state, "total_debt", None)
            else:
                selection = flows.get(metric) or balance.get(metric)
            chosen = getattr(selection, "as_of_date", None)
            value = getattr(selection, "value", None)
            existing = self.facts.get(metric)
            conflict = existing.conflict if existing else None

            if value is None or not chosen:
                freshness = (FactFreshness.CONFLICTED if conflict
                             else FactFreshness.MISSING)
                fallback_from = None
            elif conflict:
                freshness = FactFreshness.CONFLICTED
                fallback_from = None
            elif _same_period(chosen, self.resolved_period):
                freshness = FactFreshness.CURRENT_PERIOD
                fallback_from = None
            else:
                freshness = FactFreshness.FALLBACK_PRIOR_PERIOD
                fallback_from = chosen

            instant = metric in POINT_IN_TIME_METRICS
            self.facts[metric] = UnifiedActualFact(
                metric_id=metric, value=value,
                period_start=getattr(selection, "period_start", None),
                period_end=chosen,
                period_frequency=(sem.PeriodFrequency.INSTANT if instant
                                  else _frequency_of(getattr(selection, "source", None))),
                flow_or_instant=(sem.FlowOrInstant.INSTANT if instant
                                 else sem.FlowOrInstant.FLOW),
                currency=getattr(existing, "currency", None) or "USD",
                scale=None, basis=sem.AccountingBasis.GAAP,
                source_type=getattr(existing, "source_type", "NONE"),
                source_document=getattr(selection, "form", None)
                or getattr(existing, "source_document", None),
                filing_date=getattr(selection, "retrieval_timestamp", None)
                or getattr(existing, "filing_date", None),
                actual_status=(self.resolved_status
                               if freshness == FactFreshness.CURRENT_PERIOD
                               else getattr(existing, "actual_status",
                                            ActualStateStatus.HISTORICAL)),
                authority=getattr(existing, "authority", None),
                freshness=freshness, fallback_from_period=fallback_from,
                conflict=conflict)

    def to_dict(self) -> dict:
        return {
            "layer_used": self.layer_used,
            "active": self.active,
            "resolved_period": self.resolved_period,
            "resolved_status": self.resolved_status,
            "merged_periods": list(self.merged_periods),
            "dropped_non_usd": list(self.dropped_non_usd),
            "facts": {m: f.to_dict() for m, f in sorted(self.facts.items())},
            "conflicts": list(self.conflicts),
            "fallback_metrics": list(self.fallback_metrics()),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Building it
# ---------------------------------------------------------------------------

def prior_state_metrics(state) -> Dict[str, dict]:
    """The V1 state's per-metric periods, in the shape the resolver reads.

    Passed to `resolve_current_actual_state` so its `build_metric_freshness`
    can record a FALLBACK for every metric the newer release does not carry
    -- which is what makes §12's "revenue current, cash fallback" answerable.
    """
    out: Dict[str, dict] = {}
    flows = getattr(state, "flows", {}) or {}
    balance = getattr(state, "balance_sheet", {}) or {}
    total_debt = getattr(state, "total_debt", None)
    for name, selection in list(flows.items()) + list(balance.items()):
        period = getattr(selection, "as_of_date", None)
        if selection is None or getattr(selection, "value", None) is None or not period:
            continue
        out[name] = {
            "period_end": period,
            "form": getattr(selection, "form", None),
            "issued_at": getattr(selection, "retrieval_timestamp", None),
            "source_type": _source_type_for(getattr(selection, "form", None)),
            "finality": "AUDITED",
        }
    if total_debt is not None and getattr(total_debt, "value", None) is not None \
            and getattr(total_debt, "as_of_date", None):
        out["total_debt"] = {
            "period_end": total_debt.as_of_date,
            "form": getattr(total_debt, "form", None),
            "issued_at": getattr(total_debt, "retrieval_timestamp", None),
            "source_type": _source_type_for(getattr(total_debt, "form", None)),
            "finality": "AUDITED",
        }
    return out


def _source_type_for(form: Optional[str]) -> str:
    from finance.actualization import source_type_for
    return source_type_for(form)


def _cap_to_resolved_period(payload: dict, ceiling: Optional[str]
                            ) -> Tuple[dict, List[str]]:
    """Nothing newer than the resolved period may reach the freshness engine.

    `resolve_current_actual_state` already restricts CANDIDATES to `as_of`; a
    candidate reporting a period after that date is rejected outright. This
    is the same ceiling applied to FACTS, after the merge. Without it, a
    CompanyFacts row for a period newer than P -- one the resolver examined
    and correctly rejected as unable to carry the state -- remains visible to
    `build_current_financial_state`, which does not know a period was
    resolved at all and simply picks the newest balance sheet it can see.
    That is how a state ends up claiming period P while its own
    balance-sheet date is P+1.

    Overlay rows are never affected: `restrict_to_period` already limits them
    to exactly the resolved period before this function ever runs. Only rows
    the BASE payload carries for a period newer than P are excluded here.
    """
    import copy

    if not ceiling:
        return payload, []
    capped = copy.deepcopy(payload or {})
    excluded: List[str] = []
    for _taxonomy, concepts in (capped.get("facts") or {}).items():
        for concept, entry in concepts.items():
            for unit, rows in (entry.get("units") or {}).items():
                kept = []
                for row in rows:
                    end = row.get("end")
                    if end and end > ceiling:
                        excluded.append(f"{concept}:{end}")
                        continue
                    kept.append(row)
                entry["units"][unit] = kept
    return capped, excluded


def _dedupe_economic_quarters(payload: dict) -> Tuple[dict, List[str]]:
    """One row per economic quarter before TTM assembly (§10).

    `period_facts._dedupe_by_period` keys on (start, end) and keeps the
    latest-filed. That already collapses a CompanyFacts quarter and an
    overlay quarter with identical dates. This is the belt to that braces:
    two rows for one concept whose END dates are within a few days and whose
    SPANS are both ~a quarter are the same economic quarter reported with a
    day's calendar drift, and only one may enter a twelve-month sum. The
    later filing wins, matching the module's own rule.
    """
    import copy

    merged = copy.deepcopy(payload or {})
    removed: List[str] = []
    for _taxonomy, concepts in (merged.get("facts") or {}).items():
        for concept, entry in concepts.items():
            for unit, rows in (entry.get("units") or {}).items():
                duration_rows = [r for r in rows if r.get("start")]
                instant_rows = [r for r in rows if not r.get("start")]
                kept: List[dict] = []
                for row in sorted(duration_rows,
                                  key=lambda r: (r.get("end") or "",
                                                 r.get("filed") or "")):
                    span = _days_between(row.get("start"), row.get("end"))
                    clash = None
                    for existing in kept:
                        existing_span = _days_between(existing.get("start"),
                                                      existing.get("end"))
                        if span is None or existing_span is None:
                            continue
                        if abs((span or 0) - (existing_span or 0)) > 20:
                            continue
                        if _days_between(existing.get("end"), row.get("end")) is None:
                            continue
                        if abs(_days_between(existing.get("end"),
                                             row.get("end"))) <= 10:
                            clash = existing
                            break
                    if clash is None:
                        kept.append(row)
                        continue
                    # Same economic quarter. Keep whichever was filed later.
                    if (row.get("filed") or "") >= (clash.get("filed") or ""):
                        kept.remove(clash)
                        kept.append(row)
                    removed.append(f"{concept}:{row.get('end')}")
                entry["units"][unit] = instant_rows + kept
    return merged, removed


def build_unified_actual_facts(company_facts: dict,
                               *,
                               resolution=None,
                               observation=None,
                               facts_overlay: Optional[dict] = None,
                               v1_state=None,
                               extractions: Sequence = (),
                               ) -> UnifiedActualFactSet:
    """Assemble the unified fact set from the resolver's answer.

    `resolution` is the `ActualStateResolution`; `observation` the runtime's
    `ActualizationObservation` (its `layer_used` says whether V2's answer is
    authoritative for this run). `facts_overlay` is the reported-actual facts
    in company-facts shape. `v1_state` is the `CurrentFinancialState` built
    from CompanyFacts alone, used only for provenance of fallback metrics.
    """
    layer_used = getattr(observation, "layer_used", "v1") if observation else "v1"
    resolved_period = getattr(resolution, "period_end", None)
    resolved_status = getattr(resolution, "state_status",
                              ActualStateStatus.REJECTED)

    result = UnifiedActualFactSet(
        layer_used=layer_used, resolved_period=resolved_period,
        resolved_status=resolved_status, company_facts=company_facts or {})

    if layer_used != "v2" or resolution is None or not resolved_period:
        # v1 or compare: the fact base is CompanyFacts, unchanged. No overlay,
        # no rebuild -- production behaviour is untouched.
        result.company_facts = company_facts or {}
        return result

    # -- merge the resolved period's facts into the payload ----------------
    trimmed = restrict_to_period(facts_overlay or {}, resolved_period)
    merged = merge_company_facts(company_facts or {}, trimmed)
    merged, deduped = _dedupe_economic_quarters(merged)
    merged, capped = _cap_to_resolved_period(merged, resolved_period)
    result.company_facts = merged
    if capped:
        # Found live on CRWD: CompanyFacts' own revenue concept resolves to
        # NOTHING at any quarter, so the newest reported period had no
        # COMPLETE candidate and the resolver correctly fell back to an
        # older, release-supplied one. CompanyFacts' balance sheet for that
        # newer, rejected period still tags assets/equity/cash, though --
        # and nothing bounded the freshness planner's view of it, so it
        # picked a LATER balance sheet than the period Actualization had
        # just resolved as current. The state then claimed period P while
        # its own balance-sheet date was P+1, reported as "aligned" --
        # SILENT_MIXED_PERIOD_CURRENT_SNAPSHOT. The resolved period is the
        # ceiling: nothing newer may reach the freshness engine, matching
        # the `as_of` ceiling `resolve_current_actual_state` already applies
        # at the candidate level, applied consistently at the fact level.
        result.notes.append(
            f"{len(capped)} fact row(s) newer than the resolved period "
            f"{resolved_period} were excluded from the rebuild: "
            + ", ".join(sorted(set(capped))))
    if deduped:
        result.notes.append(
            f"{len(deduped)} duplicate economic quarter(s) collapsed before TTM: "
            + ", ".join(sorted(set(deduped))))

    overlay_periods = {
        row.get("end")
        for concepts in (trimmed.get("facts") or {}).values()
        for entry in concepts.values()
        for rows in (entry.get("units") or {}).values()
        for row in rows
    }
    result.merged_periods = tuple(sorted(p for p in overlay_periods if p))

    # -- per-metric provenance from the resolver's own records -------------
    by_metric = {f.metric: f for f in getattr(resolution, "fallbacks", [])}
    conflicts = {c.metric: c for c in getattr(resolution, "conflicts", [])}
    result.conflicts = [c.to_dict() for c in getattr(resolution, "conflicts", [])]

    primary = getattr(resolution, "selected_primary_source", None)
    primary_currency = getattr(primary, "currency", None) or "USD"
    primary_form = getattr(primary, "form", None)
    primary_issued = getattr(primary, "issued_at", None)

    for metric in TRACKED_METRICS:
        record = by_metric.get(metric)
        conflict = conflicts.get(metric)
        instant = metric in POINT_IN_TIME_METRICS

        if record is None:
            # The resolver kept no record for this metric -- neither the
            # release nor the prior state carried it. MISSING, and nothing
            # downstream may present it as current.
            result.facts[metric] = UnifiedActualFact(
                metric_id=metric, value=None, period_start=None,
                period_end=None,
                period_frequency=(sem.PeriodFrequency.INSTANT if instant
                                  else sem.PeriodFrequency.UNKNOWN),
                flow_or_instant=(sem.FlowOrInstant.INSTANT if instant
                                 else sem.FlowOrInstant.FLOW),
                currency=None, scale=None, basis=sem.AccountingBasis.GAAP,
                source_type="NONE", source_document=None, filing_date=None,
                actual_status=ActualStateStatus.REJECTED, authority=None,
                freshness=FactFreshness.MISSING)
            continue

        if conflict is not None:
            freshness = FactFreshness.CONFLICTED
        elif record.from_current_period:
            freshness = FactFreshness.CURRENT_PERIOD
        elif record.is_fallback:
            freshness = FactFreshness.FALLBACK_PRIOR_PERIOD
        else:
            freshness = FactFreshness.FALLBACK_PRIOR_PERIOD

        result.facts[metric] = UnifiedActualFact(
            metric_id=metric,
            value=(conflict.authoritative_value if conflict
                   else _fact_value(merged, metric, record.period_end)),
            period_start=None,
            period_end=record.period_end,
            period_frequency=(sem.PeriodFrequency.INSTANT if instant
                              else sem.PeriodFrequency.QUARTER),
            flow_or_instant=(sem.FlowOrInstant.INSTANT if instant
                             else sem.FlowOrInstant.FLOW),
            currency=primary_currency, scale=None,
            basis=sem.AccountingBasis.GAAP,
            source_type=record.source_type,
            source_document=(getattr(primary, "accession", None)
                             if record.from_current_period else record.source_form),
            filing_date=(record.issued_at or primary_issued),
            actual_status=record.actual_status,
            authority=getattr(resolution, "source_authority", None),
            freshness=freshness,
            fallback_from_period=(record.fallback_from_period
                                  if record.is_fallback else None),
            conflict=(conflict.to_dict() if conflict else None))

    return result


def _frequency_of(source: Optional[str]) -> str:
    """The period frequency implied by a `SelectedValue.source` label."""
    return {
        "ttm_calculation": sem.PeriodFrequency.TTM,
        "annual_sec_filing": sem.PeriodFrequency.ANNUAL,
        "quarterly_sec_filing": sem.PeriodFrequency.QUARTER,
    }.get(source or "", sem.PeriodFrequency.UNKNOWN)


def _fact_value(payload: dict, metric: str, period_end: Optional[str]
                ) -> Optional[float]:
    """The value the merged payload actually carries for a metric at a period."""
    if not period_end:
        return None
    try:
        from finance import period_facts as pf
        concept, rows = pf._winning_concept_facts(payload, metric)  # noqa: SLF001
        if not concept:
            return None
        for row in rows:
            if row.get("end") == period_end and row.get("val") is not None:
                return float(row["val"])
    except Exception:                                            # noqa: BLE001
        return None
    return None


# ---------------------------------------------------------------------------
# Hard-safety self-check (§25)
# ---------------------------------------------------------------------------

def check_hard_safety(unified: UnifiedActualFactSet,
                      *,
                      rebuilt_state=None,
                      canonical_evidence=None,
                      ttm=None,
                      dcf_base_assessment=None,
                      ttm_quarter_ends: Optional[Dict[str, Sequence[str]]] = None
                      ) -> Dict[str, List[str]]:
    """Cross-check what the resolver SAID against what the engine SELECTED.

    Returns a map of counter -> the detail lines behind it. An empty map is
    the required outcome. Nothing here re-decides anything; it compares the
    unified fact set, the rebuilt `CurrentFinancialState`, the twelve-month
    reconstruction and the DCF base assessment, and reports where they
    disagree about which period is current.

    Two counters -- RAW_REPORTED_ACTUAL_BYPASSES_CANONICAL_STATE and
    RAW_COMPANYFACTS_BYPASSES_CANONICAL_STATE -- are STRUCTURAL: they are
    about which code path a value took, not about a value. They are asserted
    by a static test over the workflow and reported here as zero.
    """
    findings: Dict[str, List[str]] = {}

    def flag(counter: str, detail: str) -> None:
        findings.setdefault(counter, []).append(detail)

    if not unified.active:
        return findings

    period = unified.resolved_period

    # -- selected period per metric, from the rebuilt state ---------------
    selected: Dict[str, Optional[str]] = {}
    if rebuilt_state is not None:
        for name, sv in (getattr(rebuilt_state, "flows", {}) or {}).items():
            selected[name] = getattr(sv, "as_of_date", None)
        for name, sv in (getattr(rebuilt_state, "balance_sheet", {}) or {}).items():
            selected.setdefault(name, getattr(sv, "as_of_date", None))
        td = getattr(rebuilt_state, "total_debt", None)
        if td is not None:
            selected["total_debt"] = getattr(td, "as_of_date", None)

    # 1. a metric the fact set calls current whose SELECTED period is not P,
    #    and which is not disclosed as a fallback.
    for metric, fact in unified.facts.items():
        chosen = selected.get(metric)
        if fact.freshness == FactFreshness.CURRENT_PERIOD:
            if chosen is not None and period and not _same_period(chosen, period):
                flag(HardSafety.CURRENT_METRIC_FROM_PRIOR_PERIOD_UNDISCLOSED,
                     f"{metric}: called current, selected from {chosen}, "
                     f"resolved period {period}")
        elif fact.freshness in (FactFreshness.FALLBACK_PRIOR_PERIOD,
                                FactFreshness.CONFLICTED):
            # A fallback is fine -- as long as it is actually LABELLED
            # downstream. The research-freshness view is what carries the
            # label; verify the metric is in its requalified map.
            if metric not in (unified.research_freshness_view().get("requalified") or {}):
                flag(HardSafety.CURRENT_METRIC_FROM_PRIOR_PERIOD_UNDISCLOSED,
                     f"{metric}: fallback from {fact.fallback_from_period}, "
                     "not carried in the research-freshness requalification map")

    # 2. a latest-quarter flow metric selected from a prior quarter.
    for metric in LATEST_QUARTER_FLOW_METRICS:
        chosen = selected.get(metric)
        fact = unified.facts.get(metric)
        if not period or chosen is None:
            continue
        if not _same_period(chosen, period) and (
                fact is None or fact.freshness == FactFreshness.CURRENT_PERIOD):
            flag(HardSafety.LATEST_QUARTER_METRIC_FROM_PRIOR_QUARTER,
                 f"{metric}: latest-quarter figure selected from {chosen}, "
                 f"resolved quarter {period}")

    # 3. a twelve-month window presented as current that ends before P.
    for metric, window in (getattr(ttm, "windows", {}) or {}).items():
        if not getattr(window, "ends_at_current_period", False):
            continue
        end = getattr(window, "end_date", None)
        lag = _days_between(end, period)
        if lag is not None and lag > TTM_END_TOLERANCE_DAYS:
            flag(HardSafety.CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD,
                 f"{metric}: window ends {end}, resolved period {period}")

    # 4. a duplicate economic quarter inside a twelve-month window.
    #
    # `reconstruct_ttm`'s `TtmWindow` does not carry its component quarters;
    # `ttm_quarter_ends` supplies them (`ttm.build_ttm(...).quarters_included`,
    # the same the caller renders). A repeated end date means one quarter was
    # counted twice -- a CompanyFacts quarter and an overlay near-duplicate
    # that the economic-quarter dedupe failed to collapse.
    for metric, ends in (ttm_quarter_ends or {}).items():
        ends = list(ends or ())
        if len(ends) != len(set(ends)):
            flag(HardSafety.DUPLICATE_ECONOMIC_QUARTER_IN_TTM,
                 f"{metric}: window quarters {ends}")
        # Two ends within a few days of each other are the same economic
        # quarter with calendar drift -- also a double count.
        for i, left in enumerate(ends):
            for right in ends[i + 1:]:
                lag = _days_between(left, right)
                if lag is not None and 0 < abs(lag) <= 10:
                    flag(HardSafety.DUPLICATE_ECONOMIC_QUARTER_IN_TTM,
                         f"{metric}: {left} and {right} are one quarter")

    # 5. a stale DCF base marked research-valid.
    if dcf_base_assessment is not None:
        if getattr(dcf_base_assessment, "is_stale", False) and \
                getattr(dcf_base_assessment, "may_be_research_valid", False):
            flag(HardSafety.STALE_DCF_MARKED_RESEARCH_VALID,
                 f"base {getattr(dcf_base_assessment, 'base_period_end', None)} "
                 f"against resolved {period}")

    # 8. a guidance (prospective) fact inside the actual set.
    for metric, fact in unified.facts.items():
        if fact.actual_status not in ActualStateStatus.ALL:
            flag(HardSafety.GUIDANCE_FACT_ENTERS_ACTUAL_SET,
                 f"{metric}: actual_status {fact.actual_status}")

    # 9. a current snapshot drawing on more than one period without disclosure.
    current_periods = {
        selected.get(m) for m, f in unified.facts.items()
        if f.freshness == FactFreshness.CURRENT_PERIOD and selected.get(m)
    }
    if len(current_periods) > 1:
        flag(HardSafety.SILENT_MIXED_PERIOD_CURRENT_SNAPSHOT,
             f"current metrics selected from {sorted(current_periods)}")
    if canonical_evidence is not None:
        base = getattr(canonical_evidence, "base_period", None) or \
            (canonical_evidence.get("base_period")
             if isinstance(canonical_evidence, dict) else None)
        aligned = getattr(canonical_evidence, "base_period_aligned", True)
        if isinstance(canonical_evidence, dict):
            aligned = canonical_evidence.get("base_period_aligned", True)
        if base and period and not _same_period(base, period) and aligned:
            flag(HardSafety.SILENT_MIXED_PERIOD_CURRENT_SNAPSHOT,
                 f"canonical base period {base}, resolved {period}, "
                 "reported as aligned")

    # 10. a same-period material conflict that was silently overwritten.
    for conflict in unified.conflicts:
        metric = conflict.get("metric")
        fact = unified.facts.get(metric)
        if fact is None or fact.conflict is None:
            flag(HardSafety.SAME_PERIOD_MATERIAL_CONFLICT_SILENTLY_OVERWRITTEN,
                 f"{metric}: conflict recorded by the resolver but not carried "
                 "into the unified fact set")

    return findings
