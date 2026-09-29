"""The seam where actual-state resolution is chosen. One decision, one place.

WHY A SEAM AND NOT A REPLACEMENT

`freshness.build_current_financial_state` decides which reported period is
current today, and everything downstream inherits that answer. Swapping it out
wholesale would change every stock report in one step with no way to see what
moved. So the V2 resolver runs BESIDE it and, under the default mode, changes
nothing at all.

    v1        V2 does not run. Byte-for-byte the previous behaviour.
    compare   both run, V1's answer is returned, the difference is recorded.
    v2        V2's answer is used, and FAILS CLOSED when it cannot resolve.

`compare` returning V1's answer is the point rather than a limitation. A
compare mode that quietly preferred the newer resolver would be v2-by-default
wearing a diagnostic's name, and the disagreement it exists to surface would
be the thing it hid.

WHERE IT SITS

At the single call site in `workflow.py` where current financial facts are
selected, before `CanonicalFinancialState` is finalized. The renderer, the
research pipeline and the DCF must never each decide which period is current
-- that is how two parts of one report end up describing different quarters.

FAIL CLOSED IN v2

A resolution that selects no period is not a reason to fall back to V1: the
two layers would then disagree silently and the report would carry whichever
answered last. v2 returns the failure, and the caller decides what an
unresolvable state means.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from tools import config

from finance.actualization import (
    ActualStateResolution,
    ActualStateStatus,
    ResolutionCode,
    reconstruct_ttm,
    resolve_current_actual_state,
)


class ActualizationMode:
    V1 = "v1"
    COMPARE = "compare"
    V2 = "v2"
    ALL = (V1, COMPARE, V2)


class ActualizationFailure(Exception):
    """V2 could not resolve a current state. Never a reason to fall back."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# What a run reports about itself (§6)
# ---------------------------------------------------------------------------

@dataclass
class ActualizationObservation:
    """Structured diagnostics only. No filings, no documents, no prose."""

    mode: str = ActualizationMode.V1
    layer_used: str = ActualizationMode.V1
    v1_period_end: Optional[str] = None
    v1_primary_source: Optional[str] = None
    v2_period_end: Optional[str] = None
    v2_primary_source: Optional[str] = None
    v2_state_status: Optional[str] = None
    v2_statement_completeness: Optional[str] = None
    v2_codes: List[str] = field(default_factory=list)
    period_disagreement: bool = False
    source_disagreement: bool = False
    fallback_metrics: List[str] = field(default_factory=list)
    ttm_reference_end: Optional[str] = None
    ttm_status: Optional[str] = None
    ttm_stale_metrics: List[str] = field(default_factory=list)
    realized_guidance_periods: List[str] = field(default_factory=list)
    rejected_source_count: int = 0
    conflict_metrics: List[str] = field(default_factory=list)
    # How many candidates the Reported Actuals source layer supplied. Zero
    # under its own default, which is what makes this seam inert unless both
    # switches are on.
    source_candidates_offered: int = 0
    failure_code: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = {
            "mode": self.mode,
            "layer_used": self.layer_used,
            "v1_period_end": self.v1_period_end,
            "v1_primary_source": self.v1_primary_source,
        }
        if self.v2_period_end is not None or self.failure_code:
            payload.update({
                "v2_period_end": self.v2_period_end,
                "v2_primary_source": self.v2_primary_source,
                "v2_state_status": self.v2_state_status,
                "v2_statement_completeness": self.v2_statement_completeness,
                "v2_codes": list(self.v2_codes),
                "period_disagreement": self.period_disagreement,
                "source_disagreement": self.source_disagreement,
                "fallback_metrics": list(self.fallback_metrics),
                "conflict_metrics": list(self.conflict_metrics),
                "rejected_source_count": self.rejected_source_count,
                "source_candidates_offered": self.source_candidates_offered,
            })
        if self.ttm_reference_end or self.ttm_status:
            payload.update({"ttm_reference_end": self.ttm_reference_end,
                            "ttm_status": self.ttm_status,
                            "ttm_stale_metrics": list(self.ttm_stale_metrics)})
        if self.realized_guidance_periods:
            payload["realized_guidance_periods"] = list(self.realized_guidance_periods)
        if self.failure_code:
            payload["failure_code"] = self.failure_code
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------

def _v2_resolution(company_facts: dict, as_of: Optional[str],
                   prior_state_metrics: Optional[Dict[str, dict]],
                   extra_candidates: Optional[Sequence] = None
                   ) -> ActualStateResolution:
    from finance.extraction.document_resolver import discover_reported_actuals

    candidates = list(discover_reported_actuals(company_facts or {}))
    # Reported Actuals Source Integration. Candidates read from filed
    # earnings-release exhibits arrive here ALREADY VALIDATED and are simply
    # added to the set the resolver ranks. Nothing about the ranking changes:
    # the release still has to be complete to advance the period, still loses
    # a same-period tie to a periodic filing, and is still preliminary until
    # one arrives. Making the document reachable was the whole gap; the
    # resolver was never the problem.
    candidates.extend(extra_candidates or ())
    return resolve_current_actual_state(
        candidates, as_of=as_of, prior_state_metrics=prior_state_metrics)


def resolve_actual_state(company_facts: dict,
                         v1_period_end: Optional[str] = None,
                         v1_primary_source: Optional[str] = None,
                         as_of: Optional[str] = None,
                         mode: Optional[str] = None,
                         prior_state_metrics: Optional[Dict[str, dict]] = None,
                         extra_candidates: Optional[Sequence] = None,
                         extra_facts: Optional[dict] = None
                         ) -> Tuple[Optional[ActualStateResolution],
                                    ActualizationObservation]:
    """(resolution, observation). The resolution is None under `v1`.

    `v1_period_end` is what the existing planner selected. It is passed in
    rather than recomputed so the comparison is against the answer production
    actually used, not against a second reconstruction of it.
    """
    mode = (mode or config.finance_actualization_mode() or
            ActualizationMode.V1).strip().lower()
    observation = ActualizationObservation(
        mode=mode, v1_period_end=v1_period_end,
        v1_primary_source=v1_primary_source)

    if mode == ActualizationMode.V1:
        return None, observation

    if mode not in ActualizationMode.ALL:
        # Diagnosed as a typo rather than as a resolution failure: reporting
        # "could not resolve" would send a reader to look at the data.
        observation.failure_code = "UNKNOWN_MODE"
        observation.notes.append(
            f"'{mode}' is not a recognised actualization mode; v1 was used")
        return None, observation

    resolution = _v2_resolution(company_facts, as_of, prior_state_metrics,
                                extra_candidates=extra_candidates)
    observation.source_candidates_offered = len(extra_candidates or ())
    observation.v2_period_end = resolution.period_end
    observation.v2_primary_source = (
        resolution.selected_primary_source.form
        if resolution.selected_primary_source else None)
    observation.v2_state_status = resolution.state_status
    observation.v2_statement_completeness = resolution.statement_completeness
    observation.v2_codes = list(resolution.codes)
    observation.rejected_source_count = len(resolution.rejected_candidates)
    observation.conflict_metrics = [c.metric for c in resolution.conflicts]
    observation.fallback_metrics = [f.metric for f in resolution.fallbacks
                                    if f.is_fallback]
    observation.period_disagreement = bool(
        v1_period_end and resolution.period_end
        and v1_period_end != resolution.period_end)
    observation.source_disagreement = bool(
        v1_primary_source and observation.v2_primary_source
        and v1_primary_source != observation.v2_primary_source)

    if resolution.period_end:
        # The twelve-month windows are rebuilt over the facts the SELECTED
        # sources actually reported. `extra_facts` carries the accepted
        # earnings-release figures in the same shape, so a window can follow
        # the period the resolver just advanced to rather than being marked
        # LIMITED for a quarter that WAS reported and merely was not tagged in
        # XBRL. `finance/ttm.py` is untouched and remains the only place a
        # twelve-month window is ever constructed.
        facts_for_ttm = company_facts or {}
        if extra_facts:
            from finance.reported_actuals.candidates import (
                merge_company_facts,
                restrict_to_period,
            )
            # ONLY the period the resolver actually selected. A release whose
            # candidate was refused -- incomplete, superseded, out of range --
            # must not move a twelve-month window through the back door:
            # merging its quarter anyway builds a window ending AFTER the
            # current period, which is then correctly marked LIMITED, so a
            # source the resolver declined would have degraded the very state
            # it was not allowed to advance. Measured on two live issuers.
            facts_for_ttm = merge_company_facts(
                facts_for_ttm, restrict_to_period(extra_facts, resolution.period_end))
        ttm = reconstruct_ttm(facts_for_ttm, resolution.period_end)
        observation.ttm_reference_end = ttm.reference_period_end
        observation.ttm_status = ttm.status
        observation.ttm_stale_metrics = list(ttm.stale_metrics)

    if mode == ActualizationMode.COMPARE:
        # Records, never resolves. V1's answer stands.
        observation.layer_used = ActualizationMode.V1
        return resolution, observation

    # mode == v2
    if not resolution.is_current or not resolution.period_end:
        observation.layer_used = ActualizationMode.V2
        observation.failure_code = (resolution.codes[0] if resolution.codes
                                    else ResolutionCode.NO_REPORTED_ACTUAL)
        observation.notes.append(
            "no current reported period could be resolved; the state is not "
            "replaced and no fallback to v1 is performed")
        raise ActualizationFailure(
            observation.failure_code,
            "; ".join(resolution.resolution_reasons) or
            "no current reported period could be resolved")

    observation.layer_used = ActualizationMode.V2
    return resolution, observation
