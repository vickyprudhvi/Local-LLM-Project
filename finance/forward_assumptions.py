"""Phase H.4 — the ForwardAssumptionBuilder.

THE BUG THIS EXISTS TO FIX
==========================
`propose_assumptions` set every forecast year's revenue growth to ONE number:
the historical revenue CAGR. Not a starting point that fades, not a path —
the same constant for year 1 and year 5, in every scenario. Two things are
wrong with that, and they compound.

First, a historical CAGR is a MEASUREMENT of the past. A DCF growth
assumption is a FORECAST. Copying one into the other is not conservative or
neutral; it silently asserts that the last five years are the next five, and
it does so even when the company has publicly said otherwise. AOS guided to
2-3% sales growth for 2026 while its trailing history supported a higher
number — the DCF used the history and never saw the guidance, because nothing
in the pipeline could represent "management said" as a distinct kind of
evidence.

Second, a flat path is a strong claim in itself. Real forecasts converge:
whatever a company is doing this year, competitive entry and scale drag it
toward a long-run rate. A flat 25% for five years (which is what MLI's
clamped CAGR produced) is not a neutral default — it is an aggressive one.

WHAT THIS MODULE DOES
=====================
It builds a per-YEAR path for growth and margin, from a deterministic
precedence of evidence (section 9), then optionally lets the local model
propose an alternative path within hard bounds — which is then validated,
clamped and recorded, never trusted (sections 8, 10).

Division of authority, unchanged from the rest of this project:

    the model may PROPOSE          -> `propose_paths_with_model`
    this module DECIDES            -> `validate_proposal`
    finance/dcf.py does the maths  -> untouched by anything here

Management guidance is an INPUT, not truth (section 10). A forecast may sit
below, inside or above it — but a base-case year-1 growth that departs
materially from stated guidance has to carry an explicit justification, and
if it does not, the guidance-anchored value is used instead.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import tools.config as config
from finance import freshness as fr
from finance import guidance as gm
from finance import suitability
from finance import semantics as sem
from finance.dcf import AssumptionSourceType
from finance.structural_breaks import HistoricalComparability

# Hard bounds. A proposal outside these is CLAMPED and the clamp recorded --
# never silently accepted, and never rejected outright (a clamped run still
# produces a valuation, it just says so).
GROWTH_BOUNDS = (-0.20, 0.25)
# Phase H.7, section 36 -- THE LOWER MARGIN BOUND USED TO BE +1%.
#
# That floor forbade the model from representing a loss. On a live company
# reporting a -60.1% operating margin it produced a forecast of +1.0% in
# every one of five years: a 61-point swing invented by a bound, turning a
# business burning $3.5B a year into a marginally profitable one, and then
# discounting the result into a positive value per share. The DCF validated,
# because the arithmetic was fine.
#
# A bound exists to reject the absurd, not to legislate profitability.
# finance/dcf.py models a negative operating margin perfectly well -- EBIT
# simply comes out negative and flows through NOPAT into FCFF -- so the floor
# was never protecting the engine from anything. It is now set wide enough to
# represent a company losing as much as it earns, which is far outside any
# real operating result and still bounded.
#
# What catches genuinely unmodellable companies is no longer this floor but
# finance/suitability.py, which says so explicitly instead of silently
# rewriting the input.
MARGIN_BOUNDS = (-1.00, 0.60)

# How far a base-case year-1 growth may sit outside stated guidance before it
# needs an explicit, evidence-backed justification (section 10's "2%-3%
# guidance, 7.2% base must require explicit justification").
GUIDANCE_DIVERGENCE_TOLERANCE = 0.01


@dataclass(frozen=True)
class AssumptionEntry:
    """One assumption, for one forecast year, with full provenance (section 11)."""

    field: str
    forecast_year: int
    scenario: str
    value: float
    units: str
    source_type: str
    evidence_ids: Tuple[str, ...]
    derivation: str
    approval_status: str = "proposed"
    clamped: bool = False
    original_proposed_value: Optional[float] = None
    applied_value: Optional[float] = None
    # Section 19. `raw_value` is what the evidence actually implied before any
    # bound was applied; `clamp_reason` says which bound bit and why the
    # applied value is therefore the model's limit rather than an estimate.
    raw_value: Optional[float] = None
    clamp_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "forecast_year": self.forecast_year,
            "scenario": self.scenario,
            "value": self.value,
            "units": self.units,
            "source_type": self.source_type,
            "evidence_ids": list(self.evidence_ids),
            "derivation": self.derivation,
            "approval_status": self.approval_status,
            "clamped": self.clamped,
            "original_proposed_value": self.original_proposed_value,
            "applied_value": self.applied_value if self.applied_value is not None else self.value,
            "raw_growth": self.raw_value,
            "applied_growth": (self.applied_value if self.applied_value is not None
                               else self.value),
            "clamp_reason": self.clamp_reason,
        }


@dataclass
class ForwardPath:
    """A per-year path plus how it was arrived at."""

    field: str
    values: List[float] = field(default_factory=list)
    entries: List[AssumptionEntry] = field(default_factory=list)
    anchor_source: str = AssumptionSourceType.CONFIGURED_DEFAULT
    anchor_value: Optional[float] = None
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic evidence precedence (section 9)
# ---------------------------------------------------------------------------


# A guidance record's own period vocabulary, mapped to the semantic one. The
# extractor says "annual"/"quarter"; `finance.semantics` reasons in
# frequencies, and the translation belongs in one place.
_PERIOD_TYPE_FREQUENCY = {
    "annual": sem.PeriodFrequency.ANNUAL,
    "fiscal_year": sem.PeriodFrequency.ANNUAL,
    "quarter": sem.PeriodFrequency.QUARTER,
    "quarterly": sem.PeriodFrequency.QUARTER,
    "half_year": sem.PeriodFrequency.HALF_YEAR,
    "multi_year": sem.PeriodFrequency.MULTI_YEAR,
}


@dataclass
class GrowthEvidence:
    """Every growth signal available, each labelled by KIND.

    Kept as separate named fields rather than a single "growth" number
    because section 7's whole point is that these are not interchangeable.
    The assumption builder chooses between them by explicit precedence; the
    research pipeline shows them side by side.

    Phase H.6 adds the distinction that the live AT&T run got wrong. There is
    now a `guidance_source_metric`, and a growth signal taken from a REVENUE
    COMPONENT (service revenue) is held in its own fields rather than in the
    consolidated ones — so "management guided 3-4%" can never again mean
    "management guided 3-4% of the thing the DCF is about" unless it does.
    """

    guidance_low: Optional[float] = None
    guidance_high: Optional[float] = None
    guidance_fiscal_year: Optional[int] = None
    guidance_evidence_id: Optional[str] = None
    # Which taxonomy metric the consolidated guidance above came from. Always
    # `revenue_growth` when set, because nothing else may populate it — the
    # field exists so provenance can SAY so rather than leaving a reader to
    # assume it.
    guidance_source_metric: Optional[str] = None
    guidance_period_label: Optional[str] = None
    guidance_period_type: Optional[str] = None
    guidance_basis: Optional[str] = None
    guidance_bound_type: Optional[str] = None

    # Consolidated revenue guidance for a SHORTER horizon than the assumption
    # -- a next-quarter growth rate, most often. Held in its own fields
    # rather than in the ones above, because the slot above is what a
    # twelve-month consumer reads and a quarterly rate is not a twelve-month
    # rate. Nothing is discarded: the value, its target period and its
    # evidence id all survive here, and it stays citable research evidence
    # and directional corroboration.
    near_term_guidance_low: Optional[float] = None
    near_term_guidance_high: Optional[float] = None
    near_term_guidance_period_label: Optional[str] = None
    near_term_guidance_period_type: Optional[str] = None
    near_term_guidance_evidence_id: Optional[str] = None
    near_term_guidance_source_metric: Optional[str] = None

    # A revenue COMPONENT's guidance (service, product, segment). Supporting
    # forward evidence only: it may inform a forecast and must be cited as
    # what it is, but it is never consolidated revenue guidance (section 7).
    component_guidance_low: Optional[float] = None
    component_guidance_high: Optional[float] = None
    component_guidance_metric: Optional[str] = None
    component_guidance_evidence_id: Optional[str] = None
    component_guidance_period_label: Optional[str] = None

    # Section 8. Guidance stated as an ABSOLUTE revenue amount for the next
    # quarter, converted to a growth rate against the comparable prior-year
    # quarter. NVIDIA guides exactly this way ("Revenue is expected to be
    # $91.0 billion, plus or minus 2%" for Q2 FY2027) and the figure is the
    # single most current forward statement about the company — but it
    # describes ONE QUARTER, so it is held in its own field and never
    # becomes the five-year anchor by default.
    guidance_implied_next_period_growth: Optional[float] = None
    guidance_implied_comparison_period: Optional[str] = None
    # The horizon that derivation actually covers. Stored rather than
    # inferred: everything downstream that decides what this figure may be
    # used for reads it, and a field that has to be guessed from which
    # attribute happens to be populated is the kind of implicit rule this
    # architecture exists to remove.
    guidance_implied_period_frequency: str = sem.PeriodFrequency.QUARTER

    # Phase H.10. Every guidance-to-growth derivation the semantic validator
    # refused, with the two identities and the reason. Held here so readiness
    # can name the ROOT cause (a period mismatch) instead of the symptom a
    # reader would otherwise see (a clamped assumption), and so a rejection
    # is never silently indistinguishable from "the company guided nothing".
    semantic_rejections: List[dict] = field(default_factory=list)

    ttm_yoy: Optional[float] = None
    latest_annual_yoy: Optional[float] = None
    historical_cagr: Optional[float] = None
    historical_periods: Tuple[str, ...] = ()
    # Section 13. When the history spans a structural break, the long-period
    # CAGR drops BELOW the recent trend in the precedence below.
    historical_comparability: str = "UNKNOWN"
    comparability_summary: str = ""
    # Growth on the post-break periods only, when a break was found and there
    # are enough comparable years left to measure one.
    post_break_cagr: Optional[float] = None
    post_break_periods: Tuple[str, ...] = ()

    # -- forecast-horizon eligibility, as structured metadata ------------
    #
    # Each forward signal answers the same question for itself: may it set an
    # assumption's MAGNITUDE, or only corroborate a DIRECTION? Computed from
    # the horizon the figure covers, never asserted in prose.

    @property
    def guidance_period_frequency(self) -> str:
        return _PERIOD_TYPE_FREQUENCY.get(
            (self.guidance_period_type or "").lower(), sem.PeriodFrequency.UNKNOWN)

    @property
    def guidance_eligibility(self) -> "sem.ForecastEligibility":
        """What the CONSOLIDATED revenue guidance may do to the assumption."""
        return sem.forecast_compatibility(
            evidence_metric=sem.MetricIdentity.REVENUE,
            evidence_frequency=self.guidance_period_frequency,
            assumption_metric=sem.MetricIdentity.REVENUE,
            assumption_frequency=sem.PeriodFrequency.ANNUAL)

    @property
    def guidance_forecast_compatibility(self) -> str:
        return self.guidance_eligibility.status

    @property
    def implied_growth_eligibility(self) -> "sem.ForecastEligibility":
        """What the growth DERIVED from a guided level may do.

        Always the shorter horizon in practice -- the twelve-month derivation
        lands in `guidance_low` instead -- but read from the stored frequency
        so a future derivation over a different window is classified by what
        it covers rather than by where it was put.
        """
        return sem.forecast_compatibility(
            evidence_metric=sem.MetricIdentity.REVENUE,
            evidence_frequency=self.guidance_implied_period_frequency,
            assumption_metric=sem.MetricIdentity.REVENUE,
            assumption_frequency=sem.PeriodFrequency.ANNUAL)

    @property
    def implied_growth_forecast_compatibility(self) -> str:
        return self.implied_growth_eligibility.status

    @property
    def ttm_eligibility(self) -> "sem.ForecastEligibility":
        """Trailing twelve months against a twelve-month assumption."""
        return sem.forecast_compatibility(
            evidence_metric=sem.MetricIdentity.REVENUE,
            evidence_frequency=sem.PeriodFrequency.TTM,
            assumption_metric=sem.MetricIdentity.REVENUE,
            assumption_frequency=sem.PeriodFrequency.ANNUAL)

    @property
    def near_term_guidance_midpoint(self) -> Optional[float]:
        if self.near_term_guidance_low is None or self.near_term_guidance_high is None:
            return None
        return (self.near_term_guidance_low + self.near_term_guidance_high) / 2.0

    @property
    def near_term_guidance_eligibility(self) -> "sem.ForecastEligibility":
        return sem.forecast_compatibility(
            evidence_metric=sem.MetricIdentity.REVENUE,
            evidence_frequency=_PERIOD_TYPE_FREQUENCY.get(
                (self.near_term_guidance_period_type or "quarter").lower(),
                sem.PeriodFrequency.QUARTER),
            assumption_metric=sem.MetricIdentity.REVENUE,
            assumption_frequency=sem.PeriodFrequency.ANNUAL)

    @property
    def guidance_midpoint(self) -> Optional[float]:
        if self.guidance_low is None or self.guidance_high is None:
            return None
        return (self.guidance_low + self.guidance_high) / 2.0

    @property
    def component_guidance_midpoint(self) -> Optional[float]:
        if self.component_guidance_low is None or self.component_guidance_high is None:
            return None
        return (self.component_guidance_low + self.component_guidance_high) / 2.0

    @property
    def history_is_broken(self) -> bool:
        return self.historical_comparability == HistoricalComparability.STRUCTURAL_BREAK

    @property
    def historical_span(self) -> str:
        """The exact periods the historical CAGR covers.

        Always stated alongside the figure, because a growth rate without its
        span is not checkable — and because this project currently has two
        differently-computed historical CAGRs (see `collect_growth_evidence`).
        """
        if not self.historical_periods:
            return "an unstated period"
        first = self.historical_periods[0].split("..")[0]
        last = self.historical_periods[-1].split("..")[-1]
        return f"{first} to {last} ({len(self.historical_periods)} reported years)"


# Target-period types that cover twelve months or more, and so can answer a
# question an annual assumption asks.
_ANNUAL_TARGET_TYPES = ("CURRENT_FISCAL_YEAR", "NEXT_FISCAL_YEAR", "MULTI_YEAR")


def _statement_for(statements, name_keyed, name, prefer_annual=True):
    """One statement of `name`, choosing the horizon the caller needs.

    Reads the multi-horizon list when the resolver supplied one and falls
    back to the name-keyed view otherwise, so a replayed artifact or a
    hand-built fixture keeps working unchanged.
    """
    candidates = [m for m in (statements or [])
                  if isinstance(m, dict) and m.get("name") == name
                  and m.get("low") is not None]
    if not candidates:
        entry = name_keyed.get(name)
        return entry if isinstance(entry, dict) and entry.get("low") is not None else None
    annual = [m for m in candidates
              if m.get("target_period_type") in _ANNUAL_TARGET_TYPES]
    shorter = [m for m in candidates if m not in annual]
    if prefer_annual:
        return (annual or shorter)[0]
    return (shorter or annual)[0]


def collect_growth_evidence(state: "fr.CurrentFinancialState",
                            company_facts: Optional[dict] = None,
                            comparability: Optional[dict] = None) -> GrowthEvidence:
    """Assemble every growth signal, WITHOUT choosing between them yet.

    The guidance step is where the AT&T failure landed, so it is now explicit
    about what it will and will not accept. `revenue_growth` is read ONLY if
    the taxonomy says that key is consolidated revenue growth
    (`may_anchor_revenue_growth`), and a revenue COMPONENT's guidance is kept
    in separate fields. An EBITDA-growth figure sitting in the guidance
    record cannot reach the consolidated fields by any path.
    """
    evidence = GrowthEvidence()

    published = state.management_guidance or {}
    guidance = published.get("metrics") or {}
    # Every current statement, when the resolver supplied them. The
    # name-keyed view holds one horizon per metric; a company that guided
    # both a quarter and a full year has two, and reading only the first
    # is what lost the annual outlook.
    statements = published.get("all_metrics") or []

    revenue_growth = _statement_for(
        statements, guidance, gm.GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH,
        prefer_annual=True)
    near_term = _statement_for(
        statements, guidance, gm.GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH,
        prefer_annual=False)
    if near_term is not None and near_term is not revenue_growth:
        evidence.near_term_guidance_low = float(near_term["low"])
        evidence.near_term_guidance_high = float(near_term["high"])
        evidence.near_term_guidance_period_label = near_term.get("fiscal_period")
        evidence.near_term_guidance_period_type = near_term.get("period_type")
        evidence.near_term_guidance_evidence_id = near_term.get("evidence_id")
        evidence.near_term_guidance_source_metric = near_term.get("name")

    if isinstance(revenue_growth, dict) and revenue_growth.get("low") is not None:
        source_metric = revenue_growth.get("name") or \
            gm.GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH
        # Phase 12/15. A long-term framework or an aspiration may not set a
        # forecast for a named period, however precisely it is stated. Both
        # remain in the evidence as context; neither becomes the number a
        # valuation is built on for a specific year.
        forward_kind = revenue_growth.get("forward_kind")
        anchorable = (gm.may_anchor_period_forecast(forward_kind)
                      if forward_kind else True)
        if not anchorable:
            evidence.semantic_rejections.append({
                "code": "GUIDANCE_NOT_PERIOD_COMMITMENT",
                "operation": sem.Operation.DCF_INPUT,
                "left": f"revenue_growth guidance ({forward_kind})",
                "right": "year-1 revenue growth assumption",
                "reason": (
                    f"This forward statement is a {forward_kind.replace('_', ' ').lower()} "
                    "rather than guidance for a named period, so it does not commit the "
                    "company to a figure for any particular year and cannot anchor the "
                    "forecast."),
                "context": "guidance -> year-1 revenue growth",
            })
        # Phase 2/3/12. The metric IDENTITY is checked before any value is
        # assigned, and the rejection happens here -- before a number exists
        # for a clamp to act on. A clamp must never repair a semantic error:
        # it would turn a measurement of the wrong quantity into a plausible
        # one, which is harder to notice than the original mistake.
        metric_ok, metric_reason = gm.validate_revenue_growth_source(source_metric)
        if anchorable and not metric_ok:
            evidence.semantic_rejections.append({
                "code": gm.GUIDANCE_METRIC_MISMATCH,
                "operation": sem.Operation.DCF_INPUT,
                "left": f"guidance metric {source_metric!r}",
                "right": "year-1 revenue growth assumption",
                "reason": metric_reason,
                "context": "guidance -> year-1 revenue growth",
            })
        if anchorable and metric_ok and gm.may_anchor_revenue_growth(source_metric):
            period_type = revenue_growth.get("period_type")
            frequency = _PERIOD_TYPE_FREQUENCY.get(
                (period_type or "").lower(), sem.PeriodFrequency.UNKNOWN)
            comparable = sem.forecast_compatibility(
                evidence_metric=sem.MetricIdentity.REVENUE,
                evidence_frequency=frequency,
                assumption_metric=sem.MetricIdentity.REVENUE,
                assumption_frequency=sem.PeriodFrequency.ANNUAL).may_set_magnitude
            if comparable:
                evidence.guidance_low = float(revenue_growth["low"])
                evidence.guidance_high = float(revenue_growth["high"])
                evidence.guidance_fiscal_year = revenue_growth.get("fiscal_year")
                evidence.guidance_evidence_id = revenue_growth.get("evidence_id")
                evidence.guidance_source_metric = source_metric
                evidence.guidance_period_label = revenue_growth.get("fiscal_period")
                evidence.guidance_period_type = period_type
                evidence.guidance_basis = revenue_growth.get("basis")
                evidence.guidance_bound_type = revenue_growth.get("bound_type")
            elif evidence.near_term_guidance_low is None:
                # A stated growth rate for a shorter period. Real guidance,
                # kept whole, and not the twelve-month rate the assumption
                # slot is for.
                evidence.near_term_guidance_low = float(revenue_growth["low"])
                evidence.near_term_guidance_high = float(revenue_growth["high"])
                evidence.near_term_guidance_period_label = \
                    revenue_growth.get("fiscal_period")
                evidence.near_term_guidance_period_type = period_type
                evidence.near_term_guidance_evidence_id = \
                    revenue_growth.get("evidence_id")
                evidence.near_term_guidance_source_metric = source_metric

    # A revenue COMPONENT's growth guidance, when the company guided one.
    # AT&T guides service revenue, which is most of but not all of
    # consolidated revenue; the distinction is preserved rather than
    # collapsed (section 7).
    # Only a COMPANY-WIDE revenue split may anchor. A SEGMENT's guidance is
    # recorded as evidence but never becomes the consolidated anchor: AT&T
    # guides "Advanced Connectivity service revenue growth of 5%+", which is
    # one business line inside one split, and letting it set consolidated
    # growth would be a subtler version of the same substitution that made
    # EBITDA growth into revenue growth.
    for name in (gm.GuidanceMetricName.SERVICE_REVENUE_GROWTH,
                 gm.GuidanceMetricName.PRODUCT_REVENUE_GROWTH):
        entry = guidance.get(name)
        if isinstance(entry, dict) and entry.get("low") is not None:
            evidence.component_guidance_low = float(entry["low"])
            evidence.component_guidance_high = float(entry["high"])
            evidence.component_guidance_metric = name
            evidence.component_guidance_evidence_id = entry.get("evidence_id")
            evidence.component_guidance_period_label = entry.get("fiscal_period")
            break

    # Absolute revenue guidance -> an implied growth rate, when the period it
    # covers can be matched against a reported comparable period.
    # `evidence.guidance_low is None` used to gate this, which meant ANY
    # consolidated growth guidance suppressed the derivation -- including a
    # quarterly rate that cannot set an annual assumption. A company guiding
    # 27-29% for next quarter and $118-120B for the year lost the annual
    # outlook entirely, and with it the annual growth it implies and the
    # model-bound assessment that growth would have triggered. The gate is
    # now what it always meant: derive unless a COMPARABLE annual signal is
    # already in hand.
    revenue_amount = _statement_for(
        statements, guidance, gm.GuidanceMetricName.CONSOLIDATED_REVENUE,
        prefer_annual=True)
    if isinstance(revenue_amount, dict) and revenue_amount.get("midpoint") is not None \
            and company_facts and evidence.guidance_low is None:
        _apply_absolute_revenue_guidance(evidence, revenue_amount, company_facts)

    if comparability:
        evidence.historical_comparability = comparability.get(
            "historical_comparability_status", HistoricalComparability.UNKNOWN)
        evidence.comparability_summary = comparability.get("summary", "")

    # TTM-over-TTM: the only growth measure that compares twelve months
    # against an equivalent twelve months, so it is not distorted by
    # seasonality the way a quarter-over-quarter figure is.
    if company_facts:
        current = fr.build_ttm(company_facts, "revenue")
        prior = fr.build_ttm(company_facts, "revenue", offset=4)
        if current.ok and prior.ok and prior.value:
            evidence.ttm_yoy = (current.value - prior.value) / abs(prior.value)

        from finance import period_facts as pf
        # Capped to the SAME history window the rest of the report uses, so a
        # reader never sees two different "historical CAGR" figures for one
        # company (companyfacts carries 10+ years; fundamental_metrics uses 5).
        annual = pf.annual_periods(company_facts, "revenue")[
            -config.dcf_assumption_history_max_years():]
        if len(annual) >= 2 and annual[-2].value:
            evidence.latest_annual_yoy = (
                (annual[-1].value - annual[-2].value) / abs(annual[-2].value))
        if len(annual) >= 2:
            oldest, newest = annual[0], annual[-1]
            years = len(annual) - 1
            if oldest.value and oldest.value > 0 and years > 0:
                evidence.historical_cagr = (newest.value / oldest.value) ** (1.0 / years) - 1.0
                evidence.historical_periods = tuple(
                    f"{p.start}..{p.end}" for p in annual)
                # NOTE: this CAGR is computed over the DATE-KEYED annual
                # series (finance/period_facts.py::annual_periods), which is
                # contiguous by construction. It can therefore DISAGREE with
                # `fundamental_metrics.revenue_cagr`, which is computed from
                # the (fiscal_year, fiscal_period)-bucketed history and can
                # silently skip a year -- live AOS produces the series 2025,
                # 2023, 2022, 2021, 2020 there (FY2024 missing, because a
                # 10-K's `fy` tag describes the FILING, not the fact), and
                # then divides a five-year span by four intervals: 7.25%
                # against a true ~2%. The span is spelled out in the
                # derivation below so the two figures are never mistaken for
                # each other. Fixing the bucketed history itself is a
                # separate change to finance/xbrl_mapping.py::
                # extract_statements and is NOT done here.

        # Section 13: when the history spans a structural break, measure the
        # growth of the periods that are ACTUALLY comparable as well. This is
        # not a replacement for the CAGR — it is the figure a reader needs in
        # order to see how much of the CAGR is the break.
        comparable_from = (comparability or {}).get("comparable_from")
        if comparable_from and company_facts:
            from finance import period_facts as pf_module
            post = [p for p in pf_module.annual_periods(company_facts, "revenue")
                    if p.end and p.end >= comparable_from]
            if len(post) >= 2 and post[0].value and post[0].value > 0:
                years = len(post) - 1
                evidence.post_break_cagr = (
                    (post[-1].value / post[0].value) ** (1.0 / years) - 1.0)
                evidence.post_break_periods = tuple(f"{p.start}..{p.end}" for p in post)

    historical = state.historical_metrics or {}
    if evidence.historical_cagr is None and historical.get("revenue_cagr") is not None:
        evidence.historical_cagr = float(historical["revenue_cagr"])
    if evidence.latest_annual_yoy is None and historical.get("revenue_growth_yoy") is not None:
        evidence.latest_annual_yoy = float(historical["revenue_growth_yoy"])
    return evidence


# Scale words a release uses, as multipliers onto the raw number. XBRL facts
# are in units; a release says "$91.0 billion".
_SCALE_MULTIPLIER = {"billion": 1e9, "billions": 1e9, "million": 1e6, "millions": 1e6}


GUIDANCE_PERIOD_INCOMPATIBLE = "GUIDANCE_PERIOD_INCOMPATIBLE"

# How far a guided level may sit from the issuer's own reported figures for
# the period it CLAIMS to cover before the claim is disbelieved. A company
# guiding its full year names a number in the neighbourhood of its own annual
# revenue; one that names a quarter's worth of revenue and calls it the year
# has been misread somewhere upstream.
_SCALE_DISAGREEMENT = 0.45


def _guidance_scale_contradicts_period(guided_amount, period_type, company_facts):
    """Does the issuer's own history contradict the period this guidance claims?

    Defence in depth, and the part of this fix that generalises. A regex
    corrected for one release wording will be wrong again for the next; what
    cannot go stale is the company's own reported scale. An ANNUAL revenue
    figure roughly one quarter the size of the issuer's trailing twelve
    months is not that issuer's annual revenue, whatever the surrounding
    prose says -- and a live release headed "third quarter FY2026 targets"
    produced exactly that: a quarterly level read as full-year guidance,
    divided by the trailing twelve months, and reported as a -73% decline.

    Returns a reason string when the label is contradicted, else None.
    Deliberately one-sided -- it rejects a figure far too SMALL for the
    period claimed and never one that is large, because a company growing
    quickly is not evidence of a parsing error.
    """
    from finance import period_facts as pf_module

    ttm = fr.build_ttm(company_facts, "revenue")
    if not (ttm.ok and ttm.value):
        return None
    if period_type != "annual":
        return None

    ratio = guided_amount / abs(ttm.value)
    if ratio >= _SCALE_DISAGREEMENT:
        return None

    series = pf_module.discrete_quarters(company_facts, "revenue")
    quarterly = [q.value for q in series.quarters[-4:] if q.value]
    if quarterly:
        typical = sorted(quarterly)[len(quarterly) // 2]
        if typical and abs(guided_amount - typical) < abs(guided_amount - ttm.value):
            return ("the figure is {:.0%} of the issuer's trailing-twelve-month revenue and "
                    "closer to a single reported quarter than to a year, so the full-year "
                    "period it was read as covering is contradicted by the issuer's own "
                    "filings".format(ratio))
    return ("the figure is only {:.0%} of the issuer's trailing-twelve-month revenue, which "
            "is not a plausible full-year revenue level for this issuer".format(ratio))


def _annual_growth_bases(company_facts):
    """Twelve-month revenue bases an ANNUAL outlook may be measured against.

    Yields (value, frequency, start, end, definition_id), most current first.
    Each is a real reported window carrying its own dates, so the semantic
    validator can check the pair rather than trusting the label this function
    put on it.
    """
    from finance import period_facts as pf_module

    bases = []
    ttm = fr.build_ttm(company_facts, "revenue")
    if ttm.ok and ttm.value:
        bases.append((ttm.value, sem.PeriodFrequency.TTM,
                      ttm.period_start, ttm.period_end, "reported_ttm_revenue"))

    # The latest reported FISCAL YEAR. An annual outlook is usually written
    # against exactly this -- "FY2027 revenue of $118-120 billion" means
    # against FY2026's actual -- and an issuer whose quarterly series cannot
    # build a trailing window still has it. Without this the whole annual
    # derivation was silently unavailable for such an issuer.
    annual = [p for p in pf_module.annual_periods(company_facts, "revenue")
              if p.value]
    if annual:
        latest = annual[-1]
        bases.append((latest.value, sem.PeriodFrequency.ANNUAL,
                      getattr(latest, "start", None), getattr(latest, "end", None),
                      "reported_annual_revenue"))
    return bases


def _apply_absolute_revenue_guidance(evidence, entry, company_facts):
    """Turn a guided revenue LEVEL into a growth rate, or refuse to.

    Phase H.10 rewrites this around one rule: the division that produces a
    growth rate happens only after `finance.semantics` has confirmed that
    both sides cover the same length of time. Before, the period recorded on
    the guidance CHOSE the comparison and was then never checked against it,
    so a mislabelled period silently selected the wrong denominator and the
    resulting nonsense was clamped into range rather than rejected (sections
    10-11: a clamp must never hide an upstream semantic error).

    A refusal is recorded on `evidence.semantic_rejections` and leaves the
    guidance fields unset. The guidance itself survives as evidence -- it is
    still shown and still citable -- it simply does not become an annual
    growth assumption.
    """
    from finance import period_facts as pf_module

    multiplier = _SCALE_MULTIPLIER.get((entry.get("scale") or "").lower())
    if multiplier is None:
        return
    guided_amount = float(entry["midpoint"]) * multiplier
    period_type = entry.get("period_type")

    guided = sem.SemanticFact(
        metric_id=sem.MetricIdentity.REVENUE,
        value=guided_amount,
        units="currency",
        accounting_basis=(entry.get("basis") or sem.AccountingBasis.UNKNOWN),
        period_frequency=(sem.PeriodFrequency.ANNUAL if period_type == "annual"
                          else sem.PeriodFrequency.QUARTER if period_type == "quarter"
                          else sem.PeriodFrequency.UNKNOWN),
        fiscal_year=entry.get("fiscal_year"),
        fiscal_quarter=entry.get("fiscal_quarter"),
        flow_or_instant=sem.FlowOrInstant.FLOW,
        current_or_historical=sem.CurrentOrHistorical.FORWARD,
        evidence_id=entry.get("evidence_id"),
        definition_id="guidance_revenue_level")

    def reject(reason, base_fact, code=GUIDANCE_PERIOD_INCOMPATIBLE):
        evidence.semantic_rejections.append({
            "code": code,
            "operation": sem.Operation.GROWTH,
            "left": guided.label(),
            "right": base_fact.label(),
            "reason": reason,
            "context": "guidance revenue level -> year-1 revenue growth",
        })

    contradiction = _guidance_scale_contradicts_period(
        guided_amount, period_type, company_facts)
    if contradiction:
        claimed = period_type or "an unspecified period"
        reject("The guided revenue level was read as covering {}, but {}. No growth "
               "assumption is derived from it.".format(claimed, contradiction),
               sem.SemanticFact(metric_id=sem.MetricIdentity.REVENUE,
                                period_frequency=sem.PeriodFrequency.TTM,
                                flow_or_instant=sem.FlowOrInstant.FLOW,
                                definition_id="reported_ttm_revenue"))
        return

    if period_type == "annual":
        # Two comparable bases, in preference order. A trailing twelve months
        # is the more CURRENT twelve-month window and is tried first; the
        # latest reported fiscal year is the one an annual outlook is most
        # often written against ("FY2027 revenue of $118-120B" against
        # FY2026's actual) and is the fallback for an issuer whose quarterly
        # series cannot build a TTM.
        #
        # Both are twelve months, so both pass `compatible_for(GROWTH, ...)`
        # for the same reason; neither is assumed, each is checked.
        for base_value, base_frequency, base_start, base_end, base_definition in \
                _annual_growth_bases(company_facts):
            base = sem.SemanticFact(
                metric_id=sem.MetricIdentity.REVENUE, value=base_value, units="currency",
                accounting_basis=sem.AccountingBasis.GAAP,
                period_frequency=base_frequency,
                start_date=base_start, end_date=base_end,
                flow_or_instant=sem.FlowOrInstant.FLOW,
                current_or_historical=sem.CurrentOrHistorical.CURRENT,
                definition_id=base_definition)
            verdict = sem.compatible_for(sem.Operation.GROWTH, guided, base)
            if not verdict.allowed:
                reject(verdict.reason, base, verdict.code or GUIDANCE_PERIOD_INCOMPATIBLE)
                continue
            evidence.guidance_low = (guided_amount - base_value) / abs(base_value)
            evidence.guidance_high = evidence.guidance_low
            evidence.guidance_source_metric = \
                gm.GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH
            evidence.guidance_period_label = entry.get("fiscal_period")
            evidence.guidance_period_type = "annual"
            evidence.guidance_basis = entry.get("basis")
            evidence.guidance_bound_type = entry.get("bound_type")
            evidence.guidance_evidence_id = entry.get("evidence_id")
            evidence.guidance_implied_comparison_period = "{}..{}".format(
                base_start, base_end)
            return
        return

    if period_type != "quarter":
        return

    series = pf_module.discrete_quarters(company_facts, "revenue")
    if len(series.quarters) < 4:
        return
    # The guided quarter is the one AFTER the latest reported quarter, so its
    # prior-year comparable is four quarters before that -- i.e. the quarter
    # three back from the latest reported one.
    comparable = series.quarters[-4]
    if not comparable.value:
        return
    base = sem.SemanticFact(
        metric_id=sem.MetricIdentity.REVENUE, value=comparable.value, units="currency",
        accounting_basis=sem.AccountingBasis.GAAP,
        period_frequency=sem.PeriodFrequency.QUARTER,
        start_date=comparable.start, end_date=comparable.end,
        fiscal_quarter=entry.get("fiscal_quarter"),
        flow_or_instant=sem.FlowOrInstant.FLOW,
        current_or_historical=sem.CurrentOrHistorical.HISTORICAL,
        definition_id="reported_quarter_revenue")
    verdict = sem.compatible_for(sem.Operation.GROWTH, guided, base)
    if not verdict.allowed:
        reject(verdict.reason, base, verdict.code or GUIDANCE_PERIOD_INCOMPATIBLE)
        return
    evidence.guidance_implied_next_period_growth = (
        (guided_amount - comparable.value) / abs(comparable.value))
    evidence.guidance_implied_comparison_period = "{}..{}".format(
        comparable.start, comparable.end)



# Reference margin against which an ABSOLUTE scenario delta is expressed. A
# +/-2 percentage-point bull/bear shift is a sensible perturbation of a
# 10%-margin business; applied to a 0.9%-margin distributor it is a 3x swing
# in one direction and a sign flip in the other.
_DELTA_REFERENCE_MARGIN = 0.10


def apply_margin_delta(anchor: float, delta: float) -> Tuple[float, Optional[str]]:
    """Apply a scenario's margin shift COHERENTLY (section 46).

    Scenario deltas are absolute percentage points, which is right for a
    company whose margin is of the same order as the delta and wrong for one
    whose margin is much smaller. A live distributor reporting a 0.91%
    operating margin got a bear case of -2.09% -- a profitable company turned
    loss-making purely by the arithmetic of a fixed shift -- which then
    produced a negative terminal FCFF and invalidated the whole valuation.
    Nothing about the company had suggested a loss.

    So when the delta is LARGER THAN THE ANCHOR IT ADJUSTS, the shift is
    applied proportionally instead: the scenario keeps its intended severity
    relative to a normal margin, without asserting a change of sign that no
    evidence supports. A company already reporting a loss keeps the additive
    treatment, because there a further absolute decline is meaningful and a
    proportional one would shrink toward zero (an improvement) instead.
    """
    if not delta or anchor <= 0 or abs(delta) <= abs(anchor):
        return anchor + delta, None
    scaled = anchor * (1.0 + delta / _DELTA_REFERENCE_MARGIN)
    return scaled, (
        f"The scenario's {delta:+.1%} margin shift is larger than this company's own "
        f"{anchor:.2%} operating margin, so applying it additively would have implied a "
        f"change of sign rather than a change of degree. It was applied proportionally "
        f"instead, giving {scaled:.2%}.")


def _clamp(value: float, bounds: Tuple[float, float]) -> Tuple[float, bool]:
    low, high = bounds
    clamped = max(low, min(value, high))
    return clamped, clamped != value


def _fade(anchor: float, terminal: float, years: int) -> List[float]:
    """Linear convergence from a year-1 anchor toward a long-run rate.

    Deliberately the simplest possible shape. The point is not to model a
    particular decay curve — it is that a five-year forecast should not
    assert year 5 equals year 1. Straight-line interpolation makes that
    assertion visible and adjustable rather than hidden in a constant.
    """
    if years <= 1:
        return [anchor]
    return [anchor + (terminal - anchor) * (i / (years - 1)) for i in range(years)]


def _no_annual_guidance_clause(evidence) -> str:
    """How to open a derivation that did NOT anchor on guidance.

    "No current guidance was available" is false whenever the company has
    published something -- a next-quarter revenue level, guidance for a
    component of revenue -- that simply cannot carry a five-year annual
    growth path. A live analysis said exactly that while the same report
    listed the company's Q3 revenue guidance two sections above, and a
    research role duly flagged the contradiction as an analytical
    inconsistency. It was right to. The distinction is between guidance that
    does not exist and guidance that exists but does not answer this
    question, and the derivation now says which.
    """
    if evidence.guidance_midpoint is not None             and not evidence.guidance_eligibility.may_set_magnitude:
        # Guidance EXISTS and covers a shorter period than the assumption.
        # Naming the horizon rather than the absence is the difference
        # between "the company said nothing" and "the company said something
        # about a different span of time".
        period = evidence.guidance_period_label or "a shorter period"
        return (f"Current revenue guidance targets {period}, which covers a shorter period "
                "than an annual growth assumption, so it corroborates the near-term "
                "direction but was not used to set the year-1 rate. ")
    if evidence.guidance_implied_next_period_growth is not None:
        return ("Current guidance covers the NEXT QUARTER only and cannot by itself set a "
                "multi-year annual growth path, so it was not used as the anchor. ")
    if evidence.component_guidance_midpoint is not None:
        return ("Current guidance covers a COMPONENT of revenue rather than the consolidated "
                "total, so it was not used as the anchor. ")
    return "No current guidance was available. "


def build_growth_path(evidence: GrowthEvidence, forecast_years: int,
                      long_run_growth: Optional[float] = None,
                      scenario: str = "base", growth_delta: float = 0.0) -> ForwardPath:
    """Year-1 growth by section 12's precedence, then a fade to the long-run rate.

    Precedence, highest first:

        1. explicit CURRENT CONSOLIDATED revenue guidance
        2. applicable current revenue-COMPONENT guidance
        3. the trailing-twelve-month / year-to-date trend
        4. the latest reported annual year-over-year
        5. structurally relevant normalized historical growth
        6. the configured default

    Two Phase H.6 changes, both from the live AT&T run.

    Step 2 is new. AT&T guides SERVICE revenue growth, not consolidated
    revenue growth. Previously that was either forced into slot 1 (which is
    how EBITDA growth ended up as revenue growth) or ignored entirely. It now
    has a slot of its own, and the provenance says `source_metric =
    service_revenue_growth` rather than pretending it is consolidated.

    Step 5 is conditional. A long-period CAGR that spans a structural break
    measures a company that no longer exists — AT&T's -7.1% is mostly the
    WarnerMedia and DirecTV separations — so when
    `historical_comparability` is STRUCTURAL_BREAK the CAGR drops BELOW the
    post-break trend and below the latest annual year-over-year. It is never
    deleted: it is reported as backward-looking context with the break named.
    """
    long_run = long_run_growth if long_run_growth is not None else config.dcf_long_run_growth()
    path = ForwardPath(field="revenue_growth")

    # Forecast-horizon eligibility, checked BEFORE the value is read. A
    # quarterly guide is real evidence about the near term and is not a
    # twelve-month rate; anchoring a year-1 annual assumption on it and then
    # explaining the mismatch in prose left the NUMBER wrong and the sentence
    # right, which is the harder failure to notice.
    if evidence.guidance_midpoint is not None and evidence.guidance_eligibility.may_set_magnitude:
        anchor = evidence.guidance_midpoint
        path.anchor_source = AssumptionSourceType.MANAGEMENT_GUIDANCE
        period = evidence.guidance_period_label or f"fiscal {evidence.guidance_fiscal_year}"
        derivation = (
            f"Year 1 anchored on current management guidance of "
            f"{evidence.guidance_low:.1%} to {evidence.guidance_high:.1%} "
            f"(midpoint {anchor:.2%}) for {period}. source_metric="
            f"{evidence.guidance_source_metric}.")
        if evidence.guidance_bound_type == "at_least":
            derivation += (" Management stated a MINIMUM rather than a range; the figure is a "
                           "floor, not a midpoint expectation.")
        evidence_ids = tuple(filter(None, (evidence.guidance_evidence_id,)))
    elif evidence.component_guidance_midpoint is not None:
        anchor = evidence.component_guidance_midpoint
        path.anchor_source = AssumptionSourceType.MANAGEMENT_GUIDANCE
        derivation = (
            f"No consolidated revenue guidance was published. Year 1 anchored on current "
            f"guidance for a COMPONENT of revenue: "
            f"{evidence.component_guidance_low:.1%} to {evidence.component_guidance_high:.1%} "
            f"(midpoint {anchor:.2%}) for "
            f"{evidence.component_guidance_period_label or 'the guided period'}. "
            f"source_metric={evidence.component_guidance_metric}. This is NOT consolidated "
            "revenue guidance: it covers part of the company's revenue, and total revenue can "
            "grow faster or slower than the guided component.")
        evidence_ids = tuple(filter(None, (evidence.component_guidance_evidence_id,)))
    elif evidence.ttm_yoy is not None:
        anchor = evidence.ttm_yoy
        path.anchor_source = AssumptionSourceType.TTM_CALCULATION
        derivation = (_no_annual_guidance_clause(evidence)
                      + f"Year 1 anchored on the trailing-twelve-month revenue trend "
                        f"of {anchor:.2%}.")
        evidence_ids = ("dcf.input.revenue_ttm",)
    elif evidence.history_is_broken and evidence.post_break_cagr is not None:
        anchor = evidence.post_break_cagr
        path.anchor_source = AssumptionSourceType.HISTORICAL_CALCULATION
        derivation = (
            _no_annual_guidance_clause(evidence)
            + f"No trailing-twelve-month trend was available either, and this "
              f"company's reported history contains a structural break. Year 1 anchored on the "
            f"growth of the {len(evidence.post_break_periods)} COMPARABLE periods since the "
            f"break ({anchor:.2%}) rather than on the full-history CAGR, which spans the break.")
        evidence_ids = ()
    elif evidence.latest_annual_yoy is not None:
        anchor = evidence.latest_annual_yoy
        path.anchor_source = AssumptionSourceType.HISTORICAL_CALCULATION
        derivation = (_no_annual_guidance_clause(evidence)
                      + f"No trailing-twelve-month trend was available either. "
                        f"Year 1 anchored on the latest reported annual growth of {anchor:.2%}.")
        evidence_ids = ()
    elif evidence.historical_cagr is not None and not evidence.history_is_broken:
        anchor = evidence.historical_cagr
        path.anchor_source = AssumptionSourceType.HISTORICAL_CALCULATION
        derivation = (_no_annual_guidance_clause(evidence)
                      + f"No trailing-twelve-month trend or single-year growth "
                        f"was available either. Year 1 anchored on the {len(evidence.historical_periods)}"
                      f"-period historical CAGR of {anchor:.2%}. This is a MEASUREMENT OF THE "
                      "PAST used in the absence of any forward evidence, not a forecast derived "
                      "from one.")
        evidence_ids = ()
    else:
        anchor = config.dcf_default_revenue_growth()
        path.anchor_source = AssumptionSourceType.CONFIGURED_DEFAULT
        derivation = (f"No usable growth evidence was available; using the configured "
                      f"default of {anchor:.2%}.")
        if evidence.history_is_broken and evidence.historical_cagr is not None:
            derivation += (
                f" A {len(evidence.historical_periods)}-period historical CAGR of "
                f"{evidence.historical_cagr:.2%} exists but was NOT used: it spans a structural "
                "break in this company's reported history. " + evidence.comparability_summary)
        evidence_ids = ()

    path.anchor_value = anchor
    # The scenario delta moves the YEAR-1 ANCHOR only; every scenario fades to
    # the SAME long-run rate, which the caller sets to that scenario's own
    # terminal growth.
    #
    # Applying the delta to the fade TARGET as well (the first version of
    # this) was wrong twice over. It made bear's final forecast year assume
    # -1% growth while bear's perpetuity simultaneously assumed +1.5%, a
    # discontinuity right at the handoff where a DCF is most sensitive — and
    # it silently flipped COST's bear case to a negative terminal FCFF,
    # invalidating an otherwise sound valuation. Scenarios should disagree
    # about the NEAR term, which is what is actually uncertain; they should
    # not disagree about the steady state and then contradict their own
    # terminal assumption about it.
    raw_values = _fade(anchor + growth_delta, long_run, forecast_years)

    for index, raw in enumerate(raw_values):
        value, clamped = _clamp(raw, GROWTH_BOUNDS)
        path.values.append(round(value, 6))
        clamp_reason = None
        if clamped:
            # Section 19: a clamp is a SAFETY BOUND, not an estimate. NVDA's
            # 65.5% derived growth was silently capped at 25% and the 25% was
            # then presented as the near-term forecast; the raw value, the
            # applied value and the REASON are all recorded so a reader can
            # see that the model's limit, not the company, produced the
            # number.
            bound = GROWTH_BOUNDS[1] if raw > GROWTH_BOUNDS[1] else GROWTH_BOUNDS[0]
            clamp_reason = (
                f"The derived year-{index + 1} growth of {raw:.2%} lies outside the model's "
                f"configured bound of {bound:.0%}. The bound is a safety limit on what this "
                "engine will model, NOT an estimate of this company's growth; the applied "
                f"value of {value:.2%} reflects the limit, and the derived {raw:.2%} is "
                "retained beside it.")
        path.entries.append(AssumptionEntry(
            field="revenue_growth", forecast_year=index + 1, scenario=scenario,
            value=round(value, 6), units="ratio", source_type=path.anchor_source,
            evidence_ids=evidence_ids,
            derivation=(derivation + f" Year {index + 1} of {forecast_years} fades linearly "
                        f"toward the configured long-run rate of {long_run:.2%}."
                        + (" " + clamp_reason if clamp_reason else "")),
            clamped=clamped,
            original_proposed_value=round(raw, 6) if clamped else None,
            applied_value=round(value, 6),
            raw_value=round(raw, 6),
            clamp_reason=clamp_reason))

    if evidence.historical_cagr is not None \
            and path.anchor_source != AssumptionSourceType.HISTORICAL_CALCULATION:
        note = (f"Historical revenue CAGR of {evidence.historical_cagr:.2%} over "
                f"{evidence.historical_span} is retained as backward-looking CONTEXT and was "
                "deliberately not used as the forecast anchor.")
        if evidence.history_is_broken:
            note += (" That history spans a STRUCTURAL BREAK. "
                     + evidence.comparability_summary)
            if evidence.post_break_cagr is not None:
                note += (f" Over the {len(evidence.post_break_periods)} comparable periods "
                         f"since the break, revenue compounded at "
                         f"{evidence.post_break_cagr:.2%}.")
        path.notes.append(note)
    if evidence.component_guidance_midpoint is not None \
            and path.anchor_source == AssumptionSourceType.MANAGEMENT_GUIDANCE \
            and evidence.guidance_midpoint is not None:
        path.notes.append(
            f"Management also guided {evidence.component_guidance_metric} of "
            f"{evidence.component_guidance_low:.1%} to {evidence.component_guidance_high:.1%}; "
            "that covers part of revenue and is retained as supporting evidence only.")
    return path


# ---------------------------------------------------------------------------
# Section 21 — guidance vs history, classified rather than judged
# ---------------------------------------------------------------------------

GUIDANCE_HISTORY_DIVERGENCE = "GUIDANCE_HISTORY_DIVERGENCE"


def classify_guidance_history_divergence(evidence: GrowthEvidence) -> Optional[dict]:
    """Describe a disagreement between history and guidance. Do not judge it.

    Section 21: `historical CAGR != management guidance` is NOT automatically
    a problem, and treating it as one is how the AT&T run came to weigh a
    -7.1% CAGR against management's own current outlook as though the two
    were equally good evidence about the next twelve months. They are not
    equally good evidence, and which is better depends on the COMPARABILITY
    of the history — which this returns rather than resolving.

    The research pipeline decides what the divergence means. This function's
    only job is to state both figures, both periods, the comparability
    status, the magnitude, and why they differ.
    """
    guided = evidence.guidance_midpoint
    if guided is None:
        guided = evidence.component_guidance_midpoint
    historical = evidence.historical_cagr
    if guided is None or historical is None:
        return None
    magnitude = guided - historical
    comparable = evidence.historical_comparability

    if comparable == HistoricalComparability.STRUCTURAL_BREAK:
        explanation = (
            "The historical rate spans a structural break in this company's reported history, "
            "so it measures a different business from the one management is guiding. The "
            "divergence is largely an artefact of that break rather than a disagreement about "
            "the same company's prospects. " + (evidence.comparability_summary or ""))
        if evidence.post_break_cagr is not None:
            explanation += (
                f" Measured only over the comparable periods since the break, revenue "
                f"compounded at {evidence.post_break_cagr:.2%}, against guidance of "
                f"{guided:.2%}.")
    elif comparable == HistoricalComparability.PARTIALLY_COMPARABLE:
        explanation = (
            "Part of the history behind this rate may not describe the company as it is now, "
            "so the divergence is partly a comparability question and partly a genuine "
            "difference of view. " + (evidence.comparability_summary or ""))
    elif comparable == HistoricalComparability.COMPARABLE:
        explanation = (
            "The history behind this rate is comparable with the company as it is now, so this "
            "is a genuine difference between what the company has done and what management "
            "expects it to do.")
    else:
        explanation = (
            "There is not enough filing evidence to say whether the history behind this rate "
            "describes the company as it is now, so the divergence cannot be attributed to "
            "either a comparability problem or a difference of view.")

    return {
        "code": GUIDANCE_HISTORY_DIVERGENCE,
        "historical_metric": "revenue_cagr",
        "historical_value": round(historical, 6),
        "historical_span": evidence.historical_span,
        "current_guidance_metric": (evidence.guidance_source_metric
                                    or evidence.component_guidance_metric),
        "current_guidance_value": round(guided, 6),
        "current_guidance_period": (evidence.guidance_period_label
                                    or evidence.component_guidance_period_label),
        "comparability_status": comparable,
        "magnitude": round(magnitude, 6),
        "post_break_value": (round(evidence.post_break_cagr, 6)
                             if evidence.post_break_cagr is not None else None),
        "explanation": explanation,
    }


def implied_margin_from_guidance(guidance):
    """Section 10: an operating margin DERIVED from compatible guidance parts.

    When a company guides sales, gross margin and operating expenses, an
    operating margin follows arithmetically:

        gross profit     = sales x gross margin
        operating income = gross profit - operating expenses
        operating margin = operating income / sales

    Only performed when every component is present, the periods match, the
    units are what the taxonomy says they should be, and the ACCOUNTING BASES
    agree -- an adjusted gross margin combined with a GAAP operating expense
    produces a number on neither basis. The result is labelled
    DERIVED_FROM_GUIDANCE, never reported.

    Returns (margin, derivation, evidence_ids) or None.
    """
    if not guidance:
        return None
    sales = guidance.get("revenue")
    gross_margin = (guidance.get("adjusted_gross_margin")
                    or guidance.get("gross_margin"))
    opex = (guidance.get("adjusted_operating_expenses")
            or guidance.get("operating_expenses"))
    if not all(isinstance(x, dict) for x in (sales, gross_margin, opex)):
        return None
    for entry in (sales, gross_margin, opex):
        if entry.get("midpoint") is None:
            return None

    periods = {entry.get("fiscal_period") for entry in (sales, gross_margin, opex)}
    if len(periods) != 1 or None in periods:
        return None
    scales = {entry.get("scale") for entry in (sales, opex)}
    if len(scales) != 1:
        return None
    bases = {entry.get("basis") for entry in (gross_margin, opex)}
    if len(bases) != 1:
        return None

    sales_value = float(sales["midpoint"])
    if sales_value <= 0:
        return None
    gross_profit = sales_value * float(gross_margin["midpoint"])
    operating_income = gross_profit - float(opex["midpoint"])
    margin = operating_income / sales_value
    basis = bases.pop() or "unspecified"
    period = periods.pop()
    note = (
        f"DERIVED_FROM_GUIDANCE: management guided sales of {sales_value:,.2f}, a gross "
        f"margin of {float(gross_margin['midpoint']):.1%} and operating expenses of "
        f"{float(opex['midpoint']):,.2f} for {period}, which imply an operating margin of "
        f"{margin:.2%} on a {basis} basis. This is arithmetic on guided components, not a "
        "figure management stated or a figure the company reported.")
    ids = tuple(filter(None, (sales.get("evidence_id"), gross_margin.get("evidence_id"),
                              opex.get("evidence_id"))))
    return margin, note, ids


def build_margin_path(state: "fr.CurrentFinancialState", forecast_years: int,
                      scenario: str = "base", margin_delta: float = 0.0) -> ForwardPath:
    """Operating margin by section 9's precedence.

        explicit current guidance -> TTM operating margin -> latest annual
        margin -> historical normalized margin -> configured default
    """
    path = ForwardPath(field="operating_margin")
    guidance = (state.management_guidance or {}).get("metrics") or {}
    guided = guidance.get("operating_margin")
    evidence_ids: Tuple[str, ...] = ()

    revenue = state.flows.get("revenue")
    operating_income = state.flows.get("operating_income")

    # Phase H.8, section 11 -- MARGIN PRECEDENCE.
    #
    #   1. explicit matching operating-margin guidance
    #   2. a margin deterministically IMPLIED by compatible guidance
    #      components (sales x gross margin - operating expenses)
    #   3. the NORMALIZED current margin, when a normalization actually held
    #   4. the REPORTED current margin
    #   5. a normalized historical margin
    #   6. the configured default -- LAST RESORT, and it downgrades DCF
    #      suitability rather than passing as a forecast
    #
    # Step 3 is the Phase H.8 addition and the one that matters. A live
    # large-cap pharmaceutical reported an 11.6% trailing operating margin
    # against a 27.1% margin once a single acquisition-related research
    # charge was normalized out. Forecasting five years at the reported
    # figure carries a one-off charge through every year of the forecast; the
    # normalized figure is what the recurring business actually earns.
    _implied = implied_margin_from_guidance(guidance)
    profitability = (state.profitability or {}) if hasattr(state, "profitability") else {}
    normalized = (profitability.get("normalized") or {})
    reported = (profitability.get("reported") or {})

    if isinstance(guided, dict) and guided.get("low") is not None:
        anchor = (float(guided["low"]) + float(guided["high"])) / 2.0
        path.anchor_source = AssumptionSourceType.MANAGEMENT_GUIDANCE
        basis = guided.get("basis") or "unspecified"
        derivation = (f"Anchored on current management operating-margin guidance of "
                      f"{float(guided['low']):.1%} to {float(guided['high']):.1%} "
                      f"(basis: {basis}).")
        evidence_ids = tuple(filter(None, (guided.get("evidence_id"),)))
    elif _implied is not None:
        implied, implied_note, implied_ids = _implied
        anchor = implied
        path.anchor_source = AssumptionSourceType.MANAGEMENT_GUIDANCE
        derivation = implied_note
        evidence_ids = tuple(implied_ids)
    elif normalized.get("operating_margin") is not None \
            and normalized.get("status") in ("VALID", "PARTIAL"):
        anchor = float(normalized["operating_margin"])
        path.anchor_source = AssumptionSourceType.HISTORICAL_CALCULATION
        reported_margin = reported.get("operating_margin")
        derivation = (
            f"Anchored on the NORMALIZED current operating margin of {anchor:.2%}"
            + (f", against a reported {reported_margin:.2%}"
               if reported_margin is not None else "")
            + ". The reported figure is depressed by identified unusual items; carrying it "
              "through five forecast years would carry a one-off charge into every one of "
              "them. Normalization status: " + str(normalized.get("status")) + ".")
        evidence_ids = ("financial.normalized.operating_margin.ttm",
                        "financial.reported.operating_margin.ttm")
        path.notes.append(
            "Reported and normalized operating margins differ; the forecast uses the "
            "normalized figure and both are recorded.")
    elif (revenue is not None and operating_income is not None
            and revenue.value and operating_income.value is not None
            and revenue.source == "ttm_calculation"):
        anchor = operating_income.value / revenue.value
        path.anchor_source = AssumptionSourceType.TTM_CALCULATION
        derivation = (f"Anchored on the trailing-twelve-month operating margin "
                      f"({operating_income.value:,.0f} / {revenue.value:,.0f} = {anchor:.2%}) "
                      f"for {revenue.period_start}..{revenue.as_of_date}.")
        evidence_ids = ("financial.reported.operating_margin.ttm",)
    elif (revenue is not None and operating_income is not None
            and revenue.value and operating_income.value is not None):
        anchor = operating_income.value / revenue.value
        path.anchor_source = AssumptionSourceType.HISTORICAL_CALCULATION
        derivation = (f"Anchored on the latest reported annual operating margin "
                      f"({anchor:.2%}); no trailing-twelve-month figure was available.")
    else:
        historical = (state.historical_metrics or {}).get("operating_margin")
        if historical is not None:
            anchor = float(historical)
            path.anchor_source = AssumptionSourceType.HISTORICAL_CALCULATION
            derivation = f"Anchored on the reported historical operating margin of {anchor:.2%}."
        else:
            anchor = config.dcf_default_operating_margin()
            path.anchor_source = AssumptionSourceType.CONFIGURED_DEFAULT
            derivation = (
                f"No operating margin could be read or normalized for this company; using the "
                f"configured default of {anchor:.2%}. This is a LAST RESORT, not an estimate "
                "of this company's profitability, and it downgrades the DCF suitability "
                "assessment rather than passing as a forecast.")

    path.anchor_value = anchor

    # Section 36/37 -- A BOUND MAY LIMIT AN INPUT, NEVER REPLACE IT.
    #
    # The old code clamped the anchor to MARGIN_BOUNDS and held the clamped
    # value flat for every forecast year. On a company reporting a -60.1%
    # operating margin that produced +1.0% in year one -- a 61-point swing
    # invented by the floor -- and then held it for five years. The DCF
    # validated, because the arithmetic was fine; what was wrong was that the
    # modelled company had stopped being the reported one.
    #
    # When the observed margin sits FAR outside the bound (further than half
    # the bound's own width -- see finance/suitability.py), the honest shape
    # is a normalization PATH that starts where the company actually is and
    # improves toward the bound across the horizon. Year one stays close to
    # reality, the improvement is visible and adjustable, and the DCF
    # suitability assessment records that the model is being stretched.
    #
    # The path is a REPRESENTATION, not a prediction: it asserts only that a
    # company at the observed margin does not arrive at the modelled one
    # instantly. `raw_value` keeps the observed figure on every entry.
    scenario_anchor, delta_note = apply_margin_delta(anchor, margin_delta)
    if delta_note:
        path.notes.append(delta_note)
    breach = suitability.classify_bound_breach(
        scenario_anchor, MARGIN_BOUNDS, "operating_margin")
    if breach and breach["replaces_input"]:
        raw_values = suitability.normalization_path(
            scenario_anchor, breach["bound"], forecast_years,
            floor=MARGIN_BOUNDS[0] if scenario_anchor >= MARGIN_BOUNDS[0] else None)
        path.notes.append(
            f"The observed operating margin of {anchor:.1%} lies far outside the model's "
            f"bound of {breach['bound']:.1%}. Rather than applying the bound as year-1 "
            "margin -- which would model a company that reaches that level immediately -- "
            "the forecast normalizes toward it across the horizon. This is a "
            "REPRESENTATION of the gap, not evidence that the gap closes; see the DCF "
            "suitability assessment.")
    else:
        # Margin is held flat otherwise. Unlike growth there is no neutral
        # long-run margin to fade toward -- a company's structural margin IS
        # its own history -- so inventing a convergence target would be a
        # stronger claim than holding the observed level.
        raw_values = [scenario_anchor] * forecast_years

    for index, raw in enumerate(raw_values):
        value, clamped = _clamp(raw, MARGIN_BOUNDS)
        path.values.append(round(value, 6))
        clamp_reason = None
        if clamped:
            bound = MARGIN_BOUNDS[1] if raw > MARGIN_BOUNDS[1] else MARGIN_BOUNDS[0]
            clamp_reason = (
                f"The year-{index + 1} operating margin of {raw:.2%} lies outside the "
                f"model's configured bound of {bound:.0%}. The bound is a safety limit on "
                "what this engine will model, NOT an estimate of this company's "
                f"profitability; the applied {value:.2%} reflects the limit and the derived "
                f"{raw:.2%} is retained beside it.")
        path.entries.append(AssumptionEntry(
            field="operating_margin", forecast_year=index + 1, scenario=scenario,
            value=round(value, 6), units="ratio", source_type=path.anchor_source,
            evidence_ids=evidence_ids,
            derivation=derivation + (" " + clamp_reason if clamp_reason else ""),
            clamped=clamped,
            original_proposed_value=round(raw, 6) if clamped else None,
            applied_value=round(value, 6),
            raw_value=round(raw, 6),
            clamp_reason=clamp_reason))
    return path


# ---------------------------------------------------------------------------
# The model's proposal, and its validation (sections 8 and 10)
# ---------------------------------------------------------------------------

FORWARD_ASSUMPTION_PROMPT = """You are the ForwardAssumptionBuilder for a discounted-cash-flow \
valuation of {symbol}.

Propose a year-by-year revenue-growth and operating-margin path for the next {years} years.

VALIDATED EVIDENCE AVAILABLE TO YOU
-----------------------------------
{evidence_block}

RULES
-----
1. These are FORECASTS, not measurements. Do not simply repeat the historical CAGR for every \
year unless you explicitly justify why this company's past rate is the best available estimate \
of its future rate.
2. Current management guidance is an INPUT, not the truth. You may forecast below, within, or \
above it. If your year-1 base growth differs from stated guidance by more than {tolerance:.0%}, \
you MUST explain why in "justification", citing evidence ids.
3. Growth must stay within [{growth_low:.2f}, {growth_high:.2f}] and margin within \
[{margin_low:.2f}, {margin_high:.2f}]. Values outside these bounds will be clamped.
4. Cite only evidence ids that appear above. Do not invent figures that are not shown.

Reply with STRICT JSON and nothing else:

{{
  "revenue_growth": {{{growth_year_keys}}},
  "operating_margin": {{{margin_year_keys}}},
  "reasoning_evidence_ids": [],
  "justification": "",
  "confidence": 0.0
}}"""


def build_evidence_block(state: "fr.CurrentFinancialState",
                         evidence: GrowthEvidence, baseline: Dict[str, ForwardPath]) -> str:
    """The ONLY view of the data the model gets when proposing a path.

    Deliberately a rendered summary of already-validated selections rather
    than raw filings: the model never decides freshness, and it never sees a
    document it could mine for numbers that were not selected.
    """
    lines = []
    revenue = state.flows.get("revenue")
    if revenue is not None and revenue.value is not None:
        lines.append(f"- Revenue ({revenue.source}, {revenue.period_start}..{revenue.as_of_date}): "
                     f"{revenue.value:,.0f} [dcf.input.revenue_ttm]")
    operating_income = state.flows.get("operating_income")
    if operating_income is not None and operating_income.value is not None and revenue \
            and revenue.value:
        lines.append(f"- Operating margin ({operating_income.source}): "
                     f"{operating_income.value / revenue.value:.2%} "
                     "[dcf.input.operating_income_ttm]")
    if evidence.guidance_midpoint is not None:
        lines.append(
            f"- CURRENT MANAGEMENT GUIDANCE for CONSOLIDATED revenue growth, "
            f"{evidence.guidance_period_label or evidence.guidance_fiscal_year}: "
            f"{evidence.guidance_low:.1%} to {evidence.guidance_high:.1%} "
            f"[{evidence.guidance_evidence_id}] (FORWARD-LOOKING evidence"
            + (", ONE QUARTER only" if evidence.guidance_period_type == "quarter" else "")
            + (", a stated MINIMUM not a midpoint"
               if evidence.guidance_bound_type == "at_least" else "") + ")")
    else:
        lines.append("- CURRENT MANAGEMENT GUIDANCE for consolidated revenue growth: unavailable")
    if evidence.component_guidance_midpoint is not None:
        lines.append(
            f"- Current guidance for a COMPONENT of revenue "
            f"({evidence.component_guidance_metric}), "
            f"{evidence.component_guidance_period_label or 'period as stated'}: "
            f"{evidence.component_guidance_low:.1%} to {evidence.component_guidance_high:.1%} "
            f"[{evidence.component_guidance_evidence_id}] (FORWARD-LOOKING, but covers PART of "
            "revenue — total revenue may grow faster or slower)")
    if evidence.guidance_implied_next_period_growth is not None:
        lines.append(
            f"- Guided NEXT-QUARTER revenue implies "
            f"{evidence.guidance_implied_next_period_growth:.1%} growth against the comparable "
            f"prior-year quarter ({evidence.guidance_implied_comparison_period}). This is ONE "
            "QUARTER of near-term evidence about the current trajectory. It is NOT a five-year "
            "forecast and must not be applied flat to the whole horizon.")
    if evidence.ttm_yoy is not None:
        lines.append(f"- Trailing-twelve-month revenue growth vs prior twelve months: "
                     f"{evidence.ttm_yoy:.2%} [dcf.input.revenue_ttm]")
    if evidence.latest_annual_yoy is not None:
        lines.append(f"- Latest reported annual revenue growth: {evidence.latest_annual_yoy:.2%} "
                     "(HISTORICAL)")
    if evidence.historical_cagr is not None:
        lines.append(f"- Historical revenue CAGR over {len(evidence.historical_periods)} reported "
                     f"periods: {evidence.historical_cagr:.2%} (HISTORICAL CONTEXT ONLY)")
    lines.append(f"- Historical comparability: {evidence.historical_comparability}"
                 + (f" — {evidence.comparability_summary}"
                    if evidence.comparability_summary else ""))
    if evidence.post_break_cagr is not None:
        lines.append(f"- Revenue growth over the {len(evidence.post_break_periods)} COMPARABLE "
                     f"periods since the structural break: {evidence.post_break_cagr:.2%}")
    clamped = [e for path in baseline.values() for e in path.entries if e.clamped]
    if clamped:
        first = clamped[0]
        lines.append(
            f"- NOTE: the deterministic baseline was CLAMPED. {first.field} year "
            f"{first.forecast_year} derived {first.raw_value:.2%} and applied "
            f"{first.applied_value:.2%}. A clamp is this engine's safety bound, NOT an estimate "
            "of the company's growth — do not treat the applied value as a forecast.")
    growth_baseline = baseline.get("revenue_growth")
    if growth_baseline is not None:
        lines.append("- Deterministic baseline growth path: "
                     + ", ".join(f"y{i + 1}={v:.2%}" for i, v in enumerate(growth_baseline.values)))
    return "\n".join(lines)


@dataclass
class ProposalOutcome:
    accepted: bool
    growth: Optional[List[float]] = None
    margin: Optional[List[float]] = None
    confidence: Optional[float] = None
    justification: str = ""
    cited_evidence_ids: Tuple[str, ...] = ()
    findings: List[str] = field(default_factory=list)
    clamps: List[dict] = field(default_factory=list)


def validate_proposal(raw: dict, forecast_years: int, evidence: GrowthEvidence,
                      valid_evidence_ids: Optional[Sequence[str]] = None) -> ProposalOutcome:
    """Validate a model-proposed path. Fails CLOSED to the deterministic baseline.

    Rejection is never a valuation failure — the caller keeps the
    deterministic path it already built. That asymmetry is intentional: the
    model can improve a forecast but must never be able to break one.
    """
    outcome = ProposalOutcome(accepted=False)
    if not isinstance(raw, dict):
        outcome.findings.append("The forward-assumption proposal was not a JSON object.")
        return outcome

    growth = _read_year_map(raw.get("revenue_growth"), forecast_years, "revenue_growth", outcome)
    margin = _read_year_map(raw.get("operating_margin"), forecast_years, "operating_margin",
                            outcome)
    if growth is None or margin is None:
        return outcome

    cited = tuple(str(i) for i in (raw.get("reasoning_evidence_ids") or [])
                  if isinstance(i, (str, int)))
    if valid_evidence_ids is not None:
        unknown = [i for i in cited if i not in set(valid_evidence_ids)]
        if unknown:
            outcome.findings.append(
                f"The proposal cited evidence ids that do not exist: {', '.join(unknown)}.")
            return outcome

    justification = str(raw.get("justification") or "").strip()

    # Section 10: a base-case year-1 growth that departs materially from
    # stated guidance requires an explicit justification. Without one, the
    # proposal is not used -- the guidance-anchored baseline stands.
    if evidence.guidance_low is not None and evidence.guidance_high is not None:
        year_one = growth[0]
        below = evidence.guidance_low - GUIDANCE_DIVERGENCE_TOLERANCE
        above = evidence.guidance_high + GUIDANCE_DIVERGENCE_TOLERANCE
        if (year_one < below or year_one > above) and len(justification) < 40:
            outcome.findings.append(
                f"Year-1 growth of {year_one:.2%} falls outside current management guidance of "
                f"{evidence.guidance_low:.1%}-{evidence.guidance_high:.1%} and the proposal did "
                "not justify the divergence with evidence; the guidance-anchored baseline was "
                "kept instead.")
            return outcome

    applied_growth, growth_clamps = _clamp_series(growth, GROWTH_BOUNDS, "revenue_growth")
    applied_margin, margin_clamps = _clamp_series(margin, MARGIN_BOUNDS, "operating_margin")

    confidence = raw.get("confidence")
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = None

    outcome.accepted = True
    outcome.growth = applied_growth
    outcome.margin = applied_margin
    outcome.confidence = confidence
    outcome.justification = justification
    outcome.cited_evidence_ids = cited
    outcome.clamps = growth_clamps + margin_clamps
    return outcome


def _read_year_map(raw, forecast_years: int, label: str,
                   outcome: ProposalOutcome) -> Optional[List[float]]:
    """Accept {"year_1": x, ...} or a plain list; require an exact-length series."""
    values: List[float] = []
    if isinstance(raw, dict):
        for index in range(1, forecast_years + 1):
            entry = raw.get(f"year_{index}", raw.get(str(index)))
            if entry is None:
                outcome.findings.append(f"{label} is missing year_{index}.")
                return None
            values.append(entry)
    elif isinstance(raw, (list, tuple)):
        if len(raw) != forecast_years:
            outcome.findings.append(
                f"{label} has {len(raw)} entries but the forecast horizon is {forecast_years}.")
            return None
        values = list(raw)
    else:
        outcome.findings.append(f"{label} must be an object keyed by year, or a list.")
        return None

    numeric: List[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            outcome.findings.append(f"{label} year_{index + 1} is not a number.")
            return None
        numeric.append(float(value))
    return numeric


def _clamp_series(values: Sequence[float], bounds: Tuple[float, float],
                  label: str) -> Tuple[List[float], List[dict]]:
    applied: List[float] = []
    clamps: List[dict] = []
    for index, value in enumerate(values):
        bounded, was_clamped = _clamp(value, bounds)
        applied.append(round(bounded, 6))
        if was_clamped:
            clamps.append({"field": label, "forecast_year": index + 1,
                           "original_proposed_value": round(value, 6),
                           "applied_value": round(bounded, 6),
                           "bounds": list(bounds)})
    return applied, clamps


def apply_proposal(baseline: Dict[str, ForwardPath], outcome: ProposalOutcome,
                   scenario: str = "base") -> Dict[str, ForwardPath]:
    """Replace a baseline path with an ACCEPTED proposal, preserving provenance.

    The resulting entries are labelled `llm_proposed` and keep the
    deterministic anchor's derivation text, so a reader can see both what the
    model chose and what it chose instead of.
    """
    if not outcome.accepted:
        return baseline
    updated = dict(baseline)
    for field_name, values in (("revenue_growth", outcome.growth),
                               ("operating_margin", outcome.margin)):
        if values is None or field_name not in baseline:
            continue
        original = baseline[field_name]
        path = ForwardPath(field=field_name, anchor_source=AssumptionSourceType.LLM_PROPOSED,
                           anchor_value=values[0], notes=list(original.notes))
        clamp_by_year = {c["forecast_year"]: c for c in outcome.clamps
                         if c["field"] == field_name}
        for index, value in enumerate(values):
            clamp = clamp_by_year.get(index + 1)
            baseline_value = (original.values[index] if index < len(original.values) else None)
            path.values.append(value)
            path.entries.append(AssumptionEntry(
                field=field_name, forecast_year=index + 1, scenario=scenario,
                value=value, units="ratio", source_type=AssumptionSourceType.LLM_PROPOSED,
                evidence_ids=outcome.cited_evidence_ids,
                derivation=(
                    f"Proposed by the local model in place of the deterministic baseline"
                    + (f" of {baseline_value:.2%}" if baseline_value is not None else "")
                    + f" (which was anchored on {original.anchor_source}). "
                    + (f"Justification: {outcome.justification} " if outcome.justification else "")
                    + (f"CLAMPED from {clamp['original_proposed_value']:.2%}."
                       if clamp else "")),
                clamped=bool(clamp),
                original_proposed_value=(clamp["original_proposed_value"] if clamp else None),
                applied_value=value))
        updated[field_name] = path
    return updated


# ---------------------------------------------------------------------------
# Sections 16-17 — the current-year tax rate is not the forecast tax rate
# ---------------------------------------------------------------------------

# How far a guided tax rate may sit from the issuer's own normalized history
# before the current year is treated as carrying an unusual tax effect. A
# live release guided a 35-36% effective rate against a prior guide of
# 23.5-24.5% for the same year -- an eleven-point move caused by two
# acquisitions -- and applying 35% to all five forecast years would carry a
# one-off tax consequence through the whole horizon.
UNUSUAL_TAX_EFFECT_THRESHOLD = 0.05

# The project's configured statutory-adjacent tax rate, matching the value
# finance/workflow.py::propose_assumptions has always used. Named here so
# the tax path and the scenario builder cannot drift apart.
CONFIGURED_TAX_RATE = 0.21

TAX_GUIDANCE_CONFLICT = "TAX_GUIDANCE_CONFLICT"


def build_tax_path(state, forecast_years, guidance=None, historical_tax_rate=None):
    """A year-by-year tax path (section 17), not one rate repeated.

    Year 1 may legitimately carry an unusual effect -- an acquisition, a
    one-off settlement, a rate change -- and years 2-5 should not. When the
    current guided rate and the issuer's normalized history disagree
    materially, the path starts at the guided rate and converges on the
    normalized one; when they agree, the rate is simply held.

    Returns (values, provenance) so the caller records what was used and why.
    """
    guidance = guidance or {}
    guided = guidance.get("tax_rate")
    guided_rate = None
    basis = None
    if isinstance(guided, dict) and guided.get("midpoint") is not None:
        guided_rate = float(guided["midpoint"])
        basis = guided.get("basis")

    reported = None
    profitability = getattr(state, "profitability", None) or {}
    reported_block = profitability.get("reported") or {}
    if reported_block.get("tax_rate") is not None:
        reported = float(reported_block["tax_rate"])

    normalized_rate = historical_tax_rate
    if normalized_rate is None:
        normalized_rate = CONFIGURED_TAX_RATE

    provenance = {
        "reported_tax_rate": reported,
        "current_guided_tax_rate": guided_rate,
        "guided_tax_basis": basis,
        "normalized_forward_tax_rate": normalized_rate,
        "findings": [],
    }

    if guided_rate is None:
        rate = reported if reported is not None and 0.0 <= reported <= 0.60 \
            else normalized_rate
        provenance["source"] = ("reported_ttm" if rate == reported else "configured_default")
        provenance["derivation"] = (
            f"No current tax guidance was available; the forecast holds "
            f"{rate:.1%} across all {forecast_years} years.")
        return [round(rate, 6)] * forecast_years, provenance

    divergence = abs(guided_rate - normalized_rate)
    if divergence < UNUSUAL_TAX_EFFECT_THRESHOLD:
        provenance["source"] = "management_guidance"
        provenance["derivation"] = (
            f"Current tax guidance of {guided_rate:.1%} (basis: {basis}) is close to the "
            f"normalized rate of {normalized_rate:.1%}, so it is held across the horizon.")
        return [round(guided_rate, 6)] * forecast_years, provenance

    # Year 1 takes the guided rate; the path converges on the normalized one.
    values = [guided_rate + (normalized_rate - guided_rate) * (i / max(1, forecast_years - 1))
              for i in range(forecast_years)]
    provenance["source"] = "management_guidance_normalized"
    provenance["derivation"] = (
        f"Current tax guidance of {guided_rate:.1%} (basis: {basis}) differs from the "
        f"normalized rate of {normalized_rate:.1%} by {divergence:.1%}, which indicates an "
        "unusual current-year tax effect. Year 1 uses the guided rate and the path converges "
        "on the normalized rate rather than carrying a one-off tax consequence through every "
        "forecast year.")
    provenance["findings"].append({
        "code": TAX_GUIDANCE_CONFLICT, "severity": "info",
        "message": provenance["derivation"],
        "guided_tax_rate": guided_rate, "normalized_tax_rate": normalized_rate,
    })
    return [round(v, 6) for v in values], provenance


# ---------------------------------------------------------------------------
# Sections 13-15 — is the DCF's year-1 assumption consistent with guidance?
# ---------------------------------------------------------------------------

DCF_GUIDANCE_ASSUMPTION_CONFLICT = "DCF_GUIDANCE_ASSUMPTION_CONFLICT"


class GuidanceConsistency:
    CONSISTENT = "GUIDANCE_ASSUMPTION_CONSISTENT"
    DIVERGENCE = "GUIDANCE_ASSUMPTION_DIVERGENCE"
    NOT_COMPARABLE = "GUIDANCE_METRIC_NOT_COMPARABLE"
    NONE = "NO_RELEVANT_GUIDANCE"
    ALL = (CONSISTENT, DIVERGENCE, NOT_COMPARABLE, NONE)


# How far a year-1 assumption may sit outside stated guidance before the
# divergence needs an explicit, evidence-backed rationale. Wider than the
# ordinary tolerance because a forecast is allowed to disagree with guidance
# -- section 14 is explicit that divergence is not automatically rejected --
# but a gap this size is a thesis, and a thesis has to be stated.
MATERIAL_GUIDANCE_DIVERGENCE = 0.05


def validate_guidance_against_assumption(evidence, year_one_growth,
                                         justification=""):
    """Compare the DCF's year-1 revenue growth with RELEVANT guidance only.

    Section 13's rule stated as code: a consolidated revenue-growth
    assumption is compared against consolidated revenue guidance and against
    nothing else. EBITDA growth, gross margin and EPS are not comparable to
    it, and treating them as though they were is exactly the substitution
    earlier phases fixed at extraction time -- this is the same rule applied
    one layer later.

    Divergence is REPORTED, never auto-corrected. The forecast may sit above
    or below guidance when evidence supports it; what it may not do is sit
    materially outside without saying why.
    """
    record = {
        "status": GuidanceConsistency.NONE,
        "assumption_metric": "consolidated_revenue_growth",
        "forecast_period": "year_1",
        "assumption_value": year_one_growth,
        "guidance_low": None,
        "guidance_high": None,
        "guidance_metric": None,
        "guidance_target_period": None,
        "divergence": None,
        "source_evidence_ids": [],
        "findings": [],
    }
    if year_one_growth is None:
        return record

    low, high = evidence.guidance_low, evidence.guidance_high
    metric = evidence.guidance_source_metric
    if low is None and evidence.guidance_implied_next_period_growth is not None:
        # A quarterly guide is near-term evidence about trajectory, not a
        # full-year rate (section 16). It is recorded as CORROBORATION and
        # never as the range a year-1 forecast must sit inside.
        record["status"] = GuidanceConsistency.NOT_COMPARABLE
        record["guidance_metric"] = "implied_next_quarter_revenue_growth"
        record["guidance_target_period"] = evidence.guidance_period_label
        record["findings"].append({
            "code": GuidanceConsistency.NOT_COMPARABLE, "severity": "info",
            "message": (
                "The only revenue guidance available targets a single quarter, so it is "
                "near-term evidence about trajectory rather than a full-year rate the "
                f"year-1 assumption of {year_one_growth:.1%} can be checked against."),
        })
        return record
    if low is None or high is None:
        return record

    record.update({"guidance_low": low, "guidance_high": high,
                   "guidance_metric": metric,
                   "guidance_target_period": evidence.guidance_period_label,
                   "source_evidence_ids": [evidence.guidance_evidence_id]
                   if evidence.guidance_evidence_id else []})

    if evidence.guidance_period_type == "quarter":
        record["status"] = GuidanceConsistency.NOT_COMPARABLE
        return record

    if low - MATERIAL_GUIDANCE_DIVERGENCE <= year_one_growth <= \
            high + MATERIAL_GUIDANCE_DIVERGENCE:
        record["status"] = GuidanceConsistency.CONSISTENT
        return record

    divergence = (year_one_growth - high if year_one_growth > high
                  else year_one_growth - low)
    record["divergence"] = divergence
    record["status"] = GuidanceConsistency.DIVERGENCE
    if len(justification or "") < 40:
        record["findings"].append({
            "code": DCF_GUIDANCE_ASSUMPTION_CONFLICT, "severity": "error",
            "message": (
                f"The DCF's year-1 revenue growth of {year_one_growth:.1%} sits "
                f"{abs(divergence):.1%} outside management's own guidance of {low:.1%} to "
                f"{high:.1%} for {record['guidance_target_period']}, and no evidence-backed "
                "rationale was recorded for the difference. A forecast may disagree with "
                "guidance; it may not disagree silently."),
            "assumption_value": year_one_growth, "guidance_low": low,
            "guidance_high": high, "divergence": divergence,
        })
    return record


# ---------------------------------------------------------------------------
# Sections 17-19 — a configured bound is not a forecast
# ---------------------------------------------------------------------------

DCF_MODEL_BOUND_CONFLICT = "DCF_MODEL_BOUND_CONFLICT"

# How much a bound must move the year-1 assumption before it is materially
# changing the forecast rather than trimming it.
MATERIAL_BOUND_EFFECT = 0.05


def detect_model_bound_conflict(evidence, raw_value, applied_value, bounds,
                                label="revenue_growth"):
    """Is a configured bound constraining well-corroborated current evidence?

    Section 18 requires three things at once, and all three matter:

      1. validated evidence supports a value outside the bound,
      2. the bound materially changes the forecast, and
      3. INDEPENDENT evidence -- current guidance -- corroborates the
         out-of-bound value.

    Without (3) an out-of-bound figure is ordinary assumption uncertainty and
    the clamp is doing its job. With (3) the model's representational range,
    not the company, is setting the forecast: a live issuer's trailing growth
    was 39.5% and its own next-quarter guidance implied about the same, while
    the model's ceiling was 25% -- so the valuation thesis became an artefact
    of a software limit.

    Section 19: this never raises the bound. It records the conflict so DCF
    suitability and readiness can reflect it.
    """
    if raw_value is None or applied_value is None:
        return None
    low, high = bounds
    if low <= raw_value <= high:
        return None
    if abs(raw_value - applied_value) < MATERIAL_BOUND_EFFECT:
        return None

    # -- what may ESTABLISH the conflict ----------------------------------
    #
    # Only evidence measured in the same units as the assumption. An annual
    # bound is a statement about twelve months, so a figure compared against
    # it has to cover twelve months too.
    #
    # The live failure this rule exists for: an annual bound of 25%, trailing
    # twelve-month growth of 32%, next-quarter guidance implying 84%. The
    # conflict was REAL -- 32% > 25% -- and the sentence said the model was
    # 59 percentage points wrong, because the quarterly figure supplied the
    # magnitude. Right conclusion, wrong period, and a reader sent to look at
    # a discrepancy that does not exist.
    comparable, directional = _bound_conflict_candidates(evidence)

    corroboration = None
    for value, source, eligibility in comparable:
        # Corroborating means pointing the SAME WAY past the bound, not
        # merely being a number that exists.
        if (raw_value > high and value > high) or (raw_value < low and value < low):
            corroboration = (value, source, eligibility)
            break
    if corroboration is None:
        return None

    value, source, eligibility = corroboration
    bound = high if raw_value > high else low
    conflict = {
        "code": DCF_MODEL_BOUND_CONFLICT,
        "severity": "error",
        "assumption": label,
        "raw_evidence_value": raw_value,
        "model_bound": bound,
        "applied_value": applied_value,
        "corroborating_value": value,
        "corroborating_source": source,
        "corroborating_forecast_compatibility": eligibility.status,
        "corroborating_horizon": eligibility.compatible_horizon,
        "message": (
            f"Validated evidence supports a year-1 {label} of {raw_value:.1%} and {source} "
            f"corroborates it at {value:.1%}, but the model's configured bound of "
            f"{bound:.1%} caps the applied assumption at {applied_value:.1%}. The bound is a "
            "safety control, not an estimate: here it is the model's representational range, "
            "rather than the company's economics, that is setting the forecast. The bound is "
            "deliberately NOT raised; the conflict is recorded so the valuation's standing "
            "reflects it."),
    }

    # -- what may only ACCOMPANY it ---------------------------------------
    #
    # Shorter-horizon evidence is preserved in full and attached as context.
    # It does not decide whether the conflict exists, does not supply its
    # magnitude and does not change its severity -- and its number stays out
    # of the message, because a figure printed beside a bound is a figure a
    # reader will compare against that bound.
    if directional:
        support_value, support_source, support_eligibility, period = directional[0]
        conflict["directional_support"] = {
            "value": support_value,
            "source": support_source,
            "comparison_period": period,
            "forecast_compatibility": support_eligibility.status,
            "horizon": support_eligibility.compatible_horizon,
            "reason": support_eligibility.reason,
        }
        conflict["directional_support_note"] = (
            f"{support_source} covers a shorter period than the annual assumption and "
            "corroborates the DIRECTION of this conflict only; it is not comparable with "
            "the annual bound and does not set the size of the gap.")
    return conflict


def _bound_conflict_candidates(evidence):
    """(comparable, directional) forward signals, each with its eligibility.

    Split by what each may DO rather than by where it came from, so a caller
    cannot reach for a number without also seeing the verdict on it.
    """
    comparable, directional = [], []
    if evidence is None:
        return comparable, directional

    # A twelve-month actual is the same kind of quantity as an annual
    # assumption, and it is the figure the assumption was derived from --
    # which is why it can show that a bound, rather than the company, is
    # setting the forecast.
    if evidence.ttm_yoy is not None:
        comparable.append((evidence.ttm_yoy, "the trailing twelve-month growth rate",
                           evidence.ttm_eligibility))

    guided = evidence.guidance_midpoint
    if guided is not None:
        eligibility = evidence.guidance_eligibility
        entry = (guided, "current management guidance", eligibility)
        if eligibility.may_set_magnitude:
            comparable.append(entry)
        elif eligibility.may_corroborate_direction:
            directional.append(entry + (evidence.guidance_period_label,))

    implied = evidence.guidance_implied_next_period_growth
    if implied is not None:
        eligibility = evidence.implied_growth_eligibility
        entry = (implied, "management's next-period revenue guidance", eligibility)
        if eligibility.may_set_magnitude:
            comparable.append(entry)
        elif eligibility.may_corroborate_direction:
            directional.append(entry + (evidence.guidance_implied_comparison_period,))

    return comparable, directional


def build_forward_assumptions(state: "fr.CurrentFinancialState", forecast_years: int,
                              company_facts: Optional[dict] = None,
                              scenario: str = "base", growth_delta: float = 0.0,
                              margin_delta: float = 0.0,
                              terminal_growth: Optional[float] = None,
                              comparability: Optional[dict] = None
                              ) -> Tuple[Dict[str, ForwardPath], GrowthEvidence]:
    """The deterministic baseline for one scenario.

    `terminal_growth` is the rate the explicit forecast fades TO, so the last
    forecast year joins the perpetuity continuously. Callers pass the
    scenario's own terminal growth; omitting it falls back to the configured
    long-run rate.
    """
    evidence = collect_growth_evidence(state, company_facts, comparability=comparability)
    return {
        "revenue_growth": build_growth_path(evidence, forecast_years, scenario=scenario,
                                            growth_delta=growth_delta,
                                            long_run_growth=terminal_growth),
        "operating_margin": build_margin_path(state, forecast_years, scenario=scenario,
                                              margin_delta=margin_delta),
    }, evidence
