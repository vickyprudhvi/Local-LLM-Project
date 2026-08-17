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
from finance.dcf import AssumptionSourceType
from finance.structural_breaks import HistoricalComparability

# Hard bounds. A proposal outside these is CLAMPED and the clamp recorded --
# never silently accepted, and never rejected outright (a clamped run still
# produces a valuation, it just says so).
GROWTH_BOUNDS = (-0.20, 0.25)
MARGIN_BOUNDS = (0.01, 0.60)

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

    guidance = (state.management_guidance or {}).get("metrics") or {}
    revenue_growth = guidance.get(gm.GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH)
    if isinstance(revenue_growth, dict) and revenue_growth.get("low") is not None:
        source_metric = revenue_growth.get("name") or \
            gm.GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH
        if gm.may_anchor_revenue_growth(source_metric):
            evidence.guidance_low = float(revenue_growth["low"])
            evidence.guidance_high = float(revenue_growth["high"])
            evidence.guidance_fiscal_year = revenue_growth.get("fiscal_year")
            evidence.guidance_evidence_id = revenue_growth.get("evidence_id")
            evidence.guidance_source_metric = source_metric
            evidence.guidance_period_label = revenue_growth.get("fiscal_period")
            evidence.guidance_period_type = revenue_growth.get("period_type")
            evidence.guidance_basis = revenue_growth.get("basis")
            evidence.guidance_bound_type = revenue_growth.get("bound_type")

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
    revenue_amount = guidance.get(gm.GuidanceMetricName.CONSOLIDATED_REVENUE)
    if isinstance(revenue_amount, dict) and revenue_amount.get("midpoint") is not None \
            and company_facts:
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


def _apply_absolute_revenue_guidance(evidence: GrowthEvidence, entry: dict,
                                     company_facts: dict) -> None:
    """Turn "$91.0 billion next quarter" into a growth rate, or leave it alone.

    Only ever compared against the SAME fiscal quarter of the prior year —
    a guided Q2 against the reported Q2, never against the most recent
    quarter, because a sequential comparison of a seasonal business is not a
    growth rate. If no comparable quarter can be identified the field stays
    unset: an implied growth rate that silently compared unlike periods would
    be worse than no signal at all.
    """
    from finance import period_facts as pf_module

    multiplier = _SCALE_MULTIPLIER.get((entry.get("scale") or "").lower())
    if multiplier is None or entry.get("period_type") != "quarter":
        return
    guided_amount = float(entry["midpoint"]) * multiplier

    series = pf_module.discrete_quarters(company_facts, "revenue")
    if len(series.quarters) < 4:
        return
    # The guided quarter is the one AFTER the latest reported quarter, so its
    # prior-year comparable is four quarters before that — i.e. the quarter
    # three back from the latest reported one.
    comparable = series.quarters[-4] if len(series.quarters) >= 4 else None
    if comparable is None or not comparable.value:
        return
    evidence.guidance_implied_next_period_growth = (
        (guided_amount - comparable.value) / abs(comparable.value))
    evidence.guidance_implied_comparison_period = f"{comparable.start}..{comparable.end}"


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

    if evidence.guidance_midpoint is not None:
        anchor = evidence.guidance_midpoint
        path.anchor_source = AssumptionSourceType.MANAGEMENT_GUIDANCE
        period = evidence.guidance_period_label or f"fiscal {evidence.guidance_fiscal_year}"
        derivation = (
            f"Year 1 anchored on current management guidance of "
            f"{evidence.guidance_low:.1%} to {evidence.guidance_high:.1%} "
            f"(midpoint {anchor:.2%}) for {period}. source_metric="
            f"{evidence.guidance_source_metric}.")
        if evidence.guidance_period_type == "quarter":
            derivation += (" This is guidance for ONE QUARTER, treated as near-term evidence "
                           "about the current trajectory rather than as a multi-year forecast.")
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
        derivation = (f"No current guidance was available. Year 1 anchored on the "
                      f"trailing-twelve-month revenue trend of {anchor:.2%}.")
        evidence_ids = ("dcf.input.revenue_ttm",)
    elif evidence.history_is_broken and evidence.post_break_cagr is not None:
        anchor = evidence.post_break_cagr
        path.anchor_source = AssumptionSourceType.HISTORICAL_CALCULATION
        derivation = (
            f"No current guidance or trailing-twelve-month trend was available, and this "
            f"company's reported history contains a structural break. Year 1 anchored on the "
            f"growth of the {len(evidence.post_break_periods)} COMPARABLE periods since the "
            f"break ({anchor:.2%}) rather than on the full-history CAGR, which spans the break.")
        evidence_ids = ()
    elif evidence.latest_annual_yoy is not None:
        anchor = evidence.latest_annual_yoy
        path.anchor_source = AssumptionSourceType.HISTORICAL_CALCULATION
        derivation = (f"No current guidance or trailing-twelve-month trend was available. "
                      f"Year 1 anchored on the latest reported annual growth of {anchor:.2%}.")
        evidence_ids = ()
    elif evidence.historical_cagr is not None and not evidence.history_is_broken:
        anchor = evidence.historical_cagr
        path.anchor_source = AssumptionSourceType.HISTORICAL_CALCULATION
        derivation = (f"No current guidance, trailing-twelve-month trend or single-year growth "
                      f"was available. Year 1 anchored on the {len(evidence.historical_periods)}"
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

    if isinstance(guided, dict) and guided.get("low") is not None:
        anchor = (float(guided["low"]) + float(guided["high"])) / 2.0
        path.anchor_source = AssumptionSourceType.MANAGEMENT_GUIDANCE
        derivation = (f"Anchored on current management operating-margin guidance of "
                      f"{float(guided['low']):.1%} to {float(guided['high']):.1%}.")
        evidence_ids = tuple(filter(None, (guided.get("evidence_id"),)))
    elif (revenue is not None and operating_income is not None
            and revenue.value and operating_income.value is not None
            and revenue.source == "ttm_calculation"):
        anchor = operating_income.value / revenue.value
        path.anchor_source = AssumptionSourceType.TTM_CALCULATION
        derivation = (f"Anchored on the trailing-twelve-month operating margin "
                      f"({operating_income.value:,.0f} / {revenue.value:,.0f} = {anchor:.2%}) "
                      f"for {revenue.period_start}..{revenue.as_of_date}.")
        evidence_ids = ("dcf.input.revenue_ttm", "dcf.input.operating_income_ttm")
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
            derivation = (f"No reported operating margin was available; using the configured "
                          f"default of {anchor:.2%}.")

    path.anchor_value = anchor
    # Margin is held flat by default. Unlike growth there is no neutral
    # long-run margin to fade toward -- a company's structural margin IS its
    # own history -- so inventing a convergence target would be a stronger
    # claim than holding the observed level.
    for index in range(forecast_years):
        value, clamped = _clamp(anchor + margin_delta, MARGIN_BOUNDS)
        path.values.append(round(value, 6))
        path.entries.append(AssumptionEntry(
            field="operating_margin", forecast_year=index + 1, scenario=scenario,
            value=round(value, 6), units="ratio", source_type=path.anchor_source,
            evidence_ids=evidence_ids, derivation=derivation + (
                f" CLAMPED to the configured bound." if clamped else ""),
            clamped=clamped,
            original_proposed_value=round(anchor + margin_delta, 6) if clamped else None,
            applied_value=round(value, 6)))
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
