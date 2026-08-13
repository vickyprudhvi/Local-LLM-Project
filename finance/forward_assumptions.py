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
from finance.dcf import AssumptionSourceType

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
    """

    guidance_low: Optional[float] = None
    guidance_high: Optional[float] = None
    guidance_fiscal_year: Optional[int] = None
    guidance_evidence_id: Optional[str] = None
    ttm_yoy: Optional[float] = None
    latest_annual_yoy: Optional[float] = None
    historical_cagr: Optional[float] = None
    historical_periods: Tuple[str, ...] = ()

    @property
    def guidance_midpoint(self) -> Optional[float]:
        if self.guidance_low is None or self.guidance_high is None:
            return None
        return (self.guidance_low + self.guidance_high) / 2.0

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
                            company_facts: Optional[dict] = None) -> GrowthEvidence:
    """Assemble every growth signal, WITHOUT choosing between them yet."""
    evidence = GrowthEvidence()

    guidance = (state.management_guidance or {}).get("metrics") or {}
    revenue_growth = guidance.get("revenue_growth")
    if isinstance(revenue_growth, dict) and revenue_growth.get("low") is not None:
        evidence.guidance_low = float(revenue_growth["low"])
        evidence.guidance_high = float(revenue_growth["high"])
        evidence.guidance_fiscal_year = revenue_growth.get("fiscal_year")
        evidence.guidance_evidence_id = revenue_growth.get("evidence_id")

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

    historical = state.historical_metrics or {}
    if evidence.historical_cagr is None and historical.get("revenue_cagr") is not None:
        evidence.historical_cagr = float(historical["revenue_cagr"])
    if evidence.latest_annual_yoy is None and historical.get("revenue_growth_yoy") is not None:
        evidence.latest_annual_yoy = float(historical["revenue_growth_yoy"])
    return evidence


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
    """Year-1 growth by section 9's precedence, then a fade to the long-run rate.

    Precedence, highest first:
        current management guidance
        current TTM-over-TTM trend
        latest reported annual year-over-year
        historical normalized growth (CAGR)
        configured default
    """
    long_run = long_run_growth if long_run_growth is not None else config.dcf_long_run_growth()
    path = ForwardPath(field="revenue_growth")

    if evidence.guidance_midpoint is not None:
        anchor = evidence.guidance_midpoint
        path.anchor_source = AssumptionSourceType.MANAGEMENT_GUIDANCE
        derivation = (
            f"Year 1 anchored on current management guidance of "
            f"{evidence.guidance_low:.1%} to {evidence.guidance_high:.1%} "
            f"(midpoint {anchor:.2%}) for fiscal {evidence.guidance_fiscal_year}.")
        evidence_ids = tuple(filter(None, (evidence.guidance_evidence_id,)))
    elif evidence.ttm_yoy is not None:
        anchor = evidence.ttm_yoy
        path.anchor_source = AssumptionSourceType.TTM_CALCULATION
        derivation = (f"No current guidance was available. Year 1 anchored on the "
                      f"trailing-twelve-month revenue trend of {anchor:.2%}.")
        evidence_ids = ("dcf.input.revenue_ttm",)
    elif evidence.latest_annual_yoy is not None:
        anchor = evidence.latest_annual_yoy
        path.anchor_source = AssumptionSourceType.HISTORICAL_CALCULATION
        derivation = (f"No current guidance or trailing-twelve-month trend was available. "
                      f"Year 1 anchored on the latest reported annual growth of {anchor:.2%}.")
        evidence_ids = ()
    elif evidence.historical_cagr is not None:
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
        derivation = (f"No growth evidence of any kind was available; using the configured "
                      f"default of {anchor:.2%}.")
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
        path.entries.append(AssumptionEntry(
            field="revenue_growth", forecast_year=index + 1, scenario=scenario,
            value=round(value, 6), units="ratio", source_type=path.anchor_source,
            evidence_ids=evidence_ids,
            derivation=(derivation + f" Year {index + 1} of {forecast_years} fades linearly "
                        f"toward the configured long-run rate of {long_run:.2%}."
                        + (f" CLAMPED from {raw:.2%} to the configured bound."
                           if clamped else "")),
            clamped=clamped,
            original_proposed_value=round(raw, 6) if clamped else None,
            applied_value=round(value, 6)))

    if evidence.historical_cagr is not None \
            and path.anchor_source != AssumptionSourceType.HISTORICAL_CALCULATION:
        path.notes.append(
            f"Historical revenue CAGR of {evidence.historical_cagr:.2%} over "
            f"{evidence.historical_span} is retained as backward-looking CONTEXT and was "
            "deliberately not used as the forecast anchor.")
    return path


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
            f"- CURRENT MANAGEMENT GUIDANCE, fiscal {evidence.guidance_fiscal_year} revenue "
            f"growth: {evidence.guidance_low:.1%} to {evidence.guidance_high:.1%} "
            f"[{evidence.guidance_evidence_id}] (FORWARD-LOOKING evidence)")
    else:
        lines.append("- CURRENT MANAGEMENT GUIDANCE: unavailable")
    if evidence.ttm_yoy is not None:
        lines.append(f"- Trailing-twelve-month revenue growth vs prior twelve months: "
                     f"{evidence.ttm_yoy:.2%} [dcf.input.revenue_ttm]")
    if evidence.latest_annual_yoy is not None:
        lines.append(f"- Latest reported annual revenue growth: {evidence.latest_annual_yoy:.2%} "
                     "(HISTORICAL)")
    if evidence.historical_cagr is not None:
        lines.append(f"- Historical revenue CAGR over {len(evidence.historical_periods)} reported "
                     f"periods: {evidence.historical_cagr:.2%} (HISTORICAL CONTEXT ONLY)")
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
                              terminal_growth: Optional[float] = None
                              ) -> Tuple[Dict[str, ForwardPath], GrowthEvidence]:
    """The deterministic baseline for one scenario.

    `terminal_growth` is the rate the explicit forecast fades TO, so the last
    forecast year joins the perpetuity continuously. Callers pass the
    scenario's own terminal growth; omitting it falls back to the configured
    long-run rate.
    """
    evidence = collect_growth_evidence(state, company_facts)
    return {
        "revenue_growth": build_growth_path(evidence, forecast_years, scenario=scenario,
                                            growth_delta=growth_delta,
                                            long_run_growth=terminal_growth),
        "operating_margin": build_margin_path(state, forecast_years, scenario=scenario,
                                              margin_delta=margin_delta),
    }, evidence
