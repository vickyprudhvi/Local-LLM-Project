"""Phase H.15 — an invalid figure cannot be consumed as a number.

The architecture could already DETECT a bad canonical fact. It just carried
on computing with it. Measured on a synthetic issuer whose component debt
sum ($80.0B) disagreed with its own reported total ($95B) by 16%:

    total_debt = 80_001_000_000
    findings   = [DCF_NET_DEBT_COMPONENT_OVERLAP]

A warning beside a usable number. Everything downstream — net debt, leverage,
the equity bridge, the modelled value per share, the market comparison, the
risk that followed from it — consumed the 80B without ever consulting the
finding. `validation_status` existed on every metric and was *copied*
everywhere and *checked* nowhere.

So validity here is not an annotation. It is a property of the value's
availability: an INVALID metric has no `value` to read. Code that wants the
number gets None and has to decide what to do, which is the only way a
guarantee survives contact with callers that were written before it.

Invalidity then propagates along DECLARED dependencies, and only those.
A debt conflict must not invalidate revenue.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


class Validity:
    """What may be done with a figure.

    LIMITED is deliberately usable: a trailing window that ends a little
    early is still the best available measure of the thing it measures, and
    refusing it would cost more than it protects. INVALID is not usable at
    all -- it means the number is not known to describe what it claims to.
    """

    VALID = "VALID"
    LIMITED = "LIMITED"
    INVALID = "INVALID"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNAVAILABLE = "UNAVAILABLE"

    ALL = (VALID, LIMITED, INVALID, NOT_APPLICABLE, UNAVAILABLE)

    # States a downstream calculation may read a number from.
    USABLE = (VALID, LIMITED)

    @classmethod
    def worst(cls, states) -> str:
        """The weakest state among several.

        A derived figure is never sounder than the weakest thing it was
        built from, so this is what a derivation inherits.
        """
        rank = {cls.VALID: 0, cls.LIMITED: 1, cls.UNAVAILABLE: 2,
                cls.NOT_APPLICABLE: 3, cls.INVALID: 4}
        found = [s for s in (states or ()) if s in rank]
        return max(found, key=lambda s: rank[s]) if found else cls.UNAVAILABLE


@dataclass
class ValidatedMetric:
    """A number that can refuse to be read.

    `value` is None whenever the metric is not usable. The figure itself
    survives on `raw_value` for the audit trail and the debug view -- a
    reader investigating why a valuation was withheld needs to see the
    number that failed -- but nothing on the calculation path reaches for
    `raw_value`, and the name is deliberately awkward so that reaching for
    it is a visible decision rather than an accident.
    """

    metric_id: str
    raw_value: Optional[float] = None
    validity: str = Validity.VALID
    reasons: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    source_metric_ids: Tuple[str, ...] = ()
    source_evidence_ids: Tuple[str, ...] = ()
    derivation: str = ""
    period: Optional[str] = None

    @property
    def value(self) -> Optional[float]:
        """The number, or None when it may not be used."""
        return self.raw_value if self.usable else None

    @property
    def usable(self) -> bool:
        return self.validity in Validity.USABLE and self.raw_value is not None

    def invalidate(self, code: str, reason: str) -> "ValidatedMetric":
        """Mark unusable, keeping the figure for the audit view only."""
        self.validity = Validity.INVALID
        if code and code not in self.reason_codes:
            self.reason_codes.append(code)
        if reason and reason not in self.reasons:
            self.reasons.append(reason)
        return self

    def to_dict(self) -> dict:
        return {
            "metric_id": self.metric_id,
            "value": self.value,
            "raw_value": self.raw_value,
            "validity": self.validity,
            "reasons": list(self.reasons),
            "reason_codes": list(self.reason_codes),
            "source_metric_ids": list(self.source_metric_ids),
            "source_evidence_ids": list(self.source_evidence_ids),
            "derivation": self.derivation,
            "period": self.period,
        }


# ---------------------------------------------------------------------------
# Section 2 — the dependency graph
# ---------------------------------------------------------------------------
#
# Declared, not inferred. Every edge is a statement that the left metric
# cannot be computed without the right ones, and section 21 is as important
# as section 1: a debt conflict must invalidate leverage and the equity
# bridge, and must NOT touch revenue, margin or the technicals.

DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    # -- balance sheet ------------------------------------------------------
    "net_debt": ("total_debt", "eligible_liquidity"),
    "debt_to_equity": ("total_debt", "stockholders_equity"),
    "net_debt_to_equity": ("net_debt", "stockholders_equity"),
    "cash_to_debt": ("cash_and_cash_equivalents", "total_debt"),
    "debt_to_fcf": ("total_debt", "free_cash_flow"),
    "net_debt_to_fcf": ("net_debt", "free_cash_flow"),
    "current_ratio": ("current_assets", "current_liabilities"),
    "quick_ratio": ("current_assets", "current_liabilities"),

    # -- flows and margins --------------------------------------------------
    "free_cash_flow": ("operating_cash_flow", "capital_expenditure"),
    "operating_margin": ("operating_income", "revenue"),
    "net_margin": ("net_income", "revenue"),
    "free_cash_flow_margin": ("free_cash_flow", "revenue"),
    "operating_cash_flow_margin": ("operating_cash_flow", "revenue"),
    "revenue_growth": ("revenue",),
    "roe_ending_equity": ("net_income", "stockholders_equity"),

    # -- the valuation chain (section 1's worked example) -------------------
    "dcf_equity_bridge": ("net_debt",),
    "dcf_equity_value": ("dcf_enterprise_value", "dcf_equity_bridge"),
    "modeled_value_per_share": ("dcf_equity_value", "share_basis"),
    "market_price_comparison": ("modeled_value_per_share",),
    "valuation_derived_risk": ("market_price_comparison",),
    "valuation_recommendation_reason": ("valuation_derived_risk",),
}


def dependents_of(metric_id: str) -> List[str]:
    """Every metric that depends on `metric_id`, transitively."""
    out, frontier = [], [metric_id]
    while frontier:
        current = frontier.pop()
        for name, sources in DEPENDENCIES.items():
            if current in sources and name not in out:
                out.append(name)
                frontier.append(name)
    return out


def propagate(metrics: Dict[str, ValidatedMetric]) -> Dict[str, ValidatedMetric]:
    """Section 1: invalidity flows to everything that depends on it.

    Runs to a fixed point, so a chain (total_debt -> net_debt ->
    net_debt_to_equity) invalidates end to end rather than one level deep.
    Metrics not on a dependency path are untouched -- section 21.
    """
    metrics = dict(metrics or {})
    changed = True
    while changed:
        changed = False
        for name, sources in DEPENDENCIES.items():
            metric = metrics.get(name)
            if metric is None or metric.validity == Validity.INVALID:
                continue
            broken = [s for s in sources
                      if s in metrics and metrics[s].validity == Validity.INVALID]
            if not broken:
                continue
            metric.invalidate(
                "DEPENDENT_METRIC_INVALID",
                f"{name} cannot be computed: it depends on "
                f"{', '.join(sorted(broken))}, which "
                f"{'is' if len(broken) == 1 else 'are'} invalid.")
            metric.source_metric_ids = tuple(sources)
            changed = True
    return metrics


def root_causes(metrics: Dict[str, ValidatedMetric]) -> List[str]:
    """Section 13: the metrics that failed on their OWN account.

    Anything invalidated only by propagation is a consequence, and reporting
    it beside its cause is what produced "net debt appears unusually low"
    next to the debt conflict that caused it.
    """
    return sorted(
        name for name, metric in (metrics or {}).items()
        if metric.validity == Validity.INVALID
        and "DEPENDENT_METRIC_INVALID" not in metric.reason_codes)


def invalidated_by_propagation(metrics: Dict[str, ValidatedMetric]) -> List[str]:
    return sorted(
        name for name, metric in (metrics or {}).items()
        if metric.validity == Validity.INVALID
        and "DEPENDENT_METRIC_INVALID" in metric.reason_codes)


# ---------------------------------------------------------------------------
# Sections 5-7 — resolve where a rule proves it, invalidate otherwise
# ---------------------------------------------------------------------------

TOTAL_DEBT_CONFLICT = "TOTAL_DEBT_CONFLICT"
SHARE_BASIS_CONFLICT = "SHARE_BASIS_CONFLICT"

# How far two measurements of ONE quantity may differ before the
# disagreement is material. Above this, one of them is wrong and nothing
# here can say which.
MATERIAL_CONFLICT_TOLERANCE = 0.02


def resolve_or_invalidate(metric_id: str,
                          component_sum: Optional[float],
                          reported_total: Optional[float],
                          reported_is_authoritative: bool,
                          reported_concept: Optional[str] = None) -> ValidatedMetric:
    """Sections 5-7: a material conflict resolves deterministically or fails.

    Precedence is applied ONLY after semantic identity is established -- the
    caller passes `reported_is_authoritative` having already checked that the
    reported figure covers the same quantity and period (section 7). An
    issuer's own consolidated total outranks a sum this system assembled from
    parts it may have selected incompletely; a figure whose identity is not
    established outranks nothing.

    With no authoritative resolution the metric is INVALID. Promoting one
    side of an unexplained 16% disagreement is a guess, and a guess that
    reaches a valuation is indistinguishable from a measurement.
    """
    metric = ValidatedMetric(metric_id=metric_id, raw_value=component_sum,
                             derivation="sum of reported components")

    if component_sum is None and reported_total is None:
        metric.validity = Validity.UNAVAILABLE
        metric.reasons.append(f"no {metric_id} could be selected or reported")
        return metric

    if component_sum is None:
        if reported_is_authoritative:
            metric.raw_value = reported_total
            metric.derivation = f"issuer-reported {reported_concept or 'total'}"
            return metric
        metric.validity = Validity.UNAVAILABLE
        metric.reasons.append(
            f"only a non-authoritative {reported_concept or 'reported'} figure was "
            f"available for {metric_id}")
        return metric

    if reported_total is None:
        return metric

    scale = max(abs(component_sum), abs(reported_total))
    if scale == 0 or abs(component_sum - reported_total) / scale <= MATERIAL_CONFLICT_TOLERANCE:
        return metric

    if reported_is_authoritative:
        # Section 6: resolved, and the resolution is stated.
        metric.raw_value = reported_total
        metric.derivation = (
            f"issuer-reported {reported_concept or 'total'} (the component sum of "
            f"{component_sum:,.0f} was incomplete)")
        metric.reasons.append(
            f"The components summed to {component_sum:,.0f} against an issuer-reported "
            f"{reported_total:,.0f}. The issuer's own consolidated figure is used: it covers "
            "the same quantity and period, and a sum assembled here can omit a component "
            "the issuer included.")
        return metric

    return metric.invalidate(
        TOTAL_DEBT_CONFLICT if metric_id == "total_debt" else f"{metric_id.upper()}_CONFLICT",
        f"{metric_id} could not be reconciled: components sum to {component_sum:,.0f} while "
        f"{reported_concept or 'the reported figure'} is {reported_total:,.0f}, a difference "
        f"of {abs(component_sum - reported_total) / scale:.1%}. Neither is established as "
        "authoritative, so the figure is not used.")
