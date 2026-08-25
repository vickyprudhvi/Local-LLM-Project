"""Phase H.11, sections 4-6 — growth is not one number.

A live report showed a trailing-twelve-month revenue base current to the
latest quarter and, two rows below it, a "Revenue Growth" of 10.9% that was
the prior fiscal year's annual change. Both figures were correct. Only one of
them described the period the report said it was describing, and nothing on
the page distinguished them, because both had arrived as a bare float named
`revenue_growth`.

The fix is not a better fallback. It is that "revenue growth" stops being a
single name. A trailing-twelve-month year-over-year change, a latest-quarter
year-over-year change, a fiscal-year change and a five-year compound rate are
four different measurements of four different periods, and each gets its own
identity, its own evidence id, and its own period stamped on it. A consumer
that wants the current one asks for the current one; there is no generic name
left to reach for by accident.

Every rate here is checked by `finance.semantics` before it is computed --
same metric, same currency, comparable durations, non-overlapping windows --
so a growth rate that cannot be justified is absent rather than approximate.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from finance import semantics as sem
# NOTE: `finance.ttm.build_ttm` and `finance.freshness.build_ttm` are two
# different builders with two different field vocabularies -- this one
# exposes `start_date`/`end_date`, the freshness planner's exposes
# `period_start`/`period_end`. Using the wrong pair raises, and an earlier
# version of this module swallowed that in a bare `except` and silently fell
# through to the next growth kind. Both mistakes are fixed; the duplication
# itself is recorded in the phase notes as a consolidation candidate.
from finance import ttm as ttm_module


class GrowthKind:
    """The four ways this project measures a change in a flow."""

    TTM_YOY = "TTM_YOY_GROWTH"
    LATEST_QUARTER_YOY = "LATEST_QUARTER_YOY_GROWTH"
    YTD_YOY = "YTD_YOY_GROWTH"
    FY_YOY = "FY_YOY_GROWTH"
    HISTORICAL_CAGR = "HISTORICAL_CAGR"

    ALL = (TTM_YOY, LATEST_QUARTER_YOY, YTD_YOY, FY_YOY, HISTORICAL_CAGR)


# Section 4: the evidence id each kind is published under. A bare
# `revenue_growth` appears nowhere -- that ambiguity is the bug.
EVIDENCE_IDS = {
    GrowthKind.TTM_YOY: "current.ttm.revenue_growth_yoy",
    GrowthKind.LATEST_QUARTER_YOY: "current.quarter.revenue_growth_yoy",
    GrowthKind.YTD_YOY: "current.ytd.revenue_growth_yoy",
    GrowthKind.FY_YOY: "historical.fy.revenue_growth_yoy",
    GrowthKind.HISTORICAL_CAGR: "historical.5y.revenue_cagr",
}

# Section 6: which kinds may represent CURRENT growth, best first. A fiscal
# year is deliberately last and is labelled when it is used -- it is the
# fallback, not a synonym.
CURRENT_PRECEDENCE = (GrowthKind.TTM_YOY, GrowthKind.YTD_YOY,
                      GrowthKind.LATEST_QUARTER_YOY, GrowthKind.FY_YOY)

# How each kind is described when its value is rendered, so a reader always
# knows which period a growth figure covers (section 10).
LABELS = {
    GrowthKind.TTM_YOY: "TTM YoY",
    GrowthKind.LATEST_QUARTER_YOY: "latest quarter YoY",
    GrowthKind.YTD_YOY: "year-to-date YoY",
    GrowthKind.FY_YOY: "last fiscal year YoY",
    GrowthKind.HISTORICAL_CAGR: "5-year CAGR",
}


@dataclass
class GrowthMetric:
    """One growth rate that knows what it measured and over what."""

    kind: str
    value: Optional[float]
    metric: str = "revenue"
    current_period: Optional[str] = None
    comparison_period: Optional[str] = None
    evidence_id: str = ""
    definition: str = ""
    validation_status: Optional[str] = None
    reason: Optional[str] = None

    @property
    def label(self) -> str:
        return LABELS.get(self.kind, self.kind)

    @property
    def ok(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "value": self.value, "metric": self.metric,
            "current_period": self.current_period,
            "comparison_period": self.comparison_period,
            "evidence_id": self.evidence_id, "definition": self.definition,
            "validation_status": self.validation_status, "reason": self.reason,
        }


@dataclass
class GrowthSet:
    """Every growth rate this issuer supports, plus which one is current."""

    metrics: Dict[str, GrowthMetric] = field(default_factory=dict)
    current_kind: Optional[str] = None
    warnings: list = field(default_factory=list)

    @property
    def current(self) -> Optional[GrowthMetric]:
        if self.current_kind is None:
            return None
        return self.metrics.get(self.current_kind)

    def to_dict(self) -> dict:
        return {
            "metrics": {k: m.to_dict() for k, m in self.metrics.items()},
            "current_kind": self.current_kind,
            "current_label": (self.current.label if self.current else None),
            "warnings": list(self.warnings),
        }


def _fact(value, frequency, start, end, currency=None):
    return sem.SemanticFact(
        metric_id=sem.MetricIdentity.REVENUE, value=value, units="currency",
        currency=currency, period_frequency=frequency,
        start_date=start, end_date=end,
        flow_or_instant=sem.FlowOrInstant.FLOW,
        accounting_basis=sem.AccountingBasis.GAAP)


def _windows_overlap(current_start, prior_end) -> bool:
    """Section 5: the two windows must not share months.

    A trailing twelve months against a window ending after the current one
    began is measuring part of the period against itself, and the resulting
    rate is neither a growth rate nor obviously wrong on inspection.
    """
    if not current_start or not prior_end:
        return False
    return prior_end >= current_start


def ttm_yoy_growth(company_facts: dict, metric: str = "revenue") -> GrowthMetric:
    """Section 5: current trailing twelve months over the prior comparable one.

    `finance.ttm` already knows how to step the window back four quarters;
    nothing here reimplements the construction. What this adds is the
    validation the comparison itself needs -- both windows twelve months,
    same currency, no overlap -- before the division happens.
    """
    result = GrowthMetric(kind=GrowthKind.TTM_YOY, value=None, metric=metric,
                          evidence_id=EVIDENCE_IDS[GrowthKind.TTM_YOY],
                          definition=("current trailing twelve months over the trailing "
                                      "twelve months ending one year earlier"))
    current = ttm_module.build_ttm(company_facts, metric)
    if not (current.ok and current.value):
        result.reason = f"no current trailing twelve months of {metric} could be built"
        return result
    prior = ttm_module.build_ttm(company_facts, metric, offset=4)
    if not (prior.ok and prior.value):
        result.reason = (f"no prior-year trailing twelve months of {metric} could be built, so "
                         "there is nothing comparable to measure against")
        return result

    result.current_period = f"{current.start_date}..{current.end_date}"
    result.comparison_period = f"{prior.start_date}..{prior.end_date}"

    if _windows_overlap(current.start_date, prior.end_date):
        result.reason = (f"the two trailing windows overlap ({result.comparison_period} ends on "
                         f"or after {current.start_date}), so the comparison would measure "
                         "part of the period against itself")
        return result

    verdict = sem.compatible_for(
        sem.Operation.GROWTH,
        _fact(current.value, sem.PeriodFrequency.TTM, current.start_date, current.end_date),
        _fact(prior.value, sem.PeriodFrequency.TTM, prior.start_date, prior.end_date))
    if not verdict.allowed:
        result.reason = verdict.reason
        return result

    result.value = (current.value - prior.value) / abs(prior.value)
    # A rate built from a PARTIAL window is itself no better than PARTIAL.
    # `worst_of` takes the statuses as separate arguments, not a list.
    result.validation_status = ttm_module.TtmValidation.worst_of(
        current.validation_status, prior.validation_status)
    return result


def latest_quarter_yoy_growth(company_facts: dict, metric: str = "revenue") -> GrowthMetric:
    """The latest discrete quarter against the SAME quarter a year earlier.

    Against the same quarter, never the preceding one: a sequential change in
    a seasonal business is not a growth rate, and `finance.semantics` refuses
    that comparison outright.
    """
    from finance import period_facts as pf

    result = GrowthMetric(kind=GrowthKind.LATEST_QUARTER_YOY, value=None, metric=metric,
                          evidence_id=EVIDENCE_IDS[GrowthKind.LATEST_QUARTER_YOY],
                          definition="latest reported quarter over the same quarter a year earlier")
    series = pf.discrete_quarters(company_facts, metric)
    quarters = [q for q in series.quarters if q.value is not None]
    if len(quarters) < 5:
        result.reason = (f"only {len(quarters)} discrete quarter(s) of {metric} are available; "
                         "five are needed to compare the latest against its prior-year twin")
        return result
    current, prior = quarters[-1], quarters[-5]
    result.current_period = f"{current.start}..{current.end}"
    result.comparison_period = f"{prior.start}..{prior.end}"
    if not prior.value:
        result.reason = "the prior-year quarter reported no value to compare against"
        return result

    verdict = sem.compatible_for(
        sem.Operation.GROWTH,
        _fact(current.value, sem.PeriodFrequency.QUARTER, current.start, current.end),
        _fact(prior.value, sem.PeriodFrequency.QUARTER, prior.start, prior.end))
    if not verdict.allowed:
        result.reason = verdict.reason
        return result
    result.value = (current.value - prior.value) / abs(prior.value)
    return result


def build_growth_set(company_facts: dict, historical_metrics: Optional[dict] = None,
                     metric: str = "revenue") -> GrowthSet:
    """Every growth rate this issuer supports, and which is CURRENT.

    Section 6: the current rate is chosen by explicit precedence over what
    could actually be built, and the choice is recorded so the renderer can
    label it. A fiscal-year rate reaching the current slot is not a failure
    -- for an issuer that files no usable quarters it is the only honest
    answer -- but it is named as a fiscal-year rate rather than presented as
    though it described today.
    """
    result = GrowthSet()

    for kind, builder in ((GrowthKind.TTM_YOY, ttm_yoy_growth),
                          (GrowthKind.LATEST_QUARTER_YOY, latest_quarter_yoy_growth)):
        try:
            growth = builder(company_facts, metric)
        except Exception as exc:  # a diagnostic must not take down an analysis
            # ...but it must not disappear either. A builder that raised used
            # to be skipped entirely, so the rate was absent from the set with
            # nothing saying why, and the precedence below silently fell
            # through to the next kind. The failure is now a recorded member
            # with its reason, which is how it was caught.
            growth = GrowthMetric(kind=kind, value=None, metric=metric,
                                  evidence_id=EVIDENCE_IDS[kind],
                                  reason=f"could not be computed: {exc}")
        result.metrics[growth.kind] = growth
        if growth.reason and not growth.ok:
            result.warnings.append(f"{growth.kind}: {growth.reason}")

    # The historical rates already computed from the annual statements keep
    # their own identities and stay in the historical namespace.
    for source_key, kind in (("revenue_growth_yoy", GrowthKind.FY_YOY),
                             ("revenue_cagr", GrowthKind.HISTORICAL_CAGR)):
        entry = (historical_metrics or {}).get(source_key)
        value = entry.get("value") if isinstance(entry, dict) else entry
        if value is None:
            continue
        periods = [p for p in ((entry.get("inputs") if isinstance(entry, dict) else None) or []) if p]
        result.metrics[kind] = GrowthMetric(
            kind=kind, value=value, metric=metric,
            current_period=periods[0] if periods else None,
            comparison_period=periods[1] if len(periods) > 1 else None,
            evidence_id=EVIDENCE_IDS[kind],
            definition=("change over the last two reported fiscal years"
                        if kind == GrowthKind.FY_YOY
                        else "compound annual rate across the reported fiscal years"))

    for kind in CURRENT_PRECEDENCE:
        candidate = result.metrics.get(kind)
        if candidate is not None and candidate.ok:
            result.current_kind = kind
            break
    return result
