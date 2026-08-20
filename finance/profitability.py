"""Phase H.8 — the ProfitabilityNormalizationLayer.

THE BUG THIS EXISTS TO FIX (live large-cap pharmaceutical, 2026-08-19)
======================================================================
A company with $66.6B of trailing revenue, $20.0B of operating cash flow and
$16.1B of free cash flow was valued at $0.20 per share against a market price
of $149.93 — a premium of roughly 75,800%, rendered to one decimal place. The
DCF was labelled SUITABLE_WITH_HIGH_UNCERTAINTY.

The chain:

    the issuer does not tag `OperatingIncomeLoss` at all
    -> `operating_income` resolves to None
    -> every step of the margin precedence falls through
    -> the CONFIGURED DEFAULT of 10% becomes the five-year forecast
    -> a 10% margin on this company's cost base produces almost no FCFF
    -> equity value collapses to ~zero
    -> the arithmetic validates, so the run reports a valid DCF

Two independent defects. The default silently BECAME the forecast rather than
being a last resort that degrades the verdict; and nothing noticed that a
10% assumed operating margin sits beside a 30% reported operating-cash-flow
margin, which is a contradiction visible without any judgement at all.

Underneath both sits the real problem this module addresses: the company's
trailing profitability is genuinely distorted. Its trailing net income is
$3.2B on $66.6B of revenue — a 4.8% net margin — while it generated $20.0B of
operating cash. The gap is a large acquisition-related research charge. That
is a REPORTED fact and it is not a RECURRING one, and the pipeline had no way
to hold both ideas at once.

WHAT THIS MODULE DOES
=====================
It keeps four things apart that were previously one number:

    REPORTED            what the filings say, never overwritten
    NORMALIZED          reported, adjusted for identified unusual items,
                        with every adjustment itemized and reversible
    COMPANY-ADJUSTED    the issuer's own non-GAAP measure -- evidence, not
                        authority
    FORECAST            what the DCF assumes, which is none of the above

Normalization is deterministic and evidence-bound. An unusual item must have
a TAGGED AMOUNT, a known period and a known income-statement location before
it can move a number. A charge mentioned in prose without a figure produces a
WARNING and an UNRESOLVED status -- never an adjustment, and never a silent
one. The local model never proposes an adjustment and never sees a filing.

Nothing here is issuer-specific: unusual items are read from reviewed XBRL
concepts, and the cross-metric conflict test is arithmetic on normalized
fields.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Deterministic guard codes.
PROFITABILITY_CROSS_METRIC_CONFLICT = "PROFITABILITY_CROSS_METRIC_CONFLICT"
CONFIGURED_MARGIN_CONFLICTS_WITH_REPORTED_PROFITABILITY = \
    "CONFIGURED_MARGIN_CONFLICTS_WITH_REPORTED_PROFITABILITY"
NORMALIZATION_UNRESOLVED = "NORMALIZATION_UNRESOLVED"


class UnusualItemType:
    """Section 1. Categories, not company names."""

    IPRD = "IPRD"
    RESTRUCTURING = "RESTRUCTURING"
    IMPAIRMENT = "IMPAIRMENT"
    LITIGATION = "LITIGATION"
    ACQUISITION_COST = "ACQUISITION_COST"
    OTHER = "OTHER"
    ALL = (IPRD, RESTRUCTURING, IMPAIRMENT, LITIGATION, ACQUISITION_COST, OTHER)


class RecurrenceStatus:
    """Section 5. Conservative by construction.

    A restructuring programme that has run for five consecutive years is not
    non-recurring however it is labelled, and adding back 100% of it would
    flatter the normalized margin permanently. Recurrence is therefore
    measured from the issuer's own reporting history rather than assumed from
    the category.
    """

    NON_RECURRING = "NON_RECURRING"
    POTENTIALLY_RECURRING = "POTENTIALLY_RECURRING"
    RECURRING = "RECURRING"
    UNKNOWN = "UNKNOWN"
    ALL = (NON_RECURRING, POTENTIALLY_RECURRING, RECURRING, UNKNOWN)


class NormalizationStatus:
    VALID = "VALID"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"
    ALL = (VALID, PARTIAL, UNAVAILABLE)


class MetricBasis:
    """Section 6. Never silently mixed."""

    REPORTED_GAAP = "reported_GAAP"
    COMPANY_ADJUSTED = "company_adjusted_non_GAAP"
    SYSTEM_NORMALIZED = "system_normalized"
    DERIVED_FROM_GUIDANCE = "DERIVED_FROM_GUIDANCE"
    CONFIGURED_DEFAULT = "configured_default"
    ALL = (REPORTED_GAAP, COMPANY_ADJUSTED, SYSTEM_NORMALIZED,
           DERIVED_FROM_GUIDANCE, CONFIGURED_DEFAULT)


# How many of the last N annual periods a charge must appear in before it is
# treated as part of the cost base rather than as an unusual item. Three of
# five is deliberately strict: a company that restructures in most years is
# a company whose restructuring is an operating cost.
_RECURRENCE_WINDOW = 5
_RECURRING_THRESHOLD = 3
_POTENTIALLY_RECURRING_THRESHOLD = 2

# A charge below this share of revenue is not worth adjusting for: the
# normalization would move the margin by less than the noise between two
# reporting conventions, while adding an adjustment a reader has to check.
MATERIALITY_THRESHOLD = 0.01

# How far the operating-cash-flow margin and the operating margin may diverge
# before the pair is reporting a contradiction rather than ordinary accrual
# timing. Depreciation, working capital and non-cash charges routinely put
# 5-15 points between them; 20 points sustained is a different statement
# about the business.
CROSS_METRIC_CONFLICT_THRESHOLD = 0.20

# The same test applied to a CONFIGURED DEFAULT is deliberately tighter. Two
# reported measures disagreeing by fifteen points can both be describing the
# company; a default disagreeing with the company's own cash generation by
# ten points is not describing it at all, because a default is not evidence
# about this company in the first place.
CONFIGURED_DEFAULT_CONFLICT_THRESHOLD = 0.10


@dataclass(frozen=True)
class UnusualItem:
    """Section 1. One identified non-recurring item, with its evidence."""

    item_id: str
    metric_type: str
    amount: float
    currency: str
    period: str
    income_statement_location: str
    cash_effect: str = "unknown"          # cash | noncash | mixed | unknown
    tax_effect: Optional[float] = None
    recurrence_status: str = RecurrenceStatus.UNKNOWN
    source_evidence_ids: Tuple[str, ...] = ()
    confidence: float = 0.0
    concept: Optional[str] = None
    # Section 5. The level this charge runs at in an ORDINARY year, from the
    # issuer's own annual history. A company that books an item every year at
    # one scale and this year at another has a recurring base plus a
    # non-recurring excess, and only the excess is an unusual item.
    recurring_baseline: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id, "metric_type": self.metric_type,
            "amount": self.amount, "currency": self.currency, "period": self.period,
            "income_statement_location": self.income_statement_location,
            "cash_effect": self.cash_effect, "tax_effect": self.tax_effect,
            "recurrence_status": self.recurrence_status,
            "source_evidence_ids": list(self.source_evidence_ids),
            "confidence": self.confidence, "concept": self.concept,
            "recurring_baseline": self.recurring_baseline,
            "excess_over_baseline": self.excess_over_baseline,
        }

    @property
    def is_operating(self) -> bool:
        return self.income_statement_location == "operating"

    @property
    def excess_over_baseline(self) -> Optional[float]:
        """How much of this charge sits ABOVE what the issuer books in a
        normal year. None when there is no baseline to measure against."""
        if self.recurring_baseline is None:
            return None
        return max(0.0, self.amount - self.recurring_baseline)


@dataclass
class NormalizedMetric:
    """Section 4. A normalized figure that can be re-derived from what is
    printed beside it."""

    metric: str
    reported_value: Optional[float] = None
    normalized_value: Optional[float] = None
    adjustments: List[dict] = field(default_factory=list)
    normalization_status: str = NormalizationStatus.UNAVAILABLE
    warnings: List[str] = field(default_factory=list)
    basis: str = MetricBasis.REPORTED_GAAP

    @property
    def adjustment_total(self) -> float:
        return sum(a["amount"] * (1 if a["direction"] == "add_back" else -1)
                   for a in self.adjustments)

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "reported_value": self.reported_value,
            "normalized_value": self.normalized_value,
            "adjustments": [dict(a) for a in self.adjustments],
            "normalization_status": self.normalization_status,
            "warnings": list(self.warnings),
            "basis": self.basis,
        }


@dataclass
class ProfitabilityState:
    """Reported and normalized profitability, side by side (section 2)."""

    revenue: Optional[float] = None
    period: Optional[str] = None
    reported_operating_income: Optional[float] = None
    reported_operating_margin: Optional[float] = None
    reported_net_income: Optional[float] = None
    reported_net_margin: Optional[float] = None
    reported_tax_rate: Optional[float] = None
    operating_income_source: str = "reported"     # reported | derived | unavailable
    operating_income_derivation: Optional[str] = None

    normalized_operating_income: Optional[NormalizedMetric] = None
    normalized_operating_margin: Optional[float] = None
    normalized_net_income: Optional[NormalizedMetric] = None

    unusual_items: List[UnusualItem] = field(default_factory=list)
    findings: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_material_unusual_items(self) -> bool:
        return bool(self.unusual_items)

    @property
    def normalization_status(self) -> str:
        if self.normalized_operating_income is None:
            return NormalizationStatus.UNAVAILABLE
        return self.normalized_operating_income.normalization_status

    @property
    def best_operating_margin(self) -> Tuple[Optional[float], str]:
        """The margin a forecast should start from, and its basis.

        Normalized when a normalization actually held; reported otherwise.
        Never a configured default -- that decision belongs to the
        forward-assumption builder, which has to record it as such.
        """
        if self.normalized_operating_margin is not None \
                and self.normalization_status == NormalizationStatus.VALID:
            return self.normalized_operating_margin, MetricBasis.SYSTEM_NORMALIZED
        if self.reported_operating_margin is not None:
            return self.reported_operating_margin, MetricBasis.REPORTED_GAAP
        return None, MetricBasis.CONFIGURED_DEFAULT

    def to_dict(self) -> dict:
        return {
            "revenue": self.revenue,
            "period": self.period,
            "reported": {
                "operating_income": self.reported_operating_income,
                "operating_margin": self.reported_operating_margin,
                "net_income": self.reported_net_income,
                "net_margin": self.reported_net_margin,
                "tax_rate": self.reported_tax_rate,
                "operating_income_source": self.operating_income_source,
                "operating_income_derivation": self.operating_income_derivation,
            },
            "normalized": {
                "operating_income": (self.normalized_operating_income.to_dict()
                                     if self.normalized_operating_income else None),
                "operating_margin": self.normalized_operating_margin,
                "net_income": (self.normalized_net_income.to_dict()
                               if self.normalized_net_income else None),
                "status": self.normalization_status,
            },
            "unusual_items": [i.to_dict() for i in self.unusual_items],
            "findings": [dict(f) for f in self.findings],
            "warnings": list(self.warnings),
        }


def _finding(code: str, severity: str, message: str, **detail) -> dict:
    entry = {"code": code, "severity": severity, "message": message}
    entry.update(detail)
    return entry


# Normalized field -> (unusual-item category, income-statement location,
# cash effect). Only fields whose concepts carry an AMOUNT appear here.
UNUSUAL_ITEM_FIELDS = {
    "restructuring_charges": (UnusualItemType.RESTRUCTURING, "operating", "mixed"),
    "impairment_charges": (UnusualItemType.IMPAIRMENT, "operating", "noncash"),
    "litigation_charges": (UnusualItemType.LITIGATION, "operating", "cash"),
    "acquired_in_process_research_and_development": (
        UnusualItemType.IPRD, "operating", "mixed"),
    "acquisition_transaction_costs": (UnusualItemType.ACQUISITION_COST, "operating", "cash"),
}


# How far two windows' endpoints may differ and still be the same period.
# Fiscal quarter boundaries move by a day or two between filings; the same
# allowance finance/ttm.py makes when joining quarters.
_WINDOW_BOUNDARY_TOLERANCE_DAYS = 7


def _windows_match(start_a: Optional[str], end_a: Optional[str],
                   start_b: Optional[str], end_b: Optional[str]) -> bool:
    from finance.period_facts import _span_days

    if end_a is None or end_b is None:
        return False
    for left, right in ((start_a, start_b), (end_a, end_b)):
        if left is None or right is None:
            if left != right:
                return False
            continue
        gap = _span_days(left, right)
        if gap is None or abs(gap) > _WINDOW_BOUNDARY_TOLERANCE_DAYS:
            return False
    return True


def classify_recurrence(annual_values: Sequence[Optional[float]]) -> str:
    """Section 5, from the issuer's own history rather than from the label.

    A charge the company records in most years is part of its cost base
    whatever it is called, and adding it back would permanently flatter the
    normalized margin.
    """
    recent = [v for v in annual_values[-_RECURRENCE_WINDOW:] if v is not None]
    if not recent:
        return RecurrenceStatus.UNKNOWN
    appearances = sum(1 for v in recent if abs(v) > 0)
    if len(recent) < 2:
        return RecurrenceStatus.UNKNOWN
    if appearances >= _RECURRING_THRESHOLD:
        return RecurrenceStatus.RECURRING
    if appearances >= _POTENTIALLY_RECURRING_THRESHOLD:
        return RecurrenceStatus.POTENTIALLY_RECURRING
    return RecurrenceStatus.NON_RECURRING


def recurring_baseline(annual_values):
    """The level a repeatedly-booked charge runs at in an ORDINARY year.

    The MEDIAN of the issuer's recent annual values, deliberately, not the
    mean: the point of the measure is to describe the typical year, and a
    single outlier is exactly what is being isolated. A mean would let the
    outlier raise its own baseline and shrink the excess it exists to expose.
    """
    recent = sorted(abs(v) for v in annual_values[-_RECURRENCE_WINDOW:]
                    if v is not None and abs(v) > 0)
    if len(recent) < 2:
        return None
    middle = len(recent) // 2
    if len(recent) % 2:
        return recent[middle]
    return (recent[middle - 1] + recent[middle]) / 2.0


def build_unusual_items(company_facts: dict, period_start: Optional[str],
                        period_end: Optional[str], revenue: Optional[float],
                        currency: str = "USD") -> Tuple[List[UnusualItem], List[str]]:
    """Identify material unusual items over one period. Evidence-bound.

    Every item comes from a reviewed concept the issuer TAGGED WITH AN
    AMOUNT. There is no path here by which prose, a model, or an inference
    becomes an adjustment -- section 1's "do not let the LLM invent unusual
    items" is satisfied structurally rather than by instruction.
    """
    from finance import period_facts as pf
    from finance import ttm as ttm_module

    items: List[UnusualItem] = []
    warnings: List[str] = []
    if not company_facts or not period_end:
        return items, warnings

    for field_name, (category, location, cash_effect) in sorted(UNUSUAL_ITEM_FIELDS.items()):
        metric = ttm_module.build_ttm(company_facts, field_name)
        if not metric.ok or metric.value is None or metric.value == 0:
            continue
        # The window must be the SAME one the profitability figures cover, or
        # the adjustment moves a margin by an amount from a different period.
        #
        # "Same" tolerates the few days that fiscal quarter boundaries move by
        # between filings -- the same allowance finance/ttm.py already makes
        # when joining quarters. Requiring exact equality rejected a real
        # acquisition-related charge over a ONE-DAY difference (a window
        # starting 2025-06-30 against one starting 2025-07-01), which is a
        # reporting-convention artefact and not a period mismatch.
        if not _windows_match(metric.start_date, metric.end_date, period_start, period_end):
            warnings.append(
                f"{field_name} covers {metric.start_date}..{metric.end_date}, not the "
                f"{period_start}..{period_end} window the profitability figures cover; it "
                "was not used as an adjustment.")
            continue
        if revenue and abs(metric.value) / abs(revenue) < MATERIALITY_THRESHOLD:
            continue

        annual = pf.annual_periods(company_facts, field_name)
        annual_values = [p.value for p in annual]
        recurrence = classify_recurrence(annual_values)
        baseline = recurring_baseline(annual_values)
        items.append(UnusualItem(
            item_id=f"{field_name}.{period_end}",
            metric_type=category,
            amount=abs(metric.value),
            currency=currency,
            period=f"{metric.start_date}..{metric.end_date}",
            income_statement_location=location,
            cash_effect=cash_effect,
            recurrence_status=recurrence,
            source_evidence_ids=(f"financial.unusual_item.{field_name}.{period_end}",),
            confidence=0.9 if recurrence == RecurrenceStatus.NON_RECURRING else 0.5,
            concept=(metric.components[0].get("concept") if metric.components else None),
            recurring_baseline=baseline,
        ))
    return items, warnings


# How much of a charge is added back, by recurrence status. A
# POTENTIALLY_RECURRING charge is adjusted only in part, which is section 5's
# "if recurring nature is uncertain, do not automatically add back 100%"
# expressed as a number rather than a caveat.
_ADD_BACK_FRACTION = {
    RecurrenceStatus.NON_RECURRING: 1.0,
    RecurrenceStatus.POTENTIALLY_RECURRING: 0.5,
    RecurrenceStatus.RECURRING: 0.0,
    RecurrenceStatus.UNKNOWN: 0.0,
}


def normalize_operating_income(reported: Optional[float],
                               items: Sequence[UnusualItem]) -> NormalizedMetric:
    """Section 3/4: reported plus itemized add-backs, or an explicit refusal."""
    metric = NormalizedMetric(metric="operating_income", reported_value=reported,
                              basis=MetricBasis.SYSTEM_NORMALIZED)
    if reported is None:
        metric.normalization_status = NormalizationStatus.UNAVAILABLE
        metric.warnings.append(
            "No reported operating income was available, so there is nothing to normalize.")
        return metric

    operating_items = [i for i in items if i.is_operating]
    if not operating_items:
        metric.normalized_value = reported
        metric.normalization_status = NormalizationStatus.VALID
        return metric

    total = reported
    uncertain = False
    for item in operating_items:
        fraction = _ADD_BACK_FRACTION.get(item.recurrence_status, 0.0)
        adjustable = item.amount
        note = None

        # Section 5. A charge the issuer books EVERY year is part of the cost
        # base, and adding it back would permanently flatter the margin. But a
        # charge that runs at one scale in a normal year and at a different
        # scale this year is a recurring base PLUS a non-recurring excess, and
        # only the excess is unusual. A large-cap pharmaceutical that acquires
        # in-process research most years and then takes a charge many times
        # its own typical level is exactly this shape: refusing to adjust at
        # all leaves a distorted margin, while adding back the whole charge
        # asserts the company never does this.
        if item.recurrence_status == RecurrenceStatus.RECURRING:
            excess = item.excess_over_baseline
            if excess and item.recurring_baseline and excess > item.recurring_baseline:
                adjustable = excess
                fraction = 1.0
                uncertain = True
                note = (
                    f"{item.metric_type} of {item.amount:,.0f} for {item.period} is booked in "
                    f"most years -- about {item.recurring_baseline:,.0f} in an ordinary year "
                    f"-- so only the {excess:,.0f} EXCESS over that recurring level was added "
                    "back. The baseline stays in the cost base.")

        if fraction == 0.0:
            metric.warnings.append(
                f"{item.metric_type} of {item.amount:,.0f} for {item.period} was NOT added "
                f"back: it is classified {item.recurrence_status}, so treating it as "
                "non-recurring would permanently flatter the normalized margin.")
            uncertain = uncertain or item.recurrence_status == RecurrenceStatus.UNKNOWN
            continue
        if note:
            metric.warnings.append(note)
        applied = adjustable * fraction
        total += applied
        metric.adjustments.append({
            "item_id": item.item_id,
            "amount": applied,
            "direction": "add_back",
            "recurrence_status": item.recurrence_status,
            "fraction_applied": fraction,
            "evidence_ids": list(item.source_evidence_ids),
        })
        if fraction < 1.0:
            uncertain = True
            metric.warnings.append(
                f"{item.metric_type} of {item.amount:,.0f} is classified "
                f"{item.recurrence_status}; only {fraction:.0%} was added back.")

    metric.normalized_value = total
    metric.normalization_status = (NormalizationStatus.PARTIAL if uncertain
                                   else NormalizationStatus.VALID)
    return metric


def detect_cross_metric_conflict(revenue: Optional[float],
                                 operating_margin: Optional[float],
                                 operating_cash_flow: Optional[float],
                                 net_margin: Optional[float] = None) -> Optional[dict]:
    """Section 12: does the assumed/reported margin contradict the cash flows?

    Operating cash flow is the hardest number on the statements to distort,
    so a large sustained gap between the operating-cash-flow margin and the
    operating margin is a contradiction that needs stating. It is NOT
    resolved here -- the two figures are different measures and either could
    be the odd one out -- but it must never be resolved silently either.
    """
    if not revenue or operating_margin is None or operating_cash_flow is None:
        return None
    ocf_margin = operating_cash_flow / revenue
    gap = ocf_margin - operating_margin
    if abs(gap) < CROSS_METRIC_CONFLICT_THRESHOLD:
        return None
    return _finding(
        PROFITABILITY_CROSS_METRIC_CONFLICT, "warning",
        f"The operating margin in use ({operating_margin:.1%}) and the operating-cash-flow "
        f"margin ({ocf_margin:.1%}) differ by {abs(gap):.0f} percentage points "
        f"({gap:+.1%}). Accrual timing and non-cash charges routinely separate the two by a "
        "few points; a gap this size means at least one of them does not describe the "
        "company's recurring economics, and which one is not decided here.",
        operating_margin=operating_margin, operating_cash_flow_margin=ocf_margin,
        gap=round(gap, 6))


def detect_configured_margin_conflict(configured_margin: Optional[float],
                                      revenue: Optional[float],
                                      operating_cash_flow: Optional[float],
                                      net_margin: Optional[float]) -> Optional[dict]:
    """Section 21: a configured default that contradicts known economics.

    The default exists for a company whose profitability genuinely cannot be
    read. It is not a forecast, and when it sits far from every measure of
    the company that IS available, using it is not conservative -- it is a
    different company. This is the check that was missing when a 10% default
    was applied to an issuer generating 30% operating-cash-flow margins.
    """
    if configured_margin is None or not revenue:
        return None
    observed = []
    if operating_cash_flow is not None:
        observed.append(("operating-cash-flow margin", operating_cash_flow / revenue))
    if net_margin is not None:
        observed.append(("net margin", net_margin))
    if not observed:
        return None
    conflicts = [(label, value) for label, value in observed
                 if abs(value - configured_margin) >= CONFIGURED_DEFAULT_CONFLICT_THRESHOLD]
    if not conflicts:
        return None
    detail = ", ".join(f"{label} {value:.1%}" for label, value in conflicts)
    return _finding(
        CONFIGURED_MARGIN_CONFLICTS_WITH_REPORTED_PROFITABILITY, "error",
        f"The forecast uses a CONFIGURED DEFAULT operating margin of {configured_margin:.1%} "
        f"because this company's operating margin could not be read, but its reported "
        f"economics contradict that figure ({detail}). A default is a last resort, not an "
        "estimate: applying it here models a company materially different from the one "
        "being analyzed, and every valuation figure derived from it inherits that.",
        configured_margin=configured_margin,
        observed=[{"label": label, "value": value} for label, value in observed])


def build_profitability_state(*, revenue: Optional[float], period: Optional[str],
                              operating_income: Optional[float],
                              net_income: Optional[float],
                              operating_cash_flow: Optional[float] = None,
                              income_tax_expense: Optional[float] = None,
                              income_before_tax: Optional[float] = None,
                              company_facts: Optional[dict] = None,
                              period_start: Optional[str] = None,
                              period_end: Optional[str] = None,
                              currency: str = "USD") -> ProfitabilityState:
    """Assemble reported and normalized profitability for one period."""
    state = ProfitabilityState(revenue=revenue, period=period,
                               reported_operating_income=operating_income,
                               reported_net_income=net_income)

    if revenue:
        if operating_income is not None:
            state.reported_operating_margin = operating_income / revenue
        if net_income is not None:
            state.reported_net_margin = net_income / revenue
    if income_before_tax and income_tax_expense is not None and income_before_tax != 0:
        state.reported_tax_rate = income_tax_expense / income_before_tax

    items, warnings = build_unusual_items(company_facts or {}, period_start, period_end,
                                          revenue, currency)
    state.unusual_items = items
    state.warnings.extend(warnings)

    normalized = normalize_operating_income(operating_income, items)
    state.normalized_operating_income = normalized
    if normalized.normalized_value is not None and revenue:
        state.normalized_operating_margin = normalized.normalized_value / revenue

    margin_for_check, _basis = state.best_operating_margin
    conflict = detect_cross_metric_conflict(revenue, margin_for_check, operating_cash_flow,
                                            state.reported_net_margin)
    if conflict:
        state.findings.append(conflict)
    return state
