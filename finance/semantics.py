"""Phase H.10 — the financial semantics contract.

Every previous phase fixed a real bug and none of them stopped the next
stock from exposing a variant of the same one. The reason is visible in the
four failures that opened this phase:

  * a quarterly revenue level divided by trailing-twelve-month revenue,
  * a current share count differenced against a weighted-average one,
  * a debt component sum compared against a long-term-debt tag,
  * a broker-dealer's operating cash flow read as owner free cash flow.

Not one of those is an arithmetic mistake. Every division, subtraction and
comparison was computed correctly. What was wrong in each case is that two
numbers describing DIFFERENT THINGS were combined as though they described
the same thing -- and the code could not tell, because by the time the
values reached the arithmetic they were plain floats. A float remembers its
magnitude and nothing else: not the period it covers, not the basis it was
measured on, not whether it is a stock or a flow.

So this module does not add another check to another calculation. It gives
a financial figure the identity it needs to refuse an invalid operation,
and it puts one deterministic validator between "two facts" and "an
arithmetic result". The rule it enforces is a single sentence: two facts may
be combined only when the combination is meaningful for the operation being
requested.

Nothing here knows any ticker, company or sector. Every decision is made
from the metadata a fact carries.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Section 3 — canonical metric identities
# ---------------------------------------------------------------------------
#
# Deliberately an explicit vocabulary rather than free-form strings. Two
# metrics are the same metric when their ids are equal -- never because
# their names look similar, which is how "free cash flow" came to mean
# three different quantities in one report and how a long-term-debt tag came
# to stand in for total debt.

class MetricIdentity:
    # -- income statement --
    REVENUE = "REVENUE"
    SERVICE_REVENUE = "SERVICE_REVENUE"
    PRODUCT_REVENUE = "PRODUCT_REVENUE"
    OPERATING_INCOME = "OPERATING_INCOME"
    NET_INCOME = "NET_INCOME"
    GROSS_PROFIT = "GROSS_PROFIT"
    EBITDA = "EBITDA"

    # -- cash flow --
    OPERATING_CASH_FLOW = "OPERATING_CASH_FLOW"
    CAPEX = "CAPEX"
    SIMPLE_FCF = "SIMPLE_FCF"
    COMPANY_DEFINED_FCF = "COMPANY_DEFINED_FCF"
    ADJUSTED_FCF = "ADJUSTED_FCF"

    # -- balance sheet --
    CASH = "CASH"
    SHORT_TERM_INVESTMENTS = "SHORT_TERM_INVESTMENTS"
    MARKETABLE_SECURITIES = "MARKETABLE_SECURITIES"
    SHORT_TERM_DEBT = "SHORT_TERM_DEBT"
    LONG_TERM_DEBT = "LONG_TERM_DEBT"
    TOTAL_DEBT = "TOTAL_DEBT"
    NET_DEBT = "NET_DEBT"
    STOCKHOLDERS_EQUITY = "STOCKHOLDERS_EQUITY"

    # -- share counts --
    SHARES_CURRENT_OUTSTANDING = "SHARES_CURRENT_OUTSTANDING"
    SHARES_WEIGHTED_AVERAGE_BASIC = "SHARES_WEIGHTED_AVERAGE_BASIC"
    SHARES_WEIGHTED_AVERAGE_DILUTED = "SHARES_WEIGHTED_AVERAGE_DILUTED"
    SHARES_ECONOMIC_CURRENT = "SHARES_ECONOMIC_CURRENT"
    SHARES_ADR_EQUIVALENT = "SHARES_ADR_EQUIVALENT"

    UNKNOWN = "UNKNOWN"


# The three free-cash-flow definitions are separate identities above, which
# is the point of listing them; this set exists so a check can ask "is this
# one of the things people call free cash flow" without string matching.
FCF_IDENTITIES = frozenset({
    MetricIdentity.SIMPLE_FCF,
    MetricIdentity.COMPANY_DEFINED_FCF,
    MetricIdentity.ADJUSTED_FCF,
})

SHARE_IDENTITIES = frozenset({
    MetricIdentity.SHARES_CURRENT_OUTSTANDING,
    MetricIdentity.SHARES_WEIGHTED_AVERAGE_BASIC,
    MetricIdentity.SHARES_WEIGHTED_AVERAGE_DILUTED,
    MetricIdentity.SHARES_ECONOMIC_CURRENT,
    MetricIdentity.SHARES_ADR_EQUIVALENT,
})

# Share bases that describe how many shares exist RIGHT NOW. A weighted
# average is not one of them: it is an average over a reporting period,
# correct for earnings per share and wrong for a market capitalisation.
CURRENT_SHARE_IDENTITIES = frozenset({
    MetricIdentity.SHARES_CURRENT_OUTSTANDING,
    MetricIdentity.SHARES_ECONOMIC_CURRENT,
    MetricIdentity.SHARES_ADR_EQUIVALENT,
})

WEIGHTED_AVERAGE_SHARE_IDENTITIES = frozenset({
    MetricIdentity.SHARES_WEIGHTED_AVERAGE_BASIC,
    MetricIdentity.SHARES_WEIGHTED_AVERAGE_DILUTED,
})

DEBT_COMPONENT_IDENTITIES = frozenset({
    MetricIdentity.SHORT_TERM_DEBT,
    MetricIdentity.LONG_TERM_DEBT,
})


# ---------------------------------------------------------------------------
# Section 6 — period frequency
# ---------------------------------------------------------------------------

class PeriodFrequency:
    INSTANT = "INSTANT"
    QUARTER = "QUARTER"
    YTD_6M = "YTD_6M"
    YTD_9M = "YTD_9M"
    HALF_YEAR = "HALF_YEAR"
    ANNUAL = "ANNUAL"
    TTM = "TTM"
    MULTI_YEAR = "MULTI_YEAR"
    UNKNOWN = "UNKNOWN"

    ALL = (INSTANT, QUARTER, YTD_6M, YTD_9M, HALF_YEAR, ANNUAL, TTM,
           MULTI_YEAR, UNKNOWN)


# How many months each frequency spans. INSTANT is a point in time and has
# no duration at all, which is why it is None rather than 0 -- a balance
# sheet figure is not "a flow of zero months", it is not a flow.
DURATION_MONTHS = {
    PeriodFrequency.INSTANT: None,
    PeriodFrequency.QUARTER: 3,
    PeriodFrequency.YTD_6M: 6,
    PeriodFrequency.HALF_YEAR: 6,
    PeriodFrequency.YTD_9M: 9,
    PeriodFrequency.ANNUAL: 12,
    PeriodFrequency.TTM: 12,
    PeriodFrequency.MULTI_YEAR: None,
    PeriodFrequency.UNKNOWN: None,
}

FLOW_FREQUENCIES = frozenset({
    PeriodFrequency.QUARTER, PeriodFrequency.YTD_6M, PeriodFrequency.YTD_9M,
    PeriodFrequency.HALF_YEAR, PeriodFrequency.ANNUAL, PeriodFrequency.TTM,
    PeriodFrequency.MULTI_YEAR,
})


def duration_months(frequency: str) -> Optional[int]:
    return DURATION_MONTHS.get(frequency)


def same_duration(left: str, right: str) -> bool:
    """Do these two frequencies cover the same length of time?

    ANNUAL and TTM both do (twelve months), which is why a full-year
    guidance figure may legitimately be compared against a trailing-twelve-
    month actual. QUARTER and TTM do not, which is the comparison that
    produced a -73% growth rate.
    """
    left_months = duration_months(left)
    right_months = duration_months(right)
    if left_months is None or right_months is None:
        return False
    return left_months == right_months


# ---------------------------------------------------------------------------
# Forecast-horizon compatibility
# ---------------------------------------------------------------------------
#
# Section 4 says compatibility is a property of a PAIR and an OPERATION. This
# is the pair that was missing: forward evidence against a FORECAST
# ASSUMPTION, where the operation is "set or check the assumption's numeric
# MAGNITUDE".
#
# A quarterly year-over-year growth rate can be entirely correct -- right
# metric, right comparison quarter, validated derivation -- and still not be
# the same KIND of quantity as an annual year-1 growth assumption. A company
# can grow 84% in a quarter against an easy comparable and 32% over twelve
# months; both are true, and only one of them is measured in the units a
# twelve-month forecast is measured in.
#
# The live failure: an annual bound of 25%, trailing twelve-month growth of
# 32%, next-quarter guidance implying 84%. The bound conflict was REAL --
# 32% > 25% -- and it was reported as though the model were 59 percentage
# points wrong, because the quarterly figure supplied the magnitude.
#
# The answer is not to discard the quarterly figure. It is the most current
# forward statement the company has made and it belongs in the research
# evidence. What changes is which OPERATIONS it is eligible for.


class ForecastCompatibility:
    """What a piece of forward evidence may do to a forecast assumption."""

    # Same metric, same horizon: may set or check a magnitude, and may
    # establish that a model bound is constraining real evidence.
    ASSUMPTION_COMPARABLE = "ASSUMPTION_COMPARABLE"
    # Same metric, SHORTER horizon: may support a statement about direction
    # or trajectory ("near-term growth remains elevated") and may never
    # supply a number the assumption is set to or measured against.
    DIRECTIONAL_CORROBORATION = "DIRECTIONAL_CORROBORATION"
    # Different metric, or a horizon that cannot be placed at all.
    NOT_COMPARABLE = "NOT_COMPARABLE"

    ALL = (ASSUMPTION_COMPARABLE, DIRECTIONAL_CORROBORATION, NOT_COMPARABLE)


FORECAST_HORIZON_NOT_COMPARABLE = "FORECAST_HORIZON_NOT_COMPARABLE"

# Horizons that may CORROBORATE a twelve-month assumption without supplying
# its magnitude. Each is a real, shorter window over the same metric: it says
# something about the trajectory and nothing about the annual rate.
_SHORTER_FLOW_HORIZONS = frozenset({
    PeriodFrequency.QUARTER, PeriodFrequency.YTD_6M,
    PeriodFrequency.HALF_YEAR, PeriodFrequency.YTD_9M,
})


@dataclass(frozen=True)
class ForecastEligibility:
    """The verdict, as structured metadata rather than prose.

    A consumer reads `status` and the two `may_*` questions. Nothing
    downstream should be parsing `reason` to decide what it is allowed to do
    -- the reason exists to explain the verdict to a person.
    """

    status: str
    reason: str = ""
    compatible_assumption_metric: Optional[str] = None
    compatible_horizon: Optional[str] = None
    code: Optional[str] = None

    @property
    def may_set_magnitude(self) -> bool:
        """May this evidence set, or be numerically compared against, the
        assumption's value?"""
        return self.status == ForecastCompatibility.ASSUMPTION_COMPARABLE

    @property
    def may_corroborate_direction(self) -> bool:
        """May this evidence support a claim about trajectory?"""
        return self.status in (ForecastCompatibility.ASSUMPTION_COMPARABLE,
                               ForecastCompatibility.DIRECTIONAL_CORROBORATION)

    def to_dict(self) -> dict:
        return {
            "forecast_compatibility": self.status,
            "reason": self.reason,
            "compatible_assumption_metric": self.compatible_assumption_metric,
            "compatible_horizon": self.compatible_horizon,
            "code": self.code,
        }


def forecast_compatibility(*, evidence_metric: Optional[str],
                           evidence_frequency: Optional[str],
                           assumption_metric: Optional[str],
                           assumption_frequency: str = PeriodFrequency.ANNUAL
                           ) -> ForecastEligibility:
    """May this evidence set an assumption's magnitude, or only its direction?

    Two questions, in order, and BOTH must pass for a magnitude:

      1. is the metric identity compatible?
      2. is the forecast horizon compatible?

    Metric first, because a twelve-month EBITDA growth rate shares a horizon
    with a revenue-growth assumption and is still not one. An unrecognised
    metric or an unplaceable horizon is refused rather than permitted --
    section 4's rule that a validator allowing what it does not understand
    guarantees nothing.
    """
    if not evidence_metric or not assumption_metric or evidence_metric != assumption_metric:
        return ForecastEligibility(
            status=ForecastCompatibility.NOT_COMPARABLE,
            code=INCOMPATIBLE_METRICS,
            reason=(f"{evidence_metric or 'an unidentified metric'} is not "
                    f"{assumption_metric or 'the assumption metric'}, so it cannot inform "
                    "that assumption at any horizon."),
            compatible_assumption_metric=assumption_metric)

    if same_duration(evidence_frequency, assumption_frequency):
        return ForecastEligibility(
            status=ForecastCompatibility.ASSUMPTION_COMPARABLE,
            reason=(f"{evidence_frequency} and {assumption_frequency} both span "
                    f"{duration_months(assumption_frequency)} months, so the two figures "
                    "measure the same kind of quantity."),
            compatible_assumption_metric=assumption_metric,
            compatible_horizon=evidence_frequency)

    if evidence_frequency in _SHORTER_FLOW_HORIZONS             and duration_months(assumption_frequency) is not None:
        return ForecastEligibility(
            status=ForecastCompatibility.DIRECTIONAL_CORROBORATION,
            code=FORECAST_HORIZON_NOT_COMPARABLE,
            reason=(f"A {evidence_frequency.lower()} figure covers "
                    f"{duration_months(evidence_frequency)} months against the assumption's "
                    f"{duration_months(assumption_frequency)}. It is evidence about the "
                    "near-term trajectory and is not a rate the annual assumption can be "
                    "set to or measured against."),
            compatible_assumption_metric=assumption_metric,
            compatible_horizon=evidence_frequency)

    return ForecastEligibility(
        status=ForecastCompatibility.NOT_COMPARABLE,
        code=PERIOD_FREQUENCY_MISMATCH,
        reason=(f"A {evidence_frequency or 'horizonless'} figure cannot be placed against a "
                f"{assumption_frequency} assumption."),
        compatible_assumption_metric=assumption_metric,
        compatible_horizon=evidence_frequency)


# ---------------------------------------------------------------------------
# Sections 2 and 12/14 — basis vocabularies
# ---------------------------------------------------------------------------

class AccountingBasis:
    GAAP = "GAAP"
    ADJUSTED = "ADJUSTED"
    IFRS = "IFRS"
    UNKNOWN = "UNKNOWN"


class ConsolidationScope:
    CONSOLIDATED = "CONSOLIDATED"
    SEGMENT = "SEGMENT"
    COMPONENT = "COMPONENT"
    PARENT_ONLY = "PARENT_ONLY"
    UNKNOWN = "UNKNOWN"


class FlowOrInstant:
    FLOW = "FLOW"
    INSTANT = "INSTANT"
    UNKNOWN = "UNKNOWN"


class CurrentOrHistorical:
    CURRENT = "CURRENT"
    HISTORICAL = "HISTORICAL"
    FORWARD = "FORWARD"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Section 5 — operations
# ---------------------------------------------------------------------------
#
# Compatibility is a property of a PAIR AND AN OPERATION, never of a pair
# alone. Quarterly guidance and trailing-twelve-month revenue are both
# perfectly good evidence and may appear side by side in the same report
# (COMPARE); dividing one by the other to get a growth rate (GROWTH) is
# meaningless. One relation, two answers, and any validator that returns a
# single verdict for "are these compatible" gets one of them wrong.

class Operation:
    COMPARE = "COMPARE"
    SUM = "SUM"
    SUBTRACT = "SUBTRACT"
    RATIO = "RATIO"
    GROWTH = "GROWTH"
    RECONCILE = "RECONCILE"
    ROLL_FORWARD = "ROLL_FORWARD"
    DCF_INPUT = "DCF_INPUT"
    PER_SHARE_CONVERSION = "PER_SHARE_CONVERSION"

    ALL = (COMPARE, SUM, SUBTRACT, RATIO, GROWTH, RECONCILE, ROLL_FORWARD,
           DCF_INPUT, PER_SHARE_CONVERSION)


# ---------------------------------------------------------------------------
# Section 48 — rejection codes
# ---------------------------------------------------------------------------

INCOMPATIBLE_PERIODS = "INCOMPATIBLE_PERIODS"
PERIOD_FREQUENCY_MISMATCH = "PERIOD_FREQUENCY_MISMATCH"
INCOMPATIBLE_METRICS = "INCOMPATIBLE_METRICS"
INCOMPATIBLE_SHARE_BASIS = "INCOMPATIBLE_SHARE_BASIS"
INCOMPATIBLE_DEBT_BASIS = "INCOMPATIBLE_DEBT_BASIS"
INCOMPATIBLE_ENTITY_SCOPE = "INCOMPATIBLE_ENTITY_SCOPE"
INCOMPATIBLE_CURRENCY = "INCOMPATIBLE_CURRENCY"
INCOMPATIBLE_ACCOUNTING_BASIS = "INCOMPATIBLE_ACCOUNTING_BASIS"
INCOMPATIBLE_FLOW_INSTANT = "INCOMPATIBLE_FLOW_INSTANT"
INCOMPATIBLE_FCF_DEFINITION = "INCOMPATIBLE_FCF_DEFINITION"

# Not a failure: a legitimate statement that two figures are different
# things and were never meant to agree. Section 13 -- a current share count
# and a weighted-average one differ BY CONSTRUCTION, and reporting that as a
# data-integrity problem sends a reader looking for a bug that is not there.
SHARE_BASIS_NOT_COMPARABLE = "SHARE_BASIS_NOT_COMPARABLE"


@dataclass(frozen=True)
class Compatibility:
    """The answer to one question about one pair of facts.

    `informational` marks the case that is neither allowed nor a fault:
    the operation does not apply, and nobody did anything wrong. It exists
    so callers can tell "refuse and warn" from "refuse and say nothing",
    which is the whole of section 13.
    """

    allowed: bool
    operation: str
    code: Optional[str] = None
    reason: str = ""
    informational: bool = False

    def __bool__(self) -> bool:
        return self.allowed


def _ok(operation: str, reason: str = "") -> Compatibility:
    return Compatibility(True, operation, None, reason)


def _no(operation: str, code: str, reason: str, informational: bool = False) -> Compatibility:
    return Compatibility(False, operation, code, reason, informational)


# ---------------------------------------------------------------------------
# Section 2 — the fact itself
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticFact:
    """A financial number that remembers what it is.

    Frozen because a fact's identity must not drift between the layer that
    established it and the layer that uses it -- the whole failure mode this
    module exists to prevent is metadata being lost or quietly replaced
    somewhere between normalization and arithmetic.

    Every field is optional except the identity and the value, so an
    existing call site can construct one from what it already knows and get
    stricter as more metadata becomes available. An UNKNOWN is never treated
    as a match: `compatible_for` refuses rather than assumes.
    """

    metric_id: str
    value: Optional[float] = None
    units: str = "currency"
    currency: Optional[str] = None

    entity_id: Optional[str] = None
    security_id: Optional[str] = None
    consolidation_scope: str = ConsolidationScope.CONSOLIDATED

    accounting_basis: str = AccountingBasis.UNKNOWN

    period_frequency: str = PeriodFrequency.UNKNOWN
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    instant_date: Optional[str] = None

    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None

    flow_or_instant: str = FlowOrInstant.UNKNOWN
    current_or_historical: str = CurrentOrHistorical.UNKNOWN

    source_provider: Optional[str] = None
    filing_type: Optional[str] = None
    accession: Optional[str] = None
    evidence_id: Optional[str] = None

    definition_id: Optional[str] = None
    derivation_method: Optional[str] = None

    validation_status: str = "UNVALIDATED"

    def label(self) -> str:
        """A short identity string for a rejection message and the debug log.

        Section 49 wants the two semantic identities recorded whenever an
        operation is refused; this is that identity, and it deliberately
        carries no value -- the reason an operation was rejected never
        depends on the magnitudes.
        """
        parts = [self.metric_id, self.period_frequency]
        if self.fiscal_year:
            period = f"FY{self.fiscal_year}"
            if self.fiscal_quarter:
                period = f"Q{self.fiscal_quarter} {period}"
            parts.append(period)
        elif self.end_date:
            parts.append(self.end_date)
        if self.accounting_basis != AccountingBasis.UNKNOWN:
            parts.append(self.accounting_basis)
        if self.consolidation_scope != ConsolidationScope.CONSOLIDATED:
            parts.append(self.consolidation_scope)
        return "/".join(str(p) for p in parts if p)

    def duration_months(self) -> Optional[int]:
        return duration_months(self.period_frequency)


# ---------------------------------------------------------------------------
# Section 4 — the validator
# ---------------------------------------------------------------------------

def _shared_context(operation: str, left: SemanticFact,
                    right: SemanticFact) -> Optional[Compatibility]:
    """Checks that apply to every operation on two facts.

    Currency, entity and consolidation scope come first because they
    invalidate an operation regardless of what it is: two figures in
    different currencies, or covering different entities, are not two
    measurements of one thing under any arithmetic.
    """
    if (left.currency and right.currency
            and left.currency != right.currency):
        return _no(operation, INCOMPATIBLE_CURRENCY,
                   f"{left.label()} is reported in {left.currency} and {right.label()} in "
                   f"{right.currency}; the two are not on one measurement scale.")

    if (left.entity_id and right.entity_id and left.entity_id != right.entity_id):
        return _no(operation, INCOMPATIBLE_ENTITY_SCOPE,
                   f"{left.label()} and {right.label()} describe different entities.")

    known = (ConsolidationScope.CONSOLIDATED, ConsolidationScope.SEGMENT,
             ConsolidationScope.COMPONENT, ConsolidationScope.PARENT_ONLY)
    if (left.consolidation_scope in known and right.consolidation_scope in known
            and left.consolidation_scope != right.consolidation_scope):
        return _no(operation, INCOMPATIBLE_ENTITY_SCOPE,
                   f"{left.label()} covers {left.consolidation_scope} and {right.label()} "
                   f"covers {right.consolidation_scope}; a part and a whole are not "
                   "interchangeable.")
    return None


def _check_growth(left: SemanticFact, right: SemanticFact) -> Compatibility:
    """GROWTH: (left / right) - 1.

    The single rule: a growth rate is only a growth rate when both sides
    cover the same LENGTH of time. Three months of revenue over twelve
    months of revenue is a ratio of unlike things whose value happens to
    look like a percentage -- which is exactly how a routine quarterly
    guidance figure became a -73% annual decline, and then a clamp, and then
    a model-bound warning, and then a valuation the report had to explain
    away.
    """
    operation = Operation.GROWTH

    if left.metric_id != right.metric_id:
        return _no(operation, INCOMPATIBLE_METRICS,
                   f"a growth rate needs the same metric on both sides; {left.label()} "
                   f"against {right.label()} compares two different quantities.")

    if (left.flow_or_instant == FlowOrInstant.INSTANT
            or right.flow_or_instant == FlowOrInstant.INSTANT
            or left.period_frequency == PeriodFrequency.INSTANT
            or right.period_frequency == PeriodFrequency.INSTANT):
        return _no(operation, INCOMPATIBLE_FLOW_INSTANT,
                   f"{left.label()} or {right.label()} is a balance at a point in time; "
                   "growth over a period is not defined for it here.")

    if (left.period_frequency == PeriodFrequency.UNKNOWN
            or right.period_frequency == PeriodFrequency.UNKNOWN):
        return _no(operation, PERIOD_FREQUENCY_MISMATCH,
                   f"the period covered by {left.label()} or {right.label()} is not known, "
                   "so the two cannot be shown to span the same length of time.")

    if not same_duration(left.period_frequency, right.period_frequency):
        return _no(operation, PERIOD_FREQUENCY_MISMATCH,
                   f"{left.label()} covers {duration_months(left.period_frequency)} months "
                   f"and {right.label()} covers {duration_months(right.period_frequency)}; "
                   "dividing one by the other does not produce a growth rate.")

    # Same length, but the wrong twelve months: a fiscal-year figure against
    # the SAME fiscal year is not growth, it is 0% by construction.
    if (left.fiscal_year is not None and right.fiscal_year is not None
            and left.fiscal_year == right.fiscal_year
            and left.fiscal_quarter == right.fiscal_quarter):
        return _no(operation, INCOMPATIBLE_PERIODS,
                   f"{left.label()} and {right.label()} are the same period; growth against "
                   "itself is not a rate.")

    # A quarter may only be compared with the SAME quarter of another year.
    # Q3 against Q2 is a sequential change, which for any seasonal business
    # is not a growth rate and for every business is not the one the DCF
    # wants.
    if (left.period_frequency == PeriodFrequency.QUARTER
            and right.period_frequency == PeriodFrequency.QUARTER
            and left.fiscal_quarter is not None and right.fiscal_quarter is not None
            and left.fiscal_quarter != right.fiscal_quarter):
        return _no(operation, INCOMPATIBLE_PERIODS,
                   f"{left.label()} and {right.label()} are different quarters of the year; "
                   "a sequential quarter-on-quarter change is not a year-over-year growth "
                   "rate.")

    if (left.accounting_basis != right.accounting_basis
            and AccountingBasis.UNKNOWN not in (left.accounting_basis,
                                                right.accounting_basis)):
        return _no(operation, INCOMPATIBLE_ACCOUNTING_BASIS,
                   f"{left.label()} is stated on a {left.accounting_basis} basis and "
                   f"{right.label()} on a {right.accounting_basis} basis; the difference "
                   "between them is partly a difference of definition.")

    return _ok(operation, f"{left.label()} and {right.label()} both cover "
                          f"{duration_months(left.period_frequency)} months of the same "
                          "metric on a comparable basis.")


def _check_reconcile(left: SemanticFact, right: SemanticFact) -> Compatibility:
    """RECONCILE: do these two independent measurements of ONE quantity agree?

    Section 15's requirement, and the reason two live warnings were noise:
    a reconciliation is only meaningful when both sides are supposed to be
    the same number. Before any tolerance is computed, the two sides must be
    the same quantity -- otherwise the tolerance is measuring the distance
    between a thing and a different thing, and will always eventually
    exceed itself.
    """
    operation = Operation.RECONCILE

    # -- share bases (sections 12-13) --------------------------------------
    if left.metric_id in SHARE_IDENTITIES and right.metric_id in SHARE_IDENTITIES:
        left_current = left.metric_id in CURRENT_SHARE_IDENTITIES
        right_current = right.metric_id in CURRENT_SHARE_IDENTITIES
        if left_current != right_current:
            return _no(operation, SHARE_BASIS_NOT_COMPARABLE,
                       f"{left.label()} and {right.label()} are different share bases: one "
                       "counts the shares in issue at a moment, the other averages the "
                       "shares outstanding across a reporting period. They are expected to "
                       "differ, and the difference is not a data-quality problem.",
                       informational=True)
        if (left.metric_id in WEIGHTED_AVERAGE_SHARE_IDENTITIES
                and right.metric_id in WEIGHTED_AVERAGE_SHARE_IDENTITIES
                and left.metric_id != right.metric_id):
            return _no(operation, SHARE_BASIS_NOT_COMPARABLE,
                       f"{left.label()} is a basic weighted average and {right.label()} a "
                       "diluted one; the gap between them is the dilution, not an error.",
                       informational=True)
        if (left.metric_id in WEIGHTED_AVERAGE_SHARE_IDENTITIES
                and left.metric_id == right.metric_id
                and left.period_frequency != right.period_frequency):
            return _no(operation, INCOMPATIBLE_PERIODS,
                       f"{left.label()} and {right.label()} average over different periods.",
                       informational=True)
        return _ok(operation, f"{left.label()} and {right.label()} are the same share basis.")

    # -- debt (section 14) --------------------------------------------------
    debt_identities = DEBT_COMPONENT_IDENTITIES | {MetricIdentity.TOTAL_DEBT}
    if left.metric_id in debt_identities and right.metric_id in debt_identities             and left.metric_id != right.metric_id:
        # A sum of debt components IS total debt, and reconciles against a
        # reported total debt -- same identity on both sides, two independent
        # derivations of it. What it may never be checked against is one of
        # its own parts: comparing short-term plus long-term debt with the
        # long-term-debt tag measures the short-term debt and reports it as a
        # discrepancy. Two DIFFERENT parts do not reconcile with each other
        # either; neither is a measurement of the other.
        return _no(operation, INCOMPATIBLE_DEBT_BASIS,
                   f"{left.label()} and {right.label()} are different debt measures. A sum "
                   "of debt components reconciles against a reported TOTAL debt, never "
                   "against one of its own components and never against a different "
                   "component.")

    # -- free cash flow definitions (section 30 of the prior phase) --------
    if left.metric_id in FCF_IDENTITIES and right.metric_id in FCF_IDENTITIES:
        if left.metric_id != right.metric_id:
            return _no(operation, INCOMPATIBLE_FCF_DEFINITION,
                       f"{left.label()} and {right.label()} are different definitions of free "
                       "cash flow; they are not two measurements of one quantity.")
        if (left.definition_id and right.definition_id
                and left.definition_id != right.definition_id):
            return _no(operation, INCOMPATIBLE_FCF_DEFINITION,
                       f"{left.label()} and {right.label()} share a name but were computed "
                       "under different definitions.")

    if left.metric_id != right.metric_id:
        return _no(operation, INCOMPATIBLE_METRICS,
                   f"{left.label()} and {right.label()} are different metrics and are not "
                   "expected to agree.")

    # Same metric, but a reconciliation still needs the same moment or span.
    if left.flow_or_instant != right.flow_or_instant \
            and FlowOrInstant.UNKNOWN not in (left.flow_or_instant, right.flow_or_instant):
        return _no(operation, INCOMPATIBLE_FLOW_INSTANT,
                   f"{left.label()} is a {left.flow_or_instant.lower()} and {right.label()} "
                   f"a {right.flow_or_instant.lower()}.")

    if (left.period_frequency != right.period_frequency
            and not same_duration(left.period_frequency, right.period_frequency)):
        return _no(operation, INCOMPATIBLE_PERIODS,
                   f"{left.label()} and {right.label()} cover different periods, so a "
                   "difference between them is not a disagreement.")

    if (left.instant_date and right.instant_date
            and left.instant_date != right.instant_date):
        return _no(operation, INCOMPATIBLE_PERIODS,
                   f"{left.label()} is measured at {left.instant_date} and {right.label()} "
                   f"at {right.instant_date}.")

    if (left.accounting_basis != right.accounting_basis
            and AccountingBasis.UNKNOWN not in (left.accounting_basis,
                                                right.accounting_basis)):
        return _no(operation, INCOMPATIBLE_ACCOUNTING_BASIS,
                   f"{left.label()} and {right.label()} are stated on different accounting "
                   "bases; a gap between them is partly definitional.")

    return _ok(operation, f"{left.label()} and {right.label()} are the same quantity measured "
                          "the same way.")


def _check_sum(left: SemanticFact, right: SemanticFact) -> Compatibility:
    """SUM: may these two be added into one figure?

    Adding requires the same units and the same period, and -- the part that
    matters for debt -- that neither operand already contains the other. A
    total is not a component of itself.
    """
    operation = Operation.SUM

    if left.units != right.units:
        return _no(operation, INCOMPATIBLE_METRICS,
                   f"{left.label()} is in {left.units} and {right.label()} in {right.units}.")

    if (left.metric_id == MetricIdentity.TOTAL_DEBT
            and right.metric_id in DEBT_COMPONENT_IDENTITIES) or \
       (right.metric_id == MetricIdentity.TOTAL_DEBT
            and left.metric_id in DEBT_COMPONENT_IDENTITIES):
        return _no(operation, INCOMPATIBLE_DEBT_BASIS,
                   f"{left.label()} and {right.label()} cannot be added: total debt already "
                   "includes the component.")

    if left.flow_or_instant != right.flow_or_instant \
            and FlowOrInstant.UNKNOWN not in (left.flow_or_instant, right.flow_or_instant):
        return _no(operation, INCOMPATIBLE_FLOW_INSTANT,
                   f"{left.label()} and {right.label()} are not both flows or both balances.")

    if left.flow_or_instant == FlowOrInstant.INSTANT:
        if (left.instant_date and right.instant_date
                and left.instant_date != right.instant_date):
            return _no(operation, INCOMPATIBLE_PERIODS,
                       f"{left.label()} and {right.label()} are measured at different dates.")
    elif (left.period_frequency != right.period_frequency
          and not same_duration(left.period_frequency, right.period_frequency)):
        return _no(operation, INCOMPATIBLE_PERIODS,
                   f"{left.label()} and {right.label()} cover different periods.")

    return _ok(operation)


def _check_ratio(left: SemanticFact, right: SemanticFact) -> Compatibility:
    """RATIO: a margin, a multiple, a coverage figure.

    Unlike GROWTH, a ratio does NOT require the same metric -- operating
    income over revenue is the point. It does require the same period, which
    is what stops last fiscal year's operating income from being divided by
    a trailing-twelve-month revenue to produce a margin belonging to neither.
    """
    operation = Operation.RATIO

    if left.flow_or_instant == FlowOrInstant.FLOW and right.flow_or_instant == FlowOrInstant.FLOW:
        if (left.period_frequency != right.period_frequency
                and not same_duration(left.period_frequency, right.period_frequency)):
            return _no(operation, PERIOD_FREQUENCY_MISMATCH,
                       f"{left.label()} and {right.label()} cover different periods; the "
                       "ratio would belong to neither.")
        if (left.end_date and right.end_date and left.end_date != right.end_date):
            return _no(operation, INCOMPATIBLE_PERIODS,
                       f"{left.label()} ends {left.end_date} and {right.label()} ends "
                       f"{right.end_date}.")

    if (left.accounting_basis != right.accounting_basis
            and AccountingBasis.UNKNOWN not in (left.accounting_basis,
                                                right.accounting_basis)):
        return _no(operation, INCOMPATIBLE_ACCOUNTING_BASIS,
                   f"{left.label()} and {right.label()} mix a {left.accounting_basis} "
                   f"numerator with a {right.accounting_basis} denominator.")

    return _ok(operation)


def _check_per_share(left: SemanticFact, right: SemanticFact) -> Compatibility:
    """PER_SHARE_CONVERSION: divide an amount by a share count.

    Which share count is correct depends on what the numerator is. Earnings
    per share uses the weighted average, because the earnings accrued across
    the same period the average covers. A value per share from a valuation
    uses the CURRENT count, because the valuation is of the company as it
    exists now. Using the other one in either place is a real error and a
    silent one.
    """
    operation = Operation.PER_SHARE_CONVERSION

    if right.metric_id not in SHARE_IDENTITIES:
        return _no(operation, INCOMPATIBLE_METRICS,
                   f"{right.label()} is not a share count.")

    if left.current_or_historical == CurrentOrHistorical.CURRENT \
            and right.metric_id in WEIGHTED_AVERAGE_SHARE_IDENTITIES:
        return _no(operation, INCOMPATIBLE_SHARE_BASIS,
                   f"{left.label()} describes the company as it is now and {right.label()} "
                   "averages the share count over a past reporting period.")

    if left.period_frequency in FLOW_FREQUENCIES \
            and left.current_or_historical == CurrentOrHistorical.HISTORICAL \
            and right.metric_id in CURRENT_SHARE_IDENTITIES:
        return _no(operation, INCOMPATIBLE_SHARE_BASIS,
                   f"{left.label()} accrued over a past period and {right.label()} counts "
                   "the shares in issue today.")

    return _ok(operation)


def _check_dcf_input(left: SemanticFact, right: SemanticFact) -> Compatibility:
    """DCF_INPUT: may these two facts jointly feed the valuation model?

    Strictest of the operations, because everything downstream of it is a
    number a reader will treat as a valuation. It requires what RECONCILE
    requires and, additionally, that the facts be current rather than
    historical -- a DCF of a company as it was three years ago is not the
    thing anyone asked for.
    """
    operation = Operation.DCF_INPUT

    base = _check_reconcile(left, right)
    if not base.allowed and not base.informational:
        return Compatibility(False, operation, base.code, base.reason)

    for fact in (left, right):
        if fact.current_or_historical == CurrentOrHistorical.HISTORICAL:
            return _no(operation, INCOMPATIBLE_PERIODS,
                       f"{fact.label()} is historical evidence and cannot serve as a current "
                       "valuation input.")
    return _ok(operation)


_CHECKS = {
    Operation.GROWTH: _check_growth,
    Operation.RECONCILE: _check_reconcile,
    Operation.SUM: _check_sum,
    Operation.SUBTRACT: _check_sum,          # same requirements as addition
    Operation.RATIO: _check_ratio,
    Operation.PER_SHARE_CONVERSION: _check_per_share,
    Operation.DCF_INPUT: _check_dcf_input,
}


def compatible_for(operation: str, left: SemanticFact,
                   right: SemanticFact) -> Compatibility:
    """THE entry point. Can `left` and `right` be combined under `operation`?

    Everything that combines two financial figures should ask this first.
    An unrecognised operation is refused rather than allowed: a validator
    that defaults to permitting what it does not understand provides no
    guarantee at all, which is the failure mode this whole phase exists to
    correct.
    """
    if operation not in Operation.ALL:
        return _no(operation, INCOMPATIBLE_METRICS,
                   f"{operation!r} is not a recognised operation.")

    shared = _shared_context(operation, left, right)
    if shared is not None:
        return shared

    check = _CHECKS.get(operation)
    if check is None:
        # COMPARE and ROLL_FORWARD place two figures side by side without
        # arithmetic. Both are allowed once the shared context holds --
        # showing a quarter's guidance next to a trailing-twelve-month
        # actual is exactly the right thing to do, as long as nothing
        # divides them.
        return _ok(operation)
    return check(left, right)


# ---------------------------------------------------------------------------
# Section 49 — the rejection log
# ---------------------------------------------------------------------------

@dataclass
class SemanticAudit:
    """Every operation the validator refused, in order.

    Kept as structured records rather than formatted strings so readiness
    and the report can rank them by root cause (sections 32-33) instead of
    printing whichever symptom happened to be noticed last. Values are never
    recorded -- only identities and the reason -- so this can be logged
    freely.
    """

    rejections: list = field(default_factory=list)

    def record(self, result: Compatibility, left: SemanticFact,
               right: SemanticFact, context: str = "") -> None:
        if result.allowed:
            return
        self.rejections.append({
            "operation": result.operation,
            "code": result.code,
            "left": left.label(),
            "right": right.label(),
            "reason": result.reason,
            "informational": result.informational,
            "context": context,
        })

    def blocking(self) -> list:
        """Rejections that represent a real incompatibility.

        An informational one (section 13's share bases) is deliberately not
        blocking: nothing is wrong, the operation simply does not apply.
        """
        return [r for r in self.rejections if not r["informational"]]

    def codes(self) -> Tuple[str, ...]:
        return tuple(r["code"] for r in self.rejections if r["code"])

    def to_dict(self) -> dict:
        return {"rejections": list(self.rejections),
                "blocking_count": len(self.blocking())}
