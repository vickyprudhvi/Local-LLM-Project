"""Phase H.4/H.6 — current management guidance, extracted from SEC-filed material.

WHY THIS IS NOT AN LLM READING A PRESS RELEASE
==============================================
Quantitative guidance never appears in XBRL company facts; it lives in the
earnings-release exhibit attached to an item-2.02 8-K. That is prose, and the
obvious approach — hand the document to the local model and ask for the
numbers — is exactly the approach section 5 forbids, for a good reason: a
model asked to find guidance in a document that contains none will supply
some, and a fabricated forward number is far more damaging than a missing
one because it flows straight into the DCF and looks identical to a real one.

So NOTHING here interprets. Every value is produced by a reviewed pattern
matched against the document text, and everything a pattern does not match is
DROPPED rather than guessed at. The document is untrusted input to a parser,
never instructions and never context for a model.

THE TWO BUGS PHASE H.6 EXISTS TO FIX
====================================
Both were found live, and they are opposite failures of the same missing
idea — that a guidance number without a METRIC IDENTITY is not guidance.

1. AT&T (mapped to the wrong metric). The July 2026 release says:

       Service revenue growth in the low-single-digit range annually
       Advanced Connectivity service revenue growth in the mid-single-digit ...
       Adjusted EBITDA* growth in the 3% to 4% range in 2026, improving to ...

   `revenue_growth`'s keyword matched "Service revenue growth", found no
   number in that bullet, and its 110-character window ran on into the NEXT
   bullet and captured "3% to 4%" — adjusted EBITDA growth. Nothing stopped
   it, because EBITDA was not in the metric table at all, so there was no
   "other metric keyword" in between to act as a boundary. AT&T's EBITDA
   guidance was stored as `revenue_growth`, on a GAAP basis, and fed straight
   into `dcf.assumption.revenue_growth`. Two errors in one value: the wrong
   metric AND the wrong basis.

   Fixed three ways, all of which are needed: EBITDA is now a first-class
   metric (so it acts as a boundary), keyword matching is LONGEST-MATCH-WINS
   (so "service revenue growth" can never be read as consolidated revenue
   growth), and a window now stops at a clause boundary rather than running
   a fixed number of characters into the next sentence.

2. NVIDIA (not extracted at all). The May 2026 release says:

       NVIDIA's outlook for the second quarter of fiscal 2027 is as follows:
       Revenue is expected to be $91.0 billion, plus or minus 2%.
       GAAP and non-GAAP gross margins are expected to be 74.9% and 75.0%,
       respectively, plus or minus 50 basis points.

   Three independent reasons nothing was found: the expected fiscal year was
   hard-coded to the CALENDAR year (2026) and NVDA's guidance names fiscal
   2027, so every candidate was rejected; guidance was assumed to be ANNUAL
   and this is a next-QUARTER outlook; and the value is a point with a
   tolerance rather than a range, which the range-only rule was built to
   require. All three are now handled, and the fiscal-period model records
   "Q2 FY2027" rather than flattening it to a year.

PRECISION IS STILL BOUGHT STRUCTURALLY
======================================
A guidance figure must still be expressed either as a RANGE ("between $3.60
and $3.75", "2% to 3%") or as a POINT WITH AN EXPLICIT TOLERANCE ("$91.0
billion, plus or minus 2%"). Companies state guidance that way and state
actuals as bare single values, so the requirement separates the two
automatically. It is why, in AOS's own release, the line

    Diluted EPS (GAAP)  $ 3.60-3.75   $ 3.85

yields the guidance range and silently ignores the $3.85 prior-year actual
sitting next to it. The cost is recall on bare single-point guidance, which
is accepted deliberately: a miss is visible (guidance reads as unavailable,
and the whole workflow is built to keep working that way — see section 20),
whereas a false positive is not.

Each extracted value keeps the exact excerpt it came from, so any number in
the report can be traced back to the sentence in the filing that supports it.
"""

import hashlib
import html as _html
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# The 8-K item that means "Results of Operations and Financial Condition" —
# the earnings release. Guidance in any other item type is not looked for.
EARNINGS_RELEASE_ITEM = "2.02"

# GAAP vs adjusted must never be mixed (section 5/6). A metric is "adjusted"
# only when the document says so within the matched window.
BASIS_GAAP = "GAAP"
BASIS_ADJUSTED = "adjusted"
BASIS_COMPANY_DEFINED = "company_defined"
BASIS_NONE = "not_specified"

_ADJUSTED_MARKERS = re.compile(r"(?i)\b(adjusted|non-?GAAP|core|underlying|comparable)\b")


class GuidanceUnit:
    RATIO = "ratio"                # growth rates, margins: stored as decimals
    CURRENCY = "currency"          # absolute amounts, in the document's scale
    CURRENCY_PER_SHARE = "currency_per_share"
    SHARES = "shares"


class GuidanceStatus:
    """Section 9. A guidance figure's standing at the time of the valuation."""

    CURRENT = "CURRENT"
    SUPERSEDED = "SUPERSEDED"
    WITHDRAWN = "WITHDRAWN"
    EXPIRED = "EXPIRED"
    ALL = (CURRENT, SUPERSEDED, WITHDRAWN, EXPIRED)


class GuidancePeriodType:
    ANNUAL = "annual"
    QUARTER = "quarter"
    MULTI_YEAR = "multi_year"
    ALL = (ANNUAL, QUARTER, MULTI_YEAR)


class GuidanceTargetType:
    """Section 11. WHICH FUTURE PERIOD the guidance is about.

    Distinct from the period the guidance was ISSUED WITH, which is the
    quarter whose results the release reports. A live analysis rendered
    "Q2 FY2026 guidance" for a company that had issued FULL-YEAR 2026
    guidance alongside its second-quarter results -- the report named the
    reporting quarter and called it the target.
    """

    NEXT_QUARTER = "NEXT_QUARTER"
    CURRENT_FISCAL_YEAR = "CURRENT_FISCAL_YEAR"
    NEXT_FISCAL_YEAR = "NEXT_FISCAL_YEAR"
    MULTI_YEAR = "MULTI_YEAR"
    OTHER = "OTHER"
    ALL = (NEXT_QUARTER, CURRENT_FISCAL_YEAR, NEXT_FISCAL_YEAR, MULTI_YEAR, OTHER)


# A release's own reporting period -- "Second-Quarter 2026 results", "Q2
# 2026". This is what the guidance was ISSUED WITH, never what it targets.
_REPORTING_PERIOD_RE = re.compile(
    r"(?i)\b(first|second|third|fourth)[\s-]quarter\s+(20\d{2})\b"
    r"|\bQ([1-4])\s+(20\d{2})\s+(?:results|earnings)\b"
    r"|\b(20\d{2})\s+(first|second|third|fourth)[\s-]quarter\b")

_QUARTER_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def detect_reporting_period(text: str) -> Optional[str]:
    """The fiscal quarter whose results this release reports (section 10)."""
    match = _REPORTING_PERIOD_RE.search(text or "")
    if not match:
        return None
    groups = match.groups()
    if groups[0] and groups[1]:
        return f"Q{_QUARTER_ORDINALS[groups[0].lower()]} FY{groups[1]}"
    if groups[2] and groups[3]:
        return f"Q{groups[2]} FY{groups[3]}"
    if groups[4] and groups[5]:
        return f"Q{_QUARTER_ORDINALS[groups[5].lower()]} FY{groups[4]}"
    return None


# ---------------------------------------------------------------------------
# Phase 12 — WHAT KIND of forward statement this is
# ---------------------------------------------------------------------------
#
# The target PERIOD is not the whole story. "We expect gross margin of 55-60%
# over the long term" and "we expect gross margin of 55-60% this year" name
# the same metric and the same range, and only one of them is a commitment
# about the current year. A framework, an aspiration and a guidance figure
# carry very different weight, and collapsing them means a multi-year
# ambition can be read as this year's outlook -- and then compared against
# this year's actuals.

class ForwardInformationKind:
    CURRENT_QUARTER_GUIDANCE = "CURRENT_QUARTER_GUIDANCE"
    CURRENT_FY_GUIDANCE = "CURRENT_FY_GUIDANCE"
    NEXT_FY_GUIDANCE = "NEXT_FY_GUIDANCE"
    MULTI_YEAR_GUIDANCE = "MULTI_YEAR_GUIDANCE"
    LONG_TERM_FRAMEWORK = "LONG_TERM_FRAMEWORK"
    MANAGEMENT_TARGET = "MANAGEMENT_TARGET"
    ASPIRATIONAL_TARGET = "ASPIRATIONAL_TARGET"
    OTHER_FORWARD_INFORMATION = "OTHER_FORWARD_INFORMATION"

    ALL = (CURRENT_QUARTER_GUIDANCE, CURRENT_FY_GUIDANCE, NEXT_FY_GUIDANCE,
           MULTI_YEAR_GUIDANCE, LONG_TERM_FRAMEWORK, MANAGEMENT_TARGET,
           ASPIRATIONAL_TARGET, OTHER_FORWARD_INFORMATION)

    # Kinds that are a commitment about a NAMED, BOUNDED period. Only these
    # may anchor a forecast for that period; the rest are context.
    PERIOD_GUIDANCE = (CURRENT_QUARTER_GUIDANCE, CURRENT_FY_GUIDANCE,
                       NEXT_FY_GUIDANCE, MULTI_YEAR_GUIDANCE)


# "over the long term", "long-range model", "through the cycle" -- an
# open-ended horizon rather than a period the company will be held to.
_LONG_TERM_FRAMEWORK_RE = re.compile(
    r"(?i)\b(?:long[\s-]term(?:\s+(?:model|framework|target|outlook|algorithm))?|"
    r"long[\s-]range|through[\s-]the[\s-]cycle|over\s+time|"
    r"multi[\s-]year\s+(?:framework|model|algorithm)|steady[\s-]state)\b")

# An explicit goal the company has set itself, which is a weaker statement
# than an outlook for a period.
_MANAGEMENT_TARGET_RE = re.compile(
    r"(?i)\b(?:our\s+(?:goal|target|ambition)|we\s+(?:aim|target|aspire)|"
    r"targeting|committed\s+to\s+(?:achieving|reaching)|"
    r"on\s+track\s+to\s+(?:achieve|reach))\b")

_ASPIRATIONAL_RE = re.compile(
    r"(?i)\b(?:aspir\w*|ambition|aim\s+to\s+eventually|"
    r"over\s+the\s+coming\s+years|in\s+the\s+years\s+ahead)\b")


def classify_forward_information(context: str, target_type: str) -> str:
    """Phase 12: what kind of forward statement this is.

    The surrounding language decides, and it OVERRIDES the period: a
    sentence naming a long-term framework is a framework even when a year
    appears nearby, because the year in that sentence is describing when the
    framework applies rather than a period being guided.

    Falls through to the period-based kind, which is what an ordinary
    guidance sentence gets.
    """
    text = context or ""
    if _LONG_TERM_FRAMEWORK_RE.search(text):
        return ForwardInformationKind.LONG_TERM_FRAMEWORK
    if _ASPIRATIONAL_RE.search(text):
        return ForwardInformationKind.ASPIRATIONAL_TARGET
    if _MANAGEMENT_TARGET_RE.search(text):
        return ForwardInformationKind.MANAGEMENT_TARGET

    return {
        GuidanceTargetType.NEXT_QUARTER: ForwardInformationKind.CURRENT_QUARTER_GUIDANCE,
        GuidanceTargetType.CURRENT_FISCAL_YEAR: ForwardInformationKind.CURRENT_FY_GUIDANCE,
        GuidanceTargetType.NEXT_FISCAL_YEAR: ForwardInformationKind.NEXT_FY_GUIDANCE,
        GuidanceTargetType.MULTI_YEAR: ForwardInformationKind.MULTI_YEAR_GUIDANCE,
    }.get(target_type, ForwardInformationKind.OTHER_FORWARD_INFORMATION)


def may_anchor_period_forecast(kind: str) -> bool:
    """Phase 12/15: may this statement set a forecast for its named period?

    A framework or an aspiration may not. Both are real information and both
    stay in the evidence -- they simply cannot become the number a valuation
    is built on for a specific year.
    """
    return kind in ForwardInformationKind.PERIOD_GUIDANCE


def classify_target_type(period: "GuidancePeriod",
                         reporting_period: Optional[str]) -> str:
    """Which kind of future period a guidance figure targets (section 11)."""
    if period is None:
        return GuidanceTargetType.OTHER
    if period.period_type == GuidancePeriodType.MULTI_YEAR:
        return GuidanceTargetType.MULTI_YEAR
    if period.period_type == GuidancePeriodType.QUARTER:
        return GuidanceTargetType.NEXT_QUARTER
    if reporting_period and reporting_period.startswith("Q"):
        try:
            reporting_year = int(reporting_period.split("FY")[-1])
        except (ValueError, IndexError):
            return GuidanceTargetType.CURRENT_FISCAL_YEAR
        if period.fiscal_year > reporting_year:
            return GuidanceTargetType.NEXT_FISCAL_YEAR
    return GuidanceTargetType.CURRENT_FISCAL_YEAR


class GuidanceBound:
    """How the company expressed the figure.

    A range and a floor are not the same claim. "Adjusted EPS of $2.25 to
    $2.35" states both ends; "Free cash flow of $18 billion+" states only a
    minimum, and reading its midpoint as a forecast would be inventing a
    number the company withheld.
    """

    RANGE = "range"
    AT_LEAST = "at_least"
    APPROXIMATELY = "approximately"
    ALL = (RANGE, AT_LEAST, APPROXIMATELY)


# ---------------------------------------------------------------------------
# Section 6 — the metric taxonomy
# ---------------------------------------------------------------------------


class GuidanceMetricName:
    """Deterministic metric categories.

    The legacy names (`revenue`, `revenue_growth`, `operating_margin`,
    `earnings_per_share`, `adjusted_earnings_per_share`,
    `capital_expenditure`, `free_cash_flow`, `operating_cash_flow`) are kept
    EXACTLY as they were and continue to mean the CONSOLIDATED, total-company
    measure. Everything else is new. Renaming the existing keys would have
    silently changed what every downstream consumer reads, which is the same
    class of failure this taxonomy exists to prevent.
    """

    # Revenue — consolidated vs its components. The distinction is the whole
    # point: AT&T guides service revenue, which is most of but not all of
    # consolidated revenue, and the two must never be treated as one number.
    CONSOLIDATED_REVENUE = "revenue"
    CONSOLIDATED_REVENUE_GROWTH = "revenue_growth"
    SERVICE_REVENUE = "service_revenue"
    SERVICE_REVENUE_GROWTH = "service_revenue_growth"
    PRODUCT_REVENUE = "product_revenue"
    PRODUCT_REVENUE_GROWTH = "product_revenue_growth"
    SEGMENT_REVENUE_GROWTH = "segment_revenue_growth"

    # Profitability
    OPERATING_INCOME = "operating_income"
    OPERATING_MARGIN = "operating_margin"
    ADJUSTED_OPERATING_MARGIN = "adjusted_operating_margin"
    GROSS_MARGIN = "gross_margin"
    ADJUSTED_GROSS_MARGIN = "adjusted_gross_margin"
    OPERATING_EXPENSES = "operating_expenses"
    ADJUSTED_OPERATING_EXPENSES = "adjusted_operating_expenses"

    # EBITDA. Separate from every revenue measure, separate from operating
    # income, and GAAP separate from adjusted.
    EBITDA = "ebitda"
    ADJUSTED_EBITDA = "adjusted_ebitda"
    # Phase H.14. Each of these was ABSENT, and an absent identity is not a
    # neutral gap: the nearest GENERAL pattern claims the text instead.
    # "subscription revenue growth of 11% to 12%" was extracted as
    # CONSOLIDATED revenue growth -- a component of revenue anchoring the
    # whole company's growth assumption -- because no subscription identity
    # existed for it to match. "free cash flow growth", "operating cash flow
    # growth" and "EPS growth" had the opposite failure and matched nothing
    # at all, so real guidance was silently dropped and the report said none
    # was available.
    SUBSCRIPTION_REVENUE = "subscription_revenue"
    SUBSCRIPTION_REVENUE_GROWTH = "subscription_revenue_growth"
    FREE_CASH_FLOW_GROWTH = "free_cash_flow_growth"
    OPERATING_CASH_FLOW_GROWTH = "operating_cash_flow_growth"
    EPS_GROWTH = "earnings_per_share_growth"
    NET_INCOME_GROWTH = "net_income_growth"

    EBITDA_GROWTH = "ebitda_growth"
    ADJUSTED_EBITDA_GROWTH = "adjusted_ebitda_growth"
    # An EBITDA MARGIN is a third quantity, and it was missing. A release
    # guiding "Adjusted EBITDA of 28% to 30% of projected revenue" matched
    # the absolute ADJUSTED_EBITDA pattern and stored 0.28 under a currency
    # unit -- the same "an absent identity is claimed by the nearest general
    # pattern" failure that made subscription revenue growth into
    # consolidated revenue growth.
    EBITDA_MARGIN = "ebitda_margin"
    ADJUSTED_EBITDA_MARGIN = "adjusted_ebitda_margin"

    # Per share
    EPS = "earnings_per_share"
    ADJUSTED_EPS = "adjusted_earnings_per_share"

    # Cash and capital
    CAPEX = "capital_expenditure"
    OPERATING_CASH_FLOW = "operating_cash_flow"
    FREE_CASH_FLOW = "free_cash_flow"
    SHARE_REPURCHASES = "share_repurchases"
    NET_LEVERAGE_TARGET = "net_leverage_target"
    TAX_RATE = "tax_rate"
    # Cash items a release guides ALONGSIDE free cash flow, and which the
    # free-cash-flow keyword's window would otherwise reach into. AT&T's own
    # sentence is the case: "...free cash flow* outlook anticipates annual
    # cash taxes of $1.0 billion to $1.5 billion and cash contributions to
    # its employee benefit plans...". Both are real guided quantities and
    # both are boundaries — the same mechanism that stops EBITDA growth from
    # being read as revenue growth.
    SHARE_COUNT = "share_count"
    OTHER_INCOME_EXPENSE = "other_income_expense"
    CASH_TAXES = "cash_taxes"
    PENSION_CONTRIBUTIONS = "pension_contributions"


# Metrics that describe TOTAL-COMPANY revenue. Only these may anchor a DCF
# revenue-growth assumption directly (section 7 / section 12 precedence 1).
CONSOLIDATED_REVENUE_METRICS = frozenset({
    GuidanceMetricName.CONSOLIDATED_REVENUE,
    GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH,
})

# Metrics that describe PART of revenue. Usable as supporting forward
# evidence, never as consolidated revenue guidance (section 7: "preserve that
# distinction ... do not pretend it is exact consolidated revenue guidance").
REVENUE_COMPONENT_METRICS = frozenset({
    GuidanceMetricName.SERVICE_REVENUE,
    GuidanceMetricName.SERVICE_REVENUE_GROWTH,
    GuidanceMetricName.PRODUCT_REVENUE,
    GuidanceMetricName.PRODUCT_REVENUE_GROWTH,
    GuidanceMetricName.SEGMENT_REVENUE_GROWTH,
})

# Metrics that are NOT revenue in any form. Named explicitly so the rule
# "never map EBITDA growth to revenue growth" is a table lookup rather than a
# reviewer's memory.
NON_REVENUE_METRICS = frozenset({
    GuidanceMetricName.OPERATING_INCOME, GuidanceMetricName.OPERATING_MARGIN,
    GuidanceMetricName.ADJUSTED_OPERATING_MARGIN, GuidanceMetricName.GROSS_MARGIN,
    GuidanceMetricName.ADJUSTED_GROSS_MARGIN, GuidanceMetricName.OPERATING_EXPENSES,
    GuidanceMetricName.ADJUSTED_OPERATING_EXPENSES,
    GuidanceMetricName.EBITDA, GuidanceMetricName.ADJUSTED_EBITDA,
    GuidanceMetricName.EBITDA_GROWTH, GuidanceMetricName.ADJUSTED_EBITDA_GROWTH,
    GuidanceMetricName.EBITDA_MARGIN, GuidanceMetricName.ADJUSTED_EBITDA_MARGIN,
    GuidanceMetricName.EPS, GuidanceMetricName.ADJUSTED_EPS,
    GuidanceMetricName.CAPEX, GuidanceMetricName.OPERATING_CASH_FLOW,
    GuidanceMetricName.FREE_CASH_FLOW, GuidanceMetricName.SHARE_REPURCHASES,
    GuidanceMetricName.NET_LEVERAGE_TARGET, GuidanceMetricName.TAX_RATE,
    GuidanceMetricName.CASH_TAXES, GuidanceMetricName.PENSION_CONTRIBUTIONS,
    GuidanceMetricName.SHARE_COUNT, GuidanceMetricName.OTHER_INCOME_EXPENSE,
})


def is_consolidated_revenue_metric(name: str) -> bool:
    return name in CONSOLIDATED_REVENUE_METRICS


def is_revenue_component_metric(name: str) -> bool:
    return name in REVENUE_COMPONENT_METRICS


GUIDANCE_METRIC_MISMATCH = "GUIDANCE_METRIC_MISMATCH"

# Phase 2. Metrics that are a RATE OF CHANGE of something other than revenue.
# Each is real guidance and each is worth recording; none of them describes
# how fast the top line grows, and a valuation that treats one as if it did
# is not approximately right, it is measuring a different quantity.
#
# Free cash flow can grow 25% on flat revenue through working capital alone.
# Earnings per share can grow on a buyback with no revenue change at all.
FORBIDDEN_REVENUE_GROWTH_SOURCES = frozenset({
    GuidanceMetricName.FREE_CASH_FLOW_GROWTH,
    GuidanceMetricName.OPERATING_CASH_FLOW_GROWTH,
    GuidanceMetricName.EPS_GROWTH,
    GuidanceMetricName.NET_INCOME_GROWTH,
    GuidanceMetricName.EBITDA_GROWTH,
    GuidanceMetricName.ADJUSTED_EBITDA_GROWTH,
    GuidanceMetricName.GROSS_MARGIN,
    GuidanceMetricName.ADJUSTED_GROSS_MARGIN,
    GuidanceMetricName.OPERATING_MARGIN,
    GuidanceMetricName.ADJUSTED_OPERATING_MARGIN,
    GuidanceMetricName.EBITDA_MARGIN,
    GuidanceMetricName.ADJUSTED_EBITDA_MARGIN,
    # Components of revenue. Real, and not the consolidated total.
    GuidanceMetricName.SUBSCRIPTION_REVENUE_GROWTH,
    GuidanceMetricName.SERVICE_REVENUE_GROWTH,
    GuidanceMetricName.PRODUCT_REVENUE_GROWTH,
    GuidanceMetricName.SEGMENT_REVENUE_GROWTH,
})

# The only two provenances a revenue-growth assumption may carry.
ALLOWED_REVENUE_GROWTH_SOURCES = frozenset({
    GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH,   # stated directly
    GuidanceMetricName.CONSOLIDATED_REVENUE,          # derived, period-checked
})


def validate_revenue_growth_source(source_metric):
    """Phase 2/3: may this guidance metric support a revenue-growth assumption?

    Returns (ok, reason). Defence in depth rather than a second opinion: the
    extractor's own vocabulary is what normally keeps these apart, and this
    exists because a GAP in that vocabulary is not a neutral absence -- the
    nearest general pattern claims the text instead. "Subscription revenue
    growth" was read as consolidated revenue growth for exactly that reason,
    and anchored a valuation.

    Fails closed on an unrecognised metric. A source this function has never
    heard of has not been shown to be revenue growth, and assuming it is
    would reproduce the failure this guards.
    """
    if not source_metric:
        return False, "no source metric was recorded for this assumption"
    if source_metric in FORBIDDEN_REVENUE_GROWTH_SOURCES:
        return False, (
            f"{source_metric!r} is not consolidated revenue growth. It is real guidance and "
            "is kept as evidence, but it measures a different quantity -- a rate of change "
            "of cash flow, earnings or a part of revenue -- and cannot set how fast the "
            "top line grows.")
    if source_metric in ALLOWED_REVENUE_GROWTH_SOURCES:
        return True, ""
    return False, (
        f"{source_metric!r} has not been established as consolidated revenue growth, so it "
        "may not anchor a revenue-growth assumption.")


def may_anchor_revenue_growth(name: str) -> bool:
    """Section 7: only consolidated revenue guidance may populate
    `dcf.assumption.revenue_growth.*` directly. Everything else — EBITDA
    growth above all — is barred here rather than in each consumer."""
    return name == GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH


# ---------------------------------------------------------------------------
# The structured guidance record (section 5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuidanceMetric:
    """One guidance figure, with everything needed to cite and validate it."""

    name: str
    low: float
    high: float
    unit: str
    basis: str
    fiscal_year: int
    evidence_id: str
    source_excerpt: str
    scale: Optional[str] = None      # "millions"/"billions" when stated
    # Phase H.6 additions (section 5).
    guidance_id: str = ""
    issued_at: Optional[str] = None
    fiscal_period: Optional[str] = None       # the TARGET period: "FY2026", "Q2 FY2027"
    period_type: str = GuidancePeriodType.ANNUAL
    # Section 10. The reporting period this guidance was PUBLISHED ALONGSIDE,
    # which is not what it applies to. A company routinely issues full-year
    # guidance with its second-quarter results; naming the quarter as the
    # target overstates how near-term the outlook is.
    issued_with_reporting_period: Optional[str] = None
    target_period_type: str = GuidanceTargetType.OTHER
    # Phase 12: what KIND of forward statement this is. A long-term framework
    # and this year's outlook can name the same metric and the same range;
    # only one of them may set a forecast for a named period.
    forward_kind: str = ForwardInformationKind.OTHER_FORWARD_INFORMATION
    scope: str = "consolidated"               # consolidated | service | product | segment
    source_accession: Optional[str] = None
    source_evidence_ids: Tuple[str, ...] = ()
    # How the company expressed the figure. AT&T states most of its plan as
    # floors ("$18 billion+"); a floor stored as low==high is a MINIMUM, not
    # a midpoint forecast, and nothing may read it as one.
    bound_type: str = GuidanceBound.RANGE
    status: str = GuidanceStatus.CURRENT
    status_reason: Optional[str] = None
    # WHY this figure qualified as prospective: the text that established it
    # as forward-looking. Recorded rather than asserted, because "there was a
    # forward-looking word nearby" is exactly the reasoning that turned a
    # historical share-count table into guided share count -- and a boolean
    # would have recorded that verdict just as confidently. A reader who
    # disagrees can read the sentence the verdict was made from.
    prospective_evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "low": self.low, "high": self.high, "unit": self.unit,
            "basis": self.basis, "fiscal_year": self.fiscal_year,
            "evidence_id": self.evidence_id, "source_excerpt": self.source_excerpt,
            "scale": self.scale,
            "guidance_id": self.guidance_id,
            "issued_at": self.issued_at,
            "fiscal_period": self.fiscal_period,
            "target_period": self.fiscal_period,
            "period_type": self.period_type,
            "issued_with_reporting_period": self.issued_with_reporting_period,
            "target_period_type": self.target_period_type,
            "forward_kind": self.forward_kind,
            "may_anchor_forecast": may_anchor_period_forecast(self.forward_kind),
            "midpoint": self.midpoint,
            "units": self.unit,
            "scope": self.scope,
            "source_accession": self.source_accession,
            "source_evidence_ids": list(self.source_evidence_ids),
            "bound_type": self.bound_type,
            "status": self.status,
            "status_reason": self.status_reason,
            "prospective_evidence": self.prospective_evidence,
        }

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0

    def with_status(self, status: str, reason: Optional[str] = None) -> "GuidanceMetric":
        return GuidanceMetric(
            **{**self.__dict__, "status": status, "status_reason": reason})


@dataclass(frozen=True)
class GuidanceRelease:
    """Guidance as stated by ONE filing."""

    symbol: str
    fiscal_year: Optional[int]
    accession: str
    document: str
    filed: str
    # The NAME-KEYED view, kept because every existing consumer reads it. It
    # holds ONE statement per metric, chosen by `preferred_for_assumption`.
    metrics: Dict[str, GuidanceMetric] = field(default_factory=dict)
    # EVERY current statement, keyed in practice by `guidance_identity`.
    #
    # Section 11 says guidance identity is (metric, target period, target
    # period type, basis). `guidance_identity` computed that and
    # `resolve_guidance_status` used it -- but the CONTAINER above is keyed by
    # name alone, so a release guiding both a quarter and a full year could
    # physically hold only one of them, and the extractor dropped the second
    # before the resolver ever saw it. A company guiding 27-29% for next
    # quarter and $118-120B for the year had its annual outlook disappear,
    # and with it the annual growth that outlook implied and the model-bound
    # assessment that growth would have triggered.
    all_metrics: Tuple[GuidanceMetric, ...] = ()
    warnings: Tuple[str, ...] = ()

    def statements_for(self, name: str) -> Tuple[GuidanceMetric, ...]:
        """Every current statement of one metric, across horizons."""
        return tuple(m for m in self.all_metrics if m.name == name)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "fiscal_year": self.fiscal_year,
            "accession": self.accession,
            "document": self.document,
            "filed": self.filed,
            "source_document": f"SEC 8-K exhibit {self.document} (accession {self.accession}, "
                               f"filed {self.filed})",
            "guidance_date": self.filed,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            # Every current statement, so a consumer that needs a horizon the
            # name-keyed view had to drop can find it.
            "all_metrics": [m.to_dict() for m in self.all_metrics],
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Document handling
# ---------------------------------------------------------------------------


def html_to_text(document: str) -> str:
    """Filed HTML -> flat text. No interpretation, just tag removal.

    Block-level tags become a full stop as well as a space (Phase H.6): an
    outlook table or bullet list flattens to one long line otherwise, and a
    clause boundary is what stops one bullet's metric name from claiming the
    next bullet's number — the AT&T failure exactly.
    """
    if not document:
        return ""
    text = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", document)
    # Table cell boundaries carry meaning in an outlook table; keep them as
    # separators so "$3,900 $3,950" does not become "$3,900$3,950".
    text = re.sub(r"(?i)</(td|th)\s*>", " ", text)
    # Row/paragraph/list boundaries END A STATEMENT.
    text = re.sub(r"(?i)</(tr|p|div|li)\s*>", " . ", text)
    text = re.sub(r"(?i)<br\s*/?>", " . ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = _html.unescape(text)
    # Non-breaking spaces and the bullet glyphs EDGAR filings are full of.
    text = text.replace("\xa0", " ").replace("•", " . ").replace("◦", " . ")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text).strip()
    # Collapse the runs of separators the substitutions above can produce.
    text = re.sub(r"(?:\s*\.\s*){2,}", ". ", text)
    return text.strip()


def find_earnings_release_filings(submissions: dict, limit: int = 8) -> List[dict]:
    """Item-2.02 8-K filings, NEWEST FIRST.

    Newest-first ordering is what makes supersession work: guidance from the
    most recent release that states a metric replaces every earlier statement
    of the SAME metric for the SAME period (section 9).
    """
    recent = (submissions.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    items = recent.get("items") or []
    accessions = recent.get("accessionNumber") or []
    filing_dates = recent.get("filingDate") or []
    report_dates = recent.get("reportDate") or []

    out = []
    for index, form in enumerate(forms):
        if form != "8-K":
            continue
        item_text = items[index] if index < len(items) else ""
        if EARNINGS_RELEASE_ITEM not in (item_text or ""):
            continue
        out.append({
            "accession": accessions[index] if index < len(accessions) else "",
            "filed": filing_dates[index] if index < len(filing_dates) else "",
            "report_date": report_dates[index] if index < len(report_dates) else "",
            "items": item_text,
        })
        if len(out) >= limit:
            break
    return out


# EDGAR's index.json labels each file with its ICON, not its exhibit type, so
# the authoritative type table is the filing's own index page.
_FILING_INDEX_ROW = re.compile(r"(?is)<tr[^>]*>(.*?)</tr>")
_FILING_INDEX_CELL = re.compile(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>")


def select_exhibit_document(filing_index_html: str,
                            exhibit_types: Sequence[str] = ("EX-99.1", "EX-99")) -> Optional[str]:
    """The document name of the earnings-release exhibit, from the filing's
    index page. Types are matched in the order given, so EX-99.1 (the
    conventional earnings release) wins over a bare EX-99."""
    if not filing_index_html:
        return None
    rows = []
    for row_html in _FILING_INDEX_ROW.findall(filing_index_html):
        cells = [re.sub(r"(?s)<[^>]+>", " ", cell) for cell in _FILING_INDEX_CELL.findall(row_html)]
        cells = [re.sub(r"\s+", " ", _html.unescape(c)).strip() for c in cells]
        if len(cells) >= 4:
            rows.append(cells)
    for wanted in exhibit_types:
        for cells in rows:
            # Columns are Seq | Description | Document | Type | Size.
            if cells[3].upper() == wanted.upper():
                document = cells[2].split()[0] if cells[2] else ""
                if document.lower().endswith((".htm", ".html", ".txt")):
                    return document
    return None


# ---------------------------------------------------------------------------
# Numeric patterns
# ---------------------------------------------------------------------------

_NUM = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"

# A RANGE, in the spellings real releases use. Requiring two numbers is one
# of the two precision guards — see the module docstring.
#
# A SCALE WORD may sit between the number and the connector: "$23 billion to
# $24 billion" is how most large-cap guidance is written, and requiring the
# two numbers to be adjacent to the connector silently dropped every such
# range — AT&T's "Capital investment* in the $23 billion to $24 billion range
# annually" among them. The scale itself is recorded on the metric (`scale`),
# not folded into the value, so a figure stays in the units the release used.
_SCALE = r"(?:\s*(?:billion|million|bn|mm)s?)?"

# A PERCENT MARKER is the sign or the word. Only the sign was recognised, so
# "5.0 to 6.0 percent" parsed as a unitless 5.0-6.0 and was stored as an
# absolute dollar amount -- on a live issuer, a $5-$6 adjusted EPS where the
# company had guided 5-6% GROWTH. Two spellings of one measurement, and a
# unit that depends on which one the writer chose is not a unit.
#
# The word is matched only where the SIGN would have been: immediately after
# the number, before any scale word. "6.0 percent" is a percentage; "6.0
# billion, up 3 percent" is not.
_PCT = r"(%|\s*percentage\s+points?|\s*percent(?:age)?)?"

_RANGE_PATTERNS = (
    # "between $3.60 and $3.75", "between 2% and 3%", "between $23 billion and $24 billion"
    re.compile(rf"(?i)between\s*\$?\s*({_NUM})\s*{_PCT}{_SCALE}\s*(?:and|to)\s*"
               rf"\$?\s*({_NUM})\s*{_PCT}{_SCALE}"),
    # "$3.60 to $3.75", "2% to 3%", "a range of 2% to 3%", "$23 billion to $24 billion"
    re.compile(rf"(?i)\$?\s*({_NUM})\s*{_PCT}{_SCALE}\s*to\s*\$?\s*({_NUM})\s*{_PCT}{_SCALE}"),
    # "$ 3.60-3.75", "2%-3%"
    re.compile(rf"(?i)\$?\s*({_NUM})\s*{_PCT}{_SCALE}\s*-\s*\$?\s*({_NUM})\s*{_PCT}{_SCALE}"),
)

# A POINT WITH AN EXPLICIT TOLERANCE — the other accepted guidance shape, and
# the one NVIDIA states every metric in:
#
#     "Revenue is expected to be $91.0 billion, plus or minus 2%."
#     "gross margins are expected to be 74.9% ... plus or minus 50 basis points"
#
# A bare "$91.0 billion" is still refused: without the stated tolerance there
# is nothing distinguishing a forward figure from a reported actual, which is
# the guard the range requirement exists to provide.
_TOLERANCE_PATTERN = re.compile(
    rf"(?i)\$?\s*({_NUM})\s*{_PCT}\s*(billion|million)?\s*,?\s*"
    r"(?:plus\s*or\s*minus|\+/-|±)\s*"
    rf"({_NUM})\s*(%|percent(?:age)?|basis\s*points|bps)")

# "GAAP and non-GAAP gross margins are expected to be 74.9% and 75.0%,
# respectively" — two POINT values for two different bases, which must never
# be read as a single 74.9%-75.0% range. Captured explicitly so the GAAP and
# adjusted measures land in their own metrics rather than being lost.
_DUAL_BASIS_PATTERN = re.compile(
    rf"(?i)GAAP\s+and\s+non-?GAAP\b[^.]{{0,80}}?"
    rf"(?:are\s+)?expected\s+to\s+be\s+"
    rf"(?:approximately\s+)?\$?\s*({_NUM})\s*{_PCT}\s*(billion|million)?\s*"
    rf"and\s+(?:approximately\s+)?\$?\s*({_NUM})\s*{_PCT}\s*(billion|million)?")

# A ONE-SIDED FLOOR — "Free cash flow* of $18 billion+ in 2026", "expected
# growth of 5%+ in 2026", "$45 billion+ to shareholders". AT&T states most of
# its multi-year plan this way, and a floor is real, quantified, checkable
# guidance: dropping it left the T report with no free-cash-flow guidance at
# all while the release led with it.
#
# It is NOT flattened into a range. `bound_type` records that the company
# stated a minimum, so nothing downstream can treat the number as a midpoint
# forecast — which would be reading "at least $18B" as "we expect $18B".
_FLOOR_PATTERN = re.compile(
    rf"(?i)\$?\s*({_NUM})\s*{_PCT}\s*(billion|million)?\s*\+"
    rf"|(?:at\s+least|no\s+less\s+than|or\s+better|or\s+more)\s*\$?\s*({_NUM})\s*{_PCT}")

# How far after a metric name a one-sided floor may begin. Roughly "of", "in
# the", "at least" — a connector, not a clause.
_FLOOR_ADJACENCY = 25

# An explicitly APPROXIMATE point -- "Approximately 81%", "About $1.4 billion",
# "Approximately 2.48 billion". A real outlook table states several of its
# rows this way, and requiring a two-ended range dropped them all: one live
# release guided seven metrics and only two were extracted, because the other
# five were approximate points or sat too far from a forward-looking word.
#
# This is still not a BARE point. The word "approximately" is itself the
# forward-looking marker -- a company does not describe a reported actual as
# approximate -- which is what keeps the range requirement's precision.
# Phase H.9: "estimated" and "expected" join the approximate qualifiers, and
# a couple of intervening words are allowed between the qualifier and the
# figure. A live release guides "increasing 2026 guidance with ESTIMATED
# reported sales OF $101.1 Billion" -- the qualifier, an adjective and a
# preposition all sit between the marker and the number, and requiring
# adjacency dropped the company's headline full-year revenue guidance
# entirely.
_APPROXIMATE_PATTERN = re.compile(
    rf"(?i)\b(?:approximately|about|around|roughly|estimated|expected)\s+"
    rf"(?:[a-z]+\s+){{0,2}}?(?:of\s+)?\$?\s*({_NUM})\s*{_PCT}\s*"
    rf"(billion|million|bn|mm)?")

_BASIS_POINTS_PER_PERCENT = 100.0



# ---------------------------------------------------------------------------
# Metric keywords (section 6)
# ---------------------------------------------------------------------------
#
# (canonical name, unit, percent required, scope, forced basis or None, regex)
#
# ORDER DOES NOT DECIDE PRECEDENCE — length of the matched text does (see
# `_metric_keyword_hits`). "Service revenue growth" must never be read as
# consolidated "revenue growth" merely because the latter is listed first,
# and relying on list order to prevent that is exactly the kind of implicit
# rule that produced the AT&T mis-mapping.

_SCOPE_CONSOLIDATED = "consolidated"
_SCOPE_SERVICE = "service"
_SCOPE_PRODUCT = "product"
_SCOPE_SEGMENT = "segment"
# A named part of revenue that is not a reporting segment -- a product line,
# a delivery model. Distinct from CONSOLIDATED because the company's total
# can move quite differently from it.
_SCOPE_COMPONENT = "component"

_METRIC_PATTERNS = (
    # -- revenue: components FIRST in the file for readability only --------
    (GuidanceMetricName.SERVICE_REVENUE_GROWTH, GuidanceUnit.RATIO, True, _SCOPE_SERVICE, None,
     re.compile(r"(?i)\b(?:consolidated\s+|total\s+)?service\s+revenues?\s+growth\b|"
                r"\bgrowth\s+in\s+(?:consolidated\s+)?service\s+revenues?\b")),
    (GuidanceMetricName.SERVICE_REVENUE, GuidanceUnit.CURRENCY, False, _SCOPE_SERVICE, None,
     re.compile(r"(?i)\b(?:consolidated\s+|total\s+)?service\s+revenues?\b")),
    (GuidanceMetricName.PRODUCT_REVENUE_GROWTH, GuidanceUnit.RATIO, True, _SCOPE_PRODUCT, None,
     re.compile(r"(?i)\bproduct\s+revenues?\s+growth\b")),
    (GuidanceMetricName.PRODUCT_REVENUE, GuidanceUnit.CURRENCY, False, _SCOPE_PRODUCT, None,
     re.compile(r"(?i)\bproduct\s+revenues?\b")),
    # A named business line ("Advanced Connectivity service revenue growth",
    # "Legacy service revenue", "Data Center revenue"). Deliberately matched
    # as SEGMENT rather than consolidated: AT&T's release states four
    # different segment revenue trajectories, none of which is the company's
    # total.
    (GuidanceMetricName.SEGMENT_REVENUE_GROWTH, GuidanceUnit.RATIO, True, _SCOPE_SEGMENT, None,
     re.compile(r"(?i)\b(?:advanced\s+connectivity|legacy|mobility|business\s+wireline|"
                r"consumer\s+wireline|data\s+center|gaming|automotive|segment)\s+"
                r"(?:service\s+)?revenues?\s+growth\b")),

    (GuidanceMetricName.SUBSCRIPTION_REVENUE_GROWTH, GuidanceUnit.RATIO, True,
     _SCOPE_COMPONENT, None,
     re.compile(r"(?i)\b(?:subscription|recurring|saas|software)\s+"
                r"(?:and\s+support\s+)?revenues?\s+growth\b")),
    (GuidanceMetricName.SUBSCRIPTION_REVENUE, GuidanceUnit.CURRENCY, False,
     _SCOPE_COMPONENT, None,
     re.compile(r"(?i)\b(?:subscription|recurring|saas)\s+"
                r"(?:and\s+support\s+)?revenues?\b")),

    # -- cash-flow and earnings GROWTH ------------------------------------
    #
    # These exist so the extractor can say what they ARE. None of them may
    # anchor a revenue-growth assumption -- see `may_anchor_revenue_growth`
    # and `FORBIDDEN_REVENUE_GROWTH_SOURCES` -- but recording them keeps
    # real guidance out of the "none extracted" bucket and lets the coverage
    # matrix count what management actually said.
    (GuidanceMetricName.FREE_CASH_FLOW_GROWTH, GuidanceUnit.RATIO, True,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\b(?:adjusted\s+)?free\s+cash\s+flows?\s+growth\b|"
                r"\bgrowth\s+in\s+(?:adjusted\s+)?free\s+cash\s+flows?\b")),
    (GuidanceMetricName.OPERATING_CASH_FLOW_GROWTH, GuidanceUnit.RATIO, True,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\boperating\s+cash\s+flows?\s+growth\b|"
                r"\bgrowth\s+in\s+operating\s+cash\s+flows?\b")),
    (GuidanceMetricName.EPS_GROWTH, GuidanceUnit.RATIO, True,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\b(?:adjusted\s+)?(?:eps|earnings\s+per\s+share)\s+growth\b|"
                r"\bgrowth\s+in\s+(?:adjusted\s+)?"
                r"(?:eps|earnings\s+per\s+share)\b")),
    (GuidanceMetricName.NET_INCOME_GROWTH, GuidanceUnit.RATIO, True,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\b(?:adjusted\s+)?net\s+income\s+growth\b")),

    # -- revenue: consolidated --------------------------------------------
    (GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH, GuidanceUnit.RATIO, True,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\b(?:consolidated\s+|total\s+)?(?:net\s+)?sales\s+growth\b|"
                r"\b(?:consolidated\s+|total\s+)?revenues?\s+growth\b|"
                r"\bgrowth\s+in\s+(?:consolidated\s+|total\s+)?(?:net\s+)?sales\b|"
                r"\brevenues?\s+(?:is|are)\s+expected\s+to\s+grow\b")),
    (GuidanceMetricName.CONSOLIDATED_REVENUE, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     # Bare "Revenue" and bare "Sales" are both accepted here -- one live
     # outlook table's row label is literally "Sales" and another's is
     # "Revenue" -- which is only safe because a value still has to be a
     # range, a point-with-tolerance or an explicitly approximate figure
     # inside a forward-looking block.
     re.compile(r"(?i)\b(?:consolidated\s+|total\s+|worldwide\s+)?net\s+sales\b|"
                r"\b(?:consolidated\s+|total\s+|worldwide\s+)?revenues?\b|"
                r"\b(?:consolidated\s+|total\s+|worldwide\s+)?sales\b")),
    # Section 7: a guided share count changes every per-share figure derived
    # from guided earnings, so it is a material guided metric in its own
    # right rather than a footnote.
    (GuidanceMetricName.SHARE_COUNT, GuidanceUnit.SHARES, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bshare\s+count\b|\bdiluted\s+shares\s+outstanding\b|"
                r"\bweighted[\s-]average\s+shares\b")),
    (GuidanceMetricName.OTHER_INCOME_EXPENSE, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bother\s*\(?income\)?\s*\(?expense\)?,?\s*net\b")),

    # -- EBITDA. The metric whose absence caused the AT&T mis-mapping. -----
    (GuidanceMetricName.ADJUSTED_EBITDA_GROWTH, GuidanceUnit.RATIO, True,
     _SCOPE_CONSOLIDATED, BASIS_ADJUSTED,
     re.compile(r"(?i)\b(?:adjusted|non-?GAAP)\s+EBITDA\s*\*?\s+growth\b|"
                r"\bgrowth\s+in\s+(?:adjusted|non-?GAAP)\s+EBITDA\b")),
    (GuidanceMetricName.EBITDA_GROWTH, GuidanceUnit.RATIO, True, _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bEBITDA\s*\*?\s+growth\b|\bgrowth\s+in\s+EBITDA\b")),
    (GuidanceMetricName.ADJUSTED_EBITDA_MARGIN, GuidanceUnit.RATIO, True,
     _SCOPE_CONSOLIDATED, BASIS_ADJUSTED,
     re.compile(r"(?i)\b(?:adjusted|non-?GAAP)\s+EBITDA\s+margins?\b")),
    (GuidanceMetricName.EBITDA_MARGIN, GuidanceUnit.RATIO, True, _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bEBITDA\s+margins?\b")),
    (GuidanceMetricName.ADJUSTED_EBITDA, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, BASIS_ADJUSTED,
     re.compile(r"(?i)\b(?:adjusted|non-?GAAP)\s+EBITDA\b")),
    (GuidanceMetricName.EBITDA, GuidanceUnit.CURRENCY, False, _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bEBITDA\b")),

    # -- per share ---------------------------------------------------------
    (GuidanceMetricName.ADJUSTED_EPS, GuidanceUnit.CURRENCY_PER_SHARE, False,
     _SCOPE_CONSOLIDATED, BASIS_ADJUSTED,
     re.compile(r"(?i)\b(?:adjusted|non-?GAAP)\s+(?:diluted\s+)?"
                r"(?:earnings\s+per\s+share|EPS)\s*\*?")),
    (GuidanceMetricName.EPS, GuidanceUnit.CURRENCY_PER_SHARE, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bdiluted\s+(?:earnings\s+per\s+share|EPS)\b|"
                r"(?<!adjusted\s)\bEPS\s*\(GAAP\)")),

    # -- margins and expenses ---------------------------------------------
    (GuidanceMetricName.ADJUSTED_GROSS_MARGIN, GuidanceUnit.RATIO, True,
     _SCOPE_CONSOLIDATED, BASIS_ADJUSTED,
     re.compile(r"(?i)\b(?:adjusted|non-?GAAP)\s+gross\s+margins?\b")),
    (GuidanceMetricName.GROSS_MARGIN, GuidanceUnit.RATIO, True, _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bgross\s+margins?\b")),
    (GuidanceMetricName.ADJUSTED_OPERATING_MARGIN, GuidanceUnit.RATIO, True,
     _SCOPE_CONSOLIDATED, BASIS_ADJUSTED,
     re.compile(r"(?i)\b(?:adjusted|non-?GAAP)\s+operating\s+margins?\b")),
    (GuidanceMetricName.OPERATING_MARGIN, GuidanceUnit.RATIO, True, _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\boperating\s+margins?\b")),
    (GuidanceMetricName.ADJUSTED_OPERATING_EXPENSES, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, BASIS_ADJUSTED,
     re.compile(r"(?i)\b(?:adjusted|non-?GAAP)\s+operating\s+expenses\b")),
    (GuidanceMetricName.OPERATING_EXPENSES, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\boperating\s+expenses\b")),
    (GuidanceMetricName.OPERATING_INCOME, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\boperating\s+income\b")),

    # -- cash and capital --------------------------------------------------
    (GuidanceMetricName.FREE_CASH_FLOW, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bfree\s+cash\s+flow\s*\*?")),
    (GuidanceMetricName.OPERATING_CASH_FLOW, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bcash\s+(?:provided\s+by|from)\s+operating\s+activities\b|"
                r"\boperating\s+cash\s+flow\b")),
    (GuidanceMetricName.CAPEX, GuidanceUnit.CURRENCY, False, _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bcapital\s+expenditures?\b|\bcapital\s+investment\s*\*?|\bcapex\b")),
    (GuidanceMetricName.SHARE_REPURCHASES, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bshare\s+repurchases?\b|\bstock\s+repurchases?\b|\bbuybacks?\b")),
    (GuidanceMetricName.NET_LEVERAGE_TARGET, GuidanceUnit.RATIO, False,
     _SCOPE_CONSOLIDATED, BASIS_COMPANY_DEFINED,
     re.compile(r"(?i)\bnet\s+debt[\s-]*to[\s-]*(?:adjusted\s+)?EBITDA\b|"
                r"\bleverage\s+ratio\b|\bnet\s+leverage\b")),
    (GuidanceMetricName.TAX_RATE, GuidanceUnit.RATIO, True, _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\b(?:GAAP\s+and\s+non-?GAAP\s+)?tax\s+rates?\b|"
                r"\beffective\s+tax\s+rate\b")),
    # Cash items a release guides ALONGSIDE free cash flow. AT&T's own
    # sentence is why they are here: "...free cash flow* outlook anticipates
    # annual cash taxes of $1.0 billion to $1.5 billion and cash
    # contributions to its employee benefit plans...". Both are real guided
    # quantities, and listing them makes each a BOUNDARY for the other — the
    # same mechanism that stops EBITDA growth being read as revenue growth.
    # Without them, free-cash-flow guidance came out as $1.0-1.5B.
    (GuidanceMetricName.CASH_TAXES, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\b(?:annual\s+)?cash\s+taxes\b|\bcash\s+tax\s+payments\b")),
    (GuidanceMetricName.PENSION_CONTRIBUTIONS, GuidanceUnit.CURRENCY, False,
     _SCOPE_CONSOLIDATED, None,
     re.compile(r"(?i)\bcash\s+contributions\b|\bpension\s+contributions\b|"
                r"\bbenefit\s+plan\s+contributions\b")),
)

# ---------------------------------------------------------------------------
# Units and denominator as part of metric identity (spec section 3)
# ---------------------------------------------------------------------------
#
# A live release guided "non-GAAP operating income ... approximately 21% of
# projected revenue" and it was extracted as `operating_income` -- an
# absolute currency metric -- holding 0.21.
#
# The arithmetic was not wrong; the IDENTITY was. "Operating income of $2.1
# billion" and "operating income equal to 21% of revenue" are two different
# quantities: one is a dollar amount, the other is a ratio whose denominator
# is revenue. A metric_id that cannot tell them apart is not an identity, and
# everything downstream -- the coverage matrix, the forward-assumption
# builder, the report -- reads the name and the unit and believes them.
#
# So the unit and the DENOMINATOR participate in identity. A metric whose
# taxonomy unit is an absolute amount, matched against a percentage of
# revenue, resolves to that metric's margin identity. Where no margin
# identity exists the figure is refused, because a percentage of something
# that was never established is not a measurement of anything.

# "of revenue", "of projected revenue", "of net sales", "as a percentage of
# our full-year revenues". The qualifiers between "of" and the noun are
# forward-looking adjectives a release routinely uses; none of them changes
# what the denominator IS.
_PERCENT_OF_REVENUE = re.compile(
    r"(?i)\b(?:as\s+a\s+percent(?:age)?\s+of|of)\s+"
    r"(?:its\s+|our\s+|the\s+|a\s+)?"
    r"(?:projected\s+|expected\s+|anticipated\s+|estimated\s+|forecast(?:ed)?\s+|"
    r"guided\s+|full[\s-]?year\s+|quarterly\s+|total\s+|consolidated\s+|net\s+|"
    r"worldwide\s+)*"
    r"(?:revenues?|sales)\b")

# The OTHER denominator a percentage can have: the prior period. "Free cash
# flow growth of 9.0 to 10.0 percent year-over-year" is a rate of change, and
# the vocabulary already had an identity for it -- nothing routed the figure
# there, so it was stored as $9-$10 of absolute free cash flow against a real
# figure roughly twice that.
#
# Matched on the metric's own surroundings, not on the whole paragraph: a
# release that reports one metric's growth beside another's level would
# otherwise make both growth rates.
_PERCENT_YEAR_OVER_YEAR = re.compile(
    r"(?i)\byear[\s-]over[\s-]year\b|\byear\s+on\s+year\b|\by/y\b|\byoy\b"
    r"|\bgrowth\b|\bgrow(?:s|th|ing)?\b|\bincreas\w*\s+(?:by\s+)?\d"
    r"|\bcompared\s+(?:with|to)\s+(?:fiscal\s+)?\d{4}\b"
    r"|\bversus\s+(?:fiscal\s+)?\d{4}\b")

# (absolute metric, basis) -> the identity a percentage YEAR-OVER-YEAR names.
# Every one of these already existed; the gap was the routing.
_PERCENT_GROWTH_IDENTITY = {
    GuidanceMetricName.FREE_CASH_FLOW: GuidanceMetricName.FREE_CASH_FLOW_GROWTH,
    GuidanceMetricName.OPERATING_CASH_FLOW: GuidanceMetricName.OPERATING_CASH_FLOW_GROWTH,
    GuidanceMetricName.EPS: GuidanceMetricName.EPS_GROWTH,
    GuidanceMetricName.ADJUSTED_EPS: GuidanceMetricName.EPS_GROWTH,
    GuidanceMetricName.EBITDA: GuidanceMetricName.EBITDA_GROWTH,
    GuidanceMetricName.ADJUSTED_EBITDA: GuidanceMetricName.ADJUSTED_EBITDA_GROWTH,
    GuidanceMetricName.CONSOLIDATED_REVENUE: GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH,
    GuidanceMetricName.SERVICE_REVENUE: GuidanceMetricName.SERVICE_REVENUE_GROWTH,
    GuidanceMetricName.PRODUCT_REVENUE: GuidanceMetricName.PRODUCT_REVENUE_GROWTH,
    GuidanceMetricName.SUBSCRIPTION_REVENUE: GuidanceMetricName.SUBSCRIPTION_REVENUE_GROWTH,
    # Deliberately absent: capital expenditure, share repurchases, cash taxes,
    # pension contributions, operating expenses. None has a growth identity in
    # this vocabulary, and a percentage against one of them is refused rather
    # than given a name that does not exist yet.
}

# (absolute metric, basis) -> the identity a percentage OF REVENUE actually
# names. Keyed on basis as well as name because GAAP and non-GAAP margins are
# two statements and merging them loses the distinction (section 6).
_PERCENT_OF_REVENUE_IDENTITY = {
    (GuidanceMetricName.OPERATING_INCOME, BASIS_GAAP):
        GuidanceMetricName.OPERATING_MARGIN,
    (GuidanceMetricName.OPERATING_INCOME, BASIS_ADJUSTED):
        GuidanceMetricName.ADJUSTED_OPERATING_MARGIN,
    (GuidanceMetricName.EBITDA, BASIS_GAAP):
        GuidanceMetricName.EBITDA_MARGIN,
    (GuidanceMetricName.EBITDA, BASIS_ADJUSTED):
        GuidanceMetricName.ADJUSTED_EBITDA_MARGIN,
    (GuidanceMetricName.ADJUSTED_EBITDA, BASIS_ADJUSTED):
        GuidanceMetricName.ADJUSTED_EBITDA_MARGIN,
    (GuidanceMetricName.ADJUSTED_EBITDA, BASIS_GAAP):
        GuidanceMetricName.ADJUSTED_EBITDA_MARGIN,
    # Deliberately NOT listed: operating expenses as a percentage of
    # revenue. That ratio has no identity in this vocabulary, and inventing
    # one here to avoid a refusal is how the gaps above were created. Until
    # an identity exists, the figure is refused and the refusal is recorded.
}

# Units that describe an ABSOLUTE amount. A percentage carried under one of
# these is the mismatch this machinery exists to catch.
_ABSOLUTE_UNITS = (GuidanceUnit.CURRENCY, GuidanceUnit.CURRENCY_PER_SHARE,
                   GuidanceUnit.SHARES)

# Named codes, so a refusal is greppable and the spec's failure-status list
# names something that exists rather than something that was described.
PERCENTAGE_DENOMINATOR_UNRESOLVED = "GUIDANCE_PERCENTAGE_DENOMINATOR_UNRESOLVED"
SOURCE_NOT_PROSPECTIVE = "GUIDANCE_SOURCE_NOT_PROSPECTIVE"


def percentage_denominator_stated(context: str) -> bool:
    """Does the text say what a percentage here would be a percentage OF?

    The one question that decides whether an absolute metric may carry a
    percentage at all. Asked before the value is read, so a figure whose
    denominator was never established is not parsed and then discarded -- it
    is refused at the point the ambiguity exists.
    """
    context = context or ""
    return bool(_PERCENT_OF_REVENUE.search(context)
                or _PERCENT_YEAR_OVER_YEAR.search(context))


def resolve_percentage_identity(name: str, unit: str, basis: str, context: str):
    """(identity, unit) for a percentage value, or None to refuse it.

    Returns the metric unchanged when it is already a ratio. When an ABSOLUTE
    metric carries a percentage, the denominator decides what the figure is,
    and it has to be established from the text before any identity can be
    assigned:

        % OF REVENUE      -> that metric's margin
        % YEAR-OVER-YEAR  -> that metric's growth rate
        anything else     -> refused

    Revenue is checked first because "operating income of 21% of projected
    revenue" also contains growth-ish vocabulary in most releases, and the
    stated denominator is the more specific reading.
    """
    if unit not in _ABSOLUTE_UNITS:
        return name, unit
    context = context or ""
    if _PERCENT_OF_REVENUE.search(context):
        resolved = _PERCENT_OF_REVENUE_IDENTITY.get((name, basis))
        if resolved is not None and resolved in _METRIC_BY_NAME:
            return resolved, GuidanceUnit.RATIO
        return None
    if _PERCENT_YEAR_OVER_YEAR.search(context):
        resolved = _PERCENT_GROWTH_IDENTITY.get(name)
        if resolved is not None and resolved in _METRIC_BY_NAME:
            return resolved, GuidanceUnit.RATIO
    return None


_METRIC_BY_NAME = {entry[0]: entry for entry in _METRIC_PATTERNS}

# How far after a metric keyword a value may appear and still be that
# metric's guidance. Tight on purpose: the further away a number is, the more
# likely it belongs to a different line of the release. Reducing this from
# 160 to 110 fixed a real false positive on AOS's January release, where a
# flattened reconciliation table put "Free cash flow (non-GAAP) $546.0
# $473.8" within reach of the EPS guidance range printed underneath it.
#
# Phase H.6: the window is now additionally cut at the first CLAUSE BOUNDARY
# (see `_clause_window`), which is what actually stops one bullet's metric
# name from claiming the next bullet's number.
_WINDOW = 110

# The narrower window used to decide GAAP vs adjusted. It looks only at the
# text IMMEDIATELY BEFORE the metric keyword, because "adjusted" is a
# modifier of the metric name ("adjusted EPS"), not a property of the
# paragraph. Using the wide context window instead made AOS's plain sales
# growth come out as "adjusted" purely because an adjusted-EPS sentence
# happened to sit nearby -- and, worse, made the GAAP diluted-EPS guidance
# get dropped entirely as a suspected duplicate of the adjusted measure.
_BASIS_LOOKBEHIND = 30

_SCALE_RE = re.compile(r"(?i)\b(millions?|billions?)\b")

# A range only counts as GUIDANCE when the surrounding text says it is
# forward-looking. Without this, a prior-year comparison range would qualify.
_FORWARD_MARKERS = re.compile(
    r"(?i)\b(outlook|guidance|expect\w*|anticipat\w*|forecast\w*|target\w*|"
    r"project\w*|estimat\w*|reaffirm\w*|reiterat\w*|confirm\w*|narrow\w*|"
    r"rais\w*|lower\w*|"
    r"updat\w*|plans?\s+to|on\s+track)\b")

# Section 9: an outlook the company has taken back is not current guidance.
_WITHDRAWAL_MARKERS = re.compile(
    r"(?i)\b(withdraw\w*|suspend\w*|no\s+longer\s+(?:providing|issuing)\s+"
    r"(?:full[\s-]year\s+)?guidance|rescind\w*)\b")


def _to_number(text: str) -> Optional[float]:
    try:
        return float(text.replace(",", ""))
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Fiscal period parsing (section 8)
# ---------------------------------------------------------------------------

_QUARTER_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4,
                  "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}

# "FY2026" is a fiscal-year TOKEN, not a statement that a figure is annual.
# A live release headed "<company>'s third quarter FY2026 targets" was read as
# full-year guidance because this pattern accepted "fiscal 2026" but not
# "FY2026": the quarter alternative failed, and the ANNUAL pattern below then
# matched the bare FY token. The quarter qualifier is what decides the
# frequency, so it is now recognised ahead of every spelling of the year that
# can follow it.
_FY_YEAR = r"(?:of\s+)?(?:fiscal\s+(?:year\s+)?|FY\s*)?(20\d{2})"
_QUARTER_PERIOD_RE = re.compile(
    r"(?i)\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+"
    + _FY_YEAR
    + r"|\b(?:fiscal\s+)?Q([1-4])\s*" + _FY_YEAR
    + r"|\bQ([1-4])\s*FY\s*(20\d{2}|\d{2})\b")

# A bare year immediately attached to a guidance/outlook word names the
# TARGET fiscal year: "increasing 2026 guidance with estimated reported sales
# of $101.1 Billion" is full-year 2026 guidance issued with Q2 results. Read
# without this, the nearest period declaration was the reporting quarter and
# the company's full-year outlook was labelled next-quarter guidance.
_YEAR_GUIDANCE_RE = re.compile(
    r"(?i)\b(20\d{2})\s+(?:full[\s-]year\s+)?(?:guidance|outlook|target)\b"
    r"|\b(?:guidance|outlook|target)\s+for\s+(?:full[\s-]year\s+)?(20\d{2})\b")

_ANNUAL_PERIOD_RE = re.compile(
    r"(?i)\b(?:full[\s-]year|fiscal\s+year|fiscal|full\s+fiscal\s+year|"
    r"for\s+the\s+year|calendar\s+year)\s+(?:of\s+)?(20\d{2})\b"
    r"|\bFY\s*(20\d{2})\b")

_MULTI_YEAR_RE = re.compile(r"\b(20\d{2})\s*-\s*(20\d{2})\b")

_YEAR_RE = re.compile(r"\b(20\d{2})\b")


@dataclass(frozen=True)
class GuidancePeriod:
    label: str
    period_type: str
    fiscal_year: int
    quarter: Optional[int] = None


def parse_guidance_period(context: str, filed: str) -> Optional[GuidancePeriod]:
    """The fiscal period a guidance figure applies to, from its own context.

    Non-calendar fiscal years are the reason this exists. NVIDIA's May-2026
    release guides "the second quarter of fiscal 2027"; hard-coding the
    expected year to the calendar year of the filing rejected every value in
    the document. The period is now READ, and only then checked for
    plausibility against the filing date.

    Returns None when no period can be identified — which is a rejection
    (section 11: "reject a number without a period"), not a default.
    """
    quarter_match = _QUARTER_PERIOD_RE.search(context)
    if quarter_match:
        groups = quarter_match.groups()
        if groups[0] and groups[1]:
            quarter = _QUARTER_WORDS.get(groups[0].lower())
            year = int(groups[1])
        elif groups[2] and groups[3]:
            quarter, year = int(groups[2]), int(groups[3])
        else:
            quarter = int(groups[4])
            raw_year = groups[5]
            year = int(raw_year) if len(raw_year) == 4 else 2000 + int(raw_year)
        if quarter:
            return GuidancePeriod(label=f"Q{quarter} FY{year}",
                                  period_type=GuidancePeriodType.QUARTER,
                                  fiscal_year=year, quarter=quarter)

    multi = _MULTI_YEAR_RE.search(context)
    annual_match = _ANNUAL_PERIOD_RE.search(context)
    if annual_match:
        year = int(annual_match.group(1) or annual_match.group(2))
        # A multi-year outlook ("2026-2028") that also names a specific year
        # for THIS figure ("in the 3% to 4% range in 2026") applies to that
        # year; the span is recorded as the period type so a reader is not
        # told a three-year plan is a one-year outlook.
        if multi and int(multi.group(1)) <= year <= int(multi.group(2)):
            return GuidancePeriod(label=f"FY{year}", period_type=GuidancePeriodType.ANNUAL,
                                  fiscal_year=year)
        return GuidancePeriod(label=f"FY{year}", period_type=GuidancePeriodType.ANNUAL,
                              fiscal_year=year)

    if multi:
        return GuidancePeriod(label=f"FY{multi.group(1)}-FY{multi.group(2)}",
                              period_type=GuidancePeriodType.MULTI_YEAR,
                              fiscal_year=int(multi.group(1)))

    years = sorted({int(y) for y in _YEAR_RE.findall(context)})
    if len(years) == 1:
        return GuidancePeriod(label=f"FY{years[0]}", period_type=GuidancePeriodType.ANNUAL,
                              fiscal_year=years[0])
    return None


# An outlook HEADER declares the period for everything that follows it until
# the next header. NVIDIA's release is built this way and cannot be read any
# other way:
#
#     NVIDIA's outlook for the second quarter of fiscal 2027 is as follows:
#     Revenue is expected to be $91.0 billion, plus or minus 2%.
#     GAAP and non-GAAP gross margins are expected to be 74.9% and 75.0% ...
#     GAAP and non-GAAP operating expenses are expected to be ... $8.3 billion.
#     For the full year fiscal 2027, NVIDIA expects ... tax rates ...
#
# The gross-margin and operating-expense sentences name no period at all; the
# only statement of their period is the header three sentences earlier. Worse,
# a window wide enough to reach the header ALSO reaches the full-year tax
# sentence below, which is a different period — so the header must win by
# being the nearest PRECEDING declaration, not by being anywhere in range.
_OUTLOOK_HEADER_RE = re.compile(
    r"(?i)\boutlook\s+for\s+(?:the\s+)?[^.]{0,60}?"
    r"((?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+"
    r"(?:of\s+)?(?:fiscal\s+(?:year\s+)?)?20\d{2}"
    r"|(?:full[\s-]year|fiscal\s+year|fiscal)\s+20\d{2}"
    r"|20\d{2}\s*-\s*20\d{2})"
    r"|\bfor\s+the\s+(full[\s-]year\s+fiscal\s+20\d{2}|full\s+year\s+fiscal\s+20\d{2})"
    r"|\b(?:outlook|guidance)\s+for\s+(20\d{2})\b")

# How far back a preceding outlook header may sit and still govern a figure.
# Generous enough to span the three or four sentences an outlook block runs
# to, far too tight to reach the previous quarter's block.
_HEADER_REACH = 900

# How close an outlook/guidance word must sit to a period phrase for that
# phrase to be a TABLE TITLE rather than a passing mention.
_TITLE_ADJACENCY = 40
_OUTLOOK_TITLE_WORD = re.compile(
    r"(?i)\b(outlook|guidance|forecast|expects?|expectations?|anticipat\w*)\b")

# A bare year inside the figure's OWN clause ("in the 3% to 4% range in 2026,
# improving to 5% or better in 2028") is the strongest signal there is: it
# sits with the number rather than in a header that covers several. Applied
# first, and only when the clause names exactly one year.
_CLAUSE_YEAR_REACH = 90


def _period_declarations(text: str) -> List[Tuple[int, GuidancePeriod]]:
    """Every period declaration in the document, with its position.

    Two sources, because real releases state a period in two shapes and the
    header form alone misses the one that matters most:

    1. An outlook HEADER -- "NVIDIA's outlook for the second quarter of
       fiscal 2027 is as follows:".
    2. Any standalone period phrase -- "Full-Year 2026 Financial Outlook",
       "Full Year 2026", "fiscal 2027". A guidance TABLE is titled this way
       and then lists its rows, and the rows themselves name no period at
       all. Recognising only the header form meant a live release's sales,
       gross-margin, operating-expense, other-income and share-count rows
       were all rejected for having "no fiscal period", while the two rows
       that happened to sit near prose survived -- so a seven-metric outlook
       was reported as two.
    """
    declarations: List[Tuple[int, GuidancePeriod]] = []
    for match in _OUTLOOK_HEADER_RE.finditer(text):
        phrase = next((g for g in match.groups() if g), None)
        if not phrase:
            continue
        period = parse_guidance_period(phrase, "")
        if period is not None:
            declarations.append((match.end(), period))
    # A bare period phrase counts as a DECLARATION only when it sits next to
    # an outlook/guidance word -- "Full-Year 2026 Financial Outlook" is a
    # table title, "second-quarter 2026 results" is a heading over reported
    # actuals. Without that restriction the results section's own quarter
    # phrase becomes the nearest preceding declaration for the guidance table
    # below it, and a company's FULL-YEAR outlook is relabelled as quarterly.
    for match in _YEAR_GUIDANCE_RE.finditer(text):
        year = next((g for g in match.groups() if g), None)
        if year:
            declarations.append((match.end(), GuidancePeriod(
                label=f"FY{year}", period_type=GuidancePeriodType.ANNUAL,
                fiscal_year=int(year))))
    for pattern in (_QUARTER_PERIOD_RE, _ANNUAL_PERIOD_RE):
        for match in pattern.finditer(text):
            neighbourhood = text[max(0, match.start() - _TITLE_ADJACENCY):
                                 match.end() + _TITLE_ADJACENCY]
            if not _OUTLOOK_TITLE_WORD.search(neighbourhood):
                continue
            period = parse_guidance_period(match.group(0), "")
            if period is not None:
                declarations.append((match.end(), period))
    declarations.sort(key=lambda pair: pair[0])
    return declarations


def _clause_after(text: str, position: int, reach: int = _CLAUSE_YEAR_REACH) -> str:
    """The remainder of the statement that begins at `position`.

    Cut at the first clause boundary, which is what keeps AT&T's "in the 3%
    to 4% range in 2026" from reading the "2028" of a later bullet, and keeps
    NVIDIA's "$8.3 billion, respectively." from reading the "fiscal 2027" of
    the full-year tax sentence that follows it.
    """
    tail = text[position:position + reach]
    boundary = _CLAUSE_BOUNDARY.search(tail)
    return tail[:boundary.start()] if boundary else tail


# A growth statement names TWO periods: the one it targets and the one it is
# measured against. "revenue growth of 27% to 29% compared with the first
# quarter of fiscal 2026" targets Q1 FY2027 and is measured against Q1 FY2026,
# and reading the comparison clause as the target files next year's guidance
# under last year -- where `_period_is_plausible` then rejects it outright,
# because a release cannot guide a period that has already ended.
#
# This is section 11's rule in a third pair: issue period, target period and
# COMPARISON period are three different fields and none of them may stand in
# for another.
_COMPARISON_CLAUSE_RE = re.compile(
    r"(?i)\b(?:compared\s+(?:with|to)|versus|vs\.?|against|relative\s+to|"
    r"year[\s-]over[\s-]year\s+from|up\s+from|down\s+from|from\s+the\s+"
    r"(?:prior|year[\s-]ago|comparable))\b")


def _before_comparison(fragment: str) -> str:
    """The part of a statement that is ABOUT the guided period.

    Everything from a comparison connector onward describes the BASE, so it
    is removed before any period is read out of the text. The comparison
    period is not lost -- the growth derivation records it separately -- it
    simply may not answer "which period is being guided".
    """
    match = _COMPARISON_CLAUSE_RE.search(fragment or "")
    return fragment[:match.start()] if match else fragment


def resolve_guidance_period(text: str, keyword_start: int, value_start: int, value_end: int,
                            filed: str,
                            declarations: Optional[Sequence[Tuple[int, GuidancePeriod]]] = None
                            ) -> Optional[GuidancePeriod]:
    """Which fiscal period this particular figure applies to.

    Four sources, in strict priority order, because they disagree in real
    releases and picking the wrong one attaches a number to the wrong year:

    1. A quarter named in the figure's own statement.
    2. The FIRST year named after the value, within the same statement. AT&T's
       "Adjusted EBITDA* growth in the 3% to 4% range in 2026, improving to
       5% or better in 2028" is FY2026 guidance — the year attached to the
       number, not the far end of the multi-year plan the bullet list sits
       under.
    3. The nearest PRECEDING outlook header. This is the ONLY statement of
       period NVIDIA's gross-margin and operating-expense lines have; both
       sentences name no period at all and inherit "the second quarter of
       fiscal 2027" from the header three sentences above.
    4. The surrounding context, as a last resort.

    Returns None when none of the four identifies a period — a rejection
    (section 11), never a default.
    """
    # A guidance-year phrase in the statement or its immediate lookbehind
    # names the target directly and outranks everything else.
    around = text[max(0, keyword_start - _QUALIFIER_LOOKBEHIND):value_end]
    year_guidance = _YEAR_GUIDANCE_RE.search(around)
    if year_guidance:
        year = next((g for g in year_guidance.groups() if g), None)
        if year:
            return GuidancePeriod(label=f"FY{year}",
                                  period_type=GuidancePeriodType.ANNUAL,
                                  fiscal_year=int(year))

    statement = _before_comparison(
        text[keyword_start:value_end] + _clause_after(text, value_end))
    if _QUARTER_PERIOD_RE.search(statement):
        parsed = parse_guidance_period(statement, filed)
        if parsed is not None:
            return parsed

    trailing_years = _YEAR_RE.findall(_before_comparison(_clause_after(text, value_end)))
    if trailing_years:
        year = int(trailing_years[0])
        return GuidancePeriod(label=f"FY{year}", period_type=GuidancePeriodType.ANNUAL,
                              fiscal_year=year)

    statement_years = {int(y) for y in _YEAR_RE.findall(statement)}
    if len(statement_years) == 1:
        year = statement_years.pop()
        return GuidancePeriod(label=f"FY{year}", period_type=GuidancePeriodType.ANNUAL,
                              fiscal_year=year)

    if declarations is None:
        declarations = _period_declarations(text)
    preceding = [(position, period) for position, period in declarations
                 if position <= value_start and value_start - position <= _HEADER_REACH]
    if preceding:
        return max(preceding, key=lambda pair: pair[0])[1]

    context = text[max(0, keyword_start - _WINDOW):value_end + _WINDOW]
    return parse_guidance_period(context, filed)


def _period_is_plausible(period: GuidancePeriod, filed: str) -> bool:
    """A guided period must be the filing's own year or the next one.

    Deliberately loose enough for every non-calendar fiscal year (NVDA files
    in May 2026 and guides fiscal 2027) and strict enough to reject the prior
    year's actuals, which is the whole reason a year check exists. A
    multi-year plan is accepted when its FIRST year clears the same test.
    """
    if not filed or len(filed) < 4:
        return True
    try:
        filed_year = int(filed[:4])
    except ValueError:
        return True
    return filed_year <= period.fiscal_year <= filed_year + 1


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

# Characters that end one statement in flattened filing text. `html_to_text`
# turns every block-level boundary into a full stop for exactly this reason.
_CLAUSE_BOUNDARY = re.compile(r"[.;:]\s")


def _clause_window(text: str, start: int, limit: int = _WINDOW) -> str:
    """The text after a keyword, cut at the first clause boundary.

    This is the fix for the AT&T mis-mapping stated as a rule: a metric name
    may only claim a value that appears in ITS OWN clause. "Service revenue
    growth in the low-single-digit range annually." ends there; the "3% to 4%"
    in the following sentence belongs to whatever that sentence names.
    """
    window = text[start:start + limit]
    boundary = _CLAUSE_BOUNDARY.search(window)
    return window[:boundary.start()] if boundary else window


def _metric_keyword_hits(text: str) -> List[Tuple[int, int, tuple]]:
    """Every metric keyword occurrence, LONGEST MATCH WINNING at each start.

    Longest-match is what makes "Advanced Connectivity service revenue
    growth" a segment metric and "Service revenue growth" a service metric,
    rather than both collapsing into consolidated `revenue_growth` because
    that pattern also matches the tail of the phrase. Relying on the order of
    the pattern table for this is precisely what failed on AT&T.
    """
    hits: List[Tuple[int, int, tuple]] = []
    for entry in _METRIC_PATTERNS:
        for match in entry[5].finditer(text):
            hits.append((match.start(), match.end(), entry))

    # Drop any hit fully contained in a longer hit that starts no later.
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    kept: List[Tuple[int, int, tuple]] = []
    for start, end, entry in hits:
        if any(k_start <= start and end <= k_end for k_start, k_end, _e in kept):
            continue
        kept.append((start, end, entry))
    kept.sort(key=lambda h: h[0])
    return kept


# Words that turn a following "A to B" into a CHANGE plus a LEVEL rather than
# a range: "increasing adjusted EPS guidance by $0.13 to $11.68" raises
# guidance BY $0.13, arriving AT $11.68. Reading it as a $0.13-$11.68 range
# produced a guidance record spanning two orders of magnitude.
_INCREMENT_MARKERS = re.compile(
    r"(?i)\b(?:by|up\s+by|down\s+by|increas\w*\s+by|decreas\w*\s+by|rais\w*\s+by|"
    r"lower\w*\s+by|reduc\w*\s+by)\s*\$?\s*$")
_INCREMENT_LOOKBEHIND = 30


def _is_increment_phrase(window: str, position: int) -> bool:
    """Is this "A to B" an increment arriving at a level, not a range?"""
    prefix = window[max(0, position - _INCREMENT_LOOKBEHIND):position]
    return bool(_INCREMENT_MARKERS.search(prefix))


# "…or 7.3% at the midpoint" — a company stating a single guided figure and
# then its implied rate. The rate IS the guidance; it is simply not written
# as a range.
_MIDPOINT_PATTERN = re.compile(
    rf"(?i)\bor\s+\$?\s*({_NUM})\s*{_PCT}\s*(billion|million)?\s+at\s+the\s+midpoint")


# A forward qualifier can sit before the METRIC NAME rather than before the
# number -- "increasing 2026 guidance with ESTIMATED reported sales of $101.1
# Billion". The qualifier, an adjective and a preposition separate it from the
# figure, so a pattern anchored on the number alone cannot see it, and the
# company's headline full-year revenue guidance was dropped.
#
# When such a qualifier is present immediately before the keyword, a bare
# figure sitting immediately after it is accepted as an approximate point.
# Both distances are short on purpose: this widens what counts as a forward
# statement, not what counts as a nearby number.
_QUALIFIER_LOOKBEHIND = 40
_BARE_FIGURE_ADJACENCY = 25
# The same forward vocabulary `_FORWARD_MARKERS` uses, in the same stem
# form. This list held past participles only -- "expected", "projected",
# "anticipated" -- so "we EXPECT revenue of $90 billion" qualified nothing
# while "our expected revenue" did. Two spellings of one vocabulary, one of
# them matching what releases actually write.
_FORWARD_QUALIFIER = re.compile(
    r"(?i)\b(?:estimat\w*|expect\w*|approximately|about|around|roughly|guidance|"
    r"outlook|forecast\w*|project\w*|anticipat\w*|target\w*|reaffirm\w*|"
    r"reiterat\w*|confirm\w*)\b")
_BARE_FIGURE_PATTERN = re.compile(
    rf"(?i)^[^0-9%$]{{0,{_BARE_FIGURE_ADJACENCY}}}?\$?\s*({_NUM})\s*{_PCT}\s*"
    rf"(billion|million|bn|mm)?")


# Words that mark a figure as a REPORTED ACTUAL. A live release says
# "2026 Second-Quarter REPORTED sales growth of 6.6%", and the preceding
# sentence ends with the word "outlook" -- so a lookbehind that crosses the
# sentence boundary sees a forward qualifier and accepts last quarter's
# actual as guidance. Both guards below exist for that one sentence pair.
_REPORTED_ACTUAL_MARKERS = re.compile(
    r"(?i)\b(?:results|delivered|achieved|posted|recorded|were|was|surpassing|"
    r"quarter\s+reported)\b")


def _same_statement_lookbehind(lookbehind: str) -> str:
    """Only the part of the lookbehind inside the CURRENT statement."""
    boundary = None
    for match in _CLAUSE_BOUNDARY.finditer(lookbehind):
        boundary = match.end()
    return lookbehind[boundary:] if boundary is not None else lookbehind


def _qualified_bare_figure(window: str, lookbehind: str, require_percent: bool):
    """A bare figure that a qualifier in the SAME STATEMENT makes forward-looking.

    The qualifier may sit on either side of the metric name. This function
    already knew the first half of that -- "increasing 2026 guidance with
    ESTIMATED reported sales of $101.1 Billion" puts the qualifier before the
    metric, and looking only immediately before the NUMBER missed it -- and
    then looked only in the lookbehind, which misses the mirror case:

        "guidance for revenue of $90 billion"   qualifier before the metric
        "revenue guidance of $90 billion"       qualifier after it

    Both sentences say one thing. A live issuer wrote the second and its
    FULL-YEAR outlook was dropped while the next-quarter guidance in the same
    release was kept -- so nothing downstream could derive an annual growth
    rate, and a valuation that should have been LIMITED was published as
    fully usable.

    Word order is not a semantic property. Qualification belongs to the
    statement, and both sides of the metric name are the statement.
    """
    lookbehind = _same_statement_lookbehind(lookbehind or "")
    # Only the text between the metric name and the figure counts on the
    # right: the window is already cut at a clause boundary and at the next
    # metric keyword, so this cannot reach into a neighbouring statement.
    match = _BARE_FIGURE_PATTERN.match(window or "")
    if match is None:
        return None
    leading = (window or "")[:match.start(1)]

    qualified_before = bool(lookbehind and _FORWARD_QUALIFIER.search(lookbehind))
    qualified_after = bool(_FORWARD_QUALIFIER.search(leading))
    if not (qualified_before or qualified_after):
        return None

    # A bare point is the WEAKEST value shape and is ranked last for that
    # reason. If the window also holds a properly formed RANGE, the metric's
    # value here is that range -- and a bare number taken from the same
    # window is something else: a footnote marker, a year, a count.
    #
    # "Free cash flow 1 growth of 9.0 to 10.0 percent" is the case. The "1"
    # is a footnote reference sitting inside the metric name, and reading it
    # as $1 of guided free cash flow is not a small error.
    if any(pattern.search(window or "") for pattern in _RANGE_PATTERNS):
        return None
    # A reported actual stays refused whichever side its qualifier sits on.
    if _REPORTED_ACTUAL_MARKERS.search(lookbehind) or \
            _REPORTED_ACTUAL_MARKERS.search(leading):
        return None
    value = _to_number(match.group(1))
    is_percent = bool(match.group(2))
    if value is None or require_percent != is_percent:
        return None
    return (value, value, is_percent, match.group(0).strip(),
            GuidanceBound.APPROXIMATELY)


# A unit written ONCE, at the end of a range, applies to both ends: "9.0 to
# 10.0 percent" is a percentage range, and reading the first number as
# unitless is how a 5-6% growth guide became a $5-$6 per-share figure.
#
# This does NOT relax the guard it sits next to. That guard exists for
# "6.6% to $25.3 Billion" -- a growth RATE and a sales LEVEL joined by the
# word "to" -- where the two ends carry DIFFERENT units. A unit distributes
# only when the other end carries no unit at all: no currency sign and no
# scale word anywhere in the matched span.
_CURRENCY_OR_SCALE = re.compile(r"(?i)\$|(?:billion|million|bn|mm)s?")


def _percent_unit_distributes(matched: str) -> bool:
    """Is the single percent marker in this range the unit for both ends?"""
    return not _CURRENCY_OR_SCALE.search(matched or "")


def _find_range(window: str, require_percent: bool, lookbehind: str = ""
                ) -> Optional[Tuple[float, float, bool, str, str]]:
    """(low, high, was_percent, matched_text, bound_type) for the first usable value.

    Accepts a range, a point-with-tolerance, or a one-sided floor. A bare
    point with no stated tolerance and no floor marker is refused — see the
    module docstring.
    """
    tolerance = _TOLERANCE_PATTERN.search(window)
    if tolerance is not None:
        centre = _to_number(tolerance.group(1))
        spread = _to_number(tolerance.group(4))
        if centre is not None and spread is not None:
            is_percent = bool(tolerance.group(2))
            unit_word = (tolerance.group(5) or "").lower()
            if "basis" in unit_word or "bps" in unit_word:
                delta = spread / _BASIS_POINTS_PER_PERCENT
            else:
                delta = centre * spread / 100.0
            if (is_percent and require_percent) or (not is_percent and not require_percent):
                return (centre - delta, centre + delta, is_percent,
                        tolerance.group(0).strip(), GuidanceBound.RANGE)

    for pattern in _RANGE_PATTERNS:
        for match in pattern.finditer(window):
            low = _to_number(match.group(1))
            high = _to_number(match.group(3))
            if low is None or high is None:
                continue
            left_percent = bool(match.group(2))
            right_percent = bool(match.group(4))
            # Phase H.9 -- BOTH ENDS MUST CARRY THE SAME UNIT.
            #
            # A live release says "Second-Quarter reported sales growth of
            # 6.6% to $25.3 Billion". The two numbers are a growth RATE and a
            # sales LEVEL joined by the word "to", and reading them as a
            # 6.6%-25.3% range produced a fabricated revenue-growth guidance
            # that then anchored the DCF's year-1 assumption. A range whose
            # ends disagree about their unit is not a range.
            if left_percent != right_percent:
                if not _percent_unit_distributes(match.group(0)):
                    continue
                is_percent = True
            else:
                is_percent = left_percent
            if require_percent != is_percent:
                continue
            if high < low or low == high:
                continue
            if _is_increment_phrase(window, match.start()):
                continue
            return low, high, is_percent, match.group(0).strip(), GuidanceBound.RANGE

    # A FLOOR must sit essentially ADJACENT to the metric name — "Free cash
    # flow* of $18 billion+", not "free cash flow* through 2028, its plans to
    # return $45 billion+ to shareholders". A bare "N+" is far weaker evidence
    # than a two-ended range (it is one token, and releases are full of them),
    # so it only counts when the connector between the metric and the number
    # is a preposition or two. Without this bound, AT&T's capital-return plan
    # ($45 billion+ to shareholders) was captured as free-cash-flow guidance,
    # and a SEGMENT's "expected growth of 6%+" as consolidated EBITDA growth.
    midpoint = _MIDPOINT_PATTERN.search(window)
    if midpoint is not None:
        value = _to_number(midpoint.group(1))
        is_percent = bool(midpoint.group(2))
        if value is not None and require_percent == is_percent:
            return (value, value, is_percent, midpoint.group(0).strip(),
                    GuidanceBound.APPROXIMATELY)

    approximate = _APPROXIMATE_PATTERN.search(window)
    if approximate is not None and approximate.start() <= _FLOOR_ADJACENCY:
        value = _to_number(approximate.group(1))
        is_percent = bool(approximate.group(2))
        if value is not None and require_percent == is_percent:
            return (value, value, is_percent, approximate.group(0).strip(),
                    GuidanceBound.APPROXIMATELY)

    floor = _FLOOR_PATTERN.search(window)
    if floor is not None and floor.start() <= _FLOOR_ADJACENCY:
        value = _to_number(floor.group(1) or floor.group(4))
        is_percent = bool(floor.group(2) or floor.group(5))
        if value is not None and require_percent == is_percent:
            return value, value, is_percent, floor.group(0).strip(), GuidanceBound.AT_LEAST

    # Last: a bare figure that a qualifier immediately before the metric name
    # makes forward-looking. Ranked below every explicit form so a stated
    # range, tolerance or floor always keeps its own bound type.
    qualified = _qualified_bare_figure(window, lookbehind, require_percent)
    if qualified is not None:
        return qualified
    return None


def _guidance_id(symbol: str, accession: str, name: str, period_label: str) -> str:
    digest = hashlib.sha256(
        f"{symbol}|{accession}|{name}|{period_label}".encode("utf-8")).hexdigest()[:12]
    return f"gd_{digest}"



# How far an outlook block's influence extends past its header. Long enough
# for a full guidance table (a live one runs about 700 characters across
# seven rows), short enough not to reach the next section of the release.
_OUTLOOK_BLOCK_REACH = 900

_OUTLOOK_BLOCK_RE = re.compile(
    r"(?i)\b(?:financial\s+outlook|full[\s-]year\s+outlook|guidance\s+summary|"
    r"outlook\s+for|financial\s+guidance|updated\s+guidance|"
    r"summarizes\s+the\s+[^.]{0,40}?outlook)\b")


def _outlook_block_spans(text):
    """(start, end) spans a guidance table's header governs."""
    return [(m.end(), m.end() + _OUTLOOK_BLOCK_REACH)
            for m in _OUTLOOK_BLOCK_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Blocks a release itself labels as REPORTED RESULTS
# ---------------------------------------------------------------------------
#
# An earnings release is mostly history: condensed statements, a
# reconciliation, the share-count table behind the EPS denominators. The
# forward qualification for a figure was a forward-looking WORD within reach
# of it -- and those words are everywhere in the prose wrapped around those
# tables. A live run published a historical weighted-average diluted share
# count as management guidance because "expects" appeared in a paragraph
# above the table.
#
# Proximity is not qualification. A release states, in its own headings,
# which of its numbers already happened; a figure inside such a heading's
# block is a reported actual unless the release re-qualifies it explicitly.
_REPORTED_BLOCK_RE = re.compile(
    r"(?i)\b(?:condensed\s+)?consolidated\s+"
    r"(?:statements?|balance\s+sheets?|results)\b"
    r"|\bstatements?\s+of\s+(?:operations|earnings|income|cash\s+flows)\b"
    r"|\breconciliation\s+of\s+(?:gaap|reported|net|non-?gaap)"
    # Deliberately NOT a bare "GAAP to Non-GAAP". Releases print that phrase
    # inline as a cross-reference -- "See accompanying GAAP to Non-GAAP
    # reconciliations" -- in the middle of their HIGHLIGHTS prose, and
    # reading it as a table heading laid a 700-character historical block
    # over the paragraph carrying the company's own narrowed full-year
    # outlook. A real reconciliation table is headed "Reconciliation of ...",
    # which the alternative above already matches.
    r"|\bweighted[\s-]average\s+shares\s+used\b"
    r"|\b(?:three|six|nine|twelve)\s+months\s+ended\b"
    r"|\b(?:second|third|fourth|first)[\s-]quarter\s+(?:and\s+)?"
    r"(?:full[\s-]year\s+)?results\b")

# How far a reported-results heading governs when nothing else ends it.
_REPORTED_BLOCK_REACH = 700

# What ENDS a reported-results block. Deliberately broader than
# `_OUTLOOK_BLOCK_RE`, and deliberately not the same list: that pattern
# decides what QUALIFIES as forward-looking and must stay narrow, while this
# one only decides where a historical block stops. Erring wide here can only
# shorten a block -- a figure past the terminator still has to qualify on its
# own -- whereas erring wide there would qualify figures directly.
#
# A live release heads its guidance table with the single word "Outlook",
# which `_OUTLOOK_BLOCK_RE` does not match; without this the preceding
# "Three Months Ended" block reached over the whole table and the company's
# own full-year outlook was refused as reported results.
_REPORTED_BLOCK_TERMINATOR_RE = re.compile(
    r"(?i)outlook|guidance|we\s+expect|the\s+company\s+expects")


def _reported_block_spans(text):
    """(start, end) spans a reported-results heading governs.

    Each span ends at the earlier of the next outlook/guidance heading and
    `_REPORTED_BLOCK_REACH`, so a release that puts its outlook directly
    below its statements does not have the outlook swallowed by them.
    """
    outlook_starts = sorted(m.start() for m in _REPORTED_BLOCK_TERMINATOR_RE.finditer(text))
    spans = []
    for match in _REPORTED_BLOCK_RE.finditer(text):
        start = match.start()
        stop = start + _REPORTED_BLOCK_REACH
        following = next((o for o in outlook_starts if o > start), None)
        if following is not None:
            stop = min(stop, following)
        spans.append((start, stop))
    return spans


def _inside_reported_block(position, blocks) -> bool:
    return any(start <= position <= end for start, end in blocks)


def _inside_outlook_block(position, declarations, blocks):
    """Is this metric keyword inside a declared outlook block?"""
    return any(start <= position <= end for start, end in blocks)


def _metric_from_row_label(label: str):
    """A table row label to a canonical metric identity, or None.

    Reuses `_metric_keyword_hits` -- the same longest-match resolution the
    sentence path uses -- so a row reading "Subscription revenue" and a
    sentence saying "subscription revenue" cannot be classified differently.
    A label that matches nothing returns None and the row is left alone;
    inventing an identity for an unrecognised label is how a component
    became consolidated revenue in the first place.
    """
    hits = _metric_keyword_hits(label or "")
    if not hits:
        return None
    # The hit covering the most of the label. A row label is short and names
    # one metric; the widest match is the most specific reading of it.
    _start, _end, entry = max(hits, key=lambda h: h[1] - h[0])
    return entry


def _table_guidance_metrics(text, symbol, accession, filed, existing):
    """Sections 10-11: guidance recovered from a TABLE, keyed by row+column.

    Only ADDS. Where the sentence path already produced an entry for a
    metric, that entry stands -- it carries the surrounding prose, the
    bound type and the supersession context that a bare cell does not, and
    overwriting it would trade richer evidence for a tidier parse.

    What this recovers is everything the sentence path could not see: the
    second column, the rows below the first, and the GAAP/non-GAAP pair.
    """
    from finance import guidance_tables

    table = guidance_tables.parse_guidance_table(text)
    if not table.ok:
        return {}, []

    recovered, warnings = {}, []
    for cell in table.cells:
        entry = _metric_from_row_label(cell.metric_label)
        if entry is None:
            continue
        name, unit, is_growth, scope, _basis_hint, _pattern = entry

        # A percentage cell under a metric measured in currency (or the
        # reverse) means the row and the column disagree about what this
        # number is. Refuse rather than coerce -- section 11's whole point.
        if is_growth and not cell.is_percent:
            continue
        basis = _basis_for_label(cell.metric_label)
        if not is_growth and cell.is_percent:
            # ...unless the ROW LABEL itself states the denominator ("Non-GAAP
            # operating income (% of revenue)"), in which case the cell is a
            # margin and has a margin identity. Same rule as the sentence
            # path, applied to the row label because that is where a table
            # states what its numbers are.
            resolved = resolve_percentage_identity(name, unit, basis,
                                                   cell.metric_label)
            if resolved is None:
                continue
            name, unit = resolved

        period = parse_guidance_period(cell.period_label, filed)
        if period is None:
            continue
        key = name
        if key in existing or key in recovered:
            continue

        low = cell.low / 100.0 if cell.is_percent else cell.low
        high = cell.high / 100.0 if cell.is_percent else cell.high
        evidence_id = f"dcf.guidance.{name}.current"
        target_type = classify_target_type(period, None)
        recovered[key] = GuidanceMetric(
            name=name, low=low, high=high, unit=unit,
            basis=basis,
            fiscal_year=period.fiscal_year,
            evidence_id=evidence_id,
            source_excerpt=f"{cell.metric_label} | {cell.period_label} | {cell.raw}",
            scale=cell.scale,
            guidance_id=_guidance_id(symbol, accession, name, period.label),
            issued_at=filed,
            fiscal_period=period.label,
            period_type=period.period_type,
            issued_with_reporting_period=None,
            target_period_type=target_type,
            forward_kind=classify_forward_information(cell.metric_label, target_type),
            scope=scope,
            source_accession=accession,
            source_evidence_ids=(evidence_id,),
            bound_type=(GuidanceBound.RANGE if cell.low != cell.high
                        else GuidanceBound.APPROXIMATELY),
            status=GuidanceStatus.CURRENT,
            status_reason=None,
            # A guidance TABLE qualifies its own rows: the header states that
            # the columns are periods being guided, which is a stronger
            # statement than any word in the prose around it.
            prospective_evidence=(
                f"Guidance table row {cell.metric_label!r} under the column "
                f"{cell.period_label!r}"))

    if table.skipped_rows:
        warnings.append(
            f"{len(table.skipped_rows)} guidance table row(s) did not align with the "
            "column headers and were not read; their values are not reported rather "
            "than being attached to a period that may be wrong.")
    return recovered, warnings


def _basis_for_label(label: str) -> str:
    """GAAP unless the row says otherwise. Adjacent rows in one table differ
    only by this word, and merging them loses the distinction entirely."""
    lowered = (label or "").lower()
    if "non-gaap" in lowered or "non gaap" in lowered or "adjusted" in lowered:
        return BASIS_ADJUSTED
    return BASIS_GAAP


# How much of the qualifying statement is kept. Enough for a reader to judge
# the verdict, short enough that a release's whole paragraph does not travel
# with every metric.
_PROSPECTIVE_EVIDENCE_CHARS = 180


def _prospective_evidence(text: str, keyword_start: int, forward_marker,
                          in_outlook_block: bool) -> str:
    """The text that established this figure as forward-looking.

    Prefers the sentence carrying the forward-looking verb. An outlook block
    qualifies its own rows, so where no marker sits near the figure the block
    heading is what is recorded -- naming the actual reason rather than a
    nearby word that happens to look like one.
    """
    if forward_marker is not None:
        window_start = max(0, keyword_start - _WINDOW)
        absolute = window_start + forward_marker.start()
        start = max(0, absolute - 60)
        return " ".join(text[start:absolute + _PROSPECTIVE_EVIDENCE_CHARS].split())
    if in_outlook_block:
        heading = None
        for match in _OUTLOOK_BLOCK_RE.finditer(text):
            if match.start() <= keyword_start:
                heading = match
        if heading is not None:
            return " ".join(
                text[heading.start():
                     heading.start() + _PROSPECTIVE_EVIDENCE_CHARS].split())
    return ""


def extract_guidance_from_text(text: str, symbol: str, accession: str, document: str,
                               filed: str, expected_fiscal_year: Optional[int] = None
                               ) -> GuidanceRelease:
    """Pattern-match guidance out of one earnings release. No interpretation.

    `expected_fiscal_year` is now a HINT used only to label the release, not a
    filter. Filtering on it is what silently discarded every NVIDIA value:
    the caller passed the calendar year and NVDA guides its own fiscal year,
    which is a year ahead. Each figure now carries the period IT names
    (section 8), and plausibility is checked against the FILING DATE
    (`_period_is_plausible`), which rejects prior-year actuals without
    assuming a calendar fiscal year.

    Every rejection is recorded in `warnings` so a reader can tell "this
    company published no guidance" from "a value was found and refused".
    """
    # Keyed by IDENTITY while extracting. Keying by NAME here is what made a
    # release's second horizon unreachable: the full-year outlook two
    # sentences below the next-quarter one hit `if name in metrics` and was
    # dropped before anything could weigh it.
    by_identity: Dict[tuple, GuidanceMetric] = {}
    warnings: List[str] = []
    if not text:
        return GuidanceRelease(symbol=symbol, fiscal_year=expected_fiscal_year,
                               accession=accession, document=document, filed=filed,
                               warnings=("The filing document was empty.",))

    hits = _metric_keyword_hits(text)
    boundaries = [start for start, _end, _entry in hits]
    declarations = _period_declarations(text)
    outlook_blocks = _outlook_block_spans(text)
    reported_blocks = _reported_block_spans(text)
    reporting_period = detect_reporting_period(text)

    for index, (start, end, entry) in enumerate(hits):
        name, unit, require_percent, scope, forced_basis, _regex = entry

        # The window stops at the earlier of: a clause boundary, the next
        # metric keyword, or `_WINDOW` characters. All three matter — the
        # next-keyword cut is what keeps a table row from reaching the row
        # below it even when neither carries punctuation.
        next_keyword = next((b for b in boundaries if b > end), None)
        limit = _WINDOW if next_keyword is None else min(_WINDOW, next_keyword - end)
        if limit <= 0:
            continue
        window = _clause_window(text, end, limit)
        context = text[max(0, start - _WINDOW):end + _WINDOW]

        # A row inside an OUTLOOK TABLE is forward-looking even when no
        # forward-looking word sits within reach of it. Live outlook tables
        # put the header once and then list seven metrics; requiring a marker
        # within 110 characters of each row silently dropped the rows in the
        # middle -- one release guided sales, gross margin, operating
        # expenses, other income, tax rate, EPS and share count, and only tax
        # rate and EPS survived. The block header is the marker for its own
        # rows, exactly as it is for their period.
        in_outlook_block = _inside_outlook_block(start, declarations, outlook_blocks)
        forward_marker = _FORWARD_MARKERS.search(context)
        if not forward_marker and not in_outlook_block:
            continue

        # A figure inside a block the release itself labels as reported
        # results is a reported actual. An outlook heading re-qualifies it --
        # `_reported_block_spans` already ends each block at the next one --
        # and nothing else does, because the forward-looking words in the
        # prose around a statements table are about the company, not about
        # the number in row four.
        if _inside_reported_block(start, reported_blocks) and not in_outlook_block:
            warnings.append(
                f"{SOURCE_NOT_PROSPECTIVE}: ignored a {name} value inside a "
                "reported-results section. The release presents that block as results "
                "already reported, so a figure in it is not a forward-looking statement "
                "however the surrounding prose reads.")
            continue

        prospective_evidence = _prospective_evidence(
            text, start, forward_marker, in_outlook_block)

        lookbehind = text[max(0, start - _QUALIFIER_LOOKBEHIND):start]
        found = _find_range(window, require_percent, lookbehind=lookbehind)
        if found is None and unit in _ABSOLUTE_UNITS:
            # An absolute metric normally refuses a percentage outright, and
            # that refusal is right whenever the percentage means something
            # this vocabulary cannot name. But a release that states the
            # DENOMINATOR is not ambiguous: "operating income of 21% OF
            # PROJECTED REVENUE" is a margin, and dropping it loses real
            # guidance. The percentage is accepted only when the denominator
            # is present, and the identity is corrected below.
            if percentage_denominator_stated(context):
                found = _find_range(window, True, lookbehind=lookbehind)
            if found is None and _find_range(window, True, lookbehind=lookbehind):
                warnings.append(
                    f"{PERCENTAGE_DENOMINATOR_UNRESOLVED}: ignored a {name} value. The "
                    f"figure is a percentage, {name} is measured as an absolute amount, "
                    "and the text does not say what the percentage is a percentage OF, "
                    "so no identity could be assigned to it.")
                continue
        if found is None:
            found = _dual_basis_value(text, end, name, require_percent)
            if found is None:
                continue
        low, high, is_percent, matched, bound_type = found

        # Where the matched value actually sits, so the period rules can look
        # at the text immediately after THE NUMBER rather than after the
        # metric name.
        # The dual-basis pattern searches BACKWARD from the keyword, so its
        # match can begin before `end`; search from there or the period rules
        # below would measure from the wrong place.
        value_start = text.find(matched, max(0, end - 140),
                                end + limit + len(matched) + 8)
        if value_start < 0:
            value_start = end
        value_end = value_start + len(matched)

        period = resolve_guidance_period(text, start, value_start, value_end, filed,
                                         declarations)
        if period is None:
            warnings.append(
                f"Ignored a {name} value ({matched}) because no fiscal period could be "
                "identified near it.")
            continue
        if not _period_is_plausible(period, filed):
            warnings.append(
                f"Ignored a {name} value ({matched}) for {period.label}: a release filed "
                f"{filed} cannot be stating forward guidance for that period.")
            continue

        # GAAP vs adjusted is decided from the modifier immediately before
        # the metric name, never from the wider paragraph — see
        # `_BASIS_LOOKBEHIND`. A metric whose own name carries the basis
        # (adjusted EBITDA, adjusted EPS) forces it.
        modifier = text[max(0, start - _BASIS_LOOKBEHIND):start]
        basis = forced_basis or (
            BASIS_ADJUSTED if _ADJUSTED_MARKERS.search(modifier) else BASIS_GAAP)
        if forced_basis is None and basis == BASIS_ADJUSTED:
            # "adjusted" directly modifies this mention, so it is the adjusted
            # measure. Storing it under the GAAP name would mix the two bases
            # (section 6). The adjusted variant has its own pattern and will
            # match the same text.
            adjusted_name = _ADJUSTED_TWIN.get(name)
            if adjusted_name and adjusted_name in _METRIC_BY_NAME:
                continue
            basis = BASIS_ADJUSTED

        status = GuidanceStatus.CURRENT
        status_reason = None
        if _WITHDRAWAL_MARKERS.search(context):
            status = GuidanceStatus.WITHDRAWN
            status_reason = (
                "The text around this figure states that the outlook was withdrawn or "
                "suspended, so it is retained for history and never used as current guidance.")

        # Units and denominator participate in identity. A percentage found
        # under a metric whose taxonomy unit is an absolute amount is not
        # that metric: it is either that metric's MARGIN, when the text
        # establishes revenue as the denominator, or it is unidentifiable.
        # Storing it under the absolute name is what put 0.21 into
        # `operating_income` on a live run.
        if is_percent:
            resolved = resolve_percentage_identity(name, unit, basis, context)
            if resolved is None:
                warnings.append(
                    f"{PERCENTAGE_DENOMINATOR_UNRESOLVED}: ignored a {name} value "
                    f"({matched}). It is a percentage, {name} is measured as an absolute "
                    "amount, and no denominator could be established, so the figure was "
                    "not assigned an identity.")
                continue
            name, unit = resolved

        scale_match = _SCALE_RE.search(context)
        if is_percent:
            low, high = low / 100.0, high / 100.0

        evidence_id = f"dcf.guidance.{name}.current"
        candidate = GuidanceMetric(
            name=name, low=low, high=high, unit=unit, basis=basis,
            fiscal_year=period.fiscal_year,
            evidence_id=evidence_id,
            source_excerpt=_excerpt(text, start, end + limit),
            scale=(scale_match.group(1).lower() if scale_match and
                   unit == GuidanceUnit.CURRENCY else None),
            guidance_id=_guidance_id(symbol, accession, name, period.label),
            issued_at=filed,
            fiscal_period=period.label,
            period_type=period.period_type,
            issued_with_reporting_period=reporting_period,
            target_period_type=classify_target_type(period, reporting_period),
            forward_kind=classify_forward_information(
                _excerpt(text, start, end + limit),
                classify_target_type(period, reporting_period)),
            scope=scope,
            source_accession=accession,
            source_evidence_ids=(evidence_id,),
            bound_type=bound_type,
            status=status,
            status_reason=status_reason,
            prospective_evidence=prospective_evidence)
        # One statement per IDENTITY. A repeat of the same metric for the
        # same period and basis is the same statement said twice; a different
        # period is a different statement and keeps its own place.
        by_identity.setdefault(guidance_identity(candidate), candidate)

    metrics = name_keyed_view(by_identity.values())

    # Sections 10-11. Whatever the sentence path could not see, recovered
    # from the table with its row and column identity intact. Additive only:
    # an entry the sentence path already produced stands.
    table_metrics, table_warnings = _table_guidance_metrics(
        text, symbol, accession, filed, metrics)
    for metric in table_metrics.values():
        by_identity.setdefault(guidance_identity(metric), metric)
    warnings.extend(table_warnings)

    statements = tuple(by_identity.values())
    return GuidanceRelease(symbol=symbol, fiscal_year=expected_fiscal_year,
                           accession=accession, document=document, filed=filed,
                           metrics=name_keyed_view(statements),
                           all_metrics=statements,
                           warnings=tuple(warnings))


# GAAP metric -> its adjusted counterpart, for the "an 'adjusted' modifier
# means the adjusted metric owns this text" rule above.
_ADJUSTED_TWIN = {
    GuidanceMetricName.EPS: GuidanceMetricName.ADJUSTED_EPS,
    GuidanceMetricName.GROSS_MARGIN: GuidanceMetricName.ADJUSTED_GROSS_MARGIN,
    GuidanceMetricName.OPERATING_MARGIN: GuidanceMetricName.ADJUSTED_OPERATING_MARGIN,
    GuidanceMetricName.OPERATING_EXPENSES: GuidanceMetricName.ADJUSTED_OPERATING_EXPENSES,
    GuidanceMetricName.EBITDA: GuidanceMetricName.ADJUSTED_EBITDA,
    GuidanceMetricName.EBITDA_GROWTH: GuidanceMetricName.ADJUSTED_EBITDA_GROWTH,
}

# Which of a "GAAP and non-GAAP X are expected to be A and B" pair belongs to
# which metric. The GAAP figure is stated first in every release reviewed.
_DUAL_BASIS_ADJUSTED_METRICS = frozenset({
    GuidanceMetricName.ADJUSTED_GROSS_MARGIN,
    GuidanceMetricName.ADJUSTED_OPERATING_MARGIN,
    GuidanceMetricName.ADJUSTED_OPERATING_EXPENSES,
})


def _dual_basis_value(text: str, keyword_end: int, name: str, require_percent: bool
                      ) -> Optional[Tuple[float, float, bool, str]]:
    """Handle "GAAP and non-GAAP X are expected to be A and B, respectively".

    Two POINT values for two bases, which must never be read as one A-to-B
    range (that would invent a spread the company did not state and mix the
    two bases into one figure). Each side becomes its own metric's value,
    stored as a degenerate low==high range so the record shape stays uniform.
    """
    window = text[max(0, keyword_end - 120):keyword_end + _WINDOW]
    match = _DUAL_BASIS_PATTERN.search(window)
    if match is None:
        return None
    gaap_value = _to_number(match.group(1))
    adjusted_value = _to_number(match.group(4))
    if gaap_value is None or adjusted_value is None:
        return None
    is_percent = bool(match.group(2) or match.group(5))
    if require_percent != is_percent:
        return None
    value = adjusted_value if name in _DUAL_BASIS_ADJUSTED_METRICS else gaap_value
    return value, value, is_percent, match.group(0).strip(), GuidanceBound.APPROXIMATELY


def _excerpt(text: str, start: int, end: int, pad: int = 40) -> str:
    lo = max(0, start - pad)
    hi = min(len(text), end)
    return text[lo:hi].strip()


# ---------------------------------------------------------------------------
# Section 9 — selecting CURRENT guidance
# ---------------------------------------------------------------------------


# Which horizon wins the ONE slot the name-keyed `metrics` view has.
#
# The assumption builder reads that view and builds an ANNUAL forecast, so an
# annual statement is the one that answers its question. A shorter horizon is
# not discarded -- it stays in `all_metrics` and reaches the research evidence
# and the report -- it simply does not occupy the slot a twelve-month
# consumer reads.
_HORIZON_PREFERENCE = {
    GuidanceTargetType.CURRENT_FISCAL_YEAR: 0,
    GuidanceTargetType.NEXT_FISCAL_YEAR: 1,
    GuidanceTargetType.MULTI_YEAR: 2,
    GuidanceTargetType.NEXT_QUARTER: 3,
    GuidanceTargetType.OTHER: 4,
}


def preferred_for_assumption(statements: Sequence[GuidanceMetric]
                             ) -> Optional[GuidanceMetric]:
    """The statement a twelve-month consumer should read, among several.

    Longest applicable horizon first, then the newest issue date. Returns
    None for an empty sequence rather than raising, because "this metric was
    not guided" is an ordinary answer.
    """
    def rank(metric):
        # Longest applicable horizon first; then, WITHIN a horizon, the
        # newest statement, because that is what supersession means. Sorting
        # the issue date ascending here picked a company's expired
        # next-quarter guidance over the one it had just published.
        return (_HORIZON_PREFERENCE.get(metric.target_period_type, 5),
                _descending(metric.issued_at),
                _descending(metric.fiscal_period))

    ranked = sorted(statements, key=rank)
    return ranked[0] if ranked else None


class _descending:
    """Sort one field descending inside an otherwise ascending key."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value or ""

    def __lt__(self, other):
        return self.value > other.value

    def __eq__(self, other):
        return self.value == other.value


def name_keyed_view(statements: Sequence[GuidanceMetric]) -> Dict[str, GuidanceMetric]:
    """The one-per-name projection of a multi-horizon statement set."""
    by_name: Dict[str, List[GuidanceMetric]] = {}
    for metric in statements:
        by_name.setdefault(metric.name, []).append(metric)
    return {name: preferred_for_assumption(group) for name, group in by_name.items()}


def select_current_guidance(releases: Sequence[GuidanceRelease],
                            fiscal_year: Optional[int] = None,
                            as_of: Optional[str] = None
                            ) -> Tuple[Optional[GuidanceRelease], List[GuidanceRelease]]:
    """The current guidance for each metric, plus every release it supersedes.

    PER-METRIC, not per-release, and that is the Phase H.6 correction. The
    old rule picked the newest release that had ANY metrics and discarded the
    rest. On AT&T that chose the January release — whose "3% to 4%" was
    misread as revenue growth — while the July release, which reiterates the
    same outlook, contributed nothing. Selecting per metric means the newest
    STATEMENT OF EACH METRIC wins, which is what supersession actually means.

    Withdrawn guidance is never selected as current (section 9). Guidance for
    a period that has already ended is marked EXPIRED and likewise not
    selected. Both are still returned among the superseded releases, because
    "management withdrew its outlook" is itself material.
    """
    usable = [r for r in releases if r.metrics or r.all_metrics]
    if not usable:
        return None, []
    ordered = sorted(usable, key=lambda r: (r.filed or "", r.accession or ""), reverse=True)

    # Keyed by IDENTITY -- metric, target period, target period type, basis --
    # so a full-year outlook and a next-quarter outlook for the same metric
    # are two statements that both survive, which is what section 11 has
    # always said they are.
    current_by_identity: Dict[tuple, GuidanceMetric] = {}
    supplier_of: Dict[tuple, GuidanceRelease] = {}
    for release in ordered:
        for metric in (release.all_metrics or tuple(release.metrics.values())):
            name = metric.name
            # `fiscal_year` is the EARLIEST period still relevant, not an
            # exact match. Guidance for a LATER fiscal year is the whole
            # point of asking — NVIDIA's May-2026 release guides fiscal 2027
            # — and requiring equality against the calendar year is what
            # discarded it. Only guidance for an already-past period is
            # dropped here.
            if fiscal_year is not None and metric.fiscal_year is not None \
                    and metric.fiscal_year < fiscal_year:
                continue
            if metric.status == GuidanceStatus.WITHDRAWN:
                continue
            if _is_expired(metric, as_of):
                continue
            identity = guidance_identity(metric)
            if identity in current_by_identity:
                continue
            current_by_identity[identity] = metric
            supplier_of[identity] = release

    if not current_by_identity:
        return None, list(ordered)

    statements = tuple(current_by_identity.values())
    current_metrics = name_keyed_view(statements)

    # The release that supplied the most current metrics names the result —
    # a guidance record has to cite ONE primary filing, and every individual
    # metric still carries its own `source_accession`. Counted by release
    # IDENTITY rather than accession number: two releases can share an
    # accession in a synthetic fixture, and the supersession list must not
    # silently collapse when they do.
    counts: Dict[int, int] = {}
    for release in supplier_of.values():
        counts[id(release)] = counts.get(id(release), 0) + 1
    primary = max(ordered, key=lambda r: (counts.get(id(r), 0),
                                          r.filed or "", r.accession or ""))

    combined = GuidanceRelease(
        symbol=primary.symbol,
        fiscal_year=fiscal_year if fiscal_year is not None else primary.fiscal_year,
        accession=primary.accession,
        document=primary.document,
        filed=primary.filed,
        metrics=current_metrics,
        all_metrics=statements,
        warnings=primary.warnings)
    superseded = [r for r in ordered if r is not primary]
    return combined, superseded


def _is_expired(metric: GuidanceMetric, as_of: Optional[str]) -> bool:
    """A fiscal year that ended before the valuation date is not an outlook.

    Only applied to ANNUAL and MULTI-YEAR periods, and only on the calendar
    year, because the fiscal-year-end date is not knowable from the release
    text. A quarterly outlook is never expired here: the whole point of
    next-quarter guidance is that it is the freshest forward statement there
    is, and it is superseded by the next release rather than by the calendar.
    """
    if not as_of or len(as_of) < 4:
        return False
    if metric.period_type == GuidancePeriodType.QUARTER:
        return False
    try:
        current_year = int(as_of[:4])
    except ValueError:
        return False
    return bool(metric.fiscal_year) and metric.fiscal_year < current_year


# ---------------------------------------------------------------------------
# Section 11 — validation
# ---------------------------------------------------------------------------

REQUIRED_GUIDANCE_FIELDS = ("name", "fiscal_period", "units", "basis", "source_accession")


def validate_guidance_metric(metric: GuidanceMetric) -> List[str]:
    """Section 11's checklist, as a list of problems (empty means valid).

    Every numeric guidance value must carry a source document, an evidence
    id, a metric identity, a fiscal period, units, a low/high (or an exact
    value expressed as low==high) and a GAAP/adjusted/company-defined basis.
    A value failing any of these is not published as guidance.
    """
    problems: List[str] = []
    if not metric.name or metric.name not in _METRIC_BY_NAME:
        problems.append(f"metric identity {metric.name!r} is not in the reviewed taxonomy")
    if not metric.fiscal_period:
        problems.append("no fiscal period")
    if metric.unit not in (GuidanceUnit.RATIO, GuidanceUnit.CURRENCY,
                           GuidanceUnit.CURRENCY_PER_SHARE, GuidanceUnit.SHARES):
        problems.append(f"unrecognized units {metric.unit!r}")
    if metric.basis not in (BASIS_GAAP, BASIS_ADJUSTED, BASIS_COMPANY_DEFINED, BASIS_NONE):
        problems.append(f"unrecognized basis {metric.basis!r}")
    if metric.low is None or metric.high is None or metric.high < metric.low:
        problems.append("no usable low/high value")
    if not metric.source_accession:
        problems.append("no source filing")
    if not metric.source_evidence_ids:
        problems.append("no evidence id")
    expected_unit = _METRIC_BY_NAME.get(metric.name, (None, None))[1]
    if expected_unit is not None and metric.unit != expected_unit:
        problems.append(f"units {metric.unit!r} do not match the {metric.name!r} taxonomy "
                        f"entry ({expected_unit!r})")
    # Prospective semantics. A guidance item is a statement about a period
    # that has not happened yet, and every one of these is part of saying so:
    # without the target period type nothing downstream can tell a
    # next-quarter outlook from a multi-year framework, and without the
    # qualifying text there is no way to check that the figure was forward-
    # looking at all rather than a reported actual standing near the word
    # "expects".
    if metric.target_period_type not in GuidanceTargetType.ALL:
        problems.append(f"unrecognized target period type {metric.target_period_type!r}")
    if not metric.prospective_evidence:
        problems.append(
            "no prospective evidence: nothing in the source establishes this figure as "
            "forward-looking, so it is a reported value rather than guidance")
    return problems


# ---------------------------------------------------------------------------
# Section 8 — guidance COVERAGE, not merely guidance presence
# ---------------------------------------------------------------------------


class GuidanceCoverage:
    """How much of what a valuation needs the guidance actually supplies.

    "Some guidance was extracted" and "guidance is covered" are different
    claims, and only the first was ever being made. A live release guided
    seven metrics -- sales, gross margin, operating expenses, other income,
    tax rate, EPS and share count -- of which two were extracted, and the
    report said guidance was available without qualification. A reader had no
    way to tell that the two most valuation-relevant rows were missing.
    """

    COMPLETE_FOR_RELEVANT_METRICS = "COMPLETE_FOR_RELEVANT_METRICS"
    PARTIAL = "PARTIAL"
    MINIMAL = "MINIMAL"
    UNAVAILABLE = "UNAVAILABLE"
    ALL = (COMPLETE_FOR_RELEVANT_METRICS, PARTIAL, MINIMAL, UNAVAILABLE)


# The metrics a DCF actually consumes. Coverage is scored against THESE
# rather than against everything a company might guide: a buyback target is
# real guidance and contributes nothing to a forecast of operating cash flow.
VALUATION_RELEVANT_METRICS = (
    (GuidanceMetricName.CONSOLIDATED_REVENUE, GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH,
     GuidanceMetricName.SERVICE_REVENUE_GROWTH, GuidanceMetricName.PRODUCT_REVENUE_GROWTH),
    (GuidanceMetricName.OPERATING_MARGIN, GuidanceMetricName.ADJUSTED_OPERATING_MARGIN,
     GuidanceMetricName.GROSS_MARGIN, GuidanceMetricName.ADJUSTED_GROSS_MARGIN,
     GuidanceMetricName.ADJUSTED_EBITDA_GROWTH, GuidanceMetricName.EBITDA_GROWTH),
    (GuidanceMetricName.OPERATING_EXPENSES, GuidanceMetricName.ADJUSTED_OPERATING_EXPENSES),
    (GuidanceMetricName.EPS, GuidanceMetricName.ADJUSTED_EPS),
    (GuidanceMetricName.TAX_RATE,),
    (GuidanceMetricName.CAPEX, GuidanceMetricName.FREE_CASH_FLOW,
     GuidanceMetricName.OPERATING_CASH_FLOW),
)

# Groups whose absence most changes a forecast. Revenue and profitability are
# the two the DCF cannot proceed without some view of.
_CORE_GROUP_INDEXES = (0, 1)


# ---------------------------------------------------------------------------
# Phase H.11, sections 11-15 — coverage is per METRIC, not one verdict
# ---------------------------------------------------------------------------

class GuidanceMetricStatus:
    """What is known about ONE guided metric."""

    CURRENT = "CURRENT"
    SUPERSEDED = "SUPERSEDED"
    WITHDRAWN = "WITHDRAWN"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_SEARCHED = "NOT_SEARCHED"
    INCOMPATIBLE = "INCOMPATIBLE"

    ALL = (CURRENT, SUPERSEDED, WITHDRAWN, UNAVAILABLE, NOT_SEARCHED, INCOMPATIBLE)


class GuidanceAbsence:
    """Section 14: four different reasons a guidance figure is missing.

    Collapsing them into "none extracted" makes an issuer that guides capital
    expenditure but not revenue indistinguishable from one that guides
    nothing, and both indistinguishable from a parser that failed. They call
    for different responses, so they are different values.
    """

    NO_GUIDANCE_EXISTS = "NO_GUIDANCE_EXISTS"
    NO_REVENUE_GUIDANCE = "NO_REVENUE_GUIDANCE"
    GUIDANCE_EXTRACTION_FAILED = "GUIDANCE_EXTRACTION_FAILED"
    PARTIAL_GUIDANCE_ONLY = "PARTIAL_GUIDANCE_ONLY"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# The per-metric rows of the matrix, each naming the guidance metric names
# that satisfy it. Rows are the vocabulary a reader and the assumption
# builder both think in; the names are the extractor's vocabulary.
COVERAGE_ROWS = (
    ("revenue", (GuidanceMetricName.CONSOLIDATED_REVENUE,)),
    ("revenue_growth", (GuidanceMetricName.CONSOLIDATED_REVENUE_GROWTH,
                        GuidanceMetricName.SERVICE_REVENUE_GROWTH,
                        GuidanceMetricName.PRODUCT_REVENUE_GROWTH)),
    ("gross_margin", (GuidanceMetricName.GROSS_MARGIN,
                      GuidanceMetricName.ADJUSTED_GROSS_MARGIN)),
    ("operating_margin", (GuidanceMetricName.OPERATING_MARGIN,
                          GuidanceMetricName.ADJUSTED_OPERATING_MARGIN)),
    ("operating_expenses", (GuidanceMetricName.OPERATING_EXPENSES,
                            GuidanceMetricName.ADJUSTED_OPERATING_EXPENSES)),
    ("eps", (GuidanceMetricName.EPS, GuidanceMetricName.ADJUSTED_EPS)),
    ("tax_rate", (GuidanceMetricName.TAX_RATE,)),
    ("capex", (GuidanceMetricName.CAPEX,)),
    ("operating_cash_flow", (GuidanceMetricName.OPERATING_CASH_FLOW,)),
    ("free_cash_flow", (GuidanceMetricName.FREE_CASH_FLOW,)),
    ("shares", (GuidanceMetricName.SHARE_COUNT,)),
    ("leverage", (GuidanceMetricName.NET_LEVERAGE_TARGET,)),
)

# Section 13: the rows a DCF assumption actually consumes. Earnings per share
# is real guidance and does not feed this model's inputs, so it is scored in
# the overall matrix and not here.
DCF_RELEVANT_ROWS = ("revenue", "revenue_growth", "gross_margin", "operating_margin",
                     "tax_rate", "capex", "operating_cash_flow", "free_cash_flow")


class DcfGuidanceCoverage:
    COMPLETE_FOR_DCF_RELEVANT_METRICS = "COMPLETE_FOR_DCF_RELEVANT_METRICS"
    PARTIAL = "PARTIAL"
    MINIMAL = "MINIMAL"
    UNAVAILABLE = "UNAVAILABLE"


def _row_status(present_names, metrics, superseded_names) -> str:
    """The status of one row, from what the extractor actually produced."""
    for name in present_names:
        entry = (metrics or {}).get(name)
        if isinstance(entry, dict) and entry.get("low") is not None:
            status = (entry.get("status") or "CURRENT").upper()
            if status in GuidanceMetricStatus.ALL:
                return status
            return GuidanceMetricStatus.CURRENT
    if any(name in (superseded_names or set()) for name in present_names):
        return GuidanceMetricStatus.SUPERSEDED
    return GuidanceMetricStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# Parts 9-10 — supersession resolved per metric AND target period
# ---------------------------------------------------------------------------
#
# "Latest filing wins" is the wrong rule and produces two distinct errors.
# It drops a still-current full-year outlook because a later release guided
# only the next quarter, and it keeps a revised figure and the figure it
# revised side by side when both arrive in one document. Guidance is
# superseded by NEWER GUIDANCE FOR THE SAME THING, and "the same thing" is
# the metric and the period it targets -- not the document it arrived in.


# Target-period types that differ only by HOW FAR AHEAD the target was when
# the statement was made. `classify_target_type` derives CURRENT vs NEXT
# fiscal year from the RELEASE's own reporting period, so one company
# restating one outlook produces two different types:
#
#     March release   "For fiscal year 2027 ..."   NEXT_FISCAL_YEAR
#     June release    "For fiscal year 2027 ..."   CURRENT_FISCAL_YEAR
#
# That is correct information about each STATEMENT and wrong information for
# IDENTITY. Section 11 says issue period and target period are different
# fields; the same rule read backwards is that information about the issue
# date must not leak into the identity of the target. Collapsed to one class
# so the two are recognised as one statement and the older is superseded.
#
# MULTI_YEAR and OTHER are NOT collapsed into it: a long-term framework and
# an outlook for a named year can quote the same year and remain different
# forward statements. NEXT_QUARTER is left alone because a quarter's target
# period already differs ("Q1 FY2027" against "FY2027").
_NAMED_FISCAL_YEAR_TYPES = frozenset({
    GuidanceTargetType.CURRENT_FISCAL_YEAR,
    GuidanceTargetType.NEXT_FISCAL_YEAR,
})
_NAMED_FISCAL_YEAR = "NAMED_FISCAL_YEAR"


def identity_horizon(target_period_type: Optional[str]) -> Optional[str]:
    """The horizon class an identity uses, ignoring distance from the issuer."""
    if target_period_type in _NAMED_FISCAL_YEAR_TYPES:
        return _NAMED_FISCAL_YEAR
    return target_period_type


def guidance_identity(entry) -> tuple:
    """What makes two guidance items the same statement.

    The metric, the period it targets, the KIND of horizon and the basis.

    Basis is part of it: a GAAP margin outlook and an adjusted one for the
    same period are two statements, and neither supersedes the other.
    Merging them would silently drop whichever arrived first.

    The horizon is the CLASS, not the distance -- see `identity_horizon`.
    Two releases restating one full-year outlook are one statement however
    each measured its own distance from it.
    """
    def read(name, default=None):
        if isinstance(entry, dict):
            return entry.get(name, default)
        return getattr(entry, name, default)

    return (read("name") or read("metric_id"),
            read("fiscal_period") or read("target_period"),
            identity_horizon(read("target_period_type")),
            read("basis"))


def resolve_guidance_status(entries) -> list:
    """Assign CURRENT / SUPERSEDED to a set of guidance items.

    Within one identity the newest issue date is CURRENT and everything
    older is SUPERSEDED. Across identities nothing is touched: a long-term
    framework is not superseded by this year's outlook, and a Q3 figure is
    not superseded by an FY figure -- they are different forward statements
    about different periods, and section 10 is explicit that both may stand.

    WITHDRAWN survives untouched and never becomes CURRENT again. A company
    that pulled its guidance has said something, and the newest remaining
    figure is not a replacement for it.

    Returns the entries with `status` and `status_reason` set. Input order
    is preserved so a caller can zip results back to its own records.

    A dict is updated in place; a frozen `GuidanceMetric` is REPLACED by a
    copy carrying the new status, because a validated statement must not be
    mutable after the fact. The dataclass branch used to `setattr` and raise,
    which meant this resolver could only ever be handed dicts -- fine while
    the only statements it saw were dicts, and wrong now that `all_metrics`
    carries the objects themselves.
    """
    entries = list(entries or [])
    resolved = {}          # id(original) -> replacement, for frozen entries

    def read(entry, name, default=None):
        if isinstance(entry, dict):
            return entry.get(name, default)
        return getattr(entry, name, default)

    def write(entry, name, value):
        if isinstance(entry, dict):
            entry[name] = value
            return
        current = resolved.get(id(entry), entry)
        updates = {"status": read(current, "status"),
                   "status_reason": read(current, "status_reason"),
                   name: value}
        resolved[id(entry)] = current.with_status(updates["status"],
                                                  updates["status_reason"])             if name == "status_reason" else current.with_status(
                value, read(current, "status_reason"))

    groups = {}
    for entry in entries:
        if read(entry, "status") == GuidanceStatus.WITHDRAWN:
            continue
        groups.setdefault(guidance_identity(entry), []).append(entry)

    for identity, group in groups.items():
        if len(group) == 1:
            write(group[0], "status", GuidanceStatus.CURRENT)
            continue
        # Newest issue date wins within the identity. A missing date sorts
        # oldest: an item that cannot say when it was issued cannot claim to
        # be the most recent one.
        ordered = sorted(group, key=lambda e: (read(e, "issued_at") or ""))
        newest = ordered[-1]
        for entry in ordered[:-1]:
            write(entry, "status", GuidanceStatus.SUPERSEDED)
            write(entry, "status_reason",
                  f"Superseded by guidance for the same metric and target period issued "
                  f"{read(newest, 'issued_at') or 'later'}.")
        write(newest, "status", GuidanceStatus.CURRENT)
        write(newest, "status_reason", None)
    return [resolved.get(id(entry), entry) for entry in entries]


def current_guidance_only(entries) -> list:
    """The items that may be presented as this company's current outlook."""
    def read(entry, name):
        if isinstance(entry, dict):
            return entry.get(name)
        return getattr(entry, name, None)

    return [e for e in resolve_guidance_status(entries)
            if read(e, "status") == GuidanceStatus.CURRENT]


def build_guidance_matrix(metrics, superseded=None, releases_examined=None,
                          extraction_failed=False, all_metrics=None) -> dict:
    """Sections 11-14: a status for every row, and two summary verdicts.

    The live failure this replaces: a report said "management guidance:
    unavailable" for an issuer that had published current capital-expenditure
    guidance. One missing row -- revenue -- had been allowed to speak for the
    whole matrix, and the assumption builder discarded the capital-expenditure
    figure along with it.

    `releases_examined` separates "we looked and found nothing" from "we never
    looked", which is the difference between UNAVAILABLE and NOT_SEARCHED.
    """
    # Coverage counts a metric as guided when ANY current horizon states it.
    # Reading only the name-keyed view would undercount an issuer that guided
    # a metric for a quarter and a different metric for the year, since that
    # view holds one statement per name and the matrix asks a per-name
    # question about the whole release.
    metrics = dict(metrics or {})
    for statement in (all_metrics or []):
        if isinstance(statement, dict) and statement.get("name"):
            metrics.setdefault(statement["name"], statement)
    superseded_names = {entry.get("name") for entry in (superseded or [])
                        if isinstance(entry, dict)}

    never_searched = releases_examined is None
    rows = {}
    for row, names in COVERAGE_ROWS:
        if never_searched:
            rows[row] = GuidanceMetricStatus.NOT_SEARCHED
        else:
            rows[row] = _row_status(names, metrics, superseded_names)

    current_rows = [row for row, status in rows.items()
                    if status == GuidanceMetricStatus.CURRENT]
    dcf_current = [row for row in DCF_RELEVANT_ROWS
                   if rows.get(row) == GuidanceMetricStatus.CURRENT]
    dcf_missing = [row for row in DCF_RELEVANT_ROWS
                   if rows.get(row) != GuidanceMetricStatus.CURRENT]

    if never_searched:
        overall = GuidanceCoverage.UNAVAILABLE
        dcf_status = DcfGuidanceCoverage.UNAVAILABLE
        absence = GuidanceAbsence.NOT_APPLICABLE
    elif extraction_failed and not current_rows:
        overall = GuidanceCoverage.UNAVAILABLE
        dcf_status = DcfGuidanceCoverage.UNAVAILABLE
        absence = GuidanceAbsence.GUIDANCE_EXTRACTION_FAILED
    elif not current_rows:
        overall = GuidanceCoverage.UNAVAILABLE
        dcf_status = DcfGuidanceCoverage.UNAVAILABLE
        absence = GuidanceAbsence.NO_GUIDANCE_EXISTS
    else:
        overall = (GuidanceCoverage.COMPLETE_FOR_RELEVANT_METRICS if not dcf_missing
                   else GuidanceCoverage.PARTIAL)
        if not dcf_missing:
            dcf_status = DcfGuidanceCoverage.COMPLETE_FOR_DCF_RELEVANT_METRICS
        elif dcf_current:
            dcf_status = DcfGuidanceCoverage.PARTIAL
        else:
            dcf_status = DcfGuidanceCoverage.MINIMAL
        # Section 14: partial guidance that happens to exclude revenue is
        # named for what it is, so nothing downstream reports it as absence.
        if rows.get("revenue") != GuidanceMetricStatus.CURRENT \
                and rows.get("revenue_growth") != GuidanceMetricStatus.CURRENT:
            absence = GuidanceAbsence.NO_REVENUE_GUIDANCE
        elif dcf_missing:
            absence = GuidanceAbsence.PARTIAL_GUIDANCE_ONLY
        else:
            absence = GuidanceAbsence.NOT_APPLICABLE

    return {
        "rows": rows,
        "guidance_coverage_status": overall,
        "dcf_guidance_coverage": dcf_status,
        "absence_reason": absence,
        "current_rows": sorted(current_rows),
        "dcf_rows_current": list(dcf_current),
        "dcf_rows_missing": list(dcf_missing),
        "releases_examined": releases_examined,
    }


def guidance_summary_line(matrix) -> str:
    """One sentence for the compact report (section 12).

    Never renders "unavailable" while a row is CURRENT -- that claim is the
    thing this whole section exists to stop.
    """
    if not matrix:
        return "no guidance assessment was made"
    status = matrix.get("guidance_coverage_status")
    current = matrix.get("current_rows") or []
    if status == GuidanceCoverage.UNAVAILABLE:
        reason = matrix.get("absence_reason")
        if reason == GuidanceAbsence.GUIDANCE_EXTRACTION_FAILED:
            return ("guidance could not be extracted from the filings examined (an extraction "
                    "failure, not a statement that none was issued)")
        if reason == GuidanceAbsence.NOT_APPLICABLE:
            return "guidance was not searched for in this run"
        return "no current guidance was found in the earnings releases examined"
    covered = ", ".join(current)
    missing = ", ".join(matrix.get("dcf_rows_missing") or [])
    line = f"partial — current guidance for {covered}"
    if status == GuidanceCoverage.COMPLETE_FOR_RELEVANT_METRICS:
        line = f"complete for the metrics this valuation uses — {covered}"
    elif missing:
        line += f"; no current guidance for {missing}"
    return line


def assess_guidance_coverage(metrics) -> dict:
    """Section 8. Which valuation-relevant metric groups the guidance covers."""
    names = set(metrics or {})
    covered, missing = [], []
    for index, group in enumerate(VALUATION_RELEVANT_METRICS):
        present = sorted(names & set(group))
        if present:
            covered.append({"group": group[0], "present": present})
        else:
            missing.append(group[0])

    if not names:
        status = GuidanceCoverage.UNAVAILABLE
    elif not missing:
        status = GuidanceCoverage.COMPLETE_FOR_RELEVANT_METRICS
    else:
        core_covered = sum(
            1 for index in _CORE_GROUP_INDEXES
            if names & set(VALUATION_RELEVANT_METRICS[index]))
        if core_covered == len(_CORE_GROUP_INDEXES):
            status = GuidanceCoverage.PARTIAL
        elif core_covered:
            status = GuidanceCoverage.PARTIAL
        else:
            status = GuidanceCoverage.MINIMAL
    return {
        "guidance_coverage_status": status,
        "covered_groups": covered,
        "missing_groups": missing,
        "extracted_metrics": sorted(names),
    }
