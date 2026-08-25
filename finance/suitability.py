"""Phase H.7 — is a discounted-cash-flow valuation the right instrument here?

THE BUG THIS EXISTS TO FIX (live loss-making growth company, 2026-08-19)
========================================================================
A full analysis produced a VALID DCF with a positive value per share for a
company whose trailing twelve months look like this:

    revenue                    $5,883M
    operating income          -$3,533M      operating margin  -60.1%
    operating cash flow       -$1,845M
    free cash flow            -$3,489M

The forecast it used was an operating margin of +1.0% in every one of the
five years. Not because anything suggested the company reaches break-even
next year -- nothing did -- but because +1.0% is `MARGIN_BOUNDS[0]`, the
lowest value the engine will model. The observed -60.1% was clamped to it,
and the clamped number was then treated as the forecast.

That is not a conservative assumption. It is a 61-point swing in year one,
invented by a bound, which turns a company burning $3.5B a year into a
marginally profitable one and then discounts the result. The DCF validated
cleanly, because validation checks arithmetic and the arithmetic was fine.

THE DISTINCTION THIS MODULE ENFORCES
====================================
    DCF VALIDITY      did the arithmetic hold? (finance/dcf.py)
    DCF SUITABILITY   does this instrument describe this company at all?

They are independent. A valuation can be perfectly valid and completely
unsuitable, which is precisely the state the run above was in.

Suitability is NOT a profitability test. Plenty of loss-making companies are
legitimately valued by DCF -- that is most of what a DCF is for when a
business is scaling. What makes it unsuitable is when the model cannot
REPRESENT the path: when reaching the modelled margin requires a change the
evidence does not support, when the bounds themselves are doing the
forecasting, or when the terminal value is carrying essentially the whole
answer.

Nothing here is issuer-specific. Every input is a normalized fact or a
deterministic model bound.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from finance import business_model as bm


class DcfSuitability:
    """Section 35."""

    SUITABLE = "SUITABLE"
    SUITABLE_WITH_HIGH_UNCERTAINTY = "SUITABLE_WITH_HIGH_UNCERTAINTY"
    LIMITED = "LIMITED"
    NOT_SUITABLE = "NOT_SUITABLE"
    ALL = (SUITABLE, SUITABLE_WITH_HIGH_UNCERTAINTY, LIMITED, NOT_SUITABLE)

    _RANK = {SUITABLE: 0, SUITABLE_WITH_HIGH_UNCERTAINTY: 1, LIMITED: 2, NOT_SUITABLE: 3}

    @classmethod
    def worst(cls, statuses) -> str:
        found = [s for s in statuses if s in cls._RANK]
        return max(found, key=lambda s: cls._RANK[s]) if found else cls.SUITABLE


DCF_ASSUMPTION_EXCEEDS_MODEL_RANGE = "DCF_ASSUMPTION_EXCEEDS_MODEL_RANGE"
# Section 24. A diagnostic, NOT a market-price calibration: a company with
# substantial recurring cash generation whose modelled equity collapses to
# about nothing is describing a broken input, not a cheap stock.
DCF_ECONOMIC_DISCONNECT = "DCF_ECONOMIC_DISCONNECT"


# How far an observed value may sit outside a model bound before the bound is
# doing the forecasting rather than limiting it.
#
# Set as a MULTIPLE of the bound's own width rather than an absolute number,
# so it scales with whatever bounds are configured. A margin bound of
# [1%, 60%] is 59 points wide; an observed -60% sits more than one full width
# below the floor, which is the definition of "this model cannot represent
# this company" rather than "this input needed trimming".
BOUND_BREACH_WIDTH_MULTIPLE = 0.5

# Fraction of enterprise value carried by the terminal value above which the
# explicit forecast is not really doing the work (section 45). Matches
# finance/dcf.py's own terminal-dependency warning threshold.
TERMINAL_DEPENDENCE_HIGH = 0.75
TERMINAL_DEPENDENCE_EXTREME = 0.90

# Share-count growth over the trailing period above which dilution is
# material enough that a per-share value built on today's count overstates
# what a current holder owns.
MATERIAL_DILUTION = 0.10


@dataclass
class SuitabilitySignal:
    """One reason the instrument does or does not fit."""

    code: str
    severity: str            # info | warning | blocking
    detail: str
    observed: Optional[float] = None
    modelled: Optional[float] = None

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity, "detail": self.detail,
                "observed": self.observed, "modelled": self.modelled}


@dataclass
class SuitabilityAssessment:
    status: str = DcfSuitability.SUITABLE
    signals: List[SuitabilitySignal] = field(default_factory=list)
    summary: str = ""

    @property
    def blocks_valuation_conclusion(self) -> bool:
        return self.status == DcfSuitability.NOT_SUITABLE

    def to_dict(self) -> dict:
        return {"dcf_suitability": self.status,
                "signals": [s.to_dict() for s in self.signals],
                "summary": self.summary}


def _signal(assessment: SuitabilityAssessment, code: str, severity: str, detail: str,
            observed=None, modelled=None) -> None:
    assessment.signals.append(SuitabilitySignal(code=code, severity=severity,
                                                detail=detail, observed=observed,
                                                modelled=modelled))


def classify_bound_breach(observed: Optional[float], bounds: Tuple[float, float],
                          label: str) -> Optional[dict]:
    """Did a model bound LIMIT an input, or REPLACE it? (section 36)

    A clamp that moves an input a little is a safety limit doing its job. A
    clamp that moves it further than half the bound's own width has stopped
    limiting anything -- the applied value no longer bears a relationship to
    the observed one, and calling it a forecast asserts a change in the
    business that no evidence supports.
    """
    if observed is None:
        return None
    low, high = bounds
    width = abs(high - low)
    if low <= observed <= high:
        return None
    breached = low if observed < low else high
    distance = abs(observed - breached)
    return {
        "label": label,
        "observed_value": observed,
        "bound": breached,
        "distance": distance,
        "bound_width": width,
        "replaces_input": distance > width * BOUND_BREACH_WIDTH_MULTIPLE,
    }


def assess_dcf_suitability(
        *,
        revenue: Optional[float] = None,
        operating_margin: Optional[float] = None,
        free_cash_flow: Optional[float] = None,
        operating_cash_flow: Optional[float] = None,
        margin_bounds: Optional[Tuple[float, float]] = None,
        growth_bounds: Optional[Tuple[float, float]] = None,
        observed_growth: Optional[float] = None,
        terminal_value_share: Optional[float] = None,
        share_dilution: Optional[float] = None,
        share_reconciliation_status: Optional[str] = None,
        profitability_findings: Optional[list] = None,
        assumption_conflicts: Optional[list] = None,
        base_period_aligned: Optional[bool] = None,
        margin_basis: Optional[str] = None,
        normalization_status: Optional[str] = None,
        modelled_equity_value: Optional[float] = None,
        recurring_free_cash_flow: Optional[float] = None,
        historical_comparability: Optional[str] = None,
        negative_periods: Optional[int] = None,
        total_periods: Optional[int] = None,
        cash_runway_years: Optional[float] = None,
        business_model: Optional[object] = None,
) -> SuitabilityAssessment:
    """Classify whether a DCF describes this company (section 35).

    Deliberately NOT a profitability screen. A loss-making company with a
    credible path to positive cash generation is a normal DCF subject and
    comes out SUITABLE_WITH_HIGH_UNCERTAINTY. What downgrades a run is the
    model being unable to REPRESENT the company: a bound substituting for an
    input, a terminal value carrying the entire answer, an unresolved share
    basis under every per-share figure.
    """
    assessment = SuitabilityAssessment()

    # -- sections 17-19: can this model represent this business at all? -----
    #
    # Checked FIRST because it is not a question of degree. Every other
    # signal here asks how much confidence a valuation deserves; this one
    # asks whether the quantity being discounted is the right quantity. A
    # broker-dealer's operating cash flow is mostly customer money in
    # transit, so operating cash flow less capital expenditure -- correct
    # arithmetic, and the input this model discounts -- is not cash the
    # owners can take out. Discounting it produces a number with no
    # meaning, which is worse than declining to produce one.
    if business_model is not None:
        fcff_status = getattr(business_model, "standard_fcff_suitability", None)
        if fcff_status == bm.FcffSuitability.NOT_SUITABLE:
            _signal(assessment, bm.DCF_CASH_FLOW_NOT_STANDARD_FCFF, "blocking",
                    bm.describe_cash_flow_limitation(business_model))
        elif fcff_status == bm.FcffSuitability.LIMITED:
            _signal(assessment, bm.DCF_CASH_FLOW_NOT_STANDARD_FCFF, "material",
                    bm.describe_cash_flow_limitation(business_model))

    # -- section 36: is a bound forecasting? --------------------------------
    if margin_bounds is not None:
        breach = classify_bound_breach(operating_margin, margin_bounds, "operating_margin")
        if breach and breach["replaces_input"]:
            _signal(assessment, DCF_ASSUMPTION_EXCEEDS_MODEL_RANGE, "blocking",
                    f"The observed operating margin of {operating_margin:.1%} lies "
                    f"{breach['distance']:.1%} outside the model's bound of "
                    f"{breach['bound']:.1%}, which is more than half the bound's own width. "
                    "Applying the bound would not limit the input, it would REPLACE it: the "
                    "modelled company would be one that reaches break-even immediately, "
                    "which is a claim about the business rather than a safety limit.",
                    observed=operating_margin, modelled=breach["bound"])
        elif breach:
            _signal(assessment, DCF_ASSUMPTION_EXCEEDS_MODEL_RANGE, "warning",
                    f"The observed operating margin of {operating_margin:.1%} sits outside "
                    f"the model bound of {breach['bound']:.1%} and was limited to it.",
                    observed=operating_margin, modelled=breach["bound"])
    if growth_bounds is not None:
        breach = classify_bound_breach(observed_growth, growth_bounds, "revenue_growth")
        if breach:
            severity = "warning" if not breach["replaces_input"] else "warning"
            _signal(assessment, DCF_ASSUMPTION_EXCEEDS_MODEL_RANGE, severity,
                    f"Observed revenue growth of {observed_growth:.1%} lies outside the "
                    f"model bound of {breach['bound']:.1%}; the applied value is the bound, "
                    "not an estimate of this company's growth.",
                    observed=observed_growth, modelled=breach["bound"])

    # -- cash generation ----------------------------------------------------
    if free_cash_flow is not None and free_cash_flow < 0:
        detail = f"Free cash flow is negative ({free_cash_flow:,.0f})."
        if cash_runway_years is not None:
            detail += (f" At the current burn rate, reported liquidity covers roughly "
                       f"{cash_runway_years:.1f} year(s).")
        severity = "warning"
        if (negative_periods is not None and total_periods
                and negative_periods >= max(3, total_periods - 1)):
            severity = "blocking" if operating_cash_flow is not None \
                and operating_cash_flow < 0 else "warning"
            detail += (f" Cash generation has been negative in {negative_periods} of the "
                       f"last {total_periods} reported periods, so this is a persistent "
                       "state rather than one weak year.")
        _signal(assessment, "PERSISTENT_NEGATIVE_CASH_GENERATION", severity, detail,
                observed=free_cash_flow)

    if operating_margin is not None and operating_margin < 0:
        _signal(assessment, "NEGATIVE_OPERATING_MARGIN", "warning",
                f"The operating margin is negative ({operating_margin:.1%}). A DCF can value "
                "a business that is scaling toward profitability, but the forecast must "
                "represent the path rather than assume it has already happened.",
                observed=operating_margin)

    # -- section 45: is the terminal value carrying the answer? -------------
    if terminal_value_share is not None:
        if terminal_value_share >= TERMINAL_DEPENDENCE_EXTREME:
            _signal(assessment, "TERMINAL_VALUE_DOMINANCE", "blocking",
                    f"{terminal_value_share:.0%} of enterprise value sits in the terminal "
                    "value, so the explicit forecast is almost irrelevant to the answer and "
                    "the valuation is really a statement about the perpetuity assumption.",
                    observed=terminal_value_share)
        elif terminal_value_share >= TERMINAL_DEPENDENCE_HIGH:
            _signal(assessment, "TERMINAL_VALUE_DOMINANCE", "warning",
                    f"{terminal_value_share:.0%} of enterprise value sits in the terminal "
                    "value; near-term cash flows contribute little to the result.",
                    observed=terminal_value_share)

    # -- share basis: every per-share figure depends on it ------------------
    if share_reconciliation_status in ("MATERIAL_DIFFERENCE", "INCOMPATIBLE_BASIS"):
        _signal(assessment, "UNRESOLVED_SHARE_BASIS", "blocking",
                "The share count underlying every per-share figure does not reconcile "
                "against the reported market capitalisation, so a modelled value per share "
                "cannot be compared with the market price.",
                observed=None)
    if share_dilution is not None and share_dilution > MATERIAL_DILUTION:
        _signal(assessment, "MATERIAL_DILUTION", "warning",
                f"Shares outstanding grew {share_dilution:.1%} over the trailing period. A "
                "per-share value built on today's count does not reflect further issuance "
                "the business may still require.",
                observed=share_dilution)

    # -- Phase H.9, sections 18/20: conflicts from the assumption layer -----
    for conflict in (assumption_conflicts or []):
        if not isinstance(conflict, dict):
            continue
        code = conflict.get("code")
        if code == "DCF_MODEL_BOUND_CONFLICT":
            _signal(assessment, code, "blocking", conflict.get("message", ""),
                    observed=conflict.get("raw_evidence_value"),
                    modelled=conflict.get("applied_value"))
        elif code == "DCF_GUIDANCE_ASSUMPTION_CONFLICT":
            _signal(assessment, code, "blocking", conflict.get("message", ""),
                    observed=conflict.get("assumption_value"))

    # -- section 7/20: the base is not one period ---------------------------
    if base_period_aligned is False:
        _signal(assessment, "TTM_BASE_PERIOD_MISMATCH", "warning",
                "The metrics presented as one trailing-twelve-month financial base do not all "
                "cover the same window, so the valuation's inputs are not a single period.")

    # -- section 21: assumption conflicts carried from the profitability layer
    for finding in (profitability_findings or []):
        if not isinstance(finding, dict):
            continue
        code = finding.get("code")
        if code == "CONFIGURED_MARGIN_CONFLICTS_WITH_REPORTED_PROFITABILITY":
            _signal(assessment, code, "blocking", finding.get("message", ""))
        elif code == "PROFITABILITY_CROSS_METRIC_CONFLICT":
            _signal(assessment, code, "warning", finding.get("message", ""))

    # -- section 11/23: the configured default IS the forecast ---------------
    if margin_basis == "configured_default":
        _signal(assessment, "CONFIGURED_MARGIN_IS_THE_FORECAST", "blocking",
                "The operating margin driving this valuation is a CONFIGURED DEFAULT, not a "
                "figure read or normalized from this company's own reporting. A default is a "
                "last resort that keeps the pipeline running; it is not an estimate of this "
                "company's profitability, and every valuation figure derived from it inherits "
                "that.")
    elif normalization_status == "PARTIAL":
        _signal(assessment, "NORMALIZATION_UNRESOLVED", "warning",
                "Reported profitability is materially affected by unusual items and the "
                "normalization is only partial, so the forward margin carries an unresolved "
                "adjustment.")
    elif normalization_status == "UNAVAILABLE" and margin_basis != "reported_GAAP":
        _signal(assessment, "NORMALIZATION_UNRESOLVED", "blocking",
                "Reported profitability could not be normalized and no reported margin was "
                "usable, so the forward margin rests on neither.")

    # -- section 24: economic disconnect -------------------------------------
    # A DIAGNOSTIC. It never moves an assumption toward the market price --
    # the market may be wrong -- it only says that a company generating
    # substantial recurring cash should not model to approximately nothing,
    # and that when it does the inputs are the thing to examine.
    if (modelled_equity_value is not None and recurring_free_cash_flow
            and recurring_free_cash_flow > 0):
        if modelled_equity_value <= abs(recurring_free_cash_flow):
            _signal(assessment, DCF_ECONOMIC_DISCONNECT, "blocking",
                    f"The modelled equity value ({modelled_equity_value:,.0f}) is no larger "
                    f"than a single year of this company's recurring free cash flow "
                    f"({recurring_free_cash_flow:,.0f}). A business generating that cash does "
                    "not have approximately no equity value; the disconnect points at an "
                    "assumption, not at a mispricing, and the market price plays no part in "
                    "this test.",
                    observed=modelled_equity_value, modelled=recurring_free_cash_flow)

    if historical_comparability == "STRUCTURAL_BREAK":
        _signal(assessment, "STRUCTURALLY_CHANGED_BUSINESS", "warning",
                "The reported history spans a structural break, so the trend the forecast "
                "extends describes a different business from the one being valued.")

    # -- classification -----------------------------------------------------
    blocking = [s for s in assessment.signals if s.severity == "blocking"]
    warnings = [s for s in assessment.signals if s.severity == "warning"]
    if blocking:
        assessment.status = (DcfSuitability.NOT_SUITABLE if len(blocking) > 1
                             else DcfSuitability.LIMITED)
    elif len(warnings) >= 3:
        assessment.status = DcfSuitability.LIMITED
    elif warnings:
        assessment.status = DcfSuitability.SUITABLE_WITH_HIGH_UNCERTAINTY
    else:
        assessment.status = DcfSuitability.SUITABLE

    if assessment.signals:
        assessment.summary = " ".join(s.detail for s in assessment.signals[:3])
    else:
        assessment.summary = ("No condition was found that would make a discounted-cash-flow "
                              "valuation an unsuitable instrument for this company.")
    return assessment


# ---------------------------------------------------------------------------
# Section 37 — a margin path a turnaround can actually occupy
# ---------------------------------------------------------------------------


def normalization_path(observed: float, target: float, years: int,
                       floor: Optional[float] = None) -> List[float]:
    """A year-by-year path from what the company reports toward a target.

    Section 37: for a business at -60% margin, the honest five-year shape is
    deep loss -> narrowing loss -> break-even -> normalized, not +1% flat.
    This produces that shape by straight-line interpolation from the OBSERVED
    value, so year one stays close to reality and the improvement is visible
    and adjustable rather than hidden inside a bound.

    `floor` clips the path from below WITHOUT moving its starting point --
    the difference between limiting a trajectory and replacing it.
    """
    if years <= 1:
        return [observed]
    path = [observed + (target - observed) * (index / (years - 1)) for index in range(years)]
    if floor is not None:
        path = [max(floor, value) for value in path]
    return path
