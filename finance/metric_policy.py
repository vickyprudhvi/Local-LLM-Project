"""Phase H.12 — what a metric MEANS for a given business model.

The previous phase established that operating cash flow less capital
expenditure is not owner free cash flow for a broker-dealer, a bank or an
insurer, and used that to decline the standard valuation model. It then let
the same number through to everyone else. A live insurer produced exactly
the contradiction that implies: a Valuation section stating that its
operating cash flow less capital expenditure is NOT owner free cash flow,
and a Bull Case two sections later citing "free cash flow of 23.6 billion"
as evidence of cash generation.

The classification was correct. It was just local to the DCF gate.

This module makes the decision reusable. It answers one question --
"may this metric support this KIND OF CLAIM for this business model?" --
and every consumer asks it: the evidence index, the claim validators, the
renderer, the condition builder. A metric is never deleted; what changes is
what may be concluded from it.

Nothing here knows a ticker. Policy is keyed on the business-model
classification, which itself comes from the issuer's SEC SIC code and its
own filed concepts.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from finance.business_model import CashFlowValuationProfile as Profile


class Suitability:
    """How much weight a metric can carry for a business model."""

    PRIMARY = "PRIMARY"
    SUPPORTING = "SUPPORTING"
    LOW_INFORMATION_VALUE = "LOW_INFORMATION_VALUE"
    NOT_ECONOMICALLY_COMPARABLE = "NOT_ECONOMICALLY_COMPARABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"

    ALL = (PRIMARY, SUPPORTING, LOW_INFORMATION_VALUE,
           NOT_ECONOMICALLY_COMPARABLE, NOT_APPLICABLE)


class MetricUse:
    """The KIND of claim a metric is being asked to support.

    Suitability is a property of a metric AND a use, never of a metric
    alone. An insurer's current ratio is a perfectly good input to a
    balance-sheet description and a poor basis for a liquidity-distress
    conclusion; one number, two answers, and a policy that returned a single
    verdict would get one of them wrong.
    """

    OWNER_CASH_CLAIM = "OWNER_CASH_CLAIM"
    VALUATION_INPUT = "VALUATION_INPUT"
    LIQUIDITY_CLAIM = "LIQUIDITY_CLAIM"
    LEVERAGE_CLAIM = "LEVERAGE_CLAIM"
    PROFITABILITY_CLAIM = "PROFITABILITY_CLAIM"
    GROWTH_CLAIM = "GROWTH_CLAIM"
    FACTUAL_DISPLAY = "FACTUAL_DISPLAY"

    ALL = (OWNER_CASH_CLAIM, VALUATION_INPUT, LIQUIDITY_CLAIM, LEVERAGE_CLAIM,
           PROFITABILITY_CLAIM, GROWTH_CLAIM, FACTUAL_DISPLAY)


# Diagnostic codes (sections 6, 9).
CASH_FLOW_SEMANTIC_MISUSE = "CASH_FLOW_SEMANTIC_MISUSE"
GENERIC_RATIO_INTERPRETATION_NOT_SUPPORTED = "GENERIC_RATIO_INTERPRETATION_NOT_SUPPORTED"


@dataclass(frozen=True)
class MetricSuitability:
    """One verdict, with the reason a reader would need to accept it."""

    metric_id: str
    business_model: str
    suitability: str
    use: str
    allowed: bool
    interpretation: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "metric_id": self.metric_id, "business_model": self.business_model,
            "suitability": self.suitability, "use": self.use, "allowed": self.allowed,
            "interpretation": self.interpretation, "reason": self.reason,
        }


# Business models for which operating cash flow does not measure owner
# economics. Sourced from the same set the valuation gate uses, so the two
# cannot disagree about which companies are specialized.
_CASH_FLOW_SPECIALIZED = frozenset({
    Profile.BANK, Profile.BROKER_DEALER, Profile.INSURER,
    Profile.FINANCIAL_INSTITUTION,
})

# Why the simple figure is not owner cash, per model. Kept here rather than
# in the renderer so every consumer gives the same explanation.
_CASH_FLOW_REASON = {
    Profile.INSURER: ("premiums are collected long before claims are paid, so operating cash "
                      "flow tracks the insurance cycle rather than the period's earnings"),
    Profile.BANK: ("deposit-taking and lending are the business itself, so changes in deposits "
                   "and loans dominate operating cash flow"),
    Profile.BROKER_DEALER: ("customer cash and segregated balances move through operating cash "
                            "flow, which therefore largely measures customer money in transit"),
    Profile.FINANCIAL_INSTITUTION: ("lending and funding flows run through operating cash flow, "
                                    "which therefore does not measure cash available to owners"),
}

# Ratios built on a current/non-current split. That split describes a
# working-capital business; for a balance sheet whose liabilities are policy
# reserves, customer payables or deposits, the same arithmetic carries very
# little information about whether the company can meet its obligations.
_WORKING_CAPITAL_RATIOS = frozenset({
    "current_ratio", "quick_ratio", "cash_ratio", "working_capital",
})

_CASH_FLOW_METRICS = frozenset({
    "free_cash_flow", "simple_fcf", "free_cash_flow_margin",
    "net_debt_to_fcf", "debt_to_fcf",
})


# Section 10 — which metrics actually carry the story, per model. These are
# names already present in the canonical/fundamental namespaces; nothing new
# is computed here, and a metric absent for an issuer is simply not listed.
PRIMARY_METRICS = {
    Profile.INSURER: ("revenue", "operating_income", "operating_margin", "net_income",
                      "net_margin", "stockholders_equity", "debt_to_equity",
                      "total_debt", "roe_ending_equity", "revenue_growth"),
    Profile.BANK: ("revenue", "net_income", "net_margin", "stockholders_equity",
                   "roe_ending_equity", "total_debt", "debt_to_equity"),
    Profile.BROKER_DEALER: ("revenue", "operating_income", "operating_margin", "net_income",
                            "stockholders_equity", "roe_ending_equity"),
    Profile.FINANCIAL_INSTITUTION: ("revenue", "net_income", "net_margin",
                                    "stockholders_equity", "roe_ending_equity"),
    Profile.REIT_OR_SPECIALIZED: ("revenue", "operating_income", "operating_cash_flow",
                                  "total_debt", "net_debt", "stockholders_equity"),
}

_STANDARD_PRIMARY = ("revenue", "revenue_growth", "operating_income", "operating_margin",
                     "net_income", "free_cash_flow", "operating_cash_flow", "net_debt")


def is_specialized(profile: Optional[str]) -> bool:
    return profile in _CASH_FLOW_SPECIALIZED


def evaluate(metric_id: str, use: str, profile: Optional[str]) -> MetricSuitability:
    """May `metric_id` support a claim of kind `use` for this business model?

    The default is permissive: an unrecognised metric, an unknown business
    model, or an ordinary operating company gets the project's existing
    behaviour unchanged. Only the combinations this module has an actual
    argument about are restricted -- a policy that quietly narrowed every
    claim would be a different kind of wrong.
    """
    profile = profile or Profile.UNKNOWN
    metric_id = (metric_id or "").strip()

    # Displaying a number is always allowed. Section 8 is explicit that the
    # figure stays available; what is controlled is the conclusion.
    if use == MetricUse.FACTUAL_DISPLAY:
        return MetricSuitability(
            metric_id, profile, Suitability.SUPPORTING, use, True,
            interpretation="the figure may be reported as computed")

    if metric_id in _CASH_FLOW_METRICS and is_specialized(profile):
        if use in (MetricUse.OWNER_CASH_CLAIM, MetricUse.VALUATION_INPUT,
                   MetricUse.LEVERAGE_CLAIM):
            detail = _CASH_FLOW_REASON.get(profile, "this business model's operating cash flow "
                                                    "does not correspond to owner economics")
            return MetricSuitability(
                metric_id, profile, Suitability.NOT_ECONOMICALLY_COMPARABLE, use, False,
                interpretation=("operating cash flow less capital expenditure, which for this "
                                "business model is not owner free cash flow"),
                reason=(f"For a {profile.replace('_', ' ').lower()}, {detail}. The figure is "
                        f"arithmetically correct and may be reported; it may not stand for cash "
                        f"available to the owners."))

    if metric_id in _WORKING_CAPITAL_RATIOS and is_specialized(profile):
        if use == MetricUse.LIQUIDITY_CLAIM:
            return MetricSuitability(
                metric_id, profile, Suitability.LOW_INFORMATION_VALUE, use, False,
                interpretation=("a current/non-current split of a balance sheet whose "
                                "liabilities are policy reserves, customer balances or deposits"),
                reason=(f"For a {profile.replace('_', ' ').lower()}, the current ratio does not "
                        f"measure the ability to meet near-term obligations the way it does for "
                        f"a working-capital business. A value below 1.0 is normal and is not by "
                        f"itself evidence of liquidity stress."))

    primary = PRIMARY_METRICS.get(profile, _STANDARD_PRIMARY)
    suitability = Suitability.PRIMARY if metric_id in primary else Suitability.SUPPORTING
    return MetricSuitability(metric_id, profile, suitability, use, True)


def prohibited_uses(profile: Optional[str]) -> List[dict]:
    """Every restriction this business model carries, for the evidence packet.

    Stated as data rather than prose so a validator can enforce it and the
    research prompt can state it, from one source.
    """
    out = []
    for metric in sorted(_CASH_FLOW_METRICS):
        verdict = evaluate(metric, MetricUse.OWNER_CASH_CLAIM, profile)
        if not verdict.allowed:
            out.append({"metric_id": metric, "use": MetricUse.OWNER_CASH_CLAIM,
                        "suitability": verdict.suitability, "reason": verdict.reason})
    for metric in sorted(_WORKING_CAPITAL_RATIOS):
        verdict = evaluate(metric, MetricUse.LIQUIDITY_CLAIM, profile)
        if not verdict.allowed:
            out.append({"metric_id": metric, "use": MetricUse.LIQUIDITY_CLAIM,
                        "suitability": verdict.suitability, "reason": verdict.reason})
    return out


def cash_flow_label(profile: Optional[str]) -> str:
    """Section 31: what to call operating cash flow less capital expenditure.

    "Free cash flow" carries an economic claim. Where that claim is not
    validated the arithmetic keeps its name internally and the report uses a
    description of what was actually computed.
    """
    if is_specialized(profile):
        return "Cash flow after CapEx"
    return "FCF"


# ---------------------------------------------------------------------------
# Section 13 — which guidance metrics matter for which business
# ---------------------------------------------------------------------------
#
# Rows are the coverage-matrix vocabulary from `finance.guidance`. An insurer
# that guides adjusted earnings per share and a loss ratio has told the
# market what it plans to do; scoring it against a revenue-and-capex
# checklist and reporting "guidance unavailable" describes the checklist, not
# the company.

RELEVANT_GUIDANCE_ROWS = {
    Profile.INSURER: ("eps", "operating_margin", "tax_rate", "operating_cash_flow",
                      "leverage", "shares", "revenue"),
    Profile.BANK: ("eps", "tax_rate", "leverage", "shares", "operating_margin"),
    Profile.BROKER_DEALER: ("eps", "tax_rate", "leverage", "shares", "revenue"),
    Profile.FINANCIAL_INSTITUTION: ("eps", "tax_rate", "leverage", "shares"),
    Profile.REIT_OR_SPECIALIZED: ("operating_cash_flow", "free_cash_flow", "capex",
                                  "leverage", "eps"),
}

_STANDARD_GUIDANCE_ROWS = ("revenue", "revenue_growth", "gross_margin", "operating_margin",
                           "tax_rate", "capex", "operating_cash_flow", "free_cash_flow")


def relevant_guidance_rows(profile: Optional[str]) -> Tuple[str, ...]:
    """The guidance rows that actually bear on THIS business's outlook."""
    return RELEVANT_GUIDANCE_ROWS.get(profile or Profile.UNKNOWN, _STANDARD_GUIDANCE_ROWS)


# ---------------------------------------------------------------------------
# Section 29 — the packet research agents receive
# ---------------------------------------------------------------------------

@dataclass
class BusinessModelRelevantEvidence:
    """What this business model says about the evidence in hand."""

    business_model: str = Profile.UNKNOWN
    confidence: str = "UNKNOWN"
    primary_metrics: List[str] = field(default_factory=list)
    supporting_metrics: List[str] = field(default_factory=list)
    low_information_metrics: List[str] = field(default_factory=list)
    prohibited_interpretations: List[dict] = field(default_factory=list)
    relevant_guidance_rows: List[str] = field(default_factory=list)
    relevant_guidance_found: List[str] = field(default_factory=list)
    relevant_guidance_missing: List[str] = field(default_factory=list)
    valuation_method_status: Optional[str] = None
    cash_flow_label: str = "FCF"

    def to_dict(self) -> dict:
        return {
            "business_model": self.business_model,
            "confidence": self.confidence,
            "primary_metrics": list(self.primary_metrics),
            "supporting_metrics": list(self.supporting_metrics),
            "low_information_metrics": list(self.low_information_metrics),
            "prohibited_interpretations": [dict(p) for p in self.prohibited_interpretations],
            "relevant_guidance_rows": list(self.relevant_guidance_rows),
            "relevant_guidance_found": list(self.relevant_guidance_found),
            "relevant_guidance_missing": list(self.relevant_guidance_missing),
            "valuation_method_status": self.valuation_method_status,
            "cash_flow_label": self.cash_flow_label,
        }


def build_relevant_evidence(classification, canonical_current: Optional[dict] = None,
                            guidance_matrix: Optional[dict] = None,
                            valuation_method_status: Optional[str] = None
                            ) -> BusinessModelRelevantEvidence:
    """Sort the metrics this issuer actually has by what they are worth here."""
    profile = getattr(classification, "profile", None) or Profile.UNKNOWN
    packet = BusinessModelRelevantEvidence(
        business_model=profile,
        confidence=("CORROBORATED" if getattr(classification, "corroborating_concepts", ())
                    else "SINGLE_SOURCE"),
        prohibited_interpretations=prohibited_uses(profile),
        relevant_guidance_rows=list(relevant_guidance_rows(profile)),
        valuation_method_status=valuation_method_status,
        cash_flow_label=cash_flow_label(profile))

    available = sorted((canonical_current or {}).keys())
    primary = PRIMARY_METRICS.get(profile, _STANDARD_PRIMARY)
    for name in available:
        low = (evaluate(name, MetricUse.LIQUIDITY_CLAIM, profile).allowed is False
               or evaluate(name, MetricUse.OWNER_CASH_CLAIM, profile).allowed is False)
        if low:
            packet.low_information_metrics.append(name)
        elif name in primary:
            packet.primary_metrics.append(name)
        else:
            packet.supporting_metrics.append(name)

    rows = (guidance_matrix or {}).get("rows") or {}
    for row in packet.relevant_guidance_rows:
        if rows.get(row) == "CURRENT":
            packet.relevant_guidance_found.append(row)
        else:
            packet.relevant_guidance_missing.append(row)
    return packet


# ---------------------------------------------------------------------------
# Section 15 — valid is not the same as applicable
# ---------------------------------------------------------------------------

class ValuationMethodStatus:
    VALID_AND_APPLICABLE = "VALID_AND_APPLICABLE"
    VALID_BUT_NOT_APPLICABLE = "VALID_BUT_NOT_APPLICABLE"
    LIMITED = "LIMITED"
    # Phase 27/28. The model ran correctly and the arithmetic is sound; this
    # company's own forecast path does not support a perpetuity. A business
    # whose terminal-year free cash flow to the firm is negative cannot be
    # valued by growing that figure forever, and refusing to is the model
    # working -- not failing. Kept distinct from INVALID, which is reserved
    # for the model breaking its own checks.
    NOT_VALID_FOR_CURRENT_FORECAST_PATH = "NOT_VALID_FOR_CURRENT_FORECAST_PATH"
    INVALID = "INVALID"


# Section 16: how each status is stated to a reader. "Invalid" is reserved
# for a model that ran and failed its own checks -- saying it about a model
# that was never applicable to the business misdescribes both.
VALUATION_STATUS_WORDING = {
    ValuationMethodStatus.VALID_AND_APPLICABLE: None,
    ValuationMethodStatus.VALID_BUT_NOT_APPLICABLE:
        "Standard FCFF DCF: not applicable to this business model.",
    ValuationMethodStatus.LIMITED:
        "Standard FCFF DCF: usable here only with substantial caveats.",
    ValuationMethodStatus.NOT_VALID_FOR_CURRENT_FORECAST_PATH: (
        "Standard FCFF DCF: not valid for this company's current forecast path. "
        "The projected terminal-year cash flow is negative, so a perpetuity value "
        "cannot be computed from it. The model is working; the forecast does not "
        "support this valuation method."),
    ValuationMethodStatus.INVALID:
        "Standard FCFF DCF: ran but failed its own validation checks.",
}

# Validation statuses that describe the FORECAST rather than the model.
_FORECAST_PATH_FAILURES = frozenset({"DCF_NEGATIVE_TERMINAL_FCFF"})


def valuation_method_status(classification, dcf_available: bool,
                            dcf_validation_failed: bool = False,
                            dcf_validation_status: Optional[str] = None) -> str:
    """Which state this run is actually in.

    `dcf_validation_status` separates a model that broke from a forecast the
    model correctly declined to extrapolate (Phase 27-28). Without it every
    failure read as "model invalid", which is the wrong claim for a
    loss-making company whose terminal cash flow is simply negative.
    """
    if dcf_validation_failed:
        if dcf_validation_status in _FORECAST_PATH_FAILURES:
            return ValuationMethodStatus.NOT_VALID_FOR_CURRENT_FORECAST_PATH
        return ValuationMethodStatus.INVALID
    profile = getattr(classification, "profile", None)
    fcff = getattr(classification, "standard_fcff_suitability", None)
    if fcff == "NOT_SUITABLE":
        return ValuationMethodStatus.VALID_BUT_NOT_APPLICABLE
    if fcff == "LIMITED":
        return ValuationMethodStatus.LIMITED
    if not dcf_available:
        return ValuationMethodStatus.LIMITED
    return ValuationMethodStatus.VALID_AND_APPLICABLE


# ---------------------------------------------------------------------------
# Sections 6-9, 32-33, 40 — the claim validator
# ---------------------------------------------------------------------------
#
# Deterministic, and narrow on purpose. It matches a claim ABOUT a metric of
# a kind the policy above forbids for this business model -- not the metric's
# ordinary vocabulary. "Operating cash flow was $27.0 billion" states a fact
# and stays legal; "free cash flow demonstrates strong cash generation" draws
# the conclusion the policy says the number cannot carry.
#
# The model cannot argue its way past this (section 33): the verdict comes
# from the classification and the metric identity, never from the prose.

import re as _re

# A conclusion about owner economics drawn from a cash-flow figure.
_OWNER_CASH_CLAIM = _re.compile(
    r"\b(?:free[\s-]cash[\s-]flow|fcf|cash[\s-]generation|cash[\s-]conversion)\b"
    r"(?:[^.]|\.(?=\d)){0,80}?\b(?:demonstrat\w+|show\w*|indicat\w+|support\w*|underpin\w*|"
    r"confirm\w*|reflect\w*|evidence\w*|strong|robust|healthy|ample|solid|"
    r"strength|generat\w+\s+(?:strong|substantial))\b"
    r"|\b(?:strong|robust|healthy|ample|solid|substantial|significant)\b"
    r"(?:[^.]|\.(?=\d)){0,40}?\b(?:free[\s-]cash[\s-]flow|fcf|cash[\s-]generation)\b", _re.IGNORECASE)

# A leverage or coverage conclusion resting on a cash-flow denominator.
_FCF_LEVERAGE_CLAIM = _re.compile(
    r"\b(?:net[\s-]debt|debt|leverage)\b(?:[^.]|\.(?=\d)){0,50}?"
    r"\b(?:to|/|per|against|relative\s+to|covered\s+by)\b(?:[^.]|\.(?=\d)){0,20}?"
    r"\b(?:free[\s-]cash[\s-]flow|fcf)\b", _re.IGNORECASE)

# A liquidity-distress conclusion drawn from a working-capital ratio.
_LIQUIDITY_CLAIM = _re.compile(
    r"\b(?:current|quick|cash)\s+ratio\b(?:[^.]|\.(?=\d)){0,90}?"
    r"\b(?:liquidity\s+(?:risk|stress|pressure|concern|weakness|constraint)|"
    r"liquidity\s+is\s+(?:tight|weak|strained)|"
    r"short[\s-]term\s+(?:liquidity|solvency)|unable\s+to\s+(?:meet|cover)|"
    r"cannot\s+(?:meet|cover)|strain\w*|weakness|distress|shortfall)\b"
    r"|\b(?:liquidity\s+(?:risk|stress|pressure|concern|weakness)|"
    r"short[\s-]term\s+liquidity)\b(?:[^.]|\.(?=\d)){0,60}?\b(?:current|quick|cash)\s+ratio\b", _re.IGNORECASE)

# Section 40: a valuation method that does not apply is not an invalid model.
_MODEL_INVALID_CLAIM = _re.compile(
    r"\b(?:valuation\s+model|dcf(?:\s+model)?|model)\s+(?:is\s+)?"
    r"(?:invalid|broken|failed|unusable|not\s+valid)\b"
    r"|\b(?:invalid|broken)\s+(?:valuation\s+model|dcf\s+model)\b", _re.IGNORECASE)

# Section 40: the absence of a valuation is not a fundamental company risk.
_NO_DCF_IS_COMPANY_RISK = _re.compile(
    r"\b(?:no|absent|missing|unavailable|lack\s+of)\s+"
    r"(?:a\s+)?(?:dcf|valuation|discounted[\s-]cash[\s-]flow)\b(?:[^.]|\.(?=\d)){0,70}?"
    r"\b(?:company\s+risk|business\s+risk|issuer\s+risk|high\s+risk|"
    r"risk\s+is\s+high|fundamental\s+risk)\b", _re.IGNORECASE)


def validate_claim(text: str, profile: Optional[str],
                   valuation_status: Optional[str] = None) -> List[dict]:
    """Every business-model semantic violation in one passage.

    Returns findings, empty when clean. A finding names the code, the metric
    identity and the reason -- enough for a repair prompt to state what must
    change without restating the forbidden sentence.
    """
    findings: List[dict] = []
    if not isinstance(text, str) or not text.strip():
        return findings

    if not evaluate("free_cash_flow", MetricUse.OWNER_CASH_CLAIM, profile).allowed:
        verdict = evaluate("free_cash_flow", MetricUse.OWNER_CASH_CLAIM, profile)
        if _OWNER_CASH_CLAIM.search(text) or _FCF_LEVERAGE_CLAIM.search(text):
            findings.append({
                "code": CASH_FLOW_SEMANTIC_MISUSE,
                "metric_id": "free_cash_flow",
                "business_model": profile,
                "message": (
                    "This passage draws a conclusion about cash available to the owners from "
                    "operating cash flow less capital expenditure. " + verdict.reason +
                    " State the figure as cash flow after capital expenditure if it is "
                    "relevant, and base any conclusion about cash generation on the operating "
                    "and capital metrics listed as primary for this business model."),
            })

    if not evaluate("current_ratio", MetricUse.LIQUIDITY_CLAIM, profile).allowed:
        verdict = evaluate("current_ratio", MetricUse.LIQUIDITY_CLAIM, profile)
        if _LIQUIDITY_CLAIM.search(text):
            findings.append({
                "code": GENERIC_RATIO_INTERPRETATION_NOT_SUPPORTED,
                "metric_id": "current_ratio",
                "business_model": profile,
                "message": (
                    "This passage infers a liquidity problem from a working-capital ratio. "
                    + verdict.reason +
                    " The ratio may be reported; a conclusion about the ability to meet "
                    "obligations needs capital, reserve or leverage evidence instead."),
            })

    if valuation_status == ValuationMethodStatus.VALID_BUT_NOT_APPLICABLE:
        if _MODEL_INVALID_CLAIM.search(text):
            findings.append({
                "code": "VALUATION_APPLICABILITY_MISSTATED",
                "metric_id": "valuation_method",
                "business_model": profile,
                "message": (
                    "The standard discounted-cash-flow model did not fail; it does not apply "
                    "to this business model. Describe it as not applicable rather than as "
                    "invalid, broken or failed."),
            })

    if _NO_DCF_IS_COMPANY_RISK.search(text):
        findings.append({
            "code": "VALUATION_LIMITATION_AS_COMPANY_RISK",
            "metric_id": "valuation_method",
            "business_model": profile,
            "message": (
                "The absence of a valuation is a limitation of this analysis, not a risk "
                "the company carries. Record it as an analysis limitation; company risk "
                "must rest on the issuer's own economics."),
        })
    return findings
