"""Parts 2-6 — the only way into the valuation, and the only way out.

Two boundaries, one file, because they are two halves of one rule: a
valuation is built from validated inputs or not built, and its output either
qualifies as research evidence or is withheld.

`ValidatedDCFInputPacket` fails closed at CONSTRUCTION. Before it, individual
gates existed and each consumer had to remember to consult them; a gate you
must remember to call is a gate that eventually is not called. Construction
that refuses is a gate that cannot be skipped.

`ValuationEligibility` answers the second question. A model can run perfectly
and still produce something that must not become evidence — because the
business model does not fit it, because the forecast path cannot support a
perpetuity, or because an input the bridge needed was invalid. Only
VALID_FOR_RESEARCH may support a premium, a discount, a modelled return, an
"overvalued", or a valuation-derived risk.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from finance.validity import Validity


# ---------------------------------------------------------------------------
# Part 4 — the status vocabulary
# ---------------------------------------------------------------------------

class ValuationStatus:
    """Why a valuation may or may not be used, distinguished by cause.

    "Model invalid" as a catch-all told a reader the wrong thing four
    different ways. A model that never applied to the business, a forecast
    that cannot be extrapolated, an input that failed validation and a model
    that broke its own arithmetic are four different facts about four
    different things, and only the last is the model's fault.
    """

    VALID_FOR_RESEARCH = "VALID_FOR_RESEARCH"
    LIMITED = "LIMITED"
    INPUT_PACKET_INVALID = "INPUT_PACKET_INVALID"
    MODEL_ARITHMETIC_INVALID = "MODEL_ARITHMETIC_INVALID"
    FORECAST_PATH_INVALID = "FORECAST_PATH_INVALID"
    NOT_APPLICABLE_FOR_BUSINESS_MODEL = "NOT_APPLICABLE_FOR_BUSINESS_MODEL"
    # The model ran and its arithmetic is sound, but it rests on a period the
    # company has since superseded. A fifth distinct cause, and not the
    # model's fault -- which is exactly why it needs its own name rather than
    # being folded into MODEL_ARITHMETIC_INVALID.
    FINANCIAL_BASE_STALE = "FINANCIAL_BASE_STALE"

    ALL = (VALID_FOR_RESEARCH, LIMITED, INPUT_PACKET_INVALID,
           MODEL_ARITHMETIC_INVALID, FORECAST_PATH_INVALID,
           NOT_APPLICABLE_FOR_BUSINESS_MODEL, FINANCIAL_BASE_STALE)


# How each status is explained to a reader. The wording is part of the
# contract: none of these may be rendered as "the model is invalid" unless
# the model's own arithmetic actually failed.
STATUS_EXPLANATION = {
    ValuationStatus.VALID_FOR_RESEARCH: None,
    ValuationStatus.LIMITED:
        "The valuation is usable only with substantial caveats.",
    ValuationStatus.INPUT_PACKET_INVALID:
        "A required valuation input could not be validated, so no valuation was produced.",
    ValuationStatus.MODEL_ARITHMETIC_INVALID:
        "The valuation model ran but failed its own validation checks.",
    ValuationStatus.FORECAST_PATH_INVALID:
        "This company's projected terminal cash flow does not support a perpetuity "
        "value. The model is working; the forecast does not support this method.",
    ValuationStatus.NOT_APPLICABLE_FOR_BUSINESS_MODEL:
        "A standard discounted-cash-flow valuation does not apply to this business model.",
    ValuationStatus.FINANCIAL_BASE_STALE:
        "The valuation rests on financial results the company has since superseded, "
        "so it describes a period that is over.",
}


DCF_INPUT_PACKET_INVALID = "DCF_INPUT_PACKET_INVALID"


# ---------------------------------------------------------------------------
# Parts 2-3 — the input packet
# ---------------------------------------------------------------------------

# Inputs without which a normal equity valuation cannot be produced. Absence
# or invalidity of any one of these is a refusal, not a caveat.
REQUIRED_INPUTS = (
    "revenue", "operating_margin", "tax_rate",
    "total_debt", "net_debt", "share_count", "share_basis",
    "wacc", "terminal_growth",
)


@dataclass(frozen=True)
class ValidatedDCFInputPacket:
    """The only accepted input boundary for a normal DCF.

    Frozen: an input that passed validation must not be mutated afterwards.
    The whole point of validating at construction is defeated if a caller
    can adjust a field on the way to the model.

    Build with `build_dcf_input_packet`, never directly -- the constructor
    does not validate, the builder does, and a packet that exists is a
    packet that passed.
    """

    financial_base: Optional[str] = None
    revenue: Optional[float] = None
    operating_margin: Optional[float] = None
    tax_rate: Optional[float] = None
    depreciation_amortization: Optional[float] = None
    capex: Optional[float] = None
    working_capital_change: Optional[float] = None
    cash_or_liquidity: Optional[float] = None
    total_debt: Optional[float] = None
    net_debt: Optional[float] = None
    share_basis: Optional[str] = None
    share_count: Optional[float] = None
    forecast_assumptions: tuple = ()
    wacc: Optional[float] = None
    terminal_growth: Optional[float] = None
    business_model_profile: Optional[str] = None
    standard_fcff_suitability: Optional[str] = None
    source_evidence_ids: tuple = ()
    validation_status: str = Validity.VALID
    validation_reasons: tuple = ()
    # The exact argument set the valuation engine will receive, carried
    # INSIDE the packet rather than beside it. A caller that holds the
    # arguments separately can call the model without the packet; a caller
    # that must reach through the packet to get them cannot. Stored as a
    # tuple of pairs because the packet is frozen and a dict is not.
    model_arguments: tuple = ()

    @property
    def ok(self) -> bool:
        return self.validation_status == Validity.VALID

    def arguments(self) -> dict:
        """The engine's arguments. The only way to obtain them."""
        return dict(self.model_arguments)

    def to_dict(self) -> dict:
        return {
            "financial_base": self.financial_base,
            "revenue": self.revenue, "operating_margin": self.operating_margin,
            "tax_rate": self.tax_rate, "total_debt": self.total_debt,
            "net_debt": self.net_debt, "share_basis": self.share_basis,
            "share_count": self.share_count, "wacc": self.wacc,
            "terminal_growth": self.terminal_growth,
            "business_model_profile": self.business_model_profile,
            "standard_fcff_suitability": self.standard_fcff_suitability,
            "validation_status": self.validation_status,
            "validation_reasons": list(self.validation_reasons),
            "source_evidence_ids": list(self.source_evidence_ids),
        }


@dataclass
class PacketFailure:
    """Why a packet could not be built. Structured, not a bare string."""

    code: str = DCF_INPUT_PACKET_INVALID
    reasons: List[str] = field(default_factory=list)
    invalid_inputs: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"code": self.code, "reasons": list(self.reasons),
                "invalid_inputs": list(self.invalid_inputs)}


def build_dcf_input_packet(*, values: Dict[str, object],
                           validity_graph: Optional[dict] = None,
                           business_model=None,
                           assumption_rejections=None
                           ) -> Tuple[Optional[ValidatedDCFInputPacket], Optional[PacketFailure]]:
    """Part 3: validate, then construct — or refuse.

    Returns `(packet, None)` or `(None, failure)`. Never a packet with a
    warning attached: `value + warning` passed into the engine is the exact
    pattern this boundary exists to end, because every consumer downstream
    reads the value and none reads the warning.

    Checks, in the order a reader would want them explained:
      1. the business model can be valued this way at all
      2. no forward assumption was rejected on semantic grounds
      3. every required input is present
      4. no required input is INVALID in the dependency graph
      5. the discount rate exceeds the perpetual growth rate
    """
    failure = PacketFailure()
    graph = validity_graph or {}

    # 1. Applicability first: if the method does not fit the business, the
    # remaining checks are answering a question nobody should be asking.
    suitability = getattr(business_model, "standard_fcff_suitability", None)
    if suitability == "NOT_SUITABLE":
        failure.reasons.append(
            "A standard discounted-cash-flow valuation does not apply to this business "
            "model, so no input packet is built.")
        failure.invalid_inputs.append("business_model")
        return None, failure

    # 2. An assumption the semantic layer refused must not be replaced by a
    # default and carried in (spec §13: a clamp must never repair a semantic
    # error, and neither must a fallback).
    for rejection in (assumption_rejections or []):
        code = rejection.get("code") if isinstance(rejection, dict) else None
        if code:
            failure.reasons.append(
                f"A forward assumption was rejected upstream ({code}): "
                f"{rejection.get('reason', '')}")
            failure.invalid_inputs.append("forecast_assumptions")

    # 3-4. Required inputs must be present AND not invalidated.
    for name in REQUIRED_INPUTS:
        metric = graph.get(name)
        if metric is not None and metric.validity == Validity.INVALID:
            failure.invalid_inputs.append(name)
            failure.reasons.append(
                f"{name} is invalid for this analysis"
                + (f": {metric.reasons[0]}" if metric.reasons else "."))
            continue
        if values.get(name) is None:
            failure.invalid_inputs.append(name)
            failure.reasons.append(f"{name} is required for an equity valuation and is "
                                   "not available.")

    # 5. A perpetuity needs a discount rate above its growth rate; otherwise
    # the terminal value is negative or unbounded and the arithmetic is
    # meaningless rather than merely uncertain.
    wacc, terminal = values.get("wacc"), values.get("terminal_growth")
    if isinstance(wacc, (int, float)) and isinstance(terminal, (int, float)):
        if wacc <= terminal:
            failure.invalid_inputs.append("terminal_growth")
            failure.reasons.append(
                f"The discount rate ({wacc:.2%}) does not exceed the perpetual growth rate "
                f"({terminal:.2%}), so a terminal value cannot be computed.")

    if failure.reasons:
        failure.invalid_inputs = sorted(set(failure.invalid_inputs))
        return None, failure

    return ValidatedDCFInputPacket(
        financial_base=values.get("financial_base"),
        revenue=values.get("revenue"),
        operating_margin=values.get("operating_margin"),
        tax_rate=values.get("tax_rate"),
        depreciation_amortization=values.get("depreciation_amortization"),
        capex=values.get("capex"),
        working_capital_change=values.get("working_capital_change"),
        cash_or_liquidity=values.get("cash_or_liquidity"),
        total_debt=values.get("total_debt"),
        net_debt=values.get("net_debt"),
        share_basis=values.get("share_basis"),
        share_count=values.get("share_count"),
        forecast_assumptions=tuple(values.get("forecast_assumptions") or ()),
        model_arguments=tuple(sorted((values.get("model_arguments") or {}).items())),
        wacc=wacc, terminal_growth=terminal,
        business_model_profile=getattr(business_model, "profile", None),
        standard_fcff_suitability=suitability,
        source_evidence_ids=tuple(values.get("source_evidence_ids") or ()),
    ), None


# ---------------------------------------------------------------------------
# Parts 4-6 — the output gate
# ---------------------------------------------------------------------------

def classify_valuation(*, packet_failure=None,
                       dcf_available: bool = False,
                       dcf_validation_status: Optional[str] = None,
                       business_model=None,
                       suitability_status: Optional[str] = None,
                       financial_base_stale: bool = False) -> str:
    """One status, decided by cause, in order of what a reader needs first."""
    if getattr(business_model, "standard_fcff_suitability", None) == "NOT_SUITABLE":
        return ValuationStatus.NOT_APPLICABLE_FOR_BUSINESS_MODEL
    if packet_failure is not None:
        return ValuationStatus.INPUT_PACKET_INVALID
    if dcf_validation_status == "DCF_NEGATIVE_TERMINAL_FCFF":
        return ValuationStatus.FORECAST_PATH_INVALID
    if dcf_validation_status and dcf_validation_status not in (
            "DCF_VALID", "DCF_VALID_WITH_WARNINGS"):
        return ValuationStatus.MODEL_ARITHMETIC_INVALID
    if not dcf_available:
        return ValuationStatus.INPUT_PACKET_INVALID
    if financial_base_stale:
        # Ranked BELOW the causes above and ABOVE both LIMITED and
        # VALID_FOR_RESEARCH. A model that never applied, never ran, or broke
        # its own arithmetic is a more fundamental fact than one that ran
        # correctly on old data -- but running on old data still stops the
        # answer being published.
        return ValuationStatus.FINANCIAL_BASE_STALE
    if suitability_status in ("LIMITED", "NOT_SUITABLE"):
        return ValuationStatus.LIMITED
    return ValuationStatus.VALID_FOR_RESEARCH


# Conclusions that rest on a modelled value being trustworthy. None may be
# published unless the status is VALID_FOR_RESEARCH.
VALUATION_DERIVED_CONCLUSIONS = (
    "modeled_value_per_share", "market_price_premium_pct",
    "modeled_return_to_value_pct", "valuation_view",
    "market_exceeds_bull", "valuation_derived_risk",
)


def build_valuation_research_evidence(status: str, valuation: Optional[dict] = None,
                                      root_cause: Optional[str] = None) -> dict:
    """Part 5: what the research roles are allowed to see.

    On anything other than VALID_FOR_RESEARCH this returns the STATUS and the
    reason and nothing numeric. Not a filtered-down valuation -- no
    valuation. A role cannot misuse a figure it was never given, which is a
    stronger guarantee than asking it not to.
    """
    if status == ValuationStatus.VALID_FOR_RESEARCH:
        allowed = dict(valuation or {})
        allowed["valuation_status"] = status
        return allowed

    return {
        "valuation_status": status,
        "explanation": STATUS_EXPLANATION.get(status),
        "root_cause": root_cause,
        # Stated explicitly so a role does not infer that the absence of a
        # figure means the figure was zero or unremarkable.
        "valuation_conclusions_withheld": list(VALUATION_DERIVED_CONCLUSIONS),
    }


def may_publish_valuation_conclusion(status: str) -> bool:
    """Part 6, as one question with one answer."""
    return status == ValuationStatus.VALID_FOR_RESEARCH
