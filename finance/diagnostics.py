"""Phases 43-44 — report the cause, not the five things it caused.

One upstream semantic error used to produce a chain of downstream findings,
each true in isolation and each pointing somewhere unhelpful. The live shape
was:

    a quarterly guidance figure divided by trailing-twelve-month revenue
    -> a -73% growth rate
    -> the rate clamped to the model's -20% bound
    -> a model-bound conflict, because a bound was setting the forecast
    -> a valuation far below the market price
    -> HIGH risk
    -> a recommendation resting on all of it

Five findings, one bug. A reader shown the bound conflict would go looking at
the model's configuration, which was working exactly as designed. The
earlier phases stopped the bad derivation from being made at all; this
module handles what remains, which is that several of these codes can still
co-occur legitimately and only one of them is ever worth leading with.

The rule is narrow on purpose: a symptom is suppressed ONLY when its
specific cause is present in the same run. A model-bound conflict with no
guidance error above it is a real finding about the model and is reported
normally.
"""

from typing import Dict, Iterable, List, Optional, Tuple


# (symptom, cause) — the symptom is dropped when the cause is present.
#
# Each pair is a claim that the symptom cannot occur on its own path when the
# cause is what produced the value it fired on. Adding a pair here says "this
# finding tells the reader nothing they will not learn better from the
# other one", so the list stays short and every entry is justified.
_CAUSED_BY: Tuple[Tuple[str, str], ...] = (
    # An assumption clamped to a bound only because the derivation that fed
    # it was invalid. The bound behaved correctly.
    ("DCF_MODEL_BOUND_CONFLICT", "GUIDANCE_PERIOD_INCOMPATIBLE"),
    ("DCF_MODEL_BOUND_CONFLICT", "PERIOD_FREQUENCY_MISMATCH"),
    ("DCF_MODEL_BOUND_CONFLICT", "GUIDANCE_NOT_PERIOD_COMMITMENT"),
    # Phase H.14, section 15. A guidance metric that was never revenue growth
    # cannot produce a revenue-growth assumption, so anything measured
    # against that assumption -- a bound conflict, a divergence, a valuation
    # premium -- is describing a number that should not exist.
    ("DCF_MODEL_BOUND_CONFLICT", "GUIDANCE_METRIC_MISMATCH"),
    ("GUIDANCE_ASSUMPTION_DIVERGENCE", "GUIDANCE_METRIC_MISMATCH"),
    ("VALUATION_PREMIUM_HIGH", "GUIDANCE_METRIC_MISMATCH"),
    # A guidance/assumption divergence measured against a figure that should
    # never have become an assumption.
    ("GUIDANCE_ASSUMPTION_DIVERGENCE", "GUIDANCE_PERIOD_INCOMPATIBLE"),
    ("GUIDANCE_ASSUMPTION_DIVERGENCE", "GUIDANCE_NOT_PERIOD_COMMITMENT"),
    # A derived ratio flagged as stale when its components were themselves
    # refused for a period mismatch -- the mismatch is the thing to fix.
    ("DERIVED_METRIC_STALE_SOURCE", "DERIVED_RATIO_PERIOD_MISMATCH"),
    # Cash-flow claims for a business model whose standard FCFF input was
    # already declined; the declination is the finding.
    ("DCF_INPUT_NORMALIZATION_UNRESOLVED", "DCF_CASH_FLOW_NOT_STANDARD_FCFF"),
)

# Codes that describe a ROOT condition rather than a consequence. Ranked
# ahead of everything else when readiness explains itself, because they are
# what a reader would have to change to get a different answer.
ROOT_CODES = (
    "PERIOD_FREQUENCY_MISMATCH",
    "INCOMPATIBLE_PERIODS",
    "INCOMPATIBLE_METRICS",
    "INCOMPATIBLE_SHARE_BASIS",
    "INCOMPATIBLE_DEBT_BASIS",
    "INCOMPATIBLE_CURRENCY",
    "GUIDANCE_PERIOD_INCOMPATIBLE",
    "GUIDANCE_NOT_PERIOD_COMMITMENT",
    "GUIDANCE_METRIC_MISMATCH",
    "DCF_CASH_FLOW_NOT_STANDARD_FCFF",
    "DERIVED_RATIO_PERIOD_MISMATCH",
)


def suppressed_symptoms(codes: Iterable[str]) -> Dict[str, str]:
    """Which present codes are explained by another present code.

    Returns {symptom: cause}. A symptom suppressed by more than one cause
    reports the first match, which is enough -- the point is to name a
    cause, not to enumerate every path to it.
    """
    present = {c for c in (codes or ()) if c}
    out: Dict[str, str] = {}
    for symptom, cause in _CAUSED_BY:
        if symptom in present and cause in present and symptom not in out:
            out[symptom] = cause
    return out


def filter_to_root_causes(findings: Iterable[dict],
                          code_key: str = "code") -> List[dict]:
    """Drop findings whose cause is present in the same set.

    Order is preserved, and a finding with no recognised code is always
    kept: this suppresses what it can prove is derivative and never
    silences something it does not understand.
    """
    findings = list(findings or ())
    suppressed = suppressed_symptoms(f.get(code_key) for f in findings
                                     if isinstance(f, dict))
    if not suppressed:
        return findings

    kept = []
    for finding in findings:
        code = finding.get(code_key) if isinstance(finding, dict) else None
        if code in suppressed:
            continue
        kept.append(finding)
    return kept


def explain_suppression(symptom: str, cause: str) -> str:
    """Why a symptom was not shown, for the audit view."""
    return (f"{symptom} was not reported separately: it followed from {cause}, which is "
            f"the condition that actually needs attention. Fixing {cause} removes it.")


def rank_by_root_cause(codes: Iterable[str]) -> List[str]:
    """Root conditions first, then everything else in its original order."""
    codes = [c for c in (codes or ()) if c]
    roots = [c for c in codes if c in ROOT_CODES]
    rest = [c for c in codes if c not in ROOT_CODES]
    return roots + rest


# ---------------------------------------------------------------------------
# Phase 52 — the audit view
# ---------------------------------------------------------------------------

def build_audit_trail(facts: Optional[dict]) -> dict:
    """Debug-only: how a number became a valuation input.

    Reads what the pipeline already recorded rather than instrumenting it --
    every stage in this project already keeps its provenance, and the reason
    tracing a bug is slow is that the provenance is spread across eight
    facts keys. This gathers it into one shape, in pipeline order.

    Values only, never secrets: the inputs here are filing figures and
    period labels, and no provider credential or key reaches this dict.
    """
    facts = facts or {}
    state = facts.get("current_financial_state") or {}
    canonical = facts.get("canonical_evidence") or {}
    business = facts.get("business_model") or {}
    guidance = facts.get("guidance_matrix") or {}
    dcf = facts.get("dcf") or {}

    trail = {
        "1_normalized_facts": {
            "financial_as_of": state.get("financial_as_of"),
            "flows": {name: {"value": sel.get("value"),
                             "period": f"{sel.get('period_start')}..{sel.get('as_of_date')}",
                             "source": sel.get("source"),
                             "construction": (sel.get("ttm") or {}).get("construction_method"),
                             "validation": (sel.get("ttm") or {}).get("validation_status")}
                      for name, sel in (state.get("flows") or {}).items()
                      if isinstance(sel, dict)},
        },
        "2_canonical_current": {
            name: {"value": m.get("value"), "period": m.get("period"),
                   "derivation": m.get("derivation_formula") or m.get("definition"),
                   "sources": m.get("source_metrics")}
            for name, m in (canonical.get("current") or {}).items()
            if isinstance(m, dict)
        },
        "3_base_period": {
            "base_period": canonical.get("base_period"),
            "aligned": canonical.get("base_period_aligned"),
            "current_growth_kind": canonical.get("current_growth_kind"),
        },
        "4_semantic_rejections": list(facts.get("semantic_rejections") or []),
        "5_business_model": {
            "profile": business.get("profile"),
            "sic": business.get("sic"),
            "standard_fcff": business.get("standard_fcff_suitability"),
            "reasons": business.get("reasons"),
        },
        "6_guidance": {
            "coverage": guidance.get("guidance_coverage_status"),
            "dcf_coverage": guidance.get("dcf_guidance_coverage"),
            "absence_reason": guidance.get("absence_reason"),
            "current_rows": guidance.get("current_rows"),
        },
        "7_valuation": {
            "method_status": facts.get("valuation_method_status"),
            "dcf_available": dcf.get("available"),
            "dcf_unavailable_code": dcf.get("unavailable_code"),
            "dcf_validation_status": dcf.get("validation_status"),
            "suitability": (facts.get("dcf_suitability") or {}).get("dcf_suitability"),
        },
        "8_readiness": {
            "status": (facts.get("research_readiness") or {}).get("status"),
            "reasons": (facts.get("research_readiness") or {}).get("reasons"),
        },
    }

    codes = [r.get("code") for r in (facts.get("semantic_rejections") or [])
             if isinstance(r, dict)]
    suppressed = suppressed_symptoms(codes)
    if suppressed:
        trail["9_suppressed_symptoms"] = [
            explain_suppression(symptom, cause) for symptom, cause in suppressed.items()]
    return trail
