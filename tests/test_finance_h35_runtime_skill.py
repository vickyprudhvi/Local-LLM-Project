"""Phase H.35 — the runtime financial-evidence-analysis skill actually used
by the local LLM (not merely a Claude Code development aid).

Covers section 26's A-O checklist: the skill file loads and is injected into
every interpretive stage's prompt (never into a deterministic module, which
makes no LLM calls at all), a missing skill fails the pipeline closed, the
new bull/bear direction-mismatch and negative-modeled-equity checks, and
confirming coverage of the guidance/period/metric-identity and
recommendation-coherence rules Phase H.34 already built.
"""

import inspect

import pytest

import finance.research_pipeline as rp
from finance.claim_validation import (
    GUIDANCE_METRIC_MISMATCH,
    scan_for_guidance_metric_mismatch,
    scan_for_guidance_period_mismatch,
    scan_for_unsupported_claims,
)
from finance.evidence import build_evidence_index
from finance.skill_loader import (
    SkillLoadError,
    load_runtime_skill,
    skill_path,
    try_load_runtime_skill,
)
from tests.test_finance_research_pipeline import (
    INDEX,
    _final_raw,
    make_ask_local,
)

# =============================================================================
# A. the skill file loads
# =============================================================================

def test_skill_file_exists_at_the_tracked_runtime_location():
    path = skill_path("financial-evidence-analysis")
    assert path.exists(), path
    assert path.as_posix().endswith("skills/financial-evidence-analysis/SKILL.md")


def test_skill_loads_real_non_empty_content():
    text = load_runtime_skill()
    assert isinstance(text, str) and len(text) > 500
    assert "financial-evidence-analysis" in text.lower() or "evidence" in text.lower()


def test_skill_load_is_cached():
    # lru_cache -- the same object (not merely equal) on a second call.
    assert load_runtime_skill() is load_runtime_skill()


def test_try_load_returns_none_rather_than_raising_for_a_missing_skill():
    assert try_load_runtime_skill("does-not-exist-h35") is None
    with pytest.raises(SkillLoadError):
        load_runtime_skill("does-not-exist-h35")


# =============================================================================
# B. interpretive stages receive skill text / E. structured output still parses
# =============================================================================

def test_every_interpretive_stage_prompt_contains_the_real_skill_text():
    """Runs the real staged pipeline (scripted local-LLM responses, real
    skill loading) and checks the ACTUAL system prompt sent to each stage
    contains real skill content, not a paraphrase."""
    calls = []
    result = rp.run_research_pipeline(INDEX, make_ask_local(record=calls))
    assert result.available is True
    skill_text = load_runtime_skill()
    # A short, distinctive fragment that only appears in the real skill file.
    fragment = "DIRECT_FACT"
    assert fragment in skill_text
    assert calls, "no stage calls recorded"
    for messages, _kwargs in calls:
        system_text = messages[0]["content"]
        assert fragment in system_text, "a stage prompt is missing the real skill content"


def test_structured_bull_bear_output_still_parses_end_to_end():
    """E: the existing per-claim schema (claim_id/claim/evidence_ids/
    claim_type/assumptions/confidence) is the structured claim-proposal
    contract this phase requires reusing rather than replacing -- confirms it
    still round-trips through the real pipeline with the skill now loaded."""
    result = rp.run_research_pipeline(INDEX, make_ask_local())
    assert result.available is True
    bull = result.by_stage("bull_researcher").output
    assert len(bull["claims"]) >= 2
    for claim in bull["claims"]:
        assert claim["evidence_ids"]
        assert claim["claim_type"]


# =============================================================================
# C. deterministic stages never receive the skill (they make no LLM calls)
# =============================================================================

import re

# ACTUAL import statements only -- several of these modules legitimately
# MENTION "finance/research_pipeline.py" in an explanatory comment (e.g.
# "see finance/research_pipeline.py's _risk_reviewer_prompt, which already
# tells the model this"), which is cross-reference prose, not a dependency.
_IMPORT_RE = re.compile(r"^\s*(?:import\s+finance\.(research_pipeline|skill_loader)\b"
                        r"|from\s+finance\.(research_pipeline|skill_loader)\s+import\b"
                        r"|from\s+finance\s+import\s+(?:[\w, ]*\b(research_pipeline|skill_loader)\b))",
                        re.MULTILINE)


@pytest.mark.parametrize("module_name", [
    "finance.dcf", "finance.sec_provider", "finance.sec_datasets",
    "finance.guidance", "finance.actualization", "finance.metrics",
    "finance.reporting_currency", "finance.sec_normalization",
])
def test_deterministic_modules_never_import_the_skill_loader_or_pipeline(module_name):
    import importlib
    module = importlib.import_module(module_name)
    src = inspect.getsource(module)
    match = _IMPORT_RE.search(src)
    assert match is None, f"{module_name} must never import the skill loader or LLM pipeline: {match}"


def test_research_pipeline_module_never_imports_deterministic_calculation_layer():
    """Existing architectural guarantee (test_finance_research_pipeline.py's
    own test_module_never_imports_the_deterministic_calculation_layer),
    reasserted here as the OTHER half of 'the skill is for interpretation,
    not data processing': confirms the relationship is not accidentally
    bidirectional."""
    src = inspect.getsource(rp)
    assert "finance.dcf" not in src
    assert "finance.metrics" not in src


# =============================================================================
# D. missing skill fails closed
# =============================================================================

def test_missing_skill_fails_the_whole_interpretive_pipeline_closed(monkeypatch):
    monkeypatch.setattr(rp, "try_load_runtime_skill", lambda name="financial-evidence-analysis": None)
    result = rp.run_research_pipeline(INDEX, make_ask_local())
    assert result.available is False
    assert len(result.checkpoints) == len(rp._ALL_STAGE_NAMES)
    for checkpoint in result.checkpoints:
        assert checkpoint.status == rp.StageStatus.SKIPPED
        assert "skill" in (checkpoint.error or "").lower()


def test_missing_skill_never_falls_back_to_unrestricted_reasoning(monkeypatch):
    """No ask_local_fn call happens at all when the skill cannot be loaded --
    confirms this is a hard stop, not a degrade-and-continue."""
    calls = []

    def ask_local_fn(*args, **kwargs):
        calls.append(1)
        raise AssertionError("the local LLM must never be called when the skill is unavailable")

    monkeypatch.setattr(rp, "try_load_runtime_skill", lambda name="financial-evidence-analysis": None)
    result = rp.run_research_pipeline(INDEX, ask_local_fn)
    assert result.available is False
    assert calls == []


# =============================================================================
# F/G/H. bull/bear claim-direction mismatch (sections 15-16)
# =============================================================================

def test_bear_claim_with_favourable_direction_is_rejected():
    findings = rp._claim_direction_findings(
        {"claims": [{"claim": "Revenue increased 1.4% year over year."}]}, "bear")
    assert len(findings) == 1
    assert findings[0].rule_id == rp.CLAIM_DIRECTION_MISMATCH_RULE_ID


def test_bull_claim_with_unfavourable_direction_is_rejected():
    findings = rp._claim_direction_findings(
        {"claims": [{"claim": "Operating margin deteriorated sharply this quarter."}]}, "bull")
    assert len(findings) == 1
    assert findings[0].rule_id == rp.CLAIM_DIRECTION_MISMATCH_RULE_ID


def test_neutral_bare_level_claim_is_not_rejected_from_either_side():
    """G/H, narrow scope: a claim with no up/down language (UNCLASSIFIED) is
    NOT rejected -- the classifier's own documented scope is catching clear
    contradictions, not adjudicating every phrasing. 'Operating margin is
    20%' legitimately needs other evidence to BECOME bullish/bearish; it does
    not need to be blocked outright to satisfy that rule."""
    claim = {"claims": [{"claim": "Operating margin is 20%."}]}
    assert rp._claim_direction_findings(claim, "bull") == []
    assert rp._claim_direction_findings(claim, "bear") == []


def test_price_versus_value_claims_are_exempt_from_the_direction_check():
    """A price-exceeds-modeled-value claim is bearish (overvaluation) despite
    'exceeds' reading as favourable in the shared classifier's generic
    vocabulary -- exempted via the same _PRICE_TERMS vocabulary is_circular_
    price_condition already uses, rather than teaching the shared classifier
    price semantics it was never designed to carry."""
    assert rp._claim_direction_findings(
        {"claims": [{"claim": "The market price exceeds the modeled base scenario value."}]},
        "bear") == []


def test_direction_mismatch_quarantines_rather_than_fails_the_stage():
    """The finding is SEMANTIC_MISUSE severity (stubbed), consistent with
    every other claim-fidelity check in this pipeline -- one bad claim does
    not cost the whole stage."""
    from finance.content_policy import Severity
    findings = rp._claim_direction_findings(
        {"claims": [{"claim": "Revenue increased 1.4% year over year."}]}, "bear")
    assert findings[0].severity == Severity.SEMANTIC_MISUSE


# =============================================================================
# I. blank claims cannot reach the report
# =============================================================================

def test_a_blank_claim_fails_the_stage_rather_than_reaching_output():
    index = build_evidence_index({
        "symbol": "ZZZ", "quote": {"price": 10.0, "price_basis": "delayed"},
        "fundamental_metrics": {"revenue_growth_yoy": {"value": 0.05, "formula": "..."}},
        "technical_metrics": {}, "dcf": {"available": False, "reason": "DCF_ASSUMPTION_REQUIRED"},
        "warnings": [], "data_provenance": {}, "plan": {"omitted_datasets": []},
    })
    raw = {
        "thesis": "Revenue is growing at a modest pace.",
        "claims": [
            {"claim_id": "bull-1", "claim": "Revenue grew approximately 5% year over year.",
             "evidence_ids": ["fundamental.revenue_growth_yoy"], "claim_type": "fact_interpretation",
             "assumptions": [], "confidence": 0.6},
            {"claim_id": "bull-2", "claim": "   ", "evidence_ids": ["fundamental.revenue_growth_yoy"],
             "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.5},
        ],
        "confidence": "medium",
    }
    with pytest.raises(rp._Invalid):
        rp._validate_researcher_output(raw, index, "bull")


# =============================================================================
# J/K/L. guidance metric identity, period identity, metric identity
# (Phase H.34 behavior -- confirmed still intact under the skill-loader change)
# =============================================================================

def test_guidance_metric_identity_is_preserved():
    capex_only = {"dcf.guidance.capex.fy2027.current": {
        "target_period": "FY2027", "fiscal_period": "FY2027",
        "target_period_type": "CURRENT_FISCAL_YEAR"}}
    findings = scan_for_guidance_metric_mismatch(
        "Management guidance points to revenue growth next year.", capex_only)
    assert findings and findings[0].startswith(GUIDANCE_METRIC_MISMATCH)


def test_guidance_period_identity_is_preserved():
    q1_guidance = {"revenue_growth": {
        "name": "revenue_growth", "low": 0.27, "high": 0.29,
        "target_period": "Q1 FY2027", "fiscal_period": "Q1 FY2027",
        "target_period_type": "NEXT_QUARTER"}}
    findings = scan_for_guidance_period_mismatch(
        "Management guides 27-29% growth for fiscal year 2027.", q1_guidance)
    assert findings


def test_metric_identity_net_debt_is_not_confused_with_gross_debt_claims():
    """L: the deterministic single-metric rulebook (CV-708/709) blocks a
    leverage-distress conclusion regardless of which specific debt figure is
    named -- the canonical net_debt/gross-debt distinction is never collapsed
    by free text asserting a stronger conclusion than either number alone
    supports."""
    hits = scan_for_unsupported_claims(
        "Although net debt is negative, high debt may constrain capital allocation.")
    assert any("constrain capital allocation" in h for h in hits)


# =============================================================================
# M. negative modeled equity is not called a calculation error
# =============================================================================

def test_negative_modeled_equity_called_a_calculation_error_is_rejected():
    hits = scan_for_unsupported_claims(
        "The negative modeled equity value reflects a calculation error in the DCF.")
    assert any("calculation error" in h for h in hits)


def test_negative_modeled_equity_explained_plainly_is_accepted():
    assert scan_for_unsupported_claims(
        "Under this modeled forecast path, the resulting equity value is negative.") == []


# =============================================================================
# N. analysis limitation cannot become a company risk without independent
# evidence (Phase H.34's classify_risk_category, confirmed intact)
# =============================================================================

def test_missing_guidance_stays_an_analysis_limitation_not_a_company_risk():
    category = rp.classify_risk_category(
        "No current revenue guidance is available, which limits forward visibility.")
    assert category in ("ANALYSIS_LIMITATION", "DATA_QUALITY_RISK")


def test_invalid_dcf_stays_a_valuation_limitation_not_a_company_risk():
    category = rp.classify_risk_category(
        "The forecast path does not support a valid perpetuity terminal value.")
    assert category == "VALUATION_METHOD_LIMITATION"


# =============================================================================
# O. recommendation cannot be based solely on limitations
# (Phase H.34's _require_bearish_recommendation_has_directional_evidence)
# =============================================================================

def test_recommendation_cannot_rest_solely_on_valuation_unavailability():
    from tests.test_finance_research_pipeline import INVALID_DCF_INDEX

    raw = _final_raw("inconclusive", evidence_id="valuation.status", recommendation="avoid")
    with pytest.raises(rp._Invalid) as excinfo:
        rp._validate_final_synthesizer_output(raw, INVALID_DCF_INDEX)
    assert rp.CONTENT_POLICY_VIOLATION_MARKER in excinfo.value.message


def test_recommendation_with_real_company_evidence_is_accepted():
    from tests.test_finance_research_pipeline import INVALID_DCF_INDEX

    raw = _final_raw("inconclusive", evidence_id="fundamental.revenue_growth_yoy",
                     recommendation="avoid")
    validated = rp._validate_final_synthesizer_output(raw, INVALID_DCF_INDEX)
    assert validated["recommendation"] == "avoid"
