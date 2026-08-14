"""Phase H.2 — the staged multi-agent research pipeline.

Adapts the ROLE STRUCTURE of TauricResearch/TradingAgents (reviewed at pinned
commit `a33fd4c0f134485a43553a2c23a63cb14adbd88f`, Apache-2.0 —
https://github.com/TauricResearch/TradingAgents) — independent bull/bear
researchers, a research-manager reconciliation stage, and a risk-review stage
— WITHOUT importing it as a runtime dependency, WITHOUT LangGraph, and
WITHOUT its trader / simulated-exchange / order-execution behavior. See
`docs/PHASE_H2_RESEARCH_PIPELINE.md` for the full review and the specific,
itemized divergences from the original (structured+cited output at every
stage rather than only the decision stages; one BOUNDED rebuttal round
instead of an unbounded free-text debate loop; one unified risk reviewer over
the research instead of a 3-way debate over a trader's proposal; a
`FinalInvestmentSynthesizer` that never sees or produces a transaction
proposal, position size, or order).

Report-section structure (Investment Thesis / Bull Case / Bear Case /
explicit conflict-resolution rules / Risk / Valuation Summary) is aligned with
the InvestSkill `full-report` skill's structure as a reference — its Entry
Strategy / Exit Strategy sections (position sizing, stop-loss levels) are
deliberately NOT adopted; this project has no trading capability.

Every stage:
  * receives the SAME immutable, pre-rendered evidence-index STRING (a Python
    str, not a live mutable object — see finance/evidence.py) — never the raw
    payload, never a tool, never network access
  * must return strict JSON matching its own closed schema, validated in code
  * must cite evidence IDs (`evidence_ids`/`evidence_cited`, per stage) that
    are code-verified to exist in the index — an unverifiable citation fails
    validation
  * runs under a configured token budget (Ollama `options.num_predict`) and
    timeout
  * fails CLOSED: malformed/unparseable/uncited output produces a FAILED
    checkpoint, never a guessed substitute, and every downstream stage that
    depends on it is SKIPPED rather than fed a fabricated input
  * never touches `finance/dcf.py`, `finance/metrics.py`, or any other
    deterministic calculation — these stages are pure interpretation over
    already-computed facts

All SIX stages — every stage whose output includes model-authored free text,
which turns out to mean every stage including `rebuttal_round` — get exactly
ONE constrained repair attempt specifically for a content-policy/claim-
fidelity violation (`_run_stage_with_content_policy_repair`); every other
validation failure, on every stage, still fails closed on the first attempt
— including an ORDINARY schema/parse error (e.g. malformed or truncated
JSON), which is a different failure mode from a content-policy violation and
never gets a repair on any stage. See Phase H.3 corrective patch, Problem 1
(original, final_investment_synthesizer only); the COR corrective patch,
Phase 5-6 (extended to bull/bear); COR live verification (extended to
research_manager/risk_reviewer, once repeated live runs showed their own
free-text reconciliation carries the identical content-policy exposure and,
without it, a clean bull/bear pair could still cascade-fail the pipeline one
stage later); and the GE corrective patch (extended to rebuttal_round —
originally believed exempt because "the COR corrective patch's live
verification never observed it trip this specific scan", but a live GE run
showed rebuttal_round's own bull_rebuttal/bear_rebuttal text tripping the
causal-overreach scanner in 2 of 5 runs, with the pipeline's own no-repair
design turning a single flagged phrase into a hard stage failure every time
— the exact disproportionate cost this mechanism exists to avoid on every
other free-text stage. See also the SAME patch's negation/disclaimer-
awareness fixes in finance/content_policy.py and finance/claim_validation.py
-- the majority of GE's live violations, across every stage, were the model
correctly HEDGING or DISCLAIMING a banned claim ("not guaranteed", "no
DCF ... available to model ... intrinsic value", "no benchmark ... to prove
the price is unfair") rather than making it, which the scanners' original
bare word/phrase matching could not distinguish from the claim itself).
"""

import copy
import json
import re
import time
from dataclasses import dataclass, field

import finance.content_policy as cp_module

# The key `_validate_claim_fidelity` uses to hand quarantine records back to
# `_run_stage`, which lifts them onto the checkpoint and strips them. Never
# part of any stage's schema.
_QUARANTINE_KEY = "__quarantines__"
# Phase H.5, Phase 5a: the same smuggling contract for the full finding set.
_FINDINGS_KEY = "__findings__"
from typing import Dict, List, Optional, Tuple

import tools.config as config
from finance.evidence import EvidenceItem, render_evidence_index, validate_evidence_citations

PIPELINE_VERSION = "research_pipeline_v1"
TRADINGAGENTS_REVIEWED_COMMIT = "a33fd4c0f134485a43553a2c23a63cb14adbd88f"

_CONFIDENCE_LEVELS = ("low", "medium", "high")

# Phase H.3 corrective patch, Problem 1: prefixes a validation error raised
# for a content-policy/claim-fidelity violation specifically, so
# `_run_stage_with_content_policy_repair`'s one-repair-attempt wrapper (bull_
# researcher, bear_researcher, research_manager, risk_reviewer,
# final_investment_synthesizer -- COR corrective patch extended this from
# final_investment_synthesizer alone; GE corrective patch extended it again
# to rebuttal_round, the last remaining stage) can distinguish "reject and
# offer one repair" from every OTHER validation failure ("reject, no
# repair" — including an ordinary schema error on any of these six stages).
CONTENT_POLICY_VIOLATION_MARKER = "CONTENT_POLICY_VIOLATION:"


class StageStatus:
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class StageCheckpoint:
    """A workflow checkpoint: the recorded outcome of one pipeline stage.

    This IS the "checkpoint between stages" — every stage's result (or
    documented failure/skip reason) is captured here before the next stage
    ever runs, so the pipeline's progress is inspectable at every step rather
    than being one opaque end-to-end call.
    """

    stage: str
    status: str
    output: Optional[dict] = None
    error: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_seconds: float = 0.0
    # Phase H.5, Phase 2: fields removed from `output` because they tripped an
    # OVERSTATEMENT rule. Part of the run artifact, not a log line -- a
    # quarantine is a fact about what this analysis does and does not contain,
    # and the report has to be able to say so.
    quarantines: List[dict] = field(default_factory=list)
    # Phase H.5, Phase 3: structured detail about WHY a stage failed, used to
    # build a field-scoped correction without echoing the forbidden
    # vocabulary back to the model. None for failures that carry no
    # structured detail (the structural coherence guards, parse errors,
    # transport errors).
    failure_detail: Optional[dict] = None
    # Phase H.5, Phase 5a: EVERY finding this stage produced, each tagged
    # `quarantined` or `fatal`. `quarantines` above stays the acted-on
    # subset, so nothing downstream changes shape.
    findings: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage, "status": self.status, "output": self.output,
            "error": self.error, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "duration_seconds": round(self.duration_seconds, 3),
            "quarantines": [dict(q) for q in self.quarantines],
            "findings": [dict(f) for f in self.findings],
            # Part of the replay surface: which field failed, and what it
            # said with the offending span redacted.
            "failure_detail": dict(self.failure_detail) if self.failure_detail else None,
        }


@dataclass
class ResearchPipelineResult:
    """The full pipeline outcome: every checkpoint plus convenience accessors
    for the final report renderer."""

    available: bool
    checkpoints: List[StageCheckpoint] = field(default_factory=list)
    evidence_index_size: int = 0
    pipeline_version: str = PIPELINE_VERSION

    def by_stage(self, name) -> Optional[StageCheckpoint]:
        return next((c for c in self.checkpoints if c.stage == name), None)

    def output(self, name) -> Optional[dict]:
        checkpoint = self.by_stage(name)
        return checkpoint.output if checkpoint and checkpoint.status == StageStatus.COMPLETED else None

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.checkpoints)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.checkpoints)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "pipeline_version": self.pipeline_version,
            "tradingagents_reviewed_commit": TRADINGAGENTS_REVIEWED_COMMIT,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "evidence_index_size": self.evidence_index_size,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }


# ---------------------------------------------------------------------------
# Shared guardrails, identical across every stage's system prompt
# ---------------------------------------------------------------------------

_SHARED_GUARDRAILS = (
    "Rules that apply to every response you give:\n"
    "- You have NO TOOLS and NO NETWORK ACCESS. Your only source of facts is the "
    "EVIDENCE INDEX below. Do not use outside knowledge about this company beyond "
    "what is in the evidence index.\n"
    "- Evidence may come from more than one provider (some 'provenance.<dataset>.provider' "
    "entries may say \"yahoo\", \"sec\", or \"alphavantage\"). Attribute claims to their "
    "real source when it matters (e.g. an SEC filing fact vs. an unofficial Yahoo Finance "
    "figure) — never imply a single provider supplied everything.\n"
    "- Every claim you make must cite the evidence ID(s) it is based on, in the "
    "'evidence_cited' field(s) of your JSON. Citing an ID that is not in the "
    "evidence index is a validation failure — never invent or guess an ID.\n"
    "- You NEVER perform or alter any financial calculation. The DCF, metrics, and "
    "valuation figures in the evidence index are already final — you only interpret them.\n"
    "- Every 'dcf.value_per_share.*'/'dcf.assumption.*' figure is a MODELED value from one "
    "of three scenarios (base/bull/bear), never an objective or authoritative fact and "
    "never more reliable than the assumptions behind it. Call it 'base/bull/bear modeled "
    "value', 'modeled value per share', or 'valuation model output' ONLY. NEVER call it "
    "'intrinsic value', 'authoritative value', 'authoritative intrinsic value', 'price "
    "target', 'fair-value target', or 'consensus value' — no scenario outranks another, "
    "and this system has no analyst-consensus data source. 'dcf.assumption.*' evidence "
    "items carry WHY a scenario's value differs (source/derivation/scenario tags in "
    "brackets) — cite and explain those when comparing scenarios, not just the resulting "
    "value.\n"
    "- If 'dcf.validation_status' appears in the evidence index and is NOT 'DCF_VALID' or "
    "'DCF_VALID_WITH_WARNINGS', the DCF FAILED deterministic validation (see 'dcf."
    "validation_reasons' for exactly why — e.g. a non-monotonic bull/base/bear ordering or "
    "a negative terminal-year cash flow). In that case NO 'dcf.value_per_share.*'/'dcf."
    "assumption.*'/'dcf.enterprise_value.*'/'dcf.equity_value.*' evidence will be present at "
    "all (it is withheld, not merely unreliable) — do not reference, estimate, or assume any "
    "bull/base/bear modeled value, valuation gap, or scenario spread; state plainly that "
    "valuation evidence is unavailable because the model failed validation, and base your "
    "analysis on fundamentals/technicals/other available evidence instead.\n"
    "- A fundamental metric whose evidence value shows \"status\": \"not_meaningful\" (this "
    "happens to roe_ending_equity/roe_average_equity/debt_to_equity when shareholder "
    "equity is zero or negative) must NEVER be cited as a specific percentage/ratio or as "
    "evidence of profitability deterioration or leverage severity — the ratio is not "
    "computed at all in that case. Cite shareholder_equity itself (still available) and, "
    "for leverage, prefer total_debt/net_debt/debt_to_fcf/net_debt_to_fcf/current_ratio/"
    "operating_cash_flow when they are present in the evidence index instead.\n"
    # -- Phase H.4 (section 17): four KINDS of number, never interchangeable --
    #
    # The failure this prevents is subtle and was live: a report stating
    # "revenue growth is 7.2%" when 7.2% was a five-year historical CAGR and
    # the company had publicly guided to 2-3% for the current year. Both
    # numbers are true; presenting the first as the answer to "how fast is
    # this company growing" is not. The evidence index now labels each kind
    # separately (`source_type` on every assumption and guidance item), so
    # this rule is checkable against the evidence rather than a matter of
    # tone.
    "- Evidence comes in four DIFFERENT KINDS and you must never present one as another:\n"
    "  (a) REPORTED HISTORICAL data — what the company actually filed for a completed "
    "period. Attribute it to its period.\n"
    "  (b) TRAILING-TWELVE-MONTH (TTM) data — the last four quarters, which is more "
    "current than the last fiscal year and often differs from it materially. Say "
    "'trailing twelve months' when citing it; do not call it 'annual' or imply it is a "
    "fiscal-year result.\n"
    "  (c) CURRENT MANAGEMENT GUIDANCE ('dcf.guidance.*', source_type "
    "'management_guidance') — a FORWARD-LOOKING projection BY THE COMPANY. Always "
    "attribute it ('management guides to...', 'the company expects...'). NEVER state it "
    "as a reported fact or as something that has happened.\n"
    "  (d) DCF ASSUMPTIONS and CONFIGURED DEFAULTS ('dcf.assumption.*') — inputs chosen "
    "for the model, not observations. An assumption whose source_type is "
    "'configured_default' came from configuration, NOT from this company's data; say so "
    "when it matters to the conclusion.\n"
    "- When a historical rate and current guidance both exist and differ, state BOTH and "
    "say which one your reasoning relies on. 'Historical revenue CAGR was 7.2%, while "
    "current management guidance implies 2-3% near-term sales growth' is the right shape. "
    "'Revenue growth is 7.2%' is not.\n"
    "- 'dcf.valuation_freshness' tells you whether the valuation is built on the newest "
    "data available. If it is not CURRENT, the modeled value does not fully reflect the "
    "company's latest reported position — treat conclusions drawn from it with "
    "correspondingly less confidence, and say why.\n"
    # Phase H.5, Phase 4: the addressing rules are FABRICATION-class, so a
    # stage that breaks them fails outright. Stating the rule is therefore
    # not optional -- enforcing a constraint the model was never given is how
    # you manufacture failures.
    # Stated WITHOUT demonstrating the violation. An earlier draft spelled out
    # the banned pronouns and gave a worked counter-example -- which is the
    # same priming that made the old repair prompt escalate: showing the model
    # the exact construction is how it learns to write it.
    "- NEVER address the reader. Write ABOUT the company, in the third person, never TO a "
    "person. Use no second-person pronouns anywhere. Do not state what a reader, an "
    "investor or a shareholder ought to do, and do not open a sentence with an "
    "instruction to trade. This system cannot know who is reading, what they already own, or what "
    "their circumstances are, so any such sentence would be invented rather than "
    "analysed. Describe what the evidence shows and let the separate 'recommendation' "
    "field carry the characterization on its own.\n"
    "- Respond with ONLY a single JSON object matching the schema given. No prose "
    "before or after, no markdown code fences, no explanation of the schema.\n"
)


def _stage_prompt(role_instructions, schema_instructions, evidence_text, extra_context=""):
    system = f"{role_instructions}\n\n{_SHARED_GUARDRAILS}\n{schema_instructions}"
    user = f"EVIDENCE INDEX:\n{evidence_text}\n"
    if extra_context:
        user += f"\n{extra_context}\n"
    user += "\nRespond with ONLY the JSON object."
    return system, user


# ---------------------------------------------------------------------------
# Validation helpers — hand-rolled, matching the project's existing
# no-schema-library style (see finance/dcf.py's own validators)
# ---------------------------------------------------------------------------

class _Invalid(Exception):
    """A validation failure.

    Phase H.5, Phase 3: carries OPTIONAL structured detail alongside the
    human-readable message. The message keeps its exact historical shape
    because finance/workflow.py::_classify_content_policy_violation PARSES it
    (slicing between "free-text fields: " and ". Every claim must be") to
    produce the user-facing reason line -- changing the format would silently
    degrade that classification to its generic fallback.

    The structured detail exists so the REPAIR PROMPT no longer has to be
    built by echoing that message back. That echo is what produced the
    escalation observed live on WM: the message names the forbidden
    vocabulary, the model reads it, writes disclaimers using those words, and
    trips again with MORE violations than it started with (1, then 4, then 3,
    then the stage died).

    `field_path`/`redacted_text` are set only by the vocabulary-bearing
    producer (`_validate_claim_fidelity`). The eight structural coherence
    guards leave them None, and their messages ARE the fix -- "you said
    insufficient_data but recommended buy" tells the model exactly what to
    change and names nothing forbidden.
    """

    def __init__(self, message, field_path=None, redacted_text=None, findings=None):
        super().__init__(message)
        self.message = message
        self.field_path = field_path
        self.redacted_text = redacted_text
        # Phase H.5, Phase 5a: every finding that contributed to this failure,
        # in metrics shape. Previously a fatal finding survived only as prose
        # inside `message`, so the rule_id and field_path were lost -- meaning
        # the patterns MOST worth studying (the ones that kill stages) were
        # precisely the ones with no data behind them.
        self.findings = list(findings or [])

    @property
    def detail(self):
        if self.field_path is None:
            return None
        return {"field_path": self.field_path, "redacted_text": self.redacted_text}


def _str_field(d, key, max_len=1500):
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise _Invalid(f"'{key}' must be a non-empty string")
    return v.strip()[:max_len]


def _enum_field(d, key, allowed):
    v = d.get(key)
    if v not in allowed:
        raise _Invalid(f"'{key}' must be one of {sorted(allowed)}, got {v!r}")
    return v


def _evidence_list(d, key, index, min_items=1, max_items=15):
    v = d.get(key)
    if not isinstance(v, list) or len(v) < min_items:
        raise _Invalid(f"'{key}' must be a list of at least {min_items} evidence ID(s)")
    v = v[:max_items]
    ok, unknown = validate_evidence_citations(v, index)
    if not ok:
        raise _Invalid(f"'{key}' cites unknown evidence ID(s): {unknown}")
    return v


_CLAIM_TYPE_LEVELS = ("fact_interpretation", "scenario_interpretation", "risk_offset", "catalyst")


def _claims_list(d, key, index, min_items, max_items):
    """A bounded list of per-claim objects (COR corrective patch, Phase 3):
    {claim_id, claim, evidence_ids, claim_type, assumptions, confidence}.
    Replaces the old {point, evidence_cited} shape -- every material claim
    now carries its own type classification, the specific assumptions it
    depends on (if any), and its own confidence, rather than one bare
    sentence plus a citation list. `claim_id` must be unique within the
    response (a duplicate is a data-quality problem worth failing on, not
    silently accepting)."""
    v = d.get(key)
    if not isinstance(v, list) or not (min_items <= len(v)):
        raise _Invalid(f"'{key}' must be a list of at least {min_items} item(s)")
    v = v[:max_items]
    out = []
    seen_ids = set()
    for i, item in enumerate(v):
        if not isinstance(item, dict):
            raise _Invalid(f"'{key}[{i}]' must be an object")
        claim_id = _str_field(item, "claim_id", max_len=80)
        if claim_id in seen_ids:
            raise _Invalid(f"'{key}[{i}].claim_id' {claim_id!r} is a duplicate within this response")
        seen_ids.add(claim_id)
        out.append({
            "claim_id": claim_id,
            "claim": _str_field(item, "claim", max_len=600),
            "evidence_ids": _evidence_list(item, "evidence_ids", index, min_items=1, max_items=6),
            "claim_type": _enum_field(item, "claim_type", _CLAIM_TYPE_LEVELS),
            "assumptions": _string_list(item, "assumptions", max_items=5),
            "confidence": _float_field(item, "confidence"),
        })
    return out


def _string_list(d, key, min_items=0, max_items=6, max_len=400):
    v = d.get(key)
    if not isinstance(v, list):
        raise _Invalid(f"'{key}' must be a list")
    if len(v) < min_items:
        raise _Invalid(f"'{key}' must have at least {min_items} item(s)")
    return [str(x)[:max_len] for x in v[:max_items] if isinstance(x, str) and x.strip()]


def _has_material_omissions(index) -> bool:
    """True when any dataset was omitted from this analysis (Phase H.3
    corrective patch, Problem 10) -- see finance/evidence.py's
    `plan.omitted.<dataset>` entries, sourced from
    `AnalysisPlan.omitted_datasets`. This is the deterministic trigger for
    capping self-reported confidence rather than trusting the model to have
    lowered it on its own -- the prompts already ask for that (see
    `_final_synthesizer_prompt`); this is the backstop.
    """
    return any(eid.startswith("plan.omitted.") for eid in index)


_CONFIDENCE_ENUM_RANK = {"low": 0, "medium": 1, "high": 2}


def _cap_confidence_enum_for_omissions(confidence: str, index) -> str:
    """Clamps the Bull/Bear Researchers' low/medium/high confidence DOWN
    (never up, never rejected) to `config.research_reduced_mode_confidence_enum_cap()`
    when material datasets were omitted."""
    if not _has_material_omissions(index):
        return confidence
    cap = config.research_reduced_mode_confidence_enum_cap()
    if _CONFIDENCE_ENUM_RANK.get(confidence, 0) > _CONFIDENCE_ENUM_RANK.get(cap, 1):
        return cap
    return confidence


def _cap_confidence_for_omissions(confidence: float, index) -> float:
    """Clamps the FinalInvestmentSynthesizer's 0.0-1.0 confidence DOWN (never
    up, never rejected) to `config.research_reduced_mode_confidence_cap()`
    when material datasets were omitted."""
    if not _has_material_omissions(index):
        return confidence
    return min(confidence, config.research_reduced_mode_confidence_cap())


def _require_omission_disclosure(validated: dict, index) -> dict:
    """Problem 10: when material datasets were omitted from this analysis,
    the final research stance must say so explicitly -- not merely have its
    confidence capped underneath the reader's notice. Requires at least one
    'plan.omitted.*' evidence ID cited somewhere in 'rationale', and a
    non-empty 'key_uncertainties' -- both are things `_final_synthesizer_prompt`
    already asks for; this is the deterministic backstop. Raised under the
    SAME CONTENT_POLICY_VIOLATION_MARKER as the other free-text checks in
    this validator, so a violation here also gets the one repair attempt.
    """
    if not _has_material_omissions(index):
        return validated
    cited = {eid for item in validated["rationale"] for eid in item["evidence_ids"]}
    omitted_cited = any(eid.startswith("plan.omitted.") for eid in cited)
    if not omitted_cited or not validated["key_uncertainties"]:
        raise _Invalid(
            f"{CONTENT_POLICY_VIOLATION_MARKER} material datasets were omitted from this "
            "analysis (see the 'plan.omitted.*' evidence entries) but the response did not "
            "acknowledge this: 'rationale' must cite at least one 'plan.omitted.*' evidence "
            "ID, and 'key_uncertainties' must be non-empty. A reduced-data analysis must say "
            "so explicitly, never present itself with the same completeness as a full one.")
    return validated


def _require_valid_stance_when_dcf_invalid(validated: dict, index) -> dict:
    """TSLA DCF validation patch (section 10): when the DCF failed
    deterministic validation, 'valuation_view' is FORCED to 'model_invalid'
    regardless of what the model wrote -- this is an objective fact read
    straight from 'dcf.validation_status', not a judgment call the model
    could reasonably get right or wrong, so it is set in code exactly like
    `_cap_confidence_for_omissions` deterministically caps confidence rather
    than trusting the model to have done it.

    Separately (and only when the model's own research_stance needs
    correcting, not a guaranteed on-the-first-pass fix like valuation_view
    above): 'insufficient_data' is reserved for when the UNDERLYING data is
    genuinely thin or missing. A failed valuation MODEL with fundamentals/
    technicals still available in the evidence index is a DIFFERENT
    problem, and collapsing research_stance to 'insufficient_data' anyway
    would (a) be factually wrong (data is not insufficient, the model is)
    and (b) throw away a real characterization the fundamentals/technicals
    evidence could still support. Raised under CONTENT_POLICY_VIOLATION_
    MARKER for the SAME one-repair-attempt treatment `_require_omission_
    disclosure` already gets for an analogous "the model took the easy
    wrong-in-a-different-way answer" mistake, not because this is literally
    a banned-phrase scan.
    """
    if not _dcf_validation_failed(index):
        return validated
    validated = dict(validated)
    validated["valuation_view"] = "model_invalid"
    has_other_data = any(eid.startswith("fundamental.") or eid.startswith("technical.")
                         for eid in index)
    if has_other_data and validated["research_stance"] == "insufficient_data":
        raise _Invalid(
            f"{CONTENT_POLICY_VIOLATION_MARKER} research_stance was 'insufficient_data' but "
            "fundamentals and/or technical evidence IS present in the index -- the DCF "
            "valuation model failed deterministic validation (see 'dcf.validation_status'/"
            "'dcf.validation_reasons'), which is a DIFFERENT problem from the underlying data "
            "being insufficient. Use 'inconclusive' (grounded in whatever fundamentals/"
            "technicals ARE available) for research_stance instead, and keep valuation_view "
            "as 'model_invalid'.")
    return validated


# ---------------------------------------------------------------------------
# Phase H.5, Phase 2 — field-level quarantine
# ---------------------------------------------------------------------------
#
# Per-field policy. Explicit rather than inferred, because "what happens when
# this field is removed" is a different question for every field and getting
# it wrong is silent.
#
#   DROP    the field disappears. Safe for optional prose and for list
#           elements, where losing one item costs a detail, not a section.
#   STUB    the field is replaced with renderer-generated text saying it was
#           withheld. Required where something DOWNSTREAM reads the field --
#           `research_manager.balanced_assessment` is interpolated into both
#           the risk_reviewer and final_investment_synthesizer prompts, so
#           dropping it silently degrades two later stages rather than one
#           rendered paragraph.
#   FAIL    quarantine is not acceptable; the stage fails as before. Used
#           where the schema requires a minimum that quarantine would breach.
QUARANTINE_DROP = "drop"
QUARANTINE_STUB = "stub"
QUARANTINE_FAIL = "fail"

_QUARANTINE_STUB_TEXT = (
    "[withheld: this passage used language stronger than the cited evidence "
    "supports and was removed by automated screening]")

# Keyed by the FIELD NAME as it appears at the top level of a stage's
# validated output (list indices and nested keys are resolved to their root
# field before lookup). Anything not listed defaults to DROP.
_FIELD_QUARANTINE_POLICY = {
    # Load-bearing downstream -- see STUB above.
    "balanced_assessment": QUARANTINE_STUB,
    # Load-bearing for the report's own structure: a researcher with no
    # thesis, or a risk with no description, renders as an empty section.
    "thesis": QUARANTINE_STUB,
    "primary_reason": QUARANTINE_STUB,
    # Required NESTED scalars. DROP would delete `response` from inside
    # `bull_rebuttal`, leaving a dict the rebuttal schema says must have one
    # -- a structurally invalid output rather than a redacted one. Anything
    # whose absence breaks the shape gets stubbed, not dropped; DROP is for
    # list ELEMENTS and genuinely optional scalars.
    "bull_rebuttal": QUARANTINE_STUB,
    "bear_rebuttal": QUARANTINE_STUB,
    # `rationale` carries min_items=1 and each entry pairs a statement with
    # its evidence ids; quarantining the last one would leave a synthesis
    # asserting a recommendation with no stated reasoning at all.
    "rationale": QUARANTINE_FAIL,
    # `key_risks` has min_items=1 for the same reason.
    "key_risks": QUARANTINE_FAIL,
    # `claims` carries min_items=2.
    "claims": QUARANTINE_FAIL,
}


def _root_field(field_path: str) -> str:
    """`key_risks[0].risk` -> `key_risks`; `supported_bull_points[2]` -> the
    list name. Policy is keyed by the top-level field, since that is what
    determines whether anything downstream depends on it."""
    head = field_path.split(".", 1)[0]
    return head.split("[", 1)[0]


def _policy_for(field_path: str) -> str:
    return _FIELD_QUARANTINE_POLICY.get(_root_field(field_path), QUARANTINE_DROP)


def _remove_at_path(container, field_path: str, stub: bool) -> bool:
    """Remove (or stub) the value at `field_path`. True when it was applied.

    Walks the same path grammar `_walk_fields` produces: dotted keys and
    bracketed indices. Mutates `container` in place.
    """
    tokens = []
    for part in field_path.split("."):
        name, _, rest = part.partition("[")
        if name:
            tokens.append(("key", name))
        while rest:
            index, _, rest = rest.partition("]")
            if index:
                tokens.append(("index", int(index)))
            rest = rest.lstrip("[")
    if not tokens:
        return False

    parent = container
    for kind, token in tokens[:-1]:
        try:
            parent = parent[token]
        except (KeyError, IndexError, TypeError):
            return False
    kind, token = tokens[-1]
    try:
        if stub:
            parent[token] = _QUARANTINE_STUB_TEXT
        elif kind == "index":
            del parent[token]
        else:
            parent.pop(token, None)
    except (KeyError, IndexError, TypeError):
        return False
    return True


def apply_quarantine(validated: dict, findings) -> Tuple[dict, List[dict], List]:
    """Remove overstatement-flagged fields; return (output, records, fatal).

    `fatal` is every finding that must still fail the stage: any
    FABRICATION-severity finding, plus any overstatement landing on a field
    whose policy is FAIL.

    Deterministic: findings are applied deepest-path-first so removing a list
    element never invalidates a sibling's index, and records come out sorted
    by (field_path, rule_id).
    """
    fatal = [f for f in findings if f.severity == cp_module.Severity.FABRICATION]
    quarantinable = [f for f in findings if f.severity != cp_module.Severity.FABRICATION]

    fatal += [f for f in quarantinable if _policy_for(f.field_path) == QUARANTINE_FAIL]
    quarantinable = [f for f in quarantinable if _policy_for(f.field_path) != QUARANTINE_FAIL]
    if fatal:
        return validated, [], fatal

    output = copy.deepcopy(validated)
    records: List[dict] = []
    # Deepest first, then by descending list index, so earlier removals never
    # shift a path that has not been applied yet.
    ordered = sorted(quarantinable,
                     key=lambda f: (f.field_path.count(".") + f.field_path.count("["),
                                    f.field_path),
                     reverse=True)
    for finding in ordered:
        policy = _policy_for(finding.field_path)
        applied = _remove_at_path(output, finding.field_path, stub=(policy == QUARANTINE_STUB))
        if not applied:
            continue
        record = finding.to_dict()
        record["policy"] = policy
        records.append(record)
    records.sort(key=lambda r: (r["field_path"], r["rule_id"]))
    return output, records, []


_REDACTION = "[...]"

# Phase H.5, Phase 5a. The matched span is recorded so Phase 5b can report
# "this pattern fired, and on what text" -- deleting patterns without that is
# deleting them blind. Truncated because a span is model-authored text and
# only its shape matters here; field VALUES are never recorded, only paths.
_METRIC_SPAN_MAX = 40


def _finding_metric(finding, outcome: str) -> dict:
    return {
        "rule_id": finding.rule_id,
        "label": finding.label,
        "severity": finding.severity,
        "field_path": finding.field_path,
        "matched_span": (finding.matched_span or "")[:_METRIC_SPAN_MAX],
        "outcome": outcome,
    }


def _text_at_path(container, field_path: str):
    """The string a finding's `field_path` points at, or None."""
    node = container
    for part in field_path.split("."):
        name, _, rest = part.partition("[")
        if name:
            if not isinstance(node, dict) or name not in node:
                return None
            node = node[name]
        while rest:
            index_text, _, rest = rest.partition("]")
            if index_text:
                try:
                    node = node[int(index_text)]
                except (IndexError, TypeError, ValueError):
                    return None
            rest = rest.lstrip("[")
    return node if isinstance(node, str) else None


def _redact_span(text: str, span: str) -> str:
    """Replace the matched span with a placeholder.

    The model is shown WHERE the problem is without being shown WHAT the
    forbidden phrase was. Showing it is what taught the model the vocabulary
    it then used to write more violations.
    """
    if not text or not span:
        return text or ""
    return text.replace(span, _REDACTION)


def _validate_claim_fidelity(validated: dict, index) -> dict:
    """Deterministic backstop (Phase H.3 corrective patch, Problem 5): citing
    a REAL evidence ID (finance/evidence.py) does not stop a model from
    OVERSTATING what that evidence shows in the surrounding free text --
    "industry-leading ROE" and "fortress balance sheet" were both observed
    live, each citing a real evidence ID for an unremarkable metric. Applied
    to the validated output of EVERY stage (not only the
    FinalInvestmentSynthesizer), since a bull/bear researcher's `thesis` or
    `claim` flows into the rendered report just as directly.

    Combines the Problem 1 trade-advice scan (finance/content_policy.py) with
    the Problem 5 unsupported-claim scan (finance/claim_validation.py) under
    the SAME CONTENT_POLICY_VIOLATION_MARKER prefix, so
    `_run_stage_with_content_policy_repair`'s one-repair-attempt logic
    (Problem 1; extended from final_investment_synthesizer alone to also
    cover bull_researcher/bear_researcher, then research_manager/
    risk_reviewer, by the COR corrective patch, and finally rebuttal_round by
    the GE corrective patch) covers this class of violation on all SIX
    stages with no change to that wrapper.

    GE corrective patch: rebuttal_round was previously believed to be the
    one stage that never needed this — "a bounded two-response exchange, not
    an independent synthesis" — on the strength of the COR corrective
    patch's live verification never observing it trip this scan. A live GE
    run falsified that: bull_rebuttal/bear_rebuttal text tripped the
    causal-overreach scanner in 2 of 5 runs (e.g. "...confirms sustained
    momentum that supports the current valuation"), and with no repair path
    at the time, each one was a hard, unrecoverable stage failure. It now
    gets the SAME one repair attempt as every other stage.
    """
    from finance.content_policy import find_prohibited_directives_in_structure
    from finance.claim_validation import (
        find_unsupported_claims_in_structure,
        providers_present_in_index,
    )
    # HOOD corrective patch: when the DCF failed deterministic validation, the
    # evidence index itself is the ground truth that no genuine intrinsic/
    # authoritative/fair-value claim could be citing real evidence -- see the
    # long comment above finance/claim_validation.py's `dcf_invalid` param.
    findings = (
        find_prohibited_directives_in_structure(validated)
        + find_unsupported_claims_in_structure(
            validated, providers_present_in_index(index), dcf_invalid=_dcf_validation_failed(index))
    )

    # Phase H.5, Phase 2: OVERSTATEMENT findings quarantine their field; only
    # FABRICATION (and overstatement on a FAIL-policy field) still fails the
    # stage. The claim underneath a quarantined passage still had to cite a
    # real evidence id -- that check ran earlier in this same validator and
    # remains cascade-fatal -- which is why removing the prose is a
    # proportionate response rather than a loss of grounding.
    output, quarantine_records, fatal = apply_quarantine(validated, findings)

    # Phase H.5, Phase 5a: the complete finding set, each tagged with what
    # actually happened to it. This is the evidence base for deciding, in
    # Phase 5b, which of the 39 overstatement patterns have ever fired on
    # real text and which are dead weight -- a decision that cannot be made
    # from gating behaviour alone, because a pattern that never fires and a
    # pattern that fires constantly both just look like "no failures".
    fatal_keys = {(f.rule_id, f.field_path) for f in fatal}
    metrics_findings = [
        _finding_metric(f, "fatal" if (f.rule_id, f.field_path) in fatal_keys
                        else "quarantined")
        for f in findings
    ]
    metrics_findings.sort(key=lambda m: (m["field_path"], m["rule_id"]))

    if quarantine_records:
        # Carried on the returned dict; `_run_stage` lifts it onto the
        # checkpoint and strips it, so it never reaches the report as content.
        output[_QUARANTINE_KEY] = quarantine_records
    if metrics_findings:
        output[_FINDINGS_KEY] = metrics_findings
    if not fatal:
        return output

    violations = []
    for finding in fatal:
        if finding.label not in violations:
            violations.append(finding.label)
    # Phase H.5, Phase 3: the structured detail the repair prompt is built
    # from. The first fatal finding is used -- a correction that named every
    # offending field at once would be back to sending a list of problems,
    # and one field at a time is what keeps the instruction positive and
    # specific.
    primary = fatal[0] if fatal else None
    redacted = None
    if primary is not None:
        original = _text_at_path(validated, primary.field_path)
        redacted = _redact_span(original, primary.matched_span) if original else None
    if violations:
        raise _Invalid(
            f"{CONTENT_POLICY_VIOLATION_MARKER} prohibited or unsupported language detected in "
            "free-text fields: " + ", ".join(violations) +
            ". Every claim must be a plain, evidence-grounded observation: remove any order-shaped "
            "instruction (position size, entry/exit price, stop loss, target allocation), any "
            "phrasing conditioned on whether the reader holds a position ('if you hold' / 'if you "
            "do not hold' -- this system cannot know that), unsupported superlative (e.g. "
            "'fortress', 'industry-leading', 'best-in-class', 'guaranteed'), unqualified causal "
            "claim (e.g. 'confirms', 'proves', 'protects from downside'), consensus-estimate "
            "language (this system has no analyst-consensus data source), future-tense "
            "technical-signal certainty (e.g. 'will reverse', 'poised to rally'), or a named data "
            "provider that supplied nothing in this analysis. Qualified language is fine (e.g. "
            "'may provide flexibility', 'is consistent with'). A plain buy/hold/sell/avoid "
            "characterization is NOT itself a violation -- do not remove one on that basis.",
            field_path=(primary.field_path if primary is not None else None),
            redacted_text=redacted, findings=metrics_findings)
    return output


# ---------------------------------------------------------------------------
# Stage 1-2: independent bull / bear researcher
# ---------------------------------------------------------------------------

def _researcher_prompt(side, evidence_text):
    # Phase H.3 corrective patch (Problem 7): "compelling investment case",
    # "compelling short case", "upside/downside target", "buy opportunity",
    # "sell signal" and similar promotional/action-oriented phrasing are
    # replaced throughout with research-only vocabulary (bullish/bearish
    # INTERPRETATION, MODELED scenario, evidence supports/weakens, assumption
    # REQUIRED, valuation sensitivity) — this stage identifies what the
    # evidence shows, never what the reader should do.
    #
    # COR corrective patch (Phase 3): per-claim structure (claim_type,
    # explicit assumptions, per-claim confidence) plus an explicit
    # may-say/must-not-say vocabulary list, added after a live run showed
    # promotional language ("exceptional", "must hold that X") still
    # reaching the deterministic scanners rather than being avoided at the
    # source. The scanners remain the authoritative backstop either way —
    # this is about giving the model a clearer target, not relying on it.
    vocabulary = (
        "You may say things like: 'revenue growth is positive', 'free cash flow is "
        "positive', 'net cash may provide financial flexibility', 'price is above/below "
        "the 50-day moving average', 'RSI is 71.2', 'the bull scenario produces a modeled "
        "value of X under the listed assumptions', 'the bull scenario uses a higher "
        "revenue_growth and a lower wacc than base', 'upside depends on [specific "
        "assumption]', 'historical metrics support a favorable interpretation'. You must "
        "NOT say: 'compelling investment', 'must buy', 'downside protection', 'fortress "
        "balance sheet', 'industry-leading', 'target price', 'price target', 'fair-value "
        "target', 'intrinsic value', 'authoritative value', 'achievable price', 'confirms "
        "reversal', 'guarantees upside', 'strong buy', 'invest now', 'approaching support/"
        "resistance', or any claim that a technical reading implies a future trend, "
        "reversal, or breakout unless explicitly asked. For any claim about a DCF "
        "scenario, write it as 'the scenario produces a modeled value of X under the "
        "listed assumptions' — never 'the stock can reach X' or 'X is achievable', and "
        "never 'X is the intrinsic value'. If a metric's evidence value shows \"status\": "
        "\"not_meaningful\" (ROE or debt-to-equity with negative shareholder equity), do "
        "not cite it as a percentage or ratio at all — say it is not meaningful and, for "
        "leverage, cite total_debt/net_debt/debt_to_fcf/current_ratio/operating_cash_flow "
        "instead if they are in the evidence index."
    )
    if side == "bull":
        role = (
            "You are the Bull Researcher. Build the strongest EVIDENCE-BASED bullish "
            "INTERPRETATION of this company: favorable historical trends, financial "
            "strengths, potential catalysts, and the SPECIFIC assumptions that would need "
            "to hold for the bull-scenario DCF value to be plausible. You are building an "
            "independent interpretation — you have NOT seen any bear argument yet. Do NOT "
            "call the bull DCF scenario 'achievable' or a 'target' unless the evidence "
            "specifically supports the assumptions it requires — name those assumptions "
            "explicitly instead of asserting the outcome. This is a research interpretation, "
            "never investment advice, a recommendation to buy, or a 'buy opportunity'.\n"
            + vocabulary
        )
    else:
        role = (
            "You are the Bear Researcher. Build the strongest EVIDENCE-BASED bearish "
            "INTERPRETATION of this company: valuation risk, margin sensitivity, "
            "deceleration, leverage, technical weakness, data-quality gaps and missing "
            "information, and the conditions under which downside scenarios become "
            "plausible. You are building an independent interpretation — you have NOT seen "
            "any bull argument yet. This is a research interpretation, never investment "
            "advice: do NOT call this a 'short case', do NOT recommend shorting or any "
            "other position, and do NOT describe anything as a 'sell signal'.\n"
            + vocabulary
        )
    schema = (
        'JSON schema:\n'
        '{"thesis": "<1-3 sentence core interpretation>", '
        '"claims": [{"claim_id": "<short unique id, e.g. \'bull-1\'>", '
        '"claim": "<specific, evidence-based claim>", '
        '"evidence_ids": ["<evidence id>", ...], '
        '"claim_type": "fact_interpretation"|"scenario_interpretation"|"risk_offset"|"catalyst", '
        '"assumptions": ["<assumption this claim depends on>", ...] (may be empty), '
        '"confidence": <number 0.0 to 1.0>'
        '}, ... 2 to 5 items], '
        '"confidence": "low"|"medium"|"high"}\n\n'
        "'claim_type' guide: fact_interpretation = a plain reading of a reported figure; "
        "scenario_interpretation = describes what a DCF scenario shows or requires; "
        "risk_offset = a factor that may offset a risk (never framed as proof of safety); "
        "catalyst = a specific, evidence-grounded potential future driver."
    )
    return _stage_prompt(role, schema, evidence_text)


def _validate_researcher_output(raw, index, side) -> dict:
    if not isinstance(raw, dict):
        raise _Invalid("response was not a JSON object")
    validated = {
        "role": f"{side}_researcher",
        "thesis": _str_field(raw, "thesis", max_len=500),
        "claims": _claims_list(raw, "claims", index, min_items=2, max_items=5),
        "confidence": _cap_confidence_enum_for_omissions(
            _enum_field(raw, "confidence", _CONFIDENCE_LEVELS), index),
    }
    return _validate_claim_fidelity(validated, index)


# ---------------------------------------------------------------------------
# Stage 3: ONE bounded rebuttal round (both sides, one call)
# ---------------------------------------------------------------------------

def _rebuttal_prompt(evidence_text, bull_output, bear_output):
    role = (
        "You are facilitating exactly ONE bounded rebuttal round between the Bull "
        "and Bear Researchers below. This is the ONLY rebuttal round — do not "
        "continue the debate beyond this single exchange. For EACH side, write ONE "
        "rebuttal responding to the strongest point(s) the OTHER side made."
    )
    schema = (
        'JSON schema:\n'
        '{"bull_rebuttal": {"response": "<bull rebuts bear\'s strongest point>", '
        '"evidence_cited": ["<evidence id>", ...]}, '
        '"bear_rebuttal": {"response": "<bear rebuts bull\'s strongest point>", '
        '"evidence_cited": ["<evidence id>", ...]}}\n\n'
        # Found live (COR corrective patch): the model twice wrote
        # "bull_rebuttal"/"bear_rebuttal" as a bare STRING (the rebuttal text
        # itself, with the evidence IDs either dropped or hoisted to an
        # invented top-level "bull_evidence_cited" key instead) rather than
        # the required nested object -- a schema-following miss, not a
        # content-policy violation, so spelled out here explicitly rather
        # than left to the compact one-line shape above to convey alone.
        '"bull_rebuttal" and "bear_rebuttal" are each a JSON OBJECT with '
        'EXACTLY two keys, "response" and "evidence_cited" -- never a bare '
        'string, and never any other key name for the evidence list (not '
        '"bull_evidence_cited", not "sources", not "citations").'
    )
    extra = (
        f"BULL CASE:\nThesis: {bull_output['thesis']}\n"
        + "\n".join(f"- {c['claim']}" for c in bull_output["claims"])
        + f"\n\nBEAR CASE:\nThesis: {bear_output['thesis']}\n"
        + "\n".join(f"- {c['claim']}" for c in bear_output["claims"])
    )
    return _stage_prompt(role, schema, evidence_text, extra_context=extra)


def _validate_rebuttal_output(raw, index) -> dict:
    if not isinstance(raw, dict):
        raise _Invalid("response was not a JSON object")
    out = {}
    for side, key in (("bull_rebuttal", "response"), ("bear_rebuttal", "response")):
        entry = raw.get(side)
        if not isinstance(entry, dict):
            raise _Invalid(f"'{side}' must be an object")
        out[side] = {
            "response": _str_field(entry, key, max_len=800),
            "evidence_cited": _evidence_list(entry, "evidence_cited", index, min_items=1, max_items=6),
        }
    return _validate_claim_fidelity(out, index)


# ---------------------------------------------------------------------------
# Stage 4: Research Manager — evidence reconciliation
#
# Phase H.3 corrective patch (Problem 6): the ORIGINAL schema/prompt asked
# for a "stronger_case" verdict backed by invented decision doctrine
# ("consensus overrides an outlier", "fundamentals override technicals") —
# rules that are not facts in the evidence snapshot and, observed live,
# produced fabricated statistical framing (calling the base DCF scenario
# "consensus" and the bull scenario an "outlier" when they are simply three
# modeled scenarios with no such statistical relationship). Replaced with a
# schema that only asks the model to compare EVIDENCE STRENGTH and DATA
# QUALITY — never to apply an investment doctrine or declare a scenario
# authoritative.
# ---------------------------------------------------------------------------

def _research_manager_prompt(evidence_text, bull_output, bear_output, rebuttal_output):
    role = (
        "You are the Research Manager. Compare the bull case, the bear case, and the "
        "rebuttal round below PURELY on evidence strength and data quality. You are "
        "NOT selecting a trade recommendation, and you must NOT assume the base DCF "
        "scenario is 'consensus' or that the bull/bear scenarios are 'outliers' — they "
        "are three modeled scenarios with no statistical relationship of that kind. Do "
        "not apply any investment rule or doctrine that is not itself evidence in the "
        "index below (e.g. do not claim 'fundamentals override technicals' as a rule — "
        "only note what the EVIDENCE in this specific case actually shows)."
    )
    schema = (
        'JSON schema:\n'
        '{"evidence_balance": "bull_supported_more"|"bear_supported_more"|"mixed"|"insufficient_data", '
        '"supported_bull_points": ["<point that has real cited evidence behind it>", ...], '
        '"supported_bear_points": ["<point that has real cited evidence behind it>", ...], '
        '"unsupported_points": ["<point from either side that lacks real evidentiary support>", ...], '
        '"shared_findings": ["<something both sides effectively agree on>", ...], '
        '"key_disagreements": ["<a genuine, unresolved evidence-based disagreement>", ...], '
        '"assumption_sensitive_conclusions": ["<a conclusion that only holds under specific assumptions>", ...], '
        '"data_gaps": ["<missing or stale data that limits this analysis>", ...], '
        '"balanced_assessment": "<2-4 sentences: what the evidence actually shows, with no '
        'investment doctrine or trade recommendation of any kind>", '
        '"evidence_cited": ["<evidence id>", ...]}\n\n'
        "Every list may be empty ([]) if genuinely nothing applies — do not invent content "
        "to fill a list. NEVER suppress a genuine conflict: if bull and bear both have real "
        "evidence-backed points that disagree, say so in 'key_disagreements' rather than "
        "picking a side."
    )
    extra = (
        f"BULL CASE:\nThesis: {bull_output['thesis']}\n"
        + "\n".join(f"- {c['claim']}" for c in bull_output["claims"])
        + f"\n\nBEAR CASE:\nThesis: {bear_output['thesis']}\n"
        + "\n".join(f"- {c['claim']}" for c in bear_output["claims"])
    )
    if rebuttal_output:
        extra += (f"\n\nREBUTTAL ROUND:\nBull rebuttal: {rebuttal_output['bull_rebuttal']['response']}"
                  f"\nBear rebuttal: {rebuttal_output['bear_rebuttal']['response']}")
    return _stage_prompt(role, schema, evidence_text, extra_context=extra)


def _validate_research_manager_output(raw, index) -> dict:
    if not isinstance(raw, dict):
        raise _Invalid("response was not a JSON object")
    validated = {
        "evidence_balance": _enum_field(raw, "evidence_balance",
                                        ("bull_supported_more", "bear_supported_more",
                                         "mixed", "insufficient_data")),
        "supported_bull_points": _string_list(raw, "supported_bull_points", max_items=6),
        "supported_bear_points": _string_list(raw, "supported_bear_points", max_items=6),
        "unsupported_points": _string_list(raw, "unsupported_points", max_items=6),
        "shared_findings": _string_list(raw, "shared_findings", max_items=5),
        "key_disagreements": _string_list(raw, "key_disagreements", max_items=5),
        "assumption_sensitive_conclusions": _string_list(
            raw, "assumption_sensitive_conclusions", max_items=5),
        "data_gaps": _string_list(raw, "data_gaps", max_items=5),
        "balanced_assessment": _str_field(raw, "balanced_assessment", max_len=1200),
        "evidence_cited": _evidence_list(raw, "evidence_cited", index, min_items=1, max_items=15),
    }
    return _validate_claim_fidelity(validated, index)


# ---------------------------------------------------------------------------
# Stage 5: Risk Reviewer (unified — NOT a 3-way debate over a trade proposal)
# ---------------------------------------------------------------------------

def _risk_reviewer_prompt(evidence_text, research_manager_output):
    role = (
        "You are the Risk Reviewer. Review the Research Manager's reconciled view "
        "below and identify the specific risks to this analysis and to the "
        "investment case — including analytical/data-quality risk (stale or missing "
        "data, wide valuation-scenario spread, thin history), not just business risk. "
        "You are reviewing the RESEARCH, not a trade order — there is no position "
        "size or transaction to evaluate.\n"
        "Balance-sheet/leverage risk: debt_to_equity ALONE is never sufficient evidence "
        "of severe balance-sheet risk — it can be high mainly because the equity base "
        "is small, not because absolute debt is large. Where the evidence below "
        "includes them, also weigh net_debt, net_debt_to_fcf or debt_to_fcf (debt "
        "relative to cash-generating capacity), current_ratio, and operating cash "
        "flow before characterizing leverage risk as severe; cite interest_coverage "
        "only if it actually appears in the evidence (many filings do not report "
        "interest expense in a form this system captures). Never cite or imply a "
        "ratio that is not present in the evidence below. When roe_ending_equity, "
        "roe_average_equity, or debt_to_equity is ABSENT from the evidence (its "
        "\"status\" was \"not_meaningful\" because shareholder equity is zero or "
        "negative), that absence is NOT itself evidence of profitability deterioration "
        "or severe leverage — do not describe it as a large negative ratio (it was never "
        "computed) or treat the omission as a red flag on its own; cite total_debt, "
        "net_debt, debt_to_fcf, net_debt_to_fcf, current_ratio, or operating_cash_flow "
        "for leverage risk instead, and shareholder_equity's own (possibly negative) "
        "value if it is relevant to the risk."
    )
    schema = (
        'JSON schema:\n'
        '{"key_risks": [{"risk": "<specific risk>", "severity": "low"|"medium"|"high", '
        '"evidence_cited": ["<evidence id>", ...]}, ... 1 to 6 items], '
        '"data_quality_concerns": ["<string>", ... 0 to 6 items], '
        '"evidence_cited": ["<evidence id>", ...]}'
    )
    extra = (f"RESEARCH MANAGER ASSESSMENT:\nEvidence balance: "
            f"{research_manager_output['evidence_balance']}\n"
            f"{research_manager_output['balanced_assessment']}")
    return _stage_prompt(role, schema, evidence_text, extra_context=extra)


_RISK_SEVERITY_ORDER = ("low", "medium", "high", "very_high")


def aggregate_risk_level(severities) -> Optional[str]:
    """MLI corrective patch -- the DOCUMENTED, deterministic aggregation from
    per-risk severities to one overall risk level.

    The live MLI report listed HIGH, HIGH, MEDIUM, MEDIUM in its Risk section
    and then reported "Risk: moderate" in the final synthesis. That silent
    downgrade came from the FinalInvestmentSynthesizer picking its own
    `overall_risk` free of any tie to what the RiskReviewer had actually
    found. This function is the single authority instead, and its result is
    FORCED onto the final synthesis (see `_validate_final_synthesizer_output`)
    exactly the way `valuation_view` is forced to 'model_invalid' when the
    DCF fails -- an objective read of validated upstream output, not a
    judgment the model could reasonably get right or wrong.

    Policy (deliberately conservative -- a HIGH risk is never averaged away):
      * any very_high            -> very_high
      * two or more high         -> high
      * exactly one high         -> high
      * no high, any medium      -> moderate
      * only low                 -> low
    Returns None for an empty/unrecognized input so the caller can fall back
    rather than invent a level.
    """
    counts = {level: 0 for level in _RISK_SEVERITY_ORDER}
    for severity in severities or ():
        if severity in counts:
            counts[severity] += 1
    if not any(counts.values()):
        return None
    if counts["very_high"]:
        return "very_high"
    if counts["high"]:
        # One HIGH is enough. "Do not downgrade HIGH to MODERATE without an
        # explicit validated reason" -- averaging it against MEDIUMs would be
        # exactly that downgrade.
        return "high"
    if counts["medium"]:
        return "moderate"
    return "low"


def _validate_risk_reviewer_output(raw, index) -> dict:
    if not isinstance(raw, dict):
        raise _Invalid("response was not a JSON object")
    key_risks = []
    raw_risks = raw.get("key_risks")
    if not isinstance(raw_risks, list) or not raw_risks:
        raise _Invalid("'key_risks' must be a non-empty list")
    for i, item in enumerate(raw_risks[:6]):
        if not isinstance(item, dict):
            raise _Invalid(f"'key_risks[{i}]' must be an object")
        key_risks.append({
            "risk": _str_field(item, "risk", max_len=500),
            "severity": _enum_field(item, "severity", ("low", "medium", "high")),
            "evidence_cited": _evidence_list(item, "evidence_cited", index, min_items=1, max_items=6),
        })
    validated = {
        "key_risks": key_risks,
        "data_quality_concerns": _string_list(raw, "data_quality_concerns", max_items=6),
        "evidence_cited": _evidence_list(raw, "evidence_cited", index, min_items=1, max_items=15),
        # TSLA DCF validation patch (section 10): set DETERMINISTICALLY from
        # 'dcf.validation_status' in the evidence index, never left to the
        # model's own judgment -- a failed DCF is an objective fact the
        # RiskReviewer cannot be trusted to always flag on its own (and, per
        # the shared guardrails above, does not even see the invalid
        # scenario numbers to reason about in the first place).
        "model_risk": "high" if _dcf_validation_failed(index) else "not_applicable",
        # MLI corrective patch: the authoritative overall risk, aggregated
        # deterministically from this stage's OWN per-risk severities. The
        # final synthesis is forced to match it -- see `aggregate_risk_level`.
        "aggregated_risk": aggregate_risk_level([r["severity"] for r in key_risks]),
    }
    return _validate_claim_fidelity(validated, index)


# ---------------------------------------------------------------------------
# Stage 6: FinalInvestmentSynthesizer — research characterization PLUS one
# derived recommendation
#
# Phase H.3 corrective patch (Problem 1): the ORIGINAL schema
# (verdict_not_holding: BUY/HOLD_OFF/AVOID, verdict_holding: ADD/HOLD/SELL)
# was observed live producing exactly the holding-dependent trade-advice
# output this stage was always supposed to avoid ("If you do not currently
# hold a position: AVOID" / "If you already hold a position: SELL"). A
# disclaimer after those words is not sufficient — the schema itself must not
# have a slot for a verdict. Replaced with a pure research-characterization
# schema (stance/valuation view/risk, not a buy-or-sell decision), and
# additionally backstopped by a DETERMINISTIC content-policy scan (see
# finance/content_policy.py) applied to the validated output before it can
# ever become a COMPLETED checkpoint — see `run_research_pipeline`'s
# content-policy repair step, since a model can still write prohibited
# language into a free-text field even when the schema no longer has a
# dedicated verdict field for one.
#
# Recommendation reintroduction (personal-use corrective patch): the user
# explicitly requested a real recommendation for personal use, on THIS stage
# only (bull_researcher/bear_researcher/rebuttal_round/research_manager/
# risk_reviewer are UNCHANGED and stay pure research characterization). The
# new `recommendation` field is a DIFFERENT SHAPE from the removed one in two
# load-bearing ways: (1) it is ONE enum value, never conditioned on whether
# the reader holds a position — this system still has no way to know that,
# and holding-conditional branching is exactly what produced the original bad
# output, so it is not merely discouraged by prompt wording this time, it has
# no slot to express holding-conditionality in at all; (2) it is validated
# with the SAME strictness as every other field here, but deliberately kept
# OUT of the dict passed to `_validate_claim_fidelity` until every other
# field has already been scanned and accepted — see `_validate_final_
# synthesizer_output`. `finance/content_policy.py`'s BUY/SELL/HOLD/AVOID scan
# has no field-name exemption mechanism (by design, so it stays exactly as
# strict as today for every OTHER field on every stage, including every
# OTHER field on this one); this is what keeps `rationale`/`conditions_that_
# strengthen_the_view`/`conditions_that_weaken_the_view`/`key_uncertainties`
# just as unable to say "if you hold, sell" as prose as they were before —
# only the single literal enum value is exempt, not the ability to argue for
# an action in free text.
# ---------------------------------------------------------------------------

_RESEARCH_STANCE_LEVELS = ("positive", "cautiously_positive", "neutral",
                          "cautious", "negative", "inconclusive", "insufficient_data")
_VALUATION_VIEW_LEVELS = ("undervalued", "approximately_fair", "overvalued",
                         "highly_uncertain", "model_invalid", "insufficient_data")
_OVERALL_RISK_LEVELS = ("low", "moderate", "high", "very_high", "insufficient_data")
_RECOMMENDATION_LEVELS = ("buy", "hold", "sell", "avoid", "insufficient_evidence")

# TSLA DCF validation patch: DCF validation_status values that mean the
# result must NOT be used as valuation evidence -- kept in lockstep with the
# deterministic valuation engine's own INVALID status set (this module
# intentionally never imports that engine, to keep the research pipeline's
# only dependency the evidence index, per this file's own module docstring:
# "never touches [the DCF module] ... these stages are pure interpretation
# over already-computed facts" -- so the string literals are duplicated
# here, covered by test_finance_research_pipeline.py::
# test_dcf_invalid_statuses_match_finance_dcf_module).
_DCF_INVALID_VALIDATION_STATUSES = frozenset({
    "DCF_INVALID_INPUT", "DCF_INVALID_SCENARIO_ORDER", "DCF_NEGATIVE_TERMINAL_FCFF",
    "DCF_NONFINITE_OUTPUT", "DCF_EQUITY_BRIDGE_FAILURE", "DCF_ASSUMPTION_REQUIRED",
})


def _dcf_validation_status_from_index(index) -> Optional[str]:
    item = index.get("dcf.validation_status")
    return item.value if item else None


def _dcf_validation_failed(index) -> bool:
    return _dcf_validation_status_from_index(index) in _DCF_INVALID_VALIDATION_STATUSES


def _final_synthesizer_prompt(evidence_text, research_manager_output, risk_output):
    role = (
        # NOTE: this must keep starting with "You are the FinalInvestmentSynthesizer" --
        # that prefix is how the pipeline's own stage routing and every test
        # dispatcher identify this stage.
        "You are the FinalInvestmentSynthesizer, the final research synthesizer. Your job is "
        "to MAKE A RECOMMENDATION from "
        "the validated evidence presented to you. You are NOT required to agree with the base "
        "DCF scenario, the Bull Researcher, the Bear Researcher, or any single metric — weigh "
        "the totality of the evidence and choose the recommendation that best represents it.\n"
        "Consider, together and not in isolation: business fundamentals; growth quality; "
        "profitability; cash-flow generation; balance-sheet strength; leverage and liquidity; "
        "the current market valuation; the DCF's assumptions, sensitivity and terminal-value "
        "dependency; the bull and bear evidence; the Research Manager's reconciliation; the "
        "Risk Reviewer's conclusions; technical context; data completeness; provider "
        "conflicts; and assumption quality.\n"
        "No single signal decides this. An undervalued company may still warrant 'hold' or "
        "'avoid' when confidence, data quality or risk argue against acting; an apparently "
        "overvalued one may still warrant 'hold' when the model's assumptions look "
        "conservative against validated growth and cash-flow evidence. Say WHY in "
        "'primary_reason'. Do not invent facts. Do NOT recalculate any financial value — "
        "every percentage, ratio, growth rate, modeled value and valuation comparison you "
        "need is already in the evidence index; cite it rather than deriving your own. Your "
        "'confidence' must reflect genuine uncertainty and data quality, not how strongly "
        "the argument reads.\n"
        "PLUS produce the supporting research characterization (stance, valuation view, risk) "
        "alongside that recommendation. "
        "This is PERSONAL RESEARCH OUTPUT, never a trade order: never a position size, "
        "entry/exit price, stop-loss level, target allocation, or any other order-shaped "
        "instruction. 'recommendation' is never conditioned on whether the reader holds a "
        "position — this system has no way to know that, so it is the SAME single value "
        "either way; never write 'if you hold' or 'if you do not hold' anywhere in this "
        "response, in 'recommendation' or in any other field. Describe what the evidence "
        "supports FIRST — a stance on the business, a view on valuation, a risk level, a "
        "confidence level — and let 'recommendation' follow directly from those four fields; "
        "it is not a fifth, independent judgment and must not introduce any new consideration "
        "not already reflected in them."
    )
    schema = (
        'JSON schema:\n'
        '{"research_stance": "positive"|"cautiously_positive"|"neutral"|"cautious"|'
        '"negative"|"inconclusive"|"insufficient_data", '
        '"valuation_view": "undervalued"|"approximately_fair"|"overvalued"|'
        '"highly_uncertain"|"model_invalid"|"insufficient_data", '
        '"overall_risk": "low"|"moderate"|"high"|"very_high"|"insufficient_data", '
        '"confidence": <number 0.0 to 1.0>, '
        '"recommendation": "buy"|"hold"|"sell"|"avoid"|"insufficient_evidence", '
        '"primary_reason": "<1-2 sentences: the single most important reason THIS '
        'recommendation and not the adjacent one>", '
        '"supporting_factors": ["<evidence-grounded factor FOR the recommendation>", ... 0 to 4], '
        '"limiting_factors": ["<specific validated thing holding the recommendation back, or '
        'capping confidence>", ... 0 to 4], '
        '"rationale": [{"statement": "<evidence-grounded observation>", '
        '"evidence_ids": ["<evidence id>", ...]}, ... 1 to 6 items], '
        '"conditions_that_strengthen_the_view": ["<UPGRADE condition>", ... 0 to 2 items], '
        '"conditions_that_weaken_the_view": ["<DOWNGRADE condition>", ... 0 to 2 items], '
        '"reassessment_triggers": ["<new information whose DIRECTION is unknown>", ... 0 to 2], '
        '"risk_reconciliation_reason": "<REQUIRED only when overall_risk differs from the Risk '
        'Reviewer aggregate; omit otherwise>", '
        '"key_uncertainties": ["<string>", ... 0 to 5 items]}\n\n'
        "Condition direction is checked automatically. An UPGRADE condition must make the view "
        "MORE favourable ('operating margins stay above 15% while revenue growth stays above "
        "10%'); a DOWNGRADE condition LESS favourable ('free cash flow deteriorates while debt "
        "increases'); a REASSESSMENT TRIGGER is new information that could move the view EITHER "
        "way ('a company-specific WACC replaces the configured default', 'the share-count "
        "conflict is reconciled', 'a restated filing changes historical figures'). Note "
        "carefully: if the concern is OVERvaluation, a market price falling toward the modeled "
        "value IMPROVES the case — that is an upgrade condition, never a downgrade. Better "
        "model inputs are a reassessment trigger, not a downgrade, because they could move the "
        "conclusion either way. Every condition must relate to a material driver this analysis "
        "actually identified — never generic filler like 'market sentiment improves' or "
        "'volatility falls'.\n\n"
        "Calibrate 'confidence' (0.0-1.0) using scenario_spread.spread_pct_of_base if it is "
        "in the evidence index: a wide spread (bull and bear scenarios far apart) means LOW "
        "confidence regardless of direction. If material datasets were omitted (see "
        "'plan.omitted.*' evidence entries), lower confidence accordingly, cite at least one "
        "'plan.omitted.*' evidence ID in 'rationale', and describe the impact in "
        "'key_uncertainties' — never present a reduced-data analysis with the same confidence, "
        "or the same silence about what is missing, as a complete one (an automated check "
        "enforces this and will reject a response that omits it). If 'dcf.validation_status' "
        "shows the DCF failed validation (see the shared guardrails above), use "
        "'model_invalid' for valuation_view (an automated check enforces this regardless of "
        "what you write) — and prefer 'inconclusive' over 'insufficient_data' for "
        "research_stance when fundamentals/technicals evidence is still present: the "
        "valuation MODEL failing is a different problem from the underlying DATA being "
        "insufficient, and an automated check will reject 'insufficient_data' in that specific "
        "situation. Every list may be empty ([]) only when nothing genuinely applies.\n\n"
        "'recommendation' MUST follow directly from research_stance/valuation_view/"
        "overall_risk/confidence above — never a fifth, independent judgment. Guide: a "
        "'positive' or 'cautiously_positive' stance with an 'undervalued' (or at worst "
        "'approximately_fair') valuation, 'low' or 'moderate' risk, and confidence of "
        "roughly 0.5 or higher supports 'buy'. A 'negative' stance with an 'overvalued' "
        "valuation and confidence of roughly 0.5 or higher supports 'sell'. A 'cautious' or "
        "'negative' stance, an 'overvalued' or 'highly_uncertain' valuation, low confidence, "
        "or 'high'/'very_high' risk that together fall short of what 'sell' needs supports "
        "'avoid' — evidence the reader should stay away from, without the strength or "
        "confidence to call it 'sell'. A 'neutral' stance, an 'approximately_fair' "
        "valuation, mixed signals among the fields above, or confidence too low for a "
        "directional call supports 'hold'. If research_stance is 'insufficient_data', "
        "recommendation MUST also be 'insufficient_evidence' (an automated check enforces this). "
        "If research_stance is 'inconclusive' (DCF failed validation but fundamentals/"
        "technicals remain available), 'recommendation' may still be "
        "'buy'/'hold'/'sell'/'avoid' when fundamentals/technicals clearly support one — but "
        "'rationale' must draw ONLY on fundamentals/technicals/risk, never the failed "
        "valuation model, a modeled value, or the valuation gap as support, since "
        "valuation_view is 'model_invalid' and that evidence does not exist in this case.\n\n"
        "'primary_reason' must explain why THIS recommendation and not the adjacent one -- "
        "specifically, when the valuation looks favourable ('undervalued') but the "
        "recommendation is NOT 'buy', it must name what is holding it back, and that same "
        "thing must appear in 'limiting_factors'. An automated check REJECTS a "
        "favourable-valuation 'hold' with an empty 'limiting_factors'. Legitimate limiting "
        "factors are specific and validated, e.g. a research readiness of LIMITED, a DCF "
        "assumption clamped to a configured bound, a key assumption taken from a configured "
        "default because history was missing, a wide bull-to-bear scenario spread, an "
        "unresolved cross-provider data conflict, or high/very_high risk -- never a vague "
        "'market conditions'. 'supporting_factors' is the mirror: what genuinely argues FOR "
        "the recommendation. Both draw only on evidence in the index above."
    )
    extra = (f"RESEARCH MANAGER: evidence_balance={research_manager_output['evidence_balance']}; "
            f"{research_manager_output['balanced_assessment']}\n\n"
            f"RISK REVIEWER: {len(risk_output['key_risks'])} key risk(s) identified; "
            f"data quality concerns: {risk_output['data_quality_concerns']}")
    return _stage_prompt(role, schema, evidence_text, extra_context=extra)


def _rationale_list(d, key, index, min_items=1, max_items=6):
    v = d.get(key)
    if not isinstance(v, list) or len(v) < min_items:
        raise _Invalid(f"'{key}' must be a list of at least {min_items} item(s)")
    v = v[:max_items]
    out = []
    for i, item in enumerate(v):
        if not isinstance(item, dict):
            raise _Invalid(f"'{key}[{i}]' must be an object")
        out.append({
            "statement": _str_field(item, "statement", max_len=600),
            "evidence_ids": _evidence_list(item, "evidence_ids", index, min_items=1, max_items=6),
        })
    return out


def _float_field(d, key, lo=0.0, hi=1.0):
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= float(v) <= hi):
        raise _Invalid(f"'{key}' must be a number between {lo} and {hi}")
    return float(v)


# Valuation views that make a "why not buy?" question fair to ask.
_FAVOURABLE_VALUATION_VIEWS = frozenset({"undervalued"})
# Readiness levels that cannot support an action-implying recommendation.
_READINESS_BLOCKING_ACTION = frozenset({"NOT_READY"})
_ACTION_RECOMMENDATIONS = frozenset({"buy", "sell"})


def _require_recommendation_is_explained(recommendation: str, validated: dict) -> None:
    """MLI corrective patch (spec 3/14): a recommendation that does NOT follow
    the valuation's apparent direction has to say why.

    The live MLI report paired 'Valuation view: undervalued' with
    'Recommendation: HOLD' and offered no reason for the gap -- which is
    either a hidden limitation the reader deserved to see, or an
    unsupported downgrade. Both are defects, and the fix is the same:
    require the limitation to be stated. This does NOT force
    undervalued -> buy; it forces the model to name what it is trading off.
    """
    if recommendation in ("buy", "insufficient_evidence"):
        return
    if validated["valuation_view"] not in _FAVOURABLE_VALUATION_VIEWS:
        return
    if validated.get("limiting_factors"):
        return
    raise _Invalid(
        f"{CONTENT_POLICY_VIOLATION_MARKER} valuation_view is "
        f"{validated['valuation_view']!r} but recommendation is {recommendation!r} with an "
        "EMPTY 'limiting_factors'. A recommendation that does not follow the valuation's "
        "apparent direction must name what holds it back (research readiness, a clamped or "
        "defaulted DCF assumption, wide scenario spread, an unresolved data conflict, or "
        "elevated risk) in 'limiting_factors', and 'primary_reason' must explain it.")


# LLM-owns-the-recommendation patch: confidence ceiling a non-
# INSUFFICIENT_EVIDENCE recommendation must stay under when readiness is
# NOT_READY. The recommendation itself is NOT forced -- see below.
_NOT_READY_MAX_CONFIDENCE = 0.35


def _require_recommendation_supported_by_readiness(
        recommendation: str, validated: dict, readiness_status: Optional[str]) -> None:
    """Readiness constrains how CONFIDENTLY a recommendation may be held, and
    demands an explicit reason -- it never picks the recommendation.

    Superseded the previous hard block (buy/sell simply rejected under
    NOT_READY). That was a deterministic decision rule of exactly the kind
    this patch removes: a structurally broken DCF does not mean the
    remaining fundamentals/technicals evidence supports no view, only that
    any view drawn from it is weakly held. INSUFFICIENT_EVIDENCE remains the
    NORMAL answer for a NOT_READY analysis and needs no justification; any
    other recommendation stays available but must (a) name the readiness
    limitation in 'limiting_factors' and (b) carry low confidence.
    """
    if readiness_status not in _READINESS_BLOCKING_ACTION:
        return
    if recommendation == "insufficient_evidence":
        return
    if not validated.get("limiting_factors"):
        raise _Invalid(
            f"{CONTENT_POLICY_VIOLATION_MARKER} research readiness is {readiness_status!r} "
            f"(a failed DCF or an unresolved core reconciliation conflict) but recommendation "
            f"{recommendation!r} lists no 'limiting_factors'. 'insufficient_evidence' is the "
            "normal answer here; any other recommendation is permitted only when the readiness "
            "limitation is named explicitly and confidence is low.")
    if validated["confidence"] > _NOT_READY_MAX_CONFIDENCE:
        raise _Invalid(
            f"{CONTENT_POLICY_VIOLATION_MARKER} research readiness is {readiness_status!r} but "
            f"confidence is {validated['confidence']:.2f}. A recommendation other than "
            f"'insufficient_evidence' on a structurally invalid analysis must be held at or "
            f"below {_NOT_READY_MAX_CONFIDENCE:.2f} confidence.")


# Confidence ceiling when the valuation is knowingly built on stale inputs.
# Higher than the NOT_READY ceiling: a stale input degrades a valuation, it
# does not structurally invalidate it the way a failed DCF does.
_STALE_VALUATION_MAX_CONFIDENCE = 0.55

_STALE_FRESHNESS_STATES = ("STALE_INPUT_WARNING", "STALE_INVALID")


def _require_recommendation_reflects_valuation_freshness(
        recommendation: str, validated: dict, index) -> None:
    """Phase H.4, section 18. Freshness bounds CONFIDENCE, never the choice.

    The AOS case is the argument for this existing at all: every requested
    dataset was present, the DCF validated cleanly, and the analysis would
    have reported full confidence in a valuation whose equity bridge was six
    months and one acquisition out of date. Nothing in the pipeline could
    express "complete, and also wrong". `dcf.valuation_freshness` can, and
    this makes the synthesizer answer for it.
    """
    item = (index or {}).get("dcf.valuation_freshness")
    freshness = getattr(item, "value", None)
    if freshness not in _STALE_FRESHNESS_STATES:
        return
    if recommendation == "insufficient_evidence":
        return
    if not validated.get("limiting_factors"):
        raise _Invalid(
            f"{CONTENT_POLICY_VIOLATION_MARKER} the valuation freshness is {freshness!r} "
            f"(the model does not use the newest reported data for this company) but "
            f"recommendation {recommendation!r} lists no 'limiting_factors'. Name the stale "
            "input explicitly as a limiting factor, or use 'insufficient_evidence'.")
    if validated["confidence"] > _STALE_VALUATION_MAX_CONFIDENCE:
        raise _Invalid(
            f"{CONTENT_POLICY_VIOLATION_MARKER} the valuation freshness is {freshness!r} but "
            f"confidence is {validated['confidence']:.2f}. A recommendation resting on a "
            f"valuation that does not reflect the latest reported data must be held at or "
            f"below {_STALE_VALUATION_MAX_CONFIDENCE:.2f} confidence.")


def _require_risk_reconciled_with_reviewer(validated: dict, risk_output: Optional[dict]) -> dict:
    """Spec 13: the synthesizer MAY reach a different overall risk than the
    RiskReviewer's aggregate, but never silently.

    Superseded the previous hard override (overall_risk forced to the
    aggregate unconditionally). That guaranteed consistency but removed the
    synthesizer's ability to reason -- and there are legitimate cases, e.g.
    two HIGH flags that are the same underlying DCF-sensitivity issue
    counted twice rather than two independent risks. Differing now requires
    a stated 'risk_reconciliation_reason'; matching requires nothing.
    """
    aggregated = (risk_output or {}).get("aggregated_risk")
    if not aggregated or validated["overall_risk"] == aggregated:
        return validated
    if not validated.get("risk_reconciliation_reason"):
        raise _Invalid(
            f"{CONTENT_POLICY_VIOLATION_MARKER} overall_risk is "
            f"{validated['overall_risk']!r} but the RiskReviewer's own severities aggregate to "
            f"{aggregated!r}. Differing is allowed, silently differing is not: either use "
            f"{aggregated!r}, or supply a 'risk_reconciliation_reason' explaining -- from the "
            "evidence -- why the aggregate overstates the real risk (for example, several "
            "flags tracing to one underlying issue rather than independent risks).")
    return validated


# --- Spec 8/9: condition direction ---------------------------------------
#
# An upgrade condition must be something that would make the view MORE
# favourable, a downgrade condition LESS favourable, and a reassessment
# trigger something that could move it EITHER way. The live MLI report got
# this wrong in a specific, instructive way: it listed "market price
# converges to base modeled value" as a DOWNGRADE condition while its own
# concern was overvaluation -- but a price falling toward the modeled value
# IMPROVES the valuation case. It also listed better model inputs as a
# downgrade, when better inputs could move the conclusion either way.
#
# Deliberately NOT bare keyword matching (spec 9 rules that out): the
# BIDIRECTIONAL test runs FIRST and wins, because "X becomes available" /
# "X is reconciled" / "restated" describe an information change whose
# DIRECTION is unknown -- regardless of how favourable-sounding the
# surrounding words are. Only after that does directional vocabulary apply,
# and an unclassifiable condition is ACCEPTED rather than rejected (the
# scanner is a guard against clear contradictions, not a style checker).
_BIDIRECTIONAL_MARKERS = (
    "becomes available", "become available", "is reconciled", "are reconciled",
    "is resolved", "are resolved", "is updated", "are updated", "is replaced",
    "are replaced", "replace ", "restate", "clarif", "new filing", "is disclosed",
    "are disclosed", "is published", "recalculated", "re-calculated", "revised",
    "either direction", "company-specific wacc", "is confirmed either way",
)
# Only words whose direction holds REGARDLESS of subject. Deliberately
# excludes "increase"/"rises"/"higher"/"above"/"falls"/"lower"/"below":
# rising REVENUE is favourable but rising DEBT is not, and a price FALLING
# is favourable when the concern is overvaluation. Including them made
# "free cash flow deteriorates while debt increases" score favourable and
# unfavourable simultaneously, cancelling to UNCLASSIFIED. Subject-aware
# scoring is not reachable with this approach, so ambiguous words are left
# out and the classifier falls back to UNCLASSIFIED (which is accepted).
_FAVOURABLE_MARKERS = (
    "improve", "improving", "expand", "expanding", "accelerat", "strengthen",
    "strengthening", "outperform", "recover", "beat", "exceed", "sustain",
)
_UNFAVOURABLE_MARKERS = (
    "deteriorat", "decline", "declining", "weaken", "worsen", "contract",
    "compress", "shortfall", "underperform", "erode", "breach", "impair",
)
CONDITION_FAVOURABLE = "FAVORABLE"
CONDITION_UNFAVOURABLE = "UNFAVORABLE"
CONDITION_BIDIRECTIONAL = "BIDIRECTIONAL"
CONDITION_UNCLASSIFIED = "UNCLASSIFIED"


def classify_condition_direction(text: str) -> str:
    """FAVORABLE / UNFAVORABLE / BIDIRECTIONAL / UNCLASSIFIED for one
    condition. BIDIRECTIONAL wins outright -- an information change ("a
    company-specific WACC becomes available", "the share-count conflict is
    reconciled") has no known direction even when its wording sounds
    positive. UNCLASSIFIED means "no clear signal", and callers accept it."""
    if not isinstance(text, str) or not text.strip():
        return CONDITION_UNCLASSIFIED
    lowered = text.lower()
    if any(marker in lowered for marker in _BIDIRECTIONAL_MARKERS):
        return CONDITION_BIDIRECTIONAL
    favourable = any(marker in lowered for marker in _FAVOURABLE_MARKERS)
    unfavourable = any(marker in lowered for marker in _UNFAVOURABLE_MARKERS)
    if favourable and not unfavourable:
        return CONDITION_FAVOURABLE
    if unfavourable and not favourable:
        return CONDITION_UNFAVOURABLE
    return CONDITION_UNCLASSIFIED


_CONDITION_BUCKETS = {
    CONDITION_FAVOURABLE: "conditions_that_strengthen_the_view",
    CONDITION_UNFAVOURABLE: "conditions_that_weaken_the_view",
    CONDITION_BIDIRECTIONAL: "reassessment_triggers",
}
_CONDITION_BUCKET_CAP = 2


def _route_conditions_by_direction(validated: dict) -> dict:
    """Spec 9: upgrade -> FAVORABLE, downgrade -> UNFAVORABLE, reassessment
    -> BIDIRECTIONAL. A misfiled condition is MOVED to the bucket its
    direction actually implies -- never rejected.

    Rejecting was the first design and it was wrong. Direction here is a
    HEURISTIC judgment (`classify_condition_direction`), and wiring a
    heuristic to a hard failure is the exact pattern that repeatedly took
    this pipeline down over single phrases: a live AMZN run filed "reported
    CapEx history becomes available" as an upgrade condition -- a defensible
    reading, but BIDIRECTIONAL by this spec's definition -- and the whole
    final_investment_synthesizer stage died after its one repair attempt.
    Moving it achieves precisely the outcome the spec asks for (each list
    holds conditions of the right direction) with NO failure mode, and
    without discarding a condition the model correctly identified as
    material. An UNCLASSIFIED condition stays where the model put it.

    Caps are re-applied after routing so a bucket cannot overflow.
    """
    routed = {key: [] for key in _CONDITION_BUCKETS.values()}
    for expected, source_key in (
            (CONDITION_FAVOURABLE, "conditions_that_strengthen_the_view"),
            (CONDITION_UNFAVOURABLE, "conditions_that_weaken_the_view"),
            (CONDITION_BIDIRECTIONAL, "reassessment_triggers")):
        for condition in validated.get(source_key) or []:
            actual = classify_condition_direction(condition)
            destination = (source_key if actual in (CONDITION_UNCLASSIFIED, expected)
                          else _CONDITION_BUCKETS[actual])
            if condition not in routed[destination]:
                routed[destination].append(condition)
    validated = dict(validated)
    for key, items in routed.items():
        validated[key] = items[:_CONDITION_BUCKET_CAP]
    return validated


def _require_recommendation_consistent_with_stance(recommendation: str, validated: dict) -> str:
    """A recommendation can never be MORE confident than the research_stance
    it is derived from -- an objective consistency check, enforced in code
    for the same reason `_require_valid_stance_when_dcf_invalid` is: not a
    judgment call the model could reasonably get right or wrong. Raised under
    CONTENT_POLICY_VIOLATION_MARKER so this gets the same one-repair-attempt
    treatment as every other guard in this function."""
    if (validated["research_stance"] == "insufficient_data"
            and recommendation != "insufficient_evidence"):
        raise _Invalid(
            f"{CONTENT_POLICY_VIOLATION_MARKER} research_stance was 'insufficient_data' but "
            f"'recommendation' was {recommendation!r} -- a recommendation can never be more "
            "confident than the research stance it is derived from. Use 'insufficient_evidence' "
            "for 'recommendation' too.")
    return recommendation


def _validate_final_synthesizer_output(raw, index, risk_output=None,
                                       readiness_status=None) -> dict:
    if not isinstance(raw, dict):
        raise _Invalid("response was not a JSON object")
    validated = {
        "research_stance": _enum_field(raw, "research_stance", _RESEARCH_STANCE_LEVELS),
        "valuation_view": _enum_field(raw, "valuation_view", _VALUATION_VIEW_LEVELS),
        "overall_risk": _enum_field(raw, "overall_risk", _OVERALL_RISK_LEVELS),
        "confidence": _cap_confidence_for_omissions(_float_field(raw, "confidence"), index),
        "primary_reason": _str_field(raw, "primary_reason", max_len=600),
        "supporting_factors": _string_list(raw, "supporting_factors", max_items=3),
        "limiting_factors": _string_list(raw, "limiting_factors", max_items=3),
        "rationale": _rationale_list(raw, "rationale", index, min_items=1, max_items=6),
        # Spec 8/15: upgrade/downgrade/reassessment, capped at 2 each.
        # `conditions_that_*` are kept as the on-disk names so every existing
        # consumer/fixture keeps working; `reassessment_triggers` is new and
        # is what an information-change condition (whose direction is
        # unknown) belongs in -- see `classify_condition_direction`.
        "conditions_that_strengthen_the_view": _string_list(
            raw, "conditions_that_strengthen_the_view", max_items=2),
        "conditions_that_weaken_the_view": _string_list(
            raw, "conditions_that_weaken_the_view", max_items=2),
        # OPTIONAL: a response with no bidirectional trigger to report is
        # perfectly valid, so a missing key is [] rather than a schema error.
        "reassessment_triggers": (
            _string_list(raw, "reassessment_triggers", max_items=2)
            if isinstance(raw.get("reassessment_triggers"), list) else []),
        "key_uncertainties": _string_list(raw, "key_uncertainties", max_items=5),
        # Spec 13: only required when the synthesizer's risk differs from the
        # RiskReviewer's aggregate; absent/empty otherwise.
        "risk_reconciliation_reason": (raw.get("risk_reconciliation_reason") or "").strip()[:400]
        if isinstance(raw.get("risk_reconciliation_reason"), str) else "",
    }
    # Recommendation reintroduction: validated with the SAME strictness as
    # every field above (missing/invalid -> ordinary schema error, fails
    # closed, no repair -- exactly like a bad research_stance) but
    # DELIBERATELY KEPT OUT of `validated` until every check below (including
    # `_validate_claim_fidelity`) has run. finance/content_policy.py's scan
    # has no field-name exemption mechanism -- it walks every string in
    # whatever structure it's given, and 'buy'/'sell'/'hold'/'avoid' are
    # exactly what it exists to catch. Keeping this one value out of the
    # dict that scanner ever sees (rather than teaching it a field-name
    # allowlist) keeps that shared, all-six-stage infrastructure exactly as
    # strict as it is today for every OTHER field, on every stage, including
    # every other field on THIS stage. See the module docstring's
    # "Recommendation reintroduction" note above Stage 6.
    recommendation = _enum_field(raw, "recommendation", _RECOMMENDATION_LEVELS)
    # Problem 10: when material datasets were omitted, the response must say
    # so explicitly (a cited plan.omitted.* ID plus a non-empty
    # key_uncertainties) -- raises under the same repairable marker as the
    # checks below when it doesn't.
    validated = _require_omission_disclosure(validated, index)
    # TSLA DCF validation patch: forces valuation_view='model_invalid' when
    # the DCF failed validation, and rejects (for one repair attempt) a
    # research_stance of 'insufficient_data' when fundamentals/technicals
    # are actually available -- see the function's own docstring.
    validated = _require_valid_stance_when_dcf_invalid(validated, index)
    # Spec 13: the synthesizer may differ from the RiskReviewer's aggregate,
    # but must say why. NOT silently overridden any more -- see the
    # function's own docstring for why the previous hard force was wrong.
    validated = _require_risk_reconciled_with_reviewer(validated, risk_output)
    # A recommendation of anything but 'insufficient_evidence' is rejected
    # (one repair attempt) when research_stance is itself 'insufficient_data'.
    recommendation = _require_recommendation_consistent_with_stance(recommendation, validated)
    # Spec 3/7/9/12: NONE of these pick the recommendation -- they require the
    # LLM's own choice to be internally coherent. Each raises under
    # CONTENT_POLICY_VIOLATION_MARKER, so each gets the same single repair
    # attempt as every other guard here.
    _require_recommendation_is_explained(recommendation, validated)
    _require_recommendation_supported_by_readiness(recommendation, validated, readiness_status)
    # Phase H.4 (section 18): a stale-input valuation caps confidence too.
    # Same shape as the readiness cap and for the same reason -- freshness
    # never picks the recommendation, it only bounds how firmly one can be
    # held and requires the limitation to be named.
    _require_recommendation_reflects_valuation_freshness(recommendation, validated, index)
    # Spec 9: CORRECTS rather than rejects -- a misfiled condition is moved to
    # the bucket its direction implies. See the function's own docstring for
    # why rejecting here was the wrong call.
    validated = _route_conditions_by_direction(validated)
    # The content-policy AND claim-fidelity backstop (Problem 1 requirement 4;
    # Problem 5): even with no verdict field in the schema, a model can still
    # WRITE prohibited trade-advice language, or an unsupported superlative /
    # causal overreach / consensus claim, into a free-text field above.
    # Checked here so this validator is the single source of truth for "is
    # this output acceptable" — the repair wrapper below re-invokes this same
    # validator on the repaired response, so the check never has to be
    # duplicated. _validate_claim_fidelity raises with the
    # CONTENT_POLICY_VIOLATION_MARKER prefix (rather than relying on callers
    # to string-match arbitrary prose) so the repair wrapper can distinguish
    # "reject and repair once" from every other validation failure ("reject,
    # no repair, exactly like every other stage") by a single reliable prefix
    # check. `recommendation` is STILL excluded from `validated` here, so it
    # is never scanned -- reattached only after this call returns.
    validated = _validate_claim_fidelity(validated, index)
    validated = dict(validated)
    validated["recommendation"] = recommendation
    return validated


# ---------------------------------------------------------------------------
# Generic stage execution — JSON extraction, validation, checkpointing
# ---------------------------------------------------------------------------

def _extract_json(text):
    """Best-effort extraction of a JSON object from a model response.

    Tries a direct parse first (the common case when `response_format="json"`
    is honored), then strips a markdown code fence as a single, bounded
    fallback. Anything else is a parse failure — no further guessing.
    """
    text = (text or "").strip()
    try:
        return json.loads(text), None
    except (ValueError, TypeError):
        pass
    if text.startswith("```"):
        stripped = text.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        try:
            return json.loads(stripped.strip()), None
        except (ValueError, TypeError):
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1]), None
        except (ValueError, TypeError):
            pass
    return None, "response did not contain a parseable JSON object"


def _run_stage(stage_name, system_prompt, user_prompt, validator, ask_local_fn) -> StageCheckpoint:
    """Execute exactly one stage: bounded tokens, bounded timeout, no tools,
    no network — `ask_local_fn` is called with ONLY `messages`, `options`,
    `response_format`, and `timeout`; there is no code path here that could
    pass a `tools` schema."""
    started = time.monotonic()
    messages = [{"role": "system", "content": system_prompt},
               {"role": "user", "content": user_prompt}]
    try:
        raw = ask_local_fn(
            messages,
            options={"num_predict": config.research_stage_max_output_tokens(stage_name)},
            response_format="json",
            timeout=config.research_stage_timeout_seconds(),
        )
    except Exception as e:  # noqa: BLE001 — a stage must never crash the pipeline
        return StageCheckpoint(stage_name, StageStatus.FAILED, error=f"call failed: {type(e).__name__}",
                               duration_seconds=time.monotonic() - started)

    if not raw.get("ok", True):
        return StageCheckpoint(stage_name, StageStatus.FAILED, error="local model call failed",
                               duration_seconds=time.monotonic() - started)

    content = (raw.get("message") or {}).get("content") or ""
    metrics = raw.get("metrics") or {}
    parsed, parse_error = _extract_json(content)
    if parse_error:
        # A TRUNCATED response and a MALFORMED one look identical here (both
        # fail to parse) but need opposite corrections: truncation needs a
        # SHORTER answer, malformation needs a rewritten one. Ollama tells us
        # which via done_reason="length"; say so in the error so
        # `_classify_stage_failure` can route it.
        if metrics.get("truncated"):
            parse_error = (f"{TRUNCATION_MARKER} the response hit the output-token limit and was "
                           f"cut off mid-JSON after {metrics.get('completion_tokens') or 0} "
                           "tokens, so it could not be parsed.")
        return StageCheckpoint(stage_name, StageStatus.FAILED, error=parse_error,
                               prompt_tokens=metrics.get("prompt_tokens") or 0,
                               completion_tokens=metrics.get("completion_tokens") or 0,
                               duration_seconds=time.monotonic() - started)
    try:
        validated = validator(parsed)
        # Phase H.5, Phase 2: the validator smuggles quarantine records out on
        # the returned dict (validators return one value and their signature
        # is shared across six stages). Lift them onto the checkpoint and
        # strip them here, so `output` stays exactly the stage's own schema
        # and no record can be mistaken for stage content downstream.
        quarantines = validated.pop(_QUARANTINE_KEY, []) if isinstance(validated, dict) else []
        stage_findings = validated.pop(_FINDINGS_KEY, []) if isinstance(validated, dict) else []
    except _Invalid as e:
        return StageCheckpoint(stage_name, StageStatus.FAILED, error=e.message,
                               prompt_tokens=metrics.get("prompt_tokens") or 0,
                               completion_tokens=metrics.get("completion_tokens") or 0,
                               duration_seconds=time.monotonic() - started,
                               failure_detail=e.detail, findings=e.findings)

    return StageCheckpoint(
        stage_name, StageStatus.COMPLETED, output=validated,
        prompt_tokens=metrics.get("prompt_tokens") or 0,
        completion_tokens=metrics.get("completion_tokens") or 0,
        duration_seconds=time.monotonic() - started,
        quarantines=quarantines,
        findings=stage_findings,
    )


def _skipped(stage_name, reason) -> StageCheckpoint:
    return StageCheckpoint(stage_name, StageStatus.SKIPPED, error=reason)


def _run_stage_with_content_policy_repair(stage_name, system_prompt, user_prompt, validator,
                                          ask_local_fn) -> StageCheckpoint:
    """Like `_run_stage`, but with BOUNDED, CLASSIFIED recovery attempts.

    WM corrective patch. This wrapper used to repair exactly one failure
    class — a content-policy/claim-fidelity violation — and fail closed on
    everything else. That made every stage a single attempt against a strict
    schema, and it is the reason `research_manager` appeared to "fail for a
    different stock every time": the failures are STOCHASTIC, not
    ticker-specific. See `_classify_stage_failure` for the full argument and
    the live WM token measurements.

    Now every MECHANICAL failure gets a targeted correction and a retry, up
    to `research_stage_max_attempts()`:

        truncated output      -> ask for a shorter response
        malformed JSON        -> ask for well-formed JSON
        unknown evidence id   -> name the bad ids, require exact copies
        schema error          -> quote the error, fix only that
        transport error       -> retry
        content policy        -> the original repair prompt, unchanged

    What has NOT changed: the pipeline still fails closed. Invalid content
    from ANY attempt is never placed in a COMPLETED checkpoint's `output`,
    an unrecognized failure still gets no retry, and token/time from every
    attempt is accumulated for accurate accounting.

    Historical note on the original, narrower behaviour:

    Originally built for the FinalInvestmentSynthesizer alone (Phase H.3
    corrective patch, Problem 1 requirement 5). Extended to
    bull_researcher/bear_researcher (COR corrective patch, Phase 5-6) once a
    live run showed those stages hitting this SAME class of violation —
    including outright false positives (a "hold" or "exceptional" false
    match) — with no chance to self-correct, only an expensive fall-back to
    the single-shot narrative report that discards the entire staged
    pipeline's work over one flagged phrase. Extended again to
    research_manager/risk_reviewer once further live verification showed
    their own free-text reconciliation fields (`balanced_assessment`, a
    `risk` description, ...) carry the identical exposure — a clean bull and
    bear pair could still cascade-fail the pipeline one or two stages later
    with no way to recover. Extended a final time to rebuttal_round (GE
    corrective patch) once a live GE run showed IT ALSO carries the
    identical exposure (bull_rebuttal/bear_rebuttal text tripping the
    causal-overreach scanner in 2 of 5 runs) — the belief that it was
    exempt rested on absence of evidence, not evidence of absence, and a
    fresh live run supplied the evidence. Every stage whose output includes
    model-authored free text now gets this same one repair attempt; an
    occasional truncated-JSON schema error is still a DIFFERENT failure mode
    this wrapper never repairs, on any stage.

    The repair prompt includes the ORIGINAL validation error verbatim (the
    exact prohibited/unsupported terms found), so the model knows precisely
    what to fix. If the repair attempt is STILL invalid (schema error OR
    still violates policy), this fails closed — the invalid content from
    NEITHER attempt is ever placed in a COMPLETED checkpoint's `output`.
    """
    attempts = [_run_stage(stage_name, system_prompt, user_prompt, validator, ask_local_fn)]
    if attempts[0].status == StageStatus.COMPLETED:
        return attempts[0]

    max_attempts = max(1, config.research_stage_max_attempts())
    while len(attempts) < max_attempts:
        previous = attempts[-1]
        if previous.status != StageStatus.FAILED:
            break
        kind = _classify_stage_failure(previous.error)
        correction = _correction_for(kind, previous.error, previous.failure_detail)
        if correction is None:
            break  # not a recoverable failure -- fail closed, as before
        attempts.append(_run_stage(stage_name, system_prompt, user_prompt + correction,
                                   validator, ask_local_fn))
        if attempts[-1].status == StageStatus.COMPLETED:
            return _with_attempt_totals(attempts[-1], attempts)

    # Every attempt failed. Fail closed, exactly as before -- invalid content
    # from any attempt is never exposed -- but report the LAST error and the
    # accumulated token/time cost of all attempts.
    last = attempts[-1]
    prefix = "" if len(attempts) == 1 else f"after {len(attempts)} attempts: "
    return StageCheckpoint(
        stage_name, StageStatus.FAILED,
        error=f"{prefix}{last.error}",
        prompt_tokens=sum(a.prompt_tokens for a in attempts),
        completion_tokens=sum(a.completion_tokens for a in attempts),
        duration_seconds=sum(a.duration_seconds for a in attempts),
        # Carried from the LAST attempt. Rebuilding the checkpoint without
        # these silently dropped them -- a failed stage reported zero
        # findings, which is precisely the blind spot Phase 5a exists to
        # close (the patterns that kill stages having no data behind them).
        failure_detail=last.failure_detail,
        findings=last.findings,
    )


def _with_attempt_totals(checkpoint: StageCheckpoint, attempts) -> StageCheckpoint:
    """Preserve accurate cost accounting across retries.

    Every non-cost field is carried through explicitly. Rebuilding a
    checkpoint is a standing hazard in this function: an earlier version
    omitted `quarantines`/`findings`, so a stage that succeeded on its SECOND
    attempt silently lost its quarantine records -- present on attempt 1's
    checkpoint, absent from the one actually returned.
    """
    if len(attempts) < 2:
        return checkpoint
    return StageCheckpoint(
        checkpoint.stage, checkpoint.status, output=checkpoint.output,
        error=checkpoint.error,
        prompt_tokens=sum(a.prompt_tokens for a in attempts),
        completion_tokens=sum(a.completion_tokens for a in attempts),
        duration_seconds=sum(a.duration_seconds for a in attempts),
        quarantines=checkpoint.quarantines,
        failure_detail=checkpoint.failure_detail,
        findings=checkpoint.findings,
    )


# -- failure classification -------------------------------------------------
#
# WHY THIS EXISTS
#
# Before this, a stage got exactly ONE shot unless it tripped the content
# scanner. Every other way a response can be wrong -- cut off at the token
# limit, valid JSON missing a field, a citation to an evidence id that does
# not exist, a transport blip -- killed the stage outright and cascaded
# through everything downstream of it.
#
# That is why `research_manager` "fails for a different stock every time".
# It is not one bug per ticker: it is a single-attempt stage with the LARGEST
# schema in the pipeline (ten fields, seven of them lists) whose failures are
# STOCHASTIC. Live WM run: research_manager emitted 6,038 completion tokens
# against an 8,000 cap. It succeeded. A slightly longer company description,
# one extra supported point, and the same prompt truncates -- and truncation
# was indistinguishable from malformed JSON, so even the diagnosis was wrong.
#
# All four classes below are MECHANICAL: the model can fix them given the
# specific correction, without being told anything new about the company. So
# each gets a targeted correction and a bounded retry. Anything unrecognized
# still fails closed on the first attempt.

TRUNCATION_MARKER = "STAGE_OUTPUT_TRUNCATED:"


class StageFailureKind:
    TRUNCATED = "truncated"
    CONTENT_POLICY = "content_policy"
    UNKNOWN_EVIDENCE = "unknown_evidence"
    MALFORMED_JSON = "malformed_json"
    SCHEMA = "schema"
    TRANSPORT = "transport"
    UNRECOVERABLE = "unrecoverable"


_UNKNOWN_EVIDENCE_RE = re.compile(r"cites unknown evidence ID\(s\): (.+)$")


def _classify_stage_failure(error: Optional[str]) -> str:
    text = (error or "").strip()
    if not text:
        return StageFailureKind.UNRECOVERABLE
    if text.startswith(CONTENT_POLICY_VIOLATION_MARKER):
        return StageFailureKind.CONTENT_POLICY
    if text.startswith(TRUNCATION_MARKER):
        return StageFailureKind.TRUNCATED
    if _UNKNOWN_EVIDENCE_RE.search(text):
        return StageFailureKind.UNKNOWN_EVIDENCE
    if text.startswith("call failed:") or text == "local model call failed":
        return StageFailureKind.TRANSPORT
    lowered = text.lower()
    if "json" in lowered or "not a json object" in lowered:
        return StageFailureKind.MALFORMED_JSON
    # Everything else reaching here came from a field validator: a missing
    # field, a wrong enum value, a too-short list, a duplicate claim id.
    return StageFailureKind.SCHEMA


def _correction_for(kind: str, error: Optional[str],
                    detail: Optional[dict] = None) -> Optional[str]:
    """The corrective instruction appended to the ORIGINAL user prompt.

    Appended rather than replacing it, so the model still has the full
    evidence index and schema — a correction that drops the evidence would
    just produce a different failure.

    Returns None when the failure is not mechanically recoverable, which
    preserves the previous fail-closed behaviour for anything unrecognized.
    """
    if kind == StageFailureKind.TRUNCATED:
        return (
            "\n\nYour previous response was CUT OFF because it exceeded the output-token "
            "limit, so it could not be parsed. Produce the SAME JSON schema, but SHORTER: "
            "give the MINIMUM number of items each list allows rather than the maximum, keep "
            "every free-text field under two sentences, and do not restate evidence values "
            "that are already in the evidence index. Completeness of the JSON object matters "
            "more than richness of the prose — a short valid object is useful, a long "
            "truncated one is worthless. Respond with ONLY the JSON object.")
    if kind == StageFailureKind.MALFORMED_JSON:
        return (
            "\n\nYour previous response could not be parsed as JSON. Respond with ONLY a "
            "single well-formed JSON object matching the schema exactly — no prose before or "
            "after it, no markdown code fences, no trailing commas, and every string properly "
            "quoted and closed.")
    if kind == StageFailureKind.UNKNOWN_EVIDENCE:
        match = _UNKNOWN_EVIDENCE_RE.search(error or "")
        unknown = match.group(1) if match else "the ids you used"
        return (
            f"\n\nYour previous response cited evidence ID(s) that do not exist: {unknown}. "
            "Every ID must be copied EXACTLY from the EVIDENCE INDEX above — do not "
            "abbreviate, pluralize, guess, or construct an ID from a field name. Re-check "
            "each ID against the index, drop any claim you cannot support with a real ID, "
            "and respond with ONLY the corrected JSON object.")
    if kind == StageFailureKind.SCHEMA:
        return (
            f"\n\nYour previous response did not satisfy the schema: {error}\n"
            "Fix exactly that problem, keep every other field as you already wrote it, and "
            "respond with ONLY the corrected JSON object matching the schema exactly.")
    if kind == StageFailureKind.TRANSPORT:
        return "\n\n(Retrying after a transport error; respond with ONLY the JSON object.)"
    if kind == StageFailureKind.CONTENT_POLICY:
        # Phase H.5, Phase 3. Two shapes, decided by whether the failure
        # carries structured detail.
        #
        # WITH detail -- the vocabulary-bearing producer
        # (`_validate_claim_fidelity`). Send ONLY the offending field, with
        # the matched span redacted, and a POSITIVE instruction. The previous
        # version sent the whole error message plus its own worked examples
        # of forbidden phrasings, which is how the model learned the
        # vocabulary it then used to write more violations. Measured live on
        # WM: 1 violation, then 4, then 3, then the stage died.
        #
        # WITHOUT detail -- one of the eight structural coherence guards
        # ("research_stance was 'insufficient_data' but recommendation was
        # 'buy'"). Those messages name nothing forbidden and ARE the fix, so
        # they are sent verbatim. Redacting them would leave the model with
        # no idea what to change.
        if detail and detail.get("field_path"):
            field_path = detail["field_path"]
            redacted = detail.get("redacted_text")
            excerpt = (f"\n\nThe field currently reads:\n{redacted}\n"
                       if redacted else "\n")
            return (
                f"\n\nOne field in your previous response did not pass automated screening: "
                f"'{field_path}'. Every other field was accepted.{excerpt}"
                f"Rewrite ONLY '{field_path}'. State plainly what the evidence shows and cite "
                "the evidence ID that supports it. Describe what was measured or computed, and "
                "attribute any expectation to whoever holds it. Leave every other field EXACTLY "
                "as you already wrote it, and do not add new facts, claims, or evidence IDs. "
                "Respond with ONLY the corrected JSON object.")
        return (
            "\n\nYour previous response was not internally consistent: "
            f"{error}\n"
            "Correct exactly that inconsistency, leave every other field as you already wrote "
            "it, and respond with ONLY the corrected JSON object.")
    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_research_pipeline(evidence_index: Dict[str, EvidenceItem], ask_local_fn,
                          readiness_status: Optional[str] = None) -> ResearchPipelineResult:
    """Run the full staged pipeline. `evidence_index` is built ONCE by the
    caller (`finance.evidence.build_evidence_index`) and rendered to an
    immutable string here — every stage sees the identical text.

    Cascading fail-closed: a stage that cannot run because its prerequisite(s)
    failed is recorded SKIPPED with why, never silently omitted and never fed
    a fabricated stand-in for the missing input.
    """
    checkpoints: List[StageCheckpoint] = []
    evidence_text = render_evidence_index(evidence_index)

    def index_lookup(_raw):
        return evidence_index

    # -- 1-2: independent bull and bear researchers -- each gets the SAME
    # one-repair-attempt treatment as the final synthesizer (COR corrective
    # patch, Phase 5-6): a content-policy/claim-fidelity violation here is
    # common (this is the most model-authored-prose-heavy stage), and losing
    # either researcher cascades the ENTIRE pipeline to the single-shot
    # fallback over what is sometimes a single flagged phrase. research_
    # manager, risk_reviewer, and rebuttal_round (below) all get the
    # identical treatment for the identical reason (GE corrective patch:
    # rebuttal_round was the last holdout, on the mistaken belief it had
    # never been observed tripping this scan live -- see this module's own
    # docstring and `_validate_claim_fidelity`'s).
    bull_system, bull_user = _researcher_prompt("bull", evidence_text)
    bull_checkpoint = _run_stage_with_content_policy_repair(
        "bull_researcher", bull_system, bull_user,
        lambda raw: _validate_researcher_output(raw, evidence_index, "bull"), ask_local_fn)
    checkpoints.append(bull_checkpoint)

    bear_system, bear_user = _researcher_prompt("bear", evidence_text)
    bear_checkpoint = _run_stage_with_content_policy_repair(
        "bear_researcher", bear_system, bear_user,
        lambda raw: _validate_researcher_output(raw, evidence_index, "bear"), ask_local_fn)
    checkpoints.append(bear_checkpoint)

    bull_ok = bull_checkpoint.status == StageStatus.COMPLETED
    bear_ok = bear_checkpoint.status == StageStatus.COMPLETED

    # -- 3: one bounded rebuttal round (needs BOTH sides) --
    rebuttal_output = None
    if bull_ok and bear_ok:
        rb_system, rb_user = _rebuttal_prompt(evidence_text, bull_checkpoint.output,
                                              bear_checkpoint.output)
        rebuttal_checkpoint = _run_stage_with_content_policy_repair(
            "rebuttal_round", rb_system, rb_user,
            lambda raw: _validate_rebuttal_output(raw, evidence_index), ask_local_fn)
        if rebuttal_checkpoint.status == StageStatus.COMPLETED:
            rebuttal_output = rebuttal_checkpoint.output
    else:
        rebuttal_checkpoint = _skipped(
            "rebuttal_round", "requires both bull_researcher and bear_researcher to have completed")
    checkpoints.append(rebuttal_checkpoint)

    # -- 4: research manager (needs at least one side) --
    if bull_ok or bear_ok:
        rm_system, rm_user = _research_manager_prompt(
            evidence_text, bull_checkpoint.output or {"thesis": "(unavailable)", "claims": []},
            bear_checkpoint.output or {"thesis": "(unavailable)", "claims": []}, rebuttal_output)
        # COR corrective patch, live verification: research_manager's own
        # free-text reconciliation (balanced_assessment, shared_findings, ...)
        # faces the SAME content-policy exposure as bull/bear -- observed
        # live tripping the consensus-language scanner -- and, unlike
        # bull/bear, had no repair attempt to fall back on, which cascade-
        # failed risk_reviewer and final_investment_synthesizer even though
        # both upstream researchers had completed cleanly. Same one-repair-
        # attempt wrapper, same validator, nothing weakened.
        rm_checkpoint = _run_stage_with_content_policy_repair(
            "research_manager", rm_system, rm_user,
            lambda raw: _validate_research_manager_output(raw, evidence_index), ask_local_fn)
    else:
        rm_checkpoint = _skipped(
            "research_manager", "requires at least one of bull_researcher/bear_researcher to have completed")
    checkpoints.append(rm_checkpoint)
    rm_ok = rm_checkpoint.status == StageStatus.COMPLETED

    # -- 5: risk reviewer (needs research manager) --
    if rm_ok:
        rr_system, rr_user = _risk_reviewer_prompt(evidence_text, rm_checkpoint.output)
        # Same reasoning as research_manager above: risk_reviewer's 'risk'/
        # 'data_quality_concerns' free-text fields face identical exposure,
        # and this is the last stage standing between a completed
        # research_manager and final_investment_synthesizer ever being
        # reached at all.
        risk_checkpoint = _run_stage_with_content_policy_repair(
            "risk_reviewer", rr_system, rr_user,
            lambda raw: _validate_risk_reviewer_output(raw, evidence_index), ask_local_fn)
    else:
        risk_checkpoint = _skipped("risk_reviewer", "requires research_manager to have completed")
    checkpoints.append(risk_checkpoint)
    risk_ok = risk_checkpoint.status == StageStatus.COMPLETED

    # -- 6: FinalInvestmentSynthesizer (needs research manager + risk reviewer) --
    if rm_ok and risk_ok:
        fs_system, fs_user = _final_synthesizer_prompt(
            evidence_text, rm_checkpoint.output, risk_checkpoint.output)
        final_checkpoint = _run_stage_with_content_policy_repair(
            "final_investment_synthesizer", fs_system, fs_user,
            # MLI corrective patch: the risk_reviewer's validated output is
            # threaded in so its aggregated risk can be FORCED onto the
            # synthesis rather than re-derived (or silently softened) here.
            lambda raw: _validate_final_synthesizer_output(
                raw, evidence_index, risk_checkpoint.output, readiness_status), ask_local_fn)
    else:
        final_checkpoint = _skipped(
            "final_investment_synthesizer", "requires research_manager and risk_reviewer to have completed")
    checkpoints.append(final_checkpoint)

    available = final_checkpoint.status == StageStatus.COMPLETED
    return ResearchPipelineResult(available=available, checkpoints=checkpoints,
                                  evidence_index_size=len(evidence_index))
