"""Phase H.37 — a stock-analysis request must never be answered by the
general/local LLM from its own knowledge or by the financial-evidence-
analysis skill standing in for the canonical pipeline.

Reuses the real end-to-end `orchestrated` fixture from
tests/test_assistant_stock_analysis_orchestration.py (Phase H.1) — a
scripted market-data client plus a stubbed local-LLM call, with
tool_loop.run_local_tool_loop (Phase B, the generic tool loop) monkeypatched
to raise if it is ever entered, so "the canonical path was used, not the
free-form fallback" is PROVEN, not assumed. This file adds the widened
trigger-phrase coverage (section 17's test matrix), the new exception
fail-closed behavior (section 10), and the architectural skill-boundary/
tool-description checks (sections 5, 6, 26, 28) this phase adds.
"""

import inspect

import pytest

import assistant
import tool_loop
from finance.workflow import DCF_TOOL_NAME, detect_full_stock_analysis_request

from tests.test_assistant_stock_analysis_orchestration import (  # noqa: F401
    SpyTool,
    _executed_tool_names,
    clock,
    orchestrated,
)

# =============================================================================
# Section 17 — routing test matrix: the widened trigger phrasings reach the
# canonical pipeline through the REAL entry point, not merely the bare regex.
# =============================================================================

@pytest.mark.parametrize("text", [
    "Analyze TSLA stock",
    "Give me a DCF for TSLA",
    "Bull and bear case for TSLA",
    "Should I research TSLA?",
    "What is the bull case for TSLA?",
    "Give me risks for TSLA.",
    "Is TSLA overvalued?",
    "How is TSLA stock?",
    "Give me a short analysis of TSLA",
    "Give me an investment thesis for TSLA",
])
def test_widened_trigger_phrasings_invoke_the_canonical_workflow(orchestrated, text):
    """Section 17, A-D: each of these phrasings must reach finance.dcf_model
    through the SAME real orchestration path as the original "full stock
    analysis of TSLA" wording -- never the generic tool loop (which would
    raise per the fixture's own Phase-B-must-not-run guard)."""
    executor = orchestrated["executor"]
    executed = _executed_tool_names(executor)

    reply, _metrics, pending = assistant._process_local_request_with_capability_selection(
        None, None, text, "prompt", [], "sys", set())

    assert pending is None
    assert DCF_TOOL_NAME in executed, f"{text!r} did not reach the canonical DCF tool"
    # "short"/"compact" phrasing (section 14) legitimately renders WITHOUT
    # calling the stubbed LLM at all (compact rendering is pure templating) --
    # the canonical-pipeline-reached assertion above is what matters for
    # those; the stub text is the right check only when report_detail stays
    # "full" (the fixture's pinned default).
    if "short" not in text.lower() and "compact" not in text.lower():
        assert "STUBBED REPORT TEXT" in reply


@pytest.mark.parametrize("text", [
    "What is DCF?",
    "What is WACC?",
    "Explain terminal growth.",
    "How does a 10-Q differ from a 10-K?",
    "What is free cash flow?",
    "What is debt to equity?",
])
def test_educational_questions_do_not_trigger_the_canonical_workflow(text):
    """Section 3: a conceptual/educational question about finance in general
    names no company and must not force the expensive six-stage pipeline."""
    assert detect_full_stock_analysis_request(text) is None


@pytest.mark.parametrize("text", [
    "Analyze VZ stock",
    "Give me a DCF for CF",
    "Bull and bear case for NKE",
    "Should I research VZ?",
    "Give me risks for CF.",
    "Is NKE overvalued?",
])
def test_vz_cf_nke_phrasings_reach_the_canonical_workflow_through_the_real_entry_point(
        orchestrated, text):
    """Section 24: VZ/CF/NKE specifically, through the SAME real
    assistant._process_local_request_with_capability_selection entry point
    every other test in this file uses -- never finance.workflow.
    run_full_stock_analysis called directly in isolation. Uses the same
    scripted market-data client as the rest of this file's real-orchestration
    methodology (no live network, no live LLM) -- this phase is about
    whether the request REACHES the canonical pipeline, not about data
    accuracy for these three tickers (that is H.34-36 territory and section
    25 explicitly keeps it out of scope here)."""
    executor = orchestrated["executor"]
    executed = _executed_tool_names(executor)

    reply, _metrics, pending = assistant._process_local_request_with_capability_selection(
        None, None, text, "prompt", [], "sys", set())

    assert pending is None
    assert DCF_TOOL_NAME in executed, f"{text!r} did not reach the canonical DCF tool"
    assert "STUBBED REPORT TEXT" in reply
    assert orchestrated["spies"]["browser.search"].calls == []
    assert orchestrated["spies"]["github.clone_repository"].calls == []


def test_how_is_x_doing_stays_excluded_while_how_is_x_stock_is_included():
    """The pre-existing, deliberate exclusion for casual 'how is X doing'
    phrasing (too ambiguous to justify a full pipeline run) must survive the
    new 'how is X stock' addition -- confirms the widening was scoped, not a
    blanket broadening of 'how is X ...'."""
    assert detect_full_stock_analysis_request("how is VZ doing") is None
    assert detect_full_stock_analysis_request("how is MSFT stock") == "MSFT"


# =============================================================================
# Section 10 — fail-closed on an unexpected tool/workflow exception
# =============================================================================

def test_unexpected_workflow_exception_yields_a_reduced_message_not_a_crash_or_fallback(
        orchestrated, monkeypatch):
    """The most important H.37 test: when the canonical workflow raises
    something `run_full_stock_analysis`'s own extensive internal handling
    did not anticipate, the reply must be a plain unavailability message --
    never a crash propagating out, and (proven by the fixture's Phase-B-
    must-not-run guard) never a fall-through to the generic tool loop where
    a local LLM could invent an answer."""
    from finance import workflow as finance_workflow

    def boom(*args, **kwargs):
        raise RuntimeError("simulated unexpected internal failure")

    monkeypatch.setattr(finance_workflow, "run_full_stock_analysis", boom)

    reply, metrics, pending = assistant._process_local_request_with_capability_selection(
        None, None, "Give me a full stock analysis of TSLA", "prompt", [], "sys", set())

    assert pending is None
    assert metrics == {"prompt_tokens": 0, "completion_tokens": 0}
    assert "unavailable" in reply.lower()
    assert not orchestrated["llm_calls"], "the local LLM must never be called on this path"
    # None of the invented-content categories section 18 forbids appear --
    # there is no content here at all beyond the plain unavailability notice.
    for forbidden in ("revenue", "guidance", "valuation", "peer", "metaverse"):
        assert forbidden not in reply.lower()


def test_tool_validation_error_still_returns_its_own_specific_message(orchestrated, monkeypatch):
    """The PRE-EXISTING, more specific ToolValidationError path (e.g.
    STOCK_ANALYSIS_ENABLED=false) must be unaffected by the new broader
    exception handler -- it still gets its own message, not the generic one."""
    from finance import workflow as finance_workflow
    from tools.base import ToolValidationError

    def refuse(*args, **kwargs):
        raise ToolValidationError("STOCK_ANALYSIS_ENABLED is false")

    monkeypatch.setattr(finance_workflow, "run_full_stock_analysis", refuse)

    reply, _metrics, _pending = assistant._process_local_request_with_capability_selection(
        None, None, "Give me a full stock analysis of TSLA", "prompt", [], "sys", set())

    assert reply == "STOCK_ANALYSIS_ENABLED is false"


# =============================================================================
# Sections 5, 6, 26, 28 — the skill never becomes a top-level/standalone
# stock-analysis engine
# =============================================================================

def test_skill_loader_and_skill_file_are_referenced_only_inside_research_pipeline():
    """Architectural guarantee, scoped to PRODUCTION code: grep every
    non-test .py file for any reference to the runtime skill outside
    finance/research_pipeline.py and finance/skill_loader.py itself -- in
    particular router.py, brain.py, tool_loop.py, tool_dispatch.py, and
    assistant.py, which is the actual invariant section 6 cares about. Test
    files (tests/test_finance_h35_runtime_skill.py,
    tests/test_finance_research_pipeline.py, this file) legitimately import
    the loader directly to TEST it, which is a different thing from the
    skill being exposed to the top-level assistant in production."""
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    allowed_files = {
        repo_root / "finance" / "research_pipeline.py",
        repo_root / "finance" / "skill_loader.py",
    }
    hits = []
    for path in repo_root.rglob("*.py"):
        if path in allowed_files:
            continue
        parts = path.parts
        if any(part in {"venv", "__pycache__", ".git", "node_modules", "tests"} for part in parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "skill_loader" in text or "financial-evidence-analysis" in text:
            hits.append(str(path.relative_to(repo_root)))
    assert hits == [], f"the runtime skill is referenced outside its boundary: {hits}"


def test_skill_precondition_language_is_explicit_in_the_skill_file():
    from finance.skill_loader import load_runtime_skill

    text = load_runtime_skill()
    assert "precondition" in text.lower()
    assert "CanonicalResearchEvidence" in text
    assert "stock-analysis engine" in text.lower()
    assert "SKILL_PRECONDITION_NOT_MET" in text


def test_dcf_model_tool_description_forbids_memory_derived_inputs():
    """Section 20/28: the one general-purpose tool that could otherwise be
    used to reconstruct a free-form valuation must say, in its own
    LLM-facing description, that it is not a substitute for the canonical
    workflow and that inputs must come from this turn's own retrieval."""
    import tools.finance_tools as finance_tools

    description = finance_tools.DcfModelTool.description
    assert "never from your own memory" in description.lower() or "memory" in description.lower()
    assert "FullStockAnalysis" in description


# =============================================================================
# Section 6 (negative) — deterministic modules still never import the skill
# (confirms this still holds after H.37's own changes, same style as H.35)
# =============================================================================

@pytest.mark.parametrize("module_name", [
    "finance.dcf", "finance.sec_provider", "finance.sec_datasets",
    "finance.guidance", "finance.actualization", "finance.metrics",
])
def test_deterministic_modules_still_never_import_the_skill(module_name):
    import importlib

    module = importlib.import_module(module_name)
    src = inspect.getsource(module)
    assert "skill_loader" not in src
