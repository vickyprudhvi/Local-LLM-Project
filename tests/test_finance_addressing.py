"""Phase H.5 (validation rework) — Phase 4: fabrication defined by ADDRESSING.

The eight original fabrication patterns are TOPIC bans: a list of things not
to mention. The Phase H.3 incident was not about a topic. "If you do not
currently hold a position: AVOID. If you already hold a position: SELL." is
harmful because it addresses the reader as an agent who holds a position and
should act -- and this system can know neither of those things.

Banning the addressing catches the shape rather than enumerating its
vocabulary, which is the point of the whole rework. It also closes a gap the
topic bans could not: CP-006 matched only the SECOND-PERSON framing, so three
third-person variants escaped entirely and were carried as xfail markers
through phases 2 and 3.

The design point, arrived at by testing candidates against real prose: the
harm is A DIRECTIVE WHOSE SUBJECT IS THE READER. Neither half alone works --
"The company should trim its cost base" is a directive with a different
subject, and "Investors who hold the stock have seen a 20% decline" refers to
holdings but is descriptive. Only the conjunction is a violation.
"""

import json

import pytest

import finance.content_policy as cp
import finance.research_pipeline as rp
from tests.test_finance_research_pipeline import (
    INDEX,
    _bull_response,
    make_ask_local,
)

H3_STRING = ("If you do not currently hold a position: AVOID. "
             "If you already hold a position: SELL.")


# ---------------------------------------------------------------------------
# Invariant 1 — the incident this exists for
# ---------------------------------------------------------------------------

def test_the_h3_string_trips_at_least_two_independent_rules():
    """Defence in depth: no single rule's failure can reopen it."""
    labels = {f.label for f in cp.find_prohibited_directives(H3_STRING)}
    assert len(labels) >= 2, labels
    assert "holding-dependent phrasing" in labels     # the original topic ban
    assert "reader-addressed second person" in labels  # Phase 4's addressing ban


@pytest.mark.parametrize("variant", [
    # second person
    "If you do not currently hold a position: AVOID.",
    "If you already hold a position, SELL.",
    "If you hold this stock, consider trimming.",
    "If you do not own shares, avoid entering here.",
    "You should sell into strength.",
    "Your position should be reduced.",
    # third person -- the gap Phase 4 closes
    "For readers who currently hold a position: hold.",
    "Investors who already hold shares should trim.",
    "Those holding a position may want to exit.",
    "Anyone who owns the stock should reduce exposure.",
    "Investors are advised to avoid the name.",
    "Shareholders should trim exposure into the print.",
    # bare imperative
    "Sell into strength.",
    "Buy the stock now.",
    "Trim the position.",
])
def test_reader_addressing_is_rejected_in_any_person(variant):
    findings = cp.find_prohibited_directives(variant)
    assert any(f.severity == cp.Severity.FABRICATION for f in findings), variant


# ---------------------------------------------------------------------------
# The other half: what must still pass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    # directives whose subject is the COMPANY, not the reader
    "The company should trim its cost base.",
    "Management may want to exit the China business.",
    "The board must approve the buyback before it proceeds.",
    "Management expects to add capacity in 2027.",
    # descriptive statements that merely reference holders
    "Shareholders approved the merger in June.",
    "Investors who hold the stock have seen a 20% decline.",
    "Holders of the 2031 notes rank ahead of equity.",
    # compound nouns that merely start with an action verb
    "Exit rates improved in the fourth quarter.",
    "Entry pricing improved.",
    "Add-on acquisitions continued.",
    "Hold-to-maturity securities rose.",
    "Buyback activity accelerated.",
    "Enterprise value rose 12%.",
    # ordinary analysis prose
    "Revenue growth is healthy and margins expanded.",
    "Consider the wide scenario spread when weighing this.",
    "Take the FY2025 figure as the base for the model.",
    "Increase in receivables drove the swing.",
    "The bear case rests on multiple compression.",
    # bare verdicts stay allowed -- the tool emits one as an enum
    "Overall: AVOID.",
    "Recommendation: hold",
    "This is a STRONG BUY.",
    "The company announced a share buy back program.",
    "Retail sell-through improved this quarter.",
])
def test_ordinary_analysis_prose_is_not_addressing(text):
    """The false-positive corpus. A fabrication match is cascade-fatal, so a
    false positive here costs an entire analysis -- these matter as much as
    the true positives above."""
    assert cp.scan_for_prohibited_directives(text) == [], text


# ---------------------------------------------------------------------------
# Enum fields are exempt (amendment A1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["buy", "hold", "sell", "avoid", "insufficient_evidence"])
def test_every_recommendation_enum_member_is_exempt(value):
    """Four of the five members ARE bare imperative verbs at position zero.
    Scanning every string would make every real recommendation
    cascade-fatal, deleting a feature this project deliberately built."""
    findings = cp.find_prohibited_directives(value, field_path="recommendation")
    assert findings == [], value


@pytest.mark.parametrize("field_name", sorted(cp.NON_PROSE_FIELDS))
def test_no_addressing_rule_fires_on_a_non_prose_field(field_name):
    for value in ("buy", "hold", "sell", "avoid", "you", "your"):
        findings = cp.find_prohibited_directives(value, field_path=field_name)
        prose_rules = [f for f in findings if cp.RULES_BY_ID[f.rule_id].prose_only]
        assert prose_rules == [], f"{field_name}={value}"


def test_the_same_value_in_a_prose_field_is_not_exempt():
    """The exemption is about the FIELD, not the string."""
    assert cp.find_prohibited_directives("hold", field_path="recommendation") == []
    assert cp.find_prohibited_directives("Sell into strength.", field_path="thesis")


def test_unknown_field_context_is_treated_as_prose():
    """`scan_for_prohibited_directives` is called on bare strings with no
    field path at all (the single-shot fallback report). Treating unknown
    context as non-prose would silently exempt it."""
    assert cp.find_prohibited_directives("Sell into strength.", field_path="")
    assert cp.scan_for_prohibited_directives("Sell into strength.")


def test_the_non_prose_set_covers_every_enum_field_the_stages_emit():
    """A new ENUM field not listed here would be scanned and could
    false-positive. Loud, and preferable to a silent gap -- but this pins the
    current set so the divergence is caught here rather than in a live run."""
    result = rp.run_research_pipeline(INDEX, make_ask_local())
    emitted = set()
    for checkpoint in result.checkpoints:
        for path, _text in cp._walk_fields(checkpoint.output or {}):
            emitted.add(cp._leaf_name(path))
    # Every field this project classifies as non-prose must actually exist.
    # `valuation_method_status` is set from a business-model evidence item
    # this fixture's INDEX does not carry (see
    # research_pipeline.py::_business_model_policy_from_index), so it is
    # never populated here even though it is a real, deterministic field --
    # same reason `evidence_ids` already needed this exemption.
    assert cp.NON_PROSE_FIELDS <= emitted | {"evidence_ids", "valuation_method_status"}, (
        cp.NON_PROSE_FIELDS - emitted)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_reader_addressing_fails_the_stage_and_is_never_quarantined():
    """Fabrication-class: cascade-fatal, not survivable via quarantine."""
    bad = _bull_response(extra={"thesis": "Investors who hold shares should trim."})
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull=bad))
    checkpoint = result.by_stage("bull_researcher")

    assert checkpoint.status == rp.StageStatus.FAILED
    assert checkpoint.output is None
    assert checkpoint.quarantines == []
    assert any(f["rule_id"] == "CP-015" for f in checkpoint.findings)


def test_a_real_recommendation_still_completes_end_to_end():
    """The regression the A1 amendment exists to prevent. `recommendation`
    is `hold` in the default fixture -- a bare imperative verb."""
    result = rp.run_research_pipeline(INDEX, make_ask_local())
    final = result.by_stage("final_investment_synthesizer")
    assert final.status == rp.StageStatus.COMPLETED
    assert final.output["recommendation"] in ("buy", "hold", "sell", "avoid",
                                              "insufficient_evidence")
    assert result.available is True


def test_word_boundaries_hold():
    """'your' must not fire on substrings."""
    for text in ("Yourself is not the subject here.",):
        assert cp.scan_for_prohibited_directives(text)
    for text in ("The company is in Yourkshire.", "Buyout activity rose.",
                 "Holdings increased.", "Selloff continued."):
        assert cp.scan_for_prohibited_directives(text) == [], text


def test_the_shared_guardrails_do_not_demonstrate_the_violation():
    """The instruction must state the rule without showing the construction.

    An earlier draft spelled out the banned pronouns and gave a worked
    counter-example ("investors should trim"). That is the same priming that
    made the old repair prompt escalate on WM: showing the model the exact
    phrasing is how it learns to write it.

    Scoped to the rules a DEMONSTRATION would trip, not to every rule. A
    system prompt necessarily says "you" -- it is addressing the model, and
    these rules govern stage OUTPUT, not instructions. Asserting the
    guardrail is clean of second person would be asserting it cannot be
    written at all.
    """
    demonstrable = {"reader-directed action", "bare trading imperative",
                    "trading instructions", "order instructions"}
    found = set(cp.scan_for_prohibited_directives(rp._SHARED_GUARDRAILS))
    assert found & demonstrable == set(), found & demonstrable
