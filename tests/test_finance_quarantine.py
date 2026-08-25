"""Phase H.5 (validation rework) — Phase 2: field-level quarantine.

Replaces pass/fail with pass / quarantine / fail.

A FABRICATION match still destroys the stage: there is no true version of
"use a 5% position size", because the system has no idea what the reader's
capital is. An OVERSTATEMENT match now removes the offending field and lets
the pipeline continue: the claim underneath is real, just phrased more
strongly than the evidence carries, and it still had to cite a genuine
evidence ID to get this far.

The invariant each test covers is named in its docstring.
"""

import json

import pytest

import finance.claim_validation as cv
import finance.content_policy as cp
import finance.research_pipeline as rp
from tests.test_finance_research_pipeline import (
    INDEX,
    _bull_response,
    _rebuttal_response,
    _research_manager_response,
    make_ask_local,
)

H3_STRING = ("If you do not currently hold a position: AVOID. "
             "If you already hold a position: SELL.")


# ---------------------------------------------------------------------------
# Rule identity
# ---------------------------------------------------------------------------

def test_every_pattern_has_a_stable_id_and_a_severity():
    # 13 original + 3 added by Phase 4 (addressing-based fabrication).
    assert len(cp.RULES) == 21
    assert len(cv.RULES) == 37
    for rule in list(cp.RULES) + list(cv.RULES):
        assert rule.rule_id
        assert rule.severity in cp.Severity.ALL


def test_rule_ids_are_unique_within_each_module():
    for rules in (cp.RULES, cv.RULES):
        ids = [r.rule_id for r in rules]
        assert len(ids) == len(set(ids))


def test_the_fabrication_set_is_exactly_the_information_lacking_rules():
    """The split the whole rework turns on. Fabrication guards sentences with
    no true version -- the system cannot know the reader's capital, or
    whether they hold anything. Phase 4 added three that ban the ADDRESSING
    rather than a topic; Phase H.10 added five that ban an ASSUMED POSITION,
    which is the same missing information reached through the third person
    ("investors should continue holding") rather than the second."""
    fabrication = {r.label for r in cp.RULES if r.severity == cp.Severity.FABRICATION}
    assert fabrication == {
        # topic bans (original)
        "position size", "entry price", "exit price", "stop loss", "target allocation",
        "holding-dependent phrasing", "order instructions", "trading instructions",
        "reader-directed investment imperative",
        # addressing bans (Phase 4)
        "reader-addressed second person", "reader-directed action",
        "bare trading imperative",
        # assumed-position bans (Phase H.10, section 35)
        "assumes the reader holds a position (continue holding)",
        "assumes the reader holds a position (maintain the position)",
        "assumes the reader holds a position (retain the position)",
        "directs an assumed existing holder",
        "directs a change to an assumed existing position",
    }
    assert all(r.severity == cp.Severity.OVERSTATEMENT for r in cv.RULES)


def test_the_phase_four_addressing_rules_are_prose_only():
    """Invariant: `recommendation` is an enum whose members include `buy`,
    `hold`, `sell` and `avoid` -- bare imperative verbs. A rule applied to
    every string would make every real recommendation cascade-fatal.

    The Phase H.10 assumed-position rules are prose-only for exactly the
    same reason: the enum's own HOLD member must stay legal, and only the
    sentences ABOUT holding are the problem.
    """
    prose_only = {r.rule_id for r in cp.RULES if r.prose_only}
    assert prose_only == {"CP-014", "CP-015", "CP-016",
                          "CP-017", "CP-018", "CP-019", "CP-020", "CP-021"}


def test_the_rule_table_fails_loudly_if_it_drifts_from_the_pattern_list():
    """A silently renumbered id would corrupt every stored quarantine record
    and the Phase 5 aggregation that decides which patterns get deleted."""
    with pytest.raises(RuntimeError, match="rule specs"):
        cp._build_rules(list(cp._PROHIBITED_PATTERNS)[:3],
                        cp._CONTENT_POLICY_RULE_SPECS, "test")
    shuffled = tuple([cp._CONTENT_POLICY_RULE_SPECS[1], cp._CONTENT_POLICY_RULE_SPECS[0]]
                     + list(cp._CONTENT_POLICY_RULE_SPECS[2:]))
    with pytest.raises(RuntimeError, match="drifted out of alignment"):
        cp._build_rules(cp._PROHIBITED_PATTERNS, shuffled, "test")


# ---------------------------------------------------------------------------
# Path-aware scanning
# ---------------------------------------------------------------------------

def test_a_finding_names_the_field_it_came_from():
    """You cannot quarantine a field without knowing which field it was."""
    findings = cp.find_prohibited_directives_in_structure({
        "key_risks": [{"risk": "Set a stop loss at $180."}],
        "thesis": "A price target of $250 is supported.",
    })
    by_path = {f.field_path: f for f in findings}
    assert by_path["key_risks[0].risk"].rule_id == "CP-004"
    assert by_path["key_risks[0].risk"].severity == cp.Severity.FABRICATION
    assert by_path["thesis"].severity == cp.Severity.OVERSTATEMENT
    assert by_path["thesis"].matched_span.lower() == "price target"


def test_path_aware_scanning_matches_the_label_only_scanners():
    """Same policy, different return shape -- never a second, divergent
    policy. Anything exempt in one is exempt in the other."""
    for text in ("These modeled values should not be interpreted as a price target.",
                 "The bear case guarantees nothing about future returns.",
                 "Use a position size of 5%.",
                 "A price target of $250 is supported.",
                 "Management's margin target looks achievable."):
        labels = set(cp.scan_for_prohibited_directives(text))
        paths = {f.label for f in cp.find_prohibited_directives(text)}
        assert labels == paths, text
        labels = set(cv.scan_for_unsupported_claims(text))
        paths = {f.label for f in cv.find_unsupported_claims(text)}
        assert labels == paths, text


def test_field_walk_is_deterministic():
    """Invariant 3. Dict keys sorted, list indices natural, so the same
    structure always yields the same paths in the same order."""
    payload = {"z": ["a", "b"], "a": {"m": "x", "b": "y"}}
    first = [p for p, _ in cp._walk_fields(payload)]
    second = [p for p, _ in cp._walk_fields(dict(reversed(list(payload.items()))))]
    assert first == second == ["a.b", "a.m", "z[0]", "z[1]"]


# ---------------------------------------------------------------------------
# Quarantine behaviour
# ---------------------------------------------------------------------------

def test_an_overstatement_quarantines_its_field_and_the_stage_completes():
    """Invariant 2: what lands in the checkpoint is valid. The offending text
    is gone; the rest of the stage's work survives."""
    bad = _bull_response(extra={"thesis": "This is a fortress balance sheet."})
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull=bad))
    checkpoint = result.by_stage("bull_researcher")

    assert checkpoint.status == rp.StageStatus.COMPLETED
    assert "fortress" not in json.dumps(checkpoint.output).lower()
    assert checkpoint.output["claims"], "the untouched fields must survive"
    assert len(checkpoint.quarantines) == 1
    record = checkpoint.quarantines[0]
    assert record["field_path"] == "thesis"
    assert record["severity"] == "overstatement"
    assert record["matched_span"].lower() == "fortress"
    assert record["policy"] == rp.QUARANTINE_STUB


def test_a_fabrication_still_fails_the_stage_and_stops_the_cascade():
    """Invariant 1/2. No true version of this sentence exists."""
    bad = _bull_response(extra={"thesis": "Use a position size of 5% of your portfolio."})
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull=bad))
    assert result.by_stage("bull_researcher").status == rp.StageStatus.FAILED
    assert result.by_stage("bull_researcher").output is None


def test_the_h3_string_stays_impossible_to_emit():
    """Invariant 1 — the incident this whole validation layer exists for."""
    assert cp.scan_for_prohibited_directives(H3_STRING)
    findings = cp.find_prohibited_directives(H3_STRING)
    assert any(f.severity == cp.Severity.FABRICATION for f in findings)

    bad = _bull_response(extra={"thesis": H3_STRING})
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull=bad))
    assert result.by_stage("bull_researcher").status == rp.StageStatus.FAILED
    assert H3_STRING.lower() not in json.dumps(result.to_dict()).lower()


@pytest.mark.parametrize("variant", [
    "If you do not currently hold a position: AVOID.",
    "If you already hold a position, SELL.",
    "If you hold this stock, consider trimming.",
    "If you do not own shares, avoid entering here.",
])
def test_second_person_variants_of_the_h3_string_are_rejected(variant):
    """Invariant 1. The exact string is not the hazard; the shape is."""
    findings = cp.find_prohibited_directives(variant)
    assert any(f.severity == cp.Severity.FABRICATION for f in findings), variant


@pytest.mark.parametrize("variant", [
    "For readers who currently hold a position: hold.",
    "Investors who already hold shares should trim.",
    "Those holding a position may want to exit.",
    "Anyone who owns the stock should reduce exposure.",
    "Investors are advised to avoid the name.",
])
def test_third_person_holding_conditionals_are_rejected(variant):
    """Invariant 1, CLOSED by Phase 4.

    These carried `xfail(strict=True)` markers through phases 2 and 3: output
    conditioned on the reader's holding status is the Phase H.3 harm
    regardless of grammatical person, and CP-006 only matched the
    second-person framing. Phase 4 bans the ADDRESSING rather than the
    vocabulary, which catches the shape in any person.
    """
    findings = cp.find_prohibited_directives(variant)
    assert any(f.severity == cp.Severity.FABRICATION for f in findings), variant


def test_a_reader_directed_imperative_is_fabrication_not_quarantinable():
    """The amendment. This addresses the reader as an agent who should act --
    the H.3 mechanism -- so it must not be survivable via quarantine."""
    bad = _bull_response(extra={"thesis": "Investors should invest now."})
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull=bad))
    assert result.by_stage("bull_researcher").status == rp.StageStatus.FAILED


def test_quarantine_that_would_breach_a_minimum_fails_instead():
    """Invariant 2. `claims` carries min_items=2; emptying it would leave a
    researcher asserting a thesis with no claims at all, which is not a
    redacted output but a broken one."""
    body = json.loads(_bull_response())
    body["claims"][0]["claim"] = "A fortress balance sheet supports this."
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull=json.dumps(body)))
    assert result.by_stage("bull_researcher").status == rp.StageStatus.FAILED
    assert rp._policy_for("claims[0].claim") == rp.QUARANTINE_FAIL


def test_a_load_bearing_field_is_stubbed_rather_than_dropped():
    """`balanced_assessment` is interpolated into the risk_reviewer and
    final_investment_synthesizer prompts. Dropping it degrades two later
    stages; stubbing keeps the shape."""
    body = json.loads(_research_manager_response())
    body["balanced_assessment"] = "The consensus value points toward more upside."
    result = rp.run_research_pipeline(INDEX, make_ask_local(research_manager=json.dumps(body)))
    checkpoint = result.by_stage("research_manager")

    assert checkpoint.status == rp.StageStatus.COMPLETED
    assert checkpoint.output["balanced_assessment"] == rp._QUARANTINE_STUB_TEXT
    assert checkpoint.quarantines[0]["policy"] == rp.QUARANTINE_STUB
    assert result.by_stage("risk_reviewer").status == rp.StageStatus.COMPLETED
    assert result.by_stage("final_investment_synthesizer").status == rp.StageStatus.COMPLETED


def test_a_list_element_is_dropped_leaving_its_siblings():
    body = json.loads(_research_manager_response())
    body["supported_bull_points"] = ["Revenue growth is healthy.",
                                     "The balance sheet is best-in-class.",
                                     "Margins improved."]
    result = rp.run_research_pipeline(INDEX, make_ask_local(research_manager=json.dumps(body)))
    checkpoint = result.by_stage("research_manager")

    assert checkpoint.status == rp.StageStatus.COMPLETED
    assert checkpoint.output["supported_bull_points"] == ["Revenue growth is healthy.",
                                                          "Margins improved."]
    assert checkpoint.quarantines[0]["field_path"] == "supported_bull_points[1]"
    assert checkpoint.quarantines[0]["policy"] == rp.QUARANTINE_DROP


def test_multiple_quarantines_in_one_list_do_not_shift_each_other():
    """Deepest/highest-index first, so removing one element never
    invalidates a path that has not been applied yet."""
    body = json.loads(_research_manager_response())
    body["supported_bull_points"] = ["A fortress balance sheet.", "Margins improved.",
                                     "Returns are best-in-class."]
    result = rp.run_research_pipeline(INDEX, make_ask_local(research_manager=json.dumps(body)))
    checkpoint = result.by_stage("research_manager")

    assert checkpoint.output["supported_bull_points"] == ["Margins improved."]
    assert [q["field_path"] for q in checkpoint.quarantines] == [
        "supported_bull_points[0]", "supported_bull_points[2]"]


def test_the_quarantine_record_never_becomes_stage_content():
    """The record travels on the checkpoint, never inside `output` -- nothing
    downstream may mistake it for something the model wrote."""
    bad = _bull_response(extra={"thesis": "This is a fortress balance sheet."})
    result = rp.run_research_pipeline(INDEX, make_ask_local(bull=bad))
    output = result.by_stage("bull_researcher").output
    assert rp._QUARANTINE_KEY not in output
    assert "quarantine" not in json.dumps(output).lower()
    assert result.by_stage("bull_researcher").to_dict()["quarantines"]


def test_quarantine_is_deterministic():
    """Invariant 3. Same input, byte-identical records."""
    body = json.loads(_research_manager_response())
    body["supported_bull_points"] = ["A fortress balance sheet.", "Best-in-class returns."]
    body["balanced_assessment"] = "The consensus value points toward more upside."
    runs = [rp.run_research_pipeline(INDEX, make_ask_local(research_manager=json.dumps(body)))
            for _ in range(3)]
    records = [json.dumps(r.by_stage("research_manager").quarantines, sort_keys=True)
               for r in runs]
    assert records[0] == records[1] == records[2]


def test_records_are_sorted_by_field_path_then_rule_id():
    """Invariant 3 — a stable order, not dict/iteration order."""
    body = json.loads(_research_manager_response())
    body["supported_bull_points"] = ["Best-in-class returns."]
    body["balanced_assessment"] = "The consensus value points toward more upside."
    result = rp.run_research_pipeline(INDEX, make_ask_local(research_manager=json.dumps(body)))
    records = result.by_stage("research_manager").quarantines
    assert [(r["field_path"], r["rule_id"]) for r in records] == sorted(
        (r["field_path"], r["rule_id"]) for r in records)


def test_a_clean_stage_records_no_quarantines():
    result = rp.run_research_pipeline(INDEX, make_ask_local())
    for stage in ("bull_researcher", "bear_researcher", "rebuttal_round",
                  "research_manager", "risk_reviewer", "final_investment_synthesizer"):
        assert result.by_stage(stage).quarantines == [], stage


def test_no_ticker_symbols_appear_in_the_validation_code():
    """Invariant 4."""
    import pathlib
    import re

    root = pathlib.Path(rp.__file__).parent
    # A bare 2-5 letter uppercase token in a string literal would be the
    # shape of a hardcoded ticker.
    suspicious = re.compile(r"[\"'](?:AOS|WM|TSLA|AMZN|MSFT|COST|DIS|COR|MLI|VZ|GE|HOOD|UNH|AMD)[\"']")
    for name in ("content_policy.py", "claim_validation.py", "research_pipeline.py"):
        text = (root / name).read_text(encoding="utf-8")
        code = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))
        assert not suspicious.search(code), f"{name} contains a hardcoded ticker"
