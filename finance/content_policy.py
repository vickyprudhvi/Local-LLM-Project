"""Deterministic screening for claims this system has no basis to make.

Originally (Phase H.3 corrective patch, Problem 1) this module banned ALL
trade-advice language, including the bare verdict words buy/sell/hold/avoid.
That is no longer its job: the tool now produces an explicit buy/hold/sell/
avoid `recommendation` as a validated enum field, and those words are no
longer prohibited in prose (see the long note at the top of
`_PROHIBITED_PATTERNS` for the full reasoning and what was deliberately
NOT relaxed).

What remains is narrower and has a single unifying rationale: reject
statements the system lacks the INFORMATION to support. It cannot know the
reader's capital, risk tolerance, time horizon, or whether they hold a
position — so order mechanics (position size, entry/exit price, stop-loss,
target allocation) and holding-conditional phrasing ("if you hold ...") stay
banned, not because they are advice but because they would be fabricated.
Price-target/guarantee/urgency patterns stay for the same reason
finance/claim_validation.py exists: they claim a certainty the evidence
cannot carry.

A mechanical, deterministic scan applied AFTER schema validation, not a
judgment call left to the model's own restraint — reused identically by BOTH
the staged FinalInvestmentSynthesizer stage (finance/research_pipeline.py)
and the fallback single-shot report path (finance/workflow.py).

Deliberately literal, word-boundary regex matching, not semantic
understanding — false positives are handled by the caller's ONE constrained
repair attempt; missing a genuine violation is the worse failure mode here,
so this errs toward being aggressive rather than lenient.

GE corrective patch: a live GE run (research_manager, 4 of 4 runs) showed the
single most common trip was the model correctly explaining that a banned
claim was NOT being made -- "which is not guaranteed given the 32.4%
annualized volatility" flagged 'guarantee' even though the sentence is a
textbook example of the QUALIFIED language this module asks for elsewhere.
Fixed with `_is_disclaimed`, which exempts 'guarantee'/'price target'
specifically (never the order-mechanics or holding-conditional patterns,
which stay maximally strict) when the containing text also carries an
explicit disclaimer word ("unavailable", "not guaranteed", ...) -- the same
false-positive shape the pre-existing "not a price target" lookbehind
exemption already targets, just not limited to the negation sitting
immediately before the phrase. (That patch's OTHER half -- a narrow
negative-lookahead on 'avoid' for benign continuations like "avoid
per-share metric errors" -- became moot when the bare verdict words were
removed entirely by the personal-use verdict patch, and was deleted with
them.)
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


class Severity:
    """Phase H.5 (validation rework), Phase 2 — what a match COSTS.

    The 47 patterns across this module and finance/claim_validation.py are
    two unrelated checks that have shared one consequence: kill the stage.

    FABRICATION guards statements the system has no INFORMATION to make --
    a position size needs the reader's capital and risk tolerance, "if you
    hold" needs knowledge of their holdings. There is no true version of
    these sentences, so they stay cascade-fatal. The set is small, closed
    and concrete; regex is the right tool for it.

    OVERSTATEMENT guards language stronger than the evidence supports.
    That is a judgment about how confidently a sentence asserts something,
    over unbounded English, decided by substring match -- which does not
    converge, as five rounds of vocabulary patching demonstrated. A match
    here now QUARANTINES the offending field rather than destroying six
    stages of work. The claim underneath still carries mandatory evidence
    IDs, which is the deterministic backstop that never false-positives.
    """

    FABRICATION = "fabrication"
    OVERSTATEMENT = "overstatement"
    ALL = (FABRICATION, OVERSTATEMENT)


@dataclass(frozen=True)
class Rule:
    """One pattern with a STABLE identity.

    `rule_id` is assigned explicitly, never derived from list position, so
    inserting or removing a pattern cannot silently renumber the others --
    these ids appear in quarantine records and in the Phase 5 aggregation
    that decides which patterns get deleted, and both are worthless if an
    id can change meaning between runs.
    """

    rule_id: str
    pattern: "re.Pattern"
    label: str
    severity: str
    # Phase H.5, Phase 4: applies ONLY to prose fields, never to a field
    # whose value is a closed enum. `recommendation` is validated against
    # ("buy", "hold", "sell", "avoid", "insufficient_evidence") -- four of
    # those five ARE bare imperative verbs at position zero, so a rule that
    # scanned every string would make every real recommendation
    # cascade-fatal, deleting a feature this project deliberately built.
    prose_only: bool = False

_PROHIBITED_PATTERNS = [
    # Personal-use verdict patch: the bare verdict words -- STRONG BUY,
    # STRONG SELL, BUY, SELL, HOLD, AVOID -- were REMOVED from this list.
    # They are no longer prohibited anywhere in free text.
    #
    # Why: this tool now produces an explicit buy/hold/sell/avoid
    # `recommendation` as a first-class validated enum field (see the
    # "Recommendation reintroduction" note above Stage 6 in
    # finance/research_pipeline.py). Banning the same words in the PROSE
    # around that field was enforcing a rule the project deliberately
    # reversed -- and, observed live across four tickers (GE/HOOD/UNH/AMD),
    # was a meaningful share of this scanner's false positives ("both sides
    # confirm they hold that view", "avoid per-share metric errors", "share
    # buy back"), each costing a repair round-trip or a hard stage failure.
    #
    # What this does NOT relax, and why -- the patterns kept below are not
    # about advice at all, they are about the tool lacking the INFORMATION
    # to say the thing:
    #   * order mechanics (position size, entry/exit price, stop-loss,
    #     target allocation) -- these need the reader's capital, risk
    #     tolerance, time horizon and tax situation; this system has none of
    #     them, so such a figure would be fabricated, not advisory.
    #   * holding-conditional phrasing ("if you hold" / "if you do not
    #     hold") -- this system cannot know whether the reader holds a
    #     position. This is the SPECIFIC defect behind the original Phase
    #     H.3 incident ("If you do not currently hold a position: AVOID. If
    #     you already hold a position: SELL.") -- the problem there was
    #     branching on an invented fact about the reader, not the verdict.
    #   * price target / guarantee / reader-directed urgency -- overstatement
    #     and certainty claims, the same family finance/claim_validation.py
    #     polices; unaffected by wanting a verdict.
    (re.compile(r"\bposition\s+siz(?:e|ing)\b", re.IGNORECASE), "position size"),
    (re.compile(r"\bentry\s+price\b", re.IGNORECASE), "entry price"),
    (re.compile(r"\bexit\s+price\b", re.IGNORECASE), "exit price"),
    (re.compile(r"\bstop[\s-]?loss\b", re.IGNORECASE), "stop loss"),
    (re.compile(r"\btarget\s+allocation\b", re.IGNORECASE), "target allocation"),
    # Personal-use verdict patch: the adverb may appear on EITHER side of the
    # negation ("if you do not CURRENTLY hold", "if you CURRENTLY do not
    # hold"), so both positions are optional and independent. This word order
    # matters more than it looks: "If you do not currently hold a position:
    # AVOID." is the literal sentence from the original Phase H.3 incident,
    # and the older fixed-order pattern never matched it -- it was caught
    # only INCIDENTALLY by the bare \bhold\b verdict pattern, which this same
    # patch removed. Without this fix, removing the verdict words would have
    # silently un-caught the exact output the whole guardrail exists for.
    # Covered by test_holding_dependent_phrasing_catches_both_word_orders.
    # The "hold true/up/across/steady/constant/that" exemption is carried over
    # from the removed bare \bhold\b pattern -- "if you hold that assumption
    # constant" is a modeling construction, not a question about the reader's
    # portfolio. Same accepted, documented precision gap the old pattern had:
    # a literal "if you hold that stock" is missed. Genuine holding-conditional
    # advice overwhelmingly says "hold this stock"/"hold a position"/"hold
    # shares", all still caught.
    (re.compile(r"\bif\s+you\s+"
               r"(?:(?:already|currently)\s+)?"
               r"(?:(?:do\s+not|don't)\s+)?"
               r"(?:(?:already|currently)\s+)?"
               r"(?:hold|own)\b(?!\s+(?:true|up|across|steady|constant|that)\b)",
               re.IGNORECASE),
     "holding-dependent phrasing"),
    (re.compile(r"\border\s+instructions?\b", re.IGNORECASE), "order instructions"),
    (re.compile(r"\btrading\s+instructions?\b", re.IGNORECASE), "trading instructions"),
    # NOT flagged when explicitly negated ("not a price target", "never a
    # price target", ...) -- Problem 9's own encouraged phrasing ("a modeled
    # value, not a price target") reads exactly this way, and rejecting the
    # policy-compliant phrasing would be self-defeating. Both word orders are
    # covered ("price target" and "target price" -- found live, COR
    # corrective patch: the reversed order was a real gap, matched by
    # neither this nor any other pattern). `[\s-]+` (H.4 corrective patch)
    # rather than `\s+` alone -- a hyphenated "price-target"/"target-price"
    # is the same claim and was a real gap, slipping past a whitespace-only
    # separator entirely.
    (re.compile(r"(?<!not a )(?<!not the )(?<!not an )(?<!never a )(?<!nor a )"
               r"\bprice[\s-]+targets?\b", re.IGNORECASE), "price target"),
    (re.compile(r"(?<!not a )(?<!not the )(?<!not an )(?<!never a )(?<!nor a )"
               r"\btarget[\s-]+prices?\b", re.IGNORECASE), "price target"),
    # "guarantee(s/d/ing)" -- was bare "guaranteed?", which only matched
    # "guarantee"/"guaranteed" and missed the plain verb form "guarantees"
    # (found live, COR corrective patch: "this guarantees upside" slipped
    # through entirely).
    (re.compile(r"\bguarantee(?:s|d|ing)?\b", re.IGNORECASE), "guarantee"),
    # Reader-directed investment imperatives (COR corrective patch, Phase 3's
    # "invest now" example) -- distinct from a COMPANY investing in itself
    # ("management continues to invest in automation"), which is common,
    # neutral, and must stay unflagged; only "invest" paired with an
    # immediacy adverb or an explicit "you/investors should" frame is
    # reader-directed advice.
    (re.compile(r"\binvest(?:ing)?\s+(?:now|today|immediately)\b", re.IGNORECASE),
     "reader-directed investment imperative"),
    (re.compile(r"\b(?:you\s+should|investors?\s+should)\s+invest\b", re.IGNORECASE),
     "reader-directed investment imperative"),
    # -----------------------------------------------------------------
    # Phase H.5, Phase 4 -- FABRICATION DEFINED BY ADDRESSING, NOT TOPIC.
    #
    # The eight original fabrication patterns are TOPIC bans: a list of
    # things not to mention. The Phase H.3 incident was not about a topic.
    # "If you do not currently hold a position: AVOID. If you already hold a
    # position: SELL." is harmful because it addresses the reader as an agent
    # who holds a position and should act -- and this system cannot know
    # either of those things. Banning the addressing catches the shape rather
    # than enumerating its vocabulary, which is the whole point of this
    # rework.
    #
    # The gap that made this necessary: the original CP-006 matches only the
    # SECOND-PERSON framing. Three third-person variants escaped entirely and
    # were carried as xfail markers through phases 2 and 3:
    #     "For readers who currently hold a position: hold."
    #     "Investors who already hold shares should trim."
    #     "Those holding a position may want to exit."
    #
    # The design point, learned by testing candidates against real prose: the
    # harm is a directive whose SUBJECT IS THE READER. Neither half alone
    # works. "The company should trim its cost base" is a directive with a
    # different subject and must pass; "Investors who hold the stock have
    # seen a 20% decline" references holdings but is descriptive and must
    # pass. Only the conjunction is the violation.
    #
    # -- CP-014: second person, anywhere in prose.
    # Deliberately broad. Stage output describes a company TO a reader; it
    # never addresses them. No legitimate use was found across the six
    # schemas.
    (re.compile(r"\b(?:you|your|yours|yourself)\b", re.IGNORECASE),
     "reader-addressed second person"),
    # -- CP-015: a directive whose subject is the reader.
    (re.compile(
        r"\b(?:you|readers?|investors?|shareholders?|holders?|anyone|those|whoever)\b"
        r"[^.;!?]{0,60}?"
        r"\b(?:should|must|ought\s+to|need\s+to|may\s+want\s+to|might\s+want\s+to|"
        r"are\s+advised\s+to|are\s+encouraged\s+to)\b"
        r"[^.;!?]{0,20}?"
        r"\b(?:buy|sell|hold|avoid|add|trim|exit|enter|reduce|increase|accumulate|"
        r"liquidate|divest|purchase)\b",
        re.IGNORECASE),
     "reader-directed action"),
    # -- CP-016: a bare trading imperative opening a clause.
    #
    # The verb list is NARROWER than the phase spec proposed. 'consider',
    # 'take', 'wait', 'increase' and 'reduce' were dropped after they
    # false-positived on ordinary analysis prose ("Consider the wide scenario
    # spread when weighing this", "Increase in receivables drove the swing").
    # They remain covered by CP-015, where a reader subject disambiguates
    # them. The trailing lookahead requires an object marker or end of
    # clause, so a compound noun ("Exit rates improved", "Add-on
    # acquisitions", "Hold-to-maturity securities") is not an imperative.
    (re.compile(
        r"(?:"
        # (a) opening the text outright -- "Sell into strength."
        r"^\s*(?:buy|sell|hold|avoid|add|trim|exit|enter)"
        r"(?=[.!?;:,]|\s+(?:into|out|at|on|to|the|this|that|these|those|it|some|all|"
        r"any|more|now|today|immediately|here|your|shares?|stock|position|exposure)\b)"
        r"|"
        # (b) after a colon/semicolon whose clause REFERENCES THE READER --
        # "For readers who currently hold a position: hold."
        #
        # The reader referent is required. Without it this branch also
        # flagged "Overall: AVOID." and "Recommendation: hold", which the
        # personal-use verdict patch deliberately allows: a bare verdict is
        # a characterization the tool now emits as a first-class enum, and
        # banning it in prose would re-impose a rule this project reversed
        # on purpose. What makes the H.3 shape different is not the colon,
        # it is the reader on the other side of it.
        r"\b(?:you|your|readers?|investors?|shareholders?|holders?|anyone|those|whoever)\b"
        r"[^.!?]{0,80}?[:;]\s*(?:buy|sell|hold|avoid|add|trim|exit|enter)"
        r"(?=[.!?;:,]|\s+(?:into|out|at|on|to|the|this|that|these|those|it|some|all|"
        r"any|more|now|today|immediately|here|your|shares?|stock|position|exposure)\b)"
        r")",
        re.IGNORECASE),
     "bare trading imperative"),
]

# Stable rule ids and severities for the patterns above, in list order.
#
# Held as a separate table rather than inlined into each tuple so the pattern
# list stays readable and every existing consumer of `_PROHIBITED_PATTERNS`
# keeps working unchanged. The label is repeated here purely as an assertion:
# `_build_rules` checks it against the pattern list and raises at import if
# they diverge, so reordering, inserting or deleting a pattern fails LOUDLY
# instead of silently renumbering every id after it. That matters because
# these ids appear in quarantine records and in the Phase 5 aggregation that
# decides which patterns get deleted -- an id that quietly changes meaning
# between runs would corrupt both.
_CONTENT_POLICY_RULE_SPECS = (
    # -- FABRICATION: no true version of the sentence exists, because the
    # system lacks the information (the reader's capital, risk tolerance,
    # time horizon, or whether they hold a position). Cascade-fatal.
    ("CP-001", "position size", Severity.FABRICATION, False),
    ("CP-002", "entry price", Severity.FABRICATION, False),
    ("CP-003", "exit price", Severity.FABRICATION, False),
    ("CP-004", "stop loss", Severity.FABRICATION, False),
    ("CP-005", "target allocation", Severity.FABRICATION, False),
    ("CP-006", "holding-dependent phrasing", Severity.FABRICATION, False),
    ("CP-007", "order instructions", Severity.FABRICATION, False),
    ("CP-008", "trading instructions", Severity.FABRICATION, False),
    # -- OVERSTATEMENT: a real claim, phrased more strongly than the evidence
    # carries. Quarantined, not fatal.
    ("CP-009", "price target", Severity.OVERSTATEMENT, False),
    ("CP-010", "price target", Severity.OVERSTATEMENT, False),
    ("CP-011", "guarantee", Severity.OVERSTATEMENT, False),
    # -- FABRICATION, and NOT the 8 originally specced.
    #
    # These two address the reader as an agent who should act ("you should
    # invest", "invest now") -- which IS the Phase H.3 harm mechanism, the
    # incident this whole validation layer exists because of. Grouping them
    # with the other content-policy patterns as "overstatement" would have
    # made them quarantinable, weakening invariant 1 between this phase and
    # Phase 4. They are cascade-fatal from here, which also makes Phase 4's
    # general addressing rule a GENERALIZATION of an existing ban rather
    # than a re-tightening of a relaxed one.
    ("CP-012", "reader-directed investment imperative", Severity.FABRICATION, False),
    ("CP-013", "reader-directed investment imperative", Severity.FABRICATION, False),
    # -- Phase 4: fabrication defined by ADDRESSING. Prose-only, because four
    # of the five `recommendation` enum members are bare imperative verbs.
    ("CP-014", "reader-addressed second person", Severity.FABRICATION, True),
    ("CP-015", "reader-directed action", Severity.FABRICATION, True),
    ("CP-016", "bare trading imperative", Severity.FABRICATION, True),
)


# Leaf field names whose value is a closed enum, an identifier, or a
# structured token -- never prose. Phase 4's addressing rules skip these.
#
# An ALLOWLIST OF NON-PROSE, not of prose, so the fail-safe direction is
# "scanned": a prose field added to a schema later is checked by default
# rather than silently exempt. A new ENUM field not listed here would be
# scanned and could false-positive -- loud, and preferable to a silent gap.
NON_PROSE_FIELDS = frozenset({
    "recommendation", "research_stance", "valuation_view", "overall_risk",
    "aggregated_risk", "model_risk", "evidence_balance", "severity",
    "confidence", "claim_type", "claim_id", "role",
    "evidence_cited", "evidence_ids",
})


def _leaf_name(field_path: str) -> str:
    return field_path.split(".")[-1].split("[")[0] if field_path else ""


def _is_prose_field(field_path: str) -> bool:
    """Unknown context counts as prose.

    `scan_for_prohibited_directives(text)` is called on bare strings with no
    field path at all (the single-shot fallback report path). Treating an
    unknown context as non-prose would silently exempt it.
    """
    return _leaf_name(field_path) not in NON_PROSE_FIELDS


def _build_rules(patterns, specs, module_label) -> Tuple[Rule, ...]:
    """Zip patterns with their id/severity table, asserting alignment."""
    if len(patterns) != len(specs):
        raise RuntimeError(
            f"{module_label}: {len(patterns)} patterns but {len(specs)} rule specs. "
            "Every pattern needs an explicit, stable rule id and severity.")
    rules = []
    for (pattern, label), (rule_id, expected_label, severity, prose_only) in zip(patterns, specs):
        if label != expected_label:
            raise RuntimeError(
                f"{module_label}: rule {rule_id} expects label {expected_label!r} but the "
                f"pattern at that position has label {label!r}. The pattern list and the rule "
                "table have drifted out of alignment -- fix the table rather than reordering "
                "ids, or every existing quarantine record changes meaning.")
        if severity not in Severity.ALL:
            raise RuntimeError(f"{module_label}: rule {rule_id} has unknown severity {severity!r}.")
        rules.append(Rule(rule_id=rule_id, pattern=pattern, label=label,
                          severity=severity, prose_only=prose_only))
    return tuple(rules)


RULES = _build_rules(_PROHIBITED_PATTERNS, _CONTENT_POLICY_RULE_SPECS, "content_policy")
RULES_BY_ID = {rule.rule_id: rule for rule in RULES}

# GE corrective patch: a string that explicitly says a claim is NOT available/
# supported/certain still contains the banned WORD ("...is not guaranteed
# given the volatility") -- the pre-existing "not a price target" lookbehind
# exemption only covers a disclaimer sitting IMMEDIATELY before the phrase,
# not one stated elsewhere in the same sentence/field ("there is no
# guarantee that..."). Scoped to ONLY 'guarantee' and 'price target' -- the
# two labels actually observed live tripping on disclaimed text -- never to
# buy/sell/hold/avoid/position-size/entry-exit-price/stop-loss/order-
# instructions, which stay maximally strict with no disclaimer exemption at
# all (a hedged "not a strong buy" is still adjacent to trade-advice territory
# in a way "not guaranteed" or "not a price target" is not).
_DISCLAIMER_MARKERS = (
    "unavailable", "unsupported", "impossible", "not available", "not possible",
    "not supported", "not guaranteed", "no guarantee", "cannot be", "can not be",
    "can't be",
    # HOOD corrective patch: kept in lockstep with finance/claim_validation.
    # py's identical broadening -- a live HOOD run disclaimed 'guarantee'
    # with a bare present-tense negated verb ("...do not guarantee future
    # performance"), not the past-participle "not guaranteed" this list
    # already covered. General vocabulary, not a per-ticker patch: see that
    # module's longer comment for why enumerating exact phrasings doesn't
    # scale.
    "do not guarantee", "does not guarantee", "did not guarantee", "lack of", "lacks",
    "prevents", "preventing", "prevented", "inability", "unable to", "absence of",
    # UNH corrective patch: kept in lockstep with finance/claim_validation.
    # py's identical broadening -- a live UNH run critiqued ANOTHER claim's
    # overreach ("Assertions that free cash flow 'provides ample liquidity'
    # ... imply a causal guarantee; while FCF is strong, the current ratio
    # remains below 1.0") by NAMING the fallacy type being rejected, not
    # committing it -- 'implies'/'imply' is semantically no stronger than
    # 'is consistent with', already explicitly endorsed as acceptable
    # qualified language. See that module's longer comment for the full
    # reasoning and the OTHER two live examples ('whether', 'depends on
    # which/whether') this same broadening also covers.
    "whether", "implies", "imply", "implied",
    # AMD corrective patch: kept in lockstep with finance/claim_validation.
    # py's identical broadening -- a live AMD run rejected an over-
    # deterministic framing with the gerund "rather than IMPLYING an
    # inevitable reversion", not yet covered by the base verb forms above.
    "implying",
)
# Same "no X is/are available/provided" proximity gap as finance/
# claim_validation.py's identical fix (a leading "no" quantifier rather than
# a "not"/"un-" prefix directly on the disclaiming word) -- not observed
# live for THIS module's two exempt labels yet, kept in lockstep with its
# sibling anyway since both scan the same model output for the same class of
# claim and a future live run tripping it here would be the identical bug.
# The "depends...on which/whether" pattern (UNH corrective patch) is kept in
# lockstep for the same reason -- a regex proximity check, not a literal
# substring, so filler words like "entirely" between "depends" and "on"
# don't break it.
_DISCLAIMER_PROXIMITY_PATTERNS = (
    re.compile(r"\bno\b[\s\S]{0,80}?\b(?:available|provided|exists?)\b", re.IGNORECASE),
    re.compile(r"\bdepends\b[\s\S]{0,30}?\bon\b[\s\S]{0,15}?\b(?:which|whether)\b", re.IGNORECASE),
)
_DISCLAIMER_EXEMPT_LABELS = frozenset({"guarantee", "price target"})


def _is_disclaimed(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in _DISCLAIMER_MARKERS):
        return True
    return any(pattern.search(text) for pattern in _DISCLAIMER_PROXIMITY_PATTERNS)


# WM corrective patch -- GENERAL NEGATION PROXIMITY.
#
# `_DISCLAIMER_MARKERS` above is a hand-enumerated vocabulary list that has
# been extended four times (GE, HOOD, UNH, AMD), each time by a live run
# finding one more way to say "this is not being claimed". Its own comments
# admit enumerating exact phrasings does not scale. It also matches against
# the WHOLE string, so it cannot tell WHICH phrase a negation applies to.
#
# Live WM run, research_manager, the failure this fixes:
#
#   "These modeled values should not be interpreted as a price target."
#       -> FLAGGED. The negation lookbehinds on the pattern itself only
#          cover "not a ", "not the ", "not an ", "never a ", "nor a "
#          IMMEDIATELY before the phrase; "as a price target" is not covered,
#          and "should not be interpreted" is not in the marker vocabulary.
#
#   "The bear case guarantees nothing about future returns."
#       -> FLAGGED. The negation follows the word rather than preceding it.
#
# Both are textbook examples of the qualified language this module asks for
# everywhere else, and the repair prompt cannot reliably fix them: the model
# rewrites the sentence and says the same true thing a different way, which
# trips again. That is precisely the "failed ... after one repair attempt"
# shape.
#
# So instead of adding two more phrases, this checks for ANY negation cue
# NEAR THE ACTUAL MATCH, in either direction. It only ever REMOVES findings,
# and only for the two labels already treated as disclaimable -- order
# mechanics and holding-conditional phrasing stay maximally strict, because
# those are banned for lacking information, not for overclaiming certainty.
_NEGATION_CUES = re.compile(
    r"\b(?:not|never|no|nothing|none|nor|neither|cannot|can't|cant|without|"
    r"excludes?|precludes?|rather\s+than|instead\s+of|absent)\b",
    re.IGNORECASE)

# How far from the match a negation still counts. Asymmetric: qualifying
# language usually PRECEDES the phrase ("should not be read as a price
# target") and needs more room than the trailing form ("guarantees nothing").
_NEGATION_LOOKBEHIND = 60
_NEGATION_LOOKAHEAD = 30


def _match_is_negated(text: str, match) -> bool:
    before = text[max(0, match.start() - _NEGATION_LOOKBEHIND):match.start()]
    after = text[match.end():match.end() + _NEGATION_LOOKAHEAD]
    return bool(_NEGATION_CUES.search(before) or _NEGATION_CUES.search(after))


def scan_for_prohibited_directives(text) -> List[str]:
    """Every distinct prohibited-language label found in one string, in
    pattern-list order. Empty means clean. A non-string input is always
    clean (nothing to scan) rather than raising."""
    if not isinstance(text, str) or not text:
        return []
    found = []
    for pattern, label in _PROHIBITED_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        # A negated occurrence of a disclaimable label is the model saying
        # the claim is NOT being made -- the opposite of a violation.
        if label in _DISCLAIMER_EXEMPT_LABELS and _match_is_negated(text, match):
            continue
        found.append(label)
    if found and _is_disclaimed(text):
        found = [label for label in found if label not in _DISCLAIMER_EXEMPT_LABELS]
    return found


@dataclass(frozen=True)
class Finding:
    """One match, with everything a quarantine record needs.

    The pre-existing scanners return LABELS only, which is enough to fail a
    stage but not to quarantine a field: you cannot remove the offending text
    without knowing which field it was in. `field_path` is a dotted/indexed
    path into the validated stage output (e.g.
    `key_risks[0].risk`, `supported_bull_points[2]`).
    """

    rule_id: str
    label: str
    severity: str
    field_path: str
    matched_span: str

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id, "label": self.label, "severity": self.severity,
            "field_path": self.field_path, "matched_span": self.matched_span,
        }


def find_prohibited_directives(text, field_path: str = "") -> List[Finding]:
    """Every match in ONE string, with its rule id and matched span.

    Applies the same negation and disclaimer exemptions as
    `scan_for_prohibited_directives` -- this is a different RETURN SHAPE for
    the same policy, never a second, divergent policy. Anything exempted
    there is exempted here.
    """
    if not isinstance(text, str) or not text:
        return []
    disclaimed = _is_disclaimed(text)
    is_prose = _is_prose_field(field_path)
    findings: List[Finding] = []
    for rule in RULES:
        if rule.prose_only and not is_prose:
            continue
        match = rule.pattern.search(text)
        if match is None:
            continue
        if rule.label in _DISCLAIMER_EXEMPT_LABELS:
            if _match_is_negated(text, match) or disclaimed:
                continue
        findings.append(Finding(
            rule_id=rule.rule_id, label=rule.label, severity=rule.severity,
            field_path=field_path, matched_span=match.group(0)))
    return findings


def _walk_fields(value, prefix: str = ""):
    """Yield (field_path, string) for every string in a nested structure.

    Deterministic: dict keys are visited in sorted order and list indices in
    natural order, so the same structure always produces the same paths in
    the same sequence (invariant 3).
    """
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, dict):
        for key in sorted(value.keys(), key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_fields(value[key], child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_fields(item, f"{prefix}[{index}]")


def find_prohibited_directives_in_structure(value) -> List[Finding]:
    """Path-aware scan of a whole validated stage output."""
    findings: List[Finding] = []
    for field_path, text in _walk_fields(value):
        findings.extend(find_prohibited_directives(text, field_path))
    return findings


def scan_structure_for_prohibited_directives(value) -> List[str]:
    """Recursively scans every string inside a dict/list/tuple/scalar
    structure (a validated stage output) for prohibited directives, de-
    duplicated, order preserved. Used on the FinalInvestmentSynthesizer's
    full validated output — a violation could be in 'rationale', a
    'conditions_that_*' entry, or 'key_uncertainties' just as easily as
    anywhere else, so every string field is in scope, not an allowlist of
    "likely" fields.
    """
    found: List[str] = []
    if isinstance(value, str):
        found.extend(scan_for_prohibited_directives(value))
    elif isinstance(value, dict):
        for v in value.values():
            found.extend(scan_structure_for_prohibited_directives(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            found.extend(scan_structure_for_prohibited_directives(v))
    seen = set()
    deduped = []
    for label in found:
        if label not in seen:
            seen.add(label)
            deduped.append(label)
    return deduped
