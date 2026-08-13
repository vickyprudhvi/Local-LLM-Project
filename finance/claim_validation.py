"""Phase H.3 corrective patch (Problem 5) — deterministic semantic-claim
screening: evidence IDs existing is not the same as the claim being TRUE to
that evidence.

The live COST report showed real evidence IDs cited next to text that
overstated what they show -- "fortress balance sheet" and "industry-leading
ROE" both cited a genuine metric, and "confirms a reversal" cited a genuine
MACD value, but the surrounding language claimed far more than one metric can
support. `finance/evidence.py` already rejects a HALLUCINATED citation; this
module rejects language that cites a REAL fact but overstates it.

Covers, deterministically:
  * superlative control      -- "fortress", "industry-leading", "best-in-class",
                                "safest", "guaranteed", "certain to",
                                "inevitable", and similar unearned praise.
                                ("best"/"leading"/"exceptional" on their own
                                are deliberately NOT flagged -- see
                                _SUPERLATIVE_PATTERNS's comment for why.)
  * causal-overreach control -- "confirms a reversal", "proves", "protects
                                from downside", "eliminates the risk", and
                                similar claims of certainty a single metric
                                cannot support.
  * consensus-language ban   -- "consensus value/estimate", "analyst
                                consensus". This system has NO analyst-
                                consensus data source (see finance/evidence.py
                                -- no such field is ever indexed), so a claim
                                of "consensus" is definitionally unsupported,
                                not merely risky.
  * claim-type fidelity      -- technical indicators describe HISTORY, not the
                                future ("will reverse", "poised to rally");
                                see FINANCE_REPORT_SYSTEM_INSTRUCTIONS item 5
                                for the prompt-level half of this same rule.
  * DCF terminology control  -- H.4 corrective patch: "intrinsic value",
                                "authoritative value"/"authoritative intrinsic
                                value", and "fair-value target" describing a
                                finance.dcf_model scenario output are all
                                DEFINITIONALLY unsupported -- this system
                                produces three MODELED scenarios (base/bull/
                                bear), never a single objective or
                                authoritative figure. Only "base/bull/bear
                                modeled value", "modeled value per share",
                                "valuation model output", or "scenario-based
                                modeled value" are accepted. ("price target"/
                                "target price" is a separate, pre-existing
                                ban in finance/content_policy.py; "consensus
                                value/estimate" is the existing consensus-
                                language ban directly below.)
  * provenance fidelity      -- flags a claim that names a specific data
                                provider (SEC / Yahoo / Alpha Vantage) which
                                contributed NOTHING to this analysis (its
                                `provenance.*.provider` value never appears in
                                the evidence index) -- the most damaging real-
                                world case is a reduced-mode report where SEC
                                data was omitted and every fact actually came
                                from Yahoo, but the text still says "per SEC
                                filings".

Deliberately NOT implemented: free-text arithmetic/number matching against
cited evidence values ("numeric fidelity" in the fullest sense). Bull/Bear
commentary routinely derives a correct number that is not literally present
as a scalar in the evidence index (e.g. a YoY percentage computed from two
cited revenue figures) -- a naive matcher would reject valid, well-supported
claims, and even with every pipeline stage now covered by one repair
attempt (GE corrective patch: rebuttal_round was the last holdout -- see
finance/research_pipeline.py), a false reject still costs a real repair
round-trip, and can still fail closed if the repair doesn't land. This is a
deliberate, documented scope decision, not an oversight -- see
docs/PHASE_H3_MULTI_PROVIDER.md "Known limitations". The existing
evidence-ID citation requirement (finance/evidence.py) remains the
structural backstop: every specific claim must still name a real fact.

Same literal, word-boundary regex philosophy as finance/content_policy.py:
false positives are an accepted, documented cost; missing a genuine
overstatement is the worse failure mode.

GE corrective patch: a live GE run (whose DCF failed deterministic
validation -- see finance/dcf.py's DCF_ASSUMPTION_REQUIRED -- so every stage
was explicitly prompted to explain that valuation evidence is unavailable)
showed that instruction colliding head-on with these scanners: "no DCF ...
available to model future cash flows or intrinsic value", "there is no
quantitative benchmark provided to prove the price is unfair", "both sides
confirm the stock is technically trading above its ... moving averages" were
ALL the model correctly disclaiming a banned claim or describing agreement
between the bull/bear cases, not making the claim -- yet every one tripped a
bare word/phrase match with no way to tell the difference. Two narrow,
still-deterministic fixes: `_is_disclaimed` exempts the causal-overreach,
consensus-language, and DCF-terminology labels (plus the 'guarantee'
superlative) specifically when the containing text also carries an explicit
disclaimer word ("unavailable", "not guaranteed", ...); and 'confirms' gained
a lookbehind excluding "(both/each) side(s) confirm" specifically, since the
ResearchManager schema explicitly invites describing what the bull and bear
cases agree on (`shared_findings`) and "confirm" is a natural, non-causal
synonym for "agree" in that one grammatical position. Every OTHER
superlative/causal-overreach pattern is untouched and stays maximally
strict -- this is not a general negation detector (see the "exceptional"
history above for why that was rejected), just the two specific shapes
proven live.
"""

import re
from typing import Iterable, List, Optional, Set

# Phase H.5, Phase 2: the rule/finding vocabulary is defined once, in
# finance/content_policy.py, and shared. Two divergent definitions of "what a
# match is" is exactly the kind of drift this rework exists to remove.
from finance.content_policy import Finding, Rule, Severity, _walk_fields

_SUPERLATIVE_PATTERNS = [
    (re.compile(r"\bfortress\b", re.IGNORECASE), "unsupported superlative (fortress)"),
    (re.compile(r"\b(?:industry|market|sector|category)[\s-]leading\b", re.IGNORECASE),
     "unsupported superlative (industry-leading)"),
    (re.compile(r"\bbest[\s-]in[\s-]class\b", re.IGNORECASE), "unsupported superlative (best-in-class)"),
    (re.compile(r"\bworld[\s-]class\b", re.IGNORECASE), "unsupported superlative (world-class)"),
    (re.compile(r"\bsafest\b", re.IGNORECASE), "unsupported superlative (safest)"),
    # "guarantee(s/d/ing)" -- was bare "guaranteed?", which missed the plain
    # verb form "guarantees" (found live, COR corrective patch: "this
    # guarantees upside" slipped through). finance/content_policy.py has the
    # identical fix for the same reason; both modules' hits are merged and
    # de-duplicated by _validate_claim_fidelity, so the overlap is harmless.
    (re.compile(r"\bguarantee(?:s|d|ing)?\b", re.IGNORECASE), "unsupported superlative (guarantee)"),
    (re.compile(r"\bcertain\s+to\b", re.IGNORECASE), "unsupported superlative (certain to)"),
    # "compelling investment case"/"compelling short case" -- Problem 7's
    # (prior corrective patch) exact named promotional phrases, told to the
    # model in its role-text prompt but never backstopped in code until now
    # (found unenforced live, COR corrective patch). Bare "compelling" is
    # deliberately NOT matched -- "compelling evidence" is common, neutral
    # analytical-writing language; only the specific case/thesis-framing
    # bigram is in scope.
    (re.compile(r"\bcompelling\s+(?:investment|short)(?:\s+case)?\b", re.IGNORECASE),
     "unsupported superlative (compelling investment/short case)"),
    (re.compile(r"\binevitab(?:le|ly)\b", re.IGNORECASE), "unsupported superlative (inevitable)"),
    (re.compile(r"\bunrivall?ed\b", re.IGNORECASE), "unsupported superlative (unrivaled)"),
    (re.compile(r"\bunmatched\b", re.IGNORECASE), "unsupported superlative (unmatched)"),
    (re.compile(r"\bbulletproof\b", re.IGNORECASE), "unsupported superlative (bulletproof)"),
    (re.compile(r"\bunbeatable\b", re.IGNORECASE), "unsupported superlative (unbeatable)"),
    # Deliberately NOT bare-matched: "best" and "leading" on their own.
    # "best estimate"/"best available data" and "leading indicator" (a
    # standard, neutral technical-analysis term of art) are common, correct,
    # non-promotional phrases -- only the compound promotional forms above
    # are flagged. Same false-positive-avoidance judgment call as the
    # deliberately-narrow "hold" pattern in finance/content_policy.py.
    #
    # Also deliberately NOT bare-matched (removed after a live incident):
    # "exceptional" on its own. A real bull_researcher run on COR wrote
    # "exceptional return on equity" as part of a thesis-level summary
    # sentence, with the actual grounding number (a genuinely extreme 144%
    # ROE) cited two sentences later in a separate key_point -- a completely
    # reasonable way to write a 1-3 sentence thesis (the schema explicitly
    # separates a high-level `thesis` from the specific, evidence-cited
    # `key_points` that follow it), but it tripped a bare "exceptional"
    # reject and caused a TOTAL pipeline failure with no repair path (only
    # the FinalInvestmentSynthesizer gets one). A same-sentence "is there a
    # number nearby" heuristic was tried and rejected: it still flagged this
    # exact thesis sentence (the grounding number was in a different
    # field), and widening the window further starts accepting "exceptional
    # ... 21%" pairings where the number doesn't even describe the same
    # thing the superlative does -- genuine semantic grounding isn't
    # reachable deterministically here. Unlike "fortress"/"guaranteed"/
    # "industry-leading" (essentially never legitimate in evidence-based
    # writing), "exceptional" is a common, often-reasonable intensifier for
    # a genuinely extreme statistic; the false-positive rate demonstrated
    # live outweighs what it catches, so it is excluded like "best"/
    # "leading" rather than papered over with more heuristic complexity.
]


_CAUSAL_OVERREACH_PATTERNS = [
    # Severity-scoping patch: the bare `\bconfirms?\b` and `\bproves?\b`
    # patterns were REMOVED. They were this scanner's single highest-volume
    # trip (present in EVERY round of the live GE/HOOD/UNH/AMD
    # investigations, and the dominant blocker for AMD specifically), and
    # they sit in a different severity class from everything else here:
    #
    #   * "the MACD confirms a reversal" OVERSTATES a real, cited fact. The
    #     number behind it is genuine, present in the evidence index, and
    #     visible to the reader, who can discount the verb. The separate,
    #     deterministic evidence-ID citation requirement (finance/evidence.py)
    #     -- which never false-positives -- remains the real backstop.
    #   * "industry-leading ROE" / "consensus estimate" / "fortress balance
    #     sheet" FABRICATE data that does not exist anywhere (no peer feed,
    #     no analyst feed, contradicted by the cited figure). Nothing for a
    #     reader to discount; those patterns all stay.
    #
    # The cost of blocking was never "an accurate sentence instead" -- it was
    # a FAILED stage, cascading to a report with no Risk section, no Research
    # View and no recommendation at all. Trading three whole sections for a
    # punchy verb on a cited number is the wrong trade for a personal
    # research tool whose reader knows the text is model-generated.
    #
    # The NARROW compound forms below ("support is proven", "downside is
    # confirmed") are deliberately KEPT: unlike the bare verbs they have no
    # innocent reading, and they were never a meaningful false-positive
    # source. This also retires the GE "(both/each) side(s) confirm"
    # lookbehind, which existed only to carve an exception out of the bare
    # pattern now gone.
    (re.compile(r"\bprotects?\s+(?:the\s+company\s+|it\s+|investors?\s+)?(?:from|against)\s+(?:the\s+)?downside\b",
               re.IGNORECASE), "unqualified causal claim (protects from downside)"),
    # The NOUN-PHRASE form ("offers/provides significant downside
    # protection") is a distinct pattern from the VERB form above ("protects
    # ... from downside") -- neither regex matches the other's word shape.
    # Found live (COR corrective patch, Phase 2's explicit "downside
    # protection as a factual conclusion" example): slipped through both
    # existing scanners entirely. Qualified language ("net cash may provide
    # financial flexibility") is unaffected -- this only matches the noun
    # "protection" bound to "downside", not "flexibility" or similar hedges.
    (re.compile(r"\bdownside\s+protection\b", re.IGNORECASE), "unqualified causal claim (downside protection)"),
    (re.compile(r"\beliminates?\s+(?:the\s+)?risk\b", re.IGNORECASE), "unqualified causal claim (eliminates risk)"),
    (re.compile(r"\bremoves?\s+(?:all\s+|the\s+)?risk\b", re.IGNORECASE), "unqualified causal claim (removes risk)"),
    (re.compile(r"\b(?:correction|rally|pullback|breakout|reversal)\s+is\s+"
               r"(?:due|coming|near|imminent|approaching)\b", re.IGNORECASE),
     "unqualified causal claim (X is due)"),
    (re.compile(r"\bsupport\s+is\s+proven\b", re.IGNORECASE), "unqualified causal claim (support is proven)"),
    (re.compile(r"\bdownside\s+is\s+confirmed\b", re.IGNORECASE), "unqualified causal claim (downside is confirmed)"),
    # H.4 corrective patch (goal 4): a technical reading describes HISTORY —
    # asserting it CORRELATES WITH or PREDICTS future performance/returns is
    # exactly the class of unsupported inference goal 4 names explicitly
    # ("historical price trends ... correlate with future performance").
    (re.compile(r"\bcorrelate[sd]?\s+with\s+future\b", re.IGNORECASE),
     "unqualified causal claim (correlates with future performance)"),
    (re.compile(r"\bpredicts?\s+future\s+(?:performance|returns?|price)\b", re.IGNORECASE),
     "unqualified causal claim (predicts future performance)"),
    # "achievable" applied to a DCF value/scenario asserts the outcome will
    # actually happen, without naming the assumptions required -- exactly
    # the framing Problem 7 (prior corrective patch) already told
    # bull_researcher not to use in its role-text prompt, now backstopped in
    # code (found unenforced live, COR corrective patch, Phase 3). "Achieve"/
    # "achieving" are NOT matched -- "the company achieved 10% growth" is a
    # plain past-tense factual report, not a certainty claim about the
    # future; only "achievable" (a claim ABOUT feasibility) is in scope.
    #
    # WM corrective patch -- SCOPED TO THAT STATED INTENT.
    #
    # The pattern was a bare `\bachievable\b`, which matched the word
    # ANYWHERE, not just applied to a modeled value. Measured live: WM's
    # bear_researcher tripped this in 4 of 5 consecutive runs, always on this
    # same word, and each trip cost a full extra model call to repair. The
    # rule's own comment says "applied to a DCF value/scenario" -- ordinary
    # analytical English about the COMPANY's own targets ("management's
    # margin target looks achievable given the announced cost actions") is
    # not the thing being guarded against, and never was.
    #
    # Hedging does not rescue it either: `_HEDGE_MARKERS` is only
    # whether/implies/imply/implied/implying, so "appears achievable" and
    # "achievable if margins hold" both still tripped.
    #
    # Now requires a VALUATION referent within the same sentence, on either
    # side. "The base modeled value is achievable" still flags; "the guidance
    # range looks achievable" does not.
    (re.compile(
        # NB "target" alone is deliberately NOT a referent -- a margin target
        # or a buyback target is the COMPANY's operating goal, not a
        # valuation claim. Only "price target" counts.
        r"(?:\b(?:value|valuation|price\s+target|price|per[-\s]share|modeled|scenario|dcf)\b"
        r"[^.;!?]{0,80}\bachievable\b)"
        r"|(?:\bachievable\b[^.;!?]{0,80}"
        r"\b(?:value|valuation|price\s+target|price|per[-\s]share|modeled|scenario|dcf)\b)",
        re.IGNORECASE),
     "unqualified causal claim (achievable)"),
]

_CONSENSUS_LANGUAGE_PATTERNS = [
    (re.compile(r"\bconsensus\s+(?:value|estimate|price|target)s?\b", re.IGNORECASE),
     "consensus-estimate language (no consensus data source exists)"),
    (re.compile(r"\banalyst(?:s)?\s+consensus\b", re.IGNORECASE),
     "consensus-estimate language (no consensus data source exists)"),
    (re.compile(r"\bmarket\s+consensus\b", re.IGNORECASE),
     "consensus-estimate language (no consensus data source exists)"),
    (re.compile(r"\bstreet\s+estimates?\b", re.IGNORECASE),
     "consensus-estimate language (no consensus data source exists)"),
]

# H.4 corrective patch: a DCF scenario output is a MODELED value, never an
# "intrinsic value" (implies one objective true value exists) or an
# "authoritative value" (implies one of base/bull/bear outranks the others --
# they are three modeled scenarios with no such relationship). "price
# target"/"target price" is banned separately in finance/content_policy.py;
# "consensus value/estimate" is the pre-existing ban directly above.
#  `[\s-]+` rather than `\s+` alone throughout -- a hyphenated
#  "intrinsic-value"/"authoritative-value"/"fair-value-target" is the same
#  claim and must not slip past a whitespace-only separator (same fix as
#  finance/content_policy.py's price-target pattern, H.4 corrective patch).
_DCF_TERMINOLOGY_PATTERNS = [
    (re.compile(r"\bintrinsic[\s-]+values?\b", re.IGNORECASE),
     "unsupported DCF terminology (intrinsic value -- use 'base/bull/bear modeled value')"),
    (re.compile(r"\bauthoritative[\s-]+(?:intrinsic[\s-]+)?values?\b", re.IGNORECASE),
     "unsupported DCF terminology (authoritative value -- no DCF scenario outranks another)"),
    (re.compile(r"\bfair[\s-]+value[\s-]+targets?\b", re.IGNORECASE),
     "unsupported DCF terminology (fair-value target -- use 'modeled value')"),
]

_FUTURE_TENSE_TECHNICAL_PATTERNS = [
    (re.compile(r"\bwill\s+(?:reverse|rebound|bounce|rally|correct|break\s?out)\b", re.IGNORECASE),
     "future-tense technical certainty"),
    (re.compile(r"\babout\s+to\s+(?:reverse|rebound|bounce|rally|correct|break\s?out)\b", re.IGNORECASE),
     "future-tense technical certainty"),
    (re.compile(r"\bis\s+set\s+to\s+(?:reverse|rebound|bounce|rally|correct|break\s?out)\b", re.IGNORECASE),
     "future-tense technical certainty"),
    (re.compile(r"\bpoised\s+(?:for|to)\b", re.IGNORECASE), "future-tense technical certainty (poised)"),
]

_ALL_TEXT_PATTERNS = (
    _SUPERLATIVE_PATTERNS + _CAUSAL_OVERREACH_PATTERNS
    + _CONSENSUS_LANGUAGE_PATTERNS + _FUTURE_TENSE_TECHNICAL_PATTERNS
    + _DCF_TERMINOLOGY_PATTERNS
)

# Phase H.5, Phase 2 -- stable rule ids.
#
# EVERY pattern in this module is OVERSTATEMENT by construction: each one
# judges how confidently a sentence asserts something, not whether the system
# had the information to say it at all. None of them describe a sentence with
# no true version. So all 34 are quarantine-class, and this module contributes
# nothing to the cascade-fatal set -- that lives entirely in
# finance/content_policy.py.
#
# Ids are assigned by GROUP so a group can grow without renumbering the
# others: CV-1xx superlatives, CV-2xx causal overreach, CV-3xx consensus,
# CV-4xx future-tense technical, CV-5xx DCF terminology, CV-6xx provenance.
_CLAIM_VALIDATION_RULE_GROUPS = (
    ("CV-1", _SUPERLATIVE_PATTERNS),
    ("CV-2", _CAUSAL_OVERREACH_PATTERNS),
    ("CV-3", _CONSENSUS_LANGUAGE_PATTERNS),
    ("CV-4", _FUTURE_TENSE_TECHNICAL_PATTERNS),
    ("CV-5", _DCF_TERMINOLOGY_PATTERNS),
)


def _build_claim_rules():
    rules = []
    for prefix, group in _CLAIM_VALIDATION_RULE_GROUPS:
        for index, (pattern, label) in enumerate(group, start=1):
            rules.append(Rule(rule_id=f"{prefix}{index:02d}", pattern=pattern,
                              label=label, severity=Severity.OVERSTATEMENT))
    return tuple(rules)


RULES = _build_claim_rules()
RULES_BY_ID = {rule.rule_id: rule for rule in RULES}
# Provenance mismatch is computed against the evidence index rather than by a
# fixed pattern, so it has no entry above; it gets a reserved id here so a
# quarantine record can still name it.
PROVENANCE_MISMATCH_RULE_ID = "CV-601"

# GE corrective patch: a string that explicitly says a claim is NOT
# available/supported/certain still contains the banned WORD ("no DCF ...
# available to model future cash flows or intrinsic value"; "there is no
# ... benchmark ... to prove the price is unfair" -- both found live). Scoped
# to the causal-overreach, consensus-language, and DCF-terminology label
# groups (by prefix) plus the 'guarantee' superlative specifically -- the
# label groups actually observed live tripping on disclaimed text. Every
# OTHER superlative ('fortress', 'industry-leading', ...) and the future-
# tense technical-certainty group stay maximally strict with no disclaimer
# exemption at all -- an adjectival overclaim doesn't have the same natural
# "X is unavailable" disclaiming shape a CLAIM does.
_DISCLAIMER_MARKERS = (
    "unavailable", "unsupported", "impossible", "not available", "not possible",
    "not supported", "not guaranteed", "no guarantee", "cannot be", "can not be",
    "can't be",
    # HOOD corrective patch: a SECOND live ticker's run showed the GE fix's
    # marker list was itself too narrow -- built from the exact phrasings ONE
    # ticker happened to produce, not the underlying disclaiming INTENT. Live
    # HOOD text disclaimed the same way with none of the words above: "do not
    # guarantee future performance" (a bare present-tense negated verb, not
    # "not guaranteed"), "the LACK OF analyst consensus data", "PREVENTS any
    # modeled assessment of intrinsic value", "INABILITY to quantify intrinsic
    # value". Added as general vocabulary, not per-ticker patches, since the
    # underlying pattern (this system explaining what it CANNOT compute) will
    # keep recurring in new phrasings no fixed phrase list fully anticipates
    # -- see _dcf_validation_failed-based exemption below for the more robust
    # fix on the one label group (DCF terminology) where a ground-truth
    # signal actually exists, instead of only ever chasing wording.
    "do not guarantee", "does not guarantee", "did not guarantee", "lack of", "lacks",
    "prevents", "preventing", "prevented", "inability", "unable to", "absence of",
)
# "No DCF ... values are available to model ... intrinsic value" (found live,
# rec2) and "there is no ... benchmark provided to prove ..." (found live,
# rec8) negate via a leading "no" quantifier rather than a "not"/"un-"
# prefix directly on the disclaiming word -- not caught by _DISCLAIMER_
# MARKERS above. Bounded proximity (80 chars) so "no" doesn't exempt an
# unrelated claim elsewhere in a longer field on pure coincidence.
_DISCLAIMER_PROXIMITY_PATTERNS = (
    re.compile(r"\bno\b[\s\S]{0,80}?\b(?:available|provided|exists?)\b", re.IGNORECASE),
)
_DCF_TERMINOLOGY_LABEL_PREFIX = "unsupported DCF terminology"
_FUTURE_TENSE_LABEL = "future-tense technical certainty"
# AMD corrective patch: a live AMD run wrote "Claims that the bull scenario
# value is 'achievable' or that the market price 'will reverse' based on
# technicals are NOT SUPPORTED BY THE EVIDENCE" -- the SAME critique-of-
# another-claim's-overreach pattern the UNH fix already covers (quoted claim,
# explicitly rejected), just tripping a label group ('future-tense technical
# certainty') that was never added to the exempt set. 'achievable' in that
# SAME sentence was already correctly exempted (it is 'unqualified causal
# claim'-prefixed, and the text contains 'not supported', an existing
# _DISCLAIMER_MARKERS entry) -- only the future-tense label was left
# unprotected. Added here rather than to _HEDGE_EXEMPT_LABEL_PREFIXES since
# the disclaiming word actually present ('not supported') is an absence-style
# marker, not a hedge-style one.
_DISCLAIMER_EXEMPT_LABELS = frozenset({"unsupported superlative (guarantee)"})
_DISCLAIMER_EXEMPT_LABEL_PREFIXES = (
    "unqualified causal claim", "consensus-estimate language", _DCF_TERMINOLOGY_LABEL_PREFIX,
    _FUTURE_TENSE_LABEL,
)

# UNH corrective patch: a THIRD live ticker's run showed a DIFFERENT category
# from the disclaiming-absence one above -- CONDITIONAL/hypothetical framing,
# not "this cannot be computed". A research_manager `key_disagreements` item
# genuinely doing its job ("Whether the stock's position above the 200-day
# SMA confirms a valid long-term uptrend or if the breakdown ... signals
# imminent further downside" -- presenting an unresolved disagreement, not
# asserting either side) tripped 'confirms'; an `assumption_sensitive_
# conclusions` item genuinely doing ITS job ("depends entirely on which
# scenario assumptions prove accurate") tripped 'proves'; and a critique of
# ANOTHER claim's overreach ("Assertions that free cash flow 'provides ample
# liquidity' ... imply a causal guarantee; while FCF is strong, the current
# ratio remains below 1.0") tripped 'guarantee' by NAMING the fallacy type it
# was rejecting, not committing it. All three use a banned word to describe
# an open question, a contingency, or someone ELSE's overreach -- not to
# assert the claim themselves.
#
# Deliberately a SEPARATE marker set/scope from _DISCLAIMER_MARKERS above,
# NOT folded in: these mark CONFIDENCE-level hedging (is the claim CERTAIN or
# merely suggested?), which is exactly what the guarantee/causal-overreach/
# DCF-terminology bans police -- but consensus-estimate language is a
# FORBIDDEN-TOPIC ban (this system has no analyst-consensus data source at
# all), not a confidence-level one. "Street estimates IMPLY higher margins"
# is exactly as invalid as "Street estimates CONFIRM higher margins" -- the
# problem is citing a data source that doesn't exist, and hedging the
# confidence of that citation doesn't fix it. Found live in this exact
# repo's own test suite (test_flags_consensus_language's "Street estimates
# imply..." case) the moment 'implies'/'imply' was first tried as a member
# of the single shared marker list -- corrected before merge, not caught in
# production, but kept as a cautionary comment since the mistake is an easy
# one to reintroduce.
_HEDGE_MARKERS = ("whether", "implies", "imply", "implied", "implying")
# "depends entirely on which scenario assumptions prove accurate" (found
# live) has an adverb ("entirely") between "depends" and "on", which a
# literal "depends on which" substring marker would miss -- a regex
# proximity check, same style as _DISCLAIMER_PROXIMITY_PATTERNS above,
# tolerates that and any other filler between the words.
_HEDGE_PROXIMITY_PATTERNS = (
    re.compile(r"\bdepends\b[\s\S]{0,30}?\bon\b[\s\S]{0,15}?\b(?:which|whether)\b", re.IGNORECASE),
)
# AMD corrective patch: "...provide financial flexibility that supports the
# higher growth and margin assumptions required for the bull scenario,
# RATHER THAN IMPLYING an inevitable reversion to the base case" (found
# live) -- REJECTING an over-deterministic 'inevitable' framing, not
# asserting it, using the gerund 'implying' (not yet in _HEDGE_MARKERS
# before this patch). 'inevitable' is added to the exempt set (unlike
# 'fortress'/'industry-leading', which stay maximally strict -- see the
# original scoping comment above _DISCLAIMER_MARKERS) because, like
# 'guarantee', it is fundamentally a CERTAINTY claim that gets meaningfully
# negated in ordinary hedged writing ("not inevitable", "rather than
# implying inevitable X") -- a purely adjectival overclaim like 'fortress'
# does not have that same natural negated shape.
_HEDGE_EXEMPT_LABELS = frozenset({
    "unsupported superlative (guarantee)", "unsupported superlative (inevitable)",
})
_HEDGE_EXEMPT_LABEL_PREFIXES = ("unqualified causal claim", _DCF_TERMINOLOGY_LABEL_PREFIX)


def _is_disclaimed(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in _DISCLAIMER_MARKERS):
        return True
    return any(pattern.search(text) for pattern in _DISCLAIMER_PROXIMITY_PATTERNS)


def _is_hedged(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in _HEDGE_MARKERS):
        return True
    return any(pattern.search(text) for pattern in _HEDGE_PROXIMITY_PATTERNS)


# Kept in lockstep with finance/content_policy.py's identical block.
_NEGATION_CUES = re.compile(
    r"\b(?:not|never|no|nothing|none|nor|neither|cannot|can't|cant|without|"
    r"excludes?|precludes?|rather\s+than|instead\s+of|absent)\b",
    re.IGNORECASE)

_NEGATION_LOOKBEHIND = 60
_NEGATION_LOOKAHEAD = 30


def _match_is_negated(text: str, match) -> bool:
    """Is THIS occurrence negated? Position-aware, unlike `_is_hedged`."""
    before = text[max(0, match.start() - _NEGATION_LOOKBEHIND):match.start()]
    after = text[match.end():match.end() + _NEGATION_LOOKAHEAD]
    return bool(_NEGATION_CUES.search(before) or _NEGATION_CUES.search(after))


# HOOD corrective patch: rather than keep enumerating every new way the model
# phrases "the DCF didn't run, so intrinsic value is unknown" (four MORE live
# HOOD variants slipped past the disclaimer-word list above, on top of the
# four GE ones the list already covers), use the one label group where a
# GROUND-TRUTH signal independent of the free text actually exists. When the
# DCF failed deterministic validation (finance/dcf.py; checked the same way
# finance/research_pipeline.py's own _dcf_validation_failed does, duplicated
# rather than imported for the SAME reason that module gives for not
# importing finance/dcf.py itself -- see this module's docstring), NO
# 'dcf.value_per_share.*'/'dcf.assumption.*' evidence exists in the index at
# all (withheld, not merely unreliable -- see the shared guardrails in
# finance/research_pipeline.py). That means a claim citing a REAL evidence ID
# (the separate, structural citation requirement every stage already
# enforces) categorically CANNOT be asserting a genuine DCF-based intrinsic/
# authoritative/fair-value figure in that case -- any 'intrinsic value' text
# MUST be describing its absence, regardless of which of the unbounded ways
# English has to say so. This is a strictly ADDITIVE exemption -- when the
# DCF is VALID, the DCF-terminology ban is unaffected and just as strict as
# before. See `scan_for_unsupported_claims`'s `dcf_invalid` parameter.

# Provenance fidelity: a claim naming one of these providers is only
# meaningful if that provider actually contributed something to THIS
# analysis (checked against the evidence index's provenance.*.provider
# values, not this module's own state -- see scan_for_provenance_mismatch).
_PROVIDER_NAME_PATTERNS = {
    "sec": re.compile(r"\b(?:the\s+)?SEC\b|SEC\s+filings?|SEC\s+EDGAR", re.IGNORECASE),
    "yahoo": re.compile(r"\bYahoo(?:\s+Finance)?\b", re.IGNORECASE),
    "alphavantage": re.compile(r"\bAlpha\s?Vantage\b", re.IGNORECASE),
}


def scan_for_unsupported_claims(text, dcf_invalid: bool = False) -> List[str]:
    """Every distinct unsupported-claim label found in one string, in
    pattern-list order. Empty means clean. A non-string input is always
    clean (nothing to scan) rather than raising.

    `dcf_invalid` (HOOD corrective patch): when the caller knows -- from the
    evidence index, not this text -- that the DCF failed deterministic
    validation, every DCF-terminology label is exempt unconditionally (see
    the long comment above `_DCF_TERMINOLOGY_LABEL_PREFIX`), on top of
    (never instead of) the ordinary disclaimer-word exemption below. Defaults
    to False so every existing caller that doesn't pass it is unaffected.
    """
    if not isinstance(text, str) or not text:
        return []
    # WM corrective patch -- GENERAL NEGATION PROXIMITY, kept in lockstep
    # with finance/content_policy.py's identical addition (see the long
    # comment above `_NEGATION_CUES` there for the live failure that
    # motivated it). Both modules scan the same model output for the same
    # class of overclaim, and both had the same defect: a whole-text
    # disclaimer vocabulary that cannot tell WHICH phrase a negation applies
    # to, so text DENYING a claim was flagged as making it.
    #
    # Strictly subtractive, and scoped to the same label groups the existing
    # hedge exemption already covers -- a negated occurrence of a certainty
    # claim is the model saying the claim is NOT being made.
    found = []
    for pattern, label in _ALL_TEXT_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        if (label in _HEDGE_EXEMPT_LABELS
                or label.startswith(_HEDGE_EXEMPT_LABEL_PREFIXES)) \
                and _match_is_negated(text, match):
            continue
        found.append(label)
    if found and dcf_invalid:
        found = [label for label in found if not label.startswith(_DCF_TERMINOLOGY_LABEL_PREFIX)]
    if found and _is_disclaimed(text):
        found = [label for label in found if label not in _DISCLAIMER_EXEMPT_LABELS
                and not label.startswith(_DISCLAIMER_EXEMPT_LABEL_PREFIXES)]
    # UNH corrective patch: a SEPARATE, narrower exemption from the one right
    # above -- deliberately excludes consensus-estimate language (a
    # forbidden-topic ban, not a confidence-level one) even though it shares
    # the mechanism. See the long comment above _HEDGE_MARKERS for why.
    if found and _is_hedged(text):
        found = [label for label in found if label not in _HEDGE_EXEMPT_LABELS
                and not label.startswith(_HEDGE_EXEMPT_LABEL_PREFIXES)]
    return found


def scan_for_provenance_mismatch(text, providers_present: Set[str]) -> List[str]:
    """Flags a claim that names a specific data provider not actually
    present anywhere in this analysis' evidence index -- e.g. text says "per
    SEC filings" in a reduced-mode report where every `provenance.*.provider`
    entry says "yahoo" and SEC data was omitted entirely. `providers_present`
    is the set of lower-cased provider names actually seen in the evidence
    index (see `providers_present_in_index`)."""
    if not isinstance(text, str) or not text:
        return []
    found = []
    for provider, pattern in _PROVIDER_NAME_PATTERNS.items():
        if provider not in providers_present and pattern.search(text):
            found.append(f"provenance mismatch (names {provider!r}, which supplied no data in this analysis)")
    return found


def providers_present_in_index(index) -> Set[str]:
    """The set of lower-cased provider names actually present in an evidence
    index's `provenance.<dataset>.provider` entries. `index` is the
    Dict[str, EvidenceItem] from finance/evidence.py (or anything with the
    same `.values()` / `.evidence_id` / `.value` shape)."""
    providers: Set[str] = set()
    for item in index.values():
        if item.evidence_id.startswith("provenance.") and item.evidence_id.endswith(".provider"):
            value = item.value
            if isinstance(value, str) and value.strip():
                providers.add(value.strip().lower())
    return providers


def _walk_strings(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _walk_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk_strings(v)


def find_unsupported_claims(text, field_path: str = "", dcf_invalid: bool = False,
                            providers_present: Optional[Set[str]] = None) -> List[Finding]:
    """Every match in ONE string, with its rule id and matched span.

    Same policy as `scan_for_unsupported_claims` -- same negation, hedge,
    disclaimer and `dcf_invalid` exemptions -- in the shape a quarantine
    record needs. Never a second, divergent policy.
    """
    if not isinstance(text, str) or not text:
        return []
    disclaimed = _is_disclaimed(text)
    hedged = _is_hedged(text)
    findings: List[Finding] = []
    for rule in RULES:
        match = rule.pattern.search(text)
        if match is None:
            continue
        if dcf_invalid and rule.label.startswith(_DCF_TERMINOLOGY_LABEL_PREFIX):
            continue
        exempt_by_negation = (
            (rule.label in _HEDGE_EXEMPT_LABELS
             or rule.label.startswith(_HEDGE_EXEMPT_LABEL_PREFIXES))
            and _match_is_negated(text, match))
        if exempt_by_negation:
            continue
        if disclaimed and (rule.label in _DISCLAIMER_EXEMPT_LABELS
                           or rule.label.startswith(_DISCLAIMER_EXEMPT_LABEL_PREFIXES)):
            continue
        if hedged and (rule.label in _HEDGE_EXEMPT_LABELS
                       or rule.label.startswith(_HEDGE_EXEMPT_LABEL_PREFIXES)):
            continue
        findings.append(Finding(
            rule_id=rule.rule_id, label=rule.label, severity=rule.severity,
            field_path=field_path, matched_span=match.group(0)))
    if providers_present is not None:
        for label in scan_for_provenance_mismatch(text, providers_present):
            findings.append(Finding(
                rule_id=PROVENANCE_MISMATCH_RULE_ID, label=label,
                severity=Severity.OVERSTATEMENT, field_path=field_path,
                matched_span=""))
    return findings


def find_unsupported_claims_in_structure(value, providers_present: Optional[Set[str]] = None,
                                         dcf_invalid: bool = False) -> List[Finding]:
    """Path-aware scan of a whole validated stage output."""
    findings: List[Finding] = []
    for field_path, text in _walk_fields(value):
        findings.extend(find_unsupported_claims(
            text, field_path, dcf_invalid=dcf_invalid, providers_present=providers_present))
    return findings


def scan_structure_for_unsupported_claims(value, providers_present: Optional[Set[str]] = None,
                                          dcf_invalid: bool = False) -> List[str]:
    """Recursively scans every string inside a dict/list/tuple/scalar
    structure (a validated stage output) for unsupported superlatives,
    causal overreach, consensus language, future-tense technical certainty,
    and (when `providers_present` is given) provenance mismatches.
    De-duplicated, order preserved. `dcf_invalid` is threaded straight
    through to `scan_for_unsupported_claims` -- see its docstring.
    """
    found: List[str] = []
    for text in _walk_strings(value):
        found.extend(scan_for_unsupported_claims(text, dcf_invalid=dcf_invalid))
        if providers_present is not None:
            found.extend(scan_for_provenance_mismatch(text, providers_present))
    seen = set()
    deduped = []
    for label in found:
        if label not in seen:
            seen.add(label)
            deduped.append(label)
    return deduped
