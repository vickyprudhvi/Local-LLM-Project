"""finance/content_policy.py — the deterministic backstop that rejects
statements this system has no basis to make.

Personal-use verdict patch: this module NO LONGER bans the bare verdict
words (buy/sell/hold/avoid). The tool now produces an explicit
buy/hold/sell/avoid `recommendation` as a validated enum field, so banning
the same words in prose enforced a rule the project deliberately reversed.
What remains is scoped to claims the system lacks the INFORMATION to
support: order mechanics (it does not know the reader's capital or risk
tolerance), holding-conditional phrasing (it cannot know whether the reader
holds a position), and certainty/overstatement claims.

See tests/test_finance_research_pipeline.py and
tests/test_finance_research_pipeline_integration.py for the wired-in,
end-to-end behavior (one repair attempt, fail-closed on a second violation);
this file checks the scanner itself in isolation.
"""

import pytest

from finance.content_policy import scan_for_prohibited_directives, scan_structure_for_prohibited_directives


# ---- the exact live-report regression (Problem 1's motivating example) ----

def test_flags_the_exact_live_cost_report_language():
    """The original Phase H.3 incident sentence must STILL be rejected -- but
    now for the reason that was always the real defect: it branches on
    whether the reader holds a position, which this system cannot know. The
    bare AVOID/SELL verdict words in it are no longer themselves violations
    (personal-use verdict patch); the holding-conditional framing around them
    still is."""
    text = "If you do not currently hold a position: AVOID. If you already hold a position: SELL."
    hits = scan_for_prohibited_directives(text)
    assert "holding-dependent phrasing" in hits
    assert "AVOID" not in hits and "SELL" not in hits and "HOLD" not in hits


# ---- verdict words are deliberately NOT prohibited any more ----

def test_does_not_flag_bare_verdict_words():
    """Personal-use verdict patch: the tool gives an explicit buy/hold/sell/
    avoid recommendation now, so these words carry no violation on their own.
    This is the change that eliminated a whole category of live false
    positives ("both sides hold that view", "avoid per-share metric errors",
    "share buy back") observed across GE/HOOD/UNH/AMD."""
    for text in (
        "This is a STRONG BUY.", "This is a strong sell.",
        "Investors should buy this stock.", "Investors should sell this stock.",
        "Overall: AVOID.", "Recommendation: hold",
        "Investors should HOLD given that markets are volatile.",
        "The company announced a share buy back program.",
        "Retail sell-through improved this quarter.",
        "The 2.8% discrepancy requires careful handling to avoid per-share metric errors.",
    ):
        assert scan_for_prohibited_directives(text) == [], text


def test_flags_position_sizing_and_order_language():
    assert scan_for_prohibited_directives("Use a 5% position size.") == ["position size"]
    assert scan_for_prohibited_directives("Set an entry price near $200.") == ["entry price"]
    assert scan_for_prohibited_directives("Set an exit price near $250.") == ["exit price"]
    assert scan_for_prohibited_directives("Use a stop-loss at $180.") == ["stop loss"]
    assert scan_for_prohibited_directives("Target allocation of 3%.") == ["target allocation"]
    assert scan_for_prohibited_directives("Follow these order instructions.") == ["order instructions"]
    assert scan_for_prohibited_directives("Follow these trading instructions.") == ["trading instructions"]


def test_flags_holding_dependent_phrasing():
    assert "holding-dependent phrasing" in scan_for_prohibited_directives(
        "If you already own shares, consider trimming.")
    assert "holding-dependent phrasing" in scan_for_prohibited_directives(
        "If you already hold this stock, wait.")


def test_holding_dependent_phrasing_catches_both_word_orders():
    """Personal-use verdict patch: the adverb may sit on EITHER side of the
    negation. The old fixed-order pattern only matched "if you currently do
    not hold"; "if you do NOT CURRENTLY hold" was caught only incidentally by
    the bare \\bhold\\b verdict pattern, which that same patch removed. Since
    "If you do not currently hold a position: AVOID." is the literal original
    incident sentence, losing this would have silently un-caught the exact
    output this guardrail exists for."""
    for text in (
        "If you do not currently hold this stock, wait.",
        "If you currently do not hold a position, wait.",
        "If you do not already own shares, this matters.",
        "If you don't own it yet, this matters.",
        "if you currently hold shares",
    ):
        assert "holding-dependent phrasing" in scan_for_prohibited_directives(text), text


def test_holding_dependent_pattern_does_not_swallow_the_logical_hold_that_sense():
    """The "hold true/up/across/steady/constant/that" exemption is carried
    over from the removed bare \\bhold\\b pattern -- "if you hold that
    assumption constant" is a modeling construction, not a question about the
    reader's portfolio."""
    for text in (
        "If you hold that assumption constant, the model still works.",
        "If you hold true to the base case, the value is $50.",
    ):
        assert scan_for_prohibited_directives(text) == [], text


def test_flags_price_target_language():
    assert scan_for_prohibited_directives("Our price target is $400.") == ["price target"]


def test_flags_target_price_reversed_word_order():
    """Found live (COR corrective patch): the reversed word order slipped
    through the original "price target"-only pattern entirely."""
    assert scan_for_prohibited_directives("Our target price is $400.") == ["price target"]


def test_does_not_flag_negated_target_price_either():
    assert scan_for_prohibited_directives("This is a modeled value, not a target price.") == []


def test_flags_hyphenated_price_target_too():
    """H.4 corrective patch: 'price-target' (hyphenated) is the same claim
    as 'price target' -- must not slip past a whitespace-only separator."""
    assert scan_for_prohibited_directives("Our price-target for the stock is $400.") == ["price target"]
    assert scan_for_prohibited_directives("This is a modeled value, not a price-target.") == []


def test_flags_guarantee_verb_forms():
    """Found live (COR corrective patch): bare 'guaranteed?' matched
    'guarantee'/'guaranteed' but not the plain verb form 'guarantees'."""
    for text in ("This guarantees upside from here.", "Returns are guaranteed.",
                "The company guarantees continued growth."):
        assert "guarantee" in scan_for_prohibited_directives(text)


def test_does_not_flag_guarantee_when_explicitly_negated():
    """GE corrective patch: a live GE run flagged 'which is not guaranteed
    given the 32.4% annualized volatility' -- the model correctly hedging
    (exactly the qualified language the repair prompt asks for), not making
    a guarantee claim. 'not guaranteed'/'no guarantee' must stay unflagged;
    an unqualified 'guarantees' elsewhere in the same text must still be
    caught (the exemption is per-claim in spirit even though the check
    itself is string-wide -- see finance/content_policy.py's own doc)."""
    for text in ("Returns are not guaranteed.",
                "which is not guaranteed given the 32.4% annualized volatility",
                "there is no guarantee that momentum will persist"):
        assert scan_for_prohibited_directives(text) == []


def test_does_not_flag_present_tense_guarantee_negation():
    """HOOD corrective patch: a live HOOD run disclaimed 'guarantee' with a
    bare present-tense negated verb -- '...though these do not guarantee
    future performance' -- which the GE fix's marker list missed (it only
    covered the past-participle 'not guaranteed'/'no guarantee' forms)."""
    assert scan_for_prohibited_directives(
        "Strong revenue growth and cash flow provide a counterweight, though these do "
        "not guarantee future performance.") == []
    assert "guarantee" in scan_for_prohibited_directives("This guarantees upside.")


def test_does_not_flag_a_critique_naming_another_claims_overreach():
    """UNH corrective patch: research_manager's own quality-control field
    (finance/claim_validation.py's `unsupported_points`) genuinely doing its
    job -- naming why a BULL claim overreaches -- got flagged for the word
    it used to name the fallacy: "Assertions that free cash flow 'provides
    ample liquidity' ... imply a causal guarantee; while FCF is strong, the
    current ratio remains below 1.0" critiques an overreach, it doesn't
    commit one. Kept in lockstep with the identical fix in finance/
    claim_validation.py -- see that module's longer comment for why this
    exemption must NOT extend to consensus-estimate language, which this
    module has no equivalent pattern for."""
    assert scan_for_prohibited_directives(
        "Assertions that free cash flow 'provides ample liquidity to manage' the current "
        "ratio imply a causal guarantee; while FCF is strong, the current ratio remains "
        "below 1.0 based on the snapshot data.") == []
    assert "guarantee" in scan_for_prohibited_directives("This guarantees upside.")


def test_does_not_flag_guarantee_with_the_implying_gerund():
    """AMD corrective patch: kept in lockstep with finance/claim_validation.
    py's identical broadening -- 'implying' (the gerund form) is now a
    marker alongside the existing 'implies'/'imply'/'implied'."""
    assert scan_for_prohibited_directives(
        "The data is not implying a guarantee of continued growth.") == []
    assert "guarantee" in scan_for_prohibited_directives("This guarantees upside.")


def test_flags_reader_directed_investment_imperatives():
    """Found live (COR corrective patch, Phase 3's 'invest now' example)."""
    for text in ("Investors should invest now while the price is low.",
                "You could invest today given the valuation gap.",
                "Investors should invest immediately."):
        assert "reader-directed investment imperative" in scan_for_prohibited_directives(text)


def test_does_not_flag_a_company_investing_in_itself():
    """'invest' describing the COMPANY's own capital allocation (capex, R&D)
    is common, neutral research language -- only reader-directed imperatives
    are in scope."""
    for text in ("Management continues to invest in automation and logistics.",
                "The company invested $500 million in new distribution centers.",
                "Capital investment remained elevated this year."):
        assert scan_for_prohibited_directives(text) == [], text


def test_does_not_flag_hold_that_the_logical_validity_sense():
    """Found live (COR corrective patch): a bull_researcher DCF-assumption
    claim -- 'for the bull scenario to be plausible, assumptions must hold
    that the company can scale its enterprise value...' -- was rejected on
    'hold that', an ordinary analytical-writing construction ('it must be
    true that') with no connection to a trade recommendation. Exempted
    alongside the pre-existing true/up/across/steady/constant follow-ons."""
    text = ("For the bull scenario value of ~$1,540 per share to be plausible, assumptions must "
           "hold that the company can exponentially scale its enterprise value to nearly $297 "
           "billion while increasing the terminal value share of enterprise value to over 82%.")
    assert scan_for_prohibited_directives(text) == []
    assert scan_for_prohibited_directives("This relationship is expected to hold that way going forward.") == []


# ---- explicitly allowed / negated phrasing must stay clean ----

def test_does_not_flag_negated_price_target_phrasing():
    """Problem 9's own encouraged phrasing -- rejecting it would be
    self-defeating."""
    for text in (
        "This is a modeled scenario, not a price target.",
        "The DCF output is not the target price.",
        "This figure is never a price target.",
    ):
        assert scan_for_prohibited_directives(text) == [], text


def test_does_not_flag_price_target_disclaimed_elsewhere_in_the_same_field():
    """GE corrective patch: the pre-existing negated-phrasing exemption above
    only covers a disclaimer sitting IMMEDIATELY before 'price target'
    ('not a price target'). A live GE run showed the model disclaiming
    AFTER the phrase instead -- 'future price target is impossible to
    derive', "'price target' ... is unsupported" -- which the prefix-only
    lookbehind does not catch. Both real captured sentences must stay
    unflagged; a bare, non-disclaimed price-target claim must still fire."""
    for text in (
        "Any conclusion regarding the company's intrinsic value or future price target is "
        "impossible to derive from this evidence set because no DCF scenarios were generated.",
        "Any claim regarding a specific 'price target', 'fair value', or 'intrinsic value' is "
        "unsupported as no DCF modeled values were generated.",
    ):
        assert scan_for_prohibited_directives(text) == [], text
    assert scan_for_prohibited_directives("Our price target is $400.") == ["price target"]


def test_does_not_flag_avoid_describing_a_data_or_methodology_pitfall():
    """GE corrective patch, now subsumed: 'avoid' carries no violation at all
    since the personal-use verdict patch removed the bare verdict words, so
    that patch's narrow negative-lookahead (for benign continuations like
    "avoid per-share metric errors") was deleted along with the pattern it
    guarded. These real captured sentences must still come back clean --
    kept as a regression guard in case a future patch reintroduces an
    'avoid' pattern without the exemption."""
    for text in (
        "The 2.8% discrepancy requires careful handling to avoid per-share metric errors.",
        "Use diluted shares consistently to avoid double-counting.",
        "State the basis explicitly to avoid ambiguity in the comparison.",
    ):
        assert scan_for_prohibited_directives(text) == [], text


def test_does_not_flag_share_buybacks_or_sell_through():
    assert scan_for_prohibited_directives("The company announced a share buy back program.") == []
    assert scan_for_prohibited_directives("Retail sell-through improved this quarter.") == []


def test_does_not_flag_hedge_phrasing_of_hold():
    for text in (
        "Rates are expected to hold steady this quarter.",
        "The trend continues to hold true.",
        "Margins held up across the segment.",
    ):
        assert scan_for_prohibited_directives(text) == [], text


def test_clean_research_language_produces_no_hits():
    text = ("Revenue grew year over year and the market price appears above the base "
           "modeled value. Overall risk is assessed as moderate.")
    assert scan_for_prohibited_directives(text) == []


# ---- non-string input ----

def test_non_string_input_is_clean_not_an_error():
    assert scan_for_prohibited_directives(None) == []
    assert scan_for_prohibited_directives(123) == []
    assert scan_for_prohibited_directives("") == []


# ---- scan_structure_for_prohibited_directives: recursive, deduped ----

def test_structure_scan_walks_nested_dicts_and_lists():
    structure = {
        "research_stance": "cautious",
        "rationale": [{"statement": "Use a 5% position size.", "evidence_ids": ["x"]}],
        "key_uncertainties": ["Nothing else to flag here."],
    }
    hits = scan_structure_for_prohibited_directives(structure)
    assert hits == ["position size"]


def test_structure_scan_deduplicates_repeated_violations():
    structure = {"a": "Use a 5% position size.", "b": ["Pick a position size first."]}
    hits = scan_structure_for_prohibited_directives(structure)
    assert hits.count("position size") == 1


def test_structure_scan_of_a_fully_clean_structure_is_empty():
    structure = {
        "research_stance": "neutral",
        "valuation_view": "approximately_fair",
        "overall_risk": "moderate",
        "confidence": 0.5,
        "rationale": [{"statement": "The market price is close to the base modeled value.",
                      "evidence_ids": ["valuation_gap.direction"]}],
        "conditions_that_strengthen_the_view": [],
        "conditions_that_weaken_the_view": [],
        "key_uncertainties": [],
    }
    assert scan_structure_for_prohibited_directives(structure) == []


# ---------------------------------------------------------------------------
# WM corrective patch: position-aware negation
# ---------------------------------------------------------------------------
#
# The live failure: research_manager failed with "used prohibited trade-advice
# language after one repair attempt". The trip was the model DISCLAIMING the
# banned claim -- the exact qualified language this module asks for elsewhere.
# The repair could not fix it, because rewriting the sentence still says the
# same true thing and trips again.

@pytest.mark.parametrize("text", [
    "These modeled values should not be interpreted as a price target.",
    "The bear case guarantees nothing about future returns.",
    "This is research characterization, never a price target.",
    "The modeled range is presented rather than a price target.",
    "Nothing here constitutes a guarantee.",
    "No guarantee is provided about future performance.",
    "Continued expansion is not guaranteed.",
])
def test_a_negated_certainty_claim_is_not_a_violation(text):
    """Text DENYING the claim is the opposite of text making it."""
    assert scan_for_prohibited_directives(text) == [], text


@pytest.mark.parametrize("text", [
    "The bull case supports a price target of $250 per share.",
    "We guarantee 15% returns.",
])
def test_an_unnegated_certainty_claim_is_still_a_violation(text):
    assert scan_for_prohibited_directives(text), text


@pytest.mark.parametrize("text", [
    "Use a position size of 5% of your portfolio.",
    "Set a stop loss at $180.",
    "Your entry price should be $200.",
    "Do not use a position size above 5%.",
    "There is no stop loss to recommend here.",
])
def test_order_mechanics_are_never_exempted_by_negation(text):
    """Order mechanics and holding-conditional phrasing stay maximally
    strict. They are banned because this system lacks the INFORMATION to say
    them (the reader's capital, risk tolerance, whether they hold a
    position), not because they overclaim certainty -- so a negation does not
    make them sayable."""
    assert scan_for_prohibited_directives(text), text
