"""Phase H.3 corrective patch (Problem 5) — finance/claim_validation.py: the
deterministic backstop that rejects language OVERSTATING what a cited
evidence ID actually shows (superlatives, causal overreach, consensus
language, future-tense technical certainty, provenance mismatch).

See tests/test_finance_research_pipeline.py for the wired-in, per-stage
behavior; this file checks the scanners themselves in isolation.
"""

import pytest

from finance.claim_validation import (
    providers_present_in_index,
    scan_for_provenance_mismatch,
    scan_for_unsupported_claims,
    scan_structure_for_unsupported_claims,
)


class _FakeItem:
    def __init__(self, evidence_id, value):
        self.evidence_id = evidence_id
        self.value = value


# ---- the exact live-report regressions (Problem 5's motivating examples) ----

def test_flags_fortress_balance_sheet():
    hits = scan_for_unsupported_claims("This company has a fortress balance sheet.")
    assert any("fortress" in h for h in hits)


def test_flags_industry_leading_superlative():
    hits = scan_for_unsupported_claims("Industry-leading ROE of 22% supports the bull case.")
    assert any("industry-leading" in h for h in hits)


def test_does_not_flag_bare_confirms_or_proves():
    """Severity-scoping patch: the bare 'confirms'/'proves' patterns were
    removed. They OVERSTATE a real, cited fact rather than fabricating one --
    the number behind the claim is genuine, in the evidence index, and
    visible to the reader, and the separate evidence-ID citation requirement
    (which never false-positives) remains the real backstop. They were also
    this scanner's highest-volume trip across every live GE/HOOD/UNH/AMD
    round, and the cost of blocking was a FAILED stage cascading to a report
    with no Risk section, no Research View and no recommendation -- a bad
    trade for a punchy verb on a cited number.

    Contrast test_flags_named_superlatives / test_flags_consensus_language /
    the DCF-terminology tests below: those FABRICATE data that does not
    exist anywhere, leaving the reader nothing to discount, and all stay."""
    for text in (
        "The positive MACD confirms a trend reversal.",
        "Technical breakdown below the 200-day moving average confirms a longer-term reversal.",
        "Technical indicators confirm negative momentum.",
        "This proves the thesis is correct.",
        "Both sides confirm the stock is trading above its moving averages.",
    ):
        assert scan_for_unsupported_claims(text) == [], text


def test_still_flags_the_narrow_compound_proven_and_confirmed_forms():
    """The narrow compound forms are deliberately KEPT -- unlike the bare
    verbs they have no innocent reading, and were never a meaningful
    false-positive source."""
    assert scan_for_unsupported_claims("Support is proven at the 50-day average.")
    assert scan_for_unsupported_claims(
        "Downside is confirmed by the breakdown below the 200-day average.")


# ---- superlative control ----

def test_flags_named_superlatives():
    for text, fragment in (
        ("This is a best-in-class operator.", "best-in-class"),
        ("A truly world-class management team.", "world-class"),
        ("This is the safest name in the sector.", "safest"),
        ("Continued growth is guaranteed.", "guarantee"),
        ("The stock is certain to outperform.", "certain to"),
        ("Further declines are inevitable.", "inevitable"),
        ("Scale advantages here are unrivaled.", "unrivaled"),
        ("Its distribution network is unmatched.", "unmatched"),
        ("The balance sheet looks bulletproof.", "bulletproof"),
    ):
        hits = scan_for_unsupported_claims(text)
        assert hits, f"expected a hit for: {text!r}"
        assert any(fragment in h for h in hits), (text, hits)


def test_does_not_flag_guarantee_when_explicitly_negated():
    """GE corrective patch: a live GE run flagged 'guarantee' inside
    'which is not guaranteed given the 32.4% annualized volatility' -- the
    model correctly hedging (the qualified language the repair prompt asks
    for), not asserting a guarantee. finance/content_policy.py's identical
    'guarantee' pattern gets the identical fix; a bare, unqualified
    'guarantees' must still be caught."""
    for text in ("Returns are not guaranteed.",
                "which is not guaranteed given the 32.4% annualized volatility",
                "there is no guarantee that momentum will persist"):
        assert scan_for_unsupported_claims(text) == [], text
    hits = scan_for_unsupported_claims("Continued growth is guaranteed.")
    assert any("guarantee" in h for h in hits)


def test_does_not_flag_bare_best_or_leading():
    """'best' and 'leading' alone are common, neutral epistemic/technical
    terms ('best available data', 'leading indicator') -- only the compound
    promotional forms are flagged. Deliberate false-positive avoidance,
    matching the narrow 'hold' pattern in finance/content_policy.py."""
    for text in (
        "Using the best available data, revenue grew year over year.",
        "This is a leading indicator, not a lagging one.",
        "We made our best effort to reconcile the balance sheet.",
    ):
        assert scan_for_unsupported_claims(text) == [], text


def test_does_not_flag_exceptional_at_all():
    """Removed from auto-rejection after a live incident: a real
    bull_researcher run on COR wrote 'exceptional return on equity' as part
    of a thesis-level summary sentence, with the actual grounding number (a
    genuinely extreme 144% ROE) cited two sentences later in a separate
    key_point -- a reasonable way to write a summary thesis, but it caused a
    TOTAL pipeline failure with no repair path. 'exceptional' is a common,
    often-reasonable intensifier (unlike 'fortress'/'guaranteed'), so it is
    excluded here exactly like 'best'/'leading' above -- see the long
    comment in finance/claim_validation.py for the full incident writeup."""
    for text in (
        "An exceptional quarter for margins.",
        "This is an exceptional company with a bright future.",
        "Cencora delivers an extraordinary return on average equity of 144%.",
        "The company shows an exceptional return on equity, supporting the bull case.",
    ):
        assert scan_for_unsupported_claims(text) == [], text


# ---- causal-overreach control ----

def test_flags_causal_overreach_phrases():
    """Note: bare "proves"/"confirms" are deliberately absent from this list
    -- see test_does_not_flag_bare_confirms_or_proves. The narrow compound
    forms ("support is proven", "downside is confirmed") remain."""
    for text in (
        "Net cash on the balance sheet protects the company from downside.",
        "A correction is due given the extended run-up.",
        "Support is proven at the 50-day average.",
        "Downside is confirmed by the breakdown below the 200-day average.",
        "This eliminates the risk of a margin miss.",
        "This removes all risk from the investment case.",
    ):
        assert scan_for_unsupported_claims(text), f"expected a hit for: {text!r}"


def test_does_not_flag_proves_when_the_sentence_says_nothing_proves_it():
    """GE corrective patch, now subsumed by the severity-scoping patch (bare
    'proves' carries no violation at all any more). Kept as a regression
    guard on the real captured sentence in case a future patch reintroduces
    a 'proves' pattern without the disclaimer exemption."""
    text = ("Bear claim that current price is unjustified: Without DCF modeled values or "
           "analyst consensus targets in the evidence index, there is no quantitative "
           "benchmark provided to prove the price is unfair relative to cash flow potential.")
    assert scan_for_unsupported_claims(text) == []


def test_flags_downside_protection_noun_phrase():
    """Found live (COR corrective patch, Phase 2's explicit example): the
    NOUN-PHRASE form ('offers/provides significant downside protection')
    slipped through the VERB-form pattern ('protects ... from downside')
    entirely -- distinct word shape, distinct regex needed."""
    for text in ("Net cash offers significant downside protection.",
                "The balance sheet provides meaningful downside protection in a slowdown."):
        assert scan_for_unsupported_claims(text), f"expected a hit for: {text!r}"


def test_flags_compelling_investment_or_short_case():
    """Found live (COR corrective patch): Problem 7's (prior patch) named
    promotional phrases had prompt-level guidance but no code backstop."""
    assert scan_for_unsupported_claims("This represents a compelling investment case.")
    assert scan_for_unsupported_claims("This is a compelling short case given the leverage.")


def test_does_not_flag_bare_compelling():
    """'compelling evidence' is common, neutral analytical-writing language
    -- only the investment/short case-framing bigram is in scope."""
    assert scan_for_unsupported_claims("The data provides compelling evidence of margin stability.") == []


def test_flags_achievable():
    """Found live (COR corrective patch, Phase 3): asserts an outcome will
    happen without naming the required assumptions."""
    assert scan_for_unsupported_claims("The bull scenario value is achievable given current trends.")


def test_does_not_flag_achieved_past_tense():
    assert scan_for_unsupported_claims("The company achieved 10% revenue growth last year.") == []


def test_allows_qualified_causal_language():
    """The user's own example of acceptable, hedged phrasing."""
    assert scan_for_unsupported_claims("Net cash may provide financial flexibility.") == []
    assert scan_for_unsupported_claims(
        "A pullback toward the base modeled value is possible if growth slows.") == []


# ---- consensus-language ban ----

def test_flags_consensus_language():
    for text in (
        "The consensus value points to further upside.",
        "Analyst consensus suggests this is undervalued.",
        "Market consensus has shifted more cautious.",
        "Street estimates imply higher margins next year.",
    ):
        hits = scan_for_unsupported_claims(text)
        assert hits and "consensus" in hits[0].lower(), (text, hits)


def test_does_not_flag_consensus_language_when_explaining_its_absence():
    """GE corrective patch: this system has no analyst-consensus data
    source, and the prompts explicitly tell every stage so -- a stage
    correctly explaining that NO consensus figure is available must not be
    punished for the word 'consensus' itself. A genuine consensus-value
    claim must still be caught."""
    for text in (
        "Without analyst consensus targets in the evidence index, no such figure is available.",
        "There is no market consensus data available in this evidence set.",
    ):
        assert scan_for_unsupported_claims(text) == [], text
    assert scan_for_unsupported_claims("The consensus value points to further upside.")


# ---- H.4 corrective patch: DCF terminology (goal 2) ----
# A finance.dcf_model scenario output is a MODELED value, never an
# "intrinsic value" (implies one objective true value exists) or an
# "authoritative value" (implies base/bull/bear outrank each other).

def test_intrinsic_value_is_rejected_for_a_dcf_scenario_output():
    """Test 7."""
    hits = scan_for_unsupported_claims("The stock trades 34% above its intrinsic value.")
    assert any("intrinsic value" in h.lower() for h in hits)


def test_authoritative_intrinsic_value_is_rejected():
    hits = scan_for_unsupported_claims("Base intrinsic value: $45.38, the authoritative intrinsic value.")
    assert any("intrinsic value" in h.lower() for h in hits)
    assert any("authoritative" in h.lower() for h in hits)


def test_authoritative_value_alone_is_rejected():
    hits = scan_for_unsupported_claims("The base scenario is the authoritative value for this company.")
    assert any("authoritative value" in h.lower() for h in hits)


def test_fair_value_target_is_rejected():
    hits = scan_for_unsupported_claims("Our fair-value target for the stock is $45.")
    assert any("fair-value target" in h.lower() for h in hits)


def test_hyphenated_intrinsic_value_is_also_rejected():
    """A hyphenated 'intrinsic-value' is the same claim as 'intrinsic value'
    with a space -- must not slip past a whitespace-only separator."""
    hits = scan_for_unsupported_claims("This is the intrinsic-value of the stock.")
    assert any("intrinsic value" in h.lower() for h in hits)


def test_does_not_flag_dcf_terminology_when_explaining_its_unavailability():
    """GE corrective patch: GE's own DCF failed deterministic validation, so
    every stage was explicitly prompted to explain that valuation evidence
    is unavailable -- and 4 of 4 live GE research_manager runs then tripped
    this exact scanner while doing so correctly ("No DCF ... values are
    available to model future cash flows or intrinsic value"; "Any
    conclusion regarding intrinsic value or fair value is unavailable
    because..."; "...intrinsic value or future price target is impossible
    to derive..."). All three real captured sentences must stay unflagged;
    a bare, non-disclaimed 'intrinsic value' claim must still be caught."""
    for text in (
        "No DCF base, bull, or bear scenario values are available to model future cash "
        "flows or intrinsic value.",
        "Any conclusion regarding intrinsic value or fair value is unavailable because the "
        "DCF model was not run due to insufficient data for assumption generation.",
        "Any conclusion regarding the company's intrinsic value or future price target is "
        "impossible to derive from this evidence set because no DCF scenarios were generated.",
    ):
        assert scan_for_unsupported_claims(text) == [], text
    hits = scan_for_unsupported_claims("The stock trades 34% above its intrinsic value.")
    assert any("intrinsic value" in h.lower() for h in hits)


def test_does_not_flag_dcf_terminology_when_dcf_invalid_regardless_of_wording():
    """HOOD corrective patch: a SECOND live ticker (HOOD, also a failed-DCF
    case) tripped this SAME scanner with FOUR MORE real phrasings the GE
    fix's disclaimer-word list didn't cover -- "leaving no modeled intrinsic
    value scenarios", "prevents any modeled assessment of intrinsic value",
    "prevents quantitative assessment of intrinsic value", "Inability to
    quantify intrinsic value". Rather than add a fifth, sixth, ... marker
    per new phrasing, `dcf_invalid=True` (the caller's ground truth from the
    evidence index, not this text) exempts the whole DCF-terminology label
    group unconditionally -- when the DCF failed validation, NO
    'dcf.value_per_share.*' evidence exists to cite, so a claim citing real
    evidence categorically cannot be a genuine DCF-based intrinsic-value
    assertion, regardless of which of English's many ways of saying so was
    used. A bare, non-disclaimed claim must still be caught when dcf_invalid
    is NOT set (the DCF-valid case, where a genuine claim IS possible)."""
    for text in (
        "DCF valuation model was not executed because the reported history does not "
        "support proposing assumptions, leaving no modeled intrinsic value scenarios.",
        "The absence of a DCF model output prevents any modeled assessment of intrinsic "
        "value relative to the current price.",
        "Absence of a DCF model prevents quantitative assessment of intrinsic value "
        "relative to the current price.",
        "Inability to quantify intrinsic value or valuation gaps without user-supplied "
        "DCF assumptions.",
    ):
        assert scan_for_unsupported_claims(text, dcf_invalid=True) == [], text
    hits = scan_for_unsupported_claims("The stock trades 34% above its intrinsic value.",
                                       dcf_invalid=False)
    assert any("intrinsic value" in h.lower() for h in hits)
    # dcf_invalid=True never suppresses a DIFFERENT label -- only the DCF-
    # terminology group is exempt, everything else stays exactly as strict.
    hits = scan_for_unsupported_claims("This is a fortress balance sheet.", dcf_invalid=True)
    assert any("fortress" in h for h in hits)


def test_does_not_flag_lack_of_consensus_or_present_tense_guarantee_negation():
    """HOOD corrective patch: two more disclaiming phrasings the GE fix's
    marker list didn't anticipate -- "the LACK OF analyst consensus data"
    (GE's fix only covered 'no'-quantifier constructions, not 'lack of') and
    "these do NOT GUARANTEE future performance" (GE's fix only covered the
    past-participle 'not guaranteed', not the bare present-tense verb). Both
    are general vocabulary additions, not tied to any evidence-index signal
    (unlike the DCF-terminology fix above), since neither label has an
    equivalent ground-truth check available."""
    assert scan_for_unsupported_claims(
        "The lack of analyst consensus data in the provided evidence to contextualize "
        "current market expectations.") == []
    assert scan_for_unsupported_claims(
        "Strong revenue growth and cash flow provide a counterweight, though these do "
        "not guarantee future performance.") == []
    assert scan_for_unsupported_claims("The consensus value points to further upside.")
    hits = scan_for_unsupported_claims("Continued growth is guaranteed.")
    assert any("guarantee" in h for h in hits)


def test_does_not_flag_open_questions_or_contingent_conclusions():
    """UNH corrective patch: CONDITIONAL/hypothetical framing -- a
    `key_disagreements` item presenting an unresolved disagreement, or an
    `assumption_sensitive_conclusions` item naming a contingency, rather
    than asserting either side.

    The two real captured sentences now come back clean for a simpler
    reason as well (the severity-scoping patch removed bare
    'confirms'/'proves' outright), so they are kept here as regression
    guards. The hedge mechanism ITSELF is still live and still load-bearing
    for the label groups that remain -- asserted below against 'guarantee'
    and the DCF-terminology group, which the hedge exemption still covers."""
    assert scan_for_unsupported_claims(
        "Whether the stock's position above the 200-day SMA confirms a valid long-term "
        "uptrend or if the breakdown below the 20/50-day SMAs and negative MACD signals "
        "imminent further downside.") == []
    assert scan_for_unsupported_claims(
        "Conclusion that price offers margin of safety or risk depends entirely on which "
        "scenario assumptions prove accurate.") == []
    # The hedge mechanism still applies to the label groups that remain.
    assert scan_for_unsupported_claims(
        "Whether the balance sheet implies a guarantee of continued growth is unresolved.") == []
    assert scan_for_unsupported_claims("Continued growth is guaranteed.")


def test_does_not_flag_a_critique_naming_another_claims_overreach():
    """UNH corrective patch: research_manager's `unsupported_points` field
    genuinely doing its job -- naming why a BULL claim overreaches -- got
    flagged for the word it used to name the fallacy: "Assertions that free
    cash flow 'provides ample liquidity' ... imply a causal guarantee; while
    FCF is strong, the current ratio remains below 1.0" critiques an
    overreach, it doesn't commit one. 'implies'/'imply' is treated as no
    stronger than 'is consistent with' (already explicitly acceptable
    qualified language) -- a bare, unqualified 'guarantees' must still be
    caught."""
    assert scan_for_unsupported_claims(
        "Assertions that free cash flow 'provides ample liquidity to manage' the current "
        "ratio imply a causal guarantee; while FCF is strong, the current ratio remains "
        "below 1.0 based on the snapshot data.") == []
    hits = scan_for_unsupported_claims("Continued growth is guaranteed.")
    assert any("guarantee" in h for h in hits)


def test_does_not_flag_inevitable_when_explicitly_rejected():
    """AMD corrective patch: a live AMD run rejected an over-deterministic
    framing -- "...provide financial flexibility that supports the higher
    growth and margin assumptions required for the bull scenario, RATHER
    THAN IMPLYING an inevitable reversion to the base case" -- using the
    gerund 'implying' (not the base verb form the UNH fix already covered).
    'inevitable' is treated like 'guarantee': a certainty claim that gets
    meaningfully negated in ordinary hedged writing, unlike a purely
    adjectival superlative such as 'fortress'. A bare, unqualified
    'inevitable' must still be caught."""
    assert scan_for_unsupported_claims(
        "The robust free cash flow margin and net cash position provide financial "
        "flexibility that supports the higher growth and margin assumptions required for "
        "the bull scenario, rather than implying an inevitable reversion to the base case.") == []
    hits = scan_for_unsupported_claims("Further declines are inevitable.")
    assert any("inevitable" in h for h in hits)


def test_does_not_flag_future_tense_certainty_when_explicitly_rejected():
    """AMD corrective patch: the SAME critique-of-another-claim's-overreach
    pattern the UNH fix covers ("Assertions that ... imply a causal
    guarantee"), this time tripping the future-tense-technical-certainty
    label specifically, which had not been added to the exempt set --
    "Claims that the bull scenario value is 'achievable' or that the market
    price 'will reverse' based on technicals are NOT SUPPORTED BY THE
    EVIDENCE" (found live) quotes and rejects both an 'achievable' overreach
    (already correctly exempted before this patch) and a 'will reverse'
    overreach (not yet exempted). A bare, unqualified 'will reverse' must
    still be caught."""
    assert scan_for_unsupported_claims(
        "Claims that the bull scenario value is 'achievable' or that the market price "
        "'will reverse' based on technicals are not supported by the evidence, which only "
        "provides current modeled values and static technical readings.") == []
    hits = scan_for_unsupported_claims("The price will reverse soon.")
    assert any("future-tense" in h for h in hits)


def test_hedge_markers_do_not_exempt_consensus_language():
    """UNH corrective patch: 'whether'/'implies' mark CONFIDENCE-level
    hedging (is the claim certain, or merely suggested?) -- appropriate for
    the guarantee/causal-overreach/DCF-terminology bans, which ARE about
    confidence. consensus-estimate language is a FORBIDDEN-TOPIC ban (this
    system has no analyst-consensus data source at all, full stop) -- "Street
    estimates IMPLY higher margins" is exactly as invalid as "Street
    estimates CONFIRM higher margins", so the hedge-marker exemption must
    NOT apply to it (unlike the disclaiming-absence exemption in the test
    above, which correctly DOES apply -- "the lack of consensus data" is a
    genuinely different claim: that no such data exists at all)."""
    hits = scan_for_unsupported_claims("Street estimates imply higher margins next year.")
    assert hits and "consensus" in hits[0].lower()
    hits = scan_for_unsupported_claims(
        "Whether analyst consensus implies further upside is unclear from this evidence.")
    assert hits and "consensus" in hits[0].lower()


def test_price_target_is_rejected_via_the_content_policy_scanner():
    """Test 8 -- 'price target' is banned in finance/content_policy.py (a
    separate, pre-existing scanner from this module); both are combined by
    finance/research_pipeline.py::_validate_claim_fidelity for every stage,
    so this documents the split responsibility rather than duplicating it."""
    from finance.content_policy import scan_for_prohibited_directives
    hits = scan_for_prohibited_directives("Our price target for the stock is $45.")
    assert hits == ["price target"]


def test_base_modeled_value_is_accepted():
    """Test 9."""
    assert scan_for_unsupported_claims("Base modeled value: $45.38.") == []


def test_bull_and_bear_modeled_value_are_accepted():
    assert scan_for_unsupported_claims("Bull modeled value: $52.10. Bear modeled value: $38.90.") == []
    assert scan_for_unsupported_claims("This is the modeled value per share under the base scenario.") == []
    assert scan_for_unsupported_claims("The valuation model output reflects the listed assumptions.") == []


def test_market_comparison_wording_is_accepted():
    """Test 10."""
    text = ("The delayed market price is approximately 34% above the base modeled value "
           "under the current assumptions.")
    assert scan_for_unsupported_claims(text) == []
    from finance.content_policy import scan_for_prohibited_directives
    assert scan_for_prohibited_directives(text) == []


# ---- claim-type fidelity: technical claims describe history, not the future ----

def test_flags_future_tense_technical_certainty():
    for text in (
        "The stock will reverse once earnings are reported.",
        "Price is about to break out above resistance.",
        "The chart is set to rally into next quarter.",
        "Momentum is poised for a rebound.",
    ):
        assert scan_for_unsupported_claims(text), f"expected a hit for: {text!r}"


def test_allows_descriptive_historical_technical_language():
    """The user's own examples of acceptable technical-indicator phrasing."""
    for text in (
        "The MACD histogram is positive, indicating improving recent momentum.",
        "Price is above the 20-day average but below the 50- and 200-day averages.",
        "These indicators are descriptive and not independently predictive.",
    ):
        assert scan_for_unsupported_claims(text) == [], text


def test_price_above_sma_fact_is_accepted():
    """Test 19: a plain descriptive reading of a technical indicator is not
    an inference and must never be flagged."""
    for text in ("Price is above the 200-day SMA.", "Price is below the 20-day and 50-day SMAs.",
                "MACD histogram is negative.", "RSI is 40.99."):
        assert scan_for_unsupported_claims(text) == [], text


def test_unsupported_future_trend_inference_is_rejected():
    """Test 20: goal 4's disallowed automatic inferences from a technical
    reading -- future trend, reversal, breakout -- without an explicit
    request for that interpretation."""
    for text in (
        "RSI at 40.99 suggests the stock will rebound soon.",
        "The MACD histogram being negative means a breakout is coming.",
        "Historical price trends relative to the 200-day average correlate with future performance.",
    ):
        assert scan_for_unsupported_claims(text), f"expected a hit for: {text!r}"


# ---- provenance fidelity ----

def test_providers_present_in_index_reads_provenance_provider_entries():
    index = {
        "provenance.income_statement.provider": _FakeItem("provenance.income_statement.provider", "yahoo"),
        "provenance.balance_sheet.provider": _FakeItem("provenance.balance_sheet.provider", "sec"),
        "quote.price": _FakeItem("quote.price", 123.0),  # not a provenance entry -- ignored
    }
    assert providers_present_in_index(index) == {"yahoo", "sec"}


def test_flags_a_named_provider_that_supplied_nothing():
    providers_present = {"yahoo"}
    hits = scan_for_provenance_mismatch("Per SEC filings, revenue grew steadily.", providers_present)
    assert hits and "sec" in hits[0].lower()


def test_does_not_flag_a_provider_that_actually_contributed_data():
    providers_present = {"yahoo", "sec"}
    assert scan_for_provenance_mismatch("Per SEC filings, revenue grew steadily.", providers_present) == []
    assert scan_for_provenance_mismatch("Yahoo Finance shows a delayed quote.", providers_present) == []


# ---- structure walk ----

def test_structure_scan_walks_nested_dicts_and_lists():
    structure = {
        "thesis": "This is a fortress balance sheet.",
        "key_points": [{"point": "Industry-leading ROE.", "evidence_cited": ["x"]}],
    }
    hits = scan_structure_for_unsupported_claims(structure)
    assert len(hits) == 2


def test_structure_scan_deduplicates():
    structure = {"a": "This has a fortress balance sheet.", "b": ["Truly a fortress of a company."]}
    hits = scan_structure_for_unsupported_claims(structure)
    assert len([h for h in hits if "fortress" in h]) == 1


def test_structure_scan_with_providers_present_catches_provenance_mismatch():
    structure = {"balanced_assessment": "Per SEC filings, the company is well capitalized."}
    hits = scan_structure_for_unsupported_claims(structure, providers_present={"yahoo"})
    assert any("sec" in h.lower() for h in hits)


def test_structure_scan_without_providers_present_skips_provenance_check():
    structure = {"balanced_assessment": "Per SEC filings, the company is well capitalized."}
    hits = scan_structure_for_unsupported_claims(structure)
    assert hits == []


def test_structure_scan_of_a_fully_clean_structure_is_empty():
    structure = {
        "evidence_balance": "mixed",
        "supported_bull_points": ["Revenue grew year over year, citing reported figures."],
        "supported_bear_points": ["The scenario spread is wide relative to the base value."],
        "balanced_assessment": "Both sides cite real evidence; the spread reflects genuine uncertainty.",
    }
    assert scan_structure_for_unsupported_claims(structure, providers_present={"yahoo", "sec"}) == []


# ---------------------------------------------------------------------------
# WM corrective patch: "achievable" scoped to a valuation referent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "The base modeled value of $210 is achievable.",
    "An achievable price of $210 per share.",
    "This scenario is achievable.",
    "The bull case per-share outcome looks achievable.",
    "A price target of $210 is achievable.",
])
def test_achievable_still_flags_a_valuation_certainty_claim(text):
    """The rule's actual purpose: asserting a MODELED value will be reached
    without naming the assumptions required."""
    assert any("achievable" in label for label in scan_for_unsupported_claims(text)), text


@pytest.mark.parametrize("text", [
    "Management's margin target looks achievable given announced cost actions.",
    "Revenue growth in the guided range appears achievable.",
    "Cost reductions are achievable within two years.",
    "The buyback target is achievable this year.",
    "The company achieved 10% growth.",
])
def test_achievable_no_longer_flags_ordinary_analytical_english(text):
    """Measured live: WM's bear_researcher tripped the old bare
    `\bachievable\b` pattern in 4 of 5 consecutive runs, always on this same
    word, each trip costing a full extra model call to repair. The rule's own
    comment always said "applied to a DCF value/scenario" -- a margin target
    or a cost-reduction plan is the COMPANY's operating goal, not a claim
    about a modeled valuation, and was never the thing being guarded."""
    assert not any("achievable" in label for label in scan_for_unsupported_claims(text)), text
