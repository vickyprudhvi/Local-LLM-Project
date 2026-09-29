"""Phase H.22: a grounded VALUE is not a valid MAGNITUDE until its currency
and scale are independently grounded too.

Live-traced defect: a EUR-denominated notes filing ("EUR500 million
aggregate principal amount...") had the model correctly read
`currency=EUR` but, with no non-USD scale-bearing unit token available,
fall back to `unit=UNKNOWN` -- and the PRODUCTION validator silently
defaulted that to a x1 multiplier, ACCEPTING a resolved event of 500.0 EUR
where the filing states 500,000,000 EUR. Not a benchmark scoring miss: the
deterministic acceptance boundary itself published a million-x magnitude
error. This file pins `finance.documents.monetary`'s grounding: currency
and scale are each independently supported by language in the SAME cited
evidence as the value, exactly like `EventAmountRole` grounding in
`test_finance_document_amount_roles.py` -- an unstated scale is refused,
never assumed to be UNIT.
"""

import pytest

from finance.documents.event_schema import (
    EventAmount,
    EventAmountRole,
    EventCandidate,
    EventRejectionCode,
)
from finance.documents.event_validator import EventCandidateValidator
from finance.documents.monetary import MonetaryScale
from finance.documents.spans import build_source_spans
from finance.structural_breaks import PostBalanceSheetEventType

_T = PostBalanceSheetEventType


def _spans(text):
    return build_source_spans(text)


def _span_id(spans, substring):
    return next(s.span_id for s in spans if substring in s.text)


_EUR_NOTES_TEXT = (
    "On August 1, 2026, the Company issued and sold €500 million "
    "aggregate principal amount of Senior Notes due 2033 in a Regulation S "
    "offering outside the United States."
)


# ---------------------------------------------------------------------------
# A. The live defect itself, reproduced deterministically: unit=UNKNOWN
# for a stated EUR amount must NOT silently resolve to a x1 (UNIT) scale.
# ---------------------------------------------------------------------------

def test_A_unit_unknown_with_stated_currency_is_refused_not_defaulted_to_unit():
    """The exact live-traced shape: currency correctly read as EUR, but
    unit=UNKNOWN and scale left unstated. Before H.22 this was ACCEPTED as
    500.0 (a million-x understatement); it must now be refused outright."""
    spans = _spans(_EUR_NOTES_TEXT)
    sid = _span_id(spans, "500 million")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=500.0, unit="UNKNOWN",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="EUR",
        funded=True, evidence_span_ids=(sid,), confidence=0.9, form="8-K",
        items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.SCALE_NOT_GROUNDED


def test_B_correct_scale_and_currency_grounds_the_full_magnitude():
    """The fix: an explicit scale=MILLION (independent of currency) grounds
    correctly and normalizes to the FULL 500,000,000 magnitude."""
    spans = _spans(_EUR_NOTES_TEXT)
    sid = _span_id(spans, "500 million")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=500.0, unit="CURRENCY",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="EUR",
        scale=MonetaryScale.MILLION,
        funded=True, evidence_span_ids=(sid,), confidence=0.9, form="8-K",
        items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount == 500_000_000.0
    assert resolved.currency == "EUR"
    assert resolved.scale == MonetaryScale.MILLION


# ---------------------------------------------------------------------------
# C. Model omits a required scale entirely (scale=None, no legacy unit)
# ---------------------------------------------------------------------------

def test_C_omitted_scale_on_currency_amount_is_refused():
    spans = _spans(_EUR_NOTES_TEXT)
    sid = _span_id(spans, "500 million")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=500.0, unit="CURRENCY",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="EUR",
        scale=None,
        funded=True, evidence_span_ids=(sid,), confidence=0.9, form="8-K",
        items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.SCALE_NOT_GROUNDED


def test_C2_scale_unknown_on_currency_amount_is_refused():
    """`scale=UNKNOWN` is the reader's own honest non-answer -- exactly
    like an unstated one, it must never be treated as UNIT."""
    spans = _spans(_EUR_NOTES_TEXT)
    sid = _span_id(spans, "500 million")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=500.0, unit="CURRENCY",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="EUR",
        scale=MonetaryScale.UNKNOWN,
        funded=True, evidence_span_ids=(sid,), confidence=0.9, form="8-K",
        items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.SCALE_NOT_GROUNDED


# ---------------------------------------------------------------------------
# D. Model proposes an UNSUPPORTED scale (claims BILLION when text says
# million)
# ---------------------------------------------------------------------------

def test_D_unsupported_scale_claim_is_refused():
    spans = _spans(_EUR_NOTES_TEXT)
    sid = _span_id(spans, "500 million")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=500.0, unit="CURRENCY",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="EUR",
        scale=MonetaryScale.BILLION,   # wrong -- the text says "million"
        funded=True, evidence_span_ids=(sid,), confidence=0.9, form="8-K",
        items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.SCALE_NOT_GROUNDED
    assert "BILLION" in reason


# ---------------------------------------------------------------------------
# E. Wrong currency claimed -- a well-formed ISO code the source never
# states
# ---------------------------------------------------------------------------

def test_E_wrong_currency_claim_is_refused():
    """GBP is a well-formed ISO code (passes UNKNOWN_CURRENCY's format
    check) but this source states EUR, never GBP/pounds/£."""
    spans = _spans(_EUR_NOTES_TEXT)
    sid = _span_id(spans, "500 million")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=500.0, unit="CURRENCY",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="GBP",
        scale=MonetaryScale.MILLION,
        funded=True, evidence_span_ids=(sid,), confidence=0.9, form="8-K",
        items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.CURRENCY_NOT_GROUNDED


# ---------------------------------------------------------------------------
# F. Bare amount, no scale word -- UNIT is a real, gettable-right answer
# ---------------------------------------------------------------------------

_BARE_TEXT = (
    "On August 1, 2026, the Company issued and sold a promissory note in "
    "the aggregate principal amount of €500 in a private placement."
)


def test_F_unit_scale_accepted_with_no_scale_word_present():
    spans = _spans(_BARE_TEXT)
    sid = _span_id(spans, "€500")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=500.0, unit="CURRENCY",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="EUR",
        scale=MonetaryScale.UNIT,
        funded=True, evidence_span_ids=(sid,), confidence=0.9, form="8-K",
        items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount == 500.0     # literal EUR500, never inflated


def test_F2_million_claim_on_a_bare_amount_is_refused():
    """The mirror image of F: claiming MILLION when the source states a
    bare, unscaled figure is exactly as wrong as the reverse."""
    spans = _spans(_BARE_TEXT)
    sid = _span_id(spans, "€500")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=500.0, unit="CURRENCY",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="EUR",
        scale=MonetaryScale.MILLION,
        funded=True, evidence_span_ids=(sid,), confidence=0.9, form="8-K",
        items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.SCALE_NOT_GROUNDED


# ---------------------------------------------------------------------------
# G. Legacy USD_MILLION/USD_BILLION unit tokens still work (backward
# compatibility, spec section 2)
# ---------------------------------------------------------------------------

_USD_TEXT = (
    "On August 1, 2026, the Company issued and sold $750 million aggregate "
    "principal amount of Senior Notes due 2033 in a private placement."
)


def test_G_legacy_usd_million_unit_still_grounds_correctly():
    spans = _spans(_USD_TEXT)
    sid = _span_id(spans, "$750 million")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=750.0, unit="USD_MILLION",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="USD",
        # scale intentionally left unset -- must be derived from the
        # legacy unit, exactly as every pre-H.22 candidate/fixture does.
        funded=True, evidence_span_ids=(sid,), confidence=0.9, form="8-K",
        items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount == 750_000_000.0
    assert resolved.scale == MonetaryScale.MILLION


def test_G2_legacy_unit_scale_claim_still_needs_language_grounding():
    """Backward compatibility does not mean unconditional trust: a legacy
    USD_BILLION claim against text that only says "million" must still be
    refused -- H.22 closes the gap for legacy candidates too, not just new
    ones."""
    spans = _spans(_USD_TEXT)
    sid = _span_id(spans, "$750 million")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=750.0, unit="USD_BILLION",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="USD",
        funded=True, evidence_span_ids=(sid,), confidence=0.9, form="8-K",
        items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.SCALE_NOT_GROUNDED


# ---------------------------------------------------------------------------
# H. Commitment and drawdown in DIFFERENT currencies -- each amount grounds
# its OWN currency and scale independently
# ---------------------------------------------------------------------------

_DIFFERENT_CCY_TEXT = (
    "On August 1, 2026, the Company entered into a new £750 million "
    "revolving credit facility with a syndicate of lenders; at closing, "
    "the Company drew $400 million under the facility in US Dollars for "
    "general corporate purposes."
)


def test_H_supplementary_amount_grounds_its_own_different_currency():
    spans = _spans(_DIFFERENT_CCY_TEXT)
    commitment_id = _span_id(spans, "£750 million")
    drawn_id = _span_id(spans, "$400 million")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=750.0, unit="CURRENCY",
        amount_role=EventAmountRole.FACILITY_COMMITMENT, currency="GBP",
        scale=MonetaryScale.MILLION, funded=True, committed=True,
        evidence_span_ids=(commitment_id, drawn_id),
        supplementary_amounts=(
            EventAmount(role=EventAmountRole.AMOUNT_DRAWN, value=400.0,
                       unit="CURRENCY", currency="USD",
                       scale=MonetaryScale.MILLION,
                       evidence_span_ids=(drawn_id,)),
        ),
        confidence=0.95, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.currency == "GBP"
    assert resolved.amount == 750_000_000.0
    supplement = resolved.supplementary_amounts[0]
    assert supplement.currency == "USD"
    assert supplement.value == 400_000_000.0


def test_H2_supplementary_amount_with_no_currency_inherits_the_primarys():
    """Backward compatibility: a supplementary amount that states no
    currency of its own (every pre-H.22 `EventAmount`) inherits the
    primary's -- it is never left unattributed."""
    spans = _spans(_DIFFERENT_CCY_TEXT)
    commitment_id = _span_id(spans, "£750 million")
    drawn_id = _span_id(spans, "$400 million")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=750.0, unit="CURRENCY",
        amount_role=EventAmountRole.FACILITY_COMMITMENT, currency="GBP",
        scale=MonetaryScale.MILLION, funded=True, committed=True,
        evidence_span_ids=(commitment_id, drawn_id),
        supplementary_amounts=(
            # No currency stated -- but the cited evidence only supports
            # GBP language for THIS value at 400, not USD ("400 million"
            # inherits GBP -> fails to find "$400 million" as a GBP claim,
            # so grounding correctly refuses/drops rather than mislabeling.
            EventAmount(role=EventAmountRole.AMOUNT_DRAWN, value=400.0,
                       unit="CURRENCY", scale=MonetaryScale.MILLION,
                       evidence_span_ids=(drawn_id,)),
        ),
        confidence=0.95, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    # The mislabeled (inherited-GBP) supplementary claim could not be
    # grounded against "$400 million" (no GBP language there) and was
    # correctly DROPPED -- the primary GBP750M commitment still stands.
    assert resolved.amount == 750_000_000.0
    assert resolved.supplementary_amounts == ()


# ---------------------------------------------------------------------------
# I. Scale grounding searches the FULL multi-span cited evidence, not just
# the span containing the bare number
# ---------------------------------------------------------------------------

_ADJACENT_SPANS_TEXT = (
    "On August 1, 2026, the Company entered into a new senior secured term "
    "loan facility with a syndicate of lenders. The facility's aggregate "
    "commitment is £750; that commitment figure is expressed in "
    "millions of pounds sterling."
)


def test_I_scale_language_grounds_across_multiple_cited_spans():
    spans = _spans(_ADJACENT_SPANS_TEXT)
    number_id = _span_id(spans, "£750")
    scale_word_id = _span_id(spans, "millions of pounds")
    assert number_id != scale_word_id      # genuinely two different spans
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=750.0, unit="CURRENCY",
        amount_role=EventAmountRole.FACILITY_COMMITMENT, currency="GBP",
        scale=MonetaryScale.MILLION, committed=True,
        evidence_span_ids=(number_id, scale_word_id),
        confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount == 750_000_000.0


def test_I2_scale_word_not_among_cited_spans_is_refused():
    """The mirror image of I: if the model cites ONLY the span with the
    bare number, never the one stating the scale, grounding correctly
    refuses -- citing more spans is not optional once a scale is claimed."""
    spans = _spans(_ADJACENT_SPANS_TEXT)
    number_id = _span_id(spans, "£750")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=750.0, unit="CURRENCY",
        amount_role=EventAmountRole.FACILITY_COMMITMENT, currency="GBP",
        scale=MonetaryScale.MILLION, committed=True,
        evidence_span_ids=(number_id,),   # scale-word span NOT cited
        confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.SCALE_NOT_GROUNDED


# ---------------------------------------------------------------------------
# J. Scale does not apply to a PER_SHARE amount -- no grounding required
# ---------------------------------------------------------------------------

_PER_SHARE_TEXT = (
    "Each outstanding share of common stock was converted into the right "
    "to receive $95.00 in cash, without interest."
)


def test_J_per_share_amount_needs_no_scale_grounding():
    spans = _spans(_PER_SHARE_TEXT)
    sid = _span_id(spans, "$95.00")
    candidate = EventCandidate(
        event_type=_T.ACQUISITION, amount=95.0, unit="PER_SHARE",
        amount_role=EventAmountRole.PER_SHARE_CONSIDERATION, currency="USD",
        evidence_span_ids=(sid,), confidence=0.9, form="8-K", items="2.01")
    validator = EventCandidateValidator(spans=spans, items="2.01", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount == 95.0


# ---------------------------------------------------------------------------
# Phase H.30 -- monetary MENTION binding (spec section 11's A-J, live-traced
# defect: "$2.0 billion ... EUR500 million" let a candidate claim
# value=2.0/currency=USD/scale=MILLION, because "million" existed SOMEWHERE
# in the cited evidence -- it just belonged to EUR500, not $2.0. Currency and
# scale must now be grounded in the SPECIFIC numeric mention's own window,
# never merely present in the wider cited text.
# ---------------------------------------------------------------------------

_TWO_MENTION_TEXT = (
    "On October 22, 2026, the Company entered into a new $2.0 billion "
    "multicurrency revolving credit facility with a syndicate of lenders, "
    "of which up to €500 million is available for borrowings denominated "
    "in Euro. No amounts have been drawn under the facility as of the date "
    "of this filing."
)


def _two_mention_candidate(value, currency, scale, role=EventAmountRole.FACILITY_COMMITMENT):
    spans = _spans(_TWO_MENTION_TEXT)
    all_ids = tuple(s.span_id for s in spans)
    return spans, EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=value, unit="CURRENCY",
        amount_role=role, currency=currency, scale=scale,
        funded=False, committed=True, evidence_span_ids=all_ids,
        confidence=0.9, form="8-K", items="2.03")


def test_mention_A_correct_billion_usd_claim_for_the_2_0_mention_is_accepted():
    spans, candidate = _two_mention_candidate(2.0, "USD", MonetaryScale.BILLION)
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount == 2_000_000_000.0
    assert resolved.currency == "USD"


def test_mention_B_the_h28_defect_2_0_usd_million_is_now_rejected():
    """The exact live-traced unsafe shape: MILLION belongs to EUR500, not
    2.0 -- must be refused, not accepted at a thousand-x-too-small value."""
    spans, candidate = _two_mention_candidate(2.0, "USD", MonetaryScale.MILLION)
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.SCALE_NOT_GROUNDED


def test_mention_C_500_eur_million_the_correct_mention_is_accepted():
    spans, candidate = _two_mention_candidate(500.0, "EUR", MonetaryScale.MILLION)
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount == 500_000_000.0
    assert resolved.currency == "EUR"


def test_mention_D_500_usd_million_borrows_the_wrong_mentions_currency_is_rejected():
    """500 genuinely exists, MILLION genuinely exists -- but the dollar
    sign belongs to $2.0, not to 500. USD must not be grounded for 500."""
    spans, candidate = _two_mention_candidate(500.0, "USD", MonetaryScale.MILLION)
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.CURRENCY_NOT_GROUNDED


_TWO_USD_DIFFERENT_SCALES_TEXT = (
    "On August 1, 2026, the Company issued $500 million of Senior Notes "
    "due 2031. Separately, the Company entered into a new $2 billion "
    "revolving credit facility with a syndicate of lenders; no amounts "
    "have been drawn under the facility as of the date of this filing."
)


def test_mention_E_two_usd_values_with_different_scales_stay_independent():
    spans = _spans(_TWO_USD_DIFFERENT_SCALES_TEXT)
    notes_id = _span_id(spans, "$500 million")
    facility_ids = tuple(s.span_id for s in spans if s.span_id != notes_id)
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")

    notes = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=500.0, unit="CURRENCY",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="USD",
        scale=MonetaryScale.MILLION, funded=True, evidence_span_ids=(notes_id,),
        confidence=0.9, form="8-K", items="2.03")
    resolved, code, reason = validator.validate(notes)
    assert code is None, reason
    assert resolved.amount == 500_000_000.0

    facility = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=2.0, unit="CURRENCY",
        amount_role=EventAmountRole.FACILITY_COMMITMENT, currency="USD",
        scale=MonetaryScale.BILLION, committed=True, evidence_span_ids=facility_ids,
        confidence=0.9, form="8-K", items="2.03")
    resolved2, code2, reason2 = validator.validate(facility)
    assert code2 is None, reason2
    assert resolved2.amount == 2_000_000_000.0

    # Cross-associated claims (500 as BILLION, 2 as MILLION) must both fail.
    wrong1 = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=500.0, unit="CURRENCY",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="USD",
        scale=MonetaryScale.BILLION, funded=True, evidence_span_ids=(notes_id,),
        confidence=0.9, form="8-K", items="2.03")
    r, c, _ = validator.validate(wrong1)
    assert r is None and c == EventRejectionCode.SCALE_NOT_GROUNDED

    wrong2 = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=2.0, unit="CURRENCY",
        amount_role=EventAmountRole.FACILITY_COMMITMENT, currency="USD",
        scale=MonetaryScale.MILLION, committed=True, evidence_span_ids=facility_ids,
        confidence=0.9, form="8-K", items="2.03")
    r2, c2, _ = validator.validate(wrong2)
    assert r2 is None and c2 == EventRejectionCode.SCALE_NOT_GROUNDED


_SAME_NUMBER_TWICE_TEXT = (
    "The Company drew $500 million under its revolving facility and, in a "
    "separate transaction the same week, issued €500 million of Senior "
    "Notes due 2032."
)


def test_mention_F_same_numeric_value_twice_with_different_currency_no_cross_association():
    spans = _spans(_SAME_NUMBER_TWICE_TEXT)
    all_ids = tuple(s.span_id for s in spans)
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")

    usd_claim = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=500.0, unit="CURRENCY",
        amount_role=EventAmountRole.AMOUNT_DRAWN, currency="USD",
        scale=MonetaryScale.MILLION, funded=True, evidence_span_ids=all_ids,
        confidence=0.9, form="8-K", items="2.03")
    resolved, code, reason = validator.validate(usd_claim)
    assert code is None, reason
    assert resolved.currency == "USD"

    eur_claim = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=500.0, unit="CURRENCY",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="EUR",
        scale=MonetaryScale.MILLION, funded=True, evidence_span_ids=all_ids,
        confidence=0.9, form="8-K", items="2.03")
    resolved2, code2, reason2 = validator.validate(eur_claim)
    assert code2 is None, reason2
    assert resolved2.currency == "EUR"
    # Both 500-mentions are independently, correctly grounded to their OWN
    # currency -- neither borrowed the other's.


def test_mention_G_primary_and_supplementary_amount_independence():
    """Same source as the core H.28 defect: primary $2.0B and supplementary
    EUR500M must each keep their OWN currency/scale, never cross-associated,
    exactly as the pre-existing multi-currency-facility fixture expects."""
    spans = _spans(_TWO_MENTION_TEXT)
    all_ids = tuple(s.span_id for s in spans)
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=2.0, unit="CURRENCY",
        amount_role=EventAmountRole.FACILITY_COMMITMENT, currency="USD",
        scale=MonetaryScale.BILLION, funded=False, committed=True,
        evidence_span_ids=all_ids,
        supplementary_amounts=(
            EventAmount(role=EventAmountRole.FACILITY_COMMITMENT, value=500.0,
                       unit="CURRENCY", currency="EUR", scale=MonetaryScale.MILLION,
                       evidence_span_ids=all_ids),
        ),
        confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount == 2_000_000_000.0 and resolved.currency == "USD"
    supp = resolved.supplementary_amounts[0]
    assert supp.value == 500_000_000.0 and supp.currency == "EUR"


def test_mention_H_scale_word_belonging_to_another_amount_cannot_be_borrowed():
    """Explicit restatement of the core invariant: a scale word's mere
    PRESENCE in the cited evidence is not enough once more than one
    monetary mention exists -- it must sit in the CLAIMED value's own
    window."""
    spans, wrong = _two_mention_candidate(2.0, "USD", MonetaryScale.MILLION)
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(wrong)
    assert resolved is None
    assert code == EventRejectionCode.SCALE_NOT_GROUNDED
    assert "DIFFERENT number" in reason


def test_mention_I_legacy_usd_million_billion_unaffected_by_mention_binding():
    """Backward compatibility (spec section 8.I): a single-mention legacy
    candidate is completely unaffected by the windowing change -- there is
    only one numeric mention, so its window is the whole text, exactly as
    before this phase."""
    spans = _spans(_USD_TEXT)
    sid = _span_id(spans, "$750 million")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=750.0, unit="USD_MILLION",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="USD",
        funded=True, evidence_span_ids=(sid,), confidence=0.9, form="8-K",
        items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount == 750_000_000.0


def test_mention_J_bare_500_as_unit_accepted_only_when_no_scale_word_supports_it():
    """A bare, single-mention amount with no scale word anywhere: UNIT is
    still a correct, gettable-right answer, unaffected by mention binding
    (there is nothing to disambiguate against)."""
    spans = _spans(_BARE_TEXT)
    sid = _span_id(spans, "€500")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=500.0, unit="CURRENCY",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="EUR",
        scale=MonetaryScale.UNIT, funded=True, evidence_span_ids=(sid,),
        confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount == 500.0


# ---------------------------------------------------------------------------
# Phase H.30 -- FUNDED-STATUS semantics (spec section 12's K-M). `funded`
# covers completion/proceeds-received state for EVERY capital-transaction
# type this schema handles (see `event_validator.py`'s `_FUNDED_LANGUAGE`
# docstring) -- debt drawdown and completed equity issuance alike.
# ---------------------------------------------------------------------------

_UNDRAWN_FACILITY_TEXT = (
    "On August 1, 2026, the Company entered into a new $2.0 billion "
    "revolving credit facility with a syndicate of lenders. No amounts "
    "have been drawn under the facility as of the date of this filing."
)

_DRAWN_FACILITY_TEXT = (
    "On August 1, 2026, the Company entered into a new $2.0 billion "
    "revolving credit facility with a syndicate of lenders and drew the "
    "full amount at closing."
)

_COMPLETED_EQUITY_TEXT = (
    "On August 15, 2026, the Company completed an underwritten public "
    "offering of 10,000,000 shares of common stock, issuing and selling "
    "the shares for aggregate net proceeds of approximately $450 million."
)


def test_funded_K_explicit_undrawn_facility_rejects_funded_true():
    spans = _spans(_UNDRAWN_FACILITY_TEXT)
    all_ids = tuple(s.span_id for s in spans)
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=2.0, unit="CURRENCY",
        amount_role=EventAmountRole.FACILITY_COMMITMENT, currency="USD",
        scale=MonetaryScale.BILLION, funded=True, committed=True,
        evidence_span_ids=all_ids, confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.FUNDED_STATUS_NOT_GROUNDED


def test_funded_L_explicit_draw_rejects_funded_false():
    """The NEW, symmetric direction this phase adds: cited evidence stating
    a draw occurred contradicts a funded=False claim."""
    spans = _spans(_DRAWN_FACILITY_TEXT)
    all_ids = tuple(s.span_id for s in spans)
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=2.0, unit="CURRENCY",
        amount_role=EventAmountRole.AMOUNT_DRAWN, currency="USD",
        scale=MonetaryScale.BILLION, funded=False, committed=True,
        evidence_span_ids=all_ids, confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.FUNDED_STATUS_CONTRADICTED


def test_funded_M_completed_equity_issuance_rejects_funded_false():
    """The exact H.28 live finding, reproduced deterministically: a
    completed $450M equity sale with explicit proceeds language must not
    be accepted with funded=False."""
    spans = _spans(_COMPLETED_EQUITY_TEXT)
    all_ids = tuple(s.span_id for s in spans)
    candidate = EventCandidate(
        event_type=_T.ISSUER_EQUITY_ISSUANCE, amount=450.0, unit="CURRENCY",
        amount_role=EventAmountRole.TRANSACTION_VALUE, currency="USD",
        scale=MonetaryScale.MILLION, funded=False, committed=False,
        evidence_span_ids=all_ids, confidence=0.9, form="8-K", items="3.02")
    validator = EventCandidateValidator(spans=spans, items="3.02", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.FUNDED_STATUS_CONTRADICTED


def test_funded_M2_completed_equity_issuance_accepts_funded_true():
    """Positive control: the SAME text, correctly labeled funded=True,
    must still be accepted -- this phase closes a false-negative gap, it
    does not make funded status harder to get right."""
    spans = _spans(_COMPLETED_EQUITY_TEXT)
    all_ids = tuple(s.span_id for s in spans)
    candidate = EventCandidate(
        event_type=_T.ISSUER_EQUITY_ISSUANCE, amount=450.0, unit="CURRENCY",
        amount_role=EventAmountRole.TRANSACTION_VALUE, currency="USD",
        scale=MonetaryScale.MILLION, funded=True, committed=False,
        evidence_span_ids=all_ids, confidence=0.9, form="8-K", items="3.02")
    validator = EventCandidateValidator(spans=spans, items="3.02", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.funded is True
    assert resolved.amount == 450_000_000.0
