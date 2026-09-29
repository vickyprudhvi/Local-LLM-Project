"""Phase H.20: a grounded number does not prove its own economic meaning.

Live shadow evidence: a real acquisition filing states "$95.00 in cash ...
(the 'Per Share Amount')". Citing the span containing "95.00" and reporting
`amount=95.0` passes every H.18 grounding check -- the number IS in the
source. It would still be wrong read as the deal's transaction value, which
was tens of billions of dollars. This file pins `EventAmountRole` grounding:
a role, once asserted, must be independently supported by role-specific
language in the SAME cited evidence as the value, or the whole candidate is
refused -- never silently relabelled UNKNOWN.
"""

import pytest

from finance.documents.event_schema import (
    EventAmount,
    EventAmountRole,
    EventCandidate,
    EventRejectionCode,
)
from finance.documents.event_validator import EventCandidateValidator
from finance.documents.spans import build_source_spans
from finance.structural_breaks import PostBalanceSheetEventType

_T = PostBalanceSheetEventType


def _spans(text):
    return build_source_spans(text)


def _span_id(spans, substring):
    return next(s.span_id for s in spans if substring in s.text)


# ---------------------------------------------------------------------------
# A/B/C. Acquisition prose: per-share vs. total, and both together
# ---------------------------------------------------------------------------

_ACQUISITION_PER_SHARE_TEXT = (
    "On August 1, 2026, the Company completed its acquisition of Example "
    "Target, Inc. Each outstanding share of Example Target common stock was "
    "converted into the right to receive $95.00 in cash, without interest "
    "(the \"Per Share Amount\")."
)

_ACQUISITION_TOTAL_TEXT = (
    "On August 1, 2026, the Company completed its acquisition of Example "
    "Target, Inc. for an aggregate transaction value of approximately "
    "$69.0 billion."
)

_ACQUISITION_BOTH_TEXT = (
    "On August 1, 2026, the Company completed its acquisition of Example "
    "Target, Inc. Each outstanding share of Example Target common stock was "
    "converted into the right to receive $95.00 in cash, representing an "
    "aggregate transaction value of approximately $69.0 billion."
)


def test_A_per_share_price_accepted_as_per_share_not_transaction_value():
    spans = _spans(_ACQUISITION_PER_SHARE_TEXT)
    sid = _span_id(spans, "$95.00")
    candidate = EventCandidate(
        event_type=_T.ACQUISITION, amount=95.0, unit="PER_SHARE",
        amount_role=EventAmountRole.PER_SHARE_CONSIDERATION, currency="USD",
        evidence_span_ids=(sid,), confidence=0.95, form="8-K", items="2.01")
    validator = EventCandidateValidator(spans=spans, items="2.01", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount_role == EventAmountRole.PER_SHARE_CONSIDERATION
    assert resolved.amount == 95.0


def test_A_per_share_price_rejected_when_claimed_as_transaction_value():
    """The exact live-shadow defect, reproduced generically: a per-share
    price cited as the source for a TRANSACTION_VALUE claim must be refused
    -- the cited text says 'per share', not 'transaction value'."""
    spans = _spans(_ACQUISITION_PER_SHARE_TEXT)
    sid = _span_id(spans, "$95.00")
    candidate = EventCandidate(
        event_type=_T.ACQUISITION, amount=95.0, unit="USD",
        amount_role=EventAmountRole.TRANSACTION_VALUE, currency="USD",
        evidence_span_ids=(sid,), confidence=0.95, form="8-K", items="2.01")
    validator = EventCandidateValidator(spans=spans, items="2.01", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.AMOUNT_ROLE_NOT_GROUNDED


def test_B_explicit_transaction_value_accepted():
    spans = _spans(_ACQUISITION_TOTAL_TEXT)
    sid = _span_id(spans, "$69.0 billion")
    candidate = EventCandidate(
        event_type=_T.ACQUISITION, amount=69.0, unit="USD_BILLION",
        amount_role=EventAmountRole.TRANSACTION_VALUE, currency="USD",
        evidence_span_ids=(sid,), confidence=0.95, form="8-K", items="2.01")
    validator = EventCandidateValidator(spans=spans, items="2.01", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount_role == EventAmountRole.TRANSACTION_VALUE
    assert resolved.amount == 69_000_000_000.0


def test_C_both_amounts_survive_with_distinct_roles():
    spans = _spans(_ACQUISITION_BOTH_TEXT)
    per_share_id = _span_id(spans, "$95.00")
    total_id = _span_id(spans, "$69.0 billion")
    candidate = EventCandidate(
        event_type=_T.ACQUISITION, amount=95.0, unit="PER_SHARE",
        amount_role=EventAmountRole.PER_SHARE_CONSIDERATION, currency="USD",
        evidence_span_ids=(per_share_id,),
        supplementary_amounts=(
            EventAmount(role=EventAmountRole.TRANSACTION_VALUE, value=69.0,
                       unit="USD_BILLION", evidence_span_ids=(total_id,)),
        ),
        confidence=0.95, form="8-K", items="2.01")
    validator = EventCandidateValidator(spans=spans, items="2.01", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount == 95.0
    assert resolved.amount_role == EventAmountRole.PER_SHARE_CONSIDERATION
    assert len(resolved.supplementary_amounts) == 1
    supplement = resolved.supplementary_amounts[0]
    assert supplement.role == EventAmountRole.TRANSACTION_VALUE
    assert supplement.value == 69_000_000_000.0


# ---------------------------------------------------------------------------
# D. Notes principal amount
# ---------------------------------------------------------------------------

_NOTES_TEXT = (
    "On August 1, 2026, the Company issued and sold $750 million aggregate "
    "principal amount of 5.500% Senior Notes due 2033 in a private "
    "placement."
)


def test_D_principal_amount_accepted():
    spans = _spans(_NOTES_TEXT)
    sid = _span_id(spans, "$750 million")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=750.0, unit="USD_MILLION",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="USD",
        funded=True, evidence_span_ids=(sid,), confidence=0.95, form="8-K",
        items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount_role == EventAmountRole.PRINCIPAL_AMOUNT


# ---------------------------------------------------------------------------
# E/F. Revolver: commitment alone, and commitment + drawn together
# ---------------------------------------------------------------------------

_REVOLVER_COMMITMENT_ONLY_TEXT = (
    "On August 1, 2026, the Company entered into a $5.0 billion revolving "
    "credit facility with a syndicate of lenders. No amounts have been "
    "drawn under the facility as of the date of this filing."
)

_REVOLVER_PARTIAL_DRAWDOWN_TEXT = (
    "On August 1, 2026, the Company entered into a $5.0 billion revolving "
    "credit facility with a syndicate of lenders; at closing, the Company "
    "drew $2.0 billion under the facility."
)


def test_E_facility_commitment_accepted():
    spans = _spans(_REVOLVER_COMMITMENT_ONLY_TEXT)
    sid = _span_id(spans, "$5.0 billion")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=5.0, unit="USD_BILLION",
        amount_role=EventAmountRole.FACILITY_COMMITMENT, currency="USD",
        funded=False, committed=True, evidence_span_ids=(sid,), confidence=0.95,
        form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount_role == EventAmountRole.FACILITY_COMMITMENT
    assert resolved.funded is False


def test_F_commitment_and_drawn_amount_both_survive_with_correct_roles():
    spans = _spans(_REVOLVER_PARTIAL_DRAWDOWN_TEXT)
    commitment_id = _span_id(spans, "$5.0 billion")
    drawn_id = _span_id(spans, "$2.0 billion")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=5.0, unit="USD_BILLION",
        amount_role=EventAmountRole.FACILITY_COMMITMENT, currency="USD",
        funded=True, committed=True, evidence_span_ids=(commitment_id, drawn_id),
        supplementary_amounts=(
            EventAmount(role=EventAmountRole.AMOUNT_DRAWN, value=2.0,
                       unit="USD_BILLION", evidence_span_ids=(drawn_id,)),
        ),
        confidence=0.95, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount == 5_000_000_000.0
    assert resolved.amount_role == EventAmountRole.FACILITY_COMMITMENT
    assert resolved.supplementary_amounts[0].role == EventAmountRole.AMOUNT_DRAWN
    assert resolved.supplementary_amounts[0].value == 2_000_000_000.0


# ---------------------------------------------------------------------------
# G. Repayment
# ---------------------------------------------------------------------------

_REPAYMENT_TEXT = (
    "On August 1, 2026, the Company repaid $800 million in aggregate "
    "principal amount of its outstanding senior notes at maturity."
)


def test_G_repayment_amount_accepted():
    spans = _spans(_REPAYMENT_TEXT)
    sid = _span_id(spans, "$800 million")
    candidate = EventCandidate(
        event_type=_T.REFINANCING, amount=800.0, unit="USD_MILLION",
        amount_role=EventAmountRole.REPAYMENT_AMOUNT, currency="USD",
        evidence_span_ids=(sid,), confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount_role == EventAmountRole.REPAYMENT_AMOUNT


# ---------------------------------------------------------------------------
# H. correct value, wrong role -> rejected
# ---------------------------------------------------------------------------

def test_H_correct_value_wrong_role_rejected():
    """$750 million IS the principal amount of the notes -- claiming it as a
    FACILITY_COMMITMENT (no facility/commitment language anywhere) must be
    refused even though the number itself is perfectly grounded."""
    spans = _spans(_NOTES_TEXT)
    sid = _span_id(spans, "$750 million")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=750.0, unit="USD_MILLION",
        amount_role=EventAmountRole.FACILITY_COMMITMENT, currency="USD",
        evidence_span_ids=(sid,), confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.AMOUNT_ROLE_NOT_GROUNDED


# ---------------------------------------------------------------------------
# I/L. Per-share consideration presented as a transaction total -- rejected
# ---------------------------------------------------------------------------

def test_I_transaction_total_from_per_share_evidence_rejected():
    spans = _spans(_ACQUISITION_PER_SHARE_TEXT)
    sid = _span_id(spans, "$95.00")
    candidate = EventCandidate(
        event_type=_T.ACQUISITION, amount=95.0, unit="USD",
        amount_role=EventAmountRole.TRANSACTION_VALUE, currency="USD",
        evidence_span_ids=(sid,), confidence=0.9, form="8-K", items="2.01")
    validator = EventCandidateValidator(spans=spans, items="2.01", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.AMOUNT_ROLE_NOT_GROUNDED


def test_L_unit_role_mismatch_rejected_even_with_correct_looking_text():
    """Section 3: role and unit must agree. A per-share role with a bare
    USD unit (not PER_SHARE) is refused on the unit mismatch alone, before
    the language check even runs."""
    spans = _spans(_ACQUISITION_PER_SHARE_TEXT)
    sid = _span_id(spans, "$95.00")
    candidate = EventCandidate(
        event_type=_T.ACQUISITION, amount=95.0, unit="USD",   # wrong: not PER_SHARE
        amount_role=EventAmountRole.PER_SHARE_CONSIDERATION, currency="USD",
        evidence_span_ids=(sid,), confidence=0.9, form="8-K", items="2.01")
    validator = EventCandidateValidator(spans=spans, items="2.01", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.AMOUNT_ROLE_NOT_GROUNDED
    assert "unit" in reason.lower()


# ---------------------------------------------------------------------------
# J. Model arithmetic not present in source -- rejected (reuses the H.19
# finding: a computed sum is not a value the source states)
# ---------------------------------------------------------------------------

def test_J_computed_amount_not_literally_in_source_rejected():
    text = ("On August 1, 2026, the Company drew $11.5 billion under its "
           "two-year term loan facility and $3.0 billion under its 364-day "
           "term loan facility.")
    spans = _spans(text)
    sid = _span_id(spans, "$11.5 billion")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=14.5, unit="USD_BILLION",
        amount_role=EventAmountRole.AMOUNT_DRAWN, currency="USD", funded=True,
        evidence_span_ids=(sid,), confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.AMOUNT_NOT_GROUNDED


# ---------------------------------------------------------------------------
# K. A FACILITY_COMMITMENT amount cannot make `funded` true on its own
# ---------------------------------------------------------------------------

def test_K_facility_commitment_role_does_not_make_funded_true():
    """Grounding the commitment amount and its role says nothing about
    whether money moved -- `funded` is refused independently."""
    spans = _spans(_REVOLVER_COMMITMENT_ONLY_TEXT)
    sid = _span_id(spans, "$5.0 billion")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=5.0, unit="USD_BILLION",
        amount_role=EventAmountRole.FACILITY_COMMITMENT, currency="USD",
        funded=True,   # wrong: no drawdown language anywhere in this text
        evidence_span_ids=(sid,), confidence=0.95, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.FUNDED_STATUS_NOT_GROUNDED


# ---------------------------------------------------------------------------
# Backward compatibility: UNKNOWN role is a legitimate, passing value
# ---------------------------------------------------------------------------

def test_unknown_role_is_accepted_unclassified():
    spans = _spans(_REVOLVER_COMMITMENT_ONLY_TEXT)
    sid = _span_id(spans, "$5.0 billion")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=5.0, unit="USD_BILLION",
        currency="USD", funded=False, committed=True,
        evidence_span_ids=(sid,), confidence=0.9, form="8-K", items="2.03")
    assert candidate.amount_role == EventAmountRole.UNKNOWN
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount_role == EventAmountRole.UNKNOWN


# ---------------------------------------------------------------------------
# Live H.20 finding: a bad SUPPLEMENTARY role must not sink a sound primary
# ---------------------------------------------------------------------------

_CONVERTIBLE_WITH_NET_PROCEEDS_TEXT = (
    "On August 1, 2026, the Company issued and sold $800 million aggregate "
    "principal amount of convertible senior notes due 2031 in a private "
    "placement, receiving net proceeds of approximately $784 million."
)


def test_bad_supplementary_role_is_dropped_not_fatal_to_a_sound_primary():
    """Live finding: the model tagged the note's own $800M PRINCIPAL_AMOUNT
    correctly, but tagged the supplementary "$784 million net proceeds" as
    TRANSACTION_VALUE -- a role CONVERTIBLE_ISSUANCE's compatibility matrix
    does not admit. The whole candidate must NOT be refused over the
    disputed supplementary figure: the sound, correctly-grounded primary
    amount survives, and only the unverifiable supplementary claim is
    dropped (a miss, never a wrong publication)."""
    spans = _spans(_CONVERTIBLE_WITH_NET_PROCEEDS_TEXT)
    principal_id = _span_id(spans, "$800 million")
    proceeds_id = _span_id(spans, "$784 million")
    candidate = EventCandidate(
        event_type=_T.CONVERTIBLE_ISSUANCE, amount=800.0, unit="USD_MILLION",
        amount_role=EventAmountRole.PRINCIPAL_AMOUNT, currency="USD",
        funded=True, evidence_span_ids=(principal_id,),
        supplementary_amounts=(
            EventAmount(role=EventAmountRole.TRANSACTION_VALUE, value=784.0,
                       unit="USD_MILLION", evidence_span_ids=(proceeds_id,)),
        ),
        confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.amount == 800_000_000.0
    assert resolved.amount_role == EventAmountRole.PRINCIPAL_AMOUNT
    assert resolved.supplementary_amounts == ()


def test_event_type_role_compatibility_rejects_implausible_combination():
    """A per-share consideration role makes no sense for a bare debt
    issuance with no equity/conversion dimension -- refused by the
    compatibility matrix even if some contrived language matched."""
    text = ("On August 1, 2026, the Company issued $500 million aggregate "
           "principal amount of notes, per share of which no equity was "
           "involved.")
    spans = _spans(text)
    sid = _span_id(spans, "$500 million")
    candidate = EventCandidate(
        event_type=_T.ISSUER_DEBT_ISSUANCE, amount=500.0, unit="PER_SHARE",
        amount_role=EventAmountRole.PER_SHARE_CONSIDERATION, currency="USD",
        evidence_span_ids=(sid,), confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.AMOUNT_ROLE_NOT_GROUNDED
