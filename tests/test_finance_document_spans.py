"""Phase H.18: the source-span builder, and the reproduce-first fixtures.

Section 1 of the task asks for fixtures shaped like the observed failure --
a dense multi-clause SEC sentence -- BEFORE any fix is trusted. Section 10
asks for a fixed, generalized test list (A-J). This file covers both: the
fixtures live here as the ground truth for what a real filing sentence looks
like, and the span-builder invariants (never lose text, never fabricate,
deterministic ids, cross-document rejection) are proven directly against
them, independent of any model call.

The all-important architectural test is `test_span_text_is_always_an_exact_substring`:
`finance.documents.spans.SourceSpan.text` must equal
`source[start:end]` for EVERY span this module ever builds, on every one of
these fixtures. That single property is what makes fabricated evidence
structurally impossible rather than merely checked-for.
"""

import pytest

from finance.documents.event_schema import EventCandidate, EventRejectionCode
from finance.documents.event_validator import EventCandidateValidator
from finance.documents.spans import build_source_spans, resolve_span_ids
from finance.structural_breaks import PostBalanceSheetEventType


# ---------------------------------------------------------------------------
# Section 1's reproduce-first fixtures. Fictional/generic, as required.
# ---------------------------------------------------------------------------

# A. Single dense multi-clause financing sentence (the task's own example).
#    Amount and funding status share ONE clause-free sentence -- no
#    semicolon to split on, so this is exactly the shape that broke
#    transcription: a model asked to "copy the exact sentence" had to
#    reproduce all of this verbatim, including "subject to customary
#    conditions", and sometimes didn't.
FIXTURE_A_DENSE_SENTENCE = (
    "The Company entered into a five-year revolving credit agreement which "
    "provides for commitments of $3.0 billion, subject to customary "
    "conditions, and no amounts were drawn as of the effective date."
)

# B. Amount appears in a middle clause of a longer, comma-heavy sentence.
FIXTURE_B_AMOUNT_IN_MIDDLE_CLAUSE = (
    "Pursuant to the agreement, dated as of August 3, 2026, among the "
    "Company, the lenders party thereto, and Example Bank, N.A., as "
    "administrative agent, the lenders committed to provide term loans in "
    "an aggregate principal amount of $1.2 billion to the Company."
)

# C. Event meaning in one clause, funding status in another, semicolon-joined.
FIXTURE_C_EVENT_AND_FUNDING_SEPARATE_CLAUSES = (
    "The Company entered into a $900 million term loan credit agreement "
    "with a syndicate of lenders; on the closing date, the Company borrowed "
    "the full amount available under the agreement."
)

# D. Semicolon-separated clauses, three deep.
FIXTURE_D_SEMICOLON_CHAIN = (
    "The credit agreement provides for revolving commitments of $500 "
    "million; it matures on August 1, 2031; and no amounts were drawn at "
    "closing."
)

# E. Parenthetical clause embedded in the operative sentence.
FIXTURE_E_PARENTHETICAL = (
    "The Company entered into a $2.5 billion term loan agreement (the "
    "\"Term Loan Agreement\") and, substantially concurrently with such "
    "entry, drew the full amount available thereunder."
)

# F. Defined terms in quotes, referenced later in the same document.
FIXTURE_F_DEFINED_TERMS = (
    "On August 10, 2026, the Company, as borrower, entered into a Credit "
    "Agreement (as defined below) with the lenders party thereto. The "
    "\"Credit Agreement\" provides commitments of $1.0 billion. No amounts "
    "were drawn under the Credit Agreement as of the date of this report."
)

# G. Text as it would arrive after HTML-to-text normalization: collapsed
# whitespace, a stray non-breaking space, entities already decoded.
FIXTURE_G_NORMALIZED_HTML_TEXT = (
    "The Company entered into a  $750 million  bridge loan facility "
    "with Example Bank, N.A.; the facility remains undrawn as of the "
    "date of this filing."
)


ALL_FIXTURES = (
    FIXTURE_A_DENSE_SENTENCE, FIXTURE_B_AMOUNT_IN_MIDDLE_CLAUSE,
    FIXTURE_C_EVENT_AND_FUNDING_SEPARATE_CLAUSES, FIXTURE_D_SEMICOLON_CHAIN,
    FIXTURE_E_PARENTHETICAL, FIXTURE_F_DEFINED_TERMS,
    FIXTURE_G_NORMALIZED_HTML_TEXT,
)


# ---------------------------------------------------------------------------
# The architectural invariant
# ---------------------------------------------------------------------------

def test_span_text_is_always_an_exact_substring():
    for fixture in ALL_FIXTURES:
        for span in build_source_spans(fixture):
            assert fixture[span.start:span.end] == span.text
            assert span.text == fixture[span.start:span.end]  # both directions, no surprises


def test_span_builder_never_fabricates_or_loses_a_word():
    """Every non-whitespace token in the source appears in some span's
    text, in the same order -- splitting drops boundary whitespace only,
    never content."""
    for fixture in ALL_FIXTURES:
        spans = build_source_spans(fixture, min_chars=0)
        recombined = " ".join(span.text for span in spans)
        assert recombined.split() == fixture.split()


def test_span_ids_are_deterministic_across_calls():
    for fixture in ALL_FIXTURES:
        first = build_source_spans(fixture)
        second = build_source_spans(fixture)
        assert [s.span_id for s in first] == [s.span_id for s in second]
        assert [s.text for s in first] == [s.text for s in second]


# ---------------------------------------------------------------------------
# A. dense multi-clause sentence -> accepted through exact span selection
# ---------------------------------------------------------------------------

def test_A_dense_sentence_accepted_via_one_cited_span():
    spans = build_source_spans(FIXTURE_A_DENSE_SENTENCE)
    assert len(spans) == 1, "no semicolon/second sentence -- must stay one span"
    candidate = EventCandidate(
        event_type=PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE,
        amount=3.0, unit="USD_BILLION", currency="USD", funded=False, committed=True,
        evidence_span_ids=(spans[0].span_id,), confidence=0.95, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.funded is False
    assert resolved.amount == 3_000_000_000.0
    # The resolved evidence is the fixture's own text, not a rewritten one.
    assert resolved.source_evidence == FIXTURE_A_DENSE_SENTENCE


# ---------------------------------------------------------------------------
# B. model paraphrases evidence field but correct span_id exists ->
#    authoritative evidence comes from the span, never the paraphrase
# ---------------------------------------------------------------------------

def test_B_authoritative_evidence_is_the_span_not_anything_the_model_wrote():
    """`EventCandidate` (Phase H.18) has NO field the model can put a
    paraphrase into at all -- `source_evidence` is not part of the model's
    JSON schema (see event_extractor.py's `_schema_text`). This test proves
    the CONSEQUENCE: even if a caller tries to hand-construct a candidate
    with a `source_evidence` string, the validator's output ignores it
    entirely and uses only the resolved span text."""
    spans = build_source_spans(FIXTURE_B_AMOUNT_IN_MIDDLE_CLAUSE)
    target = next(s for s in spans if "$1.2 billion" in s.text)
    candidate = EventCandidate(
        event_type=PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE,
        amount=1.2, unit="USD_BILLION", currency="USD", funded=False, committed=True,
        evidence_span_ids=(target.span_id,),
        # A deliberately WRONG/rewritten string in the field a pre-H.18
        # candidate would have used for evidence. It must have zero effect.
        source_evidence="totally fabricated text nowhere in the filing",
        confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.source_evidence == target.text
    assert "fabricated" not in resolved.source_evidence


# ---------------------------------------------------------------------------
# C. invented span_id -> rejected
# ---------------------------------------------------------------------------

def test_C_invented_span_id_rejected():
    spans = build_source_spans(FIXTURE_A_DENSE_SENTENCE)
    candidate = EventCandidate(
        event_type=PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE,
        amount=3.0, unit="USD_BILLION", currency="USD",
        evidence_span_ids=("totally-invented-id",), confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.EVIDENCE_NOT_IN_SOURCE


# ---------------------------------------------------------------------------
# D. correct event span but no funding witness -> funded=true rejected
# ---------------------------------------------------------------------------

def test_D_funded_true_rejected_without_a_drawdown_witness():
    spans = build_source_spans(FIXTURE_D_SEMICOLON_CHAIN)
    event_span = next(s for s in spans if "$500" in s.text)
    candidate = EventCandidate(
        event_type=PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE,
        amount=500.0, unit="USD_MILLION", currency="USD", funded=True,
        evidence_span_ids=(event_span.span_id,), confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.FUNDED_STATUS_NOT_GROUNDED


# ---------------------------------------------------------------------------
# E. event semantics in one span + amount in second span -> accepted when
#    both referenced (funding witness in a THIRD span here, section C fixture)
# ---------------------------------------------------------------------------

def test_E_event_and_funding_accepted_when_both_spans_cited():
    spans = build_source_spans(FIXTURE_C_EVENT_AND_FUNDING_SEPARATE_CLAUSES)
    assert len(spans) == 2
    event_span, funding_span = spans[0], spans[1]
    candidate = EventCandidate(
        event_type=PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE,
        amount=900.0, unit="USD_MILLION", currency="USD", funded=True,
        evidence_span_ids=(event_span.span_id, funding_span.span_id),
        confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.funded is True
    assert resolved.amount == 900_000_000.0


# ---------------------------------------------------------------------------
# F. undrawn status in a separate span -> correctly remains unfunded
# ---------------------------------------------------------------------------

def test_F_undrawn_status_in_its_own_span_keeps_candidate_unfunded():
    spans = build_source_spans(FIXTURE_D_SEMICOLON_CHAIN)
    undrawn_span = next(s for s in spans if "no amounts were drawn" in s.text)
    event_span = next(s for s in spans if "$500" in s.text)
    candidate = EventCandidate(
        event_type=PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE,
        amount=500.0, unit="USD_MILLION", currency="USD", funded=False, committed=True,
        evidence_span_ids=(event_span.span_id, undrawn_span.span_id),
        confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.funded is False
    assert resolved.committed is True


# ---------------------------------------------------------------------------
# G. semicolon-heavy SEC prose -> exact evidence preserved
# ---------------------------------------------------------------------------

def test_G_semicolon_heavy_prose_splits_into_exact_clauses():
    spans = build_source_spans(FIXTURE_D_SEMICOLON_CHAIN)
    assert len(spans) == 3
    for span in spans:
        assert FIXTURE_D_SEMICOLON_CHAIN[span.start:span.end] == span.text


# ---------------------------------------------------------------------------
# H. HTML/text normalization -> span identity remains deterministic
# ---------------------------------------------------------------------------

def test_H_span_identity_stable_across_a_normalization_boundary():
    """Collapsed double-spaces and a stray non-breaking space (exactly what
    an HTML-to-text pass leaves behind) must not change span ids between
    two identical calls, and every span must still be an exact substring of
    the (already-normalized) text handed in."""
    first = build_source_spans(FIXTURE_G_NORMALIZED_HTML_TEXT)
    second = build_source_spans(FIXTURE_G_NORMALIZED_HTML_TEXT)
    assert [s.span_id for s in first] == [s.span_id for s in second]
    for span in first:
        assert FIXTURE_G_NORMALIZED_HTML_TEXT[span.start:span.end] == span.text
    undrawn_span = next(s for s in first if "undrawn" in s.text)
    candidate = EventCandidate(
        event_type=PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE,
        amount=750.0, unit="USD_MILLION", currency="USD", funded=False, committed=True,
        evidence_span_ids=(next(s for s in first if "$750" in s.text).span_id,
                          undrawn_span.span_id),
        confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=first, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.funded is False


# ---------------------------------------------------------------------------
# I. candidate references another document's span -> rejected
# ---------------------------------------------------------------------------

def test_I_span_from_another_document_rejected():
    spans_a = build_source_spans(FIXTURE_A_DENSE_SENTENCE)
    spans_b = build_source_spans(FIXTURE_D_SEMICOLON_CHAIN)
    assert not ({s.span_id for s in spans_a} & {s.span_id for s in spans_b})
    candidate = EventCandidate(
        event_type=PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE,
        amount=3.0, unit="USD_BILLION", currency="USD",
        evidence_span_ids=(spans_b[0].span_id,), confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans_a, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.EVIDENCE_NOT_IN_SOURCE


# ---------------------------------------------------------------------------
# J. event reader emits no valid span -> fail closed
# ---------------------------------------------------------------------------

def test_J_no_span_ids_fails_closed():
    spans = build_source_spans(FIXTURE_A_DENSE_SENTENCE)
    candidate = EventCandidate(
        event_type=PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE,
        amount=3.0, unit="USD_BILLION", currency="USD",
        evidence_span_ids=(), confidence=0.9, form="8-K", items="2.03")
    validator = EventCandidateValidator(spans=spans, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.NO_EVIDENCE


# ---------------------------------------------------------------------------
# resolve_span_ids -- the retrieval primitive, tested directly
# ---------------------------------------------------------------------------

def test_resolve_span_ids_orders_by_document_position_not_citation_order():
    spans = build_source_spans(FIXTURE_D_SEMICOLON_CHAIN)
    reversed_ids = tuple(s.span_id for s in reversed(spans))
    text, unknown = resolve_span_ids(reversed_ids, spans)
    assert not unknown
    assert text == " ".join(s.text for s in spans)  # document order, not citation order


def test_resolve_span_ids_reports_every_unknown_id():
    spans = build_source_spans(FIXTURE_A_DENSE_SENTENCE)
    text, unknown = resolve_span_ids(("bogus-1", "bogus-2"), spans)
    assert text is None
    assert set(unknown) == {"bogus-1", "bogus-2"}
