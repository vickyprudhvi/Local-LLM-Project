"""Prospectivity is a property of the BLOCK a figure sits in.

THE INVARIANT, AND WHY IT IS NOT BEING WEAKENED

A figure may only be read as guidance when something in the document
establishes it as forward-looking. That is what stops a reported-results
table being published as management guidance -- V1's worst failure mode, and
the reason the check exists.

The V2 validator implemented only half of it. `prospective_semantics` tested
the CITED SENTENCE, and an issuer who tabulates guidance cites a table row:
"Non-GAAP diluted EPS $2.05 - $2.25." A row carries figures, never
vocabulary. So correctly-read, correctly-grounded guidance could not pass the
boundary at all, however well the model read it.

What establishes a guidance row as forward-looking is its CAPTION -- "The
following table summarizes GAAP and Non-GAAP guidance based on the current
outlook" -- which governs the rows beneath it. V1 has always modelled this
(`_OUTLOOK_BLOCK_RE` / `_REPORTED_BLOCK_RE`); V2 modelled only the
sentence-scoped half.

So this ADDS a second way to establish prospectivity, under conditions strict
enough that it cannot admit history:

  * the governing caption must itself be prospective
  * EVERY fragment between the caption and the row must be a table row --
    prose in between means the row does not belong to that caption
  * nothing marking reported results may intervene, and a reported caption
    never establishes anything

FOUND BY

A live QCOM compare run. After the section-selection fix the model proposed
all three Q4 statements correctly and grounded each in the document; all
three were refused NOT_PROSPECTIVE. That is a validator failure by the change
policy in `docs/finance_extraction_v2.md`, and a CLASS -- it costs the
guidance of every issuer who tabulates rather than narrates.
"""

import pytest

from finance import guidance as gm
from finance.extraction.schema import GuidanceCandidate, ValueType
from finance.extraction.validator import GuidanceCandidateValidator

# A captioned guidance table, flattened the way `html_to_text` leaves one.
# Not any real issuer's text -- the SHAPE is the class.
GUIDANCE_TABLE = (
    "Acme Corporation Reports Third Quarter Results. "
    "The following table summarizes GAAP and Non-GAAP guidance based on the "
    "current outlook. "
    "Current Guidance . "
    "Q4 FY2026 Estimates . "
    "Revenues $9.7B - $10.5B. "
    "GAAP diluted EPS $1.22 - $1.42. "
    "Non-GAAP diluted EPS $2.05 - $2.25. "
)

# The same shape, but the caption says these numbers already happened.
REPORTED_TABLE = (
    "Acme Corporation Reports Third Quarter Results. "
    "Condensed Consolidated Statements of Operations . "
    "Three Months Ended June 30 . "
    "Revenues $9.7B . "
    "GAAP diluted EPS $1.22 . "
    "Non-GAAP diluted EPS $2.05 . "
)


def _row(metric_id, low, high, sentence, unit="USD", period="FY2026",
         value_type=ValueType.RANGE, basis="GAAP"):
    return GuidanceCandidate(
        metric_id=metric_id, value_type=value_type, low=low, high=high,
        unit=unit, target_period=period, target_period_type="annual",
        basis=basis, prospective=True, source_sentence=sentence,
        confidence=0.95)


def _validate(candidate, document):
    return GuidanceCandidateValidator(
        document_text=document, issued_at="2026-07-29").validate(candidate)


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric_id,low,high,unit,sentence", [
    ("revenue", 9.7, 10.5, "USD_BILLION", "Revenues $9.7B - $10.5B."),
    ("earnings_per_share", 1.22, 1.42, "PER_SHARE",
     "GAAP diluted EPS $1.22 - $1.42."),
    ("adjusted_earnings_per_share", 2.05, 2.25, "PER_SHARE",
     "Non-GAAP diluted EPS $2.05 - $2.25."),
])
def test_a_row_under_a_guidance_caption_is_prospective(metric_id, low, high,
                                                       unit, sentence):
    metric, code, reason = _validate(
        _row(metric_id, low, high, sentence, unit=unit), GUIDANCE_TABLE)
    assert metric is not None, f"{code}: {reason}"
    assert metric.name == metric_id


def test_the_caption_is_recorded_as_the_evidence_that_qualified_it():
    """The reason must name the caption, not the row.

    A row is not self-qualifying, so recording the row as its own prospective
    evidence would assert something the document does not say.
    """
    metric, _code, _reason = _validate(
        _row("revenue", 9.7, 10.5, "Revenues $9.7B - $10.5B."), GUIDANCE_TABLE)
    assert metric is not None
    evidence = (metric.prospective_evidence or "").lower()
    # The NEAREST caption governing the row, which here is the table's own
    # "Current Guidance" label rather than the fuller sentence above it.
    # Either is a caption; what matters is that the row is not recorded as
    # qualifying itself.
    assert "guidance" in evidence or "outlook" in evidence
    assert "9.7" not in evidence, "the row was recorded as its own evidence"


# ---------------------------------------------------------------------------
# ...and every way it must still refuse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric_id,value,unit,sentence", [
    ("revenue", 9.7, "USD_BILLION", "Revenues $9.7B ."),
    ("earnings_per_share", 1.22, "PER_SHARE", "GAAP diluted EPS $1.22 ."),
    ("adjusted_earnings_per_share", 2.05, "PER_SHARE",
     "Non-GAAP diluted EPS $2.05 ."),
])
def test_the_same_row_under_a_REPORTED_caption_is_still_refused(
        metric_id, value, unit, sentence):
    """The failure this whole check exists to prevent.

    Identical row shape, identical absence of vocabulary. Only the caption
    differs, and it is the caption that decides.
    """
    metric, code, _reason = _validate(
        _row(metric_id, value, value, sentence, unit=unit,
             value_type=ValueType.POINT),
        REPORTED_TABLE)
    assert metric is None, "a reported-results row was published as guidance"
    assert code


def test_prose_between_the_caption_and_the_row_breaks_the_block():
    """A row only belongs to a caption when nothing prose-like intervenes.

    Otherwise any figure anywhere below any guidance sentence inherits its
    prospectivity, which is the "proximity is not qualification" failure
    `finance/guidance.py` documents at length.
    """
    document = (
        "The following table summarizes guidance based on the current outlook. "
        "Current Guidance . "
        "Revenues $9.7B - $10.5B. "
        "Separately, the company completed the acquisition of a subsidiary "
        "during the quarter and recognised a gain on the transaction, which "
        "is reflected in the reported results for the period just ended. "
        "Legacy segment contribution $3.3B - $3.7B. ")
    metric, code, _reason = _validate(
        _row("revenue", 3.3, 3.7, "Legacy segment contribution $3.3B - $3.7B."),
        document)
    assert metric is None
    assert code


def test_a_row_with_no_caption_at_all_is_refused():
    document = ("Acme Corporation Reports Results. "
                "Revenues $9.7B - $10.5B. "
                "GAAP diluted EPS $1.22 - $1.42. ")
    metric, code, _reason = _validate(
        _row("revenue", 9.7, 10.5, "Revenues $9.7B - $10.5B."), document)
    assert metric is None
    assert code


def test_an_analyst_caption_never_qualifies_a_row():
    """A caption establishes WHOSE claim it is as well as when."""
    document = ("The following table summarizes the outlook according to a "
                "survey of analysts covering the company. "
                "Consensus estimates . "
                "Revenues $9.7B - $10.5B. ")
    metric, code, _reason = _validate(
        _row("revenue", 9.7, 10.5, "Revenues $9.7B - $10.5B."), document)
    assert metric is None
    assert code


def test_a_sentence_that_qualifies_itself_still_does():
    """No regression on the path that already worked."""
    document = ("Outlook. For the full year 2026, the company expects revenue "
                "of $9.7 billion to $10.5 billion. ")
    metric, code, reason = _validate(
        _row("revenue", 9.7, 10.5,
             "For the full year 2026, the company expects revenue of $9.7 "
             "billion to $10.5 billion."),
        document)
    assert metric is not None, f"{code}: {reason}"


def test_an_ungrounded_row_is_still_refused_before_any_of_this():
    """Block scope must not become a way in for a sentence not in the document."""
    metric, code, _reason = _validate(
        _row("revenue", 9.7, 10.5, "Revenues $9.9B - $10.9B."), GUIDANCE_TABLE)
    assert metric is None
    assert code


def test_a_candidate_the_model_marked_not_prospective_is_still_refused():
    candidate = GuidanceCandidate(
        metric_id="revenue", value_type=ValueType.RANGE, low=9.7, high=10.5,
        unit="USD", target_period="FY2026", target_period_type="annual",
        prospective=False, source_sentence="Revenues $9.7B - $10.5B.",
        confidence=0.95)
    metric, code, _reason = _validate(candidate, GUIDANCE_TABLE)
    assert metric is None
    assert code


def test_the_block_lookup_needs_the_document():
    """With no document supplied there is no block, so the row cannot pass.

    Fail closed: a validator constructed without text must not become the
    permissive path.
    """
    metric, code, _reason = GuidanceCandidateValidator(
        document_text="", issued_at="2026-07-29").validate(
            _row("revenue", 9.7, 10.5, "Revenues $9.7B - $10.5B."))
    assert metric is None
    assert code
