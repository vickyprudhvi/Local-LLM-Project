"""Who gets blamed when the boundary refuses a proposal.

`PROPOSED_AND_REJECTED` used to mean `VALIDATOR_FAILURE` automatically. That
is wrong in both directions: it flattered the reader, whose malformed
candidates were counted as the boundary's fault, and it slandered the
boundary, which was refusing them correctly.

It also broke the one signal the change policy depends on. A validator change
is permitted only where the reader proposed something semantically correct and
grounded and the boundary refused it anyway. If every rejection reads as a
validator failure, that signal is noise and the policy cannot be applied.

So a refusal is attributed by its CODE:

  READER_WRONG_PROPOSAL   the candidate was malformed -- a number not in the
                          sentence, a unit contradicting the metric, a period
                          that will not resolve. The boundary was right.
  READER_CORRECT_BUT_UNSUPPORTED_FORM
                          the reader understood the sentence and the schema
                          could not carry it. Fixed by a schema change, not by
                          loosening anything.
  VALIDATOR_FALSE_REJECTION
                          everything checkable looks right and it was refused
                          anyway. The DEFAULT, so an unrecognised code surfaces
                          as needing a decision rather than being absolved.
"""

import importlib.util
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "live_extraction_benchmark",
    os.path.join(_ROOT, "scripts", "run_live_extraction_benchmark.py"))
bench_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench_script)

from finance.extraction.schema import RejectionCode  # noqa: E402
from tests.fixtures.extraction_benchmark import ExpectedStatement  # noqa: E402

EXPECTED = ExpectedStatement("revenue", "FY2027", "currency", 91.0, 91.0)


def _attribute(code):
    return bench_script._attribute_rejection(None, code, EXPECTED)  # noqa: SLF001


# ---------------------------------------------------------------------------
# J. a wrong proposal correctly refused is the READER's failure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code", [
    RejectionCode.VALUE_NOT_IN_EVIDENCE,     # computed endpoint as reported
    RejectionCode.UNIT_METRIC_MISMATCH,      # wrong unit
    RejectionCode.AMBIGUOUS_TARGET_PERIOD,   # wrong/unresolvable period
    RejectionCode.UNKNOWN_METRIC,            # unsupported metric identity
    RejectionCode.EVIDENCE_NOT_IN_SOURCE,    # unsupported evidence
    RejectionCode.ANALYST_ESTIMATE,
    RejectionCode.HISTORICAL_TABLE,
    # §12: the reader named a metric the evidence does not describe. The
    # boundary refusing it is the boundary working.
    RejectionCode.METRIC_NOT_GROUNDED,
])
def test_a_malformed_candidate_is_a_reader_wrong_proposal(code):
    blame, attribution = _attribute(code)
    assert blame == "reader"
    assert attribution == bench_script.Attribution.READER_WRONG_PROPOSAL


def test_the_computed_endpoint_case_specifically():
    """The refusal that started this: "89.18 does not appear in the sentence".

    The boundary was right every time it said that. Counting it as a
    validator failure is what made the acceptance rate look like a boundary
    problem when it was a reader-and-schema problem.
    """
    blame, attribution = _attribute(RejectionCode.VALUE_NOT_IN_EVIDENCE)
    assert (blame, attribution) == (
        "reader", bench_script.Attribution.READER_WRONG_PROPOSAL)


# ---------------------------------------------------------------------------
# K. a correct grounded proposal wrongly refused is the VALIDATOR's failure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code", [
    RejectionCode.NOT_PROSPECTIVE,
    RejectionCode.DENOMINATOR_NOT_IN_EVIDENCE,
    RejectionCode.INCOMPATIBLE_BASIS,
])
def test_a_semantic_refusal_is_a_validator_false_rejection(code):
    blame, attribution = _attribute(code)
    assert blame == "validator"
    assert attribution == bench_script.Attribution.VALIDATOR_FALSE_REJECTION


def test_an_unrecognised_code_defaults_to_needing_a_decision():
    """Fail loud, not silent.

    A new rejection code must show up as a validator false rejection so
    somebody looks at it, rather than being quietly absolved into the reader's
    column and disappearing.
    """
    blame, attribution = _attribute("SOME_NEW_CODE_NOBODY_CLASSIFIED")
    assert blame == "validator"
    assert attribution == bench_script.Attribution.VALIDATOR_FALSE_REJECTION


# ---------------------------------------------------------------------------
# The schema-shaped middle class
# ---------------------------------------------------------------------------

def test_an_inexpressible_form_is_neither_a_misread_nor_a_bad_boundary():
    blame, attribution = _attribute(RejectionCode.TOLERANCE_NOT_DERIVABLE)
    assert blame == "reader"
    assert attribution == (
        bench_script.Attribution.READER_CORRECT_BUT_UNSUPPORTED_FORM)


def test_every_rejection_code_is_classified_somewhere():
    """No code may fall through unnoticed as the vocabulary grows."""
    codes = [v for k, v in vars(RejectionCode).items()
             if not k.startswith("_") and isinstance(v, str)]
    assert codes
    for code in codes:
        blame, attribution = _attribute(code)
        assert blame in ("reader", "validator")
        assert attribution in (
            bench_script.Attribution.READER_WRONG_PROPOSAL,
            bench_script.Attribution.VALIDATOR_FALSE_REJECTION,
            bench_script.Attribution.READER_CORRECT_BUT_UNSUPPORTED_FORM)


def test_reader_and_validator_columns_stay_separate():
    """§9: the attribution must remain explicit, not merged into one number."""
    reader, _ = _attribute(RejectionCode.VALUE_NOT_IN_EVIDENCE)
    validator, _ = _attribute(RejectionCode.NOT_PROSPECTIVE)
    assert reader != validator
