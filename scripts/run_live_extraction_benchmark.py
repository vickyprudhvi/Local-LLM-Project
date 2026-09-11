"""Measure the REAL semantic reader against the extraction benchmark.

WHY THIS IS A SCRIPT AND NOT A TEST

It calls a model, once per document section, and takes minutes. A suite that
must stay fast and deterministic cannot own it. The offline suite
(`tests/test_finance_extraction_benchmark.py`) measures the deterministic
acceptance boundary by feeding the validator candidates from an oracle; that
oracle is a sentence-finder, not a reader, and its recall column measures
itself. THIS script replaces the oracle with the configured model, which is
the only way the reader's recall becomes a real number.

WHAT IT SEPARATES

The whole point is telling a READER failure from a VALIDATOR failure, because
the two have opposite fixes and only one of them is ever a reason to touch
the acceptance boundary:

    NOT_PROPOSED          the reader never saw it          -> reader
    WRONG_PROPOSAL        the reader mislabelled it        -> reader
    PROPOSED_AND_REJECTED it was refused -- by WHOSE fault
                          depends on the rejection CODE    -> see below
    PROPOSED_AND_ACCEPTED end to end correct
    DUPLICATE_PROPOSAL    same economic identity twice     -> dedup

A rejection is NOT automatically a validator failure, and treating it as one
flattered the reader while making the boundary look broken. `_attribute_
rejection` splits them by code:

    READER_WRONG_PROPOSAL   malformed candidate -- a number not in the
                            sentence, a unit contradicting the metric, an
                            unresolvable period. The boundary was right.
    READER_CORRECT_BUT_UNSUPPORTED_FORM
                            understood correctly, inexpressible in the schema.
                            Fixed by a schema change.
    VALIDATOR_FALSE_REJECTION
                            everything checkable looks right and it was
                            refused anyway. The only class that may justify a
                            boundary change, and the DEFAULT so an
                            unclassified code surfaces rather than hides.

Loosening validation because a NOT_PROPOSED item is missing would trade
precision away for nothing: the candidate was never offered, so no rule
change could have accepted it.

NONDETERMINISM

The reader is nondeterministic -- the same section has returned three
statements on one run and four on the next. A single run is a SAMPLE. Use
`--repeat N` (which forces `--no-cache`, since a cache would make every run
identical) and read the per-item 3/3, 2/3, 1/3, 0/3 stability table rather
than any one run's recall figure.

CACHING

Model responses are cached on disk, keyed by section text, model identity and
schema version. A rerun after a scoring change costs nothing, and a crash
halfway through does not throw away an hour. `--no-cache` forces fresh reads.

USAGE
    python scripts/run_live_extraction_benchmark.py [--case ID] [--no-cache]
                                                    [--json PATH]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import brain  # noqa: E402
from finance.extraction import EXTRACTOR_VERSION, SCHEMA_VERSION  # noqa: E402
from finance.extraction.schema import RejectionCode  # noqa: E402
from finance.extraction.semantic_extractor import (  # noqa: E402
    ExtractionFailure,
    LocalModelGuidanceExtractor,
    select_sections,
)
from finance.extraction.validator import GuidanceCandidateValidator  # noqa: E402
from tests.fixtures import extraction_benchmark as bench  # noqa: E402
from tools import config  # noqa: E402

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "app_data", "extraction_benchmark_cache.json")


# ---------------------------------------------------------------------------
# Outcome vocabulary (§7)
# ---------------------------------------------------------------------------

class Outcome:
    PROPOSED_AND_ACCEPTED = "PROPOSED_AND_ACCEPTED"
    PROPOSED_AND_REJECTED = "PROPOSED_AND_REJECTED"
    NOT_PROPOSED = "NOT_PROPOSED"
    WRONG_PROPOSAL = "WRONG_PROPOSAL"
    DUPLICATE_PROPOSAL = "DUPLICATE_PROPOSAL"


class Attribution:
    """Who was wrong when a proposal was refused (§9).

    The distinction the whole benchmark turns on, because it decides whether
    a validator change is even allowed. `PROPOSED_AND_REJECTED` used to mean
    "validator failure" automatically, which flattered the reader and
    slandered the boundary.
    """

    READER_WRONG_PROPOSAL = "READER_WRONG_PROPOSAL"
    VALIDATOR_FALSE_REJECTION = "VALIDATOR_FALSE_REJECTION"
    READER_CORRECT_BUT_UNSUPPORTED_FORM = "READER_CORRECT_BUT_UNSUPPORTED_FORM"


# Rejection codes that mean THE CANDIDATE WAS MALFORMED, not that the boundary
# misjudged a good one. Each names something the reader got wrong: a number
# that is not in the sentence, a unit that contradicts the metric, a period
# that cannot be resolved, a name the taxonomy does not have.
_READER_AT_FAULT_CODES = frozenset({
    "VALUE_NOT_IN_EVIDENCE",
    "EVIDENCE_NOT_IN_SOURCE",
    "NO_EVIDENCE",
    "UNKNOWN_METRIC",
    "UNIT_METRIC_MISMATCH",
    "AMBIGUOUS_TARGET_PERIOD",
    "TARGET_PERIOD_IMPLAUSIBLE",
    "ANALYST_ESTIMATE",
    "HISTORICAL_TABLE",
    "LOW_CONFIDENCE",
    # The reader named a metric the cited sentence does not describe -- a
    # component measure claimed as the consolidated one, most often. The
    # boundary refusing it is the boundary working, and §12 requires this to
    # score as the reader's error rather than the validator's.
    "METRIC_NOT_GROUNDED",
})

# Refusals that mean the reader UNDERSTOOD the sentence but could not express
# it in the schema. Kept separate because the fix is a schema change, not a
# validator change and not a prompt change. If a form becomes expressible this
# class empties out -- which is what structured tolerance did to
# TOLERANCE_NOT_DERIVABLE's predecessor, a computed endpoint refused as
# ungrounded.
_UNSUPPORTED_FORM_CODES = frozenset({
    "TOLERANCE_NOT_DERIVABLE",
    "DENOMINATOR_MISSING",
})


def _attribute_rejection(candidate, code, expected) -> Tuple[str, str]:
    """(blame, failure class) for a refused proposal that named the right key."""
    if code in _READER_AT_FAULT_CODES:
        return "reader", Attribution.READER_WRONG_PROPOSAL
    if code in _UNSUPPORTED_FORM_CODES:
        return "reader", Attribution.READER_CORRECT_BUT_UNSUPPORTED_FORM
    # Anything else: the boundary refused a candidate that looks right on
    # every axis this harness can check. That is the signal a validator
    # change may be warranted, and it is deliberately the DEFAULT so a new
    # rejection code shows up as needing a decision rather than being
    # silently absolved.
    return "validator", Attribution.VALIDATOR_FALSE_REJECTION


class FailureClass:
    SECTION_SELECTION = "SECTION_SELECTION"
    TOLERANCE_SEMANTICS = "TOLERANCE_SEMANTICS"
    MODEL_MISREAD = "MODEL_MISREAD"
    VALUE_EXTRACTION = "VALUE_EXTRACTION"
    UNIT_IDENTITY = "UNIT_IDENTITY"
    PERIOD_RESOLUTION = "PERIOD_RESOLUTION"
    BASIS_RESOLUTION = "BASIS_RESOLUTION"
    REVISION_ACTION = "REVISION_ACTION"
    EVIDENCE_GROUNDING = "EVIDENCE_GROUNDING"
    ECONOMIC_DEDUP = "ECONOMIC_DEDUP"
    VALIDATOR_REJECTION = "VALIDATOR_REJECTION"


@dataclass
class ItemResult:
    case_id: str
    metric: str
    target_period: str
    outcome: str
    failure_class: Optional[str] = None
    blame: Optional[str] = None          # "reader" | "validator" | None
    detail: str = ""


@dataclass
class CaseResult:
    case_id: str
    sections: int = 0
    proposed: int = 0
    accepted: int = 0
    rejected: int = 0
    rejection_codes: Dict[str, int] = field(default_factory=dict)
    read_failure: Optional[str] = None
    elapsed: float = 0.0
    items: List[ItemResult] = field(default_factory=list)
    critical_false_positives: List[Tuple[str, str]] = field(default_factory=list)
    unsupported_accepted: List[str] = field(default_factory=list)
    unit_errors: List[str] = field(default_factory=list)
    period_errors: List[str] = field(default_factory=list)
    basis_errors: List[str] = field(default_factory=list)
    action_errors: List[str] = field(default_factory=list)
    action_checked: int = 0
    duplicate_accepted: List[str] = field(default_factory=list)
    extra_accepted: List[str] = field(default_factory=list)
    complete_case: bool = False
    complete_accepted: int = 0
    complete_correct: int = 0
    metric_id_errors: List[str] = field(default_factory=list)
    period_id_errors: List[str] = field(default_factory=list)
    reader_wrong_proposals: List[str] = field(default_factory=list)
    validator_false_rejections: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _close(observed, expected, tolerance=0.011) -> bool:
    """Is `observed` the same number as `expected`?

    The tolerance is anchored on EXPECTED, deliberately. An earlier version
    anchored it on the observed value, which gave a +/-1.5 band on a figure
    stated as 74.9 -- wide enough that 74.5 "matched" 74.9, and wide enough to
    misreport a validator rejection as a reader mislabel. The band must be a
    property of the target, not of whatever the model happened to say.
    """
    if observed is None or expected is None:
        return observed is None and expected is None
    return abs(observed - expected) <= max(tolerance, abs(expected) * 0.02)


def _comparable_bounds(candidate):
    """A candidate's (low, high) in the units the expected data uses.

    Two conversions, both of which the validator performs downstream, so a
    scorer that skipped them would blame the reader for the validator's
    normalisation:

      * a POINT candidate carries `value` and no bounds; its range is the
        point itself
      * a PERCENT candidate is stated as 74.9 where the expected data holds
        the ratio 0.749
    """
    # Works for a raw candidate and for an accepted GuidanceMetric alike; the
    # latter is already normalised and has no `value` field.
    low, high = getattr(candidate, "low", None), getattr(candidate, "high", None)
    point = getattr(candidate, "value", None)
    if low is None and high is None and point is not None:
        low = high = point
    if low is None and high is None:
        return None
    if low is None:
        low = high
    if high is None:
        high = low
    unit = str(candidate.unit or "").upper()
    if "PERCENT" in unit:
        return (low / 100.0, high / 100.0)
    return (low, high)


def _values_match(candidate, expected) -> bool:
    """Does the candidate carry the expected numbers?

    Where the fixture pins no numbers (`low`/`high` unset) any value matches --
    the case is testing that the statement was found at all.
    """
    if expected.low is None and expected.high is None:
        return True
    bounds = _comparable_bounds(candidate)
    if bounds is None:
        return False
    return (_close(bounds[0], expected.low)
            and _close(bounds[1], expected.high))


def _carries_expected_numbers(candidate, expected) -> bool:
    """Looser than `_values_match`: did the reader look at the right sentence?

    Used to tell WRONG_PROPOSAL (right sentence, wrong label) from
    NOT_PROPOSED (never seen). Deliberately checks the raw figures in either
    scaling, because a percentage read as a ratio and a ratio read as a
    percentage carry the same digits -- exactly the confusion this has to see
    through. It does NOT widen the tolerance to do so.
    """
    if expected.low is None and expected.high is None:
        return False
    wanted = [v for v in (expected.low, expected.high) if v is not None]
    got = [v for v in (candidate.low, candidate.high, candidate.value)
           if v is not None]
    for value in wanted:
        forms = {value, value * 100.0, value / 100.0}
        if not any(any(_close(g, f) for g in got) for f in forms):
            return False
    return True


def _accepted_key(metric):
    return (getattr(metric, "name", None), getattr(metric, "fiscal_period", None))


def _candidate_key(candidate):
    return (candidate.metric_id, candidate.target_period)


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def run_case(case, extractor, verbose=True) -> CaseResult:
    text = case.document_text()
    result = CaseResult(case_id=case.case_id)
    sections = select_sections(text)
    result.sections = len(sections)

    started = time.time()
    try:
        candidates = extractor.extract(sections, issued_at=case.filed,
                                       document_id=case.case_id)
    except ExtractionFailure as failure:
        result.read_failure = failure.code
        result.elapsed = time.time() - started
        # Every expected item is NOT_PROPOSED, blamed on the reader. Scoring
        # a failed read as a validator problem would be exactly backwards.
        for expected in case.expected_guidance:
            result.items.append(ItemResult(
                case.case_id, expected.metric, expected.target_period,
                Outcome.NOT_PROPOSED, FailureClass.MODEL_MISREAD, "reader",
                f"the read did not complete ({failure.code})"))
        return result
    result.elapsed = time.time() - started
    result.proposed = len(candidates)

    validator = GuidanceCandidateValidator(document_text=text, issued_at=case.filed)
    accepted, rejected = validator.validate_all(candidates)
    result.accepted = len(accepted)
    result.rejected = len(rejected)
    for _candidate, code, _reason in rejected:
        result.rejection_codes[code] = result.rejection_codes.get(code, 0) + 1

    rejected_by_key: Dict[Tuple[str, str], List] = {}
    for candidate, code, reason in rejected:
        rejected_by_key.setdefault(_candidate_key(candidate), []).append(
            (candidate, code, reason))
    accepted_by_key = {}
    duplicates = Counter()
    for metric in accepted:
        key = _accepted_key(metric)
        duplicates[key] += 1
        accepted_by_key.setdefault(key, metric)

    # -- per expected item ------------------------------------------------
    for expected in case.expected_guidance:
        key = expected.key
        item = ItemResult(case.case_id, expected.metric, expected.target_period,
                          Outcome.NOT_PROPOSED)

        match = accepted_by_key.get(key)
        if match is not None and _values_match(match, expected):
            if duplicates[key] > 1:
                item.outcome = Outcome.DUPLICATE_PROPOSAL
                item.failure_class = FailureClass.ECONOMIC_DEDUP
                item.blame = "validator"
                item.detail = f"{duplicates[key]} accepted statements share this identity"
            else:
                item.outcome = Outcome.PROPOSED_AND_ACCEPTED
                # Attribute accuracy, scored only on items we got right --
                # a wrong unit on a statement that was never found is not a
                # unit error, it is the same miss counted twice.
                if expected.unit and match.unit != expected.unit:
                    item.failure_class = FailureClass.UNIT_IDENTITY
                    result.unit_errors.append(f"{case.case_id}:{key} "
                                              f"{match.unit} != {expected.unit}")
                if expected.basis and str(match.basis or "").lower() != expected.basis.lower():
                    result.basis_errors.append(f"{case.case_id}:{key} "
                                               f"{match.basis} != {expected.basis}")
                if expected.action:
                    result.action_checked += 1
                    got = str(getattr(match, "action", "") or "").upper()
                    if got != expected.action.upper():
                        result.action_errors.append(
                            f"{case.case_id}:{key} {got or '(none)'} != {expected.action}")
            result.items.append(item)
            continue

        refusals = rejected_by_key.get(key) or []
        correct_refusal = [r for r in refusals if _values_match(r[0], expected)]
        if correct_refusal:
            candidate, code, reason = correct_refusal[0]
            item.outcome = Outcome.PROPOSED_AND_REJECTED
            # A rejection is NOT automatically a validator failure. The
            # candidate landed on the right (metric, period) with roughly the
            # right numbers, which is what got it here -- but it may still
            # have been malformed in a way the boundary is right to refuse.
            # Blaming the boundary for those made the validator look broken
            # while the reader's real defects went uncounted, and it is the
            # exact signal that decides whether a validator change is allowed.
            item.blame, item.failure_class = _attribute_rejection(
                candidate, code, expected)
            item.detail = f"{code}: {reason}"
            result.items.append(item)
            if item.blame == "reader":
                result.reader_wrong_proposals.append(
                    f"{case.case_id}:{key} {code}")
            else:
                result.validator_false_rejections.append(
                    f"{case.case_id}:{key} {code}: {reason[:70]}")
            continue

        # Right numbers under a wrong label, accepted or refused: the reader
        # found the sentence and named it wrong.
        everything = list(candidates)
        mislabelled = [c for c in everything
                       if _candidate_key(c) != key
                       and _carries_expected_numbers(c, expected)]
        if mislabelled:
            wrong = mislabelled[0]
            item.outcome = Outcome.WRONG_PROPOSAL
            item.blame = "reader"
            if wrong.metric_id != expected.metric:
                result.metric_id_errors.append(
                    f"{case.case_id}:{key} read as {wrong.metric_id}")
            if wrong.target_period != expected.target_period:
                result.period_id_errors.append(
                    f"{case.case_id}:{key} read as {wrong.target_period}")
            if wrong.metric_id != expected.metric and wrong.target_period != expected.target_period:
                item.failure_class = FailureClass.MODEL_MISREAD
            elif wrong.metric_id != expected.metric:
                item.failure_class = FailureClass.UNIT_IDENTITY
            else:
                item.failure_class = FailureClass.PERIOD_RESOLUTION
            item.detail = (f"proposed {wrong.metric_id}@{wrong.target_period} "
                           f"({wrong.low}-{wrong.high} {wrong.unit})")
            result.items.append(item)
            continue

        # Same key, wrong numbers -> a value error, not a miss.
        same_key = [c for c in everything if _candidate_key(c) == key]
        if same_key:
            wrong = same_key[0]
            item.outcome = Outcome.WRONG_PROPOSAL
            item.failure_class = FailureClass.VALUE_EXTRACTION
            item.blame = "reader"
            item.detail = (f"proposed {wrong.low}-{wrong.high}, "
                           f"expected {expected.low}-{expected.high}")
            result.items.append(item)
            continue

        # Never proposed at all. Was the sentence even shown to the model?
        item.outcome = Outcome.NOT_PROPOSED
        item.blame = "reader"
        item.failure_class = (FailureClass.SECTION_SELECTION
                              if not _numbers_in_sections(sections, expected)
                              else FailureClass.MODEL_MISREAD)
        item.detail = ("the expected numbers are not in any selected section"
                       if item.failure_class == FailureClass.SECTION_SELECTION
                       else "the numbers were in a section the model read")
        result.items.append(item)

    # -- forbidden statements --------------------------------------------
    for metric, period in case.forbidden_guidance:
        if (metric, period) in accepted_by_key:
            result.critical_false_positives.append((metric, period))

    # -- grounding of everything accepted --------------------------------
    lowered = " ".join(text.lower().split())
    for metric in accepted:
        sentence = (getattr(metric, "source_sentence", "") or "").strip()
        if sentence and " ".join(sentence.lower().split()) not in lowered:
            result.unsupported_accepted.append(
                f"{_accepted_key(metric)}: {sentence[:60]}")

    for key, count in duplicates.items():
        if count > 1:
            result.duplicate_accepted.append(f"{key} x{count}")

    # Precision is only meaningful where the expected set is EXHAUSTIVE. On a
    # real case the expected statements are the ones verified by reading, not
    # everything the document contains, so a correct extra statement there is
    # not an error -- counting it as one would punish the reader for finding
    # something true that nobody wrote down.
    if case.expected_complete:
        result.complete_case = True
        expected_keys = {s.key for s in case.expected_guidance}
        result.complete_accepted = len(accepted_by_key)
        result.complete_correct = len(set(accepted_by_key) & expected_keys)
        for key in accepted_by_key:
            if key not in expected_keys:
                result.extra_accepted.append(f"{case.case_id}:{key}")

    if verbose:
        print(f"  {result.accepted} accepted / {result.proposed} proposed "
              f"/ {result.sections} sections / {result.elapsed:.0f}s")
    return result


def _numbers_in_sections(sections, expected) -> bool:
    """Were the expected numbers even visible to the model?

    Distinguishes SECTION_SELECTION (we never showed it) from MODEL_MISREAD
    (we showed it and the model did not report it). Blaming the reader for
    text it was never given would send the next fix to the wrong layer.
    """
    import re

    wanted = [v for v in (expected.low, expected.high) if v is not None]
    if not wanted:
        return True
    blob = " ".join(s.text for s in sections)
    numbers = [float(t.replace(",", "")) for t in
               re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", blob)]
    for value in wanted:
        forms = {value, value * 100.0, value / 100.0}
        if not any(any(abs(f - n) < 0.011 for n in numbers) for f in forms):
            return False
    return True


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(numerator, denominator) -> str:
    if not denominator:
        return "n/a"
    return f"{numerator / denominator:.0%} ({numerator}/{denominator})"


def report(results: List[CaseResult]) -> str:
    items = [i for r in results for i in r.items]
    total = len(items)
    counts = Counter(i.outcome for i in items)

    accepted_ok = counts[Outcome.PROPOSED_AND_ACCEPTED]
    proposed_ok = accepted_ok + counts[Outcome.PROPOSED_AND_REJECTED] \
        + counts[Outcome.DUPLICATE_PROPOSAL]

    total_accepted = sum(r.accepted for r in results)
    total_proposed = sum(r.proposed for r in results)
    critical = [(r.case_id, m, p) for r in results
                for m, p in r.critical_false_positives]
    unsupported = [(r.case_id, u) for r in results for u in r.unsupported_accepted]
    unit_errors = [e for r in results for e in r.unit_errors]
    basis_errors = [e for r in results for e in r.basis_errors]
    action_errors = [e for r in results for e in r.action_errors]
    action_checked = sum(r.action_checked for r in results)
    duplicates = [d for r in results for d in r.duplicate_accepted]
    extras = [e for r in results for e in r.extra_accepted]
    period_wrong = [i for i in items
                    if i.outcome == Outcome.WRONG_PROPOSAL
                    and i.failure_class == FailureClass.PERIOD_RESOLUTION]

    lines = []
    lines.append("Live V2 semantic extraction benchmark")
    lines.append("=" * 78)
    lines.append(f"model            : {config.finance_extraction_model_identity()}")
    lines.append(f"extractor/schema : {EXTRACTOR_VERSION} / {SCHEMA_VERSION}")
    lines.append(f"cases            : {len(results)}")
    lines.append(f"expected items   : {total}")
    lines.append(f"wall clock       : {sum(r.elapsed for r in results):.0f}s")
    lines.append("")

    reader_wrong = [e for r in results for e in r.reader_wrong_proposals]
    false_rejections = [e for r in results for e in r.validator_false_rejections]

    lines.append("PER-ITEM OUTCOMES")
    for name in (Outcome.PROPOSED_AND_ACCEPTED, Outcome.PROPOSED_AND_REJECTED,
                 Outcome.WRONG_PROPOSAL, Outcome.NOT_PROPOSED,
                 Outcome.DUPLICATE_PROPOSAL):
        lines.append(f"  {name:24s} {counts[name]}")
    lines.append("")

    complete_accepted = sum(r.complete_accepted for r in results if r.complete_case)
    complete_correct = sum(r.complete_correct for r in results if r.complete_case)
    complete_cases = sum(1 for r in results if r.complete_case)
    metric_errors = [e for r in results for e in r.metric_id_errors]
    period_errors = [e for r in results for e in r.period_id_errors]
    # Of the items the reader proposed AT ALL -- correctly or mislabelled --
    # how many carried the right metric / period. That is the population an
    # identity score is about; scoring it over items the reader never saw
    # would fold a recall miss into an identity error and count it twice.
    proposed_any = proposed_ok + counts[Outcome.WRONG_PROPOSAL]

    lines.append(f"  {'READER_WRONG_PROPOSAL':24s} {len(reader_wrong)}")
    lines.append(f"  {'VALIDATOR_FALSE_REJECTION':24s} {len(false_rejections)}")
    lines.append("")

    lines.append("MEASURES")
    lines.append(f"  semantic reader recall      {_pct(proposed_ok, total)}")
    lines.append(f"  validator acceptance rate   {_pct(accepted_ok, proposed_ok)}")
    lines.append(f"  end-to-end accepted recall  {_pct(accepted_ok, total)}")
    lines.append(f"  accepted precision          {_pct(complete_correct, complete_accepted)}"
                 f"   [{complete_cases} exhaustive case(s) only]")
    lines.append(f"  metric identity accuracy    {_pct(proposed_any - len(metric_errors), proposed_any)}")
    lines.append(f"  target-period accuracy      {_pct(proposed_any - len(period_errors), proposed_any)}")
    lines.append(f"  unit accuracy               {_pct(accepted_ok - len(unit_errors), accepted_ok)}")
    lines.append(f"  basis accuracy              {_pct(accepted_ok - len(basis_errors), accepted_ok)}")
    lines.append(f"  revision-action accuracy    {_pct(action_checked - len(action_errors), action_checked)}")
    lines.append(f"  economic dedup accuracy     {_pct(total_accepted - len(duplicates), total_accepted)}")
    lines.append("")
    lines.append("  NOTE precision is scored ONLY on cases whose expected set is")
    lines.append("       exhaustive. On the real cases the expected statements are the")
    lines.append("       ones verified by reading, not everything the document says, so")
    lines.append("       a correct extra statement there is not an error.")
    lines.append("")

    lines.append("HARD REQUIREMENTS (§10)")
    lines.append(f"  critical false positives    {len(critical)}")
    lines.append(f"  unsupported accepted facts  {len(unsupported)}")
    lines.append(f"  wrong-unit accepted facts   {len(unit_errors)}")
    lines.append(f"  wrong-period accepted facts {len(period_wrong)}")
    lines.append("")
    if unit_errors:
        lines.append("  unit errors:")
        for error in unit_errors:
            lines.append(f"    {error}")
    if basis_errors:
        lines.append("  basis errors:")
        for error in basis_errors:
            lines.append(f"    {error}")
    if action_errors:
        lines.append("  revision-action errors:")
        for error in action_errors:
            lines.append(f"    {error}")
    if metric_errors or period_errors:
        for error in metric_errors + period_errors:
            lines.append(f"  identity error: {error}")
    lines.append("")

    lines.append("RAW COUNTS")
    lines.append(f"  candidates proposed         {total_proposed}")
    lines.append(f"  candidates accepted         {total_accepted}")
    lines.append(f"  candidates rejected         {sum(r.rejected for r in results)}")
    codes = Counter()
    for r in results:
        codes.update(r.rejection_codes)
    for code, count in codes.most_common():
        lines.append(f"    {code:34s} {count}")
    if extras:
        lines.append(f"  accepted beyond an exhaustive expected set: {len(extras)}")
        for extra in extras:
            lines.append(f"    {extra}")
    lines.append("")

    blame = Counter(i.blame for i in items if i.blame)
    lines.append("FAILURE ATTRIBUTION (§8/§9)")
    if false_rejections:
        lines.append("  validator false rejections -- the only class that may")
        lines.append("  justify a boundary change:")
        for entry in false_rejections:
            lines.append(f"    {entry}")
    if reader_wrong:
        lines.append("  reader wrong proposals (boundary was right to refuse):")
        for entry in reader_wrong:
            lines.append(f"    {entry}")
    lines.append(f"  reader failures             {blame['reader']}")
    lines.append(f"  validator failures          {blame['validator']}")
    classes = Counter(i.failure_class for i in items if i.failure_class)
    for name, count in classes.most_common():
        lines.append(f"    {name:26s} {count}")
    lines.append("")

    problems = [i for i in items if i.outcome != Outcome.PROPOSED_AND_ACCEPTED]
    if problems:
        lines.append("EVERY MISS, NAMED")
        for i in problems:
            lines.append(f"  [{i.outcome}] {i.case_id}: {i.metric}@{i.target_period}")
            lines.append(f"      class={i.failure_class} blame={i.blame}")
            if i.detail:
                lines.append(f"      {i.detail}")
    if critical:
        lines.append("")
        lines.append("CRITICAL FALSE POSITIVES")
        for case_id, metric, period in critical:
            lines.append(f"  {case_id}: {metric}@{period}")
    if unsupported:
        lines.append("")
        lines.append("UNSUPPORTED ACCEPTED FACTS")
        for case_id, detail in unsupported:
            lines.append(f"  {case_id}: {detail}")
    read_failures = [(r.case_id, r.read_failure) for r in results if r.read_failure]
    if read_failures:
        lines.append("")
        lines.append("READ FAILURES")
        for case_id, code in read_failures:
            lines.append(f"  {case_id}: {code}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Variance across independent runs (§11/§12)
# ---------------------------------------------------------------------------

def _run_metrics(results: List[CaseResult]) -> Dict[str, float]:
    items = [i for r in results for i in r.items]
    total = len(items)
    counts = Counter(i.outcome for i in items)
    accepted_ok = counts[Outcome.PROPOSED_AND_ACCEPTED]
    proposed_ok = (accepted_ok + counts[Outcome.PROPOSED_AND_REJECTED]
                   + counts[Outcome.DUPLICATE_PROPOSAL])
    complete_accepted = sum(r.complete_accepted for r in results if r.complete_case)
    complete_correct = sum(r.complete_correct for r in results if r.complete_case)
    # Reader period mislabels, reported separately. NOT added to the accepted
    # count below: a candidate the boundary refused was never published, and
    # counting it as an accepted wrong-period fact turns the boundary doing
    # its job into a safety failure.
    period_mislabels = sum(1 for i in items
                           if i.outcome == Outcome.WRONG_PROPOSAL
                           and i.failure_class == FailureClass.PERIOD_RESOLUTION)
    return {
        "expected": total,
        "reader_recall": proposed_ok / total if total else 0.0,
        "accepted_recall": accepted_ok / total if total else 0.0,
        "precision": (complete_correct / complete_accepted
                      if complete_accepted else 1.0),
        "critical_fp": sum(len(r.critical_false_positives) for r in results),
        "unsupported": sum(len(r.unsupported_accepted) for r in results),
        "unit_errors": sum(len(r.unit_errors) for r in results),
        # ACCEPTED wrong-period facts only -- what §13 asks about.
        "period_errors": sum(len(r.period_errors) for r in results),
        "period_mislabels": period_mislabels,
        "reader_wrong": sum(len(r.reader_wrong_proposals) for r in results),
        "false_rejections": sum(len(r.validator_false_rejections) for r in results),
        "not_proposed": counts[Outcome.NOT_PROPOSED],
        "duplicates": counts[Outcome.DUPLICATE_PROPOSAL],
    }


def variance_report(all_runs: List[List[CaseResult]]) -> str:
    """What repeated runs say that one run cannot.

    A nondeterministic reader makes a single recall figure a sample. What is
    stable across runs is worth acting on; what flickers is a reliability
    problem, and averaging the two into one number hides exactly the items
    that need attention.
    """
    runs = [_run_metrics(r) for r in all_runs]
    n = len(runs)
    lines = ["Variance across %d independent runs" % n, "=" * 78]

    header = (f"{'run':>4} | {'reader recall':>14} | {'accepted recall':>16} | "
              f"{'precision':>10} | {'crit FP':>7} | {'unsup':>5} | "
              f"{'unit':>4} | {'period':>6} | {'pd mislbl':>9} | "
              f"{'rdr wrong':>9} | {'val false':>9}")
    lines.append(header)
    lines.append("-" * len(header))
    for index, run in enumerate(runs, 1):
        lines.append(
            f"{index:>4} | {run['reader_recall']:>13.0%} | "
            f"{run['accepted_recall']:>15.0%} | {run['precision']:>9.0%} | "
            f"{run['critical_fp']:>7} | {run['unsupported']:>5} | "
            f"{run['unit_errors']:>4} | {run['period_errors']:>6} | "
            f"{run['period_mislabels']:>9} | "
            f"{run['reader_wrong']:>9} | {run['false_rejections']:>9}")

    def spread(key):
        values = [r[key] for r in runs]
        return sum(values) / len(values), min(values), max(values)

    lines.append("")
    for label, key in (("semantic reader recall", "reader_recall"),
                       ("end-to-end accepted recall", "accepted_recall")):
        mean, low, high = spread(key)
        lines.append(f"  {label:28s} mean {mean:.0%}  min {low:.0%}  max {high:.0%}")

    lines.append("")
    lines.append("HARD REQUIREMENTS ACROSS ALL RUNS (§13)")
    for label, key in (("critical false positives", "critical_fp"),
                       ("unsupported accepted facts", "unsupported"),
                       ("accepted wrong-unit facts", "unit_errors"),
                       ("accepted wrong-period facts", "period_errors")):
        worst = max(r[key] for r in runs)
        lines.append(f"  {label:30s} max {worst}   "
                     + ("OK" if worst == 0 else "*** FAILED ***"))

    # -- per-item stability ------------------------------------------------
    accepted_counts: Dict[Tuple[str, str, str], int] = {}
    classes: Dict[Tuple[str, str, str], Counter] = {}
    for results in all_runs:
        for case in results:
            for item in case.items:
                key = (case.case_id, item.metric, item.target_period)
                accepted_counts.setdefault(key, 0)
                classes.setdefault(key, Counter())
                if item.outcome == Outcome.PROPOSED_AND_ACCEPTED:
                    accepted_counts[key] += 1
                elif item.failure_class:
                    classes[key][item.failure_class] += 1

    buckets = Counter(accepted_counts.values())
    lines.append("")
    lines.append(f"PER-ITEM STABILITY (§12), {len(accepted_counts)} expected items")
    for hits in range(n, -1, -1):
        lines.append(f"  accepted {hits}/{n}: {buckets.get(hits, 0)}")

    unstable = {k: v for k, v in accepted_counts.items() if 0 < v < n}
    never = {k: v for k, v in accepted_counts.items() if v == 0}
    if unstable:
        lines.append("")
        lines.append("  UNSTABLE items, by failure class:")
        for key, hits in sorted(unstable.items(), key=lambda kv: kv[1]):
            kinds = ", ".join(f"{c}x{n_}" for c, n_ in classes[key].most_common()) or "-"
            lines.append(f"    {hits}/{n}  {key[1]}@{key[2]}  [{kinds}]")
    if never:
        lines.append("")
        lines.append("  NEVER accepted, by failure class:")
        for key in sorted(never):
            kinds = ", ".join(f"{c}x{n_}" for c, n_ in classes[key].most_common()) or "-"
            lines.append(f"    0/{n}  {key[1]}@{key[2]}  [{kinds}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cache + main
# ---------------------------------------------------------------------------

def _load_cache(path, enabled):
    if not enabled or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _save_cache(path, cache):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(cache, handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", default=None)
    parser.add_argument("--repeat", type=int, default=1,
                        help="independent runs; forces --no-cache so the runs "
                             "really are independent")
    args = parser.parse_args()
    if args.repeat > 1:
        # A cache would make runs 2..N replays of run 1 and the measured
        # variance would read as exactly zero. Measuring a nondeterministic
        # reader requires real re-reads.
        args.no_cache = True

    cache_path = os.path.abspath(CACHE_PATH)
    cache = _load_cache(cache_path, not args.no_cache)
    extractor = LocalModelGuidanceExtractor(brain.ask_local_raw, cache=cache)

    cases = bench.all_cases()
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c.case_id in wanted]

    print(f"model {config.finance_extraction_model_identity()}  "
          f"budget {config.finance_extraction_max_output_tokens()} tokens  "
          f"timeout {config.finance_extraction_timeout_seconds()}s")
    print(f"{len(cases)} case(s), {len(cache)} cached section read(s)\n")

    all_runs = []
    for run_number in range(1, args.repeat + 1):
        if args.repeat > 1:
            print(f"\n{'=' * 78}\nRUN {run_number} of {args.repeat}\n{'=' * 78}")
            # A fresh extractor per run, so nothing carries over in memory
            # either. Independence is the whole point of repeating.
            extractor = LocalModelGuidanceExtractor(brain.ask_local_raw, cache={})
        results = []
        for index, case in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] {case.case_id}")
            try:
                results.append(run_case(case, extractor))
            finally:
                if not args.no_cache:
                    _save_cache(cache_path, cache)
        print("\n" + report(results))
        all_runs.append(results)

    if args.repeat > 1:
        print("\n" + variance_report(all_runs))
    results = all_runs[-1]

    if args.json:
        payload = {
            "model": config.finance_extraction_model_identity(),
            "extractor_version": EXTRACTOR_VERSION,
            "schema_version": SCHEMA_VERSION,
            "cases": [
                {"case_id": r.case_id, "sections": r.sections,
                 "proposed": r.proposed, "accepted": r.accepted,
                 "rejected": r.rejected, "rejection_codes": r.rejection_codes,
                 "read_failure": r.read_failure, "elapsed": round(r.elapsed, 1),
                 "items": [vars(i) for i in r.items],
                 "critical_false_positives": r.critical_false_positives,
                 "unsupported_accepted": r.unsupported_accepted}
                for r in results],
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
