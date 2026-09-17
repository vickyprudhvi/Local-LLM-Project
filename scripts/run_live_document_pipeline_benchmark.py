"""Live-model benchmark for `finance/documents/` (Phase H.17).

Drives the REAL configured semantic model (via `brain.ask_local_raw`,
registered through `finance.documents.runtime` exactly as `assistant.py`
does at import) against the golden document-package benchmark's cases that
carry real document text -- the actual-table-fallback cases and the
financing-event cases. Nothing here scripts or oracles the model's output;
`LocalModelActualsExtractor`/`LocalModelEventExtractor` are the same classes
`finance/documents/runtime.py` constructs in production.

Every expected item is classified into exactly one of:

    PROPOSED_AND_ACCEPTED   a candidate matching ground truth was proposed
                            and the deterministic validator accepted it
    PROPOSED_AND_REJECTED   a candidate for this item was proposed but the
                            validator refused it
    NOT_PROPOSED            the model never proposed anything for this item
    WRONG_PROPOSAL          a candidate was proposed AND accepted, but its
                            value disagrees with the hand-verified ground
                            truth -- this is the hard-safety-relevant case
    DUPLICATE_PROPOSAL      more than one raw candidate was proposed for the
                            same underlying item

Run three independent times (fresh extractor + empty cache each run, so a
model call actually happens three times, not once with two cache hits) via
`--runs 3` (the default), matching the nondeterminism-measurement
requirement. Mirrors `scripts/run_live_extraction_benchmark.py`'s role for
the guidance layer -- a live measurement tool, not a pytest test, because it
makes real model calls and its numbers are a sample, not a fixed assertion.
"""

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, ".")

from brain import ask_local_raw, LOCAL_MODEL                                  # noqa: E402
from finance.documents import DOCUMENT_PIPELINE_VERSION, SCHEMA_VERSION       # noqa: E402
from finance.documents.actuals_bridge import _period_ends_by_label            # noqa: E402
from finance.documents.actuals_extractor import (                             # noqa: E402
    LocalModelActualsExtractor,
    select_actual_sections,
)
from finance.documents.actuals_schema import ActualRejectionCode              # noqa: E402
from finance.documents.actuals_validator import ActualFactCandidateValidator  # noqa: E402
from finance.documents.event_extractor import (                              # noqa: E402
    LocalModelEventExtractor,
    select_event_sections,
)
from finance.documents.event_schema import EventRejectionCode                 # noqa: E402
from finance.documents.event_validator import EventCandidateValidator         # noqa: E402
from finance.reported_actuals.tables import document_scale, parse_filing_tables  # noqa: E402
from finance.structural_breaks import PostBalanceSheetEventType               # noqa: E402
from tests.fixtures.document_package_benchmark import all_cases               # noqa: E402


class Outcome:
    PROPOSED_AND_ACCEPTED = "PROPOSED_AND_ACCEPTED"
    PROPOSED_AND_REJECTED = "PROPOSED_AND_REJECTED"
    NOT_PROPOSED = "NOT_PROPOSED"
    WRONG_PROPOSAL = "WRONG_PROPOSAL"
    DUPLICATE_PROPOSAL = "DUPLICATE_PROPOSAL"


class FailureClass:
    """Section 7's fixed, generalized taxonomy. No ticker-specific classes."""

    DOCUMENT_SELECTION = "DOCUMENT_SELECTION"
    SECTION_SELECTION = "SECTION_SELECTION"
    MODEL_MISREAD = "MODEL_MISREAD"
    VALUE_EXTRACTION = "VALUE_EXTRACTION"
    METRIC_IDENTITY = "METRIC_IDENTITY"
    METRIC_SCOPE = "METRIC_SCOPE"
    UNIT_RESOLUTION = "UNIT_RESOLUTION"
    CURRENCY_RESOLUTION = "CURRENCY_RESOLUTION"
    PERIOD_RESOLUTION = "PERIOD_RESOLUTION"
    BASIS_RESOLUTION = "BASIS_RESOLUTION"
    EVIDENCE_GROUNDING = "EVIDENCE_GROUNDING"
    EVENT_CLASSIFICATION = "EVENT_CLASSIFICATION"
    FUNDED_STATUS = "FUNDED_STATUS"
    ACTUAL_GUIDANCE_SEPARATION = "ACTUAL_GUIDANCE_SEPARATION"
    VALIDATOR_FALSE_REJECTION = "VALIDATOR_FALSE_REJECTION"
    SHARED_XBRL_COVERAGE = "SHARED_XBRL_COVERAGE"
    OTHER_GENERALIZED = "OTHER_GENERALIZED"


_REFINEMENTS = {
    PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE: {
        PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE,
        PostBalanceSheetEventType.CONVERTIBLE_ISSUANCE,
        PostBalanceSheetEventType.DEBT_FINANCED_REPURCHASE,
        PostBalanceSheetEventType.REFINANCING,
    },
    PostBalanceSheetEventType.ISSUER_EQUITY_ISSUANCE: {
        PostBalanceSheetEventType.ISSUER_EQUITY_ISSUANCE,
        PostBalanceSheetEventType.CONVERTIBLE_ISSUANCE,
        PostBalanceSheetEventType.WARRANT_ISSUANCE,
    },
}


@dataclass
class ItemResult:
    case_id: str
    item_key: str
    outcome: str
    detail: str = ""
    failure_class: Optional[str] = None
    accepted_value: Optional[object] = None
    rejection_code: Optional[str] = None


@dataclass
class RunResult:
    run_index: int
    actual_items: List[ItemResult] = field(default_factory=list)
    event_items: List[ItemResult] = field(default_factory=list)
    # Items whose ground truth is `should_be_accepted=False` (a negative
    # control -- an outlook figure adjacent to a reported table, say).
    # Tracked SEPARATELY from `actual_items`/`event_items`: these measure
    # correct-refusal behavior, not reader recall, and mixing them into the
    # recall denominator would make a hard-safety pass look like a recall
    # miss.
    negative_control_items: List[ItemResult] = field(default_factory=list)
    fallback_activation: Dict[str, bool] = field(default_factory=dict)
    hard_safety_hits: Dict[str, int] = field(default_factory=dict)
    call_seconds: float = 0.0


def _record_hard_safety(run: RunResult, code: str) -> None:
    run.hard_safety_hits[code] = run.hard_safety_hits.get(code, 0) + 1


def _close_enough(a: float, b: float, rel: float = 0.01) -> bool:
    return abs(a - b) <= max(abs(b) * rel, 1.0)


def _extract_with_retry(label: str, fn, *args, retries: int = 2, **kwargs):
    """A transient network/timeout failure must degrade ONE item, not the
    whole multi-hour run. A live model call over a real network genuinely
    times out sometimes (measured: a `requests.exceptions.ReadTimeout` at
    read timeout=560 killed an entire in-progress run outright before this
    existed). Retried a bounded number of times with a short backoff, then
    surfaces as `None` so the caller can record NOT_PROPOSED and move on --
    never silently invented, never a crash."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:                                  # noqa: BLE001
            last_error = exc
            print(f"    [WARN] {label} attempt {attempt + 1}/{retries + 1} failed: "
                 f"{type(exc).__name__}: {exc}", flush=True)
            if attempt < retries:
                time.sleep(5)
    print(f"    [WARN] {label} exhausted retries: {traceback.format_exception_only(type(last_error), last_error)}",
         flush=True)
    return None


# ---------------------------------------------------------------------------
# Actual-table fallback
# ---------------------------------------------------------------------------

def _score_actuals_case(case, run: RunResult) -> None:
    if not case.actual_document_text:
        return

    should_trigger = bool(case.expected_fallback_triggers)
    activated = should_trigger   # this script drives the extractor directly
                                 # rather than through actuals_bridge, so it
                                 # records what the FIXTURE says should happen
                                 # and separately re-derives it below.
    from finance.documents.actuals_bridge import should_attempt_fallback
    really_should, reason = should_attempt_fallback(
        case.target_period_end, case.structured_completeness, ())
    run.fallback_activation[case.case_id] = really_should
    if really_should != should_trigger:
        run.actual_items.append(ItemResult(
            case.case_id, "FALLBACK_GATE", Outcome.WRONG_PROPOSAL,
            detail=f"expected trigger={should_trigger}, gate returned {really_should} ({reason})",
            failure_class=FailureClass.OTHER_GENERALIZED))
        return

    if not really_should:
        # An "unnecessary activation" positive control: confirm NOTHING is
        # proposed and the model is never invoked for this case.
        return

    tables = parse_filing_tables(case.actual_document_text,
                                 document_scale_hint=document_scale(
                                     case.actual_document_text))
    sections = select_actual_sections(tables)
    if not sections:
        for expected in case.expected_actual_facts:
            if expected.should_be_accepted:
                run.actual_items.append(ItemResult(
                    case.case_id, f"{expected.metric_id}@{expected.period_end}",
                    Outcome.NOT_PROPOSED, detail="no reported-statement section was selected",
                    failure_class=FailureClass.SECTION_SELECTION))
        return

    extractor = LocalModelActualsExtractor(ask_local_raw)
    started = time.monotonic()
    raw_candidates = _extract_with_retry(
        f"actuals/{case.case_id}", extractor.extract, sections,
        issued_at=case.target_period_end, document_id=case.actual_document_accession,
        form="8-K")
    run.call_seconds += time.monotonic() - started
    if raw_candidates is None:
        for expected in case.expected_actual_facts:
            if expected.should_be_accepted:
                run.actual_items.append(ItemResult(
                    case.case_id, f"{expected.metric_id}@{expected.period_end}",
                    Outcome.NOT_PROPOSED, detail="model call failed after retries",
                    failure_class=FailureClass.OTHER_GENERALIZED))
        return

    grid_text = "\n\n".join(s.text for s in sections)
    period_ends = _period_ends_by_label(sections, tables)
    validator = ActualFactCandidateValidator(
        grid_text=grid_text, period_ends_by_label=period_ends, as_of=case.target_period_end,
        accession=case.actual_document_accession, form="8-K", filed=case.target_period_end)

    accepted, rejected = validator.validate_all(raw_candidates)

    # Duplicate detection at the RAW proposal stage: >1 raw candidate naming
    # the same (metric, resolved period_end).
    raw_keys: Dict[Tuple[str, Optional[str]], int] = {}
    for candidate in raw_candidates:
        resolved_end = period_ends.get(candidate.period_label or "")
        raw_keys[(candidate.metric_id, resolved_end)] = \
            raw_keys.get((candidate.metric_id, resolved_end), 0) + 1

    accepted_by_key = {(f.field, f.period_end): f for f in accepted}
    rejected_by_key: Dict[Tuple[str, Optional[str]], Tuple[str, str]] = {}
    for candidate, code, reason in rejected:
        resolved_end = period_ends.get(candidate.period_label or "")
        rejected_by_key[(candidate.metric_id, resolved_end)] = (code, reason)

    for expected in case.expected_actual_facts:
        key = (expected.metric_id, expected.period_end)
        item_key = f"{expected.metric_id}@{expected.period_end}"
        duplicate = raw_keys.get(key, 0) > 1
        # Negative controls (`should_be_accepted=False`) go to their own
        # list: they measure correct-refusal behavior, not reader recall,
        # and folding a correct non-proposal into the recall denominator
        # would make a hard-safety PASS look like a recall MISS.
        target = run.actual_items if expected.should_be_accepted else run.negative_control_items

        if key in accepted_by_key:
            fact = accepted_by_key[key]
            if _close_enough(fact.value, expected.value) and expected.should_be_accepted:
                outcome = Outcome.DUPLICATE_PROPOSAL if duplicate else Outcome.PROPOSED_AND_ACCEPTED
                target.append(ItemResult(case.case_id, item_key, outcome,
                                         accepted_value=fact.value))
            elif not expected.should_be_accepted:
                _record_hard_safety(run, "GUIDANCE_AS_ACTUAL")
                target.append(ItemResult(
                    case.case_id, item_key, Outcome.WRONG_PROPOSAL,
                    detail=f"accepted a negative-control value: {fact.value}",
                    accepted_value=fact.value,
                    failure_class=FailureClass.ACTUAL_GUIDANCE_SEPARATION))
            else:
                _record_hard_safety(run, "UNSUPPORTED_ACCEPTED_FACT")
                target.append(ItemResult(
                    case.case_id, item_key, Outcome.WRONG_PROPOSAL,
                    detail=f"accepted {fact.value}, expected {expected.value}",
                    accepted_value=fact.value, failure_class=FailureClass.VALUE_EXTRACTION))
            continue

        if key in rejected_by_key:
            code, reason = rejected_by_key[key]
            if expected.should_be_accepted:
                fclass = (FailureClass.VALIDATOR_FALSE_REJECTION
                         if code not in (ActualRejectionCode.LOW_CONFIDENCE,)
                         else FailureClass.OTHER_GENERALIZED)
                target.append(ItemResult(
                    case.case_id, item_key, Outcome.PROPOSED_AND_REJECTED,
                    detail=f"{code}: {reason}", rejection_code=code, failure_class=fclass))
            else:
                # Correctly rejected an item that should never have been
                # accepted -- the validator did its job.
                target.append(ItemResult(
                    case.case_id, item_key, Outcome.PROPOSED_AND_REJECTED,
                    detail=f"correctly refused: {code}: {reason}", rejection_code=code))
            continue

        if expected.should_be_accepted:
            target.append(ItemResult(
                case.case_id, item_key, Outcome.NOT_PROPOSED,
                failure_class=FailureClass.MODEL_MISREAD))
        else:
            # Correctly never proposed -- the best outcome for a
            # should-be-refused item (e.g. a nearby guidance figure). Still
            # explicitly recorded, so a hard-safety negative control that
            # passed is visible rather than silently absent -- section 10's
            # "do not accept 0 failures from an unexercised check" applies to
            # the REPORT, not only to the check itself.
            target.append(ItemResult(
                case.case_id, item_key, Outcome.NOT_PROPOSED,
                detail="correctly never proposed -- negative control verified"))


# ---------------------------------------------------------------------------
# Financing events
# ---------------------------------------------------------------------------

def _score_event_case(case, run: RunResult) -> None:
    for expected in case.expected_events:
        text = case.event_documents.get(expected.accession, "")
        if not text:
            continue
        items = _items_for(case, expected.accession)
        sections = select_event_sections(text)
        extractor = LocalModelEventExtractor(ask_local_raw)
        started = time.monotonic()
        raw_candidates = _extract_with_retry(
            f"event/{case.case_id}", extractor.extract, sections, items=items,
            filed=None, accession=expected.accession, document_id=None)
        run.call_seconds += time.monotonic() - started
        if raw_candidates is None:
            run.event_items.append(ItemResult(
                case.case_id, expected.accession, Outcome.NOT_PROPOSED,
                detail="model call failed after retries",
                failure_class=FailureClass.OTHER_GENERALIZED))
            continue

        all_spans = tuple(span for section in sections for span in section.spans)
        validator = EventCandidateValidator(spans=all_spans, items=items, form="8-K")
        accepted, rejected = validator.validate_all(raw_candidates)

        item_key = f"{expected.accession}"
        duplicate = len(raw_candidates) > 1

        def _type_matches(proposed: str, expected_type: str) -> bool:
            if proposed == expected_type:
                return True
            return expected_type in _REFINEMENTS.get(proposed, set()) \
                or proposed in _REFINEMENTS.get(expected_type, set())

        matching_accepted = [a for a in accepted if _type_matches(a.event_type, expected.event_type)]

        if matching_accepted:
            resolved = matching_accepted[0]
            funded_ok = resolved.funded == bool(expected.funded)
            amount_ok = (expected.amount is None or resolved.amount is None
                        or _close_enough(resolved.amount, expected.amount))
            if not funded_ok:
                _record_hard_safety(
                    run, "UNDRAWN_FACILITY_COUNTED_AS_FUNDED_DEBT"
                    if resolved.funded and not expected.funded
                    else "FUNDED_STATUS_MISREAD")
            if funded_ok and amount_ok:
                outcome = Outcome.DUPLICATE_PROPOSAL if duplicate else Outcome.PROPOSED_AND_ACCEPTED
                run.event_items.append(ItemResult(
                    case.case_id, item_key, outcome,
                    accepted_value={"event_type": resolved.event_type,
                                    "funded": resolved.funded, "amount": resolved.amount}))
            else:
                run.event_items.append(ItemResult(
                    case.case_id, item_key, Outcome.WRONG_PROPOSAL,
                    detail=f"funded={resolved.funded} amount={resolved.amount}, expected "
                           f"funded={expected.funded} amount={expected.amount}",
                    accepted_value={"event_type": resolved.event_type,
                                    "funded": resolved.funded, "amount": resolved.amount},
                    failure_class=(FailureClass.FUNDED_STATUS if not funded_ok
                                  else FailureClass.VALUE_EXTRACTION)))
            continue

        if accepted:
            # Something was accepted, but not of a compatible type.
            run.event_items.append(ItemResult(
                case.case_id, item_key, Outcome.WRONG_PROPOSAL,
                detail=f"accepted {[a.event_type for a in accepted]}, expected "
                       f"{expected.event_type}",
                failure_class=FailureClass.EVENT_CLASSIFICATION))
            continue

        if rejected:
            codes = [code for _c, code, _r in rejected]
            fclass = (FailureClass.VALIDATOR_FALSE_REJECTION
                     if expected.should_be_accepted else None)
            run.event_items.append(ItemResult(
                case.case_id, item_key, Outcome.PROPOSED_AND_REJECTED,
                detail=str(codes), rejection_code=(codes[0] if codes else None),
                failure_class=fclass))
            continue

        run.event_items.append(ItemResult(
            case.case_id, item_key, Outcome.NOT_PROPOSED,
            failure_class=FailureClass.MODEL_MISREAD))


def _items_for(case, accession: str) -> str:
    recent = case.submissions["filings"]["recent"]
    for index, acc in enumerate(recent["accessionNumber"]):
        if acc == accession:
            return recent["items"][index]
    return ""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_once(run_index: int, cases) -> RunResult:
    run = RunResult(run_index=run_index)
    for case in cases:
        # A case-level try/except is a second, coarser safety net beyond
        # `_extract_with_retry`'s per-call one: an unanticipated exception
        # anywhere in scoring degrades ONE case's measurement for this run,
        # never the whole multi-run benchmark.
        if case.actual_document_text:
            print(f"  [run {run_index}] actuals case: {case.case_id}", flush=True)
            try:
                _score_actuals_case(case, run)
            except Exception as exc:                              # noqa: BLE001
                print(f"    [ERROR] {case.case_id} actuals scoring raised "
                     f"{type(exc).__name__}: {exc} -- skipped for this run", flush=True)
        if case.expected_events and any(case.event_documents.values()):
            print(f"  [run {run_index}] event case: {case.case_id}", flush=True)
            try:
                _score_event_case(case, run)
            except Exception as exc:                              # noqa: BLE001
                print(f"    [ERROR] {case.case_id} event scoring raised "
                     f"{type(exc).__name__}: {exc} -- skipped for this run", flush=True)
    return run


def _rate(items: List[ItemResult], outcome: str) -> Tuple[int, int]:
    relevant = [i for i in items]
    hit = sum(1 for i in relevant if i.outcome == outcome)
    return hit, len(relevant)


def summarize(runs: List[RunResult]) -> dict:
    def recall(get_items):
        values = []
        for run in runs:
            items = get_items(run)
            total = len(items)
            proposed = sum(1 for i in items if i.outcome in (
                Outcome.PROPOSED_AND_ACCEPTED, Outcome.PROPOSED_AND_REJECTED,
                Outcome.WRONG_PROPOSAL, Outcome.DUPLICATE_PROPOSAL))
            values.append((proposed / total) if total else 1.0)
        return values

    def accepted_recall(get_items):
        values = []
        for run in runs:
            items = [i for i in get_items(run)]
            total = len(items)
            accepted = sum(1 for i in items if i.outcome in (
                Outcome.PROPOSED_AND_ACCEPTED, Outcome.DUPLICATE_PROPOSAL))
            values.append((accepted / total) if total else 1.0)
        return values

    def accepted_precision(get_items):
        values = []
        for run in runs:
            items = [i for i in get_items(run)]
            accepted_total = sum(1 for i in items if i.outcome in (
                Outcome.PROPOSED_AND_ACCEPTED, Outcome.DUPLICATE_PROPOSAL, Outcome.WRONG_PROPOSAL))
            correct = sum(1 for i in items if i.outcome in (
                Outcome.PROPOSED_AND_ACCEPTED, Outcome.DUPLICATE_PROPOSAL))
            values.append((correct / accepted_total) if accepted_total else 1.0)
        return values

    actual_recall = recall(lambda r: r.actual_items)
    actual_accepted_recall = accepted_recall(lambda r: r.actual_items)
    actual_accepted_precision = accepted_precision(lambda r: r.actual_items)
    event_recall = recall(lambda r: r.event_items)
    event_accepted_recall = accepted_recall(lambda r: r.event_items)
    event_accepted_precision = accepted_precision(lambda r: r.event_items)

    hard_safety: Dict[str, int] = {}
    for run in runs:
        for code, count in run.hard_safety_hits.items():
            hard_safety[code] = hard_safety.get(code, 0) + count

    stability: Dict[str, List[str]] = {}
    for run in runs:
        for item in run.actual_items + run.event_items:
            key = f"{item.case_id}/{item.item_key}"
            stability.setdefault(key, []).append(item.outcome)

    stability_buckets = {"3/3": 0, "2/3": 0, "1/3": 0, "0/3": 0}
    for key, outcomes in stability.items():
        accepted_count = sum(1 for o in outcomes if o in (
            Outcome.PROPOSED_AND_ACCEPTED, Outcome.DUPLICATE_PROPOSAL))
        bucket = f"{accepted_count}/{len(runs)}" if len(runs) == 3 else str(accepted_count)
        stability_buckets[bucket] = stability_buckets.get(bucket, 0) + 1

    # Negative controls: correct-refusal rate, reported separately from
    # recall/precision (see `RunResult.negative_control_items`'s docstring).
    negative_control_pass = 0
    negative_control_total = 0
    negative_control_detail: Dict[str, List[str]] = {}
    for run in runs:
        for item in run.negative_control_items:
            negative_control_total += 1
            passed = item.outcome in (Outcome.NOT_PROPOSED, Outcome.PROPOSED_AND_REJECTED)
            negative_control_pass += int(passed)
            key = f"{item.case_id}/{item.item_key}"
            negative_control_detail.setdefault(key, []).append(
                f"{item.outcome}{'' if passed else ' <-- FAILED'}")

    return {
        "model": LOCAL_MODEL,
        "document_pipeline_version": DOCUMENT_PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "actual_reader_recall": actual_recall,
        "actual_accepted_recall": actual_accepted_recall,
        "actual_accepted_precision": actual_accepted_precision,
        "event_reader_recall": event_recall,
        "event_accepted_recall": event_accepted_recall,
        "event_accepted_precision": event_accepted_precision,
        "hard_safety_hits": hard_safety,
        "stability_buckets": stability_buckets,
        "per_item_stability": {k: v for k, v in stability.items()},
        "negative_control_pass": negative_control_pass,
        "negative_control_total": negative_control_total,
        "negative_control_detail": negative_control_detail,
        "call_seconds_total": sum(r.call_seconds for r in runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    from finance.documents import runtime as doc_runtime
    doc_runtime.register_actuals_model_client(ask_local_raw)
    doc_runtime.register_event_model_client(ask_local_raw)

    cases = [c for c in all_cases() if c.live_reader_case and
             (c.actual_document_text or
              (c.expected_events and any(c.event_documents.values())))]
    excluded = [c.case_id for c in all_cases() if not c.live_reader_case]
    if excluded:
        print(f"Excluded from live run (validator-boundary negative controls, "
             f"not reader tests): {excluded}")
    print(f"Model: {LOCAL_MODEL}")
    print(f"Live-testable cases: {[c.case_id for c in cases]}")

    runs = []
    for run_index in range(1, args.runs + 1):
        print(f"\n=== RUN {run_index}/{args.runs} ===")
        runs.append(run_once(run_index, cases))

    summary = summarize(runs)
    print(f"\nNegative controls (correct-refusal checks): "
         f"{summary['negative_control_pass']}/{summary['negative_control_total']} passed")
    for key, outcomes in summary["negative_control_detail"].items():
        print(f"  {key}: {outcomes}")
    print("\n" + json.dumps({k: v for k, v in summary.items()
                             if k not in ("per_item_stability", "negative_control_detail")},
                            indent=2, default=str))
    print("\nPer-item stability:")
    for key, outcomes in summary["per_item_stability"].items():
        print(f"  {key}: {outcomes}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
