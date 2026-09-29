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
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, ".")

from brain import ask_local_raw, LOCAL_MODEL, OLLAMA_URL                      # noqa: E402
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
from finance.documents.monetary import resolve_scale, scale_multiplier       # noqa: E402
from finance.reported_actuals.tables import document_scale, parse_filing_tables  # noqa: E402
from finance.structural_breaks import PostBalanceSheetEventType               # noqa: E402
from tests.fixtures.document_package_benchmark import all_cases               # noqa: E402


def check_model_available(model: str, ollama_url: str, timeout: float = 20.0) -> Optional[str]:
    """A cheap, real preflight probe -- not a silent substitution.

    Returns None when `model` answers a trivial prompt successfully, or a
    human-readable reason string (e.g. the server's own "retired" error) when
    it does not. Phase H.25/H.26 both burned a full multi-hour benchmark run
    against a model that had been retired that same day, discovering it only
    from a wall of MODEL_ERROR failures at the very end. This makes that an
    immediate, explicit MODEL_UNAVAILABLE exit instead -- never a fallback to
    a different model chosen by this script.
    """
    body = json.dumps({"model": model, "prompt": "ok", "stream": False}).encode()
    req = urllib.request.Request(f"{ollama_url}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return f"could not reach {ollama_url}: {e}"
    except (ValueError, json.JSONDecodeError) as e:
        return f"invalid response from {ollama_url}: {e}"
    if isinstance(payload, dict) and payload.get("error"):
        return str(payload["error"])
    return None


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
    # Phase H.21 additions -- section 8's fixed taxonomy also names these
    # three explicitly; AMOUNT_ROLE and COMMITTED_STATUS parallel the
    # existing VALUE_EXTRACTION/FUNDED_STATUS split (a wrong role or a wrong
    # committed flag is a distinct failure from a wrong bare value), and
    # ITEM_CODE_MISMATCH promotes what was previously folded into
    # MODEL_MISREAD into its own named class.
    AMOUNT_ROLE = "AMOUNT_ROLE"
    COMMITTED_STATUS = "COMMITTED_STATUS"
    ITEM_CODE_MISMATCH = "ITEM_CODE_MISMATCH"
    ACTUAL_GUIDANCE_SEPARATION = "ACTUAL_GUIDANCE_SEPARATION"
    VALIDATOR_FALSE_REJECTION = "VALIDATOR_FALSE_REJECTION"
    SHARED_XBRL_COVERAGE = "SHARED_XBRL_COVERAGE"
    OTHER_GENERALIZED = "OTHER_GENERALIZED"


# Roles that never assert a SPECIFIC economic meaning -- see
# `EventAmountRole`'s own docstring ("OTHER is an honest 'I can tell it's
# economically material but not which named role', never a way to skip
# grounding a role that DOES fit one of the above"; UNKNOWN is the
# no-role-asserted default). A value grounded under either can never be the
# WRONG role, because it never claimed to BE a role in the first place.
_GENERIC_ROLES = {"OTHER", "UNKNOWN"}

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
    # Phase H.21, spec section 5: per-dimension reader accuracy, measured
    # against the READER'S OWN raw proposal (before validation) so a
    # validator rejection never hides whether the model itself got the
    # dimension right -- section 8's "reader wrong + validator rejects is a
    # safe reader failure" needs the reader's claim measured independently.
    dimension_hits: Dict[str, int] = field(default_factory=dict)
    dimension_totals: Dict[str, int] = field(default_factory=dict)
    # Corpus-wide (not per-expected-item): of every RAW candidate the model
    # proposed for an event case, how many were refused specifically for an
    # item-code mismatch or for citing evidence outside the source. These
    # are validator-side, deterministic outcomes -- tracked to see whether
    # the deterministic backstop, not just the reader, is doing its job.
    event_candidates_proposed: int = 0
    item_code_mismatch_rejections: int = 0
    evidence_not_in_source_rejections: int = 0


def _record_hard_safety(run: RunResult, code: str) -> None:
    run.hard_safety_hits[code] = run.hard_safety_hits.get(code, 0) + 1


def _record_dimension(run: RunResult, name: str, correct: bool) -> None:
    run.dimension_totals[name] = run.dimension_totals.get(name, 0) + 1
    if correct:
        run.dimension_hits[name] = run.dimension_hits.get(name, 0) + 1


def _scale_raw_amount(value: Optional[float], unit: Optional[str],
                      scale: Optional[str] = None) -> Optional[float]:
    """Phase H.22: uses the SAME central scale mechanism
    (`finance.documents.monetary`) `EventCandidateValidator` normalizes
    with -- a RAW candidate's amount is unscaled exactly like the resolved
    one; dimension accuracy compares against `expected.amount`, which is
    always the full magnitude, so raw values need the same scaling before
    comparison. Before H.22 this was a second, silently-divergent
    scale-multiplier table; that divergence is exactly the class of bug
    this phase closes."""
    if value is None:
        return None
    resolved = resolve_scale(unit, scale)
    return value * scale_multiplier(resolved)


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
        sections, section_failure = select_event_sections(text)
        if not sections:
            run.event_items.append(ItemResult(
                case.case_id, expected.accession, Outcome.NOT_PROPOSED,
                detail=f"no section to show the model ({section_failure})",
                failure_class=FailureClass.SECTION_SELECTION))
            continue
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

        run.event_candidates_proposed += len(raw_candidates)
        for _c, rcode, _r in rejected:
            if rcode == EventRejectionCode.ITEM_CODE_MISMATCH:
                run.item_code_mismatch_rejections += 1
            elif rcode == EventRejectionCode.EVIDENCE_NOT_IN_SOURCE:
                run.evidence_not_in_source_rejections += 1

        item_key = f"{expected.accession}"
        duplicate = len(raw_candidates) > 1

        def _type_matches(proposed: str, expected_type: str) -> bool:
            if proposed == expected_type:
                return True
            return expected_type in _REFINEMENTS.get(proposed, set()) \
                or proposed in _REFINEMENTS.get(expected_type, set())

        # Phase H.21, spec section 5/8: per-dimension READER accuracy,
        # measured against the model's own RAW claim -- independent of
        # whether the validator went on to accept or refuse it. Prefers a
        # type-matching raw candidate (the one actually meant to answer
        # this expected event); falls back to the first raw candidate
        # proposed for this document if none matched, so a badly-classified
        # candidate still contributes an (expected) wrong answer to the
        # other dimensions rather than silently dropping out of the
        # denominator.
        type_matching_raw = [c for c in raw_candidates
                             if _type_matches(c.event_type, expected.event_type)]
        best_raw = type_matching_raw[0] if type_matching_raw else (
            raw_candidates[0] if raw_candidates else None)
        _record_dimension(run, "event_type", bool(type_matching_raw))
        if best_raw is not None:
            if expected.amount is not None:
                raw_pairs = [(best_raw.amount_role,
                             _scale_raw_amount(best_raw.amount, best_raw.unit, best_raw.scale))] + \
                    [(s.role, _scale_raw_amount(s.value, s.unit, s.scale))
                     for s in best_raw.supplementary_amounts]
                amount_present = any(v is not None and _close_enough(v, expected.amount)
                                     for _r, v in raw_pairs)
                _record_dimension(run, "amount", amount_present)
                if expected.expected_amount_role is not None:
                    role_present = any(
                        r == expected.expected_amount_role and v is not None
                        and _close_enough(v, expected.amount) for r, v in raw_pairs)
                    _record_dimension(run, "amount_role", role_present)
                # Phase H.22, spec section 7: reader SCALE accuracy,
                # independent of role -- did the reader's own resolved
                # magnitude (after applying whatever scale IT claimed) land
                # on the expected value at all, regardless of which slot.
                _record_dimension(run, "scale", amount_present)
            if expected.expected_currency is not None:
                _record_dimension(run, "currency", (best_raw.currency or "").upper()
                                  == expected.expected_currency.upper())
            if expected.funded is not None:
                _record_dimension(run, "funded_status",
                                  best_raw.funded == bool(expected.funded))
            if expected.committed is not None:
                _record_dimension(run, "committed_status",
                                  best_raw.committed == bool(expected.committed))

        matching_accepted = [a for a in accepted if _type_matches(a.event_type, expected.event_type)]

        if matching_accepted:
            resolved = matching_accepted[0]
            funded_ok = resolved.funded == bool(expected.funded)
            if not funded_ok:
                _record_hard_safety(
                    run, "UNDRAWN_FACILITY_COUNTED_AS_FUNDED_DEBT"
                    if resolved.funded and not expected.funded
                    else "FUNDED_STATUS_MISREAD")

            # Phase H.20: check every expected (role, value) pair against the
            # model's COMBINED primary+supplementary amounts, regardless of
            # which one it called "primary" -- a model reporting the
            # transaction total as primary and the per-share price as
            # supplementary (or vice versa) is not a defect; both amounts
            # present with their OWN correct role is what matters. This
            # order-independence is what distinguishes a genuine wrong-role
            # acceptance (the expected role is ABSENT at that value, and a
            # different, economically incompatible role is present there
            # instead) from a harmless reordering.
            resolved_amounts = [(resolved.amount_role, resolved.amount)] + \
                [(s.role, s.value) for s in resolved.supplementary_amounts]
            expected_amounts = [(expected.expected_amount_role, expected.amount)] + \
                list(expected.expected_supplementary)
            # Phase H.22: same (role, value) pairs, plus each amount's OWN
            # resolved currency -- for the order-independent
            # WRONG_CURRENCY_ACCEPTED check below.
            resolved_amounts_with_currency = [
                (resolved.amount_role, resolved.amount, resolved.currency)] + \
                [(s.role, s.value, s.currency) for s in resolved.supplementary_amounts]

            # `expected.amount` is only a fixture-authoring convenience for
            # "which value did I write first" -- it carries no claim about
            # which slot (primary vs. supplementary) the MODEL must use. A
            # model that reports the same value the fixture calls "primary"
            # under its supplementary slot instead (live H.20 finding: FIN-
            # ACQ/FIN-PARTIAL) is not a defect, so this check is a plain
            # value-membership test across BOTH slots; role correctness for
            # every explicitly-typed pair is enforced separately below.
            resolved_values = [v for _r, v in resolved_amounts]
            amount_ok = (expected.amount is None
                        or any(v is not None and _close_enough(v, expected.amount)
                               for v in resolved_values))

            # Phase H.22, spec section 7's hard requirement:
            # WRONG_MAGNITUDE_ACCEPTED = 0. Independent of `amount_ok` --
            # checked directly against the ACCEPTED, validator-resolved
            # amounts, never relying on the scorer's own pass/fail
            # classification (section 1: "do not rely solely on the
            # scorer's classification"). A resolved value that is a
            # thousand/million/billion multiple (or fraction) of the
            # expected one, but not itself close to it, means a wrong
            # SCALE reached a published event.
            if expected.amount not in (None, 0.0):
                for v in resolved_values:
                    if v is None or _close_enough(v, expected.amount):
                        continue
                    ratio = v / expected.amount
                    for factor in (1_000.0, 1_000_000.0, 1_000_000_000.0):
                        if (abs(ratio - factor) / factor < 0.02
                                or abs(ratio - (1.0 / factor)) / (1.0 / factor) < 0.02):
                            _record_hard_safety(run, "WRONG_MAGNITUDE_ACCEPTED")
                            break

            # Currency accepted under a claim that does not match the
            # fixture's own stated currency -- WRONG_CURRENCY_ACCEPTED.
            # Order-independent by VALUE, the same discipline as
            # `amount_ok` above and for the identical reason (live H.22
            # finding: FIN-DIFFCCY) -- a model that puts the GBP commitment
            # in the supplementary slot and the USD draw in the primary one
            # is a harmless reordering, not a wrong currency, as long as
            # EACH value's OWN currency is correct wherever it landed.
            if expected.expected_currency is not None and expected.amount is not None:
                currencies_at_primary_value = [
                    c for _r, v, c in resolved_amounts_with_currency
                    if v is not None and _close_enough(v, expected.amount)]
                if currencies_at_primary_value and not any(
                        c is not None and c.upper() == expected.expected_currency.upper()
                        for c in currencies_at_primary_value):
                    _record_hard_safety(run, "WRONG_CURRENCY_ACCEPTED")

            def _roles_at_value(pairs, value):
                return [role for role, v in pairs
                       if v is not None and _close_enough(v, value)]

            # Phase H.21 scorer fix (live finding: FIN-NETPROCEEDS). A
            # MISSING supplementary pair and a WRONG one are not the same
            # failure: the hard-safety branch below already only fires on
            # WRONG (present under an incompatible role), never on missing
            # -- but `amounts_ok` was gating PROPOSED_AND_ACCEPTED on EVERY
            # expected pair equally, so a correctly-grounded, safely-typed
            # PRIMARY amount was being scored as WRONG_PROPOSAL solely
            # because one optional SUPPLEMENTARY figure (the H.20
            # drop-on-role-mismatch fix's own intended, safe outcome) never
            # made it into the resolved candidate. The PRIMARY pair (index
            # 0 -- the fixture's headline claim) still requires PRESENCE,
            # not just absence-of-wrongness; only supplementary pairs get
            # the recall-miss-is-not-a-failure treatment.
            amounts_ok = True
            for index, (exp_role, exp_value) in enumerate(expected_amounts):
                if exp_role is None or exp_value is None:
                    continue
                roles_here = _roles_at_value(resolved_amounts, exp_value)
                if exp_role in roles_here:
                    continue
                # A SPECIFIC, named role at this value, other than a
                # generic OTHER/UNKNOWN catch-all. Live H.21 finding (FIN-
                # NETPROCEEDS): `OTHER` is the schema's own honest "I can
                # tell it's material but not which named role" bucket (see
                # `EventAmountRole`'s docstring) -- it never asserts a
                # SPECIFIC economic meaning, so it can never be the WRONG
                # one, and must never trigger a hard-safety hit.
                specific_wrong_roles = [r for r in roles_here if r not in _GENERIC_ROLES]
                if exp_role == "PER_SHARE_CONSIDERATION" and "TRANSACTION_VALUE" in roles_here:
                    _record_hard_safety(run, "PER_SHARE_AS_TRANSACTION_TOTAL")
                    amounts_ok = False
                elif exp_role == "FACILITY_COMMITMENT" and "AMOUNT_DRAWN" in roles_here:
                    _record_hard_safety(run, "FACILITY_COMMITMENT_AS_FUNDED_AMOUNT")
                    amounts_ok = False
                elif specific_wrong_roles:
                    _record_hard_safety(run, "WRONG_AMOUNT_ROLE_ACCEPTED")
                    amounts_ok = False
                elif index == 0:
                    # The primary pair is missing outright, or grounded
                    # only under a generic OTHER/UNKNOWN label -- either
                    # way the fixture's headline claim needs its OWN
                    # specific role to count as answered; a recall/
                    # precision miss, never a hard-safety hit.
                    amounts_ok = False
                # else: a SUPPLEMENTARY pair is missing, or present only
                # under a generic OTHER/UNKNOWN label -- a recall footnote,
                # never a reason to discard an otherwise-correct primary
                # answer.

            if funded_ok and amount_ok and amounts_ok:
                outcome = Outcome.DUPLICATE_PROPOSAL if duplicate else Outcome.PROPOSED_AND_ACCEPTED
                run.event_items.append(ItemResult(
                    case.case_id, item_key, outcome,
                    accepted_value={"event_type": resolved.event_type,
                                    "funded": resolved.funded, "amount": resolved.amount,
                                    "amount_role": resolved.amount_role,
                                    "supplementary_amounts":
                                        [a.to_dict() for a in resolved.supplementary_amounts]}))
            else:
                run.event_items.append(ItemResult(
                    case.case_id, item_key, Outcome.WRONG_PROPOSAL,
                    detail=(f"funded={resolved.funded} amount={resolved.amount} "
                           f"role={resolved.amount_role} "
                           f"supplementary={[a.to_dict() for a in resolved.supplementary_amounts]}, "
                           f"expected funded={expected.funded} amounts={expected_amounts}"),
                    accepted_value={"event_type": resolved.event_type,
                                    "funded": resolved.funded, "amount": resolved.amount,
                                    "amount_role": resolved.amount_role},
                    failure_class=(FailureClass.FUNDED_STATUS if not funded_ok
                                  else FailureClass.AMOUNT_ROLE if not amounts_ok
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
            # Section 8's three-way split: a should-be-accepted item that
            # got refused is a VALIDATOR_FALSE_REJECTION only if the READER
            # itself proposed the right type (`type_matching_raw` above,
            # computed before validation) -- if the reader never proposed a
            # compatible type at all, refusal is a safe reader failure
            # (EVENT_CLASSIFICATION), not the validator's fault. A specific
            # ITEM_CODE_MISMATCH/AMOUNT_ROLE_NOT_GROUNDED code gets its own
            # class when the reader WAS right but the deterministic
            # boundary still refused it.
            if expected.should_be_accepted:
                if not type_matching_raw:
                    fclass = FailureClass.EVENT_CLASSIFICATION
                elif codes and codes[0] == EventRejectionCode.ITEM_CODE_MISMATCH:
                    fclass = FailureClass.ITEM_CODE_MISMATCH
                elif codes and codes[0] == EventRejectionCode.AMOUNT_ROLE_NOT_GROUNDED:
                    fclass = FailureClass.AMOUNT_ROLE
                elif codes and codes[0] == EventRejectionCode.EVIDENCE_NOT_IN_SOURCE:
                    fclass = FailureClass.EVIDENCE_GROUNDING
                else:
                    fclass = FailureClass.VALIDATOR_FALSE_REJECTION
            else:
                fclass = None
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

    # Phase H.21, spec section 5: per-dimension reader accuracy, one value
    # per run (mean/min/max computed by the caller, matching every other
    # per-run metric here).
    def _dimension_series(name: str) -> List[float]:
        values = []
        for run in runs:
            total = run.dimension_totals.get(name, 0)
            hits = run.dimension_hits.get(name, 0)
            values.append((hits / total) if total else 1.0)
        return values

    event_type_accuracy = _dimension_series("event_type")
    amount_accuracy = _dimension_series("amount")
    amount_role_accuracy = _dimension_series("amount_role")
    currency_accuracy = _dimension_series("currency")
    scale_accuracy = _dimension_series("scale")
    funded_status_accuracy = _dimension_series("funded_status")
    committed_status_accuracy = _dimension_series("committed_status")

    item_code_consistency: List[float] = []
    evidence_grounding_accuracy: List[float] = []
    for run in runs:
        total = run.event_candidates_proposed
        item_code_consistency.append(
            1.0 - (run.item_code_mismatch_rejections / total) if total else 1.0)
        evidence_grounding_accuracy.append(
            1.0 - (run.evidence_not_in_source_rejections / total) if total else 1.0)

    # Section 8: a corpus-wide tally of WHY each miss/wrong-proposal
    # happened, not a stock-by-stock list -- the generalized failure-class
    # breakdown the completion summary reports from directly.
    failure_class_counts: Dict[str, int] = {}
    for run in runs:
        for item in run.event_items + run.actual_items:
            if item.failure_class:
                failure_class_counts[item.failure_class] = \
                    failure_class_counts.get(item.failure_class, 0) + 1

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
        "event_type_accuracy": event_type_accuracy,
        "amount_accuracy": amount_accuracy,
        "amount_role_accuracy": amount_role_accuracy,
        "currency_accuracy": currency_accuracy,
        "scale_accuracy": scale_accuracy,
        "funded_status_accuracy": funded_status_accuracy,
        "committed_status_accuracy": committed_status_accuracy,
        "item_code_consistency": item_code_consistency,
        "evidence_grounding_accuracy": evidence_grounding_accuracy,
        "failure_class_counts": failure_class_counts,
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
    parser.add_argument("--skip-model-check", action="store_true",
                        help="Skip the model-availability preflight probe.")
    args = parser.parse_args()

    if not args.skip_model_check:
        unavailable_reason = check_model_available(LOCAL_MODEL, OLLAMA_URL)
        if unavailable_reason is not None:
            # Explicit, immediate MODEL_UNAVAILABLE exit -- never a silent
            # substitution of a different model this script chooses itself.
            print(json.dumps({
                "status": "MODEL_UNAVAILABLE",
                "model": LOCAL_MODEL,
                "ollama_url": OLLAMA_URL,
                "reason": unavailable_reason,
            }, indent=2))
            sys.exit(2)

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
