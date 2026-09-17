"""The acceptance boundary for a financing/capital-structure event candidate.

PHASE H.18 -- EVIDENCE IS RETRIEVED, NOT SEARCHED FOR

`evidence_is_grounded`-style substring search against a model's own
transcription is GONE. A candidate names `evidence_span_ids`; this module
resolves them against the SAME span list the model was shown
(`finance.documents.spans.resolve_span_ids`) and refuses outright --
`EVIDENCE_NOT_IN_SOURCE` -- if any id is unknown (invented, mistyped, or
left over from a different document's span set). The reconstructed text is
by construction an exact, verbatim slice of the source document: there is no
longer a class of failure where a semantically correct reading is refused
for imperfect transcription, because nothing is transcribed.

THE CHECK THAT MATTERS MOST, UNCHANGED

`funded=True` is refused -- never silently downgraded -- unless the
RESOLVED evidence (the actual cited spans, concatenated) contains language
that says money moved: drew, borrowed, issued and sold, received net
proceeds, closed the offering. "Creation of a Direct Financial Obligation"
(the language of 8-K item 2.03 itself) is NOT drawdown language; it proves a
facility exists, not that it was used.
"""

import re
from typing import Optional, Sequence, Tuple

import tools.config as config
from finance.documents.event_schema import EventCandidate, EventRejectionCode, ResolvedFinancingEvent
from finance.documents.spans import SourceSpan, resolve_span_ids
from finance.extraction.validator import _numbers_in
from finance.extraction.schema import domain_unit
from finance.structural_breaks import (
    PostBalanceSheetEventType,
    classify_security_event,
    event_impact,
)

_ISO_CURRENCY = re.compile(r"^[A-Z]{3}$")

# Language that says money actually MOVED. Deliberately narrower than
# `finance.structural_breaks._DEBT_LANGUAGE` (which recognises the SUBJECT of
# a filing, debt vs equity) -- this recognises the ACT of funding, which that
# module's own docstring says it never attempts.
_FUNDED_LANGUAGE = re.compile(
    r"(?i)\b(?:drew|borrowed|drawdown|draw-down|"
    r"(?:issued\s+and\s+sold|sold\s+and\s+issued)|"
    r"issued\s+(?:\$[\d,.]+|notes|shares|bonds|securities)|"
    r"received\s+(?:net\s+)?proceeds|net\s+proceeds\s+(?:of|were)|"
    r"closed\s+the\s+(?:offering|sale|transaction)|"
    r"completed\s+the\s+(?:offering|issuance)|"
    r"\bfunded\b)\b")

# Which deterministic item-code classification a model's claimed event type
# may REFINE. `finance.structural_breaks.classify_security_event` already
# performs the convertible/repurchase-funding refinements itself when its own
# language patterns match; this is the model's read being held to the same
# menu rather than a competing one. UNKNOWN (the item codes did not name a
# type on their own -- items 1.01/2.04 have no entry in `_ITEM_EVENT_TYPES`)
# imposes no constraint.
_T = PostBalanceSheetEventType
_COMPATIBLE_REFINEMENTS = {
    _T.ISSUER_DEBT_ISSUANCE: {_T.ISSUER_DEBT_ISSUANCE, _T.CONVERTIBLE_ISSUANCE,
                             _T.DEBT_FINANCED_REPURCHASE, _T.REFINANCING},
    _T.ACQUISITION: {_T.ACQUISITION},
    _T.ISSUER_EQUITY_ISSUANCE: {_T.ISSUER_EQUITY_ISSUANCE, _T.CONVERTIBLE_ISSUANCE,
                               _T.WARRANT_ISSUANCE},
}


def _matches_any(value: float, present) -> bool:
    forms = {value, value * 100.0, value / 100.0}
    return any(any(abs(f - p) <= max(abs(p) * 1e-6, 1e-9) for p in present)
               for f in forms)


class EventCandidateValidator:
    """One centralized acceptance boundary for LLM-read financing events.

    `spans` must be the EXACT span list the extractor showed the model for
    this document (`finance.documents.event_extractor.select_event_sections`
    builds it once; the caller threads the same list to both). A candidate
    whose `evidence_span_ids` do not all resolve against `spans` is refused
    before any other check runs -- there is nothing left to check against.
    """

    def __init__(self, spans: Sequence[SourceSpan] = (), items: Optional[str] = None,
                 form: str = "8-K", description: str = "",
                 min_confidence: Optional[float] = None):
        self.spans = tuple(spans)
        self.items = items or ""
        self.form = form
        self.description = description
        self.min_confidence = (min_confidence if min_confidence is not None
                               else config.finance_event_extraction_min_confidence())

    def validate(self, candidate: EventCandidate
                ) -> Tuple[Optional[ResolvedFinancingEvent], Optional[str], str]:
        if not candidate.evidence_span_ids:
            return None, EventRejectionCode.NO_EVIDENCE, (
                "the candidate cites no evidence spans, so nothing about it can be checked")

        resolved_text, unknown_ids = resolve_span_ids(candidate.evidence_span_ids, self.spans)
        if resolved_text is None:
            return None, EventRejectionCode.EVIDENCE_NOT_IN_SOURCE, (
                f"span id(s) {list(unknown_ids)} do not exist in the spans this run showed "
                "the model -- invented, mistyped, or from a different document")

        if candidate.event_type not in PostBalanceSheetEventType.ALL:
            return None, EventRejectionCode.UNSUPPORTED_EVENT_TYPE, (
                f"{candidate.event_type!r} is not a reviewed event type")

        deterministic_type, _confidence = classify_security_event(
            self.form, self.items, self.description)
        if deterministic_type != PostBalanceSheetEventType.UNKNOWN:
            allowed = _COMPATIBLE_REFINEMENTS.get(deterministic_type, {deterministic_type})
            if candidate.event_type not in allowed:
                return None, EventRejectionCode.ITEM_CODE_MISMATCH, (
                    f"the filing's own item code(s) classify this as "
                    f"{deterministic_type!r}, which does not admit "
                    f"{candidate.event_type!r}")

        if candidate.amount is not None:
            present = _numbers_in(resolved_text)
            if not present or not _matches_any(candidate.amount, present):
                return None, EventRejectionCode.AMOUNT_NOT_GROUNDED, (
                    f"{candidate.amount} does not appear in the cited spans")
            currency = (candidate.currency or "").upper()
            if not _ISO_CURRENCY.match(currency):
                return None, EventRejectionCode.UNKNOWN_CURRENCY, (
                    f"{candidate.currency!r} is not a recognisable ISO currency code "
                    "for a stated amount")

        # THE check. `funded=True` requires its own drawdown-language
        # evidence, IN THE CITED SPANS; the item code and the amount being
        # grounded prove neither. See the module docstring.
        if candidate.funded and not _FUNDED_LANGUAGE.search(resolved_text):
            return None, EventRejectionCode.FUNDED_STATUS_NOT_GROUNDED, (
                "the candidate asserts the event is funded but the cited spans contain no "
                "drawdown/proceeds-received language -- they may describe a facility being "
                "established rather than money moving")

        if candidate.confidence < self.min_confidence:
            return None, EventRejectionCode.LOW_CONFIDENCE, (
                f"confidence {candidate.confidence:.2f} is below the floor "
                f"{self.min_confidence:.2f}")

        currency = (candidate.currency or "").upper() or None
        amount = (candidate.amount * self._scale(candidate.unit)
                  if candidate.amount is not None else None)
        resolved = ResolvedFinancingEvent(
            event_type=candidate.event_type, amount=amount, currency=currency,
            funded=candidate.funded, committed=candidate.committed,
            counterparty=candidate.counterparty,
            effective_date=candidate.effective_date,
            # The AUTHORITATIVE evidence text -- reconstructed from the
            # document's own spans, never from anything the model wrote.
            source_evidence=resolved_text,
            confidence=candidate.confidence, accession=candidate.accession,
            form=candidate.form, filed=candidate.filed,
            impact=event_impact(candidate.event_type))
        return resolved, None, ""

    @staticmethod
    def _scale(unit: Optional[str]) -> float:
        _domain, scale = domain_unit(unit)
        return {None: 1.0, "million": 1_000_000.0, "billion": 1_000_000_000.0}.get(scale, 1.0)

    def validate_all(self, candidates: Sequence[EventCandidate]):
        accepted, rejected = [], []
        for candidate in candidates:
            resolved, code, reason = self.validate(candidate)
            if resolved is None:
                rejected.append((candidate, code, reason))
            else:
                accepted.append(resolved)
        return accepted, rejected
