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

PHASE H.20 -- A GROUNDED NUMBER STILL NEEDS A GROUNDED MEANING

Live shadow evidence: "$95.00 in cash ... (the 'Per Share Amount')" grounds
`amount=95.0` perfectly under H.18's rules -- the number IS in the cited
span. It would still be wrong to publish as the deal's transaction value,
which was tens of billions of dollars. So amount checking is now TWO
questions, not one: does the value exist in the cited evidence (unchanged),
and -- when a role is asserted -- does that SAME evidence contain language
supporting that specific economic role. `EventAmountRole.UNKNOWN` asks
nothing of the second question and always passes it; a role that IS
asserted must earn it or the whole candidate is refused
(`AMOUNT_ROLE_NOT_GROUNDED`), never silently relabelled UNKNOWN.

THE CHECK THAT MATTERS MOST, UNCHANGED

`funded=True` is refused -- never silently downgraded -- unless the
RESOLVED evidence (the actual cited spans, concatenated) contains language
that says money moved: drew, borrowed, issued and sold, received net
proceeds, closed the offering. "Creation of a Direct Financial Obligation"
(the language of 8-K item 2.03 itself) is NOT drawdown language; it proves a
facility exists, not that it was used. This check is independent of amount
role by design -- a `FACILITY_COMMITMENT` amount and the `funded` flag are
two different claims about two different things, and neither validates the
other.
"""

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import tools.config as config
from finance import guidance as gm
from finance.documents.event_schema import (
    EventAmount,
    EventAmountRole,
    EventCandidate,
    EventRejectionCode,
    ResolvedAmount,
    ResolvedFinancingEvent,
)
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


# ---------------------------------------------------------------------------
# Phase H.20 -- amount ROLE grounding
# ---------------------------------------------------------------------------
#
# Structural KEEP-class language cues, same footing as `_FUNDED_LANGUAGE`
# above: each recognises VOCABULARY a role's own sentence conventionally
# uses, never an inference about what the number "should" mean. A role
# asserted with none of its language present in the number's own cited
# evidence is refused -- this is what turns "the number is in the source"
# into "the number's CLAIMED MEANING is in the source".
#
# "net proceeds" appears under BOTH TRANSACTION_VALUE/PURCHASE_PRICE and (via
# `_FUNDED_LANGUAGE`) AMOUNT_DRAWN -- a real, common ambiguity in filing
# prose itself (an equity offering's "net proceeds" IS its economic value; a
# debt draw's "net proceeds" IS the funded amount). The event-type/role
# compatibility matrix below, not this dict, is what disambiguates: only
# transaction-shaped types admit TRANSACTION_VALUE/PURCHASE_PRICE, only
# debt-shaped types admit AMOUNT_DRAWN.
_ROLE_LANGUAGE: Dict[str, re.Pattern] = {
    EventAmountRole.PER_SHARE_CONSIDERATION: re.compile(
        r"(?i)\bper\s+share\b|\beach\s+(?:outstanding\s+)?share\b|/\s*share\b|"
        r"\bper-share\b"),
    EventAmountRole.TRANSACTION_VALUE: re.compile(
        r"(?i)\btransaction\s+value\b|\bequity\s+value\b|\baggregate\s+value\b|"
        r"\benterprise\s+value\b|\btotal\s+consideration\b|\bimplied\s+value\b|"
        r"\b(?:aggregate\s+)?net\s+proceeds\b"),
    EventAmountRole.PURCHASE_PRICE: re.compile(
        r"(?i)\bpurchase\s+price\b|\b(?:aggregate\s+)?net\s+proceeds\b"),
    EventAmountRole.PRINCIPAL_AMOUNT: re.compile(
        r"(?i)\bprincipal\s+amount\b|\baggregate\s+principal\b|"
        r"\bterm\s+loan\b|\bnotes?\s+due\s+20\d{2}\b"),
    EventAmountRole.FACILITY_COMMITMENT: re.compile(
        r"(?i)\bcommitment(?:s)?\b|\bfacility\b|\bavailable\s+(?:for|to)\b|"
        r"\bprovides?\s+for\b|\brevolving\s+credit\b"),
    EventAmountRole.AMOUNT_DRAWN: _FUNDED_LANGUAGE,
    EventAmountRole.AMOUNT_OUTSTANDING: re.compile(r"(?i)\boutstanding\b"),
    EventAmountRole.REPAYMENT_AMOUNT: re.compile(
        r"(?i)\brepa(?:id|yment|y|ying)\b|\bredeem(?:ed|ing)?\b|\bretired\b"),
    EventAmountRole.REFINANCED_AMOUNT: re.compile(r"(?i)\brefinanc\w*\b"),
    EventAmountRole.MAXIMUM_CAPACITY: re.compile(
        r"(?i)\bmaximum\b|\bup\s+to\b|\bnot\s+to\s+exceed\b|\baggregate\s+maximum\b"),
    # OTHER and UNKNOWN have no language requirement -- OTHER is an honest
    # "I can tell it's economically material but not which named role",
    # never a way to skip grounding a role that DOES fit one of the above.
}

# section 6: which amount roles an event type may plausibly carry. A type
# absent from this map is UNRESTRICTED (return None from `_roles_allowed_for`)
# -- the matrix exists to reject clearly impossible combinations, not to
# encode a rule for every corporate-finance sentence (spec section 6). Debt-
# shaped types are deliberately generous: `ISSUER_DEBT_ISSUANCE` alone covers
# term loans, revolvers and notes issuances in this taxonomy (structural_
# breaks.py has no finer-grained type), so it must admit every debt-shaped
# role rather than guessing which instrument a filing describes.
_DEBT_ROLES: Set[str] = {
    EventAmountRole.PRINCIPAL_AMOUNT, EventAmountRole.FACILITY_COMMITMENT,
    EventAmountRole.AMOUNT_DRAWN, EventAmountRole.AMOUNT_OUTSTANDING,
    EventAmountRole.REFINANCED_AMOUNT, EventAmountRole.REPAYMENT_AMOUNT,
    EventAmountRole.MAXIMUM_CAPACITY, EventAmountRole.OTHER, EventAmountRole.UNKNOWN,
}
_TRANSACTION_ROLES: Set[str] = {
    EventAmountRole.TRANSACTION_VALUE, EventAmountRole.PER_SHARE_CONSIDERATION,
    EventAmountRole.PURCHASE_PRICE, EventAmountRole.OTHER, EventAmountRole.UNKNOWN,
}
_EVENT_TYPE_ROLES: Dict[str, Set[str]] = {
    _T.ISSUER_DEBT_ISSUANCE: _DEBT_ROLES,
    _T.CONVERTIBLE_ISSUANCE: _DEBT_ROLES | {EventAmountRole.PER_SHARE_CONSIDERATION},
    _T.DEBT_FINANCED_REPURCHASE: _DEBT_ROLES | _TRANSACTION_ROLES,
    _T.REFINANCING: _DEBT_ROLES,
    _T.ACQUISITION: _TRANSACTION_ROLES,
    _T.DIVESTITURE: _TRANSACTION_ROLES,
    _T.ISSUER_EQUITY_ISSUANCE: _TRANSACTION_ROLES,
    _T.WARRANT_ISSUANCE: _TRANSACTION_ROLES,
    _T.SHARE_REPURCHASE: _TRANSACTION_ROLES | {EventAmountRole.MAXIMUM_CAPACITY},
    _T.ACCELERATED_SHARE_REPURCHASE: _TRANSACTION_ROLES,
    _T.SPECIAL_DIVIDEND: _TRANSACTION_ROLES,
    # OTHER_MATERIAL_FINANCING / UNKNOWN / anything unlisted: unrestricted.
}


def _roles_allowed_for(event_type: str) -> Optional[Set[str]]:
    return _EVENT_TYPE_ROLES.get(event_type)


def _unit_agrees_with_role(unit: Optional[str], role: str) -> bool:
    """Spec section 3: role and unit must agree. A per-share role demands a
    per-share unit; a per-share UNIT under a non-per-share role is the same
    contradiction the other way round."""
    domain, _scale = domain_unit(unit)
    is_per_share_unit = domain == gm.GuidanceUnit.CURRENCY_PER_SHARE
    is_per_share_role = role in EventAmountRole.PER_UNIT_ROLES
    if is_per_share_role and not is_per_share_unit:
        return False
    if is_per_share_unit and not is_per_share_role and role not in (
            EventAmountRole.UNKNOWN, EventAmountRole.OTHER):
        return False
    return True


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

    # -- one amount: value grounding + role grounding + unit/role agreement --

    def _ground_amount(self, value: Optional[float], unit: Optional[str], role: str,
                       span_ids: Sequence[str], allowed_roles: Optional[Set[str]],
                       event_type: str) -> Tuple[Optional[str], Optional[str], str]:
        """(resolved evidence text, rejection code or None, reason).

        Applies to BOTH the candidate's primary amount and every entry in
        `supplementary_amounts` -- one rule, not two, so a supplementary
        amount cannot smuggle in a claim the primary amount would have been
        refused for.
        """
        if value is None:
            return "", None, ""            # nothing asserted, nothing to check

        resolved_text, unknown_ids = resolve_span_ids(span_ids, self.spans)
        if resolved_text is None:
            return None, EventRejectionCode.EVIDENCE_NOT_IN_SOURCE, (
                f"span id(s) {list(unknown_ids)} do not exist in the spans this run "
                "showed the model")

        present = _numbers_in(resolved_text)
        if not present or not _matches_any(value, present):
            return None, EventRejectionCode.AMOUNT_NOT_GROUNDED, (
                f"{value} does not appear in the cited spans -- a computed or derived "
                "number is not a value the source itself states")

        if role != EventAmountRole.UNKNOWN:
            if not _unit_agrees_with_role(unit, role):
                return None, EventRejectionCode.AMOUNT_ROLE_NOT_GROUNDED, (
                    f"unit {unit!r} does not agree with role {role!r} (spec section 3: "
                    "role and unit must agree)")
            pattern = _ROLE_LANGUAGE.get(role)
            if pattern is not None and not pattern.search(resolved_text):
                return None, EventRejectionCode.AMOUNT_ROLE_NOT_GROUNDED, (
                    f"the cited spans contain no {role} language -- a number existing "
                    "in the source does not prove what economic role it plays")
            if allowed_roles is not None and role not in allowed_roles:
                return None, EventRejectionCode.AMOUNT_ROLE_NOT_GROUNDED, (
                    f"{role} is not a plausible amount role for event type {event_type!r}")
        return resolved_text, None, ""

    def _scale_value(self, value: Optional[float], unit: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        return value * self._scale(unit)

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

        allowed_roles = _roles_allowed_for(candidate.event_type)

        # Primary amount: value grounding (unchanged) + role grounding (new).
        primary_text, code, reason = self._ground_amount(
            candidate.amount, candidate.unit, candidate.amount_role,
            candidate.evidence_span_ids, allowed_roles, candidate.event_type)
        if code is not None:
            return None, code, reason
        if candidate.amount is not None:
            currency = (candidate.currency or "").upper()
            if not _ISO_CURRENCY.match(currency):
                return None, EventRejectionCode.UNKNOWN_CURRENCY, (
                    f"{candidate.currency!r} is not a recognisable ISO currency code "
                    "for a stated amount")

        # Supplementary amounts: same grounding rule as the primary amount,
        # but a failure here DROPS that one amount rather than refusing the
        # whole candidate. A supplementary figure is additional to the
        # primary claim already grounded above; an ungrounded value or an
        # implausible role on it means "this extra fact could not be
        # verified" -- exactly the miss the spec allows ("the reader is
        # allowed to miss a fact; it is not allowed to publish an
        # unsupported one") -- never a reason to also discard an
        # already-verified, correctly-typed primary amount over a role
        # dispute about a DIFFERENT number. Live H.20 finding: a convertible
        # note's grounded $800M PRINCIPAL_AMOUNT was being refused whole
        # because its supplementary "$784 million net proceeds" was tagged
        # TRANSACTION_VALUE, a role CONVERTIBLE_ISSUANCE's matrix does not
        # admit -- the fix is to drop the disputed $784M, not the sound
        # $800M. The primary amount keeps the atomic all-or-nothing rule
        # (a candidate with no trustworthy headline amount is not published).
        resolved_supplementary: List[ResolvedAmount] = []
        for supplement in candidate.supplementary_amounts:
            supp_text, code, reason = self._ground_amount(
                supplement.value, supplement.unit, supplement.role,
                supplement.evidence_span_ids, allowed_roles, candidate.event_type)
            if code is not None:
                continue    # drop this one unverified claim; primary stands
            if supplement.value is not None:
                resolved_supplementary.append(ResolvedAmount(
                    role=supplement.role,
                    value=self._scale_value(supplement.value, supplement.unit),
                    unit=supplement.unit, source_evidence=supp_text))

        # THE check. `funded=True` requires its own drawdown-language
        # evidence, IN THE CITED SPANS; the item code and the amount being
        # grounded prove neither. See the module docstring. Independent of
        # amount role by design: a FACILITY_COMMITMENT amount and `funded`
        # are two different claims about two different things.
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
        amount = self._scale_value(candidate.amount, candidate.unit)
        resolved = ResolvedFinancingEvent(
            event_type=candidate.event_type, amount=amount, currency=currency,
            amount_role=candidate.amount_role,
            funded=candidate.funded, committed=candidate.committed,
            counterparty=candidate.counterparty,
            effective_date=candidate.effective_date,
            # The AUTHORITATIVE evidence text -- reconstructed from the
            # document's own spans, never from anything the model wrote.
            source_evidence=resolved_text,
            supplementary_amounts=tuple(resolved_supplementary),
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
