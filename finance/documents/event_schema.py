"""The candidate type for a financing/capital-structure EVENT.

Reuses `finance.structural_breaks.PostBalanceSheetEventType` directly (spec
section 8: "Use existing types/enums whenever possible") rather than
inventing a parallel vocabulary. `structural_breaks.classify_security_event`
already classifies an event's TYPE from its 8-K item code and filing
description; what it explicitly does not do -- by design, see that module's
own docstring -- is read the filing's own text for an amount, a currency, or
whether the facility is actually FUNDED. That is the genuinely new
capability this schema exists for.

THE ONE RULE THAT MATTERS MOST

`funded` is a claim requiring its own evidence, never an inference from the
event TYPE. "Creation of a Direct Financial Obligation" (item 2.03) proves a
facility was established; it does not prove a dollar was drawn.
`event_validator.py` enforces this by grounding, not by trusting the model's
`funded` flag.

EVIDENCE IS SELECTED, NEVER AUTHORED (Phase H.18)

`EventCandidate.evidence_span_ids` is the ONLY evidence field the model
populates -- a list of ids into the numbered spans
`finance.documents.spans.build_source_spans` cut from the document BEFORE
the model ever sees it. `source_evidence` stays on the dataclass because
`ResolvedFinancingEvent` and every downstream reader still wants one string
to cite, but its value is always RECONSTRUCTED by
`finance.documents.spans.resolve_span_ids` from the referenced spans'
verbatim text -- never taken from anything the model wrote. A model that
elides or rewords a clause can point at the wrong span; it cannot fabricate
one, because span ids are assigned deterministically before the model call
and never accepted as authored by it.

A NUMBER BEING GROUNDED DOES NOT SAY WHAT IT MEANS (Phase H.20)

Live shadow evidence: a real acquisition filing states "$95.00 in cash ...
(the 'Per Share Amount')". A candidate citing the span containing "95.00"
and reporting `amount=95.0` passes every H.18 grounding check -- the number
IS in the source -- and would still be economically wrong if read as the
deal's transaction value (which was in fact tens of billions of dollars).
Exact-substring grounding proves a number exists; it says nothing about
which of a document's many distinct economic quantities (a per-share price,
a facility's total commitment, the amount actually drawn, a note's
principal, a repayment) that number IS.

`amount_role` (`EventAmountRole`) makes that economic role an explicit,
independently-grounded claim rather than an assumption baked into which
dataclass field a number happens to occupy. `EventAmountRole.UNKNOWN` is a
legitimate, PASSING value -- section 2's "do not make every amount
mandatory" -- so an amount with no asserted role is accepted as before,
unclassified rather than presumed. A role that IS asserted must be grounded
by role-specific language inside the SAME cited spans as the value itself,
or the candidate is refused (`AMOUNT_ROLE_NOT_GROUNDED`), never silently
downgraded to UNKNOWN -- downgrading would let a wrong claim through wearing
a blank label instead of a false one.

`supplementary_amounts` (of type `EventAmount`) lets one candidate carry
MORE than one economically distinct, independently grounded and typed
number -- a facility's $5B commitment alongside its $2B drawn portion, or a
$95/share price alongside an explicitly stated $69B transaction value.
Never a derivation of one from the other: `event_validator.py` grounds each
amount's VALUE against its own cited evidence exactly as before, so a
model-computed product or difference is refused for the same reason a
model-computed sum already was (Phase H.19's live finding).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from finance.structural_breaks import PostBalanceSheetEventType


class EventAmountRole:
    """The economic MEANING of one grounded monetary value.

    Deliberately generic across financing/corporate-event prose (spec
    section 2) rather than issuer- or instrument-specific. `UNKNOWN` is not
    a failure state -- it is what an amount with no asserted role holds,
    and passes validation on that basis alone.
    """

    TRANSACTION_VALUE = "TRANSACTION_VALUE"
    PER_SHARE_CONSIDERATION = "PER_SHARE_CONSIDERATION"
    PRINCIPAL_AMOUNT = "PRINCIPAL_AMOUNT"
    FACILITY_COMMITMENT = "FACILITY_COMMITMENT"
    AMOUNT_DRAWN = "AMOUNT_DRAWN"
    AMOUNT_OUTSTANDING = "AMOUNT_OUTSTANDING"
    REFINANCED_AMOUNT = "REFINANCED_AMOUNT"
    REPAYMENT_AMOUNT = "REPAYMENT_AMOUNT"
    PURCHASE_PRICE = "PURCHASE_PRICE"
    MAXIMUM_CAPACITY = "MAXIMUM_CAPACITY"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"

    ALL = (TRANSACTION_VALUE, PER_SHARE_CONSIDERATION, PRINCIPAL_AMOUNT,
           FACILITY_COMMITMENT, AMOUNT_DRAWN, AMOUNT_OUTSTANDING,
           REFINANCED_AMOUNT, REPAYMENT_AMOUNT, PURCHASE_PRICE,
           MAXIMUM_CAPACITY, OTHER, UNKNOWN)

    # A role expressed per unit of stock, never a whole-transaction figure.
    # Its UNIT must agree (spec section 3): `finance.extraction.schema`'s
    # "PER_SHARE" model unit, never a bare currency amount.
    PER_UNIT_ROLES = (PER_SHARE_CONSIDERATION,)


@dataclass(frozen=True)
class EventAmount:
    """One ADDITIONAL monetary value beyond a candidate's primary amount --
    e.g. the drawn portion of a facility, or an explicit transaction value
    stated alongside a per-share price. Always independently grounded: its
    own `value` must appear in its own `evidence_span_ids`, and its `role`
    (if not UNKNOWN) must be supported by role-specific language in that
    same evidence -- see `event_validator.py`.
    """

    role: str = EventAmountRole.UNKNOWN
    value: Optional[float] = None
    unit: Optional[str] = None
    evidence_span_ids: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"role": self.role, "value": self.value, "unit": self.unit,
                "evidence_span_ids": list(self.evidence_span_ids)}


@dataclass(frozen=True)
class ResolvedAmount:
    """A validated `EventAmount`, with its evidence RESOLVED to text (never
    span ids) -- the same "reader selects, deterministic code retrieves"
    contract `ResolvedFinancingEvent.source_evidence` already follows."""

    role: str
    value: Optional[float]
    unit: Optional[str]
    source_evidence: str

    def to_dict(self) -> dict:
        return {"role": self.role, "value": self.value, "unit": self.unit,
                "source_evidence": self.source_evidence}


class EventRejectionCode:
    NO_EVIDENCE = "NO_EVIDENCE"
    EVIDENCE_NOT_IN_SOURCE = "EVIDENCE_NOT_IN_SOURCE"
    VALUE_NOT_IN_EVIDENCE = "VALUE_NOT_IN_EVIDENCE"
    UNSUPPORTED_EVENT_TYPE = "UNSUPPORTED_EVENT_TYPE"
    ITEM_CODE_MISMATCH = "ITEM_CODE_MISMATCH"
    FUNDED_STATUS_NOT_GROUNDED = "FUNDED_STATUS_NOT_GROUNDED"
    AMOUNT_NOT_GROUNDED = "AMOUNT_NOT_GROUNDED"
    # Phase H.20: the value is grounded (it exists in the cited spans) but
    # the ECONOMIC ROLE asserted for it -- transaction value, per-share
    # price, principal, facility commitment, ... -- is not, or is not one
    # the event's own type can plausibly carry. Never conflated with
    # AMOUNT_NOT_GROUNDED: that code means the NUMBER is not in the source;
    # this one means the number is there but its claimed MEANING is not.
    AMOUNT_ROLE_NOT_GROUNDED = "AMOUNT_ROLE_NOT_GROUNDED"
    UNKNOWN_CURRENCY = "UNKNOWN_CURRENCY"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MALFORMED = "MALFORMED"

    ALL = (NO_EVIDENCE, EVIDENCE_NOT_IN_SOURCE, VALUE_NOT_IN_EVIDENCE,
           UNSUPPORTED_EVENT_TYPE, ITEM_CODE_MISMATCH,
           FUNDED_STATUS_NOT_GROUNDED, AMOUNT_NOT_GROUNDED,
           AMOUNT_ROLE_NOT_GROUNDED, UNKNOWN_CURRENCY, LOW_CONFIDENCE,
           MALFORMED)

    # The one a benchmark must count as critical (spec section 27): a
    # facility asserted funded with nothing in its own text supporting a
    # drawdown is `UNDRAWN_FACILITY_COUNTED_AS_FUNDED_DEBT` if it slipped
    # through; this is the code that stops it before that. A per-share price
    # accepted as a transaction value is the same CLASS of failure --
    # grounded value, wrong economic meaning -- so AMOUNT_ROLE_NOT_GROUNDED
    # belongs in this set too.
    UNSUPPORTED = (EVIDENCE_NOT_IN_SOURCE, VALUE_NOT_IN_EVIDENCE,
                   FUNDED_STATUS_NOT_GROUNDED, AMOUNT_NOT_GROUNDED,
                   AMOUNT_ROLE_NOT_GROUNDED, ITEM_CODE_MISMATCH)


@dataclass(frozen=True)
class EventCandidate:
    """One proposed financing/capital event, with the SPANS that support it.

    `evidence_span_ids` is what the model fills. `source_evidence` is filled
    afterward, deterministically, by resolving those ids against the same
    span list the model was shown (see the module docstring) -- it is never
    parsed from the model's own JSON. A candidate straight off the model
    (before that resolution step) has `source_evidence=""`.
    """

    event_type: str = PostBalanceSheetEventType.UNKNOWN
    amount: Optional[float] = None
    unit: Optional[str] = None                 # USD / USD_MILLION / USD_BILLION
    # The economic role of `amount` specifically. UNKNOWN (default) is a
    # legitimate, passing value -- an amount with no asserted role is
    # accepted unclassified, never presumed to be any particular one.
    amount_role: str = EventAmountRole.UNKNOWN
    currency: Optional[str] = None
    funded: bool = False
    committed: bool = False
    counterparty: Optional[str] = None
    effective_date: Optional[str] = None
    evidence_span_ids: Tuple[str, ...] = ()
    # Additional, independently-typed-and-grounded amounts beyond the
    # primary one above -- a facility's drawn portion, an explicit
    # transaction value stated alongside a per-share price. Never required;
    # never a substitute for grounding the primary amount.
    supplementary_amounts: Tuple[EventAmount, ...] = ()
    source_evidence: str = ""
    confidence: float = 0.0
    # Filled by the extractor, never by the model.
    source_document_id: Optional[str] = None
    accession: Optional[str] = None
    form: Optional[str] = None
    filed: Optional[str] = None
    items: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type, "amount": self.amount,
            "unit": self.unit, "amount_role": self.amount_role,
            "currency": self.currency,
            "funded": self.funded, "committed": self.committed,
            "counterparty": self.counterparty,
            "effective_date": self.effective_date,
            "evidence_span_ids": list(self.evidence_span_ids),
            "supplementary_amounts": [a.to_dict() for a in self.supplementary_amounts],
            "source_evidence": self.source_evidence,
            "confidence": self.confidence,
            "source_document_id": self.source_document_id,
            "accession": self.accession, "form": self.form,
            "filed": self.filed, "items": self.items,
        }


@dataclass(frozen=True)
class ResolvedFinancingEvent:
    """A validated candidate, joined with the deterministic item-code impact
    flags from `finance.structural_breaks.event_impact`.

    Phase-1 boundary: this is evidence and diagnostics only. It is surfaced
    to the report and to the assumption builder exactly as
    `finance.structural_breaks.PostBalanceSheetEvent` already is; it does not
    itself adjust `finance/net_debt.py` or any DCF input in this phase (spec
    section 35: "Do NOT change DCF mathematics"). Wiring a FUNDED, amount-
    grounded event into the numeric debt graph is future work once this
    reader has a live-shadow track record.
    """

    event_type: str
    amount: Optional[float]
    currency: Optional[str]
    # UNKNOWN when the candidate asserted none. Never populated with a
    # guess -- see the module docstring's "downgrading would let a wrong
    # claim through wearing a blank label" note.
    amount_role: str
    funded: bool
    committed: bool
    counterparty: Optional[str]
    effective_date: Optional[str]
    source_evidence: str
    confidence: float
    accession: Optional[str]
    form: Optional[str]
    filed: Optional[str]
    impact: dict
    supplementary_amounts: Tuple[ResolvedAmount, ...] = ()

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type, "amount": self.amount,
            "currency": self.currency, "amount_role": self.amount_role,
            "funded": self.funded,
            "committed": self.committed, "counterparty": self.counterparty,
            "effective_date": self.effective_date,
            "source_evidence": self.source_evidence,
            "supplementary_amounts": [a.to_dict() for a in self.supplementary_amounts],
            "confidence": self.confidence, "accession": self.accession,
            "form": self.form, "filed": self.filed, "impact": dict(self.impact),
        }
