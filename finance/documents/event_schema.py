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
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from finance.structural_breaks import PostBalanceSheetEventType


class EventRejectionCode:
    NO_EVIDENCE = "NO_EVIDENCE"
    EVIDENCE_NOT_IN_SOURCE = "EVIDENCE_NOT_IN_SOURCE"
    VALUE_NOT_IN_EVIDENCE = "VALUE_NOT_IN_EVIDENCE"
    UNSUPPORTED_EVENT_TYPE = "UNSUPPORTED_EVENT_TYPE"
    ITEM_CODE_MISMATCH = "ITEM_CODE_MISMATCH"
    FUNDED_STATUS_NOT_GROUNDED = "FUNDED_STATUS_NOT_GROUNDED"
    AMOUNT_NOT_GROUNDED = "AMOUNT_NOT_GROUNDED"
    UNKNOWN_CURRENCY = "UNKNOWN_CURRENCY"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MALFORMED = "MALFORMED"

    ALL = (NO_EVIDENCE, EVIDENCE_NOT_IN_SOURCE, VALUE_NOT_IN_EVIDENCE,
           UNSUPPORTED_EVENT_TYPE, ITEM_CODE_MISMATCH,
           FUNDED_STATUS_NOT_GROUNDED, AMOUNT_NOT_GROUNDED, UNKNOWN_CURRENCY,
           LOW_CONFIDENCE, MALFORMED)

    # The one a benchmark must count as critical (spec section 27): a
    # facility asserted funded with nothing in its own text supporting a
    # drawdown is `UNDRAWN_FACILITY_COUNTED_AS_FUNDED_DEBT` if it slipped
    # through; this is the code that stops it before that.
    UNSUPPORTED = (EVIDENCE_NOT_IN_SOURCE, VALUE_NOT_IN_EVIDENCE,
                   FUNDED_STATUS_NOT_GROUNDED, AMOUNT_NOT_GROUNDED,
                   ITEM_CODE_MISMATCH)


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
    currency: Optional[str] = None
    funded: bool = False
    committed: bool = False
    counterparty: Optional[str] = None
    effective_date: Optional[str] = None
    evidence_span_ids: Tuple[str, ...] = ()
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
            "unit": self.unit, "currency": self.currency,
            "funded": self.funded, "committed": self.committed,
            "counterparty": self.counterparty,
            "effective_date": self.effective_date,
            "evidence_span_ids": list(self.evidence_span_ids),
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

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type, "amount": self.amount,
            "currency": self.currency, "funded": self.funded,
            "committed": self.committed, "counterparty": self.counterparty,
            "effective_date": self.effective_date,
            "source_evidence": self.source_evidence,
            "confidence": self.confidence, "accession": self.accession,
            "form": self.form, "filed": self.filed, "impact": dict(self.impact),
        }
