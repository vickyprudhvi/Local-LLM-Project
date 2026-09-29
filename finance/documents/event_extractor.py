"""The LLM boundary for a financing/capital-structure EVENT.

Runs only on documents `finance.documents.package` already classified
`FINANCING_EVENT` -- an 8-K whose OWN item codes are eligible
(`finance.documents.package.FINANCING_EVENT_ITEMS`). Eligibility is
deterministic; what the filing's own text actually commits the company to
(an amount, a currency, whether it was drawn) is read here and never
inferred from the item code alone (spec section 8).

PHASE H.18 -- EVIDENCE IS SELECTED, NOT TRANSCRIBED

The model is shown the section as a list of NUMBERED SPANS
(`finance.documents.spans.build_source_spans`, cut before the model call)
and returns `evidence_span_ids` -- which spans support the candidate. It
never returns evidence text. See `finance.documents.event_schema`'s and
`finance.documents.spans`' module docstrings for why: transcription is where
the live benchmark found the model eliding a clause of a dense sentence,
and a candidate cannot fabricate a span id the way it could fabricate a
"close enough" quotation.

PHASE H.19 -- NORMALIZE BEFORE BOUNDING

`select_event_sections` used to bound a RAW, un-stripped document to N
characters before doing anything else -- and a real 8-K's raw document
opens with inline-XBRL's own hidden-facts header, so that bound was mostly
spent on markup no reader should ever see. It now runs
`finance.documents.text_normalization.normalize_sec_document` (readable
text) and `select_relevant_item_blocks` (one section per financing-eligible
Item code) FIRST, and only bounds the SELECTED, NORMALIZED block. See that
module's docstring for the measured before/after on a real filing.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import tools.config as config
from finance.documents.event_schema import EventAmount, EventAmountRole, EventCandidate
from finance.documents.monetary import MonetaryScale
from finance.documents.spans import SourceSpan, build_source_spans, render_spans
from finance.documents.text_normalization import (
    NormalizationFailure,
    normalize_sec_document,
    select_relevant_item_blocks,
)
from finance.structural_breaks import PostBalanceSheetEventType

# Phase H.22: a currency-neutral unit vocabulary for the reader contract.
# CURRENCY replaces USD/USD_MILLION/USD_BILLION as what the model is OFFERED
# for a monetary amount -- currency and scale are separate fields, never
# baked into the unit name (see `finance.documents.monetary`'s docstring for
# the live defect that coupling produced: a EUR amount with no non-USD
# scale-bearing unit to choose fell back to UNKNOWN, silently losing its
# scale). `EventCandidateValidator` still ACCEPTS the legacy USD_MILLION/
# USD_BILLION/USD tokens for backward compatibility; they are simply no
# longer what this prompt asks the model to produce.
_MODEL_UNIT_NAMES = ("CURRENCY", "PER_SHARE", "RATIO", "PERCENT", "SHARES", "UNKNOWN")

# The item codes a financing-event reader ever needs to see. Duplicated as a
# bare tuple (rather than importing `finance.documents.package.
# FINANCING_EVENT_ITEMS`) to avoid a package.py <-> event_extractor.py
# import edge -- package.py is the deterministic ELIGIBILITY decision this
# already agrees with, and a test pins the two lists equal.
FINANCING_EVENT_ITEMS = ("1.01", "2.01", "2.03", "2.04", "3.02")


@dataclass(frozen=True)
class EventDocumentSection:
    label: str
    text: str
    spans: Tuple[SourceSpan, ...] = ()
    item_code: Optional[str] = None


def select_event_sections(text: str, max_chars: Optional[int] = None,
                          max_spans: Optional[int] = None,
                          max_sections: Optional[int] = None
                          ) -> Tuple[List[EventDocumentSection], Optional[str]]:
    """Normalized, Item-block-selected sections. Never a raw prefix.

    Returns `(sections, failure_code)`. `failure_code` (from
    `finance.documents.text_normalization.NormalizationFailure`) is set,
    and `sections` is empty, whenever there is nothing safe to show the
    model: the document did not normalize to readable text at all, or it
    normalized fine but names no financing-eligible Item code. Section 11's
    fail-closed rule -- no candidates, a structured reason, never a silent
    fall back to raw markup.
    """
    max_chars = max_chars or config.finance_event_extraction_max_section_chars()
    max_sections = max_sections or config.finance_event_extraction_max_sections()

    normalized = normalize_sec_document(text)
    if not normalized.ok:
        return [], normalized.failure_code

    blocks = select_relevant_item_blocks(
        normalized.text, FINANCING_EVENT_ITEMS, max_blocks=max_sections,
        max_chars_per_block=max_chars)
    if not blocks:
        return [], NormalizationFailure.NO_RELEVANT_ITEM_SECTION

    sections = [
        EventDocumentSection(
            label=f"Item {block.item_code}", text=block.text,
            spans=tuple(build_source_spans(block.text, max_spans=max_spans)),
            item_code=block.item_code)
        for block in blocks]
    return sections, None


_SYSTEM = (
    "You read one SEC 8-K filing about a financing or capital-structure "
    "event, already split into NUMBERED SOURCE SPANS, and report the "
    "STRUCTURED facts it states. You are a reader, not an analyst.\n\n"
    "THE MOST IMPORTANT DISTINCTION: a facility being ESTABLISHED (a company "
    "signs a credit agreement, is granted access to borrow) is NOT the same "
    "as money being RECEIVED (the company actually drew on it, issued notes "
    "and received proceeds, or closed a sale). Set funded=true ONLY when a "
    "span says money was actually drawn, borrowed, issued and sold, or "
    "received as proceeds -- and that span must be one of your "
    "evidence_span_ids. Set funded=false, committed=true when a span "
    "describes a facility, commitment or availability with no drawdown "
    "stated. If nothing you cite says anything was drawn, use funded=false.\n\n"
    "EVIDENCE RULES -- THESE MATTER MOST:\n"
    "1. You do NOT write evidence. You SELECT it. evidence_span_ids is a "
    "list of the exact span ids shown in the SOURCE SPANS list below (copy "
    "the id in brackets, e.g. [id] text -> the id is what goes in "
    "evidence_span_ids) that support this candidate. Never put sentence "
    "text in evidence_span_ids and never invent an id that is not in the "
    "list you were given.\n"
    "2. NEVER rewrite, summarize, shorten, or normalize a span's wording "
    "anywhere in your answer. You are not asked to reproduce any text at "
    "all -- only to point at spans.\n"
    "3. Cite EVERY span you actually relied on. The event, the amount and "
    "the funded/committed status may each come from a DIFFERENT span (a "
    "compound sentence, or separate sentences) -- cite all of them, not "
    "just one.\n"
    "4. If no span you were given fully supports a property (amount, "
    "funded status, event type), leave that property null/false rather "
    "than guessing, or omit the whole candidate if nothing supports it.\n\n"
    "A NUMBER EXISTING IS NOT ENOUGH -- YOU MUST ALSO SAY WHAT IT MEANS:\n"
    "9. A document routinely states SEVERAL economically different amounts: "
    "a per-share price is not the deal's total value; a facility's total "
    "commitment is not the amount actually drawn; a note's principal is not "
    "a repayment amount. amount_role names which one your PRIMARY amount "
    "is. Use UNKNOWN if no span makes the role clear -- never guess a role "
    "to fill in the field.\n"
    "10. NEVER compute one amount from another. If a span says '$95.00 per "
    "share' and a DIFFERENT span separately states an aggregate/total "
    "dollar figure, report them as TWO separate amounts with two roles -- "
    "do not multiply a per-share price by a share count, and do not add, "
    "subtract or otherwise calculate any amount that is not itself written "
    "as a single number in a span.\n"
    "11. If the document states more than one economically distinct amount "
    "worth recording (a commitment AND a drawn amount; a per-share price "
    "AND an explicitly stated total value), report the primary one as "
    "amount/amount_role and any others in supplementary_amounts, each with "
    "its OWN role and its OWN evidence_span_ids. Do not omit a second "
    "amount just because the schema has one primary slot.\n\n"
    "OTHER RULES:\n"
    "5. event_type must be one of the EVENT TYPES listed below.\n"
    "6. Never infer a unit or currency; if no span states one, use UNKNOWN. "
    "A per-share price uses unit PER_SHARE, never a bare currency unit.\n"
    "7. counterparty is the lender/underwriter/counterparty a cited span "
    "names, or null.\n"
    "8. effective_date is the date a cited span states, or null.\n\n"
    "CURRENCY AND SCALE ARE TWO SEPARATE FACTS -- NEVER LET ONE BLOCK THE "
    "OTHER:\n"
    "12. currency (an ISO code like USD, EUR, GBP, JPY) and scale (the "
    "magnitude word a span actually uses) are INDEPENDENT properties of a "
    "monetary amount. Report each from what its OWN language in the cited "
    "spans states, regardless of what the other one is. A non-USD currency "
    "is NOT a reason to leave scale unstated, and an unfamiliar scale is "
    "NOT a reason to leave currency unstated.\n"
    "13. scale is exactly one of: UNIT (no magnitude word -- the span "
    "states a bare number, e.g. \"$500\"), THOUSAND, MILLION, or BILLION "
    "(the span literally says that word), or UNKNOWN if a span clearly "
    "states a currency amount but its magnitude word is genuinely "
    "unclear. NEVER default scale to UNIT just because you are not sure -- "
    "an uncertain scale must be UNKNOWN, never guessed.\n"
    "14. Worked example -- \"€500 million of senior notes\": "
    "value=500, currency=EUR, scale=MILLION, unit=CURRENCY, "
    "role=PRINCIPAL_AMOUNT. The unfamiliar currency (EUR, not USD) does "
    "NOT change how you read the scale word \"million\" -- it is stated "
    "in the same span regardless of currency.\n"
    "15. scale only applies to a currency-domain amount (unit=CURRENCY). "
    "Leave it UNIT for a PER_SHARE/RATIO/PERCENT/SHARES amount -- those "
    "have no thousand/million/billion magnitude to state.\n\n"
    "Respond with ONLY a JSON object, no prose."
)


def _amount_schema_text() -> str:
    return ('{"value": <number or null>, '
           f'"unit": "{"|".join(_MODEL_UNIT_NAMES)}", '
           '"currency": "<ISO code or UNKNOWN>", '
           f'"scale": "{"|".join(MonetaryScale.ALL)}", '
           f'"role": "{"|".join(EventAmountRole.ALL)}", '
           '"evidence_span_ids": ["<span id>", "..."]}')


def _schema_text() -> str:
    return (
        '{"events": [{'
        f'"event_type": "{"|".join(PostBalanceSheetEventType.ALL)}", '
        '"amount": <number or null>, '
        f'"unit": "{"|".join(_MODEL_UNIT_NAMES)}", '
        f'"amount_role": "{"|".join(EventAmountRole.ALL)}", '
        '"currency": "<ISO code or UNKNOWN>", '
        f'"scale": "{"|".join(MonetaryScale.ALL)}", '
        '"funded": <true|false>, '
        '"committed": <true|false>, '
        '"counterparty": "<name or null>", '
        '"effective_date": "<YYYY-MM-DD or null>", '
        '"evidence_span_ids": ["<one of the span ids shown below>", "..."], '
        f'"supplementary_amounts": [{_amount_schema_text()}, "..."], '
        '"confidence": <0.0 to 1.0>'
        '}]}'
    )


def build_prompt(section: EventDocumentSection, items: Optional[str],
                 filed: Optional[str]) -> Tuple[str, str]:
    context = []
    if items:
        context.append(f"8-K item(s): {items}")
    if filed:
        context.append(f"Filed: {filed}")
    user = (
        f"{chr(10).join(context)}\n\n"
        f"EVENT TYPES you may use for event_type:\n{', '.join(PostBalanceSheetEventType.ALL)}\n\n"
        f"JSON schema:\n{_schema_text()}\n\n"
        f"SOURCE SPANS (cite by id; do not copy their text):\n{render_spans(section.spans)}"
    )
    return _SYSTEM, user


class EventSemanticExtractor(Protocol):
    version: str

    def extract(self, sections: Sequence[EventDocumentSection], *,
                items: Optional[str] = None, filed: Optional[str] = None,
                accession: Optional[str] = None,
                document_id: Optional[str] = None) -> List[EventCandidate]:
        ...


class EventExtractionFailure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise EventExtractionFailure("EMPTY_RESPONSE", "the model returned nothing")
    match = _JSON_BLOCK.search(text)
    if match is None:
        raise EventExtractionFailure("NO_JSON", "the response contained no JSON object")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise EventExtractionFailure("INVALID_JSON", str(exc)) from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("events"), list):
        raise EventExtractionFailure("SCHEMA", "'events' must be a list")
    return parsed


def _as_float(value) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("$", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _as_span_ids(value) -> Tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ()
    return tuple(str(v).strip() for v in value if isinstance(v, (str, int)) and str(v).strip())


def _amount_from(raw) -> Optional[EventAmount]:
    if not isinstance(raw, dict):
        return None
    value = _as_float(raw.get("value"))
    span_ids = _as_span_ids(raw.get("evidence_span_ids"))
    if value is None and not span_ids:
        return None
    return EventAmount(
        role=str(raw.get("role") or EventAmountRole.UNKNOWN).upper(),
        value=value, unit=(raw.get("unit") or None),
        currency=(raw.get("currency") or None),
        scale=(raw.get("scale") or None),
        evidence_span_ids=span_ids)


def _supplementary_amounts_from(raw) -> Tuple[EventAmount, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(a for a in (_amount_from(item) for item in raw) if a is not None)


def _candidate_from(raw: dict, items: Optional[str], filed: Optional[str],
                    accession: Optional[str], document_id: Optional[str]
                    ) -> Optional[EventCandidate]:
    if not isinstance(raw, dict):
        return None
    confidence = _as_float(raw.get("confidence"))
    return EventCandidate(
        event_type=str(raw.get("event_type") or PostBalanceSheetEventType.UNKNOWN).upper(),
        amount=_as_float(raw.get("amount")),
        unit=(raw.get("unit") or None),
        amount_role=str(raw.get("amount_role") or EventAmountRole.UNKNOWN).upper(),
        currency=(raw.get("currency") or None),
        scale=(raw.get("scale") or None),
        funded=bool(raw.get("funded", False)),
        committed=bool(raw.get("committed", False)),
        counterparty=(raw.get("counterparty") or None),
        effective_date=(raw.get("effective_date") or None),
        evidence_span_ids=_as_span_ids(raw.get("evidence_span_ids")),
        supplementary_amounts=_supplementary_amounts_from(raw.get("supplementary_amounts")),
        # source_evidence is deliberately NOT read from the model's JSON --
        # the schema no longer even asks for it. It stays "" until
        # `event_validator.py` reconstructs it from the cited spans.
        confidence=confidence if confidence is not None else 0.0,
        source_document_id=document_id, accession=accession, form="8-K",
        filed=filed, items=items)


class LocalModelEventExtractor:
    """Backed by the project's injected model client. Same pattern as
    `finance.extraction.semantic_extractor.LocalModelGuidanceExtractor`."""

    version = "events-v4.0"

    def __init__(self, ask_local_fn: Callable, cache: Optional[dict] = None):
        self._ask = ask_local_fn
        self._cache = cache if cache is not None else {}

    def _cache_key(self, section: EventDocumentSection, filed: Optional[str]) -> str:
        digest = hashlib.sha256(section.text.encode("utf-8")).hexdigest()[:32]
        return "|".join([digest, filed or "", self.version,
                         config.finance_extraction_model_identity()])

    def _ask_once(self, system: str, user: str) -> str:
        response = self._ask(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            response_format="json",
            timeout=config.finance_event_extraction_timeout_seconds(),
            options={"num_predict": config.finance_event_extraction_max_output_tokens()})
        if not isinstance(response, dict) or not response.get("ok", True):
            raise EventExtractionFailure("MODEL_ERROR", "the model call did not succeed")
        metrics = response.get("metrics") or {}
        content = ((response.get("message") or {}).get("content")) or ""
        if metrics.get("truncated") and not content.strip():
            raise EventExtractionFailure(
                "TRUNCATED_RESPONSE",
                "the response hit the output-token limit without emitting content")
        return content

    def extract(self, sections: Sequence[EventDocumentSection], *,
                items: Optional[str] = None, filed: Optional[str] = None,
                accession: Optional[str] = None,
                document_id: Optional[str] = None) -> List[EventCandidate]:
        candidates: List[EventCandidate] = []
        for section in sections:
            if not section.spans:
                continue
            key = self._cache_key(section, filed)
            if key in self._cache:
                payload = self._cache[key]
            else:
                system, user = build_prompt(section, items, filed)
                try:
                    payload = _parse_response(self._ask_once(system, user))
                except EventExtractionFailure as first:
                    correction = (
                        f"Your previous response could not be read "
                        f"({first.code}). Respond with ONLY the JSON object "
                        "described above.")
                    payload = _parse_response(
                        self._ask_once(system, f"{user}\n\n{correction}"))
                self._cache[key] = payload
            for raw in payload.get("events") or []:
                candidate = _candidate_from(raw, items, filed, accession, document_id)
                if candidate is not None:
                    candidates.append(candidate)
        return candidates
