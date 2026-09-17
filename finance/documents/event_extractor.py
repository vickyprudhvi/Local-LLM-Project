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
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import tools.config as config
from finance.documents.event_schema import EventCandidate
from finance.documents.spans import SourceSpan, build_source_spans, render_spans
from finance.extraction.schema import MODEL_UNIT_NAMES
from finance.structural_breaks import PostBalanceSheetEventType


@dataclass(frozen=True)
class EventDocumentSection:
    label: str
    text: str
    spans: Tuple[SourceSpan, ...] = ()


def select_event_sections(text: str, max_chars: Optional[int] = None,
                          max_spans: Optional[int] = None
                          ) -> List[EventDocumentSection]:
    """The whole filing body, bounded, pre-split into citable spans.

    An 8-K item narrative is short; there is no heading structure worth
    parsing the way a guidance release has, so this stays one section --
    but that section now carries its own span list, built once here rather
    than separately by every caller that needs to validate against it.
    """
    if not (text or "").strip():
        return []
    max_chars = max_chars or config.finance_event_extraction_max_section_chars()
    bounded = text[:max_chars]
    spans = build_source_spans(bounded, max_spans=max_spans)
    return [EventDocumentSection(label="Filing body", text=bounded, spans=tuple(spans))]


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
    "OTHER RULES:\n"
    "5. event_type must be one of the EVENT TYPES listed below.\n"
    "6. Never infer a unit or currency; if no span states one, use UNKNOWN.\n"
    "7. counterparty is the lender/underwriter/counterparty a cited span "
    "names, or null.\n"
    "8. effective_date is the date a cited span states, or null.\n\n"
    "Respond with ONLY a JSON object, no prose."
)


def _schema_text() -> str:
    return (
        '{"events": [{'
        f'"event_type": "{"|".join(PostBalanceSheetEventType.ALL)}", '
        '"amount": <number or null>, '
        f'"unit": "{"|".join(MODEL_UNIT_NAMES)}", '
        '"currency": "<ISO code or UNKNOWN>", '
        '"funded": <true|false>, '
        '"committed": <true|false>, '
        '"counterparty": "<name or null>", '
        '"effective_date": "<YYYY-MM-DD or null>", '
        '"evidence_span_ids": ["<one of the span ids shown below>", "..."], '
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
        currency=(raw.get("currency") or None),
        funded=bool(raw.get("funded", False)),
        committed=bool(raw.get("committed", False)),
        counterparty=(raw.get("counterparty") or None),
        effective_date=(raw.get("effective_date") or None),
        evidence_span_ids=_as_span_ids(raw.get("evidence_span_ids")),
        # source_evidence is deliberately NOT read from the model's JSON --
        # the schema no longer even asks for it. It stays "" until
        # `event_validator.py` reconstructs it from the cited spans.
        confidence=confidence if confidence is not None else 0.0,
        source_document_id=document_id, accession=accession, form="8-K",
        filed=filed, items=items)


class LocalModelEventExtractor:
    """Backed by the project's injected model client. Same pattern as
    `finance.extraction.semantic_extractor.LocalModelGuidanceExtractor`."""

    version = "events-v2.0"

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
