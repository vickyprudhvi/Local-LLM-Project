"""The LLM boundary for ACTUAL statement lines: a controlled fallback.

WHEN THIS RUNS AT ALL

Only when `actuals_bridge.should_attempt_fallback` says the structured SEC
path AND the deterministic table reader in `finance/reported_actuals/`
together could not safely resolve the latest relevant period (spec section
5: "structured-first ... a controlled fallback, not an uncontrolled
replacement"). Most runs never construct this extractor.

WHAT THE MODEL READS

Not raw HTML. `finance/reported_actuals/tables.py::parse_filing_tables`
already parses the document's tables STRUCTURALLY -- caption, rows, period
columns, scale, currency -- with no financial interpretation. This module
renders each REPORTED-STATEMENT table (income statement, balance sheet, cash
flow; never a guidance table, never a non-GAAP reconciliation) as a small
plain-text grid and asks the model to identify which of a fixed set of line
items ('ACTUAL_METRIC_NAMES') each row is. Structure stays deterministic;
only "what does this row mean" is asked of the model, same division as
`finance/extraction/semantic_extractor.py`.

WHAT THE MODEL MAY NOT DO

No arithmetic (the rendered grid already carries the scaled currency amount
-- the model copies a cell, it does not compute one), no unit conversion, no
guess at a row the grid does not contain, and no proposal for a table the
deterministic classifier did not first accept as a reported statement.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import tools.config as config
from finance.documents.actuals_schema import (
    ACTUAL_METRIC_NAMES,
    ActualBasis,
    ActualFactCandidate,
    ActualPeriodType,
)
from finance.extraction.schema import MODEL_UNIT_NAMES
from finance.reported_actuals.tables import FilingTable, StatementKind


@dataclass(frozen=True)
class ActualsTableSection:
    """One reported-statement table, rendered as plain text."""

    label: str
    text: str
    table_index: int
    period_labels: Tuple[str, ...]
    statement_kind: str = ""


def render_table(table: FilingTable) -> str:
    """A deterministic plain-text grid. No financial interpretation."""
    lines = [f"Table caption: {table.caption}".strip()]
    if table.currency or table.scale:
        lines.append(f"Currency: {table.currency or 'UNKNOWN'}, "
                     f"Scale: {table.scale or 'units'}")
    if table.period_columns:
        lines.append("Columns: " + " | ".join(
            c.label or c.band_label or "?" for c in table.period_columns))
    for row in table.rows:
        cells = []
        for position in sorted(row.values):
            if position < len(table.period_columns):
                cells.append(str(row.values[position].number))
        if cells:
            lines.append(f"Row: {row.label}: " + " | ".join(cells))
    return "\n".join(lines)


def select_actual_sections(tables: Sequence[FilingTable],
                           max_sections: Optional[int] = None,
                           max_chars: Optional[int] = None
                           ) -> List[ActualsTableSection]:
    """Reported-statement tables only, rendered and bounded.

    A guidance table (`StatementKind.GUIDANCE`) or a non-GAAP reconciliation
    is never sent here -- exactly the boundary
    `finance/reported_actuals/facts.py::facts_from_table` already enforces
    for the deterministic path, so the two readers can never disagree about
    which tables are eligible.
    """
    max_sections = max_sections or config.finance_actuals_extraction_max_sections()
    max_chars = max_chars or config.finance_actuals_extraction_max_section_chars()
    out: List[ActualsTableSection] = []
    for table in tables:
        if table.kind not in StatementKind.REPORTED_STATEMENTS:
            continue
        if not table.rows:
            continue
        text = render_table(table)[:max_chars]
        out.append(ActualsTableSection(
            label=f"{table.kind} (table {table.index})", text=text,
            table_index=table.index,
            period_labels=tuple(c.label for c in table.period_columns),
            statement_kind=table.kind))
        if len(out) >= max_sections:
            break
    return out


_SYSTEM = (
    "You read one financial statement table, already extracted as a plain-"
    "text grid, and identify which REPORTED (not guided, not projected) line "
    "item each row is, from a FIXED list of names. You are a reader, not an "
    "analyst -- never compute a new number, never convert a unit, never "
    "report a row that is not in the grid.\n\n"
    "RULES:\n"
    "1. metric_id must be one of the METRIC NAMES listed below, or omit the "
    "row.\n"
    "2. Copy the number EXACTLY as printed in the grid into value. Do not "
    "rescale it -- the grid's Scale line already says what one unit means.\n"
    "3. source_evidence must be the EXACT 'Row: ...' line you read it from, "
    "copied verbatim.\n"
    "4. period_label is the column header you read the value from, copied "
    "verbatim from the Columns line.\n"
    "5. scope is CONSOLIDATED unless the row label itself says segment, "
    "region or discontinued operations, in which case omit the row.\n"
    "6. basis is GAAP unless the table caption says non-GAAP/adjusted.\n"
    "7. If you are unsure about any field, omit the row. A missed row costs "
    "nothing; a wrong one is used.\n\n"
    "Respond with ONLY a JSON object, no prose."
)


def _schema_text() -> str:
    return (
        '{"facts": [{'
        '"metric_id": "<one of the metric names listed below>", '
        '"value": <number, copied exactly>, '
        f'"unit": "{"|".join(MODEL_UNIT_NAMES)}", '
        '"currency": "<ISO code, e.g. USD>", '
        '"period_label": "<the exact column header>", '
        f'"period_type": "{"|".join(ActualPeriodType.ALL)}", '
        f'"basis": "{"|".join(ActualBasis.ALL)}", '
        '"scope": "CONSOLIDATED", '
        '"source_evidence": "<the exact Row: ... line, copied>", '
        '"confidence": <0.0 to 1.0>'
        '}]}'
    )


def build_prompt(section: ActualsTableSection, issued_at: Optional[str]
                 ) -> Tuple[str, str]:
    context = [f"Table: {section.label}"]
    if issued_at:
        context.append(f"Filed: {issued_at}")
    user = (
        f"{chr(10).join(context)}\n\n"
        f"METRIC NAMES you may use for metric_id:\n{', '.join(ACTUAL_METRIC_NAMES)}\n\n"
        f"JSON schema:\n{_schema_text()}\n\n"
        f"TABLE GRID:\n{section.text}"
    )
    return _SYSTEM, user


class ActualsSemanticExtractor(Protocol):
    version: str

    def extract(self, sections: Sequence[ActualsTableSection], *,
                issued_at: Optional[str] = None,
                document_id: Optional[str] = None,
                form: Optional[str] = None) -> List[ActualFactCandidate]:
        ...


class ActualsExtractionFailure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ActualsExtractionFailure("EMPTY_RESPONSE", "the model returned nothing")
    match = _JSON_BLOCK.search(text)
    if match is None:
        raise ActualsExtractionFailure("NO_JSON", "the response contained no JSON object")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ActualsExtractionFailure("INVALID_JSON", str(exc)) from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("facts"), list):
        raise ActualsExtractionFailure("SCHEMA", "'facts' must be a list")
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


def _candidate_from(raw: dict, section: ActualsTableSection,
                    issued_at: Optional[str], document_id: Optional[str],
                    form: Optional[str]) -> Optional[ActualFactCandidate]:
    if not isinstance(raw, dict):
        return None
    confidence = _as_float(raw.get("confidence"))
    return ActualFactCandidate(
        metric_id=(raw.get("metric_id") or None),
        value=_as_float(raw.get("value")),
        unit=(raw.get("unit") or None),
        currency=(raw.get("currency") or None),
        period_label=(raw.get("period_label") or None),
        period_type=str(raw.get("period_type") or ActualPeriodType.OTHER).upper(),
        basis=str(raw.get("basis") or ActualBasis.UNKNOWN).upper(),
        scope=str(raw.get("scope") or "CONSOLIDATED").upper(),
        prospective=False,
        source_evidence=str(raw.get("source_evidence") or ""),
        section_label=section.label,
        statement_kind=section.statement_kind,
        confidence=confidence if confidence is not None else 0.0,
        source_document_id=document_id, form=form, filed=issued_at)


class LocalModelActualsExtractor:
    """Backed by the project's injected model client. Same pattern as
    `finance.extraction.semantic_extractor.LocalModelGuidanceExtractor`."""

    version = "actuals-v1.0"

    def __init__(self, ask_local_fn: Callable, cache: Optional[dict] = None):
        self._ask = ask_local_fn
        self._cache = cache if cache is not None else {}

    def _cache_key(self, section: ActualsTableSection, issued_at: Optional[str]) -> str:
        digest = hashlib.sha256(section.text.encode("utf-8")).hexdigest()[:32]
        return "|".join([digest, section.label, issued_at or "", self.version,
                         config.finance_extraction_model_identity()])

    def _ask_once(self, system: str, user: str) -> str:
        response = self._ask(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            response_format="json",
            timeout=config.finance_actuals_extraction_timeout_seconds(),
            options={"num_predict": config.finance_actuals_extraction_max_output_tokens()})
        if not isinstance(response, dict) or not response.get("ok", True):
            raise ActualsExtractionFailure("MODEL_ERROR", "the model call did not succeed")
        metrics = response.get("metrics") or {}
        content = ((response.get("message") or {}).get("content")) or ""
        if metrics.get("truncated") and not content.strip():
            raise ActualsExtractionFailure(
                "TRUNCATED_RESPONSE",
                "the response hit the output-token limit without emitting content")
        return content

    def extract(self, sections: Sequence[ActualsTableSection], *,
                issued_at: Optional[str] = None,
                document_id: Optional[str] = None,
                form: Optional[str] = None) -> List[ActualFactCandidate]:
        candidates: List[ActualFactCandidate] = []
        for section in sections:
            key = self._cache_key(section, issued_at)
            if key in self._cache:
                payload = self._cache[key]
            else:
                system, user = build_prompt(section, issued_at)
                try:
                    payload = _parse_response(self._ask_once(system, user))
                except ActualsExtractionFailure as first:
                    correction = (
                        f"Your previous response could not be read "
                        f"({first.code}). Respond with ONLY the JSON object "
                        "described above.")
                    payload = _parse_response(
                        self._ask_once(system, f"{user}\n\n{correction}"))
                self._cache[key] = payload
            for raw in payload.get("facts") or []:
                candidate = _candidate_from(raw, section, issued_at, document_id, form)
                if candidate is not None:
                    candidates.append(candidate)
        return candidates
