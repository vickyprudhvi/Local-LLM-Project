"""The LLM boundary: a semantic READER of guidance sentences.

WHAT THE MODEL IS FOR

Reading an English sentence and saying what quantity management committed to.
That is the one thing here a regex cannot do, and the one thing this module
asks a model to do. Everything else -- which document, which period, whether
the answer is acceptable -- is decided elsewhere by code.

The model NEVER:
    computes a DCF assumption, decides valuation eligibility, judges a
    company, infers a missing unit, invents a target period, converts a
    margin into an amount or an amount into a margin, treats a historical
    table as guidance, treats an analyst estimate as management guidance, or
    fabricates a value.

It proposes candidates. `validator.py` accepts or refuses them, and a
candidate that cannot point at the sentence it came from is refused before
anything else is checked.

PROVIDER NEUTRALITY

`GuidanceSemanticExtractor` is the interface. The implementation takes an
injected `ask_local_fn` -- the same pattern `synthesize_report` uses so that
`finance/` never imports a model client and never learns a vendor's name.
Domain validation is downstream of the interface and sees only candidates.

BOUNDING

Deterministic section selection runs FIRST, so the model reads an outlook
section rather than a whole filing. Sections, characters and attempts are all
capped, and one structured repair is allowed for malformed JSON -- the same
one-repair convention `research_pipeline` already uses.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import tools.config as config
from finance.extraction import EXTRACTOR_VERSION, SCHEMA_VERSION
from finance.extraction.schema import (
    MODEL_PERIOD_TYPE_NAMES,
    MODEL_UNIT_NAMES,
    REPORTED_RESULTS_MARKER,
    GuidanceAction,
    GuidanceCandidate,
    ToleranceBasis,
    ToleranceUnit,
    ValueType,
)


# ---------------------------------------------------------------------------
# Deterministic section selection (a KEEP use of regex: boundaries, not meaning)
# ---------------------------------------------------------------------------

# Headings that introduce forward-looking commentary. Matching a HEADING is a
# boundary question -- where does this section start -- not a semantic one,
# which is why a pattern is the right instrument here and the wrong one for
# the sentences inside.
_SECTION_HEADINGS = re.compile(
    r"(?im)^[^\S\n]*(?P<label>"
    r"(?:financial\s+)?outlook"
    r"|business\s+outlook"
    r"|(?:financial\s+|full[\s-]year\s+|updated\s+)?guidance"
    r"|guidance\s+summary"
    r"|forward[\s-]looking\s+information"
    r"|management\s+commentary"
    r"|(?:ceo|cfo|chief\s+\w+\s+officer)\s+commentary"
    r"|conference\s+call\s+commentary"
    r")\s*:?\s*$")

# When no heading is found, sentences carrying forward vocabulary are
# gathered instead. Many releases state guidance in a paragraph with no
# heading at all, and refusing to look at them would trade one brittleness
# for another.
_FORWARD_SENTENCE = re.compile(
    r"(?i)\b(?:expect\w*|guidance|outlook|anticipat\w*|forecast\w*|project\w*|"
    r"target\w*|reaffirm\w*|reiterat\w*|confirm\w*|rais\w*\s+(?:its|our)|"
    r"lower\w*\s+(?:its|our)|withdraw\w*|plans?\s+to)\b")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# A flattened TABLE ROW is short; a paragraph is not. That is the whole rule,
# and it is a boundary question rather than a semantic one -- it decides where
# a region stops, never what a row means.
#
# It exists because the fallback below filters on forward-looking vocabulary,
# and a table row has none. "Revenues $9.7B - $10.5B." says "Revenues", never
# "we expect revenues", so an issuer who TABULATES its guidance had every row
# holding a figure dropped and only the prose caption kept. The model was then
# handed the safe-harbor boilerplate and asked to find guidance in it.
_TABLE_ROW_MAX_WORDS = 12
_TABLE_RUN_MAX_ROWS = 40
# A table is a REPETITION of short rows. One short sentence after a paragraph
# is a sentence -- "Shares outstanding were 139,933." following a guidance
# sentence is a share count, not a guidance row, and absorbing it would drag
# reported history into a prompt asking for forward-looking statements.
# Requiring a RUN, most of it numeric, is what separates the two structurally
# without reading either.
_TABLE_RUN_MIN_ROWS = 3
_TABLE_RUN_MIN_NUMERIC_ROWS = 2
_HAS_DIGIT = re.compile(r"\d")


def _looks_like_a_table_row(fragment: str) -> bool:
    words = fragment.split()
    return 0 < len(words) <= _TABLE_ROW_MAX_WORDS


@dataclass(frozen=True)
class DocumentSection:
    """A bounded slice of a document, with where it came from."""

    label: str
    text: str
    start: int
    end: int


def select_sections(text: str,
                    max_sections: Optional[int] = None,
                    max_chars: Optional[int] = None) -> List[DocumentSection]:
    """The parts of a document worth reading, newest-relevant first.

    Headings first, because a release that labels its outlook is telling us
    exactly where to look. Forward-looking sentences second, for releases
    that do not. Both are bounded: a whole 10-K is never sent.
    """
    if not text:
        return []
    max_sections = max_sections or config.finance_extraction_max_sections()
    max_chars = max_chars or config.finance_extraction_max_section_chars()

    sections: List[DocumentSection] = []
    matches = list(_SECTION_HEADINGS.finditer(text))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:min(end, start + max_chars)].strip()
        if body:
            sections.append(DocumentSection(
                label=" ".join(match.group("label").split()).title(),
                text=body, start=start, end=start + len(body)))
        if len(sections) >= max_sections:
            break

    if sections:
        return sections

    # No heading. Gather the forward-looking sentences with a little context
    # around each, still bounded.
    gathered, used = [], 0
    fragments = _SENTENCE_SPLIT.split(text)
    index = 0
    while index < len(fragments):
        if used >= max_chars or len(gathered) >= max_sections * 8:
            break
        fragment = fragments[index]
        index += 1
        if not _FORWARD_SENTENCE.search(fragment):
            continue
        cleaned = " ".join(fragment.split())
        gathered.append(cleaned)
        # +1 for the newline this fragment costs in the joined body. `used`
        # has to measure the STRING that gets sent, not the sum of the pieces,
        # or the section overruns `max_chars` by one per fragment.
        used += len(cleaned) + 1

        # A guidance sentence is very often the CAPTION of a table, and the
        # rows under it carry the numbers this whole layer exists to read.
        # Absorb the run of short fragments that follows, stopping at the
        # first one long enough to be prose -- which is where the table ends
        # and the safe-harbor footnote begins.
        run_start, run, numeric_rows = index, [], 0
        while index < len(fragments) and len(run) < _TABLE_RUN_MAX_ROWS:
            row = " ".join(fragments[index].split())
            if not row or not _looks_like_a_table_row(row):
                break
            # A reported-results caption ENDS the run. Without this the rule
            # walked a guidance sentence straight into the condensed income
            # statement below it -- "Three Months Ended April 30", "Total
            # revenue $4,571,779" -- and sent reported history to a model
            # asked for an outlook. Measured: it put a results table in the
            # prompt for a retail release and the reader then published a
            # component metric as consolidated revenue growth.
            if REPORTED_RESULTS_MARKER.search(row):
                break
            if used + sum(len(r) + 1 for r in run) + len(row) + 1 > max_chars:
                break
            run.append(row)
            numeric_rows += bool(_HAS_DIGIT.search(row))
            index += 1
        # Kept only when the run actually looks like a table: several rows,
        # more than one of them carrying a figure. A caption followed by a few
        # terse sentences is not a table, and pulling those in would be
        # padding rather than evidence.
        if (len(run) >= _TABLE_RUN_MIN_ROWS
                and numeric_rows >= _TABLE_RUN_MIN_NUMERIC_ROWS):
            gathered.extend(run)
            used += sum(len(r) + 1 for r in run)
        else:
            index = run_start
    if not gathered:
        return []
    body = "\n".join(gathered)
    position = text.find(gathered[0][:40]) if gathered else 0
    return [DocumentSection(label="Forward-looking statements", text=body,
                            start=max(position, 0),
                            end=max(position, 0) + len(body))]


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You read one section of a company's earnings release and report the "
    "FORWARD-LOOKING statements management makes about future financial "
    "periods. You are a reader, not an analyst.\n\n"
    "Report a statement ONLY when management commits the company to a figure "
    "for a future period. Do NOT report:\n"
    "- figures for periods that have already been reported (results tables, "
    "condensed statements, reconciliations, share-count tables)\n"
    "- analyst or consensus estimates\n"
    "- statements with no number\n"
    "- anything you inferred rather than read\n\n"
    "RULES THAT MATTER MOST:\n"
    "1. Copy the exact sentence into source_sentence. If you cannot, omit "
    "the statement.\n"
    "2. Never infer a unit. If the text says 'percent', the unit is PERCENT. "
    "If it says '$90 billion', the unit is USD_BILLION. If you cannot tell, "
    "use UNKNOWN.\n"
    "3. A percentage OF SOMETHING is a ratio and must name what it is a "
    "ratio of in denominator_metric. 'operating income of 21% of revenue' is "
    "value_type=margin, denominator_metric=revenue -- NOT an amount.\n"
    "4. Never convert. A margin stays a margin; an amount stays an amount.\n"
    "4b. NEVER COMPUTE AN ENDPOINT. For 'approximately $91.0 billion, plus or "
    "minus 2%' report value_type=tolerance, value=91.0, unit=USD_BILLION, "
    "tolerance_value=2, tolerance_unit=PERCENT, "
    "tolerance_basis=OF_POINT_VALUE, and leave low and high null. Do NOT "
    "report 89.18 and 92.82 -- those are not in the text and will be "
    "refused. The endpoints are calculated for you.\n"
    "4c. tolerance_basis says what the tolerance is measured against. "
    "'$91B plus or minus 2%' is OF_POINT_VALUE. 'margin of 75.0% plus or "
    "minus 50 bps' is ABSOLUTE with tolerance_unit=BASIS_POINTS, giving 74.5 "
    "to 75.5. 'EPS of $2.50 plus or minus $0.10' is ABSOLUTE with "
    "tolerance_unit=SAME_AS_POINT.\n"
    "4d. A range the company STATES -- '$89 billion to $93 billion' -- is "
    "value_type=range with low and high copied from the text. Use tolerance "
    "ONLY where the text says plus or minus.\n"
    "5. target_period is the period being GUIDED, never the period it is "
    "compared against. In 'Q1 FY2027 revenue growth of 8% compared with Q1 "
    "FY2026', the target is Q1 FY2027 and the comparison is Q1 FY2026.\n"
    "6. action says what this does to an earlier statement: NEW, RAISED, "
    "LOWERED, REITERATED, REAFFIRMED, WITHDRAWN, or UNKNOWN.\n"
    "7. If you are unsure about any field, use null or UNKNOWN and lower "
    "confidence. An omitted statement costs nothing; a wrong one is used.\n\n"
    "Respond with ONLY a JSON object, no prose."
)


def _schema_text() -> str:
    return (
        '{"statements": [{'
        '"metric_id": "<one of the metric names listed below>", '
        f'"value_type": "{"|".join(ValueType.ALL)}", '
        '"low": <number or null>, "high": <number or null>, '
        '"value": <number or null>, '
        f'"unit": "{"|".join(MODEL_UNIT_NAMES)}", '
        '"denominator_metric": "<metric name or null>", '
        '"target_period": "<e.g. FY2027 or Q1 FY2027>", '
        f'"target_period_type": "{"|".join(MODEL_PERIOD_TYPE_NAMES)}", '
        '"comparison_period": "<period or null>", '
        '"tolerance_value": <number or null>, '
        f'"tolerance_unit": "{"|".join(ToleranceUnit.ALL)}|null", '
        f'"tolerance_basis": "{"|".join(ToleranceBasis.ALL)}", '
        '"basis": "GAAP|NON_GAAP|ADJUSTED|UNKNOWN", '
        f'"action": "{"|".join(GuidanceAction.ALL)}", '
        '"prospective": true, '
        '"source_sentence": "<the exact sentence, copied>", '
        '"confidence": <0.0 to 1.0>'
        '}]}'
    )


def _metric_vocabulary() -> str:
    from finance import guidance as gm

    names = sorted({value for key, value in vars(gm.GuidanceMetricName).items()
                    if not key.startswith("_") and isinstance(value, str)})
    return ", ".join(names)


def build_prompt(section: DocumentSection, issued_at: Optional[str],
                 fiscal_year_hint: Optional[int] = None) -> Tuple[str, str]:
    context = [f"Section: {section.label}"]
    if issued_at:
        context.append(f"Released: {issued_at}")
    if fiscal_year_hint:
        context.append(f"Calendar year at release: {fiscal_year_hint}")
    user = (
        f"{chr(10).join(context)}\n\n"
        f"METRIC NAMES you may use for metric_id:\n{_metric_vocabulary()}\n\n"
        f"JSON schema:\n{_schema_text()}\n\n"
        f"SECTION TEXT:\n{section.text}"
    )
    return _SYSTEM, user


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------

class GuidanceSemanticExtractor(Protocol):
    """Reads document sections, returns candidates. Nothing else."""

    version: str

    def extract(self, sections: Sequence[DocumentSection], *,
                issued_at: Optional[str] = None,
                document_id: Optional[str] = None) -> List[GuidanceCandidate]:
        ...


class ExtractionFailure(Exception):
    """The model could not be read. Never a reason to fall back silently.

    §20: a failed semantic extraction must not quietly become an unvalidated
    regex extraction. The caller decides what to do and the provenance says
    which layer answered.
    """

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ExtractionFailure("EMPTY_RESPONSE", "the model returned nothing")
    match = _JSON_BLOCK.search(text)
    if match is None:
        raise ExtractionFailure("NO_JSON", "the response contained no JSON object")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ExtractionFailure("INVALID_JSON", str(exc)) from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("statements"), list):
        raise ExtractionFailure("SCHEMA", "'statements' must be a list")
    return parsed


def _as_float(value) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("$", "").replace("%", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _candidate_from(raw: dict, section: DocumentSection,
                    issued_at: Optional[str], document_id: Optional[str]
                    ) -> Optional[GuidanceCandidate]:
    if not isinstance(raw, dict):
        return None
    confidence = _as_float(raw.get("confidence"))
    return GuidanceCandidate(
        metric_id=(raw.get("metric_id") or None),
        value_type=str(raw.get("value_type") or ValueType.UNKNOWN).lower(),
        low=_as_float(raw.get("low")), high=_as_float(raw.get("high")),
        value=_as_float(raw.get("value")),
        unit=(raw.get("unit") or None),
        denominator_metric=(raw.get("denominator_metric") or None),
        target_period=(raw.get("target_period") or None),
        target_period_type=(raw.get("target_period_type") or None),
        comparison_period=(raw.get("comparison_period") or None),
        basis=(raw.get("basis") or None),
        tolerance_value=_as_float(raw.get("tolerance_value")),
        tolerance_unit=((raw.get("tolerance_unit") or "").upper() or None),
        tolerance_basis=str(raw.get("tolerance_basis")
                            or ToleranceBasis.UNKNOWN).upper(),
        action=str(raw.get("action") or GuidanceAction.UNKNOWN).upper(),
        prospective=bool(raw.get("prospective", True)),
        source_sentence=str(raw.get("source_sentence") or ""),
        section_label=section.label,
        confidence=confidence if confidence is not None else 0.0,
        document_id=document_id, issued_at=issued_at)


class LocalModelGuidanceExtractor:
    """The interface, backed by the project's injected model client.

    `ask_local_fn` has the signature `synthesize_report` already injects, so
    this module -- like the rest of `finance/` -- never imports a client and
    never names a vendor.
    """

    version = EXTRACTOR_VERSION

    def __init__(self, ask_local_fn: Callable, cache: Optional[dict] = None):
        self._ask = ask_local_fn
        self._cache = cache if cache is not None else {}

    # -- caching ---------------------------------------------------------
    #
    # Keyed on what would change the answer: the text, the extractor, the
    # schema and the model identity. No secret and no auth material enters a
    # key -- the model identity is a NAME, taken from configuration.
    def _cache_key(self, section: DocumentSection, issued_at: Optional[str]) -> str:
        digest = hashlib.sha256(section.text.encode("utf-8")).hexdigest()[:32]
        return "|".join([digest, section.label, issued_at or "",
                         self.version, SCHEMA_VERSION,
                         config.finance_extraction_model_identity()])

    def _ask_once(self, system: str, user: str) -> str:
        response = self._ask(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            response_format="json",
            timeout=config.finance_extraction_timeout_seconds(),
            options={"num_predict": config.finance_extraction_max_output_tokens()})
        if not isinstance(response, dict) or not response.get("ok", True):
            raise ExtractionFailure("MODEL_ERROR", "the model call did not succeed")

        # A truncated response and a malformed one both fail to parse, and
        # they need opposite corrections -- truncation needs a shorter answer
        # or a bigger budget, malformation needs a rewritten one. Ollama
        # distinguishes them via done_reason="length", so this is diagnosed
        # HERE rather than surfacing downstream as a generic parse error.
        # A reasoning model can spend an entire budget on thinking tokens and
        # return an empty body, which looks exactly like "the model found no
        # guidance" unless this check exists.
        metrics = response.get("metrics") or {}
        content = ((response.get("message") or {}).get("content")) or ""
        if metrics.get("truncated") and not content.strip():
            raise ExtractionFailure(
                "TRUNCATED_RESPONSE",
                f"the response hit the {config.finance_extraction_max_output_tokens()}"
                f"-token limit after {metrics.get('completion_tokens') or 0} tokens "
                "without emitting any content")
        return content

    def extract(self, sections: Sequence[DocumentSection], *,
                issued_at: Optional[str] = None,
                document_id: Optional[str] = None) -> List[GuidanceCandidate]:
        candidates: List[GuidanceCandidate] = []
        for section in sections:
            key = self._cache_key(section, issued_at)
            if key in self._cache:
                payload = self._cache[key]
            else:
                system, user = build_prompt(section, issued_at)
                try:
                    payload = _parse_response(self._ask_once(system, user))
                except ExtractionFailure as first:
                    # ONE structured repair, the same bounded convention the
                    # research pipeline uses. A second failure is a failure.
                    #
                    # The correction depends on which failure it was: a
                    # truncated answer needs a SHORTER one, a malformed answer
                    # needs a rewritten one. Sending "rewrite it" to a model
                    # that ran out of budget just burns the budget again.
                    if first.code == "TRUNCATED_RESPONSE":
                        correction = (
                            "Your previous response never finished. Do not "
                            "explain your reasoning. Emit ONLY the JSON object "
                            "described above, with the most clearly stated "
                            "guidance statements only.")
                    else:
                        correction = (
                            f"Your previous response could not be read "
                            f"({first.code}). Respond with ONLY the JSON object "
                            "described above.")
                    payload = _parse_response(
                        self._ask_once(system, f"{user}\n\n{correction}"))
                self._cache[key] = payload
            for raw in payload.get("statements") or []:
                candidate = _candidate_from(raw, section, issued_at, document_id)
                if candidate is not None:
                    candidates.append(candidate)
        return candidates
