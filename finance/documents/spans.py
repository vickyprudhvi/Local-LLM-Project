"""ONE deterministic source-span builder for every semantic reader.

WHY THIS EXISTS (Phase H.18)

The live event-reader benchmark found a reproducible defect: asked to "copy
the exact sentence," the model sometimes elided or lightly reworded a middle
clause of a dense, multi-clause SEC sentence. The validator correctly
refused every one of those (`EVIDENCE_NOT_IN_SOURCE`) -- it was never wrong
to refuse a citation that does not literally appear in the document. The
defect was asking the model to TRANSCRIBE evidence at all.

THE FIX, ARCHITECTURALLY

    LLM selects evidence     -- picks span_ids from a list this module built
    deterministic code retrieves evidence  -- looks the ids up, verbatim

never:

    LLM writes evidence      -- transcribes/paraphrases source text
    validator searches for similar text    -- fuzzy matching as authority

A span's `text` is guaranteed to be an EXACT substring of the document it was
built from (`document[span.start:span.end] == span.text`, always, by
construction -- never by verification after the fact). A reader can point at
the wrong span; it cannot invent one, because span_ids are assigned here, not
by the model.

GRANULARITY

Splits on sentence boundaries AND semicolons -- a compound SEC sentence
routinely states the event in one clause and the funding status in another,
joined by a semicolon, and the whole point of `evidence_span_ids` being a
LIST (finance/documents/event_schema.py) is that one candidate may cite
several spans rather than being forced to find one sentence that says
everything.

Parentheticals and quoted defined terms are deliberately NOT split out
further: they stay inside whichever sentence/clause contains them, because a
parenthetical clause read in isolation ("(subject to customary conditions)")
is not independently meaningful evidence for anything.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Sentence end (.!?) followed by whitespace and (usually) a capital letter or
# digit, OR a semicolon followed by whitespace. The capital/digit lookahead on
# the period case is what keeps "Corp." or "$3.0 billion" from being read as
# a sentence end -- an abbreviation or a decimal point is followed by a
# lowercase word or another digit far more often than a new clause is.
_SPLIT_POINT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\"“])"
    r"|;\s+")


@dataclass(frozen=True)
class SourceSpan:
    """One clause/sentence, with the EXACT offsets it came from.

    `text` is always `source[start:end]` for the `source` passed to
    `build_source_spans` -- never re-derived, never cleaned after slicing.
    Only pure leading/trailing WHITESPACE is trimmed from the boundary
    (adjusting `start`/`end` to match), which is the same tolerance
    `finance.extraction.validator._normalise` already treats as free for
    every other evidence check in this codebase -- a span never drops or
    rewrites a word.
    """

    span_id: str
    text: str
    start: int
    end: int


def _strip_bounds(text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def build_source_spans(text: str, max_spans: Optional[int] = None,
                       min_chars: int = 8) -> List[SourceSpan]:
    """Deterministic spans over `text`, each an exact, citable substring.

    Every span id is prefixed with a short hash of `text` itself
    (`<8-hex>-s001`, `<8-hex>-s002`, ...) rather than a bare sequence
    number. This is what makes "a candidate references another document's
    span" (spec section 10.I) an actual REJECTION rather than an accidental
    resolution: two different documents' span lists both start counting at
    `s001`, so a bare sequential id from document A would silently exist --
    and resolve to the WRONG text -- in document B's span map too. Hashing
    the source text into the id means a foreign id practically never
    collides, so `resolve_span_ids` simply fails to find it, the same path
    that already refuses an invented or mistyped id.

    `min_chars` drops spans too short to be evidence for anything (stray
    fragments left by the split, a lone "Inc" after an abbreviation period)
    -- dropped, not merged, so a neighboring span's own boundaries stay
    exact rather than absorbing a fragment that was not part of it.
    """
    if not text:
        return []
    prefix = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    spans: List[SourceSpan] = []
    cursor = 0
    boundaries = [m.end() for m in _SPLIT_POINT.finditer(text)]
    boundaries.append(len(text))
    index = 1
    for boundary in boundaries:
        start, end = _strip_bounds(text, cursor, boundary)
        cursor = boundary
        if end - start < min_chars:
            continue
        spans.append(SourceSpan(span_id=f"{prefix}-s{index:03d}",
                                text=text[start:end], start=start, end=end))
        index += 1
        if max_spans is not None and len(spans) >= max_spans:
            break
    return spans


def render_spans(spans: Sequence[SourceSpan]) -> str:
    """The numbered-span block shown to the model. Nothing else is shown."""
    return "\n".join(f"[{span.span_id}] {span.text}" for span in spans)


def resolve_span_ids(span_ids: Sequence[str], spans: Sequence[SourceSpan]
                     ) -> Tuple[Optional[str], Tuple[str, ...]]:
    """(reconstructed evidence text or None, the unknown ids if any).

    Text is reconstructed in DOCUMENT ORDER (by `start`), not in whatever
    order the model listed the ids -- a candidate that cites its own spans
    out of order still gets the source's own order, so evidence never reads
    as something the document did not say.

    `None` (with the offending ids) means at least one id does not exist in
    `spans` -- invented, mistyped, or copied from a different document's
    span set. The caller refuses the candidate; nothing is reconstructed
    from a partial or best-effort match.
    """
    by_id = {span.span_id: span for span in spans}
    unknown = tuple(sid for sid in span_ids if sid not in by_id)
    if unknown or not span_ids:
        return None, unknown
    ordered = sorted((by_id[sid] for sid in span_ids), key=lambda s: s.start)
    return " ".join(span.text for span in ordered), ()
