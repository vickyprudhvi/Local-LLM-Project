"""Raw SEC document -> readable text. Structural only, never financial.

WHY THIS EXISTS (Phase H.19)

The live shadow re-run after H.18's evidence-grounding fix produced ZERO
event proposals on real filings -- not because the fix regressed anything,
but because `event_extractor.py` was truncating RAW, un-stripped HTML/
inline-XBRL to 8,000 characters before doing anything else. A real 8-K's
raw document starts with XML namespace declarations and an
`<ix:header>...</ix:header>` block: inline XBRL's own container for tagged
facts that are NOT meant to be read (entity identifiers, context/unit
definitions, schema references, duplicate-tagged numbers), invisible in a
browser via `display:none` and INVISIBLE to that same truncation. Measured
on one real filing: 7,045 of the first 8,000 raw characters were this
metadata; the actual "Item 2.03 ... the Company drew $11.5 billion ..."
prose did not begin until character 67,176 of a 77,468-character document.
The model was shown pure markup and correctly returned no candidates.

WHAT ALREADY EXISTED AND WHY IT IS REUSED

`finance.guidance.html_to_text` already turns filed HTML into flat,
readable text -- strips `<script>/<style>/<head>`, turns table cells and
row/paragraph/list boundaries into word/clause separators, decodes
entities, normalizes whitespace and typography. Measured directly against
the same real filing: it alone takes 77,468 raw characters down to 9,928
readable ones. It is reused here UNCHANGED (this module never edits
`finance/guidance.py`) for exactly the tag-stripping and entity-decoding
work it already does correctly. What it does NOT do -- because guidance
prose (an EX-99 press-release exhibit) essentially never needs it -- is
recognise inline XBRL's own hidden-facts wrapper, which is why that survived
as 7,000+ characters of readable-looking junk ahead of the real content.

WHAT IS NEW HERE

    1. Strip `<ix:header>` (and any other `display:none` element) BEFORE
       handing off to `html_to_text`, so its content never becomes "flat
       text" in the first place.
    2. Split the normalized text into Item-code blocks ("Item 2.03 ..." to
       the next "Item N.NN" or end of document) -- a KEEP-class structural
       pattern: the SEC mandates this exact numbering, there is nothing to
       interpret. Bounding then applies to a SELECTED item block, never to
       a raw document prefix.

This module never decides what a sentence MEANS. It decides what text a
reader is even shown.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from finance.guidance import html_to_text

# ---------------------------------------------------------------------------
# Size bounds (section 12) -- pathological input must cost bounded work,
# never an unbounded prompt.
# ---------------------------------------------------------------------------

# A raw filing document larger than this is truncated BEFORE any regex work
# runs on it, so a pathological input cannot make normalization itself
# unbounded. Comfortably above any real 8-K (even AT&T's 77KB example is
# under 1% of this).
MAX_RAW_CHARS = 2_000_000


class NormalizationFailure:
    NORMALIZATION_EMPTY = "NORMALIZATION_EMPTY"
    NO_RELEVANT_ITEM_SECTION = "NO_RELEVANT_ITEM_SECTION"
    DOCUMENT_NOT_READABLE = "DOCUMENT_NOT_READABLE"
    NORMALIZATION_FAILED = "NORMALIZATION_FAILED"


@dataclass(frozen=True)
class NormalizedDocumentText:
    """The whole document, readable. No section selection yet."""

    text: str
    ok: bool
    failure_code: Optional[str] = None
    raw_chars: int = 0
    normalized_chars: int = 0

    def to_dict(self) -> dict:
        return {"ok": self.ok, "failure_code": self.failure_code,
                "raw_chars": self.raw_chars, "normalized_chars": self.normalized_chars}


@dataclass(frozen=True)
class ItemBlock:
    """One SEC Item-numbered section of a normalized document."""

    item_code: str
    heading: str
    text: str
    start: int
    end: int


# ---------------------------------------------------------------------------
# 1. Strip what must never reach html_to_text as "readable" text
# ---------------------------------------------------------------------------

# Inline XBRL's own hidden-facts container. Part of the Inline XBRL 2013
# specification itself (not a filer convention), so this is structural
# recognition, not an inference about meaning -- the same KEEP/DEPRECATE
# line `docs/finance_extraction_v2.md` §24 draws. Non-nested by the spec
# (it holds ix:hidden/ix:references/ix:resources, not arbitrary <div>
# structure), so a non-greedy match is safe here.
_IX_HEADER = re.compile(r"(?is)<ix:header\b.*?</ix:header\s*>")

# XML/DOCTYPE preamble. `html_to_text`'s generic `<[^>]+>` strip already
# removes the TAGS; stripping the whole declaration here (content included,
# though an XML PI has none) keeps the two passes' responsibilities clean.
_XML_DECLARATION = re.compile(r"(?is)<\?xml\b.*?\?>|<!DOCTYPE\b[^>]*>")

_DIV_OPEN = re.compile(r"(?i)<div\b[^>]*>")
_DIV_CLOSE = re.compile(r"(?i)</div\s*>")
_HIDDEN_DIV_OPEN = re.compile(
    r'(?i)<div\b[^>]*\bstyle\s*=\s*["\'][^"\']*display\s*:\s*none[^"\']*["\'][^>]*>')


def _strip_hidden_divs(text: str) -> str:
    """Remove every `<div style="display:none">...</div>`, matching nested
    divs by DEPTH rather than the first `</div>` seen -- a naive non-greedy
    regex would stop at an inner close tag and leave the remainder of a
    nested hidden block looking like visible prose. Defense in depth beyond
    `_IX_HEADER`: most SEC hidden content IS an ix:header, but a filer may
    mark something else invisible too.
    """
    out: List[str] = []
    cursor = 0
    while True:
        match = _HIDDEN_DIV_OPEN.search(text, cursor)
        if not match:
            out.append(text[cursor:])
            break
        out.append(text[cursor:match.start()])
        depth = 1
        pos = match.end()
        while depth > 0:
            next_open = _DIV_OPEN.search(text, pos)
            next_close = _DIV_CLOSE.search(text, pos)
            if next_close is None:
                # No matching close found. Fail toward dropping too much
                # rather than too little -- everything from here on is
                # discarded rather than risking hidden content leaking
                # through as if it were visible.
                pos = len(text)
                depth = 0
                break
            if next_open and next_open.start() < next_close.start():
                depth += 1
                pos = next_open.end()
            else:
                depth -= 1
                pos = next_close.end()
        cursor = pos
    return "".join(out)


def _strip_non_visible_markup(raw: str) -> str:
    text = _XML_DECLARATION.sub(" ", raw)
    text = _IX_HEADER.sub(" ", text)
    text = _strip_hidden_divs(text)
    return text


# ---------------------------------------------------------------------------
# 2. The public entry point
# ---------------------------------------------------------------------------

def normalize_sec_document(raw_document: Optional[str]) -> NormalizedDocumentText:
    """Raw filing markup -> readable text. Fails closed, never raises.

    Ordering is the whole point (section 5): every non-visible-markup strip
    happens BEFORE `html_to_text` flattens tags away, and no character
    bound is applied here at all -- bounding happens after item-section
    selection (`select_relevant_item_blocks`), against MEANINGFUL text, not
    against a raw prefix.
    """
    if not raw_document or not raw_document.strip():
        return NormalizedDocumentText(text="", ok=False,
                                      failure_code=NormalizationFailure.NORMALIZATION_EMPTY,
                                      raw_chars=0, normalized_chars=0)
    raw_chars = len(raw_document)
    bounded_raw = raw_document[:MAX_RAW_CHARS]
    try:
        visible = _strip_non_visible_markup(bounded_raw)
        text = html_to_text(visible)
    except Exception:                                          # noqa: BLE001
        return NormalizedDocumentText(text="", ok=False,
                                      failure_code=NormalizationFailure.NORMALIZATION_FAILED,
                                      raw_chars=raw_chars, normalized_chars=0)
    if not text.strip():
        return NormalizedDocumentText(text="", ok=False,
                                      failure_code=NormalizationFailure.DOCUMENT_NOT_READABLE,
                                      raw_chars=raw_chars, normalized_chars=0)
    return NormalizedDocumentText(text=text, ok=True, raw_chars=raw_chars,
                                  normalized_chars=len(text))


# ---------------------------------------------------------------------------
# 3. Item-heading section selection (structural only -- KEEP class)
# ---------------------------------------------------------------------------

# The SEC's own Form 8-K item numbering (Item 1.01, Item 2.03, ...). This is
# a MANDATED FORMAT, not a phrase the reader must interpret -- recognising it
# is exactly the "structure a document's own shape determines" KEEP class
# `docs/finance_extraction_v2.md` §24 defines, same footing as the filing-
# index and period-token patterns already kept there.
_ITEM_HEADING = re.compile(r"(?i)\bItem\s+(\d{1,2}\.\d{2})\b\.?\s*")


def split_by_item(text: str) -> List[ItemBlock]:
    """Every `Item N.NN` block in `text`, in document order.

    One block per heading found, running to the next heading or the end of
    the document. A filing with no Item headings at all (a plain exhibit,
    say) returns an empty list -- callers decide what that means; this
    function only recognises structure that is actually there.
    """
    matches = list(_ITEM_HEADING.finditer(text))
    blocks: List[ItemBlock] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        blocks.append(ItemBlock(item_code=match.group(1), heading=match.group(0).strip(),
                                text=body, start=start, end=end))
    return blocks


def select_relevant_item_blocks(text: str, relevant_items: Sequence[str],
                                max_blocks: Optional[int] = None,
                                max_chars_per_block: Optional[int] = None
                                ) -> List[ItemBlock]:
    """Only the blocks whose own item code is in `relevant_items`.

    Bounding (`max_chars_per_block`) is applied HERE -- after normalization
    and after selecting the block that actually matters -- never against a
    raw document prefix (section 5's ordering invariant). A block longer
    than the bound is truncated at its own end, not the document's start.
    """
    wanted = set(relevant_items)
    blocks = [b for b in split_by_item(text) if b.item_code in wanted]
    if max_chars_per_block is not None:
        blocks = [b if len(b.text) <= max_chars_per_block
                 else ItemBlock(b.item_code, b.heading, b.text[:max_chars_per_block],
                                b.start, b.start + max_chars_per_block)
                 for b in blocks]
    if max_blocks is not None:
        blocks = blocks[:max_blocks]
    return blocks


# ---------------------------------------------------------------------------
# 4. MD&A section selection for periodic filings (10-Q/10-K) -- Phase H.32
#    section 1-4: current management guidance may appear outside the
#    earnings-release exhibit, most commonly in the periodic filing's own
#    MD&A item. `_ITEM_HEADING` above requires the 8-K's decimal sub-item
#    format ("Item 2.03") and deliberately does not match a 10-Q/10-K's
#    bare-number item headings ("Item 2", "Item 7") -- reusing it here would
#    risk 8-K sub-item text matching a periodic filing's own cross-references
#    to ITS items. This is a SEPARATE, narrowly-scoped pattern for exactly
#    one purpose: finding the MD&A item and nothing else in the filing, most
#    importantly never Item 1A (Risk Factors), which sits textually
#    adjacent to it in a 10-K and must never be fed to the guidance reader.
# ---------------------------------------------------------------------------

# The SEC's own mandated title (Regulation S-K Item 303) for the MD&A item --
# "Item 2" in a 10-Q, "Item 7" in a 10-K. Matched on the FULL TITLE WORDING,
# not the bare item number, because the item NUMBER alone ("Item 2") also
# names a different section in a 10-K (Properties) and recurs across a
# 10-Q's Part I and Part II -- the title is what SEC rules make this exact
# section state verbatim at its own heading, so matching it is recognising
# document STRUCTURE, not interpreting content (same KEEP-class discipline
# `_ITEM_HEADING` above already uses for 8-K item codes).
_MDA_HEADING = re.compile(
    r"(?i)\bItem\s+\d{1,2}[A-Z]?\.?\s*"
    r"Management'?s?\s+Discussion\s+and\s+Analysis\s+of\s+Financial\s+Condition\s+"
    r"and\s+Results\s+of\s+Operations\b")

# Full, SEC-mandated titles of the items that conventionally follow MD&A.
# Matched the same way (full title, not bare number) so an in-prose
# cross-reference like "see Item 7A" -- which never repeats the complete
# official title immediately after the item number -- cannot be mistaken
# for the section's own heading and truncate the MD&A block early.
_NEXT_MAJOR_SECTION_TITLE = re.compile(
    r"(?i)\bItem\s+\d{1,2}[A-Z]?\.?\s*(?:"
    r"Quantitative\s+and\s+Qualitative\s+Disclosures\s+About\s+Market\s+Risk"
    r"|Controls\s+and\s+Procedures"
    r"|Financial\s+Statements(?:\s+and\s+Supplementary\s+Data)?"
    r"|Legal\s+Proceedings"
    r"|Risk\s+Factors"
    r"|Unregistered\s+Sales\s+of\s+Equity\s+Securities"
    r"|Defaults\s+Upon\s+Senior\s+Securities"
    r"|Mine\s+Safety\s+Disclosures"
    r"|Other\s+Information"
    r"|Exhibits(?:\s+and\s+Financial\s+Statement\s+Schedules)?"
    r")\b")


def select_mda_section(text: str, max_chars: Optional[int] = None) -> Optional[ItemBlock]:
    """The MD&A item block of a 10-Q/10-K, structurally bounded -- or None
    if the filing's own heading structure cannot be found.

    Bounded at the end by the NEXT recognised major item title (never a bare
    item number, to avoid an in-text cross-reference ending the block early)
    or the end of the document. `max_chars` truncates the block at its own
    end, the same discipline `select_relevant_item_blocks` already uses.

    Phase H.32 live finding (TSLA): EVERY real 10-Q/10-K opens with a Table
    of Contents that lists the MD&A item's own title again, with only a page
    number between it and the NEXT item's title -- so the FIRST occurrence
    of the MD&A heading is routinely the TOC entry, not the body, and
    bounding from there produced a body of a few dozen characters while the
    real, multi-thousand-character MD&A sat much later in the document. Not
    a TSLA-specific condition: a TOC is a universal 10-Q/10-K structural
    convention, so the first-match assumption was wrong for every filer, not
    just this one. Fixed by taking whichever occurrence of the heading
    produces the LARGEST span before the next major item title -- the TOC
    entry's span is always tiny (a title and a page number), while the body
    section's span is the section itself. No length threshold is tuned: the
    comparison is relative, between the filing's own candidate spans.
    """
    matches = list(_MDA_HEADING.finditer(text))
    if not matches:
        return None
    best_match, best_end = None, None
    for match in matches:
        following = _NEXT_MAJOR_SECTION_TITLE.search(text, match.end())
        end = following.start() if following is not None else len(text)
        if best_match is None or (end - match.start()) > (best_end - best_match.start()):
            best_match, best_end = match, end
    match, end = best_match, best_end
    start = match.start()
    body = text[start:end].strip()
    if not body:
        return None
    if max_chars is not None and len(body) > max_chars:
        body = body[:max_chars]
        end = start + len(body)
    return ItemBlock(item_code="MDA", heading=match.group(0).strip(),
                     text=body, start=start, end=end)
