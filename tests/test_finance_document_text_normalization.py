"""Phase H.19: normalize real SEC HTML/inline-XBRL before section selection.

A live shadow re-run after H.18's evidence-grounding fix produced ZERO event
proposals on real filings. Traced directly: `event_extractor.py` bounded a
RAW, un-stripped document to 8,000 characters, and a real 8-K's raw text
opens with an `<ix:header>` block -- inline XBRL's own hidden-facts
container -- so the model was shown markup, not prose, and correctly
returned nothing. This file's fixtures reproduce that shape generically
(fictional entity/CIK/amounts) and pin the fix: normalize -> select the
relevant Item block -> THEN bound, never bound-then-strip.
"""

import re

from finance.documents.event_extractor import FINANCING_EVENT_ITEMS, select_event_sections
from finance.documents.text_normalization import (
    NormalizationFailure,
    normalize_sec_document,
    select_relevant_item_blocks,
    split_by_item,
)


# ---------------------------------------------------------------------------
# A realistic (fictional) raw inline-XBRL 8-K, shaped like the real one that
# exposed the defect: XML declaration, a large hidden ix:header block full
# of duplicate-tagged facts, THEN the visible body with Item headings, a
# table, and readable financing prose.
# ---------------------------------------------------------------------------

def _hidden_header(fact_count: int = 40) -> str:
    facts = "".join(
        f'<ix:nonNumeric id="Hidden_dei_Fact{i}" name="dei:Fact{i}" '
        f'contextRef="duration_2026-08-01_to_2026-08-01">0009999999</ix:nonNumeric> '
        for i in range(fact_count))
    return (
        '<div style="display:none"> <ix:header> <ix:hidden> ' + facts +
        '</ix:hidden> <ix:references> <xbrli:context id="duration_2026-08-01_to_2026-08-01">'
        '<xbrli:entity><xbrli:identifier scheme="http://www.sec.gov/CIK">0009999999'
        '</xbrli:identifier></xbrli:entity></xbrli:context> </ix:references> '
        '</ix:header> </div>')


def _raw_filing(body: str, fact_count: int = 40) -> str:
    return (
        "<?xml version='1.0' encoding='ASCII'?>\n"
        '<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL" '
        'xmlns:dei="http://xbrl.sec.gov/dei/2025" xmlns="http://www.w3.org/1999/xhtml">\n'
        "<head><title>8-K</title>"
        '<style type="text/css">.hidden{display:none}</style>'
        "<script>function track(){/* analytics */}</script>"
        "</head>\n"
        f"<body>{_hidden_header(fact_count)}\n{body}</body></html>"
    )


_ITEM_101_TITLE = "Entry into a Material Definitive Agreement."
_ITEM_203_TITLE = ("Creation of a Direct Financial Obligation or an Obligation under an "
                   "Off-Balance Sheet Arrangement of a Registrant.")
_ITEM_502_TITLE = ("Departure of Directors or Certain Officers; Election of Directors; "
                   "Appointment of Certain Officers; Compensatory Arrangements of "
                   "Certain Officers.")

_FINANCING_BODY = (
    f"<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>"
    f"<p>FORM 8-K</p>"
    f"<p>Item 2.03 {_ITEM_203_TITLE}</p>"
    f"<p>On August 1, 2026, the Company entered into a $600 million term loan "
    f"credit agreement with a syndicate of lenders (the &quot;Credit "
    f"Agreement&quot;). No amounts were drawn under the Credit Agreement "
    f"as of the effective date.</p>"
    f"<p>Item 9.01 Financial Statements and Exhibits.</p>"
    f"<p>None.</p>"
)

_LARGE_RAW_FILING = _raw_filing(_FINANCING_BODY, fact_count=200)


# ---------------------------------------------------------------------------
# A. 12k+ raw-character XBRL header + financing body -> normalized body
#    reaches the reader
# ---------------------------------------------------------------------------

def test_A_large_hidden_header_does_not_block_the_financing_body():
    assert len(_LARGE_RAW_FILING) > 12_000
    result = normalize_sec_document(_LARGE_RAW_FILING)
    assert result.ok
    assert "the Company entered into a $600 million term loan" in result.text
    assert "no amounts were drawn" in result.text.lower()


# ---------------------------------------------------------------------------
# B. normalization occurs before context truncation
# ---------------------------------------------------------------------------

def test_B_the_raw_first_8000_characters_contain_no_financing_prose():
    """The regression this phase exists to fix, stated as an assertion: the
    OLD behavior (bound raw text to 8000 chars) would have shown the model
    none of the real content."""
    raw_prefix = _LARGE_RAW_FILING[:8000]
    assert "$600 million" not in raw_prefix
    assert "term loan" not in raw_prefix
    # ... and the FIXED behavior recovers it despite that.
    sections, failure = select_event_sections(_LARGE_RAW_FILING, max_chars=8000)
    assert failure is None
    assert any("$600 million" in s.text for s in sections)


# ---------------------------------------------------------------------------
# C. scripts/styles/hidden metadata excluded
# ---------------------------------------------------------------------------

def test_C_scripts_styles_and_hidden_metadata_excluded():
    result = normalize_sec_document(_LARGE_RAW_FILING)
    assert result.ok
    assert "function track" not in result.text
    assert "display:none" not in result.text
    assert "Hidden_dei_Fact" not in result.text
    assert "0009999999" not in result.text  # the hidden CIK/context facts


# ---------------------------------------------------------------------------
# D. visible numbers/currencies preserved exactly
# ---------------------------------------------------------------------------

def test_D_visible_numbers_and_currencies_preserved_exactly():
    result = normalize_sec_document(_LARGE_RAW_FILING)
    assert "$600 million" in result.text


# ---------------------------------------------------------------------------
# E. "no amounts were drawn" preserved exactly
# ---------------------------------------------------------------------------

def test_E_negation_preserved_exactly():
    result = normalize_sec_document(_LARGE_RAW_FILING)
    assert "No amounts were drawn under the Credit Agreement" in result.text


# ---------------------------------------------------------------------------
# F. Item 2.03 block selected
# ---------------------------------------------------------------------------

def test_F_item_203_block_selected():
    sections, failure = select_event_sections(_LARGE_RAW_FILING)
    assert failure is None
    codes = {s.item_code for s in sections}
    assert "2.03" in codes


# ---------------------------------------------------------------------------
# G. Item 1.01 + Item 2.03 both selected where relevant
# ---------------------------------------------------------------------------

def test_G_multiple_relevant_items_both_selected():
    body = (
        f"<p>Item 1.01 {_ITEM_101_TITLE}</p>"
        f"<p>On August 1, 2026, the Company entered into an amendment to its "
        f"master supply agreement with Example Vendor, Inc.</p>"
        f"<p>Item 2.03 {_ITEM_203_TITLE}</p>"
        f"<p>On August 1, 2026, the Company borrowed $400 million under its "
        f"existing revolving credit facility.</p>"
    )
    raw = _raw_filing(body)
    sections, failure = select_event_sections(raw)
    assert failure is None
    codes = {s.item_code for s in sections}
    assert codes == {"1.01", "2.03"}


# ---------------------------------------------------------------------------
# H. irrelevant Item 5.02 excluded from the event reader
# ---------------------------------------------------------------------------

def test_H_irrelevant_item_502_excluded():
    body = (
        f"<p>Item 5.02 {_ITEM_502_TITLE}</p>"
        f"<p>On August 1, 2026, Jane Smith was appointed Chief Financial "
        f"Officer of the Company.</p>"
        f"<p>Item 2.03 {_ITEM_203_TITLE}</p>"
        f"<p>On August 1, 2026, the Company borrowed $400 million under its "
        f"existing revolving credit facility.</p>"
    )
    raw = _raw_filing(body)
    sections, failure = select_event_sections(raw)
    assert failure is None
    codes = {s.item_code for s in sections}
    assert "5.02" not in codes
    assert "2.03" in codes
    assert not any("Jane Smith" in s.text for s in sections)


# ---------------------------------------------------------------------------
# I. tables remain readable
# ---------------------------------------------------------------------------

def test_I_table_rows_remain_readable_and_ordered():
    body = (
        f"<p>Item 2.03 {_ITEM_203_TITLE}</p>"
        "<p>The Company's outstanding facilities are as follows.</p>"
        "<table><tr><td>Facility</td><td>Commitment</td><td>Drawn</td></tr>"
        "<tr><td>Revolver</td><td>$5.0 billion</td><td>$0</td></tr>"
        "<tr><td>Term Loan A</td><td>$2.0 billion</td><td>$2.0 billion</td></tr>"
        "</table>")
    raw = _raw_filing(body)
    result = normalize_sec_document(raw)
    assert result.ok
    # Cell content survives, in document order, with cells space-separated
    # and rows separated -- never merged into one unreadable run.
    assert "Facility" in result.text and "Commitment" in result.text
    assert "Revolver" in result.text and "$5.0 billion" in result.text
    assert "Term Loan A" in result.text and "$2.0 billion" in result.text
    facility_pos = result.text.find("Facility")
    revolver_pos = result.text.find("Revolver")
    term_loan_pos = result.text.find("Term Loan A")
    assert facility_pos < revolver_pos < term_loan_pos


# ---------------------------------------------------------------------------
# J. HTML entities decode correctly
# ---------------------------------------------------------------------------

def test_J_html_entities_decode_correctly():
    result = normalize_sec_document(_LARGE_RAW_FILING)
    assert '"Credit Agreement"' in result.text
    assert "&quot;" not in result.text


# ---------------------------------------------------------------------------
# K. nested inline-XBRL tags preserve visible prose
# ---------------------------------------------------------------------------

def test_K_nested_inline_xbrl_value_tags_preserve_visible_prose():
    """A VISIBLE (not hidden) ix:nonFraction tag wraps a number inline, the
    way real filings tag the numbers they also display -- the tag must be
    stripped and the number kept, not treated as more hidden metadata."""
    body = (
        f"<p>Item 2.03 {_ITEM_203_TITLE}</p>"
        '<p>The Company borrowed <ix:nonFraction name="us-gaap:ProceedsFromNotesPayable" '
        'contextRef="c1" unitRef="usd" scale="6">400</ix:nonFraction> million dollars '
        "under the facility.</p>")
    raw = _raw_filing(body)
    result = normalize_sec_document(raw)
    assert result.ok
    assert "The Company borrowed 400 million dollars under the facility" in result.text


# ---------------------------------------------------------------------------
# L. source-span ids deterministic after normalization
# ---------------------------------------------------------------------------

def test_L_span_ids_deterministic_after_normalization():
    sections_1, _ = select_event_sections(_LARGE_RAW_FILING)
    sections_2, _ = select_event_sections(_LARGE_RAW_FILING)
    assert [s.item_code for s in sections_1] == [s.item_code for s in sections_2]
    for section_1, section_2 in zip(sections_1, sections_2):
        assert [sp.span_id for sp in section_1.spans] == [sp.span_id for sp in section_2.spans]
        assert [sp.text for sp in section_1.spans] == [sp.text for sp in section_2.spans]


# ---------------------------------------------------------------------------
# M. model and validator operate on identical normalized spans
# ---------------------------------------------------------------------------

def test_M_reader_and_validator_share_the_same_span_text():
    """The exact ordering invariant section 10 cares about: whatever spans
    `select_event_sections` builds for the model are BYTE-IDENTICAL to what
    a validator constructed from the same call would check against -- there
    is only one normalization pass, not two that could drift apart."""
    sections, failure = select_event_sections(_LARGE_RAW_FILING)
    assert failure is None
    from finance.documents.event_validator import EventCandidateValidator
    from finance.documents.spans import resolve_span_ids

    all_spans = tuple(span for section in sections for span in section.spans)
    validator = EventCandidateValidator(spans=all_spans, items="2.03", form="8-K")
    # The validator's own span map IS the extractor's span list -- resolve
    # every id from it and confirm the text is exactly what was shown.
    ids = tuple(span.span_id for span in all_spans)
    text, unknown = resolve_span_ids(ids, validator.spans)
    assert not unknown
    assert text == " ".join(span.text for span in sorted(all_spans, key=lambda s: s.start))


# ---------------------------------------------------------------------------
# N. malformed HTML fails closed
# ---------------------------------------------------------------------------

def test_N_malformed_html_fails_closed_not_raises():
    malformed = "<html><body><div style=\"display:none\"><ix:header>" * 50 + "no closing tags at all"
    result = normalize_sec_document(malformed)
    # Never raises. Either it produces SOMETHING readable (the malformed
    # markup itself, stripped as best-effort) or it fails closed with a
    # code -- but it must not crash the pipeline.
    assert isinstance(result.ok, bool)


def test_N_unreadable_document_produces_no_sections():
    only_markup = "<html><head><script>var x = 1;</script></head><body></body></html>"
    sections, failure = select_event_sections(only_markup)
    assert sections == []
    assert failure in (NormalizationFailure.DOCUMENT_NOT_READABLE,
                       NormalizationFailure.NO_RELEVANT_ITEM_SECTION)


# ---------------------------------------------------------------------------
# O. normalized empty document produces no model call
# ---------------------------------------------------------------------------

def test_O_empty_document_fails_closed_before_any_section_is_built():
    sections, failure = select_event_sections("")
    assert sections == []
    assert failure == NormalizationFailure.NORMALIZATION_EMPTY

    sections, failure = select_event_sections(None)
    assert sections == []
    assert failure == NormalizationFailure.NORMALIZATION_EMPTY


def test_O_document_with_no_financing_item_produces_no_sections():
    """A filing that normalizes fine but names no financing-eligible item
    (only Item 5.02, an executive appointment) must produce zero sections
    -- not fall back to sending the whole document."""
    body = f"<p>Item 5.02 {_ITEM_502_TITLE}</p><p>Jane Smith was appointed CFO.</p>"
    raw = _raw_filing(body)
    sections, failure = select_event_sections(raw)
    assert sections == []
    assert failure == NormalizationFailure.NO_RELEVANT_ITEM_SECTION


# ---------------------------------------------------------------------------
# Direct coverage of the lower-level primitives
# ---------------------------------------------------------------------------

def test_split_by_item_finds_every_heading_in_order():
    body = (
        f"<p>Item 1.01 {_ITEM_101_TITLE}</p><p>Agreement text.</p>"
        f"<p>Item 2.03 {_ITEM_203_TITLE}</p><p>Financing text.</p>"
        f"<p>Item 9.01 Financial Statements and Exhibits.</p><p>None.</p>")
    result = normalize_sec_document(_raw_filing(body))
    blocks = split_by_item(result.text)
    assert [b.item_code for b in blocks] == ["1.01", "2.03", "9.01"]


def test_select_relevant_item_blocks_truncates_a_block_not_the_document():
    long_body = f"<p>Item 2.03 {_ITEM_203_TITLE}</p><p>" + ("word " * 3000) + "$500 million drawn.</p>"
    result = normalize_sec_document(_raw_filing(long_body))
    blocks = select_relevant_item_blocks(result.text, FINANCING_EVENT_ITEMS,
                                         max_chars_per_block=500)
    assert len(blocks) == 1
    assert len(blocks[0].text) <= 500
    assert blocks[0].text.startswith("Item 2.03")


def test_financing_event_items_matches_package_module():
    """Pins the two lists' agreement (event_extractor.py deliberately does
    not import package.py to avoid the edge -- see that module's own
    docstring)."""
    from finance.documents.package import FINANCING_EVENT_ITEMS as PACKAGE_ITEMS
    assert set(FINANCING_EVENT_ITEMS) == set(PACKAGE_ITEMS)
