"""Guidance stated as a TABLE must reach the reader.

THE FAILURE CLASS

`select_sections` has two paths. When a release labels its outlook with a
heading on its own line, the heading path takes the region after it. When it
does not, a fallback gathers the sentences carrying forward-looking
vocabulary.

An issuer that states guidance in a TABLE defeats the fallback. `html_to_text`
flattens a table to fragments -- "Revenues $9.7B - $10.5B." is its own
"sentence" -- and a table row carries NUMBERS, not vocabulary. It says
"Revenues", never "we expect revenues". So every row holding a figure was
dropped and only the prose caption survived, and the model was handed a page
of safe-harbor boilerplate and asked to find guidance in it.

That is a SECTION_SELECTION failure, and it is generalized: it costs the
guidance of any issuer who tabulates rather than narrates. The reader is not
at fault and no prompt or model change could fix it -- the numbers were never
in the prompt.

THE RULE

A guidance sentence is often the CAPTION of a table. So the run of short
fragments following a forward-looking sentence is absorbed with it, and the
run ends at the first fragment long enough to be prose. A short fragment is a
table row; a paragraph is not.

The run is kept only if something in it carries a digit, so a caption followed
by ordinary short prose does not drag unrelated text along.

Not tested here but worth stating: this is deliberately a BOUNDARY rule, the
`[KEEP]` use of a pattern under the §24 classification. It decides where a
region starts and stops. It reads no meaning out of any row -- that remains
the model's job, and the validator's.
"""

import pytest

from finance import guidance as gm
from finance.extraction.semantic_extractor import select_sections

# A release that captions a table and then tabulates. No heading on its own
# line, which is what sends this down the fallback path. Deliberately not any
# real issuer's text -- the SHAPE is the failure class.
TABULATED = (
    "Acme Corporation Reports Third Quarter Results. "
    "Revenue for the third quarter was $8.9 billion. "
    "The following table summarizes GAAP and Non-GAAP guidance based on the "
    "current outlook. "
    "Current Guidance . "
    "Q4 FY2026 Estimates . "
    "Revenues $9.7B - $10.5B. "
    "Segment revenues $8.4B - $9.0B . "
    "GAAP diluted EPS $1.22 - $1.42. "
    "Non-GAAP diluted EPS $2.05 - $2.25. "
    "(1) Our outlook does not include provisions for proposed tax law changes, "
    "future asset impairments or for pending legal matters, other than future "
    "legal amounts that are probable and estimable, and accordingly we include "
    "such items only when they can be accurately forecast for the period. "
)


def _sent(text):
    sections = select_sections(text)
    assert sections, "no section was selected at all"
    return sections[0].text


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("figure", ["9.7", "10.5", "1.22", "1.42", "2.05", "2.25"])
def test_every_tabulated_figure_reaches_the_reader(figure):
    """The whole failure in one assertion: the numbers must be in the prompt.

    Before the fix the caption survived and every row holding a figure was
    dropped, so the model was asked to find guidance in text that contained
    none of it.
    """
    assert figure in _sent(TABULATED)


def test_the_caption_still_comes_along():
    """The row says "Revenues $9.7B - $10.5B" and nothing about the period.

    Without the caption and the "Q4 FY2026 Estimates" label the reader cannot
    tell what period the row targets, so absorbing the rows without the text
    that introduces them would trade one failure for another.
    """
    body = _sent(TABULATED)
    assert "guidance based on the current outlook" in body
    assert "Q4 FY2026" in body


# ---------------------------------------------------------------------------
# ...and what absorbing a run must NOT do
# ---------------------------------------------------------------------------

def test_the_run_stops_at_prose():
    """A paragraph ends the table. Otherwise this swallows the whole filing.

    Asserted against a paragraph carrying NO forward-looking vocabulary, so
    the only way it could appear in the section is by being absorbed as a
    table row. The safe-harbor footnote in `TABULATED` is a poor probe for
    this: it says "outlook" and "forecast", so the fallback picks it up on its
    own merits whether or not any run absorption happens, and asserting its
    absence would be asserting something unrelated to the rule under test.
    """
    text = (TABULATED
            + "Acme is a manufacturer of industrial equipment headquartered "
              "in Ohio, with roughly eleven thousand employees across "
              "seventeen countries, and it has paid a dividend in every year "
              "since its founding, which the board reviews each quarter. ")
    body = _sent(text)
    assert "industrial equipment headquartered" not in body
    # ...while the rows before it still made it in.
    assert "2.05" in body


def test_a_caption_followed_by_short_prose_drags_nothing_along():
    """A run is kept only when something in it carries a digit.

    Otherwise any forward-looking sentence followed by a few terse sentences
    would pull them in, which is not a table and not evidence of anything.
    """
    text = ("We expect conditions to remain stable. "
            "Thank you all. "
            "Questions follow. "
            "Please hold. ")
    body = _sent(text)
    assert "We expect conditions to remain stable." in body
    assert "Thank you all" not in body
    assert "Questions follow" not in body


def test_a_lone_short_numeric_sentence_is_not_a_table():
    """The dangerous direction, and the one this rule got wrong first.

    A share count following a guidance sentence is short and carries a
    figure, so a rule keyed on "short and numeric" absorbs it -- dragging
    REPORTED HISTORY into a prompt that asks for forward-looking statements.
    That is how a results table becomes guidance, the single worst failure
    this layer has.

    A table is a REPETITION of short rows. One short sentence after a
    paragraph is a sentence.
    """
    text = ("The quarter closed well. For fiscal year 2027, we expect revenue "
            "of $90 billion. Shares outstanding were 139,933.")
    body = _sent(text)
    assert "expect revenue" in body
    assert "Shares outstanding" not in body


def test_two_reported_lines_after_a_guidance_sentence_are_not_absorbed():
    """Same rule, at the boundary. Two rows is still not a run."""
    text = ("We expect continued momentum next year. "
            "Revenue was $1.0 billion. "
            "Net income was $200 million. "
            "The company operates in a highly competitive market where "
            "pricing pressure and customer concentration remain relevant "
            "considerations for the periods ahead. ")
    body = _sent(text)
    assert "expect continued momentum" in body
    assert "Revenue was $1.0 billion" not in body
    assert "Net income was" not in body


def test_a_results_table_below_a_guidance_sentence_is_not_absorbed():
    """The defect this rule actually shipped with, found on the benchmark.

    A release states its outlook and then, further down, prints the condensed
    income statement. Those rows are short and numeric, so the run walked
    straight out of the outlook and into reported history -- putting "Three
    Months Ended April 30 / Total revenue $4,571,779" in a prompt asking for
    forward-looking statements. The reader then published a component metric
    as consolidated revenue growth: a CRITICAL false positive, caused by what
    the section rule chose to send.

    A reported-results caption ends the run, the same marker the validator
    stops at when it walks back for a governing caption.
    """
    text = ("The Company expects full-year revenue growth of 2% to 5%. "
            "Guidance Summary . "
            "Revenue growth 2% - 5% . "
            "Margin 42% - 44% . "
            "Casey's General Stores, Inc. "
            "and Subsidiaries . "
            "Condensed Consolidated Statements of Income . "
            "Three Months Ended April 30, Twelve Months Ended April 30, . "
            "Total revenue $ 4,571,779 $ 3,992,758 . ")
    body = _sent(text)
    assert "Revenue growth 2% - 5%" in body, "the outlook rows must still arrive"
    assert "Consolidated Statements of Income" not in body
    assert "Three Months Ended" not in body
    assert "4,571,779" not in body


def test_the_reader_and_the_boundary_share_one_reported_results_pattern():
    """One definition, two callers.

    Two definitions of the same idea is how the denominator defect happened:
    `resolve_identity` and `denominator_is_grounded` each decided what a ratio
    metric was, disagreed, and refused correct data. The reader must not send
    what the boundary would refuse to read.
    """
    from finance.extraction import semantic_extractor, validator
    from finance.extraction.schema import REPORTED_RESULTS_MARKER

    assert semantic_extractor.REPORTED_RESULTS_MARKER is REPORTED_RESULTS_MARKER
    assert validator._HISTORICAL_TABLE is REPORTED_RESULTS_MARKER  # noqa: SLF001


def test_the_section_stays_within_its_character_budget():
    """Absorbing rows must not become a way around the bound.

    A long table is still a bounded read; `max_chars` is what stops a whole
    filing reaching a model, and a run that ignored it would be a hole in the
    §3 bounding contract rather than a fix.
    """
    rows = " ".join(f"Line item {n} $1.0B - $2.0B." for n in range(400))
    text = "We expect the following results. " + rows
    sections = select_sections(text, max_chars=800)
    assert sections
    assert all(len(s.text) <= 800 for s in sections), \
        [len(s.text) for s in sections]


# ---------------------------------------------------------------------------
# No regression on the path that already worked
# ---------------------------------------------------------------------------

def test_a_release_with_a_real_heading_still_uses_the_heading_path():
    text = ("Acme Corporation Reports Results.\n"
            "Outlook\n"
            "For the full year 2027, the company expects revenue of $4.10 "
            "billion to $4.30 billion.\n")
    sections = select_sections(text)
    assert sections[0].label.lower().startswith("outlook")
    assert "4.10" in sections[0].text


def test_a_document_with_nothing_forward_looking_still_yields_nothing():
    text = ("Acme Corporation Reports Results. "
            "Revenue for the quarter was $1.0 billion. "
            "Net income was $200 million. ")
    assert select_sections(text) == []


def test_an_empty_document_is_still_empty():
    assert select_sections("") == []


# ---------------------------------------------------------------------------
# End to end through the real flattener
# ---------------------------------------------------------------------------

def test_the_fix_survives_html_to_text():
    """The failure only appears after flattening, so the test must flatten too.

    A table in filed HTML is where this starts; asserting against
    already-flat text would test a shape the pipeline never actually sees.
    """
    html = ("<p>Acme Corporation Reports Third Quarter Results.</p>"
            "<p>The following table summarizes GAAP and Non-GAAP guidance "
            "based on the current outlook.</p>"
            "<table>"
            "<tr><td>Current Guidance</td><td>Q4 FY2026 Estimates</td></tr>"
            "<tr><td>Revenues</td><td>$9.7B - $10.5B</td></tr>"
            "<tr><td>Non-GAAP diluted EPS</td><td>$2.05 - $2.25</td></tr>"
            "</table>")
    body = _sent(gm.html_to_text(html))
    assert "9.7" in body and "10.5" in body
    assert "2.05" in body and "2.25" in body
