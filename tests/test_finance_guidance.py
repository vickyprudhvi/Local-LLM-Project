"""Phase H.4 — management-guidance extraction (spec sections 4, 5, 16).

The property under test throughout is PRECISION, not recall. A missed
guidance figure is visible (guidance simply reads as unavailable and the
whole workflow is built to keep working that way — section 20). A fabricated
one is not: it flows into the DCF and looks exactly like a real number. So
every test that matters here is about what must NOT be extracted.
"""

import pytest

from finance import guidance as G

ACCN = "0000091142-26-000096"
DOC = "exhibit991.htm"


def _extract(text, fiscal_year=2026, filed="2026-07-30", accession=ACCN):
    return G.extract_guidance_from_text(text, "TEST", accession, DOC, filed,
                                        expected_fiscal_year=fiscal_year)


# ---------------------------------------------------------------------------
# What SHOULD be extracted
# ---------------------------------------------------------------------------

def test_a_percentage_range_becomes_a_decimal_ratio():
    release = _extract(
        "2026 full year outlook: Sales growth of between 2% and 3% for the year.")
    growth = release.metrics["revenue_growth"]
    assert (growth.low, growth.high) == pytest.approx((0.02, 0.03))
    assert growth.unit == G.GuidanceUnit.RATIO
    assert growth.fiscal_year == 2026


def test_a_per_share_range_keeps_its_currency_units():
    release = _extract(
        "The Company expects full-year 2026 Diluted EPS of between $3.60 and $3.75.")
    eps = release.metrics["earnings_per_share"]
    assert (eps.low, eps.high) == pytest.approx((3.60, 3.75))
    assert eps.unit == G.GuidanceUnit.CURRENCY_PER_SHARE
    assert eps.basis == G.BASIS_GAAP


def test_the_dash_and_to_spellings_are_both_understood():
    for text in ("2026 Guidance: Diluted EPS (GAAP) $ 3.60-3.75",
                 "For 2026 we expect diluted EPS of $3.60 to $3.75"):
        release = _extract(text)
        eps = release.metrics.get("earnings_per_share")
        assert eps is not None, text
        assert (eps.low, eps.high) == pytest.approx((3.60, 3.75))


def test_every_extracted_value_carries_the_text_that_supports_it():
    release = _extract(
        "The Company narrowed its full-year 2026 sales growth outlook to a range of "
        "2% to 3%, compared to its previous range of 2% to 4%.")
    growth = release.metrics["revenue_growth"]
    assert "2026" in growth.source_excerpt
    assert growth.evidence_id == "dcf.guidance.revenue_growth.current"


def test_gaap_and_adjusted_are_stored_separately_never_merged():
    """Section 5: mixing the two bases is a rejection condition. Both appear
    in the same sentence in a real release, one word apart."""
    release = _extract(
        "2026 full year guidance updated to Diluted EPS of between $3.60 and $3.75 "
        "and Adjusted EPS of between $3.70 and $3.85.")
    gaap = release.metrics["earnings_per_share"]
    adjusted = release.metrics["adjusted_earnings_per_share"]
    assert (gaap.low, gaap.high) == pytest.approx((3.60, 3.75))
    assert (adjusted.low, adjusted.high) == pytest.approx((3.70, 3.85))
    assert gaap.basis == G.BASIS_GAAP
    assert adjusted.basis == G.BASIS_ADJUSTED


def test_a_neighbouring_adjusted_metric_does_not_relabel_an_unrelated_one():
    """The basis modifier attaches to the METRIC NAME, not the paragraph.
    Reading the wider context instead made plain sales growth come out as
    "adjusted" and silently dropped the GAAP EPS line as a duplicate."""
    release = _extract(
        "2026 outlook: Sales growth of between 2% and 3%. Adjusted EPS of between "
        "$3.70 and $3.85.")
    assert release.metrics["revenue_growth"].basis == G.BASIS_GAAP


# ---------------------------------------------------------------------------
# What must NOT be extracted
# ---------------------------------------------------------------------------

def test_a_single_reported_actual_is_never_read_as_guidance():
    """Guidance is stated as a RANGE; actuals are single values. Requiring
    two numbers is what separates them without interpreting anything."""
    release = _extract("For 2026, diluted EPS was $3.85 for the full year as reported.")
    assert release.metrics == {}


def test_a_range_for_the_wrong_fiscal_year_is_rejected():
    """A release discusses last year's actuals beside this year's outlook."""
    release = _extract(
        "In 2025 the Company expects diluted EPS of between $3.70 and $3.85.",
        fiscal_year=2026)
    assert "earnings_per_share" not in release.metrics
    assert any("2025" in w for w in release.warnings)


def test_a_range_with_no_identifiable_fiscal_year_is_rejected():
    release = _extract("The Company expects diluted EPS of between $3.70 and $3.85.")
    assert release.metrics == {}
    # Phase H.6 wording: the extractor now resolves a fiscal PERIOD (which may
    # be a quarter), not just a year, so the rejection says so.
    assert any("no fiscal period could be identified" in w for w in release.warnings)


def test_a_range_with_no_forward_looking_language_is_not_guidance():
    """Historical comparisons are also ranges."""
    release = _extract(
        "Diluted EPS across 2026 segments ranged from $3.60 to $3.75 by division.")
    assert release.metrics == {}


def test_a_range_belonging_to_a_different_metric_is_not_claimed():
    """Filed releases are tables; flattening one puts unrelated rows next to
    each other. This exact shape produced $3.85-$4.15 of EPS being extracted
    as free-cash-flow guidance."""
    release = _extract(
        "Free cash flow (non-GAAP) $ 546.0 $ 473.8 2026 EPS Guidance and 2025 EPS "
        "Diluted EPS (GAAP) $ 3.85 - 4.15 $ 3.85")
    assert "free_cash_flow" not in release.metrics


def test_an_empty_or_unparseable_document_yields_no_guidance_and_says_so():
    assert _extract("").metrics == {}
    assert _extract("").warnings
    assert _extract("The Company held its earnings call today.").metrics == {}


def test_no_numeric_value_is_ever_invented():
    """A release with plenty of forward-looking language and no numbers at
    all must produce nothing."""
    release = _extract(
        "For fiscal 2026 the Company expects continued growth, anticipates margin "
        "improvement, and reaffirms its outlook for the full year.")
    assert release.metrics == {}


# ---------------------------------------------------------------------------
# Supersession (section 5)
# ---------------------------------------------------------------------------

def _release(filed, low, high, fiscal_year=2026, accession="a"):
    return G.GuidanceRelease(
        symbol="TEST", fiscal_year=fiscal_year, accession=accession, document=DOC,
        filed=filed,
        metrics={"revenue_growth": G.GuidanceMetric(
            name="revenue_growth", low=low, high=high, unit=G.GuidanceUnit.RATIO,
            basis=G.BASIS_GAAP, fiscal_year=fiscal_year,
            evidence_id="dcf.guidance.revenue_growth.current", source_excerpt="x")})


def test_the_newest_release_for_a_fiscal_year_is_the_current_guidance():
    releases = [_release("2026-01-29", 0.02, 0.05),
                _release("2026-04-30", 0.02, 0.04),
                _release("2026-07-30", 0.02, 0.03)]
    current, superseded = G.select_current_guidance(releases, 2026)
    assert current.filed == "2026-07-30"
    assert current.metrics["revenue_growth"].high == pytest.approx(0.03)
    assert [r.filed for r in superseded] == ["2026-04-30", "2026-01-29"]


def test_superseded_guidance_is_retained_for_comparison():
    releases = [_release("2026-01-29", 0.02, 0.05), _release("2026-07-30", 0.02, 0.03)]
    _current, superseded = G.select_current_guidance(releases, 2026)
    assert superseded and superseded[0].metrics["revenue_growth"].high == pytest.approx(0.05)


def test_narrowed_raised_and_lowered_guidance_all_resolve_to_the_newest():
    for older, newer in ((0.02, 0.05), (0.05, 0.02), (0.03, 0.03)):
        releases = [_release("2026-01-29", 0.02, older),
                    _release("2026-07-30", 0.02, newer)]
        current, _ = G.select_current_guidance(releases, 2026)
        assert current.metrics["revenue_growth"].high == pytest.approx(newer)


def test_guidance_for_an_ALREADY_PAST_fiscal_year_is_not_selected():
    """Phase H.6 changed what `fiscal_year` means here, deliberately.

    It is now the EARLIEST period still relevant, not an exact match. The old
    exact-match rule is what discarded every NVIDIA value: NVDA's May-2026
    release guides fiscal 2027 throughout, and a caller passing the calendar
    year 2026 got nothing at all. Guidance for a LATER period is the whole
    point of asking; only guidance for a period already behind us is stale.
    """
    stale = [_release("2025-07-30", 0.02, 0.03, fiscal_year=2025)]
    current, _superseded = G.select_current_guidance(stale, 2026)
    assert current is None


def test_guidance_for_a_LATER_fiscal_year_is_selected():
    """A non-calendar fiscal year guides ahead of the calendar year, and that
    is not a reason to discard it."""
    ahead = [_release("2026-05-20", 0.02, 0.03, fiscal_year=2027)]
    current, _superseded = G.select_current_guidance(ahead, 2026)
    assert current is not None
    assert current.metrics["revenue_growth"].fiscal_year == 2027


def test_no_releases_at_all_is_a_supported_outcome():
    assert G.select_current_guidance([], 2026) == (None, [])


# ---------------------------------------------------------------------------
# Document handling
# ---------------------------------------------------------------------------

def test_html_to_text_keeps_table_cells_apart():
    """Without cell boundaries, '$3,900' and '$3,950' become one token."""
    text = G.html_to_text(
        "<table><tr><td>Net sales</td><td>$3,900</td><td>$3,950</td></tr></table>")
    assert "$3,900 $3,950" in text


def test_html_to_text_drops_scripts_and_styles():
    text = G.html_to_text("<style>p{color:red}</style><p>2026 outlook</p>"
                          "<script>var x=1;</script>")
    assert "color" not in text and "var x" not in text
    assert "2026 outlook" in text


def test_the_exhibit_is_chosen_from_the_authoritative_type_column():
    """EDGAR's index.json labels each file with its ICON, so the exhibit TYPE
    has to come from the filing index page's own table."""
    html = """
    <table class="tableFile">
      <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
      <tr><td>1</td><td>8-K</td><td>aos-20260730.htm</td><td>8-K</td><td>25833</td></tr>
      <tr><td>2</td><td>EX-99.1</td><td>a6302026exhibit991.htm</td><td>EX-99.1</td><td>292819</td></tr>
    </table>"""
    assert G.select_exhibit_document(html) == "a6302026exhibit991.htm"


def test_a_filing_with_no_earnings_release_exhibit_returns_none():
    html = """
    <table class="tableFile">
      <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
      <tr><td>1</td><td>8-K</td><td>x.htm</td><td>8-K</td><td>100</td></tr>
    </table>"""
    assert G.select_exhibit_document(html) is None


def test_only_item_2_02_filings_are_treated_as_earnings_releases():
    submissions = {"filings": {"recent": {
        "form": ["8-K", "8-K", "10-Q"],
        "items": ["5.02,9.01", "2.02,9.01", ""],
        "accessionNumber": ["a-1", "a-2", "a-3"],
        "filingDate": ["2026-06-22", "2026-07-30", "2026-07-30"],
        "reportDate": ["2026-06-22", "2026-07-30", "2026-06-30"],
    }}}
    found = G.find_earnings_release_filings(submissions)
    assert [f["accession"] for f in found] == ["a-2"]
