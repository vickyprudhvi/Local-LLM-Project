"""Phase H.32 -- current management guidance may live in the CURRENT 10-Q/
10-K's own MD&A item, not only the earnings-release exhibit. This is a
SOURCE-COVERAGE fix, not a semantic-reader fix: the same V1 regex extractor,
the same forward-marker/reported-block filtering, and the same current-
guidance resolver (`select_current_guidance`) are reused unchanged -- only
WHICH TEXT reaches them is new.

Live-traced gap: a report said "Management guidance: none extracted (3 SEC
earnings releases examined)" while the issuer's current 10-Q stated an
explicit, quantitative, forward capital-expenditure figure in its own MD&A
item -- a document this tool never read at all.
"""

import pytest

from finance import guidance as gm
from finance.documents.text_normalization import select_mda_section


_TENQ_TEXT = (
    "Item 1. Financial Statements\n"
    "[balance sheet and income statement tables omitted]\n"
    "Item 2. Management's Discussion and Analysis of Financial Condition and "
    "Results of Operations\n"
    "Results of Operations. During the three months ended September 30, 2026, "
    "capital expenditures were $1.2 billion, compared to $1.0 billion in the "
    "prior-year period.\n"
    "Capital Resources and Liquidity. We expect fiscal year 2027 capital "
    "expenditures of approximately $8 billion, consistent with our long-term "
    "network capacity plan.\n"
    "Item 3. Quantitative and Qualitative Disclosures About Market Risk\n"
    "There have been no material changes to our market risk exposures.\n"
)

_TENK_WITH_RISK_FACTORS_TEXT = (
    "Item 1A. Risk Factors\n"
    "Our capital expenditures could exceed $8 billion if commodity prices "
    "rise sharply or network demand accelerates beyond current projections, "
    "which could adversely affect our liquidity.\n"
    "Item 7. Management's Discussion and Analysis of Financial Condition and "
    "Results of Operations\n"
    "We anticipate fiscal year 2027 capital expenditures of approximately "
    "$8 billion.\n"
    "Item 7A. Quantitative and Qualitative Disclosures About Market Risk\n"
    "No material changes.\n"
)

_ANALYST_TEXT = (
    "Item 2. Management's Discussion and Analysis of Financial Condition and "
    "Results of Operations\n"
    "Analysts estimate our fiscal year 2027 capital expenditures will be "
    "approximately $8 billion, though we have not confirmed this figure.\n"
)


def _extract(text, accession="acc-10q-1", filed="2026-09-15", fiscal_year=2027):
    return gm.extract_guidance_from_text(text, "ACME", accession, "10q.htm", filed,
                                         expected_fiscal_year=fiscal_year)


# ---------------------------------------------------------------------------
# A. 10-Q MD&A quantitative forward statement is accepted as current guidance.
# ---------------------------------------------------------------------------

_TENQ_WITH_TOC_TEXT = (
    "PART I - FINANCIAL INFORMATION\n"
    "Item 1. Financial Statements\n"
    "Item 2. Management's Discussion and Analysis of Financial Condition and "
    "Results of Operations\n"
    "Item 3. Quantitative and Qualitative Disclosures About Market Risk\n"
    "Item 4. Controls and Procedures\n"
    "PART II - OTHER INFORMATION\n"
    "[balance sheet and income statement tables omitted]\n"
    "Item 2. Management's Discussion and Analysis of Financial Condition and "
    "Results of Operations\n"
    "Results of Operations. Revenue for the period increased year over year.\n"
    "Capital Resources and Liquidity. We expect fiscal year 2027 capital "
    "expenditures of approximately $8 billion.\n"
    "Item 3. Quantitative and Qualitative Disclosures About Market Risk\n"
    "No material changes.\n"
)


def test_mda_section_selection_skips_the_table_of_contents_entry():
    """Phase H.32 live finding (TSLA real 10-Q): EVERY real 10-Q/10-K opens
    with a Table of Contents that restates the MD&A item's own title, with
    only a page number before the next item's title -- so the FIRST
    occurrence of the heading is routinely the TOC entry, not the body.
    Taking it produced a body of a few dozen characters while the real
    section sat thousands of characters later. Not TSLA-specific: a TOC is
    a universal 10-Q/10-K convention.
    """
    mda = select_mda_section(_TENQ_WITH_TOC_TEXT)
    assert mda is not None
    assert "capital expenditures of approximately $8 billion" in mda.text
    assert "Results of Operations. Revenue" in mda.text
    # The TOC's bare restatement of every item title must not be what was
    # selected (that span would be tiny and would end before any content).
    assert len(mda.text) > 150


def test_A_10q_mda_quantitative_forward_statement_is_accepted():
    mda = select_mda_section(_TENQ_TEXT)
    assert mda is not None
    release = _extract(mda.text)
    assert "capital_expenditure" in release.metrics
    metric = release.metrics["capital_expenditure"]
    assert metric.fiscal_year == 2027
    assert metric.low == pytest.approx(8_000_000_000.0) or metric.low == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# B. The SAME 10-Q's historical capex table (prior-quarter actuals) is not
#    read as guidance -- the existing reported-block filter handles this
#    without any new code, since it already applies to any text run through
#    `extract_guidance_from_text`.
# ---------------------------------------------------------------------------

def test_B_historical_capex_in_same_filing_is_not_guidance():
    mda = select_mda_section(_TENQ_TEXT)
    release = _extract(mda.text)
    # The historical $1.2 billion / $1.0 billion figures must never surface
    # as the accepted capex guidance value -- only the $8 billion forward one.
    metric = release.metrics.get("capital_expenditure")
    assert metric is not None
    assert metric.low not in (1_200_000_000.0, 1.2, 1_000_000_000.0, 1.0)


# ---------------------------------------------------------------------------
# C. A risk-factor hypothetical ("could exceed") is never read as guidance --
#    structurally impossible here because Item 1A text never reaches the
#    selected MD&A block at all (select_mda_section excludes it by
#    construction), independent of any keyword filtering.
# ---------------------------------------------------------------------------

def test_C_risk_factor_hypothetical_never_reaches_the_extractor():
    mda = select_mda_section(_TENK_WITH_RISK_FACTORS_TEXT)
    assert mda is not None
    assert "could exceed" not in mda.text
    assert "Risk Factors" not in mda.text
    release = _extract(mda.text, fiscal_year=2027)
    # The only capex figure in the selected text is the genuine MD&A one.
    metric = release.metrics.get("capital_expenditure")
    assert metric is not None
    assert metric.low == pytest.approx(8_000_000_000.0) or metric.low == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# D. An analyst/consensus-attributed figure embedded in a filing is not
#    management guidance.
# ---------------------------------------------------------------------------

def test_D_analyst_estimate_embedded_in_filing_is_not_management_guidance():
    mda = select_mda_section(_ANALYST_TEXT)
    assert mda is not None
    release = _extract(mda.text)
    assert "capital_expenditure" not in release.metrics
    assert any(gm.SOURCE_NOT_MANAGEMENT in w for w in release.warnings)


# ---------------------------------------------------------------------------
# E. Older earnings-release guidance + newer 10-Q reaffirmation -> ONE
#    current economic signal, correct provenance (the 10-Q's own accession).
#    Reuses `select_current_guidance` UNCHANGED -- this is the whole point
#    of routing periodic-filing releases through the SAME resolver.
# ---------------------------------------------------------------------------

def test_E_newer_10q_reaffirmation_supersedes_older_earnings_release_guidance():
    earlier = gm.extract_guidance_from_text(
        "We expect fiscal year 2027 capital expenditures of approximately "
        "$8 billion.",
        "ACME", "acc-er-1", "ex99.htm", "2026-07-20", expected_fiscal_year=2027)
    mda = select_mda_section(_TENQ_TEXT)
    later = _extract(mda.text, accession="acc-10q-1", filed="2026-09-15")

    current, superseded = gm.select_current_guidance([earlier, later])
    assert current is not None
    metric = current.metrics["capital_expenditure"]
    assert metric.fiscal_year == 2027
    # The NEWER (10-Q) release supplied the current statement.
    assert current.accession == "acc-10q-1"
    assert earlier in superseded


# ---------------------------------------------------------------------------
# F. A genuinely REVISED 10-Q figure supersedes the older compatible
#    guidance for the same metric/period/basis.
# ---------------------------------------------------------------------------

def test_F_revised_10q_guidance_supersedes_older_compatible_guidance():
    earlier = gm.extract_guidance_from_text(
        "We expect fiscal year 2027 capital expenditures of approximately "
        "$7 billion.",
        "ACME", "acc-er-1", "ex99.htm", "2026-07-20", expected_fiscal_year=2027)
    revised_text = (
        "Item 2. Management's Discussion and Analysis of Financial Condition "
        "and Results of Operations\n"
        "We now expect fiscal year 2027 capital expenditures of approximately "
        "$8 billion, an increase from our prior outlook.\n"
    )
    mda = select_mda_section(revised_text)
    later = _extract(mda.text, accession="acc-10q-2", filed="2026-09-15")

    current, superseded = gm.select_current_guidance([earlier, later])
    assert current is not None
    metric = current.metrics["capital_expenditure"]
    assert metric.low == pytest.approx(8_000_000_000.0) or metric.low == pytest.approx(8.0)
    assert earlier in superseded


# ---------------------------------------------------------------------------
# G. Guidance for a period that has already ended still retires normally,
#    whether it was sourced from an earnings release or a periodic filing --
#    `select_current_guidance`'s `as_of`/expiry logic is source-agnostic.
# ---------------------------------------------------------------------------

def test_G_completed_period_periodic_filing_guidance_still_retires():
    mda = select_mda_section(_TENQ_TEXT)
    release = _extract(mda.text, filed="2026-09-15", fiscal_year=2027)
    assert "capital_expenditure" in release.metrics
    # The guided fiscal year (2027) has since ended -- `as_of` in 2028 expires
    # it under `_is_expired`'s calendar-year rule for ANNUAL periods, same as
    # it would for an earnings-release-sourced statement.
    current, superseded = gm.select_current_guidance([release], as_of="2028-01-01")
    assert current is None
    assert release in superseded


# ---------------------------------------------------------------------------
# Config default / opt-out
# ---------------------------------------------------------------------------

def test_periodic_filing_coverage_defaults_to_enabled(monkeypatch):
    import tools.config as config

    monkeypatch.delenv("GUIDANCE_PERIODIC_FILING_COVERAGE_ENABLED", raising=False)
    assert config.guidance_periodic_filing_coverage_enabled() is True


def test_periodic_filing_coverage_can_be_disabled(monkeypatch):
    import tools.config as config

    monkeypatch.setenv("GUIDANCE_PERIODIC_FILING_COVERAGE_ENABLED", "false")
    assert config.guidance_periodic_filing_coverage_enabled() is False


# ---------------------------------------------------------------------------
# End-to-end: `SecCurrentGuidanceTool.execute()` actually reaches the 10-Q's
# MD&A, not only the earnings-release exhibit path. Stub shape follows
# `tests/test_finance_extraction_wiring.py`'s `_StubOutcome`/`_StubLedger`/
# `_StubCoordinator` convention -- no network, no SEC.
# ---------------------------------------------------------------------------

class _StubOutcome:
    def __init__(self, payload):
        self.payload = payload

    def freshness_dict(self, _now):
        return {"fetched_at": "2026-09-16T00:00:00Z"}


class _StubLedger:
    def _clock(self):
        return "2026-09-16T00:00:00Z"


class _StubPeriodicCoordinator:
    """Serves ONLY a 10-Q (no 8-K) so the periodic-filing path is the sole
    source of any releases, isolating it from the earnings-release path.
    """

    ledger = _StubLedger()

    def fetch(self, dataset, _symbol, params=None):
        if dataset == "company_submissions":
            return _StubOutcome({"filings": {"recent": {
                "form": ["10-Q"], "accessionNumber": ["0000000000-26-000099"],
                "filingDate": ["2026-09-15"], "primaryDocument": ["tenq.htm"],
                "items": [""], "reportDate": ["2026-08-02"]}}})
        return _StubOutcome({"document_text": f"<p>{_TENQ_TEXT}</p>"})


def test_tool_execute_reaches_10q_mda_guidance_end_to_end(monkeypatch):
    import tools.config as config
    import tools.finance_tools as ft

    monkeypatch.setattr(ft, "get_sec_coordinator", lambda: _StubPeriodicCoordinator())
    monkeypatch.setattr(ft, "resolve_cik", lambda _c, _s: ("0000000001", "Acme Corp"))
    monkeypatch.setattr(config, "sec_edgar_enabled", lambda: True)

    result = ft.SecCurrentGuidanceTool().execute({"symbol": "ACME", "fiscal_year": 2027})

    assert result["releases_examined"] == 0
    assert result["periodic_filings_examined"] == 1
    assert result["guidance"] is not None
    assert "capital_expenditure" in result["guidance"]["metrics"]
    metric = result["guidance"]["metrics"]["capital_expenditure"]
    assert metric["source_accession"] == "0000000000-26-000099"
