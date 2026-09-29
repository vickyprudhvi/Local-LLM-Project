"""Reported Actuals Source Integration: discovery, tables, facts, and the seam.

WHAT THIS FILE IS FOR

The benchmark next door measures the layer end to end. This one pins the
individual invariants, so a regression names itself instead of moving a
percentage.

Three of these tests exist because the first run of the benchmark failed them,
and each was a generalized defect rather than a fixture's problem:

  * "PRESS RELEASE" counted as evidence that a filing reported results. It
    names the vehicle, not the subject.
  * an EX-99 filename counted as the same. It names the slot.
  * `6-K` sat in the periodic-form set, so every foreign private issuer's
    every filing was a reported-actual source with its evidence unread.

All three are the same mistake at three levels: treating the CONTAINER as
proof of the CONTENT, which is exactly what section 3 forbids.
"""

import pytest

from finance import semantics as sem
from finance.extraction.schema import StatementCompleteness
from finance.reported_actuals import (
    extract_reported_actuals,
    parse_filing_tables,
)
from finance.reported_actuals.candidates import (
    company_facts_overlay,
    merge_company_facts,
)
from finance.reported_actuals.discovery import (
    SourceDiscoveryCode,
    classify_filing,
    find_reported_actual_filings,
    select_results_document,
)
from finance.reported_actuals.tables import StatementKind, band_frequency
from tests.fixtures import reported_actuals_releases as R


# ---------------------------------------------------------------------------
# Section 3: a form code is eligibility, not proof
# ---------------------------------------------------------------------------

def test_item_2_02_is_evidence_on_its_own():
    """The SEC's own classification of the filing, not an inference about it."""
    qualifies, evidence, _reason = classify_filing(
        R.filing("8-K", "a", "2026-09-02", items="2.02,9.01",
                 description="PRESS RELEASE", document="exhibit991.htm"))
    assert qualifies
    assert SourceDiscoveryCode.EARNINGS_ITEM in evidence


def test_a_press_release_about_an_appointment_is_not_reported_results():
    """PRESS RELEASE NAMES THE VEHICLE, NOT THE SUBJECT.

    An appointment, an acquisition and a quarter's results are all announced
    by press release, all as EX-99.1, all on an 8-K. Counting the phrase as
    evidence admitted every one of them -- a form allowlist one level down.
    """
    qualifies, evidence, reason = classify_filing(
        R.filing("8-K", "a", "2026-08-20", items="5.02",
                 description="PRESS RELEASE - LEADERSHIP APPOINTMENT",
                 document="exhibit991.htm"))
    assert not qualifies, evidence
    assert "no earnings evidence" in reason


def test_an_acquisition_announcement_is_not_reported_results():
    qualifies, _evidence, _reason = classify_filing(
        R.filing("8-K", "a", "2026-07-14", items="8.01",
                 description="PRESS RELEASE - ACQUISITION ANNOUNCEMENT",
                 document="ex991.htm"))
    assert not qualifies


def test_an_exhibit_99_filename_is_not_evidence_of_results():
    """The slot, not the subject. Every 8-K with any press release has one."""
    qualifies, _evidence, _reason = classify_filing(
        R.filing("8-K", "a", "2026-08-11", items="1.01,2.03",
                 description="INDENTURE", document="exhibit991.htm"))
    assert not qualifies


def test_a_6k_is_a_container_and_not_a_periodic_filing():
    """`taxonomy.ALL_REPORT_FORMS` groups 6-K with the interim forms, which is
    right for the question it answers and wrong for this one. A 6-K carries
    interim results, a notice of a general meeting, a change of auditor. The
    form proves nothing, and treating it as proof made every filing a foreign
    private issuer had ever made a reported-actual source."""
    qualifies, _evidence, _reason = classify_filing(
        R.filing("6-K", "a", "2026-06-02",
                 description="NOTICE OF ANNUAL GENERAL MEETING",
                 document="agm.htm"))
    assert not qualifies

    qualifies, evidence, _reason = classify_filing(
        R.filing("6-K", "b", "2026-08-05", report_date="2026-06-30",
                 description="INTERIM RESULTS FOR THE HALF YEAR ENDED 30 JUNE 2026",
                 document="results6k.htm"))
    assert qualifies
    assert SourceDiscoveryCode.DESCRIPTION_NAMES_RESULTS in evidence


def test_a_periodic_filing_needs_no_further_evidence():
    qualifies, evidence, _reason = classify_filing(
        R.filing("10-Q", "a", "2026-06-05", description="FORM 10-Q"))
    assert qualifies
    assert evidence == (SourceDiscoveryCode.PERIODIC_FILING,)


def test_discovery_returns_the_refusals_with_their_reasons():
    """A skip nobody can see is indistinguishable from a source that was
    never filed."""
    accepted, refused = find_reported_actual_filings(
        R.submissions(
            R.filing("8-K", "yes", "2026-09-02", items="2.02"),
            R.filing("8-K", "no", "2026-08-20", items="5.02",
                     description="PRESS RELEASE - LEADERSHIP APPOINTMENT")),
        limit=10)
    assert [c.accession for c in accepted] == ["yes"]
    assert [c.accession for c in refused] == ["no"]
    assert refused[0].reason


# ---------------------------------------------------------------------------
# Section 4: which document inside the filing
# ---------------------------------------------------------------------------

def test_the_conventional_exhibit_slot_is_not_assumed():
    """EX-99.1 IS FREQUENTLY SOMETHING ELSE.

    Here it is an investor deck and the results are in EX-99.2. A reader that
    takes the numbering extracts a presentation's rounded restatements and
    calls them the filed statements.
    """
    assert select_results_document(R.INVERTED_INDEX).document_id == "earnings992.htm"
    assert select_results_document(R.EARNINGS_INDEX).document_id == "exhibit991.htm"


def test_a_filing_with_no_results_exhibit_selects_nothing():
    for index in (R.LEADERSHIP_INDEX, R.NO_EXHIBIT_INDEX):
        selection = select_results_document(index)
        assert selection.document_id is None
        assert selection.selection_reason


def test_the_selection_records_why():
    selection = select_results_document(
        R.INVERTED_INDEX, accession="acc", form="8-K", filing_date="2026-09-02")
    assert selection.to_dict()["filing_id"] == "acc"
    assert "results" in selection.selection_reason.lower()


# ---------------------------------------------------------------------------
# Section 8: the period a column covers
# ---------------------------------------------------------------------------

def test_three_fiscal_quarters_is_nine_months_not_three():
    """COUNTING QUARTERS AND COUNTING MONTHS ARE TWO DIFFERENT SPELLINGS.

    "Three Months Ended" and "Three Fiscal Quarters Ended" differ by one word
    and by a factor of three, and both appear as bands over the SAME date in
    one filed income statement.
    """
    assert band_frequency("Three Months Ended") == sem.PeriodFrequency.QUARTER
    assert band_frequency("Three Fiscal Quarters Ended") == sem.PeriodFrequency.YTD_9M
    assert band_frequency("Nine Months Ended") == sem.PeriodFrequency.YTD_9M
    assert band_frequency("Six Months Ended") == sem.PeriodFrequency.YTD_6M
    assert band_frequency("Year Ended") == sem.PeriodFrequency.ANNUAL
    assert band_frequency("Twelve Months Ended") == sem.PeriodFrequency.ANNUAL
    assert band_frequency("Fiscal Quarter Ended") == sem.PeriodFrequency.QUARTER


def test_a_table_with_no_duration_band_is_point_in_time():
    tables = parse_filing_tables(R.balance_sheet())
    balance = next(t for t in tables if t.kind == StatementKind.BALANCE_SHEET)
    assert {c.frequency for c in balance.period_columns} == {sem.PeriodFrequency.INSTANT}


def test_a_quarter_column_and_a_ytd_column_sharing_one_date_stay_apart():
    """Both columns say "April 30, 2026". Only the band says which is which."""
    result = extract_reported_actuals(R.QUARTER_AND_YTD_RELEASE, form="8-K",
                                      filed="2026-06-04")
    quarter = [f for f in result.canonical_facts
               if f.field == "revenue" and f.frequency == sem.PeriodFrequency.QUARTER]
    ytd = [f for f in result.canonical_facts
           if f.field == "revenue" and f.frequency == sem.PeriodFrequency.YTD_9M]
    assert [f.value for f in quarter] == [R.Q3_QUARTER_REVENUE]
    assert [f.value for f in ytd] == [R.Q3_YTD_REVENUE]

    candidate = result.candidates[0]
    assert candidate.period_type == sem.PeriodFrequency.QUARTER
    assert candidate.values["revenue"] == R.Q3_QUARTER_REVENUE


def test_a_bare_year_with_no_month_anywhere_resolves_no_period():
    """A fiscal year does not necessarily end on 31 December, and assuming it
    does moves a September filer's year by a quarter."""
    document = R.statement_table(
        "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS (IN MILLIONS)",
        ["2026", "2025"],
        [("Total revenue", ["1,450.0", "1,247.0"])],
        bands=[("Year Ended", 2)])
    tables = parse_filing_tables(document)
    assert tables[0].period_columns == []


# ---------------------------------------------------------------------------
# Section 9: scale and currency
# ---------------------------------------------------------------------------

def test_a_table_that_states_no_scale_produces_nothing():
    """"1,450.0" is a plausible figure in every scale there is."""
    result = extract_reported_actuals(R.NO_SCALE_RELEASE, form="8-K",
                                      filed="2026-09-02")
    assert result.candidates == []
    assert "UNKNOWN_SCALE" in result.rejection_codes()


def test_the_stated_scale_reaches_the_value():
    thousands = extract_reported_actuals(R.THOUSANDS_RELEASE, form="8-K",
                                         filed="2026-09-02")
    millions = extract_reported_actuals(R.COMPLETE_Q4_RELEASE, form="8-K",
                                        filed="2026-09-02")
    assert thousands.candidates[0].values["revenue"] == R.Q4["revenue"] * R.MILLIONS
    assert millions.candidates[0].values["revenue"] == R.Q4["revenue"] * R.MILLIONS


def test_a_euro_release_is_never_recorded_as_dollars():
    result = extract_reported_actuals(R.EURO_RELEASE, form="6-K",
                                      filed="2027-02-10")
    assert {f.currency for f in result.canonical_facts} == {"EUR"}
    assert result.candidates[0].currency == "EUR"
    assert result.candidates[0].values["revenue"] == R.EURO_REVENUE


# ---------------------------------------------------------------------------
# Sections 6 and 10: actuals, guidance and adjusted measures
# ---------------------------------------------------------------------------

def test_an_outlook_table_never_becomes_a_reported_actual():
    """The outlook table has the same shape and the same row names as the
    income statement above it. Only the caption differs."""
    result = extract_reported_actuals(R.RELEASE_WITH_OUTLOOK, form="8-K",
                                      filed="2026-09-02")
    values = [f.value for f in result.canonical_facts if f.field == "revenue"]
    assert R.GUIDED_Q1_REVENUE not in values
    assert R.GUIDED_FY_REVENUE not in values
    assert R.Q4["revenue"] * R.MILLIONS in values
    assert result.candidates[0].statement_completeness == StatementCompleteness.COMPLETE


def test_an_analyst_consensus_table_is_not_the_companys_report():
    result = extract_reported_actuals(R.RELEASE_WITH_ANALYST_TABLE, form="8-K",
                                      filed="2026-09-02")
    values = [f.value for f in result.canonical_facts if f.field == "revenue"]
    assert R.ANALYST_REVENUE not in values


def test_an_adjusted_figure_never_fills_the_gaap_line():
    """The reconciliation's rows are called "Operating income" and "Net
    income", for the same period, with different numbers. Nothing in the row
    distinguishes them and nothing ever will."""
    result = extract_reported_actuals(R.COMPLETE_Q4_RELEASE, form="8-K",
                                      filed="2026-09-02")
    candidate = next(c for c in result.candidates if c.period_end == R.Q4_END_ISO)
    assert candidate.values["operating_income"] == R.Q4["operating_income"] * R.MILLIONS
    assert candidate.values["operating_income"] != R.ADJUSTED_OPERATING_INCOME
    assert candidate.values["net_income"] != R.ADJUSTED_NET_INCOME

    adjusted = [f for f in result.facts
                if f.accounting_basis == sem.AccountingBasis.ADJUSTED]
    assert adjusted, "the adjusted figures were dropped rather than labelled"
    assert all(not f.is_canonical for f in adjusted)


def test_a_row_label_never_decides_which_statement_it_is_in():
    tables = parse_filing_tables(R.non_gaap_reconciliation())
    assert tables[0].kind == StatementKind.NON_GAAP_RECONCILIATION


# ---------------------------------------------------------------------------
# Section 7: row identity
# ---------------------------------------------------------------------------

def test_total_current_assets_is_not_total_assets():
    """One word apart, six rows apart, a factor of three apart."""
    from finance.reported_actuals.facts import identify_row

    assert identify_row("Total assets") == "assets"
    assert identify_row("Total current assets") == "current_assets"
    assert identify_row("Total liabilities") == "liabilities"
    assert identify_row("Total current liabilities") == "current_liabilities"


def test_a_longer_phrase_containing_an_alias_is_a_different_line():
    from finance.reported_actuals.facts import identify_row

    assert identify_row("Cost of revenue") is None
    assert identify_row("Deferred revenue") is None
    assert identify_row("Total revenue") == "revenue"


def test_a_bare_basis_row_is_resolved_by_its_table_or_refused():
    """"Diluted" is a share count in a weighted-average table and an
    earnings-per-share figure in an income statement, nine orders of
    magnitude apart, and the row says which only through a heading."""
    from finance.reported_actuals.facts import identify_row

    assert identify_row("Diluted") is None
    assert identify_row("Diluted", StatementKind.INCOME_STATEMENT) is None
    assert identify_row("Diluted", StatementKind.SHARE_COUNT) == "diluted_shares"


def test_a_balance_under_a_duration_header_is_refused():
    """A cash-flow statement ends with "Cash and cash equivalents at end of
    period" under a "Three Months Ended" band. It is a balance, the column is
    a flow, and the two do not describe one fact."""
    from finance.reported_actuals.facts import FactRejection, facts_from_table

    document = R.statement_table(
        "CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS (IN MILLIONS)",
        [R.Q4_END, R.Q4_PRIOR_END],
        [("Net cash provided by operating activities", ["358.0", "322.0"]),
         ("Cash and cash equivalents", ["812.0", "690.0"])],
        bands=[("Three Months Ended", 2)])
    table = parse_filing_tables(document)[0]
    facts, rejections = facts_from_table(table)
    assert {f.field for f in facts} == {"operating_cash_flow"}
    assert any(code == FactRejection.FLOW_INSTANT_MISMATCH
               for code, _detail in rejections)


# ---------------------------------------------------------------------------
# Section 13: completeness is not redefined
# ---------------------------------------------------------------------------

def test_completeness_comes_from_the_existing_gate():
    complete = extract_reported_actuals(R.COMPLETE_Q4_RELEASE, form="8-K",
                                        filed="2026-09-02")
    headline = extract_reported_actuals(R.HEADLINE_ONLY_RELEASE, form="8-K",
                                        filed="2026-09-02")
    assert complete.candidates[0].statement_completeness == \
        StatementCompleteness.COMPLETE
    assert headline.candidates[0].statement_completeness == \
        StatementCompleteness.HEADLINE_ONLY


def test_prose_is_not_a_financial_statement():
    result = extract_reported_actuals(R.PROSE_ONLY_RELEASE, form="8-K",
                                      filed="2026-09-02")
    assert result.candidates == []


# ---------------------------------------------------------------------------
# Section 18: preliminary until the periodic filing arrives
# ---------------------------------------------------------------------------

def test_a_filed_release_is_preliminary_not_final():
    from finance.extraction.schema import FinalityStatus, SourceType

    result = extract_reported_actuals(R.COMPLETE_Q4_RELEASE, form="8-K",
                                      filed="2026-09-02")
    candidate = result.candidates[0]
    assert candidate.finality == FinalityStatus.UNAUDITED_PRELIMINARY
    assert candidate.source_type == SourceType.PRELIMINARY_EARNINGS_RELEASE


def test_the_same_statements_on_a_10k_are_audited():
    from finance.extraction.schema import FinalityStatus, SourceType

    result = extract_reported_actuals(R.PERIODIC_FILING_MATCHING, form="10-K",
                                      filed="2026-10-15")
    candidate = result.candidates[0]
    assert candidate.finality == FinalityStatus.AUDITED
    assert candidate.source_type == SourceType.FORMAL_PERIODIC_FILING


# ---------------------------------------------------------------------------
# Section 17: provenance
# ---------------------------------------------------------------------------

def test_every_accepted_fact_carries_the_row_and_column_it_came_from():
    result = extract_reported_actuals(
        R.COMPLETE_Q4_RELEASE, accession="acc-1", form="8-K",
        filed="2026-09-02", document="exhibit991.htm")
    for fact in result.canonical_facts:
        assert fact.row_label and fact.row_label in R.COMPLETE_Q4_RELEASE
        assert fact.column_label and fact.column_label in R.COMPLETE_Q4_RELEASE
        assert fact.accession == "acc-1"
        assert fact.document == "exhibit991.htm"
        assert fact.period_end
        assert fact.currency and fact.scale


# ---------------------------------------------------------------------------
# Sections 11, 12 and 14: the resolver decides, with values to decide on
# ---------------------------------------------------------------------------

def _facts_through_q3():
    from tests.fixtures import actualization_benchmark as AB
    return AB.build_company_facts(AB.HISTORY_THROUGH_Q3)


def _q4_release_candidates(document=R.COMPLETE_Q4_RELEASE, form="8-K",
                           filed="2026-09-02", accession="acc-8k"):
    result = extract_reported_actuals(document, accession=accession, form=form,
                                      filed=filed, document="ex991.htm")
    return [c for c in result.candidates if c.period_end == R.Q4_END_ISO], result


def test_a_complete_release_advances_the_period_past_the_10q():
    """THE CASE THE WHOLE PHASE EXISTS FOR.

    CompanyFacts stops at the Q3 10-Q. The Q4 release is filed and the 10-K is
    weeks away. The resolver already knew what to do with this candidate; it
    had never been given one.
    """
    from finance import actualization_runtime as ar

    facts = _facts_through_q3()
    candidates, _result = _q4_release_candidates()

    without, _obs = ar.resolve_actual_state(facts, as_of="2026-09-15", mode="v2")
    assert without.period_end == "2026-04-30"

    with_release, observation = ar.resolve_actual_state(
        facts, as_of="2026-09-15", mode="v2", extra_candidates=candidates)
    assert with_release.period_end == R.Q4_END_ISO
    assert with_release.selected_primary_source.form == "8-K"
    assert with_release.state_status == "PRELIMINARY_REPORTED_ACTUAL"
    assert observation.source_candidates_offered == 1


def test_an_incomplete_release_does_not_replace_the_state():
    from finance import actualization_runtime as ar

    facts = _facts_through_q3()
    candidates, _result = _q4_release_candidates(R.HEADLINE_ONLY_RELEASE)
    resolution, _obs = ar.resolve_actual_state(
        facts, as_of="2026-09-15", mode="v2", extra_candidates=candidates)
    assert resolution.period_end == "2026-04-30"
    assert "NEWER_SOURCE_INCOMPLETE" in resolution.codes


def test_the_later_periodic_filing_supersedes_without_a_second_period():
    from finance import actualization_runtime as ar

    facts = _facts_through_q3()
    release, _r1 = _q4_release_candidates()
    filing, _r2 = _q4_release_candidates(
        R.PERIODIC_FILING_MATCHING, form="10-K", filed="2026-10-15",
        accession="acc-10k")

    resolution, _obs = ar.resolve_actual_state(
        facts, as_of="2026-11-01", mode="v2",
        extra_candidates=release + filing)
    assert resolution.period_end == R.Q4_END_ISO
    assert resolution.selected_primary_source.form == "10-K"
    assert [s.form for s in resolution.superseded_sources] == ["8-K"]
    assert resolution.conflicts == []


def test_a_material_same_period_disagreement_is_recorded():
    """SECTION 12, AND THE REASON `values` HAD TO BE POPULATED.

    Until candidates carried numbers, `detect_same_period_conflicts` compared
    nothing: a preliminary release could disagree with the 10-K that followed
    it by any margin at all and the state said nothing.
    """
    from finance import actualization_runtime as ar

    facts = _facts_through_q3()
    release, _r1 = _q4_release_candidates()
    filing, _r2 = _q4_release_candidates(
        R.PERIODIC_FILING_CONFLICTING, form="10-K", filed="2026-10-15",
        accession="acc-10k")
    assert release[0].values["revenue"] != filing[0].values["revenue"]

    resolution, _obs = ar.resolve_actual_state(
        facts, as_of="2026-11-01", mode="v2",
        extra_candidates=release + filing)
    assert [c.metric for c in resolution.conflicts] == ["revenue"]
    conflict = resolution.conflicts[0]
    assert conflict.authoritative_source == "10-K"
    assert conflict.other_source == "8-K"
    # One economic period, not two, and the release is still provenance.
    assert resolution.period_end == R.Q4_END_ISO
    assert [s.form for s in resolution.superseded_sources] == ["8-K"]


def test_a_quarter_is_never_reconciled_against_a_year():
    """One period end carries a quarter and a full year. Two sources that
    reported different durations disagree about nothing."""
    from finance.actualization import detect_same_period_conflicts
    from finance.extraction.schema import ReportedActualCandidate

    quarter = ReportedActualCandidate(
        period_end=R.Q4_END_ISO, period_type=sem.PeriodFrequency.QUARTER,
        form="8-K", values={"revenue": 1_450.0})
    annual = ReportedActualCandidate(
        period_end=R.Q4_END_ISO, period_type=sem.PeriodFrequency.ANNUAL,
        form="10-K", values={"revenue": 5_460.0})
    assert detect_same_period_conflicts(annual, [quarter]) == []

    same = ReportedActualCandidate(
        period_end=R.Q4_END_ISO, period_type=sem.PeriodFrequency.QUARTER,
        form="10-K", values={"revenue": 1_385.0})
    assert [c.metric for c in detect_same_period_conflicts(same, [quarter])] == ["revenue"]


def test_the_companyfacts_path_now_populates_values():
    """Section 11. Reachability is the whole point: a detector nothing can
    reach is a detector that reports nothing."""
    from finance.extraction.document_resolver import discover_reported_actuals

    facts = _facts_through_q3()
    candidates = discover_reported_actuals(facts)
    assert candidates
    for candidate in candidates:
        assert candidate.values, candidate.period_end
        assert candidate.period_type


# ---------------------------------------------------------------------------
# Section 19: the existing TTM layer follows the new period
# ---------------------------------------------------------------------------

def test_the_twelve_month_window_rolls_forward_onto_the_release_quarter():
    """No second calculator. The accepted facts are projected into the shape
    `finance/ttm.py` already reads, and it builds the window it always did."""
    from finance import ttm as ttm_module
    from finance.actualization import reconstruct_ttm

    facts = _facts_through_q3()
    _candidates, result = _q4_release_candidates()
    merged = merge_company_facts(
        facts, company_facts_overlay(result.facts, base=facts))

    before = reconstruct_ttm(facts, R.Q4_END_ISO)
    after = reconstruct_ttm(merged, R.Q4_END_ISO)
    assert before.status == "LIMITED"
    assert after.status == "COMPLETE"

    window = ttm_module.build_ttm(merged, "revenue", reference_end=R.Q4_END_ISO)
    assert window.quarters_included == (
        "2025-08-01..2025-10-31", "2025-11-01..2026-01-31",
        "2026-02-01..2026-04-30", "2026-05-01..2026-07-31")
    assert len(set(window.quarters_included)) == 4
    assert "acc-8k" in window.source_accessions


def test_the_overlay_never_mutates_the_payload_it_was_built_from():
    facts = _facts_through_q3()
    import copy
    before = copy.deepcopy(facts)
    _candidates, result = _q4_release_candidates()
    merge_company_facts(facts, company_facts_overlay(result.facts, base=facts))
    assert facts == before


def test_an_overlay_row_joins_the_issuers_own_concept_series():
    """`period_facts` builds a series from ONE winning concept and prefers the
    one reaching furthest forward. An overlay filed under a different spelling
    of the same line becomes a new one-fact series that wins on recency and
    can build nothing -- measured: revenue went from a clean four-quarter
    window to no window at all."""
    facts = _facts_through_q3()
    _candidates, result = _q4_release_candidates()
    overlay = company_facts_overlay(result.facts, base=facts)
    concepts = set((overlay.get("facts") or {}).get("us-gaap") or {})
    assert "Revenues" in concepts
    assert "RevenueFromContractWithCustomerExcludingAssessedTax" not in concepts


def test_a_missing_quarter_still_leaves_the_window_limited():
    """Section 19's other half: when the quarterly information genuinely is
    not there, the existing LIMITED behaviour stands rather than a number
    being invented for it."""
    from finance.actualization import reconstruct_ttm

    facts = _facts_through_q3()
    _candidates, result = _q4_release_candidates(R.HEADLINE_ONLY_RELEASE)
    merged = merge_company_facts(
        facts, company_facts_overlay(result.facts, base=facts))
    rebuilt = reconstruct_ttm(merged, R.Q4_END_ISO)
    assert rebuilt.status == "LIMITED"
    assert "operating_cash_flow" in rebuilt.stale_metrics


# ---------------------------------------------------------------------------
# Section 28: the runtime seam
# ---------------------------------------------------------------------------

class _Fetcher:
    """A stand-in for the SEC coordinator. No network, same interface."""

    def __init__(self, index_html, exhibit_html, fail_on=None):
        self.index_html = index_html
        self.exhibit_html = exhibit_html
        self.fail_on = fail_on
        self.calls = []

    def filing_index(self, accession):
        self.calls.append(("index", accession))
        if self.fail_on == "index":
            raise RuntimeError("the filing index is unavailable")
        return self.index_html

    def exhibit(self, accession, document):
        self.calls.append(("exhibit", accession, document))
        if self.fail_on == "exhibit":
            raise RuntimeError("the exhibit is unavailable")
        return self.exhibit_html


_SUBMISSIONS = R.submissions(
    R.filing("8-K", "acc-1", "2026-09-02", items="2.02,9.01",
             report_date="2026-09-02", description="PRESS RELEASE",
             document="exhibit991.htm"))


def _run(mode, fetcher=None, **kwargs):
    from finance.reported_actuals import runtime as rt

    return rt.discover_release_candidates(
        "BENCH", _SUBMISSIONS, mode=mode,
        fetcher=fetcher or _Fetcher(R.EARNINGS_INDEX, R.COMPLETE_Q4_RELEASE),
        **kwargs)


def test_v1_fetches_nothing_and_produces_nothing():
    fetcher = _Fetcher(R.EARNINGS_INDEX, R.COMPLETE_Q4_RELEASE)
    result = _run("v1", fetcher)
    assert result.candidates == ()
    assert fetcher.calls == [], "v1 reached the network"
    assert result.observation.to_dict()["mode"] == "v1"


def test_compare_reads_everything_and_offers_nothing():
    """A compare mode that changed which period was selected would be v2 by
    default wearing a diagnostic's name."""
    result = _run("compare")
    assert result.candidates == ()
    assert result.facts_overlay == {}
    observation = result.observation
    assert observation.candidates_offered is False
    assert observation.releases_detected == 1
    assert observation.candidate_count >= 1
    assert observation.latest_release_period == R.Q4_END_ISO


def test_v2_offers_the_candidates():
    result = _run("v2", companyfacts_latest_period="2026-04-30")
    assert result.candidates
    assert result.observation.candidates_offered is True
    assert result.observation.advances_period is True
    assert result.observation.latest_companyfacts_period == "2026-04-30"


def test_an_unknown_mode_is_diagnosed_as_a_typo_not_as_a_data_failure():
    result = _run("V2-please")
    assert result.candidates == ()
    assert result.observation.failure_code == "UNKNOWN_MODE"


def test_an_unreachable_document_is_structured_never_fabricated():
    for failing in ("index", "exhibit"):
        result = _run("v2", _Fetcher(R.EARNINGS_INDEX, R.COMPLETE_Q4_RELEASE,
                                     fail_on=failing))
        assert result.candidates == ()
        assert result.observation.failure_code == "NO_ACTUAL_FACTS_EXTRACTED"
        assert result.observation.rejection_codes


def test_a_missing_filing_index_fails_closed_under_v2():
    from finance.reported_actuals import runtime as rt

    with pytest.raises(rt.ReportedActualsFailure):
        rt.discover_release_candidates("BENCH", None, mode="v2")

    result = rt.discover_release_candidates("BENCH", None, mode="compare")
    assert result.observation.failure_code == "NO_SUBMISSIONS_INDEX"


def test_the_observation_carries_no_document_text():
    """Section 29. Diagnostics, never contents."""
    result = _run("v2")
    payload = result.observation.to_dict()
    rendered = repr(payload)
    assert "<table" not in rendered
    assert "Total revenue" not in rendered
    assert payload["facts_accepted"] > 0
    assert payload["tables_read"] > 0


def test_production_defaults_are_unchanged():
    from tools import config

    assert config.finance_reported_actuals_mode() == "v1"
    assert config.finance_actualization_mode() == "v1"
    assert config.finance_extraction_mode() == "v1"


def test_no_sec_retrieval_lives_in_the_semantic_modules():
    """Section 15. The boundary, asserted rather than described."""
    import pathlib

    for name in ("finance/actualization.py", "finance/ttm.py",
                 "finance/freshness.py", "finance/dcf.py",
                 "finance/dcf_packet.py", "finance/research_pipeline.py",
                 "finance/reported_actuals/tables.py",
                 "finance/reported_actuals/facts.py",
                 "finance/reported_actuals/candidates.py"):
        text = pathlib.Path(name).read_text(encoding="utf-8")
        # Names that FETCH, not the letters that spell one. A URL quoted in a
        # docstring is documentation; `import requests` is a socket.
        for forbidden in ("import requests", "import urllib", "urllib.request",
                          "requests.get", "get_sec_coordinator",
                          "SecEdgarClient", "safe_get(", "coordinator.fetch"):
            assert forbidden not in text, f"{name} reaches the network"


# ---------------------------------------------------------------------------
# Found by the live/historical run, kept here so they stay found
# ---------------------------------------------------------------------------

def test_a_statement_title_behind_heavy_markup_is_still_found():
    """ACTUAL_SECTION_SELECTION.

    Measured on a live exhibit: "CONDENSED CONSOLIDATED BALANCE SHEETS" sat
    718 HTML characters above its table, behind inline styling, and a
    700-character caption window missed it by eighteen. The balance sheet was
    then an unidentified table and every figure in it was discarded.
    """
    from finance.reported_actuals.tables import preceding_caption

    padding = '<div style="' + ("x" * 3000) + '">&nbsp;</div>'
    document = ("<div>CONDENSED CONSOLIDATED BALANCE SHEETS</div>"
                "<div>($ in millions)</div>" + padding + "<table><tr></tr></table>")
    start = document.index("<table")
    assert "BALANCE SHEET" in preceding_caption(document, start).upper()


def test_the_caption_walk_stops_at_the_previous_table():
    """The other half: a bigger window must not reach the PREVIOUS statement's
    caption and label this table with it."""
    from finance.reported_actuals.tables import preceding_caption

    document = ("<div>CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS</div>"
                "<table><tr><td>x</td></tr></table>"
                "<div>($ in millions)</div><table><tr></tr></table>")
    floor = document.index("</table>") + len("</table>")
    start = document.rindex("<table")
    caption = preceding_caption(document, start, floor=floor)
    assert "CASH FLOWS" not in caption.upper()


def test_a_release_headline_never_becomes_a_tables_caption():
    """The walk stops as soon as the text names a statement, so "Reports
    Fourth Quarter Results and Provides Outlook" cannot classify a condensed
    income statement as an outlook table."""
    result = extract_reported_actuals(R.RELEASE_WITH_OUTLOOK, form="8-K",
                                      filed="2026-09-02")
    kinds = {t["kind"] for t in result.tables}
    assert StatementKind.INCOME_STATEMENT in kinds
    assert result.candidates[0].values["revenue"] == R.Q4["revenue"] * R.MILLIONS


def test_the_overlay_defers_to_a_period_the_payload_already_reports():
    """TTM_INTEGRATION.

    A release's quarter boundaries are derived here, not read from the
    document, so an overlay row for a period CompanyFacts already holds is a
    SECOND quarter with a slightly different start. `finance/ttm.py` then
    correctly refuses to sum two overlapping quarters and a COMPLETE window
    becomes LIMITED -- measured on two live issuers, one of whose period had
    not moved at all.
    """
    from finance.actualization import reconstruct_ttm

    facts = _facts_through_q3()
    result = extract_reported_actuals(
        R.QUARTER_AND_YTD_RELEASE, accession="acc", form="8-K",
        filed="2026-06-04", document="ex991.htm")
    # This release describes 2026-04-30, which the Q3 10-Q already reports.
    overlay = company_facts_overlay(result.facts, base=facts)
    assert overlay == {} or not any(
        row.get("end") == "2026-04-30"
        for concepts in (overlay.get("facts") or {}).values()
        for entry in concepts.values()
        for rows in (entry.get("units") or {}).values()
        for row in rows)

    merged = merge_company_facts(facts, overlay)
    assert reconstruct_ttm(merged, "2026-04-30").status == "COMPLETE"


def test_an_overlay_quarter_begins_where_the_last_reported_period_ended():
    """A 52/53-week filer's quarter is not three calendar months.

    Counting three months back from a 2 August period end gives 1 June: a
    62-day "quarter" leaving a 29-day hole after the previously filed period.
    Consecutive reporting periods abut, and both dates come from filings.
    """
    from finance.reported_actuals.candidates import _overlay_start
    from finance.reported_actuals.facts import ActualFinancialFact

    fact = ActualFinancialFact(
        field="revenue", value=1.0, raw_value=1.0, currency="USD",
        scale="millions", period_end="2026-08-02", period_start="2026-06-01",
        frequency=sem.PeriodFrequency.QUARTER,
        flow_or_instant=sem.FlowOrInstant.FLOW,
        accounting_basis=sem.AccountingBasis.GAAP,
        statement_kind=StatementKind.INCOME_STATEMENT,
        row_label="Net revenue", column_label="August 2, 2026", band_label="")
    assert _overlay_start(fact, None, {"2026-05-03", "2026-02-01"}) == "2026-05-04"
    # No plausible previous period: the derived start stands, labelled as such.
    assert _overlay_start(fact, None, {"2019-01-01"}) == "2026-06-01"


def test_a_refused_release_cannot_move_a_twelve_month_window():
    """A source the resolver declined must not degrade the state it was not
    allowed to advance."""
    from finance import actualization_runtime as ar
    from finance.reported_actuals.candidates import restrict_to_period

    facts = _facts_through_q3()
    result = extract_reported_actuals(
        R.HEADLINE_ONLY_RELEASE, accession="acc-8k", form="8-K",
        filed="2026-09-02", document="ex991.htm")
    candidates = [c for c in result.candidates if c.period_end == R.Q4_END_ISO]
    overlay = company_facts_overlay(result.facts, base=facts)

    resolution, observation = ar.resolve_actual_state(
        facts, as_of="2026-09-15", mode="v2",
        extra_candidates=candidates, extra_facts=overlay)
    # The release could not carry the state, so the period held...
    assert resolution.period_end == "2026-04-30"
    # ...and the window is the one it always was.
    assert observation.ttm_status == "COMPLETE"
    assert restrict_to_period(overlay, "2026-04-30") == {}
