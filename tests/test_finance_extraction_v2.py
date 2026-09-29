"""Finance Extraction V2 — the eighteen cases of §25.

The point of every one of these is the same: the model may propose anything,
and only what the SOURCE supports is accepted. A case that reads like a
capability test ("does it find the guidance?") is paired with the refusal
that makes the capability safe ("and does it refuse the one that is not
there?").

No live model is called. A scripted extractor stands in for one, because the
subject under test is the ACCEPTANCE BOUNDARY, not a particular model's
reading ability — and a boundary tested against a real model's good day is a
boundary nobody has tested.
"""

import json

import pytest

from finance import guidance as gm
from finance import semantics as sem
from finance.extraction import document_resolver as dr
from finance.extraction.schema import (
    GuidanceAction,
    GuidanceCandidate,
    RejectionCode,
    SourceType,
    StatementCompleteness,
    ValueType,
)
from finance.extraction.semantic_extractor import (
    DocumentSection,
    ExtractionFailure,
    LocalModelGuidanceExtractor,
    select_sections,
)
from finance.extraction.validator import GuidanceCandidateValidator

N = gm.GuidanceMetricName


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _candidate(**kwargs):
    base = dict(confidence=0.9, issued_at="2026-06-10", prospective=True,
                target_period="FY2027", target_period_type="FISCAL_YEAR",
                basis="GAAP", action=GuidanceAction.NEW)
    base.update(kwargs)
    return GuidanceCandidate(**base)


def _validate(candidate, document=None, issued_at="2026-06-10", actuals=None):
    validator = GuidanceCandidateValidator(
        document_text=document if document is not None else candidate.source_sentence,
        issued_at=issued_at, reported_actuals=actuals)
    return validator.validate(candidate)


def _accept(candidate, **kwargs):
    metric, code, reason = _validate(candidate, **kwargs)
    assert metric is not None, f"rejected {code}: {reason}"
    return metric


def _reject(candidate, **kwargs):
    metric, code, reason = _validate(candidate, **kwargs)
    assert metric is None, f"accepted {metric.name}={metric.low}"
    return code, reason


def _scripted(statements_by_section):
    """A stand-in model that returns exactly what the test scripts."""
    payloads = list(statements_by_section)

    def ask(messages, tools=None, timeout=120, options=None, response_format=None):
        body = payloads.pop(0) if payloads else {"statements": []}
        return {"message": {"content": json.dumps(body)},
                "metrics": {"prompt_tokens": 1, "completion_tokens": 1}, "ok": True}

    return LocalModelGuidanceExtractor(ask)


# ---------------------------------------------------------------------------
# 1-3. an annual revenue guide, however the sentence is written
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence, action", [
    ("For fiscal year 2027, our revenue guidance is $90 billion.",
     GuidanceAction.NEW),
    ("For fiscal year 2027, we expect revenue of approximately $90 billion.",
     GuidanceAction.NEW),
    ("For fiscal year 2027, we confirm our prior revenue guidance of $90 billion.",
     GuidanceAction.REAFFIRMED),
])
def test_an_annual_revenue_guide_is_accepted_however_it_is_phrased(sentence, action):
    """Cases 1-3. Word order and verb form are a writer's choice; the
    quantity committed to is the same, and V1 needed a pattern for each."""
    metric = _accept(_candidate(
        metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT, value=90.0,
        unit="USD_BILLION", source_sentence=sentence, action=action))
    assert metric.name == N.CONSOLIDATED_REVENUE
    assert metric.low == pytest.approx(90.0)
    assert metric.unit == gm.GuidanceUnit.CURRENCY
    assert metric.scale == "billion"
    assert metric.fiscal_period == "FY2027"


def test_a_reaffirmation_is_recorded_as_one_statement_not_two():
    """Case 13. A company repeating itself has said one thing once."""
    sentence = "For fiscal year 2027, we confirm our prior revenue guidance of $90 billion."
    earlier = _candidate(metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT,
                         value=90.0, unit="USD_BILLION", source_sentence=sentence,
                         action=GuidanceAction.NEW, issued_at="2026-03-10")
    later = _candidate(metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT,
                       value=90.0, unit="USD_BILLION", source_sentence=sentence,
                       action=GuidanceAction.REAFFIRMED, issued_at="2026-06-10")
    validator = GuidanceCandidateValidator(document_text=sentence, issued_at="2026-06-10")
    accepted, rejected = validator.validate_all([earlier, later])
    assert len(accepted) == 1, [(m.name, m.issued_at) for m in accepted]
    assert accepted[0].issued_at == "2026-06-10"
    assert any(code == RejectionCode.DUPLICATE_ECONOMIC_IDENTITY
               for _c, code, _r in rejected)


# ---------------------------------------------------------------------------
# 4. a margin is a margin
# ---------------------------------------------------------------------------

_EBITDA_MARGIN = ("Third quarter fiscal year 2027 Adjusted EBITDA guidance of "
                  "approximately 68 percent of projected revenue.")


def test_a_percentage_of_revenue_is_a_margin():
    """Case 4."""
    metric = _accept(_candidate(
        metric_id=N.ADJUSTED_EBITDA, value_type=ValueType.MARGIN, value=0.68,
        unit="PERCENT", denominator_metric="revenue",
        target_period="Q3 FY2027", target_period_type="QUARTER",
        basis="NON_GAAP", source_sentence=_EBITDA_MARGIN))
    assert metric.name == N.ADJUSTED_EBITDA_MARGIN
    assert metric.low == pytest.approx(0.68)
    assert metric.unit == gm.GuidanceUnit.RATIO


def test_the_same_sentence_read_as_an_amount_is_refused():
    """§7's worked example, and the single most important test here.

    The number 68 IS in the sentence, so every check that asks only "is this
    number present" passes it -- and what reaches the canonical layer is a
    quarterly profit larger than the company's revenue.
    """
    code, reason = _reject(_candidate(
        metric_id=N.ADJUSTED_EBITDA, value_type=ValueType.POINT, value=68.0,
        unit="USD_BILLION", target_period="Q3 FY2027",
        target_period_type="QUARTER", source_sentence=_EBITDA_MARGIN))
    assert code == RejectionCode.UNIT_METRIC_MISMATCH
    assert "percent" in reason


def test_a_margin_claimed_without_a_stated_denominator_is_refused():
    code, _reason = _reject(_candidate(
        metric_id=N.OPERATING_INCOME, value_type=ValueType.MARGIN, value=0.21,
        unit="PERCENT", denominator_metric="revenue",
        source_sentence="For fiscal year 2027 operating income is expected to rise 21 "
                        "percent."))
    assert code in (RejectionCode.DENOMINATOR_NOT_IN_EVIDENCE,
                    RejectionCode.UNIT_METRIC_MISMATCH)


# ---------------------------------------------------------------------------
# 5. a historical table is not guidance
# ---------------------------------------------------------------------------

def test_a_historical_share_table_is_refused_even_beside_forward_language():
    """Case 5. The release says "expects" a paragraph above; the table is
    still a record of a quarter that has closed."""
    sentence = ("Weighted-average shares used in computing earnings per share, "
                "Three Months Ended June 30, 2026: 139,933")
    code, _reason = _reject(_candidate(
        metric_id=N.SHARE_COUNT, value_type=ValueType.POINT, value=139933.0,
        unit="SHARES", source_sentence=sentence))
    assert code == RejectionCode.HISTORICAL_TABLE


def test_an_analyst_estimate_is_not_management_guidance():
    sentence = ("Analysts surveyed expect fiscal year 2027 revenue of $92 billion.")
    code, _reason = _reject(_candidate(
        metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT, value=92.0,
        unit="USD_BILLION", source_sentence=sentence))
    assert code == RejectionCode.ANALYST_ESTIMATE


# ---------------------------------------------------------------------------
# 6. two horizons from one release
# ---------------------------------------------------------------------------

def test_a_quarter_and_a_full_year_from_one_release_both_survive():
    """Case 6."""
    document = ("For the first quarter of fiscal 2027, we expect revenue growth of "
                "27% to 29%. For fiscal year 2027, our revenue guidance is $90 billion.")
    quarter = _candidate(
        metric_id=N.CONSOLIDATED_REVENUE_GROWTH, value_type=ValueType.GROWTH_RATE,
        low=0.27, high=0.29, unit="PERCENT", target_period="Q1 FY2027",
        target_period_type="QUARTER",
        source_sentence="For the first quarter of fiscal 2027, we expect revenue "
                        "growth of 27% to 29%.")
    year = _candidate(
        metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT, value=90.0,
        unit="USD_BILLION",
        source_sentence="For fiscal year 2027, our revenue guidance is $90 billion.")
    validator = GuidanceCandidateValidator(document_text=document, issued_at="2026-06-10")
    accepted, rejected = validator.validate_all([quarter, year])
    horizons = {(m.name, m.fiscal_period) for m in accepted}
    assert (N.CONSOLIDATED_REVENUE_GROWTH, "Q1 FY2027") in horizons, (horizons, rejected)
    assert (N.CONSOLIDATED_REVENUE, "FY2027") in horizons, horizons


# ---------------------------------------------------------------------------
# 7. actualization
# ---------------------------------------------------------------------------

def test_guidance_for_a_reported_period_is_no_longer_current():
    """Case 7. Once the results are in, the outlook describes the past."""
    actuals = [dr.ReportedActualCandidate(
        period_end="2026-05-31", fiscal_year=2026, fiscal_period="FY",
        statement_completeness=StatementCompleteness.COMPLETE, form="10-K")]
    code, reason = _reject(
        _candidate(metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT,
                   value=67.0, unit="USD_BILLION", target_period="FY2026",
                   source_sentence="For fiscal year 2026, we expect revenue of "
                                   "$67 billion."),
        issued_at="2025-12-10", actuals=actuals)
    assert code == RejectionCode.TARGET_PERIOD_COMPLETED
    assert "actual results" in reason


def test_guidance_for_a_still_forward_period_survives_the_same_check():
    actuals = [dr.ReportedActualCandidate(
        period_end="2026-05-31", fiscal_year=2026, fiscal_period="FY",
        statement_completeness=StatementCompleteness.COMPLETE, form="10-K")]
    metric = _accept(
        _candidate(metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT,
                   value=90.0, unit="USD_BILLION", target_period="FY2027",
                   source_sentence="For fiscal year 2027, our revenue guidance is "
                                   "$90 billion."),
        actuals=actuals)
    assert metric.fiscal_period == "FY2027"


# ---------------------------------------------------------------------------
# 8-9. actual-source precedence
# ---------------------------------------------------------------------------

def _actual(end, form, completeness, filed, **kwargs):
    return dr.ReportedActualCandidate(
        period_end=end, form=form, issued_at=filed,
        source_type=dr.source_type_for_form(form),
        statement_completeness=completeness,
        finality=dr.finality_for_form(form), **kwargs)


def test_a_complete_newer_earnings_release_advances_the_current_period():
    """Case 8. The 10-K is weeks away; the results are already public."""
    resolution = dr.resolve_current_period([
        _actual("2026-02-28", "10-Q", StatementCompleteness.COMPLETE, "2026-03-15"),
        _actual("2026-05-31", "8-K", StatementCompleteness.COMPLETE, "2026-06-10"),
    ], as_of="2026-06-20")
    assert resolution.period_end == "2026-05-31"
    assert resolution.selected.source_type == SourceType.PRELIMINARY_EARNINGS_RELEASE
    assert resolution.selected.finality == "UNAUDITED_PRELIMINARY"
    assert "preliminary and unaudited" in resolution.reason


def test_an_incomplete_release_does_not_displace_a_complete_filing():
    """Case 9, and §13's guard. Newer is not the question; complete is."""
    resolution = dr.resolve_current_period([
        _actual("2026-02-28", "10-Q", StatementCompleteness.COMPLETE, "2026-03-15"),
        _actual("2026-05-31", "8-K", StatementCompleteness.HEADLINE_ONLY, "2026-06-10",
                missing_metrics=("assets", "operating_cash_flow")),
    ], as_of="2026-06-20")
    assert resolution.period_end == "2026-02-28"
    assert resolution.rejected, "the newer incomplete source was not explained"
    _candidate_, why = resolution.rejected[0]
    assert "does not advance the state" in why


def test_a_periodic_filing_outranks_a_release_for_the_same_period():
    """Authority breaks a TIE on period; it never beats a newer period."""
    resolution = dr.resolve_current_period([
        _actual("2026-05-31", "8-K", StatementCompleteness.COMPLETE, "2026-06-10"),
        _actual("2026-05-31", "10-K", StatementCompleteness.COMPLETE, "2026-07-20"),
    ], as_of="2026-08-01")
    assert resolution.selected.form == "10-K"


def test_completeness_is_decided_by_the_three_statements():
    complete, missing = dr.classify_completeness(
        {"revenue", "assets", "stockholders_equity", "cash_and_cash_equivalents",
         "operating_cash_flow"})
    assert complete == StatementCompleteness.COMPLETE and missing == ()
    headline, missing = dr.classify_completeness({"revenue"})
    assert headline == StatementCompleteness.HEADLINE_ONLY
    assert "operating_cash_flow" in missing


def test_a_balance_sheet_that_does_not_total_to_liabilities_is_still_complete():
    """TOTAL LIABILITIES IS AN OPTIONAL LINE, AND REQUIRING IT REJECTED 10-Ks.

    A classified balance sheet may present current and non-current liabilities
    and total straight to `LiabilitiesAndStockholdersEquity`, never tagging
    `Liabilities` at all. That is ordinary US GAAP presentation, not an
    incomplete filing -- and an issuer that files that way had EVERY document
    it has ever filed classified PARTIAL, so no period could be resolved for
    it and the whole layer went dark on the issuer.

    The anchors are the pair `finance/period_facts.py` already uses to decide
    that a balance sheet was filed at all, so both callers now agree on what a
    balance sheet is. Two definitions of one idea is how a correct filing gets
    refused.
    """
    complete, missing = dr.classify_completeness(
        {"revenue", "assets", "stockholders_equity", "cash_and_cash_equivalents",
         "operating_cash_flow", "net_income", "operating_income"})
    assert complete == StatementCompleteness.COMPLETE, missing


def test_the_guard_against_a_headline_release_still_holds():
    """The other half. Loosening the anchors must not admit a press release
    that never published a balance sheet."""
    headline, missing = dr.classify_completeness(
        {"revenue", "net_income", "earnings_per_share"})
    assert headline == StatementCompleteness.HEADLINE_ONLY
    assert set(missing) >= {"assets", "stockholders_equity",
                            "cash_and_cash_equivalents", "operating_cash_flow"}

    partial, missing = dr.classify_completeness(
        {"revenue", "net_income", "operating_income", "assets",
         "stockholders_equity", "cash_and_cash_equivalents"})
    assert partial == StatementCompleteness.PARTIAL
    assert "operating_cash_flow" in missing


def test_discovery_reads_the_issuers_own_reporting_framework():
    """A FOREIGN PRIVATE ISSUER'S FACTS WERE INVISIBLE TO DISCOVERY.

    `finance/period_facts.py::_concept_map` exists because every reader that
    used the module-level us-gaap `CONCEPT_MAP` directly made an ifrs-full
    filer's entire history unreadable -- `revenue` resolved to nothing and the
    pipeline concluded the company had published no financials. Discovery was
    written later and reintroduced exactly that: it found `Assets` (spelled
    the same in both frameworks) and nothing else, classified every filing
    HEADLINE_ONLY, and resolved no period for the issuer at all.

    The taxonomy is the issuer's, not the reader's.
    """
    facts = {"facts": {"ifrs-full": {
        "Revenue": {"units": {"USD": [
            {"start": "2026-01-01", "end": "2026-06-30", "val": 100.0,
             "form": "20-F", "accn": "a1", "filed": "2026-08-01",
             "fy": 2026, "fp": "FY"}]}},
        "Assets": {"units": {"USD": [
            {"end": "2026-06-30", "val": 900.0, "form": "20-F", "accn": "a1",
             "filed": "2026-08-01", "fy": 2026, "fp": "FY"}]}},
        "Equity": {"units": {"USD": [
            {"end": "2026-06-30", "val": 400.0, "form": "20-F", "accn": "a1",
             "filed": "2026-08-01", "fy": 2026, "fp": "FY"}]}},
        "CashAndCashEquivalents": {"units": {"USD": [
            {"end": "2026-06-30", "val": 120.0, "form": "20-F", "accn": "a1",
             "filed": "2026-08-01", "fy": 2026, "fp": "FY"}]}},
        "CashFlowsFromUsedInOperatingActivities": {"units": {"USD": [
            {"start": "2026-01-01", "end": "2026-06-30", "val": 60.0,
             "form": "20-F", "accn": "a1", "filed": "2026-08-01",
             "fy": 2026, "fp": "FY"}]}},
    }}}
    candidates = dr.discover_reported_actuals(facts)
    assert candidates, "the issuer's facts were not seen at all"
    assert candidates[0].statement_completeness == StatementCompleteness.COMPLETE,         candidates[0].missing_metrics
    resolution = dr.resolve_current_period(candidates, as_of="2026-09-10")
    assert resolution.period_end == "2026-06-30"
    assert resolution.selected.form == "20-F"


def test_an_issuer_that_never_tags_total_liabilities_still_resolves_a_period():
    """The failure end to end: not one candidate, but every one of them.

    Without an anchor the issuer actually reports, `resolve_current_period`
    finds no COMPLETE source anywhere in the issuer's history and returns no
    period -- for a company whose 10-K is on file.
    """
    present = ("revenue", "net_income", "operating_income", "assets",
               "stockholders_equity", "cash_and_cash_equivalents",
               "operating_cash_flow", "capital_expenditure")
    completeness, _missing = dr.classify_completeness(set(present))
    resolution = dr.resolve_current_period([
        _actual("2026-02-28", "10-Q", completeness, "2026-03-10"),
        _actual("2026-05-31", "10-K", completeness, "2026-06-20"),
    ], as_of="2026-09-10")
    assert resolution.period_end == "2026-05-31", resolution.reason
    assert resolution.selected.form == "10-K"


# ---------------------------------------------------------------------------
# 10-11. forecast-horizon eligibility is unchanged and still consulted
# ---------------------------------------------------------------------------

def test_quarterly_growth_is_directional_for_an_annual_assumption():
    """Case 10. V2 changes how a statement is READ, not what it may do."""
    eligibility = sem.forecast_compatibility(
        evidence_metric=sem.MetricIdentity.REVENUE,
        evidence_frequency=sem.PeriodFrequency.QUARTER,
        assumption_metric=sem.MetricIdentity.REVENUE,
        assumption_frequency=sem.PeriodFrequency.ANNUAL)
    assert eligibility.status == sem.ForecastCompatibility.DIRECTIONAL_CORROBORATION
    assert not eligibility.may_set_magnitude


def test_annual_absolute_guidance_is_assumption_comparable():
    """Case 11. The accepted metric carries an ANNUAL target, which is what
    makes the existing derivation eligible to use it."""
    metric = _accept(_candidate(
        metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT, value=90.0,
        unit="USD_BILLION",
        source_sentence="For fiscal year 2027, our revenue guidance is $90 billion."))
    assert metric.period_type == gm.GuidancePeriodType.ANNUAL
    eligibility = sem.forecast_compatibility(
        evidence_metric=sem.MetricIdentity.REVENUE,
        evidence_frequency=sem.PeriodFrequency.ANNUAL,
        assumption_metric=sem.MetricIdentity.REVENUE,
        assumption_frequency=sem.PeriodFrequency.ANNUAL)
    assert eligibility.status == sem.ForecastCompatibility.ASSUMPTION_COMPARABLE


# ---------------------------------------------------------------------------
# 12. one economic target, one signal
# ---------------------------------------------------------------------------

def test_two_releases_naming_one_target_produce_one_signal():
    """Case 12, and the rule §9 asks to preserve: the economic target is what
    identifies a statement; how far away it was is metadata."""
    sentence = "For fiscal year 2027, our revenue guidance is $90 billion."
    march = _candidate(metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT,
                       value=88.0, unit="USD_BILLION", issued_at="2026-03-10",
                       source_sentence="For fiscal year 2027, our revenue guidance is "
                                       "$88 billion.")
    june = _candidate(metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT,
                      value=90.0, unit="USD_BILLION", issued_at="2026-06-10",
                      action=GuidanceAction.RAISED, source_sentence=sentence)
    document = march.source_sentence + " " + sentence
    validator = GuidanceCandidateValidator(document_text=document, issued_at="2026-06-10")
    accepted, _rejected = validator.validate_all([march, june])
    assert len(accepted) == 1
    assert accepted[0].low == pytest.approx(90.0), "the raised figure must be the signal"


# ---------------------------------------------------------------------------
# 14-18. the refusals that make the rest safe
# ---------------------------------------------------------------------------

def test_a_number_not_in_the_evidence_is_refused():
    """Case 14."""
    code, reason = _reject(_candidate(
        metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT, value=95.0,
        unit="USD_BILLION",
        source_sentence="For fiscal year 2027, our revenue guidance is $90 billion."))
    assert code == RejectionCode.VALUE_NOT_IN_EVIDENCE
    assert "95.0" in reason


def test_a_sentence_not_in_the_document_is_refused():
    """The model quoting something the document does not contain."""
    code, _reason = _reject(
        _candidate(metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT,
                   value=90.0, unit="USD_BILLION",
                   source_sentence="For fiscal year 2027, revenue will be $90 billion."),
        document="A completely different document about something else.")
    assert code == RejectionCode.EVIDENCE_NOT_IN_SOURCE


def test_a_candidate_with_no_evidence_is_refused():
    """Case 18."""
    code, reason = _reject(_candidate(
        metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT, value=90.0,
        unit="USD_BILLION", source_sentence=""), document="anything")
    assert code == RejectionCode.NO_EVIDENCE
    assert "nothing about it can be checked" in reason


def test_an_ambiguous_target_period_is_refused():
    """Case 17. "next year" is not a fiscal period."""
    code, _reason = _reject(_candidate(
        metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT, value=90.0,
        unit="USD_BILLION", target_period="next year",
        source_sentence="Next year we expect revenue guidance of $90 billion."))
    assert code == RejectionCode.AMBIGUOUS_TARGET_PERIOD


def test_an_unknown_metric_is_refused_rather_than_guessed():
    code, _reason = _reject(_candidate(
        metric_id="synergy_run_rate", value_type=ValueType.POINT, value=90.0,
        unit="USD_BILLION",
        source_sentence="For fiscal year 2027, we expect synergies of $90 billion."))
    assert code == RejectionCode.UNKNOWN_METRIC


def test_a_low_confidence_candidate_is_refused():
    code, _reason = _reject(_candidate(
        metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT, value=90.0,
        unit="USD_BILLION", confidence=0.2,
        source_sentence="For fiscal year 2027, our revenue guidance is $90 billion."))
    assert code == RejectionCode.LOW_CONFIDENCE


def test_a_release_cannot_guide_a_period_that_has_already_ended():
    code, _reason = _reject(
        _candidate(metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT,
                   value=90.0, unit="USD_BILLION", target_period="FY2020",
                   source_sentence="For fiscal year 2020, our revenue guidance is "
                                   "$90 billion."),
        issued_at="2026-06-10")
    assert code == RejectionCode.TARGET_PERIOD_IMPLAUSIBLE


# ---------------------------------------------------------------------------
# 16. the model failing is not a reason to trust something else
# ---------------------------------------------------------------------------

def test_invalid_json_fails_closed_after_one_repair():
    """Case 16. One structured repair, then a refusal -- never a silent
    fall back to an unvalidated read (§20)."""
    attempts = []

    def ask(messages, tools=None, timeout=120, options=None, response_format=None):
        attempts.append(messages)
        return {"message": {"content": "I think revenue will be around $90 billion."},
                "metrics": {}, "ok": True}

    from finance.extraction.semantic_extractor import DocumentSection

    extractor = LocalModelGuidanceExtractor(ask)
    sections = [DocumentSection(label="Outlook", text="Outlook text", start=0, end=12)]
    with pytest.raises(ExtractionFailure) as excinfo:
        extractor.extract(sections, issued_at="2026-06-10")
    assert excinfo.value.code in ("NO_JSON", "INVALID_JSON", "SCHEMA")
    assert len(attempts) == 2, "exactly one repair attempt is allowed"


def test_a_model_error_is_a_failure_not_an_empty_result():
    def ask(messages, tools=None, timeout=120, options=None, response_format=None):
        return {"ok": False, "message": {"content": ""}, "metrics": {}}

    from finance.extraction.semantic_extractor import DocumentSection
    extractor = LocalModelGuidanceExtractor(ask)
    with pytest.raises(ExtractionFailure):
        extractor.extract([DocumentSection("Outlook", "text", 0, 4)],
                          issued_at="2026-06-10")


def test_a_well_formed_response_produces_candidates():
    """The positive case, so the failure tests above are not vacuous."""
    from finance.extraction.semantic_extractor import DocumentSection
    extractor = _scripted([{"statements": [{
        "metric_id": "revenue", "value_type": "point", "value": 90.0,
        "unit": "USD_BILLION", "target_period": "FY2027",
        "target_period_type": "FISCAL_YEAR", "basis": "GAAP", "action": "NEW",
        "prospective": True, "confidence": 0.9,
        "source_sentence": "For fiscal year 2027, our revenue guidance is $90 billion."}]}])
    candidates = extractor.extract(
        [DocumentSection("Outlook", "For fiscal year 2027, our revenue guidance is "
                                    "$90 billion.", 0, 60)],
        issued_at="2026-06-10", document_id="acc-1")
    assert len(candidates) == 1
    assert candidates[0].metric_id == "revenue"
    assert candidates[0].document_id == "acc-1"


# ---------------------------------------------------------------------------
# section selection is deterministic and bounded
# ---------------------------------------------------------------------------

def test_section_selection_finds_a_labelled_outlook():
    text = ("Results\n\nRevenue was $67 billion.\n\nFinancial Outlook\n\n"
            "For fiscal year 2027, our revenue guidance is $90 billion.\n")
    sections = select_sections(text)
    assert sections and sections[0].label == "Financial Outlook"
    assert "90 billion" in sections[0].text
    assert "Revenue was" not in sections[0].text


def test_section_selection_falls_back_to_forward_sentences():
    text = ("The quarter closed well. For fiscal year 2027, we expect revenue of "
            "$90 billion. Shares outstanding were 139,933.")
    sections = select_sections(text)
    assert sections
    assert "expect revenue" in sections[0].text
    assert "Shares outstanding" not in sections[0].text


def test_section_selection_is_bounded():
    text = "Financial Outlook\n\n" + ("x" * 50_000)
    sections = select_sections(text, max_sections=2, max_chars=1000)
    assert len(sections) <= 2
    assert all(len(s.text) <= 1000 for s in sections)


def test_no_section_means_no_model_call():
    """Cost bounding: a document with nothing forward-looking is not sent."""
    assert select_sections("Revenue was $67 billion for the year.") == []


# ---------------------------------------------------------------------------
# GENERALIZED READER-RECALL phase. A live release stated next-quarter
# guidance as "Outlook for Q3 2026 ... For Q3 2026, we anticipate: <revenue-
# like KPI>, <growth rate>, <adjusted EPS>, <adjusted EBITDA>" and the report
# came back "management guidance: none extracted". Two independent defects
# compounded: section selection dropped bulleted rows with no forward verb
# of their own (covered in test_finance_extraction_section_selection.py),
# and the ONE candidate that WAS proposed was refused as AMBIGUOUS_TARGET_
# PERIOD because `parse_target_period` kept a narrower, duplicate regex that
# required the literal token "FY" and rejected the plain calendar phrasing
# ("Q3 2026") the release actually used -- while `finance.guidance`'s own
# canonical resolver, which V1 already relies on, has always accepted it.
# ---------------------------------------------------------------------------

# Fixture B: next-quarter adjusted EPS in plain calendar-quarter phrasing.
def test_calendar_quarter_phrasing_resolves_to_the_correct_fiscal_period():
    metric = _accept(_candidate(
        metric_id=N.ADJUSTED_EPS, value_type=ValueType.RANGE, low=0.84, high=0.88,
        unit="PER_SHARE", basis="NON_GAAP",
        target_period="Q3 2026", target_period_type="QUARTER",
        source_sentence="Non-GAAP EPS of $0.84 to $0.88."),
        document="For Q3 2026, we anticipate: Non-GAAP EPS of $0.84 to $0.88.")
    assert metric.name == N.ADJUSTED_EPS
    assert metric.fiscal_period == "Q3 FY2026"
    assert metric.target_period_type == "NEXT_QUARTER"
    assert metric.basis == "adjusted"


@pytest.mark.parametrize("label, expected_label", [
    ("Q3 2026", "Q3 FY2026"),
    ("Q3 FY2026", "Q3 FY2026"),
    ("third quarter 2026", "Q3 FY2026"),
    ("FY2027", "FY2027"),
    ("fiscal 2027", "FY2027"),
])
def test_the_validator_and_v1_agree_on_every_period_spelling(label, expected_label):
    """One canonical period resolver, not two disagreeing on the same label."""
    from finance.extraction.validator import parse_target_period
    period = parse_target_period(label)
    assert period is not None, f"{label!r} was rejected as ambiguous"
    assert period.label == expected_label


def test_a_genuinely_ambiguous_period_is_still_refused():
    """The fix must not turn AMBIGUOUS_TARGET_PERIOD into an always-accept.

    "next year" names no year at all and stays refused -- the canonical
    resolver is more capable, not less strict.
    """
    code, _reason = _reject(_candidate(
        metric_id=N.CONSOLIDATED_REVENUE, value_type=ValueType.POINT, value=90.0,
        unit="USD_BILLION", target_period="next year",
        source_sentence="Next year we expect revenue guidance of $90 billion."))
    assert code == RejectionCode.AMBIGUOUS_TARGET_PERIOD


# Fixture C: an operating KPI unsupported by the revenue taxonomy must not
# become REVENUE_GROWTH -- neither under its own unsupported name nor
# re-labelled as revenue_growth by a misreading.
_BOOKINGS_DOC = ("For Q3 2026, we anticipate: Gross Bookings of $58.25 billion to "
                 "$60.25 billion, representing growth of 18% to 22% YoY on a "
                 "constant-currency basis.")
_BOOKINGS_SENTENCE = ("Gross Bookings of $58.25 billion to $60.25 billion, "
                      "representing growth of 18% to 22% YoY on a "
                      "constant-currency basis.")


def test_an_unsupported_operating_kpi_is_refused_under_its_own_name():
    code, reason = _reject(_candidate(
        metric_id="gross_bookings", value_type=ValueType.RANGE, low=58.25, high=60.25,
        unit="USD_BILLION", target_period="Q3 2026", target_period_type="QUARTER",
        basis="NON_GAAP", source_sentence=_BOOKINGS_SENTENCE),
        document=_BOOKINGS_DOC)
    assert code == RejectionCode.UNKNOWN_METRIC


def test_an_unsupported_operating_kpi_cannot_be_relabelled_as_revenue_growth():
    """The dangerous direction: the same sentence, proposed AS consolidated
    revenue growth. Grounding must catch that the sentence names a component
    KPI, not consolidated revenue, regardless of how the model labels it."""
    code, reason = _reject(_candidate(
        metric_id=N.CONSOLIDATED_REVENUE_GROWTH, value_type=ValueType.RANGE,
        low=18.0, high=22.0, unit="PERCENT",
        target_period="Q3 2026", target_period_type="QUARTER", basis="NON_GAAP",
        source_sentence=_BOOKINGS_SENTENCE),
        document=_BOOKINGS_DOC)
    assert code == RejectionCode.METRIC_NOT_GROUNDED


# Fixture D: a SUPPORTED operating KPI (a segment/component identity the
# taxonomy does carry) retains its own identity rather than being refused or
# folded into the consolidated metric.
def test_a_supported_operating_kpi_keeps_its_own_identity():
    doc = "For Q3 2026, we anticipate: Cloud segment revenue growth of 20% to 24% YoY."
    metric = _accept(_candidate(
        metric_id=N.SEGMENT_REVENUE_GROWTH, value_type=ValueType.RANGE,
        low=20.0, high=24.0, unit="PERCENT",
        target_period="Q3 2026", target_period_type="QUARTER", basis="GAAP",
        source_sentence="Cloud segment revenue growth of 20% to 24% YoY."),
        document=doc)
    assert metric.name == N.SEGMENT_REVENUE_GROWTH
    assert metric.name != N.CONSOLIDATED_REVENUE_GROWTH


# Fixture F: GAAP EPS and adjusted (non-GAAP) EPS guided in the same release
# must both survive, with distinct basis -- neither overwrites the other
# merely because the metric family is EPS.
def test_gaap_and_adjusted_eps_both_survive_with_distinct_basis():
    doc = ("For Q3 2026, we anticipate: GAAP diluted EPS of $0.55 to $0.60. "
          "Non-GAAP EPS of $0.84 to $0.88.")
    validator = GuidanceCandidateValidator(document_text=doc, issued_at="2026-08-05")
    gaap = _candidate(metric_id=N.EPS, value_type=ValueType.RANGE, low=0.55, high=0.60,
                      unit="PER_SHARE", basis="GAAP",
                      target_period="Q3 2026", target_period_type="QUARTER",
                      source_sentence="GAAP diluted EPS of $0.55 to $0.60.")
    adjusted = _candidate(metric_id=N.ADJUSTED_EPS, value_type=ValueType.RANGE,
                          low=0.84, high=0.88, unit="PER_SHARE", basis="NON_GAAP",
                          target_period="Q3 2026", target_period_type="QUARTER",
                          source_sentence="Non-GAAP EPS of $0.84 to $0.88.")
    accepted, rejected = validator.validate_all([gaap, adjusted])
    assert len(accepted) == 2, [(c, code, r) for c, code, r in rejected]
    names = {m.name for m in accepted}
    assert names == {N.EPS, N.ADJUSTED_EPS}
    bases = {m.name: m.basis for m in accepted}
    assert bases[N.EPS] == "GAAP"
    assert bases[N.ADJUSTED_EPS] == "adjusted"


# Fixture A (reader layer): an outlook with three distinct guided metrics in
# ONE section must all be proposed and all survive the boundary -- no
# accidental cross-metric collapsing when several statements share a period.
def test_three_distinct_guided_metrics_in_one_section_all_survive():
    section_text = ("For Q3 2026, we anticipate: Segment revenue growth of "
                    "10% to 14% YoY. Non-GAAP EPS of $0.84 to $0.88. "
                    "Adjusted EBITDA of $2.86 billion to $2.96 billion.")
    payload = {"statements": [
        {"metric_id": "segment_revenue_growth", "value_type": "range",
         "low": 10.0, "high": 14.0, "unit": "PERCENT", "target_period": "Q3 2026",
         "target_period_type": "QUARTER", "basis": "GAAP", "action": "NEW",
         "prospective": True,
         "source_sentence": "Segment revenue growth of 10% to 14% YoY.",
         "confidence": 1.0},
        {"metric_id": "adjusted_earnings_per_share", "value_type": "range",
         "low": 0.84, "high": 0.88, "unit": "PER_SHARE", "target_period": "Q3 2026",
         "target_period_type": "QUARTER", "basis": "NON_GAAP", "action": "NEW",
         "prospective": True, "source_sentence": "Non-GAAP EPS of $0.84 to $0.88.",
         "confidence": 1.0},
        {"metric_id": "adjusted_ebitda", "value_type": "range",
         "low": 2.86, "high": 2.96, "unit": "USD_BILLION", "target_period": "Q3 2026",
         "target_period_type": "QUARTER", "basis": "NON_GAAP", "action": "NEW",
         "prospective": True,
         "source_sentence": "Adjusted EBITDA of $2.86 billion to $2.96 billion.",
         "confidence": 1.0},
    ]}
    extractor = _scripted([payload])
    sections = [DocumentSection(label="Forward-looking statements", text=section_text,
                                start=0, end=len(section_text))]
    candidates = extractor.extract(sections, issued_at="2026-08-05")
    assert len(candidates) == 3
    validator = GuidanceCandidateValidator(document_text=section_text, issued_at="2026-08-05")
    accepted, rejected = validator.validate_all(candidates)
    assert len(accepted) == 3, [(code, r) for _c, code, r in rejected]
    names = {m.name for m in accepted}
    assert names == {N.SEGMENT_REVENUE_GROWTH, N.ADJUSTED_EPS, N.ADJUSTED_EBITDA}
