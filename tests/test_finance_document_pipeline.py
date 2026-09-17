"""Finance Document Package pipeline (Phase H.16): unit + hard-safety tests.

Covers the modules under `finance/documents/`: package resolution
(`package.py`), the actual-fact fallback boundary (`actuals_validator.py`),
the financing-event boundary (`event_validator.py`), and the `v1|compare|v2`
mode seam (`runtime.py`).

Hard-safety positive controls (spec section 27) each get their own test:
a fixture engineered to trip the failure and an assertion that it is
refused. `UNDRAWN_FACILITY_COUNTED_AS_FUNDED_DEBT` and `GUIDANCE_AS_ACTUAL`
are this phase's two new invariants; the rest of section 27's list is
already covered by `finance/extraction` and `finance/reported_actuals`'s own
suites and is untouched here.
"""

import pytest

from finance import semantics as sem
from finance.documents.actuals_schema import (
    ActualBasis,
    ActualFactCandidate,
    ActualPeriodType,
    ActualRejectionCode,
)
from finance.documents.actuals_validator import ActualFactCandidateValidator
from finance.documents.event_schema import EventCandidate, EventRejectionCode
from finance.documents.event_validator import EventCandidateValidator
from finance.documents.package import DocumentClass, resolve_document_package
from finance.documents.spans import build_source_spans
from finance.documents.runtime import (
    DocumentFetcher,
    DocumentPipelineMode,
    build_document_pipeline,
    register_actuals_model_client,
    register_event_model_client,
)
from finance.reported_actuals.tables import StatementKind
from finance.structural_breaks import PostBalanceSheetEventType


# ---------------------------------------------------------------------------
# package.py -- document classification
# ---------------------------------------------------------------------------

def _submissions(rows):
    recent = {"accessionNumber": [], "filingDate": [], "reportDate": [],
             "form": [], "primaryDocDescription": [], "primaryDocument": [],
             "items": []}
    for row in rows:
        recent["accessionNumber"].append(row["accession"])
        recent["filingDate"].append(row["filed"])
        recent["reportDate"].append(row.get("report_date", ""))
        recent["form"].append(row["form"])
        recent["primaryDocDescription"].append(row.get("description", ""))
        recent["primaryDocument"].append(row.get("document", ""))
        recent["items"].append(row.get("items", ""))
    return {"cik": "1234", "filings": {"recent": recent}}


def _domestic_submissions():
    return _submissions([
        {"accession": "A-10K", "form": "10-K", "filed": "2026-08-15",
         "report_date": "2026-06-30"},
        {"accession": "A-10Q", "form": "10-Q", "filed": "2026-05-01",
         "report_date": "2026-03-31"},
        {"accession": "A-8K-ER", "form": "8-K", "filed": "2026-08-01",
         "items": "2.02,9.01", "description": "Q2 2026 Earnings Release",
         "document": "ex991.htm"},
        {"accession": "A-8K-DEBT", "form": "8-K", "filed": "2026-07-01",
         "items": "2.03", "description": "Entry into Credit Agreement",
         "document": "debt8k.htm"},
        {"accession": "A-3", "form": "3", "filed": "2026-06-01"},
        {"accession": "A-DEF14A", "form": "DEF 14A", "filed": "2026-04-01"},
    ])


def test_package_classifies_periodic_earnings_financing_and_corporate():
    package = resolve_document_package("TICK", _domestic_submissions())
    assert [d.accession for d in package.periodic_annual] == ["A-10K"]
    assert [d.accession for d in package.periodic_quarterly] == ["A-10Q"]
    assert [d.accession for d in package.earnings_releases] == ["A-8K-ER"]
    assert DocumentClass.GUIDANCE_UPDATE in package.earnings_releases[0].classifications
    assert DocumentClass.ACTUAL_EARNINGS_RELEASE in package.earnings_releases[0].classifications
    assert [d.accession for d in package.financing_events] == ["A-8K-DEBT"]
    corporate_accessions = {d.accession for d in package.corporate_events}
    assert {"A-3", "A-DEF14A"} <= corporate_accessions
    assert package.reporting_status == "domestic_registrant"


def test_package_never_classifies_a_6k_as_periodic_on_form_alone():
    """The exact fix `reported_actuals/discovery.py` made, preserved here:
    a 6-K is a container, and membership in the interim-forms taxonomy list
    must not be read as proof of periodic financial statements."""
    submissions = _submissions([
        {"accession": "F-20F", "form": "20-F", "filed": "2026-06-01",
         "report_date": "2026-03-31"},
        {"accession": "F-6K", "form": "6-K", "filed": "2026-07-01",
         "description": "Notice of Annual General Meeting"},
    ])
    package = resolve_document_package("FPI", submissions)
    assert package.reporting_status == "foreign_private_issuer"
    assert [d.accession for d in package.periodic_annual] == ["F-20F"]
    assert package.periodic_quarterly == ()
    accessions = {d.accession for d in package.periodic_annual + package.periodic_quarterly}
    assert "F-6K" not in accessions


def test_package_v1_mode_never_fetches_anything():
    class ExplodingFetcher:
        def filing_index(self, accession):
            raise AssertionError("v1 must never fetch a filing index")

        def exhibit(self, accession, document):
            raise AssertionError("v1 must never fetch an exhibit")

    result = build_document_pipeline(
        "TICK", _domestic_submissions(), {}, mode=DocumentPipelineMode.V1,
        fetcher=ExplodingFetcher())
    assert result.package is None
    assert result.candidates == ()
    assert result.facts_overlay == {}
    assert result.events == ()


# ---------------------------------------------------------------------------
# actuals_validator.py -- the acceptance boundary for LLM-read actual facts
# ---------------------------------------------------------------------------

_GRID = (
    "Table caption: Condensed Consolidated Statements of Operations\n"
    "Currency: USD, Scale: millions\n"
    "Columns: Three Months Ended June 30, 2026 | Three Months Ended June 30, 2025\n"
    "Row: Total revenues: 29600.0 | 26044.0\n"
    "Row: Operating income: 8700.0 | 7100.0\n"
    "Row: Net income attributable to segment operations: 500.0 | 400.0\n"
    "Row: Total revenues: 29600.0 | 26044.0 (the company expects continued growth)\n"
)

_PERIOD_ENDS = {"Three Months Ended June 30, 2026": "2026-06-30",
               "Three Months Ended June 30, 2025": "2025-06-30"}


def _actual_candidate(**overrides):
    base = dict(
        metric_id="revenue", value=29600.0, unit="USD_MILLION", currency="USD",
        period_label="Three Months Ended June 30, 2026",
        period_type=ActualPeriodType.QUARTER, basis=ActualBasis.GAAP,
        scope=sem.ConsolidationScope.CONSOLIDATED, prospective=False,
        source_evidence="Row: Total revenues: 29600.0 | 26044.0",
        confidence=0.9, statement_kind=StatementKind.INCOME_STATEMENT)
    base.update(overrides)
    return ActualFactCandidate(**base)


def _actual_validator(**overrides):
    kwargs = dict(grid_text=_GRID, period_ends_by_label=_PERIOD_ENDS,
                 as_of="2026-08-15")
    kwargs.update(overrides)
    return ActualFactCandidateValidator(**kwargs)


def test_actual_candidate_accepted_when_fully_grounded():
    fact, code, reason = _actual_validator().validate(_actual_candidate())
    assert code is None, reason
    assert fact.field == "revenue"
    assert fact.value == 29_600_000_000.0          # millions, scaled
    assert fact.period_end == "2026-06-30"
    assert fact.is_canonical


def test_actual_candidate_refused_when_evidence_not_in_document():
    """UNSUPPORTED_ACCEPTED_FACT positive control."""
    candidate = _actual_candidate(source_evidence="Row: Total revenues: 999.0 | 1.0")
    fact, code, _reason = _actual_validator().validate(candidate)
    assert fact is None
    assert code == ActualRejectionCode.EVIDENCE_NOT_IN_SOURCE


def test_actual_candidate_refused_for_unknown_metric():
    """WRONG_METRIC_ACCEPTED positive control."""
    candidate = _actual_candidate(metric_id="same_store_sales_growth")
    fact, code, _reason = _actual_validator().validate(candidate)
    assert fact is None
    assert code == ActualRejectionCode.UNKNOWN_METRIC


def test_actual_candidate_refused_for_segment_scope():
    """WRONG_SCOPE_ACCEPTED positive control."""
    candidate = _actual_candidate(
        metric_id="net_income", value=500.0,
        source_evidence="Row: Net income attributable to segment operations: 500.0 | 400.0",
        scope=sem.ConsolidationScope.SEGMENT)
    fact, code, _reason = _actual_validator().validate(candidate)
    assert fact is None
    assert code == ActualRejectionCode.SCOPE_NOT_CONSOLIDATED


def test_actual_candidate_refused_when_adjusted():
    candidate = _actual_candidate(basis=ActualBasis.ADJUSTED)
    fact, code, _reason = _actual_validator().validate(candidate)
    assert fact is None
    assert code == ActualRejectionCode.ADJUSTED_NOT_CANONICAL


def test_actual_candidate_refused_when_forward_looking():
    """GUIDANCE_AS_ACTUAL positive control -- a hard-safety invariant this
    phase adds: an actual-fact candidate whose own cited text reads as an
    outlook must never become a reported fact."""
    candidate = _actual_candidate(
        source_evidence="Row: Total revenues: 29600.0 | 26044.0 (the company "
                        "expects continued growth)")
    fact, code, _reason = _actual_validator().validate(candidate)
    assert fact is None
    assert code == ActualRejectionCode.PROSPECTIVE_NOT_ACTUAL


def test_actual_candidate_refused_for_unresolved_period():
    candidate = _actual_candidate(period_label="Some Unlisted Column")
    fact, code, _reason = _actual_validator().validate(candidate)
    assert fact is None
    assert code == ActualRejectionCode.PERIOD_UNRESOLVED


def test_actual_candidate_refused_for_a_period_that_has_not_ended():
    """WRONG_PERIOD_ACCEPTED positive control: a period after `as_of` is not
    a reported actual, whatever the table appears to say."""
    validator = _actual_validator(as_of="2026-01-01")
    fact, code, _reason = validator.validate(_actual_candidate())
    assert fact is None
    assert code == ActualRejectionCode.PERIOD_NOT_YET_ENDED


def test_actual_candidate_refused_below_confidence_floor():
    candidate = _actual_candidate(confidence=0.1)
    fact, code, _reason = _actual_validator().validate(candidate)
    assert fact is None
    assert code == ActualRejectionCode.LOW_CONFIDENCE


def test_actual_candidate_refused_for_unknown_currency():
    candidate = _actual_candidate(currency="dollars")
    fact, code, _reason = _actual_validator().validate(candidate)
    assert fact is None
    assert code == ActualRejectionCode.UNKNOWN_CURRENCY


# ---------------------------------------------------------------------------
# event_validator.py -- funded vs. committed is the hard-safety invariant
# ---------------------------------------------------------------------------

_CREDIT_AGREEMENT_TEXT = (
    "On July 1, 2026, the Company entered into a $2.0 billion revolving "
    "credit facility with a syndicate of lenders led by Example Bank. "
    "The facility is available to the Company for general corporate "
    "purposes. No amounts have been drawn under the facility as of the "
    "date of this filing."
)

_DRAWN_TERM_LOAN_TEXT = (
    "On July 1, 2026, the Company entered into a $2.0 billion term loan "
    "agreement and drew the full $2.0 billion, receiving net proceeds of "
    "approximately $1.98 billion."
)

_CREDIT_AGREEMENT_SPANS = build_source_spans(_CREDIT_AGREEMENT_TEXT)
_DRAWN_TERM_LOAN_SPANS = build_source_spans(_DRAWN_TERM_LOAN_TEXT)

# A compound sentence, semicolon-joined: the amount is established in the
# first clause, the drawdown witness only in the second. Deliberately
# different fixture text from `_DRAWN_TERM_LOAN_TEXT` above -- that one is a
# single sentence with nothing to split, which is the wrong shape for
# proving multi-span citation actually works.
_MULTI_SPAN_TERM_LOAN_TEXT = (
    "On July 1, 2026, the Company entered into a $2.0 billion term loan "
    "agreement with a syndicate of lenders; the Company drew the full "
    "$2.0 billion under the agreement and received net proceeds of "
    "approximately $1.98 billion."
)
_MULTI_SPAN_TERM_LOAN_SPANS = build_source_spans(_MULTI_SPAN_TERM_LOAN_TEXT)


def _span_ids_containing(spans, *substrings) -> tuple:
    """The span id(s) whose text contains each given substring -- lets a
    test name evidence by MEANING ("the sentence naming the amount") rather
    than by a span number that would silently drift if the splitter's
    boundaries ever change."""
    return tuple(
        next(span.span_id for span in spans if substring in span.text)
        for substring in substrings)


def _event_candidate(**overrides):
    base = dict(
        event_type=PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE,
        amount=2.0, unit="USD_BILLION", currency="USD", funded=False,
        committed=True,
        evidence_span_ids=_span_ids_containing(
            _CREDIT_AGREEMENT_SPANS, "entered into a $2.0 billion revolving"),
        confidence=0.9, form="8-K", items="2.03")
    base.update(overrides)
    return EventCandidate(**base)


def test_event_candidate_accepted_as_committed_not_funded():
    validator = EventCandidateValidator(
        spans=_CREDIT_AGREEMENT_SPANS, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(_event_candidate())
    assert code is None, reason
    assert resolved.funded is False
    assert resolved.committed is True
    assert resolved.amount == 2_000_000_000.0


def test_event_candidate_evidence_is_an_exact_span_slice_not_a_transcription():
    """Phase H.18's core invariant: the resolved evidence is the document's
    OWN text, retrieved by id, never anything the model wrote."""
    validator = EventCandidateValidator(
        spans=_CREDIT_AGREEMENT_SPANS, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(_event_candidate())
    assert code is None, reason
    assert resolved.source_evidence in _CREDIT_AGREEMENT_TEXT


def test_event_candidate_refused_when_funded_claim_is_ungrounded():
    """UNDRAWN_FACILITY_COUNTED_AS_FUNDED_DEBT positive control: the model
    asserts funded=True but the cited SPAN only describes a facility being
    established, never a drawdown."""
    candidate = _event_candidate(funded=True)
    validator = EventCandidateValidator(
        spans=_CREDIT_AGREEMENT_SPANS, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.FUNDED_STATUS_NOT_GROUNDED


def test_event_candidate_accepted_as_funded_when_drawdown_is_grounded():
    candidate = _event_candidate(
        funded=True, committed=False,
        evidence_span_ids=_span_ids_containing(
            _DRAWN_TERM_LOAN_SPANS, "drew the full $2.0 billion"))
    validator = EventCandidateValidator(
        spans=_DRAWN_TERM_LOAN_SPANS, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.funded is True


def test_event_candidate_refused_when_span_id_is_invented():
    """Replaces the old substring-search 'evidence not in document' test:
    there is no longer a text search to fool -- an unknown span id is
    refused outright, whether invented, mistyped, or from another run."""
    candidate = _event_candidate(evidence_span_ids=("s999",))
    validator = EventCandidateValidator(
        spans=_CREDIT_AGREEMENT_SPANS, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.EVIDENCE_NOT_IN_SOURCE


def test_event_candidate_refused_when_span_id_is_from_another_document():
    """A span id from a DIFFERENT document's span set must be refused
    against this one. `build_source_spans` hashes the source text into
    every id precisely so this is not a coincidence to rely on: the two
    documents' ids are drawn from disjoint id spaces by construction."""
    other_spans = build_source_spans(_DRAWN_TERM_LOAN_TEXT)
    assert not ({s.span_id for s in other_spans}
               & {s.span_id for s in _CREDIT_AGREEMENT_SPANS}), (
        "the two fixtures' span ids must be disjoint for this test to mean anything")
    candidate = _event_candidate(evidence_span_ids=(other_spans[0].span_id,))
    # Deliberately validated against the CREDIT AGREEMENT's spans, not the
    # term loan's -- a real id, just the wrong document.
    validator = EventCandidateValidator(
        spans=_CREDIT_AGREEMENT_SPANS, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.EVIDENCE_NOT_IN_SOURCE


def test_event_candidate_refused_when_no_span_cited():
    candidate = _event_candidate(evidence_span_ids=())
    validator = EventCandidateValidator(
        spans=_CREDIT_AGREEMENT_SPANS, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.NO_EVIDENCE


def test_event_candidate_refused_when_amount_not_in_evidence():
    candidate = _event_candidate(amount=99.0)
    validator = EventCandidateValidator(
        spans=_CREDIT_AGREEMENT_SPANS, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.AMOUNT_NOT_GROUNDED


def test_event_candidate_amount_and_funding_may_come_from_different_spans():
    """Section 7's multi-span support: the amount is in one span, the
    funding witness in another, and BOTH must be cited for the candidate to
    be accepted as funded."""
    amount_span, funding_span = _span_ids_containing(
        _MULTI_SPAN_TERM_LOAN_SPANS, "entered into a $2.0 billion term loan",
        "drew the full $2.0 billion")
    assert amount_span != funding_span, "fixture must exercise genuinely distinct spans"
    candidate = _event_candidate(
        funded=True, committed=False,
        evidence_span_ids=(amount_span, funding_span))
    validator = EventCandidateValidator(
        spans=_MULTI_SPAN_TERM_LOAN_SPANS, items="2.03", form="8-K")
    resolved, code, reason = validator.validate(candidate)
    assert code is None, reason
    assert resolved.funded is True
    assert resolved.amount == 2_000_000_000.0


def test_event_candidate_refused_when_only_amount_span_cited_no_funding_witness():
    """The mirror of the above: citing ONLY the amount span must still be
    refused for funded=true -- multi-span citation is additive, not a way
    to smuggle a funding claim past a span that never mentions funding."""
    amount_span, = _span_ids_containing(
        _MULTI_SPAN_TERM_LOAN_SPANS, "entered into a $2.0 billion term loan")
    candidate = _event_candidate(
        funded=True, committed=False, evidence_span_ids=(amount_span,))
    validator = EventCandidateValidator(
        spans=_MULTI_SPAN_TERM_LOAN_SPANS, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.FUNDED_STATUS_NOT_GROUNDED


def test_event_candidate_refused_when_item_code_does_not_admit_the_type():
    """A filing whose OWN item code identifies it as equity (3.02) must
    refuse a model claiming it is an acquisition."""
    candidate = _event_candidate(
        event_type=PostBalanceSheetEventType.ACQUISITION, items="3.02")
    validator = EventCandidateValidator(
        spans=_CREDIT_AGREEMENT_SPANS, items="3.02", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.ITEM_CODE_MISMATCH


def test_event_candidate_refused_below_confidence_floor():
    candidate = _event_candidate(confidence=0.1)
    validator = EventCandidateValidator(
        spans=_CREDIT_AGREEMENT_SPANS, items="2.03", form="8-K")
    resolved, code, _reason = validator.validate(candidate)
    assert resolved is None
    assert code == EventRejectionCode.LOW_CONFIDENCE


# ---------------------------------------------------------------------------
# runtime.py -- mode gating and registration
# ---------------------------------------------------------------------------

def test_no_extractor_registered_is_diagnosed_not_silently_v1():
    register_actuals_model_client(None)
    register_event_model_client(None)
    from finance.documents import runtime as document_pipeline_runtime

    assert document_pipeline_runtime.actuals_extractor_registered() is False
    assert document_pipeline_runtime.event_extractor_registered() is False


def test_compare_mode_withholds_candidates_even_when_registered():
    """compare must never let a new candidate change the resolver's input."""
    calls = []

    class StubFetcher:
        def filing_index(self, accession):
            calls.append(("index", accession))
            return "<html></html>"

        def exhibit(self, accession, document):
            calls.append(("exhibit", accession, document))
            return ""

    register_actuals_model_client(None)
    register_event_model_client(None)
    result = build_document_pipeline(
        "TICK", _domestic_submissions(), {}, mode=DocumentPipelineMode.COMPARE,
        fetcher=StubFetcher(), as_of="2026-08-15",
        target_period_end="2026-06-30")
    assert result.candidates == ()
    assert result.facts_overlay == {}
    assert result.diagnostics.mode == DocumentPipelineMode.COMPARE
