"""Golden Document Benchmark — Phase H.16/H.17, spec sections 24-27.

Phase H.16 built twelve cases covering document classification and the
EVENT validator boundary (committed-only, funded, convertible refinement,
equity issuance, item-code mismatch, undrawn-claimed-funded). Phase H.17
adds the class Phase H.16 explicitly named as its own gap -- these twelve
carried no earnings-release TABLE text, so the LLM actual-fact fallback was
never exercised, live or otherwise -- plus two more financing-event shapes
(bridge facility, debt refinancing) and a guidance-adjacent-to-actuals
contamination case. This is still short of spec section 24's 25-30 case
target; the completion summary says so honestly.

Ground truth is hand-authored from the fixture's OWN construction, never
derived from running the pipeline and reading back its answer -- the same
discipline `tests/fixtures/reported_actuals_benchmark.py` documents. The new
actual-fact cases reuse `tests/fixtures/reported_actuals_releases.py`'s
`statement_table`/`income_statement`/`balance_sheet`/`cash_flow` helpers
directly (the SAME fixture machinery `reported_actuals`'s own benchmark
uses) rather than writing a second HTML-table builder, and their expected
values are that module's own `Q4`/`FY` dicts -- the numbers a reader would
get by reading the fixture's source, not by running any extractor.

Ticker-shaped names below are fixture data only and never appear in any
`finance/` or `tools/` production branch (see
`test_no_benchmark_ticker_appears_in_production_code` in
`tests/test_finance_extraction_benchmark.py` for the same discipline applied
to the extraction benchmark; this suite is small enough to check by
inspection and is covered by the parallel test in
`tests/test_finance_document_package_benchmark.py`).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tests.fixtures.reported_actuals_releases import (
    Q4,
    Q4_END_ISO,
    Q4_PRIOR_END_ISO,
    balance_sheet,
    cash_flow,
    income_statement,
    outlook_table,
    release,
)


@dataclass(frozen=True)
class ExpectedClassification:
    accession: str
    expected_bucket: str        # "periodic_annual" | "periodic_quarterly" |
                                 # "earnings_releases" | "financing_events" |
                                 # "corporate_events" | "unknown"


@dataclass(frozen=True)
class ExpectedEvent:
    accession: str
    event_type: str
    funded: Optional[bool]
    committed: Optional[bool]
    amount: Optional[float]
    should_be_accepted: bool
    note: str = ""


@dataclass(frozen=True)
class ExpectedActualFact:
    """One statement-line ground truth for the LLM actual-table fallback.

    `value` is the FULL magnitude (spec section 9's `resolved_value`, after
    the model's stated unit/scale is applied) -- what
    `ActualFactCandidateValidator.validate` would put on the resulting
    `ActualFinancialFact.value`, not the as-printed table figure.
    """

    metric_id: str
    value: float
    period_end: str
    should_be_accepted: bool = True
    note: str = ""


@dataclass(frozen=True)
class DocumentPackageCase:
    case_id: str
    classes: Tuple[str, ...]
    submissions: dict
    expected: Tuple[ExpectedClassification, ...]
    expected_reporting_status: str = "domestic_registrant"
    event_documents: Dict[str, str] = field(default_factory=dict)   # accession -> body text
    expected_events: Tuple[ExpectedEvent, ...] = ()
    # -- actual-table fallback (Phase H.17) --
    # The accession whose exhibit carries `actual_document_text`, and whether
    # the structured/deterministic paths already resolve the target period --
    # the exact two inputs `actuals_bridge.should_attempt_fallback` takes.
    actual_document_accession: Optional[str] = None
    actual_document_text: str = ""
    target_period_end: Optional[str] = None
    structured_completeness: Optional[str] = None
    expected_fallback_triggers: Optional[bool] = None
    expected_actual_facts: Tuple[ExpectedActualFact, ...] = ()
    # False for a case whose `expected_events`/`expected_actual_facts` are
    # DELIBERATELY WRONG relative to the document's true content -- a
    # validator-boundary negative control (see cases 10-11 below). Running
    # the live reader against one of these would score the model's CORRECT
    # reading against an intentionally-wrong ground truth and manufacture a
    # false defect; `scripts/run_live_document_pipeline_benchmark.py`
    # excludes any case with this set to False.
    live_reader_case: bool = True
    notes: str = ""


def _recent(rows: List[dict]) -> dict:
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
    return recent


def _submissions(rows: List[dict]) -> dict:
    return {"cik": "9999999999", "filings": {"recent": _recent(rows)}}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

_CASES: List[DocumentPackageCase] = []


def _add(case: DocumentPackageCase) -> None:
    _CASES.append(case)


# 1. Mature profitable domestic issuer, one of everything, nothing surprising.
_add(DocumentPackageCase(
    case_id="mature-profitable-baseline",
    classes=("mature_profitable",),
    submissions=_submissions([
        {"accession": "MAT-10K", "form": "10-K", "filed": "2026-08-15",
         "report_date": "2026-06-30"},
        {"accession": "MAT-10Q", "form": "10-Q", "filed": "2026-05-01",
         "report_date": "2026-03-31"},
        {"accession": "MAT-8K-ER", "form": "8-K", "filed": "2026-08-01",
         "items": "2.02,9.01", "description": "Q2 2026 Earnings Release",
         "document": "ex991.htm"},
    ]),
    expected=(
        ExpectedClassification("MAT-10K", "periodic_annual"),
        ExpectedClassification("MAT-10Q", "periodic_quarterly"),
        ExpectedClassification("MAT-8K-ER", "earnings_releases"),
    ),
))

# 2. Loss-making growth issuer -- no distinguishing document-selection
# behaviour, included as its own class per spec section 24.
_add(DocumentPackageCase(
    case_id="loss-making-growth-baseline",
    classes=("loss_making_growth",),
    submissions=_submissions([
        {"accession": "LOSS-10Q", "form": "10-Q", "filed": "2026-05-10",
         "report_date": "2026-03-31"},
        {"accession": "LOSS-8K-ER", "form": "8-K", "filed": "2026-08-05",
         "items": "2.02", "description": "Earnings Press Release",
         "document": "ex991.htm"},
    ]),
    expected=(
        ExpectedClassification("LOSS-10Q", "periodic_quarterly"),
        ExpectedClassification("LOSS-8K-ER", "earnings_releases"),
    ),
))

# 3. Foreign private issuer: 20-F is periodic annual, 6-K is NOT periodic on
# form alone (must carry earnings evidence to become an earnings release; a
# meeting-notice 6-K is neither).
_add(DocumentPackageCase(
    case_id="foreign-private-issuer-6k-not-periodic",
    classes=("foreign_private_issuer",),
    submissions=_submissions([
        {"accession": "FPI-20F", "form": "20-F", "filed": "2026-04-01",
         "report_date": "2026-01-31"},
        {"accession": "FPI-6K-RESULTS", "form": "6-K", "filed": "2026-08-01",
         "description": "Interim Results Press Release", "document": "results.htm"},
        {"accession": "FPI-6K-MEETING", "form": "6-K", "filed": "2026-06-01",
         "description": "Notice of Annual General Meeting", "document": "agm.htm"},
    ]),
    expected_reporting_status="foreign_private_issuer",
    expected=(
        ExpectedClassification("FPI-20F", "periodic_annual"),
        ExpectedClassification("FPI-6K-RESULTS", "earnings_releases"),
        ExpectedClassification("FPI-6K-MEETING", "unknown"),
    ),
))

# 4. Guidance-heavy issuer: the earnings release must carry BOTH
# classifications on one document (spec section 3).
_add(DocumentPackageCase(
    case_id="guidance-heavy-single-document",
    classes=("guidance_heavy",),
    submissions=_submissions([
        {"accession": "GUID-8K-ER", "form": "8-K", "filed": "2026-08-01",
         "items": "2.02,9.01", "description": "Q2 2026 Results and Business Outlook",
         "document": "ex991.htm"},
    ]),
    expected=(ExpectedClassification("GUID-8K-ER", "earnings_releases"),),
))

# 5. Routine corporate filings must never enter the financial-source buckets.
_add(DocumentPackageCase(
    case_id="routine-corporate-filings-excluded",
    classes=("routine_corporate",),
    submissions=_submissions([
        {"accession": "ROUT-3", "form": "3", "filed": "2026-06-01"},
        {"accession": "ROUT-4", "form": "4", "filed": "2026-06-15"},
        {"accession": "ROUT-DEF14A", "form": "DEF 14A", "filed": "2026-04-01"},
        {"accession": "ROUT-13G", "form": "SC 13G", "filed": "2026-03-01"},
    ]),
    expected=(
        ExpectedClassification("ROUT-3", "corporate_events"),
        ExpectedClassification("ROUT-4", "corporate_events"),
        ExpectedClassification("ROUT-DEF14A", "corporate_events"),
        ExpectedClassification("ROUT-13G", "corporate_events"),
    ),
))

# 6. Financing event: committed facility, no drawdown language anywhere.
_CREDIT_FACILITY_TEXT = (
    "On July 1, 2026, the Company entered into a $2.0 billion five-year "
    "revolving credit facility with a syndicate of lenders led by Example "
    "Bank, N.A. The facility is available to the Company for working "
    "capital and general corporate purposes. No amounts have been drawn "
    "under the facility as of the date of this filing."
)
_add(DocumentPackageCase(
    case_id="financing-committed-only",
    classes=("financing_event", "capital_structure"),
    submissions=_submissions([
        {"accession": "FIN-CREDIT", "form": "8-K", "filed": "2026-07-01",
         "items": "2.03", "description": "Entry into Credit Agreement",
         "document": "credit8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-CREDIT", "financing_events"),),
    event_documents={"FIN-CREDIT": _CREDIT_FACILITY_TEXT},
    expected_events=(
        ExpectedEvent("FIN-CREDIT", "ISSUER_DEBT_ISSUANCE", funded=False,
                      committed=True, amount=2_000_000_000.0,
                      should_be_accepted=True,
                      note="committed but not drawn -- must never be counted as funded debt"),
    ),
))

# 7. Financing event: funded term loan, drawdown language present.
_DRAWN_TERM_LOAN_TEXT = (
    "On July 1, 2026, the Company entered into a $1.5 billion term loan "
    "agreement and drew the full $1.5 billion, receiving net proceeds of "
    "approximately $1.48 billion after fees."
)
_add(DocumentPackageCase(
    case_id="financing-funded-term-loan",
    classes=("financing_event", "capital_structure"),
    submissions=_submissions([
        {"accession": "FIN-TERMLOAN", "form": "8-K", "filed": "2026-07-01",
         "items": "2.03", "description": "Entry into Term Loan Agreement",
         "document": "termloan8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-TERMLOAN", "financing_events"),),
    event_documents={"FIN-TERMLOAN": _DRAWN_TERM_LOAN_TEXT},
    expected_events=(
        ExpectedEvent("FIN-TERMLOAN", "ISSUER_DEBT_ISSUANCE", funded=True,
                      committed=False, amount=1_500_000_000.0,
                      should_be_accepted=True, note="drawn -- correctly funded"),
    ),
))

# 8. Financing event: convertible-debt refinement of a bare 2.03 item.
_CONVERTIBLE_TEXT = (
    "On August 1, 2026, the Company issued and sold $800 million aggregate "
    "principal amount of convertible senior notes due 2031 in a private "
    "placement, receiving net proceeds of approximately $784 million."
)
_add(DocumentPackageCase(
    case_id="financing-convertible-refinement",
    classes=("financing_event", "capital_structure"),
    submissions=_submissions([
        {"accession": "FIN-CONV", "form": "8-K", "filed": "2026-08-01",
         "items": "2.03", "description": "Creation of a Direct Financial Obligation",
         "document": "conv8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-CONV", "financing_events"),),
    event_documents={"FIN-CONV": _CONVERTIBLE_TEXT},
    expected_events=(
        ExpectedEvent("FIN-CONV", "CONVERTIBLE_ISSUANCE", funded=True,
                      committed=False, amount=800_000_000.0,
                      should_be_accepted=True,
                      note="issued and sold -- funded, and a convertible "
                           "refinement of a bare 2.03 item"),
    ),
))

# 9. Financing event: equity issuance, item 3.02.
_EQUITY_TEXT = (
    "On August 15, 2026, the Company completed an underwritten public "
    "offering of 10,000,000 shares of common stock, issuing and selling "
    "the shares for aggregate net proceeds of approximately $450 million."
)
_add(DocumentPackageCase(
    case_id="financing-equity-issuance",
    classes=("financing_event", "capital_structure"),
    submissions=_submissions([
        {"accession": "FIN-EQUITY", "form": "8-K", "filed": "2026-08-15",
         "items": "3.02", "description": "Unregistered Sales of Equity Securities",
         "document": "equity8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-EQUITY", "financing_events"),),
    event_documents={"FIN-EQUITY": _EQUITY_TEXT},
    expected_events=(
        ExpectedEvent("FIN-EQUITY", "ISSUER_EQUITY_ISSUANCE", funded=True,
                      committed=False, amount=450_000_000.0,
                      should_be_accepted=True, note="issued and sold shares -- funded"),
    ),
))

# 10. HARD-SAFETY NEGATIVE CONTROL: an item-code mismatch. The filing's own
# item code is 2.03 (debt); a hallucinated ACQUISITION claim must be refused.
_add(DocumentPackageCase(
    case_id="financing-item-code-mismatch-negative-control",
    classes=("financing_event", "hard_safety"),
    submissions=_submissions([
        {"accession": "FIN-MISMATCH", "form": "8-K", "filed": "2026-07-01",
         "items": "2.03", "description": "Entry into Credit Agreement",
         "document": "mismatch8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-MISMATCH", "financing_events"),),
    event_documents={"FIN-MISMATCH": _CREDIT_FACILITY_TEXT},
    expected_events=(
        ExpectedEvent("FIN-MISMATCH", "ACQUISITION", funded=None, committed=None,
                      amount=None, should_be_accepted=False,
                      note="a debt-item filing cannot support an ACQUISITION claim"),
    ),
    # This is a VALIDATOR-boundary test: the ACQUISITION claim is injected
    # directly, not something the real model would propose from this
    # (truthful, debt-only) text. Excluded from the live-reader benchmark.
    live_reader_case=False,
))

# 11. HARD-SAFETY NEGATIVE CONTROL: funded asserted with no drawdown language
# anywhere in the filing -- the exact UNDRAWN_FACILITY_COUNTED_AS_FUNDED_DEBT
# shape, on the SAME committed-only text as case 6.
_add(DocumentPackageCase(
    case_id="financing-undrawn-claimed-funded-negative-control",
    classes=("financing_event", "hard_safety"),
    submissions=_submissions([
        {"accession": "FIN-FALSEFUND", "form": "8-K", "filed": "2026-07-01",
         "items": "2.03", "description": "Entry into Credit Agreement",
         "document": "falsefund8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-FALSEFUND", "financing_events"),),
    event_documents={"FIN-FALSEFUND": _CREDIT_FACILITY_TEXT},
    expected_events=(
        ExpectedEvent("FIN-FALSEFUND", "ISSUER_DEBT_ISSUANCE", funded=True,
                      committed=False, amount=2_000_000_000.0,
                      should_be_accepted=False,
                      note="no drawdown language exists in the source text"),
    ),
    # Same reasoning as case 10: `funded=True` is injected to test the
    # validator, not read by a model. Excluded from the live-reader benchmark.
    live_reader_case=False,
))

# 12. Multi-document package: one of everything at once, to prove the buckets
# don't interfere with each other's counts.
_add(DocumentPackageCase(
    case_id="multi-document-full-package",
    classes=("mature_profitable", "capital_structure"),
    submissions=_submissions([
        {"accession": "FULL-10K", "form": "10-K", "filed": "2026-08-15",
         "report_date": "2026-06-30"},
        {"accession": "FULL-10Q-OLD", "form": "10-Q", "filed": "2026-02-01",
         "report_date": "2025-12-31"},
        {"accession": "FULL-10Q-NEW", "form": "10-Q", "filed": "2026-05-01",
         "report_date": "2026-03-31"},
        {"accession": "FULL-8K-ER", "form": "8-K", "filed": "2026-08-01",
         "items": "2.02,9.01", "description": "Q2 2026 Earnings Release",
         "document": "ex991.htm"},
        {"accession": "FULL-8K-DEBT", "form": "8-K", "filed": "2026-07-01",
         "items": "2.03", "description": "Entry into Credit Agreement",
         "document": "debt8k.htm"},
        {"accession": "FULL-4", "form": "4", "filed": "2026-06-01"},
    ]),
    expected=(
        ExpectedClassification("FULL-10K", "periodic_annual"),
        ExpectedClassification("FULL-10Q-NEW", "periodic_quarterly"),
        ExpectedClassification("FULL-8K-ER", "earnings_releases"),
        ExpectedClassification("FULL-8K-DEBT", "financing_events"),
        ExpectedClassification("FULL-4", "corporate_events"),
    ),
))

# ---------------------------------------------------------------------------
# Phase H.17 additions
# ---------------------------------------------------------------------------

# 13. Actual-table fallback: structured/deterministic paths are NOT complete
# for the target period, so the LLM reader must run against a genuine
# three-statement earnings release and read it correctly. The ground truth
# is `reported_actuals_releases.Q4` -- the SAME dict the deterministic
# `reported_actuals` benchmark's own income/balance-sheet/cash-flow blocks
# are built from.
_FALLBACK_RELEASE_TEXT = release(
    "Q4 FY2026 Results",
    income_statement(quarter=True, annual=False),
    balance_sheet(),
    cash_flow(quarter=True, annual=False),
)
_FALLBACK_EXPECTED_FACTS = (
    ExpectedActualFact("revenue", Q4["revenue"] * 1_000_000.0, Q4_END_ISO),
    ExpectedActualFact("operating_income", Q4["operating_income"] * 1_000_000.0, Q4_END_ISO),
    ExpectedActualFact("net_income", Q4["net_income"] * 1_000_000.0, Q4_END_ISO),
    ExpectedActualFact("cash_and_cash_equivalents",
                       Q4["cash_and_cash_equivalents"] * 1_000_000.0, Q4_END_ISO),
    ExpectedActualFact("short_term_investments",
                       Q4["short_term_investments"] * 1_000_000.0, Q4_END_ISO),
    ExpectedActualFact("current_assets", Q4["current_assets"] * 1_000_000.0, Q4_END_ISO),
    ExpectedActualFact("assets", Q4["assets"] * 1_000_000.0, Q4_END_ISO),
    ExpectedActualFact("current_liabilities",
                       Q4["current_liabilities"] * 1_000_000.0, Q4_END_ISO),
    ExpectedActualFact("liabilities", Q4["liabilities"] * 1_000_000.0, Q4_END_ISO),
    ExpectedActualFact("stockholders_equity",
                       Q4["stockholders_equity"] * 1_000_000.0, Q4_END_ISO),
    ExpectedActualFact("short_term_debt", Q4["short_term_debt"] * 1_000_000.0, Q4_END_ISO),
    ExpectedActualFact("long_term_debt", Q4["long_term_debt"] * 1_000_000.0, Q4_END_ISO),
    ExpectedActualFact("operating_cash_flow",
                       Q4["operating_cash_flow"] * 1_000_000.0, Q4_END_ISO),
    ExpectedActualFact("capital_expenditure",
                       Q4["capital_expenditure"] * 1_000_000.0, Q4_END_ISO),
)
_add(DocumentPackageCase(
    case_id="actual-table-fallback-full-statements",
    classes=("actual_table_fallback", "incomplete_structured_actuals"),
    submissions=_submissions([
        {"accession": "ATF-8K-ER", "form": "8-K", "filed": "2026-08-15",
         "items": "2.02,9.01", "description": "Q4 and FY2026 Results",
         "document": "ex991.htm"},
    ]),
    expected=(ExpectedClassification("ATF-8K-ER", "earnings_releases"),),
    actual_document_accession="ATF-8K-ER",
    actual_document_text=_FALLBACK_RELEASE_TEXT,
    target_period_end=Q4_END_ISO,
    structured_completeness="PARTIAL",
    expected_fallback_triggers=True,
    expected_actual_facts=_FALLBACK_EXPECTED_FACTS,
    notes="XBRL/CompanyFacts has not yet tagged Q4; the release is the only "
         "COMPLETE-eligible source and the fallback must trigger and read it.",
))

# 14. The SAME release text, but the structured path already resolves the
# period COMPLETE. The fallback must NOT trigger and the model must NOT be
# called -- an "unnecessary fallback activation" positive control.
_add(DocumentPackageCase(
    case_id="actual-table-fallback-not-needed",
    classes=("actual_table_fallback",),
    submissions=_submissions([
        {"accession": "ATFN-8K-ER", "form": "8-K", "filed": "2026-08-15",
         "items": "2.02,9.01", "description": "Q4 and FY2026 Results",
         "document": "ex991.htm"},
    ]),
    expected=(ExpectedClassification("ATFN-8K-ER", "earnings_releases"),),
    actual_document_accession="ATFN-8K-ER",
    actual_document_text=_FALLBACK_RELEASE_TEXT,
    target_period_end=Q4_END_ISO,
    structured_completeness="COMPLETE",
    expected_fallback_triggers=False,
    expected_actual_facts=(),
    notes="Structured CompanyFacts already resolves Q4 COMPLETE; the model "
         "must never be invoked for this case.",
))

# 15. Guidance-adjacent-to-actuals contamination: the SAME reported income
# statement, plus a FINANCIAL OUTLOOK table in the same document. The
# document resolver/actuals extractor must never let the outlook's forward
# figures reach the model as candidate actual facts -- `select_actual_sections`
# excludes any table not classified a reported statement before the model
# ever sees the document, and this case proves that holds end to end.
_CONTAMINATION_RELEASE_TEXT = release(
    "Q4 FY2026 Results and Business Outlook",
    income_statement(quarter=True, annual=False),
    outlook_table(),
)
_add(DocumentPackageCase(
    case_id="actual-table-guidance-adjacent-contamination",
    classes=("actual_table_fallback", "guidance_near_actual_table"),
    submissions=_submissions([
        {"accession": "CONTAM-8K-ER", "form": "8-K", "filed": "2026-08-15",
         "items": "2.02,9.01", "description": "Q4 FY2026 Results and Business Outlook",
         "document": "ex991.htm"},
    ]),
    expected=(ExpectedClassification("CONTAM-8K-ER", "earnings_releases"),),
    actual_document_accession="CONTAM-8K-ER",
    actual_document_text=_CONTAMINATION_RELEASE_TEXT,
    target_period_end=Q4_END_ISO,
    structured_completeness="PARTIAL",
    expected_fallback_triggers=True,
    expected_actual_facts=(
        ExpectedActualFact("revenue", Q4["revenue"] * 1_000_000.0, Q4_END_ISO),
        ExpectedActualFact("operating_income", Q4["operating_income"] * 1_000_000.0,
                           Q4_END_ISO),
        ExpectedActualFact("net_income", Q4["net_income"] * 1_000_000.0, Q4_END_ISO),
        # The outlook's own figures must NEVER be accepted as Q4-end actuals.
        ExpectedActualFact("revenue", 1_600.0 * 1_000_000.0, "2026-10-31",
                           should_be_accepted=False,
                           note="this is the OUTLOOK table's Q1 FY2027 guided "
                                "revenue, not a Q4 FY2026 actual"),
    ),
))

# 16-17. Two more financing-event shapes named in spec section 13.
_BRIDGE_FACILITY_TEXT = (
    "On September 1, 2026, the Company entered into a $750 million senior "
    "unsecured bridge loan facility with a syndicate of lenders led by "
    "Example Bank, N.A., to provide interim financing pending the issuance "
    "of long-term notes. The facility remains fully undrawn and available "
    "as of the date of this filing."
)
_add(DocumentPackageCase(
    case_id="financing-bridge-facility-undrawn",
    classes=("financing_event", "bridge_facility"),
    submissions=_submissions([
        {"accession": "FIN-BRIDGE", "form": "8-K", "filed": "2026-09-01",
         "items": "2.03", "description": "Entry into Bridge Loan Facility",
         "document": "bridge8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-BRIDGE", "financing_events"),),
    event_documents={"FIN-BRIDGE": _BRIDGE_FACILITY_TEXT},
    expected_events=(
        ExpectedEvent("FIN-BRIDGE", "ISSUER_DEBT_ISSUANCE", funded=False,
                      committed=True, amount=750_000_000.0,
                      should_be_accepted=True,
                      note="a bridge facility that remains fully undrawn -- "
                           "committed, not funded"),
    ),
))

_REFINANCING_TEXT = (
    "On September 15, 2026, the Company completed a refinancing of its "
    "existing 5.500% Senior Notes due 2027. The Company issued and sold "
    "$600 million aggregate principal amount of new 4.750% Senior Notes due "
    "2033 and used the net proceeds, together with cash on hand, to redeem "
    "in full the outstanding 2027 notes."
)
_add(DocumentPackageCase(
    case_id="financing-debt-refinancing",
    classes=("financing_event", "debt_refinancing"),
    submissions=_submissions([
        {"accession": "FIN-REFI", "form": "8-K", "filed": "2026-09-15",
         "items": "2.03", "description": "Completion of Debt Refinancing",
         "document": "refi8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-REFI", "financing_events"),),
    event_documents={"FIN-REFI": _REFINANCING_TEXT},
    expected_events=(
        ExpectedEvent("FIN-REFI", "REFINANCING", funded=True, committed=False,
                      amount=600_000_000.0, should_be_accepted=True,
                      note="issued and sold new notes to redeem the old ones "
                           "-- funded, and a REFINANCING refinement of a bare "
                           "2.03 item"),
    ),
))


def all_cases() -> Tuple[DocumentPackageCase, ...]:
    return tuple(_CASES)


def classes_covered() -> Tuple[str, ...]:
    seen = []
    for case in _CASES:
        for cls in case.classes:
            if cls not in seen:
                seen.append(cls)
    return tuple(seen)
