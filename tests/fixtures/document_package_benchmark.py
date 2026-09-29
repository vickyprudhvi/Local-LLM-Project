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
from finance.documents.event_schema import EventAmountRole


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
    # Phase H.20. None means "not checked" (pre-H.20 cases, or a case whose
    # point is something other than amount semantics); a real role means the
    # live benchmark's per-item scoring also verifies the ACCEPTED event
    # carries this exact role, not merely a grounded value.
    expected_amount_role: Optional[str] = None
    # Additional independently-typed amounts a case expects to survive
    # alongside the primary one (spec section 8's "$5B commitment, $2B
    # drawn" / "$95/share, $69B total" shapes). Each is (role, amount).
    expected_supplementary: Tuple[Tuple[str, float], ...] = ()
    # Phase H.21. None means "not checked" -- most pre-H.21 cases are USD
    # and the currency dimension was never separately measured. A real
    # ISO code means the live benchmark's per-item scoring also verifies
    # the reader's OWN currency claim (independent of acceptance).
    expected_currency: Optional[str] = None


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

# Real SEC 8-Ks are always ITEM-numbered (finance.documents.package only
# ever classifies a form-8-K document FINANCING_EVENT, and a form 8-K is
# defined by its Item structure) -- Phase H.19's `select_event_sections`
# now selects a normalized ITEM BLOCK rather than a raw document prefix, so
# a fixture with no "Item N.NN" heading at all would never reach the model
# in production and is not a realistic financing-event document. Every
# event fixture below is prefixed with the item's own SEC boilerplate
# title, exactly as a real filing states it.
_ITEM_TITLES = {
    "1.01": "Entry into a Material Definitive Agreement.",
    "2.01": "Completion of Acquisition or Disposition of Assets.",
    "2.03": "Creation of a Direct Financial Obligation or an Obligation under "
           "an Off-Balance Sheet Arrangement of a Registrant.",
    "3.02": "Unregistered Sales of Equity Securities.",
}


def _item_prefixed(item_code: str, body: str) -> str:
    return f"Item {item_code} {_ITEM_TITLES[item_code]} {body}"


# 6. Financing event: committed facility, no drawdown language anywhere.
_CREDIT_FACILITY_TEXT = _item_prefixed("2.03",
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
_DRAWN_TERM_LOAN_TEXT = _item_prefixed("2.03",
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
_CONVERTIBLE_TEXT = _item_prefixed("2.03",
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
_EQUITY_TEXT = _item_prefixed("3.02",
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
_BRIDGE_FACILITY_TEXT = _item_prefixed("2.03",
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

_REFINANCING_TEXT = _item_prefixed("2.03",
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

# ---------------------------------------------------------------------------
# Phase H.20 additions -- amount-role coverage
# ---------------------------------------------------------------------------

# 18. Acquisition stating BOTH a per-share price and an explicit aggregate
# transaction value -- section 8's "$95/share, $69B total" shape. The two
# numbers describe two different economic quantities and must survive as
# two independently-grounded, distinctly-typed amounts, never collapsed or
# used to derive one from the other.
_ACQUISITION_PER_SHARE_AND_TOTAL_TEXT = _item_prefixed("2.01",
    "On October 1, 2026, the Company completed its acquisition of Example "
    "Target, Inc. Each outstanding share of Example Target common stock was "
    "converted into the right to receive $95.00 in cash, without interest, "
    "representing an aggregate transaction value of approximately "
    "$69.0 billion."
)
_add(DocumentPackageCase(
    case_id="financing-acquisition-per-share-and-total",
    classes=("financing_event", "amount_role", "acquisition"),
    submissions=_submissions([
        {"accession": "FIN-ACQ", "form": "8-K", "filed": "2026-10-01",
         "items": "2.01", "description": "Completion of Acquisition of Assets",
         "document": "acq8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-ACQ", "financing_events"),),
    event_documents={"FIN-ACQ": _ACQUISITION_PER_SHARE_AND_TOTAL_TEXT},
    expected_events=(
        ExpectedEvent("FIN-ACQ", "ACQUISITION", funded=False, committed=False,
                      amount=95.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.PER_SHARE_CONSIDERATION,
                      expected_supplementary=((EventAmountRole.TRANSACTION_VALUE,
                                              69_000_000_000.0),),
                      note="the live-shadow defect, reproduced generically: $95.00 "
                           "is a PER-SHARE price, never the deal's transaction value, "
                           "which is separately and explicitly $69.0 billion"),
    ),
))

# 19. Notes issuance: a clean, single PRINCIPAL_AMOUNT.
_NOTES_PRINCIPAL_TEXT = _item_prefixed("2.03",
    "On October 5, 2026, the Company issued and sold $750 million aggregate "
    "principal amount of 5.500% Senior Notes due 2033 in a registered "
    "public offering."
)
_add(DocumentPackageCase(
    case_id="financing-notes-principal-amount",
    classes=("financing_event", "amount_role", "notes_issuance"),
    submissions=_submissions([
        {"accession": "FIN-NOTES", "form": "8-K", "filed": "2026-10-05",
         "items": "2.03", "description": "Issuance of Senior Notes",
         "document": "notes8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-NOTES", "financing_events"),),
    event_documents={"FIN-NOTES": _NOTES_PRINCIPAL_TEXT},
    expected_events=(
        ExpectedEvent("FIN-NOTES", "ISSUER_DEBT_ISSUANCE", funded=True,
                      committed=False, amount=750_000_000.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.PRINCIPAL_AMOUNT,
                      note="issued and sold -- funded, and the amount is the "
                           "notes' own aggregate principal, not a facility "
                           "commitment or a transaction value"),
    ),
))

# 20. Revolving facility with a PARTIAL drawdown -- section 8's "$5B
# commitment, of which $2B was drawn" shape. The commitment and the drawn
# portion are different amounts with different roles, both grounded.
_FACILITY_PARTIAL_DRAWDOWN_TEXT = _item_prefixed("2.03",
    "On October 10, 2026, the Company entered into a $5.0 billion revolving "
    "credit facility with a syndicate of lenders; at closing, the Company "
    "drew $2.0 billion under the facility for general corporate purposes."
)
_add(DocumentPackageCase(
    case_id="financing-facility-partial-drawdown",
    classes=("financing_event", "amount_role", "partial_drawdown"),
    submissions=_submissions([
        {"accession": "FIN-PARTIAL", "form": "8-K", "filed": "2026-10-10",
         "items": "2.03", "description": "Entry into Credit Agreement",
         "document": "partial8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-PARTIAL", "financing_events"),),
    event_documents={"FIN-PARTIAL": _FACILITY_PARTIAL_DRAWDOWN_TEXT},
    expected_events=(
        ExpectedEvent("FIN-PARTIAL", "ISSUER_DEBT_ISSUANCE", funded=True,
                      committed=True, amount=5_000_000_000.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.FACILITY_COMMITMENT,
                      expected_supplementary=((EventAmountRole.AMOUNT_DRAWN,
                                              2_000_000_000.0),),
                      note="the $5B commitment and the $2B drawn are different "
                           "amounts; funded=true is grounded by the draw, not by "
                           "the commitment"),
    ),
))

# 21. Debt repayment/refinancing: a clean REPAYMENT_AMOUNT.
_DEBT_REPAYMENT_TEXT = _item_prefixed("2.03",
    "On October 15, 2026, the Company repaid $800 million in aggregate "
    "principal amount of its outstanding senior notes at maturity, using "
    "cash on hand."
)
_add(DocumentPackageCase(
    case_id="financing-debt-repayment",
    classes=("financing_event", "amount_role", "debt_repayment"),
    submissions=_submissions([
        {"accession": "FIN-REPAY", "form": "8-K", "filed": "2026-10-15",
         "items": "2.03", "description": "Repayment of Senior Notes",
         "document": "repay8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-REPAY", "financing_events"),),
    event_documents={"FIN-REPAY": _DEBT_REPAYMENT_TEXT},
    expected_events=(
        ExpectedEvent("FIN-REPAY", "REFINANCING", funded=False, committed=False,
                      amount=800_000_000.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.REPAYMENT_AMOUNT,
                      note="a repayment, not a new draw -- funded describes money "
                           "RECEIVED by the company, which did not happen here"),
    ),
))


# ---------------------------------------------------------------------------
# Phase H.21 additions -- generalized shapes not yet in the benchmark
# ---------------------------------------------------------------------------

# 22. Bridge facility DRAWDOWN (case E) -- the undrawn-bridge shape (case 7
# above) proved committed-not-funded; this is its funded counterpart, a
# bridge facility that IS fully drawn.
_BRIDGE_DRAWN_TEXT = _item_prefixed("2.03",
    "On September 1, 2026, the Company drew the full $750 million available "
    "under its previously established senior unsecured bridge loan facility "
    "with Example Bank, N.A. to fund a portion of the cash consideration for "
    "the Company's pending acquisition of Example Target Corp., pending "
    "permanent financing."
)
_add(DocumentPackageCase(
    case_id="financing-bridge-facility-drawn",
    classes=("financing_event", "amount_role", "bridge_facility"),
    submissions=_submissions([
        {"accession": "FIN-BRIDGE-DRAWN", "form": "8-K", "filed": "2026-09-01",
         "items": "2.03", "description": "Drawdown under Bridge Loan Facility",
         "document": "bridgedrawn8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-BRIDGE-DRAWN", "financing_events"),),
    event_documents={"FIN-BRIDGE-DRAWN": _BRIDGE_DRAWN_TEXT},
    expected_events=(
        ExpectedEvent("FIN-BRIDGE-DRAWN", "ISSUER_DEBT_ISSUANCE", funded=True,
                      committed=True, amount=750_000_000.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.AMOUNT_DRAWN,
                      note="a bridge facility fully drawn -- unlike case 7's "
                           "undrawn bridge, this one MUST be recognized as funded"),
    ),
))

# 23. Facility amendment / maturity extension (case J) -- no new money, no
# new commitment: an existing facility's maturity date moves, nothing else.
# The safety property is the mirror image of the undrawn-facility cases:
# nothing here should be read as a NEW financing amount, drawn or committed.
_MATURITY_EXTENSION_TEXT = _item_prefixed("1.01",
    "On September 10, 2026, the Company entered into a Second Amendment to "
    "its existing $1.5 billion revolving credit facility, extending the "
    "maturity date from March 2028 to March 2031. The amendment did not "
    "increase the aggregate commitment under the facility, and no "
    "additional amounts were drawn in connection with the amendment."
)
_add(DocumentPackageCase(
    case_id="financing-facility-maturity-extension",
    classes=("financing_event", "amount_role", "facility_amendment"),
    submissions=_submissions([
        {"accession": "FIN-MATEXT", "form": "8-K", "filed": "2026-09-10",
         "items": "1.01", "description": "Amendment to Revolving Credit Facility",
         "document": "matext8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-MATEXT", "financing_events"),),
    event_documents={"FIN-MATEXT": _MATURITY_EXTENSION_TEXT},
    expected_events=(
        # `amount=None`: this event asserts NO new financing figure -- a
        # maturity-only amendment is not itself a fresh draw or commitment.
        # Item 1.01 is deterministically UNRESTRICTED (spec section 6), so
        # ISSUER_DEBT_ISSUANCE here is the most natural reading, not the
        # only defensible one -- OTHER_MATERIAL_FINANCING is a reasonable
        # alternative this phase measures rather than forces.
        ExpectedEvent("FIN-MATEXT", "ISSUER_DEBT_ISSUANCE", funded=False,
                      committed=False, amount=None, should_be_accepted=True,
                      note="maturity-only amendment -- no new amount, no new "
                           "draw; must not be misread as new debt issuance"),
    ),
))

# 24. Multiple facilities in one paragraph (case K) -- two independently
# drawn amounts in the SAME sentence must survive as two distinct figures,
# never summed or collapsed into one (the live analog of the H.19 unit
# test's synthetic two-facility-draw scenario).
_MULTI_FACILITY_TEXT = _item_prefixed("2.03",
    "On September 15, 2026, the Company entered into a new credit "
    "agreement with a syndicate of lenders. At closing, the Company drew "
    "$1.0 billion under Term Loan A and $500 million under Term Loan B to "
    "refinance existing indebtedness."
)
_add(DocumentPackageCase(
    case_id="financing-multiple-facilities-one-paragraph",
    classes=("financing_event", "amount_role", "multi_facility"),
    submissions=_submissions([
        {"accession": "FIN-MULTI", "form": "8-K", "filed": "2026-09-15",
         "items": "2.03", "description": "Entry into New Credit Agreement",
         "document": "multi8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-MULTI", "financing_events"),),
    event_documents={"FIN-MULTI": _MULTI_FACILITY_TEXT},
    expected_events=(
        # No `expected_amount_role`/`expected_supplementary` role pinned
        # here: for a FULLY drawn term loan, "amount drawn" and "principal
        # amount" are the SAME true fact about the SAME number (unlike
        # FIN-PARTIAL, where the commitment and the draw are genuinely
        # DIFFERENT amounts) -- AMOUNT_DRAWN and PRINCIPAL_AMOUNT are both
        # correct, non-dangerous readings, so pinning one as "the" answer
        # would penalize a safe, legitimate alternative (live H.22 finding:
        # a role-language rewrite meant to close this ambiguity just moved
        # it from FACILITY_COMMITMENT to PRINCIPAL_AMOUNT, confirming the
        # ambiguity is inherent to the shape, not a defect to fix away).
        # The test's actual point -- $1.0B and $500M survive as two
        # distinct figures, never summed into $1.5B -- is still enforced
        # by `amount_ok` requiring the PRIMARY value ($1.0B) be found.
        ExpectedEvent("FIN-MULTI", "ISSUER_DEBT_ISSUANCE", funded=True,
                      committed=True, amount=1_000_000_000.0, should_be_accepted=True,
                      note="two distinct tranches, $1.0B and $500M, both drawn "
                           "-- must never be summed into a single $1.5B figure"),
    ),
))

# 25. Foreign issuer financing disclosure (case L) -- a non-USD principal
# amount, to measure currency accuracy specifically (spec section 5).
_FOREIGN_ISSUER_NOTES_TEXT = _item_prefixed("2.03",
    "On September 20, 2026, the Company, a Netherlands-incorporated public "
    "limited company, issued and sold €500 million aggregate principal "
    "amount of 4.250% Senior Notes due 2033 in a Regulation S offering "
    "outside the United States."
)
_add(DocumentPackageCase(
    case_id="financing-foreign-issuer-eur-notes",
    classes=("financing_event", "amount_role", "foreign_issuer"),
    submissions=_submissions([
        {"accession": "FIN-EURNOTES", "form": "8-K", "filed": "2026-09-20",
         "items": "2.03", "description": "Issuance of Euro-Denominated Notes",
         "document": "eurnotes8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-EURNOTES", "financing_events"),),
    # `_detect_reporting_status` derives foreign_private_issuer status from
    # 20-F/40-F filings actually present in `submissions` (none here -- this
    # fixture's issuer files domestically); the "foreign" dimension under
    # test is the disclosure's own CURRENCY, not the SEC reporting-status
    # bucket, so the default domestic_registrant is the correct ground truth.
    event_documents={"FIN-EURNOTES": _FOREIGN_ISSUER_NOTES_TEXT},
    expected_events=(
        ExpectedEvent("FIN-EURNOTES", "ISSUER_DEBT_ISSUANCE", funded=True,
                      committed=False, amount=500_000_000.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.PRINCIPAL_AMOUNT,
                      expected_currency="EUR",
                      note="a EUR-denominated notes issuance -- currency must "
                           "be read as EUR, never defaulted to USD"),
    ),
))

# 26. Facility commitment INCREASE (case M) -- an upsize amendment. The
# headline figure is the NEW total commitment, not the delta, and not the
# superseded old commitment; nothing was drawn.
_COMMITMENT_INCREASE_TEXT = _item_prefixed("2.03",
    "On September 25, 2026, the Company entered into an amendment to its "
    "existing revolving credit facility increasing the aggregate "
    "commitments thereunder from $2.0 billion to $2.75 billion. No amounts "
    "were drawn under the facility in connection with the amendment."
)
_add(DocumentPackageCase(
    case_id="financing-facility-commitment-increase",
    classes=("financing_event", "amount_role", "facility_amendment"),
    submissions=_submissions([
        {"accession": "FIN-UPSIZE", "form": "8-K", "filed": "2026-09-25",
         "items": "2.03", "description": "Amendment Increasing Revolving Commitments",
         "document": "upsize8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-UPSIZE", "financing_events"),),
    event_documents={"FIN-UPSIZE": _COMMITMENT_INCREASE_TEXT},
    expected_events=(
        ExpectedEvent("FIN-UPSIZE", "ISSUER_DEBT_ISSUANCE", funded=False,
                      committed=True, amount=2_750_000_000.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.FACILITY_COMMITMENT,
                      note="the NEW total commitment after the upsize -- not "
                           "the $2.0B superseded figure, not the $750M delta, "
                           "and never funded (no drawdown language)"),
    ),
))

# 27. Debt issuance with explicit net proceeds (case N) -- a straight,
# non-convertible notes issuance stating both its own principal AND net
# proceeds. Distinct from case 8's convertible-notes shape: here
# AMOUNT_DRAWN (via `_FUNDED_LANGUAGE`'s "net proceeds" language) IS in
# ISSUER_DEBT_ISSUANCE's allowed role set, so this also live-regression-
# tests the H.20 supplementary-drop fix on a case where the supplementary
# role SHOULD ground cleanly.
_NOTES_WITH_NET_PROCEEDS_TEXT = _item_prefixed("2.03",
    "On September 28, 2026, the Company issued and sold $500 million "
    "aggregate principal amount of 5.750% Senior Notes due 2034 in an "
    "underwritten public offering, receiving net proceeds of approximately "
    "$493 million after underwriting discounts and offering expenses."
)
_add(DocumentPackageCase(
    case_id="financing-notes-with-net-proceeds",
    classes=("financing_event", "amount_role", "notes_issuance"),
    submissions=_submissions([
        {"accession": "FIN-NETPROCEEDS", "form": "8-K", "filed": "2026-09-28",
         "items": "2.03", "description": "Issuance of Senior Notes",
         "document": "netproceeds8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-NETPROCEEDS", "financing_events"),),
    event_documents={"FIN-NETPROCEEDS": _NOTES_WITH_NET_PROCEEDS_TEXT},
    expected_events=(
        ExpectedEvent("FIN-NETPROCEEDS", "ISSUER_DEBT_ISSUANCE", funded=True,
                      committed=False, amount=500_000_000.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.PRINCIPAL_AMOUNT,
                      expected_supplementary=((EventAmountRole.AMOUNT_DRAWN,
                                              493_000_000.0),),
                      note="the $493M net-proceeds figure is a recall target, "
                           "not a hard-safety one -- missing it is a miss; "
                           "the $500M principal amount is the amount that "
                           "must never be lost or misrole'd"),
    ),
))

# 28. Acquisition financing combining facility + funded borrowing (case O)
# -- the real T/EchoStar shadow shape, reproduced generically. An
# acquisition is mentioned BY NAME as the funding's purpose, but the only
# amounts with financing evidence are the draw and its facility's
# commitment; this is section 4's "a background acquisition mention inside
# a financing item must not become a separate ACQUISITION event unless
# evidence supports it" -- the deterministic item-code backstop (item 2.03
# does not admit ACQUISITION, see `_COMPATIBLE_REFINEMENTS`) already
# protects this, and this case exercises it against real prose rather than
# an injected claim.
_ACQUISITION_FUNDED_VIA_FACILITY_TEXT = _item_prefixed("2.03",
    "On October 1, 2026, in connection with the Company's previously "
    "announced acquisition of Example Target Holdings, Inc., the Company "
    "drew $2.0 billion under its existing $3.0 billion delayed-draw term "
    "loan credit agreement with Example Bank, N.A., as agent, to finance a "
    "portion of the cash consideration for the acquisition, with the "
    "balance funded from cash on hand."
)
_add(DocumentPackageCase(
    case_id="financing-acquisition-funded-via-facility",
    classes=("financing_event", "amount_role", "acquisition_financing"),
    submissions=_submissions([
        {"accession": "FIN-ACQFUNDED", "form": "8-K", "filed": "2026-10-01",
         "items": "2.03", "description": "Drawdown to Fund Pending Acquisition",
         "document": "acqfunded8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-ACQFUNDED", "financing_events"),),
    event_documents={"FIN-ACQFUNDED": _ACQUISITION_FUNDED_VIA_FACILITY_TEXT},
    expected_events=(
        ExpectedEvent("FIN-ACQFUNDED", "ISSUER_DEBT_ISSUANCE", funded=True,
                      committed=True, amount=2_000_000_000.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.AMOUNT_DRAWN,
                      expected_supplementary=((EventAmountRole.FACILITY_COMMITMENT,
                                              3_000_000_000.0),),
                      note="the acquisition is named as the PURPOSE of the "
                           "draw, not itself evidenced -- exactly one debt "
                           "event is expected; a spurious separate "
                           "ACQUISITION event must not be accepted (the "
                           "item-code backstop already guards this)"),
    ),
))


# ---------------------------------------------------------------------------
# Phase H.22 additions -- currency-independent monetary scale
# ---------------------------------------------------------------------------

# 29. GBP-denominated facility (case: "GBP750 million facility").
_GBP_FACILITY_TEXT = _item_prefixed("2.03",
    "On October 12, 2026, the Company entered into a new senior unsecured "
    "revolving credit facility with a syndicate of lenders led by Example "
    "Bank plc, providing for commitments of £750 million. No amounts "
    "have been drawn under the facility as of the date of this filing."
)
_add(DocumentPackageCase(
    case_id="financing-foreign-issuer-gbp-facility",
    classes=("financing_event", "amount_role", "monetary_scale"),
    submissions=_submissions([
        {"accession": "FIN-GBPFAC", "form": "8-K", "filed": "2026-10-12",
         "items": "2.03", "description": "Entry into GBP Revolving Credit Facility",
         "document": "gbpfac8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-GBPFAC", "financing_events"),),
    event_documents={"FIN-GBPFAC": _GBP_FACILITY_TEXT},
    expected_events=(
        ExpectedEvent("FIN-GBPFAC", "ISSUER_DEBT_ISSUANCE", funded=False,
                      committed=True, amount=750_000_000.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.FACILITY_COMMITMENT,
                      expected_currency="GBP",
                      note="a GBP-denominated facility -- currency must be "
                           "read as GBP and scale as MILLION, independently"),
    ),
))

# 30. JPY-denominated borrowing (case: "JPY5 billion borrowing").
_JPY_BORROWING_TEXT = _item_prefixed("2.03",
    "On October 14, 2026, the Company, through its wholly-owned Japanese "
    "subsidiary, issued and sold ¥5 billion aggregate principal amount "
    "of unsecured loan notes to a syndicate of Japanese lenders, receiving "
    "net proceeds in the same amount."
)
_add(DocumentPackageCase(
    case_id="financing-foreign-issuer-jpy-borrowing",
    classes=("financing_event", "amount_role", "monetary_scale"),
    submissions=_submissions([
        {"accession": "FIN-JPYBORROW", "form": "8-K", "filed": "2026-10-14",
         "items": "2.03", "description": "Issuance of Yen-Denominated Loan Notes",
         "document": "jpyborrow8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-JPYBORROW", "financing_events"),),
    event_documents={"FIN-JPYBORROW": _JPY_BORROWING_TEXT},
    expected_events=(
        ExpectedEvent("FIN-JPYBORROW", "ISSUER_DEBT_ISSUANCE", funded=True,
                      committed=False, amount=5_000_000_000.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.PRINCIPAL_AMOUNT,
                      expected_currency="JPY",
                      note="a JPY-denominated borrowing -- BILLION scale must "
                           "survive even though the currency is neither USD "
                           "nor a currency this reader sees often"),
    ),
))

# 31. Clean USD baseline under the NEW currency-neutral contract (unit=
# CURRENCY + independent scale, not the legacy USD_MILLION token) -- proves
# the decoupling did not regress the common case.
_USD_NOTES_BASELINE_TEXT = _item_prefixed("2.03",
    "On October 16, 2026, the Company issued and sold $500 million "
    "aggregate principal amount of 5.125% Senior Notes due 2032 in an "
    "underwritten public offering."
)
_add(DocumentPackageCase(
    case_id="financing-usd-notes-baseline",
    classes=("financing_event", "amount_role", "monetary_scale"),
    submissions=_submissions([
        {"accession": "FIN-USDBASE", "form": "8-K", "filed": "2026-10-16",
         "items": "2.03", "description": "Issuance of Senior Notes",
         "document": "usdbase8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-USDBASE", "financing_events"),),
    event_documents={"FIN-USDBASE": _USD_NOTES_BASELINE_TEXT},
    expected_events=(
        ExpectedEvent("FIN-USDBASE", "ISSUER_DEBT_ISSUANCE", funded=True,
                      committed=False, amount=500_000_000.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.PRINCIPAL_AMOUNT,
                      expected_currency="USD",
                      note="the common USD case under the new currency/scale "
                           "contract -- must not regress"),
    ),
))

# 32. A second clean USD baseline at BILLION scale, deliberately a facility
# (not notes) to pair with case 31.
_USD_FACILITY_BASELINE_TEXT = _item_prefixed("2.03",
    "On October 18, 2026, the Company entered into a new $1.2 billion "
    "revolving credit facility with a syndicate of lenders. No amounts "
    "have been drawn under the facility as of the date of this filing."
)
_add(DocumentPackageCase(
    case_id="financing-usd-facility-baseline",
    classes=("financing_event", "amount_role", "monetary_scale"),
    submissions=_submissions([
        {"accession": "FIN-USDFACBASE", "form": "8-K", "filed": "2026-10-18",
         "items": "2.03", "description": "Entry into Revolving Credit Facility",
         "document": "usdfacbase8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-USDFACBASE", "financing_events"),),
    event_documents={"FIN-USDFACBASE": _USD_FACILITY_BASELINE_TEXT},
    expected_events=(
        ExpectedEvent("FIN-USDFACBASE", "ISSUER_DEBT_ISSUANCE", funded=False,
                      committed=True, amount=1_200_000_000.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.FACILITY_COMMITMENT,
                      expected_currency="USD",
                      note="a second USD baseline at BILLION scale"),
    ),
))

# 33. A bare currency amount with NO scale word at all -- the correct
# reading is scale=UNIT (a literal, small EUR500, not 500 million). Proves
# UNIT is a real, gettable-right answer, not merely "the thing that's never
# tested". Framed as a small demand note (realistically un-scaled in real
# filings, unlike a notes/facility headline figure).
_BARE_EUR_TEXT = _item_prefixed("2.03",
    "On October 20, 2026, the Company issued and sold a promissory note in "
    "the aggregate principal amount of €500 in a private placement to "
    "Example Financing B.V."
)
_add(DocumentPackageCase(
    case_id="financing-bare-currency-no-scale-word",
    classes=("financing_event", "amount_role", "monetary_scale"),
    submissions=_submissions([
        {"accession": "FIN-BAREEUR", "form": "8-K", "filed": "2026-10-20",
         "items": "2.03", "description": "Issuance of Promissory Note",
         "document": "bareeur8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-BAREEUR", "financing_events"),),
    event_documents={"FIN-BAREEUR": _BARE_EUR_TEXT},
    expected_events=(
        ExpectedEvent("FIN-BAREEUR", "ISSUER_DEBT_ISSUANCE", funded=True,
                      committed=False, amount=500.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.PRINCIPAL_AMOUNT,
                      expected_currency="EUR",
                      note="NO scale word anywhere -- the correct reading is "
                           "literal EUR500, never inflated to 500 thousand/"
                           "million/billion by assuming a 'normal' facility "
                           "size"),
    ),
))

# 34. Mixed USD/EUR amounts in ONE document -- a multicurrency facility
# whose Euro sub-limit must keep its OWN currency, never inherit USD from
# the primary commitment.
_MIXED_CURRENCY_TEXT = _item_prefixed("2.03",
    "On October 22, 2026, the Company entered into a new $2.0 billion "
    "multicurrency revolving credit facility with a syndicate of lenders, "
    "of which up to €500 million is available for borrowings "
    "denominated in Euro. No amounts have been drawn under the facility as "
    "of the date of this filing."
)
_add(DocumentPackageCase(
    case_id="financing-mixed-currency-facility",
    classes=("financing_event", "amount_role", "monetary_scale"),
    submissions=_submissions([
        {"accession": "FIN-MIXEDCCY", "form": "8-K", "filed": "2026-10-22",
         "items": "2.03", "description": "Entry into Multicurrency Credit Facility",
         "document": "mixedccy8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-MIXEDCCY", "financing_events"),),
    event_documents={"FIN-MIXEDCCY": _MIXED_CURRENCY_TEXT},
    expected_events=(
        ExpectedEvent("FIN-MIXEDCCY", "ISSUER_DEBT_ISSUANCE", funded=False,
                      committed=True, amount=2_000_000_000.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.FACILITY_COMMITMENT,
                      expected_currency="USD",
                      expected_supplementary=((EventAmountRole.FACILITY_COMMITMENT,
                                              500_000_000.0),),
                      note="the EUR500M Euro sub-limit is a DIFFERENT "
                           "currency from the USD2.0B total commitment -- "
                           "each amount must keep its own currency"),
    ),
))

# 35. Facility commitment and drawdown stated in DIFFERENT currencies --
# a real multicurrency-facility shape (draw made in one currency out of a
# commitment denominated in another).
_COMMITMENT_DRAWDOWN_DIFFERENT_CCY_TEXT = _item_prefixed("2.03",
    "On October 24, 2026, the Company entered into a new £750 million "
    "revolving credit facility with a syndicate of lenders; at closing, the "
    "Company drew $400 million-equivalent under the facility in US Dollars "
    "for general corporate purposes."
)
_add(DocumentPackageCase(
    case_id="financing-commitment-drawdown-different-currencies",
    classes=("financing_event", "amount_role", "monetary_scale"),
    submissions=_submissions([
        {"accession": "FIN-DIFFCCY", "form": "8-K", "filed": "2026-10-24",
         "items": "2.03", "description": "Entry into Multicurrency Facility with Drawdown",
         "document": "diffccy8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-DIFFCCY", "financing_events"),),
    event_documents={"FIN-DIFFCCY": _COMMITMENT_DRAWDOWN_DIFFERENT_CCY_TEXT},
    expected_events=(
        ExpectedEvent("FIN-DIFFCCY", "ISSUER_DEBT_ISSUANCE", funded=True,
                      committed=True, amount=750_000_000.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.FACILITY_COMMITMENT,
                      expected_currency="GBP",
                      expected_supplementary=((EventAmountRole.AMOUNT_DRAWN,
                                              400_000_000.0),),
                      note="the GBP750M commitment and the USD400M draw are "
                           "in DIFFERENT currencies -- neither may borrow "
                           "the other's currency label"),
    ),
))

# 36. Amount and its scale word in ADJACENT evidence spans (not the same
# sentence) -- grounding must work across the model's cited span SET, not
# require the number and its magnitude word to share one span.
_SCALE_ADJACENT_SPANS_TEXT = _item_prefixed("2.03",
    "On October 26, 2026, the Company entered into a new senior secured "
    "term loan facility with Example Bank plc as administrative agent. The "
    "facility's aggregate commitment is £750; that commitment figure "
    "is expressed in millions of pounds sterling. No amounts have been "
    "drawn under the facility as of the date of this filing."
)
_add(DocumentPackageCase(
    case_id="financing-scale-adjacent-spans",
    classes=("financing_event", "amount_role", "monetary_scale"),
    submissions=_submissions([
        {"accession": "FIN-ADJSPAN", "form": "8-K", "filed": "2026-10-26",
         "items": "2.03", "description": "Entry into Term Loan Facility",
         "document": "adjspan8k.htm"},
    ]),
    expected=(ExpectedClassification("FIN-ADJSPAN", "financing_events"),),
    event_documents={"FIN-ADJSPAN": _SCALE_ADJACENT_SPANS_TEXT},
    expected_events=(
        ExpectedEvent("FIN-ADJSPAN", "ISSUER_DEBT_ISSUANCE", funded=False,
                      committed=True, amount=750_000_000.0, should_be_accepted=True,
                      expected_amount_role=EventAmountRole.FACILITY_COMMITMENT,
                      expected_currency="GBP",
                      note="the scale word 'millions' is in the sentence "
                           "AFTER the number -- grounding must search the "
                           "full multi-span cited evidence, not just the "
                           "span containing the bare figure"),
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
