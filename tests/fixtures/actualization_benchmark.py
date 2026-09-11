"""The Actualization benchmark: historical filing SEQUENCES and what is true of them.

WHAT A CASE IS

A sequence of filings, expressed the way the production layers actually
receive one: an SEC company-facts payload. Both layers are then run over the
SAME payload, so neither is measured through a harness the other does not
get.

    V1   finance/freshness.py::build_current_financial_state
    V2   finance/extraction/document_resolver.py::discover_reported_actuals
         -> finance/actualization.py::resolve_current_actual_state

WHERE THE GROUND TRUTH COMES FROM

From the case, never from either layer. Every case is a filing sequence whose
correct reading is decided by the calendar and the documents alone -- which
period ended last, which source could describe it, which figures were in it --
and that reading is written out in full below. A case whose expected answer
was read off an implementation would measure nothing.

The sequences are the generalized SHAPES section 3 names (A-N), not issuers:
what makes a Q4 earnings release beat a Q3 10-Q is that one period ended after
the other and both sources are complete, and no issuer's name appears in that
sentence. Ticker names appear nowhere in this file. There is nothing to name.

WHAT IS SIMPLIFIED, STATED RATHER THAN HIDDEN

  * Quarterly flows are tagged as DISCRETE quarters. Real 10-Qs report
    year-to-date columns for Q2/Q3 and `finance/period_facts.py` reconstructs
    the discrete quarter by differencing them. That reconstruction has its own
    tests; carrying it here would measure it a second time and make every TTM
    expectation depend on it.

  * `fiscal_period` labels use the SEC's own vocabulary (`FY`, `Q1`..`Q3`),
    because that is what `discover_reported_actuals` reads out of a fact and
    therefore what guidance retirement matches against.

  * A case may carry `extra_candidates`: candidates the EXTRACTION layer
    supplies that XBRL discovery cannot, namely a release's forward outlook
    (which is prospective, never an actual) and two sources' reported VALUES
    for one period (which `discover_reported_actuals` does not populate).
    They are passed to the same production resolver; nothing is stubbed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from finance.extraction.schema import (
    ReportedActualCandidate,
    SourceType,
    StatementCompleteness,
)


# ---------------------------------------------------------------------------
# Ground-truth vocabulary (section 4)
# ---------------------------------------------------------------------------

class Truth:
    """Values a ground-truth field may take besides a date or a status name."""

    UNKNOWN = "UNKNOWN"                    # excluded from every denominator
    MISSING = "MISSING"                    # the layer must report no value
    NOT_TRACKED = "NOT_TRACKED_BY_LAYER"   # outside the layer's field set
    NONE_CURRENT = "NONE_CURRENT"          # no period may be published


class DcfBase:
    CURRENT = "CURRENT"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class ConflictStatus:
    NO_CONFLICT = "NO_CONFLICT"
    CONFLICT_RECORDED = "CONFLICT_RECORDED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Currentness:
    CURRENT = "CURRENT"              # may be published as current/latest/TTM
    MUST_QUALIFY = "MUST_QUALIFY"    # real, but may not wear the word


class Hard:
    """Hard-safety names, so a case can declare which one it controls for."""

    WRONG_CURRENT_PERIOD_PUBLISHED = "WRONG_CURRENT_PERIOD_PUBLISHED"
    GUIDANCE_CLASSIFIED_AS_ACTUAL = "GUIDANCE_CLASSIFIED_AS_ACTUAL"
    SILENT_MIXED_PERIOD_STATE = "SILENT_MIXED_PERIOD_STATE"
    STALE_DCF_MARKED_VALID_FOR_RESEARCH = "STALE_DCF_MARKED_VALID_FOR_RESEARCH"
    DUPLICATE_CURRENT_STATE_SAME_PERIOD = "DUPLICATE_CURRENT_STATE_SAME_PERIOD"
    LATEST_QUARTER_METRIC_FROM_OLDER_QUARTER = (
        "LATEST_QUARTER_METRIC_FROM_OLDER_QUARTER")
    CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD = (
        "CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD")

    ALL = (WRONG_CURRENT_PERIOD_PUBLISHED, GUIDANCE_CLASSIFIED_AS_ACTUAL,
           SILENT_MIXED_PERIOD_STATE, STALE_DCF_MARKED_VALID_FOR_RESEARCH,
           DUPLICATE_CURRENT_STATE_SAME_PERIOD,
           LATEST_QUARTER_METRIC_FROM_OLDER_QUARTER,
           CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD)


class FailureClass:
    """Section 20's generalized classes. The only vocabulary a failure gets."""

    ACTUAL_SOURCE_PRECEDENCE = "ACTUAL_SOURCE_PRECEDENCE"
    STATEMENT_COMPLETENESS = "STATEMENT_COMPLETENESS"
    PERIOD_RESOLUTION = "PERIOD_RESOLUTION"
    PER_METRIC_FRESHNESS = "PER_METRIC_FRESHNESS"
    TTM_RECONSTRUCTION = "TTM_RECONSTRUCTION"
    SAME_PERIOD_RECONCILIATION = "SAME_PERIOD_RECONCILIATION"
    ACTUAL_GUIDANCE_SEPARATION = "ACTUAL_GUIDANCE_SEPARATION"
    GUIDANCE_ACTUALIZATION = "GUIDANCE_ACTUALIZATION"
    DCF_BASE_STALE = "DCF_BASE_STALE"
    RESEARCH_FRESHNESS = "RESEARCH_FRESHNESS"
    BENCHMARK_SCORER = "BENCHMARK_SCORER"

    ALL = (ACTUAL_SOURCE_PRECEDENCE, STATEMENT_COMPLETENESS, PERIOD_RESOLUTION,
           PER_METRIC_FRESHNESS, TTM_RECONSTRUCTION, SAME_PERIOD_RECONCILIATION,
           ACTUAL_GUIDANCE_SEPARATION, GUIDANCE_ACTUALIZATION, DCF_BASE_STALE,
           RESEARCH_FRESHNESS, BENCHMARK_SCORER)


# ---------------------------------------------------------------------------
# Building a company-facts payload from a filing sequence
# ---------------------------------------------------------------------------

# One concept per field, chosen from the reviewed map, so a case's numbers
# arrive under a tag the production mapping actually resolves.
US_GAAP_CONCEPTS = {
    "revenue": ("Revenues", False),
    "net_income": ("NetIncomeLoss", False),
    "operating_income": ("OperatingIncomeLoss", False),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities", False),
    "capital_expenditure": ("PaymentsToAcquirePropertyPlantAndEquipment", False),
    "assets": ("Assets", True),
    "liabilities": ("Liabilities", True),
    "cash_and_cash_equivalents": ("CashAndCashEquivalentsAtCarryingValue", True),
    "stockholders_equity": ("StockholdersEquity", True),
    "short_term_debt": ("ShortTermBorrowings", True),
    "long_term_debt": ("LongTermDebtNoncurrent", True),
}

IFRS_CONCEPTS = {
    "revenue": ("Revenue", False),
    "net_income": ("ProfitLoss", False),
    "operating_income": ("ProfitLossFromOperatingActivities", False),
    "operating_cash_flow": ("CashFlowsFromUsedInOperatingActivities", False),
    "capital_expenditure": (
        "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", False),
    "assets": ("Assets", True),
    "liabilities": ("Liabilities", True),
    "cash_and_cash_equivalents": ("CashAndCashEquivalents", True),
    "stockholders_equity": ("Equity", True),
}

# Everything a source needs before it can describe a whole period, plus the
# flows a case cares about. A case names a SUBSET to make a source incomplete.
FULL_SET = ("revenue", "net_income", "operating_income", "operating_cash_flow",
            "capital_expenditure", "assets", "liabilities",
            "cash_and_cash_equivalents", "stockholders_equity",
            "short_term_debt", "long_term_debt")

# A release naming two headline numbers and nothing else.
HEADLINE_SET = ("revenue", "net_income")

# A release with an income statement and a balance sheet but no cash-flow
# statement -- PARTIAL, and the shape that most invites a wholesale replace.
NO_CASH_FLOW_SET = ("revenue", "net_income", "operating_income", "assets",
                    "liabilities", "cash_and_cash_equivalents",
                    "stockholders_equity")

# Complete, but the release did not tag capital expenditure.
NO_CAPEX_SET = tuple(f for f in FULL_SET if f != "capital_expenditure")

# Complete, but neither capital expenditure nor operating income.
NO_CAPEX_NO_OPINC_SET = tuple(
    f for f in FULL_SET if f not in ("capital_expenditure", "operating_income"))

# A classified balance sheet that totals to liabilities-and-equity and never
# tags total liabilities. Ordinary US GAAP presentation, and the shape a live
# run found had made EVERY document an issuer filed look incomplete.
NO_TOTAL_LIABILITIES_SET = tuple(f for f in FULL_SET if f != "liabilities")


@dataclass(frozen=True)
class Filing:
    """One document, and the facts it tagged."""

    form: str
    filed: str
    period_end: str
    fiscal_period: str                     # FY | Q1 | Q2 | Q3
    fiscal_year: int
    accession: str
    period_start: Optional[str] = None     # duration facts' start
    fields: Tuple[str, ...] = FULL_SET
    # Per-field overrides, so two sources can disagree about one period.
    values: Dict[str, float] = field(default_factory=dict)
    # A second duration the same document tags: a Q4 release also states the
    # full year, and so does a 10-K.
    annual_start: Optional[str] = None


def _value(field_name: str, filing: Filing, scale: float = 1.0) -> float:
    """A deterministic number. Values are provenance carriers here, not data.

    Derived from the period end so two sources for one period AGREE unless a
    case deliberately overrides one -- which is what makes an override a
    material disagreement rather than noise.
    """
    if field_name in filing.values:
        return float(filing.values[field_name])
    seed = int(filing.period_end.replace("-", ""))
    base = (seed % 9973) + len(field_name) * 17
    return round(float(base) * scale, 2)


def build_company_facts(filings: Tuple[Filing, ...],
                        taxonomy: str = "us-gaap") -> dict:
    """The SEC company-facts shape both layers read."""
    concepts = IFRS_CONCEPTS if taxonomy == "ifrs-full" else US_GAAP_CONCEPTS
    payload: Dict[str, Dict[str, dict]] = {}

    for filing in filings:
        for name in filing.fields:
            mapping = concepts.get(name)
            if mapping is None:
                continue
            concept, is_instant = mapping
            rows = payload.setdefault(concept, {"units": {"USD": []}})["units"]["USD"]
            stamp = {"form": filing.form, "filed": filing.filed,
                     "accn": filing.accession, "fy": filing.fiscal_year,
                     "fp": filing.fiscal_period}
            if is_instant:
                rows.append(dict(stamp, end=filing.period_end,
                                 val=_value(name, filing)))
                continue
            if filing.period_start:
                rows.append(dict(stamp, start=filing.period_start,
                                 end=filing.period_end, val=_value(name, filing)))
            if filing.annual_start:
                rows.append(dict(stamp, start=filing.annual_start,
                                 end=filing.period_end,
                                 val=_value(name, filing, scale=4.0)))
    return {"facts": {taxonomy: payload}}


# ---------------------------------------------------------------------------
# The fiscal calendar every case shares (fiscal year ends 31 July)
# ---------------------------------------------------------------------------
#
# Deliberately NOT a calendar year. A fiscal year ending in July is where
# "the latest periodic filing" and "the latest reported period" come apart
# most visibly, and a benchmark run entirely on December year-ends would
# never exercise the difference.

QUARTERS = {
    ("FY2025", "Q1"): ("2024-08-01", "2024-10-31"),
    ("FY2025", "Q2"): ("2024-11-01", "2025-01-31"),
    ("FY2025", "Q3"): ("2025-02-01", "2025-04-30"),
    ("FY2025", "Q4"): ("2025-05-01", "2025-07-31"),
    ("FY2026", "Q1"): ("2025-08-01", "2025-10-31"),
    ("FY2026", "Q2"): ("2025-11-01", "2026-01-31"),
    ("FY2026", "Q3"): ("2026-02-01", "2026-04-30"),
    ("FY2026", "Q4"): ("2026-05-01", "2026-07-31"),
}

FY2025_START, FY2025_END = "2024-08-01", "2025-07-31"
FY2026_START, FY2026_END = "2025-08-01", "2026-07-31"

Q1_FY26_END = "2025-10-31"
Q2_FY26_END = "2026-01-31"
Q3_FY26_END = "2026-04-30"
Q4_FY26_END = "2026-07-31"

# Every metric a case states expectations about.
SCORED_METRICS = ("revenue", "net_income", "operating_income",
                  "cash_and_cash_equivalents", "total_debt",
                  "operating_cash_flow", "capital_expenditure")

# `total_debt` is an AGGREGATE the freshness planner derives from balance-sheet
# components; it is not one of the fields `discover_reported_actuals` walks, so
# V2 keeps no freshness record for it. Named here rather than quietly dropped:
# a coverage boundary reported is a boundary, a coverage boundary omitted from
# the schema is a flattering denominator.
V2_UNTRACKED_METRICS = ("total_debt",)


def _q(year: str, quarter: str) -> Tuple[str, str]:
    return QUARTERS[(year, quarter)]


def _quarterly(year: str, quarter: str, filed: str, fiscal_year: int,
               fields: Tuple[str, ...] = FULL_SET,
               form: str = "10-Q", values=None) -> Filing:
    start, end = _q(year, quarter)
    return Filing(form=form, filed=filed, period_end=end, period_start=start,
                  fiscal_period=quarter, fiscal_year=fiscal_year,
                  accession=f"{form}-{end}", fields=fields, values=values or {})


# -- the shared history every case starts from ------------------------------
#
# FY2025 closed and filed, then FY2026 Q1..Q3 filed as 10-Qs. Seven filings,
# which is enough for a twelve-month window to be built at either of the last
# two period ends.

FY2025_10K = Filing(
    form="10-K", filed="2025-09-19", period_end=FY2025_END,
    period_start=_q("FY2025", "Q4")[0], annual_start=FY2025_START,
    fiscal_period="FY", fiscal_year=2025, accession="10-K-2025")

FY26_Q1 = _quarterly("FY2026", "Q1", "2025-12-04", 2026)
FY26_Q2 = _quarterly("FY2026", "Q2", "2026-03-05", 2026)
FY26_Q3 = _quarterly("FY2026", "Q3", "2026-06-05", 2026)

HISTORY_THROUGH_Q3: Tuple[Filing, ...] = (
    _quarterly("FY2025", "Q1", "2024-12-05", 2025),
    _quarterly("FY2025", "Q2", "2025-03-06", 2025),
    _quarterly("FY2025", "Q3", "2025-06-05", 2025),
    FY2025_10K, FY26_Q1, FY26_Q2, FY26_Q3,
)

HISTORY_THROUGH_Q2: Tuple[Filing, ...] = HISTORY_THROUGH_Q3[:-1]


def _q4_release(fields: Tuple[str, ...] = FULL_SET, values=None,
                filed: str = "2026-09-02", form: str = "8-K") -> Filing:
    """The fourth-quarter/full-year earnings release, filed before the 10-K."""
    start, end = _q("FY2026", "Q4")
    return Filing(form=form, filed=filed, period_end=end, period_start=start,
                  annual_start=FY2026_START, fiscal_period="FY",
                  fiscal_year=2026, accession=f"{form}-{filed}", fields=fields,
                  values=values or {})


def _fy_10k(values=None, filed: str = "2026-10-15") -> Filing:
    start, end = _q("FY2026", "Q4")
    return Filing(form="10-K", filed=filed, period_end=end, period_start=start,
                  annual_start=FY2026_START, fiscal_period="FY",
                  fiscal_year=2026, accession="10-K-2026", fields=FULL_SET,
                  values=values or {})


# ---------------------------------------------------------------------------
# Guidance stand-ins (sections 14/15)
# ---------------------------------------------------------------------------

class GuidanceStatement:
    """The three attributes `retire_realized_guidance` reads, and no others.

    A real `GuidanceMetric` is frozen and carries twenty fields; constructing
    one here would make the case about the domain object rather than about
    whether a completed period retires its own outlook.
    """

    def __init__(self, name: str, fiscal_period: str, status: str = "CURRENT"):
        self.name = name
        self.fiscal_period = fiscal_period
        self.status = status
        self.status_reason = ""

    def with_status(self, status, reason):
        clone = GuidanceStatement(self.name, self.fiscal_period, status)
        clone.status_reason = reason
        return clone

    def __repr__(self):
        return f"Guidance({self.name}@{self.fiscal_period}:{self.status})"


# ---------------------------------------------------------------------------
# A case
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActualizationCase:
    """One filing sequence and everything true of it. Section 4's schema."""

    case_id: str
    case_class: str                        # the section 3 letter
    description: str
    filings: Tuple[Filing, ...]
    as_of: str
    taxonomy: str = "us-gaap"

    extra_candidates: Tuple[ReportedActualCandidate, ...] = ()
    prior_state_metrics: Dict[str, dict] = field(default_factory=dict)
    guidance: Tuple[GuidanceStatement, ...] = ()
    dcf_base_period_end: Optional[str] = None

    # -- ground truth ------------------------------------------------------
    expected_current_period: str = Truth.UNKNOWN
    expected_primary_source: str = Truth.UNKNOWN          # a form name
    expected_actual_status: str = Truth.UNKNOWN
    expected_statement_completeness: str = Truth.UNKNOWN
    expected_current_metrics: Dict[str, str] = field(default_factory=dict)
    expected_fallback_metrics: Dict[str, str] = field(default_factory=dict)
    expected_ttm_end_period: str = Truth.UNKNOWN
    expected_ttm_status: str = Truth.UNKNOWN
    expected_ttm_window: Tuple[str, ...] = ()             # quarter end dates
    expected_ttm_stale_metrics: Tuple[str, ...] = ()
    expected_completed_guidance_status: str = Truth.UNKNOWN
    expected_current_guidance_periods: Tuple[str, ...] = ()
    expected_same_period_conflict_status: str = ConflictStatus.NOT_APPLICABLE
    expected_dcf_base_status: str = Truth.UNKNOWN
    expected_research_currentness: Dict[str, str] = field(default_factory=dict)
    forbidden_actual_periods: Tuple[str, ...] = ()
    positive_controls: Tuple[str, ...] = ()
    expect_v1_v2_agreement: Optional[bool] = None
    notes: str = ""

    def company_facts(self) -> dict:
        return build_company_facts(self.filings, taxonomy=self.taxonomy)


# ---------------------------------------------------------------------------
# Shared expectation blocks
# ---------------------------------------------------------------------------

def _all_from(period_end: str) -> Dict[str, str]:
    """Every scored metric current, from one period."""
    return {name: (Truth.NOT_TRACKED if name in V2_UNTRACKED_METRICS
                   else period_end)
            for name in SCORED_METRICS}


def _all_current(period_end: str) -> Dict[str, str]:
    return {name: Currentness.CURRENT for name in
            ("revenue", "net_income", "operating_income",
             "operating_cash_flow", "capital_expenditure")}


FY26_WINDOW = (Q1_FY26_END, Q2_FY26_END, Q3_FY26_END, Q4_FY26_END)
THROUGH_Q3_WINDOW = ("2025-07-31", Q1_FY26_END, Q2_FY26_END, Q3_FY26_END)


def _prior_state(metrics: Tuple[str, ...], period_end: str,
                 form: str = "10-Q") -> Dict[str, dict]:
    """The state being replaced, in the shape `build_metric_freshness` reads."""
    return {name: {"period_end": period_end, "form": form,
                   "issued_at": "2026-06-05",
                   "source_type": "PERIODIC_INTERIM",
                   "finality": "AUDITED"}
            for name in metrics}


# ---------------------------------------------------------------------------
# The cases (section 3, classes A-N)
# ---------------------------------------------------------------------------
#
# Read each `expected_*` field as a claim about the DOCUMENTS, not about any
# implementation: "the fourth quarter ended 31 July, the release describing it
# was filed on 2 September and carries all three statements, so 31 July is the
# latest reported period and the release is its source." Everything below was
# written from the calendar above.


def _outlook_candidate(period_end: str, fiscal_period: str,
                       issued_at: str) -> ReportedActualCandidate:
    """The forward half of an earnings release, as the extraction layer offers it.

    Complete, newer than everything, and PROSPECTIVE. A resolver that ranks on
    period end alone takes it; that is precisely the failure section 15 names.
    """
    return ReportedActualCandidate(
        period_end=period_end, period_start="2026-08-01",
        fiscal_period=fiscal_period, issued_at=issued_at, form="8-K",
        accession="8-K-outlook", source_type=SourceType.PRELIMINARY_EARNINGS_RELEASE,
        statement_completeness=StatementCompleteness.COMPLETE,
        present_metrics=FULL_SET, is_prospective=True)


def _valued_candidate(form: str, issued_at: str, values: Dict[str, float],
                      period_end: str = Q4_FY26_END) -> ReportedActualCandidate:
    """One source's reported figures for a period, as extraction supplies them."""
    return ReportedActualCandidate(
        period_end=period_end, period_start=_q("FY2026", "Q4")[0],
        fiscal_period="FY", issued_at=issued_at, form=form,
        accession=f"{form}-valued", statement_completeness=StatementCompleteness.COMPLETE,
        source_type=(SourceType.FORMAL_PERIODIC_FILING if form != "8-K"
                     else SourceType.PRELIMINARY_EARNINGS_RELEASE),
        present_metrics=FULL_SET, values=dict(values))


_FLOW_METRICS = ("revenue", "net_income", "operating_income",
                 "operating_cash_flow", "capital_expenditure")


def _currentness(qualify: Tuple[str, ...] = (),
                 include_cash: bool = True) -> Dict[str, str]:
    view = {name: (Currentness.MUST_QUALIFY if name in qualify
                   else Currentness.CURRENT)
            for name in _FLOW_METRICS}
    if include_cash:
        view["cash_and_cash_equivalents"] = (
            Currentness.MUST_QUALIFY if "cash_and_cash_equivalents" in qualify
            else Currentness.CURRENT)
    return view


ALL_CASES: Tuple[ActualizationCase, ...] = (

    # -- A ------------------------------------------------------------------
    # The fourth quarter ended 31 July. The release describing it was filed on
    # 2 September and carries an income statement, a balance sheet and a cash
    # flow statement. The 10-K has not been filed. 31 July is the latest
    # reported period; the release is its source, and the figures are
    # preliminary because no periodic filing has carried them yet.
    ActualizationCase(
        case_id="A-complete-release-past-10q", case_class="A",
        description="Complete Q4/FY earnings release, newer than the last 10-Q.",
        filings=HISTORY_THROUGH_Q3 + (_q4_release(),),
        as_of="2026-09-15",
        dcf_base_period_end=Q4_FY26_END,
        expected_current_period=Q4_FY26_END,
        expected_primary_source="8-K",
        expected_actual_status="PRELIMINARY_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics=_all_from(Q4_FY26_END),
        expected_fallback_metrics={},
        expected_ttm_end_period=Q4_FY26_END,
        expected_ttm_status="COMPLETE",
        expected_ttm_window=FY26_WINDOW,
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(),
        positive_controls=(Hard.WRONG_CURRENT_PERIOD_PUBLISHED,),
        notes="The period must advance. A newer DOCUMENT DATE alone would not "
              "be enough -- the release also has to be able to describe the "
              "whole period, and here it can."),

    # -- B ------------------------------------------------------------------
    # The same fourth quarter, announced in a release naming revenue and net
    # income and nothing else. It cannot supply a balance sheet or a cash flow
    # statement, so it cannot carry the state. The last period any source can
    # fully describe is still 30 April.
    ActualizationCase(
        case_id="B-headline-release-does-not-replace", case_class="B",
        description="Headline-only Q4 release against a complete Q3 10-Q.",
        filings=HISTORY_THROUGH_Q3 + (_q4_release(fields=HEADLINE_SET),),
        as_of="2026-09-15",
        dcf_base_period_end=Q3_FY26_END,
        expected_current_period=Q3_FY26_END,
        expected_primary_source="10-Q",
        expected_actual_status="FINAL_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics=_all_from(Q3_FY26_END),
        expected_fallback_metrics={},
        expected_ttm_end_period=Q3_FY26_END,
        expected_ttm_status="LIMITED",
        expected_ttm_stale_metrics=("net_income", "revenue"),
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(
            qualify=("revenue", "net_income")),
        positive_controls=(Hard.SILENT_MIXED_PERIOD_STATE,),
        notes="Revenue and net income now have twelve-month windows ending "
              "31 July while the state ends 30 April. Two periods in one "
              "state: permitted only if every part of it is labelled."),

    # -- B2 -----------------------------------------------------------------
    # A fuller release -- income statement and balance sheet -- but no cash
    # flow statement. Still not a whole period.
    ActualizationCase(
        case_id="B2-partial-release-no-cash-flow", case_class="B",
        description="Q4 release with two statements out of three.",
        filings=HISTORY_THROUGH_Q3 + (_q4_release(fields=NO_CASH_FLOW_SET),),
        as_of="2026-09-15",
        dcf_base_period_end=Q3_FY26_END,
        expected_current_period=Q3_FY26_END,
        expected_primary_source="10-Q",
        expected_actual_status="FINAL_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics=_all_from(Q3_FY26_END),
        expected_ttm_end_period=Q3_FY26_END,
        expected_ttm_status="LIMITED",
        expected_ttm_stale_metrics=("net_income", "operating_income", "revenue"),
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(
            qualify=("revenue", "net_income", "operating_income")),
        notes="A balance sheet without a cash flow statement is still a "
              "source that cannot answer what it claims."),

    # -- C ------------------------------------------------------------------
    # The release and the 10-K describe ONE economic period. The 10-K is the
    # authority; the release stays in the provenance. There is no second
    # current period, and the figures agree.
    ActualizationCase(
        case_id="C-release-then-10k-same-period", case_class="C",
        description="Same-period earnings release followed by the 10-K.",
        filings=HISTORY_THROUGH_Q3 + (_q4_release(), _fy_10k()),
        as_of="2026-11-01",
        dcf_base_period_end=Q4_FY26_END,
        expected_current_period=Q4_FY26_END,
        expected_primary_source="10-K",
        expected_actual_status="FINAL_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics=_all_from(Q4_FY26_END),
        expected_ttm_end_period=Q4_FY26_END,
        expected_ttm_status="COMPLETE",
        expected_ttm_window=FY26_WINDOW,
        expected_same_period_conflict_status=ConflictStatus.NO_CONFLICT,
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(),
        positive_controls=(Hard.DUPLICATE_CURRENT_STATE_SAME_PERIOD,),
        notes="Two sources, one period. A layer that made two current states "
              "would double-count the year."),

    # -- D ------------------------------------------------------------------
    ActualizationCase(
        case_id="D-normal-q3-progression", case_class="D",
        description="Ordinary Q1/Q2/Q3 10-Q progression, nothing newer filed.",
        filings=HISTORY_THROUGH_Q3, as_of="2026-06-20",
        dcf_base_period_end=Q3_FY26_END,
        expected_current_period=Q3_FY26_END,
        expected_primary_source="10-Q",
        expected_actual_status="FINAL_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics=_all_from(Q3_FY26_END),
        expected_ttm_end_period=Q3_FY26_END,
        expected_ttm_status="COMPLETE",
        expected_ttm_window=THROUGH_Q3_WINDOW,
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(),
        notes="The common case. It must not be disturbed by anything the "
              "harder cases need."),

    # -- D2 -----------------------------------------------------------------
    ActualizationCase(
        case_id="D2-normal-q2-progression", case_class="D",
        description="The same progression one quarter earlier.",
        filings=HISTORY_THROUGH_Q2, as_of="2026-03-20",
        dcf_base_period_end=Q2_FY26_END,
        expected_current_period=Q2_FY26_END,
        expected_primary_source="10-Q",
        expected_actual_status="FINAL_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics=_all_from(Q2_FY26_END),
        expected_ttm_end_period=Q2_FY26_END,
        expected_ttm_status="COMPLETE",
        expected_ttm_window=("2025-04-30", "2025-07-31", Q1_FY26_END, Q2_FY26_END),
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(),
        notes="Proves the window MOVES with the period rather than always "
              "landing on the newest fact in the payload."),

    # -- E ------------------------------------------------------------------
    # The full year is guided, then reported. The guidance for it is no longer
    # an outlook. The next year's guidance still is.
    ActualizationCase(
        case_id="E-fy-actual-retires-fy-guidance", case_class="E",
        description="FY results arrive against FY guidance and next-year guidance.",
        filings=HISTORY_THROUGH_Q3 + (_q4_release(), _fy_10k()),
        as_of="2026-11-01",
        guidance=(GuidanceStatement("revenue", "FY"),
                  GuidanceStatement("revenue", "Q1 FY2027"),
                  GuidanceStatement("earnings_per_share", "FY2027")),
        dcf_base_period_end=Q4_FY26_END,
        expected_current_period=Q4_FY26_END,
        expected_primary_source="10-K",
        expected_actual_status="FINAL_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics=_all_from(Q4_FY26_END),
        expected_ttm_end_period=Q4_FY26_END,
        expected_ttm_status="COMPLETE",
        expected_ttm_window=FY26_WINDOW,
        expected_completed_guidance_status="REALIZED",
        expected_current_guidance_periods=("FY2027", "Q1 FY2027"),
        expected_same_period_conflict_status=ConflictStatus.NO_CONFLICT,
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(),
        notes="REALIZED, not EXPIRED: the number is in, which is a different "
              "fact from the calendar having moved past the period."),

    # -- E2 -----------------------------------------------------------------
    ActualizationCase(
        case_id="E2-guidance-survives-without-actuals", case_class="E",
        description="The guided year has not been reported; the outlook stands.",
        filings=HISTORY_THROUGH_Q3, as_of="2026-06-20",
        guidance=(GuidanceStatement("revenue", "FY"),
                  GuidanceStatement("revenue", "Q1 FY2027")),
        dcf_base_period_end=Q3_FY26_END,
        expected_current_period=Q3_FY26_END,
        expected_primary_source="10-Q",
        expected_actual_status="FINAL_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics=_all_from(Q3_FY26_END),
        expected_ttm_end_period=Q3_FY26_END,
        expected_ttm_status="COMPLETE",
        expected_ttm_window=THROUGH_Q3_WINDOW,
        expected_completed_guidance_status="CURRENT",
        expected_current_guidance_periods=("FY", "Q1 FY2027"),
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(),
        notes="The negative control for retirement. A layer that retired "
              "everything would score perfectly on E alone."),

    # -- F ------------------------------------------------------------------
    ActualizationCase(
        case_id="F-exact-ttm-roll-forward", case_class="F",
        description="The 10-K arrives; every window rolls forward one quarter.",
        filings=HISTORY_THROUGH_Q3 + (_fy_10k(),),
        as_of="2026-11-01",
        dcf_base_period_end=Q4_FY26_END,
        expected_current_period=Q4_FY26_END,
        expected_primary_source="10-K",
        expected_actual_status="FINAL_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics=_all_from(Q4_FY26_END),
        expected_ttm_end_period=Q4_FY26_END,
        expected_ttm_status="COMPLETE",
        expected_ttm_window=FY26_WINDOW,
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(),
        notes="Exactly four quarters, contiguous, the prior-year quarter "
              "dropped and the new one added. No quarter twice, none missing, "
              "and the annual figure never summed alongside them."),

    # -- G ------------------------------------------------------------------
    # The release carries the quarter for everything except net income, which
    # it does not tag. The window for it cannot follow.
    ActualizationCase(
        case_id="G-incomplete-ttm-history", case_class="G",
        description="A new quarter that one metric's history cannot reach.",
        filings=HISTORY_THROUGH_Q3 + (
            _q4_release(fields=tuple(f for f in FULL_SET if f != "net_income")),),
        as_of="2026-09-15",
        prior_state_metrics=_prior_state(("net_income",), Q3_FY26_END),
        dcf_base_period_end=Q4_FY26_END,
        expected_current_period=Q4_FY26_END,
        expected_primary_source="8-K",
        expected_actual_status="PRELIMINARY_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics={
            **_all_from(Q4_FY26_END), "net_income": Q3_FY26_END},
        expected_fallback_metrics={"net_income": Q3_FY26_END},
        expected_ttm_end_period=Q4_FY26_END,
        expected_ttm_status="LIMITED",
        expected_ttm_window=FY26_WINDOW,
        expected_ttm_stale_metrics=("net_income",),
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(qualify=("net_income",)),
        positive_controls=(Hard.CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD,),
        notes="An incomplete window must be LIMITED, never silently filled "
              "and never quietly reported as current."),

    # -- H ------------------------------------------------------------------
    # The release said revenue was 1,000 and the 10-K says 1,400. Same period,
    # two sources, a 40% difference: a different claim about the same quarter,
    # not a rounding move.
    ActualizationCase(
        case_id="H-same-period-conflict-from-facts", case_class="H",
        description="Release and 10-K disagree materially about one period.",
        filings=HISTORY_THROUGH_Q3 + (
            _q4_release(values={"revenue": 1000.0}),
            _fy_10k(values={"revenue": 1400.0})),
        as_of="2026-11-01",
        dcf_base_period_end=Q4_FY26_END,
        expected_current_period=Q4_FY26_END,
        expected_primary_source="10-K",
        expected_actual_status="FINAL_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics=_all_from(Q4_FY26_END),
        expected_ttm_end_period=Q4_FY26_END,
        expected_ttm_status="COMPLETE",
        expected_same_period_conflict_status=ConflictStatus.CONFLICT_RECORDED,
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(),
        notes="The 10-K is the authority; the disagreement is recorded rather "
              "than overwritten. Reached through XBRL discovery, which is how "
              "production reaches it."),

    # -- H2 -----------------------------------------------------------------
    # The same disagreement, offered as the extraction layer offers it: two
    # candidates for one period, each carrying its own reported figures.
    ActualizationCase(
        case_id="H2-same-period-conflict-with-values", case_class="H",
        description="The same conflict, with both sources' figures supplied.",
        filings=HISTORY_THROUGH_Q3, as_of="2026-11-01",
        extra_candidates=(
            _valued_candidate("8-K", "2026-09-02", {"revenue": 1000.0,
                                                    "net_income": 90.0}),
            _valued_candidate("10-K", "2026-10-15", {"revenue": 1400.0,
                                                     "net_income": 90.0})),
        dcf_base_period_end=Q4_FY26_END,
        expected_current_period=Q4_FY26_END,
        expected_primary_source="10-K",
        expected_actual_status="FINAL_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics=_all_from(Q4_FY26_END),
        expected_ttm_end_period=Q4_FY26_END,
        expected_ttm_status="LIMITED",
        expected_ttm_stale_metrics=_FLOW_METRICS,
        expected_same_period_conflict_status=ConflictStatus.CONFLICT_RECORDED,
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(qualify=_FLOW_METRICS,
                                                   include_cash=False),
        notes="The positive control for conflict detection: net income agrees "
              "and must NOT be reported as a conflict, revenue differs by 40% "
              "and must be. The fourth quarter reaches this case only as "
              "extraction candidates -- the XBRL payload stops at Q3 -- so no "
              "twelve-month window can follow the period, and LIMITED is the "
              "only honest status."),

    # -- I ------------------------------------------------------------------
    ActualizationCase(
        case_id="I-latest-quarter-metric-provenance", case_class="I",
        description="The current period cannot supply one metric.",
        filings=HISTORY_THROUGH_Q3 + (_q4_release(fields=NO_CAPEX_SET),),
        as_of="2026-09-15",
        prior_state_metrics=_prior_state(("capital_expenditure",), Q3_FY26_END),
        dcf_base_period_end=Q4_FY26_END,
        expected_current_period=Q4_FY26_END,
        expected_primary_source="8-K",
        expected_actual_status="PRELIMINARY_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics={
            **_all_from(Q4_FY26_END), "capital_expenditure": Q3_FY26_END},
        expected_fallback_metrics={"capital_expenditure": Q3_FY26_END},
        expected_ttm_end_period=Q4_FY26_END,
        expected_ttm_status="LIMITED",
        expected_ttm_window=FY26_WINDOW,
        expected_ttm_stale_metrics=("capital_expenditure",),
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(
            qualify=("capital_expenditure",)),
        positive_controls=(Hard.LATEST_QUARTER_METRIC_FROM_OLDER_QUARTER,),
        notes="The April figure may be RETAINED. It may not be presented as "
              "the latest quarter's, and the label is what decides which."),

    # -- J ------------------------------------------------------------------
    ActualizationCase(
        case_id="J-actual-and-outlook-in-one-release", case_class="J",
        description="One release carrying reported results AND next year's outlook.",
        filings=HISTORY_THROUGH_Q3 + (_q4_release(),),
        as_of="2026-09-15",
        extra_candidates=(_outlook_candidate("2027-07-31", "FY2027", "2026-09-02"),),
        guidance=(GuidanceStatement("revenue", "FY2027"),),
        dcf_base_period_end=Q4_FY26_END,
        expected_current_period=Q4_FY26_END,
        expected_primary_source="8-K",
        expected_actual_status="PRELIMINARY_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics=_all_from(Q4_FY26_END),
        expected_ttm_end_period=Q4_FY26_END,
        expected_ttm_status="COMPLETE",
        expected_ttm_window=FY26_WINDOW,
        expected_completed_guidance_status="CURRENT",
        expected_current_guidance_periods=("FY2027",),
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(),
        forbidden_actual_periods=("2027-07-31",),
        positive_controls=(Hard.GUIDANCE_CLASSIFIED_AS_ACTUAL,),
        notes="The outlook is complete, newer than everything, and must never "
              "become the current reported period. A resolver ranking on "
              "period end alone takes it."),

    # -- K ------------------------------------------------------------------
    # A foreign private issuer: 6-K interims and a 20-F annual, reported under
    # IFRS. The fiscal year ended 31 July and the 20-F describing it was filed
    # in November, so 31 July is the latest reported period.
    ActualizationCase(
        case_id="K-foreign-private-issuer", case_class="K",
        description="20-F annual and 6-K interims under IFRS.",
        filings=(
            Filing("6-K", "2024-12-05", "2024-10-31", "Q1", 2025, "6-K-1",
                   period_start="2024-08-01", fields=tuple(IFRS_CONCEPTS)),
            Filing("6-K", "2025-03-06", "2025-01-31", "Q2", 2025, "6-K-2",
                   period_start="2024-11-01", fields=tuple(IFRS_CONCEPTS)),
            Filing("6-K", "2025-06-05", "2025-04-30", "Q3", 2025, "6-K-3",
                   period_start="2025-02-01", fields=tuple(IFRS_CONCEPTS)),
            Filing("20-F", "2025-11-20", "2025-07-31", "FY", 2025, "20-F-2025",
                   period_start="2025-05-01", annual_start="2024-08-01",
                   fields=tuple(IFRS_CONCEPTS)),
            Filing("6-K", "2025-12-04", "2025-10-31", "Q1", 2026, "6-K-4",
                   period_start="2025-08-01", fields=tuple(IFRS_CONCEPTS)),
            Filing("6-K", "2026-03-05", "2026-01-31", "Q2", 2026, "6-K-5",
                   period_start="2025-11-01", fields=tuple(IFRS_CONCEPTS)),
            Filing("6-K", "2026-06-05", "2026-04-30", "Q3", 2026, "6-K-6",
                   period_start="2026-02-01", fields=tuple(IFRS_CONCEPTS)),
            Filing("20-F", "2026-11-20", "2026-07-31", "FY", 2026, "20-F-2026",
                   period_start="2026-05-01", annual_start="2025-08-01",
                   fields=tuple(IFRS_CONCEPTS)),
        ),
        as_of="2026-12-01", taxonomy="ifrs-full",
        dcf_base_period_end=Q4_FY26_END,
        expected_current_period=Q4_FY26_END,
        expected_primary_source="20-F",
        expected_actual_status="FINAL_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics=_all_from(Q4_FY26_END),
        expected_ttm_end_period=Q4_FY26_END,
        expected_ttm_status="COMPLETE",
        expected_ttm_window=FY26_WINDOW,
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(),
        notes="The same shapes in another reporting framework. A layer that "
              "reads only one taxonomy sees an issuer with no financials."),

    # -- L ------------------------------------------------------------------
    ActualizationCase(
        case_id="L-annual-only-filer-both-agree", case_class="L",
        description="A filer with nothing but 10-Ks. Both layers must agree.",
        filings=(FY2025_10K, _fy_10k()), as_of="2026-11-01",
        dcf_base_period_end=Q4_FY26_END,
        expected_current_period=Q4_FY26_END,
        expected_primary_source="10-K",
        expected_actual_status="FINAL_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics=_all_from(Q4_FY26_END),
        expected_ttm_end_period=Q4_FY26_END,
        expected_ttm_status="LIMITED",
        expected_ttm_stale_metrics=_FLOW_METRICS,
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(
            qualify=_FLOW_METRICS),
        expect_v1_v2_agreement=True,
        notes="No quarterly history, so no twelve-month window can be built "
              "from quarters. LIMITED is the correct answer; a number would "
              "be an invented one."),

    # -- M ------------------------------------------------------------------
    ActualizationCase(
        case_id="M-stale-dcf-base", case_class="M",
        description="The company has reported Q4; the DCF was built on Q3.",
        filings=HISTORY_THROUGH_Q3 + (_q4_release(),),
        as_of="2026-09-15",
        dcf_base_period_end=Q3_FY26_END,
        expected_current_period=Q4_FY26_END,
        expected_primary_source="8-K",
        expected_actual_status="PRELIMINARY_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics=_all_from(Q4_FY26_END),
        expected_ttm_end_period=Q4_FY26_END,
        expected_ttm_status="COMPLETE",
        expected_ttm_window=FY26_WINDOW,
        expected_dcf_base_status=DcfBase.STALE,
        expected_research_currentness=_currentness(),
        positive_controls=(Hard.STALE_DCF_MARKED_VALID_FOR_RESEARCH,),
        notes="Ninety-two days between the base and the reported period: a "
              "whole quarter, not a calendar wobble. The model describes a "
              "period that is over and may supply no valuation evidence."),

    # -- N ------------------------------------------------------------------
    ActualizationCase(
        case_id="N-mixed-period-fallback-labelled", case_class="N",
        description="Two metrics retained from the prior quarter, deliberately.",
        filings=HISTORY_THROUGH_Q3 + (
            _q4_release(fields=NO_CAPEX_NO_OPINC_SET),),
        as_of="2026-09-15",
        prior_state_metrics=_prior_state(
            ("capital_expenditure", "operating_income"), Q3_FY26_END),
        dcf_base_period_end=Q4_FY26_END,
        expected_current_period=Q4_FY26_END,
        expected_primary_source="8-K",
        expected_actual_status="PRELIMINARY_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics={
            **_all_from(Q4_FY26_END),
            "capital_expenditure": Q3_FY26_END,
            "operating_income": Q3_FY26_END},
        expected_fallback_metrics={"capital_expenditure": Q3_FY26_END,
                                   "operating_income": Q3_FY26_END},
        expected_ttm_end_period=Q4_FY26_END,
        expected_ttm_status="LIMITED",
        expected_ttm_window=FY26_WINDOW,
        expected_ttm_stale_metrics=("capital_expenditure", "operating_income"),
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(
            qualify=("capital_expenditure", "operating_income")),
        positive_controls=(Hard.SILENT_MIXED_PERIOD_STATE,),
        notes="A mixed-period state is allowed. A mixed-period state nobody "
              "can see is the failure."),

    # -- D3 -----------------------------------------------------------------
    # Found by the live set, kept here so it stays found. An issuer whose
    # balance sheet totals to liabilities-and-equity never tags total
    # liabilities; requiring that line classified every document it had ever
    # filed as PARTIAL, and the layer resolved no period for the company at
    # all while its 10-K sat on file.
    ActualizationCase(
        case_id="D3-no-total-liabilities-line", case_class="D",
        description="A complete filer that never tags total liabilities.",
        filings=tuple(
            Filing(f.form, f.filed, f.period_end, f.fiscal_period, f.fiscal_year,
                   f.accession, period_start=f.period_start,
                   fields=tuple(x for x in f.fields if x != "liabilities"),
                   values=dict(f.values), annual_start=f.annual_start)
            for f in HISTORY_THROUGH_Q3),
        as_of="2026-06-20",
        dcf_base_period_end=Q3_FY26_END,
        expected_current_period=Q3_FY26_END,
        expected_primary_source="10-Q",
        expected_actual_status="FINAL_REPORTED_ACTUAL",
        expected_statement_completeness="COMPLETE",
        expected_current_metrics=_all_from(Q3_FY26_END),
        expected_ttm_end_period=Q3_FY26_END,
        expected_ttm_status="COMPLETE",
        expected_ttm_window=THROUGH_Q3_WINDOW,
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_research_currentness=_currentness(),
        notes="Total liabilities is an optional presentation line. Assets and "
              "stockholders' equity are the anchors the rest of the project "
              "already uses to decide a balance sheet was filed."),
)


def cases() -> List[ActualizationCase]:
    return list(ALL_CASES)


def classes_covered() -> List[str]:
    return sorted({case.case_class for case in ALL_CASES})


def controls_covered() -> Dict[str, List[str]]:
    """Hard-safety name -> the cases that are its positive control."""
    found: Dict[str, List[str]] = {name: [] for name in Hard.ALL}
    for case in ALL_CASES:
        for name in case.positive_controls:
            found[name].append(case.case_id)
    return found

