"""The Canonical Actual Fact Integration benchmark: cases and independent truth.

WHAT A CASE IS

A CompanyFacts payload (Q1-Q3 and the prior year, in millions) paired with a
filed earnings-release document. The harness runs the REAL pipeline over the
pair -- reported-actuals discovery, the Actualization resolver, the unified
fact set, a rebuild of `CurrentFinancialState`, canonical evidence, the
twelve-month reconstruction and the DCF base assessment -- exactly as
`finance/workflow.py` does under `FINANCE_ACTUALIZATION_MODE=v2` and
`FINANCE_REPORTED_ACTUALS_MODE=v2`.

WHERE THE GROUND TRUTH COMES FROM

From the CALENDAR and the DOCUMENTS. The CompanyFacts quarters carry the
values written into `_Q` below; the release carries the values in
`tests/fixtures/reported_actuals_releases.py`, generated from the same
constants a correct reader must return. Every `expected_*` field is derived
from those two by hand -- which quarter ended last, which source could
describe it, what the four-quarter sum is -- and written before any code runs.
Nothing is read off the pipeline's output.

Ticker names appear nowhere. A case is a set of quarterly figures and a
release document.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from tests.fixtures import reported_actuals_releases as R


MILLIONS = R.MILLIONS

# The fiscal calendar, shared with the Actualization benchmark: FY ends 31 July.
FY2025 = {
    "Q1": ("2024-08-01", "2024-10-31"),
    "Q2": ("2024-11-01", "2025-01-31"),
    "Q3": ("2025-02-01", "2025-04-30"),
    "Q4": ("2025-05-01", "2025-07-31"),
}
FY2026 = {
    "Q1": ("2025-08-01", "2025-10-31"),
    "Q2": ("2025-11-01", "2026-01-31"),
    "Q3": ("2026-02-01", "2026-04-30"),
    "Q4": ("2026-05-01", "2026-07-31"),
}
FY2025_END = "2025-07-31"
FY2026_END = "2026-07-31"
Q3_FY26_END = "2026-04-30"
Q4_FY26_END = "2026-07-31"

# Discrete quarterly flow figures the CompanyFacts payload reports, in
# millions. Q4 FY2026 is NOT here -- it is what the earnings release supplies.
_FLOW_QUARTERS = {
    ("FY2025", "Q1"): {"revenue": 980.0, "net_income": 150.0, "operating_income": 190.0,
                       "operating_cash_flow": 230.0, "capital_expenditure": 48.0},
    ("FY2025", "Q2"): {"revenue": 1010.0, "net_income": 160.0, "operating_income": 200.0,
                       "operating_cash_flow": 240.0, "capital_expenditure": 50.0},
    ("FY2025", "Q3"): {"revenue": 1050.0, "net_income": 168.0, "operating_income": 210.0,
                       "operating_cash_flow": 250.0, "capital_expenditure": 52.0},
    ("FY2025", "Q4"): {"revenue": 1120.0, "net_income": 180.0, "operating_income": 226.0,
                       "operating_cash_flow": 268.0, "capital_expenditure": 55.0},
    ("FY2026", "Q1"): {"revenue": 1180.0, "net_income": 190.0, "operating_income": 240.0,
                       "operating_cash_flow": 286.0, "capital_expenditure": 58.0},
    ("FY2026", "Q2"): {"revenue": 1260.0, "net_income": 202.0, "operating_income": 256.0,
                       "operating_cash_flow": 300.0, "capital_expenditure": 62.0},
    ("FY2026", "Q3"): {"revenue": 1340.0, "net_income": 214.0, "operating_income": 272.0,
                       "operating_cash_flow": 318.0, "capital_expenditure": 66.0},
}

# The balance sheet CompanyFacts reports at each quarter end (instants).
_BALANCE = {
    Q3_FY26_END: {"cash_and_cash_equivalents": 690.0, "short_term_investments": 160.0,
                  "assets": 5710.0, "current_assets": 1880.0, "liabilities": 3190.0,
                  "current_liabilities": 1010.0, "stockholders_equity": 2520.0,
                  "short_term_debt": 120.0, "long_term_debt": 1480.0},
    FY2025_END: {"cash_and_cash_equivalents": 640.0, "short_term_investments": 150.0,
                 "assets": 5480.0, "current_assets": 1790.0, "liabilities": 3120.0,
                 "current_liabilities": 980.0, "stockholders_equity": 2360.0,
                 "short_term_debt": 110.0, "long_term_debt": 1520.0},
}

# TTM through the Q4 release, in millions: Q1+Q2+Q3 (CompanyFacts) + Q4 (release).
TTM_THROUGH_Q4 = {
    metric: (_FLOW_QUARTERS[("FY2026", "Q1")][metric]
             + _FLOW_QUARTERS[("FY2026", "Q2")][metric]
             + _FLOW_QUARTERS[("FY2026", "Q3")][metric]
             + R.Q4[metric])
    for metric in ("revenue", "net_income", "operating_income",
                   "operating_cash_flow", "capital_expenditure")
}

# TTM through Q3 (CompanyFacts only): the prior-year Q4 + this year Q1+Q2+Q3.
TTM_THROUGH_Q3 = {
    metric: (_FLOW_QUARTERS[("FY2025", "Q4")][metric]
             + _FLOW_QUARTERS[("FY2026", "Q1")][metric]
             + _FLOW_QUARTERS[("FY2026", "Q2")][metric]
             + _FLOW_QUARTERS[("FY2026", "Q3")][metric])
    for metric in ("revenue", "net_income", "operating_income",
                   "operating_cash_flow", "capital_expenditure")
}


_US_GAAP = {
    "revenue": ("Revenues", False),
    "net_income": ("NetIncomeLoss", False),
    "operating_income": ("OperatingIncomeLoss", False),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities", False),
    "capital_expenditure": ("PaymentsToAcquirePropertyPlantAndEquipment", False),
    "cash_and_cash_equivalents": ("CashAndCashEquivalentsAtCarryingValue", True),
    "short_term_investments": ("ShortTermInvestments", True),
    "assets": ("Assets", True),
    "current_assets": ("AssetsCurrent", True),
    "liabilities": ("Liabilities", True),
    "current_liabilities": ("LiabilitiesCurrent", True),
    "stockholders_equity": ("StockholdersEquity", True),
    "short_term_debt": ("ShortTermBorrowings", True),
    "long_term_debt": ("LongTermDebtNoncurrent", True),
}


def _row(form, filed, accn, **extra):
    return dict(form=form, filed=filed, accn=accn, fy=None, fp=None, **extra)


def companyfacts_through_q3(*, include_q3_balance: bool = True,
                            drop_fields: Tuple[str, ...] = (),
                            revenue_override: Optional[Dict[Tuple[str, str], float]] = None
                            ) -> dict:
    """A CompanyFacts payload: FY2025 (10-K) + FY2026 Q1-Q3 (10-Q), in dollars."""
    payload: Dict[str, dict] = {}
    revenue_override = revenue_override or {}

    def add(field_name, *, start, end, val, form, filed, accn):
        if field_name in drop_fields:
            return
        concept, _instant = _US_GAAP[field_name]
        rows = payload.setdefault(concept, {"units": {"USD": []}})["units"]["USD"]
        row = _row(form, filed, accn, end=end, val=val * MILLIONS)
        if start is not None:
            row["start"] = start
        rows.append(row)

    quarter_filings = {
        ("FY2025", "Q1"): ("10-Q", "2024-12-05", "10-Q-2024-10-31"),
        ("FY2025", "Q2"): ("10-Q", "2025-03-06", "10-Q-2025-01-31"),
        ("FY2025", "Q3"): ("10-Q", "2025-06-05", "10-Q-2025-04-30"),
        ("FY2025", "Q4"): ("10-K", "2025-09-19", "10-K-2025"),
        ("FY2026", "Q1"): ("10-Q", "2025-12-04", "10-Q-2025-10-31"),
        ("FY2026", "Q2"): ("10-Q", "2026-03-05", "10-Q-2026-01-31"),
        ("FY2026", "Q3"): ("10-Q", "2026-06-05", "10-Q-2026-04-30"),
    }
    for key, flows in _FLOW_QUARTERS.items():
        year, quarter = key
        start, end = (FY2025 if year == "FY2025" else FY2026)[quarter]
        form, filed, accn = quarter_filings[key]
        for metric, value in flows.items():
            value = revenue_override.get(key, value) if metric == "revenue" else value
            add(metric, start=start, end=end, val=value, form=form,
                filed=filed, accn=accn)

    # Balance-sheet instants: the FY2025 10-K carries the year-end, the Q3
    # 10-Q carries 30 April 2026.
    for as_of, form, filed, accn in (
            (FY2025_END, "10-K", "2025-09-19", "10-K-2025"),
            (Q3_FY26_END, "10-Q", "2026-06-05", "10-Q-2026-04-30")):
        if as_of == Q3_FY26_END and not include_q3_balance:
            continue
        for metric, value in _BALANCE[as_of].items():
            add(metric, start=None, end=as_of, val=value, form=form,
                filed=filed, accn=accn)

    return {"facts": {"us-gaap": payload}}


# ---------------------------------------------------------------------------
# Ground-truth vocabulary
# ---------------------------------------------------------------------------

class Fresh:
    CURRENT = "CURRENT_PERIOD"
    FALLBACK = "FALLBACK_PRIOR_PERIOD"
    MISSING = "MISSING"
    CONFLICTED = "CONFLICTED"


class DcfBase:
    CURRENT = "CURRENT"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class Hard:
    CURRENT_METRIC_FROM_PRIOR_PERIOD_UNDISCLOSED = (
        "CURRENT_METRIC_FROM_PRIOR_PERIOD_UNDISCLOSED")
    LATEST_QUARTER_METRIC_FROM_PRIOR_QUARTER = "LATEST_QUARTER_METRIC_FROM_PRIOR_QUARTER"
    CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD = "CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD"
    DUPLICATE_ECONOMIC_QUARTER_IN_TTM = "DUPLICATE_ECONOMIC_QUARTER_IN_TTM"
    STALE_DCF_MARKED_RESEARCH_VALID = "STALE_DCF_MARKED_RESEARCH_VALID"
    RAW_REPORTED_ACTUAL_BYPASSES_CANONICAL_STATE = (
        "RAW_REPORTED_ACTUAL_BYPASSES_CANONICAL_STATE")
    RAW_COMPANYFACTS_BYPASSES_CANONICAL_STATE = (
        "RAW_COMPANYFACTS_BYPASSES_CANONICAL_STATE")
    GUIDANCE_FACT_ENTERS_ACTUAL_SET = "GUIDANCE_FACT_ENTERS_ACTUAL_SET"
    SILENT_MIXED_PERIOD_CURRENT_SNAPSHOT = "SILENT_MIXED_PERIOD_CURRENT_SNAPSHOT"
    SAME_PERIOD_MATERIAL_CONFLICT_SILENTLY_OVERWRITTEN = (
        "SAME_PERIOD_MATERIAL_CONFLICT_SILENTLY_OVERWRITTEN")

    ALL = (CURRENT_METRIC_FROM_PRIOR_PERIOD_UNDISCLOSED,
           LATEST_QUARTER_METRIC_FROM_PRIOR_QUARTER,
           CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD,
           DUPLICATE_ECONOMIC_QUARTER_IN_TTM,
           STALE_DCF_MARKED_RESEARCH_VALID,
           RAW_REPORTED_ACTUAL_BYPASSES_CANONICAL_STATE,
           RAW_COMPANYFACTS_BYPASSES_CANONICAL_STATE,
           GUIDANCE_FACT_ENTERS_ACTUAL_SET,
           SILENT_MIXED_PERIOD_CURRENT_SNAPSHOT,
           SAME_PERIOD_MATERIAL_CONFLICT_SILENTLY_OVERWRITTEN)


class FailureClass:
    UNIFIED_FACT_SELECTION = "UNIFIED_FACT_SELECTION"
    PER_METRIC_FRESHNESS = "PER_METRIC_FRESHNESS"
    PERIOD_IDENTITY = "PERIOD_IDENTITY"
    TTM_DEDUP = "TTM_DEDUP"
    TTM_RECONSTRUCTION = "TTM_RECONSTRUCTION"
    SAME_PERIOD_RECONCILIATION = "SAME_PERIOD_RECONCILIATION"
    DCF_BASE_ALIGNMENT = "DCF_BASE_ALIGNMENT"
    RESEARCH_FRESHNESS = "RESEARCH_FRESHNESS"
    ACTUAL_GUIDANCE_SEPARATION = "ACTUAL_GUIDANCE_SEPARATION"

    ALL = (UNIFIED_FACT_SELECTION, PER_METRIC_FRESHNESS, PERIOD_IDENTITY,
           TTM_DEDUP, TTM_RECONSTRUCTION, SAME_PERIOD_RECONCILIATION,
           DCF_BASE_ALIGNMENT, RESEARCH_FRESHNESS, ACTUAL_GUIDANCE_SEPARATION)


_FLOW = ("revenue", "net_income", "operating_income", "operating_cash_flow",
         "capital_expenditure")
_INSTANT = ("cash_and_cash_equivalents", "stockholders_equity", "total_debt",
            "current_assets", "current_liabilities")


def _all_current(*, instants_from: str = Q4_FY26_END,
                 flow_fallback: Tuple[str, ...] = (),
                 instant_fallback: Tuple[str, ...] = ()) -> Dict[str, str]:
    out = {m: (Fresh.FALLBACK if m in flow_fallback else Fresh.CURRENT)
           for m in _FLOW}
    for m in _INSTANT:
        out[m] = Fresh.FALLBACK if m in instant_fallback else Fresh.CURRENT
    return out


# ---------------------------------------------------------------------------
# A case
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntegrationCase:
    """One (CompanyFacts, release) pair and everything true of the merged state."""

    case_id: str
    case_class: str
    description: str
    company_facts: dict
    release_document: Optional[str]
    release_form: str = "8-K"
    release_filed: str = "2026-09-02"
    release_accession: str = "0000000000-26-000001"
    later_10k_document: Optional[str] = None
    later_10k_filed: str = "2026-10-15"
    as_of: str = "2026-09-15"
    # A packet built from this period, for the stale-base positive control.
    dcf_packet_base_period: Optional[str] = None

    # -- ground truth ----------------------------------------------------
    expected_resolved_period: str = "UNKNOWN"
    expected_active: bool = True
    # canonical current metric -> period end it must come from (ISO) or Fresh.*
    expected_metric_period: Dict[str, str] = field(default_factory=dict)
    expected_freshness: Dict[str, str] = field(default_factory=dict)
    expected_canonical_revenue: Optional[float] = None
    expected_canonical_net_income: Optional[float] = None
    expected_ttm_end_period: str = "UNKNOWN"
    expected_ttm_revenue: Optional[float] = None
    expected_latest_quarter_revenue: Optional[float] = None
    expected_dcf_base_status: str = "UNKNOWN"
    expected_same_period_conflict: bool = False
    expected_ttm_quarter_ends: Tuple[str, ...] = ()
    positive_controls: Tuple[str, ...] = ()
    notes: str = ""


def _cf():
    return companyfacts_through_q3()


ALL_CASES: Tuple[IntegrationCase, ...] = (

    # -- A: complete Q4 release advances everything ----------------------
    IntegrationCase(
        case_id="A-complete-release-advances-all", case_class="A",
        description="Q1-Q3 from CompanyFacts, complete Q4 earnings release.",
        company_facts=_cf(), release_document=R.COMPLETE_Q4_RELEASE,
        dcf_packet_base_period=Q4_FY26_END,
        expected_resolved_period=Q4_FY26_END, expected_active=True,
        expected_metric_period={m: Q4_FY26_END for m in _FLOW + _INSTANT},
        expected_freshness=_all_current(),
        expected_canonical_revenue=TTM_THROUGH_Q4["revenue"] * MILLIONS,
        expected_canonical_net_income=TTM_THROUGH_Q4["net_income"] * MILLIONS,
        expected_ttm_end_period=Q4_FY26_END,
        expected_ttm_revenue=TTM_THROUGH_Q4["revenue"] * MILLIONS,
        expected_latest_quarter_revenue=R.Q4["revenue"] * MILLIONS,
        expected_dcf_base_status=DcfBase.CURRENT,
        expected_ttm_quarter_ends=("2025-10-31", "2026-01-31",
                                   "2026-04-30", "2026-07-31"),
        positive_controls=(Hard.CURRENT_METRIC_FROM_PRIOR_PERIOD_UNDISCLOSED,
                           Hard.LATEST_QUARTER_METRIC_FROM_PRIOR_QUARTER,
                           Hard.CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD),
        notes="The whole phase in one case: resolved period and resolved "
              "numbers advance together."),

    # -- B: Q4 release lacks the balance sheet ---------------------------
    IntegrationCase(
        case_id="B-release-lacks-balance-sheet", case_class="B",
        description="Q4 release with income statement and cash flow only.",
        company_facts=_cf(), release_document=R.RELEASE_WITH_OUTLOOK_NO_BS,
        dcf_packet_base_period=Q4_FY26_END,
        expected_resolved_period=Q3_FY26_END, expected_active=False,
        expected_metric_period={},
        expected_freshness={},
        expected_ttm_end_period=Q3_FY26_END,
        expected_dcf_base_status=DcfBase.CURRENT,
        notes="An income statement and a cash-flow statement without a balance "
              "sheet cannot carry the whole period. No unsafe wholesale "
              "advance; the state stays on Q3."),

    # -- C: Q4 release has the balance sheet ---------------------------
    IntegrationCase(
        case_id="C-release-has-balance-sheet", case_class="C",
        description="Complete Q4 release; cash/debt/ratios use Q4 values.",
        company_facts=_cf(), release_document=R.COMPLETE_Q4_RELEASE,
        expected_resolved_period=Q4_FY26_END, expected_active=True,
        expected_metric_period={m: Q4_FY26_END for m in _INSTANT},
        expected_freshness=_all_current(),
        expected_dcf_base_status=DcfBase.CURRENT,
        notes="cash_and_cash_equivalents, debt and the current ratio come from "
              "the Q4 balance sheet, not the Q3 one."),

    # -- D: Q4 release then same-period 10-K ---------------------------
    IntegrationCase(
        case_id="D-release-then-10k", case_class="D",
        description="Q4 release, then the FY 10-K for the same period.",
        company_facts=_cf(), release_document=R.COMPLETE_Q4_RELEASE,
        later_10k_document=R.PERIODIC_FILING_MATCHING,
        as_of="2026-11-01",
        expected_resolved_period=Q4_FY26_END, expected_active=True,
        expected_metric_period={m: Q4_FY26_END for m in _FLOW},
        expected_freshness=_all_current(),
        expected_ttm_end_period=Q4_FY26_END,
        expected_ttm_quarter_ends=("2025-10-31", "2026-01-31",
                                   "2026-04-30", "2026-07-31"),
        positive_controls=(Hard.DUPLICATE_ECONOMIC_QUARTER_IN_TTM,),
        notes="One economic period, two sources. The 10-K is authoritative; "
              "no duplicate Q4 in the twelve-month window."),

    # -- E: same-period material disagreement ------------------------
    IntegrationCase(
        case_id="E-same-period-conflict", case_class="E",
        description="Q4 release and later 10-K disagree on revenue by 4.5%.",
        company_facts=_cf(), release_document=R.COMPLETE_Q4_RELEASE,
        later_10k_document=R.PERIODIC_FILING_CONFLICTING,
        as_of="2026-11-01",
        expected_resolved_period=Q4_FY26_END, expected_active=True,
        expected_same_period_conflict=True,
        expected_freshness={**_all_current(), "revenue": Fresh.CONFLICTED},
        expected_metric_period={"net_income": Q4_FY26_END},
        positive_controls=(Hard.SAME_PERIOD_MATERIAL_CONFLICT_SILENTLY_OVERWRITTEN,),
        notes="The conflict survives into the unified fact set; revenue is "
              "CONFLICTED, not silently overwritten."),

    # -- F: release supplies a quarter already in CompanyFacts -------
    IntegrationCase(
        case_id="F-duplicate-quarter", case_class="F",
        description="Release restates Q3, which CompanyFacts already carries.",
        company_facts=_cf(), release_document=R.RELEASE_Q3_RESTATEMENT,
        release_filed="2026-06-04",
        expected_resolved_period=Q3_FY26_END, expected_active=False,
        expected_same_period_conflict=False,
        expected_ttm_end_period=Q3_FY26_END,
        expected_ttm_quarter_ends=("2025-07-31", "2025-10-31",
                                   "2026-01-31", "2026-04-30"),
        positive_controls=(Hard.DUPLICATE_ECONOMIC_QUARTER_IN_TTM,),
        notes="The release's Q3 is the same economic quarter CompanyFacts "
              "reports. One observation in the window, not two."),

    # -- G: resolved period P but a metric stuck at P-1 (control) ----
    IntegrationCase(
        case_id="G-current-metric-stuck-at-prior", case_class="G",
        description="Positive control: a current metric selected from P-1.",
        company_facts=_cf(), release_document=R.COMPLETE_Q4_RELEASE,
        expected_resolved_period=Q4_FY26_END, expected_active=True,
        expected_freshness=_all_current(),
        positive_controls=(Hard.CURRENT_METRIC_FROM_PRIOR_PERIOD_UNDISCLOSED,),
        notes="The harness injects a P-1 revenue into the rebuilt state and "
              "asserts the hard-safety check flags it."),

    # -- H: new period, DCF rebuilt from P ---------------------------
    IntegrationCase(
        case_id="H-dcf-base-current", case_class="H",
        description="Q4 resolved, DCF base built from Q4.",
        company_facts=_cf(), release_document=R.COMPLETE_Q4_RELEASE,
        dcf_packet_base_period=Q4_FY26_END,
        expected_resolved_period=Q4_FY26_END, expected_active=True,
        expected_dcf_base_status=DcfBase.CURRENT,
        notes="The DCF financial base matches the resolved state -- NOT stale."),

    # -- I: new period, DCF still on P-1 (control) -------------------
    IntegrationCase(
        case_id="I-dcf-base-stale", case_class="I",
        description="Positive control: Q4 resolved, packet base still Q3.",
        company_facts=_cf(), release_document=R.COMPLETE_Q4_RELEASE,
        dcf_packet_base_period=Q3_FY26_END,
        expected_resolved_period=Q4_FY26_END, expected_active=True,
        expected_dcf_base_status=DcfBase.STALE,
        positive_controls=(Hard.STALE_DCF_MARKED_RESEARCH_VALID,),
        notes="A packet built from Q3 while Q4 is resolved must be STALE and "
              "may supply no valuation evidence."),

    # -- J: point-in-time currentness judged by balance-sheet date --
    IntegrationCase(
        case_id="J-point-in-time-currentness", case_class="J",
        description="Cash from the resolved balance-sheet date, with a TTM present.",
        company_facts=_cf(), release_document=R.COMPLETE_Q4_RELEASE,
        expected_resolved_period=Q4_FY26_END, expected_active=True,
        expected_freshness={"cash_and_cash_equivalents": Fresh.CURRENT,
                            "stockholders_equity": Fresh.CURRENT,
                            "revenue": Fresh.CURRENT},
        expected_metric_period={"cash_and_cash_equivalents": Q4_FY26_END},
        notes="A balance-sheet metric is CURRENT because its date is the "
              "resolved period's, even though it has no twelve-month window."),

    # -- K: a flow metric from a prior period stays a fallback ------
    IntegrationCase(
        case_id="K-flow-lags-instants", case_class="K",
        description="Complete Q4 release that omits the capex line; capex "
                    "stays on Q3 while everything else advances.",
        company_facts=_cf(), release_document=R.RELEASE_NO_CASH_FLOW,
        expected_resolved_period=Q4_FY26_END, expected_active=True,
        expected_freshness={
            "revenue": Fresh.CURRENT, "net_income": Fresh.CURRENT,
            "operating_income": Fresh.CURRENT,
            "operating_cash_flow": Fresh.CURRENT,
            "cash_and_cash_equivalents": Fresh.CURRENT,
            "stockholders_equity": Fresh.CURRENT,
            "capital_expenditure": Fresh.FALLBACK},
        expected_metric_period={"capital_expenditure": Q3_FY26_END,
                                "revenue": Q4_FY26_END,
                                "operating_cash_flow": Q4_FY26_END},
        positive_controls=(Hard.CURRENT_METRIC_FROM_PRIOR_PERIOD_UNDISCLOSED,),
        notes="The one flow the release omits stays on Q3 and is LABELLED a "
              "fallback -- never presented as current. Everything else is Q4."),
)


def cases() -> Tuple[IntegrationCase, ...]:
    return ALL_CASES


def classes_covered():
    return sorted({c.case_class for c in ALL_CASES})


def controls_covered() -> Dict[str, list]:
    found = {name: [] for name in Hard.ALL}
    for case in ALL_CASES:
        for name in case.positive_controls:
            found[name].append(case.case_id)
    # Structural counters: asserted by a static test over the workflow, and
    # exercised on every active case by the harness feeding only the rebuilt
    # state to canonical/DCF.
    for name in (Hard.RAW_REPORTED_ACTUAL_BYPASSES_CANONICAL_STATE,
                 Hard.RAW_COMPANYFACTS_BYPASSES_CANONICAL_STATE,
                 Hard.GUIDANCE_FACT_ENTERS_ACTUAL_SET,
                 Hard.SILENT_MIXED_PERIOD_CURRENT_SNAPSHOT):
        found[name] = [c.case_id for c in ALL_CASES if c.expected_active]
    return found
