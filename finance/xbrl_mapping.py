"""Phase H.3 — deterministic SEC XBRL concept -> normalized-field mapping.

No fuzzy/LLM tag matching anywhere in this file: each normalized field has a
fixed, reviewed, PRECEDENCE-ORDERED list of candidate `us-gaap` concept names
(from finance/xbrl_mapping.py's own review of real filings, cross-checked
live against AAPL's actual companyfacts payload this session — see
docs/security/YAHOO_SEC_PROVIDER_REVIEW.md). The first candidate with a
usable fact for the requested period wins; an issuer using a genuinely
unlisted custom tag produces a controlled `unresolved` entry, never a guess.

Feeds directly into the SAME normalized field names finance/normalization.py
already produces from Alpha Vantage (cash_and_cash_equivalents,
short_term_investments, total_debt, ...) — this file's job ends at "one
resolved value + its provenance per concept", not at building the final
report-ready statement shape.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

# Each entry: normalized_field -> (is_instant, candidate concepts in precedence order).
# is_instant=True means the concept is a POINT-IN-TIME balance (only an `end`
# date in the XBRL fact, e.g. cash on the balance sheet); False means a
# DURATION over a period (both `start` and `end`, e.g. revenue for a quarter).
# Getting this wrong would silently mix a snapshot with a flow.
CONCEPT_MAP: Dict[str, Tuple[bool, Tuple[str, ...]]] = {
    "revenue": (False, (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
    )),
    "net_income": (False, ("NetIncomeLoss", "ProfitLoss")),
    "operating_income": (False, ("OperatingIncomeLoss",)),
    "cash_and_cash_equivalents": (True, ("CashAndCashEquivalentsAtCarryingValue",)),
    "short_term_investments": (True, (
        "ShortTermInvestments",
        "MarketableSecuritiesCurrent",
        "ShortTermInvestmentsAndMarketableSecurities",
    )),
    "assets": (True, ("Assets",)),
    "current_assets": (True, ("AssetsCurrent",)),
    "liabilities": (True, ("Liabilities",)),
    "current_liabilities": (True, ("LiabilitiesCurrent",)),
    "stockholders_equity": (True, (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    )),
    "short_term_debt": (True, ("ShortTermBorrowings", "DebtCurrent")),
    # Equity-bridge deductions (Phase H.4). finance/dcf.py subtracts both from
    # equity value and has always accepted them as inputs; until now nothing
    # ever RESOLVED them from SEC data, so both silently defaulted to 0 for
    # every SEC-sourced valuation. A company with real preferred stock or a
    # material non-controlling interest was therefore overvalued by exactly
    # those amounts. Absence still means 0 -- but now it means "the company
    # reported none", not "nobody looked".
    "preferred_equity": (True, (
        "PreferredStockValue",
        "PreferredStockValueOutstanding",
    )),
    "minority_interest": (True, (
        "MinorityInterest",
        "StockholdersEquityAttributableToNoncontrollingInterest",
    )),
    # Phase H.4 -- LEASE-INCLUSIVE DEBT TAGS.
    #
    # This project's documented debt policy excludes finance/capital lease
    # obligations (see finance/normalization.py::_derive_balance_sheet_
    # aggregates). That policy is about which COMPONENTS to add up; it was
    # never meant to mean "report no debt at all", which is what it silently
    # produced for issuers that report debt only on a lease-inclusive line.
    #
    # Verizon is the case that exposed it: VZ reports
    # `LongTermDebtAndCapitalLeaseObligations` ($143.4B at 2026-06-30) and
    # uses neither `LongTermDebtNoncurrent` nor (since 2013) `LongTermDebt`.
    # long_term_debt therefore resolved to None and VZ's total debt came out
    # as $21.8B -- the current portion alone -- against an actual ~$165B. A
    # ~$143B hole in the equity bridge, silently, on one of the most
    # heavily indebted issuers in the index.
    #
    # The lease-inclusive spellings are listed LAST, so an issuer that
    # separates the two is unaffected and keeps the stricter policy. When a
    # lease-inclusive tag does win, finance/freshness.py records it in the
    # selection's derivation so the report can say the figure includes
    # capitalized leases rather than quietly changing what "total debt"
    # means.
    "current_portion_of_long_term_debt": (True, (
        "LongTermDebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
    )),
    "long_term_debt": (True, (
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
    )),
    "total_debt_combined": (True, ("DebtLongtermAndShorttermCombinedAmount",)),
    "operating_cash_flow": (False, ("NetCashProvidedByUsedInOperatingActivities",)),
    # VZ corrective patch -- the productive-assets tags were MISSING, and
    # their absence is expensive rather than merely incomplete. Verizon
    # reports capital expenditure as PaymentsToAcquireOtherProductiveAssets
    # ($17.011B for FY2025) and uses neither PropertyPlantAndEquipment tag
    # anywhere in its filings, so capex resolved to None and
    # `propose_assumptions` fell back to the configured 5%-of-revenue
    # default. Against VZ's ~$134B revenue that is ~12.7% actual vs 5%
    # assumed -- a ~2.5x understatement of the single largest drag on FCFF,
    # for one of the most capital-intensive issuers there is. The DCF still
    # "validated" cleanly, because validation checks arithmetic, not whether
    # an input was ever found. Older VZ filings (through 2018) use the
    # non-"Other" spelling, so both are listed; `resolve_concept` falls
    # through per-period, so a company that switched tags mid-history
    # resolves correctly on both sides of the switch.
    "capital_expenditure": (False, (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForAdditionsToPropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquireOtherProductiveAssets",
        "PaymentsForCapitalImprovements",
    )),
    "depreciation_and_amortization": (False, (
        "DepreciationDepletionAndAmortization",
        "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
        "Depreciation",
    )),
    "diluted_shares": (False, ("WeightedAverageNumberOfDilutedSharesOutstanding",)),
    "diluted_eps": (False, ("EarningsPerShareDiluted",)),
}

ANNUAL_FORMS = ("10-K", "10-K/A")
QUARTERLY_FORMS = ("10-Q", "10-Q/A")


@dataclass(frozen=True)
class ResolvedFact:
    field: str
    value: float
    unit: str
    concept: str
    accession: str
    fiscal_year: int
    fiscal_period: str
    form: str
    filed: str
    start: Optional[str]
    end: Optional[str]

    def to_dict(self) -> dict:
        return {
            "field": self.field, "value": self.value, "unit": self.unit,
            "concept": self.concept, "accession_number": self.accession,
            "fiscal_year": self.fiscal_year, "fiscal_period": self.fiscal_period,
            "form": self.form, "filed": self.filed, "start": self.start, "end": self.end,
        }


def _candidate_facts(company_facts: dict, concept: str) -> List[dict]:
    usgaap = (company_facts.get("facts") or {}).get("us-gaap") or {}
    entry = usgaap.get(concept)
    if not entry:
        return []
    units = entry.get("units") or {}
    # Prefer USD (or USD/shares for share-count concepts); a concept reported
    # in a unit this project doesn't handle is treated as absent, not guessed.
    for unit_name in ("USD", "USD/shares", "shares"):
        if unit_name in units and isinstance(units[unit_name], list):
            return [dict(f, _unit=unit_name) for f in units[unit_name]]
    return []


def _matches_period(fact: dict, forms: Tuple[str, ...], fiscal_year: Optional[int],
                    fiscal_period: Optional[str]) -> bool:
    if forms and fact.get("form") not in forms:
        return False
    if fiscal_year is not None and fact.get("fy") != fiscal_year:
        return False
    if fiscal_period is not None and fact.get("fp") != fiscal_period:
        return False
    return True


# MLI corrective patch -- DURATION WINDOWS.
#
# A DURATION fact's fiscal-period TAG is not sufficient to know what span it
# actually covers. In SEC companyfacts a Q4 duration fact is routinely tagged
# `fp: "FY"` (companies do not file a separate Q4 10-Q, so the fourth-quarter
# figure rides along in the 10-K carrying the annual period tag). A 10-K
# therefore commonly contains BOTH, with the SAME `fy`, `fp`, `form`, `filed`
# AND `end` -- differing only in `start`.
#
# Found live on MLI: annual revenue came back as FY2025 $4.18B (a true year)
# but FY2024 $923.5M, FY2023 $732.4M, FY2022 $877.6M, FY2021 $956.4M -- all
# fourth-QUARTER magnitudes (MLI's real FY2024 revenue is ~$3.77B). Because
# the old sort key `(filed, end)` tied on every field for both facts, which
# one `matching[-1]` returned was effectively arbitrary. That produced
# revenue_growth_yoy = (4.18B - 0.92B)/0.92B = +352.5% (a full year divided
# by a single quarter), net_income_growth_yoy = +456%, and a revenue_cagr of
# +44.6% that then fed the DCF base growth assumption and hit its 25% clamp
# -- so the clamp was masking a data bug rather than capping a real
# company characteristic.
#
# Generous windows: fiscal years are 52/53 weeks (364-371 days) and vary
# with 4-4-5 calendars, so anything from ~10 to ~13 months counts as annual
# and ~1 to ~4 months as quarterly. The windows are deliberately wide enough
# to accept irregular real-world calendars but far too narrow to confuse a
# quarter with a year, which is the only distinction that matters here.
_ANNUAL_DURATION_DAYS = (300, 400)
_QUARTERLY_DURATION_DAYS = (45, 130)


def _duration_days(fact: dict) -> Optional[int]:
    """Actual span of a duration fact in days, or None if not computable."""
    start, end = fact.get("start"), fact.get("end")
    if not start or not end:
        return None
    try:
        return (date.fromisoformat(end) - date.fromisoformat(start)).days
    except (ValueError, TypeError):
        return None


def duration_window_for_forms(forms: Tuple[str, ...]) -> Optional[Tuple[int, int]]:
    """The expected duration window for the period type `forms` selects."""
    if forms == ANNUAL_FORMS:
        return _ANNUAL_DURATION_DAYS
    if forms == QUARTERLY_FORMS:
        return _QUARTERLY_DURATION_DAYS
    return None


def resolve_concept(company_facts: dict, field: str, forms: Tuple[str, ...],
                    fiscal_year: Optional[int] = None,
                    fiscal_period: Optional[str] = None,
                    duration_days: Optional[Tuple[int, int]] = None) -> Optional[ResolvedFact]:
    """Resolve ONE normalized field for ONE period, trying candidates in
    precedence order. Among matching facts for the WINNING concept, the most
    recently FILED one wins — this is how an amended/restated filing
    (10-K/A) correctly supersedes the original without the caller having to
    know that happened.

    `duration_days` (MLI corrective patch) is an inclusive (min, max) day
    window a DURATION fact's actual start->end span must fall inside; it is
    ignored for instant facts. Defaults to the window implied by `forms` --
    see `duration_window_for_forms` and the long comment above -- so callers
    that pass ANNUAL_FORMS/QUARTERLY_FORMS get correct filtering without
    changing their call. Pass an explicit window to override, or `(0, 10**6)`
    to accept any duration (the pre-patch behavior).
    """
    if field not in CONCEPT_MAP:
        return None
    if duration_days is None:
        duration_days = duration_window_for_forms(forms)
    is_instant, candidates = CONCEPT_MAP[field]
    for concept in candidates:
        facts = _candidate_facts(company_facts, concept)
        matching = [f for f in facts if _matches_period(f, forms, fiscal_year, fiscal_period)]
        if is_instant:
            matching = [f for f in matching if not f.get("start")]
        else:
            matching = [f for f in matching if f.get("start")]
            if duration_days is not None:
                low, high = duration_days
                matching = [f for f in matching
                           if (d := _duration_days(f)) is not None and low <= d <= high]
        if not matching:
            continue
        # Tie-break on the actual span too: with `end` and `filed` equal
        # between an annual and a Q4 fact, span is the only field that
        # distinguishes them, so an explicit key beats relying on source
        # ordering. (The window above already excludes the wrong one; this
        # keeps selection deterministic if a filing carries two facts that
        # both land inside the window.)
        matching.sort(key=lambda f: (f.get("filed") or "", f.get("end") or "",
                                     _duration_days(f) or 0))
        best = matching[-1]
        return ResolvedFact(
            field=field, value=float(best["val"]), unit=best.get("_unit", "USD"),
            concept=concept, accession=best.get("accn", ""),
            fiscal_year=best.get("fy"), fiscal_period=best.get("fp"),
            form=best.get("form", ""), filed=best.get("filed", ""),
            start=best.get("start"), end=best.get("end"),
        )
    return None


def list_available_periods(company_facts: dict, forms: Tuple[str, ...]) -> List[Tuple[int, str]]:
    """Every distinct (fiscal_year, fiscal_period) available for ANY mapped
    concept under the given form filter, most recent first -- used to decide
    which periods to actually extract without guessing at fiscal-year-end."""
    seen = set()
    usgaap = (company_facts.get("facts") or {}).get("us-gaap") or {}
    # NetIncomeLoss/Assets are near-universally reported, so they anchor
    # period discovery without scanning every concept in the payload.
    for concept in ("NetIncomeLoss", "Assets", "Revenues",
                    "RevenueFromContractWithCustomerExcludingAssessedTax"):
        for fact in _candidate_facts(company_facts, concept):
            if fact.get("form") in forms and fact.get("fy") is not None and fact.get("fp"):
                seen.add((fact["fy"], fact["fp"]))
    return sorted(seen, key=lambda p: (p[0], p[1]), reverse=True)


def extract_statements(company_facts: dict, period_type: str, max_periods: int) -> List[dict]:
    """One dict per period (most recent first), each mapping normalized field
    name -> ResolvedFact.to_dict() (or omitted entirely if unresolved for
    that period -- a controlled absence, never a guessed zero)."""
    forms = ANNUAL_FORMS if period_type == "annual" else QUARTERLY_FORMS
    periods = list_available_periods(company_facts, forms)[:max_periods]
    out = []
    for fy, fp in periods:
        period_dict = {"fiscal_year": fy, "fiscal_period": fp, "form": forms[0]}
        values = {}
        unresolved = []
        for field_name in CONCEPT_MAP:
            resolved = resolve_concept(company_facts, field_name, forms, fy, fp)
            if resolved is not None:
                values[field_name] = resolved.to_dict()
            else:
                unresolved.append(field_name)
        period_dict["values"] = values
        period_dict["unresolved_fields"] = unresolved
        out.append(period_dict)
    return out
