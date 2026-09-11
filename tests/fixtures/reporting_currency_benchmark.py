"""Generalized fixtures for finance/reporting_currency.py.

Ground truth for every case is decided INDEPENDENTLY of the resolver's own
logic -- by reading the dates and currencies built into the fixture, not by
running `resolve_reporting_series` and copying its answer (spec section 22).

Cases correspond to the lettered list in the REPORTING_CURRENCY_SERIES_
SELECTION phase prompt, section 21. Two are deliberately not modelled as
standalone cases here:

  D. "Old series complete but years stale, cannot become current" -- when
     only ONE currency is present at all, there is nothing for a currency
     resolver to prefer it OVER; the staleness of an unrivalled series is
     Actualization's job (`finance.actualization.MAX_COMPLETE_CANDIDATE_
     LAG_DAYS`, phase 4), not a currency-selection question. Modelled below
     as CASE_D to prove this resolver correctly stays out of that lane
     (RESOLVED, not a false currency conflict) rather than skipped.

  H. "TTM uses comparative values restated into the new currency" -- this
     project has NO reading path for any non-USD unit at all (finance/
     xbrl_mapping.py::_candidate_facts only ever reads `USD`), so even a
     perfectly restated JPY comparative series cannot enter the numeric TTM
     path regardless of provenance. There is no shortcut fix for this that
     is not FX conversion (forbidden) or a parallel reading path (also
     forbidden as scope creep). Documented, not modelled as a passing case.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from finance.reporting_currency import ReportingSeriesStatus


def instant(end: str, value: float, form: str = "10-Q", filed: Optional[str] = None,
           accn: str = "acc") -> dict:
    return {"end": end, "val": value, "form": form, "filed": filed or end,
            "accn": accn, "fy": None, "fp": None}


def duration(start: str, end: str, value: float, form: str = "10-Q",
            filed: Optional[str] = None, accn: str = "acc") -> dict:
    return {"start": start, "end": end, "val": value, "form": form,
            "filed": filed or end, "accn": accn, "fy": None, "fp": None}


def company_facts(concepts: Dict[str, Dict[str, List[dict]]],
                  taxonomy: str = "us-gaap") -> dict:
    """concepts: {concept_name: {unit_name: [rows]}}"""
    return {"facts": {taxonomy: {
        concept: {"units": units} for concept, units in concepts.items()
    }}}


def balance_sheet_anchor_rows(entries: List[Tuple[str, float, str]],
                              currency: str = "USD") -> Dict[str, dict]:
    """`Assets` and `StockholdersEquity` rows for the given (end, value, form)
    entries, in one currency -- `finance/period_facts.py::_BALANCE_SHEET_
    ANCHORS` requires BOTH present at a date before `balance_sheet_date()`
    resolves at all, which every fixture that exercises the built
    `CurrentFinancialState` (rather than just the resolver in isolation)
    needs wired up or its balance sheet silently stays empty."""
    assets_rows = [instant(end, value, form=form) for end, value, form in entries]
    equity_rows = [instant(end, value * 0.4, form=form) for end, value, form in entries]
    return {
        "Assets": {currency: assets_rows},
        "StockholdersEquity": {currency: equity_rows},
    }


@dataclass(frozen=True)
class ReportingCurrencyCase:
    case_id: str
    description: str
    company_facts: dict
    expected_status: str
    expected_currency: Optional[str] = None
    expected_period: Optional[str] = None
    expected_rejection_codes: Tuple[str, ...] = ()
    notes: str = ""


CASES: List[ReportingCurrencyCase] = []


def _add(case: ReportingCurrencyCase) -> None:
    CASES.append(case)


# -- A. No currency switch: USD throughout -----------------------------------
_add(ReportingCurrencyCase(
    case_id="A_usd_no_switch",
    description="USD throughout a long quarterly history; current USD series selected.",
    company_facts=company_facts({
        **balance_sheet_anchor_rows([
            ("2015-12-31", 10_000_000_000.0, "10-K"),
            ("2020-12-31", 14_000_000_000.0, "10-K"),
            ("2026-04-30", 18_500_000_000.0, "10-Q"),
            ("2026-07-31", 19_200_000_000.0, "10-Q"),
        ]),
        "CashAndCashEquivalentsAtCarryingValue": {"USD": [
            instant("2015-12-31", 10_000_000_000.0, form="10-K"),
            instant("2020-12-31", 14_000_000_000.0, form="10-K"),
            instant("2026-04-30", 18_500_000_000.0, form="10-Q"),
            instant("2026-07-31", 19_200_000_000.0, form="10-Q"),
        ]},
        "Revenues": {"USD": [
            duration("2026-02-01", "2026-04-30", 4_100_000_000.0, form="10-Q"),
            duration("2026-05-01", "2026-07-31", 4_300_000_000.0, form="10-Q"),
        ]},
    }),
    expected_status=ReportingSeriesStatus.RESOLVED,
    expected_currency="USD",
    expected_period="2026-07-31",
))

# -- B. Currency switch: old USD history, new JPY current series -------------
_add(ReportingCurrencyCase(
    case_id="B_currency_switch",
    description="Issuer reported in USD through FY2017, switched to JPY from FY2018 onward.",
    company_facts=company_facts({
        **balance_sheet_anchor_rows([
            ("2010-12-31", 900_000_000.0, "10-K"),
            ("2015-12-31", 1_400_000_000.0, "10-K"),
            ("2017-12-31", 1_600_000_000.0, "10-K"),
        ]),
        "CashAndCashEquivalentsAtCarryingValue": {
            "USD": [
                instant("2010-12-31", 900_000_000.0, form="10-K"),
                instant("2015-12-31", 1_400_000_000.0, form="10-K"),
                instant("2017-12-31", 1_600_000_000.0, form="10-K"),
            ],
            "JPY": [
                instant("2018-12-31", 210_000_000_000.0, form="20-F"),
                instant("2022-12-31", 260_000_000_000.0, form="20-F"),
                instant("2026-06-30", 305_000_000_000.0, form="6-K"),
            ],
        },
        "Revenues": {
            "USD": [
                duration("2017-01-01", "2017-12-31", 950_000_000.0, form="10-K"),
            ],
            "JPY": [
                duration("2026-01-01", "2026-06-30", 140_000_000_000.0, form="6-K"),
            ],
        },
    }),
    expected_status=ReportingSeriesStatus.UNRESOLVED_CURRENCY_SWITCH,
    expected_currency="JPY",
    expected_period=None,
    expected_rejection_codes=("STALE_CURRENCY_SERIES",),
    notes="The stale USD series (ending 2017-12-31) must never be published as current.",
))

# -- C. Old series longer than new series, new still wins ---------------------
_add(ReportingCurrencyCase(
    case_id="C_old_series_longer_still_loses",
    description=("A 10-year EUR history (2005-2015) is far longer than a 2-year USD "
                "history (2024-2026), but USD is current and readable, so USD wins."),
    company_facts=company_facts({
        **balance_sheet_anchor_rows([
            ("2024-12-31", 2_100_000_000.0, "10-K"),
            ("2026-06-30", 2_400_000_000.0, "10-Q"),
        ]),
        "CashAndCashEquivalentsAtCarryingValue": {
            "EUR": [instant(f"{year}-12-31", 500_000_000.0 + year, form="20-F")
                   for year in range(2005, 2016)],
            "USD": [
                instant("2024-12-31", 2_100_000_000.0, form="10-K"),
                instant("2026-06-30", 2_400_000_000.0, form="10-Q"),
            ],
        },
    }),
    expected_status=ReportingSeriesStatus.RESOLVED,
    expected_currency="USD",
    expected_period="2026-06-30",
    notes="History length must not outrank recency + readability (spec section 6).",
))

# -- D. Old series complete but years stale, no rival currency ---------------
_add(ReportingCurrencyCase(
    case_id="D_stale_unrivalled_series",
    description=("Only USD ever reported, newest fact is years old. Currency selection "
                "still resolves cleanly; PERIOD staleness is Actualization's job."),
    company_facts=company_facts({
        **balance_sheet_anchor_rows([
            ("2015-12-31", 500_000_000.0, "10-K"),
            ("2017-12-31", 560_000_000.0, "10-K"),
        ]),
        "CashAndCashEquivalentsAtCarryingValue": {"USD": [
            instant("2015-12-31", 500_000_000.0, form="10-K"),
            instant("2017-12-31", 560_000_000.0, form="10-K"),
        ]},
    }),
    expected_status=ReportingSeriesStatus.RESOLVED,
    expected_currency="USD",
    expected_period="2017-12-31",
    notes="No currency conflict exists here; do not invent one. See module docstring.",
))

# -- E. Same period, reporting currency + convenience translation ------------
_add(ReportingCurrencyCase(
    case_id="E_convenience_translation",
    description=("At the same period end, USD backs the primary statements (4 lines); "
                "EUR backs only a convenience-translated revenue line."),
    company_facts=company_facts({
        "Assets": {"USD": [instant("2026-06-30", 3_000_000_000.0, form="10-Q")]},
        "StockholdersEquity": {"USD": [instant("2026-06-30", 1_200_000_000.0, form="10-Q")]},
        "NetIncomeLoss": {"USD": [
            duration("2026-04-01", "2026-06-30", 90_000_000.0, form="10-Q")]},
        "Revenues": {
            "USD": [duration("2026-04-01", "2026-06-30", 800_000_000.0, form="10-Q")],
            "EUR": [duration("2026-04-01", "2026-06-30", 740_000_000.0, form="10-Q")],
        },
    }),
    expected_status=ReportingSeriesStatus.RESOLVED,
    expected_currency="USD",
    expected_period="2026-06-30",
    expected_rejection_codes=("CONVENIENCE_TRANSLATION_OR_SEGMENT",),
))

# -- F. Same metric, current period, incompatible currencies tied ------------
_add(ReportingCurrencyCase(
    case_id="F_same_period_conflict",
    description="USD and JPY each back exactly two statement lines at the same period end.",
    company_facts=company_facts({
        "Assets": {"USD": [instant("2026-06-30", 3_000_000_000.0, form="10-Q")]},
        "Revenues": {"USD": [
            duration("2026-04-01", "2026-06-30", 800_000_000.0, form="10-Q")]},
        "StockholdersEquity": {"JPY": [instant("2026-06-30", 400_000_000_000.0, form="6-K")]},
        "NetIncomeLoss": {"JPY": [
            duration("2026-04-01", "2026-06-30", 12_000_000_000.0, form="6-K")]},
    }),
    expected_status=ReportingSeriesStatus.CONFLICT_SAME_PERIOD_MULTIPLE_CURRENCIES,
    expected_currency=None,
    expected_period=None,
    expected_rejection_codes=("SAME_PERIOD_AMBIGUOUS_CURRENCY",),
))

# -- I. Foreign issuer, 20-F/IFRS, no USD ever -------------------------------
_add(ReportingCurrencyCase(
    case_id="I_foreign_issuer_no_usd_ever",
    description="A 20-F filer under ifrs-full, reporting only in JPY, with no USD history.",
    company_facts=company_facts({
        "Assets": {"JPY": [
            instant("2022-12-31", 500_000_000_000.0, form="20-F"),
            instant("2026-03-31", 620_000_000_000.0, form="6-K"),
        ]},
        "Revenue": {"JPY": [
            duration("2025-04-01", "2026-03-31", 900_000_000_000.0, form="20-F"),
        ]},
    }, taxonomy="ifrs-full"),
    expected_status=ReportingSeriesStatus.UNRESOLVED_FOREIGN_CURRENCY_ONLY,
    expected_currency="JPY",
    expected_period=None,
))

# -- J. Normal domestic issuer control ---------------------------------------
_add(ReportingCurrencyCase(
    case_id="J_domestic_control",
    description="A single-period domestic USD filer -- the simplest possible control case.",
    company_facts=company_facts({
        **balance_sheet_anchor_rows([("2026-03-31", 1_000_000_000.0, "10-Q")]),
        "CashAndCashEquivalentsAtCarryingValue": {
            "USD": [instant("2026-03-31", 1_000_000_000.0, form="10-Q")]},
    }),
    expected_status=ReportingSeriesStatus.RESOLVED,
    expected_currency="USD",
    expected_period="2026-03-31",
))

# -- K. Reporting currency differs from listing currency ---------------------
# The resolver's function signature accepts only `company_facts`: it has no
# listing-currency, ticker or exchange input at all, so
# LISTING_CURRENCY_ASSUMED_AS_REPORTING_CURRENCY cannot occur by
# construction. This fixture is a foreign issuer that reports only in EUR
# (e.g. a US-listed ADR whose statements are never restated to USD).
_add(ReportingCurrencyCase(
    case_id="K_listing_currency_differs",
    description="Issuer's ADRs trade in USD on a US exchange; its statements are in EUR only.",
    company_facts=company_facts({
        "Assets": {"EUR": [instant("2026-03-31", 8_000_000_000.0, form="20-F")]},
    }),
    expected_status=ReportingSeriesStatus.UNRESOLVED_FOREIGN_CURRENCY_ONLY,
    expected_currency="EUR",
    expected_period=None,
))

# -- L. No resolvable current reporting currency -----------------------------
_add(ReportingCurrencyCase(
    case_id="L1_no_taxonomy",
    description="Payload has no us-gaap or ifrs-full facts at all.",
    company_facts={"facts": {"dei": {"EntityCommonStockSharesOutstanding": {}}}},
    expected_status=ReportingSeriesStatus.UNRESOLVED_NO_TAXONOMY,
))
_add(ReportingCurrencyCase(
    case_id="L2_no_statement_facts",
    description="us-gaap taxonomy present, but none of the wanted concepts have any rows.",
    company_facts=company_facts({
        "SomeUnmappedConcept": {"USD": [instant("2026-03-31", 1.0, form="10-Q")]},
    }),
    expected_status=ReportingSeriesStatus.UNRESOLVED_NO_STATEMENT_FACTS,
))


CASE_IDS = tuple(c.case_id for c in CASES)
