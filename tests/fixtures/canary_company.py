"""Part 18 — SANITIZED BUSINESS-CLASS fixtures, one per model class.

Spec §19 is explicit about why these exist: *ticker fixtures prove a
regression did not come back; they cannot prove an architecture.* Each of the
nine ticker fixtures in this directory exercises only the path its own issuer
happens to take, which is exactly why every new stock kept finding a fresh
route to the same class of error.

A canary is the opposite construction. It is a synthetic issuer built to sit
squarely inside ONE business-model class, with the class's defining
characteristic dialled up until it is unmistakable, and nothing else about it
distinctive at all. If the architecture handles the class, it handles the
canary; if it handles only the issuers it has already seen, the canary
catches that.

So these are deliberately NOT renamed ticker fixtures. Every figure here is
invented. Where a canary was inspired by a live failure the inspiration is
the SHAPE of the problem — an insurer's operating cash flow, a broker-dealer
holding customer money, a foreign private issuer filing a 20-F — never the
issuer's numbers.

The builder produces the same two payload families a real run consumes:

    SEC   company_facts (XBRL us-gaap facts), company_submissions, ticker map
    Yahoo stock_quote, company_profile, price_history, corporate_actions

so a canary runs through `run_full_stock_analysis` with the REAL executor,
the REAL normalization, the REAL freshness planner and the REAL research
gate. Nothing is stubbed except the two network clients.
"""

import datetime
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# The canaries are dated relative to one another, not to the wall clock, so
# their period selection is stable whenever the suite runs.
TODAY = datetime.date(2026, 8, 20)


# ---------------------------------------------------------------------------
# The description of one synthetic issuer
# ---------------------------------------------------------------------------

@dataclass
class Period:
    """One reported period, as a bag of normalized field values.

    Field names are `finance.xbrl_mapping.CONCEPT_MAP` keys; the builder
    converts each into the concept the mapping actually resolves, so a canary
    describes ECONOMICS and the fixture describes XBRL.
    """

    end: str
    start: Optional[str] = None          # None for an annual period: derived
    fiscal_year: int = 2026
    fiscal_period: str = "FY"
    form: str = "10-K"
    filed: Optional[str] = None
    accession: Optional[str] = None
    values: Dict[str, float] = field(default_factory=dict)


@dataclass
class CanaryCompany:
    """One business-model class, in the smallest form that exercises it."""

    key: str
    symbol: str
    cik: int
    name: str
    sic: str
    sic_description: str
    price: float
    shares_outstanding: float
    market_cap: Optional[float] = None
    annual: List[Period] = field(default_factory=list)
    quarterly: List[Period] = field(default_factory=list)
    # us-gaap concepts the issuer reports that only its class reports. These
    # are what `finance.business_model` corroborates the SIC code against.
    corroborating_concepts: Tuple[str, ...] = ()
    # Extra concept -> [facts] entries the CONCEPT_MAP does not cover, used
    # where the point of the canary is a concept outside the standard map.
    extra_concepts: Dict[str, List[dict]] = field(default_factory=dict)
    guidance_release: Optional[dict] = None
    notes: str = ""

    def effective_market_cap(self) -> float:
        return self.market_cap if self.market_cap is not None \
            else self.price * self.shares_outstanding


# ---------------------------------------------------------------------------
# normalized field -> the us-gaap concept a real filing would carry it under
# ---------------------------------------------------------------------------
#
# The FIRST candidate in `finance.xbrl_mapping.CONCEPT_MAP` for each field,
# so a canary resolves through the same precedence a live filing does. Where
# a canary deliberately tests a LATER candidate (the foreign issuer's
# lease-inclusive debt line, say) it names the concept explicitly through
# `extra_concepts` instead.

_FIELD_CONCEPT = {
    "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", False),
    "operating_income": ("OperatingIncomeLoss", False),
    "net_income": ("NetIncomeLoss", False),
    "income_before_tax": (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItems"
        "NoncontrollingInterest", False),
    "income_tax_expense": ("IncomeTaxExpenseBenefit", False),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities", False),
    "capital_expenditure": ("PaymentsToAcquirePropertyPlantAndEquipment", False),
    "depreciation_and_amortization": ("DepreciationDepletionAndAmortization", False),
    "diluted_shares": ("WeightedAverageNumberOfDilutedSharesOutstanding", False),
    "diluted_eps": ("EarningsPerShareDiluted", False),
    "restructuring_charges": ("RestructuringCharges", False),
    "impairment_charges": ("AssetImpairmentCharges", False),

    "cash_and_cash_equivalents": ("CashAndCashEquivalentsAtCarryingValue", True),
    "short_term_investments": ("ShortTermInvestments", True),
    "assets": ("Assets", True),
    "current_assets": ("AssetsCurrent", True),
    "liabilities": ("Liabilities", True),
    "current_liabilities": ("LiabilitiesCurrent", True),
    "stockholders_equity": ("StockholdersEquity", True),
    "short_term_debt": ("ShortTermBorrowings", True),
    "current_portion_of_long_term_debt": ("LongTermDebtCurrent", True),
    "long_term_debt": ("LongTermDebtNoncurrent", True),
    "total_debt_combined": ("DebtLongtermAndShorttermCombinedAmount", True),
    "preferred_equity": ("PreferredStockValue", True),
    "minority_interest": ("MinorityInterest", True),
    "restricted_cash": ("RestrictedCashAndCashEquivalentsAtCarryingValue", True),
}

# us-gaap concepts reported in shares rather than dollars.
_SHARE_UNIT_CONCEPTS = {"WeightedAverageNumberOfDilutedSharesOutstanding"}
_PER_SHARE_CONCEPTS = {"EarningsPerShareDiluted"}


def _unit_for(concept: str) -> str:
    if concept in _SHARE_UNIT_CONCEPTS:
        return "shares"
    if concept in _PER_SHARE_CONCEPTS:
        return "USD/shares"
    return "USD"


def _annual_start(end: str) -> str:
    end_date = datetime.date.fromisoformat(end)
    try:
        return end_date.replace(year=end_date.year - 1).isoformat()
    except ValueError:                       # 29 February
        return end_date.replace(year=end_date.year - 1, day=28).isoformat()


def _default_filed(end: str) -> str:
    """A filing lands a few weeks after the period it covers."""
    return (datetime.date.fromisoformat(end) + datetime.timedelta(days=30)).isoformat()


# ---------------------------------------------------------------------------
# SEC payloads
# ---------------------------------------------------------------------------

def build_company_facts(company: CanaryCompany) -> dict:
    """The `companyfacts` payload for one canary.

    One entry per (concept, period). Values that a period does not report are
    simply absent — which is the point: a canary whose issuer genuinely does
    not tag `OperatingIncomeLoss` must produce a fixture in which that
    concept does not exist, not one where it exists and is zero.
    """
    us_gaap: Dict[str, dict] = {}

    def emit(concept: str, is_instant: bool, period: Period, value: float, index: int):
        entry = us_gaap.setdefault(concept, {
            "label": concept, "description": f"{concept} (canary fixture)",
            "units": {},
        })
        unit = _unit_for(concept)
        accession = period.accession or f"9999999999-{period.fiscal_year % 100:02d}-{index:06d}"
        fact = {
            "accn": accession,
            "end": period.end,
            "filed": period.filed or _default_filed(period.end),
            "form": period.form,
            "fp": period.fiscal_period,
            "fy": period.fiscal_year,
            "val": value,
        }
        if not is_instant:
            fact["start"] = period.start or _annual_start(period.end)
        entry["units"].setdefault(unit, []).append(fact)

    for index, period in enumerate(list(company.annual) + list(company.quarterly)):
        for field_name, value in period.values.items():
            mapping = _FIELD_CONCEPT.get(field_name)
            if mapping is None:
                raise KeyError(
                    f"{company.key}: no us-gaap concept is mapped for {field_name!r}. "
                    "Add it to _FIELD_CONCEPT or supply it through extra_concepts.")
            concept, is_instant = mapping
            emit(concept, is_instant, period, value, index)

    # Concepts whose PRESENCE is the evidence: a broker-dealer's customer
    # payables, an insurer's policy benefits. `finance.business_model`
    # corroborates the SIC code against these, so a canary that claims to be
    # an insurer files what an insurer files.
    latest = (company.annual or company.quarterly)[0]
    for concept in company.corroborating_concepts:
        us_gaap.setdefault(concept, {
            "label": concept, "description": f"{concept} (canary fixture)",
            "units": {"USD": [{
                "accn": latest.accession or "9999999999-26-000001",
                "end": latest.end, "filed": latest.filed or _default_filed(latest.end),
                "form": latest.form, "fp": latest.fiscal_period,
                "fy": latest.fiscal_year, "val": 0.0,
            }]},
        })

    for concept, facts in (company.extra_concepts or {}).items():
        entry = us_gaap.setdefault(concept, {
            "label": concept, "description": f"{concept} (canary fixture)", "units": {}})
        for fact in facts:
            entry["units"].setdefault(fact.get("unit", "USD"), []).append(
                {k: v for k, v in fact.items() if k != "unit"})

    return {"cik": company.cik, "entityName": company.name,
            "facts": {"us-gaap": us_gaap}}


def build_submissions(company: CanaryCompany) -> dict:
    """The submissions index: the SIC code, and the filing history.

    The SIC code is where business-model classification starts (spec §12),
    and the filing list is what structural-break and post-balance-sheet-event
    detection read.
    """
    periods = sorted(list(company.annual) + list(company.quarterly),
                     key=lambda p: p.filed or _default_filed(p.end), reverse=True)
    recent = {
        "accessionNumber": [], "filingDate": [], "reportDate": [], "form": [],
        "primaryDocument": [], "primaryDocDescription": [], "items": [],
        "isXBRL": [], "isInlineXBRL": [], "core_type": [],
        "acceptanceDateTime": [],
    }
    for index, period in enumerate(periods):
        filed = period.filed or _default_filed(period.end)
        recent["accessionNumber"].append(
            period.accession or f"9999999999-{period.fiscal_year % 100:02d}-{index:06d}")
        recent["filingDate"].append(filed)
        recent["reportDate"].append(period.end)
        recent["form"].append(period.form)
        recent["primaryDocument"].append(f"{company.symbol.lower()}-{period.end}.htm")
        recent["primaryDocDescription"].append(period.form)
        recent["items"].append("")
        recent["isXBRL"].append(1)
        recent["isInlineXBRL"].append(1)
        recent["core_type"].append("XBRL")
        recent["acceptanceDateTime"].append(f"{filed}T21:00:00.000Z")

    if company.guidance_release:
        release = company.guidance_release
        recent["accessionNumber"].insert(0, release["accession"])
        recent["filingDate"].insert(0, release["filed"])
        recent["reportDate"].insert(0, release["filed"])
        recent["form"].insert(0, "8-K")
        recent["primaryDocument"].insert(0, release["document"])
        recent["primaryDocDescription"].insert(0, "8-K")
        recent["items"].insert(0, "2.02,9.01")
        recent["isXBRL"].insert(0, 1)
        recent["isInlineXBRL"].insert(0, 1)
        recent["core_type"].insert(0, "XBRL")
        recent["acceptanceDateTime"].insert(0, f"{release['filed']}T21:00:00.000Z")

    return {
        "cik": str(company.cik).zfill(10),
        "name": company.name,
        "tickers": [company.symbol],
        "sic": company.sic,
        "sicDescription": company.sic_description,
        "filings": {"recent": recent},
    }


def build_filing_documents(company: CanaryCompany) -> Dict[str, str]:
    if not company.guidance_release:
        return {}
    release = company.guidance_release
    key = f"{release['accession']}/{release['document']}"
    index_key = f"{release['accession']}/{release['accession']}-index.html"
    return {
        key: release["text"],
        # The guidance path reads the filing index to find the exhibit.
        # EDGAR's own five-column layout: Seq | Description | Document |
        # Type | Size. `finance/guidance.py::select_exhibit_document` reads
        # the TYPE column, because EDGAR's index.json labels each file by its
        # icon rather than its exhibit type -- so a fixture that abbreviates
        # this table is testing a parser that would never run in production.
        index_key: (
            "<html><body><table>"
            "<tr><th>Seq</th><th>Description</th><th>Document</th>"
            "<th>Type</th><th>Size</th></tr>"
            "<tr><td>1</td><td>8-K</td>"
            f"<td><a href=\"/Archives/edgar/data/{company.cik}/x/"
            f"{company.symbol.lower()}-8k.htm\">{company.symbol.lower()}-8k.htm</a></td>"
            "<td>8-K</td><td>25776</td></tr>"
            "<tr><td>2</td><td>EX-99.1</td>"
            f"<td><a href=\"/Archives/edgar/data/{company.cik}/x/"
            f"{release['document']}\">{release['document']}</a></td>"
            "<td>EX-99.1</td><td>280576</td></tr>"
            "</table></body></html>"),
    }


# ---------------------------------------------------------------------------
# Yahoo payloads
# ---------------------------------------------------------------------------

def build_yahoo(company: CanaryCompany) -> Dict[str, dict]:
    """Market data only. Nothing fundamental comes from here.

    A canary's fundamentals are always SEC-sourced, so a failure can never be
    blamed on a vendor's own normalization — and the vendor sector string is
    deliberately generic, because spec §12 forbids classification from
    reading it at all.
    """
    bars = []
    day = TODAY - datetime.timedelta(days=400)
    price = company.price * 0.85
    index = 0
    while day <= TODAY:
        if day.weekday() < 5:
            # A deterministic, gently rising path: no randomness, so a
            # technical assertion means the same thing on every run.
            price = company.price * (0.85 + 0.15 * (index / 260.0))
            bars.append({
                "Date": f"{day.isoformat()}T00:00:00-04:00",
                "Open": round(price * 0.995, 4), "High": round(price * 1.01, 4),
                "Low": round(price * 0.99, 4), "Close": round(price, 4),
                "Adj Close": round(price, 4), "Volume": 1_000_000,
                "Dividends": 0.0, "Stock Splits": 0.0,
            })
            index += 1
        day += datetime.timedelta(days=1)
    bars[-1]["Close"] = company.price
    bars[-1]["Adj Close"] = company.price

    return {
        "stock_quote": {"quote": {
            "currency": "USD", "exchange": "NYQ",
            "lastPrice": company.price, "previousClose": round(company.price * 0.99, 4),
            "open": round(company.price * 0.995, 4),
            "dayHigh": round(company.price * 1.01, 4),
            "dayLow": round(company.price * 0.98, 4),
            "marketCap": company.effective_market_cap(),
            "shares": company.shares_outstanding,
            "lastVolume": 1_000_000, "tenDayAverageVolume": 1_000_000,
            "threeMonthAverageVolume": 1_000_000,
            "fiftyDayAverage": round(company.price * 0.98, 4),
            "twoHundredDayAverage": round(company.price * 0.95, 4),
            "yearHigh": round(company.price * 1.25, 4),
            "yearLow": round(company.price * 0.72, 4),
        }},
        "company_profile": {"profile": {
            "country": "United States", "currency": "USD", "exchange": "NYQ",
            "fullExchangeName": "NYSE", "industry": "Diversified",
            "sector": "Diversified", "fullTimeEmployees": 10_000,
            "longBusinessSummary": (
                f"{company.name} is a synthetic issuer built for the "
                f"{company.key} business-model canary. {company.notes}"),
            "longName": company.name, "shortName": company.name,
            "website": "https://example.invalid",
        }},
        "price_history": {"bars": bars, "interval": "1d", "period": "1y"},
        "corporate_actions": {"dividends": [], "splits": []},
        "analyst_estimates": {"analyst_price_targets": {
            "current": company.price, "high": round(company.price * 1.3, 2),
            "low": round(company.price * 0.8, 2), "mean": round(company.price * 1.05, 2),
            "median": round(company.price * 1.05, 2)}},
    }


def build_ticker_map(company: CanaryCompany) -> dict:
    return {company.symbol: {"cik_str": company.cik, "ticker": company.symbol,
                             "title": company.name}}


def build_fixture(company: CanaryCompany) -> dict:
    """Everything a canary run needs, in the same shape as a ticker fixture."""
    return {
        "cik": str(company.cik),
        "company_name": company.name,
        "note": f"Synthetic {company.key} canary. Every figure is invented.",
        "sec": {
            "company_facts": build_company_facts(company),
            "company_submissions": build_submissions(company),
            "filing_documents": build_filing_documents(company),
            "ticker_cik_map": build_ticker_map(company),
        },
        "yahoo": build_yahoo(company),
    }


def fixture_bytes(company: CanaryCompany) -> int:
    return len(json.dumps(build_fixture(company), default=str))
