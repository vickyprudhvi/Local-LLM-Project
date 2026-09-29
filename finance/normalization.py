"""Phase H.1 — deterministic normalization of provider payloads.

Turns raw Alpha Vantage JSON into structured records that carry their own
provenance, so nothing downstream has to guess where a number came from.

Rules this module will not break:

* Missing is missing. `"None"`, `"-"`, `""`, `null` and unparseable text all
  become `None` — never 0, never an interpolated estimate, never a carried-
  forward prior year. A zero and an absent value mean different things.
* Annual and quarterly periods are kept in SEPARATE lists and are never mixed
  into one series.
* A REPORTED annual figure and a CALCULATED trailing-twelve-month figure are
  labelled differently and never substituted for one another.
* The reported currency and the source units travel with every record.
* Inconsistent currency or period across statements raises a controlled warning
  rather than being silently reconciled.

Every normalized field records the dataset it came from and the fiscal date it
belongs to, which is what lets the final report separate provider-reported facts
from locally calculated metrics.
"""

import datetime
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_MISSING_TOKENS = frozenset({"", "none", "None", "-", "n/a", "na", "null", "nan"})
_FISCAL_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def parse_number(raw) -> Optional[float]:
    """Provider numbers arrive as strings. Anything not clearly numeric is None.

    Returning None (rather than 0.0) is deliberate: downstream metrics must be
    able to tell "reported as zero" from "not reported".
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return None if (value != value or value in (float("inf"), float("-inf"))) else value
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.lower() in _MISSING_TOKENS:
        return None
    text = text.replace(",", "").replace("$", "")
    try:
        value = float(text)
    except ValueError:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def parse_fiscal_date(raw) -> Optional[str]:
    """Normalize a fiscal date to ISO `YYYY-MM-DD`, or None."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not _FISCAL_DATE_RE.match(text):
        return None
    try:
        datetime.date.fromisoformat(text)
    except ValueError:
        return None
    return text


class PeriodType:
    ANNUAL = "annual"
    QUARTERLY = "quarterly"


class ValueBasis:
    """How a figure was arrived at. Never blurred."""

    REPORTED = "reported"                    # straight from the provider
    CALCULATED = "calculated"                # derived locally from reported values
    TRAILING_TWELVE_MONTHS = "trailing_ttm"  # locally summed from four quarters


@dataclass(frozen=True)
class FinancialPeriod:
    """One reporting period of one statement, with provenance attached."""

    fiscal_date: str
    period_type: str
    currency: Optional[str]
    values: Dict[str, Optional[float]]
    dataset_id: str
    basis: str = ValueBasis.REPORTED

    def get(self, key) -> Optional[float]:
        return self.values.get(key)

    def to_dict(self) -> dict:
        return {
            "fiscal_date": self.fiscal_date,
            "period_type": self.period_type,
            "currency": self.currency,
            "basis": self.basis,
            "dataset_id": self.dataset_id,
            "values": dict(self.values),
        }


@dataclass
class NormalizedStatements:
    """All statements for one symbol, plus every warning raised along the way."""

    symbol: str
    annual: Dict[str, List[FinancialPeriod]] = field(default_factory=dict)
    quarterly: Dict[str, List[FinancialPeriod]] = field(default_factory=dict)
    currency: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    provenance: Dict[str, dict] = field(default_factory=dict)

    def latest_annual(self, statement) -> Optional[FinancialPeriod]:
        periods = self.annual.get(statement) or []
        return periods[0] if periods else None

    def annual_series(self, statement, key) -> List[Tuple[str, Optional[float]]]:
        """(fiscal_date, value) newest-first. Absent values stay None."""
        return [(p.fiscal_date, p.get(key)) for p in self.annual.get(statement, [])]

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "currency": self.currency,
            "annual": {k: [p.to_dict() for p in v] for k, v in sorted(self.annual.items())},
            "quarterly": {k: [p.to_dict() for p in v]
                          for k, v in sorted(self.quarterly.items())},
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }


# Provider field -> our canonical name. Only fields we actually consume are
# mapped; an unmapped provider field is ignored rather than guessed at.
_INCOME_FIELDS = {
    "totalRevenue": "revenue",
    "grossProfit": "gross_profit",
    "operatingIncome": "operating_income",
    "netIncome": "net_income",
    "ebit": "ebit",
    "ebitda": "ebitda",
    "incomeBeforeTax": "pretax_income",
    "incomeTaxExpense": "tax_expense",
    "interestExpense": "interest_expense",
    "researchAndDevelopment": "research_and_development",
}
# Cash and debt are mapped to their SEPARATE reported components, never a
# single combined figure taken at face value — see _derive_balance_sheet_
# aggregates below for why. "*_provider_reported" fields are kept ONLY for
# reconciliation against the locally-derived sum; nothing downstream (metrics,
# DCF) reads them directly.
_BALANCE_FIELDS = {
    "totalAssets": "total_assets",
    "totalCurrentAssets": "current_assets",
    "totalCurrentLiabilities": "current_liabilities",
    "totalLiabilities": "total_liabilities",
    "totalShareholderEquity": "shareholder_equity",
    "cashAndCashEquivalentsAtCarryingValue": "cash_and_cash_equivalents",
    "shortTermInvestments": "short_term_investments",
    "cashAndShortTermInvestments": "cash_and_short_term_investments_provider_reported",
    "shortTermDebt": "short_term_debt",
    "currentLongTermDebt": "current_portion_of_long_term_debt",
    "longTermDebt": "long_term_debt",
    "shortLongTermDebtTotal": "total_debt_provider_reported",
    "inventory": "inventory",
    "commonStockSharesOutstanding": "shares_outstanding",
    "retainedEarnings": "retained_earnings",
}

# Reconciliation tolerances: how far a provider-reported combined/total figure
# may diverge from the locally-derived sum before it is flagged as a conflict
# rather than ordinary rounding. Debt gets a looser tolerance than cash+STI
# because provider "total debt" aggregates commonly include items (finance
# lease obligations, etc.) that this project's documented debt policy (see
# `_derive_balance_sheet_aggregates`) deliberately excludes.
_CASH_RECONCILE_RELATIVE_TOLERANCE = 0.01
_DEBT_RECONCILE_RELATIVE_TOLERANCE = 0.05
_RECONCILE_ABSOLUTE_FLOOR = 1.0


def _reconciles(derived, reported, relative_tolerance) -> bool:
    """True when there is nothing to compare, or the two values agree within
    tolerance. False only on a genuine, material conflict."""
    if derived is None or reported is None:
        return True
    tolerance = max(abs(derived) * relative_tolerance, _RECONCILE_ABSOLUTE_FLOOR)
    return abs(derived - reported) <= tolerance


def _derive_balance_sheet_aggregates(values: dict):
    """Compute cash-and-short-term-investments and total-debt from their
    DOCUMENTED, SEPARATELY-REPORTED components. Mutates `values` in place.

    A provider-reported combined field is NEVER trusted at face value: Alpha
    Vantage's own `cashAndShortTermInvestments` has been observed, for a real
    issuer, to silently equal cash-and-equivalents alone while a genuine
    nonzero `shortTermInvestments` field sat right next to it unused — see
    docs/security/... no, see the MSFT regression fixture in
    tests/fixtures/msft_balance_sheet_regression.json. The combined figure
    used everywhere downstream is always cash_and_cash_equivalents +
    short_term_investments, computed here, ONLY WHEN BOTH are known (never
    guessed from one alone).

    Debt aggregation policy (documented): total_debt = short_term_debt +
    current_portion_of_long_term_debt + long_term_debt. `long_term_debt` is
    treated as the NON-CURRENT portion — Alpha Vantage reports the current
    slice separately as `currentLongTermDebt` precisely so the two are not
    double-counted. total_debt is the sum of whichever of the three components
    are actually known (missing ones excluded, not zero-filled), so a company
    that doesn't break out a current portion still gets a usable total from
    the components it does report; which components fed it is always visible
    in the record (values stay individually accessible). Finance/capital
    lease obligations are NOT included in this project's total_debt (a
    documented limitation, not a silent omission — see
    docs/PHASE_H1_STOCK_ANALYSIS.md).
    """
    cash = values.get("cash_and_cash_equivalents")
    sti = values.get("short_term_investments")
    values["cash_and_short_term_investments"] = (
        cash + sti if cash is not None and sti is not None else None)

    debt_parts = [values.get(k) for k in
                  ("short_term_debt", "current_portion_of_long_term_debt", "long_term_debt")]
    known_debt_parts = [p for p in debt_parts if p is not None]
    values["total_debt"] = sum(known_debt_parts) if known_debt_parts else None


def _normalize_cash_flow_signs(values: dict):
    """Capital expenditure's sign varies by provider/feed (some report it as
    a negative outflow line within the cash-flow statement, others as a
    positive magnitude). Normalized HERE to one documented convention —
    outflow_positive (capital_expenditure is always >= 0, representing the
    amount spent) — so nothing downstream (propose_assumptions' CapEx/
    revenue hierarchy, the DCF) has to guess or re-derive the sign itself.
    `capital_expenditure_sign_convention` records that this normalization
    happened, rather than silently changing the number with no trace.
    """
    capex = values.get("capital_expenditure")
    if capex is not None:
        values["capital_expenditure"] = abs(capex)
        values["capital_expenditure_sign_convention"] = "outflow_positive"


def _balance_sheet_reconciliation_warnings(dataset_id, latest_period: "FinancialPeriod"):
    """Compare the LATEST annual period's derived aggregates against whatever
    provider-reported combined figures exist. Restricted to the latest period
    (not every historical year) so a systemic provider data-quality issue
    produces one clear warning instead of one per fiscal year."""
    v = latest_period.values
    warnings = []

    derived_cash_sti = v.get("cash_and_short_term_investments")
    reported_cash_sti = v.get("cash_and_short_term_investments_provider_reported")
    if not _reconciles(derived_cash_sti, reported_cash_sti, _CASH_RECONCILE_RELATIVE_TOLERANCE):
        warnings.append(
            f"{dataset_id} {latest_period.fiscal_date}: the provider's own "
            f"cash-and-short-term-investments figure ({reported_cash_sti:,.0f}) does not "
            f"reconcile with cash_and_cash_equivalents + short_term_investments computed "
            f"from separately reported components ({derived_cash_sti:,.0f}); the derived "
            "sum is used, and cash is never reported as if it already included "
            "short-term investments.")

    derived_debt = v.get("total_debt")
    reported_debt = v.get("total_debt_provider_reported")
    if not _reconciles(derived_debt, reported_debt, _DEBT_RECONCILE_RELATIVE_TOLERANCE):
        warnings.append(
            f"{dataset_id} {latest_period.fiscal_date}: the provider's own total-debt "
            f"figure ({reported_debt:,.0f}) does not reconcile with short_term_debt + "
            f"current_portion_of_long_term_debt + long_term_debt "
            f"({derived_debt:,.0f}); the derived sum is used under this project's "
            "documented debt policy, which excludes items the provider figure may include "
            "(e.g. lease obligations).")
    return warnings
_CASHFLOW_FIELDS = {
    "operatingCashflow": "operating_cash_flow",
    "capitalExpenditures": "capital_expenditure",
    "depreciationDepletionAndAmortization": "depreciation_amortization",
    "dividendPayout": "dividends_paid",
    "changeInOperatingLiabilities": "change_in_operating_liabilities",
    "changeInOperatingAssets": "change_in_operating_assets",
    "netIncome": "net_income",
}

_STATEMENT_FIELD_MAPS = {
    "income_statement": _INCOME_FIELDS,
    "balance_sheet": _BALANCE_FIELDS,
    "cash_flow": _CASHFLOW_FIELDS,
}


def normalize_statement(payload, dataset_id, symbol) -> Tuple[List[FinancialPeriod],
                                                              List[FinancialPeriod],
                                                              List[str]]:
    """Normalize one statement payload into (annual, quarterly, warnings)."""
    warnings = []
    field_map = _STATEMENT_FIELD_MAPS.get(dataset_id)
    if field_map is None:
        return [], [], [f"No field mapping is defined for {dataset_id!r}."]
    if not isinstance(payload, dict):
        return [], [], [f"The {dataset_id} payload was not an object."]

    def build(reports, period_type):
        out = []
        if not isinstance(reports, list):
            return out
        for report in reports:
            if not isinstance(report, dict):
                continue
            fiscal_date = parse_fiscal_date(report.get("fiscalDateEnding"))
            if fiscal_date is None:
                warnings.append(
                    f"{dataset_id}: a {period_type} period had an unusable fiscal date "
                    "and was skipped.")
                continue
            currency = report.get("reportedCurrency")
            currency = currency.strip().upper() if isinstance(currency, str) and \
                currency.strip() else None
            values = {canonical: parse_number(report.get(provider_key))
                      for provider_key, canonical in field_map.items()}
            if dataset_id == "balance_sheet":
                # A no-op for income_statement/cash_flow: their field maps
                # never populate these keys, so both stay None.
                _derive_balance_sheet_aggregates(values)
            elif dataset_id == "cash_flow":
                _normalize_cash_flow_signs(values)
            out.append(FinancialPeriod(
                fiscal_date=fiscal_date, period_type=period_type, currency=currency,
                values=values, dataset_id=dataset_id, basis=ValueBasis.REPORTED))
        # Newest first, deterministically.
        out.sort(key=lambda p: p.fiscal_date, reverse=True)
        return out

    annual = build(payload.get("annualReports"), PeriodType.ANNUAL)
    quarterly = build(payload.get("quarterlyReports"), PeriodType.QUARTERLY)

    if dataset_id == "balance_sheet" and annual:
        warnings.extend(_balance_sheet_reconciliation_warnings(dataset_id, annual[0]))

    currencies = {p.currency for p in annual + quarterly if p.currency}
    if len(currencies) > 1:
        warnings.append(
            f"{dataset_id}: periods are reported in more than one currency "
            f"({', '.join(sorted(currencies))}); figures were NOT converted.")
    return annual, quarterly, warnings


def normalize_all(payloads, symbol) -> NormalizedStatements:
    """Normalize every statement payload for one symbol.

    `payloads` maps dataset_id -> (payload, provenance dict).
    """
    result = NormalizedStatements(symbol=symbol)
    currencies = set()

    for dataset_id in ("income_statement", "balance_sheet", "cash_flow"):
        entry = payloads.get(dataset_id)
        if entry is None:
            continue
        payload, provenance = entry
        annual, quarterly, warnings = normalize_statement(payload, dataset_id, symbol)
        result.annual[dataset_id] = annual
        result.quarterly[dataset_id] = quarterly
        result.warnings.extend(warnings)
        result.provenance[dataset_id] = dict(provenance or {})
        currencies.update(p.currency for p in annual if p.currency)

    if len(currencies) > 1:
        result.warnings.append(
            "Statements are reported in more than one currency "
            f"({', '.join(sorted(currencies))}); no conversion was applied.")
    result.currency = next(iter(currencies)) if len(currencies) == 1 else None

    _warn_on_period_misalignment(result)
    return result


def _warn_on_period_misalignment(result: NormalizedStatements):
    """Flag statements whose latest annual periods do not line up.

    Comparing revenue from FY2025 against capex from FY2024 would silently
    corrupt every derived metric, so it is surfaced rather than reconciled.
    """
    latest = {}
    for dataset_id, periods in result.annual.items():
        if periods:
            latest[dataset_id] = periods[0].fiscal_date
    if len(set(latest.values())) > 1:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(latest.items()))
        result.warnings.append(
            f"The latest annual periods do not align across statements ({detail}); "
            "metrics combining them may mix reporting periods.")


def normalize_overview(payload, provenance=None) -> dict:
    """Normalize the company overview into typed fields with provenance."""
    if not isinstance(payload, dict):
        return {"available": False, "warnings": ["The overview payload was not an object."]}

    numeric = {
        "market_capitalisation": "MarketCapitalization",
        "pe_ratio": "PERatio",
        "forward_pe": "ForwardPE",
        "peg_ratio": "PEGRatio",
        "price_to_book": "PriceToBookRatio",
        "price_to_sales_ttm": "PriceToSalesRatioTTM",
        "ev_to_ebitda": "EVToEBITDA",
        "profit_margin": "ProfitMargin",
        "operating_margin_ttm": "OperatingMarginTTM",
        "return_on_equity_ttm": "ReturnOnEquityTTM",
        "return_on_assets_ttm": "ReturnOnAssetsTTM",
        "diluted_eps_ttm": "DilutedEPSTTM",
        "revenue_ttm": "RevenueTTM",
        "ebitda": "EBITDA",
        "beta": "Beta",
        "dividend_yield": "DividendYield",
        "shares_outstanding": "SharesOutstanding",
        "book_value": "BookValue",
        "week_52_high": "52WeekHigh",
        "week_52_low": "52WeekLow",
    }
    text = {
        "name": "Name", "description": "Description", "sector": "Sector",
        "industry": "Industry", "exchange": "Exchange", "country": "Country",
        "currency": "Currency", "fiscal_year_end": "FiscalYearEnd",
    }

    out = {"available": bool(payload), "warnings": []}
    for canonical, provider_key in numeric.items():
        out[canonical] = parse_number(payload.get(provider_key))
    for canonical, provider_key in text.items():
        value = payload.get(provider_key)
        out[canonical] = value.strip() if isinstance(value, str) and value.strip() else None

    missing = [k for k in ("shares_outstanding", "market_capitalisation") if out.get(k) is None]
    if missing:
        out["warnings"].append(
            f"The overview is missing {', '.join(missing)}; any metric needing them "
            "cannot be calculated.")
    out["provenance"] = dict(provenance or {})
    return out


def normalize_quote(payload, provenance=None) -> dict:
    """Normalize the latest-price payload.

    Freshness is reported as DELAYED because that is what the free tier returns.
    Nothing here claims a realtime price.
    """
    quote = payload.get("Global Quote") if isinstance(payload, dict) else None
    if not isinstance(quote, dict) or not quote:
        return {"available": False, "warnings": ["No quote data was returned."],
                "provenance": dict(provenance or {})}
    return {
        "available": True,
        "symbol": (quote.get("01. symbol") or "").strip().upper() or None,
        "price": parse_number(quote.get("05. price")),
        "open": parse_number(quote.get("02. open")),
        "high": parse_number(quote.get("03. high")),
        "low": parse_number(quote.get("04. low")),
        "volume": parse_number(quote.get("06. volume")),
        "latest_trading_day": parse_fiscal_date(quote.get("07. latest trading day")),
        "previous_close": parse_number(quote.get("08. previous close")),
        "change": parse_number(quote.get("09. change")),
        "change_percent": parse_number((quote.get("10. change percent") or "").rstrip("%")),
        "price_basis": "delayed",
        "warnings": [],
        "provenance": dict(provenance or {}),
    }


def normalize_yahoo_overview(payload, provenance=None) -> dict:
    """Yahoo equivalent of normalize_overview() -- SAME output contract, so
    finance/workflow.py::build_facts consumes either source unchanged.
    `payload` is finance/yahoo_provider.py's {"profile": {...curated .info
    subset...}}."""
    profile = (payload or {}).get("profile") if isinstance(payload, dict) else None
    if not isinstance(profile, dict) or not profile:
        return {"available": False, "warnings": ["No profile data was returned."],
                "provenance": dict(provenance or {})}
    out = {
        "available": True, "warnings": [],
        "name": profile.get("shortName") or profile.get("longName"),
        "description": profile.get("longBusinessSummary"),
        "sector": profile.get("sector"), "industry": profile.get("industry"),
        "exchange": profile.get("fullExchangeName") or profile.get("exchange"),
        "country": profile.get("country"), "currency": profile.get("currency"),
        "fiscal_year_end": None,  # not exposed by the curated Yahoo profile fields
        "market_capitalisation": parse_number(profile.get("marketCap")),
        "shares_outstanding": parse_number(profile.get("sharesOutstanding")),
        # Yahoo's curated profile intentionally excludes valuation ratios
        # (pe_ratio, peg_ratio, ...) -- those are quote-adjacent, not profile
        # data, and this project already gets them from fundamental_metrics'
        # own deterministic calculation rather than trusting a provider ratio.
        "pe_ratio": None, "forward_pe": None, "peg_ratio": None,
        "price_to_book": None, "price_to_sales_ttm": None, "ev_to_ebitda": None,
        "profit_margin": None, "operating_margin_ttm": None,
        "return_on_equity_ttm": None, "return_on_assets_ttm": None,
        "diluted_eps_ttm": None, "revenue_ttm": None, "ebitda": None,
        "beta": None, "dividend_yield": None, "book_value": None,
        "week_52_high": None, "week_52_low": None,
    }
    missing = [k for k in ("shares_outstanding", "market_capitalisation") if out.get(k) is None]
    if missing:
        out["warnings"].append(
            f"The Yahoo profile is missing {', '.join(missing)}; any metric needing "
            "them cannot be calculated.")
    out["provenance"] = dict(provenance or {})
    return out


def normalize_yahoo_quote(payload, provenance=None) -> dict:
    """Yahoo equivalent of normalize_quote() -- SAME output contract.
    `payload` is finance/yahoo_provider.py's {"quote": {...fast_info...}}.
    Yahoo's fast_info has no single "change"/"change_percent" field the way
    Alpha Vantage's Global Quote does; both are DERIVED here from
    lastPrice/previousClose, same arithmetic a caller would otherwise do."""
    quote = (payload or {}).get("quote") if isinstance(payload, dict) else None
    if not isinstance(quote, dict) or quote.get("lastPrice") is None:
        return {"available": False, "warnings": ["No quote data was returned."],
                "provenance": dict(provenance or {})}
    price = parse_number(quote.get("lastPrice"))
    previous_close = parse_number(quote.get("previousClose"))
    change = (price - previous_close) if price is not None and previous_close is not None else None
    change_percent = (change / previous_close * 100.0) if change is not None and previous_close else None
    return {
        "available": True,
        "symbol": None,  # not carried in fast_info; caller already knows the requested symbol
        "price": price,
        "open": parse_number(quote.get("open")),
        "high": parse_number(quote.get("dayHigh")),
        "low": parse_number(quote.get("dayLow")),
        "volume": parse_number(quote.get("lastVolume")),
        "latest_trading_day": None,  # fast_info carries no explicit trading-day timestamp
        "previous_close": previous_close,
        "change": round(change, 6) if change is not None else None,
        "change_percent": round(change_percent, 6) if change_percent is not None else None,
        "price_basis": "delayed",
        "warnings": [],
        "provenance": dict(provenance or {}),
    }


def normalize_yahoo_price_history(payload, provenance=None) -> dict:
    """Yahoo equivalent of normalize_price_history() -- SAME output contract.
    `payload` is finance/yahoo_provider.py's {"bars": [...], "period":,
    "interval":}. Fetched with auto_adjust=False, so unlike Alpha Vantage's
    free tier, a genuine split/dividend-adjusted close IS available here
    (yfinance's own "Adj Close" column) -- reported as `adjusted=True`."""
    bars_raw = (payload or {}).get("bars") if isinstance(payload, dict) else None
    if not isinstance(bars_raw, list) or not bars_raw:
        return {"available": False, "bars": [], "dropped": 0,
                "warnings": ["No price history was returned."],
                "provenance": dict(provenance or {})}

    bars, dropped = [], 0
    for raw in bars_raw:
        if not isinstance(raw, dict):
            dropped += 1
            continue
        date_raw = raw.get("Date")
        date = parse_fiscal_date(str(date_raw)[:10]) if date_raw else None
        close = parse_number(raw.get("Close"))
        adjusted = parse_number(raw.get("Adj Close"))
        if date is None or (close is None and adjusted is None):
            dropped += 1
            continue
        reference = close if close is not None else adjusted
        bars.append(PriceBar(
            date=date,
            open=parse_number(raw.get("Open")) or reference,
            high=parse_number(raw.get("High")) or reference,
            low=parse_number(raw.get("Low")) or reference,
            close=reference,
            adjusted_close=adjusted if adjusted is not None else reference,
            volume=parse_number(raw.get("Volume")) or 0.0,
        ))
    bars.sort(key=lambda b: b.date)

    warnings = []
    if dropped:
        warnings.append(f"{dropped} price bar(s) were unusable and were excluded.")
    return {"available": bool(bars), "bars": bars, "dropped": dropped,
            "adjusted": True, "warnings": warnings, "provenance": dict(provenance or {})}


@dataclass(frozen=True)
class PriceBar:
    date: str
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: float


def normalize_price_history(payload, provenance=None) -> dict:
    """Normalize adjusted daily bars into an OLDEST-FIRST series.

    A bar with an unusable date or a missing close is dropped and counted, never
    interpolated — an invented price would silently corrupt every indicator.
    """
    series = None
    if isinstance(payload, dict):
        for key in payload:
            if isinstance(key, str) and key.lower().startswith("time series"):
                series = payload[key]
                break
    if not isinstance(series, dict) or not series:
        return {"available": False, "bars": [], "dropped": 0,
                "warnings": ["No price history was returned."],
                "provenance": dict(provenance or {})}

    # Two different response shapes share this series, and key "5." means
    # DIFFERENT things in each:
    #   TIME_SERIES_DAILY           1.open 2.high 3.low 4.close 5.volume
    #   TIME_SERIES_DAILY_ADJUSTED  1.open 2.high 3.low 4.close 5.adjusted close
    #                               6.volume 7.dividend 8.split coefficient
    # Reading "5." positionally would silently treat VOLUME as a price. Detect
    # the shape by the presence of the explicitly-named adjusted-close key.
    sample = next((v for v in series.values() if isinstance(v, dict)), {})
    is_adjusted = "5. adjusted close" in sample

    bars, dropped = [], 0
    for raw_date, raw in series.items():
        date = parse_fiscal_date(raw_date)
        if date is None or not isinstance(raw, dict):
            dropped += 1
            continue
        close = parse_number(raw.get("4. close"))
        adjusted = parse_number(raw.get("5. adjusted close")) if is_adjusted else None
        volume = parse_number(raw.get("6. volume") if is_adjusted else raw.get("5. volume"))
        if close is None and adjusted is None:
            dropped += 1
            continue
        reference = close if close is not None else adjusted
        bars.append(PriceBar(
            date=date,
            open=parse_number(raw.get("1. open")) or reference,
            high=parse_number(raw.get("2. high")) or reference,
            low=parse_number(raw.get("3. low")) or reference,
            close=reference,
            # Without the premium endpoint there is no adjusted close; using the
            # raw close is correct and is reported via `adjusted`, so nothing
            # downstream mistakes it for split/dividend-adjusted data.
            adjusted_close=adjusted if adjusted is not None else reference,
            volume=volume or 0.0,
        ))
    bars.sort(key=lambda b: b.date)

    warnings = []
    if dropped:
        warnings.append(f"{dropped} price bar(s) were unusable and were excluded.")
    if not is_adjusted and bars:
        warnings.append(
            "Price history is NOT split/dividend adjusted (the adjusted endpoint "
            "requires a premium Alpha Vantage plan), so long-run returns and "
            "moving averages may be distorted across any split or large dividend.")
    return {"available": bool(bars), "bars": bars, "dropped": dropped,
            "adjusted": is_adjusted, "warnings": warnings,
            "provenance": dict(provenance or {})}


def normalize_earnings(payload, provenance=None, max_annual=5, max_quarterly=8) -> dict:
    """Normalize EARNINGS into a BOUNDED summary.

    Alpha Vantage's EARNINGS endpoint can return 80+ quarterly entries; the raw
    payload must never be forwarded verbatim (that was the earlier
    `facts["earnings_raw"]` shape — a major, uncontrolled contributor to the
    83k-token synthesis prompt this fixes). `max_annual`/`max_quarterly` bound
    how many of the newest periods are kept; the ORIGINAL reported counts are
    preserved too, so truncation is visible rather than silent.
    """
    if not isinstance(payload, dict):
        return {"available": False, "warnings": ["The earnings payload was not an object."],
                "provenance": dict(provenance or {})}

    def annual_row(report):
        return {
            "fiscal_date": parse_fiscal_date(report.get("fiscalDateEnding")),
            "reported_eps": parse_number(report.get("reportedEPS")),
        }

    def quarterly_row(report):
        return {
            "fiscal_date": parse_fiscal_date(report.get("fiscalDateEnding")),
            "reported_date": parse_fiscal_date(report.get("reportedDate")),
            "reported_eps": parse_number(report.get("reportedEPS")),
            "estimated_eps": parse_number(report.get("estimatedEPS")),
            "surprise": parse_number(report.get("surprise")),
            "surprise_percent": parse_number(report.get("surprisePercentage")),
        }

    annual_raw = payload.get("annualEarnings")
    quarterly_raw = payload.get("quarterlyEarnings")
    annual_raw = annual_raw if isinstance(annual_raw, list) else []
    quarterly_raw = quarterly_raw if isinstance(quarterly_raw, list) else []

    annual = [annual_row(r) for r in annual_raw[:max_annual] if isinstance(r, dict)]
    quarterly = [quarterly_row(r) for r in quarterly_raw[:max_quarterly] if isinstance(r, dict)]
    latest_quarterly = quarterly[0] if quarterly else None

    warnings = []
    if len(annual_raw) > max_annual or len(quarterly_raw) > max_quarterly:
        warnings.append(
            f"Earnings history was truncated to the most recent {max_annual} annual and "
            f"{max_quarterly} quarterly period(s) of {len(annual_raw)} and "
            f"{len(quarterly_raw)} reported.")

    return {
        "available": bool(annual or quarterly),
        "annual": annual,
        "quarterly_recent": quarterly,
        "latest_quarterly_surprise_percent": (
            latest_quarterly.get("surprise_percent") if latest_quarterly else None),
        "total_annual_periods_reported": len(annual_raw),
        "total_quarterly_periods_reported": len(quarterly_raw),
        "warnings": warnings,
        "provenance": dict(provenance or {}),
    }
