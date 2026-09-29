"""Phase H.1 — locally calculated metrics.

Everything here is computed from data already in the cache, so it costs no
provider quota. Technical indicators in particular are calculated from cached
OHLCV rather than from Alpha Vantage's dedicated SMA/EMA/RSI/MACD endpoints:
each of those would be a separate billable call for a number we can derive
exactly, and on a 25-call daily budget that is the difference between one
analysis and none.

Every metric carries:
  * `formula` — how it was calculated
  * `calculation_version` — bumped when a formula changes
  * `inputs` — the fiscal periods or bar count it consumed
  * `value` — `None` when an input was missing, never a filled-in guess

A metric whose inputs are absent reports `None` with a `reason`. It is never
approximated, carried forward, or defaulted to zero.
"""

import math
from typing import List, Optional

from finance.dcf import NetDebtPolicy, compute_net_debt
from finance.normalization import NormalizedStatements, PriceBar, ValueBasis

CALCULATION_VERSION = "metrics_v1"

# H.4 corrective patch — negative-equity applicability. A ratio dividing by
# non-positive shareholder equity (ROE, debt-to-equity) is mathematically
# computable but economically misleading (e.g. ROE = -198% reads like a
# catastrophic profitability collapse when it is actually an artifact of a
# negative denominator — common for companies with large buyback programs or
# accumulated-deficit histories). Such a metric is reported as `value: None`
# with `status: "not_meaningful"` and a stable, machine-readable `reason`
# code (as opposed to the free-text sentence every other missing metric
# carries) so callers — the report renderer, the research-pipeline prompts —
# can branch on it deterministically instead of string-matching prose.
STATUS_NOT_MEANINGFUL = "not_meaningful"
REASON_NEGATIVE_SHAREHOLDER_EQUITY = "negative_shareholder_equity"
REASON_NEGATIVE_AVERAGE_SHAREHOLDER_EQUITY = "negative_average_shareholder_equity"


def _metric(name, value, formula, inputs, reason=None, unit=None, accounting_basis=None,
           status=None) -> dict:
    out = {
        "metric": name,
        "value": value,
        "formula": formula,
        "inputs": inputs,
        "calculation_version": CALCULATION_VERSION,
        # This is the VALUE basis (calculated here vs. reported vs. TTM) —
        # distinct from `accounting_basis` (GAAP vs. adjusted), which is a
        # separate, optional field so the two are never confused.
        "basis": ValueBasis.CALCULATED,
    }
    if unit:
        out["unit"] = unit
    if accounting_basis:
        out["accounting_basis"] = accounting_basis
    if status:
        out["status"] = status
    if value is None:
        out["reason"] = reason or "A required input was not reported."
    return out


def _safe_divide(numerator, denominator):
    """Division that refuses to invent a result.

    None in, None out; a zero or non-finite denominator yields None rather than
    an exception or an infinity.
    """
    if numerator is None or denominator is None:
        return None
    try:
        if denominator == 0:
            return None
        result = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _growth(current, prior):
    """Period-over-period growth. Undefined when the prior value is <= 0, because
    a percentage change from a negative or zero base is not meaningful."""
    if current is None or prior is None or prior <= 0:
        return None
    return (float(current) - float(prior)) / float(prior)


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def fundamental_metrics(statements: NormalizedStatements) -> List[dict]:
    """Compute the fundamental metric set from normalized ANNUAL periods only.

    Annual and quarterly are never mixed: a margin built from annual revenue and
    quarterly costs would be silently wrong.
    """
    income = statements.annual.get("income_statement") or []
    balance = statements.annual.get("balance_sheet") or []
    cash = statements.annual.get("cash_flow") or []

    metrics = []
    latest_income = income[0] if income else None
    prior_income = income[1] if len(income) > 1 else None
    latest_balance = balance[0] if balance else None
    latest_cash = cash[0] if cash else None

    income_periods = [p.fiscal_date for p in income[:2]]
    balance_periods = [p.fiscal_date for p in balance[:1]]
    cash_periods = [p.fiscal_date for p in cash[:1]]

    # -- growth --
    # Alpha Vantage's INCOME_STATEMENT reports GAAP figures only; no adjusted /
    # non-GAAP series is available from this provider. Both growth metrics are
    # labeled accordingly so neither the report nor the LLM can describe a GAAP
    # number as "underlying" or "adjusted" operating growth — see Problem 5.
    # If a reviewed, separately-provenanced adjusted-figures source is ever
    # added, it must store its own metric under its own name, never overwrite
    # these.
    metrics.append(_metric(
        "revenue_growth_yoy",
        _growth(latest_income.get("revenue") if latest_income else None,
                prior_income.get("revenue") if prior_income else None),
        "(revenue_t - revenue_{t-1}) / revenue_{t-1}",
        income_periods, unit="ratio", accounting_basis="GAAP"))
    metrics.append(_metric(
        "net_income_growth_yoy",
        _growth(latest_income.get("net_income") if latest_income else None,
                prior_income.get("net_income") if prior_income else None),
        "(net_income_t - net_income_{t-1}) / net_income_{t-1}",
        income_periods, unit="ratio", accounting_basis="GAAP"))
    metrics.append(_metric(
        "revenue_cagr",
        _cagr([p.get("revenue") for p in income]),
        "(revenue_latest / revenue_oldest)^(1/periods) - 1",
        [p.fiscal_date for p in income], unit="ratio"))

    # -- margins --
    revenue = latest_income.get("revenue") if latest_income else None
    metrics.append(_metric(
        "gross_margin",
        _safe_divide(latest_income.get("gross_profit") if latest_income else None, revenue),
        "gross_profit / revenue", income_periods[:1], unit="ratio"))
    metrics.append(_metric(
        "operating_margin",
        _safe_divide(latest_income.get("operating_income") if latest_income else None,
                     revenue),
        "operating_income / revenue", income_periods[:1], unit="ratio"))
    metrics.append(_metric(
        "net_margin",
        _safe_divide(latest_income.get("net_income") if latest_income else None, revenue),
        "net_income / revenue", income_periods[:1], unit="ratio"))

    # -- free cash flow --
    operating_cf = latest_cash.get("operating_cash_flow") if latest_cash else None
    capex = latest_cash.get("capital_expenditure") if latest_cash else None
    # A directly reported pass-through, not a calculation -- exposed as its
    # own metric (COR corrective patch, Phase 9) purely so it gets a stable
    # evidence ID via finance/evidence.py's generic fundamental_metrics loop;
    # it was already used internally to derive free_cash_flow below but had
    # no citable ID of its own.
    metrics.append(_metric(
        "operating_cash_flow", operating_cf,
        "directly reported operating cash flow", cash_periods, unit="currency"))
    free_cash_flow = (operating_cf - abs(capex)
                      if operating_cf is not None and capex is not None else None)
    metrics.append(_metric(
        "free_cash_flow", free_cash_flow,
        "operating_cash_flow - |capital_expenditure|", cash_periods, unit="currency"))
    metrics.append(_metric(
        "free_cash_flow_margin", _safe_divide(free_cash_flow, revenue),
        "free_cash_flow / revenue", cash_periods + income_periods[:1], unit="ratio"))

    # -- balance sheet health --
    metrics.append(_metric(
        "current_ratio",
        _safe_divide(latest_balance.get("current_assets") if latest_balance else None,
                     latest_balance.get("current_liabilities") if latest_balance else None),
        "current_assets / current_liabilities", balance_periods, unit="ratio"))

    # total_debt is already the reconciled aggregate from normalization (short-
    # term debt + current portion of long-term debt + long-term debt) — reused
    # here rather than re-derived, so this can never regress to the
    # current-portion-of-LTD undercount Problem 2 identified.
    total_debt = latest_balance.get("total_debt") if latest_balance else None
    equity = latest_balance.get("shareholder_equity") if latest_balance else None
    prior_balance = balance[1] if len(balance) > 1 else None
    prior_equity = prior_balance.get("shareholder_equity") if prior_balance else None

    # H.4 corrective patch: a directly reported pass-through, exposed as its
    # own metric purely so negative (or any) shareholder equity gets a stable
    # evidence ID via finance/evidence.py's generic fundamental_metrics loop —
    # same rationale as operating_cash_flow above. This is what lets a
    # negative-equity company's equity value remain CITABLE even when the
    # ratios computed from it (roe_ending_equity, roe_average_equity,
    # debt_to_equity) are reported as not_meaningful below: the underlying
    # fact is never hidden, only the misleading ratio is suppressed.
    metrics.append(_metric(
        "shareholder_equity", equity,
        "directly reported total shareholder equity", balance_periods, unit="currency"))
    # Same rationale for total_debt: the RiskReviewer is instructed (see
    # finance/research_pipeline.py's _risk_reviewer_prompt) to prefer
    # total_debt/net_debt/debt_to_fcf/current_ratio/operating_cash_flow over
    # debt_to_equity when equity is negative -- those metrics must actually
    # be citable for that instruction to be followable at all.
    metrics.append(_metric(
        "total_debt", total_debt,
        "short_term_debt + current_portion_of_long_term_debt + long_term_debt",
        balance_periods, unit="currency"))

    # H.4 corrective patch: debt_to_equity divided by non-positive equity is
    # mathematically computable but economically misleading (a small negative
    # denominator can produce an enormous or wildly-signed ratio that reads
    # like a leverage judgment it cannot actually support) -- reported as
    # not_meaningful rather than as an ordinary ratio. The equity figure
    # itself stays available above (shareholder_equity), never hidden.
    if equity is not None and equity <= 0:
        metrics.append(_metric(
            "debt_to_equity", None,
            "total_debt / shareholder_equity "
            "(total_debt = short_term_debt + current_portion_of_long_term_debt + long_term_debt)",
            balance_periods, unit="ratio", status=STATUS_NOT_MEANINGFUL,
            reason=REASON_NEGATIVE_SHAREHOLDER_EQUITY))
    else:
        metrics.append(_metric(
            "debt_to_equity", _safe_divide(total_debt, equity),
            "total_debt / shareholder_equity "
            "(total_debt = short_term_debt + current_portion_of_long_term_debt + long_term_debt)",
            balance_periods, unit="ratio"))

    # COR corrective patch (Phase 9): debt_to_equity alone can overstate
    # balance-sheet risk when it is high mainly because the equity base is
    # small rather than because absolute debt is large (COR's own motivating
    # case: debt_to_equity ~5.08, driven by ~$1.5B equity against ~$7.7B
    # debt). finance/research_pipeline.py's _risk_reviewer_prompt instructs
    # the RiskReviewer to weigh these alongside debt_to_equity, never treat
    # debt_to_equity in isolation as sufficient evidence of severe risk.
    # net_debt reuses finance.dcf.compute_net_debt's CASH_ONLY policy --
    # documented there as "the ONLY place net debt is computed" -- rather
    # than re-deriving the same subtraction a second, potentially-drifting
    # way; None (never 0) when an input is genuinely unreported.
    cash = latest_balance.get("cash_and_cash_equivalents") if latest_balance else None
    net_debt = (compute_net_debt(total_debt, cash, None, NetDebtPolicy.CASH_ONLY)
               if total_debt is not None and cash is not None else None)
    metrics.append(_metric(
        "net_debt", net_debt,
        "total_debt - cash_and_cash_equivalents (cash_only policy, matching the DCF default)",
        balance_periods, unit="currency"))
    metrics.append(_metric(
        "net_debt_to_fcf", _safe_divide(net_debt, free_cash_flow),
        "net_debt / free_cash_flow", cash_periods + balance_periods, unit="ratio"))
    metrics.append(_metric(
        "debt_to_fcf", _safe_divide(total_debt, free_cash_flow),
        "total_debt / free_cash_flow", cash_periods + balance_periods, unit="ratio"))

    # Interest coverage: only when interest_expense was actually reported.
    # This project's SEC concept map (finance/xbrl_mapping.py) does not
    # currently capture an interest-expense concept, so this is None (with
    # the standard reason) for SEC-sourced statements; genuinely computed
    # when the income-statement provider does report it (e.g. Alpha
    # Vantage's own interestExpense field). Never invented when absent --
    # this is the "if supported" the corrective patch asked for.
    interest_expense = latest_income.get("interest_expense") if latest_income else None
    ebit_for_coverage = ((latest_income.get("ebit") or latest_income.get("operating_income"))
                        if latest_income else None)
    metrics.append(_metric(
        "interest_coverage", _safe_divide(ebit_for_coverage, interest_expense),
        "EBIT / interest_expense", income_periods[:1], unit="ratio"))

    # -- ROE: two methodologies, never presented as a single undifferentiated
    # "ROE". `roe_ending_equity` matches how a quick back-of-envelope figure is
    # usually computed; `roe_average_equity` matches how most providers
    # (including Alpha Vantage's own OVERVIEW.ReturnOnEquityTTM, kept
    # separately under `facts.overview` — see finance/normalization.py)
    # actually compute it, and needs the PRIOR period's equity.
    # H.4 corrective patch: net_income / a non-positive ending equity is
    # mathematically computable but economically misleading (e.g. a small
    # negative equity base turns an ordinary net income into an ROE reading
    # like -198%, which looks like a profitability collapse it does not
    # actually represent) -- reported as not_meaningful rather than as an
    # ordinary percentage. shareholder_equity itself remains a separate,
    # citable metric above regardless.
    if equity is not None and equity <= 0:
        metrics.append(_metric(
            "roe_ending_equity", None,
            "net_income / ending_shareholder_equity",
            income_periods[:1] + balance_periods, unit="ratio", status=STATUS_NOT_MEANINGFUL,
            reason=REASON_NEGATIVE_SHAREHOLDER_EQUITY))
    else:
        metrics.append(_metric(
            "roe_ending_equity",
            _safe_divide(latest_income.get("net_income") if latest_income else None, equity),
            "net_income / ending_shareholder_equity",
            income_periods[:1] + balance_periods, unit="ratio"))

    if prior_equity is not None and equity is not None:
        average_equity = (equity + prior_equity) / 2.0
        roe_average_periods = income_periods[:1] + [p.fiscal_date for p in balance[:2]]
        if average_equity <= 0:
            metrics.append(_metric(
                "roe_average_equity", None,
                "net_income / ((beginning_shareholder_equity + ending_shareholder_equity) / 2)",
                roe_average_periods, unit="ratio", status=STATUS_NOT_MEANINGFUL,
                reason=REASON_NEGATIVE_AVERAGE_SHAREHOLDER_EQUITY))
        else:
            roe_average = _safe_divide(
                latest_income.get("net_income") if latest_income else None, average_equity)
            metrics.append(_metric(
                "roe_average_equity", roe_average,
                "net_income / ((beginning_shareholder_equity + ending_shareholder_equity) / 2)",
                roe_average_periods, unit="ratio"))
    else:
        metrics.append(_metric(
            "roe_average_equity", None,
            "net_income / ((beginning_shareholder_equity + ending_shareholder_equity) / 2)",
            balance_periods, unit="ratio",
            reason="The prior period's shareholder equity was not available."))

    # -- ROIC (only when every input is present) --
    metrics.append(_roic(latest_income, latest_balance,
                         income_periods[:1] + balance_periods))

    # -- share count trend --
    share_series = [(p.fiscal_date, p.get("shares_outstanding")) for p in balance]
    metrics.append(_metric(
        "share_count_change",
        _growth(share_series[0][1] if share_series else None,
                share_series[1][1] if len(share_series) > 1 else None),
        "(shares_t - shares_{t-1}) / shares_{t-1}",
        [d for d, _ in share_series[:2]], unit="ratio"))

    return metrics


def _cagr(values):
    """Compound annual growth over the reported span, newest-first input."""
    clean = [v for v in values if v is not None and v > 0]
    if len(clean) < 2:
        return None
    latest, oldest = clean[0], clean[-1]
    periods = len(clean) - 1
    try:
        return (latest / oldest) ** (1.0 / periods) - 1.0
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _roic(income, balance, periods):
    """Return on invested capital — only when every input is genuinely reported."""
    formula = "EBIT x (1 - effective_tax_rate) / (total_debt + shareholder_equity)"
    if income is None or balance is None:
        return _metric("return_on_invested_capital", None, formula, periods,
                       "Income statement or balance sheet was unavailable.")

    ebit = income.get("ebit")
    if ebit is None:
        ebit = income.get("operating_income")
    pretax = income.get("pretax_income")
    tax_expense = income.get("tax_expense")
    equity = balance.get("shareholder_equity")
    # Reuses the reconciled total_debt from normalization (short-term debt +
    # current portion of long-term debt + long-term debt) rather than
    # re-deriving it here — this used to sum only short+long term debt,
    # undercounting invested capital by the same current-portion-of-LTD gap
    # Problem 2 identified for debt_to_equity.
    total_debt = balance.get("total_debt")

    if ebit is None or equity is None:
        return _metric("return_on_invested_capital", None, formula, periods,
                       "EBIT or shareholder equity was not reported.")
    effective_tax = _safe_divide(tax_expense, pretax)
    if effective_tax is None or not (0.0 <= effective_tax < 1.0):
        return _metric("return_on_invested_capital", None, formula, periods,
                       "A usable effective tax rate could not be derived.")
    invested = (total_debt or 0.0) + equity
    value = _safe_divide(ebit * (1.0 - effective_tax), invested)
    return _metric("return_on_invested_capital", value, formula, periods,
                   "Invested capital was zero or unusable.", unit="ratio")


# ---------------------------------------------------------------------------
# Technical indicators — all from cached OHLCV, zero extra provider calls
# ---------------------------------------------------------------------------

def _closes(bars: List[PriceBar]) -> List[float]:
    return [b.adjusted_close for b in bars if b.adjusted_close is not None]


def simple_moving_average(values, window) -> Optional[float]:
    if len(values) < window or window <= 0:
        return None
    return sum(values[-window:]) / float(window)


def exponential_moving_average(values, window) -> Optional[float]:
    """Standard EMA seeded with the SMA of the first `window` observations."""
    if len(values) < window or window <= 0:
        return None
    multiplier = 2.0 / (window + 1.0)
    ema = sum(values[:window]) / float(window)
    for value in values[window:]:
        ema = (value - ema) * multiplier + ema
    return ema


def relative_strength_index(values, window=14) -> Optional[float]:
    """Wilder's RSI. None when there is not enough history."""
    if len(values) <= window:
        return None
    gains, losses = [], []
    for previous, current in zip(values, values[1:]):
        change = current - previous
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    average_gain = sum(gains[:window]) / float(window)
    average_loss = sum(losses[:window]) / float(window)
    for gain, loss in zip(gains[window:], losses[window:]):
        average_gain = (average_gain * (window - 1) + gain) / float(window)
        average_loss = (average_loss * (window - 1) + loss) / float(window)

    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0
    rs = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(values, fast=12, slow=26, signal=9):
    """MACD line, signal line and histogram. None when history is too short."""
    if len(values) < slow + signal:
        return None, None, None
    fast_ema = exponential_moving_average(values, fast)
    slow_ema = exponential_moving_average(values, slow)
    if fast_ema is None or slow_ema is None:
        return None, None, None
    macd_line = fast_ema - slow_ema

    # Build the MACD history so the signal line is a real EMA of it.
    history = []
    for end in range(slow, len(values) + 1):
        window = values[:end]
        f = exponential_moving_average(window, fast)
        s = exponential_moving_average(window, slow)
        if f is not None and s is not None:
            history.append(f - s)
    signal_line = exponential_moving_average(history, signal) if len(history) >= signal else None
    histogram = macd_line - signal_line if signal_line is not None else None
    return macd_line, signal_line, histogram


def annualized_volatility(values, trading_days=252) -> Optional[float]:
    """Annualized standard deviation of daily log returns."""
    if len(values) < 3:
        return None
    returns = []
    for previous, current in zip(values, values[1:]):
        if previous <= 0 or current <= 0:
            continue
        returns.append(math.log(current / previous))
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(trading_days)


def maximum_drawdown(values) -> Optional[float]:
    """Largest peak-to-trough decline over the series, as a negative ratio."""
    if len(values) < 2:
        return None
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, (value - peak) / peak)
    return worst


def total_return(values) -> Optional[float]:
    if len(values) < 2 or values[0] <= 0:
        return None
    return (values[-1] - values[0]) / values[0]


def technical_metrics(bars: List[PriceBar]) -> List[dict]:
    """Every indicator we can derive from the cached bars we actually have."""
    closes = _closes(bars)
    span = [bars[0].date, bars[-1].date] if bars else []
    count = len(closes)
    inputs = {"bar_count": count, "date_range": span}

    def add(name, value, formula, reason=None, unit=None):
        return _metric(name, value, formula, inputs, reason, unit)

    metrics = [
        add("latest_close", closes[-1] if closes else None,
            "most recent adjusted close", unit="currency"),
        add("sma_20", simple_moving_average(closes, 20),
            "mean of the last 20 adjusted closes", "Fewer than 20 bars available.",
            unit="currency"),
        add("sma_50", simple_moving_average(closes, 50),
            "mean of the last 50 adjusted closes", "Fewer than 50 bars available.",
            unit="currency"),
        add("sma_200", simple_moving_average(closes, 200),
            "mean of the last 200 adjusted closes", "Fewer than 200 bars available.",
            unit="currency"),
        add("ema_12", exponential_moving_average(closes, 12),
            "12-period EMA seeded with the 12-period SMA", unit="currency"),
        add("ema_26", exponential_moving_average(closes, 26),
            "26-period EMA seeded with the 26-period SMA", unit="currency"),
        add("rsi_14", relative_strength_index(closes, 14),
            "Wilder's 14-period RSI", "Fewer than 15 bars available.", unit="index"),
        add("annualized_volatility", annualized_volatility(closes),
            "stdev(daily log returns) x sqrt(252)", unit="ratio"),
        add("maximum_drawdown", maximum_drawdown(closes),
            "min((value - running_peak) / running_peak)", unit="ratio"),
        add("total_return_over_window", total_return(closes),
            "(last_close - first_close) / first_close", unit="ratio"),
    ]

    macd_line, signal_line, histogram = macd(closes)
    metrics.append(add("macd_line", macd_line, "EMA(12) - EMA(26)",
                       "Insufficient history for MACD."))
    metrics.append(add("macd_signal", signal_line, "EMA(9) of the MACD line",
                       "Insufficient history for the MACD signal."))
    metrics.append(add("macd_histogram", histogram, "MACD line - signal line",
                       "Insufficient history for the MACD histogram."))

    # Trend classification is a FACT about the moving averages, not a forecast.
    latest = closes[-1] if closes else None
    sma_50 = simple_moving_average(closes, 50)
    sma_200 = simple_moving_average(closes, 200)
    if latest is not None and sma_50 is not None and sma_200 is not None:
        if latest > sma_50 > sma_200:
            trend = "above both the 50- and 200-day averages"
        elif latest < sma_50 < sma_200:
            trend = "below both the 50- and 200-day averages"
        else:
            trend = "mixed relative to the 50- and 200-day averages"
    else:
        trend = None
    metrics.append(add("price_vs_moving_averages", trend,
                       "compare latest close to SMA(50) and SMA(200)",
                       "Fewer than 200 bars available."))
    return metrics


def metrics_to_dict(metrics) -> dict:
    """Name -> metric, for the structured fact block handed to the local LLM."""
    return {m["metric"]: m for m in metrics}


def available_only(metrics) -> dict:
    return {m["metric"]: m["value"] for m in metrics if m["value"] is not None}


def missing_only(metrics) -> dict:
    return {m["metric"]: m.get("reason") for m in metrics if m["value"] is None}
