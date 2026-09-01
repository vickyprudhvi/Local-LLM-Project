"""Parts 9-13 — the report boundary the renderer was standing in for.

The compact renderer had grown into the last unaudited decision-maker in the
system. It chose which of two revenue figures to show, whether a ratio was
current or a year old, whether "free cash flow" was an honest label for this
issuer, whether a price-versus-value percentage was safe to print, whether a
risk belonged to the company or to the analysis, and whether a condition was
worth stating -- all from raw `compact.get(...)` reads, in text-formatting
functions, with no test able to reach the decision without going through the
markdown.

Those are finance decisions. They belong where every other finance decision
in this system now lives: in a validated object built once from canonical
inputs, testable on its own, with the renderer downstream of it.

So this module holds `StockAnalysisReportModel` -- display-ready SEMANTIC
FACTS, never raw provider data -- and one deterministic builder. After it,
the renderer's job is genuinely only formatting: numbers into strings,
strings into headings, bullets into a list.

The division is stated once here and is the rule for every future change:

    this module   decides WHETHER a figure may be shown, WHAT it means,
                  and WHICH of several candidates is the current one
    the renderer  decides how many decimal places it gets
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from finance import guidance as guidance_module
from finance import metric_policy
from finance.dcf_packet import STATUS_EXPLANATION, ValuationStatus
from finance.metrics import STATUS_NOT_MEANINGFUL


# ---------------------------------------------------------------------------
# Value kinds — what a number IS, so the renderer knows how to print it
# ---------------------------------------------------------------------------

class ValueKind:
    """How a figure should be rendered, decided by what it measures.

    This is the ONLY formatting-adjacent thing the model carries, and it is
    here rather than in the renderer because it follows from the metric's
    identity: a margin is a percentage because it is a ratio of two
    currencies, not because someone chose to print it that way.
    """

    CURRENCY = "currency"
    PRICE = "price"
    PERCENT = "percent"
    RATIO = "ratio"
    TEXT = "text"


@dataclass(frozen=True)
class SnapshotRow:
    """One Snapshot line: a settled label, a settled value, a settled kind.

    `text` is set only where the row is deliberately NOT a number -- a metric
    that is not meaningful against negative equity, for instance. A row with
    both would be two answers to one question, so the builder sets exactly
    one.
    """

    label: str
    kind: str = ValueKind.CURRENCY
    value: Optional[float] = None
    text: Optional[str] = None


@dataclass(frozen=True)
class ReportStatus:
    headline: str
    reason: Optional[str] = None
    missing_datasets: Tuple[str, ...] = ()
    unofficial_source_note: Optional[str] = None


@dataclass(frozen=True)
class ValuationBasis:
    """Which periods, which guidance, which caveats. Every field decided."""

    financial_base_label: Optional[str] = None
    financial_base_caveat: Optional[str] = None
    balance_sheet_label: Optional[str] = None
    balance_sheet_kind: Optional[str] = None
    guidance_lines: Tuple[str, ...] = ()
    freshness_note: Optional[str] = None
    suitability_note: Optional[str] = None
    method_lines: Tuple[str, ...] = ()
    unavailable_line: Optional[str] = None
    share_basis_note: Optional[str] = None


@dataclass(frozen=True)
class ValuationView:
    """What the report may say about value, and nothing it may not.

    Every "shown" flag was a branch in the renderer reading a different
    field. They are decided together here because they are one decision:
    whether this analysis established a modelled value a reader may act on.
    """

    status: str = ValuationStatus.INPUT_PACKET_INVALID
    publishable: bool = False
    market_price: Optional[float] = None
    basis: ValuationBasis = field(default_factory=ValuationBasis)
    bear_value_per_share: Optional[float] = None
    base_value_per_share: Optional[float] = None
    bull_value_per_share: Optional[float] = None
    premium_pct: Optional[float] = None
    modeled_return_pct: Optional[float] = None
    comparison_withheld_reason: Optional[str] = None
    model_invalid_status: Optional[str] = None
    model_invalid_explanation: Optional[str] = None
    model_invalid_reasons: Tuple[str, ...] = ()
    assumption_lines: Tuple[str, ...] = ()
    sensitivity_level: Optional[str] = None
    terminal_dependence_note: Optional[str] = None
    unavailable_reason: Optional[str] = None


@dataclass(frozen=True)
class TechnicalView:
    trend_sentence: Optional[str] = None
    rsi: Optional[float] = None
    macd_histogram: Optional[float] = None
    macd_sign: Optional[str] = None
    return_label: Optional[str] = None
    total_return: Optional[float] = None
    annualized_volatility: Optional[float] = None


def _reject_empty(label: str, texts) -> None:
    """A published claim has to say something, and say it to a reader.

    Two contracts, one place, because they fail the same way. The report
    model is the last boundary before a reader, and both refusals are
    STRUCTURAL rather than filters: a bad bullet silently dropped here looks
    identical to a claim that was never made, and the stage that produced it
    goes on producing them. Refusing construction surfaces the failure where
    it can be fixed. Filtering in the renderer -- which is where both were
    noticed -- would have hidden them one layer further down.

    1. CONTENT. "", "   ", ".", "N/A" and "TBD" are placeholders. The
       minimum-content rule already applied to claims and risks at the schema
       boundary; an analysis limitation reached the report through a field it
       had not been applied to.

    2. AUDIENCE. "see dcf.warnings / dcf.validation_reasons" is a field path
       inside this program. A reader cannot open it and it tells them nothing
       except that the sentence was written for somebody else.
    """
    from finance.content_policy import contains_internal_reference

    for text in texts or ():
        if not _carries_content(text):
            raise ValueError(
                f"{label} cannot contain an empty claim; {text!r} is a placeholder, and a "
                "bullet that says nothing is a published claim that says nothing")
        if contains_internal_reference(text):
            raise ValueError(
                f"{label} cannot contain internal implementation text; {text[:60]!r} names "
                "a field inside this program rather than something a reader can act on")


# Placeholder text that is not empty and is not a claim either. Kept beside
# the schema's own `_MIN_CLAIM_CHARS` rule rather than duplicating the
# threshold: this boundary is about the SHAPE of a non-answer, and the schema
# is about length.
_PLACEHOLDER_TEXT = frozenset({
    "", ".", "-", "--", "...", "n/a", "na", "n.a.", "tbd", "tba", "none",
    "no", "unknown", "?", "pending",
})


def _carries_content(text) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    return bool(stripped) and stripped.lower().rstrip(".") not in _PLACEHOLDER_TEXT


@dataclass(frozen=True)
class ClaimSet:
    claims: Tuple[str, ...] = ()
    unavailable_note: Optional[str] = None

    def __post_init__(self):
        _reject_empty("a research claim set", self.claims)


@dataclass(frozen=True)
class RiskItem:
    text: str
    severity: Optional[str] = None

    def __post_init__(self):
        _reject_empty("a risk item", (self.text,))


@dataclass(frozen=True)
class ResearchView:
    pipeline_status: str = "DISABLED"
    readiness_status: Optional[str] = None
    readiness_reason: Optional[str] = None
    available: bool = False
    research_stance: Optional[str] = None
    valuation_view: Optional[str] = None
    overall_risk: Optional[str] = None
    confidence: Optional[float] = None
    confidence_band: Optional[str] = None
    recommendation: Optional[str] = None
    primary_reason: Optional[str] = None
    condition_groups: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class SourceLine:
    provider: str
    labels: Tuple[str, ...] = ()


@dataclass(frozen=True)
class StockAnalysisReportModel:
    """Everything the report says, decided. Nothing it says, formatted."""

    symbol: str
    status: ReportStatus
    snapshot: Tuple[SnapshotRow, ...] = ()
    valuation: ValuationView = field(default_factory=ValuationView)
    technical: TechnicalView = field(default_factory=TechnicalView)
    bull_case: ClaimSet = field(default_factory=ClaimSet)
    bear_case: ClaimSet = field(default_factory=ClaimSet)
    company_risk: Tuple[RiskItem, ...] = ()
    analysis_limitations: Tuple[RiskItem, ...] = ()
    risk_unavailable_note: Optional[str] = None
    research_pipeline: str = "DISABLED"
    research_readiness: Optional[str] = None
    research_view: ResearchView = field(default_factory=ResearchView)
    recommendation: Optional[str] = None
    conditions: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    sources: Tuple[SourceLine, ...] = ()
    missing_dataset_note: Optional[str] = None
    currency: str = "USD"

    def __post_init__(self):
        # `ClaimSet` and `RiskItem` guard their own text. Everything else the
        # model holds that a reader will SEE is checked here, so the contract
        # covers the whole surface rather than the two fields it started on.
        for label, items in self.conditions or ():
            _reject_empty(f"the {label!r} conditions", items)
        view = self.research_view
        for label, items in (view.condition_groups or ()):
            _reject_empty(f"the {label!r} conditions", items)
        _reject_empty("the research readiness reason",
                      [t for t in (view.readiness_reason,) if t])
        _reject_empty("the recommendation reason",
                      [t for t in (view.primary_reason,) if t])
        _reject_empty("the analysis status",
                      [t for t in (self.status.reason,) if t])

    # -- questions the tests and the renderer both ask -------------------
    def snapshot_values(self) -> Dict[str, Optional[float]]:
        return {row.label: row.value for row in self.snapshot}

    def numeric_snapshot_values(self) -> List[float]:
        return [row.value for row in self.snapshot if row.value is not None]

    def all_risk_text(self) -> List[str]:
        return [item.text for item in self.company_risk + self.analysis_limitations]


# ---------------------------------------------------------------------------
# Part 10 — the builder
# ---------------------------------------------------------------------------

_MONTH_ABBREVIATIONS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Spec §7's own worked example reads "Confidence: 45% (low)", so the low band
# runs to 0.50 -- deliberately conservative, since the failure this guards
# against is a hedged number being read with unhedged conviction.
_CONFIDENCE_BANDS = ((0.50, "low"), (0.70, "moderate"))

_PROVIDER_DISPLAY_NAME = {"yahoo": "Yahoo Finance", "sec": "SEC",
                          "alphavantage": "Alpha Vantage"}
_DATASET_SOURCE_LABEL = {
    "quote": "market data", "stock_quote": "market data",
    "price_history": "price history", "daily_prices": "price history",
    "daily_prices_adjusted": "price history",
    "company_profile": "company profile", "company_overview": "company profile",
    "us_fundamentals": "filings", "income_statement": "filings",
    "balance_sheet": "filings", "cash_flow": "filings",
    "corporate_actions": "dividends/splits",
    "analyst_estimates": "analyst estimates", "earnings": "earnings/news",
    "news": "earnings/news",
}

_NOT_MEANINGFUL_REASON_TEXT: Dict[str, str] = {}


def period_label(date_text) -> Optional[str]:
    """'2026-04-30' -> '30 Apr 2026'.

    An explicit date needs no fiscal-calendar inference and cannot be
    misread. The previous calendar-quarter shorthand turned an April
    fiscal-year end into "Q2 2026" and produced a line that contradicted
    itself: "Balance sheet: Q2 2026 (latest annual filing)".
    """
    if not isinstance(date_text, str) or len(date_text) < 10:
        return None
    try:
        year, month, day = (int(date_text[0:4]), int(date_text[5:7]), int(date_text[8:10]))
        return f"{day} {_MONTH_ABBREVIATIONS[month - 1]} {year}"
    except (ValueError, IndexError):
        return None


def confidence_band(confidence: float) -> str:
    for ceiling, label in _CONFIDENCE_BANDS:
        if confidence < ceiling:
            return label
    return "high"


def return_window_label(total_return_metric) -> str:
    """'1Y Return' ONLY when the underlying window really is about a year.

    A ~100-session window is not a year and must not be labelled as one --
    the same "never describe data as something it is not" rule that makes
    `price_basis` always say "delayed".
    """
    import datetime

    inputs = (total_return_metric or {}).get("inputs") or {}
    date_range = inputs.get("date_range") or []
    if len(date_range) == 2:
        try:
            start = datetime.date.fromisoformat(str(date_range[0])[:10])
            end = datetime.date.fromisoformat(str(date_range[1])[:10])
            days = (end - start).days
        except (ValueError, TypeError):
            days = None
        if days is not None:
            if 300 <= days <= 420:
                return "1Y Return"
            if days > 0:
                return f"Return ({days}d)"
    return "Return (period)"


def _join_labels(labels) -> str:
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def _build_snapshot(compact: dict) -> Tuple[SnapshotRow, ...]:
    """Which figure each row shows, and what it is honestly called.

    Every choice here used to be a branch in `_snapshot_rows`. The canonical
    CURRENT value wins wherever one exists; the annual entry is the labelled
    fallback for an issuer with no newer components; and a metric whose
    ordinary name asserts economics this business model does not support is
    named for the subtraction that was actually performed.
    """
    from finance.metrics import (REASON_NEGATIVE_AVERAGE_SHAREHOLDER_EQUITY,
                                 REASON_NEGATIVE_SHAREHOLDER_EQUITY)
    reason_text = {
        REASON_NEGATIVE_SHAREHOLDER_EQUITY: "negative equity",
        REASON_NEGATIVE_AVERAGE_SHAREHOLDER_EQUITY: "negative average equity",
    }

    quote = compact.get("quote") or {}
    company = compact.get("company") or {}
    fundamentals = compact.get("fundamental_metrics") or {}
    technicals = compact.get("technical_metrics") or {}
    canonical_current = (compact.get("canonical_evidence") or {}).get("current") or {}
    income_periods = (compact.get("financial_history") or {}).get("income_statement") or []
    latest_income = (income_periods[0].get("values") or {}) if income_periods else {}

    def fm(name) -> dict:
        entry = fundamentals.get(name)
        return entry if isinstance(entry, dict) else {}

    rows: List[SnapshotRow] = []

    def add(label, value, kind=ValueKind.CURRENCY, text=None):
        if value is None and text is None:
            return
        rows.append(SnapshotRow(label=label, value=value, kind=kind, text=text))

    def flow(field_name, fallback):
        """The canonical current value and the period it covers, or the
        annual fallback with no period claim attached."""
        selection = canonical_current.get(field_name)
        if isinstance(selection, dict) and selection.get("value") is not None:
            suffix = (" (TTM)" if selection.get("period_type") in ("TTM", "DERIVED")
                      else " (FY)")
            return selection["value"], suffix
        return fallback, ""

    add("Price", quote.get("price"), ValueKind.PRICE)
    add("Market Cap", company.get("market_capitalisation"))

    revenue, revenue_suffix = flow("revenue", latest_income.get("revenue"))
    add(f"Revenue{revenue_suffix}", revenue)

    growth_metric = canonical_current.get("revenue_growth") or {}
    growth_label = (compact.get("canonical_evidence") or {}).get("current_growth_label")
    if growth_metric.get("value") is not None:
        heading = f"Revenue Growth ({growth_label})" if growth_label else "Revenue Growth"
        add(heading, growth_metric["value"], ValueKind.PERCENT)
    elif fm("revenue_growth_yoy").get("value") is not None:
        # Nothing current could be built at all. Say which period this is,
        # rather than letting a bare heading imply the current one.
        add("Revenue Growth (last fiscal year YoY)",
            fm("revenue_growth_yoy").get("value"), ValueKind.PERCENT)

    net_income, income_suffix = flow("net_income", latest_income.get("net_income"))
    add(f"Net Income{income_suffix}", net_income)

    fcf_value, fcf_suffix = flow("free_cash_flow", fm("free_cash_flow").get("value"))
    # §12: "FCF" asserts owner economics. Where the business model does not
    # support that claim the row is named for what was actually computed, so
    # the Snapshot and the Valuation section cannot describe one number two
    # different ways.
    fcf_label = ((compact.get("business_model_evidence") or {}).get("cash_flow_label")
                 or "FCF")
    add(f"{fcf_label}{fcf_suffix}", fcf_value)

    canonical_net_debt = canonical_current.get("net_debt") or {}
    net_debt = canonical_net_debt.get("value")
    if net_debt is None:
        # §15's remaining unlabelled fallback, now labelled. An annual net
        # debt shown under a bare "Net Debt" heading beside a quarterly
        # balance-sheet date is the same two-periods-one-name failure the
        # rest of the Snapshot was fixed for.
        state_net_debt = (compact.get("current_financial_state") or {}).get("net_debt")
        if state_net_debt is not None:
            add("Net Debt", state_net_debt)
        elif fm("net_debt").get("value") is not None:
            add("Net Debt (FY)", fm("net_debt").get("value"))
    else:
        add("Net Debt", net_debt)

    operating_margin = canonical_current.get("operating_margin") or {}
    if operating_margin.get("value") is not None:
        add("Operating Margin (TTM)", operating_margin["value"], ValueKind.PERCENT)
    else:
        add("Operating Margin", fm("operating_margin").get("value"), ValueKind.PERCENT)

    def instant_ratio(label, key) -> bool:
        current = canonical_current.get(key) or {}
        if current.get("value") is not None:
            add(label, current["value"], ValueKind.RATIO)
            return True
        return False

    if not instant_ratio("Current Ratio", "current_ratio"):
        annual = fm("current_ratio")
        if annual.get("value") is not None:
            add("Current Ratio (FY)", annual.get("value"), ValueKind.RATIO)

    for label, metric_name, kind in (("ROE", "roe_ending_equity", ValueKind.PERCENT),
                                     ("Debt-to-Equity", "debt_to_equity", ValueKind.RATIO)):
        if metric_name == "debt_to_equity" and instant_ratio("Debt-to-Equity",
                                                             "debt_to_equity"):
            continue
        entry = fm(metric_name)
        if entry.get("status") == STATUS_NOT_MEANINGFUL:
            # A not-meaningful verdict is about the DENOMINATOR, not the
            # period: it holds on any basis, so the row keeps the plain
            # label and says why instead of quoting a number.
            reason = reason_text.get(entry.get("reason"), "not meaningful")
            add(label, None, ValueKind.TEXT, text=f"N/M — {reason}")
        elif entry.get("value") is not None:
            # §7. These reach this line only from the HISTORICAL namespace --
            # ROE has no current derivation at all, and debt-to-equity is
            # here only when the canonical instant form could not be built.
            # A bare heading beside trailing-twelve-month figures above it is
            # the two-periods-one-name ambiguity the Snapshot was fixed for.
            add(f"{label} (FY)", entry["value"], kind)

    total_return_entry = technicals.get("total_return_over_window") or {}
    add(return_window_label(total_return_entry), total_return_entry.get("value"),
        ValueKind.PERCENT)

    return tuple(rows)


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------

def _build_valuation_basis(compact: dict) -> ValuationBasis:
    basis = compact.get("dcf_financial_basis") or {}
    guidance = compact.get("management_guidance") or {}

    financial_base_label = None
    financial_base_caveat = None
    flow_end = basis.get("flow_period_end")
    validation = basis.get("flow_base_validation")
    # §6/24: the label may say "trailing twelve months" only when a
    # twelve-month window was actually constructed AND validated. Deriving it
    # from `base_revenue_basis` alone printed that phrase above a headline
    # revenue that was the prior fiscal year's.
    if basis.get("base_revenue_basis") == "ttm_calculation" and flow_end \
            and validation in (None, "valid", "partial"):
        financial_base_label = ("Financial base: trailing twelve months to "
                                f"{period_label(flow_end) or flow_end}")
        if validation == "partial":
            financial_base_caveat = (
                "  (that twelve-month window ends before this company's latest "
                "reported period — see the research view)")
    elif flow_end:
        financial_base_label = (f"Financial base: fiscal year to "
                                f"{period_label(flow_end) or flow_end}")

    balance_label = balance_kind = None
    balance_as_of = basis.get("balance_sheet_as_of")
    if balance_as_of:
        balance_label = period_label(balance_as_of) or balance_as_of
        balance_kind = ("latest quarterly filing"
                        if basis.get("balance_sheet_source") == "quarterly_sec_filing"
                        else "fiscal year end, latest annual filing")

    guidance_lines: List[str] = []
    metrics = (guidance or {}).get("metrics") or {}
    coverage_matrix = compact.get("guidance_matrix") or {}
    if metrics:
        period = basis.get("guidance_period")
        if period:
            guidance_lines.append(f"Management guidance: {period} current guidance")
            issued_with = basis.get("guidance_issued_with")
            if issued_with and issued_with != period:
                guidance_lines.append(f"  Issued with {issued_with} results")
        else:
            fiscal_year = guidance.get("fiscal_year")
            guidance_lines.append(
                f"Management guidance: FY{fiscal_year} current guidance"
                if fiscal_year else "Management guidance: current guidance")
        guided = sorted(name for name in metrics)
        if guided:
            guidance_lines.append(f"  (guided metrics: {', '.join(guided)})")
        if coverage_matrix.get("dcf_rows_missing"):
            guidance_lines.append(
                f"  Coverage: {guidance_module.guidance_summary_line(coverage_matrix)}")
    elif coverage_matrix.get("current_rows"):
        # Partial coverage is not absence. A live issuer published current
        # capital-expenditure guidance and no revenue guidance, and the
        # report said guidance was unavailable.
        guidance_lines.append("Management guidance: partial")
        guidance_lines.append(
            f"  {guidance_module.guidance_summary_line(coverage_matrix)}")
    else:
        # "Unavailable" asserts the company published nothing. What is known
        # is that the extractor found nothing, which is a weaker claim and
        # the only one the evidence supports.
        examined = compact.get("guidance_releases_examined")
        if examined:
            guidance_lines.append(
                f"Management guidance: none extracted ({examined} SEC earnings release"
                f"{'s' if examined != 1 else ''} examined)")
        elif examined == 0:
            guidance_lines.append("Management guidance: no SEC earnings release found")
        else:
            guidance_lines.append("Management guidance: not retrieved")

    freshness_note = None
    freshness = basis.get("valuation_freshness")
    if freshness and freshness != "CURRENT":
        freshness_note = f"Valuation freshness: {freshness.replace('_', ' ').lower()}"

    suitability_note = None
    suitability = (compact.get("dcf_suitability") or {}).get("dcf_suitability")
    if suitability and suitability != "SUITABLE":
        suitability_note = f"DCF suitability: {suitability.replace('_', ' ').lower()}"

    method_lines: List[str] = []
    model = compact.get("business_model") or {}
    dcf = compact.get("dcf") or {}
    method_status = compact.get("valuation_method_status")
    method_line = metric_policy.VALUATION_STATUS_WORDING.get(method_status)
    if method_line and not dcf.get("available"):
        method_lines.append(method_line)
        profile = (model.get("profile") or "").replace("_", " ").lower()
        if profile and model.get("standard_fcff_suitability") == "NOT_SUITABLE":
            method_lines.append(
                f"  This issuer is classified as {profile} (SEC SIC {model.get('sic')}, "
                f"{model.get('sic_description')}); for that business model operating cash "
                f"flow less capital expenditure is not owner free cash flow. The figure is "
                f"still reported, under that definition.")
        method_lines.append("  Research therefore relies on operating, capital, guidance "
                            "and market evidence.")

    unavailable_line = None
    if not method_lines and not dcf.get("available") and dcf.get("reason"):
        # A declined valuation says WHY. Without this the reader sees a
        # Valuation section with no valuation and no explanation, which
        # reads as a malfunction -- and for a euro-reporting issuer the old
        # silence was worse than that: nothing distinguished "this project
        # cannot convert the currency" from "the company reported nothing".
        unavailable_line = ("Valuation model: no discounted-cash-flow valuation was "
                            f"produced. {dcf['reason']}")

    share_basis_note = None
    share = basis.get("share_reconciliation") or {}
    if share.get("status") in ("MATERIAL_DIFFERENCE", "INCOMPATIBLE_BASIS"):
        gap = share.get("market_cap_gap")
        share_basis_note = (
            "Share basis: UNRESOLVED"
            + (f" — price x shares differs from reported market capitalisation by "
               f"{gap:+.1%}" if isinstance(gap, (int, float)) else "")
            + "; per-share figures are not comparable with the market price")

    return ValuationBasis(
        financial_base_label=financial_base_label,
        financial_base_caveat=financial_base_caveat,
        balance_sheet_label=balance_label, balance_sheet_kind=balance_kind,
        guidance_lines=tuple(guidance_lines), freshness_note=freshness_note,
        suitability_note=suitability_note, method_lines=tuple(method_lines),
        unavailable_line=unavailable_line, share_basis_note=share_basis_note)


def _base_growth_clamp(compact: dict) -> Optional[dict]:
    """The base scenario's year-1 revenue-growth clamp, if it bound.

    Reads the PROVENANCE the assumption builder already recorded rather than
    re-deriving a bound: the report's job is to show what happened, not to
    recompute it.
    """
    dcf = compact.get("dcf") or {}
    candidates = []
    for scenario in (dcf.get("scenarios") or []):
        if scenario.get("scenario") != "base":
            continue
        candidates.append(((scenario.get("assumptions") or {})
                           .get("assumption_provenance") or {}).get("revenue_growth"))
    candidates.append((dcf.get("shared_assumption_provenance") or {}).get("revenue_growth"))
    for entry in candidates:
        if isinstance(entry, dict) and entry.get("clamped") \
                and entry.get("raw_value") is not None:
            return {"raw_growth": entry["raw_value"],
                    "applied_growth": entry.get("applied_value"),
                    "clamp_reason": entry.get("clamp_reason")}
    return None


def _fmt_pct_inline(value, decimals=1) -> Optional[str]:
    return None if value is None else f"{value * 100:.{decimals}f}%"


def _build_valuation(compact: dict, valuation_status: str) -> ValuationView:
    """Part 12's central rule: the renderer never decides valuation validity.

    `publishable` is the one flag, derived from the one status. The modelled
    values, the premium, the modelled return and the scenario spread are all
    absent from this object unless it is True -- not present-but-suppressed,
    absent. A renderer cannot print what it was not given.
    """
    quote = compact.get("quote") or {}
    dcf = compact.get("dcf") or {}
    spread = compact.get("dcf_scenario_spread") or {}
    gap = compact.get("valuation_gap") or {}
    publishable = valuation_status == ValuationStatus.VALID_FOR_RESEARCH

    # §14, and the wording rule it exists to enforce. The DCF RAN and its
    # output may not be published: say which of the four reasons that is.
    # "Status: MODEL_INVALID" was printed for all of them, including the one
    # case the spec names explicitly as forbidden -- a model that correctly
    # declined to grow a negative terminal cash flow into a perpetuity is
    # working, and telling a reader it failed sends them to debug a model
    # that has nothing wrong with it.
    model_invalid_status = None
    model_invalid_explanation = None
    model_invalid_reasons: Tuple[str, ...] = ()
    if dcf.get("available") and not publishable             and valuation_status != ValuationStatus.LIMITED:
        model_invalid_status = valuation_status
        model_invalid_explanation = STATUS_EXPLANATION.get(valuation_status)
        model_invalid_reasons = tuple(dcf.get("validation_reasons") or ())

    unavailable_reason = None
    if not dcf.get("available"):
        unavailable_reason = dcf.get("reason")

    comparison_withheld = None
    premium_pct = modeled_return_pct = None
    if gap.get("available"):
        if publishable:
            premium_pct = gap.get("market_price_premium_pct")
            modeled_return_pct = gap.get("modeled_return_to_value_pct")
        else:
            basis = compact.get("dcf_financial_basis") or {}
            share = basis.get("share_reconciliation") or {}
            if share.get("status") in ("MATERIAL_DIFFERENCE", "INCOMPATIBLE_BASIS"):
                comparison_withheld = (
                    "Comparison with the market price is WITHHELD: the share count "
                    "underlying the modeled per-share values does not reconcile against "
                    "the reported market capitalisation, so the two figures are not on "
                    "the same basis.")
            elif valuation_status == ValuationStatus.LIMITED:
                comparison_withheld = (
                    "Market-price comparison: not meaningful — DCF input normalization is "
                    "unresolved, so a percentage comparison against the modeled value "
                    "would state a precision the inputs do not support.")

    assumption_lines: List[str] = []
    sensitivity_level = None
    terminal_note = None
    scenarios = dcf.get("scenarios") or []
    primary = next((s for s in scenarios
                    if s.get("scenario") == dcf.get("primary_scenario")), None)
    # NOT gated on `publishable`. Part 4's withheld list is valuation
    # CONCLUSIONS -- a modelled value, a premium, a modelled return. The
    # assumptions are the model's INPUTS, and the clamp disclosure in
    # particular exists to stop a reader mistaking the engine's configured
    # bound for a forecast. Hiding it whenever the valuation is unusable
    # would remove the explanation exactly where it is most needed.
    if primary:
        assumptions = primary.get("assumptions") or {}
        revenue_growth = assumptions.get("revenue_growth")
        if isinstance(revenue_growth, list):
            revenue_growth = revenue_growth[0] if revenue_growth else None
        if revenue_growth is not None:
            clamp = _base_growth_clamp(compact)
            if clamp:
                # A clamped growth rate is the MODEL'S BOUND, not an estimate
                # of the company's growth. Printing it bare invites exactly
                # the reading a live report got: "25.0%" beside a 65.5%
                # reported growth rate, with nothing saying which was which.
                assumption_lines.append(
                    f"- Revenue growth: {_fmt_pct_inline(revenue_growth)} "
                    f"(CLAMPED — the evidence implied "
                    f"{_fmt_pct_inline(clamp['raw_growth'])}; this is the model's "
                    "configured bound, not an estimate of the company's growth)")
            else:
                assumption_lines.append(
                    f"- Revenue growth: {_fmt_pct_inline(revenue_growth)}")
        if assumptions.get("wacc") is not None:
            assumption_lines.append(f"- WACC: {_fmt_pct_inline(assumptions['wacc'])}")
        if assumptions.get("terminal_growth") is not None:
            assumption_lines.append(
                f"- Terminal growth: {_fmt_pct_inline(assumptions['terminal_growth'])}")

    if publishable and spread.get("available") \
            and spread.get("spread_pct_of_base") is not None:
        spread_pct = abs(spread["spread_pct_of_base"])
        sensitivity_level = ("LOW" if spread_pct < 0.15
                             else ("MODERATE" if spread_pct < 0.40 else "HIGH"))
        terminal_share = (primary or {}).get("terminal_value_share_of_enterprise_value")
        # 0.75 matches the threshold finance/dcf.py::_collect_warnings already
        # uses for its own terminal-value-dependency warning.
        if terminal_share is not None and terminal_share > 0.75:
            terminal_note = ("The base modeled value depends heavily on the terminal-value "
                             "assumption (WACC and terminal growth), not near-term cash "
                             "flows.")

    return ValuationView(
        status=valuation_status,
        publishable=publishable,
        market_price=quote.get("price"),
        basis=_build_valuation_basis(compact),
        bear_value_per_share=(spread.get("bear_value_per_share")
                              if publishable and spread.get("available") else None),
        base_value_per_share=(spread.get("base_value_per_share")
                              if publishable and spread.get("available") else None),
        bull_value_per_share=(spread.get("bull_value_per_share")
                              if publishable and spread.get("available") else None),
        premium_pct=premium_pct, modeled_return_pct=modeled_return_pct,
        comparison_withheld_reason=comparison_withheld,
        model_invalid_status=model_invalid_status,
        model_invalid_explanation=model_invalid_explanation,
        model_invalid_reasons=model_invalid_reasons,
        assumption_lines=tuple(assumption_lines),
        sensitivity_level=sensitivity_level,
        terminal_dependence_note=terminal_note,
        unavailable_reason=unavailable_reason)


# ---------------------------------------------------------------------------
# Technical
# ---------------------------------------------------------------------------

def _build_technical(compact: dict) -> TechnicalView:
    technicals = compact.get("technical_metrics") or {}

    def value(name):
        entry = technicals.get(name)
        return entry.get("value") if isinstance(entry, dict) else None

    trend = None
    latest = value("latest_close")
    smas = (("20-day", value("sma_20")), ("50-day", value("sma_50")),
            ("200-day", value("sma_200")))
    if latest is not None and any(sma is not None for _label, sma in smas):
        above = [label for label, sma in smas if sma is not None and latest >= sma]
        below = [label for label, sma in smas if sma is not None and latest < sma]
        parts = []
        if above:
            parts.append(f"above the {_join_labels(above)} SMA{'s' if len(above) > 1 else ''}")
        if below:
            parts.append(f"below the {_join_labels(below)} SMA{'s' if len(below) > 1 else ''}")
        trend = f"Price is {' and '.join(parts)}."

    macd = value("macd_histogram")
    macd_sign = None
    if macd is not None:
        macd_sign = "positive" if macd > 0 else ("negative" if macd < 0 else "flat")

    total_return_entry = technicals.get("total_return_over_window") or {}
    return TechnicalView(
        trend_sentence=trend, rsi=value("rsi_14"),
        macd_histogram=macd, macd_sign=macd_sign,
        return_label=return_window_label(total_return_entry),
        total_return=total_return_entry.get("value"),
        annualized_volatility=value("annualized_volatility"))


# ---------------------------------------------------------------------------
# Research stages
# ---------------------------------------------------------------------------

def _stage_claims(pipeline_result, stage_name, max_bullets, unavailable_note_fn) -> ClaimSet:
    output, note = unavailable_note_fn(pipeline_result, stage_name)
    if output is None:
        return ClaimSet(unavailable_note=note)
    return ClaimSet(claims=tuple(claim["claim"] for claim in output["claims"][:max_bullets]))


def _build_risks(pipeline_result, unavailable_note_fn, max_bullets=4):
    """§18: company risk and analysis limitations are separate dimensions.

    A live insurer's risk list opened with "[HIGH] Valuation evidence is
    unavailable because the discounted cash flow model produced no output" --
    read as the company's largest risk, when the company had done nothing.
    """
    output, note = unavailable_note_fn(pipeline_result, "risk_reviewer")
    if output is None:
        return (), (), note

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    company = output.get("company_risks")
    limitations = output.get("analysis_limitations")
    # Older stage output (a replayed artifact) has no categories; treating
    # everything as company risk there reproduces the previous rendering
    # exactly rather than silently reclassifying it.
    if company is None and limitations is None:
        company, limitations = output.get("key_risks") or [], []

    ordered = sorted(company, key=lambda r: severity_rank.get(r["severity"], 1))
    company_items = tuple(RiskItem(text=r["risk"], severity=r["severity"])
                          for r in ordered[:max_bullets])
    if not company_items:
        company_items = (RiskItem(
            text="No company-specific risk was identified at this severity."),)

    remaining = max(0, max_bullets - len(company_items))
    limitation_items: List[RiskItem] = []
    for entry in sorted(limitations, key=lambda r: severity_rank.get(r["severity"], 1)):
        if len(limitation_items) >= remaining:
            break
        limitation_items.append(RiskItem(text=entry["risk"], severity=entry.get("severity")))
    for concern in (output.get("data_quality_concerns") or []):
        if len(limitation_items) >= remaining:
            break
        limitation_items.append(RiskItem(text=concern))
    return company_items, tuple(limitation_items), None


_CONDITION_GROUPS = (("Limiting factors", "limiting_factors", 3),
                     ("Upgrade conditions", "conditions_that_strengthen_the_view", 2),
                     ("Downgrade conditions", "conditions_that_weaken_the_view", 2),
                     ("Reassessment triggers", "reassessment_triggers", 2))


def _build_research_view(compact, pipeline_result, pipeline_status, readiness,
                         readiness_reason) -> ResearchView:
    output = (pipeline_result.output("final_investment_synthesizer")
              if pipeline_result else None)
    base = ResearchView(
        pipeline_status=pipeline_status,
        readiness_status=readiness.get("status"),
        readiness_reason=readiness_reason)
    if output is None:
        return base

    confidence = output["confidence"]
    groups = []
    for label, key, cap in _CONDITION_GROUPS:
        # §16: `min_items` counts entries that SURVIVE filtering. A blank
        # string is not a condition, and a report that prints one has
        # published an empty claim.
        items = tuple(i for i in (output.get(key) or []) if str(i).strip())[:cap]
        if items:
            groups.append((label, items))

    return ResearchView(
        pipeline_status=pipeline_status,
        readiness_status=readiness.get("status"),
        readiness_reason=readiness_reason,
        available=True,
        research_stance=output["research_stance"].replace("_", " "),
        valuation_view=output["valuation_view"].replace("_", " "),
        overall_risk=output["overall_risk"].replace("_", " "),
        confidence=confidence,
        confidence_band=confidence_band(confidence),
        recommendation=output["recommendation"].replace("_", " ").upper(),
        primary_reason=(output.get("primary_reason") or "").strip() or None,
        condition_groups=tuple(groups))


def _build_sources(compact: dict) -> Tuple[Tuple[SourceLine, ...], Optional[str]]:
    provenance = compact.get("data_provenance") or {}
    by_provider: Dict[str, List[str]] = {}
    for dataset, entry in provenance.items():
        provider = entry.get("provider") if isinstance(entry, dict) else None
        if not provider:
            continue
        label = _DATASET_SOURCE_LABEL.get(dataset, dataset.replace("_", " "))
        labels = by_provider.setdefault(provider, [])
        if label not in labels:
            labels.append(label)
    lines = tuple(SourceLine(provider=_PROVIDER_DISPLAY_NAME.get(p, p.title()),
                             labels=tuple(labels))
                  for p, labels in by_provider.items())

    plan = compact.get("plan") or {}
    omitted = plan.get("omitted_datasets") or []
    note = None
    if omitted:
        effect = (plan.get("omission_effects") or {}).get(omitted[0], "")
        note = (f"Missing: {omitted[0].replace('_', ' ')}"
                + (f" — {effect}" if effect else "") + ".")
    return lines, note



# ---------------------------------------------------------------------------
# Section builders, public
# ---------------------------------------------------------------------------
#
# Each sub-view is buildable on its own from a compact payload. That is not a
# convenience: it is what makes the decisions testable WITHOUT rendering
# markdown and reading it back, which is how they were previously asserted
# and why several of them went unchecked for so long.

def valuation_basis_from(compact: dict) -> ValuationBasis:
    return _build_valuation_basis(compact)


def valuation_from(compact: dict) -> ValuationView:
    from finance.evidence import valuation_evidence_status
    return _build_valuation(compact, valuation_evidence_status(compact))


def research_view_from(compact: dict, pipeline_result) -> ResearchView:
    from finance.workflow import (
        _effective_research_readiness,
        _pipeline_overall_status,
        _pipeline_stage_cascade,
        _research_view_reason_line,
    )
    cascade = _pipeline_stage_cascade(pipeline_result)
    readiness = _effective_research_readiness(
        compact.get("research_readiness") or {"status": None, "reasons": []}, cascade)
    return _build_research_view(
        compact, pipeline_result, _pipeline_overall_status(cascade), readiness,
        _research_view_reason_line(readiness, pipeline_result, cascade))


def risks_from(pipeline_result):
    """(company risks, analysis limitations, unavailable note)."""
    from finance.workflow import _compact_pipeline_stage_output
    return _build_risks(pipeline_result, _compact_pipeline_stage_output)


_STATUS_WORDS = {"full": "COMPLETE", "reduced": "REDUCED",
                 "cached_only": "PARTIAL", "stale": "PARTIAL"}


def build_stock_analysis_report_model(result, compact: dict, pipeline_result
                                      ) -> StockAnalysisReportModel:
    """Part 10: one deterministic builder, from canonical validated outputs.

    Imports from `finance.workflow` are done inside the function because the
    workflow imports this module; the alternative was duplicating the
    pipeline-cascade helpers here, and two implementations of "which stages
    completed" is the class of duplication this phase exists to remove.
    """
    from finance.evidence import valuation_evidence_status
    from finance.workflow import (
        _compact_pipeline_stage_output,
        _effective_research_readiness,
        _pipeline_overall_status,
        _pipeline_stage_cascade,
        _research_view_reason_line,
    )

    symbol = compact.get("symbol") or result.symbol
    mode = result.plan.mode

    sentence = f"**{_STATUS_WORDS.get(mode, 'REDUCED')}**."
    reason = result.plan.reason if mode != "full" else None
    provenance = compact.get("data_provenance") or {}
    unofficial = ("Yahoo Finance data is an unofficial, personal-use source."
                  if any(isinstance(p, dict) and p.get("provider") == "yahoo"
                         for p in provenance.values()) else None)
    status = ReportStatus(
        headline=sentence, reason=reason,
        missing_datasets=tuple(result.plan.omitted_datasets),
        unofficial_source_note=unofficial)

    cascade = _pipeline_stage_cascade(pipeline_result)
    pipeline_status = _pipeline_overall_status(cascade)
    base_readiness = compact.get("research_readiness") or {"status": None, "reasons": []}
    readiness = _effective_research_readiness(base_readiness, cascade)
    readiness_reason = _research_view_reason_line(readiness, pipeline_result, cascade)

    company_risk, limitations, risk_note = _build_risks(
        pipeline_result, _compact_pipeline_stage_output)
    research_view = _build_research_view(compact, pipeline_result, pipeline_status,
                                         readiness, readiness_reason)
    sources, missing_note = _build_sources(compact)

    return StockAnalysisReportModel(
        symbol=symbol,
        status=status,
        snapshot=_build_snapshot(compact),
        valuation=_build_valuation(compact, valuation_evidence_status(compact)),
        technical=_build_technical(compact),
        bull_case=_stage_claims(pipeline_result, "bull_researcher", 3,
                                _compact_pipeline_stage_output),
        bear_case=_stage_claims(pipeline_result, "bear_researcher", 3,
                                _compact_pipeline_stage_output),
        company_risk=company_risk,
        analysis_limitations=limitations,
        risk_unavailable_note=risk_note,
        research_pipeline=pipeline_status,
        research_readiness=readiness.get("status"),
        research_view=research_view,
        recommendation=research_view.recommendation,
        conditions=research_view.condition_groups,
        sources=sources,
        missing_dataset_note=missing_note,
        currency=compact.get("financial_history_currency") or "USD",
    )
