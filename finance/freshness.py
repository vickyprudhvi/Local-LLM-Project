"""Phase H.4 — DcfFreshnessPlanner: which reported figure the DCF actually uses.

THE BUG THIS EXISTS TO FIX (live AOS, valuation run in 2026):

    long-term debt   FY2025 (2025-12-31)   $112.7M   <- what the DCF used
                     Q2 2026 (2026-06-30)  $598.0M   <- what was available
    total debt       FY2025                ~$155M
                     Q2 2026               ~$637M    (4.1x)

AOS financed the Leonard Valve acquisition in January 2026 (there is an
8-K item 2.03, "Creation of a Direct Financial Obligation", filed
2026-01-06). Six months and two 10-Qs later, the DCF's equity bridge still
subtracted December's net debt, understating it by ~$482M. Nothing warned,
because `_dcf_inputs_from_facts` read `statements["annual"]["balance_sheet"]
[0]` and there is no such thing as a stale annual statement — FY2025 WAS the
latest annual filing. The quarterly data was fetched, normalized and sat
unread.

THE RULE: there is no single "latest financial year". Freshness is decided
PER FIELD, because different kinds of figure go stale differently:

* A balance-sheet field is a POINT IN TIME. The newest one wins outright,
  and it comes from a 10-Q far more often than a 10-K.
* A flow (revenue, cash flow) is a PERIOD. The newest twelve months wins —
  which usually means a trailing-twelve-month roll built from quarters, not
  the last completed fiscal year.
* Management guidance is FORWARD-looking. It is not a reported fact at all
  and never supersedes one; it is separate evidence (see finance/guidance.py).

Two independent statuses come out of this, and conflating them is what let
the AOS run look healthy:

    data_completeness  — did every dataset we asked for arrive?
    valuation_freshness — is the DCF actually USING the newest data it has?

AOS scored COMPLETE on the first and would have scored STALE_INPUT_WARNING
on the second. Only the pair describes the run honestly.

Nothing here computes a valuation. This module SELECTS inputs and records
why; finance/dcf.py remains the only place arithmetic happens.
"""

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import tools.config as config
from finance import period_facts as pf

# ---------------------------------------------------------------------------
# Statuses
# ---------------------------------------------------------------------------


class DataCompleteness:
    """Did the datasets we ASKED FOR actually arrive? Says nothing at all
    about whether the valuation used the freshest of what arrived."""

    COMPLETE = "COMPLETE"
    REDUCED = "REDUCED"
    PARTIAL = "PARTIAL"
    ALL = (COMPLETE, REDUCED, PARTIAL)


class ValuationFreshness:
    """Is the DCF using the newest data available to it?

    Deliberately ORTHOGONAL to DataCompleteness. A run can have every
    requested dataset present (COMPLETE) and still value the company off
    six-month-old balances (STALE_INPUT_WARNING) — that is precisely the AOS
    failure, and a single combined status cannot express it.
    """

    CURRENT = "CURRENT"
    MOSTLY_CURRENT = "MOSTLY_CURRENT"
    STALE_INPUT_WARNING = "STALE_INPUT_WARNING"
    STALE_INVALID = "STALE_INVALID"
    ALL = (CURRENT, MOSTLY_CURRENT, STALE_INPUT_WARNING, STALE_INVALID)

    # Ordered worst-last so a run's overall status is max() over its findings.
    _RANK = {CURRENT: 0, MOSTLY_CURRENT: 1, STALE_INPUT_WARNING: 2, STALE_INVALID: 3}

    @classmethod
    def worst(cls, statuses) -> str:
        found = [s for s in statuses if s in cls._RANK]
        if not found:
            return cls.CURRENT
        return max(found, key=lambda s: cls._RANK[s])


class FreshnessStatus:
    """Per-FIELD freshness classification."""

    CURRENT_QUARTER = "current_quarter"
    CURRENT_ANNUAL = "current_annual"
    TTM = "ttm"
    STALE = "stale"
    MISSING = "missing"
    ALL = (CURRENT_QUARTER, CURRENT_ANNUAL, TTM, STALE, MISSING)


# Deterministic stale-input guard codes (section 12). Emitted as structured
# findings, never raised — a stale input degrades and annotates a valuation,
# it does not delete it.
DCF_STALE_BALANCE_SHEET_INPUT = "DCF_STALE_BALANCE_SHEET_INPUT"
DCF_STALE_FLOW_INPUT = "DCF_STALE_FLOW_INPUT"
DCF_STALE_DEBT_INPUT = "DCF_STALE_DEBT_INPUT"
DCF_CURRENT_GUIDANCE_NOT_CONSIDERED = "DCF_CURRENT_GUIDANCE_NOT_CONSIDERED"

ALL_STALE_GUARD_CODES = (DCF_STALE_BALANCE_SHEET_INPUT, DCF_STALE_FLOW_INPUT,
                         DCF_STALE_DEBT_INPUT, DCF_CURRENT_GUIDANCE_NOT_CONSIDERED)


# ---------------------------------------------------------------------------
# Selected values
# ---------------------------------------------------------------------------

# Point-in-time balance-sheet fields, per section 2. `total_debt` and
# `net_debt` are DERIVED from components rather than selected directly (see
# `_derive_total_debt`), matching finance/normalization.py's documented
# aggregation policy exactly so a SEC-sourced bridge and an Alpha-Vantage-
# sourced one apply identical debt policy.
BALANCE_SHEET_FIELDS = (
    "cash_and_cash_equivalents",
    "short_term_investments",
    "short_term_debt",
    "current_portion_of_long_term_debt",
    "long_term_debt",
    "current_assets",
    "current_liabilities",
    "stockholders_equity",
    "preferred_equity",
    "minority_interest",
)

# Flow fields that get a trailing-twelve-month treatment, per section 3.
TTM_FLOW_FIELDS = (
    "revenue",
    "operating_income",
    "net_income",
    "depreciation_and_amortization",
    "capital_expenditure",
    "operating_cash_flow",
)


@dataclass(frozen=True)
class SelectedValue:
    """One DCF input plus the full provenance section 2 requires.

    Every field here answers a question a reader of the report can ask:
    where did this number come from, when was it true, which filing said so,
    and how do I cite it.
    """

    field: str
    value: Optional[float]
    unit: str
    source: str                       # quarterly_sec_filing | annual_sec_filing | ttm_calculation | derived | unavailable
    provider: str
    accession: Optional[str]
    form: Optional[str]
    fiscal_period: Optional[str]
    as_of_date: Optional[str]
    retrieval_timestamp: Optional[str]
    evidence_id: str
    freshness_status: str
    period_start: Optional[str] = None
    derivation: Optional[str] = None
    components: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "provider": self.provider,
            "accession": self.accession,
            "form": self.form,
            "fiscal_period": self.fiscal_period,
            "as_of_date": self.as_of_date,
            "period_start": self.period_start,
            "retrieval_timestamp": self.retrieval_timestamp,
            "evidence_id": self.evidence_id,
            "freshness_status": self.freshness_status,
            "derivation": self.derivation,
            "components": list(self.components),
        }


# XBRL concepts whose value INCLUDES capitalized/finance lease obligations.
# This project's documented debt policy otherwise excludes them (see
# finance/normalization.py::_derive_balance_sheet_aggregates), so whenever one
# of these wins, the selection says so rather than letting "total debt"
# silently change meaning between issuers.
_LEASE_INCLUSIVE_CONCEPTS = frozenset({
    "LongTermDebtAndCapitalLeaseObligations",
    "LongTermDebtAndCapitalLeaseObligationsCurrent",
    "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    "DebtAndCapitalLeaseObligations",
})


def _unavailable(field_name: str, evidence_id: str, reason: str) -> SelectedValue:
    return SelectedValue(
        field=field_name, value=None, unit="USD", source="unavailable",
        provider="sec", accession=None, form=None, fiscal_period=None,
        as_of_date=None, retrieval_timestamp=None, evidence_id=evidence_id,
        freshness_status=FreshnessStatus.MISSING, derivation=reason)


# ---------------------------------------------------------------------------
# TTM construction
# ---------------------------------------------------------------------------

# A trailing twelve months must actually span twelve months. Four quarters of
# a 52/53-week or 4-4-5 fiscal calendar land in this window; anything outside
# it means the four "quarters" are not a contiguous year and the roll is
# refused rather than reported as TTM.
TTM_SPAN_DAYS = (350, 380)
# Largest tolerated gap/overlap between consecutive quarters, in days. Fiscal
# period boundaries do not always align to the day across filings.
_MAX_QUARTER_JOIN_GAP = 7


@dataclass
class TtmResult:
    value: Optional[float] = None
    quarters: List[pf.PeriodFact] = field(default_factory=list)
    ok: bool = False
    reason: Optional[str] = None
    used_reconstruction: bool = False

    @property
    def period_start(self) -> Optional[str]:
        return self.quarters[0].start if self.quarters else None

    @property
    def period_end(self) -> Optional[str]:
        return self.quarters[-1].end if self.quarters else None


def build_ttm(company_facts: dict, field_name: str, offset: int = 0) -> TtmResult:
    """Sum the latest four CONTIGUOUS discrete quarters, or fail closed.

    "Fail closed" is load-bearing here. A TTM that silently sums three
    quarters, or sums across a gap, understates a flow by a quarter and there
    is no way to see that from the resulting number alone. Every rejection
    states its reason so the caller can fall back to the annual figure
    explicitly and say so in the report.

    `offset` steps the window BACK by whole quarters, so offset=4 gives the
    prior-year trailing twelve months. That is what makes a TTM-over-TTM
    growth rate possible — the only growth measure that compares a current
    twelve-month period against an equivalent one.
    """
    series = pf.discrete_quarters(company_facts, field_name)
    available = series.quarters[:len(series.quarters) - offset] if offset else series.quarters
    quarters = available[-4:] if len(available) >= 4 else []
    if len(quarters) < 4:
        return TtmResult(reason=(
            f"Only {len(available)} discrete quarter(s) of {field_name} could be built "
            f"{'before the requested offset ' if offset else ''}; four are required for a "
            "trailing-twelve-month figure."))

    for earlier, later in zip(quarters, quarters[1:]):
        gap = pf._span_days(earlier.end, later.start)
        if gap is None or abs(gap) > _MAX_QUARTER_JOIN_GAP:
            return TtmResult(reason=(
                f"{field_name} quarters are not contiguous: {earlier.start}..{earlier.end} is "
                f"followed by {later.start}..{later.end}, so summing them would "
                f"{'double-count' if (gap or 0) < 0 else 'skip'} part of the year."))

    total_span = pf._span_days(quarters[0].start, quarters[-1].end)
    low, high = TTM_SPAN_DAYS
    if total_span is None or not (low <= total_span <= high):
        return TtmResult(reason=(
            f"{field_name}'s four quarters span {total_span} days, outside the {low}-{high} day "
            "window a trailing twelve months must cover."))

    units = {q.unit for q in quarters}
    if len(units) > 1:
        return TtmResult(reason=(
            f"{field_name} quarters mix units ({', '.join(sorted(units))}); they cannot be summed."))

    return TtmResult(
        value=sum(q.value for q in quarters),
        quarters=list(quarters),
        ok=True,
        used_reconstruction=any(q.reconstructed_from for q in quarters),
    )


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _evidence_id(kind: str, name: str) -> str:
    return f"dcf.{kind}.{name}"


class DcfFreshnessPlanner:
    """Selects the freshest valid input for every DCF field, deterministically.

    The LLM is never consulted here and never sees a raw filing. It receives
    the RESULT of this selection (a `CurrentFinancialState`) and may reason
    about it, but the decision of which period is freshest, which value
    supersedes which, and whether an input is stale belongs entirely to this
    class.
    """

    def __init__(self, company_facts: dict, symbol: str,
                 retrieval_timestamp: Optional[str] = None,
                 provider: str = "sec"):
        self.company_facts = company_facts or {}
        self.symbol = symbol
        self.retrieved_at = retrieval_timestamp or _now_iso()
        self.provider = provider

    # -- balance sheet -----------------------------------------------------

    def balance_sheet_date(self) -> Optional[str]:
        return pf.latest_balance_sheet_date(self.company_facts)

    def latest_annual_balance_sheet_date(self) -> Optional[str]:
        periods = pf.annual_periods(self.company_facts, "revenue")
        return periods[-1].end if periods else None

    def select_balance_sheet(self) -> Tuple[Dict[str, SelectedValue], List[str]]:
        """Every point-in-time field AT ONE COHERENT DATE.

        The date is chosen first (`latest_balance_sheet_date`), then each
        field is required to have been reported at exactly that date. A field
        that was not is ABSENT — never back-filled from an older filing.

        Live AOS shows why back-filling would be actively harmful: its
        `short_term_debt` candidate concepts are (`ShortTermBorrowings`,
        `DebtCurrent`); it used `ShortTermBorrowings` twice, both times in
        2010, and has never used `DebtCurrent`. A "latest available value"
        rule injects a sixteen-year-old $158M balance into a 2026 net-debt
        bridge. Requiring the selected date instead correctly reports the
        field as not present on the current balance sheet, which for debt
        components means the company has none (matching
        finance/dcf.py::compute_net_debt's own convention for these fields).
        """
        selections: Dict[str, SelectedValue] = {}
        warnings: List[str] = []
        as_of = self.balance_sheet_date()
        if not as_of:
            for name in BALANCE_SHEET_FIELDS:
                selections[name] = _unavailable(
                    name, _evidence_id("input", f"{name}.latest"),
                    "No balance-sheet date could be identified in these filings.")
            warnings.append("No filed balance sheet could be located; the equity bridge has no "
                            "point-in-time inputs.")
            return selections, warnings

        annual_end = self.latest_annual_balance_sheet_date()
        is_quarterly = bool(annual_end and as_of > annual_end)

        for name in BALANCE_SHEET_FIELDS:
            fact = pf.instant_as_of(self.company_facts, name, as_of)
            evidence_id = _evidence_id("input", f"{name}.latest")
            if fact is None:
                stale_fact = pf.latest_instant(self.company_facts, name)
                if stale_fact is not None:
                    # The field EXISTS in the company's history but not on the
                    # current balance sheet. Recorded explicitly so a reader
                    # can see the difference between "never reported" and
                    # "not reported any more".
                    selections[name] = SelectedValue(
                        field=name, value=None, unit=stale_fact.unit, source="unavailable",
                        provider=self.provider, accession=stale_fact.accession,
                        form=stale_fact.form, fiscal_period=stale_fact.fiscal_period,
                        as_of_date=None, retrieval_timestamp=self.retrieved_at,
                        evidence_id=evidence_id, freshness_status=FreshnessStatus.MISSING,
                        derivation=(
                            f"Not reported on the {as_of} balance sheet. The most recent "
                            f"{name} figure in these filings is from {stale_fact.end}, which is "
                            "too old to belong to this balance sheet and is NOT used."))
                else:
                    selections[name] = _unavailable(
                        name, evidence_id,
                        f"Not reported on the {as_of} balance sheet.")
                continue
            selections[name] = SelectedValue(
                field=name, value=fact.value, unit=fact.unit,
                source="quarterly_sec_filing" if is_quarterly else "annual_sec_filing",
                provider=self.provider, accession=fact.accession, form=fact.form,
                fiscal_period=fact.fiscal_period, as_of_date=fact.end,
                retrieval_timestamp=self.retrieved_at, evidence_id=evidence_id,
                freshness_status=(FreshnessStatus.CURRENT_QUARTER if is_quarterly
                                  else FreshnessStatus.CURRENT_ANNUAL),
                derivation=(f"Reported on the {fact.form} balance sheet dated {fact.end} "
                            f"as {fact.concept}."
                            + (" This figure INCLUDES capitalized lease obligations, which "
                               "this project's debt policy otherwise excludes; the issuer does "
                               "not report a lease-free debt line."
                               if fact.concept in _LEASE_INCLUSIVE_CONCEPTS else "")))
            if fact.concept in _LEASE_INCLUSIVE_CONCEPTS:
                warnings.append(
                    f"{name} was taken from {fact.concept}, which includes capitalized lease "
                    "obligations; this issuer does not report a lease-free debt line.")
        return selections, warnings

    # -- derived debt ------------------------------------------------------

    def derive_total_debt(self, balance: Dict[str, SelectedValue]) -> SelectedValue:
        """total_debt = short_term_debt + current portion of LTD + long-term debt.

        The SAME documented policy as finance/normalization.py::
        _derive_balance_sheet_aggregates — the sum of whichever components
        are actually reported, missing ones excluded rather than zero-filled,
        finance/capital-lease obligations deliberately not included.
        """
        parts = ("short_term_debt", "current_portion_of_long_term_debt", "long_term_debt")
        known = [(p, balance[p]) for p in parts
                 if p in balance and balance[p].value is not None]
        evidence_id = _evidence_id("input", "debt.latest")
        if not known:
            return _unavailable("total_debt", evidence_id,
                                "No debt component was reported on the current balance sheet.")
        total = sum(sel.value for _p, sel in known)
        anchor = known[0][1]
        breakdown = ", ".join(f"{p}={sel.value:,.0f}" for p, sel in known)
        return SelectedValue(
            field="total_debt", value=total, unit=anchor.unit, source="derived",
            provider=self.provider, accession=anchor.accession, form=anchor.form,
            fiscal_period=anchor.fiscal_period, as_of_date=anchor.as_of_date,
            retrieval_timestamp=self.retrieved_at, evidence_id=evidence_id,
            freshness_status=anchor.freshness_status,
            derivation=f"Sum of reported debt components at {anchor.as_of_date}: {breakdown}.",
            components=tuple(p for p, _sel in known))

    # -- flows -------------------------------------------------------------

    # How far a trailing-twelve-month window may END before the balance-sheet
    # date and still be "trailing". Generous enough for a company whose
    # cash-flow tagging lags its balance-sheet tagging by a quarter, far too
    # tight to admit a window from a different era.
    #
    # This guard exists because its absence produced a genuinely absurd
    # result: on the AMZN fixture, capital expenditure's newest usable
    # quarters were from 2016-2017 (that concept stopped being used), and
    # `latest(4)` dutifully summed them into a "trailing twelve months"
    # ending 2017-03-31 -- $7.4B, against a real current figure an order of
    # magnitude larger. Every structural check passed: four quarters,
    # contiguous, 365 days, one unit. "Trailing" was the only property never
    # actually tested.
    _MAX_FLOW_LAG_DAYS = 200

    def select_flow(self, field_name: str, reference_date: Optional[str] = None) -> SelectedValue:
        """TTM first, latest annual second, absent third (section 3's hierarchy).

        The annual fallback is not a failure — for a company that files only
        10-Ks, or whose interim tagging cannot be reconciled, the last
        completed fiscal year IS the freshest honest flow figure. What must
        never happen is using it while a valid TTM exists and not saying so;
        that is what `DCF_STALE_FLOW_INPUT` detects.
        """
        evidence_id = _evidence_id("input", f"{field_name}_ttm")
        ttm = build_ttm(self.company_facts, field_name)
        if ttm.ok and reference_date and ttm.period_end:
            lag = pf._span_days(ttm.period_end, reference_date)
            if lag is not None and lag > self._MAX_FLOW_LAG_DAYS:
                ttm = TtmResult(reason=(
                    f"The newest four quarters of {field_name} end {ttm.period_end}, "
                    f"{lag} days before the current balance-sheet date {reference_date}; that "
                    "is not a trailing twelve months. This company appears to have stopped "
                    "reporting the line under the concept that resolved."))
        if ttm.ok:
            last = ttm.quarters[-1]
            note = ""
            if ttm.used_reconstruction:
                note = (" Some quarters were reconstructed by differencing consecutive "
                        "year-to-date figures, because this company reports the line "
                        "cumulatively rather than per quarter.")
            return SelectedValue(
                field=field_name, value=ttm.value, unit=last.unit, source="ttm_calculation",
                provider=self.provider, accession=last.accession, form=last.form,
                fiscal_period=last.fiscal_period, as_of_date=ttm.period_end,
                period_start=ttm.period_start, retrieval_timestamp=self.retrieved_at,
                evidence_id=evidence_id, freshness_status=FreshnessStatus.TTM,
                derivation=(f"Trailing twelve months {ttm.period_start}..{ttm.period_end}, "
                            f"summed from 4 discrete quarters." + note),
                components=tuple(f"{q.start}..{q.end}" for q in ttm.quarters))

        annual = pf.annual_periods(self.company_facts, field_name)
        if annual and reference_date:
            # The same "is it actually recent?" test, applied to the fallback.
            # A fiscal year may legitimately be up to ~15 months behind the
            # latest quarterly balance sheet; anything older means the line
            # is no longer reported under the concept that resolved.
            lag = pf._span_days(annual[-1].end, reference_date)
            if lag is not None and lag > 460:
                return _unavailable(
                    field_name, evidence_id,
                    f"The most recent annual {field_name} figure ends {annual[-1].end}, "
                    f"{lag} days before the current balance-sheet date; it is too old to "
                    "describe this company's current economics and is NOT used.")
        if annual:
            latest = annual[-1]
            return SelectedValue(
                field=field_name, value=latest.value, unit=latest.unit,
                source="annual_sec_filing", provider=self.provider,
                accession=latest.accession, form=latest.form,
                fiscal_period=latest.fiscal_period, as_of_date=latest.end,
                period_start=latest.start, retrieval_timestamp=self.retrieved_at,
                evidence_id=evidence_id, freshness_status=FreshnessStatus.CURRENT_ANNUAL,
                derivation=(f"Latest reported fiscal year {latest.start}..{latest.end}. "
                            f"No trailing-twelve-month figure was built: {ttm.reason}"))
        return _unavailable(field_name, evidence_id,
                            f"Neither a trailing-twelve-month nor an annual {field_name} figure "
                            f"could be built. {ttm.reason or ''}".strip())


# ---------------------------------------------------------------------------
# CurrentFinancialState
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurrentFinancialState:
    """The normalized state the DCF consumes, INSTEAD of the latest annual
    statement object (section 6).

    The point of routing everything through one immutable structure is that
    "which period is this number from" stops being implicit in which list
    index somebody happened to read. Every value here is a `SelectedValue`
    carrying its own as-of date, filing and evidence id, so a mismatch
    between two inputs is detectable rather than invisible.

    `historical_metrics` is BACKWARD-looking context and is deliberately kept
    in its own compartment, separate from `management_guidance` (forward-
    looking evidence). Section 7 exists because the two were previously
    interchangeable: a historical revenue CAGR was copied directly into the
    year-1 DCF growth assumption, which is a forecast, not a measurement.
    """

    symbol: str
    valuation_date: str
    financial_as_of: Optional[str]
    balance_sheet: Dict[str, SelectedValue]
    flows: Dict[str, SelectedValue]
    total_debt: SelectedValue
    net_debt: Optional[float]
    latest_annual_period: Optional[str]
    latest_quarterly_period: Optional[str]
    management_guidance: Optional[dict]
    historical_metrics: Dict[str, object]
    data_completeness: str
    valuation_freshness: str
    findings: Tuple[dict, ...] = ()
    warnings: Tuple[str, ...] = ()

    def value(self, name: str) -> Optional[float]:
        """The plain number for a field, from whichever compartment holds it."""
        if name in self.balance_sheet:
            return self.balance_sheet[name].value
        if name in self.flows:
            return self.flows[name].value
        if name == "total_debt":
            return self.total_debt.value
        return None

    def selection(self, name: str) -> Optional[SelectedValue]:
        if name in self.balance_sheet:
            return self.balance_sheet[name]
        if name in self.flows:
            return self.flows[name]
        if name == "total_debt":
            return self.total_debt
        return None

    @property
    def guidance_available(self) -> bool:
        return bool(self.management_guidance
                    and (self.management_guidance.get("metrics") or {}))

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "valuation_date": self.valuation_date,
            "financial_as_of": self.financial_as_of,
            "balance_sheet": {k: v.to_dict() for k, v in self.balance_sheet.items()},
            "flows": {k: v.to_dict() for k, v in self.flows.items()},
            "total_debt": self.total_debt.to_dict(),
            "net_debt": self.net_debt,
            "latest_annual_period": self.latest_annual_period,
            "latest_quarterly_period": self.latest_quarterly_period,
            "management_guidance": self.management_guidance,
            "historical_metrics": dict(self.historical_metrics),
            "data_completeness": self.data_completeness,
            "valuation_freshness": self.valuation_freshness,
            "findings": [dict(f) for f in self.findings],
            "warnings": list(self.warnings),
        }


def _finding(code: str, severity: str, message: str, **detail) -> dict:
    entry = {"code": code, "severity": severity, "message": message}
    entry.update(detail)
    return entry


# How much total debt must move between the last fiscal year end and the
# current balance sheet before the difference is reported as a finding.
_MATERIAL_DEBT_CHANGE = 0.20


def build_current_financial_state(company_facts: dict, symbol: str,
                                  historical_metrics: Optional[dict] = None,
                                  management_guidance: Optional[dict] = None,
                                  guidance_considered: bool = True,
                                  valuation_date: Optional[str] = None,
                                  retrieval_timestamp: Optional[str] = None,
                                  data_completeness: str = DataCompleteness.COMPLETE
                                  ) -> CurrentFinancialState:
    """Run the planner and classify the result's freshness (sections 6, 12, 13)."""
    planner = DcfFreshnessPlanner(company_facts, symbol,
                                  retrieval_timestamp=retrieval_timestamp)
    valuation_date = valuation_date or datetime.datetime.now(
        datetime.timezone.utc).date().isoformat()

    balance, warnings = planner.select_balance_sheet()
    total_debt = planner.derive_total_debt(balance)
    # The balance-sheet date is the reference point for "is this flow
    # actually trailing?" -- see `DcfFreshnessPlanner._MAX_FLOW_LAG_DAYS`.
    reference_date = planner.balance_sheet_date()
    flows = {name: planner.select_flow(name, reference_date=reference_date)
             for name in TTM_FLOW_FIELDS}

    # Free cash flow is DERIVED, never selected: OCF - capex, both on the SAME
    # basis. Mixing a trailing-twelve-month operating cash flow with an annual
    # capex would produce a figure belonging to no period at all.
    flows["free_cash_flow"] = _derive_free_cash_flow(
        flows.get("operating_cash_flow"), flows.get("capital_expenditure"),
        planner.retrieved_at)

    annual_end = planner.latest_annual_balance_sheet_date()
    bs_date = planner.balance_sheet_date()
    quarterly_end = bs_date if (annual_end and bs_date and bs_date > annual_end) else None

    findings: List[dict] = []

    # -- guard: a newer figure existed and was not used --------------------
    # Checked against what the filings ACTUALLY contain, so this can never
    # fire when there is genuinely nothing newer to have used.
    for name, selection in balance.items():
        newest = pf.latest_instant(company_facts, name)
        if (selection.value is not None and newest is not None and newest.end
                and selection.as_of_date and newest.end > selection.as_of_date):
            findings.append(_finding(
                DCF_STALE_BALANCE_SHEET_INPUT, "warning",
                f"{name} was taken from {selection.as_of_date} although a newer figure dated "
                f"{newest.end} exists.", field=name,
                used_as_of=selection.as_of_date, available_as_of=newest.end))

    debt_finding = _debt_change_finding(company_facts, total_debt, annual_end)
    if debt_finding:
        findings.append(debt_finding)

    # -- guard: an annual flow was used while a valid TTM existed ----------
    for name, selection in flows.items():
        if selection.source != "annual_sec_filing":
            continue
        probe = build_ttm(company_facts, name)
        if probe.ok and reference_date and probe.period_end:
            # Do not report a stale window as an available TTM -- the same
            # recency test `select_flow` applied when it declined to use it.
            lag = pf._span_days(probe.period_end, reference_date)
            if lag is not None and lag > DcfFreshnessPlanner._MAX_FLOW_LAG_DAYS:
                probe = TtmResult(reason="stale")
        if probe.ok:
            findings.append(_finding(
                DCF_STALE_FLOW_INPUT, "warning",
                f"{name} used the latest fiscal year although a trailing-twelve-month figure "
                f"through {probe.period_end} could be built.", field=name))

    # -- guard: guidance retrieved but not routed to the assumption builder -
    if management_guidance and (management_guidance.get("metrics") or {}) \
            and not guidance_considered:
        findings.append(_finding(
            DCF_CURRENT_GUIDANCE_NOT_CONSIDERED, "warning",
            "Current management guidance was retrieved but did not reach the forward-assumption "
            "builder.", fiscal_year=management_guidance.get("fiscal_year")))

    net_debt = None
    cash = balance.get("cash_and_cash_equivalents")
    if total_debt.value is not None and cash is not None and cash.value is not None:
        net_debt = total_debt.value - cash.value

    freshness = _classify_valuation_freshness(findings, balance, flows, bs_date, annual_end)

    return CurrentFinancialState(
        symbol=symbol,
        valuation_date=valuation_date,
        financial_as_of=bs_date,
        balance_sheet=balance,
        flows=flows,
        total_debt=total_debt,
        net_debt=net_debt,
        latest_annual_period=annual_end,
        latest_quarterly_period=quarterly_end,
        management_guidance=management_guidance,
        historical_metrics=dict(historical_metrics or {}),
        data_completeness=data_completeness,
        valuation_freshness=freshness,
        findings=tuple(findings),
        warnings=tuple(warnings),
    )


def _derive_free_cash_flow(ocf: Optional[SelectedValue], capex: Optional[SelectedValue],
                           retrieved_at: str) -> SelectedValue:
    """FCF = operating cash flow - capital expenditure, on ONE basis."""
    evidence_id = _evidence_id("input", "free_cash_flow_ttm")
    if ocf is None or capex is None or ocf.value is None or capex.value is None:
        return _unavailable("free_cash_flow", evidence_id,
                            "Operating cash flow and capital expenditure are not both available.")
    if ocf.source != capex.source:
        return _unavailable(
            "free_cash_flow", evidence_id,
            f"Operating cash flow is on a {ocf.source} basis but capital expenditure is on a "
            f"{capex.source} basis; subtracting them would produce a figure belonging to no "
            "single period.")
    return SelectedValue(
        field="free_cash_flow", value=ocf.value - abs(capex.value), unit=ocf.unit,
        source=ocf.source, provider=ocf.provider, accession=ocf.accession, form=ocf.form,
        fiscal_period=ocf.fiscal_period, as_of_date=ocf.as_of_date,
        period_start=ocf.period_start, retrieval_timestamp=retrieved_at,
        evidence_id=evidence_id, freshness_status=ocf.freshness_status,
        derivation=(f"Operating cash flow ({ocf.value:,.0f}) less capital expenditure "
                    f"({abs(capex.value):,.0f}), both {ocf.source}."),
        components=("operating_cash_flow", "capital_expenditure"))


def _debt_change_finding(company_facts: dict, total_debt: SelectedValue,
                         annual_end: Optional[str]) -> Optional[dict]:
    """Report a MATERIAL debt change since the last annual balance sheet.

    This is the AOS case stated as a rule. When total debt at the current
    balance-sheet date differs materially from total debt at the last fiscal
    year end, a valuation built on the annual figure is not slightly behind —
    it bridges to the wrong capital structure entirely. The finding is
    emitted whenever the change is material, INCLUDING on a correct run that
    uses the current figure, so the report can explain why the two differ
    (acquisition, refinancing) rather than leaving a reader to wonder.
    """
    if total_debt.value is None or not annual_end or not total_debt.as_of_date:
        return None
    if total_debt.as_of_date <= annual_end:
        return None
    parts = ("short_term_debt", "current_portion_of_long_term_debt", "long_term_debt")
    prior_components = []
    for name in parts:
        fact = pf.instant_as_of(company_facts, name, annual_end)
        if fact is not None:
            prior_components.append(fact.value)
    if not prior_components:
        return None
    prior_total = sum(prior_components)
    if prior_total <= 0:
        return None
    change = (total_debt.value - prior_total) / prior_total
    if abs(change) < _MATERIAL_DEBT_CHANGE:
        return None
    return _finding(
        DCF_STALE_DEBT_INPUT, "warning",
        f"Total debt changed {change:+.0%} between the last fiscal year end ({annual_end}: "
        f"{prior_total:,.0f}) and the current balance sheet ({total_debt.as_of_date}: "
        f"{total_debt.value:,.0f}). The current figure is used; a valuation built on the "
        "annual figure would bridge to the wrong capital structure.",
        prior_total_debt=prior_total, current_total_debt=total_debt.value,
        change_ratio=round(change, 4), prior_as_of=annual_end,
        current_as_of=total_debt.as_of_date)


def _classify_valuation_freshness(findings, balance, flows, bs_date, annual_end) -> str:
    """Section 13. Distinct from data completeness — see the module docstring."""
    codes = {f["code"] for f in findings}
    if (DCF_STALE_BALANCE_SHEET_INPUT in codes or DCF_STALE_FLOW_INPUT in codes
            or DCF_CURRENT_GUIDANCE_NOT_CONSIDERED in codes):
        return ValuationFreshness.STALE_INPUT_WARNING

    if not any(sel.value is not None for sel in balance.values()):
        return ValuationFreshness.STALE_INVALID

    quarterly_balance = bool(annual_end and bs_date and bs_date > annual_end)
    ttm_flows = any(sel.source == "ttm_calculation" for sel in flows.values())
    if quarterly_balance and ttm_flows:
        return ValuationFreshness.CURRENT
    # An annual-only filer genuinely has nothing fresher, so this is not a
    # warning — but it is also not "current" in the sense a reader assumes.
    return ValuationFreshness.MOSTLY_CURRENT
