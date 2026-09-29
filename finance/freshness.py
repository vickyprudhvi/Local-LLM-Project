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
from finance import net_debt as nd
from finance import period_facts as pf
from finance import profitability as prof
from finance import reporting_currency as reporting_currency_module
from finance import structural_breaks as sb
from finance import taxonomy as taxonomy_module
from finance import ttm as ttm_module

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
    # Section 18: every input IS the newest filed, and something material was
    # filed after the balance sheet anyway. Distinct from MOSTLY_CURRENT,
    # which means "nothing fresher exists"; this means "something fresher
    # exists and it is not an accounting balance we may apply".
    MOSTLY_CURRENT_WITH_EVENT_WARNING = "MOSTLY_CURRENT_WITH_EVENT_WARNING"
    MOSTLY_CURRENT = "MOSTLY_CURRENT"
    STALE_INPUT_WARNING = "STALE_INPUT_WARNING"
    STALE_INVALID = "STALE_INVALID"
    ALL = (CURRENT, MOSTLY_CURRENT, MOSTLY_CURRENT_WITH_EVENT_WARNING,
           STALE_INPUT_WARNING, STALE_INVALID)

    # Ordered worst-last so a run's overall status is max() over its findings.
    _RANK = {CURRENT: 0, MOSTLY_CURRENT: 1, MOSTLY_CURRENT_WITH_EVENT_WARNING: 2,
             STALE_INPUT_WARNING: 3, STALE_INVALID: 4}

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
# Section 18. A balance sheet can be the newest one filed and still not
# describe the company: something material was filed after it.
DCF_POST_BALANCE_SHEET_EVENT = "DCF_POST_BALANCE_SHEET_EVENT"

ALL_STALE_GUARD_CODES = (DCF_STALE_BALANCE_SHEET_INPUT, DCF_STALE_FLOW_INPUT,
                         DCF_STALE_DEBT_INPUT, DCF_CURRENT_GUIDANCE_NOT_CONSIDERED,
                         DCF_POST_BALANCE_SHEET_EVENT)


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
    # Section 27: reported so the bridge's exclusions are visible, never
    # netted -- restricted cash cannot repay debt, and lease liabilities sit
    # outside this project's documented total-debt policy.
    "restricted_cash",
    "lease_liabilities",
    # Phase H.6, section 14: kept as its own component so a net-debt policy
    # that does NOT treat equity holdings as cash can still report them.
    "equity_securities_at_fair_value",
    "short_term_debt",
    "current_portion_of_long_term_debt",
    "long_term_debt",
    # Not a component of the bridge: an issuer-reported combined debt figure,
    # selected purely so finance/net_debt.py can cross-check the component
    # sum against it (section 16). NVDA reports `LongTermDebt` = $8,470M,
    # which is what proves its $1,000M `DebtCurrent`/`LongTermDebtCurrent`
    # pair is one obligation and not two.
    "total_debt_combined",
    "current_assets",
    "current_liabilities",
    "stockholders_equity",
    "preferred_equity",
    "minority_interest",
)

# Balance-sheet fields that are POINT-IN-TIME by definition and must never be
# rolled into a trailing-twelve-month figure (section 4). Enforced in
# finance/ttm.py::build_ttm as well; stated here so the rule is visible at
# the place the two kinds of field sit side by side.
POINT_IN_TIME_ONLY_FIELDS = frozenset(BALANCE_SHEET_FIELDS)

# Flow fields that get a trailing-twelve-month treatment, per section 3.
TTM_FLOW_FIELDS = (
    "revenue",
    "operating_income",
    # Phase H.8: the components an operating income is DERIVED from when the
    # issuer does not tag `OperatingIncomeLoss`, plus the lines the
    # profitability layer needs. Selected through the same freshness planner
    # as everything else so they carry matching periods.
    "income_before_tax",
    "interest_expense_nonoperating",
    "income_tax_expense",
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
    # The winning XBRL concept. Phase H.6: previously this was recorded only
    # inside the human-readable `derivation` text, which meant a rule that
    # depends on WHICH TAG supplied a value -- the debt-overlap rule in
    # finance/net_debt.py most of all -- had to parse prose to work. NVDA's
    # $1.0B double-count is exactly a which-tag question (`DebtCurrent`
    # already contains `LongTermDebtCurrent`), so the concept is now a field.
    concept: Optional[str] = None
    # Section 3's TTM record, present only on a ttm_calculation selection.
    ttm: Optional[dict] = None

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
            "concept": self.concept,
            "ttm": dict(self.ttm) if self.ttm else None,
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
    """Backwards-compatible view over finance/ttm.py's `TtmMetric`.

    Phase H.6 moved TTM construction into its own module so a second
    construction (fiscal year + current year-to-date - prior-year
    year-to-date) could be added and every window could carry an explicit
    `construction_method`/`validation_status`. This wrapper keeps the shape
    the rest of this module and its tests already read.
    """

    value: Optional[float] = None
    quarters: List[pf.PeriodFact] = field(default_factory=list)
    ok: bool = False
    reason: Optional[str] = None
    used_reconstruction: bool = False
    metric: Optional["ttm_module.TtmMetric"] = None

    @property
    def period_start(self) -> Optional[str]:
        if self.metric is not None:
            return self.metric.start_date
        return self.quarters[0].start if self.quarters else None

    @property
    def period_end(self) -> Optional[str]:
        if self.metric is not None:
            return self.metric.end_date
        return self.quarters[-1].end if self.quarters else None


def build_ttm(company_facts: dict, field_name: str, offset: int = 0,
              reference_end: Optional[str] = None) -> TtmResult:
    """The trailing twelve months for one flow field, or a stated failure.

    Delegates to finance/ttm.py::build_ttm, which owns both supported
    constructions and the section-3 invariants. "Fail closed" is still
    load-bearing: a TTM that silently sums three quarters, or sums across a
    gap, understates a flow by a quarter and there is no way to see that from
    the resulting number alone.

    `offset` steps the window BACK by whole quarters, so offset=4 gives the
    prior-year trailing twelve months — what makes a TTM-over-TTM growth rate
    possible, the only growth measure comparing a twelve-month period against
    an equivalent one.
    """
    metric = ttm_module.build_ttm(company_facts, field_name, offset=offset,
                                  reference_end=reference_end)
    return TtmResult(
        value=metric.value,
        quarters=list(metric.quarter_facts),
        ok=metric.ok,
        reason=metric.reason,
        used_reconstruction=metric.used_reconstruction,
        metric=metric,
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
                concept=fact.concept,
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

    def derive_total_debt(self, balance: Dict[str, SelectedValue]
                          ) -> Tuple[SelectedValue, "nd.NetDebtComponents"]:
        """total_debt = short_term_debt + current portion of LTD + long-term debt,
        MINUS whichever of those is contained inside another.

        The SAME documented policy as finance/normalization.py::
        _derive_balance_sheet_aggregates — the sum of whichever components
        are actually reported, missing ones excluded rather than zero-filled,
        finance/capital-lease obligations deliberately not included — with
        one Phase H.6 correction: the components must not OVERLAP.

        NVDA is the case. `DebtCurrent` ($1,000M) and `LongTermDebtCurrent`
        ($1,000M) are the same obligation — us-gaap defines `DebtCurrent` as
        short-term debt AND the current portion of long-term debt — and
        summing all three components gave $9,470M against a true $8,470M,
        which NVDA reports directly as `LongTermDebt`. The overlap rule lives
        in finance/net_debt.py so the same policy applies wherever net debt
        is computed; this method just drives it and preserves the
        `SelectedValue` shape the rest of the module reads.
        """
        evidence_id = _evidence_id("input", "debt.latest")
        combined = balance.get("total_debt_combined")
        components = nd.collect_components(
            balance,
            reported_total_debt=getattr(combined, "value", None),
            reported_total_debt_concept=getattr(combined, "concept", None),
            as_of_date=self.balance_sheet_date())

        if components.total_debt is None:
            return _unavailable(
                "total_debt", evidence_id,
                "No debt component was reported on the current balance sheet."), components

        included = [name for name in nd.DEBT_COMPONENT_FIELDS
                    if getattr(components, name) is not None
                    and name not in {e["field"] for e in components.excluded}]
        anchor = next((balance[name] for name in included if name in balance), None)
        breakdown = ", ".join(f"{name}={getattr(components, name):,.0f}" for name in included)
        note = ""
        if components.excluded:
            note = " " + " ".join(e["reason"] for e in components.excluded)
        return SelectedValue(
            field="total_debt", value=components.total_debt,
            unit=components.unit, source="derived",
            provider=self.provider,
            accession=getattr(anchor, "accession", None),
            form=getattr(anchor, "form", None),
            fiscal_period=getattr(anchor, "fiscal_period", None),
            as_of_date=components.as_of_date,
            retrieval_timestamp=self.retrieved_at, evidence_id=evidence_id,
            freshness_status=getattr(anchor, "freshness_status", FreshnessStatus.MISSING),
            derivation=(f"Sum of reported debt components at {components.as_of_date}: "
                        f"{breakdown}." + note),
            components=tuple(included)), components

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

        Phase H.6: the selection now carries the full `TtmMetric` record
        (construction method, validation status, the exact periods that were
        combined) so nothing downstream has to infer what "TTM" meant for
        this particular field. A PARTIAL window — a real twelve months that
        ends materially before the company's latest reported period — is
        still used, but it is labelled as such and `_derive_free_cash_flow`
        refuses to combine it with a window ending somewhere else.
        """
        evidence_id = _evidence_id("input", f"{field_name}_ttm")
        ttm = build_ttm(self.company_facts, field_name, reference_end=reference_date)
        if ttm.ok and reference_date and ttm.period_end:
            lag = pf._span_days(ttm.period_end, reference_date)
            if lag is not None and lag > self._MAX_FLOW_LAG_DAYS:
                ttm = TtmResult(reason=(
                    f"The newest twelve months of {field_name} end {ttm.period_end}, "
                    f"{lag} days before the current balance-sheet date {reference_date}; that "
                    "is not a trailing twelve months. This company appears to have stopped "
                    "reporting the line under the concept that resolved."))
        if ttm.ok:
            metric = ttm.metric
            note = ""
            if ttm.used_reconstruction:
                note = (" Some periods were reconstructed by differencing consecutive "
                        "year-to-date figures, because this company reports the line "
                        "cumulatively rather than per quarter.")
            if metric.validation_status == ttm_module.TtmValidation.PARTIAL and metric.reason:
                note += " " + metric.reason
            return SelectedValue(
                field=field_name, value=ttm.value, unit=metric.unit, source="ttm_calculation",
                provider=self.provider, accession=metric.latest_accession,
                form=metric.latest_form,
                fiscal_period=metric.latest_fiscal_period, as_of_date=ttm.period_end,
                period_start=ttm.period_start, retrieval_timestamp=self.retrieved_at,
                evidence_id=evidence_id, freshness_status=FreshnessStatus.TTM,
                derivation=(f"Trailing twelve months {ttm.period_start}..{ttm.period_end}, "
                            f"built by {metric.construction_method}." + note),
                components=tuple(metric.quarters_included),
                ttm=metric.to_dict())

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
    # Phase H.6 additions.
    net_debt_detail: Optional[dict] = None
    historical_comparability: Optional[dict] = None
    post_balance_sheet_events: Tuple[dict, ...] = ()
    freshness_audit: Optional[dict] = None
    # Phase H.7 additions.
    taxonomy: Optional[str] = None
    reporting_framework_note: Optional[str] = None
    # REPORTING_CURRENCY_SERIES_SELECTION phase. `reporting_currency_status`
    # is one of finance/reporting_currency.py::ReportingSeriesStatus; a value
    # in `BLOCKS_CURRENT_STATE` means a readable (USD) series exists but a
    # newer, incompatible-currency series has superseded it, so this state
    # must not be treated as current downstream (see `valuation_freshness`
    # below, and finance/dcf_packet.py's FINANCIAL_BASE_STALE gate).
    reporting_currency_status: Optional[str] = None
    reporting_currency_resolution: Optional[dict] = None
    # Phase H.8. Reported and normalized profitability, side by side, with
    # every unusual-item adjustment itemized (sections 1-5).
    profitability: Optional[dict] = None

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
            "net_debt_detail": self.net_debt_detail,
            "historical_comparability": self.historical_comparability,
            "post_balance_sheet_events": [dict(e) for e in self.post_balance_sheet_events],
            "freshness_audit": self.freshness_audit,
            "taxonomy": self.taxonomy,
            "reporting_framework_note": self.reporting_framework_note,
            "profitability": self.profitability,
            "reporting_currency_status": self.reporting_currency_status,
            "reporting_currency_resolution": self.reporting_currency_resolution,
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
                                  data_completeness: str = DataCompleteness.COMPLETE,
                                  submissions: Optional[dict] = None
                                  ) -> CurrentFinancialState:
    """Run the planner and classify the result's freshness (sections 6, 12, 13).

    `submissions` (Phase H.6) is the SEC submissions index. It is what makes
    two things knowable that company facts alone cannot answer: whether the
    reported history spans a structural break (finance/structural_breaks.py),
    and whether anything material was filed AFTER the balance sheet the
    equity bridge uses. Both are optional — the state is built without them,
    with the corresponding fields reported as unknown rather than as clean.
    """
    planner = DcfFreshnessPlanner(company_facts, symbol,
                                  retrieval_timestamp=retrieval_timestamp)
    valuation_date = valuation_date or datetime.datetime.now(
        datetime.timezone.utc).date().isoformat()

    balance, warnings = planner.select_balance_sheet()
    framework_note = (taxonomy_module.unsupported_taxonomy_reason(company_facts)
                      or taxonomy_module.reporting_currency_note(company_facts))
    # REPORTING_CURRENCY_SERIES_SELECTION: the ONE place this is decided (see
    # finance/reporting_currency.py's module docstring for the failure this
    # closes). `reporting_currency_note` above only fires when NO USD fact
    # exists anywhere in the issuer's history; it stays silent for an issuer
    # that reported in USD for years and then switched -- exactly the case
    # that requires this resolver.
    currency_resolution = reporting_currency_module.resolve_reporting_series(company_facts)
    if (currency_resolution.selection_status
            in reporting_currency_module.ReportingSeriesStatus.BLOCKS_CURRENT_STATE):
        framework_note = framework_note or currency_resolution.resolution_reason
    if framework_note:
        # Phase H.7: without this the pipeline reported "this company
        # published no financials", which is a far stronger claim than "this
        # project cannot read this issuer's reporting framework".
        warnings.append(framework_note)
    total_debt, debt_components = planner.derive_total_debt(balance)
    warnings.extend(exclusion["reason"] for exclusion in debt_components.excluded)
    # The balance-sheet date is the reference point for "is this flow
    # actually trailing?" -- see `DcfFreshnessPlanner._MAX_FLOW_LAG_DAYS`.
    reference_date = planner.balance_sheet_date()
    # Flows are measured against the company's latest REPORTED PERIOD, not
    # against its balance-sheet date. They are usually the same; when they
    # are not, the reported period is the honest yardstick for "is this
    # twelve months current?".
    flow_reference = ttm_module.latest_reported_period_end(company_facts) or reference_date
    flows = {name: planner.select_flow(name, reference_date=flow_reference)
             for name in TTM_FLOW_FIELDS}

    # Phase H.8 -- OPERATING INCOME, DERIVED WHEN IT IS NOT TAGGED.
    #
    # Not every issuer reports `OperatingIncomeLoss`. Two live examples had
    # 775 and 394 us-gaap concepts respectively and neither included it, so
    # `operating_income` resolved to None and the forward-assumption builder
    # fell all the way through to the configured default margin. On one of
    # them that produced a base modelled value of $0.20 against a $149.93
    # market price.
    #
    # Both DO report pre-tax income and non-operating interest, which is the
    # standard bridge. The result is labelled `derived` -- never `reported` --
    # so nothing downstream mistakes it for a figure the issuer published.
    if flows.get("operating_income") is None or flows["operating_income"].value is None:
        derived = _derive_operating_income(
            flows.get("income_before_tax"), flows.get("interest_expense_nonoperating"),
            planner.retrieved_at)
        if derived.value is not None:
            flows["operating_income"] = derived

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

    # -- net debt, under a NAMED policy, from named components -------------
    # Phase H.6: previously this was `total_debt - cash`, unconditionally,
    # regardless of the configured policy — so a run configured for
    # cash-and-marketable-securities reported one net debt here and a
    # different one inside the DCF. NVDA made the gap $37B.
    policy = config.dcf_net_debt_policy()
    net_debt_result = nd.compute_net_debt(
        debt_components, policy,
        marketable_securities_eligible=config.dcf_short_term_investments_eligible())
    net_debt = net_debt_result.value
    findings.extend(net_debt_result.findings)

    # -- section 13: is the reported history comparable with today? --------
    comparability = sb.assess_historical_comparability(
        company_facts, submissions,
        history_years=config.dcf_assumption_history_max_years())

    # -- section 18: material events after the balance-sheet date ----------
    all_events = sb.find_post_balance_sheet_events(submissions, bs_date, valuation_date)
    # Phase H.9, sections 24-25. Only events that change the ISSUER's own
    # capital structure make the equity bridge stale. A Form 144 -- an
    # insider selling shares he already owns -- changes no share count, raises
    # no capital and moves no debt, and reporting it as a freshness risk was
    # describing a different fact about the company. Every event is still
    # RECORDED; only the reassessment-worthy ones raise a finding.
    events = [e for e in all_events if (e.impact or {}).get("requires_reassessment")]
    non_material = [e for e in all_events if e not in events]
    if non_material:
        warnings.append(
            f"{len(non_material)} securities filing(s) after {bs_date} were reviewed and do "
            "not affect the company's share count, debt or cash "
            f"({', '.join(sorted({e.event_type for e in non_material}))}).")
    for event in events:
        findings.append(_finding(
            DCF_POST_BALANCE_SHEET_EVENT, "warning",
            f"A {event.form} filed {event.filed} reports {event.description}, AFTER the "
            f"{bs_date} balance sheet this valuation bridges to. The balance sheet is not "
            "adjusted for it — no deterministic adjustment is available from a filing index — "
            "but the equity bridge may no longer describe the company's current capital "
            "structure.",
            filed=event.filed, form=event.form, items=event.items,
            accession=event.accession))

    # -- Phase H.8, sections 1-5: reported vs normalized profitability ------
    revenue_flow = flows.get("revenue")
    profitability_state = prof.build_profitability_state(
        revenue=getattr(revenue_flow, "value", None),
        period=(f"{getattr(revenue_flow, 'period_start', None)}.."
                f"{getattr(revenue_flow, 'as_of_date', None)}"),
        operating_income=getattr(flows.get("operating_income"), "value", None),
        net_income=getattr(flows.get("net_income"), "value", None),
        operating_cash_flow=getattr(flows.get("operating_cash_flow"), "value", None),
        income_tax_expense=getattr(flows.get("income_tax_expense"), "value", None),
        income_before_tax=getattr(flows.get("income_before_tax"), "value", None),
        company_facts=company_facts,
        period_start=getattr(revenue_flow, "period_start", None),
        period_end=getattr(revenue_flow, "as_of_date", None))
    operating_selection = flows.get("operating_income")
    if operating_selection is not None and operating_selection.components:
        profitability_state.operating_income_source = "derived"
        profitability_state.operating_income_derivation = operating_selection.derivation
    findings.extend(profitability_state.findings)
    warnings.extend(profitability_state.warnings)

    freshness = _classify_valuation_freshness(
        findings, balance, flows, bs_date, annual_end,
        currency_status=currency_resolution.selection_status)

    audit = _build_freshness_audit(
        symbol, valuation_date, flows, bs_date, annual_end, quarterly_end,
        management_guidance, events, comparability, net_debt_result, findings,
        company_facts=company_facts)

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
        net_debt_detail=net_debt_result.to_dict(),
        historical_comparability=comparability.to_dict(),
        taxonomy=taxonomy_module.detect_taxonomy(company_facts),
        profitability=profitability_state.to_dict(),
        reporting_framework_note=framework_note,
        post_balance_sheet_events=tuple(e.to_dict() for e in all_events),
        freshness_audit=audit,
        reporting_currency_status=currency_resolution.selection_status,
        reporting_currency_resolution=currency_resolution.to_dict(),
    )


def _build_freshness_audit(symbol, valuation_date, flows, bs_date, annual_end, quarterly_end,
                           management_guidance, events, comparability, net_debt_result,
                           findings, company_facts=None) -> dict:
    """Section 23 — the compact internal audit built BEFORE the DCF runs.

    One structure answering "what period is each side of this valuation
    actually on?", so the answer stops being distributed across a dozen
    fields nobody reads together. Deliberately NOT rendered into compact
    reports (section 23's last line); the compact report shows the four
    summary lines section 24 specifies and this is what they are derived from.
    """
    revenue = flows.get("revenue")
    ttm_record = (revenue.ttm if revenue is not None else None) or {}
    guidance_metrics = ((management_guidance or {}).get("metrics") or {})
    return {
        "symbol": symbol,
        "valuation_date": valuation_date,
        "flow_base": {
            "type": ("TTM" if revenue is not None and revenue.source == "ttm_calculation"
                     else "ANNUAL" if revenue is not None and revenue.value is not None
                     else "UNAVAILABLE"),
            "through": getattr(revenue, "as_of_date", None),
            "from": getattr(revenue, "period_start", None),
            "valid": bool(ttm_record.get("validation_status") in ("valid", "partial")),
            "validation_status": ttm_record.get("validation_status"),
            "construction_method": ttm_record.get("construction_method"),
        },
        "flow_periods": {
            name: {"source": selection.source, "from": selection.period_start,
                   "through": selection.as_of_date}
            for name, selection in flows.items()
        },
        "balance_sheet": {
            "as_of": bs_date,
            "form": "10-Q" if quarterly_end else "10-K",
            "latest_annual_period": annual_end,
        },
        "guidance": {
            "available": bool(guidance_metrics),
            "issued_at": (management_guidance or {}).get("filed"),
            "metrics": sorted(guidance_metrics),
            "periods": sorted({(entry or {}).get("fiscal_period")
                               for entry in guidance_metrics.values()
                               if isinstance(entry, dict) and entry.get("fiscal_period")}),
        },
        "net_debt": {
            "policy": net_debt_result.policy,
            "value": net_debt_result.value,
            "reconciled": net_debt_result.reconciled,
        },
        "post_balance_sheet_events": [e.to_dict() for e in events],
        "historical_comparability": comparability.status,
        # Section 48. `shares`, `dcf_suitability` and the reporting framework
        # are filled in by the workflow once the share reconciliation and the
        # valuation have run -- they are not knowable at this point, and
        # leaving the keys present-but-empty is what makes the audit a fixed
        # shape rather than a dict whose fields come and go.
        "shares": {"basis": None, "reconciliation": None},
        "dcf_suitability": None,
        "reporting_framework": {
            "taxonomy": taxonomy_module.detect_taxonomy(company_facts),
            "note": taxonomy_module.unsupported_taxonomy_reason(company_facts)
                    or taxonomy_module.reporting_currency_note(company_facts),
        },
        "warnings": [f["message"] for f in findings if f.get("severity") in ("warning", "error")],
    }


def _derive_operating_income(pre_tax: Optional[SelectedValue],
                             interest: Optional[SelectedValue],
                             retrieved_at: str) -> SelectedValue:
    """operating income ~= pre-tax income + non-operating interest expense.

    Only ever attempted when the issuer does not tag operating income
    directly, and only when both components cover the SAME period -- adding
    an interest expense from a different window to a pre-tax income is the
    same class of error as any other period mismatch.

    The bridge is approximate by nature: it recovers operating income from
    below the line, so any other non-operating item the issuer nets into
    pre-tax income (investment income, equity-method results, one-off gains)
    stays inside the result. That is stated in the derivation rather than
    hidden, and the value is labelled `derived` so the profitability layer
    and the report can both say so.
    """
    evidence_id = _evidence_id("input", "operating_income_ttm")
    if pre_tax is None or pre_tax.value is None:
        return _unavailable(
            "operating_income", evidence_id,
            "This issuer does not report operating income, and no pre-tax income figure was "
            "available to derive it from.")
    if interest is None or interest.value is None:
        return _unavailable(
            "operating_income", evidence_id,
            "This issuer does not report operating income. Pre-tax income is available but "
            "non-operating interest expense is not, so the two cannot be bridged.")
    if pre_tax.as_of_date != interest.as_of_date             or pre_tax.period_start != interest.period_start:
        return _unavailable(
            "operating_income", evidence_id,
            f"Pre-tax income covers {pre_tax.period_start}..{pre_tax.as_of_date} but "
            f"non-operating interest covers {interest.period_start}..{interest.as_of_date}; "
            "bridging them would produce a figure belonging to neither period.")
    value = pre_tax.value + abs(interest.value)
    return SelectedValue(
        field="operating_income", value=value, unit=pre_tax.unit, source=pre_tax.source,
        provider=pre_tax.provider, accession=pre_tax.accession, form=pre_tax.form,
        fiscal_period=pre_tax.fiscal_period, as_of_date=pre_tax.as_of_date,
        period_start=pre_tax.period_start, retrieval_timestamp=retrieved_at,
        evidence_id=evidence_id, freshness_status=pre_tax.freshness_status,
        derivation=(
            f"DERIVED, not reported: this issuer does not tag operating income. Pre-tax "
            f"income ({pre_tax.value:,.0f}) plus non-operating interest expense "
            f"({abs(interest.value):,.0f}) over {pre_tax.period_start}..{pre_tax.as_of_date}. "
            "Any other non-operating item the issuer nets into pre-tax income remains inside "
            "this figure."),
        components=("income_before_tax", "interest_expense_nonoperating"),
        concept=pre_tax.concept, ttm=pre_tax.ttm)


def _derive_free_cash_flow(ocf: Optional[SelectedValue], capex: Optional[SelectedValue],
                           retrieved_at: str) -> SelectedValue:
    """FCF = operating cash flow - capital expenditure, over ONE period.

    Phase H.6 — the period test, not just the basis test. Matching `source`
    ("both are ttm_calculation") was never sufficient: live AT&T produced an
    operating-cash-flow TTM ending 2025-12-31 (the concept it used stopped
    being tagged after fiscal 2025) and a capital-expenditure TTM ending
    2026-06-30. Both were `ttm_calculation`, so the old check passed, and the
    difference was published as free cash flow "through 2026-06-30" — a
    figure covering neither window and belonging to no period at all.
    """
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
    if ocf.as_of_date != capex.as_of_date or ocf.period_start != capex.period_start:
        return _unavailable(
            "free_cash_flow", evidence_id,
            f"Operating cash flow covers {ocf.period_start}..{ocf.as_of_date} but capital "
            f"expenditure covers {capex.period_start}..{capex.as_of_date}. Subtracting one from "
            "the other would produce a figure belonging to neither period, so no free cash flow "
            "is reported for this company.")
    if ocf.unit != capex.unit:
        return _unavailable(
            "free_cash_flow", evidence_id,
            f"Operating cash flow is reported in {ocf.unit} and capital expenditure in "
            f"{capex.unit}; they cannot be subtracted.")
    # Section 4: a DERIVED value that carries a TTM label needs a TTM record
    # too, or the label is unverifiable. The record is inherited from the
    # components -- which the checks above have just proved cover the same
    # window -- and names the derivation as its construction method, so
    # nothing mistakes it for a directly summed series.
    derived_ttm = None
    if ocf.source == "ttm_calculation" and ocf.ttm:
        component_record = dict(ocf.ttm)
        derived_ttm = {
            **component_record,
            "metric": "free_cash_flow",
            "value": ocf.value - abs(capex.value),
            "construction_method": "operating_cash_flow_less_capital_expenditure",
            "validation_status": ttm_module.TtmValidation.worst_of(
                component_record.get("validation_status"),
                (capex.ttm or {}).get("validation_status")),
            "source_accessions": sorted(set(
                (component_record.get("source_accessions") or [])
                + ((capex.ttm or {}).get("source_accessions") or []))),
        }
    return SelectedValue(
        field="free_cash_flow", value=ocf.value - abs(capex.value), unit=ocf.unit,
        source=ocf.source, provider=ocf.provider, accession=ocf.accession, form=ocf.form,
        fiscal_period=ocf.fiscal_period, as_of_date=ocf.as_of_date,
        period_start=ocf.period_start, retrieval_timestamp=retrieved_at,
        evidence_id=evidence_id, freshness_status=ocf.freshness_status,
        derivation=(f"Operating cash flow ({ocf.value:,.0f}) less capital expenditure "
                    f"({abs(capex.value):,.0f}), both {ocf.source} over "
                    f"{ocf.period_start}..{ocf.as_of_date}."),
        components=("operating_cash_flow", "capital_expenditure"),
        ttm=derived_ttm)


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


def _classify_valuation_freshness(findings, balance, flows, bs_date, annual_end,
                                  currency_status: Optional[str] = None) -> str:
    """Section 13. Distinct from data completeness — see the module docstring."""
    if currency_status in reporting_currency_module.ReportingSeriesStatus.BLOCKS_CURRENT_STATE:
        # REPORTING_CURRENCY_SERIES_SELECTION: whatever balance/flow figures
        # were selected below come from a currency series the issuer has
        # since superseded (or one that cannot be disambiguated at all). They
        # are not a valid "current" state at any freshness grade above this.
        return ValuationFreshness.STALE_INVALID

    codes = {f["code"] for f in findings}
    if (DCF_STALE_BALANCE_SHEET_INPUT in codes or DCF_STALE_FLOW_INPUT in codes
            or DCF_CURRENT_GUIDANCE_NOT_CONSIDERED in codes):
        return ValuationFreshness.STALE_INPUT_WARNING

    if not any(sel.value is not None for sel in balance.values()):
        return ValuationFreshness.STALE_INVALID

    quarterly_balance = bool(annual_end and bs_date and bs_date > annual_end)
    ttm_flows = any(sel.source == "ttm_calculation" for sel in flows.values())
    if DCF_POST_BALANCE_SHEET_EVENT in codes:
        # Section 18: the inputs are the freshest that exist and a material
        # event has been filed since. Not a stale-input warning (nothing
        # newer could have been used) and not CURRENT either.
        return ValuationFreshness.MOSTLY_CURRENT_WITH_EVENT_WARNING
    if quarterly_balance and ttm_flows:
        return ValuationFreshness.CURRENT
    # An annual-only filer genuinely has nothing fresher, so this is not a
    # warning — but it is also not "current" in the sense a reader assumes.
    return ValuationFreshness.MOSTLY_CURRENT
