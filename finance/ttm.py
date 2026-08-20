"""Phase H.6 — TRUE trailing-twelve-month construction, with a stated method.

THE BUG THIS EXISTS TO FIX (live NVDA, valuation run 2026-08-17)
================================================================
The NVDA report carried the line

    Financial base: trailing twelve months to 26 Apr 2026

directly above a headline revenue of $215.94B. That figure is NVDA's FY2026
annual revenue (the year ended 2026-01-25) — not a trailing twelve months
through 2026-04-26 at all. The real TTM is $253.5B. A twelve-month window and
a fiscal year ARE different periods, and one quarter of NVDA's growth is a
$37B difference; labelling the annual figure "TTM" understates the base by
15% while telling the reader it does not.

The label came from one place and the number from another, and nothing
checked that they agreed. That is the whole failure mode this module closes:

    A value may not be called TTM unless a twelve-month window was actually
    constructed, from periods that actually tile that window, ending where
    the label says it ends.

WHAT IS DIFFERENT FROM finance/freshness.py::build_ttm
=====================================================
`build_ttm` summed four discrete quarters and gave up otherwise. That is one
valid construction, not the only one, and its failure mode was silent: the
caller fell back to the latest fiscal year and kept a TTM-shaped label
(`select_flow`'s `freshness_status` said `ttm` for the sum and
`current_annual` for the fallback, but everything downstream — the compact
report's "Financial base" line most of all — rendered both as a period ending
on the balance-sheet date).

This module adds the construction a reader would actually do by hand:

    TTM = latest fiscal year
          + year-to-date through the current quarter
          - year-to-date through the SAME quarter of the prior fiscal year

which needs only figures a 10-Q reports directly, works for a Q1, Q2 or Q3
roll-forward alike, and never depends on reconstructing a fourth quarter out
of a 10-K. When only one new quarter has been reported since fiscal year end
this reduces to exactly the arithmetic section 1 specifies: FY + Q1(current)
- Q1(prior).

Every result carries its `construction_method` and a `validation_status`, and
an unconstructable window returns TTM_INVALID_PERIOD_RECONSTRUCTION with a
reason rather than a number. Nothing here fabricates or interpolates a
period: every value is either a figure a filing reported or an explicit
arithmetic combination of such figures, and the combination is named.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from finance import period_facts as pf
from finance.period_facts import _concept_map

# Deterministic guard code (section 3). Emitted as a structured status on the
# result, never raised: an unconstructable TTM degrades a valuation to the
# annual basis with an explicit reason, it does not delete it.
TTM_INVALID_PERIOD_RECONSTRUCTION = "TTM_INVALID_PERIOD_RECONSTRUCTION"

# A trailing twelve months must actually span twelve months. Four quarters of
# a 52/53-week or 4-4-5 fiscal calendar land in this window; anything outside
# it means the periods do not tile a year.
TTM_SPAN_DAYS = (350, 380)

# Largest tolerated gap/overlap between two consecutive quarters, in days.
# Fiscal period boundaries do not always align to the day across filings
# (NVDA's own quarters join at 2025-10-26/2025-10-26 in one filing and
# 2025-10-26/2025-10-27 in the next).
MAX_QUARTER_JOIN_GAP = 7

# How far a metric's TTM may end BEFORE the company's latest reported period
# and still be called a current trailing twelve months. A company whose
# cash-flow tagging lags its income-statement tagging by one quarter is
# normal; six months behind is not the same period as the balance sheet and
# must not be silently combined with one.
#
# Live AT&T is exactly this case: `NetCashProvidedByUsedInOperatingActivities`
# stops at 2025-12-31 because T moved to the ...ContinuingOperations spelling
# in 2026, so operating cash flow produced a "TTM" ending 2025-12-31 while
# revenue's ended 2026-06-30 — and free cash flow was then derived by
# subtracting a capex window ending 2026-06-30 from an operating-cash-flow
# window ending 2025-12-31, a figure belonging to no period at all.
MAX_TRAILING_LAG_DAYS = 100


class TtmValidation:
    """Whether the constructed window is what its label claims."""

    VALID = "valid"

    @classmethod
    def worst_of(cls, *statuses) -> str:
        """The weakest status among several, for a value derived from more
        than one window. A figure built from a PARTIAL component is itself
        no better than PARTIAL."""
        rank = {cls.VALID: 0, cls.PARTIAL: 1, cls.INVALID: 2}
        found = [s for s in statuses if s in rank]
        return max(found, key=lambda s: rank[s]) if found else cls.INVALID

    # Constructed and arithmetically sound, but something a reader needs to
    # know is true of it — most often that it ends materially earlier than
    # the company's latest reported period.
    PARTIAL = "partial"
    INVALID = "invalid"
    ALL = (VALID, PARTIAL, INVALID)


class TtmConstruction:
    """How the twelve-month window was actually assembled."""

    FOUR_DISCRETE_QUARTERS = "four_discrete_quarters"
    ANNUAL_ROLL_FORWARD = "fiscal_year_plus_current_ytd_less_prior_ytd"
    NOT_CONSTRUCTED = "not_constructed"


@dataclass(frozen=True)
class TtmMetric:
    """One trailing-twelve-month figure and everything needed to check it.

    The shape section 3 requires, plus the unit (two facts in different units
    can never be summed) and the human-readable `reason` an invalid or
    partial result carries.
    """

    metric: str
    value: Optional[float]
    unit: str
    start_date: Optional[str]
    end_date: Optional[str]
    quarters_included: Tuple[str, ...]
    source_accessions: Tuple[str, ...]
    construction_method: str
    validation_status: str
    reason: Optional[str] = None
    used_reconstruction: bool = False
    components: Tuple[dict, ...] = ()
    # The underlying facts, for callers that need filing provenance (form,
    # accession, fiscal period) rather than just the arithmetic. Empty for
    # the roll-forward construction, whose inputs are an annual fact and two
    # year-to-date facts rather than quarters — `components` describes those.
    quarter_facts: Tuple[pf.PeriodFact, ...] = ()
    # The filing that supplied the LATEST period in the window, whichever
    # construction was used. This is what a citation should name.
    latest_accession: Optional[str] = None
    latest_form: Optional[str] = None
    latest_fiscal_period: Optional[str] = None

    @property
    def ok(self) -> bool:
        """Usable as a trailing-twelve-month figure. PARTIAL counts — it is a
        real twelve-month window with a caveat, not a broken one."""
        return self.validation_status in (TtmValidation.VALID, TtmValidation.PARTIAL)

    @property
    def span_days(self) -> Optional[int]:
        return pf._span_days(self.start_date, self.end_date)  # noqa: SLF001

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "quarters_included": list(self.quarters_included),
            "source_accessions": list(self.source_accessions),
            "construction_method": self.construction_method,
            "validation_status": self.validation_status,
            "reason": self.reason,
            "used_reconstruction": self.used_reconstruction,
            "span_days": self.span_days,
        }


def _invalid(metric: str, reason: str,
             method: str = TtmConstruction.NOT_CONSTRUCTED) -> TtmMetric:
    return TtmMetric(
        metric=metric, value=None, unit="USD", start_date=None, end_date=None,
        quarters_included=(), source_accessions=(), construction_method=method,
        validation_status=TtmValidation.INVALID,
        reason=f"{TTM_INVALID_PERIOD_RECONSTRUCTION}: {reason}")


# ---------------------------------------------------------------------------
# The company's latest reported period
# ---------------------------------------------------------------------------

# Flow lines nearly every filer tags, used to establish how current this
# company's reporting actually is. Deliberately income-statement lines: a
# company that filed a 10-Q reported revenue in it, whatever it did or did not
# do with its cash-flow tagging.
_REPORTING_ANCHORS = ("revenue", "net_income", "operating_income")


def latest_reported_period_end(company_facts: dict) -> Optional[str]:
    """The newest period END any anchor flow line was reported for.

    This is the yardstick for "is this metric's trailing twelve months
    actually trailing?" — a per-metric window is compared against what the
    company as a whole has reported, not against today's date (a filer is
    legitimately weeks or months behind the calendar) and not against the
    metric's own history (which is what made a stale concept look current).
    """
    ends = []
    for anchor in _REPORTING_ANCHORS:
        series = pf.discrete_quarters(company_facts, anchor, max_quarters=4)
        if series.quarters:
            ends.append(series.quarters[-1].end or "")
        annual = pf.annual_periods(company_facts, anchor)
        if annual:
            ends.append(annual[-1].end or "")
    ends = [e for e in ends if e]
    return max(ends) if ends else None


# ---------------------------------------------------------------------------
# Construction 1 — four contiguous discrete quarters
# ---------------------------------------------------------------------------


def _check_contiguous(quarters: Sequence[pf.PeriodFact], metric: str) -> Optional[str]:
    """No quarter counted twice, no quarter silently missing (section 3)."""
    for earlier, later in zip(quarters, quarters[1:]):
        gap = pf._span_days(earlier.end, later.start)  # noqa: SLF001
        if gap is None:
            return (f"{metric} quarters {earlier.start}..{earlier.end} and "
                    f"{later.start}..{later.end} cannot be joined; one of the dates is missing.")
        if gap < -MAX_QUARTER_JOIN_GAP:
            return (f"{metric} quarters {earlier.start}..{earlier.end} and "
                    f"{later.start}..{later.end} OVERLAP by {abs(gap)} days, so summing them "
                    "would count part of the year twice.")
        if gap > MAX_QUARTER_JOIN_GAP:
            return (f"{metric} quarters {earlier.start}..{earlier.end} and "
                    f"{later.start}..{later.end} leave a {gap}-day gap, so summing them would "
                    "skip part of the year.")
    ends = [q.end for q in quarters]
    if len(set(ends)) != len(ends):
        return f"{metric} has two quarters ending on the same date; one would be counted twice."
    return None


def _from_four_quarters(company_facts: dict, metric: str, offset: int = 0
                        ) -> Tuple[Optional[TtmMetric], Optional[str]]:
    """Sum the latest four contiguous discrete quarters, or say why not."""
    series = pf.discrete_quarters(company_facts, metric)
    available = series.quarters[:len(series.quarters) - offset] if offset else series.quarters
    quarters = available[-4:] if len(available) >= 4 else []
    if len(quarters) < 4:
        return None, (
            f"only {len(available)} discrete quarter(s) of {metric} could be built"
            + (" before the requested offset" if offset else "")
            + "; four are required")

    problem = _check_contiguous(quarters, metric)
    if problem:
        return None, problem

    total_span = pf._span_days(quarters[0].start, quarters[-1].end)  # noqa: SLF001
    low, high = TTM_SPAN_DAYS
    if total_span is None or not (low <= total_span <= high):
        return None, (f"{metric}'s four quarters span {total_span} days, outside the "
                      f"{low}-{high} day window a trailing twelve months must cover")

    units = {q.unit for q in quarters}
    if len(units) > 1:
        return None, (f"{metric} quarters mix units ({', '.join(sorted(units))}); "
                      "they cannot be summed")

    return TtmMetric(
        metric=metric,
        value=sum(q.value for q in quarters),
        unit=quarters[0].unit,
        start_date=quarters[0].start,
        end_date=quarters[-1].end,
        quarters_included=tuple(f"{q.start}..{q.end}" for q in quarters),
        source_accessions=tuple(dict.fromkeys(q.accession for q in quarters if q.accession)),
        construction_method=TtmConstruction.FOUR_DISCRETE_QUARTERS,
        validation_status=TtmValidation.VALID,
        used_reconstruction=any(q.reconstructed_from for q in quarters),
        components=tuple(q.to_dict() for q in quarters),
        quarter_facts=tuple(quarters),
        latest_accession=quarters[-1].accession or None,
        latest_form=quarters[-1].form or None,
        latest_fiscal_period=quarters[-1].fiscal_period,
    ), None


# ---------------------------------------------------------------------------
# Construction 2 — fiscal year + current YTD - prior-year YTD
# ---------------------------------------------------------------------------


def _ytd_facts_by_year(company_facts: dict, metric: str
                       ) -> Tuple[Optional[str], Dict[str, List[dict]]]:
    """Year-to-date duration facts grouped by their fiscal-year START date.

    Facts sharing a start date are one fiscal year's YTD chain (Q1 ~90d,
    H1 ~181d, 9M ~273d), because year-to-date resets at the fiscal year start.
    """
    concept_map = _concept_map(company_facts)
    if metric not in concept_map:
        return None, {}
    is_instant, _ = concept_map[metric]
    if is_instant:
        return None, {}
    concept, facts = pf._winning_concept_facts(company_facts, metric)  # noqa: SLF001
    if not concept:
        return None, {}
    usable = [f for f in facts
              if f.get("start") and f.get("end") and f.get("val") is not None]
    grouped: Dict[str, List[dict]] = {}
    for raw in pf._dedupe_by_period(usable):  # noqa: SLF001
        span = pf._span_days(raw["start"], raw["end"])  # noqa: SLF001
        if span is not None and 0 < span <= pf.YTD_MAX_DAYS:
            grouped.setdefault(raw["start"], []).append(raw)
    return concept, grouped


def _from_annual_roll_forward(company_facts: dict, metric: str
                              ) -> Tuple[Optional[TtmMetric], Optional[str]]:
    """TTM = latest fiscal year + current YTD - prior-year YTD of equal length.

    The construction section 1 specifies. `_ytd_quarter_index` supplies the
    "equal length" test by fiscal quarter rather than by raw day count, so a
    52/53-week calendar whose H1 is 181 days one year and 188 the next still
    matches its own prior-year column.
    """
    annual = pf.annual_periods(company_facts, metric)
    if not annual:
        return None, f"no annual {metric} figure is available to roll forward from"
    latest_annual = annual[-1]

    concept, ytd_by_year = _ytd_facts_by_year(company_facts, metric)
    if not concept:
        return None, f"no reviewed concept for {metric} appears in these filings"

    # The CURRENT fiscal year's YTD chain starts where the latest annual
    # period ended (allowing for the one-day boundary convention filers vary
    # on). Anything else is a prior year and cannot roll the window forward.
    current_year_starts = [
        start for start in ytd_by_year
        if start > (latest_annual.start or "")
        and (gap := pf._span_days(latest_annual.end, start)) is not None  # noqa: SLF001
        and -1 <= gap <= MAX_QUARTER_JOIN_GAP
    ]
    if not current_year_starts:
        return None, (f"no year-to-date {metric} figure has been reported since the fiscal "
                      f"year ending {latest_annual.end}, so there is nothing to roll forward")
    current_start = max(current_year_starts)

    current = None
    for raw in sorted(ytd_by_year[current_start], key=lambda r: r["end"]):
        index = pf._ytd_quarter_index(raw)  # noqa: SLF001
        if index is not None and (current is None or raw["end"] > current[1]["end"]):
            current = (index, raw)
    if current is None:
        return None, (f"the {metric} figures reported since {latest_annual.end} do not have a "
                      "recognizable year-to-date length")
    current_index, current_raw = current

    prior_chain = ytd_by_year.get(latest_annual.start or "") or []
    prior_raw = None
    for raw in prior_chain:
        if pf._ytd_quarter_index(raw) == current_index:  # noqa: SLF001
            prior_raw = raw
            break
    if prior_raw is None:
        return None, (
            f"{metric} was reported year-to-date through fiscal quarter {current_index} of the "
            f"current year ({current_raw['start']}..{current_raw['end']}) but the SAME "
            f"year-to-date period of the prior fiscal year (starting {latest_annual.start}) was "
            "not; subtracting a different-length period would not produce twelve months")

    if latest_annual.unit != current_raw.get("_unit", "USD") \
            or current_raw.get("_unit", "USD") != prior_raw.get("_unit", "USD"):
        return None, (f"the annual and year-to-date {metric} figures are reported in different "
                      "units and cannot be combined")

    # The window: (prior_ytd_end, current_ytd_end]. The annual figure supplies
    # (prior_ytd_end, annual_end] once its own first months are removed, and
    # the current year-to-date supplies (annual_end, current_ytd_end].
    start_date = prior_raw["end"]
    end_date = current_raw["end"]
    span = pf._span_days(start_date, end_date)  # noqa: SLF001
    low, high = TTM_SPAN_DAYS
    if span is None or not (low <= span <= high):
        return None, (f"rolling {metric} forward from {latest_annual.end} produces a window of "
                      f"{span} days ({start_date}..{end_date}), outside the {low}-{high} day "
                      "window a trailing twelve months must cover")

    value = (float(latest_annual.value) + float(current_raw["val"]) - float(prior_raw["val"]))
    components = (
        {"role": "fiscal_year", "start": latest_annual.start, "end": latest_annual.end,
         "value": float(latest_annual.value), "accession": latest_annual.accession},
        {"role": "current_year_to_date", "start": current_raw["start"], "end": current_raw["end"],
         "value": float(current_raw["val"]), "accession": current_raw.get("accn", "")},
        {"role": "prior_year_to_date", "start": prior_raw["start"], "end": prior_raw["end"],
         "value": float(prior_raw["val"]), "accession": prior_raw.get("accn", "")},
    )
    return TtmMetric(
        metric=metric,
        value=value,
        unit=latest_annual.unit,
        start_date=start_date,
        end_date=end_date,
        quarters_included=(
            f"{latest_annual.start}..{latest_annual.end} (fiscal year)",
            f"{current_raw['start']}..{current_raw['end']} (year-to-date, added)",
            f"{prior_raw['start']}..{prior_raw['end']} (prior-year year-to-date, subtracted)",
        ),
        source_accessions=tuple(dict.fromkeys(
            c["accession"] for c in components if c["accession"])),
        construction_method=TtmConstruction.ANNUAL_ROLL_FORWARD,
        validation_status=TtmValidation.VALID,
        used_reconstruction=True,
        components=components,
        latest_accession=current_raw.get("accn") or None,
        latest_form=current_raw.get("form") or None,
        latest_fiscal_period=current_raw.get("fp"),
    ), None


# ---------------------------------------------------------------------------
# The public builder
# ---------------------------------------------------------------------------


def build_ttm(company_facts: dict, metric: str, offset: int = 0,
              reference_end: Optional[str] = None) -> TtmMetric:
    """The trailing twelve months for one FLOW metric, or a stated failure.

    Tries the four-discrete-quarter construction first — it needs no annual
    figure and its provenance is the simplest to read — then the annual
    roll-forward, which is what rescues a company whose fourth quarter cannot
    be reconstructed but whose 10-Q reports a clean year-to-date column.

    `offset` steps the window back by whole quarters (offset=4 gives the
    prior-year trailing twelve months), which is what makes a TTM-over-TTM
    growth rate possible. Only the four-quarter construction supports it: a
    roll-forward has no meaningful "one quarter earlier" form.

    `reference_end` is the company's latest reported period; when the
    constructed window ends materially before it, the result is PARTIAL and
    says so rather than passing as a current trailing twelve months.

    Balance-sheet fields are refused outright (section 4): a point-in-time
    balance has no twelve-month sum, and averaging one would invent a figure
    no filing reported.
    """
    concept_map = _concept_map(company_facts)
    if metric not in concept_map:
        return _invalid(metric, f"{metric!r} has no reviewed XBRL concept mapping")
    is_instant, _ = concept_map[metric]
    if is_instant:
        return _invalid(
            metric,
            f"{metric!r} is a point-in-time balance-sheet field, not a flow; balance-sheet "
            "values are never rolled into a trailing twelve months (they stay point-in-time)")

    result, quarters_reason = _from_four_quarters(company_facts, metric, offset=offset)
    roll_reason = None
    if result is None and not offset:
        result, roll_reason = _from_annual_roll_forward(company_facts, metric)

    if result is None:
        reasons = [r for r in (quarters_reason, roll_reason) if r]
        return _invalid(metric, "; and ".join(reasons) or "no twelve-month window could be built")

    if reference_end and result.end_date:
        lag = pf._span_days(result.end_date, reference_end)  # noqa: SLF001
        if lag is not None and lag > MAX_TRAILING_LAG_DAYS:
            return TtmMetric(
                metric=result.metric, value=result.value, unit=result.unit,
                start_date=result.start_date, end_date=result.end_date,
                quarters_included=result.quarters_included,
                source_accessions=result.source_accessions,
                construction_method=result.construction_method,
                validation_status=TtmValidation.PARTIAL,
                reason=(f"This twelve-month window ends {result.end_date}, {lag} days before "
                        f"{reference_end}, the latest period this company has reported. It is a "
                        "valid twelve months but NOT a current one, and must not be combined "
                        "with figures from the latest period."),
                used_reconstruction=result.used_reconstruction,
                components=result.components,
                quarter_facts=result.quarter_facts,
                latest_accession=result.latest_accession,
                latest_form=result.latest_form,
                latest_fiscal_period=result.latest_fiscal_period)

    return result


def build_ttm_set(company_facts: dict, metrics: Sequence[str],
                  reference_end: Optional[str] = None) -> Dict[str, TtmMetric]:
    """Every requested metric's TTM, against one shared reference period."""
    reference = reference_end or latest_reported_period_end(company_facts)
    return {metric: build_ttm(company_facts, metric, reference_end=reference)
            for metric in metrics}


def periods_are_comparable(left: Optional[TtmMetric], right: Optional[TtmMetric]) -> bool:
    """May these two twelve-month figures be combined arithmetically?

    Required before deriving free cash flow (operating cash flow less capital
    expenditure) — subtracting a window ending 2026-06-30 from one ending
    2025-12-31 produces a figure belonging to no period, which is exactly what
    the live AT&T run did and then labelled TTM.
    """
    if left is None or right is None or not left.ok or not right.ok:
        return False
    if left.unit != right.unit:
        return False
    if left.end_date != right.end_date:
        return False
    return left.start_date == right.start_date
