"""Phase H.4 — SEC facts keyed on the period they ACTUALLY cover.

Everything else in this project groups XBRL facts by their `(fy, fp)` tags
(see finance/xbrl_mapping.py::list_available_periods). That is correct for
"give me the last five annual statements", but it is the WRONG key for
freshness, and the AOS root-cause investigation showed exactly why:

    fy=2026 fp=Q2 form=10-Q start=2025-04-01 end=2025-06-30  val=1,011.3M
    fy=2026 fp=Q2 form=10-Q start=2026-04-01 end=2026-06-30  val=1,004.3M
    fy=2026 fp=Q2 form=10-Q start=2025-01-01 end=2025-06-30  val=1,975.2M
    fy=2026 fp=Q2 form=10-Q start=2026-01-01 end=2026-06-30  val=1,949.9M

FOUR facts, one `(fy, fp)` bucket. In SEC companyfacts `fy`/`fp` describe the
FILING the fact appeared in, not the period the fact measures — so a 10-Q's
prior-year comparative columns carry the CURRENT filing's fiscal tags. A
"latest quarter" chosen by `(fy, fp)` can therefore silently be a
year-old comparative figure, and a "quarterly revenue" can silently be a
half-year YTD total.

This module keys on `start`/`end` alone. `fy`/`fp`/`form`/`accn`/`filed` are
carried through as PROVENANCE (every selection has to name its filing), never
used to decide which period a fact belongs to.

The second thing this module exists for is YTD cumulative facts. A 10-Q's
cash-flow statement is reported year-to-date, not per-quarter — live AOS has
NO 90-day operating-cash-flow fact at all, only 180d/272d/364d spans:

    start=2026-01-01 end=2026-06-30  span=180d  val=178.3M   (H1, not Q2)

Summing those would double-count Q1. `discrete_quarters()` reconstructs the
discrete quarter by DIFFERENCING consecutive YTD facts within the same fiscal
year (Q2 = H1YTD - Q1YTD), and refuses to guess when the chain is broken.

Nothing here fabricates a value. Every function returns either a fact that a
filing actually reported (or an explicit arithmetic difference of two such
facts, labelled as reconstructed) or None with a stated reason.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from finance.xbrl_mapping import CONCEPT_MAP, _candidate_facts


def _concept_map(company_facts: dict):
    """The reviewed field -> concept mapping for THIS issuer's framework.

    Phase H.7. Every function below used the module-level us-gaap
    `CONCEPT_MAP` directly, which made a foreign private issuer's entire
    history unreadable: an ifrs-full filer has no `RevenueFromContractWith-
    CustomerExcludingAssessedTax`, so `revenue` resolved to nothing and the
    pipeline reported the company as having published no financials.
    """
    from finance import taxonomy as taxonomy_module
    return taxonomy_module.concept_map_for(
        taxonomy_module.detect_taxonomy(company_facts))

# A discrete quarter is ~13 weeks. 4-4-5 retail calendars and 52/53-week
# fiscal years stretch this, so the window is generous on both sides but
# nowhere near wide enough to admit a half-year (180d) or a year (364d) —
# the only distinctions that actually matter here.
QUARTER_DAYS = (75, 115)
# Year-to-date spans a 10-Q can legitimately carry: Q1 (~90d), H1 (~181d),
# 9M (~273d). The upper bound stops short of a full year so an annual fact
# riding along in a 10-K is never mistaken for a YTD column.
YTD_MAX_DAYS = 300
ANNUAL_DAYS = (300, 400)

# Fiscal quarter index by YTD length. A YTD fact ending N days after the
# fiscal year began is the Nth quarter cumulative.
_YTD_QUARTER_BOUNDS = ((0, 115, 1), (116, 210, 2), (211, 300, 3))


class DurationType:
    """What KIND of period a fact covers (section 1).

    The distinction that matters is not "quarterly vs annual" but whether two
    facts may be COMPARED OR SUMMED at all. A 6-month year-to-date column and
    a discrete quarter are both "interim" and adding them double-counts three
    months; a trailing twelve months and a fiscal year are both twelve months
    and are still not the same period.

    `OTHER` is deliberate and load-bearing: an issuer that tags an
    11-month transition period, or a 45-day stub after a reorganization,
    produces a real fact that no rule here should silently treat as a quarter.
    """

    INSTANT = "INSTANT"
    QUARTER = "QUARTER"
    YTD_6M = "YTD_6M"
    YTD_9M = "YTD_9M"
    HALF_YEAR = "HALF_YEAR"
    ANNUAL = "ANNUAL"
    TTM = "TTM"
    OTHER = "OTHER"
    ALL = (INSTANT, QUARTER, YTD_6M, YTD_9M, HALF_YEAR, ANNUAL, TTM, OTHER)

    # Types that are cumulative from a fiscal-year start. Summing two of
    # these, or a cumulative one with a discrete one, double-counts.
    CUMULATIVE = frozenset({YTD_6M, YTD_9M, HALF_YEAR})
    # Types that measure one non-overlapping slice of time.
    DISCRETE = frozenset({QUARTER})
    # Types that already span twelve months.
    FULL_YEAR = frozenset({ANNUAL, TTM})


# Day-span windows for each duration type. Generous on both sides because
# 52/53-week and 4-4-5 fiscal calendars stretch every boundary, but never
# wide enough for two adjacent types to overlap -- that is the only property
# these windows must guarantee.
_DURATION_WINDOWS = (
    (75, 115, DurationType.QUARTER),
    (150, 210, DurationType.HALF_YEAR),
    (240, 300, DurationType.YTD_9M),
    (300, 400, DurationType.ANNUAL),
)


def classify_duration(start: Optional[str], end: Optional[str],
                      fiscal_year_start: Optional[str] = None) -> str:
    """The duration type of one fact, from its actual dates.

    `fiscal_year_start` disambiguates the one genuinely ambiguous case: a
    ~181-day span is a HALF_YEAR if it stands alone and a YTD_6M if it starts
    at the fiscal year start, and only the caller knows which. Without it the
    span is reported as HALF_YEAR, which is the weaker claim.
    """
    if start is None:
        return DurationType.INSTANT if end else DurationType.OTHER
    span = _span_days(start, end)
    if span is None:
        return DurationType.OTHER
    for low, high, kind in _DURATION_WINDOWS:
        if low <= span <= high:
            if kind == DurationType.HALF_YEAR and fiscal_year_start                     and start == fiscal_year_start:
                return DurationType.YTD_6M
            if kind == DurationType.YTD_9M and fiscal_year_start                     and start != fiscal_year_start:
                return DurationType.OTHER
            return kind
    return DurationType.OTHER


def periods_are_summable(left: "PeriodFact", right: "PeriodFact") -> Tuple[bool, str]:
    """May these two facts be added? (ok, reason)

    Section 1's "do not compare or sum periods without validating
    compatibility", as one function. Every caller that combines two facts
    goes through here rather than re-deriving the rule.
    """
    if left.unit != right.unit:
        return False, f"units differ ({left.unit} vs {right.unit})"
    if left.currency != right.currency:
        return False, f"currencies differ ({left.currency} vs {right.currency})"
    if left.duration_type == DurationType.INSTANT or             right.duration_type == DurationType.INSTANT:
        return False, "a point-in-time balance cannot be summed with a period"
    if left.duration_type in DurationType.CUMULATIVE or             right.duration_type in DurationType.CUMULATIVE:
        return False, ("a cumulative year-to-date figure cannot be summed; difference "
                       "consecutive year-to-date facts instead")
    if left.start and right.start and left.end and right.end:
        if left.start < right.end and right.start < left.end:
            return False, f"periods overlap ({left.start}..{left.end}, {right.start}..{right.end})"
    return True, ""


@dataclass(frozen=True)
class PeriodFact:
    """One fact plus every piece of provenance a citation needs.

    `reconstructed_from` is empty for a value a filing reported directly, and
    names the two YTD facts that were differenced when the value was
    reconstructed (see `discrete_quarters`) — a reconstructed value is never
    presented as a directly reported one.
    """

    field: str
    value: float
    unit: str
    concept: str
    accession: str
    form: str
    filed: str
    fiscal_year: Optional[int]
    fiscal_period: Optional[str]
    start: Optional[str]
    end: Optional[str]
    reconstructed_from: Tuple[str, ...] = ()
    # Section 1. `currency` is separate from `unit` because a fact can be
    # USD-denominated shares or a pure ratio; `amended` records that the
    # value arrived on an amending form (10-K/A, 10-Q/A, 20-F/A), which is
    # why it supersedes an earlier value for the same period; `taxonomy`
    # records which reporting framework produced it, so a us-gaap figure is
    # never silently combined with an ifrs-full one.
    currency: str = "USD"
    amended: bool = False
    taxonomy: str = "us-gaap"
    entity_id: Optional[str] = None

    @property
    def duration_type(self) -> str:
        return classify_duration(self.start, self.end)

    @property
    def is_instant(self) -> bool:
        return self.start is None

    @property
    def duration_days(self) -> Optional[int]:
        return _span_days(self.start, self.end)

    @property
    def as_of_date(self) -> Optional[str]:
        """The date this fact speaks to — its `end` in both the instant
        (balance-sheet snapshot) and duration (period close) cases."""
        return self.end

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "value": self.value,
            "unit": self.unit,
            "concept": self.concept,
            "accession_number": self.accession,
            "form": self.form,
            "filed": self.filed,
            "fiscal_year": self.fiscal_year,
            "fiscal_period": self.fiscal_period,
            "start": self.start,
            "end": self.end,
            "as_of_date": self.as_of_date,
            "duration_days": self.duration_days,
            "duration_type": self.duration_type,
            "currency": self.currency,
            "amended": self.amended,
            "taxonomy": self.taxonomy,
            "entity_id": self.entity_id,
            "reconstructed_from": list(self.reconstructed_from),
        }


def _span_days(start, end) -> Optional[int]:
    if not start or not end:
        return None
    try:
        return (date.fromisoformat(end) - date.fromisoformat(start)).days
    except (ValueError, TypeError):
        return None


def _to_fact(field_name: str, raw: dict, concept: str,
             reconstructed_from: Sequence[str] = ()) -> PeriodFact:
    return PeriodFact(
        field=field_name,
        value=float(raw["val"]),
        unit=raw.get("_unit", "USD"),
        concept=concept,
        accession=raw.get("accn", ""),
        form=raw.get("form", ""),
        filed=raw.get("filed", ""),
        fiscal_year=raw.get("fy"),
        fiscal_period=raw.get("fp"),
        start=raw.get("start"),
        end=raw.get("end"),
        reconstructed_from=tuple(reconstructed_from),
        currency=raw.get("_currency", "USD"),
        # An amending form supersedes the original for the same period. This
        # is already handled by latest-filed-wins in `_dedupe_by_period`; the
        # flag exists so a report can SAY the figure was restated rather than
        # leaving a reader to compare accession numbers.
        amended=str(raw.get("form", "")).endswith("/A"),
        taxonomy=raw.get("_taxonomy", "us-gaap"),
    )


def _winning_concept_facts(company_facts: dict, field_name: str) -> Tuple[Optional[str], List[dict]]:
    """The concept whose data actually reaches FURTHEST FORWARD, with
    finance/xbrl_mapping.py's precedence order as the tie-break.

    NOT simply "the first candidate that has any fact" — that is the rule
    `resolve_concept` uses, and it is wrong for a freshness question. Live
    AOS proves it: `short_term_debt`'s candidates are
    (`ShortTermBorrowings`, `DebtCurrent`). AOS reported
    `ShortTermBorrowings` exactly twice, both in 2010, and has never used
    `DebtCurrent`. First-candidate-wins returns a 2010 balance as today's
    short-term debt. Recency-first at least prefers whichever tag the
    company still uses; a tag that is stale in ABSOLUTE terms is then caught
    by the balance-sheet-date coherence rule in `instant_as_of`.

    Only ONE concept ever wins for the whole series. Mixing concepts across
    periods would mean differencing two year-to-date facts that came from
    different tags, which is not a defensible reconstruction.
    """
    concept_map = _concept_map(company_facts)
    if field_name not in concept_map:
        return None, []
    _is_instant, candidates = concept_map[field_name]
    best = None
    for order, concept in enumerate(candidates):
        facts = _candidate_facts(company_facts, concept)
        if not facts:
            continue
        latest_end = max((f.get("end") or "") for f in facts)
        # Greatest `end` wins; earlier precedence breaks a tie (negated order,
        # since we are taking a maximum).
        key = (latest_end, -order)
        if best is None or key > best[0]:
            best = (key, concept, facts)
    if best is None:
        return None, []
    winner_concept, winner_facts = best[1], best[2]
    return winner_concept, _merge_equivalent_concepts(
        company_facts, field_name, winner_concept, winner_facts, candidates)


# How closely two concepts must agree on a shared period to be treated as the
# same series. Exact equality in practice; the tolerance only absorbs the
# float round-tripping of a JSON payload.
_CONCEPT_EQUIVALENCE_TOLERANCE = 1e-6
# At least this many periods must be reported under BOTH concepts before one
# can be declared a continuation of the other. One coincidental match is not
# evidence; three identical periods is a tag rename.
_MIN_OVERLAP_FOR_EQUIVALENCE = 3
# How far back the equivalence test looks, in days before the winning
# concept's latest period.
#
# Scoped to the RECENT era on purpose. Two spellings of the same line can
# legitimately differ in the past and coincide now: AT&T's total-company and
# continuing-operations operating cash flow differ in 2017, 2020 and 2021 —
# the years it HAD discontinued operations — and are identical from 2023
# onward, because the separations are complete. Testing over all history
# would refuse the merge on evidence about a company that no longer exists,
# which is the same mistake finance/structural_breaks.py exists to prevent.
# Merging is likewise confined to this window, so the chain never crosses the
# era where the two genuinely disagreed.
_EQUIVALENCE_WINDOW_DAYS = 3 * 366


def _merge_equivalent_concepts(company_facts: dict, field_name: str, winner_concept: str,
                               winner_facts: List[dict],
                               candidates: Sequence[str]) -> List[dict]:
    """Extend the winning concept's series with an EARLIER concept that is
    demonstrably the same series under a different tag.

    Phase H.6. AT&T renamed its operating-cash-flow tag for fiscal 2026:
    `NetCashProvidedByUsedInOperatingActivities` stops at 2025-12-31 and
    `...ContinuingOperations` starts. Choosing one concept for the whole
    series — the rule that keeps this module from differencing two unrelated
    tags — then leaves EITHER a stale window (the old tag, ending six months
    early) or too little history to build twelve months from (the new tag,
    which has no fiscal-year figure to roll forward from). T's operating cash
    flow and therefore its free cash flow came out unavailable.

    The merge is only performed when the two tags are PROVED interchangeable
    on this issuer's own data: every period reported under both must carry
    the same value, and there must be at least three such periods. AT&T's do
    (Q1 2025 = $9,049M and H1 2025 = $18,812M under both spellings) because
    the change is a tagging change, not a change of basis. An issuer whose
    continuing-operations figures genuinely DIFFER from its total-company
    ones fails the test and keeps the single-concept behaviour, which is the
    conservative outcome — the periods stay unmixed.
    """
    latest_end = max((f.get("end") or "") for f in winner_facts)
    if not latest_end:
        return winner_facts
    try:
        cutoff = (date.fromisoformat(latest_end)
                  - timedelta(days=_EQUIVALENCE_WINDOW_DAYS)).isoformat()
    except (ValueError, TypeError):
        return winner_facts

    winner_by_period = {(f.get("start"), f.get("end")): f for f in winner_facts}
    merged = list(winner_facts)
    for concept in candidates:
        if concept == winner_concept:
            continue
        other = [f for f in _candidate_facts(company_facts, concept)
                 if (f.get("end") or "") >= cutoff]
        if not other:
            continue
        agreements = 0
        for fact in other:
            key = (fact.get("start"), fact.get("end"))
            existing = winner_by_period.get(key)
            if existing is None:
                continue
            try:
                if abs(float(fact["val"]) - float(existing["val"])) > max(
                        abs(float(existing["val"])) * _CONCEPT_EQUIVALENCE_TOLERANCE, 1.0):
                    agreements = -1
                    break
            except (TypeError, ValueError, KeyError):
                agreements = -1
                break
            agreements += 1
        if agreements < _MIN_OVERLAP_FOR_EQUIVALENCE:
            continue
        for fact in other:
            if (fact.get("start"), fact.get("end")) not in winner_by_period:
                merged.append(fact)
                winner_by_period[(fact.get("start"), fact.get("end"))] = fact
    return merged


def _dedupe_by_period(facts: List[dict]) -> List[dict]:
    """One fact per (start, end), keeping the LATEST-FILED one.

    The same period is re-reported in every subsequent filing (a 10-K repeats
    two prior years; each 10-Q repeats the prior-year comparative). Later
    filings supersede earlier ones for the same period — this is also how an
    amended 10-K/A correctly replaces the original without the caller having
    to know a restatement happened.
    """
    best: Dict[Tuple, dict] = {}
    for fact in facts:
        key = (fact.get("start"), fact.get("end"))
        current = best.get(key)
        if current is None or (fact.get("filed") or "") > (current.get("filed") or ""):
            best[key] = fact
    return sorted(best.values(), key=lambda f: (f.get("end") or "", f.get("start") or ""))


def latest_instant(company_facts: dict, field_name: str,
                   not_after: Optional[str] = None) -> Optional[PeriodFact]:
    """The most recent POINT-IN-TIME value for a balance-sheet field.

    "Most recent" means the largest `end` date — NOT the newest filing. Every
    10-Q restates the prior year-end balance as its comparative column, so
    the newest FILING routinely contains an OLD balance:

        filed=2026-07-30  end=2025-12-31  cash=174.5M   <- comparative
        filed=2026-07-30  end=2026-06-30  cash=181.3M   <- current

    Ties on `end` are broken by filing date so a restatement wins.
    """
    concept_map = _concept_map(company_facts)
    if field_name not in concept_map:
        return None
    is_instant, _candidates = concept_map[field_name]
    if not is_instant:
        return None
    concept, facts = _winning_concept_facts(company_facts, field_name)
    if not concept:
        return None
    usable = [f for f in facts if not f.get("start") and f.get("end") and f.get("val") is not None]
    if not_after:
        usable = [f for f in usable if f["end"] <= not_after]
    if not usable:
        return None
    usable.sort(key=lambda f: (f.get("end") or "", f.get("filed") or ""))
    return _to_fact(field_name, usable[-1], concept)


def instant_as_of(company_facts: dict, field_name: str, as_of: str) -> Optional[PeriodFact]:
    """A balance-sheet field AT ONE SPECIFIC DATE, or None.

    A balance sheet is a coherent snapshot of one moment. Reading cash from
    2026-06-30 and short-term debt from 2010-09-30 into the same net-debt
    calculation is not a balance sheet — it is two unrelated numbers
    subtracted from each other. So every field is required to have been
    reported AT the selected date; a field that was not is ABSENT (the
    caller decides whether absence means "the company has none" or "unknown"
    — see finance/freshness.py), never back-filled from an older filing.
    """
    if not as_of:
        return None
    fact = latest_instant(company_facts, field_name, not_after=as_of)
    if fact is None or fact.end != as_of:
        return None
    return fact


# Fields that must be present for a date to count as a real balance-sheet
# date. Every filed balance sheet reports total assets; requiring a second,
# independent field guards against a stray one-off tag defining a date no
# actual balance sheet was published for.
_BALANCE_SHEET_ANCHORS = ("assets", "stockholders_equity")


def latest_balance_sheet_date(company_facts: dict) -> Optional[str]:
    """The most recent date at which a full balance sheet was actually filed.

    This is the date the equity bridge is built AT. Derived from anchor
    fields rather than from "the newest filing", because every 10-Q also
    restates the PRIOR year-end as its comparative column — so the newest
    filing contains two balance-sheet dates and only one of them is current.
    """
    dates = []
    for anchor in _BALANCE_SHEET_ANCHORS:
        fact = latest_instant(company_facts, anchor)
        if fact is not None and fact.end:
            dates.append(fact.end)
    if not dates:
        return None
    # The EARLIEST of the anchors' latest dates: a date is only usable when
    # every anchor was reported at it. (They are normally identical; they
    # diverge when one anchor appears in a later filing than the other.)
    return min(dates)


def annual_periods(company_facts: dict, field_name: str) -> List[PeriodFact]:
    """Every distinct full-year duration fact, oldest first."""
    concept_map = _concept_map(company_facts)
    if field_name not in concept_map:
        return []
    is_instant, _ = concept_map[field_name]
    if is_instant:
        return []
    concept, facts = _winning_concept_facts(company_facts, field_name)
    if not concept:
        return []
    low, high = ANNUAL_DAYS
    annual = [f for f in facts
              if f.get("start") and f.get("end") and f.get("val") is not None
              and (d := _span_days(f["start"], f["end"])) is not None and low <= d <= high]
    return [_to_fact(field_name, f, concept) for f in _dedupe_by_period(annual)]


def _ytd_quarter_index(fact: dict) -> Optional[int]:
    """Which fiscal quarter a YTD fact cumulates THROUGH (1..3), by its span."""
    span = _span_days(fact.get("start"), fact.get("end"))
    if span is None:
        return None
    for low, high, index in _YTD_QUARTER_BOUNDS:
        if low <= span <= high:
            return index
    return None


# How far back a reconstruction gap is still worth reporting. Companyfacts
# carries 15+ years of filings, and interim-period tagging in the early years
# is patchy for most issuers; a broken chain in 2009 says nothing about a
# valuation being built today.
_RECENT_WARNING_YEARS = 3


def _is_recent(end: Optional[str], today: Optional[date] = None) -> bool:
    if not end:
        return False
    try:
        parsed = date.fromisoformat(end)
    except (ValueError, TypeError):
        return False
    reference = today or date.today()
    return (reference - parsed).days <= _RECENT_WARNING_YEARS * 366


@dataclass
class QuarterSeries:
    """Discrete quarters plus an explicit account of what could not be built."""

    field: str
    quarters: List[PeriodFact] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # True when at least one quarter had to be reconstructed by differencing
    # YTD facts rather than read directly from a reported discrete figure.
    used_reconstruction: bool = False

    def latest(self, n: int) -> List[PeriodFact]:
        return self.quarters[-n:] if len(self.quarters) >= n else []


def discrete_quarters(company_facts: dict, field_name: str,
                      max_quarters: int = 12) -> QuarterSeries:
    """Discrete (non-cumulative) quarterly values for a FLOW field.

    Two sources, in this order:

    1. A directly reported ~quarter-length fact. Preferred always — no
       arithmetic, no reconstruction risk.
    2. Reconstruction by differencing consecutive YTD facts inside one fiscal
       year: Q(n) = YTD(n) - YTD(n-1). Only ever within a single fiscal year
       and only between ADJACENT quarter indices, so a missing Q2 can never
       be papered over by differencing Q3YTD against Q1YTD (that difference
       is two quarters of activity, not one).

    3. The FOURTH quarter, from the annual fact minus the same fiscal year's
       nine-month year-to-date fact. No company files a Q4 10-Q — the
       fourth-quarter figure exists only inside the 10-K — so without this
       step every trailing-twelve-month window that spans a year end has a
       hole in it and no TTM can ever be built for the first three quarters
       of any year. (Found exactly this way: AOS's quarters ran
       ...2025-09-30, then jumped to 2026-01-01, and every flow silently
       fell back to the FY2025 annual figure.)

    Fails closed: an unreconcilable chain produces a warning and a SHORTER
    series, never an interpolated quarter.
    """
    series = QuarterSeries(field=field_name)
    concept_map = _concept_map(company_facts)
    if field_name not in concept_map:
        series.warnings.append(f"{field_name!r} has no reviewed XBRL concept mapping.")
        return series
    is_instant, _ = concept_map[field_name]
    if is_instant:
        series.warnings.append(
            f"{field_name!r} is a point-in-time balance, not a flow; it has no discrete quarters.")
        return series

    concept, facts = _winning_concept_facts(company_facts, field_name)
    if not concept:
        series.warnings.append(f"No reviewed concept for {field_name!r} appears in these filings.")
        return series

    usable = [f for f in facts
              if f.get("start") and f.get("end") and f.get("val") is not None]
    deduped = _dedupe_by_period(usable)

    q_low, q_high = QUARTER_DAYS
    interim = [f for f in deduped
               if (d := _span_days(f["start"], f["end"])) is not None and d <= YTD_MAX_DAYS]

    # Key every discrete quarter by its END date: that is what makes a
    # reconstructed quarter comparable to a directly reported one.
    by_end: Dict[str, PeriodFact] = {}

    # Group interim facts by their START date. Facts sharing a start date are
    # the year-to-date series for one fiscal year (Q1 ~90d, H1 ~181d, 9M
    # ~273d) — YTD resets at the fiscal year start, so grouping this way is
    # what keeps differencing inside a single year.
    #
    # A ~quarter-length fact starting at the fiscal year start is BOTH the
    # discrete first quarter AND the first link of that YTD chain, so it must
    # stay in the group. Classifying it as "direct, therefore not YTD" is
    # what silently broke Q2 reconstruction for AOS's operating cash flow:
    # the H1 fact was then the only member of its group, had no Q1 to
    # difference against, and the current quarter went missing entirely.
    by_start: Dict[str, List[dict]] = {}
    for raw in interim:
        by_start.setdefault(raw["start"], []).append(raw)

    # Pass 1 — directly reported discrete quarters. These never need
    # arithmetic, so they win over any reconstruction for the same end date.
    for raw in interim:
        span = _span_days(raw["start"], raw["end"])
        if span is not None and q_low <= span <= q_high:
            by_end[raw["end"]] = _to_fact(field_name, raw, concept)

    # Pass 2 — reconstruct whatever pass 1 could not supply.
    for _year_start, group in sorted(by_start.items()):
        indexed = []
        for raw in group:
            index = _ytd_quarter_index(raw)
            if index is not None:
                indexed.append((index, raw))
        indexed.sort(key=lambda pair: pair[0])
        for position, (index, raw) in enumerate(indexed):
            if raw["end"] in by_end:
                continue  # already covered by a directly reported quarter
            if index == 1:
                # A Q1 year-to-date fact IS the discrete first quarter.
                by_end[raw["end"]] = _to_fact(field_name, raw, concept)
                continue
            prior = indexed[position - 1] if position > 0 else None
            if prior is None or prior[0] != index - 1:
                # Only warn about periods recent enough to matter to a
                # current valuation; a broken chain in 2009 is not a finding.
                if _is_recent(raw.get("end")):
                    series.warnings.append(
                        f"{field_name}: cannot reconstruct the quarter ending {raw['end']} - the "
                        f"preceding year-to-date figure (through fiscal quarter {index - 1}) is "
                        "missing, so differencing would span more than one quarter.")
                continue
            prior_raw = prior[1]
            reconstructed = dict(raw)
            reconstructed["val"] = float(raw["val"]) - float(prior_raw["val"])
            reconstructed["start"] = prior_raw["end"]
            by_end[raw["end"]] = _to_fact(
                field_name, reconstructed, concept,
                reconstructed_from=(
                    f"{concept} {raw['start']}..{raw['end']} ({float(raw['val']):,.0f})",
                    f"{concept} {prior_raw['start']}..{prior_raw['end']} "
                    f"({float(prior_raw['val']):,.0f})",
                ))
            series.used_reconstruction = True

    # Pass 3 — the fourth quarter, from annual minus nine-month year-to-date.
    a_low, a_high = ANNUAL_DAYS
    annual_by_start: Dict[str, dict] = {}
    for raw in deduped:
        span = _span_days(raw["start"], raw["end"])
        if span is not None and a_low <= span <= a_high:
            existing = annual_by_start.get(raw["start"])
            if existing is None or (raw.get("filed") or "") > (existing.get("filed") or ""):
                annual_by_start[raw["start"]] = raw
    for year_start, annual_raw in sorted(annual_by_start.items()):
        if annual_raw["end"] in by_end:
            continue
        nine_month = None
        for raw in by_start.get(year_start, []):
            if _ytd_quarter_index(raw) == 3:
                nine_month = raw
                break
        if nine_month is None:
            continue
        reconstructed = dict(annual_raw)
        reconstructed["val"] = float(annual_raw["val"]) - float(nine_month["val"])
        reconstructed["start"] = nine_month["end"]
        by_end[annual_raw["end"]] = _to_fact(
            field_name, reconstructed, concept,
            reconstructed_from=(
                f"{concept} {annual_raw['start']}..{annual_raw['end']} "
                f"({float(annual_raw['val']):,.0f}, full year)",
                f"{concept} {nine_month['start']}..{nine_month['end']} "
                f"({float(nine_month['val']):,.0f}, nine months)",
            ))
        series.used_reconstruction = True

    series.quarters = sorted(by_end.values(), key=lambda f: f.end or "")[-max_quarters:]
    return series
