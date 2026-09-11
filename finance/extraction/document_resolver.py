"""Which document, and which period, is CURRENT. Deterministic throughout.

THE FAILURE THIS EXISTS FOR

"Latest reported financial period" was answered by "latest periodic filing".
Those are different questions, and they diverge for one or two months after
every fiscal year end: the company has announced complete audited-in-all-but-
name results in an 8-K, the 10-K is weeks away, and the analysis is still
reporting the state as of the previous 10-Q.

The fix is not to prefer newer documents. §13 is explicit that an incomplete
press release must not displace a complete filing, and it is right: a
headline-only release naming revenue and EPS cannot supply a balance sheet,
and letting it advance the state would produce a "current" period whose debt
figure silently came from somewhere else.

So the resolution reasons over structured metadata and says why:

    period end          the question is about a PERIOD, so this ranks first
    completeness        a candidate must be able to answer what it claims
    authority           a filing outranks a release ON A TIE
    issue date          the newest statement of a period wins

Every decision is returned with its reason. A resolver that picks correctly
and cannot say why is one debugging session away from being reverted.
"""

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from finance import period_facts as pf
from finance import taxonomy as taxonomy_module
from finance.extraction.schema import (
    FinalityStatus,
    ReportedActualCandidate,
    SourceType,
    StatementCompleteness,
)

# The metrics a source must carry before it may claim to describe a whole
# reported period. One from each statement, because "complete" means the
# three statements are present, not that some large number of tags is.
#
# Deliberately small: requiring more would reject a genuinely complete
# release over a line this project does not read, and requiring less would
# let a headline release claim a balance sheet it never published.
#
# TOTAL LIABILITIES IS NOT ONE OF THE ANCHORS, and used to be. A classified
# balance sheet may present current and non-current liabilities and total
# straight to `LiabilitiesAndStockholdersEquity`, tagging `Liabilities` never.
# That is ordinary presentation, not an incomplete filing -- but it made every
# document such an issuer had ever filed PARTIAL, so no period resolved for it
# at all and the layer went dark on the whole company while its 10-K sat on
# file. `assets` + `stockholders_equity` is the pair `finance/period_facts.py`
# already uses to decide a balance sheet was filed (`_BALANCE_SHEET_ANCHORS`),
# so both callers now answer "is this a balance sheet?" the same way. Two
# definitions of one idea is how a correct filing gets refused.
REQUIRED_FOR_COMPLETE = {
    "income_statement": ("revenue",),
    "balance_sheet": ("assets", "stockholders_equity",
                      "cash_and_cash_equivalents"),
    "cash_flow": ("operating_cash_flow",),
}

# Metrics whose absence makes a source PARTIAL rather than HEADLINE_ONLY.
_SUBSTANTIVE = ("revenue", "net_income", "operating_income", "assets",
                "cash_and_cash_equivalents", "operating_cash_flow",
                "capital_expenditure", "stockholders_equity")

_EARNINGS_RELEASE_FORMS = frozenset(taxonomy_module.EARNINGS_MATERIAL_FORMS)
_PERIODIC_FORMS = frozenset(taxonomy_module.ALL_REPORT_FORMS)


def source_type_for_form(form: Optional[str]) -> str:
    """A form name to the kind of authority it carries."""
    name = (form or "").upper()
    if name in {f.upper() for f in _PERIODIC_FORMS}:
        return SourceType.FORMAL_PERIODIC_FILING
    if name in {f.upper() for f in _EARNINGS_RELEASE_FORMS}:
        return SourceType.PRELIMINARY_EARNINGS_RELEASE
    return SourceType.OTHER


def finality_for_form(form: Optional[str]) -> str:
    name = (form or "").upper()
    if name in ("10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"):
        return FinalityStatus.AUDITED
    if name in {f.upper() for f in _EARNINGS_RELEASE_FORMS}:
        return FinalityStatus.UNAUDITED_PRELIMINARY
    return FinalityStatus.UNKNOWN


def classify_completeness(present: set) -> Tuple[str, Tuple[str, ...]]:
    """(completeness, missing required metrics) for a set of present metrics."""
    missing = []
    for _statement, required in sorted(REQUIRED_FOR_COMPLETE.items()):
        missing += [name for name in required if name not in present]
    if not missing:
        return StatementCompleteness.COMPLETE, ()
    substantive = [name for name in _SUBSTANTIVE if name in present]
    if len(substantive) >= 3:
        return StatementCompleteness.PARTIAL, tuple(missing)
    if substantive:
        return StatementCompleteness.HEADLINE_ONLY, tuple(missing)
    return StatementCompleteness.NONE, tuple(missing)


# ---------------------------------------------------------------------------
# Candidate discovery from XBRL facts
# ---------------------------------------------------------------------------

def _fact_rows(company_facts: dict, field_name: str) -> List[dict]:
    """Every raw fact for one mapped field, with its provenance intact.

    The mapping is THIS ISSUER'S framework's, via `period_facts._concept_map`,
    which is the function that exists because reading the module-level us-gaap
    map directly made a foreign private issuer's entire history unreadable.
    This resolver was written later and had reintroduced exactly that: it
    found `Assets` -- spelled the same under both frameworks -- and nothing
    else, so every filing an ifrs-full issuer had made was HEADLINE_ONLY and
    no period resolved for the company at all.
    """
    from finance.xbrl_mapping import _candidate_facts

    concept_map = pf._concept_map(company_facts)
    if field_name not in concept_map:
        return []
    _is_instant, concepts = concept_map[field_name]
    rows: List[dict] = []
    for concept in concepts:
        rows.extend(_candidate_facts(company_facts, concept))
    return rows


def discover_reported_actuals(company_facts: dict,
                              fields: Optional[Tuple[str, ...]] = None
                              ) -> List[ReportedActualCandidate]:
    """Every (period end, source) pair that reported actual results.

    Grouped by (period end, accession) rather than by (fiscal year, fiscal
    period): the fiscal tags describe the FILING a fact appeared in, not the
    period it measures, which is the whole reason `finance/period_facts.py`
    exists. Two sources reporting the same period are two candidates and the
    precedence rule below decides between them.
    """
    if fields is None:
        required = [name for group in REQUIRED_FOR_COMPLETE.values() for name in group]
        fields = tuple(sorted(set(required) | set(_SUBSTANTIVE)))
    groups: Dict[Tuple[str, str], dict] = {}
    for field_name in fields:
        for raw in _fact_rows(company_facts, field_name):
            end = raw.get("end")
            if not end:
                continue
            key = (end, raw.get("accn") or "")
            entry = groups.setdefault(key, {
                "end": end, "accession": raw.get("accn") or "",
                "form": raw.get("form"), "filed": raw.get("filed"),
                "fy": raw.get("fy"), "fp": raw.get("fp"),
                "present": set(), "starts": set(), "by_duration": {},
            })
            entry["present"].add(field_name)
            if raw.get("start"):
                entry["starts"].add(raw["start"])
            # The REPORTED NUMBER, kept per duration. One filing states a
            # figure for the quarter and for the year to date under the same
            # period end, and folding them together would make the group's
            # revenue mean nothing -- which is exactly what a same-period
            # reconciliation would then be comparing.
            duration = pf.classify_duration(raw.get("start"), end)
            bucket = entry["by_duration"].setdefault(duration, {})
            if raw.get("val") is not None and field_name not in bucket:
                try:
                    bucket[field_name] = float(raw["val"])
                except (TypeError, ValueError):
                    pass
            # The newest filing date among the group's facts is the group's.
            if (raw.get("filed") or "") > (entry["filed"] or ""):
                entry["filed"] = raw.get("filed")

    candidates = []
    for entry in groups.values():
        completeness, missing = classify_completeness(entry["present"])
        form = entry["form"]
        period_type, values = _values_for(entry["by_duration"])
        candidates.append(ReportedActualCandidate(
            period_end=entry["end"],
            period_start=min(entry["starts"]) if entry["starts"] else None,
            period_type=period_type,
            fiscal_year=entry["fy"], fiscal_period=entry["fp"],
            issued_at=entry["filed"], form=form,
            accession=entry["accession"],
            source_type=source_type_for_form(form),
            statement_completeness=completeness,
            finality=finality_for_form(form),
            present_metrics=tuple(sorted(entry["present"])),
            missing_metrics=missing,
            values=values))
    return candidates


# Which flow duration a candidate's `values` are taken from, best first. A
# cumulative year-to-date column measures the same end date over a different
# span and is never the period's own figure; putting one in `values` is how a
# nine-month revenue gets compared against a three-month one and reported as a
# disagreement between two sources that in fact agree.
_PREFERRED_DURATIONS = (pf.DurationType.QUARTER, pf.DurationType.ANNUAL)


def _values_for(by_duration: Dict[str, Dict[str, float]]
                ) -> Tuple[Optional[str], Dict[str, float]]:
    """(period type, values) for one group of facts sharing a period end.

    THE POINT OF THIS FUNCTION, which had no equivalent before: a candidate
    that carries no numbers cannot be reconciled against another candidate for
    the same period. `detect_same_period_conflicts` was reachable only from a
    caller that filled `values` by hand, so in production it compared nothing
    and a preliminary release could disagree materially with the 10-K that
    followed it without a word being said.

    Balance-sheet instants belong to the period end whatever the flow duration
    is, so they are always included.
    """
    instants = dict(by_duration.get(pf.DurationType.INSTANT) or {})
    chosen = next((d for d in _PREFERRED_DURATIONS if by_duration.get(d)), None)
    values = dict(by_duration.get(chosen) or {}) if chosen else {}
    for name, value in instants.items():
        values.setdefault(name, value)
    period_type = chosen or (pf.DurationType.INSTANT if instants else None)
    return period_type, values


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------

@dataclass
class PeriodResolution:
    """Which period is current, which source said so, and why."""

    selected: Optional[ReportedActualCandidate] = None
    reason: str = ""
    considered: List[ReportedActualCandidate] = field(default_factory=list)
    rejected: List[Tuple[ReportedActualCandidate, str]] = field(default_factory=list)

    @property
    def period_end(self) -> Optional[str]:
        return self.selected.period_end if self.selected else None

    def to_dict(self) -> dict:
        return {
            "selected": self.selected.to_dict() if self.selected else None,
            "reason": self.reason,
            "considered": [c.to_dict() for c in self.considered],
            "rejected": [{"candidate": c.to_dict(), "reason": why}
                         for c, why in self.rejected],
        }


# Completeness levels a candidate may have and still advance the whole state.
_ADVANCING_COMPLETENESS = (StatementCompleteness.COMPLETE,)


def resolve_current_period(candidates: List[ReportedActualCandidate],
                           as_of: Optional[str] = None) -> PeriodResolution:
    """The newest COMPLETE reported period, with the reason it won.

    Ordering, in the order the questions matter:

      1. period end, descending -- the question is about a period
      2. authority, ascending -- a filing outranks a release on a tie
      3. issue date, descending -- the newest statement of that period

    A candidate that is not COMPLETE is recorded as rejected with its
    missing metrics rather than dropped, because "an 8-K arrived but could
    not carry the state" is the answer to a question a reader will ask.
    """
    as_of = as_of or datetime.date.today().isoformat()
    usable = [c for c in candidates if c.period_end and c.period_end <= as_of]
    resolution = PeriodResolution(considered=list(usable))
    if not usable:
        resolution.reason = "No source reported a completed period on or before this date."
        return resolution

    complete, incomplete = [], []
    for candidate in usable:
        (complete if candidate.statement_completeness in _ADVANCING_COMPLETENESS
         else incomplete).append(candidate)

    newest_any = max(c.period_end for c in usable)
    for candidate in incomplete:
        if candidate.period_end == newest_any:
            resolution.rejected.append((
                candidate,
                f"{candidate.form or 'this source'} reports the newest period "
                f"({candidate.period_end}) but is {candidate.statement_completeness}: "
                f"missing {', '.join(candidate.missing_metrics) or 'required statements'}. "
                "A source that cannot describe the whole period does not advance the "
                "state; its individual figures remain available per metric."))

    if not complete:
        resolution.reason = (
            "No source carried a complete statement set, so the current period is "
            "unchanged.")
        return resolution

    complete.sort(key=lambda c: (c.period_end or "", -c.authority, c.issued_at or ""),
                  reverse=True)
    selected = complete[0]
    resolution.selected = selected

    beaten = [c for c in usable
              if c is not selected and (c.period_end or "") > (selected.period_end or "")]
    if beaten:
        resolution.reason = (
            f"{selected.form or 'the selected source'} reports a complete statement set "
            f"for {selected.period_end}; the newer {beaten[0].period_end} source is "
            f"{beaten[0].statement_completeness} and cannot carry the state.")
    elif selected.source_type == SourceType.PRELIMINARY_EARNINGS_RELEASE:
        resolution.reason = (
            f"A {selected.form} earnings release dated {selected.issued_at} carries a "
            f"complete statement set for {selected.period_end}, newer than any periodic "
            "filing. The figures are preliminary and unaudited until the periodic "
            "filing arrives, which is recorded rather than assumed away.")
    else:
        resolution.reason = (
            f"{selected.form or 'A periodic filing'} is the newest complete statement "
            f"set, covering {selected.period_end}.")
    return resolution


def resolve(company_facts: dict, as_of: Optional[str] = None) -> PeriodResolution:
    """Discovery and precedence in one call."""
    return resolve_current_period(discover_reported_actuals(company_facts), as_of=as_of)


# ---------------------------------------------------------------------------
# Actualization: a period whose results are in is no longer a forecast
# ---------------------------------------------------------------------------

def reported_period_ends(candidates: List[ReportedActualCandidate]) -> List[str]:
    """Period ends for which ACTUAL results exist, newest first."""
    return sorted({c.period_end for c in candidates if c.period_end}, reverse=True)


def period_has_actuals(target_period: Optional[str],
                       target_period_end: Optional[str],
                       candidates: List[ReportedActualCandidate]) -> bool:
    """Have actual results been reported for this guided period?

    §11: once they have, the guidance for it is realized -- it describes
    something that already happened and is no longer an outlook. Keyed on the
    period END rather than on the label, because the label is the issuer's
    vocabulary and the date is the fact.
    """
    if not target_period_end:
        return False
    for candidate in candidates:
        if not candidate.period_end:
            continue
        # A period is reported when a source covers its end date. Exact
        # equality: a source covering a LATER end has not necessarily
        # reported this one (a fiscal year end is not its own Q3).
        if candidate.period_end == target_period_end \
                and candidate.statement_completeness in (
                    StatementCompleteness.COMPLETE, StatementCompleteness.PARTIAL):
            return True
    return False


def latest_reported_period_end(company_facts: dict) -> Optional[str]:
    """The newest period any source reported actuals for.

    Kept beside the resolver because callers asking "is my base stale?" want
    this rather than the resolved period: a base is stale when ANYTHING newer
    has been reported, complete or not.
    """
    ends = [c.period_end for c in discover_reported_actuals(company_facts)
            if c.period_end]
    return max(ends) if ends else None
