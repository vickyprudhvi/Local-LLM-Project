"""Central reporting-currency / financial-series resolver.

THE FAILURE CLASS THIS EXISTS TO CLOSE: REPORTING_CURRENCY_SERIES_SELECTION.

A live foreign issuer showed both V1 and V2 anchoring a "current" financial
state to a series more than a decade stale. The mechanism: `finance/
xbrl_mapping.py::_candidate_facts` deliberately reads only a `USD` (or
`USD/shares`, `shares`) unit -- there is no FX conversion anywhere in this
project, so a non-USD series is treated as absent rather than guessed into a
USD equity bridge (see that module's docstring). That refusal is correct in
isolation. The gap is what happens ABOVE it: `finance/taxonomy.py::
reporting_currency_note` only fires when NO USD fact exists for the wanted
concepts at ANY point in the issuer's history. An issuer that reported in USD
for years and then switched its primary statements to another currency has
plenty of historical USD facts -- `usable` comes back True, no note fires --
and every period-selection routine downstream (which searches company_facts
for "the newest fact it can find") walks straight past the switch and
happily calls the pre-switch USD data current, because that is the newest
data it is able to see at all.

THE RULE (spec section 6): current-period compatibility and current
reporting currency outrank recency, and recency outranks continuity/history
length. A shorter, newer, incompatible-currency series is not "worse" than a
longer historical one just because this project cannot read it -- it means
this project cannot currently answer "what is this company's current
state?" at all, and must SAY that rather than answering with a stale one.

WHAT THIS MODULE DOES AND DOES NOT DO.

`resolve_reporting_series` answers exactly one question: which currency do
this issuer's PRIMARY FINANCIAL STATEMENTS use as of the newest period they
actually cover, and is that the currency (`USD`) this project can read? It:

  * never infers currency from listing venue, ticker, share price or country
    (section 4) -- currency comes only from the unit attached to the actual
    filed statement-line facts;
  * never performs FX conversion (section 8) -- a non-USD current series is
    reported as UNRESOLVED, never converted or estimated;
  * never prefers a series for having more observations or longer history
    (section 6) -- selection is decided by the MOST RECENT usable statement
    fact, in whichever currency that turns out to be;
  * is issuer-agnostic -- nothing here keys off a ticker, CIK or company
    name; the concept lists it scans are the same reviewed, taxonomy-wide
    maps `finance/xbrl_mapping.py::CONCEPT_MAP` and `finance/taxonomy.py::
    IFRS_CONCEPT_MAP` every other reader already uses.

Callers gate on `ReportingSeriesResolution.selection_status` rather than on
`selected_reporting_currency` alone, because "USD" being unresolved (no
statement facts at all) and "not USD" (a real switch) both mean the same
thing to a consumer: this project has nothing safe to call current.
`ReportingSeriesStatus.BLOCKS_CURRENT_STATE` names the two statuses under
which a previously-readable (USD) series exists but must NOT be published as
current -- see `finance/freshness.py::build_current_financial_state`, the
single call site that wires this in.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

from finance import taxonomy as taxonomy_module

# The only currency finance/xbrl_mapping.py::_candidate_facts reads into the
# numeric path. Named once, here, so the constant is not duplicated between
# modules that must agree on it.
READABLE_CURRENCY = "USD"

# A duration fact outside this window is not a quarter or a year of this
# issuer's ordinary statements -- it is far more likely a multi-year
# cumulative or an unrelated disclosure that happens to share a concept name.
# Generous on purpose (see finance/xbrl_mapping.py's own 300-400/45-130 day
# windows); this module only needs "plausibly one statement period", not an
# exact classification of which one.
_MAX_DURATION_DAYS = 400


class ReportingSeriesStatus:
    RESOLVED = "RESOLVED"
    UNRESOLVED_NO_TAXONOMY = "UNRESOLVED_NO_TAXONOMY"
    UNRESOLVED_NO_STATEMENT_FACTS = "UNRESOLVED_NO_STATEMENT_FACTS"
    UNRESOLVED_FOREIGN_CURRENCY_ONLY = "UNRESOLVED_FOREIGN_CURRENCY_ONLY"
    UNRESOLVED_CURRENCY_SWITCH = "UNRESOLVED_CURRENCY_SWITCH"
    CONFLICT_SAME_PERIOD_MULTIPLE_CURRENCIES = "CONFLICT_SAME_PERIOD_MULTIPLE_CURRENCIES"

    ALL = (RESOLVED, UNRESOLVED_NO_TAXONOMY, UNRESOLVED_NO_STATEMENT_FACTS,
           UNRESOLVED_FOREIGN_CURRENCY_ONLY, UNRESOLVED_CURRENCY_SWITCH,
           CONFLICT_SAME_PERIOD_MULTIPLE_CURRENCIES)

    # A READABLE (USD) series exists but a newer, incompatible-currency
    # series has superseded it, or the current period is genuinely
    # ambiguous. In both cases the readable series must not be published as
    # the issuer's current financial state -- see the module docstring.
    BLOCKS_CURRENT_STATE = (UNRESOLVED_CURRENCY_SWITCH,
                            CONFLICT_SAME_PERIOD_MULTIPLE_CURRENCIES)


def _statement_forms() -> Tuple[str, ...]:
    return tuple(taxonomy_module.ANNUAL_FORMS) + tuple(taxonomy_module.INTERIM_FORMS)


def _unit_currency(unit_name: str) -> Optional[str]:
    """The ISO-ish currency a companyfacts unit token represents, or None for
    a unit this resolver has no opinion about (e.g. `shares`)."""
    if unit_name in ("USD", "USD/shares"):
        return READABLE_CURRENCY
    if unit_name == "shares":
        return None
    if len(unit_name) == 3 and unit_name.isalpha():
        return unit_name.upper()
    return None


def _duration_days(start: Optional[str], end: Optional[str]) -> Optional[int]:
    if not start or not end:
        return None
    try:
        return (date.fromisoformat(end) - date.fromisoformat(start)).days
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class SeriesObservation:
    """One currency's newest usable statement evidence, for provenance."""

    currency: str
    taxonomy: str
    latest_period_end: str
    concept: str
    form: Optional[str]
    filed: Optional[str]
    accession: Optional[str]
    statement_line_count: int

    def to_dict(self) -> dict:
        return {
            "currency": self.currency, "taxonomy": self.taxonomy,
            "latest_period_end": self.latest_period_end, "concept": self.concept,
            "form": self.form, "filed": self.filed, "accession": self.accession,
            "statement_line_count": self.statement_line_count,
        }


@dataclass(frozen=True)
class RejectedSeries:
    currency: str
    reason_code: str
    reason: str
    latest_period_end: Optional[str] = None

    def to_dict(self) -> dict:
        return {"currency": self.currency, "reason_code": self.reason_code,
                "reason": self.reason, "latest_period_end": self.latest_period_end}


@dataclass(frozen=True)
class ReportingSeriesResolution:
    selected_reporting_currency: Optional[str]
    selected_series_id: Optional[str]
    selected_period: Optional[str]
    selection_status: str
    currency_source: str
    series_source: Optional[str]
    candidate_series: Tuple[SeriesObservation, ...] = ()
    rejected_series: Tuple[RejectedSeries, ...] = ()
    rejection_codes: Tuple[str, ...] = ()
    resolution_reason: str = ""

    @property
    def resolved(self) -> bool:
        return self.selection_status == ReportingSeriesStatus.RESOLVED

    def to_dict(self) -> dict:
        return {
            "selected_reporting_currency": self.selected_reporting_currency,
            "selected_series_id": self.selected_series_id,
            "selected_period": self.selected_period,
            "selection_status": self.selection_status,
            "currency_source": self.currency_source,
            "series_source": self.series_source,
            "candidate_series": [c.to_dict() for c in self.candidate_series],
            "rejected_series": [r.to_dict() for r in self.rejected_series],
            "rejection_codes": list(self.rejection_codes),
            "resolution_reason": self.resolution_reason,
        }


def _unresolved(status: str, reason: str,
                currency_source: str = "none") -> ReportingSeriesResolution:
    return ReportingSeriesResolution(
        selected_reporting_currency=None, selected_series_id=None, selected_period=None,
        selection_status=status, currency_source=currency_source, series_source=None,
        resolution_reason=reason)


def resolve_reporting_series(company_facts: dict) -> ReportingSeriesResolution:
    """Which currency represents this issuer's CURRENT primary statements.

    Pure function of one companyfacts payload -- no filing_context/issuer_
    metadata argument is needed today because every signal this resolver
    uses (which concepts are tagged, in which unit, dated when, on which
    form) already lives in that one payload; a future caller with
    additional filing-level metadata may extend this without changing the
    contract for existing callers.
    """
    taxonomy = taxonomy_module.detect_taxonomy(company_facts)
    if not taxonomy:
        return _unresolved(
            ReportingSeriesStatus.UNRESOLVED_NO_TAXONOMY,
            "No supported reporting framework (us-gaap or ifrs-full) was found in this "
            "issuer's filed facts.")

    concept_map = taxonomy_module.concept_map_for(taxonomy)
    # concept -> is_instant, built once from the SAME reviewed maps every
    # other reader uses (never a locally-invented list).
    concept_is_instant: Dict[str, bool] = {}
    for is_instant, concepts in concept_map.values():
        for concept in concepts:
            concept_is_instant.setdefault(concept, is_instant)

    facts_for_taxonomy = ((company_facts or {}).get("facts") or {}).get(taxonomy) or {}
    forms = _statement_forms()

    # currency -> {end_date -> {concepts tagged with that end date}}
    lines_by_currency: Dict[str, Dict[str, set]] = {}
    # currency -> (end_date, concept, form, filed, accession) of its newest row
    newest_by_currency: Dict[str, tuple] = {}

    for concept, is_instant in concept_is_instant.items():
        entry = facts_for_taxonomy.get(concept)
        if not entry:
            continue
        for unit_name, rows in (entry.get("units") or {}).items():
            currency = _unit_currency(unit_name)
            if currency is None or not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict) or row.get("form") not in forms:
                    continue
                start, end = row.get("start"), row.get("end")
                if not end:
                    continue
                if is_instant:
                    if start:
                        continue
                else:
                    if not start:
                        continue
                    days = _duration_days(start, end)
                    if days is None or not (1 <= days <= _MAX_DURATION_DAYS):
                        continue
                lines_by_currency.setdefault(currency, {}).setdefault(end, set()).add(concept)
                current_newest = newest_by_currency.get(currency)
                if current_newest is None or end > current_newest[0]:
                    newest_by_currency[currency] = (
                        end, concept, row.get("form"), row.get("filed"), row.get("accn"))

    if not newest_by_currency:
        return _unresolved(
            ReportingSeriesStatus.UNRESOLVED_NO_STATEMENT_FACTS,
            f"No usable statement-line facts (in any currency) were found under the "
            f"{taxonomy} taxonomy for the concepts this project reads.")

    def line_count(currency: str, end: str) -> int:
        return len(lines_by_currency.get(currency, {}).get(end, ()))

    candidate_series = tuple(
        SeriesObservation(
            currency=currency, taxonomy=taxonomy, latest_period_end=newest[0],
            concept=newest[1], form=newest[2], filed=newest[3], accession=newest[4],
            statement_line_count=line_count(currency, newest[0]))
        for currency, newest in sorted(newest_by_currency.items())
    )

    global_latest = max(newest[0] for newest in newest_by_currency.values())
    tied = sorted(c for c, newest in newest_by_currency.items() if newest[0] == global_latest)

    rejected: List[RejectedSeries] = []
    current_currency: Optional[str]

    if len(tied) > 1:
        # Section 11/17 -- same current period, more than one currency. The
        # currency backing the most statement LINES at that date is the
        # primary financial statements; a convenience translation or a
        # segment/subsidiary disclosure covers only a handful of lines, not
        # the whole statement set.
        counts = {c: line_count(c, global_latest) for c in tied}
        top = max(counts.values())
        winners = sorted(c for c, n in counts.items() if n == top)
        if len(winners) > 1:
            for c in tied:
                rejected.append(RejectedSeries(
                    currency=c, reason_code="SAME_PERIOD_AMBIGUOUS_CURRENCY",
                    reason=(f"{c} backs the same number of primary statement lines "
                            f"({top}) at {global_latest} as at least one other currency; "
                            "which is the issuer's primary reporting currency cannot be "
                            "determined deterministically."),
                    latest_period_end=global_latest))
            return ReportingSeriesResolution(
                selected_reporting_currency=None, selected_series_id=None,
                selected_period=None,
                selection_status=ReportingSeriesStatus.CONFLICT_SAME_PERIOD_MULTIPLE_CURRENCIES,
                currency_source="same_period_multiple_currencies_tied",
                series_source=taxonomy, candidate_series=candidate_series,
                rejected_series=tuple(rejected),
                rejection_codes=("SAME_PERIOD_AMBIGUOUS_CURRENCY",),
                resolution_reason=(
                    f"Multiple currencies ({', '.join(tied)}) report the same number of "
                    f"primary statement lines at {global_latest}; this project will not "
                    "guess which is canonical, so no current currency is selected."))
        current_currency = winners[0]
        for c in tied:
            if c == current_currency:
                continue
            rejected.append(RejectedSeries(
                currency=c, reason_code="CONVENIENCE_TRANSLATION_OR_SEGMENT",
                reason=(f"{c} backs only {counts[c]} statement line(s) at {global_latest} "
                        f"against {top} in {current_currency}, consistent with a "
                        "convenience translation or a segment/subsidiary disclosure rather "
                        "than the primary financial statements."),
                latest_period_end=global_latest))
    else:
        current_currency = tied[0]

    for currency, newest in newest_by_currency.items():
        if currency == current_currency or any(r.currency == currency for r in rejected):
            continue
        rejected.append(RejectedSeries(
            currency=currency, reason_code="HISTORICAL_SUPERSEDED",
            reason=(f"{currency}'s newest usable statement fact is dated {newest[0]}, before "
                    f"the current reporting currency's newest fact on {global_latest}."),
            latest_period_end=newest[0]))

    if current_currency == READABLE_CURRENCY:
        return ReportingSeriesResolution(
            selected_reporting_currency=READABLE_CURRENCY,
            selected_series_id=f"{taxonomy}:{READABLE_CURRENCY}",
            selected_period=global_latest,
            selection_status=ReportingSeriesStatus.RESOLVED,
            currency_source="current_period_statement_units",
            series_source=taxonomy,
            candidate_series=candidate_series,
            rejected_series=tuple(rejected),
            rejection_codes=tuple(sorted({r.reason_code for r in rejected})),
            resolution_reason=(
                f"The newest primary statement facts ({global_latest}) are reported in "
                f"{READABLE_CURRENCY}, this project's readable currency."))

    # The current reporting currency is not one this project can read
    # numerically -- there is no FX conversion here (section 8).
    usd_newest = newest_by_currency.get(READABLE_CURRENCY)
    if usd_newest is None:
        status = ReportingSeriesStatus.UNRESOLVED_FOREIGN_CURRENCY_ONLY
        reason = (
            f"This issuer's primary statements are reported in {current_currency}. This "
            f"project has no currency conversion, and no {READABLE_CURRENCY} series exists "
            "for it at any period, so its current financial state cannot be read "
            "numerically.")
    else:
        status = ReportingSeriesStatus.UNRESOLVED_CURRENCY_SWITCH
        rejected = [r for r in rejected if r.currency != READABLE_CURRENCY]
        if usd_newest[0] == global_latest:
            # Section 11 -- the USD facts are not older, they are a minority
            # convenience translation AT the current period (fewer statement
            # lines than the primary currency): still not a currency this
            # project may treat as the primary current series.
            reason = (
                f"This issuer's newest primary statement facts ({global_latest}) are "
                f"reported in {current_currency}. A {READABLE_CURRENCY} series exists at the "
                f"same date but backs fewer statement lines ({line_count(READABLE_CURRENCY, global_latest)} "
                f"vs {line_count(current_currency, global_latest)}), consistent with a "
                "convenience translation rather than the primary financial statements. This "
                "project has no currency conversion, so that translation must not be "
                "published as this issuer's current financial state.")
            rejected.append(RejectedSeries(
                currency=READABLE_CURRENCY, reason_code="CONVENIENCE_TRANSLATION_OR_SEGMENT",
                reason=reason, latest_period_end=usd_newest[0]))
        else:
            gap_days = _duration_days(usd_newest[0], global_latest)
            reason = (
                f"This issuer's newest primary statement facts ({global_latest}) are "
                f"reported in {current_currency}, not {READABLE_CURRENCY}. The newest "
                f"{READABLE_CURRENCY} series available ends {usd_newest[0]}"
                + (f", {gap_days} days earlier" if gap_days is not None else "")
                + f". This project has no currency conversion, so the {READABLE_CURRENCY} "
                "series must not be published as this issuer's current financial state.")
            rejected.append(RejectedSeries(
                currency=READABLE_CURRENCY, reason_code="STALE_CURRENCY_SERIES",
                reason=reason, latest_period_end=usd_newest[0]))

    return ReportingSeriesResolution(
        selected_reporting_currency=current_currency,
        selected_series_id=None,
        selected_period=None,
        selection_status=status,
        currency_source="current_period_statement_units",
        series_source=(taxonomy if status != ReportingSeriesStatus.UNRESOLVED_FOREIGN_CURRENCY_ONLY
                      else None),
        candidate_series=candidate_series,
        rejected_series=tuple(rejected),
        rejection_codes=tuple(sorted({r.reason_code for r in rejected})),
        resolution_reason=reason)
