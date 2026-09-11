"""Facts to a `ReportedActualCandidate` the Actualization resolver can rank.

WHAT THIS LAYER DECIDES, AND WHAT IT MUST NOT

It decides which facts describe ONE period coherently. It does not decide
which period is current, which source is authoritative, or which of two
disagreeing figures is right. Those are the Actualization resolver's, and a
source layer that answered them would be a second resolver with a different
name -- the failure the whole architecture is organised against.

ONE CANDIDATE PER PERIOD, ONE DURATION INSIDE IT

A Q3 earnings release states revenue twice for the same period end: once for
the quarter and once for the nine months. Both are true, and putting both in
one candidate's `values` would make its revenue mean nothing. So a candidate
is built for ONE flow duration, and the others are recorded as facts that
exist without being part of the state.

The duration preferred is the QUARTER, then the fiscal YEAR. The quarter is
what advances the reported state and what a trailing-twelve-month window rolls
forward on; the year-to-date columns are a different measurement of the same
end date and never a substitute for it.

`period_type` on the candidate says which duration was used, so two candidates
for one period end can be compared only when they are measuring the same thing.

VALUES, WHICH IS THE POINT

`ReportedActualCandidate.values` is what makes same-period reconciliation
possible: a preliminary release and the 10-K that follows it describe one
economic period, and whether they AGREE is a question about numbers. Until
this layer, `values` was empty on every candidate production ever built, so
`detect_same_period_conflicts` compared nothing and reported nothing.

COMPLETENESS IS NOT REDEFINED HERE

`document_resolver.classify_completeness` already answers "can this source
describe a whole period?", and the Actualization benchmark measures it. This
module calls it. A press release naming revenue and earnings per share comes
out HEADLINE_ONLY exactly as it should, and the resolver then declines to let
it replace a complete filing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from finance import semantics as sem
from finance.extraction.schema import (
    FinalityStatus,
    ReportedActualCandidate,
    SourceType,
    StatementCompleteness,
)
from finance.reported_actuals.facts import (
    ActualFinancialFact,
    FactRejection,
    facts_from_table,
)
from finance.reported_actuals.tables import (
    StatementKind,
    document_scale,
    parse_filing_tables,
)


class CandidateRejection:
    NO_TABLES = "NO_TABLES"
    NO_FACTS = "NO_FACTS"
    MIXED_CURRENCY = "MIXED_CURRENCY"
    NON_PREFERRED_DURATION = "NON_PREFERRED_DURATION"
    ADJUSTED_ONLY = "ADJUSTED_ONLY"
    VALUE_DISAGREEMENT_WITHIN_DOCUMENT = "VALUE_DISAGREEMENT_WITHIN_DOCUMENT"


# The flow durations a candidate may be built on, best first.
#
# QUARTER and ANNUAL are reporting periods in their own right. YTD_9M is not,
# and is deliberately absent: a nine-month column exists in a release only
# BESIDE the quarter it belongs to, so taking it would mean reading three
# months of results off a nine-month figure -- the masquerade section 8 names,
# in the one place it could actually happen.
#
# YTD_6M is the exception, and only as a last resort. A semi-annual reporter's
# half year IS its reported period; refusing it would leave every foreign
# private issuer that files interim results on a 6-K with no flow facts at
# all. It ranks below both real reporting periods, so a release that states
# its quarter is always read as the quarter, and the candidate records
# `period_type=YTD_6M` so nothing downstream can mistake it for one.
PREFERRED_FLOW_FREQUENCIES = (sem.PeriodFrequency.QUARTER,
                              sem.PeriodFrequency.ANNUAL,
                              sem.PeriodFrequency.YTD_6M)

# How far two statements of one figure inside ONE document may differ before
# the document is contradicting itself. Matches
# `finance/actualization.py::MATERIAL_DIFFERENCE`, imported rather than
# restated so one definition of "materially different" serves both.
from finance.actualization import MATERIAL_DIFFERENCE  # noqa: E402


@dataclass
class ReportedActualsExtraction:
    """Everything one document produced, including what it refused."""

    candidates: List[ReportedActualCandidate] = field(default_factory=list)
    facts: List[ActualFinancialFact] = field(default_factory=list)
    tables: List[dict] = field(default_factory=list)
    rejections: List[Tuple[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    accession: Optional[str] = None
    form: Optional[str] = None
    filed: Optional[str] = None
    document: Optional[str] = None

    @property
    def canonical_facts(self) -> List[ActualFinancialFact]:
        return [f for f in self.facts if f.is_canonical]

    def rejection_codes(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for code, _detail in self.rejections:
            counts[code] = counts.get(code, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "accession": self.accession, "form": self.form,
            "filed": self.filed, "document": self.document,
            "candidates": [c.to_dict() for c in self.candidates],
            "candidate_count": len(self.candidates),
            "fact_count": len(self.facts),
            "canonical_fact_count": len(self.canonical_facts),
            "tables": list(self.tables),
            "rejection_codes": self.rejection_codes(),
            "notes": list(self.notes),
        }


def _flow_frequency_for(period_end: str,
                        facts: Sequence[ActualFinancialFact]) -> Optional[str]:
    available = {f.frequency for f in facts
                 if f.period_end == period_end
                 and f.flow_or_instant == sem.FlowOrInstant.FLOW}
    for frequency in PREFERRED_FLOW_FREQUENCIES:
        if frequency in available:
            return frequency
    return None


def build_candidates(facts: Sequence[ActualFinancialFact], *,
                     accession: Optional[str] = None,
                     form: Optional[str] = None,
                     filed: Optional[str] = None,
                     entity_id: Optional[str] = None
                     ) -> Tuple[List[ReportedActualCandidate], List[Tuple[str, str]]]:
    """(candidates, rejections). One candidate per period end.

    Only CANONICAL facts -- reported GAAP/IFRS statement lines -- take part.
    An adjusted figure is a real number about a real period and is simply not
    the line whose name it shares.
    """
    from finance.extraction.document_resolver import (
        classify_completeness,
        finality_for_form,
        source_type_for_form,
    )

    rejections: List[Tuple[str, str]] = []
    canonical = [f for f in facts if f.is_canonical]
    if not canonical:
        rejections.append((CandidateRejection.NO_FACTS,
                           "no reported statement line was identified"))
        return [], rejections

    currencies = {f.currency for f in canonical}
    if len(currencies) > 1:
        rejections.append((CandidateRejection.MIXED_CURRENCY,
                           ", ".join(sorted(currencies))))
        return [], rejections
    currency = currencies.pop()

    candidates: List[ReportedActualCandidate] = []
    for period_end in sorted({f.period_end for f in canonical if f.period_end},
                             reverse=True):
        chosen = _flow_frequency_for(period_end, canonical)
        selected: List[ActualFinancialFact] = []
        for fact in canonical:
            if fact.period_end != period_end:
                continue
            if fact.flow_or_instant == sem.FlowOrInstant.INSTANT:
                selected.append(fact)
            elif fact.frequency == chosen:
                selected.append(fact)
            else:
                rejections.append((
                    CandidateRejection.NON_PREFERRED_DURATION,
                    f"{fact.field} {fact.frequency} at {period_end} is not the "
                    f"period's own {chosen or 'flow'} figure"))
        if not selected:
            continue

        values: Dict[str, float] = {}
        starts = set()
        for fact in selected:
            existing = values.get(fact.field)
            if existing is None:
                values[fact.field] = fact.value
                if fact.period_start:
                    starts.add(fact.period_start)
                continue
            # The same line stated twice in one document -- the income
            # statement's net income and the cash-flow statement's opening
            # line, for instance. They must agree; a document that
            # contradicts itself is not a source of one of the two numbers.
            scale = max(abs(existing), 1e-9)
            if abs(existing - fact.value) / scale > MATERIAL_DIFFERENCE:
                rejections.append((
                    CandidateRejection.VALUE_DISAGREEMENT_WITHIN_DOCUMENT,
                    f"{fact.field} at {period_end}: {existing} vs {fact.value}"))
                values.pop(fact.field, None)

        present = tuple(sorted(values))
        completeness, missing = classify_completeness(set(present))
        candidates.append(ReportedActualCandidate(
            period_end=period_end,
            period_start=min(starts) if starts else None,
            period_type=chosen or sem.PeriodFrequency.INSTANT,
            fiscal_year=None, fiscal_period=None,
            issued_at=filed, form=form, accession=accession,
            source_type=source_type_for_form(form),
            statement_completeness=completeness,
            finality=finality_for_form(form),
            present_metrics=present, missing_metrics=missing,
            currency=currency, entity_id=entity_id,
            values=dict(values), is_prospective=False))
    return candidates, rejections


def extract_reported_actuals(document_text: str, *,
                             accession: Optional[str] = None,
                             form: Optional[str] = None,
                             filed: Optional[str] = None,
                             document: Optional[str] = None,
                             entity_id: Optional[str] = None,
                             gaap_basis: str = sem.AccountingBasis.GAAP
                             ) -> ReportedActualsExtraction:
    """One filed document to reported-actual candidates. Deterministic throughout."""
    result = ReportedActualsExtraction(accession=accession, form=form,
                                       filed=filed, document=document)
    tables = parse_filing_tables(document_text,
                                 document_scale_hint=document_scale(document_text))
    if not tables:
        result.rejections.append((CandidateRejection.NO_TABLES,
                                  "the document contains no readable table"))
        return result

    finality = FinalityStatus.UNKNOWN
    try:
        from finance.extraction.document_resolver import finality_for_form
        finality = finality_for_form(form)
    except Exception:                                        # noqa: BLE001
        pass

    for table in tables:
        result.tables.append(table.to_dict())
        result.rejections.extend(table.rejections)
        facts, rejected = facts_from_table(
            table, accession=accession, form=form, filed=filed,
            document=document, finality=finality, gaap_basis=gaap_basis)
        result.facts.extend(facts)
        result.rejections.extend(rejected)

    candidates, rejected = build_candidates(
        result.facts, accession=accession, form=form, filed=filed,
        entity_id=entity_id)
    result.candidates = candidates
    result.rejections.extend(rejected)
    return result


# ---------------------------------------------------------------------------
# The overlay (section 19)
# ---------------------------------------------------------------------------
#
# A release's quarterly figures are not in CompanyFacts, so once the resolver
# selects the release's period the existing twelve-month windows still end one
# quarter earlier and are correctly marked LIMITED. That is safe and it is not
# the whole answer: the quarter WAS reported, and the window could follow it.
#
# `finance/ttm.py` already builds the window and already owns every invariant
# about it. What it needs is the facts. So this projects the accepted facts
# back into the SAME company-facts shape that module already reads, and the
# caller merges it -- no second calculator, no new arithmetic, and the real
# payload is never mutated.
#
# ONLY the facts that were accepted appear, each carrying the accession and
# form it came from, so a window built over one is as traceable as a window
# built over XBRL.

def company_facts_overlay(facts: Sequence[ActualFinancialFact],
                          base: Optional[dict] = None,
                          taxonomy: Optional[str] = None) -> dict:
    """Accepted release facts in the SEC company-facts shape.

    THE CONCEPT MUST BE THE ISSUER'S OWN. `period_facts` builds a series from
    ONE winning concept and picks the one reaching furthest forward, so an
    overlay row filed under a different spelling of the same line does not
    extend the issuer's series -- it becomes a new one-fact series that wins on
    recency and can build nothing. Measured: revenue went from a clean
    four-quarter window to no window at all, while the four fields whose
    spellings happened to coincide rolled forward correctly.

    So when the base payload already reports a field, the overlay joins THAT
    concept. The reviewed map's first candidate is used only for a field the
    issuer has never reported, where there is no series to join.
    """
    concepts: Dict[str, Dict[str, list]] = {}
    taxonomy = taxonomy or _base_taxonomy(base)
    resolved: Dict[str, Optional[str]] = {}
    already: Dict[str, set] = {}
    for fact in facts:
        if not fact.is_canonical or not fact.period_end:
            continue
        if fact.currency != "USD":
            # NON-USD FACTS NEVER ENTER THE NUMERIC PATH. `xbrl_mapping.
            # _candidate_facts` reads only `units["USD"]`, so a EUR figure
            # written under a USD unit key would be silently wrong by the
            # exchange rate -- roughly fifteen percent, and invisible to
            # every downstream check. The release's period is still resolved;
            # only its numbers stay out of the overlay. The reporting-
            # currency-switch problem is a separate, deferred matter -- this
            # is the narrow guard that keeps this integration from adding a
            # currency error.
            continue
        if fact.field not in resolved:
            resolved[fact.field] = _concept_for(fact.field, base)
            already[fact.field] = _period_ends_in(base, fact.field)
        concept = resolved[fact.field]
        if concept is None:
            continue
        if fact.period_end in already[fact.field]:
            # THE BASE ALREADY REPORTS THIS PERIOD, so the overlay has nothing
            # to add and a great deal to break. A release's fiscal quarter
            # boundaries are derived by month arithmetic here, not read from
            # the document, so an overlay row for a period the payload already
            # holds is a SECOND quarter with almost the same end date and a
            # slightly different start. `finance/ttm.py` then correctly refuses
            # to sum two overlapping quarters, and a twelve-month window that
            # was COMPLETE becomes LIMITED -- measured on two live issuers,
            # including one whose period had not moved at all.
            #
            # The periodic filing is the authority for a period it covers.
            # That is section 14's rule, and this is the same rule applied to
            # the facts rather than to the candidates.
            continue
        row = {"end": fact.period_end, "val": fact.value,
               "form": fact.form, "accn": fact.accession,
               "filed": fact.filed, "fy": None, "fp": None}
        if fact.flow_or_instant == sem.FlowOrInstant.FLOW:
            start = _overlay_start(fact, base, already[fact.field])
            if not start:
                continue
            row["start"] = start
        entry = concepts.setdefault(concept, {"units": {"USD": []}})
        entry["units"]["USD"].append(row)
    return {"facts": {taxonomy: concepts}} if concepts else {}


def merge_company_facts(base: Optional[dict], overlay: Optional[dict]) -> dict:
    """`base` with `overlay`'s rows appended. Neither input is modified.

    Rows are APPENDED rather than replacing anything. `period_facts` already
    dedupes by period and prefers the latest filing, so a release figure the
    periodic filing later restates loses to the filing on its own merits
    rather than by an ordering rule invented here.
    """
    import copy

    merged = copy.deepcopy(base or {})
    root = merged.setdefault("facts", {})
    for taxonomy, concepts in ((overlay or {}).get("facts") or {}).items():
        target = root.setdefault(taxonomy, {})
        for concept, entry in concepts.items():
            existing = target.setdefault(concept, {"units": {}})
            for unit, rows in (entry.get("units") or {}).items():
                existing.setdefault("units", {}).setdefault(unit, [])
                existing["units"][unit] = list(existing["units"][unit]) + list(rows)
    return merged


def _base_taxonomy(base: Optional[dict]) -> str:
    from finance import taxonomy as taxonomy_module
    return taxonomy_module.detect_taxonomy(base) or taxonomy_module.US_GAAP


def _overlay_start(fact: ActualFinancialFact, base: Optional[dict],
                   known_ends: set) -> Optional[str]:
    """When this flow period BEGAN, from the issuer's own filed boundaries.

    THE MONTH ARITHMETIC IS NOT GOOD ENOUGH HERE, and this is where it shows.
    A 52/53-week filer's third quarter ended 2 August 2026 and began 4 May
    2026, the day after the previous quarter ended. Counting three months back
    gives 1 June: a 62-day "quarter" leaving a 29-day hole after the last
    filed period. `finance/ttm.py` then refuses to sum across the gap -- quite
    correctly, since the periods it was given do not tile the year -- and a
    window that could have followed the release is marked LIMITED instead.

    Consecutive reporting periods abut. So the start is the day after the
    previous period end the ISSUER reported, when there is one at a plausible
    distance. Nothing is invented: both dates come from filings.
    """
    import datetime

    end = fact.period_end
    months = sem.duration_months(fact.frequency)
    if not end or not months:
        return None
    try:
        end_date = datetime.date.fromisoformat(end)
    except ValueError:
        return None

    # A period of `months` months is this many days, give or take. Wide enough
    # for a 4-4-5 calendar, never wide enough to admit the period before it.
    low, high = months * 30 - 20, months * 30 + 25
    previous = None
    for candidate in known_ends:
        try:
            span = (end_date - datetime.date.fromisoformat(candidate)).days
        except (TypeError, ValueError):
            continue
        if low <= span <= high and (previous is None or candidate > previous):
            previous = candidate
    if previous:
        return (datetime.date.fromisoformat(previous)
                + datetime.timedelta(days=1)).isoformat()
    return fact.period_start


def _period_ends_in(base: Optional[dict], field_name: str) -> set:
    """Period ends the base payload already reports for one field."""
    from finance import period_facts as pf

    if not base:
        return set()
    _winner, rows = pf._winning_concept_facts(base, field_name)   # noqa: SLF001
    return {row.get("end") for row in (rows or []) if row.get("end")}


def _concept_for(field_name: str, base: Optional[dict]) -> Optional[str]:
    """The concept an overlay row for this field must be filed under."""
    from finance import period_facts as pf

    if base:
        winner, _rows = pf._winning_concept_facts(base, field_name)  # noqa: SLF001
        if winner:
            return winner
    concept_map = pf._concept_map(base or {})                       # noqa: SLF001
    entry = concept_map.get(field_name)
    return entry[1][0] if entry and entry[1] else None


def restrict_to_period(overlay: Optional[dict], period_end: Optional[str]) -> dict:
    """The overlay's rows for ONE period end, and nothing else.

    The overlay is built before anything is resolved, so it carries a row for
    every period the release described -- including periods the resolver then
    declines to advance to. Handing all of them to the twelve-month
    reconstruction lets a REFUSED source move a window, which is the one thing
    a refusal is supposed to prevent.
    """
    if not overlay or not period_end:
        return {}
    trimmed: Dict[str, Dict[str, dict]] = {}
    for taxonomy, concepts in (overlay.get("facts") or {}).items():
        for concept, entry in concepts.items():
            for unit, rows in (entry.get("units") or {}).items():
                kept = [row for row in rows if row.get("end") == period_end]
                if not kept:
                    continue
                target = trimmed.setdefault(taxonomy, {}).setdefault(
                    concept, {"units": {}})
                target["units"][unit] = kept
    return {"facts": trimmed} if trimmed else {}
