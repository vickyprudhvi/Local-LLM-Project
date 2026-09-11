"""Running the Reported Actuals Source Integration benchmark, and scoring it.

TWO STAGES, DELIBERATELY SEPARATE

    measure(case) -> Measurement    RAW. What the layer did, no verdicts.
    score(...)    -> Score          Verdicts only, no execution.

The same split the Actualization benchmark uses and for the same reason: a
scoring rule that turns out to be wrong can be corrected and the stored
measurements re-scored without running production code again, and without the
temptation to adjust a rule while watching a number move.

WHAT IS COMPARED AGAINST WHAT

Expected data comes from `tests/fixtures/reported_actuals_benchmark.py`, which
is generated from the same constants the fixture DOCUMENTS are generated from.
Nothing here reads an extractor's output to decide what is true.

THE FACT-SUPPORT CHECK

`UNSUPPORTED_ACCEPTED_FACT` is not scored against the expected list. It is
scored against the DOCUMENT: every accepted fact must quote a row label and a
column label that appear in the text it came from, and its printed magnitude
must appear there too. A layer that invented a number would pass an expected-
value comparison on every field it happened to get right and fail this.
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from finance import semantics as sem
from tests.fixtures import reported_actuals_benchmark as bench
from tests.fixtures.reported_actuals_benchmark import FailureClass, Hard, SourceCase


# A value is "the same" as an expected one within this relative tolerance.
# Tight: these are exact figures printed in a document, not estimates.
TOLERANCE = 1e-6


def _close(left: Optional[float], right: Optional[float]) -> bool:
    if left is None or right is None:
        return False
    scale = max(abs(right), 1.0)
    return abs(left - right) / scale <= TOLERANCE


# ---------------------------------------------------------------------------
# The raw record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Measurement:
    """What the source layer did on ONE case."""

    case_id: str
    candidate_count: int = 0
    primary_period: Optional[str] = None
    primary_period_type: Optional[str] = None
    primary_completeness: Optional[str] = None
    primary_currency: Optional[str] = None
    values: Dict[str, float] = field(default_factory=dict)
    values_populated: int = 0
    # Every canonical fact accepted for the primary period, with its evidence.
    accepted_facts: List[dict] = field(default_factory=list)
    # Every canonical fact at any period, for the period-accuracy check.
    all_periods: List[str] = field(default_factory=list)
    scales_seen: List[str] = field(default_factory=list)
    currencies_seen: List[str] = field(default_factory=list)
    frequencies_seen: List[str] = field(default_factory=list)
    statement_kinds_used: List[str] = field(default_factory=list)
    table_kinds: List[str] = field(default_factory=list)
    rejection_codes: Dict[str, int] = field(default_factory=dict)
    # discovery
    discovery_is_source: Optional[bool] = None
    selected_document_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def measure(case: SourceCase) -> Measurement:
    """Run the source layer over one case document. Records, never judges."""
    from finance.reported_actuals import extract_reported_actuals
    from finance.reported_actuals.discovery import (
        classify_filing,
        select_results_document,
    )

    try:
        extraction = extract_reported_actuals(
            case.document, accession=case.accession, form=case.form,
            filed=case.filed, document=case.document_id,
            gaap_basis=(sem.AccountingBasis.IFRS if case.form.startswith("6-K")
                        or case.form.startswith("20-F")
                        else sem.AccountingBasis.GAAP))
    except Exception as failure:                              # noqa: BLE001
        return Measurement(case_id=case.case_id,
                           error=f"{type(failure).__name__}: {failure}")

    discovery_is_source = None
    if case.submissions_row is not None:
        discovery_is_source, _evidence, _reason = classify_filing(case.submissions_row)

    selected = None
    if case.filing_index_html is not None:
        selected = select_results_document(
            case.filing_index_html, accession=case.accession, form=case.form,
            filing_date=case.filed).document_id

    primary = None
    if case.expected_period_end:
        primary = next((c for c in extraction.candidates
                        if c.period_end == case.expected_period_end), None)
    if primary is None and extraction.candidates:
        primary = extraction.candidates[0]

    accepted = [f.to_dict() for f in extraction.canonical_facts
                if primary is not None and f.period_end == primary.period_end
                and (f.frequency == primary.period_type
                     or f.flow_or_instant == sem.FlowOrInstant.INSTANT)]

    return Measurement(
        case_id=case.case_id,
        candidate_count=len(extraction.candidates),
        primary_period=(primary.period_end if primary else None),
        primary_period_type=(primary.period_type if primary else None),
        primary_completeness=(primary.statement_completeness if primary else None),
        primary_currency=(primary.currency if primary else None),
        values=dict(primary.values) if primary else {},
        values_populated=len(primary.values or {}) if primary else 0,
        accepted_facts=accepted,
        all_periods=sorted({f.period_end for f in extraction.canonical_facts}),
        scales_seen=sorted({f.scale or "?" for f in extraction.canonical_facts}),
        currencies_seen=sorted({f.currency for f in extraction.canonical_facts}),
        frequencies_seen=sorted({f.frequency for f in extraction.canonical_facts}),
        statement_kinds_used=sorted({f.statement_kind
                                     for f in extraction.canonical_facts}),
        table_kinds=[t.get("kind") for t in extraction.tables],
        rejection_codes=extraction.rejection_codes(),
        discovery_is_source=discovery_is_source,
        selected_document_id=selected)


def measure_all(cases: Optional[Sequence[SourceCase]] = None) -> List[Measurement]:
    return [measure(case) for case in (cases if cases is not None else bench.cases())]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class Tally:
    correct: int = 0
    scored: int = 0
    excluded: int = 0
    misses: List[str] = field(default_factory=list)

    def hit(self, ok: bool, label: str = "") -> None:
        self.scored += 1
        if ok:
            self.correct += 1
        elif label:
            self.misses.append(label)

    def skip(self) -> None:
        self.excluded += 1

    @property
    def rate(self) -> float:
        return self.correct / self.scored if self.scored else 1.0

    def text(self) -> str:
        return "n/a" if not self.scored else \
            f"{self.rate:.0%} ({self.correct}/{self.scored})"


@dataclass
class Score:
    cases: int = 0
    discovery_precision: Tally = field(default_factory=Tally)
    discovery_recall: Tally = field(default_factory=Tally)
    document_selection: Tally = field(default_factory=Tally)
    table_precision: Tally = field(default_factory=Tally)
    fact_precision: Tally = field(default_factory=Tally)
    fact_recall: Tally = field(default_factory=Tally)
    metric_identity: Tally = field(default_factory=Tally)
    period_accuracy: Tally = field(default_factory=Tally)
    unit_scale_accuracy: Tally = field(default_factory=Tally)
    currency_accuracy: Tally = field(default_factory=Tally)
    guidance_separation: Tally = field(default_factory=Tally)
    values_population: Tally = field(default_factory=Tally)
    completeness_accuracy: Tally = field(default_factory=Tally)
    hard: Dict[str, int] = field(default_factory=dict)
    hard_detail: Dict[str, List[str]] = field(default_factory=dict)
    failure_classes: Dict[str, List[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def hard_total(self) -> int:
        return sum(self.hard.values())

    def note(self, failure_class: str, label: str) -> None:
        self.failure_classes.setdefault(failure_class, []).append(label)

    def flag(self, name: str, label: str) -> None:
        self.hard[name] = self.hard.get(name, 0) + 1
        self.hard_detail.setdefault(name, []).append(label)


def _blank() -> Score:
    return Score(hard={name: 0 for name in Hard.ALL})


# Numbers a document prints, for the support check. Any spelling of the digits
# the reader could have taken them from.
_DIGITS = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _document_numbers(document: str) -> set:
    return {match.group(0).replace(",", "") for match in _DIGITS.finditer(document)}


def _supported(fact: dict, document: str, numbers: set) -> bool:
    """Does the document actually contain this fact's row, column and figure?"""
    if fact.get("row_label") not in document:
        return False
    if fact.get("column_label") not in document:
        return False
    raw = fact.get("raw_value")
    if raw is None:
        return False
    printed = {f"{abs(raw):,.1f}".replace(",", ""), f"{abs(raw):g}",
               f"{abs(raw):.0f}", f"{abs(raw):.1f}", f"{abs(raw):.2f}"}
    return bool(printed & numbers)


def score(measurements: Sequence[Measurement],
          cases: Optional[Sequence[SourceCase]] = None) -> Score:
    result = _blank()
    by_id = {case.case_id: case for case in
             (cases if cases is not None else bench.cases())}
    for record in measurements:
        case = by_id.get(record.case_id)
        if case is None:
            continue
        result.cases += 1
        if record.error:
            result.errors.append(f"{record.case_id}: {record.error}")
            continue
        _score_one(case, record, result)
    _score_discovery(result)
    _score_document_selection(result)
    return result


def _score_one(case: SourceCase, m: Measurement, s: Score) -> None:
    tag = case.case_id

    # -- a case that must produce nothing ----------------------------------
    if case.expects_no_candidate:
        ok = m.candidate_count == 0
        s.fact_precision.hit(ok, f"{tag}: produced {m.candidate_count} candidate(s)")
        if not ok:
            s.note(FailureClass.ACTUAL_SECTION_SELECTION,
                   f"{tag}: {m.candidate_count} candidates")
            if case.case_id.startswith("F-"):
                s.flag(Hard.UNRELATED_8K_AS_EARNINGS_ACTUAL, tag)
            if case.case_id.startswith("I2"):
                s.flag(Hard.WRONG_UNIT_ACCEPTED,
                       f"{tag}: a table with no stated scale produced facts")
        return

    if not case.expected_facts:
        return

    # -- period ------------------------------------------------------------
    period_ok = m.primary_period == case.expected_period_end
    s.period_accuracy.hit(period_ok, f"{tag}: {m.primary_period!r}")
    type_ok = m.primary_period_type == case.expected_period_type
    s.period_accuracy.hit(type_ok, f"{tag}: type {m.primary_period_type!r}")
    if not (period_ok and type_ok):
        s.note(FailureClass.PERIOD_RESOLUTION, f"{tag}: {m.primary_period!r}")

    # -- units and currency ------------------------------------------------
    if case.expected_scale:
        ok = m.scales_seen == [case.expected_scale]
        s.unit_scale_accuracy.hit(ok, f"{tag}: {m.scales_seen}")
        if not ok:
            s.note(FailureClass.UNIT_SCALE, f"{tag}: {m.scales_seen}")
    if case.expected_currency:
        ok = m.currencies_seen == [case.expected_currency]
        s.currency_accuracy.hit(ok, f"{tag}: {m.currencies_seen}")
        if not ok:
            s.note(FailureClass.CURRENCY, f"{tag}: {m.currencies_seen}")
            s.flag(Hard.WRONG_CURRENCY_ACCEPTED,
                   f"{tag}: recorded {m.currencies_seen} for "
                   f"{case.expected_currency}")

    # -- completeness ------------------------------------------------------
    if case.expected_completeness:
        ok = m.primary_completeness == case.expected_completeness
        s.completeness_accuracy.hit(
            ok, f"{tag}: {m.primary_completeness!r} for {case.expected_completeness!r}")
        if not ok:
            s.note(FailureClass.STATEMENT_COMPLETENESS, f"{tag}: {m.primary_completeness}")

    # -- facts -------------------------------------------------------------
    for name, expected in sorted(case.expected_facts.items()):
        found = m.values.get(name)
        ok = _close(found, expected)
        s.fact_recall.hit(ok, f"{tag}/{name}: {found!r} for {expected!r}")
        if not ok:
            s.note(FailureClass.METRIC_IDENTITY, f"{tag}/{name}")

    for name, found in sorted(m.values.items()):
        expected = case.expected_facts.get(name)
        ok = expected is not None and _close(found, expected)
        s.fact_precision.hit(ok, f"{tag}/{name}: {found!r} not expected")
        if not ok:
            s.note(FailureClass.METRIC_IDENTITY, f"{tag}/{name}: unexpected")

    # -- forbidden figures -------------------------------------------------
    for name, value, why in case.forbidden_values:
        taken = _close(m.values.get(name), value)
        s.guidance_separation.hit(not taken, f"{tag}/{name}: {why}")
        if not taken:
            continue
        s.note(FailureClass.ACTUAL_GUIDANCE_SEPARATION, f"{tag}/{name}: {why}")
        if "guidance" in why:
            s.flag(Hard.GUIDANCE_AS_ACTUAL, f"{tag}/{name}: {why}")
        elif "estimate" in why:
            s.flag(Hard.ANALYST_ESTIMATE_AS_ACTUAL, f"{tag}/{name}: {why}")
        elif "months" in why or "year" in why:
            s.flag(Hard.YTD_AS_QUARTER_ACCEPTED, f"{tag}/{name}: {why}")
            s.flag(Hard.WRONG_PERIOD_ACCEPTED, f"{tag}/{name}: {why}")
        elif "scale" in why or "millions" in why or "unscaled" in why:
            s.flag(Hard.WRONG_UNIT_ACCEPTED, f"{tag}/{name}: {why}")
        else:
            s.flag(Hard.UNSUPPORTED_ACCEPTED_FACT, f"{tag}/{name}: {why}")

    # -- metric identity: every accepted fact came from a reported statement
    for fact in m.accepted_facts:
        kind = fact.get("statement_kind")
        ok = kind in ("INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW",
                      "SHARE_COUNT")
        s.metric_identity.hit(ok, f"{tag}/{fact.get('field')}: from {kind}")
        if not ok:
            s.note(FailureClass.ACTUAL_SECTION_SELECTION,
                   f"{tag}/{fact.get('field')}: {kind}")

    # -- values populated --------------------------------------------------
    populated = m.values_populated >= len(case.expected_facts)
    s.values_population.hit(populated,
                            f"{tag}: {m.values_populated} of "
                            f"{len(case.expected_facts)}")

    # -- table classification precision ------------------------------------
    for kind in m.table_kinds:
        s.table_precision.hit(kind is not None, f"{tag}: unclassified table")

    # -- every accepted fact must be IN the document -----------------------
    numbers = _document_numbers(case.document)
    for fact in m.accepted_facts:
        if not _supported(fact, case.document, numbers):
            s.flag(Hard.UNSUPPORTED_ACCEPTED_FACT,
                   f"{tag}/{fact.get('field')}: {fact.get('row_label')!r} / "
                   f"{fact.get('column_label')!r} / {fact.get('raw_value')!r}")

    # -- a flow frequency that is not the declared one is a wrong period ---
    for fact in m.accepted_facts:
        if fact.get("flow_or_instant") != sem.FlowOrInstant.FLOW:
            continue
        if fact.get("frequency") != m.primary_period_type:
            s.flag(Hard.WRONG_PERIOD_ACCEPTED,
                   f"{tag}/{fact.get('field')}: {fact.get('frequency')} inside a "
                   f"{m.primary_period_type} candidate")
            if fact.get("frequency", "").startswith("YTD"):
                s.flag(Hard.YTD_AS_QUARTER_ACCEPTED, f"{tag}/{fact.get('field')}")


def _score_discovery(s: Score) -> None:
    """Section 24's discovery precision and recall, over one submissions mix."""
    from finance.reported_actuals.discovery import find_reported_actual_filings

    accepted, refused = find_reported_actual_filings(
        bench.DISCOVERY_SUBMISSIONS, limit=20)
    accepted_ids = {c.accession for c in accepted}

    for accession, truth in sorted(bench.DISCOVERY_TRUTH.items()):
        if truth is None:
            s.discovery_precision.skip()
            s.discovery_recall.skip()
            continue
        found = accession in accepted_ids
        if found:
            s.discovery_precision.hit(truth, f"{accession}: accepted, is not a source")
            if not truth:
                s.note(FailureClass.SOURCE_DISCOVERY, f"{accession}: false positive")
                s.flag(Hard.UNRELATED_8K_AS_EARNINGS_ACTUAL, accession)
        if truth:
            s.discovery_recall.hit(found, f"{accession}: not discovered")
            if not found:
                s.note(FailureClass.SOURCE_DISCOVERY, f"{accession}: missed")


def _score_document_selection(s: Score) -> None:
    from finance.reported_actuals.discovery import select_results_document

    for name, index_html, expected in bench.DOCUMENT_SELECTION_CASES:
        chosen = select_results_document(index_html).document_id
        ok = chosen == expected
        s.document_selection.hit(ok, f"{name}: {chosen!r} for {expected!r}")
        if not ok:
            s.note(FailureClass.DOCUMENT_SELECTION, f"{name}: {chosen!r}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

ROWS = (
    ("Source-discovery precision", "discovery_precision"),
    ("Source-discovery recall", "discovery_recall"),
    ("Document selection", "document_selection"),
    ("Actual-table classification", "table_precision"),
    ("Actual-fact precision", "fact_precision"),
    ("Actual-fact recall", "fact_recall"),
    ("Metric-identity accuracy", "metric_identity"),
    ("Period accuracy", "period_accuracy"),
    ("Unit/scale accuracy", "unit_scale_accuracy"),
    ("Currency accuracy", "currency_accuracy"),
    ("Actual-vs-guidance separation", "guidance_separation"),
    ("Candidate-values population", "values_population"),
    ("Completeness accuracy", "completeness_accuracy"),
)


def render_table(result: Score) -> str:
    rows = [("Metric", "Result")]
    rows += [(label, getattr(result, attribute).text()) for label, attribute in ROWS]
    rows.append(("Hard safety failures", str(result.hard_total)))
    width = max(len(a) for a, _b in rows)
    lines = [f"{rows[0][0].ljust(width)} | {rows[0][1]}",
             f"{'-' * width}-|-{'-' * 16}"]
    for label, value in rows[1:]:
        lines.append(f"{label.ljust(width)} | {value}")
    return "\n".join(lines)


def render_hard(result: Score) -> str:
    return "\n".join(f"  {name}: {result.hard.get(name, 0)}" for name in Hard.ALL)
