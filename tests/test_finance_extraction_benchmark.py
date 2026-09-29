"""Scoring the extraction layers against the benchmark.

WHAT EACH NUMBER MEANS

`precision` is the number that decides. §18 is explicit: a layer producing
fewer statements and no invented ones beats a layer producing more with any.
So a FORBIDDEN statement -- a historical table read as guidance, a margin read
as an amount, an analyst estimate read as management's -- is counted as a
CRITICAL false positive and reported separately from an ordinary miss.

`recall` is only meaningful where a case declares `expected_complete`. On the
real cases the expected set is what was verified by reading, not everything
the document contains, so recall there answers "did it find the statements we
checked" and nothing more. The scorer keeps the two populations apart rather
than averaging them into a number that would mean neither.

WHAT IS AND IS NOT MEASURED HERE

V1 is measured end to end: it is deterministic, so running it IS the
measurement.

V2's ACCEPTANCE BOUNDARY is measured end to end, by feeding the validator both
the correct candidates and the wrong ones each case invites. What is NOT
measured is a live model's reading ability. That needs a model call per case,
and reporting an oracle-fed recall number as though it were a model
measurement would be exactly the flattering nonsense this benchmark exists to
prevent. Two consequences, stated rather than smoothed over:

  * V2 RECALL IS NOT MEASURED. The oracle only proposes a candidate where it
    can locate a sentence carrying the expected numbers, so the recall column
    for V2 scores the oracle's sentence-finder. What IS measured, and is
    reported instead, is ACCEPTANCE RATE: of the correct candidates actually
    proposed, how many the validator accepted. A validator that rejects truth
    is as broken as one that accepts fiction, and that number catches it.

  * V1 RECALL IS FLATTERED. On the real cases the expected statements were
    read from the document, but the choice of WHICH statements to verify was
    informed by seeing V1's output first. V1 scoring 100% on the named set is
    therefore partly a measurement of where the set came from. The
    forbidden-statement and precision columns carry no such bias -- those
    were written from the document against V1's behaviour, not from it.

Precision and critical false positives are the numbers to read.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pytest

from finance import guidance as gm
from finance.extraction.schema import (
    GuidanceAction,
    GuidanceCandidate,
    RejectionCode,
    ValueType,
)
from finance.extraction.validator import GuidanceCandidateValidator
from tests.fixtures import extraction_benchmark as bench

N = gm.GuidanceMetricName


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class Score:
    layer: str
    cases: int = 0
    expected_total: int = 0
    found_expected: int = 0
    produced_total: int = 0
    critical_false_positives: List[Tuple[str, str, str]] = field(default_factory=list)
    wrong_unit: List[Tuple[str, str]] = field(default_factory=list)
    wrong_period: List[Tuple[str, str]] = field(default_factory=list)
    complete_cases: int = 0
    complete_expected: int = 0
    complete_found: int = 0
    complete_extra: int = 0
    proposed_correct: int = 0        # correct candidates actually offered
    accepted_correct: int = 0        # ...of which the layer produced

    @property
    def acceptance_rate(self) -> float:
        """Of the correct statements OFFERED, how many survived acceptance."""
        return (self.accepted_correct / self.proposed_correct
                if self.proposed_correct else 1.0)

    @property
    def recall_named(self) -> float:
        return self.found_expected / self.expected_total if self.expected_total else 1.0

    @property
    def recall_complete(self) -> float:
        return (self.complete_found / self.complete_expected
                if self.complete_expected else 1.0)

    @property
    def precision_complete(self) -> float:
        produced = self.complete_found + self.complete_extra
        return self.complete_found / produced if produced else 1.0

    def row(self) -> Dict[str, str]:
        return {
            "layer": self.layer,
            "cases": str(self.cases),
            "found (named)": f"{self.found_expected}/{self.expected_total}",
            "accepted of offered": f"{self.acceptance_rate:.0%} "
                                   f"({self.accepted_correct}/{self.proposed_correct})",
            "precision (complete cases)": f"{self.precision_complete:.0%}",
            "critical false positives": str(len(self.critical_false_positives)),
            "wrong unit": str(len(self.wrong_unit)),
            "wrong period": str(len(self.wrong_period)),
            "statements produced": str(self.produced_total),
        }


def _key(metric) -> Tuple[str, str]:
    return (getattr(metric, "name", None), getattr(metric, "fiscal_period", None))


def _score(layer: str, per_case) -> Score:
    """`per_case(case) -> (produced metrics, keys of correct candidates offered)`.

    The second element is what makes the acceptance rate meaningful. A layer
    that was never offered a statement did not reject it, and counting that
    as a miss would blame the validator for the harness.
    """
    score = Score(layer=layer)
    for case in bench.all_cases():
        score.cases += 1
        produced, offered = per_case(case)
        score.produced_total += len(produced)
        produced_keys = {_key(m): m for m in produced}
        for key in offered:
            score.proposed_correct += 1
            if key in produced_keys:
                score.accepted_correct += 1

        for statement in case.expected_guidance:
            score.expected_total += 1
            match = produced_keys.get(statement.key)
            if match is None:
                continue
            score.found_expected += 1
            if statement.unit and match.unit != statement.unit:
                score.wrong_unit.append((case.case_id, str(statement.key)))

        for metric, period in case.forbidden_guidance:
            if (metric, period) in produced_keys:
                score.critical_false_positives.append(
                    (case.case_id, metric, period))

        # A statement for a period no expected statement names, on a case
        # whose expected set is exhaustive, is a period error.
        if case.expected_complete:
            score.complete_cases += 1
            expected_keys = {s.key for s in case.expected_guidance}
            score.complete_expected += len(expected_keys)
            found = expected_keys & set(produced_keys)
            score.complete_found += len(found)
            extra = set(produced_keys) - expected_keys
            score.complete_extra += len(extra)
            expected_periods = {p for _m, p in expected_keys}
            for metric, period in extra:
                if period not in expected_periods:
                    score.wrong_period.append((case.case_id, f"{metric}@{period}"))
    return score


# ---------------------------------------------------------------------------
# The two layers
# ---------------------------------------------------------------------------

def _v1(case):
    release = gm.extract_guidance_from_text(
        case.document_text(), "BENCH", "acc", "doc", case.filed,
        expected_fiscal_year=case.fiscal_year_hint)
    produced = list(release.all_metrics or release.metrics.values())
    # V1 has no candidate stage; every expected statement is "offered" to it
    # in the sense that the document contains it.
    return produced, [s.key for s in case.expected_guidance]


def _oracle_candidates(case) -> List[GuidanceCandidate]:
    """What a PERFECT reader would propose, plus what a poor one would.

    The correct statements come from the expected set. The wrong ones come
    from `forbidden_guidance` -- the specific misreadings each case invites --
    so the validator is scored on refusing them, not merely on accepting the
    truth.
    """
    text = case.document_text()
    candidates: List[GuidanceCandidate] = []

    for statement in case.expected_guidance:
        sentence = _sentence_for(text, statement)
        if sentence is None:
            continue
        ratio = statement.unit == "ratio"
        unit = ("PERCENT" if ratio else
                "PER_SHARE" if statement.unit == "currency_per_share" else
                "USD_BILLION" if _billions(sentence) else "USD_MILLION")
        candidates.append(GuidanceCandidate(
            metric_id=statement.metric,
            value_type=(ValueType.MARGIN if ratio and "margin" in statement.metric
                        else ValueType.GROWTH_RATE if ratio else ValueType.RANGE),
            low=statement.low, high=statement.high, unit=unit,
            denominator_metric=("revenue" if "margin" in statement.metric else None),
            target_period=statement.target_period,
            target_period_type=("QUARTER" if statement.target_period.startswith("Q")
                                else "FISCAL_YEAR"),
            basis=("NON_GAAP" if statement.basis == "adjusted" else "GAAP"),
            action=GuidanceAction.NEW, prospective=True,
            source_sentence=sentence, confidence=0.9,
            issued_at=case.filed, document_id="acc"))

    for metric, period in case.forbidden_guidance:
        sentence = _any_sentence_mentioning(text, metric, period)
        if sentence is None:
            continue
        candidates.append(GuidanceCandidate(
            metric_id=metric, value_type=ValueType.POINT,
            value=_first_number(sentence), unit="USD_BILLION",
            target_period=period,
            target_period_type=("QUARTER" if period.startswith("Q") else "FISCAL_YEAR"),
            basis="GAAP", action=GuidanceAction.NEW, prospective=True,
            source_sentence=sentence, confidence=0.9,
            issued_at=case.filed, document_id="acc"))
    return candidates


def _v2_validator(case):
    candidates = _oracle_candidates(case)
    expected_keys = {s.key for s in case.expected_guidance}
    offered = [(c.metric_id, c.target_period) for c in candidates
               if (c.metric_id, c.target_period) in expected_keys]
    validator = GuidanceCandidateValidator(
        document_text=case.document_text(), issued_at=case.filed)
    accepted, _rejected = validator.validate_all(candidates)
    return accepted, offered


# -- sentence helpers: the oracle must quote the document, like a model would

import re  # noqa: E402

# Sentences wrap across newlines in a text-converted release, so the splitter
# must not treat a newline as a boundary. An earlier version did, and the
# fragments it produced had lost their forward-looking clause -- which the
# validator then correctly refused. The oracle was wrong, not the boundary.
_SENTENCE = re.compile(r"[^.!?]{15,400}[.!?]")

_FORWARD_IN_SENTENCE = re.compile(
    r"(?i)\b(?:expects?|expected|expecting|guidance|outlook|anticipates?|"
    r"forecasts?|projects?|projected|targets?|reaffirms?|reiterates?)\b")

_NUMBER_TOKEN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _billions(sentence: str) -> bool:
    return bool(re.search(r"(?i)\bbillion\b", sentence))


def _first_number(sentence: str):
    match = re.search(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?", sentence)
    return float(match.group(0).replace(",", "")) if match else None


def _sentence_for(text: str, statement) -> Optional[str]:
    """The sentence carrying this statement's numbers AND its forward clause.

    Both conditions matter. A sentence with the right numbers and no forward
    vocabulary is a results line that happens to share a figure; offering it
    as a guidance candidate tests only whether the validator notices, which is
    already covered by the forbidden-candidate cases.

    Returns None when no such sentence exists -- including for an expected
    statement with no pinned numbers, which cannot be located this way. The
    score counts what was OFFERED, so an unlocatable statement does not become
    a phantom rejection charged against the validator.
    """
    wanted = [v for v in (statement.low, statement.high) if v is not None]
    if not wanted:
        return None
    matches = []
    for sentence in _SENTENCE.findall(text):
        if not _FORWARD_IN_SENTENCE.search(sentence):
            continue
        numbers = [float(t.replace(",", ""))
                   for t in _NUMBER_TOKEN.findall(sentence)]
        if not numbers:
            continue
        hits = 0
        for value in wanted:
            forms = {value, value * 100.0, round(value * 100.0, 4)}
            if any(any(abs(f - n) < 0.011 for n in numbers) for f in forms):
                hits += 1
        if hits == len(wanted):
            matches.append((len(numbers), " ".join(sentence.split())))
    if not matches:
        return None
    # The TIGHTEST citation wins, not the first one found. A press release
    # converted to text can be a single unbroken line, in which case the
    # chunker returns a slab carrying a dozen unrelated figures; the expected
    # numbers appear in it by coincidence, and quoting it as evidence is a
    # misattribution the validator is right to refuse. Fewest competing
    # numbers is the sentence a careful reader would actually cite.
    return min(matches)[1]


def _any_sentence_mentioning(text: str, metric: str, _period: str) -> Optional[str]:
    words = [w for w in metric.split("_") if len(w) > 3]
    for sentence in _SENTENCE.findall(text):
        lowered = sentence.lower()
        if all(w in lowered for w in words) and re.search(r"\d", sentence):
            return sentence.strip()
    return None


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def render_table(scores: List[Score]) -> str:
    rows = [s.row() for s in scores]
    columns = list(rows[0])
    widths = {c: max(len(c), *(len(r[c]) for r in rows)) for c in columns}
    lines = [" | ".join(c.ljust(widths[c]) for c in columns),
             "-|-".join("-" * widths[c] for c in columns)]
    for row in rows:
        lines.append(" | ".join(row[c].ljust(widths[c]) for c in columns))
    return "\n".join(lines)


@pytest.fixture(scope="module")
def scores():
    return [_score("V1 (regex)", _v1), _score("V2-validator", _v2_validator)]


def test_the_benchmark_covers_the_named_classes():
    covered = set(bench.classes_covered())
    for required in ("high_growth", "loss_making", "foreign_private_issuer",
                     "multi_horizon", "guidance_heavy", "table_contamination",
                     "unit_identity", "economic_dedup"):
        assert required in covered, f"{required} is not represented"
    assert len(bench.all_cases()) >= 10


def test_the_benchmark_reports_both_layers(scores, capsys):
    """Prints the comparison table. Always passes; the assertions are below."""
    with capsys.disabled():
        print("\n\nFinance Extraction benchmark\n" + "=" * 78)
        print(render_table(scores))
        print(f"\ncases: {len(bench.all_cases())}  "
              f"classes: {len(bench.classes_covered())}")
        for score in scores:
            if score.critical_false_positives:
                print(f"\n{score.layer} critical false positives:")
                for case_id, metric, period in score.critical_false_positives:
                    print(f"  {case_id}: {metric} @ {period}")
    assert scores


def test_v2_produces_no_critical_false_positive(scores):
    """§19's first acceptance criterion, and the one that matters most.

    The validator is fed the exact misreadings each case invites. Accepting
    any of them would mean the boundary does not hold.
    """
    v2 = next(s for s in scores if s.layer == "V2-validator")
    assert v2.critical_false_positives == [], v2.critical_false_positives


def test_v2_units_are_correct_on_everything_it_accepts(scores):
    v2 = next(s for s in scores if s.layer == "V2-validator")
    assert v2.wrong_unit == [], v2.wrong_unit


def test_v2_target_periods_are_correct_on_everything_it_accepts(scores):
    v2 = next(s for s in scores if s.layer == "V2-validator")
    assert v2.wrong_period == [], v2.wrong_period


def test_the_validator_accepts_the_statements_that_are_true(scores):
    """The other half of the guarantee.

    Refusing everything would score a perfect precision and be useless. Of
    the correct candidates actually offered, the validator must accept all of
    them -- a boundary that rejects truth is as broken as one that admits
    fiction, and this is the number that catches it.

    Deliberately NOT a recall comparison against V1: this harness cannot
    measure V2 recall (see the module docstring), and asserting a number it
    cannot measure would make the suite claim something untrue.
    """
    v2 = next(s for s in scores if s.layer == "V2-validator")
    assert v2.proposed_correct > 0, "no correct candidate was offered at all"
    assert v2.acceptance_rate == 1.0, (
        f"the validator rejected {v2.proposed_correct - v2.accepted_correct} "
        "correct candidate(s)")


def test_v1_critical_false_positives_are_recorded_not_asserted_away(scores):
    """V1's false positives are the measurement, not a failure of this suite.

    They are what V2 exists to prevent, and pinning the count means a change
    to V1 that silently fixes or worsens them is visible here.
    """
    v1 = next(s for s in scores if s.layer == "V1 (regex)")
    kinds = {metric for _case, metric, _period in v1.critical_false_positives}
    assert v1.critical_false_positives, (
        "V1 produced none, which would make the comparison uninformative")
    assert kinds <= {"revenue_growth", "revenue"}, kinds


def test_no_benchmark_ticker_appears_in_production_code():
    """§16. The names are DATA. A production path that knows one of them has
    stopped being general, which is the failure this whole architecture is
    organised against."""
    import pathlib

    names = ("MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "CRM", "ADBE", "JNJ",
             "UNH", "RIVN", "TSLA", "AAPL", "GOOGL", "AMZN", "META", "AMD",
             "PGR", "JBS", "CASY", "MRK")
    offenders = []
    for path in list(pathlib.Path("finance").rglob("*.py")) + \
            list(pathlib.Path("tools").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for name in names:
            # A NAME IN A COMMENT is documentation of a live failure and is
            # how this codebase records why a rule exists. A name in CODE is
            # a branch. Only the second is forbidden.
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or name not in line:
                    continue
                if re.search(rf'["\']{name}["\']', line):
                    offenders.append(f"{path}: {stripped[:90]}")
    assert not offenders, offenders
