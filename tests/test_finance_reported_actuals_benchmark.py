"""The Reported Actuals Source Integration benchmark, and its own safety.

WHAT THIS FILE ASSERTS, IN ORDER OF WHAT MATTERS

  1. Every section 25 counter is ZERO across the whole case set.
  2. Every one of those counters is EXERCISED -- there is a case for each
     whose score changes the moment the corresponding check stops working
     (section 26). A counter that reads zero because nothing reached it is
     not a measurement.
  3. The scorer cannot flatter anything: an accepted fact is checked against
     the DOCUMENT, not against the expected list, so a layer that invented a
     number fails even on a field it happened to get right.

The secondary accuracy numbers are printed rather than pinned, for the same
reason the Actualization benchmark prints its own: pinning one creates
pressure to move production to satisfy it.

WHAT THE WRONG ANSWERS ARE

They are IN the documents. A guided revenue, an analyst consensus, a
nine-month total, an adjusted operating income, the same digits in the wrong
scale -- each printed in the same release, each a plausible thing for a
careless reader to return. Measuring precision against wrong answers that
were never present would measure nothing.
"""

import dataclasses
import hashlib
import pathlib

import pytest

from tests import reported_actuals_benchmark_harness as harness
from tests.reported_actuals_benchmark_harness import Measurement
from tests.fixtures import reported_actuals_benchmark as bench
from tests.fixtures.reported_actuals_benchmark import Hard


@pytest.fixture(scope="module")
def measurements():
    """The raw measurement. Produced once, scored many times."""
    return harness.measure_all()


@pytest.fixture(scope="module")
def result(measurements):
    return harness.score(measurements)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_the_benchmark_covers_the_named_case_classes():
    covered = set(bench.classes_covered())
    assert covered == set("ABCDEFGHIJK"), sorted(covered)
    assert len(bench.ALL_CASES) >= 11, len(bench.ALL_CASES)


def test_every_hard_requirement_has_a_positive_control_case():
    """Section 26. A zero nobody could have made non-zero is not a result."""
    missing = [name for name, ids in bench.controls_covered().items() if not ids]
    assert not missing, missing


def test_no_issuer_is_named_anywhere_in_the_fixtures():
    """A case is a calendar, a set of statements and a set of column headers."""
    for path in ("tests/fixtures/reported_actuals_releases.py",
                 "tests/fixtures/reported_actuals_benchmark.py"):
        text = pathlib.Path(path).read_text(encoding="utf-8")
        for name in ("MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "CRM", "ADBE",
                     "AAPL", "GOOGL", "AMZN", "META", "AMD", "JBS", "CASY",
                     "MRK", "SHEL", "NVS", "Broadcom", "Oracle", "Microsoft"):
            assert name not in text, f"{path}: {name}"


def test_no_ticker_specific_branch_reached_production():
    import re

    names = ("MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "CRM", "ADBE", "JNJ",
             "UNH", "RIVN", "TSLA", "AAPL", "GOOGL", "AMZN", "META", "AMD",
             "PGR", "JBS", "CASY", "MRK", "SHEL", "NVS")
    offenders = []
    for path in list(pathlib.Path("finance").rglob("*.py")) + \
            list(pathlib.Path("tools").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for name in names:
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or name not in line:
                    continue
                if re.search(rf'["\']{name}["\']', line):
                    offenders.append(f"{path}: {stripped[:90]}")
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_the_benchmark_reports_its_numbers(result, capsys):
    with capsys.disabled():
        print("\n\nReported Actuals Source Integration benchmark\n" + "=" * 78)
        print(harness.render_table(result))
        print("\nHard safety failures, by requirement:")
        print(harness.render_hard(result))
        print(f"\ncases: {len(bench.ALL_CASES)}  "
              f"classes: {len(bench.classes_covered())}")
        if result.failure_classes:
            print("\ngeneralized failure classes:")
            for name, entries in sorted(result.failure_classes.items()):
                print(f"  {name}: {len(entries)}")
        for label, attribute in harness.ROWS:
            for miss in getattr(result, attribute).misses:
                print(f"  MISS {label}: {miss}")
    assert result.cases == len(bench.ALL_CASES)
    assert not result.errors, result.errors


# ---------------------------------------------------------------------------
# Section 25: the hard requirements
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("requirement", Hard.ALL)
def test_no_hard_safety_failure(result, requirement):
    detail = result.hard_detail.get(requirement, [])
    assert result.hard.get(requirement, 0) == 0, detail


def test_the_layer_did_not_crash_on_any_case(measurements):
    broken = [m.error for m in measurements if m.error]
    assert not broken, broken


# ---------------------------------------------------------------------------
# Section 26: the checks are exercised, not merely zero
# ---------------------------------------------------------------------------
#
# Each test takes the REAL measurement for the case that controls one
# requirement and injects exactly the defect that requirement's check exists
# to catch -- what the measurement would have looked like had the check been
# disconnected. If the counter does not move, its zero above meant nothing.

def _for(measurements, case_id) -> Measurement:
    return next(m for m in measurements if m.case_id == case_id)


def _rescore(measurements, replacement):
    others = [m for m in measurements if m.case_id != replacement.case_id]
    return harness.score(others + [replacement])


def _control(requirement) -> str:
    ids = bench.controls_covered()[requirement]
    assert ids, requirement
    return ids[0]


def _with_value(record: Measurement, field: str, value: float) -> Measurement:
    values = dict(record.values)
    values[field] = value
    return dataclasses.replace(record, values=values)


def test_control_guidance_as_actual(measurements):
    from tests.fixtures import reported_actuals_releases as R

    case_id = _control(Hard.GUIDANCE_AS_ACTUAL)
    real = _for(measurements, case_id)
    assert real.values["revenue"] != R.GUIDED_Q1_REVENUE
    broken = _with_value(real, "revenue", R.GUIDED_Q1_REVENUE)
    assert _rescore(measurements, broken).hard[Hard.GUIDANCE_AS_ACTUAL] == 1


def test_control_analyst_estimate_as_actual(measurements):
    from tests.fixtures import reported_actuals_releases as R

    case_id = _control(Hard.ANALYST_ESTIMATE_AS_ACTUAL)
    real = _for(measurements, case_id)
    assert real.values["revenue"] != R.ANALYST_REVENUE
    broken = _with_value(real, "revenue", R.ANALYST_REVENUE)
    assert _rescore(measurements, broken).hard[Hard.ANALYST_ESTIMATE_AS_ACTUAL] == 1


def test_control_ytd_as_quarter_accepted(measurements):
    from tests.fixtures import reported_actuals_releases as R

    case_id = "J-quarter-and-ytd-columns"
    real = _for(measurements, case_id)
    assert real.values["revenue"] == R.Q3_QUARTER_REVENUE
    broken = _with_value(real, "revenue", R.Q3_YTD_REVENUE)
    rescored = _rescore(measurements, broken)
    assert rescored.hard[Hard.YTD_AS_QUARTER_ACCEPTED] >= 1
    assert rescored.hard[Hard.WRONG_PERIOD_ACCEPTED] >= 1


def test_control_wrong_period_accepted(measurements):
    """A flow whose duration is not the candidate's own is the wrong period,
    whatever its end date says."""
    case_id = "A-complete-q4-release"
    real = _for(measurements, case_id)
    facts = [dict(f) for f in real.accepted_facts]
    flow = next(f for f in facts if f["flow_or_instant"] == "FLOW")
    flow["frequency"] = "YTD_9M"
    broken = dataclasses.replace(real, accepted_facts=facts)
    assert _rescore(measurements, broken).hard[Hard.WRONG_PERIOD_ACCEPTED] >= 1


def test_control_wrong_unit_accepted(measurements):
    case_id = "I-thousands-scale"
    real = _for(measurements, case_id)
    assert real.scales_seen == ["thousands"]
    broken = _with_value(real, "revenue", 1_450_000.0)
    assert _rescore(measurements, broken).hard[Hard.WRONG_UNIT_ACCEPTED] >= 1

    # And the table that states no scale must keep producing nothing.
    no_scale = _for(measurements, "I2-no-scale-stated")
    assert no_scale.candidate_count == 0
    invented = dataclasses.replace(no_scale, candidate_count=1)
    assert _rescore(measurements, invented).hard[Hard.WRONG_UNIT_ACCEPTED] >= 1


def test_control_wrong_currency_accepted(measurements):
    case_id = _control(Hard.WRONG_CURRENCY_ACCEPTED)
    real = _for(measurements, case_id)
    assert real.currencies_seen == ["EUR"]
    broken = dataclasses.replace(real, currencies_seen=["USD"])
    assert _rescore(measurements, broken).hard[Hard.WRONG_CURRENCY_ACCEPTED] == 1


def test_control_unrelated_8k_as_earnings_actual(measurements):
    case_id = _control(Hard.UNRELATED_8K_AS_EARNINGS_ACTUAL)
    real = _for(measurements, case_id)
    assert real.discovery_is_source is False
    assert real.candidate_count == 0
    broken = dataclasses.replace(real, candidate_count=1)
    assert _rescore(measurements, broken).hard[
        Hard.UNRELATED_8K_AS_EARNINGS_ACTUAL] >= 1


def test_control_silent_same_period_value_conflict(measurements):
    """The control is that the release and the filing DISAGREE and the
    disagreement survives into the values the resolver reconciles.

    Disconnected, the two candidates carry the same number and there is
    nothing left to detect -- which is precisely the state production was in
    before `values` was populated.
    """
    from finance.actualization import detect_same_period_conflicts
    from tests.fixtures import reported_actuals_releases as R

    release = _for(measurements, "A-complete-q4-release")
    filing = _for(measurements, "D-periodic-filing-conflicts")
    assert release.values["revenue"] != filing.values["revenue"]

    from finance.extraction.schema import ReportedActualCandidate

    def candidate(record, form):
        return ReportedActualCandidate(
            period_end=record.primary_period, period_type=record.primary_period_type,
            form=form, values=dict(record.values))

    conflicts = detect_same_period_conflicts(candidate(filing, "10-K"),
                                             [candidate(release, "8-K")])
    assert [c.metric for c in conflicts] == ["revenue"]

    # Disconnected: values never populated, so nothing can be compared.
    empty = ReportedActualCandidate(period_end=filing.primary_period,
                                    period_type=filing.primary_period_type,
                                    form="8-K", values={})
    assert detect_same_period_conflicts(candidate(filing, "10-K"), [empty]) == []


def test_control_unsupported_accepted_fact(measurements):
    """Scored against the DOCUMENT, not against the expected list."""
    case_id = "A-complete-q4-release"
    real = _for(measurements, case_id)
    facts = [dict(f) for f in real.accepted_facts]
    facts[0]["raw_value"] = 999_999.0
    facts[0]["row_label"] = "A line no filing contains"
    broken = dataclasses.replace(real, accepted_facts=facts)
    assert _rescore(measurements, broken).hard[Hard.UNSUPPORTED_ACCEPTED_FACT] >= 1


# ---------------------------------------------------------------------------
# Scorer safety
# ---------------------------------------------------------------------------

def test_the_scorer_does_not_infer_truth_from_the_layers_output(measurements):
    """Expected data is FIXED. A layer that produced nothing scores nothing."""
    empty = [Measurement(case_id=case.case_id) for case in bench.ALL_CASES]
    scored = harness.score(empty)
    assert scored.fact_recall.correct == 0
    assert scored.period_accuracy.correct == 0
    assert harness.score(measurements).fact_recall.correct > 0


def test_scoring_is_a_pure_function_of_the_stored_measurements(measurements):
    first = harness.score(measurements)
    second = harness.score(measurements)
    assert first.hard == second.hard
    assert first.fact_recall.correct == second.fact_recall.correct


def test_reporting_failure_cannot_alter_a_completed_measurement(measurements):
    before = [m.to_dict() for m in measurements]

    class Exploding:
        def write(self, _text):
            raise IOError("the checkpoint device is gone")

    with pytest.raises(IOError):
        Exploding().write(harness.render_table(harness.score(measurements)))

    assert [m.to_dict() for m in measurements] == before
    assert harness.score(measurements).hard_total == 0


def test_a_case_that_must_produce_nothing_is_scored_as_such(measurements):
    for case_id in ("F-unrelated-8k", "B2-prose-only-release", "I2-no-scale-stated"):
        assert _for(measurements, case_id).candidate_count == 0


# ---------------------------------------------------------------------------
# Section 33: the benchmark measured ONE fixed implementation
# ---------------------------------------------------------------------------

MEASURED_MODULES = (
    "finance/reported_actuals/discovery.py",
    "finance/reported_actuals/tables.py",
    "finance/reported_actuals/facts.py",
    "finance/reported_actuals/candidates.py",
    "finance/reported_actuals/runtime.py",
    "finance/extraction/document_resolver.py",
    "finance/actualization.py",
    "finance/actualization_runtime.py",
    "finance/ttm.py",
)


def _hashes():
    return {name: hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()[:16]
            for name in MEASURED_MODULES}


def test_the_measured_production_code_did_not_change_during_the_run(measurements):
    before = _hashes()
    harness.measure_all()
    assert _hashes() == before
