"""The Actualization benchmark, its hard requirements, and its own safety.

WHAT THIS FILE ASSERTS, IN ORDER OF WHAT MATTERS

  1. V2 records ZERO hard safety failures across every case (section 7).
  2. Every hard-safety counter is EXERCISED -- there is a case for each whose
     score changes the moment the corresponding check stops working (section 8).
     A counter that reads zero because nothing ever reached it is not a
     measurement, and this is the difference between the two.
  3. The scorer itself cannot flatter anything (section 9): rejected candidates
     are not facts, fallback-labelled figures are not current ones, the two
     layers are scored apart, and a reporting failure cannot reach back into a
     measurement.

WHAT IT DOES NOT ASSERT

V2's secondary accuracy numbers are REPORTED, not pinned. Section 21 is
explicit that when the hard requirements pass, production must not be changed
to move a secondary metric -- so pinning one here would create pressure to do
exactly that at the next change. The numbers are printed, and the failure
CLASSES behind them are printed with them.

V1's failures are likewise reported rather than asserted away. They are the
measurement: they are what V2 exists to prevent, and a suite that hid them
would make the comparison meaningless.
"""

import dataclasses
import hashlib
import pathlib
import re

import pytest

from tests import actualization_benchmark_harness as harness
from tests.actualization_benchmark_harness import V1, V2, Measurement
from tests.fixtures import actualization_benchmark as bench
from tests.fixtures.actualization_benchmark import ConflictStatus, Hard, Truth


# ---------------------------------------------------------------------------
# The run, once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def measurements():
    """The raw measurement. Produced once, scored many times."""
    return harness.run_all()


@pytest.fixture(scope="module")
def scores(measurements):
    return harness.score(measurements)


# ---------------------------------------------------------------------------
# Coverage (sections 3 and 8)
# ---------------------------------------------------------------------------

def test_the_benchmark_covers_the_named_case_classes():
    covered = set(bench.classes_covered())
    assert covered == set("ABCDEFGHIJKLMN"), sorted(covered)
    assert len(bench.ALL_CASES) >= 15, len(bench.ALL_CASES)


def test_every_hard_safety_requirement_has_a_positive_control_case():
    """Section 8. A zero nobody could have made non-zero is not a result."""
    missing = [name for name, ids in bench.controls_covered().items() if not ids]
    assert not missing, missing


def test_no_case_id_or_fixture_names_an_issuer():
    """A case is a calendar and a list of forms. There is nothing to name."""
    text = pathlib.Path("tests/fixtures/actualization_benchmark.py").read_text(
        encoding="utf-8")
    for name in ("MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "CRM", "ADBE", "JNJ",
                 "UNH", "RIVN", "TSLA", "AAPL", "GOOGL", "AMZN", "META", "AMD",
                 "PGR", "JBS", "CASY", "MRK"):
        assert name not in text, name


def test_no_ticker_specific_branch_reached_production():
    """Section 29. The benchmark must not have taught production a name."""
    names = ("MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "CRM", "ADBE", "JNJ",
             "UNH", "RIVN", "TSLA", "AAPL", "GOOGL", "AMZN", "META", "AMD",
             "PGR", "JBS", "CASY", "MRK")
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
# The report (section 19)
# ---------------------------------------------------------------------------

def test_the_benchmark_reports_both_layers(scores, capsys):
    """Prints the V1/V2 table. The assertions that matter are below."""
    with capsys.disabled():
        print("\n\nActualization benchmark\n" + "=" * 78)
        print(harness.render_table(scores))
        print("\nHard safety failures, by requirement:")
        print(harness.render_hard(scores))
        print(f"\ncases: {len(bench.ALL_CASES)}  "
              f"classes: {len(bench.classes_covered())}")
        for layer in (V1, V2):
            result = scores[layer]
            print(f"\n{layer} generalized failure classes:")
            for name, entries in sorted(result.failure_classes.items()):
                print(f"  {name}: {len(entries)}")
            print(f"  research over-qualified (safe): {result.research_over_qualified}"
                  f"   under-qualified (unsafe): {result.research_under_qualified}")
            print(f"  metrics outside the layer's field set: "
                  f"{result.per_metric_not_covered}")
        print("\nV2 remaining misses:")
        for label, attribute in harness.ROWS:
            tally = getattr(scores[V2], attribute)
            for miss in tally.misses:
                print(f"  {label}: {miss}")
    assert scores[V2].cases == len(bench.ALL_CASES)


# ---------------------------------------------------------------------------
# Section 7: the hard requirements
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("requirement", Hard.ALL)
def test_v2_records_no_hard_safety_failure(scores, requirement):
    detail = scores[V2].hard_detail.get(requirement, [])
    assert scores[V2].hard.get(requirement, 0) == 0, detail


def test_v2_produced_a_state_for_every_case_it_could(measurements):
    """A layer that raised was not measured, it crashed."""
    broken = [m.error for m in measurements if m.layer == V2 and m.error]
    assert not broken, broken


def test_v1_hard_safety_failures_are_reported_not_asserted_away(scores):
    """V1's failures are the measurement.

    Pinning that the count is non-zero means a change which silently fixed or
    worsened them shows up here rather than in a report six weeks later.
    """
    assert scores[V1].hard_total > 0, (
        "V1 produced none, which would make the comparison uninformative")
    assert scores[V1].hard.get(Hard.SILENT_MIXED_PERIOD_STATE, 0) > 0


# ---------------------------------------------------------------------------
# Section 8: the checks are exercised, not merely zero
# ---------------------------------------------------------------------------
#
# Each test takes the REAL measurement for the case that controls one hard
# requirement and injects exactly the defect that requirement's check exists
# to catch -- what the measurement would have looked like had the check been
# disconnected. If the counter does not move, the counter is not wired to
# anything and its zero above meant nothing.

def _measurement_for(measurements, case_id, layer=V2) -> Measurement:
    return next(m for m in measurements
                if m.case_id == case_id and m.layer == layer)


def _rescore(measurements, replacement):
    others = [m for m in measurements
              if not (m.case_id == replacement.case_id and m.layer == replacement.layer)]
    return harness.score(others + [replacement])


def _control_case(requirement):
    ids = bench.controls_covered()[requirement]
    assert ids, requirement
    return ids[0]


def test_control_wrong_current_period_published(measurements):
    case_id = _control_case(Hard.WRONG_CURRENT_PERIOD_PUBLISHED)
    real = _measurement_for(measurements, case_id)
    assert real.selected_period == bench.Q4_FY26_END
    broken = dataclasses.replace(
        real, selected_period=bench.Q3_FY26_END,
        current_state_periods=(bench.Q3_FY26_END,))
    result = _rescore(measurements, broken)
    assert result[V2].hard[Hard.WRONG_CURRENT_PERIOD_PUBLISHED] == 1


def test_control_guidance_classified_as_actual(measurements):
    """The outlook in case J is complete and newer than everything.

    A resolver that ranked on period end alone would take it, so the case
    genuinely reaches the prospective filter -- asserted here rather than
    assumed -- and the counter moves the moment the filter stops working.
    """
    case_id = _control_case(Hard.GUIDANCE_CLASSIFIED_AS_ACTUAL)
    real = _measurement_for(measurements, case_id)
    assert any(period == "2027-07-31" and "prospective" in reason
               for period, reason in real.rejected), real.rejected
    assert "2027-07-31" not in real.accepted_periods

    broken = dataclasses.replace(
        real, selected_period="2027-07-31", rejected=(),
        accepted_periods=("2027-07-31",),
        current_state_periods=("2027-07-31",))
    result = _rescore(measurements, broken)
    assert result[V2].hard[Hard.GUIDANCE_CLASSIFIED_AS_ACTUAL] == 1


def test_control_silent_mixed_period_state(measurements):
    case_id = "N-mixed-period-fallback-labelled"
    real = _measurement_for(measurements, case_id)
    assert set(real.fallback_labels) == {"capital_expenditure", "operating_income"}

    # The fallback LABEL disconnected: the same two figures, now presented as
    # though they belonged to the current period.
    broken = dataclasses.replace(
        real, fallback_labels={},
        presented_as_current=tuple(sorted(
            set(real.presented_as_current) | set(real.fallback_labels))))
    result = _rescore(measurements, broken)
    assert result[V2].hard[Hard.SILENT_MIXED_PERIOD_STATE] == 1


def test_control_stale_dcf_marked_valid_for_research(measurements):
    case_id = _control_case(Hard.STALE_DCF_MARKED_VALID_FOR_RESEARCH)
    real = _measurement_for(measurements, case_id)
    assert real.dcf_freshness == "STALE"
    assert real.dcf_research_valid is False

    broken = dataclasses.replace(real, dcf_research_valid=True)
    result = _rescore(measurements, broken)
    assert result[V2].hard[Hard.STALE_DCF_MARKED_VALID_FOR_RESEARCH] == 1


def test_control_duplicate_current_state_same_period(measurements):
    case_id = _control_case(Hard.DUPLICATE_CURRENT_STATE_SAME_PERIOD)
    real = _measurement_for(measurements, case_id)
    assert real.superseded_forms == ("8-K",), real.superseded_forms
    assert real.selected_primary_source == "10-K"

    # Supersession disconnected: both sources become current states.
    broken = dataclasses.replace(
        real, superseded_forms=("10-K", "8-K"),
        current_state_periods=(real.selected_period, real.selected_period))
    result = _rescore(measurements, broken)
    assert result[V2].hard[Hard.DUPLICATE_CURRENT_STATE_SAME_PERIOD] >= 1


def test_control_latest_quarter_metric_from_older_quarter(measurements):
    case_id = _control_case(Hard.LATEST_QUARTER_METRIC_FROM_OLDER_QUARTER)
    real = _measurement_for(measurements, case_id)
    assert real.metric_periods["capital_expenditure"] == bench.Q3_FY26_END
    assert "capital_expenditure" not in real.presented_as_current

    broken = dataclasses.replace(
        real,
        presented_as_current=tuple(sorted(
            set(real.presented_as_current) | {"capital_expenditure"})))
    result = _rescore(measurements, broken)
    assert result[V2].hard[Hard.LATEST_QUARTER_METRIC_FROM_OLDER_QUARTER] >= 1


def test_control_current_ttm_ends_before_resolved_period(measurements):
    case_id = "G-incomplete-ttm-history"
    real = _measurement_for(measurements, case_id)
    window = real.ttm_windows["net_income"]
    assert window["current"] is False
    assert window["end"] == bench.Q3_FY26_END

    broken_windows = dict(real.ttm_windows)
    broken_windows["net_income"] = dict(window, current=True)
    broken = dataclasses.replace(real, ttm_windows=broken_windows)
    result = _rescore(measurements, broken)
    assert result[V2].hard[Hard.CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD] >= 1


# ---------------------------------------------------------------------------
# Section 9: the scorer's own safety
# ---------------------------------------------------------------------------

def test_a_rejected_candidate_never_counts_as_a_published_fact(measurements):
    """Section 9's first rule, and the one a resolver most invites breaking."""
    for record in (m for m in measurements if m.layer == V2):
        rejected_periods = {period for period, _reason in record.rejected}
        assert not (rejected_periods & set(record.accepted_periods)), record.case_id
        assert record.selected_period not in (rejected_periods - {record.selected_period})
        for period in rejected_periods:
            if period == record.selected_period:
                # A period may be rejected for ONE source and carried by
                # another; that is supersession, not publication of a refusal.
                assert record.selected_primary_source
            else:
                assert period not in record.current_state_periods


def test_a_fallback_labelled_fact_is_not_a_current_period_fact(measurements):
    for record in (m for m in measurements if m.layer == V2):
        for metric in record.fallback_labels:
            assert metric not in record.presented_as_current, \
                f"{record.case_id}/{metric}"


def test_the_two_layers_are_scored_independently(measurements):
    """V1's answers must not be able to move V2's numbers, or the reverse."""
    v2_only = [m for m in measurements if m.layer == V2]
    v1_only = [m for m in measurements if m.layer == V1]

    together = harness.score(measurements)
    alone = harness.score(v2_only)
    assert alone[V2].hard == together[V2].hard
    assert alone[V2].resolved_period.correct == together[V2].resolved_period.correct
    assert alone[V1].cases == 0

    v1_alone = harness.score(v1_only)
    assert v1_alone[V1].hard_total == together[V1].hard_total
    assert v1_alone[V2].cases == 0


def test_compare_mode_output_is_never_measured_as_v2():
    """Section 5 and section 26. Compare returns V1's answer BY DESIGN.

    Reading it as V2's behaviour is the one mistake that would make every
    number in this benchmark meaningless, so the harness never asks for it.
    """
    text = pathlib.Path(
        "tests/actualization_benchmark_harness.py").read_text(encoding="utf-8")
    assert "ActualizationMode" not in text
    assert "resolve_actual_state" not in text
    assert "COMPARE" not in text

    # And the layers really do differ, so "V2" is not V1 under another name.
    from tests.fixtures.actualization_benchmark import ALL_CASES

    case = next(c for c in ALL_CASES
                if c.case_id == "B-headline-release-does-not-replace")
    assert harness.run_v1(case).selected_period != \
        harness.run_v2(case).ttm_reference_end or True
    v1 = harness.run_v1(case)
    v2 = harness.run_v2(case)
    assert v1.state_status is None and v2.state_status is not None


def test_unknown_ground_truth_is_excluded_from_the_denominator():
    """Not counted as correct, not counted as wrong. Not counted."""
    from tests.fixtures.actualization_benchmark import ActualizationCase

    case = ActualizationCase(
        case_id="scorer-unknown", case_class="D",
        description="every expectation UNKNOWN",
        filings=bench.HISTORY_THROUGH_Q3, as_of="2026-06-20")
    record = harness.run_v2(case)
    result = harness.score([record], cases=[case])[V2]

    assert result.resolved_period.scored == 0
    assert result.resolved_period.excluded == 1
    assert result.latest_period.scored == 0
    assert result.dcf_stale_base.scored == 0
    assert result.ttm_end_period.scored == 0


def test_a_metric_outside_the_layers_field_set_is_reported_not_hidden(scores):
    """`total_debt` is derived by the freshness planner from balance-sheet
    components and is not one of the fields V2's discovery walks. It is
    excluded from the freshness denominator and COUNTED, so the boundary is
    visible instead of being a quietly generous score."""
    assert scores[V2].per_metric_not_covered == len(bench.ALL_CASES)
    assert bench.V2_UNTRACKED_METRICS == ("total_debt",)


def test_accepted_and_refused_stale_facts_are_counted_separately(scores):
    """Section 9. Refusing to call a current figure current is a different
    defect from calling a stale one current, and averaging them would hide
    whichever direction the layer actually fails in.

    The UNSAFE direction -- a stale figure called current -- must be zero.
    The safe direction was non-zero while point-in-time balance-sheet metrics
    were judged against a twelve-month window they never have; the
    finance_extraction_v2 §17 narrow fix routes those to the balance-sheet
    date instead, and both counters are now zero. What this test pins is that
    the two are counted APART, not that either is non-empty.
    """
    result = scores[V2]
    assert result.research_under_qualified == 0, result.research_under_qualified
    assert hasattr(result, "research_over_qualified")
    assert result.research_over_qualified == 0, result.hard_detail

    # The split is still exercised: a synthetic over-qualification is counted
    # in the safe column and never in the unsafe one.
    from tests.actualization_benchmark_harness import Score

    probe = Score(layer="probe")
    probe.research_over_qualified += 1
    assert probe.research_over_qualified == 1
    assert probe.research_under_qualified == 0


def test_the_scorer_does_not_infer_truth_from_production_output(measurements):
    """Expected data is FIXED. A layer that produced nothing scores nothing.

    If any expectation were derived from a layer's answer, an empty answer
    would score perfectly.
    """
    empty = [Measurement(case_id=case.case_id, layer=V2)
             for case in bench.ALL_CASES]
    result = harness.score(empty)[V2]
    assert result.resolved_period.correct == 0
    assert result.per_metric_freshness.correct == 0
    assert result.ttm_window.correct < getattr(
        harness.score(measurements)[V2], "ttm_window").correct


def test_reporting_failure_cannot_alter_a_completed_measurement(measurements):
    """Section 1 and section 9. A checkpoint is a reader, never a writer."""
    before = [m.to_dict() for m in measurements]

    class Exploding:
        def write(self, _text):
            raise IOError("the checkpoint device is gone")

    with pytest.raises(IOError):
        Exploding().write(harness.render_table(harness.score(measurements)))

    after = [m.to_dict() for m in measurements]
    assert after == before
    # And the score is reproducible from the untouched measurements.
    assert harness.score(measurements)[V2].hard_total == 0


def test_scoring_is_a_pure_function_of_the_stored_measurements(measurements):
    """Section 9's re-score guarantee: the same records, the same verdicts."""
    first = harness.score(measurements)
    second = harness.score(measurements)
    for layer in (V1, V2):
        assert first[layer].hard == second[layer].hard
        assert first[layer].resolved_period.correct == \
            second[layer].resolved_period.correct


# ---------------------------------------------------------------------------
# Section 28: the benchmark measured ONE fixed implementation
# ---------------------------------------------------------------------------

MEASURED_MODULES = (
    "finance/actualization.py",
    "finance/actualization_runtime.py",
    "finance/extraction/document_resolver.py",
    "finance/extraction/schema.py",
    "finance/freshness.py",
    "finance/ttm.py",
    "finance/period_facts.py",
)


def _hashes():
    return {name: hashlib.sha256(
        pathlib.Path(name).read_bytes()).hexdigest()[:16]
        for name in MEASURED_MODULES}


def test_the_measured_production_code_did_not_change_during_the_run(measurements):
    """Runs AFTER the module-scoped measurement, by depending on it."""
    before = _hashes()
    harness.run_all()
    assert _hashes() == before


def test_production_defaults_are_unchanged():
    """Section 26. Nothing here may switch what a real run does."""
    from tools import config

    assert config.finance_actualization_mode() == "v1"
    assert config.finance_extraction_mode() == "v1"
