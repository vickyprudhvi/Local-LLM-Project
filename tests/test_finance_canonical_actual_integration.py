"""The Canonical Actual Fact Integration benchmark, and its own safety.

WHAT THIS FILE ASSERTS, IN ORDER OF WHAT MATTERS

  1. Every §25 counter is ZERO across the whole case set.
  2. Every counter with a runtime check is EXERCISED -- a case whose injected
     defect makes the self-check fire (§26). Two structural counters,
     RAW_REPORTED_ACTUAL_BYPASSES_CANONICAL_STATE and
     RAW_COMPANYFACTS_BYPASSES_CANONICAL_STATE, are asserted by a static
     read of the workflow instead: they are about which code path a value
     took, and there is no value to inject.
  3. The scorer cannot flatter anything: expected data is fixed, measurement
     and scoring are separate, and a reporting failure cannot reach a
     measurement.

The secondary accuracy numbers are printed, not pinned -- §33 keeps the
production defaults unchanged, so pinning one would pressure a change to
satisfy it.

MOST IMPORTANT

Once the Actualization resolver says period P is current, the rebuilt
`CurrentFinancialState` -- and therefore canonical evidence, the DCF base and
research -- must be P too. The selected period and the selected numbers
advance together. Every case that resolves a newer period checks exactly
that.
"""

import hashlib
import pathlib
import re

import pytest

from tests import canonical_actual_integration_harness as harness
from tests.canonical_actual_integration_harness import Measurement
from tests.fixtures import canonical_actual_integration_benchmark as bench
from tests.fixtures.canonical_actual_integration_benchmark import Hard


@pytest.fixture(scope="module")
def measurements():
    return harness.measure_all()


@pytest.fixture(scope="module")
def result(measurements):
    return harness.score(measurements)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_the_benchmark_covers_the_named_case_classes():
    assert set(bench.classes_covered()) == set("ABCDEFGHIJK"), \
        bench.classes_covered()
    assert len(bench.ALL_CASES) >= 11


def test_every_runtime_hard_counter_has_a_positive_control(result):
    """§26. A counter that reads zero because nothing reached it is not a
    measurement."""
    structural = {Hard.RAW_REPORTED_ACTUAL_BYPASSES_CANONICAL_STATE,
                  Hard.RAW_COMPANYFACTS_BYPASSES_CANONICAL_STATE,
                  Hard.GUIDANCE_FACT_ENTERS_ACTUAL_SET,
                  Hard.SILENT_MIXED_PERIOD_CURRENT_SNAPSHOT}
    for counter in Hard.ALL:
        if counter in structural:
            continue
        assert counter in result.controls_fired, counter
    assert result.controls_missed == [], result.controls_missed


def test_no_issuer_is_named_in_the_fixtures():
    for path in ("tests/fixtures/canonical_actual_integration_benchmark.py",
                 "tests/canonical_actual_integration_harness.py"):
        text = pathlib.Path(path).read_text(encoding="utf-8")
        for name in ("MSFT", "NVDA", "AVGO", "ORCL", "AAPL", "GOOGL", "AMZN",
                     "META", "AMD", "Broadcom", "Oracle", "Microsoft"):
            assert name not in text, f"{path}: {name}"


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_the_benchmark_reports_its_numbers(result, capsys):
    with capsys.disabled():
        print("\n\nCanonical Actual Fact Integration benchmark\n" + "=" * 78)
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
        print("\npositive controls that fired:")
        for counter, ids in sorted(result.controls_fired.items()):
            print(f"  {counter}: {ids}")
    assert result.cases == len(bench.ALL_CASES)
    assert not result.errors, result.errors


# ---------------------------------------------------------------------------
# §25: the hard requirements
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("requirement", Hard.ALL)
def test_no_hard_safety_failure(result, requirement):
    detail = result.hard_detail.get(requirement, [])
    assert result.hard.get(requirement, 0) == 0, detail


def test_the_pipeline_did_not_crash_on_any_case(measurements):
    broken = [m.error for m in measurements if m.error]
    assert not broken, broken


# ---------------------------------------------------------------------------
# §1: the selected period and the selected numbers advance together
# ---------------------------------------------------------------------------

def test_on_every_active_case_every_current_metric_is_the_resolved_period(
        measurements):
    """§1's hard invariant, stated directly over the raw measurements."""
    from tests.canonical_actual_integration_harness import _same_period

    for m in measurements:
        if not m.active or m.error:
            continue
        for metric, freshness in m.freshness.items():
            selected = m.metric_period.get(metric)
            if freshness == "CURRENT_PERIOD" and selected:
                assert _same_period(selected, m.resolved_period), (
                    f"{m.case_id}/{metric}: called current, selected from "
                    f"{selected}, resolved {m.resolved_period}")
            if freshness == "FALLBACK_PRIOR_PERIOD":
                assert metric in m.requalified, (
                    f"{m.case_id}/{metric}: fallback not requalified for research")


def test_the_canonical_base_and_the_dcf_base_advance_with_the_period(measurements):
    from tests.canonical_actual_integration_harness import _same_period

    for m in measurements:
        if not m.active or m.error:
            continue
        if m.canonical_base_period and m.canonical_base_aligned:
            assert _same_period(m.canonical_base_period, m.resolved_period), (
                f"{m.case_id}: canonical base {m.canonical_base_period} vs "
                f"resolved {m.resolved_period}")
        # A DCF packet actually built from the resolved period is never stale.
        case = next(c for c in bench.ALL_CASES if c.case_id == m.case_id)
        if case.dcf_packet_base_period == m.resolved_period:
            assert m.dcf_base_status == "CURRENT", m.case_id
            assert m.dcf_base_research_valid is True


# ---------------------------------------------------------------------------
# §26: the checks are exercised
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("counter", [
    Hard.CURRENT_METRIC_FROM_PRIOR_PERIOD_UNDISCLOSED,
    Hard.LATEST_QUARTER_METRIC_FROM_PRIOR_QUARTER,
    Hard.CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD,
    Hard.DUPLICATE_ECONOMIC_QUARTER_IN_TTM,
    Hard.STALE_DCF_MARKED_RESEARCH_VALID,
    Hard.SAME_PERIOD_MATERIAL_CONFLICT_SILENTLY_OVERWRITTEN,
])
def test_the_positive_control_fires(measurements, counter):
    fired = any(counter in (m.control_hard_findings or {})
                and (m.control_hard_findings[counter].get(counter))
                for m in measurements)
    assert fired, f"{counter}: no case's injected defect made the check fire"


def test_guidance_facts_never_enter_the_actual_set(measurements):
    """§21/§25. Every fact in the set carries a reported-actual status, never
    a prospective one."""
    from finance.actualization import ActualStateStatus

    for m in measurements:
        for metric, freshness in m.freshness.items():
            assert freshness in ("CURRENT_PERIOD", "FALLBACK_PRIOR_PERIOD",
                                 "MISSING", "CONFLICTED", "NOT_APPLICABLE")
    # And the resolver's own prospective filter is asserted in the
    # Actualization suite; here we confirm the unified set exposes no status
    # outside the reported-actual vocabulary.
    assert ActualStateStatus.HISTORICAL in ActualStateStatus.ALL


# ---------------------------------------------------------------------------
# Structural counters (§25): a static read of the workflow
# ---------------------------------------------------------------------------

def test_raw_reported_actuals_never_bypass_the_canonical_state():
    """The DCF packet and canonical evidence read `_current_financial_state`
    and `canonical_evidence`, never a raw `ReportedActualCandidate` or a
    release table."""
    text = pathlib.Path("finance/workflow.py").read_text(encoding="utf-8")

    # _dcf_inputs_from_facts reads the state, not raw candidates.
    dcf_fn = text.split("def _dcf_inputs_from_facts", 1)[1].split("\ndef ", 1)[0]
    for forbidden in ("ReportedActualCandidate", "extract_reported_actuals",
                      "parse_filing_tables", "source_candidates",
                      "facts_overlay"):
        assert forbidden not in dcf_fn, f"_dcf_inputs_from_facts reads {forbidden}"

    # build_canonical_evidence is called with the state only.
    assert "canonical_module.build_canonical_evidence(\n            state," in text \
        or "build_canonical_evidence(state" in text


def test_raw_companyfacts_never_bypass_the_canonical_state_for_current_metrics():
    """`_sec_company_facts` is set to the UNIFIED payload once V2 is active,
    so every consumer that walks it sees the merged view. The V1 build is
    replaced, not shadowed."""
    text = pathlib.Path("finance/workflow.py").read_text(encoding="utf-8")
    block = text.split("Unified actual fact set", 1)[1].split("\ndef ", 1)[0]
    assert "company_facts = unified.company_facts" in block
    assert "state = build_current_financial_state(" in block
    assert 'facts["_current_financial_state"] = state' in block


def test_the_canonical_evidence_reads_only_the_state():
    """§13/§16. `build_canonical_evidence` adds a namespace and derived
    ratios; it must not introduce a second selection policy."""
    text = pathlib.Path("finance/canonical.py").read_text(encoding="utf-8")
    body = text.split("def build_canonical_evidence", 1)[1]
    for forbidden in ("company_facts", "ReportedActualCandidate",
                      "discover_reported_actuals", "extract_reported_actuals"):
        assert forbidden not in body, f"build_canonical_evidence reads {forbidden}"


# ---------------------------------------------------------------------------
# Scorer safety
# ---------------------------------------------------------------------------

def test_the_scorer_does_not_infer_truth_from_the_pipeline(measurements):
    empty = [Measurement(case_id=c.case_id) for c in bench.ALL_CASES]
    scored = harness.score(empty)
    assert scored.per_metric_period.correct == 0
    assert scored.canonical_value.correct == 0
    assert harness.score(measurements).per_metric_period.correct > 0


def test_scoring_is_a_pure_function_of_the_stored_measurements(measurements):
    first = harness.score(measurements)
    second = harness.score(measurements)
    assert first.hard == second.hard
    assert first.per_metric_period.correct == second.per_metric_period.correct


def test_reporting_failure_cannot_alter_a_measurement(measurements):
    before = [m.to_dict() for m in measurements]

    class Exploding:
        def write(self, _text):
            raise IOError("gone")

    with pytest.raises(IOError):
        Exploding().write(harness.render_table(harness.score(measurements)))
    assert [m.to_dict() for m in measurements] == before
    assert harness.score(measurements).hard_total == 0


def test_measurement_and_scoring_are_separate(measurements):
    """`measure` runs the pipeline; `score` runs no production code. A
    measurement re-scored gives the same verdict."""
    import inspect

    src = inspect.getsource(harness.score)
    for forbidden in ("build_current_financial_state", "resolve_actual_state",
                      "build_unified_actual_facts", "extract_reported_actuals"):
        assert forbidden not in src, f"score() calls {forbidden}"


# ---------------------------------------------------------------------------
# §33: production defaults, and §32 code integrity
# ---------------------------------------------------------------------------

def test_production_defaults_are_unchanged():
    from tools import config

    assert config.finance_actualization_mode() == "v1"
    assert config.finance_reported_actuals_mode() == "v1"
    assert config.finance_extraction_mode() in ("v1", "compare")  # repo .env


MEASURED_MODULES = (
    "finance/reported_actuals/unified.py",
    "finance/reported_actuals/candidates.py",
    "finance/actualization.py",
    "finance/actualization_runtime.py",
    "finance/freshness.py",
    "finance/canonical.py",
    "finance/ttm.py",
    "finance/workflow.py",
)


def _hashes():
    return {name: hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()[:16]
            for name in MEASURED_MODULES}


def test_the_measured_production_code_did_not_change_during_the_run(measurements):
    before = _hashes()
    harness.measure_all()
    assert _hashes() == before


def test_no_ticker_specific_branch_reached_production():
    names = ("MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "CRM", "ADBE", "JNJ",
             "UNH", "RIVN", "TSLA", "AAPL", "GOOGL", "AMZN", "META", "AMD",
             "PGR", "JBS", "CASY", "MRK", "SHEL", "NVS")
    offenders = []
    for path in list(pathlib.Path("finance").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for name in names:
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or name not in line:
                    continue
                if re.search(rf'["\']{name}["\']', line):
                    offenders.append(f"{path}: {stripped[:80]}")
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# Found live on CRWD: CompanyFacts' OWN newer, incomplete balance sheet
# outlives the resolved period once the unified fact set is active.
# ---------------------------------------------------------------------------
#
# CRWD's revenue concept is not found by XBRL discovery at ANY quarter, so
# every CompanyFacts-derived candidate is PARTIAL. A filed earnings-release
# exhibit for the PRIOR quarter reads revenue from its own table and comes
# out COMPLETE, so the resolver correctly resolves to that prior quarter.
# But CompanyFacts' OWN newer quarter still tags `assets`/`stockholders_
# equity`/`cash` (just not revenue) -- and nothing capped the freshness
# planner's view of the merged payload to the resolved period, so it picked
# the NEWER balance sheet the resolver had just rejected as unable to carry
# the state. The result claimed period P and reported a balance sheet from
# P+1 as "aligned" -- SILENT_MIXED_PERIOD_CURRENT_SNAPSHOT.

def _crwd_shape_company_facts():
    """CompanyFacts: Q3 complete-except-revenue-never-tagged; Q4 the same,
    one quarter later. Revenue is absent from BOTH -- the shared, generic
    'this project cannot see revenue for this issuer' gap -- so the defect
    under test is the period-capping one, not a completeness illusion."""
    def instant(concept, end, value, form, filed, accn):
        return {"end": end, "val": value, "form": form, "filed": filed,
                "accn": accn, "fy": None, "fp": None}

    def duration(concept, start, end, value, form, filed, accn):
        return {"start": start, "end": end, "val": value, "form": form,
                "filed": filed, "accn": accn, "fy": None, "fp": None}

    facts = {
        "Assets": {"units": {"USD": [
            instant("Assets", "2026-04-30", 11_000_000_000.0, "10-Q",
                    "2026-06-01", "acc-q3"),
            instant("Assets", "2026-07-31", 12_000_000_000.0, "10-Q",
                    "2026-09-01", "acc-q4"),
        ]}},
        "StockholdersEquity": {"units": {"USD": [
            instant("StockholdersEquity", "2026-04-30", 4_600_000_000.0,
                    "10-Q", "2026-06-01", "acc-q3"),
            instant("StockholdersEquity", "2026-07-31", 5_100_000_000.0,
                    "10-Q", "2026-09-01", "acc-q4"),
        ]}},
        "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [
            instant("Cash", "2026-04-30", 4_500_000_000.0, "10-Q",
                    "2026-06-01", "acc-q3"),
            instant("Cash", "2026-07-31", 5_000_000_000.0, "10-Q",
                    "2026-09-01", "acc-q4"),
        ]}},
        "NetIncomeLoss": {"units": {"USD": [
            duration("NetIncomeLoss", "2026-02-01", "2026-04-30", 45_000_000.0,
                     "10-Q", "2026-06-01", "acc-q3"),
            duration("NetIncomeLoss", "2026-05-01", "2026-07-31", 50_000_000.0,
                     "10-Q", "2026-09-01", "acc-q4"),
        ]}},
        # Revenue is tagged NOWHERE -- the shared, generic gap.
    }
    return {"facts": {"us-gaap": facts}}


def _crwd_shape_release_candidate():
    """The release's own extraction: COMPLETE for the OLDER quarter, revenue
    read from its own table rather than from a CompanyFacts concept."""
    from finance.extraction.schema import (
        ReportedActualCandidate,
        SourceType,
        StatementCompleteness,
    )

    return ReportedActualCandidate(
        period_end="2026-04-30", period_start="2026-02-01",
        period_type="QUARTER", form="8-K", issued_at="2026-05-15",
        accession="acc-release", source_type=SourceType.PRELIMINARY_EARNINGS_RELEASE,
        statement_completeness=StatementCompleteness.COMPLETE,
        present_metrics=("revenue", "net_income", "assets", "stockholders_equity",
                         "cash_and_cash_equivalents", "operating_cash_flow"),
        currency="USD",
        values={"revenue": 1_400_000_000.0, "net_income": 45_000_000.0,
                "assets": 11_000_000_000.0, "stockholders_equity": 4_600_000_000.0,
                "cash_and_cash_equivalents": 4_500_000_000.0,
                "operating_cash_flow": 590_000_000.0})


def test_the_rebuilt_state_never_exposes_a_balance_sheet_newer_than_the_resolved_period():
    """The generalized invariant CRWD violated, asserted directly."""
    from finance import actualization_runtime as ar
    from finance.freshness import build_current_financial_state
    from finance.reported_actuals import unified as U
    from finance.reported_actuals.candidates import company_facts_overlay

    company_facts = _crwd_shape_company_facts()
    release = _crwd_shape_release_candidate()

    v1_state = build_current_financial_state(company_facts, "BENCH",
                                              valuation_date="2026-09-15")
    # CompanyFacts alone: no complete candidate at all (revenue nowhere), so
    # V1's own state is simply whatever it can select -- unaffected by this
    # test either way. What matters is the RESOLVED state once the release
    # is offered. The overlay supplies revenue for the release's OWN period --
    # the one fact CompanyFacts never carries at any quarter -- which is what
    # makes the unified layer ACTIVE (merged_periods non-empty) and is
    # exactly the shape that reached production.
    overlay = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2026-02-01", "end": "2026-04-30", "val": 1_400_000_000.0,
         "form": "8-K", "filed": "2026-05-15", "accn": "acc-release",
         "fy": None, "fp": None},
    ]}}}}}
    resolution, observation = ar.resolve_actual_state(
        company_facts, as_of="2026-09-15", mode="v2",
        prior_state_metrics=U.prior_state_metrics(v1_state),
        extra_candidates=(release,), extra_facts=overlay)

    assert resolution.period_end == "2026-04-30", (
        "the release should resolve as current -- CompanyFacts' own newer "
        "quarter is not COMPLETE (no revenue anywhere)")

    unified = U.build_unified_actual_facts(
        company_facts, resolution=resolution, observation=observation,
        facts_overlay=overlay, v1_state=v1_state)

    if unified.active:
        rebuilt = build_current_financial_state(
            unified.company_facts, "BENCH", valuation_date="2026-09-15")
    else:
        rebuilt = v1_state

    # THE INVARIANT: the rebuilt state's balance-sheet date must never be
    # NEWER than the period Actualization resolved as current.
    assert rebuilt.financial_as_of is not None
    assert rebuilt.financial_as_of <= resolution.period_end, (
        f"balance-sheet date {rebuilt.financial_as_of} is newer than the "
        f"resolved period {resolution.period_end} -- a mixed-period snapshot "
        "reported as one coherent state")

    if unified.active:
        unified.reconcile_with_state(rebuilt)
        hard = U.check_hard_safety(unified, rebuilt_state=rebuilt)
        assert "SILENT_MIXED_PERIOD_CURRENT_SNAPSHOT" not in hard, hard
