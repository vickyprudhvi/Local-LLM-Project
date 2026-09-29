"""The actualization seam: which layer decides the current period, and what it may decide.

The risk is not that V2 resolves badly -- the fixtures cover that -- but that
it becomes the production answer by accident. The accidents are specific:
a default that drifts, a compare mode that quietly prefers the newer resolver,
a v2 mode that falls back to v1 and reports v1's period as V2's.

The current reported period is the answer everything downstream inherits:
revenue, margins, debt, latest-quarter growth, the DCF base, the research
claims. A wrong one is wrong everywhere at once.
"""

import json

import pytest

from finance import actualization_runtime as runtime
from finance.actualization import ActualStateStatus
from finance.actualization_runtime import (
    ActualizationFailure,
    ActualizationMode,
)
from tools import config

# XBRL-shaped facts: a complete Q3, then a complete FY reported on an 8-K.
FULL_FIELDS = {
    "Revenues": "revenue",
    "OperatingIncomeLoss": "operating_income",
    "NetIncomeLoss": "net_income",
    "CashAndCashEquivalentsAtCarryingValue": "cash_and_cash_equivalents",
    "StockholdersEquity": "stockholders_equity",
    "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capital_expenditure",
    # Required for COMPLETE by `REQUIRED_FOR_COMPLETE`; without them the
    # candidate is PARTIAL and cannot advance the state, which is correct
    # behaviour and would make this fixture test the wrong thing.
    "Assets": "assets",
    "Liabilities": "liabilities",
}


def _row(start, end, val, form, accn):
    return {"start": start, "end": end, "val": val, "form": form,
            "accn": accn, "fy": int(end[:4]), "fp": "Q3", "filed": end}


def _facts_with_release():
    concepts = {}
    for concept in FULL_FIELDS:
        concepts[concept] = [
            _row("2026-02-01", "2026-04-30", 100, "10-Q", "acc-q3"),
            _row("2025-08-01", "2026-07-31", 420, "8-K", "acc-fy"),
        ]
    return {"facts": {"us-gaap": {c: {"units": {"USD": r}}
                                  for c, r in concepts.items()}}}


AS_OF = "2026-11-01"


# ---------------------------------------------------------------------------
# The default
# ---------------------------------------------------------------------------

def test_the_shipped_default_is_v1(monkeypatch):
    """The one line that decides whether V2 shipped by accident."""
    monkeypatch.delenv("FINANCE_ACTUALIZATION_MODE", raising=False)
    assert config.finance_actualization_mode() == "v1"


def test_an_unrecognised_mode_falls_back_to_v1(monkeypatch):
    monkeypatch.setenv("FINANCE_ACTUALIZATION_MODE", "V2!")
    assert config.finance_actualization_mode() == "v1"


def test_v1_does_no_work_at_all():
    """Not merely "v1 wins" -- the V2 resolver must not run.

    A default that resolves a second time and discards the answer costs the
    work on every ordinary run, and gives a second decision the chance to
    diverge unnoticed.
    """
    resolution, observation = runtime.resolve_actual_state(
        _facts_with_release(), v1_period_end="2026-04-30", as_of=AS_OF,
        mode=ActualizationMode.V1)
    assert resolution is None
    assert observation.layer_used == ActualizationMode.V1
    assert observation.v2_period_end is None
    assert "v2_period_end" not in observation.to_dict()


def test_v1_observation_carries_nothing_about_v2():
    _resolution, observation = runtime.resolve_actual_state(
        _facts_with_release(), v1_period_end="2026-04-30", as_of=AS_OF,
        mode=ActualizationMode.V1)
    payload = observation.to_dict()
    assert payload["mode"] == "v1"
    assert set(payload) == {"mode", "layer_used", "v1_period_end",
                            "v1_primary_source"}


# ---------------------------------------------------------------------------
# compare records; it does not resolve
# ---------------------------------------------------------------------------

def test_compare_runs_v2_but_leaves_the_answer_to_v1():
    resolution, observation = runtime.resolve_actual_state(
        _facts_with_release(), v1_period_end="2026-04-30", as_of=AS_OF,
        mode=ActualizationMode.COMPARE)
    assert resolution is not None, "V2 did not run"
    assert observation.layer_used == ActualizationMode.V1, (
        "compare mode preferred V2")


def test_compare_records_the_disagreement_it_found():
    _resolution, observation = runtime.resolve_actual_state(
        _facts_with_release(), v1_period_end="2026-04-30", as_of=AS_OF,
        mode=ActualizationMode.COMPARE)
    payload = observation.to_dict()
    assert payload["v1_period_end"] == "2026-04-30"
    assert payload["v2_period_end"]
    if payload["v2_period_end"] != payload["v1_period_end"]:
        assert payload["period_disagreement"] is True


def test_compare_diagnostics_are_metadata_only():
    """§6: structured fields, never filings or documents."""
    _resolution, observation = runtime.resolve_actual_state(
        _facts_with_release(), v1_period_end="2026-04-30", as_of=AS_OF,
        mode=ActualizationMode.COMPARE)
    blob = json.dumps(observation.to_dict())
    for leaked in ("us-gaap", "Revenues", "document_text", "<"):
        assert leaked not in blob
    for expected in ("v2_state_status", "v2_codes", "fallback_metrics",
                     "ttm_status"):
        assert expected in blob


def test_compare_reports_the_ttm_window_against_the_resolved_period():
    _resolution, observation = runtime.resolve_actual_state(
        _facts_with_release(), v1_period_end="2026-04-30", as_of=AS_OF,
        mode=ActualizationMode.COMPARE)
    if observation.v2_period_end:
        assert observation.ttm_reference_end == observation.v2_period_end


# ---------------------------------------------------------------------------
# v2 fails closed
# ---------------------------------------------------------------------------

def test_v2_raises_rather_than_falling_back_when_nothing_resolves():
    """The failure that would be easiest to hide.

    Falling back to v1 here would make the two layers disagree silently and
    let the report carry whichever answered last.
    """
    with pytest.raises(ActualizationFailure) as caught:
        runtime.resolve_actual_state({}, v1_period_end="2026-04-30",
                                     as_of=AS_OF, mode=ActualizationMode.V2)
    assert caught.value.code


def test_v2_returns_its_own_resolution_when_it_can():
    resolution, observation = runtime.resolve_actual_state(
        _facts_with_release(), v1_period_end="2026-04-30", as_of=AS_OF,
        mode=ActualizationMode.V2)
    assert resolution is not None
    assert observation.layer_used == ActualizationMode.V2
    assert resolution.state_status in ActualStateStatus.CURRENT


def test_an_unknown_mode_is_diagnosed_as_a_typo_not_as_a_data_problem():
    resolution, observation = runtime.resolve_actual_state(
        _facts_with_release(), v1_period_end="2026-04-30", as_of=AS_OF,
        mode="v3-experimental")
    assert resolution is None
    assert observation.failure_code == "UNKNOWN_MODE"
    assert observation.layer_used == ActualizationMode.V1


# ---------------------------------------------------------------------------
# One decision, one place (§5)
# ---------------------------------------------------------------------------

def test_only_the_workflow_calls_the_seam():
    """§5: the renderer, research pipeline and DCF must not each decide.

    Checked as source text, because the failure mode is someone adding a
    second call where it is convenient.
    """
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    callers = []
    for folder, _dirs, files in os.walk(os.path.join(root, "finance")):
        for name in files:
            if not name.endswith(".py") or name.startswith("actualization"):
                continue
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8") as handle:
                if "resolve_actual_state(" in handle.read():
                    callers.append(os.path.basename(path))
    assert callers == ["workflow.py"], callers


def test_the_seam_never_sees_a_ticker():
    import inspect

    source = inspect.getsource(runtime)
    for forbidden in ("ticker", "cik"):
        assert forbidden not in source.lower()


def test_the_production_payload_is_untouched_under_v1(monkeypatch):
    """§4: v1 behaviour unchanged, asserted at the shape the workflow attaches."""
    monkeypatch.setattr(config, "finance_actualization_mode", lambda: "v1")
    resolution, observation = runtime.resolve_actual_state(
        _facts_with_release(), v1_period_end="2026-04-30", as_of=AS_OF)
    attaches = observation.mode != ActualizationMode.V1 or observation.failure_code
    assert not attaches, "v1 would have attached a diagnostics key"
