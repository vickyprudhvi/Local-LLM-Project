"""§9: research evidence may only be called current when it is.

THE FAILURE

A research role reads `current.revenue_growth`, and writes "current revenue
growth is X" when X belongs to the previous quarter. Nothing was wrong with
the figure; the namespace it sat in asserted a currency its provenance did not
support.

WHY NOT PROSE INSPECTION

Checking the sentence afterwards is checking the symptom. The role wrote
"current" because it was handed something labelled current, and any wording
guard is one rephrasing away from being wrong. So the LABEL is corrected at
the single boundary where roles receive facts -- a role cannot misuse a claim
it was never given, the same guarantee `build_valuation_research_evidence`
relies on.

WHY REQUALIFY RATHER THAN DROP

A prior-quarter figure is real and a role may legitimately need it. Dropping
it loses information; leaving it unqualified is the failure. Naming its period
is the only option that does neither.
"""

import dataclasses

import pytest

from finance.actualization import (
    CURRENT_EVIDENCE_PREFIX,
    FreshnessEligibility,
    build_research_freshness_view,
    reconstruct_ttm,
    requalify_evidence_label,
    resolve_current_actual_state,
)
from finance.evidence import EvidenceItem, apply_research_freshness
from finance.extraction.schema import (
    ReportedActualCandidate,
    SourceType,
    StatementCompleteness,
)

FULL = ("revenue", "operating_income", "net_income", "cash_and_cash_equivalents",
        "total_debt", "stockholders_equity", "operating_cash_flow",
        "capital_expenditure", "shares_diluted")
TODAY = "2026-11-01"
CURRENT_END = "2026-07-31"
PRIOR_END = "2026-04-30"


def _actual(period_end, form, issued_at, present=FULL,
            completeness=StatementCompleteness.COMPLETE):
    return ReportedActualCandidate(
        period_end=period_end, form=form, issued_at=issued_at,
        fiscal_period="FY", statement_completeness=completeness,
        present_metrics=tuple(present),
        source_type=(SourceType.FORMAL_PERIODIC_FILING if form != "8-K"
                     else SourceType.PRELIMINARY_EARNINGS_RELEASE))


def _resolution_with_fallback():
    """Current period reports revenue only; total_debt falls back."""
    return resolve_current_actual_state(
        [_actual(CURRENT_END, "8-K", "2026-09-02", present=("revenue",))],
        as_of=TODAY,
        prior_state_metrics={"total_debt": {"period_end": PRIOR_END,
                                            "form": "10-Q"}})


# ---------------------------------------------------------------------------
# The view
# ---------------------------------------------------------------------------

def test_a_current_period_metric_may_keep_its_label():
    view = build_research_freshness_view(_resolution_with_fallback())
    assert "revenue" in view.may_claim_current
    assert "revenue" not in view.requalified


def test_a_fallback_metric_must_be_qualified():
    view = build_research_freshness_view(_resolution_with_fallback())
    assert "total_debt" in view.must_be_qualified
    assert view.requalified["total_debt"] == PRIOR_END
    assert "RESEARCH_EVIDENCE_REQUALIFIED" in view.codes


def test_every_metric_lands_in_exactly_one_bucket():
    view = build_research_freshness_view(_resolution_with_fallback())
    overlap = set(view.may_claim_current) & set(view.must_be_qualified)
    assert not overlap
    assert set(view.may_claim_current) | set(view.must_be_qualified) == set(view.eligibility)


def test_the_view_reports_the_period_it_judged_against():
    view = build_research_freshness_view(_resolution_with_fallback())
    assert view.current_period_end == CURRENT_END
    payload = view.to_dict()
    assert payload["current_period_end"] == CURRENT_END
    assert payload["eligibility"]["total_debt"]["may_be_called_current"] is False


# ---------------------------------------------------------------------------
# Label requalification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", [
    "Current revenue growth", "current revenue growth",
    "Latest revenue growth", "latest revenue growth",
])
def test_the_word_current_is_removed_not_decorated(label):
    """"Current revenue (as of 2026-04-30)" still reads as current at a glance.

    A role skimming for the latest figure would take it, which is the whole
    failure repeated with extra words.
    """
    result = requalify_evidence_label(label, PRIOR_END)
    assert "current" not in result.lower()
    assert "latest" not in result.lower()
    assert PRIOR_END in result


def test_the_metric_survives_requalification():
    result = requalify_evidence_label("Current revenue growth", PRIOR_END)
    assert "revenue growth" in result


# ---------------------------------------------------------------------------
# Applying it to the evidence index
# ---------------------------------------------------------------------------

def _index():
    return {
        "current.revenue": EvidenceItem(
            evidence_id="current.revenue", label="Current revenue", value=100.0),
        "current.total_debt": EvidenceItem(
            evidence_id="current.total_debt", label="Current total debt",
            value=50.0),
        "fundamental.revenue": EvidenceItem(
            evidence_id="fundamental.revenue", label="Annual revenue",
            value=90.0),
    }


def test_a_stale_current_item_is_requalified_in_place():
    index = _index()
    view = build_research_freshness_view(_resolution_with_fallback())
    changed = apply_research_freshness(index, view.to_dict())

    assert changed == ["current.total_debt"]
    item = index["current.total_debt"]
    assert "current" not in item.label.lower()
    assert PRIOR_END in item.label
    assert item.source_periods == [PRIOR_END]


def test_the_requalified_item_keeps_its_value():
    index = _index()
    view = build_research_freshness_view(_resolution_with_fallback())
    apply_research_freshness(index, view.to_dict())
    assert index["current.total_debt"].value == 50.0


def test_a_current_item_is_left_alone():
    index = _index()
    before = index["current.revenue"]
    view = build_research_freshness_view(_resolution_with_fallback())
    apply_research_freshness(index, view.to_dict())
    assert index["current.revenue"] == before


def test_items_outside_the_current_namespace_are_untouched():
    """Only the current namespace asserts currency by being what it is."""
    index = _index()
    before = index["fundamental.revenue"]
    view = build_research_freshness_view(_resolution_with_fallback())
    apply_research_freshness(index, view.to_dict())
    assert index["fundamental.revenue"] == before


def test_no_freshness_view_means_no_change():
    """v1 attaches nothing, so the index must be exactly what it was."""
    index = _index()
    snapshot = dict(index)
    assert apply_research_freshness(index, None) == []
    assert apply_research_freshness(index, {}) == []
    assert index == snapshot


def test_the_namespace_prefix_is_shared_not_restated():
    assert CURRENT_EVIDENCE_PREFIX == "current."


# ---------------------------------------------------------------------------
# TTM claims (§9's second example)
# ---------------------------------------------------------------------------

REVENUE = "Revenues"
Q_ENDS = [("2025-04-01", "2025-06-30"), ("2025-07-01", "2025-09-30"),
          ("2025-10-01", "2025-12-31"), ("2026-01-01", "2026-03-31"),
          ("2026-04-01", "2026-06-30")]


def _facts():
    rows = [{"start": s, "end": e, "val": 100 + i * 10, "form": "10-Q",
             "accn": f"a-{e}", "fy": int(e[:4]), "fp": "Q1", "filed": e}
            for i, (s, e) in enumerate(Q_ENDS)]
    return {"facts": {"us-gaap": {REVENUE: {"units": {"USD": rows}}}}}


def test_a_ttm_window_ending_early_may_not_be_called_current():
    """"TTM revenue is Y" when the window ends a quarter before the period."""
    resolution = resolve_current_actual_state(
        [_actual("2026-09-30", "8-K", "2026-11-01")], as_of=TODAY)
    ttm = reconstruct_ttm(_facts(), "2026-09-30", metrics=("revenue",))
    view = build_research_freshness_view(resolution, ttm=ttm,
                                         metrics=("revenue",))
    assert view.eligibility["revenue"].eligibility == FreshnessEligibility.STALE
    assert "revenue" in view.must_be_qualified


def test_a_ttm_window_reaching_the_period_keeps_its_label():
    resolution = resolve_current_actual_state(
        [_actual("2026-06-30", "8-K", "2026-08-01")], as_of=TODAY)
    ttm = reconstruct_ttm(_facts(), "2026-06-30", metrics=("revenue",))
    view = build_research_freshness_view(resolution, ttm=ttm,
                                         metrics=("revenue",))
    assert view.eligibility["revenue"].eligibility == FreshnessEligibility.CURRENT
    assert "revenue" in view.may_claim_current
