"""A number in a historical table is not guidance.

Spec §11 defines guidance as a PROSPECTIVE statement about a named period.
The extractor's qualification for that was a forward-looking WORD somewhere
within reach of the number -- and an earnings release is full of such words,
sitting a few hundred characters from condensed statements that report what
already happened. A live run turned a historical weighted-average diluted
share count into "management guidance: share_count", because "expects"
appeared in the paragraph above the table.

The general invariant: a GuidanceItem must have VALIDATED PROSPECTIVE
SEMANTICS, not merely a forward-looking word in the vicinity. Concretely it
must carry

    forward-looking context that is about THIS figure
    a target period
    a target period type
    a metric identity
    a prospective value or range
    an evidence location

and it must not sit inside a block the release itself labels as reported
results. Proximity is not qualification.

`tests/fixtures/earnings_release.py` is the hard case on purpose: three
historical tables and three prospective figures, close together, in the order
a real release uses them.
"""

import pytest

from finance import guidance as G
from tests.fixtures.earnings_release import (
    EARNINGS_RELEASE_HTML,
    PRIOR_WEIGHTED_AVERAGE_DILUTED_SHARES,
    Q3_EBITDA_MARGIN_OUTLOOK,
    Q3_OPERATING_MARGIN_OUTLOOK,
    Q3_REVENUE_OUTLOOK,
    REPORTED_Q2_OPERATING_INCOME,
    REPORTED_Q2_REVENUE,
    REPORTED_WEIGHTED_AVERAGE_DILUTED_SHARES,
)

N = G.GuidanceMetricName


@pytest.fixture(scope="module")
def release():
    return G.extract_guidance_from_text(
        G.html_to_text(EARNINGS_RELEASE_HTML), "ZZ", "9999999999-26-000001",
        "release.htm", "2026-07-28", expected_fiscal_year=2026)


@pytest.fixture(scope="module")
def metrics(release):
    return dict(release.metrics or {})


# ---------------------------------------------------------------------------
# The prospective figures ARE extracted
# ---------------------------------------------------------------------------
#
# Asserted first, because a qualification rule that keeps historical numbers
# out by keeping everything out has not solved anything.

def test_the_quarter_revenue_outlook_is_extracted(metrics):
    entry = metrics[N.CONSOLIDATED_REVENUE]
    assert (entry.low, entry.high) == pytest.approx(Q3_REVENUE_OUTLOOK)
    assert entry.target_period_type == G.GuidanceTargetType.NEXT_QUARTER


def test_the_operating_margin_outlook_is_extracted_as_a_margin(metrics):
    entry = metrics[N.ADJUSTED_OPERATING_MARGIN]
    assert (entry.low, entry.high) == pytest.approx(Q3_OPERATING_MARGIN_OUTLOOK)
    assert entry.unit == G.GuidanceUnit.RATIO
    assert entry.basis == G.BASIS_ADJUSTED


def test_the_ebitda_margin_outlook_is_extracted_as_a_margin(metrics):
    entry = metrics[N.ADJUSTED_EBITDA_MARGIN]
    assert (entry.low, entry.high) == pytest.approx(Q3_EBITDA_MARGIN_OUTLOOK)
    assert entry.unit == G.GuidanceUnit.RATIO


def test_exactly_the_three_prospective_figures_are_guidance(metrics):
    """The whole point, in one assertion. Three outlook sentences, three
    guidance metrics -- and the eleven historical numbers above them are not
    among them."""
    assert set(metrics) == {N.CONSOLIDATED_REVENUE, N.ADJUSTED_OPERATING_MARGIN,
                            N.ADJUSTED_EBITDA_MARGIN}, sorted(metrics)


# ---------------------------------------------------------------------------
# The historical figures are NOT
# ---------------------------------------------------------------------------

def test_a_historical_share_count_table_is_not_share_count_guidance(metrics):
    """The live failure. A weighted-average diluted share count is an EPS
    DENOMINATOR for a quarter that has already closed; reading it as a
    guided share count makes every per-share figure derived from guided
    earnings rest on a number the company never guided."""
    assert N.SHARE_COUNT not in metrics


def test_a_historical_revenue_table_is_not_revenue_guidance(metrics):
    entry = metrics.get(N.CONSOLIDATED_REVENUE)
    assert entry is not None
    for reported in (REPORTED_Q2_REVENUE, 1_042.9):
        assert entry.low != pytest.approx(reported)
        assert entry.high != pytest.approx(reported)


def test_a_reported_operating_income_is_not_operating_income_guidance(metrics):
    assert N.OPERATING_INCOME not in metrics
    margin = metrics[N.ADJUSTED_OPERATING_MARGIN]
    assert margin.low != pytest.approx(REPORTED_Q2_OPERATING_INCOME)


def test_a_reconciliation_table_margin_is_not_margin_guidance(metrics):
    """The reconciliation states an 18.0% GAAP operating margin FOR THE
    QUARTER JUST REPORTED. It is a real margin and it is not an outlook."""
    gaap = metrics.get(N.OPERATING_MARGIN)
    assert gaap is None or gaap.low != pytest.approx(0.18)


@pytest.mark.parametrize("value", [
    REPORTED_WEIGHTED_AVERAGE_DILUTED_SHARES,
    PRIOR_WEIGHTED_AVERAGE_DILUTED_SHARES,
    REPORTED_Q2_REVENUE,
    REPORTED_Q2_OPERATING_INCOME,
])
def test_no_reported_actual_appears_as_a_guidance_value(metrics, value):
    """A property test over the whole release: no figure printed in a
    historical table may appear as the low or high of any guidance item,
    whatever identity it was given."""
    for name, entry in metrics.items():
        assert entry.low != pytest.approx(value), f"{name} carries a reported actual"
        assert entry.high != pytest.approx(value), f"{name} carries a reported actual"


# ---------------------------------------------------------------------------
# Every published item carries validated prospective semantics
# ---------------------------------------------------------------------------

def test_every_guidance_item_carries_prospective_semantics(metrics):
    for name, entry in metrics.items():
        problems = G.validate_guidance_metric(entry)
        assert problems == [], f"{name}: {problems}"


@pytest.mark.parametrize("field", [
    "fiscal_period", "target_period_type", "name", "source_accession",
    "forward_kind", "prospective_evidence",
])
def test_a_guidance_item_names_each_required_property(metrics, field):
    for name, entry in metrics.items():
        assert getattr(entry, field, None), f"{name} has no {field}"


def test_the_prospective_evidence_locates_the_forward_statement(metrics):
    """"Forward-looking context" is recorded as the text that qualified THIS
    figure, not asserted as a boolean. A reader disagreeing with the verdict
    can see the sentence it was made from."""
    for name, entry in metrics.items():
        assert "expect" in entry.prospective_evidence.lower(), name


def test_an_item_without_prospective_semantics_is_refused_by_the_validator():
    """The validator is the boundary, so it must reject directly -- not only
    when the extractor happens to feed it well-formed input."""
    entry = metrics_stub = G.GuidanceMetric(
        name=N.SHARE_COUNT, low=139_933.0, high=139_933.0,
        unit=G.GuidanceUnit.SHARES, basis=G.BASIS_GAAP, fiscal_year=2026,
        evidence_id="dcf.guidance.share_count.current",
        source_excerpt="Diluted 139,933 145,758",
        fiscal_period="Q2 FY2026", source_accession="9999999999-26-000001",
        source_evidence_ids=("dcf.guidance.share_count.current",),
        prospective_evidence="")
    assert any("prospective" in p for p in G.validate_guidance_metric(metrics_stub)), \
        G.validate_guidance_metric(entry)


# ---------------------------------------------------------------------------
# The whole release, end to end through the coverage matrix
# ---------------------------------------------------------------------------

def test_the_coverage_matrix_reports_only_the_prospective_rows(metrics):
    matrix = G.build_guidance_matrix({k: v.to_dict() for k, v in metrics.items()},
                                     releases_examined=1)
    assert "share_count" not in (matrix.get("current_rows") or [])
    assert "revenue" in (matrix.get("current_rows") or [])
