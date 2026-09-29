"""Does the evidence describe the metric the candidate claims?

THE CRITICAL FALSE POSITIVE THIS CLOSES

    evidence:  "The Company expects inside same-store sales to increase 2% to 5%"
    candidate: revenue_growth, FY2027, 2%-5%   -> ACCEPTED

Every check passed. The numbers were in the sentence, the unit was a ratio,
the period resolved, the sentence was prospective. Nothing asked whether the
sentence was about consolidated revenue, and it is not -- same-store sales
excludes new stores, closures and acquisitions, so a retailer's total revenue
can move quite differently.

Accepted under that name it becomes a DCF revenue-growth assumption:
`guidance.may_anchor_revenue_growth` gates that path by NAME, and by then the
name was already wrong. So the earliest boundary is metric identity itself.

THE INVARIANT

    A candidate is acceptable only where the cited evidence supports both its
    METRIC IDENTITY and its ECONOMIC SCOPE. Numeric grounding is not enough.

The reader is allowed to misread a sentence. The validator is not allowed to
publish that misreading as a financial fact.
"""

import pytest

from finance import guidance as gm
from finance.extraction.metric_grounding import (
    MatchType,
    ground_metric,
    narrowing_qualifier,
)
from finance.extraction.schema import (
    GuidanceCandidate,
    RejectionCode,
    ValueType,
)
from finance.extraction.validator import GuidanceCandidateValidator

SAME_STORE = ("The Company expects inside same-store sales to increase 2% to "
              "5% with an inside margin above 42%.")
CONSOLIDATED = "The Company expects total company revenue to grow 8% to 10%."
CLOUD = "Cloud revenue is expected to grow 25%."
NET_SALES = "Net sales are expected to increase 6%."
EBITDA_MARGIN = "Adjusted EBITDA is expected to be 68% of revenue."


def _candidate(metric_id, low, high, sentence, unit="PERCENT",
               value_type=ValueType.GROWTH_RATE, period="FY2027", **kwargs):
    return GuidanceCandidate(
        metric_id=metric_id, value_type=value_type, low=low, high=high,
        unit=unit, target_period=period, target_period_type="annual",
        prospective=True, source_sentence=sentence, confidence=0.95, **kwargs)


def _validate(candidate, document=None):
    document = document or ("Outlook. " + candidate.source_sentence)
    return GuidanceCandidateValidator(
        document_text=document, issued_at="2026-06-09").validate(candidate)


# ---------------------------------------------------------------------------
# A / B. the failing case
# ---------------------------------------------------------------------------

def test_same_store_sales_is_not_consolidated_revenue_growth():
    """The critical false positive, at the boundary that must stop it."""
    metric, code, reason = _validate(
        _candidate("revenue_growth", 2.0, 5.0, SAME_STORE))
    assert metric is None, "a component measure was published as the consolidated one"
    assert code == RejectionCode.METRIC_NOT_GROUNDED
    assert "same-store" in reason.lower()


def test_the_grounding_result_names_the_scope_conflict():
    result = ground_metric("revenue_growth", SAME_STORE)
    assert not result.supported
    assert result.match_type == MatchType.COMPONENT_METRIC
    assert result.code == "METRIC_NOT_GROUNDED"


def test_the_component_is_not_silently_retained_under_another_name():
    """§7: it may be kept under its OWN identity, never promoted.

    The taxonomy has no same-store-sales metric, so the correct outcome is
    refusal -- not invention of a metric to hold it.
    """
    assert "same_store_sales_growth" not in gm._METRIC_BY_NAME  # noqa: SLF001


# ---------------------------------------------------------------------------
# C / D / E / F. scope and aliases
# ---------------------------------------------------------------------------

def test_consolidated_revenue_growth_is_accepted():
    metric, code, reason = _validate(
        _candidate("revenue_growth", 8.0, 10.0, CONSOLIDATED))
    assert metric is not None, f"{code}: {reason}"
    assert metric.name == "revenue_growth"


def test_an_approved_alias_is_accepted():
    """"Net sales" is consolidated revenue. A literal-name check would refuse
    most real releases, which is why the taxonomy's own pattern is used."""
    metric, code, reason = _validate(
        _candidate("revenue_growth", 6.0, 6.0, NET_SALES))
    assert metric is not None, f"{code}: {reason}"


@pytest.mark.parametrize("sentence,qualifier", [
    (CLOUD, "cloud"),
    ("International revenue is expected to grow 12%.", "international"),
    ("Comparable-store sales are expected to increase 3%.", "comparable-store"),
    ("Segment revenue is expected to grow 4%.", "segment"),
    ("Subscription revenue is expected to grow 30%.", "subscription"),
])
def test_a_component_is_never_the_consolidated_metric(sentence, qualifier):
    result = ground_metric("revenue_growth", sentence)
    assert not result.supported, f"{qualifier} was read as consolidated"
    assert result.match_type == MatchType.COMPONENT_METRIC


def test_a_component_candidate_is_accepted_where_the_taxonomy_has_one():
    """§7/E: the component survives under its OWN identity."""
    metric, code, reason = _validate(
        _candidate("subscription_revenue_growth", 30.0, 30.0,
                   "Subscription revenue is expected to grow 30%."))
    assert metric is not None, f"{code}: {reason}"
    assert metric.name == "subscription_revenue_growth"


@pytest.mark.parametrize("qualifier,expected", [
    ("total", False), ("consolidated", False), ("worldwide", False),
    ("net", False), ("company", False),
    ("cloud", True), ("same-store", True), ("international", True),
])
def test_narrowing_qualifiers_are_told_from_harmless_ones(qualifier, expected):
    sentence = f"{qualifier} revenue is expected to grow 5%."
    assert bool(narrowing_qualifier(sentence)) is expected


# ---------------------------------------------------------------------------
# G / H. identities that must keep working
# ---------------------------------------------------------------------------

def test_a_margin_stated_as_a_percentage_of_revenue_still_grounds():
    """§G. Resolved through `guidance.resolve_percentage_identity`, the
    project's own rule, rather than a second one written here."""
    result = ground_metric("adjusted_ebitda_margin", EBITDA_MARGIN)
    assert result.supported, result.reason
    assert result.match_type == MatchType.APPROVED_DERIVATION


def test_absolute_revenue_guidance_still_grounds_the_level_metric():
    """§H: the annual-guidance growth derivation happens downstream from a
    LEVEL candidate, so the level must keep being accepted."""
    metric, code, reason = _validate(_candidate(
        "revenue", 4.10, 4.30,
        "For fiscal 2027 the company expects revenue of $4.10 billion to "
        "$4.30 billion.",
        unit="USD_BILLION", value_type=ValueType.RANGE))
    assert metric is not None, f"{code}: {reason}"
    assert metric.name == "revenue"


def test_a_growth_metric_against_a_named_comparison_period_is_a_derivation():
    result = ground_metric(
        "revenue_growth",
        "Full year 2027 revenue is expected to be $4.30 billion.",
        comparison_period="FY2026")
    assert result.supported
    assert result.match_type == MatchType.APPROVED_DERIVATION


def test_an_unknown_metric_name_is_still_refused():
    result = ground_metric("vibe_growth", CONSOLIDATED)
    assert not result.supported
    assert result.match_type == MatchType.UNSUPPORTED


def test_a_sentence_about_something_else_entirely_is_refused():
    metric, code, _reason = _validate(_candidate(
        "revenue_growth", 5.0, 5.0,
        "The effective tax rate is expected to increase 5%."))
    assert metric is None
    assert code == RejectionCode.METRIC_NOT_GROUNDED


# ---------------------------------------------------------------------------
# I / J. table rows
# ---------------------------------------------------------------------------

TABLE_DOC = ("The following table summarizes our guidance based on the current "
             "outlook. Current Guidance . FY2027 Estimates . "
             "Total revenue $9.7B - $10.5B. "
             "Same-store sales growth 2% - 5%. ")


def test_a_correct_row_under_a_guidance_caption_is_accepted():
    metric, code, reason = _validate(
        _candidate("revenue", 9.7, 10.5, "Total revenue $9.7B - $10.5B.",
                   unit="USD_BILLION", value_type=ValueType.RANGE),
        document=TABLE_DOC)
    assert metric is not None, f"{code}: {reason}"


def test_a_prospective_caption_cannot_turn_a_component_row_into_revenue_growth():
    """§9: the caption qualifies the row as forward-looking. It says nothing
    about what the row MEASURES, and must not be allowed to."""
    metric, code, _reason = _validate(
        _candidate("revenue_growth", 2.0, 5.0,
                   "Same-store sales growth 2% - 5%."),
        document=TABLE_DOC)
    assert metric is None
    assert code == RejectionCode.METRIC_NOT_GROUNDED


# ---------------------------------------------------------------------------
# K / L. existing guards
# ---------------------------------------------------------------------------

def test_a_historical_table_row_is_still_refused():
    document = ("Condensed Consolidated Statements of Operations . "
                "Three Months Ended June 30 . Total revenue $9.7B . ")
    metric, code, _reason = _validate(
        _candidate("revenue", 9.7, 9.7, "Total revenue $9.7B .",
                   unit="USD_BILLION", value_type=ValueType.POINT),
        document=document)
    assert metric is None
    assert code


def test_an_analyst_estimate_is_still_refused():
    sentence = "Analysts expect total company revenue to grow 8% to 10%."
    metric, code, _reason = _validate(
        _candidate("revenue_growth", 8.0, 10.0, sentence))
    assert metric is None
    assert code == RejectionCode.ANALYST_ESTIMATE


def test_an_ungrounded_sentence_is_still_refused_first():
    metric, code, _reason = _validate(
        _candidate("revenue_growth", 8.0, 10.0, CONSOLIDATED),
        document="Outlook. Something else entirely.")
    assert metric is None
    assert code == RejectionCode.EVIDENCE_NOT_IN_SOURCE
