"""Phase H.14 — a guidance metric cannot become a different one.

The failure this phase corrects, found by testing the extractor directly
rather than by reading a report:

    "For fiscal 2027 we expect subscription revenue growth of 11% to 12%."

extracted as `revenue_growth`, scope `consolidated`, and reported
`may_anchor_revenue_growth = True`. A component of revenue was set to anchor
the whole company's growth assumption. Separately, "free cash flow growth",
"operating cash flow growth" and "EPS growth" matched nothing at all, so real
guidance was dropped and the coverage matrix reported none.

Both have one cause, and it is the point of these tests: **a gap in the
metric vocabulary is not a neutral absence.** When no identity exists for a
phrase, the nearest GENERAL pattern claims it. Absent identities do not
produce missing data; they produce wrong data.

No issuer is named in any production path exercised here.
"""

import pytest

from finance import guidance as G


N = G.GuidanceMetricName


def _extract(sentence):
    release = G.extract_guidance_from_text(sentence, "ZZ", "acc", "doc", "2026-06-01")
    return dict(getattr(release, "metrics", None) or {})


# ---------------------------------------------------------------------------
# Phase 1 — every growth metric extracts as ITSELF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence, expected", [
    ("For fiscal 2027 we expect revenue growth of 8% to 9%.",
     N.CONSOLIDATED_REVENUE_GROWTH),
    ("For fiscal 2027 we expect subscription revenue growth of 11% to 12%.",
     N.SUBSCRIPTION_REVENUE_GROWTH),
    ("For fiscal 2027 we expect free cash flow growth of 20% to 25%.",
     N.FREE_CASH_FLOW_GROWTH),
    ("For fiscal 2027 we expect operating cash flow growth of 12% to 14%.",
     N.OPERATING_CASH_FLOW_GROWTH),
    ("For fiscal 2027 we expect EPS growth of 15% to 17%.",
     N.EPS_GROWTH),
    ("For fiscal 2027 we expect adjusted EBITDA growth of 10% to 12%.",
     N.ADJUSTED_EBITDA_GROWTH),
    ("For fiscal 2027 we expect service revenue growth of 5% to 6%.",
     N.SERVICE_REVENUE_GROWTH),
])
def test_each_growth_metric_keeps_its_own_identity(sentence, expected):
    assert expected in _extract(sentence)


def test_a_revenue_component_is_never_read_as_the_consolidated_total():
    """The live cross-wiring, stated as the rule it broke."""
    metrics = _extract("For fiscal 2027 we expect subscription revenue growth of 11% to 12%.")
    assert N.SUBSCRIPTION_REVENUE_GROWTH in metrics
    assert N.CONSOLIDATED_REVENUE_GROWTH not in metrics


def test_cash_flow_growth_is_recorded_rather_than_dropped():
    """Phase 9/10: dropping real guidance is what produced 'none extracted'."""
    for sentence in ("For fiscal 2027 we expect free cash flow growth of 20% to 25%.",
                     "For fiscal 2027 we expect operating cash flow growth of 12% to 14%.",
                     "For fiscal 2027 we expect EPS growth of 15% to 17%."):
        assert _extract(sentence), f"guidance was silently dropped: {sentence!r}"


def test_the_extracted_value_stays_with_its_own_metric():
    """Two growth figures in one sentence must not swap."""
    metrics = _extract(
        "For fiscal 2027 we expect revenue growth of 8% to 9% and free cash flow "
        "growth of 20% to 25%.")
    assert metrics[N.CONSOLIDATED_REVENUE_GROWTH].low == pytest.approx(0.08)
    assert metrics[N.FREE_CASH_FLOW_GROWTH].low == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Phase 35 — the source-metric matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source, allowed", [
    # The only two provenances a revenue-growth assumption may carry.
    (N.CONSOLIDATED_REVENUE_GROWTH, True),
    (N.CONSOLIDATED_REVENUE, True),
    # Rates of change of something that is not revenue.
    (N.FREE_CASH_FLOW_GROWTH, False),
    (N.OPERATING_CASH_FLOW_GROWTH, False),
    (N.EPS_GROWTH, False),
    (N.NET_INCOME_GROWTH, False),
    (N.EBITDA_GROWTH, False),
    (N.ADJUSTED_EBITDA_GROWTH, False),
    # Margins are not growth at all.
    (N.OPERATING_MARGIN, False),
    (N.GROSS_MARGIN, False),
    # Parts of revenue, which the total can move quite differently from.
    (N.SUBSCRIPTION_REVENUE_GROWTH, False),
    (N.SERVICE_REVENUE_GROWTH, False),
    (N.PRODUCT_REVENUE_GROWTH, False),
    (N.SEGMENT_REVENUE_GROWTH, False),
])
def test_revenue_growth_source_matrix(source, allowed):
    ok, _reason = G.validate_revenue_growth_source(source)
    assert ok is allowed


def test_an_unrecognised_metric_fails_closed():
    """The guard exists because vocabulary gaps recur.

    A source this function has never heard of has not been shown to be
    revenue growth, and assuming it is would reproduce the exact failure it
    guards against.
    """
    ok, reason = G.validate_revenue_growth_source("some_metric_added_next_year")
    assert ok is False
    assert "has not been established" in reason


def test_a_missing_source_fails_closed():
    ok, reason = G.validate_revenue_growth_source(None)
    assert ok is False
    assert "no source metric" in reason


def test_the_rejection_explains_the_economics():
    """A reason a reader can act on, not just a refusal."""
    _ok, reason = G.validate_revenue_growth_source(N.FREE_CASH_FLOW_GROWTH)
    assert "measures a different quantity" in reason
    assert "kept as evidence" in reason


# ---------------------------------------------------------------------------
# Phase 38 — the cascade stops at the earliest boundary
# ---------------------------------------------------------------------------

def test_a_mismapped_metric_is_rejected_before_any_value_is_assigned():
    """Phase 12: the rejection happens before a number exists to clamp.

    A clamp must never repair a semantic error -- it turns a measurement of
    the wrong quantity into a plausible one, which is harder to notice than
    the original mistake.
    """
    from finance import forward_assumptions as FA

    evidence = FA.GrowthEvidence()

    # Drive the guard directly: a forbidden source must produce a rejection
    # and no anchor.
    ok, reason = G.validate_revenue_growth_source(N.FREE_CASH_FLOW_GROWTH)
    assert not ok
    evidence.semantic_rejections.append({"code": G.GUIDANCE_METRIC_MISMATCH,
                                         "reason": reason})

    path = FA.build_growth_path(evidence, forecast_years=5)
    # With no guidance anchor and no other evidence, the path falls through
    # to the configured default -- it does NOT carry the rejected figure.
    assert evidence.guidance_low is None
    assert path.entries[0].clamped is False


def test_the_mismatch_carries_a_stable_code():
    """Phase 15: a root cause needs an id downstream can suppress against."""
    assert G.GUIDANCE_METRIC_MISMATCH == "GUIDANCE_METRIC_MISMATCH"


def test_the_forbidden_set_and_the_allowed_set_do_not_overlap():
    """An invariant about the invariant: no metric may be both."""
    assert not (G.FORBIDDEN_REVENUE_GROWTH_SOURCES
                & G.ALLOWED_REVENUE_GROWTH_SOURCES)


def test_only_consolidated_revenue_growth_may_anchor():
    """The two gates agree. `may_anchor_revenue_growth` is the older check
    and `validate_revenue_growth_source` the newer one; a metric the second
    forbids must never pass the first."""
    for metric in G.FORBIDDEN_REVENUE_GROWTH_SOURCES:
        assert G.may_anchor_revenue_growth(metric) is False


# ---------------------------------------------------------------------------
# Units and denominator are part of the identity
# ---------------------------------------------------------------------------
#
# Found by a live run. A release stated:
#
#     "non-GAAP operating income is expected to be approximately 21% of
#      projected revenue"
#
# and it extracted as `operating_income` -- an ABSOLUTE currency metric --
# carrying the value 0.21. A margin stored under an absolute identity, with a
# currency unit, and a magnitude three orders out.
#
# The general invariant this violated is spec 3's: two figures are the same
# metric when their metric_id matches, and a metric_id is not just a name.
# "Operating income of $2.1 billion" and "operating income equal to 21% of
# revenue" are not the same quantity -- one is a dollar amount, the other is
# a RATIO WITH REVENUE AS ITS DENOMINATOR -- and the identity has to say so
# or a downstream consumer cannot tell them apart.
#
# So the unit and the denominator participate in identity: an absolute metric
# expressed as a percentage OF REVENUE resolves to that metric's margin
# identity, and where no margin identity exists the figure is REFUSED rather
# than stored under the absolute one. Fail closed: a percentage whose
# denominator was never established has not been shown to be anything.

def test_operating_income_as_a_percent_of_revenue_is_a_margin():
    metrics = _extract(
        "For the third quarter of fiscal 2027, non-GAAP operating income is expected to "
        "be 20% to 22% of projected revenue.")
    assert N.OPERATING_INCOME not in metrics
    entry = metrics[N.ADJUSTED_OPERATING_MARGIN]
    assert entry.unit == G.GuidanceUnit.RATIO
    assert (entry.low, entry.high) == pytest.approx((0.20, 0.22))
    assert entry.basis == G.BASIS_ADJUSTED


def test_gaap_operating_income_as_a_percent_of_revenue_is_the_gaap_margin():
    metrics = _extract(
        "For the third quarter of fiscal 2027, operating income is expected to be "
        "12% to 13% of revenue.")
    assert N.OPERATING_INCOME not in metrics
    assert metrics[N.OPERATING_MARGIN].unit == G.GuidanceUnit.RATIO
    assert metrics[N.OPERATING_MARGIN].basis == G.BASIS_GAAP


def test_adjusted_ebitda_as_a_percent_of_revenue_is_an_ebitda_margin():
    """The identity did not exist at all, which is the worse failure mode:
    the nearest general pattern claimed the text and an EBITDA MARGIN was
    recorded as an absolute adjusted EBITDA amount."""
    metrics = _extract(
        "For fiscal 2027 we expect Adjusted EBITDA to be 28% to 30% of projected revenue.")
    assert N.ADJUSTED_EBITDA not in metrics
    entry = metrics[N.ADJUSTED_EBITDA_MARGIN]
    assert entry.unit == G.GuidanceUnit.RATIO
    assert (entry.low, entry.high) == pytest.approx((0.28, 0.30))


def test_a_dollar_value_keeps_the_absolute_identity():
    """The other half of the pair. Nothing about this fix may change how an
    ordinary dollar-denominated outlook is read."""
    metrics = _extract(
        "For fiscal 2027 we expect operating income of $2.1 billion to $2.3 billion.")
    assert N.OPERATING_MARGIN not in metrics
    entry = metrics[N.OPERATING_INCOME]
    assert entry.unit == G.GuidanceUnit.CURRENCY
    assert (entry.low, entry.high) == pytest.approx((2.1, 2.3))


def test_absolute_adjusted_ebitda_in_dollars_keeps_its_identity():
    metrics = _extract(
        "For fiscal 2027 we expect Adjusted EBITDA of $900 million to $950 million.")
    assert N.ADJUSTED_EBITDA_MARGIN not in metrics
    assert metrics[N.ADJUSTED_EBITDA].unit == G.GuidanceUnit.CURRENCY


def test_a_percentage_with_no_stated_denominator_is_refused_not_guessed():
    """Fail closed. "Operating income is expected to grow 12% to 14%" is a
    growth rate, not a margin, and this vocabulary has no operating-income
    growth identity -- so the figure is refused rather than stored as either
    an absolute amount or a margin."""
    metrics = _extract(
        "For fiscal 2027 operating income is expected to increase 12% to 14%.")
    assert N.OPERATING_INCOME not in metrics
    assert N.OPERATING_MARGIN not in metrics
    assert N.ADJUSTED_OPERATING_MARGIN not in metrics


def test_the_refusal_is_recorded_rather_than_silent():
    release = G.extract_guidance_from_text(
        "For fiscal 2027 operating income is expected to increase 12% to 14%.",
        "ZZ", "acc", "doc", "2026-06-01")
    assert any("percentage" in w.lower() for w in release.warnings), release.warnings


def test_percent_of_sales_is_the_same_denominator_as_percent_of_revenue():
    """A release may say sales where another says revenue. The denominator is
    the same quantity and the identity must not depend on the wording."""
    metrics = _extract(
        "For fiscal 2027 we expect operating income to be approximately 15% to 16% "
        "of net sales.")
    assert metrics[N.OPERATING_MARGIN].unit == G.GuidanceUnit.RATIO


def test_a_margin_identity_can_never_anchor_revenue_growth():
    """Spec 11's hard invariant reaches the new identities too: an EBITDA
    margin is not revenue growth, however it was spelled."""
    assert not G.may_anchor_revenue_growth(N.ADJUSTED_EBITDA_MARGIN)
    assert not G.may_anchor_revenue_growth(N.EBITDA_MARGIN)
    assert N.ADJUSTED_EBITDA_MARGIN in G.FORBIDDEN_REVENUE_GROWTH_SOURCES
    assert N.EBITDA_MARGIN in G.FORBIDDEN_REVENUE_GROWTH_SOURCES


def test_the_new_margin_identities_are_in_the_reviewed_taxonomy():
    """`validate_guidance_metric` refuses a name it does not know, so an
    identity that is produced but not registered would be produced and then
    discarded."""
    metrics = _extract(
        "For fiscal 2027 we expect Adjusted EBITDA to be 28% to 30% of projected revenue.")
    assert G.validate_guidance_metric(metrics[N.ADJUSTED_EBITDA_MARGIN]) == []


# ---------------------------------------------------------------------------
# The OTHER denominator: year-over-year
# ---------------------------------------------------------------------------
#
# Found by the live rerun of the fix above, on two more issuers. The same
# invariant, the other denominator:
#
#     "raising our Adjusted EPS guidance to year-over-year GROWTH of 5.0 to
#      6.0 percent"          -> adjusted_earnings_per_share = $5.00-$6.00
#     "Free cash flow growth of 9.0 to 10.0 percent year-over-year"
#                            -> free_cash_flow = $9-$10
#
# Both were stored as ABSOLUTE amounts, roughly a fifth of the real figure,
# and both had a growth identity waiting in the vocabulary that nothing
# routed them to.
#
# Two things were wrong and both are general:
#
#   1. The word "percent" was not read as a percent marker. Only the SIGN
#      was. "5.0 to 6.0 percent" and "5.0% to 6.0%" are the same measurement
#      written two ways, and a unit that depends on which one a writer chose
#      is not a unit.
#
#   2. The identity resolver knew one denominator (revenue) and not the
#      other (the prior period). A percentage of revenue is a margin; a
#      percentage year-over-year is a growth rate; the vocabulary has both,
#      and the resolver has to reach both or the gap it does not cover fills
#      with absolute amounts.

@pytest.mark.parametrize("sentence, expected", [
    ("For fiscal 2027 we expect free cash flow growth of 9.0 to 10.0 percent "
     "year-over-year.", N.FREE_CASH_FLOW_GROWTH),
    ("For fiscal 2027 we are raising our adjusted EPS guidance to year-over-year "
     "growth of 5.0 to 6.0 percent.", N.EPS_GROWTH),
    ("For fiscal 2027 we expect operating cash flow to grow 3.0 to 4.0 percent "
     "year-over-year.", N.OPERATING_CASH_FLOW_GROWTH),
])
def test_a_percentage_year_over_year_is_a_growth_rate(sentence, expected):
    metrics = _extract(sentence)
    assert expected in metrics, sorted(metrics)
    assert metrics[expected].unit == G.GuidanceUnit.RATIO
    assert metrics[expected].high <= 1.0, "a growth rate is a ratio, not a percentage point"


@pytest.mark.parametrize("absolute", [N.FREE_CASH_FLOW, N.ADJUSTED_EPS, N.EPS,
                                      N.OPERATING_CASH_FLOW])
def test_no_absolute_identity_holds_a_year_over_year_percentage(absolute):
    metrics = _extract(
        "For fiscal 2027 we expect free cash flow growth of 9.0 to 10.0 percent "
        "year-over-year, and we are raising our adjusted EPS guidance to "
        "year-over-year growth of 5.0 to 6.0 percent.")
    assert absolute not in metrics, f"{absolute} holds a growth rate as an absolute amount"


def test_the_word_percent_is_the_same_unit_as_the_sign():
    """Two spellings of one measurement. A unit that depends on which one the
    writer chose is not a unit."""
    spelled = _extract("For fiscal 2027 we expect revenue growth of 8.0 to 9.0 percent.")
    signed = _extract("For fiscal 2027 we expect revenue growth of 8.0% to 9.0%.")
    assert spelled[N.CONSOLIDATED_REVENUE_GROWTH].low == pytest.approx(
        signed[N.CONSOLIDATED_REVENUE_GROWTH].low)
    assert spelled[N.CONSOLIDATED_REVENUE_GROWTH].high == pytest.approx(0.09)


def test_a_footnote_marker_does_not_change_a_metrics_identity():
    """Releases print footnote markers inside metric names ("Free cash flow 1
    growth"). The marker is typography; the identity is the same, and a
    pattern that misses it sends the figure to the absolute identity."""
    metrics = _extract(
        "For fiscal 2027 we expect Free cash flow 1 growth of 9.0 to 10.0 percent "
        "year-over-year.")
    assert N.FREE_CASH_FLOW not in metrics
    assert metrics[N.FREE_CASH_FLOW_GROWTH].unit == G.GuidanceUnit.RATIO


def test_a_dollar_free_cash_flow_outlook_is_still_absolute():
    metrics = _extract(
        "For fiscal 2027 we expect free cash flow of $19.0 billion to $20.0 billion.")
    assert N.FREE_CASH_FLOW_GROWTH not in metrics
    assert metrics[N.FREE_CASH_FLOW].unit == G.GuidanceUnit.CURRENCY


# ---------------------------------------------------------------------------
# The percent marker means the same thing in EVERY value shape
# ---------------------------------------------------------------------------
#
# "Spelling is not a unit" was added to the extractor for RANGES and nowhere
# else. Guidance arrives in four value shapes, each with its own pattern, and
# each pattern carried its own private copy of the percent marker:
#
#     range                 "20% to 22%"                 fixed
#     approximate point     "approximately 67 percent"   NOT fixed
#     point with tolerance  "74.9%, plus or minus 50bps" NOT fixed
#     midpoint restatement  "or 7.3% at the midpoint"    NOT fixed
#
# A live release guided "Non-GAAP operating income of approximately 67 percent
# of projected revenue" and "Adjusted EBITDA of approximately 68 percent of
# projected revenue". Both are approximate POINTS, so both went through the
# one shape that still read the word "percent" as no unit at all -- and were
# stored as $67 billion and $68 billion of quarterly profit against $29.4
# billion of quarterly revenue.
#
# One rule, one definition, every shape. This is the same defect class as two
# minimum-content rules for claims: the rule was right and it was applied in
# one of the places that needed it.

@pytest.mark.parametrize("sentence, expected, value", [
    # approximate point
    ("Third quarter fiscal 2027 non-GAAP operating income guidance of approximately "
     "67 percent of projected revenue.", N.ADJUSTED_OPERATING_MARGIN, 0.67),
    ("Third quarter fiscal 2027 Adjusted EBITDA guidance of approximately 68 percent "
     "of projected revenue.", N.ADJUSTED_EBITDA_MARGIN, 0.68),
    # the same sentence written with the sign must give the same answer
    ("Third quarter fiscal 2027 non-GAAP operating income guidance of approximately "
     "67% of projected revenue.", N.ADJUSTED_OPERATING_MARGIN, 0.67),
])
def test_an_approximate_point_reads_percent_as_a_unit(sentence, expected, value):
    metrics = _extract(sentence)
    assert expected in metrics, sorted(metrics)
    entry = metrics[expected]
    assert entry.unit == G.GuidanceUnit.RATIO
    assert entry.low == pytest.approx(value)
    assert entry.high == pytest.approx(value)


@pytest.mark.parametrize("absolute", [N.OPERATING_INCOME, N.ADJUSTED_EBITDA, N.EBITDA])
def test_no_absolute_identity_survives_an_approximate_percentage(absolute):
    metrics = _extract(
        "Third quarter fiscal 2027 non-GAAP operating income guidance of approximately "
        "67 percent of projected revenue. Third quarter fiscal 2027 Adjusted EBITDA "
        "guidance of approximately 68 percent of projected revenue.")
    assert absolute not in metrics, (
        f"{absolute} holds a percentage as an absolute amount: {metrics.get(absolute)}")


def test_the_two_spellings_of_an_approximate_point_agree():
    """The invariant itself, stated as a property: the same measurement
    written two ways must produce the same metric, unit and value."""
    spelled = _extract("For fiscal 2027 we expect a gross margin of approximately "
                       "74.5 percent.")
    signed = _extract("For fiscal 2027 we expect a gross margin of approximately 74.5%.")
    assert set(spelled) == set(signed)
    for name in spelled:
        assert spelled[name].unit == signed[name].unit
        assert spelled[name].low == pytest.approx(signed[name].low)


def test_a_tolerance_point_reads_percent_as_a_unit():
    spelled = _extract("For fiscal 2027 we expect a gross margin of 74.9 percent, "
                       "plus or minus 50 basis points.")
    signed = _extract("For fiscal 2027 we expect a gross margin of 74.9%, "
                      "plus or minus 50 basis points.")
    assert spelled and set(spelled) == set(signed)
    for name in spelled:
        assert spelled[name].low == pytest.approx(signed[name].low)


def test_a_scale_word_is_still_not_a_percent_marker():
    """The other half. Nothing here may turn a dollar figure into a ratio."""
    metrics = _extract("Third quarter fiscal 2027 revenue guidance of approximately "
                       "$29.4 billion.")
    entry = metrics[N.CONSOLIDATED_REVENUE]
    assert entry.unit == G.GuidanceUnit.CURRENCY
    assert entry.low == pytest.approx(29.4)
    assert entry.scale == "billion"


def test_every_value_shape_uses_the_one_percent_marker():
    """The durable half of the fix.

    "The sign and the word are one unit" was correct and was applied to ONE
    of the six patterns that read a value. Six private copies of `(%?)`
    meant six chances for the rule to be true in the abstract and false in
    the code, and the shape that carried a live release's guidance was one
    of the five that had not been updated.

    So the rule is a name, and no pattern spells the marker itself. A
    seventh value shape added later either refers to `_PCT` or fails here.
    """
    import inspect
    import re as _re

    source = inspect.getsource(G)
    # Strip the definition of `_PCT` itself; it is the one place the marker
    # is allowed to be written out.
    source = source.replace(G._PCT, "<_PCT>")
    offenders = [line.strip() for line in source.splitlines()
                 if _re.search(r"\(%\?\)", line)]
    assert not offenders, (
        "a value pattern spells its own percent marker instead of using _PCT: "
        + "; ".join(offenders))


@pytest.mark.parametrize("pattern_name", [
    "_RANGE_PATTERNS", "_TOLERANCE_PATTERN", "_APPROXIMATE_PATTERN",
    "_FLOOR_PATTERN", "_MIDPOINT_PATTERN", "_BARE_FIGURE_PATTERN",
    "_DUAL_BASIS_PATTERN",
])
def test_each_named_value_shape_still_exists(pattern_name):
    """Names the shapes, so removing one is a visible decision rather than a
    silent narrowing of what counts as guidance."""
    assert getattr(G, pattern_name, None) is not None
