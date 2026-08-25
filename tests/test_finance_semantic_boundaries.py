"""Phase H.10, sections 37/41/53 — the contract at each system boundary.

The semantics module is only worth having if the pipeline actually asks it.
`test_finance_semantics.py` proves the validator answers correctly; these
tests prove the CALLERS consult it, which is the failure mode section 46
names: validators that exist but run too late, or beside the path rather
than on it.

Each test drives a real production entry point and asserts that a specific
bad transition cannot occur -- not that some output looks better.
"""

import pytest

from finance import business_model as B
from finance import claim_validation as CV
from finance import content_policy as CP
from finance import entity as E
from finance import forward_assumptions as FA
from finance import semantics as S


# ---------------------------------------------------------------------------
# Boundary: guidance -> forward assumption (sections 8-11)
# ---------------------------------------------------------------------------

class _Quarter:
    def __init__(self, value, start, end):
        self.value, self.start, self.end = value, start, end


class _Series:
    def __init__(self, quarters):
        self.quarters = quarters


class _Ttm:
    def __init__(self, value, ok=True):
        self.ok, self.value = ok, value
        self.period_start, self.period_end = "2025-06-01", "2026-05-31"


def _install(monkeypatch, ttm_value, quarter_values):
    """Point the assumption builder at a synthetic issuer.

    No ticker, no fixture file -- just an issuer whose annual revenue is
    `ttm_value` and whose recent quarters are `quarter_values`.
    """
    from finance import freshness as fr
    from finance import period_facts as pf

    monkeypatch.setattr(fr, "build_ttm", lambda *a, **k: _Ttm(ttm_value))
    quarters = [_Quarter(v, "2025-%02d-01" % (i * 3 + 1), "2025-%02d-28" % (i * 3 + 3))
                for i, v in enumerate(quarter_values)]
    monkeypatch.setattr(pf, "discrete_quarters", lambda *a, **k: _Series(quarters))


def test_a_quarterly_guidance_level_never_becomes_an_annual_growth_rate(monkeypatch):
    """The ADBE path, end to end at the boundary that produced it.

    A guided quarter of 6.7 against a trailing twelve months of 25.2 is the
    -73% that reached the model. The guidance must not set `guidance_low`.
    """
    _install(monkeypatch, ttm_value=25.2e9, quarter_values=[5.9e9, 6.1e9, 6.3e9, 6.5e9])
    evidence = FA.GrowthEvidence()
    FA._apply_absolute_revenue_guidance(  # noqa: SLF001
        evidence,
        {"midpoint": 6.695, "scale": "billion", "period_type": "quarter",
         "fiscal_year": 2026, "fiscal_quarter": 3, "basis": "GAAP"},
        company_facts={"facts": {}})
    assert evidence.guidance_low is None
    assert evidence.guidance_high is None
    # It becomes supporting evidence about the NEXT QUARTER instead, measured
    # against the comparable quarter a year earlier.
    assert evidence.guidance_implied_next_period_growth is not None


def test_a_mislabelled_annual_guidance_is_refused_rather_than_clamped(monkeypatch):
    """Sections 10-11, and the generalisation that matters.

    Here the extractor has got the period WRONG -- a quarterly figure
    labelled annual, exactly what a live release headed "third quarter
    FY2026 targets" produced. The regex fix for that wording is elsewhere;
    this asserts the system survives the next wording it has not seen, by
    disbelieving a label the issuer's own scale contradicts.
    """
    _install(monkeypatch, ttm_value=25.2e9, quarter_values=[5.9e9, 6.1e9, 6.3e9, 6.5e9])
    evidence = FA.GrowthEvidence()
    FA._apply_absolute_revenue_guidance(  # noqa: SLF001
        evidence,
        {"midpoint": 6.695, "scale": "billion", "period_type": "annual",
         "fiscal_year": 2026, "basis": "GAAP"},
        company_facts={"facts": {}})

    assert evidence.guidance_low is None, "a -73% growth rate was derived"
    assert evidence.semantic_rejections, "the refusal must be recorded, not silent"
    rejection = evidence.semantic_rejections[0]
    assert rejection["code"] == FA.GUIDANCE_PERIOD_INCOMPATIBLE
    assert "trailing-twelve-month" in rejection["reason"]


def test_genuine_full_year_guidance_still_produces_a_growth_rate(monkeypatch):
    """The contract must not cost a normal company its guidance anchor."""
    _install(monkeypatch, ttm_value=97.9e9, quarter_values=[24e9, 24.5e9, 24.6e9, 24.8e9])
    evidence = FA.GrowthEvidence()
    FA._apply_absolute_revenue_guidance(  # noqa: SLF001
        evidence,
        {"midpoint": 101.1, "scale": "billion", "period_type": "annual",
         "fiscal_year": 2026, "basis": "GAAP", "fiscal_period": "FY2026"},
        company_facts={"facts": {}})
    assert evidence.guidance_low == pytest.approx((101.1e9 - 97.9e9) / 97.9e9)
    assert evidence.semantic_rejections == []


def test_a_rejected_derivation_produces_no_clamp_to_hide(monkeypatch):
    """Section 31: one error, one message -- not a cascade of five.

    The old order was derive -> clamp -> warn about the clamp -> widen the
    scenario spread -> flag the valuation gap. With the derivation refused,
    there is no out-of-range value for a clamp to act on.
    """
    _install(monkeypatch, ttm_value=25.2e9, quarter_values=[5.9e9, 6.1e9, 6.3e9, 6.5e9])
    evidence = FA.GrowthEvidence(ttm_yoy=0.08)
    FA._apply_absolute_revenue_guidance(  # noqa: SLF001
        evidence,
        {"midpoint": 6.695, "scale": "billion", "period_type": "annual",
         "fiscal_year": 2026, "basis": "GAAP"},
        company_facts={"facts": {}})

    path = FA.build_growth_path(evidence, forecast_years=5)
    assert evidence.guidance_low is None
    # The anchor falls through to the trailing trend, in range, unclamped.
    assert path.entries[0].clamped is False
    assert path.values[0] == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# Boundary: share counts -> reconciliation finding (sections 12-13)
# ---------------------------------------------------------------------------

def _security():
    return E.SecurityIdentity(
        ticker="ZZ", entity_id="0000000001", exchange="NYSE",
        security_type="common_stock", depositary_ratio=1.0, currency="USD",
        share_classes=())


def test_incompatible_share_bases_never_raise_a_conflict_finding():
    """Section 13, at the production entry point.

    Two counts 40% apart, on bases that differ by construction. Before this
    phase that produced a share-count conflict; the correct output is a
    recorded, non-comparable pair and no finding.
    """
    counts = E.ShareCountSet()
    counts.set(E.ShareCountType.CURRENT_OUTSTANDING, 1_000, "provider")
    counts.set(E.ShareCountType.WEIGHTED_AVERAGE_DILUTED, 1_400, "sec")
    result = E.reconcile_share_basis(counts, _security())

    assert result.status != E.ReconciliationStatus.MATERIAL_DIFFERENCE
    assert result.findings == []
    assert all(c["status"] == E.ReconciliationStatus.NOT_COMPARABLE
               for c in result.comparisons)


def test_like_for_like_counts_that_disagree_still_raise_one():
    """The other half of the contract: a real disagreement must survive.

    Two CURRENT counts from two providers are the same quantity measured
    twice, so a 40% gap between them is a genuine data-integrity problem and
    must not be silenced by the same change that quieted the basis mismatch.
    """
    counts = E.ShareCountSet()
    counts.set(E.ShareCountType.CURRENT_OUTSTANDING, 1_000, "provider")
    counts.set(E.ShareCountType.ECONOMIC_OUTSTANDING, 1_400, "sec")
    result = E.reconcile_share_basis(counts, _security())
    statuses = {c["status"] for c in result.comparisons}
    assert E.ReconciliationStatus.NOT_COMPARABLE not in statuses


# ---------------------------------------------------------------------------
# Boundary: business model -> DCF (sections 17-19)
# ---------------------------------------------------------------------------

def test_a_specialized_business_cannot_reach_the_standard_model():
    broker = B.classify_business_model(submissions={"sic": "6211"})
    assert B.may_enter_standard_fcff(broker) is False


def test_the_gate_is_checked_before_inputs_are_assembled():
    """Section 19: refused, not annotated after the fact.

    `_business_model_blocks_dcf` is what the input builder calls first. If
    the check lived only in the suitability assessment, the scenario values
    would already exist by the time anything said they should not.
    """
    from finance import workflow as W

    broker = B.classify_business_model(submissions={"sic": "6211"})
    blocked = W._business_model_blocks_dcf({"_business_model": broker})  # noqa: SLF001
    assert blocked is not None
    code, message = blocked
    assert code == W.DCF_CASH_FLOW_NOT_STANDARD_FCFF
    assert "owner free cash flow" in message

    ordinary = B.classify_business_model(submissions={"sic": "3674"})
    assert W._business_model_blocks_dcf({"_business_model": ordinary}) is None  # noqa: SLF001


def test_no_dcf_is_not_the_same_as_a_failed_analysis():
    """Section 20: the two states are separately representable.

    A refusal carries its own code, so a reader can be told the valuation
    model does not fit this business rather than that something broke.
    """
    from finance import workflow as W

    broker = B.classify_business_model(submissions={"sic": "6211"})
    code, _message = W._business_model_blocks_dcf({"_business_model": broker})  # noqa: SLF001
    assert code != W.DCF_INPUT_NORMALIZATION_UNRESOLVED


# ---------------------------------------------------------------------------
# Boundary: research output -> report (sections 26, 35)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, flagged", [
    ("RSI above 70 historically predicts a correction.", True),
    ("The MACD crossover signals an imminent reversal.", True),
    ("A golden cross typically precedes a rally.", True),
    # Descriptive statements of the same indicators, which must stay legal.
    ("RSI is above 70, indicating relatively elevated recent momentum.", False),
    ("Price is above the 200-day SMA and below the 50-day SMA.", False),
    ("The stock is overbought with RSI at 72.", False),
    ("The MACD histogram is positive at 0.18.", False),
])
def test_technical_language_stays_descriptive(text, flagged):
    assert bool(CV.scan_for_unsupported_claims(text)) is flagged


@pytest.mark.parametrize("text, flagged", [
    ("Investors should continue holding the position.", True),
    ("Shareholders can maintain their current position.", True),
    ("Existing holders should reassess after the next filing.", True),
    # HOLD as a classification, and ordinary uses of the same verbs.
    ("HOLD reflects the research classification only.", False),
    ("The company continues to hold market share in its core segment.", False),
    ("Cash holdings rose over the period.", False),
])
def test_hold_never_assumes_the_reader_owns_the_security(text, flagged):
    """Section 35. The system has no portfolio data source, so a sentence
    presuming a position invents a fact about the reader."""
    assert bool(CP.scan_for_prohibited_directives(text)) is flagged


def test_the_recommendation_enum_itself_stays_legal():
    """The assumed-position rules are prose-only for this reason."""
    for value in ("buy", "hold", "sell", "avoid", "insufficient_evidence"):
        assert CP.scan_for_prohibited_directives(value) == []


# ---------------------------------------------------------------------------
# Section 38 — the invariants, stated as invariants
# ---------------------------------------------------------------------------

def test_invariant_no_quarterly_absolute_over_ttm_yields_growth():
    quarter = S.SemanticFact(metric_id=S.MetricIdentity.REVENUE, value=1.0,
                             period_frequency=S.PeriodFrequency.QUARTER,
                             flow_or_instant=S.FlowOrInstant.FLOW)
    ttm = S.SemanticFact(metric_id=S.MetricIdentity.REVENUE, value=4.0,
                         period_frequency=S.PeriodFrequency.TTM,
                         flow_or_instant=S.FlowOrInstant.FLOW)
    assert not S.compatible_for(S.Operation.GROWTH, quarter, ttm)


def test_invariant_no_weighted_average_reconciles_against_a_current_count():
    current = S.SemanticFact(metric_id=S.MetricIdentity.SHARES_CURRENT_OUTSTANDING,
                             value=1.0, period_frequency=S.PeriodFrequency.INSTANT,
                             flow_or_instant=S.FlowOrInstant.INSTANT)
    diluted = S.SemanticFact(metric_id=S.MetricIdentity.SHARES_WEIGHTED_AVERAGE_DILUTED,
                             value=1.0, period_frequency=S.PeriodFrequency.ANNUAL,
                             flow_or_instant=S.FlowOrInstant.FLOW)
    assert not S.compatible_for(S.Operation.RECONCILE, current, diluted)


def test_invariant_no_component_sum_is_compared_to_long_term_debt():
    total = S.SemanticFact(metric_id=S.MetricIdentity.TOTAL_DEBT, value=100.0,
                           period_frequency=S.PeriodFrequency.INSTANT,
                           flow_or_instant=S.FlowOrInstant.INSTANT)
    long_term = S.SemanticFact(metric_id=S.MetricIdentity.LONG_TERM_DEBT, value=80.0,
                               period_frequency=S.PeriodFrequency.INSTANT,
                               flow_or_instant=S.FlowOrInstant.INSTANT)
    verdict = S.compatible_for(S.Operation.RECONCILE, total, long_term)
    assert not verdict
    assert verdict.code == S.INCOMPATIBLE_DEBT_BASIS


def test_invariant_a_financial_company_simple_fcf_stays_out_of_the_model():
    for sic in ("6021", "6211", "6311", "6199"):
        classification = B.classify_business_model(submissions={"sic": sic})
        assert B.may_enter_standard_fcff(classification) is False


# ---------------------------------------------------------------------------
# Section 20 — a declined valuation is explained, not just absent
# ---------------------------------------------------------------------------

def _analysis_result():
    """The minimum an AnalysisResult needs to render a compact report."""
    from finance.workflow import AnalysisMode, AnalysisPlan, AnalysisResult

    plan = AnalysisPlan(
        symbol="ZZ", mode=AnalysisMode.FULL, datasets=(), cached_datasets=(),
        uncached_datasets=(), max_external_calls=0, estimated_remaining_quota=0,
        reason="synthetic plan for a renderer test")
    return AnalysisResult(symbol="ZZ", plan=plan, facts={}, errors=[], warnings=[])


def test_the_report_says_why_no_valuation_was_produced():
    """An absent DCF reads as a malfunction unless the report says otherwise.

    Section 20 separates "research completeness" from "DCF availability";
    that separation is only useful if a reader can see it on the page.
    """
    from finance.workflow import render_compact_report

    compact = {
        "symbol": "ZZ",
        "quote": {"price": 9.01, "currency": "USD"},
        "dcf": {"available": False, "reason": "not owner free cash flow",
                "unavailable_code": "DCF_CASH_FLOW_NOT_STANDARD_FCFF"},
        "dcf_suitability": {"dcf_suitability": "LIMITED"},
        "business_model": {
            "profile": "BROKER_DEALER", "standard_fcff_suitability": "NOT_SUITABLE",
            "sic": "6211", "sic_description": "Security Brokers, Dealers",
        },
        # Phase H.12 rewrote this passage. It used to render TWICE -- once
        # from the H.10 block and once from the H.11 block -- and described a
        # model that never applied as one that had not been "produced". The
        # single statement now names the applicability, which is the actual
        # state, and the explanation still follows it.
        "valuation_method_status": "VALID_BUT_NOT_APPLICABLE",
    }
    report = render_compact_report(_analysis_result(), compact, pipeline_result=None)
    assert "not applicable to this business model" in report
    assert "broker dealer" in report
    assert "6211" in report
    assert "Research therefore relies on" in report
    # Section 44: said once, not twice.
    assert report.count("not applicable to this business model") == 1
    # Section 16: never described as invalid.
    assert "model invalid" not in report.lower()


def test_an_ordinary_company_gets_no_such_note():
    from finance.workflow import render_compact_report

    compact = {
        "symbol": "ZZ",
        "quote": {"price": 100.0, "currency": "USD"},
        "dcf": {"available": False, "reason": "no annual income statement"},
        "business_model": {"profile": "STANDARD_OPERATING_COMPANY",
                           "standard_fcff_suitability": "SUITABLE", "sic": "3674"},
    }
    report = render_compact_report(_analysis_result(), compact, pipeline_result=None)
    assert "was NOT produced" not in report


# ---------------------------------------------------------------------------
# Reporting currency — a filer this project cannot convert
# ---------------------------------------------------------------------------

def _facts_in(currency, concepts=("Revenues", "Assets", "OperatingIncomeLoss")):
    """Company facts whose MAPPED statement concepts are in `currency`."""
    return {"facts": {"us-gaap": {
        name: {"units": {currency: [{"val": 1.0, "form": "20-F", "fy": 2025,
                                     "fp": "FY", "start": "2025-01-01",
                                     "end": "2025-12-31"}]}}
        for name in concepts}}}


def test_a_foreign_currency_filer_is_named_rather_than_called_empty():
    """The ASML failure: complete accounts, reported as "not reported".

    An issuer filing full statements in euros has not failed to report
    anything. Saying so sends a reader looking for missing data that is
    present, and hides the actual limit -- this project has no FX
    conversion.
    """
    from finance import taxonomy as T

    note = T.reporting_currency_note(_facts_in("EUR"))
    assert note is not None
    assert "EUR" in note
    assert "currency conversion" in note


def test_a_dollar_filer_gets_no_currency_note():
    from finance import taxonomy as T

    assert T.reporting_currency_note(_facts_in("USD")) is None


def test_an_incidental_dollar_disclosure_does_not_make_a_filer_readable():
    """The bug that kept the note silent.

    A euro reporter carried three USD-denominated tags -- a derivative
    notional, a purchase commitment, a tax-benefit lapse -- none of them a
    statement line. Only the concepts the numeric path actually reads may
    decide the reporting currency.
    """
    from finance import taxonomy as T

    facts = _facts_in("EUR")
    facts["facts"]["us-gaap"]["NotionalAmountOfForeignCurrencyDerivatives"] = {
        "units": {"USD": [{"val": 5.0, "form": "20-F"}]}}
    facts["facts"]["us-gaap"]["PurchaseCommitmentRemainingMinimumAmountCommitted"] = {
        "units": {"USD": [{"val": 7.0, "form": "20-F"}]}}
    assert T.reporting_currency_note(facts) is not None


def test_a_share_count_never_establishes_the_reporting_currency():
    """Every filer reports share counts in `shares`, whatever the accounts
    are denominated in -- the first version of this check was fooled by it."""
    from finance import taxonomy as T

    facts = _facts_in("EUR")
    facts["facts"]["us-gaap"]["WeightedAverageNumberOfDilutedSharesOutstanding"] = {
        "units": {"shares": [{"val": 400.0, "form": "20-F"}]}}
    assert T.reporting_currency_note(facts) is not None


def test_foreign_private_issuer_annual_forms_are_recognised():
    """A 20-F filer's annual facts must not be filtered out as non-annual.

    `xbrl_mapping` carried its own ("10-K", "10-K/A") list while
    `finance/taxonomy.py` already had the right one, so for every foreign
    private issuer the numeric path saw no annual facts at all.
    """
    from finance import taxonomy as T
    from finance.xbrl_mapping import ANNUAL_FORMS, QUARTERLY_FORMS

    assert "20-F" in ANNUAL_FORMS and "40-F" in ANNUAL_FORMS
    assert "10-K" in ANNUAL_FORMS
    assert tuple(ANNUAL_FORMS) == tuple(T.ANNUAL_FORMS)
    assert tuple(QUARTERLY_FORMS) == tuple(T.INTERIM_FORMS)


def test_the_declined_valuation_states_the_currency_reason():
    from finance.workflow import render_compact_report

    compact = {
        "symbol": "ZZ",
        "quote": {"price": 1748.63, "currency": "USD"},
        "dcf": {"available": False,
                "unavailable_code": "DCF_REPORTING_CURRENCY_UNSUPPORTED",
                "reason": "This issuer reports in EUR. This project has no currency "
                          "conversion, so its financial statements are not read into a "
                          "valuation that ends in a US-dollar price per share."},
    }
    report = render_compact_report(_analysis_result(), compact, pipeline_result=None)
    assert "no discounted-cash-flow valuation was produced" in report
    assert "reports in EUR" in report
    assert "not reported" not in report
