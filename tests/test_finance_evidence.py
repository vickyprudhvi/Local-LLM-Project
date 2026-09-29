"""Phase H.2 — finance/evidence.py: the evidence-ID index behind the staged
research pipeline (finance/research_pipeline.py).

Uses small, hand-built compact-payload dicts (not a real AnalysisResult) so
each indexing/validation rule is checked in isolation. See
tests/test_finance_research_pipeline_integration.py for the real,
end-to-end-through-synthesize_report proof.
"""

from finance.evidence import build_evidence_index, render_evidence_index, validate_evidence_citations

MINIMAL_PAYLOAD = {
    "symbol": "TEST",
    "analysis_mode": "full",
    "company": {"name": "Test Corp", "sector": "TECHNOLOGY", "industry": "SOFTWARE",
                "market_capitalisation": 20000, "pe_ratio": 25.5, "forward_pe": None,
                "price_to_book": 3.2, "beta": 1.1, "dividend_yield": None,
                "shares_outstanding": 100},
    "quote": {"price": 200.0, "price_basis": "delayed", "change_percent": "1.01%",
              "latest_trading_day": "2026-08-04", "previous_close": 198.0},
    "fundamental_metrics": {
        "roe_ending_equity": {"value": 0.25, "formula": "net_income / ending_equity"},
        "roe_average_equity": {"value": 0.27, "formula": "net_income / average_equity"},
        "revenue_growth_yoy": {"value": None, "formula": "..."},  # None -> not indexed
    },
    "technical_metrics": {
        "rsi_14": {"value": 71.2, "formula": "..."},
    },
    "dcf": {
        "available": True,
        "net_debt": 150.0,
        "net_debt_policy": "cash_only",
        "value_per_share": 210.5,
        "scenarios": [
            {"scenario": "bull", "value_per_share": 250.0, "enterprise_value": 5000.0,
             "equity_value": 4850.0, "net_debt": 150.0,
             "terminal_value_share_of_enterprise_value": 0.6},
            {"scenario": "base", "value_per_share": 210.5, "enterprise_value": 4200.0,
             "equity_value": 4050.0, "net_debt": 150.0,
             "terminal_value_share_of_enterprise_value": 0.55},
            {"scenario": "bear", "value_per_share": 170.0, "enterprise_value": 3400.0,
             "equity_value": 3250.0, "net_debt": 150.0,
             "terminal_value_share_of_enterprise_value": 0.5},
        ],
    },
    "valuation_gap": {
        "available": True, "market_price": 200.0, "market_price_basis": "delayed",
        "estimated_base_modeled_value_per_share": 210.5, "difference": 10.5,
        "difference_pct": 5.25, "direction": "undervalued",
    },
    "dcf_scenario_spread": {
        "available": True, "bull_value_per_share": 250.0, "base_value_per_share": 210.5,
        "bear_value_per_share": 170.0, "spread": 80.0, "spread_pct_of_base": 38.0,
    },
    "warnings": ["cash-and-short-term-investments does not reconcile with provider total"],
    "data_provenance": {
        "stock_quote": {"origin": "provider", "stale": False},
        "balance_sheet": {"origin": "cache", "stale": True},
    },
    "plan": {
        "omitted_datasets": ["earnings"],
        "omission_effects": {"earnings": "EPS surprise history is unavailable this run."},
    },
}


# ---- build_evidence_index: presence ----

def test_indexes_company_and_quote_fields():
    index = build_evidence_index(MINIMAL_PAYLOAD)
    assert index["company.name"].value == "Test Corp"
    assert index["company.sector"].value == "TECHNOLOGY"
    assert index["quote.price"].value == 200.0
    assert index["quote.price_basis"].value == "delayed"


def test_indexes_meta_symbol_and_analysis_mode():
    index = build_evidence_index(MINIMAL_PAYLOAD)
    assert index["meta.symbol"].value == "TEST"
    assert index["meta.analysis_mode"].value == "full"


def test_indexes_fundamental_and_technical_metric_values_only():
    index = build_evidence_index(MINIMAL_PAYLOAD)
    assert index["fundamental.roe_ending_equity"].value == 0.25
    assert index["fundamental.roe_average_equity"].value == 0.27
    assert index["technical.rsi_14"].value == 71.2


def test_none_valued_fields_are_never_indexed():
    """A stage cannot cite an ID for a fact that was never reported, because
    the ID must not exist at all -- this is what makes a missing input
    unciteable rather than citeable-with-a-null-value."""
    index = build_evidence_index(MINIMAL_PAYLOAD)
    assert "fundamental.revenue_growth_yoy" not in index
    assert "company.forward_pe" not in index
    assert "company.dividend_yield" not in index


def test_indexes_dcf_top_level_and_per_scenario_fields():
    index = build_evidence_index(MINIMAL_PAYLOAD)
    assert index["dcf.net_debt"].value == 150.0
    assert index["dcf.net_debt_policy"].value == "cash_only"
    assert index["dcf.value_per_share"].value == 210.5
    assert index["dcf.value_per_share.bull"].value == 250.0
    assert index["dcf.value_per_share.base"].value == 210.5
    assert index["dcf.value_per_share.bear"].value == 170.0
    assert index["dcf.enterprise_value.bull"].value == 5000.0
    assert index["dcf.net_debt.bear"].value == 150.0


def _dcf_scenario(name, revenue_growth, operating_margin, wacc, terminal_growth,
                  terminal_value_share):
    """One scenario shaped like the REAL finance.dcf.value_scenario().to_dict()
    output: assumptions carry per-year arrays (finance/dcf.py expands a scalar
    to one entry per forecast year) plus a full assumption_provenance block —
    exactly what finance/workflow.py::_compact_dcf hands to build_evidence_index."""
    return {
        "scenario": name,
        "value_per_share": 210.5, "enterprise_value": 4200.0, "equity_value": 4050.0,
        "net_debt": 150.0, "terminal_value_share_of_enterprise_value": terminal_value_share,
        "assumptions": {
            "name": name,
            "revenue_growth": [revenue_growth] * 5, "operating_margin": [operating_margin] * 5,
            "wacc": wacc, "terminal_growth": terminal_growth, "tax_rate": 0.21,
            "capex_pct_revenue": [0.018] * 5, "depreciation_pct_revenue": [0.009] * 5,
            "working_capital_pct_revenue": [0.003] * 5,
            "assumption_provenance": {
                "revenue_growth": {
                    "value": revenue_growth, "source_type": "deterministic_calculation",
                    "source_periods": ["2025-12-31"], "source_evidence_ids": [],
                    "derivation": f"Derived from reported revenue_cagr for the {name!r} scenario.",
                    "approval_status": "proposed", "units": "ratio",
                },
                "operating_margin": {
                    "value": operating_margin, "source_type": "deterministic_calculation",
                    "source_periods": ["2025-12-31"], "source_evidence_ids": [],
                    "derivation": f"Derived from reported operating_margin for the {name!r} scenario.",
                    "approval_status": "proposed", "units": "ratio",
                },
                "wacc": {
                    "value": wacc, "source_type": "configured_default", "source_periods": [],
                    "source_evidence_ids": [], "derivation": f"Configured default WACC for {name!r}.",
                    "approval_status": "proposed", "units": "ratio",
                },
                "terminal_growth": {
                    "value": terminal_growth, "source_type": "configured_default",
                    "source_periods": [], "source_evidence_ids": [],
                    "derivation": f"Configured default terminal growth for {name!r}.",
                    "approval_status": "proposed", "units": "ratio",
                },
                # capex/depreciation/working_capital/tax_rate deliberately
                # OMITTED here -- in the real payload these are hoisted out to
                # 'shared_assumption_provenance' (see DCF_PAYLOAD_WITH_ASSUMPTIONS
                # below) since they are byte-identical across every scenario.
            },
        },
    }


DCF_PAYLOAD_WITH_ASSUMPTIONS = dict(MINIMAL_PAYLOAD)
DCF_PAYLOAD_WITH_ASSUMPTIONS["dcf"] = {
    "available": True, "net_debt": 150.0, "net_debt_policy": "cash_only",
    "value_per_share": 210.5, "calculation_version": "fcff_enterprise_v1",
    "primary_scenario": "base",
    "scenarios": [
        _dcf_scenario("bull", 0.13, 0.22, 0.08, 0.030, 0.6),
        _dcf_scenario("base", 0.09, 0.20, 0.09, 0.025, 0.55),
        _dcf_scenario("bear", 0.05, 0.17, 0.11, 0.015, 0.5),
    ],
    "shared_assumption_provenance": {
        "capex_pct_revenue": {
            "value": 0.018, "source_type": "deterministic_calculation",
            "source_periods": ["2025-12-31", "2024-12-31"], "source_evidence_ids": [],
            "derivation": "Average of 2 reported CapEx/revenue ratios.",
            "approval_status": "proposed", "units": "ratio",
        },
        "depreciation_pct_revenue": {
            "value": 0.009, "source_type": "deterministic_calculation",
            "source_periods": ["2025-12-31", "2024-12-31"], "source_evidence_ids": [],
            "derivation": "Average of 2 reported D&A/revenue ratios.",
            "approval_status": "proposed", "units": "ratio",
        },
        "working_capital_pct_revenue": {
            "value": 0.003, "source_type": "deterministic_calculation",
            "source_periods": ["2025-12-31", "2024-12-31"], "source_evidence_ids": [],
            "derivation": "Average of 2 reported net-working-capital/revenue ratios.",
            "approval_status": "proposed", "units": "ratio",
        },
        "tax_rate": {
            "value": 0.21, "source_type": "configured_default", "source_periods": [],
            "source_evidence_ids": [],
            "derivation": "Configured default statutory-adjacent tax rate.",
            "approval_status": "proposed", "units": "ratio",
        },
    },
}


# ---- H.4 corrective patch (Problem 3): DCF assumptions as evidence ----

def test_per_scenario_dcf_assumption_evidence_ids_exist_for_base_bull_and_bear():
    """Tests 11-13 (revenue_growth) plus operating_margin, for good measure."""
    index = build_evidence_index(DCF_PAYLOAD_WITH_ASSUMPTIONS)
    for field in ("revenue_growth", "operating_margin"):
        for scenario, expected in (("base", None), ("bull", None), ("bear", None)):
            assert f"dcf.assumption.{field}.{scenario}" in index, f"{field}.{scenario}"
    assert index["dcf.assumption.revenue_growth.base"].value == 0.09
    assert index["dcf.assumption.revenue_growth.bull"].value == 0.13
    assert index["dcf.assumption.revenue_growth.bear"].value == 0.05
    assert index["dcf.assumption.operating_margin.bull"].value == 0.22


def test_wacc_scenario_assumption_evidence_ids_exist():
    """Test 14."""
    index = build_evidence_index(DCF_PAYLOAD_WITH_ASSUMPTIONS)
    assert index["dcf.assumption.wacc.base"].value == 0.09
    assert index["dcf.assumption.wacc.bull"].value == 0.08
    assert index["dcf.assumption.wacc.bear"].value == 0.11


def test_terminal_growth_scenario_assumption_evidence_ids_exist():
    """Test 15."""
    index = build_evidence_index(DCF_PAYLOAD_WITH_ASSUMPTIONS)
    assert index["dcf.assumption.terminal_growth.base"].value == 0.025
    assert index["dcf.assumption.terminal_growth.bull"].value == 0.030
    assert index["dcf.assumption.terminal_growth.bear"].value == 0.015


def test_capex_depreciation_nwc_and_tax_rate_assumption_evidence_ids_exist_once_shared():
    """Test 16: these four are IDENTICAL across scenarios by construction, so
    they get ONE evidence ID each (no .base/.bull/.bear suffix), not three."""
    index = build_evidence_index(DCF_PAYLOAD_WITH_ASSUMPTIONS)
    assert index["dcf.assumption.capex_pct_revenue"].value == 0.018
    assert index["dcf.assumption.depreciation_pct_revenue"].value == 0.009
    assert index["dcf.assumption.working_capital_pct_revenue"].value == 0.003
    assert index["dcf.assumption.tax_rate"].value == 0.21
    for field in ("capex_pct_revenue", "depreciation_pct_revenue",
                 "working_capital_pct_revenue", "tax_rate"):
        for scenario in ("base", "bull", "bear"):
            assert f"dcf.assumption.{field}.{scenario}" not in index
        assert index[f"dcf.assumption.{field}"].scenario == "shared"


def test_dcf_assumption_evidence_carries_provenance_so_scenarios_can_be_explained():
    """Tests 17-18: an evidence item is not just a bare number -- it carries
    WHY (source_type/derivation/scenario), which is what lets a research
    stage explain that bull uses a higher revenue_growth and a lower wacc
    than base, not merely that the two modeled values differ."""
    index = build_evidence_index(DCF_PAYLOAD_WITH_ASSUMPTIONS)
    bull_growth = index["dcf.assumption.revenue_growth.bull"]
    assert bull_growth.evidence_type == "dcf_assumption"
    assert bull_growth.scenario == "bull"
    assert bull_growth.units == "ratio"
    assert bull_growth.source_type == "deterministic_calculation"
    assert "bull" in bull_growth.derivation
    assert bull_growth.calculation_version == "fcff_enterprise_v1"

    rendered = render_evidence_index(index)
    rendered_line = next(l for l in rendered.splitlines()
                         if l.startswith("dcf.assumption.revenue_growth.bull:"))
    assert "scenario=bull" in rendered_line
    assert "derivation=" in rendered_line


def test_terminal_value_share_alias_evidence_ids_exist_per_scenario():
    index = build_evidence_index(DCF_PAYLOAD_WITH_ASSUMPTIONS)
    assert index["dcf.terminal_value_share.bull"].value == 0.6
    assert index["dcf.terminal_value_share.base"].value == 0.55
    assert index["dcf.terminal_value_share.bear"].value == 0.5
    # The pre-existing longer-named id (generic _DCF_SCENARIO_FIELDS loop)
    # still exists too -- nothing is removed, only a shorter alias is added.
    assert index["dcf.terminal_value_share_of_enterprise_value.bull"].value == 0.6


def test_dcf_assumption_evidence_absent_when_dcf_unavailable():
    payload = dict(DCF_PAYLOAD_WITH_ASSUMPTIONS)
    payload["dcf"] = {"available": False}
    index = build_evidence_index(payload)
    assert not any(k.startswith("dcf.assumption.") for k in index)
    assert not any(k.startswith("dcf.terminal_value_share.") for k in index)


# ---- H.4 corrective patch (goal 1): negative equity remains evidence ----

def test_negative_shareholder_equity_is_indexed_as_evidence():
    """Test 4 at the evidence-index level: a negative equity VALUE is still
    citable even though the ratios computed from it are not."""
    payload = dict(MINIMAL_PAYLOAD)
    payload["fundamental_metrics"] = dict(MINIMAL_PAYLOAD["fundamental_metrics"])
    payload["fundamental_metrics"]["shareholder_equity"] = {"value": -800.0, "formula": "..."}
    index = build_evidence_index(payload)
    assert index["fundamental.shareholder_equity"].value == -800.0


def test_not_meaningful_roe_and_debt_to_equity_are_never_indexed():
    """A not_meaningful metric's value is None, and finance/evidence.py's
    generic fundamental_metrics loop only indexes non-None values -- so a
    stage structurally CANNOT cite 'fundamental.roe_ending_equity' or
    'fundamental.debt_to_equity' for a negative-equity company; the ID
    simply does not exist, exactly like any other absent fact."""
    payload = dict(MINIMAL_PAYLOAD)
    payload["fundamental_metrics"] = {
        "roe_ending_equity": {"value": None, "status": "not_meaningful",
                              "reason": "negative_shareholder_equity"},
        "debt_to_equity": {"value": None, "status": "not_meaningful",
                          "reason": "negative_shareholder_equity"},
        "shareholder_equity": {"value": -800.0, "formula": "..."},
    }
    index = build_evidence_index(payload)
    assert "fundamental.roe_ending_equity" not in index
    assert "fundamental.debt_to_equity" not in index
    assert index["fundamental.shareholder_equity"].value == -800.0


def test_dcf_unavailable_indexes_nothing_under_dcf():
    payload = dict(MINIMAL_PAYLOAD)
    payload["dcf"] = {"available": False}
    index = build_evidence_index(payload)
    assert not any(k.startswith("dcf.") for k in index), \
        "no DCF evidence may exist when the DCF itself is unavailable"


def test_dcf_assumption_required_still_indexes_its_validation_status():
    """HOOD corrective patch: finance/workflow.py::run_full_stock_analysis
    represents a DCF that was never even ATTEMPTED (insufficient reported
    history to auto-propose assumptions) as {"available": False, "reason":
    "DCF_ASSUMPTION_REQUIRED", "detail": ...} -- a DIFFERENT dict shape from
    a DCF that ran but failed validation ({"available": True,
    "validation_status": ..., "validation_reasons": [...]}, covered by
    test_invalid_dcf_still_indexes_its_own_validation_status_and_reasons
    below). Before this fix, ONLY the second shape ever populated
    'dcf.validation_status' in the index -- so a research-pipeline stage
    citing it for the FIRST shape (exactly what the shared guardrails' own
    "if 'dcf.validation_status' appears in the evidence index..." framing
    invites every stage to expect) failed evidence-ID citation validation. A
    live HOOD run hit this in 2 of 5 runs, always on
    final_investment_synthesizer, with no repair path (a schema/citation
    error, not a content-policy violation). 'DCF_ASSUMPTION_REQUIRED' is
    already a member of finance.dcf.DcfValidationStatus.INVALID, so this
    keeps the evidence index consistent with how research_pipeline.py's own
    `_dcf_validation_failed` already treats the two shapes identically."""
    payload = dict(MINIMAL_PAYLOAD)
    payload["dcf"] = {"available": False, "reason": "DCF_ASSUMPTION_REQUIRED",
                      "detail": "base_working_capital could not be derived from reported history."}
    index = build_evidence_index(payload)
    assert index["dcf.validation_status"].value == "DCF_ASSUMPTION_REQUIRED"
    assert "base_working_capital" in index["dcf.validation_reasons"].value
    # Still never indexes anything ELSE under dcf.* -- no scenario values,
    # no assumptions, exactly like the "ran but failed" shape.
    assert not any(k.startswith("dcf.") and k not in
                  ("dcf.validation_status", "dcf.validation_reasons") for k in index)


def test_dcf_assumption_required_with_no_detail_indexes_only_the_status():
    payload = dict(MINIMAL_PAYLOAD)
    payload["dcf"] = {"available": False, "reason": "DCF_ASSUMPTION_REQUIRED"}
    index = build_evidence_index(payload)
    assert index["dcf.validation_status"].value == "DCF_ASSUMPTION_REQUIRED"
    assert "dcf.validation_reasons" not in index


# ---- TSLA DCF validation patch: invalid DCF results must never become
# citable evidence (section 9/10) ----

def _invalid_dcf_payload(validation_status="DCF_INVALID_SCENARIO_ORDER"):
    payload = dict(MINIMAL_PAYLOAD)
    dcf = dict(MINIMAL_PAYLOAD["dcf"])
    dcf["validation_status"] = validation_status
    dcf["validation_reasons"] = [f"{validation_status}: the calculated values violate the "
                                 "expected bull >= base >= bear ordering."]
    payload["dcf"] = dcf
    return payload


def test_invalid_dcf_scenario_values_are_never_indexed():
    index = build_evidence_index(_invalid_dcf_payload())
    for name in ("bull", "base", "bear"):
        assert f"dcf.value_per_share.{name}" not in index
        assert f"dcf.enterprise_value.{name}" not in index
        assert f"dcf.equity_value.{name}" not in index
    assert "dcf.value_per_share" not in index


def test_invalid_dcf_assumption_evidence_is_never_indexed():
    index = build_evidence_index(_invalid_dcf_payload())
    assert not any(k.startswith("dcf.assumption.") for k in index)
    assert not any(k.startswith("dcf.terminal_value_share") for k in index)


def test_invalid_dcf_still_indexes_its_own_validation_status_and_reasons():
    """The status itself IS citable -- a bull/bear researcher, risk
    reviewer, or synthesizer must be able to see AND cite that the DCF
    failed, and why, even though none of its numbers are usable."""
    index = build_evidence_index(_invalid_dcf_payload("DCF_NEGATIVE_TERMINAL_FCFF"))
    assert index["dcf.validation_status"].value == "DCF_NEGATIVE_TERMINAL_FCFF"
    assert "DCF_NEGATIVE_TERMINAL_FCFF" in index["dcf.validation_reasons"].value


def test_valid_dcf_still_indexes_its_validation_status():
    payload = dict(MINIMAL_PAYLOAD)
    dcf = dict(MINIMAL_PAYLOAD["dcf"])
    dcf["validation_status"] = "DCF_VALID"
    payload["dcf"] = dcf
    index = build_evidence_index(payload)
    assert index["dcf.validation_status"].value == "DCF_VALID"
    # And the ordinary scenario evidence remains present -- a VALID status
    # does not withhold anything.
    assert "dcf.value_per_share.bull" in index


def test_dcf_with_no_validation_status_field_is_treated_as_usable():
    """Backward compatibility: MINIMAL_PAYLOAD's own 'dcf' block (used by
    every other test in this file) sets no 'validation_status' at all --
    confirms that absence defaults to 'usable', never silently withheld."""
    assert "validation_status" not in MINIMAL_PAYLOAD["dcf"]
    index = build_evidence_index(MINIMAL_PAYLOAD)
    assert "dcf.value_per_share.bull" in index
    assert "dcf.validation_status" not in index  # nothing to index -- the field is absent


def test_a_research_stage_citing_an_invalid_dcf_value_id_fails_citation_validation():
    """The structural guarantee end to end: even if a stage's raw JSON tries
    to cite 'dcf.value_per_share.bull' when the DCF is invalid, the ID does
    not exist in the index, so citation validation rejects it exactly like
    any other hallucinated ID."""
    index = build_evidence_index(_invalid_dcf_payload())
    ok, unknown = validate_evidence_citations(["dcf.value_per_share.bull"], index)
    assert ok is False
    assert "dcf.value_per_share.bull" in unknown


def test_indexes_valuation_gap_and_scenario_spread():
    index = build_evidence_index(MINIMAL_PAYLOAD)
    assert index["valuation_gap.direction"].value == "undervalued"
    assert index["valuation_gap.difference_pct"].value == 5.25
    assert index["scenario_spread.spread_pct_of_base"].value == 38.0


def test_valuation_gap_and_spread_unavailable_indexes_nothing():
    payload = dict(MINIMAL_PAYLOAD)
    payload["valuation_gap"] = {"available": False}
    payload["dcf_scenario_spread"] = {"available": False}
    index = build_evidence_index(payload)
    assert not any(k.startswith("valuation_gap.") for k in index)
    assert not any(k.startswith("scenario_spread.") for k in index)


def test_indexes_warnings_by_position():
    index = build_evidence_index(MINIMAL_PAYLOAD)
    assert index["warning.0"].value == MINIMAL_PAYLOAD["warnings"][0]


def test_indexes_provenance_origin_and_staleness_per_dataset():
    index = build_evidence_index(MINIMAL_PAYLOAD)
    assert index["provenance.stock_quote.origin"].value == "provider"
    assert index["provenance.stock_quote.stale"].value is False
    assert index["provenance.balance_sheet.stale"].value is True


def test_indexes_omitted_datasets_with_their_effect():
    index = build_evidence_index(MINIMAL_PAYLOAD)
    assert "plan.omitted.earnings" in index
    assert "unavailable" in index["plan.omitted.earnings"].value.lower() \
        or "EPS" in index["plan.omitted.earnings"].value


def test_build_evidence_index_is_deterministic():
    index_a = build_evidence_index(MINIMAL_PAYLOAD)
    index_b = build_evidence_index(MINIMAL_PAYLOAD)
    assert set(index_a) == set(index_b)
    assert [item.value for item in index_a.values()] == [item.value for item in index_b.values()]


def test_empty_payload_produces_an_empty_or_near_empty_index():
    index = build_evidence_index({})
    assert "dcf.net_debt" not in index
    assert "quote.price" not in index


# ---- render_evidence_index ----

def test_render_produces_one_line_per_item_with_id_prefix():
    index = build_evidence_index(MINIMAL_PAYLOAD)
    rendered = render_evidence_index(index)
    lines = rendered.splitlines()
    assert len(lines) == len(index)
    assert lines[0].startswith(f"{next(iter(index))}: ")


def test_render_is_a_plain_string_not_the_original_nested_json():
    index = build_evidence_index(MINIMAL_PAYLOAD)
    rendered = render_evidence_index(index)
    assert isinstance(rendered, str)
    assert "dcf.value_per_share.bull: " in rendered
    assert "250.0" in rendered


# ---- validate_evidence_citations: the code-level "never invent facts" backstop ----

def test_valid_known_ids_pass():
    index = build_evidence_index(MINIMAL_PAYLOAD)
    ok, unknown = validate_evidence_citations(["quote.price", "dcf.net_debt"], index)
    assert ok is True
    assert unknown == []


def test_a_single_fabricated_id_fails_the_whole_citation_list():
    index = build_evidence_index(MINIMAL_PAYLOAD)
    ok, unknown = validate_evidence_citations(
        ["quote.price", "fundamental.made_up_metric_no_such_thing"], index)
    assert ok is False
    assert "fundamental.made_up_metric_no_such_thing" in unknown


def test_citing_a_field_that_is_none_and_therefore_unindexed_fails():
    """A model that has SEEN a field name elsewhere (e.g. in training data or
    by guessing a plausible-looking ID) but the value was actually null in
    THIS analysis must not be able to cite it -- the ID simply isn't real."""
    index = build_evidence_index(MINIMAL_PAYLOAD)
    ok, unknown = validate_evidence_citations(["fundamental.revenue_growth_yoy"], index)
    assert ok is False
    assert unknown == ["fundamental.revenue_growth_yoy"]


def test_non_list_citations_fail_closed():
    index = build_evidence_index(MINIMAL_PAYLOAD)
    ok, _unknown = validate_evidence_citations("quote.price", index)
    assert ok is False
    ok, _unknown = validate_evidence_citations(None, index)
    assert ok is False
    ok, _unknown = validate_evidence_citations({"quote.price": True}, index)
    assert ok is False


def test_non_string_entries_in_citation_list_fail_closed():
    index = build_evidence_index(MINIMAL_PAYLOAD)
    ok, unknown = validate_evidence_citations(["quote.price", 12345, None], index)
    assert ok is False
    assert 12345 in unknown and None in unknown


def test_empty_citation_list_is_valid_but_cites_nothing():
    index = build_evidence_index(MINIMAL_PAYLOAD)
    ok, unknown = validate_evidence_citations([], index)
    assert ok is True
    assert unknown == []
