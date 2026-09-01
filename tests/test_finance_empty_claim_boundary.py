"""An empty research claim must not survive validation.

A live compact report rendered a blank bullet under Bull Case. Tracing it
back, the earliest boundary that could have stopped it is the stage schema
validator -- and it did not, because two different minimum-content rules had
grown up side by side:

    `_string_list`   (assumptions, conditions, data-quality concerns)
                     drops entries shorter than `_MIN_CLAIM_CHARS` and counts
                     `min_items` on what SURVIVES.

    `_str_field`     (claim, risk, thesis, statement)
                     rejects "" and "   " and accepts everything else.

So a `claim` of "." or "-" or "N/A" passed the schema, passed quarantine,
reached the report model and rendered as a bullet that said nothing. The
schema exists to guarantee content; a bullet reading "-" meets it only on a
technicality, and the rule that already knew this was applied to the plain
string lists and not to the object-shaped claims that are the ones a reader
actually reads.

The fix is the same rule in both places, plus a structural backstop: the
report model cannot HOLD an empty claim, so even a future path that skips the
schema cannot put one in a report.

Spec §16: "Empty and placeholder claims cannot satisfy a schema. `min_items`
is counted on entries that SURVIVE filtering, not on entries supplied."
"""

import pytest

from finance import research_pipeline as R
from finance.report_model import ClaimSet, RiskItem, StockAnalysisReportModel, ReportStatus


def _index():
    from finance.evidence import EvidenceItem
    return {"current.revenue": EvidenceItem(evidence_id="current.revenue",
                                            label="Current revenue", value=1.0)}


def _claim(text, claim_id="c1"):
    return {"claim_id": claim_id, "claim": text,
            "evidence_ids": ["current.revenue"],
            "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.6}


# ---------------------------------------------------------------------------
# The earliest boundary: the stage schema
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blank", ["", "   ", "\n", "\t  \n"])
def test_a_whitespace_claim_fails_the_schema(blank):
    with pytest.raises(R._Invalid):
        R._claims_list({"claims": [_claim(blank), _claim("A real claim about revenue.", "c2")]},
                       "claims", _index(), min_items=2, max_items=5)


@pytest.mark.parametrize("placeholder", [".", "-", "N/A", "n/a", "TBD", "...", "?"])
def test_a_placeholder_claim_fails_the_schema(placeholder):
    """Not empty, and not a claim either. A bullet reading "-" occupies the
    space of one that would have said something."""
    with pytest.raises(R._Invalid):
        R._claims_list(
            {"claims": [_claim(placeholder), _claim("A real claim about revenue.", "c2")]},
            "claims", _index(), min_items=2, max_items=5)


def test_the_schema_counts_claims_that_survive_filtering():
    """Spec 16's actual wording. Two supplied, one contentless, min_items=2 --
    the count a reader sees is one, so the schema must not pass."""
    with pytest.raises(R._Invalid) as excinfo:
        R._claims_list({"claims": [_claim("Revenue grew materially year over year."),
                                   _claim(" ", "c2")]},
                       "claims", _index(), min_items=2, max_items=5)
    message = str(excinfo.value).lower()
    assert "claim" in message and ("content" in message or "non-empty" in message)


def test_real_claims_still_validate():
    out = R._claims_list(
        {"claims": [_claim("Revenue grew materially year over year."),
                    _claim("Operating margin expanded on the same basis.", "c2")]},
        "claims", _index(), min_items=2, max_items=5)
    assert len(out) == 2


@pytest.mark.parametrize("blank", ["", "   ", ".", "N/A"])
def test_a_blank_risk_fails_the_risk_reviewer_schema(blank):
    raw = {
        "key_risks": [{"risk": blank, "severity": "high",
                       "evidence_cited": ["current.revenue"]}],
        "data_quality_concerns": [],
        "evidence_cited": ["current.revenue"],
    }
    with pytest.raises(R._Invalid):
        R._validate_risk_reviewer_output(raw, _index())


def test_a_real_risk_still_validates():
    raw = {
        "key_risks": [{"risk": "Leverage is elevated relative to cash generation.",
                       "severity": "high", "evidence_cited": ["current.revenue"]}],
        "data_quality_concerns": [],
        "evidence_cited": ["current.revenue"],
    }
    assert R._validate_risk_reviewer_output(raw, _index())["key_risks"]


# ---------------------------------------------------------------------------
# The structural backstop: the report model cannot hold one
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_a_claim_set_refuses_an_empty_claim(blank):
    with pytest.raises(ValueError, match="empty"):
        ClaimSet(claims=("A real claim about revenue.", blank))


@pytest.mark.parametrize("blank", ["", "  ", "\t"])
def test_a_risk_item_refuses_empty_text(blank):
    with pytest.raises(ValueError, match="empty"):
        RiskItem(text=blank, severity="high")


def test_the_report_model_refuses_to_hold_an_empty_condition():
    with pytest.raises(ValueError, match="empty"):
        StockAnalysisReportModel(
            symbol="TEST", status=ReportStatus(headline="**COMPLETE**."),
            conditions=(("Upgrade conditions", ("A real condition.", "  ")),))


def test_a_report_model_with_real_content_constructs():
    model = StockAnalysisReportModel(
        symbol="TEST", status=ReportStatus(headline="**COMPLETE**."),
        bull_case=ClaimSet(claims=("Revenue grew materially year over year.",)),
        company_risk=(RiskItem(text="Leverage is elevated.", severity="high"),))
    assert model.bull_case.claims
    assert model.company_risk


# ---------------------------------------------------------------------------
# End to end: a stage that returns a blank claim cannot reach the report
# ---------------------------------------------------------------------------

def test_a_stage_returning_a_blank_claim_does_not_reach_the_report_model(
        tmp_path, monkeypatch):
    """The runtime check. The model is driven to emit a blank bullet; the
    pipeline must refuse the stage rather than pass it through, and the
    report model must contain no empty claim either way."""
    import json

    from finance.workflow import build_compact_synthesis_payload, synthesize_report
    from tests.canary_harness import wire_canary
    from tests.fixtures import canary_definitions as CANARIES
    from finance.workflow import run_full_stock_analysis

    blank_claims = json.dumps({
        "thesis": "The company is growing and generating cash.",
        "claims": [
            {"claim_id": "b1", "claim": "   ", "evidence_ids": ["current.revenue"],
             "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.6},
            {"claim_id": "b2", "claim": " ", "evidence_ids": ["current.revenue"],
             "claim_type": "fact_interpretation", "assumptions": [], "confidence": 0.6},
        ],
        "confidence": "medium",
    })

    def ask_local(messages, tools=None, timeout=120, options=None, response_format=None):
        return {"message": {"content": blank_claims},
                "metrics": {"prompt_tokens": 10, "completion_tokens": 10}, "ok": True}

    monkeypatch.setenv("STOCK_ANALYSIS_RESEARCH_PIPELINE_ENABLED", "true")
    company = CANARIES.MATURE_PROFITABLE
    executor = wire_canary(company, tmp_path, monkeypatch)
    try:
        result = run_full_stock_analysis(executor, company.symbol)
        text, _metrics = synthesize_report(result, ask_local, report_detail="compact")
    finally:
        import tools.finance_tools as finance_tools
        finance_tools.set_coordinator(None)
        finance_tools.set_yahoo_coordinator(None)
        finance_tools.set_sec_coordinator(None)

    # No blank bullet reached the rendered report.
    assert "\n- \n" not in text
    assert not any(line.strip() == "-" for line in text.splitlines())

    # And the stage was refused rather than published.
    pipeline = result.research_pipeline or {}
    bull = next((c for c in (pipeline.get("checkpoints") or [])
                 if c.get("stage") == "bull_researcher"), None)
    assert bull is not None
    assert bull.get("status") != "completed", bull
