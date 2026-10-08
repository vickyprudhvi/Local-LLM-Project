"""Phase H.32 — live verification against the REAL TSLA filings/market data.

OPT-IN, not part of the default test suite: makes real Yahoo Finance, SEC
EDGAR, and (for the staged research pipeline) local-model calls. Verifies,
against the live issuer, the three H.32 fixes:

  1. guidance source coverage (current 10-Q MD&A, not only earnings releases)
  2. financing-event funded-vs-committed semantics
  3. DCF model-limitation vs company-risk attribution

and general report-content quality (section 4 of the verification request).

Uses the existing production configuration unconditionally -- no mode is
overridden here. Everything below is READ-ONLY against real providers; it
changes no code and writes nothing but a scratch cache and a copy of the
rendered report for manual inspection.

    venv/Scripts/python.exe scripts/manual_verify_h32_tsla_live.py

Exit code 0 means every check passed.
"""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.config as config  # noqa: E402
import tools.finance_tools as finance_tools  # noqa: E402
from finance.cache import MarketDataCache  # noqa: E402
from finance.coordinator import MarketDataRequestCoordinator  # noqa: E402
from finance.provider import AlphaVantageClient  # noqa: E402
from finance.quota import AlphaVantageQuotaLedger, QuotaLedger  # noqa: E402
from finance.sec_datasets import resolve_sec_dataset  # noqa: E402
from finance.sec_provider import SecEdgarClient  # noqa: E402
from finance.workflow import (  # noqa: E402
    AnalysisMode,
    build_compact_synthesis_payload,
    run_full_stock_analysis,
    synthesize_report,
)
from finance.yahoo_datasets import resolve_yahoo_dataset  # noqa: E402
from finance.yahoo_provider import YahooFinanceClient  # noqa: E402
from tools.executor import ToolExecutor  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402

SYMBOL = "TSLA"
SCRATCH_DIR = "app_data/finance_cache"
YAHOO_CACHE = f"{SCRATCH_DIR}/h32_live_yahoo_cache.sqlite3"
YAHOO_LEDGER = f"{SCRATCH_DIR}/h32_live_yahoo_quota.sqlite3"
SEC_CACHE = f"{SCRATCH_DIR}/h32_live_sec_cache.sqlite3"
SEC_LEDGER = f"{SCRATCH_DIR}/h32_live_sec_quota.sqlite3"
AV_CACHE = f"{SCRATCH_DIR}/h32_live_av_cache.sqlite3"
AV_LEDGER = f"{SCRATCH_DIR}/h32_live_av_quota.sqlite3"

_failures = []
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        _failures.append(label)
    return condition


def note(label, detail=""):
    print(f"  [INFO] {label}" + (f" — {detail}" if detail else ""))


def _abspath(relative):
    return os.path.join(_ROOT, relative)


def main():
    print(f"Phase H.32 live verification — symbol {SYMBOL}\n")

    print("0. Production configuration (unmodified)")
    check("FINANCE_DOCUMENT_PIPELINE_MODE is 'v1'",
          config.finance_document_pipeline_mode() == "v1",
          config.finance_document_pipeline_mode())
    check("FINANCE_SEC_PROVIDER is 'current'",
          config.finance_sec_provider() == "current", config.finance_sec_provider())
    check("guidance periodic-filing coverage is enabled (H.32 default)",
          config.guidance_periodic_filing_coverage_enabled() is True)

    for path in (YAHOO_CACHE, YAHOO_LEDGER, SEC_CACHE, SEC_LEDGER, AV_CACHE, AV_LEDGER):
        absolute = _abspath(path)
        if os.path.exists(absolute):
            os.remove(absolute)

    yahoo_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=YAHOO_CACHE, base_dir=_ROOT),
        ledger=QuotaLedger(path=YAHOO_LEDGER, base_dir=_ROOT, daily_limit=1_000_000),
        client=YahooFinanceClient(), provider_id="yahoo",
        dataset_resolver=resolve_yahoo_dataset, min_request_interval_ms=0)
    sec_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=SEC_CACHE, base_dir=_ROOT),
        ledger=QuotaLedger(path=SEC_LEDGER, base_dir=_ROOT, daily_limit=1_000_000),
        client=SecEdgarClient(), provider_id="sec", dataset_resolver=resolve_sec_dataset,
        min_request_interval_ms=config.sec_min_request_interval_ms())
    av_coordinator = MarketDataRequestCoordinator(
        cache=MarketDataCache(path=AV_CACHE, base_dir=_ROOT),
        ledger=AlphaVantageQuotaLedger(path=AV_LEDGER, base_dir=_ROOT, daily_limit=100),
        client=AlphaVantageClient())

    finance_tools.set_coordinator(av_coordinator)
    finance_tools.set_yahoo_coordinator(yahoo_coordinator)
    finance_tools.set_sec_coordinator(sec_coordinator)
    registry = ToolRegistry()
    for tool_cls in finance_tools.ALL_FINANCE_TOOL_CLASSES:
        registry.register(tool_cls())
    executor = ToolExecutor(registry)

    print("\n1. Running the full TSLA stock analysis (real Yahoo + SEC + Alpha Vantage)")
    result = run_full_stock_analysis(executor, SYMBOL, include_news=False)
    print(f"   plan.mode = {result.plan.mode}")
    if result.plan.mode != AnalysisMode.FULL:
        print(f"   omitted_datasets = {result.plan.omitted_datasets}")

    # ------------------------------------------------------------------
    # GOAL 1 -- guidance verification
    # ------------------------------------------------------------------
    print("\n2. GOAL 1 — guidance verification")
    guidance = result.facts.get("management_guidance")
    releases_examined = result.facts.get("guidance_releases_examined")
    periodic_examined = result.facts.get("guidance_periodic_filings_examined")
    note("guidance_releases_examined", str(releases_examined))
    note("guidance_periodic_filings_examined", str(periodic_examined))
    check("the periodic-filing (10-Q/10-K MD&A) guidance path was examined",
          bool(periodic_examined), str(periodic_examined))

    boundary = None
    if not guidance:
        note("no current guidance resolved; see warnings for extraction notes",
             json.dumps([w for w in (result.warnings or [])
                        if "GUIDANCE" in w.upper()])[:2000])
        if not periodic_examined:
            boundary = "SOURCE_SELECTION"
        else:
            boundary = "SECTION_SELECTION / V1_EXTRACTION / METRIC_IDENTITY / VALIDATION / "\
                      "CURRENT_GUIDANCE_RESOLUTION (see notes above for which)"
        check("guidance is currently absent — see boundary classification below", False,
              f"boundary candidate: {boundary}")
    else:
        metrics = guidance.get("metrics") or {}
        capex = metrics.get("capital_expenditure")
        note("guidance metrics present", ", ".join(metrics.keys()) or "(none)")
        check("a capital_expenditure guidance statement is present", capex is not None)
        if capex:
            note("capex guidance", json.dumps(capex)[:500])
            check("capex guidance carries a source accession",
                  bool(capex.get("source_accession")))
            check("capex guidance carries a fiscal period/year",
                  bool(capex.get("fiscal_year") or capex.get("fiscal_period")))
            check("capex guidance has a numeric low bound",
                  capex.get("low") is not None)
        note("source_document", guidance.get("source_document") or guidance.get("document") or "")
        note("accession", guidance.get("accession") or "")
        note("filed", guidance.get("filed") or "")

    # ------------------------------------------------------------------
    # GOAL 2 -- financing-event verification
    # ------------------------------------------------------------------
    print("\n3. GOAL 2 — financing-event verification")
    events = (result.facts.get("current_financial_state") or {}).get(
        "post_balance_sheet_events") or []
    note("post_balance_sheet_events count", str(len(events)))
    sept_events = [e for e in events if (e.get("filed") or "").startswith("2026-09")]
    for e in events:
        note("event", json.dumps({k: e.get(k) for k in
                                  ("filed", "form", "event_type", "funded", "description")}))
    if not sept_events:
        note("no post-balance-sheet event filed in September 2026 was found in this live run",
             "(filing calendar may differ from when the original bug was observed)")
    for e in sept_events:
        desc = e.get("description") or ""
        impact = e.get("impact") or {}
        funded = e.get("funded")
        check(f"[{e.get('filed')}] description does not claim 'issuance of debt' unless funded",
              funded is True or "issuance of debt" not in desc, desc)
        check(f"[{e.get('filed')}] affects_debt is only True when funded is True",
              impact.get("affects_debt") is not True or funded is True,
              f"funded={funded} affects_debt={impact.get('affects_debt')}")
        check(f"[{e.get('filed')}] funded is a tri-state, never a silent assumption",
              funded in (True, False, None), str(funded))
        if e.get("event_type") == "ISSUER_DEBT_ISSUANCE":
            check(f"[{e.get('filed')}] a debt-shaped event is still retained for "
                  "reassessment regardless of funded status",
                  impact.get("requires_reassessment") is True)

    # ------------------------------------------------------------------
    # GOAL 3 -- DCF risk attribution (requires the staged research pipeline)
    # ------------------------------------------------------------------
    print("\n4. GOAL 3 — DCF / risk attribution (staged research pipeline, real local model)")
    dcf = result.facts.get("dcf") or {}
    validation_status = dcf.get("validation_status")
    note("dcf.validation_status", str(validation_status))
    note("dcf.available", str(dcf.get("available")))

    import brain
    print(f"   calling the local model ({brain.LOCAL_MODEL}) for the staged research pipeline...")
    text, usage = synthesize_report(result, brain.ask_local_raw)
    print(f"   local-model tokens this call: {usage}")

    report_path = _abspath(f"{SCRATCH_DIR}/h32_live_verify_report_{SYMBOL}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"   full report text written to {report_path} for manual inspection")

    pipeline_result = result.research_pipeline
    risk_items = []
    if check("research pipeline object was populated", pipeline_result is not None):
        checkpoints = pipeline_result.get("checkpoints") or []
        risk_cp = next((c for c in checkpoints if c["stage"] == "risk_reviewer"), None)
        if risk_cp and risk_cp.get("status") == "completed":
            risk_items = (risk_cp["output"] or {}).get("key_risks") or []
            model_risk = (risk_cp["output"] or {}).get("model_risk")
            note("risk_reviewer.model_risk", str(model_risk))
            for item in risk_items:
                note("risk item", f"[{item.get('category')}] {item.get('risk')}")
        final_cp = next((c for c in checkpoints if c["stage"] == "final_investment_synthesizer"),
                        None)
        if final_cp and final_cp.get("status") == "completed":
            out = final_cp["output"]
            note("final_investment_synthesizer",
                 f"research_stance={out.get('research_stance')} "
                 f"valuation_view={out.get('valuation_view')} "
                 f"overall_risk={out.get('overall_risk')}")
            if validation_status and validation_status != "DCF_VALID":
                check("valuation_view reflects the withheld DCF ('model_invalid' or "
                      "'insufficient_data')",
                      out.get("valuation_view") in ("model_invalid", "insufficient_data"),
                      out.get("valuation_view"))

    if validation_status and validation_status != "DCF_VALID":
        check("no risk item restates the DCF failure as a company cash-flow-deficit claim",
              not any(
                  ("cash-flow deficit" in (i.get("risk") or "").lower()
                   or "cash flow deficit" in (i.get("risk") or "").lower())
                  and i.get("category") == "COMPANY_RISK"
                  for i in risk_items))
        model_limitation_present = any(
            i.get("category") in ("VALUATION_METHOD_LIMITATION", "MODEL_RISK")
            for i in risk_items)
        note("a VALUATION_METHOD_LIMITATION/MODEL_RISK item is present",
             str(model_limitation_present))

    # ------------------------------------------------------------------
    # GOAL 4 -- report quality
    # ------------------------------------------------------------------
    print("\n5. GOAL 4 — report content quality")
    from finance.claim_validation import scan_for_unsupported_claims
    from finance.content_policy import scan_for_prohibited_directives
    from finance.workflow import _COMPACT_DISCLAIMER

    # The deterministic, renderer-authored disclaimer footer explains in
    # plain English that the report never gives a position size and never
    # assumes the reader holds a position -- which necessarily uses the
    # literal words "you"/"position size" to say so. It is not model
    # output, is identical on every report regardless of symbol, and is
    # never passed through the content-policy repair loop (that loop
    # scopes to each stage's own JSON fields -- see content_policy.py's
    # module docstring). Scanning it here would fail on every successful
    # run for every ticker, not just this one.
    scannable_text = text.replace(_COMPACT_DISCLAIMER, "")
    directive_hits = scan_for_prohibited_directives(scannable_text)
    claim_hits = scan_for_unsupported_claims(text)
    check("no prohibited trade-advice directive in the generated report "
          "(excluding the fixed disclaimer footer)",
          not directive_hits, str(directive_hits))
    check("no unsupported superlative/causal/consensus claim in the generated report",
          not claim_hits, str(claim_hits))

    lines = text.splitlines()
    blank_bullets = [ln for ln in lines if ln.strip() in ("-", "*", "•", "- ", "* ")]
    check("no blank bullets in the rendered report", not blank_bullets, str(blank_bullets[:5]))

    diagnostic_tokens = ("None examined", "null", "NoneType", "Traceback", "<class ",
                         "KeyError", "AttributeError")
    found_tokens = [t for t in diagnostic_tokens if t in text]
    check("no internal diagnostic tokens leaked into the report", not found_tokens,
          str(found_tokens))

    quote_section_timed_out = "quote" in (result.plan.omitted_datasets or ())
    if quote_section_timed_out:
        check("no fabricated market price appears when the quote timed out",
              "lastPrice" not in text and "$0.00" not in text)
    else:
        note("quote did not time out in this run", "")

    readiness = (result.facts.get("research_readiness") or {}).get("status") \
        if isinstance(result.facts.get("research_readiness"), dict) else None
    note("research_readiness.status", str(readiness))
    if validation_status and validation_status != "DCF_VALID":
        check("research readiness is NOT_READY/limited when valuation evidence is withheld",
              readiness in ("NOT_READY", "LIMITED", None) or readiness != "READY",
              str(readiness))

    finance_tools.set_coordinator(None)
    finance_tools.set_yahoo_coordinator(None)
    finance_tools.set_sec_coordinator(None)

    print("\n" + ("ALL CHECKS PASSED" if not _failures
                  else f"{len(_failures)} CHECK(S) FAILED: {', '.join(_failures)}"))
    return 0 if not _failures else 1


if __name__ == "__main__":
    sys.exit(main())
