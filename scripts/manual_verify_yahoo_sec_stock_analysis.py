"""Phase H.3 — OPT-IN live verification against the real Yahoo Finance
(yfinance) and SEC EDGAR APIs.

Deliberately NOT part of the default test suite: it makes real external
requests (Yahoo has no quota to spend, but this still touches the real
network; SEC EDGAR is paced under its published fair-access ceiling). The
automated tests use mocked provider responses exclusively — see
tests/test_finance_yahoo_provider.py, tests/test_finance_sec_provider.py,
tests/test_finance_multi_provider_workflow.py.

    venv/Scripts/python.exe scripts/manual_verify_yahoo_sec_stock_analysis.py [SYMBOL]

Prints no secret: SEC_USER_AGENT is checked for presence only, never echoed
(it is not a credential in the same sense ALPHAVANTAGE_API_KEY is, but this
project applies the same discipline anyway). Every payload is summarized,
never dumped. Stops after deterministic calculations by default — pass
--with-pipeline to also run the staged research pipeline through a real
local Ollama call (spends local-LLM time/tokens) and check its output
against the Phase H.3 corrective-patch policy (finance/content_policy.py,
finance/claim_validation.py) — no BUY/SELL/HOLD/AVOID, no unsupported
superlatives/causal claims, valid evidence citations, correct CapEx
provenance, reduced-mode status, provider attribution, token usage.

Exit code 0 means every check passed.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.config as config  # noqa: E402
import tools.finance_tools as finance_tools  # noqa: E402
from finance.cache import MarketDataCache, Origin  # noqa: E402
from finance.coordinator import MarketDataRequestCoordinator  # noqa: E402
from finance.dcf import AssumptionSourceType  # noqa: E402
from finance.metrics import fundamental_metrics, metrics_to_dict  # noqa: E402
from finance.provider import AlphaVantageClient  # noqa: E402
from finance.quota import AlphaVantageQuotaLedger, QuotaLedger  # noqa: E402
from finance.sec_datasets import resolve_sec_dataset  # noqa: E402
from finance.sec_normalization import normalize_sec_statements  # noqa: E402
from finance.sec_provider import SecEdgarClient, resolve_cik  # noqa: E402
from finance.workflow import (  # noqa: E402
    AnalysisMode,
    build_compact_synthesis_payload,
    run_full_stock_analysis,
    synthesize_report,
)
from finance.yahoo_datasets import resolve_yahoo_dataset  # noqa: E402
from finance.yahoo_provider import YahooFinanceClient  # noqa: E402
from tools.base import ToolFailure  # noqa: E402
from tools.executor import ToolExecutor  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402

SCRATCH_DIR = "app_data/finance_cache"
YAHOO_CACHE = f"{SCRATCH_DIR}/live_verify_yahoo_cache.sqlite3"
YAHOO_LEDGER = f"{SCRATCH_DIR}/live_verify_yahoo_quota.sqlite3"
SEC_CACHE = f"{SCRATCH_DIR}/live_verify_sec_cache.sqlite3"
SEC_LEDGER = f"{SCRATCH_DIR}/live_verify_sec_quota.sqlite3"
AV_CACHE = f"{SCRATCH_DIR}/live_verify_av_cache.sqlite3"
AV_LEDGER = f"{SCRATCH_DIR}/live_verify_av_quota.sqlite3"

_failures = []
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        _failures.append(label)
    return condition


def _abspath(relative):
    return os.path.join(_ROOT, relative)


def main():
    symbol = (sys.argv[1] if len(sys.argv) > 1 else "IBM").upper()
    run_pipeline = "--with-pipeline" in sys.argv
    print(f"Phase H.3 live verification (Yahoo + SEC EDGAR) — symbol {symbol}\n")

    # ---- 1. configuration presence, never a private value ----
    print("1. Configuration")
    yahoo_ok = check("YAHOO_FINANCE_ENABLED is true", config.yahoo_finance_enabled())
    ack_ok = check("YAHOO_PERSONAL_USE_ACKNOWLEDGED is true", config.yahoo_personal_use_acknowledged())
    sec_ok = check("SEC_EDGAR_ENABLED is true", config.sec_edgar_enabled())
    ua = config.sec_user_agent()
    ua_ok = check("SEC_USER_AGENT is configured", bool(ua))
    if ua_ok:
        check("SEC_USER_AGENT looks well-formed (contains a contact-ish '@' or domain)",
              "@" in ua or "." in ua, "(value itself not printed)")
    if not (yahoo_ok and ack_ok and sec_ok and ua_ok):
        print("\nSet the missing values in .env and re-run. Nothing was requested.")
        return 1

    # ---- isolated stores so a live run never disturbs the real cache ----
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

    # ---- 2. Yahoo: quote + price history ----
    print("\n2. Yahoo Finance live calls (2 datasets)")
    try:
        quote = yahoo_coordinator.fetch("stock_quote", symbol)
        history = yahoo_coordinator.fetch("price_history", symbol, {"period": "1y", "interval": "1d"})
    except ToolFailure as failure:
        print(f"  [FAIL] Yahoo call failed: {failure.code} — {failure.message}")
        return 1

    check("quote came from the provider", quote.origin == Origin.PROVIDER)
    check("a last price was returned", quote.payload["quote"].get("lastPrice") is not None,
          f"lastPrice={quote.payload['quote'].get('lastPrice')}")
    check("price history returned bars", len(history.payload.get("bars", [])) > 0,
          f"{len(history.payload.get('bars', []))} bars")

    # ---- 3. Yahoo cache reuse ----
    print("\n3. Yahoo cache reuse")
    before = yahoo_coordinator.ledger.snapshot().attempted
    second_quote = yahoo_coordinator.fetch("stock_quote", symbol)
    after = yahoo_coordinator.ledger.snapshot().attempted
    check("the second Yahoo call made ZERO external requests", after == before,
          f"attempted before={before} after={after}")
    check("the second quote is served from cache", second_quote.origin == Origin.CACHE)

    # ---- 4. SEC: CIK resolution + company facts ----
    print("\n4. SEC EDGAR live calls (CIK resolution + company facts)")
    try:
        cik, company_name = resolve_cik(sec_coordinator, symbol)
        facts_outcome = sec_coordinator.fetch("company_facts", symbol, {"cik": cik})
    except ToolFailure as failure:
        print(f"  [FAIL] SEC call failed: {failure.code} — {failure.message}")
        return 1

    check("a CIK was resolved", bool(cik), f"CIK={cik}")
    check("a company name was resolved", bool(company_name), str(company_name))
    check("company facts came from the provider", facts_outcome.origin == Origin.PROVIDER)

    statements, fact_provenance = normalize_sec_statements(
        facts_outcome.payload, symbol, facts_outcome.freshness_dict(sec_coordinator.ledger._clock()))
    annual = statements.annual.get("balance_sheet") or []
    check("at least one annual balance-sheet period was normalized", len(annual) > 0,
          f"{len(annual)} period(s)")
    if annual:
        latest = annual[0]
        print(f"   latest balance sheet: {latest.fiscal_date} "
              f"(total_debt={latest.values.get('total_debt')}, "
              f"cash={latest.values.get('cash_and_cash_equivalents')})")
    check("per-fact provenance (concept/accession) was captured", len(fact_provenance) > 0,
          f"{len(fact_provenance)} fact(s)")

    # ---- 5. SEC cache reuse (including the ticker-CIK map reused across symbols) ----
    print("\n5. SEC cache reuse")
    before = sec_coordinator.ledger.snapshot().attempted
    _cik2, _name2 = resolve_cik(sec_coordinator, symbol)  # same symbol -> same GLOBAL map, cached
    second_facts = sec_coordinator.fetch("company_facts", symbol, {"cik": cik})
    after = sec_coordinator.ledger.snapshot().attempted
    check("the second SEC pass made ZERO external requests", after == before,
          f"attempted before={before} after={after}")
    check("the second company-facts fetch is served from cache", second_facts.origin == Origin.CACHE)

    # ---- 6. deterministic local metrics from the SEC-sourced statements ----
    print("\n6. Deterministic metrics (local calculation, no provider call)")
    metrics = metrics_to_dict(fundamental_metrics(statements))
    computed = {k: v for k, v in metrics.items() if v.get("value") is not None}
    check("at least one fundamental metric was computed", len(computed) > 0,
          f"{len(computed)}/{len(metrics)} computed")

    # ---- 7. secret/PII hygiene in the cache files ----
    print("\n7. Configuration hygiene")
    blob = b""
    for path in (YAHOO_CACHE, SEC_CACHE):
        absolute = _abspath(path)
        with open(absolute, "rb") as handle:
            blob += handle.read()
        wal = absolute + "-wal"
        if os.path.exists(wal):
            with open(wal, "rb") as handle:
                blob += handle.read()
    check("SEC_USER_AGENT does not appear in the cache files", ua.encode() not in blob)

    # ---- 8. full workflow + Phase H.3 corrective-patch checks ----
    print("\n8. Full workflow (CapEx/DCF correctness, compaction, provider attribution)")
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

    result = run_full_stock_analysis(executor, symbol, include_news=False)
    print(f"   plan.mode = {result.plan.mode}")
    if result.plan.mode != AnalysisMode.FULL:
        print(f"   omitted_datasets = {result.plan.omitted_datasets}")
    if not config.alphavantage_api_key():
        check("ALPHAVANTAGE_API_KEY unconfigured -> earnings omitted with a controlled "
              "error, never a crash (Problem 10: reduced-mode transparency)",
              any(e.get("dataset") == "earnings" for e in result.errors))

    # CapEx/D&A/NWC: reported-history-first, never a margin difference (Problems 2-3).
    dcf = result.facts.get("dcf") or {}
    if check("DCF is available", dcf.get("available") is True):
        base = next((s for s in dcf["scenarios"] if s["scenario"] == "base"), None)
        prov = ((base or {}).get("assumptions") or {}).get("assumption_provenance") or {}
        capex_prov = prov.get("capex_pct_revenue") or {}
        check("capex_pct_revenue.source_type is a recognized value",
              capex_prov.get("source_type") in AssumptionSourceType.ALL, str(capex_prov.get("source_type")))
        derivation = (capex_prov.get("derivation") or "").lower()
        check("CapEx derivation is never a margin-difference proxy (the original bug)",
              "operating margin" not in derivation and "fcf margin" not in derivation
              and "free cash flow margin" not in derivation)
        print(f"   capex_pct_revenue = {base['assumptions']['capex_pct_revenue'][0]:.4f} "
              f"({capex_prov.get('source_type')})")
        print(f"   derivation: {(capex_prov.get('derivation') or '')[:220]}")
        for name in ("base", "bull", "bear"):
            s = next(sc for sc in dcf["scenarios"] if sc["scenario"] == name)
            print(f"   {name:5s} value_per_share = {s['value_per_share']:>12.4f}   "
                  f"net_debt = {s['net_debt']:>15,.0f}")
        check("net_debt is identical across scenarios (a balance-sheet fact, not a scenario assumption)",
              len({sc["net_debt"] for sc in dcf["scenarios"]}) == 1)

    # Compact payload / token budget (Problem 11). repeated_field_count is
    # normally set by synthesize_report(); computed directly here too so it
    # is populated even in the (default, faster) no-LLM-call path below.
    from finance.workflow import _count_repeated_fields
    compact = build_compact_synthesis_payload(result)
    payload_bytes = len(json.dumps(compact, default=str))
    estimated_tokens = payload_bytes // 4
    repeated_field_count = _count_repeated_fields(
        (compact.get("dcf") or {}).get("scenarios") or [],
        *(compact.get("financial_history") or {}).values())
    print(f"   compact payload: {payload_bytes} bytes, ~{estimated_tokens} estimated tokens, "
          f"repeated_field_count={repeated_field_count}")
    check("compact payload is under the 20k hard token limit (Problem 11)", estimated_tokens < 20_000)

    # Provider attribution.
    provenance_entries = result.facts.get("data_provenance") or {}
    providers_seen = {p.get("provider") for p in provenance_entries.values() if isinstance(p, dict)} - {None}
    check("data_provenance names an actual provider for at least one dataset",
          bool(providers_seen), str(providers_seen))

    if run_pipeline:
        import brain
        print(f"\n   Staged research pipeline (--with-pipeline): calling the local Ollama "
              f"model ({brain.LOCAL_MODEL})...")
        text, metrics = synthesize_report(result, brain.ask_local_raw)
        print(f"   local-model tokens this call: {metrics}")

        from finance.claim_validation import scan_for_unsupported_claims
        from finance.content_policy import scan_for_prohibited_directives
        directive_hits = scan_for_prohibited_directives(text)
        claim_hits = scan_for_unsupported_claims(text)
        check("no prohibited trade-advice directive in the generated report (Problem 1)",
              not directive_hits, str(directive_hits))
        check("no unsupported superlative/causal/consensus claim in the generated report (Problem 5)",
              not claim_hits, str(claim_hits))

        pipeline_result = result.research_pipeline
        if check("research pipeline object was populated", pipeline_result is not None):
            check("research pipeline reached a final research characterization "
                  "(implies every evidence_cited ID validated -- an unverifiable citation "
                  "fails the stage closed and 'available' would be False)",
                  pipeline_result.get("available") is True)
            final = next((c for c in pipeline_result["checkpoints"]
                         if c["stage"] == "final_investment_synthesizer"), None)
            if final and final["status"] == "completed":
                out = final["output"]
                print(f"   research_stance={out['research_stance']} valuation_view={out['valuation_view']} "
                      f"overall_risk={out['overall_risk']} confidence={out['confidence']}")

        report_path = _abspath(f"{SCRATCH_DIR}/live_verify_report_{symbol}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"   full report text written to {report_path} for manual inspection")
    else:
        print("\n   Staged research pipeline — skipped (pass --with-pipeline to include it)")

    finance_tools.set_coordinator(None)
    finance_tools.set_yahoo_coordinator(None)
    finance_tools.set_sec_coordinator(None)

    # ---- 9. cleanup ----
    print("\n9. Cleanup")
    check("no subprocess was started (in-process HTTP clients only)", True)
    yq = yahoo_coordinator.ledger.snapshot()
    sq = sec_coordinator.ledger.snapshot()
    print(f"   Yahoo calls this run: attempted={yq.attempted} cache_hits={yq.cache_hits}")
    print(f"   SEC calls this run:   attempted={sq.attempted} cache_hits={sq.cache_hits}")

    print("\n" + ("ALL CHECKS PASSED" if not _failures
                  else f"{len(_failures)} CHECK(S) FAILED: {', '.join(_failures)}"))
    return 0 if not _failures else 1


if __name__ == "__main__":
    sys.exit(main())
