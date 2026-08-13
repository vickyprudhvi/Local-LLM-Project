"""Phase H.4 — OPT-IN live verification of DCF input freshness against the
real SEC EDGAR and Yahoo Finance APIs.

Deliberately NOT part of the default test suite: it makes real external
requests. The automated tests use captured fixtures exclusively — see
tests/test_finance_aos_regression.py (real AOS data), plus
tests/test_finance_period_facts.py / test_finance_freshness.py /
test_finance_guidance.py for the synthetic unit cases.

    venv/Scripts/python.exe scripts/manual_verify_h4_dcf_freshness.py [SYMBOL]

Prints the provenance the phase spec asks to see: which annual and quarterly
filings were identified, the balance-sheet date actually used, current cash
and debt, the trailing-twelve-month period, current management guidance,
historical CAGR, the proposed year-by-year growth path with its evidence
ids, the final DCF inputs, and the valuation freshness status.

Nothing here forces a valuation toward any external analysis — it reports
what the deterministic pipeline produced from the corrected inputs.

Exit code 0 means every structural check passed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import tools.config as config
from finance.dcf import AssumptionSourceType
from finance.freshness import ValuationFreshness
from finance.workflow import (
    build_compact_synthesis_payload,
    render_compact_report,
    run_full_stock_analysis,
)
from tools.executor import ToolExecutor
from tools.registry import default_registry

SYMBOL = (sys.argv[1] if len(sys.argv) > 1 else "AOS").upper()

failures = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def money(value):
    return f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)


def main():
    if not config.sec_edgar_enabled():
        print("SEC_EDGAR_ENABLED is not set; nothing to verify.")
        return 1
    if not (config.sec_user_agent() or "").strip():
        print("SEC_USER_AGENT is not configured (presence checked, never echoed).")
        return 1

    print(f"=== Phase H.4 live verification — {SYMBOL} ===\n")
    executor = ToolExecutor(default_registry())
    result = run_full_stock_analysis(executor, SYMBOL)

    state = result.facts.get("current_financial_state") or {}
    basis = result.facts.get("dcf_financial_basis") or {}
    dcf = result.facts.get("dcf") or {}

    # ---- filings identified ----
    print("-- filings identified --")
    print(f"  latest annual period    : {state.get('latest_annual_period')}")
    print(f"  latest quarterly period : {state.get('latest_quarterly_period')}")
    print(f"  balance sheet as of     : {state.get('financial_as_of')}")
    check("an annual period was identified", bool(state.get("latest_annual_period")))
    check("a balance-sheet date was identified", bool(state.get("financial_as_of")))

    # ---- the equity bridge ----
    print("\n-- current balance sheet (equity bridge) --")
    for name in ("cash_and_cash_equivalents", "short_term_investments",
                 "short_term_debt", "current_portion_of_long_term_debt",
                 "long_term_debt", "preferred_equity", "minority_interest"):
        entry = (state.get("balance_sheet") or {}).get(name) or {}
        print(f"  {name:34} {money(entry.get('value')):>18}  "
              f"as_of={entry.get('as_of_date')}  [{entry.get('freshness_status')}]")
    total_debt = state.get("total_debt") or {}
    print(f"  {'total_debt (derived)':34} {money(total_debt.get('value')):>18}  "
          f"as_of={total_debt.get('as_of_date')}")
    print(f"  {'net_debt':34} {money(state.get('net_debt')):>18}")

    annual_end = state.get("latest_annual_period")
    bs_date = state.get("financial_as_of")
    if annual_end and bs_date and bs_date > annual_end:
        check("the equity bridge uses the QUARTERLY balance sheet, not the annual one",
              total_debt.get("as_of_date") == bs_date,
              f"debt as of {total_debt.get('as_of_date')}, latest annual was {annual_end}")
    else:
        print("  (no quarterly balance sheet newer than the annual one for this issuer)")

    # ---- trailing twelve months ----
    print("\n-- flows --")
    for name, entry in (state.get("flows") or {}).items():
        print(f"  {name:32} {money(entry.get('value')):>18}  "
              f"{entry.get('period_start')}..{entry.get('as_of_date')}  [{entry.get('source')}]")
    revenue = (state.get("flows") or {}).get("revenue") or {}
    check("base revenue came from the freshest available flow basis",
          dcf.get("base_revenue") == revenue.get("value"),
          f"{basis.get('base_revenue_basis')} "
          f"{basis.get('flow_period_start')}..{basis.get('flow_period_end')}")

    # ---- guidance ----
    print("\n-- current management guidance --")
    guidance = result.facts.get("management_guidance") or {}
    if guidance.get("metrics"):
        print(f"  source: {guidance.get('source_document')}")
        for name, metric in guidance["metrics"].items():
            print(f"    {name:32} {metric['low']} .. {metric['high']}  "
                  f"[{metric['unit']}/{metric['basis']}] fy={metric['fiscal_year']}")
        superseded = result.facts.get("superseded_guidance") or []
        print(f"  superseded releases retained: {[r['guidance_date'] for r in superseded]}")
    else:
        print("  unavailable (this is a supported outcome — see section 20)")

    # ---- historical context vs the forecast ----
    print("\n-- historical context --")
    cagr = (result.facts.get("fundamental_metrics") or {}).get("revenue_cagr") or {}
    print(f"  historical revenue CAGR : {cagr.get('value')}")

    base = next((s for s in dcf.get("scenarios") or [] if s["scenario"] == "base"), None)
    if base is None:
        check("a base scenario was produced", False)
        return 1

    print("\n-- proposed year-by-year growth path (base) --")
    provenance = base["assumptions"]["assumption_provenance"]["revenue_growth"]
    for index, value in enumerate(base["assumptions"]["revenue_growth"], start=1):
        print(f"    year {index}: {value:.4%}")
    print(f"  source_type   : {provenance.get('source_type')}")
    print(f"  evidence_ids  : {provenance.get('source_evidence_ids')}")
    print(f"  derivation    : {str(provenance.get('derivation'))[:220]}")

    growth_path = base["assumptions"]["revenue_growth"]
    check("growth is a per-year path, not one scalar",
          isinstance(growth_path, list) and len(growth_path) == base["forecast_years"])
    check("the forecast converges on the scenario's own terminal growth",
          abs(growth_path[-1] - base["assumptions"]["terminal_growth"]) < 1e-6)
    if cagr.get("value") is not None and guidance.get("metrics"):
        check("the historical CAGR was NOT copied into year-1 growth",
              abs(growth_path[0] - cagr["value"]) > 1e-9,
              f"year1={growth_path[0]:.4%} vs CAGR={cagr['value']:.4%}")
        check("year-1 growth is anchored on management guidance",
              provenance.get("source_type") == AssumptionSourceType.MANAGEMENT_GUIDANCE)

    print("\n-- assumption provenance (base) --")
    for field, entry in base["assumptions"]["assumption_provenance"].items():
        print(f"    {field:30} {entry.get('source_type'):26} clamped={entry.get('clamped')}")
    check("every assumption carries a recognized source_type",
          all(e.get("source_type") in AssumptionSourceType.ALL
              for e in base["assumptions"]["assumption_provenance"].values()))

    # ---- final DCF inputs and freshness ----
    print("\n-- final DCF inputs --")
    print(f"  base_revenue    : {money(dcf.get('base_revenue'))}")
    print(f"  net_debt        : {money(dcf.get('net_debt'))}")
    print(f"  diluted_shares  : {money(dcf.get('diluted_shares'))}")
    print(f"  value_per_share : {dcf.get('value_per_share')}")
    print(f"  validation      : {dcf.get('validation_status')}")

    print("\n-- freshness --")
    print(f"  data_completeness   : {state.get('data_completeness')}")
    print(f"  valuation_freshness : {state.get('valuation_freshness')}")
    for finding in state.get("findings") or []:
        print(f"    [{finding['code']}] {finding['message'][:180]}")
    check("valuation freshness was classified",
          state.get("valuation_freshness") in ValuationFreshness.ALL)

    # ---- the compact report says which periods it used ----
    compact = build_compact_synthesis_payload(result)
    text = render_compact_report(result, compact, pipeline_result=None)
    print("\n-- compact report, Valuation section --")
    inside = False
    for line in text.splitlines():
        if line.startswith("## Valuation"):
            inside = True
        elif inside and line.startswith("## "):
            break
        if inside:
            print(f"  {line}")
    check("the report states the financial base",
          "Financial base:" in text or "Balance sheet:" in text)
    check("the report states the guidance position", "Management guidance:" in text)

    print(f"\n=== {'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + str(failures)} ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
