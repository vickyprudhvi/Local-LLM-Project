"""Reporting-currency resolver against LIVE SEC company facts.

Same rationale as scripts/run_live_actualization_benchmark.py: a live
issuer's answer changes the day it files, so this stays a script, not a
test. It fetches one issuer's companyfacts and runs `finance.reporting_
currency.resolve_reporting_series` over it, changing no configuration and
switching no default.

USAGE

    python -m scripts.run_live_reporting_currency AAA BBB CCC
"""

import argparse
import sys


def _company_facts(ticker: str):
    from finance.sec_provider import resolve_cik
    from tools import finance_tools

    coordinator = finance_tools.get_sec_coordinator()
    cik, name = resolve_cik(coordinator, ticker)
    outcome = coordinator.fetch("company_facts", ticker, {"cik": cik})
    return outcome.payload or {}, name


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="+")
    args = parser.parse_args(argv)

    from finance import reporting_currency as rc
    from finance import freshness as fr

    for ticker in args.tickers:
        try:
            payload, name = _company_facts(ticker)
        except Exception as exc:  # noqa: BLE001 -- a live-network script may fail any way
            print(f"{ticker}: FETCH FAILED -- {exc}")
            continue

        result = rc.resolve_reporting_series(payload)
        print(f"\n=== {ticker} ({name}) ===")
        print(f"  selection_status               : {result.selection_status}")
        print(f"  selected_reporting_currency     : {result.selected_reporting_currency}")
        print(f"  selected_series_id              : {result.selected_series_id}")
        print(f"  selected_period                 : {result.selected_period}")
        print(f"  candidate_series                :")
        for c in result.candidate_series:
            print(f"      {c.currency:5s} latest={c.latest_period_end} "
                  f"concept={c.concept} form={c.form} lines={c.statement_line_count}")
        print(f"  rejected_series                 :")
        for r in result.rejected_series:
            print(f"      {r.currency:5s} [{r.reason_code}] latest={r.latest_period_end}")
        print(f"  resolution_reason               : {result.resolution_reason}")

        try:
            state = fr.build_current_financial_state(payload, ticker,
                                                      valuation_date="2026-09-11")
            print(f"  state.financial_as_of           : {state.financial_as_of}")
            print(f"  state.valuation_freshness       : {state.valuation_freshness}")
            print(f"  state.reporting_currency_status : {state.reporting_currency_status}")
        except Exception as exc:  # noqa: BLE001
            print(f"  build_current_financial_state FAILED -- {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
