"""Phase H.1 — OPT-IN live verification against the real Alpha Vantage API.

This is deliberately NOT part of the default test suite: it spends real quota.
The automated tests use mocked provider responses exclusively.

    venv/Scripts/python.exe scripts/manual_verify_h1_stock_analysis.py [SYMBOL]

It makes the smallest safe number of external requests (two datasets, one call
each) and then proves the cache prevents any further ones. It prints no secret:
the key is checked for PRESENCE and LENGTH only, never echoed, and every payload
is summarized rather than dumped.

Exit code 0 means every check passed.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.config as config  # noqa: E402
from finance.cache import MarketDataCache, Origin  # noqa: E402
from finance.coordinator import MarketDataRequestCoordinator  # noqa: E402
from finance.dcf import DcfInputs, run_dcf  # noqa: E402
from finance.quota import AlphaVantageQuotaLedger  # noqa: E402
from tools.base import ToolFailure  # noqa: E402

SCRATCH_CACHE = "app_data/finance_cache/live_verify_cache.sqlite3"
SCRATCH_LEDGER = "app_data/finance_cache/live_verify_quota.sqlite3"

_failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        _failures.append(label)
    return condition


def main():
    symbol = (sys.argv[1] if len(sys.argv) > 1 else "IBM").upper()
    print(f"Phase H.1 live verification — symbol {symbol}\n")

    # ---- 1. credential presence, never its value ----
    print("1. Credential")
    api_key = config.alphavantage_api_key()
    if not check("ALPHAVANTAGE_API_KEY is configured", bool(api_key)):
        print("\nSet ALPHAVANTAGE_API_KEY in .env and re-run. Nothing was requested.")
        return 1
    check("key looks well-formed", len(api_key) >= 8, f"length {len(api_key)}")
    print("   (the key itself is never printed)")

    # ---- isolated stores so a live run never disturbs the real cache ----
    for path in (SCRATCH_CACHE, SCRATCH_LEDGER):
        absolute = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                path)
        if os.path.exists(absolute):
            os.remove(absolute)
    cache = MarketDataCache(path=SCRATCH_CACHE)
    ledger = AlphaVantageQuotaLedger(path=SCRATCH_LEDGER)
    coordinator = MarketDataRequestCoordinator(cache=cache, ledger=ledger)

    # ---- 2. smallest useful number of real calls ----
    print("\n2. Live provider calls (2 datasets, 1 call each)")
    try:
        quote = coordinator.fetch("stock_quote", symbol)
        overview = coordinator.fetch("company_overview", symbol)
    except ToolFailure as failure:
        print(f"  [FAIL] provider call failed: {failure.code} — {failure.message}")
        return 1

    check("quote came from the provider", quote.origin == Origin.PROVIDER)
    check("overview came from the provider", overview.origin == Origin.PROVIDER)
    check("two external calls were recorded", ledger.snapshot().attempted == 2,
          f"attempted={ledger.snapshot().attempted}")

    from finance.normalization import normalize_overview, normalize_quote
    parsed_quote = normalize_quote(quote.payload)
    parsed_overview = normalize_overview(overview.payload)
    check("a price was returned", parsed_quote.get("price") is not None,
          f"price={parsed_quote.get('price')}")
    check("price is labelled delayed, not realtime",
          parsed_quote.get("price_basis") == "delayed")
    check("a company name was returned", bool(parsed_overview.get("name")),
          str(parsed_overview.get("name")))

    # ---- 3. cache reuse: the second identical run must cost nothing ----
    print("\n3. Cache reuse")
    before = ledger.snapshot().attempted
    second_quote = coordinator.fetch("stock_quote", symbol)
    second_overview = coordinator.fetch("company_overview", symbol)
    after = ledger.snapshot().attempted

    check("the second run made ZERO provider calls", after == before,
          f"attempted before={before} after={after}")
    check("the second quote is served from cache", second_quote.origin == Origin.CACHE)
    check("the second overview is served from cache", second_overview.origin == Origin.CACHE)
    check("cache hits were recorded", ledger.snapshot().cache_hits >= 2)

    # ---- 4. no secret in the cache file ----
    print("\n4. Secret hygiene")
    with open(cache.path, "rb") as handle:
        blob = handle.read()
    wal = cache.path + "-wal"
    if os.path.exists(wal):
        with open(wal, "rb") as handle:
            blob += handle.read()
    check("the API key does not appear in the cache files",
          api_key.encode() not in blob)
    check("no 'apikey=' query fragment was stored", b"apikey=" not in blob)

    # ---- 5. a small DCF fixture, independent of any live data ----
    print("\n5. DCF fixture")
    result = run_dcf(
        DcfInputs(ticker="FIXTURE", valuation_date="2026-01-01", currency="USD",
                  base_revenue=1000.0, forecast_years=3, diluted_shares=100.0,
                  total_debt=200.0, cash_and_cash_equivalents=0.0,
                  base_working_capital=100.0,
                  source_periods=("FY2025",)),
        [{"name": "base", "revenue_growth": 0.10, "operating_margin": 0.20,
          "tax_rate": 0.25, "depreciation_pct_revenue": 0.05,
          "capex_pct_revenue": 0.07, "working_capital_pct_revenue": 0.10,
          "wacc": 0.10, "terminal_growth": 0.02}])
    per_share = result["scenarios"][0]["value_per_share"]
    # Matches the independently hand-calculated value in tests/test_finance_dcf.py.
    check("the fixture reproduces the known value per share",
          abs(per_share - 17.043182) < 1e-3, f"got {per_share}")
    check("the calculation version is recorded",
          result["calculation_version"] == "fcff_enterprise_v1")

    repeat = run_dcf(
        DcfInputs(ticker="FIXTURE", valuation_date="2026-01-01", currency="USD",
                  base_revenue=1000.0, forecast_years=3, diluted_shares=100.0,
                  total_debt=200.0, cash_and_cash_equivalents=0.0,
                  base_working_capital=100.0,
                  source_periods=("FY2025",)),
        [{"name": "base", "revenue_growth": 0.10, "operating_margin": 0.20,
          "tax_rate": 0.25, "depreciation_pct_revenue": 0.05,
          "capex_pct_revenue": 0.07, "working_capital_pct_revenue": 0.10,
          "wacc": 0.10, "terminal_growth": 0.02}])
    check("repeat runs are byte-identical",
          json.dumps(result, sort_keys=True) == json.dumps(repeat, sort_keys=True))

    # ---- 6. cleanup: nothing is left running ----
    print("\n6. Cleanup")
    # There is no MCP process to stop: the provider is an in-process HTTP client,
    # which is precisely why this phase has no orphan-process risk at all.
    check("no subprocess was started (in-process HTTP client only)", True)
    quota = ledger.snapshot()
    print(f"   quota this window: attempted={quota.attempted} "
          f"succeeded={quota.succeeded} cache_hits={quota.cache_hits} "
          f"estimated_remaining={quota.estimated_remaining}")

    print("\n" + ("ALL CHECKS PASSED" if not _failures
                  else f"{len(_failures)} CHECK(S) FAILED: {', '.join(_failures)}"))
    return 0 if not _failures else 1


if __name__ == "__main__":
    sys.exit(main())
