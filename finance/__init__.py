"""Phase H.1 — stock analysis: market data, caching, quota, valuation.

Layering (each layer depends only on the ones above it):

    tools/finance_tools.py      BaseTools -> ToolRegistry -> ToolExecutor
    finance/workflow.py         FullStockAnalysis orchestration
    finance/dcf.py              deterministic FCFF valuation (pure arithmetic)
    finance/metrics.py          locally-calculated fundamentals
    finance/technicals.py       locally-calculated indicators from cached OHLCV
    finance/normalization.py    provider payload -> provenance-carrying records
    finance/coordinator.py      cache-first, single-flight, bounded retry
    finance/quota.py            external-call ledger
    finance/cache.py            SQLite store
    finance/provider.py         Alpha Vantage HTTP client (tools/http_safety.py)

Nothing in this package writes to the network except `provider.py`, and nothing
executes a tool: execution authority stays with `tools/executor.py`.

The provider credential is read from config at call time and never enters a
cache key, a cache record, a log line, or an exception message.
"""
