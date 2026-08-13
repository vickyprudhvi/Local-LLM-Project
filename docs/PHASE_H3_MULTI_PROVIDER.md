# Phase H.3 — Yahoo Finance + SEC EDGAR multi-provider architecture

**Date:** 2026-08-06

Adds Yahoo Finance (via `yfinance`) and SEC EDGAR as dataset-specific
providers alongside Alpha Vantage, per an explicit per-capability policy —
never one provider for everything, never a silent fallback. See
`docs/security/YAHOO_SEC_PROVIDER_REVIEW.md` for the provider reviews
(package/API inspection, ToS/fair-access posture, live verification) this
document assumes as background.

## Default provider policy

| Capability | Default provider | Config knob |
|---|---|---|
| Quote | Yahoo | `FINANCE_QUOTE_PROVIDER` |
| Price history | Yahoo | `FINANCE_PRICE_HISTORY_PROVIDER` |
| Corporate actions | Yahoo | `FINANCE_CORPORATE_ACTIONS_PROVIDER` |
| Company profile | Yahoo | `FINANCE_COMPANY_PROFILE_PROVIDER` |
| Analyst estimates | Yahoo | `FINANCE_ANALYST_ESTIMATES_PROVIDER` |
| US fundamentals (income/balance/cash flow) | SEC | `FINANCE_US_FUNDAMENTALS_PROVIDER` |
| Earnings history (EPS/surprises) | Alpha Vantage (always — no Yahoo/SEC equivalent) | n/a |
| News | Alpha Vantage | `FINANCE_NEWS_PROVIDER` |

`FINANCE_AUTOMATIC_FALLBACK` (default `false`) — a primary-provider failure
never silently tries the secondary; every omission is reported with a reason
in `plan.omission_reasons`/`omission_effects`, the same Problem-10
transparency mechanism from the Phase H.1 accuracy patch, now covering every
provider uniformly.

**Provenance, stated plainly:** Yahoo Finance is **unofficial** (unsanctioned
scraping via `yfinance`, no SLA, can break without notice — see "Known
limitations" below); SEC EDGAR is the **authoritative** source for US
fundamentals (an official, documented, keyless government API); Alpha Vantage
is an **explicit secondary** provider, used only where configured (by default:
earnings history alone, since neither Yahoo nor SEC has an equivalent) — never
a silent fallback for anything Yahoo/SEC already cover. A report's `data_
provenance` names which provider actually supplied each dataset
(`provenance.<dataset>.provider`), and every evidence-citing research stage is
explicitly instructed not to imply a single provider supplied everything (see
`docs/PHASE_H2_RESEARCH_PIPELINE.md`). Missing earnings (e.g. Alpha Vantage
unconfigured or its quota exhausted) does not stop or fail the analysis — Yahoo
quote/profile/price-history and SEC income/balance/cash-flow are gathered
independently and unmetered — but it does reduce completeness, which the
research pipeline's confidence calibration accounts for in code, not just by
prompt instruction (Problem 10 in the corrective patch below).

## Architecture

**One coordinator instance per provider**, sharing the same
`MarketDataRequestCoordinator`/`MarketDataCache` machinery via dependency
injection (`finance/coordinator.py` gained `provider_id`/`dataset_resolver`/
`min_request_interval_ms` params, all defaulting to the original Alpha
Vantage behavior):

```
tools/finance_tools.py
  get_coordinator()        -> Alpha Vantage (existing, unchanged default)
  get_yahoo_coordinator()  -> Yahoo, own quota-ledger FILE (unmetered, high ceiling)
  get_sec_coordinator()    -> SEC, own quota-ledger FILE, paced under 10 req/s
```

Cache rows are already partitioned by a `provider` column
(`finance/cache.py`) — three providers safely share one `market_data.sqlite3`
file with zero key collisions. Quota ledgers are NOT safely shareable
(`window_key` has no provider dimension), so each of the three gets its own
ledger file — a real bug found and fixed during this phase: Yahoo/SEC
coordinators originally defaulted to a fresh `AlphaVantageQuotaLedger`
pointed at the SAME file/window as the real Alpha Vantage ledger, so an
exhausted AV quota silently blocked Yahoo/SEC calls too.

**Fetch flow** (`finance/workflow.py`):

```
plan_analysis()         -- ONLY for whatever is still configured for Alpha
                            Vantage (by default: "earnings" alone)
gather()                 -- executes that AV-scoped plan
gather_yahoo_and_sec()    -- separately gathers everything configured for
                            Yahoo/SEC; UNMETERED, so never part of AV's
                            quota-driven FULL/REDUCED/STOPPED decision
build_facts()             -- provider-agnostic per capability: reads EITHER
                            Alpha-Vantage-shaped or Yahoo-shaped payload keys,
                            and EITHER calls normalize_all() (AV) or accepts
                            a pre-normalized NormalizedStatements (SEC)
reconcile_facts()          -- compares facts sourced from different providers
                            (the one real overlap this architecture
                            produces: Yahoo vs. SEC share counts)
```

A real control-flow bug was found and fixed here too: `run_full_stock_
analysis` originally returned early whenever `plan.mode == STOPPED`, before
Yahoo/SEC were even attempted — meaning an exhausted Alpha Vantage quota
(now only guarding "earnings") aborted the ENTIRE analysis even though Yahoo
and SEC could still serve everything else. Fixed: STOPPED is now decided
AFTER every provider has been tried, from what is actually available.

**Normalization seam** (`finance/sec_normalization.py`,
`finance/normalization.py::normalize_yahoo_*`): both produce the EXACT same
`FinancialPeriod`/`NormalizedStatements` contract and quote/overview shape
Alpha Vantage's normalization already used — verified live, byte-for-byte,
against real AAPL data. This is what lets `finance/metrics.py` and
`_dcf_inputs_from_facts` (the DCF input assembly) work completely unchanged
regardless of which provider supplied the data — neither file was modified
in this phase.

**XBRL concept mapping** (`finance/xbrl_mapping.py`): a fixed,
precedence-ordered candidate-concept list per normalized field (never fuzzy/
LLM matching), instant-vs-duration correctness enforced structurally, and
amended filings (10-K/A) correctly superseding the original via
most-recently-filed selection — all verified against real, live AAPL data
including one genuine amended-filing scenario in the test fixture.

## What's deliberately NOT built in this phase

- **Checkpoint/resume across process restarts.** Still not requested at the
  granularity of "persist mid-analysis state to disk and resume after a
  restart" — `StageCheckpoint`/`ResearchPipelineResult` (Phase H.2) remain
  in-process-only. The workflow-level checkpoints this phase's spec asked
  for (`REQUEST_VALIDATED` → ... → `SYNTHESIS_COMPLETE`) were not added as a
  literal state machine; `AnalysisPlan`/`AnalysisMode` plus the omission-
  transparency fields already give equivalent visibility into what
  succeeded/was omitted for one run.
- **Full 20-section report restructure.** The report gained provider
  attribution (item 1 of `FINANCE_REPORT_SYSTEM_INSTRUCTIONS`/
  `FINANCE_FACTS_REPORT_SYSTEM_INSTRUCTIONS`) and a prompt-injection
  instruction for provider free-text, not a full rewrite into the spec's
  exact 20-numbered-section layout.
- **Yahoo `symbol_search`.** The other five Yahoo tools are built; symbol
  search was deprioritized since `FullStockAnalysis` always receives an
  explicit ticker already (see `finance/workflow.py::detect_full_stock_
  analysis_request`), not a company name needing resolution.

## Configuration

See `.env.example`'s "Phase H.3" section for the complete, documented list.
Key points: `YAHOO_PERSONAL_USE_ACKNOWLEDGED` defaults to `false` (an
explicit opt-in separate from `YAHOO_FINANCE_ENABLED` — unofficial scraping
needs more than "not disabled"); `SEC_USER_AGENT` has no default (SEC
rejects requests without one) and is never hardcoded in source.

## Testing

| File | Covers |
|---|---|
| `tests/test_finance_yahoo_provider.py` | Client-level: payload shaping, the fast_info-silently-returns-None-for-invalid-tickers bug (found and fixed live), error classification, cache reuse (18 tests) |
| `tests/test_finance_sec_provider.py` | Client-level: URL building, host allowlist, credential gating, HTTP error classification, `resolve_cik`'s one-fetch-serves-every-ticker cache reuse (17 tests) |
| `tests/test_finance_xbrl_mapping.py` | Concept precedence, instant-vs-duration correctness, amended-filing precedence, unresolved fields (10 tests) |
| `tests/test_finance_sec_normalization.py` | Exact shape/field-name match with Alpha Vantage's contract, derived aggregates, per-fact provenance (7 tests) |
| `tests/test_finance_reconciliation.py` | The real Yahoo-vs-SEC share-count overlap (6 tests) |
| `tests/test_finance_multi_provider_workflow.py` | End-to-end through `run_full_stock_analysis` with fake multi-provider clients: default routing, the STOPPED-recovery fix, Yahoo/SEC failure-mode degradation, reconciliation firing on real divergence (11 tests) |

Plus `tests/conftest.py` gained a session-wide `autouse` fixture pinning
every pre-existing finance test to `alphavantage`-only provider config (they
predate Yahoo/SEC and build Alpha-Vantage-shaped fixtures) — the alternative
would have meant editing ~50 existing test files individually.

Full regression: **1510 passed, 3 skipped** (pre-existing, unrelated Windows
symlink-permission skips), 0 failed, at the time this document was written —
up from 1441 before this phase (69 new tests). A subsequent corrective patch
(2026-08-06 — see `docs/PHASE_H2_RESEARCH_PIPELINE.md` §9) added a real,
live-captured COST regression fixture (`tests/fixtures/cost_regression.json`,
`tests/test_finance_cost_regression.py`) exercising this exact Yahoo+SEC path
end to end; full regression at that point: **1603 passed, 3 skipped**, 0
failed.

Live verification: `scripts/manual_verify_yahoo_sec_stock_analysis.py
[SYMBOL]` — run against AAPL this session, all checks passed against the
real Yahoo Finance and SEC EDGAR APIs.

## Known limitations

- **Yahoo can break without notice** — see the provider review's honest
  limitations section; unofficial scraping has no SLA.
- **SEC XBRL tag coverage is issuer-specific** — a genuinely custom/unlisted
  tag produces a controlled unresolved-field entry, never a guess.
- **Yahoo's own local pickle cache** (outside this project's control) is a
  low-severity residual risk in this single-user desktop context — see the
  provider review §1.2.
- **Confidence in the reconciliation warning's 2% threshold is a judgment
  call**, not derived from any published tolerance — the two figures being
  compared (basic vs. diluted-weighted-average shares) are legitimately
  different measures, not necessarily an error, at any threshold.
- **This is still not investment advice** — unchanged from the Phase H.1/H.2
  guardrail; richer, more authoritative sourcing changes confidence in the
  facts, not the nature of the research output. Strengthened materially by
  the 2026-08-06 corrective patch: see `docs/PHASE_H2_RESEARCH_PIPELINE.md`
  §9 for the schema/content-policy changes a live COST report's trade-advice
  output prompted.
