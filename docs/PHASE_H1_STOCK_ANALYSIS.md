# Phase H.1 — Stock analysis

Production stock-analysis capability: reviewed market-data provider, quota-aware
cache, deterministic local metrics, a deterministic DCF valuation tool, and a
FullStockAnalysis workflow whose report is written by the local LLM.

**Provider decision:** both candidate Alpha Vantage **MCP servers were reviewed and
rejected**; see [`security/ALPHAVANTAGE_PROVIDER_REVIEW.md`](security/ALPHAVANTAGE_PROVIDER_REVIEW.md).
Market data is a **trusted local tool** over the existing SSRF-safe fetch path.

---

## Control flow

```
User: "/FullStockAnalysis AAPL"
  -> Router                       (local vs Claude only — unchanged)
  -> capability preflight         (unchanged)
  -> Phase B bounded shortlist    (finance tools gated on relevance)
  -> local LLM selects a tool
  -> ToolRegistry -> ToolExecutor (sole execution authority)
       -> finance.* market data
            -> MarketDataRequestCoordinator
                 1 cache lookup           (fresh hit -> zero quota)
                 2 mode policy            (cached_only / offline never hit network)
                 3 quota affordability    (refuse BEFORE spending)
                 4 per-key single-flight  (concurrent identical -> ONE call)
                 5 re-check cache
                 6 bounded retry          (transient only, backoff + jitter)
                 7 validated cache write
                 8 stale-if-error         (labelled stale, never "fresh")
       -> finance.dcf_model       (authoritative arithmetic)
  -> structured facts + provenance
  -> local LLM writes the report
```

`router.py` and `tools/executor.py` are **unchanged**.

## Layers

| Module | Responsibility |
|---|---|
| `finance/datasets.py` | The nine reviewed datasets. `resolve_dataset` fails closed. |
| `finance/provider.py` | Alpha Vantage HTTP client over `tools/http_safety.py`. Classifies in-band errors into retryable vs. deterministic. |
| `finance/cache.py` | SQLite store, SHA-256 keys, payload-digest integrity, per-key single-flight locks. |
| `finance/quota.py` | Per-UTC-day ledger. Provider rate-limit signals override the local estimate. |
| `finance/coordinator.py` | The only path that can cause an external call. |
| `finance/normalization.py` | Provider payloads -> provenance-carrying records. Missing stays missing. |
| `finance/metrics.py` | Fundamentals + technical indicators, all from cached data. |
| `finance/dcf.py` | FCFF enterprise-value DCF. Pure arithmetic. |
| `finance/workflow.py` | FullStockAnalysis orchestration. |
| `tools/finance_tools.py` | The `BaseTool`s: 8 market-data + `finance.dcf_model`. |

## Cache

Key = SHA-256 of canonical JSON over provider, dataset, function, normalized
symbol, normalized arguments, freshness mode, schema version. **The API key is
never an input** — a rotated credential must not invalidate the cache, and a
cache file must not be able to leak one.

| Dataset | Freshness | TTL |
|---|---|---|
| `stock_quote` | delayed | 15 min |
| `intraday_prices` | delayed | 15 min |
| `daily_prices` | end_of_day | 12 h |
| `earnings` | periodic | 7 d |
| `company_overview` | reference | 7 d |
| `income_statement` / `balance_sheet` / `cash_flow` | periodic | 30 d |
| `news` | streaming | 1 h |
| `symbol_search` | reference | 30 d |

Nothing is labelled `realtime`. The free tier returns delayed data and entitlement
is not verified, so the claim is never made.

Supported behaviours: fresh hit, miss, expired, stale-if-error, forced refresh,
offline, negative caching (**only** for a provider-confirmed invalid symbol —
never a transport failure), and schema-version invalidation.

## Quota and retries

`ALPHAVANTAGE_DAILY_CALL_LIMIT` (default 25) is an **estimate, not a plan claim**.
A provider rate-limit response is authoritative and zeroes the remaining budget
for the window regardless of the estimate.

Retried (transient only): `MARKET_DATA_TIMEOUT`, `MARKET_DATA_PROVIDER_ERROR`.
Never retried: quota exceeded, rate limited, auth failure, entitlement required,
invalid symbol, invalid response. Classification **fails closed** — an unknown
code is treated as non-retryable.

The workflow chooses: `full` / `reduced` / `cached_only` / `stale` / `stopped`.

## DCF

```
FCFF_t   = EBIT_t x (1 - tax) + D&A_t - capex_t - ΔNWC_t
TV       = FCFF_n x (1 + g) / (WACC - g)        [g < WACC, enforced everywhere]
EV       = Σ FCFF_t x DF_t + PV(TV)
net_debt = total_debt - cash_and_cash_equivalents [- eligible_short_term_investments]
Equity   = EV - net_debt + other_non_operating_assets - preferred - minority
/share   = Equity / diluted shares
```

`net_debt` may be **negative** (a net-cash position) and is never clamped to
zero. Two net-debt policies are supported (`finance/dcf.py::NetDebtPolicy`),
recorded on every result:

- `cash_only` (default) — nets `total_debt` against `cash_and_cash_equivalents` alone.
- `cash_and_marketable_securities` — additionally nets off `eligible_short_term_investments`.
  Short-term investments are only *eligible* when both the policy is selected
  **and** `DCF_SHORT_TERM_INVESTMENTS_ELIGIBLE=true` — nothing infers liquidity
  automatically. `short_term_investments` (the raw figure) is always reported
  even when not eligible, so the tradeoff stays visible.

Validated: `g < WACC` strictly; horizon within configured bounds; positive shares;
WACC within bounds; `net_debt_policy` is a recognized value; no division by zero;
no NaN/infinity anywhere in the output; consistent currency; unique scenario
names; forecast arrays exactly matching the horizon. **No hidden clamps** — a
breach raises a named error rather than moving the number.

A missing assumption returns `DCF_ASSUMPTION_REQUIRED` naming every absent field.
Nothing is defaulted or invented.

Intermediates are aggregated at **full float precision** and rounded only on
output, so results match an independent calculation to 1e-4 while repeat runs stay
byte-identical.

Every scenario result carries the **full per-year forecast** (revenue, growth,
EBIT, tax rate, NOPAT, D&A, capex, ΔNWC, FCFF, discount factor, PV of FCFF),
`terminal_year_fcff`, the complete equity bridge, `terminal_value_share_of_
enterprise_value`, and its own `calculation_version` — for bear, base, and bull
alike, so the result is independently auditable without re-running anything.

### Assumption provenance

Every scenario assumption carries `assumption_provenance`: `value`,
`source_type` (`provider_fact` / `deterministic_calculation` /
`configured_default` / `user_supplied` / `llm_proposed`), `source_periods`
(every period an average actually used — plural; `source_period` singular is
kept as a backward-compatible alias), `source_evidence_ids`, `derivation` (a
human-readable audit-trail sentence — `reason` is kept as a legacy alias),
`approval_status`, `units`. The LLM may *propose* assumptions this way; it
cannot alter the arithmetic afterward — `finance.dcf_model` only ever reads
the numeric assumption fields for calculation, never the provenance metadata,
and the result reflects exactly what was validated at call time.
`propose_assumptions()` labels everything it derives as either
`deterministic_calculation` (anchored to reported figures) or
`configured_default` (e.g. WACC, tax rate — not derived from the company),
and every scenario's `approval_status` is `"proposed"`: this phase has no
user-approval step wired up yet (see "Deferred" below).

**CapEx / D&A / net-working-capital hierarchy** (Phase H.3 corrective patch,
2026-08-06): these three assumptions are NEVER derived from the difference
between two unrelated margins (e.g. `abs(operating_margin - fcf_margin)`) —
that difference also reflects taxes, interest, and working-capital swings, so
it is not a CapEx figure at all. A live COST report exposed this: it produced
a 1.00% CapEx/revenue assumption against a real reported ratio near 2%.
`propose_assumptions()` instead follows a reported-history-first hierarchy,
implemented in `finance/workflow.py::_reported_ratio_assumption`:

1. **Multiple reported annual periods available** (up to
   `DCF_ASSUMPTION_HISTORY_MAX_YEARS`, default 5) — the simple average across
   them, `source_type=deterministic_calculation`, every period and its own
   ratio spelled out in `derivation`.
2. **Exactly one reported period** — that period's ratio directly,
   `source_type=deterministic_calculation`, with an explicit single-sample
   limitation noted in `derivation` (one year may not be representative).
3. **No reported history at all** — the configured default
   (`DCF_DEFAULT_CAPEX_PCT_REVENUE` / `_DEPRECIATION_PCT_REVENUE` /
   `_WORKING_CAPITAL_PCT_REVENUE`), `source_type=configured_default`, never
   silently presented as calculated.

CapEx and D&A ratios are always clamped to `[0.0, 0.20]`; net-working-capital
may legitimately be negative and is clamped to `[-0.20, 0.30]`. All three are
identical across base/bull/bear (they come from history, not from a
scenario's own revenue-growth/margin deltas) — verified live against COST
(see `tests/test_finance_cost_regression.py`): CapEx/revenue 1.84% (5-year
average of 2.00%, 1.85%, 1.78%, 1.71%, 1.83%), D&A/revenue 0.87%,
**operating** working-capital/revenue -5.02% (re-derived by the TSLA DCF
validation patch below; COST is a well-known negative-operating-working-
capital business once its own cash/short-term-investments balance is
correctly excluded from the ratio).

### TSLA DCF validation patch (2026-08-07)

A live TSLA `FullStockAnalysis` report produced internally inconsistent
bear/base/bull modeled values (bull MORE negative than base, base MORE
negative than bear — the reverse of the intended ordering) and a nonsensical
valuation-gap percentage against a negative modeled value. Root cause, traced
against real live TSLA data (`tests/fixtures/tsla_regression.json`): the
working-capital assumption (`finance/workflow.py::_historical_nwc_ratio_
pairs`) computed net working capital as the RAW `current_assets -
current_liabilities` balance-sheet aggregate, which for a cash-rich issuer
like TSLA is dominated by cash and short-term investments (~$44B on ~$95B
revenue) — financing/investing balances, not operating working capital, and
already accounted for separately in the DCF's own net-debt equity bridge
(`finance/dcf.py::compute_net_debt`). This produced a +24.4%-of-revenue
working-capital assumption instead of the correct **operating** figure
(~-8.6%, excluding cash/short-term investments from current assets and
short-term debt/the current portion of long-term debt from current
liabilities), forcing deeply negative FCFF in every forecast year for every
scenario and, because the resulting fake "cash build" scaled fastest under
whichever scenario had the highest revenue growth, inverting the intended
bull ≥ base ≥ bear ordering.

Fixed at the root (`_historical_nwc_ratio_pairs` now computes operating net
working capital) and backstopped with deterministic, non-corrective
validation that NEVER reorders or clamps a value, only detects and reports:

* **Per-scenario status** (`finance/dcf.py::ScenarioResultStatus`) — `valid`
  / `negative_equity_value` (preserved exactly, never clamped to zero) /
  `invalid` (a negative terminal-year FCFF fed into the Gordon-growth
  perpetuity, gated by `DCF_ALLOW_NEGATIVE_TERMINAL_FCFF`, default false).
* **Scenario monotonicity check** (`finance/dcf.py::
  _scenario_monotonicity_check`) — only when scenario names are exactly
  `base`/`bull`/`bear` AND their assumptions are actually constructed to
  imply bull ≥ base ≥ bear, validates that the CALCULATED `value_per_share`
  satisfies it too; a violation is recorded with the exact assumptions,
  values, and violated relation, never silently reordered.
* **Overall `validation_status`** on every `finance.dcf_model` result
  (`finance/dcf.py::DcfValidationStatus`: `DCF_VALID`,
  `DCF_VALID_WITH_WARNINGS`, `DCF_INVALID_SCENARIO_ORDER`,
  `DCF_NEGATIVE_TERMINAL_FCFF`, plus the pre-existing raised-on-bad-input
  codes) — never raised as an exception; a full, auditable result is always
  returned so the specific reason survives into the evidence snapshot and
  report even when the model is invalid.
* **Valuation-gap math** (`finance/workflow.py::_valuation_gap`) — a
  percentage comparison against a non-positive base modeled value is
  mathematically not meaningful; `difference_pct` is `null` and
  `valuation_gap_status` is `"not_meaningful"` (`finance/metrics.py`'s
  existing `STATUS_NOT_MEANINGFUL` convention, reused rather than
  reinvented) in that case, and unavailable entirely when the DCF failed
  validation.
* **Evidence gating** (`finance/evidence.py`) — when `validation_status` is
  invalid, no `dcf.value_per_share.*` / `dcf.assumption.*` / etc. evidence is
  indexed at all (only `dcf.validation_status`/`dcf.validation_reasons`), so
  a bull/bear researcher structurally CANNOT cite an invalid DCF value — the
  ID does not exist to cite, not merely "unreliable."
* **Research pipeline** (`finance/research_pipeline.py`) — the Risk Reviewer
  deterministically sets `model_risk: "high"`; the FinalInvestmentSynthesizer
  deterministically forces `valuation_view: "model_invalid"` and rejects (one
  repair attempt) a `research_stance` of `"insufficient_data"` when
  fundamentals/technicals are still available — `"inconclusive"` is the
  correct value for a failed valuation MODEL with otherwise-usable data, a
  different situation from insufficient underlying data.
* **`decision_readiness`** (`finance/workflow.py::_decision_readiness`,
  `READY` / `LIMITED` / `NOT_READY`) — a deterministic, research-only signal
  (never a BUY/SELL/HOLD/AVOID input) computed from DCF validity, dataset
  coverage, and cross-provider reconciliation conflicts.

For TSLA specifically: the working-capital fix alone restores correct
monotonic ordering (bull ≥ base ≥ bear), but base and bear's forecasted FCFF
still goes negative within the 5-year horizon under current assumptions
(TSLA's heavy CapEx intensity against thinner base/bear operating margins) —
`validation_status` correctly reports `DCF_NEGATIVE_TERMINAL_FCFF` for that
run rather than hiding it. This is the intended behavior: fail valuation
closed and report the exact reason, per this patch's own live-verification
requirement, rather than force a clean-looking number the underlying
assumptions do not support.

### Financial normalization corrections

Balance-sheet cash and debt are **derived from separately-reported
components**, never taken from a provider's own combined field at face value —
Alpha Vantage's `cashAndShortTermInvestments` has been observed, for a real
issuer, to silently equal cash-and-equivalents alone while a genuine nonzero
`shortTermInvestments` sat unused beside it (see
`tests/fixtures/msft_regression.json`). `finance/normalization.py` computes
`cash_and_short_term_investments = cash_and_cash_equivalents +
short_term_investments` only when both are known, retains the provider's own
figure separately for comparison, and emits a reconciliation warning (scoped to
the latest period only, so a systemic issue produces one warning, not one per
fiscal year) when the two disagree by more than 1%. `total_debt = short_term_
debt + current_portion_of_long_term_debt + long_term_debt` — using only the
components actually reported, never double-counting the current portion — with
the same reconciliation treatment (5% tolerance) against a provider-reported
total-debt field. `debt_to_equity` and ROIC's invested-capital both consume
this same reconciled `total_debt`.

ROE is reported under **two named methodologies**, never as a single
undifferentiated "ROE": `roe_ending_equity` (net income / ending equity) and
`roe_average_equity` (net income / average of beginning and ending equity —
null with a stated reason when the prior period's equity is unavailable). The
provider's own `ReturnOnEquityTTM` is kept separately under `overview` and
never conflated with either. Growth metrics (`revenue_growth_yoy`,
`net_income_growth_yoy`) carry `accounting_basis: "GAAP"` — Alpha Vantage's
income statement is GAAP-only; nothing infers or labels an adjusted figure.

## What the local LLM may and may not do

> **Superseded 2026-08-06 (Phase H.3 corrective patch).** This system
> produces **research, never trade instructions.** An earlier version of this
> phase required an explicit BUY/HOLD/ADD/SELL judgment (covering both the
> "don't hold" and "already hold" cases) — a live COST report showed exactly
> what that produces in practice: "If you do not currently hold a position:
> AVOID. If you already hold a position: SELL." That requirement is REMOVED,
> not just discouraged by prompt wording. See `docs/PHASE_H2_RESEARCH_
> PIPELINE.md`'s "Corrective patch" section for the full current policy,
> schema, and the deterministic content-policy/claim-fidelity backstops
> (`finance/content_policy.py`, `finance/claim_validation.py`) that enforce
> it in code, not just in prompt text.

**May:** explain the business, identify trends, compare scenarios, discuss
catalysts and risks, explain valuation sensitivity, describe uncertainty,
summarize what the evidence supports, and state a research stance (positive /
cautiously_positive / neutral / cautious / negative / insufficient_data), a
valuation view (undervalued / approximately_fair / overvalued /
highly_uncertain / insufficient_data), and an overall risk level, each with a
stated confidence and the specific facts it's grounded in.

**May not:** invent figures; alter DCF assumptions; perform the valuation
arithmetic; describe stale or delayed data as realtime; claim provider data was
retrieved when it came from cache; guarantee a return or claim certainty about
the future; give a buy/sell/hold/avoid recommendation in any form; say "if you
hold" / "if you do not hold"; give a position size, entry/exit price,
stop-loss level, target allocation, or any other order-shaped instruction;
call a DCF scenario a "price target" or present it as a consensus estimate
(this system has no analyst-consensus data source); treat the base DCF
scenario as automatically authoritative over bull/bear ("consensus" vs.
"outlier" is not a real relationship between three modeled scenarios); use an
unsupported superlative ("fortress", "industry-leading", "best-in-class",
"guaranteed") or an unqualified causal claim ("confirms a reversal",
"protects from downside") that the cited evidence does not itself support.

These are enforced structurally, not by prompt wording: the model receives
`data_provenance` with `origin` (`provider`/`cache`), `retrieved_at_utc`,
`age_seconds`, `stale`, and `source_freshness` per dataset; `price_basis` is
always `delayed`; and `valuation_gap` is computed in `finance/workflow.py`, not by
the model. The system prompt additionally forbids deterministic technical-analysis
claims ("a correction is due") in favor of probabilistic framing, and requires
naming which ROE methodology is cited and that GAAP growth is never described as
"adjusted" or "underlying." The staged research pipeline (Phase H.2) additionally
backstops every one of these with a deterministic scan (`finance/content_policy.py`,
`finance/claim_validation.py`) applied to EVERY stage's output, not just the
final one — see `docs/PHASE_H2_RESEARCH_PIPELINE.md`.

## Compact synthesis payload

`AnalysisResult.facts` is **unbounded** — every fiscal period the provider
returned, the full DCF detail, everything (this is what full auditability
means). What actually reaches the local LLM's prompt is a much smaller,
separate view built by `finance/workflow.py::build_compact_synthesis_payload`:

- financial history capped to the most recent `STOCK_ANALYSIS_COMPACT_
  HISTORY_YEARS` **annual** periods per statement (quarterly is dropped from
  synthesis entirely — nothing that calculates from it needs it there)
- earnings normalized and bounded (`normalize_earnings`, capped to
  `STOCK_ANALYSIS_EARNINGS_MAX_ANNUAL`/`_QUARTERLY`) — replacing an earlier
  `facts["earnings_raw"]` that forwarded the provider payload unmodified
- a bounded list of recent daily closes (`STOCK_ANALYSIS_COMPACT_PRICE_
  OBSERVATIONS`), never the full OHLCV series used to calculate the technical
  indicators
- warnings and provenance entries capped to their configured maximums
- the DCF result kept **whole** — every scenario's full forecast, equity
  bridge, and assumption provenance survive compaction unchanged; only the
  WACC x terminal-growth sensitivity **grid** is reduced to a summary (min/max
  across accepted cells plus the value at the scenario's own assumptions),
  since Problem 9 asks for a sensitivity *summary* specifically and the full
  grid remains in the unbounded result. **Phase H.3 corrective patch:** a
  provenance entry identical across every scenario (CapEx/D&A/working-capital/
  tax-rate — these come from reported history, not from a scenario's own
  growth/margin deltas) is stated ONCE under `dcf.shared_assumption_
  provenance` instead of once per scenario (`_hoist_shared_assumption_
  provenance`, dynamically verified byte-identical, never assumed) — still
  whole, just not repeated 3x; a per-period statement `currency` identical to
  the top-level `financial_history_currency` is likewise omitted (a genuine
  anomaly, e.g. a redomicile, still survives). Richer per-assumption audit
  detail (Problem 2's multi-year `derivation` sentence) pushed a real
  fixture over the preferred token threshold before this de-duplication.
- `dcf_scenario_spread` — how much the bull/base/bear scenarios themselves
  disagree (`spread_pct_of_base`), computed in `finance/workflow.py::
  _scenario_spread`. This is the objective signal the FinalInvestmentSynthesizer's
  self-reported confidence is calibrated against: a wide spread means the
  valuation is highly assumption-sensitive (low confidence, regardless of
  direction); a narrow spread with a large `valuation_gap` supports higher
  confidence. Confidence is additionally capped in code, not just by prompt
  instruction, when a material dataset was omitted — see `docs/PHASE_H2_
  RESEARCH_PIPELINE.md`'s corrective-patch section.

On the captured MSFT regression fixture (20 years annual + 81 quarters per
statement — the actual shape that produced the original bug): raw provider
payload 356KB, normalized (pre-compaction, DCF included) ~203KB (~50.7k
estimated tokens), compact payload ~39KB (**~9.8k estimated tokens**) — an
80.6% reduction, comfortably under the 20k hard threshold and the 10k
preferred one. `AnalysisResult.instrumentation` carries all five figures
(`raw_provider_payload_bytes`, `normalized_payload_bytes`,
`compact_synthesis_payload_bytes`, `estimated_synthesis_tokens`,
`repeated_field_count` — cross-item redundancy remaining in the compact
payload's repeating groups, a Phase H.3 corrective-patch regression signal —
plus post-call `actual_prompt_tokens`/`actual_completion_tokens`) for any
caller to inspect. Re-verified on real, live-fetched COST data (Problem 4):
~34.6KB, ~8.6k estimated tokens (`tests/test_finance_cost_regression.py`).

## Deferred to a future general skills framework

The project has **no Agent-Skills loader**, and this phase deliberately did not
build one. When one is added, these should move into it:

1. **Invocation.** `/FullStockAnalysis <SYMBOL>` is currently a workflow function
   (`run_full_stock_analysis`) rather than a registered slash command; there is no
   command router to register it with.
2. **Capability declaration.** `REQUIRED_CAPABILITY = "stock_analysis"` is a
   module constant. It should become skill frontmatter resolved by the trusted
   catalog, so the workflow declares a capability and never names tools.
3. **The dataset -> tool map** (`_DATASET_TOOLS`) should be capability-resolved
   rather than hardcoded tool names.
4. **Report structure** (the 17 sections) should be a skill-owned template.
5. **Assumption approval.** `propose_assumptions` derives base/bull/bear
   deterministically; a skills framework should offer an explicit approval gate
   before the valuation runs.

## Configuration

See `.env.example`. Key flags: `ALPHAVANTAGE_API_KEY`, `MARKET_DATA_ENABLED`,
`STOCK_ANALYSIS_ENABLED`, `ALPHAVANTAGE_DAILY_CALL_LIMIT`,
`MARKET_DATA_STALE_IF_ERROR_SECONDS`, `MARKET_DATA_MAX_RETRIES`,
`STOCK_ANALYSIS_INCLUDE_NEWS`.

## Testing

```bash
venv/Scripts/python.exe -m pytest tests/test_finance_cache.py tests/test_finance_quota.py \
    tests/test_finance_dcf.py tests/test_finance_normalization.py \
    tests/test_finance_workflow.py -q          # 161 focused tests
venv/Scripts/python.exe -m pytest tests/ -q    # full suite
venv/Scripts/python.exe scripts/manual_verify_h1_stock_analysis.py IBM   # OPT-IN, live
```

**Phase H.3 corrective patch** additionally added
`tests/test_finance_cost_regression.py` (17 tests, real live-captured Yahoo +
SEC data for COST — see `tests/fixtures/cost_regression.json` — the exact
company/data shape that exposed the CapEx bug), plus
`tests/test_finance_content_policy.py` and `tests/test_finance_claim_
validation.py` for the two deterministic language-screening modules. Full
regression at the time this patch was written: **1603 passed, 3 skipped**, 0
failed.

**TSLA DCF validation patch** additionally added
`tests/test_finance_tsla_regression.py` (26 tests, real live-captured Yahoo +
SEC data for TSLA — see `tests/fixtures/tsla_regression.json` — the exact
company/data shape that exposed the working-capital bug in the section
above), plus focused tests for the new validation layer across
`tests/test_finance_dcf.py` (sign-convention/terminal/monotonicity/negative-
equity), `tests/test_finance_workflow.py` (operating-NWC ratio,
valuation-gap non-meaningful handling, `decision_readiness`),
`tests/test_finance_evidence.py` (invalid-DCF evidence gating), and
`tests/test_finance_research_pipeline.py` (`model_risk`, forced
`valuation_view=model_invalid`, the `insufficient_data`-vs-`inconclusive`
stance guard). `tests/test_finance_cost_regression.py` and `tests/test_
finance_cor_regression.py`'s own working-capital-derived expected values were
re-derived and independently re-verified against the same fix (both
companies' corrected working-capital ratios are negative — a real,
well-known characteristic for COST specifically). Full regression at the
time this patch was written: **1828 passed, 3 skipped**, 0 failed.

Live Alpha Vantage calls are **never** part of the default suite.
