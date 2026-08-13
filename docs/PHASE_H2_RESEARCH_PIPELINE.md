# Phase H.2 — Staged research pipeline

**Date:** 2026-08-05 (original) · **corrective patch:** 2026-08-06 — see §9
**What this adds:** an independent bull/bear/risk/research-synthesis stage,
appended to the existing Phase H.1 `FullStockAnalysis` report, built from role
structure **adapted** from `TauricResearch/TradingAgents` (reviewed at a
pinned commit) — **not** imported as a runtime dependency, and **not** using
LangGraph. **This produces research characterization only — never a
buy/sell/hold/avoid recommendation, a position size, an order, or a guarantee
of future performance** (see §4.1 and §9.1; this was tightened materially by
the 2026-08-06 corrective patch after a live report showed the original
schema producing exactly that kind of trade-advice output).

This document records what was inspected in TradingAgents, exactly which
concepts were adapted vs. deliberately excluded, the evidence-ID scheme that
enforces "never invent provider facts" in code (not just in a prompt), and the
fail-closed cascade design. See `docs/PHASE_H1_STOCK_ANALYSIS.md` for the
underlying deterministic facts/DCF pipeline this stage consumes but never
modifies, and §9 below for the 2026-08-06 corrective patch that closed
several gaps a live report exposed.

---

## 1. What was inspected

`TauricResearch/TradingAgents`, commit `a33fd4c0f134485a43553a2c23a63cb14adbd88f`
(committed 2026-07-18), **Apache-2.0** (`LICENSE`, verified). Read directly
from GitHub at that pinned ref (`gh api repos/.../contents/...?ref=<sha>`) —
nothing below is quoted from the README; every claim was read from source at
that exact commit. TradingAgents was never installed, imported, or added as a
dependency anywhere in this project.

Relevant structure at that commit:

```
tradingagents/agents/researchers/bull_researcher.py
tradingagents/agents/researchers/bear_researcher.py
tradingagents/agents/managers/research_manager.py
tradingagents/agents/managers/portfolio_manager.py
tradingagents/agents/risk_mgmt/aggressive_debator.py
tradingagents/agents/risk_mgmt/conservative_debator.py
tradingagents/agents/risk_mgmt/neutral_debator.py
tradingagents/agents/trader/trader.py
tradingagents/graph/conditional_logic.py   (round-count / routing logic)
tradingagents/graph/checkpointer.py        (LangGraph state persistence)
tradingagents/graph/trading_graph.py       (LangGraph StateGraph assembly)
```

### 1.1 — Bull/bear debate is an unbounded, config-driven LOOP, not one exchange

`tradingagents/graph/conditional_logic.py::ConditionalLogic.should_continue_debate`:

```python
def should_continue_debate(self, state: AgentState) -> str:
    if state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds:
        return "Research Manager"
    if state["investment_debate_state"]["current_response"].startswith("Bull"):
        return "Bear Researcher"
    return "Bull Researcher"
```

`max_debate_rounds` defaults to 1, meaning even the *default* configuration
runs 2 full bull/bear exchanges ("3 rounds of back-and-forth between 2
agents", per the code's own comment counting the opening statement) before
handing off — and it is uncapped if `max_debate_rounds` is raised. There is no
fixed "one rebuttal" boundary anywhere in the original.

### 1.2 — The risk-review stage debates the TRADER'S PROPOSAL, not the research

`tradingagents/agents/risk_mgmt/aggressive_debator.py` takes
`trader_decision = state["trader_investment_plan"]` as its central input and
is prompted to "create a compelling case for **the trader's decision**" —
confirmed identically in the conservative/neutral variants and in
`ConditionalLogic.should_continue_risk_analysis` (a 3-way
Aggressive → Conservative → Neutral loop, `3 * max_risk_discuss_rounds`
exchanges). This stage exists to argue about a concrete transaction, not to
review research quality.

### 1.3 — There is a real Trader stage and a real Portfolio Manager stage

`tradingagents/agents/trader/trader.py`: *"Trader: turns the Research
Manager's investment plan into a **concrete transaction proposal**"* — a
`TraderProposal`-typed output.

`tradingagents/agents/managers/portfolio_manager.py`: *"Portfolio Manager:
synthesises the risk-analyst debate into the **final decision**"* — a
`PortfolioDecision`-typed output ("Buy" / "Overweight" / "Hold" /
"Underweight" / "Sell", explicitly framed as *"the final trading decision"*),
consuming both the research plan and the trader's proposal.

### 1.4 — No evidence-citation mechanism exists anywhere in the original

Every agent (bull, bear, research manager, risk debators, trader, portfolio
manager) either free-text-generates or produces a typed rating/plan
(`invoke_structured_or_freetext`, `bind_structured`), but **nothing in the
reviewed source requires or validates that a claim is tied to a specific,
checkable upstream fact.** The evidence-ID scheme in this project
(`finance/evidence.py`) has no counterpart in TradingAgents — it was designed
for this project specifically to satisfy "cite internal evidence IDs" and
"never invent provider facts" as a **code-level**, not prompt-level,
guarantee (see §3).

### 1.5 — Orchestration is LangGraph, with its own checkpointing

`tradingagents/graph/checkpointer.py` / `trading_graph.py` / `propagation.py`
build and run a LangGraph `StateGraph`; "workflow checkpoints between stages"
in the original means LangGraph's own state-persistence mechanism. This
project's checkpoints (`finance.research_pipeline.StageCheckpoint`) are an
unrelated, much smaller in-process dataclass — see §4.

---

## 2. Adapted vs. excluded — itemized

| TradingAgents concept | Adapted here as | Divergence |
|---|---|---|
| Independent bull researcher | `bull_researcher` stage | Structured JSON + evidence citation, not free text; no debate loop |
| Independent bear researcher | `bear_researcher` stage | Same |
| Bull/bear debate loop (§1.1) | ONE `rebuttal_round` stage | Single call, both rebuttals, no further looping regardless of config — "bounded" is a hard code property, not a config default that happens to be small |
| Research Manager reconciliation | `research_manager` stage | Compares evidence strength/data quality ONLY — evidence-cited; **no rating scale, no "for the trader" framing, and (2026-08-06 corrective patch) no invented decision doctrine either**: an earlier version of this stage applied named conflict-resolution rules (fundamentals-over-technicals, consensus-over-outlier) that are not facts in the evidence snapshot and, observed live, produced fabricated statistical framing (calling the base DCF scenario "consensus" and the bull scenario an "outlier" when they are three modeled scenarios with no such relationship) — removed; see §9 |
| Risk debate (§1.2, 3-way, over a trade proposal) | ONE `risk_reviewer` stage | Reviews the Research Manager's reconciliation, **never a trade proposal — no trader stage exists in this pipeline at all**; adds explicit data-quality/analytical-risk framing (stale data, wide scenario spread) that the original's business-risk-only framing doesn't have |
| Trader (§1.3) | **excluded entirely** | No transaction proposal is ever produced anywhere in this pipeline |
| Portfolio Manager (§1.3) | **excluded by name and by function** | Renamed concept is `FinalInvestmentSynthesizer`; schema has no position size, order type, quantity, or **(2026-08-06 corrective patch) buy/sell/hold/avoid verdict field of any kind** — see §4.1 |
| LangGraph `StateGraph` + checkpointer (§1.5) | Plain Python function calls + `StageCheckpoint`/`ResearchPipelineResult` dataclasses | No new dependency; no persistence/resumability across process restarts (not requested) |
| Evidence citation | **new in this project** | No counterpart in TradingAgents (§1.4) |

Report-section framing (Bull Case / Bear Case / explicit conflict-resolution
notes / Risk / final verdict) is also aligned with the Alpha Vantage
`global-stock-analysis` skill's report structure as a secondary structural
reference (see the module docstring in `finance/research_pipeline.py`) —
specifically its Investment Thesis / Bull Case / Bear Case / Risk / Valuation
Summary sections. Its Entry Strategy / Exit Strategy sections (position
sizing, stop-loss levels) were deliberately **not** adopted, for the same
reason the Trader/Portfolio Manager stages were excluded: this project has no
trading capability and must not imply one.

---

## 3. The evidence-ID scheme (`finance/evidence.py`)

Every citable fact in a `build_compact_synthesis_payload()` snapshot (Phase
H.1) gets a stable, deterministic ID — `fundamental.roe_ending_equity`,
`dcf.value_per_share.bull`, `valuation_gap.direction`, `plan.omitted.earnings`
— built once per analysis by `build_evidence_index()` and rendered to a single
immutable string (`render_evidence_index()`) that every stage receives
identically.

**Absent facts are simply not indexed.** A metric whose value is `None` (not
reported this run) produces no ID at all — a stage cannot cite "the metric
that wasn't there" because there is nothing to cite; there is no null-valued
placeholder ID a model could point to and claim as support.

**Citations are validated in code, not just requested in the prompt.**
`validate_evidence_citations(cited_ids, index)` rejects any ID a stage's
output cites that is not actually a key in the index. Every per-stage
validator (`finance/research_pipeline.py::_evidence_list`) calls this and
raises on the first unknown ID — a model that fabricates a plausible-looking
ID (or cites a real-sounding field that happens to be null this run) fails
validation exactly like malformed JSON does: the stage is marked `FAILED`,
never silently accepted. `tests/test_finance_evidence.py` and
`tests/test_finance_research_pipeline.py` cover this directly, including the
specific case of citing a field that exists as a *name* elsewhere but was
`None` in this analysis.

---

## 4. The staged pipeline and its fail-closed cascade

`finance/research_pipeline.py::run_research_pipeline` runs six stages, each a
**separate** local-model call:

```
bull_researcher ─┐
bear_researcher ─┴──> rebuttal_round ──> research_manager ──> risk_reviewer ──> final_investment_synthesizer
```

Every stage:

- receives the **same immutable rendered evidence-index string** — never the
  raw payload, never a tool, never network access (`_run_stage` calls
  `ask_local_fn` with only `messages`, `options`, `response_format`, and
  `timeout` — there is no code path that can pass a `tools` schema; see
  `test_no_stage_call_ever_receives_a_tools_argument`)
- must return strict JSON matching its own closed schema (hand-rolled
  validators, matching this project's existing no-schema-library style in
  `finance/dcf.py`) — unknown/extra keys in the model's raw JSON are dropped,
  never passed through (`test_extra_unexpected_fields_are_dropped_not_passed_through`)
- must cite `evidence_cited` IDs that are code-verified against the index
- runs under a configured token budget (`options.num_predict`) and timeout
  (both threaded through per call, not hardcoded — see
  `test_configured_token_and_timeout_budgets_actually_propagate`)
- **fails closed**: malformed/unparseable/uncited output, an `ask_local_fn`
  exception, or an `ok: False` response all produce a `FAILED`
  `StageCheckpoint` — never a guessed substitute

**Checkpoints ARE the "workflow checkpoints between stages"** the spec asked
for: `StageCheckpoint` records every stage's outcome (or documented
failure/skip reason) before the next stage ever runs, so progress is
inspectable at every step. This is an in-process dataclass, not
LangGraph — there is no persistence or resumability across process restarts,
because none was requested.

**Cascading dependency logic** — a stage whose prerequisite failed is
`SKIPPED` with a recorded reason, never fed a fabricated stand-in:

| Stage | Requires |
|---|---|
| `rebuttal_round` | BOTH `bull_researcher` AND `bear_researcher` completed |
| `research_manager` | AT LEAST ONE of `bull_researcher` / `bear_researcher` completed |
| `risk_reviewer` | `research_manager` completed |
| `final_investment_synthesizer` | `research_manager` AND `risk_reviewer` completed |

One researcher failing does **not** kill the pipeline (the survivor is enough
evidence to reconcile from); both failing cascades every downstream stage to
`SKIPPED` and the pipeline never reaches a verdict
(`ResearchPipelineResult.available = final_stage.status == COMPLETED`).
`tests/test_finance_research_pipeline.py` exercises every row of this table
directly, plus an `ask_local_fn` that raises and a response with `ok: False`.

### 4.1 — `FinalInvestmentSynthesizer`, not `PortfolioManager`

> **Superseded 2026-08-06 (corrective patch).** The schema described in the
> original version of this section — `verdict_not_holding` (`BUY`/`HOLD_OFF`/
> `AVOID`), `verdict_holding` (`ADD`/`HOLD`/`SELL`) — was observed live
> producing exactly the holding-dependent trade-advice output this stage was
> always supposed to avoid: *"If you do not currently hold a position: AVOID.
> If you already hold a position: SELL."* A disclaimer after those words was
> not sufficient. The schema itself no longer has a slot for a verdict. Full
> current design in §9.

Named per the spec, and structurally incapable of producing a position size,
order, or buy/sell/hold/avoid verdict of any kind. Current validated schema
(`finance/research_pipeline.py::_validate_final_synthesizer_output`):

```
research_stance   "positive" | "cautiously_positive" | "neutral" |
                   "cautious" | "negative" | "insufficient_data"
valuation_view    "undervalued" | "approximately_fair" | "overvalued" |
                   "highly_uncertain" | "insufficient_data"
overall_risk      "low" | "moderate" | "high" | "very_high" | "insufficient_data"
confidence        float, 0.0-1.0 (capped in code when a material dataset was
                   omitted -- see §9)
rationale         [{statement: str, evidence_ids: [str]}, 1-6 items]
conditions_that_strengthen_the_view   [str], 0-4 items
conditions_that_weaken_the_view       [str], 0-4 items
key_uncertainties                     [str], 0-5 items
```

No field for a quantity, order type, position size, entry/exit price,
stop-loss, target allocation, or price target anywhere in the schema.
`confidence` is explicitly calibrated against `dcf_scenario_spread` (a wide
bull/bear spread caps confidence low regardless of direction) AND, since the
corrective patch, deterministically clamped down (never up, never rejected)
when a material dataset was omitted from the analysis, regardless of what the
model itself reports.
`test_final_investment_synthesizer_output_never_contains_order_or_size_fields`
and `test_final_investment_synthesizer_enum_values_are_research_language_only`
lock this in — the enum values themselves, not just a scan of free text,
contain no BUY/SELL/HOLD/AVOID/ADD.

Even with no verdict-shaped field left in the schema, a model can still WRITE
prohibited language into a free-text field (`rationale`, a `conditions_that_
*` entry, `key_uncertainties`) — this is backstopped by a deterministic
content-policy AND claim-fidelity scan (§9), not left to the schema alone.

### 4.2 — Never touches deterministic calculations

`finance/research_pipeline.py` imports nothing from `finance.dcf`,
`finance.metrics`, or `finance.normalization`
(`test_module_never_imports_the_deterministic_calculation_layer`), and the
integration test
`test_enabled_happy_path_never_mutates_deterministic_facts` proves
`result.facts["dcf"]` and `result.facts["fundamental_metrics"]` are
byte-identical before and after a full `synthesize_report()` run with the
pipeline enabled.

---

## 5. Wiring into the report (`finance/workflow.py::synthesize_report`)

The pipeline runs **before** the report-writing call, against an evidence
index built from the same compact payload the report call receives:

- **Pipeline reaches a final research characterization** (`available = True`):
  the report call uses `FINANCE_FACTS_REPORT_SYSTEM_INSTRUCTIONS` — the same
  facts/cache/metrics/DCF/technical/omitted-datasets instructions as the
  original single-shot prompt, but explicitly told **not** to write its own
  bull case, bear case, risk discussion, or research stance. That content is
  instead rendered deterministically from the pipeline's own validated stage
  outputs (`_render_research_pipeline_section`) and appended after the
  model's text — so there is never a second, disagreeing, prompt-written
  characterization. That free-text report call itself goes through the SAME
  content-policy/claim-fidelity scan-and-repair as the staged pipeline (§9) —
  never a lower bar just because it's plain text with no JSON schema.
- **Pipeline disabled, or enabled but fails to reach a final
  characterization**: falls back to exactly the original
  `FINANCE_REPORT_SYSTEM_INSTRUCTIONS` (which includes its own embedded
  Research Stance section — research characterization only, never a
  buy/sell/hold/avoid verdict, per the corrective patch), so a report is
  never left without one merely because the newer multi-stage pipeline
  couldn't complete. A short note names which stage broke the chain (e.g.
  *"could not complete (bull_researcher: 'key_points' must be a list of at
  least 2 item(s))"*).
- **Metrics**: the `(prompt_tokens, completion_tokens)` returned from
  `synthesize_report` sum the report call **and every pipeline stage that
  ran** — consistent with how `assistant.py` already sums tool-loop and
  synthesis tokens elsewhere for one turn's accounting.

A SKIPPED/FAILED individual stage still renders — as
*"Not available: \<reason\>"* under its own section heading — rather than
silently vanishing, mirroring the existing `plan.omission_reasons`
transparency pattern from the Phase H.1 accuracy patch.

---

## 6. Configuration

```
STOCK_ANALYSIS_RESEARCH_PIPELINE_ENABLED=true   # default on; false = today's single-shot report only
RESEARCH_STAGE_MAX_OUTPUT_TOKENS=700            # per-stage Ollama options.num_predict
RESEARCH_STAGE_TIMEOUT_SECONDS=90               # per-stage wall-clock timeout
RESEARCH_REDUCED_MODE_CONFIDENCE_CAP=0.6        # 2026-08-06: ceiling for FinalInvestmentSynthesizer's
                                                 # confidence (0.0-1.0) when a dataset was omitted
RESEARCH_REDUCED_MODE_CONFIDENCE_ENUM_CAP=medium # ceiling for Bull/Bear's low/medium/high confidence
```

See `.env.example` for the full block with inline documentation.

---

## 7. Testing

| File | Covers |
|---|---|
| `tests/test_finance_evidence.py` | Index construction, absence-of-null-fields, determinism, citation validation (21 tests) |
| `tests/test_finance_research_pipeline.py` | Every stage's validator, the full cascade table, malformed-JSON fallbacks, no-tools/budget/immutable-snapshot structural guarantees, module-level no-network/no-deterministic-calculation-import checks, the content-policy repair mechanism (repair succeeds/fails-closed/not-triggered-for-schema-errors), claim-fidelity violations per stage, reduced-mode confidence capping and omission-disclosure enforcement (2026-08-06 corrective patch additions — 51 tests) |
| `tests/test_finance_research_pipeline_integration.py` | Real `AnalysisResult` through `synthesize_report`: enabled happy path, enabled total-failure fallback, disabled parity with the original behavior, a STOPPED plan never invoking the pipeline, deterministic-facts immutability (7 tests) |
| `tests/test_finance_content_policy.py` | The trade-advice-directive scanner in isolation: every prohibited category, negation-aware "not a price target" handling, structure-walk deduplication (15 tests, corrective patch) |
| `tests/test_finance_claim_validation.py` | The unsupported-claim scanner in isolation: superlatives, causal overreach, consensus-language ban, future-tense technical certainty, provenance mismatch (18 tests, corrective patch) |

Full project regression: `pytest tests/` — **1603 passed, 3 skipped**
(pre-existing, unrelated: Windows symlink-permission skips in
`test_repo_files.py` / `test_repo_store.py`), 0 failed, at the time this
corrective patch was written (up from 1441 at Phase H.2's original writing).

Two pre-existing tests (`test_finance_msft_regression.py::
test_synthesis_prompt_is_compact_for_the_msft_fixture` and the
`orchestrated` fixture in `test_assistant_stock_analysis_orchestration.py`)
used a stubbed `ask_local_raw` with the OLD 3-argument signature
(`messages, tools=None, timeout=120`). Because every pipeline-stage call
passes `options=`/`response_format=`, those calls raised `TypeError`, which
`_run_stage`'s `except Exception` silently turned into ordinary `FAILED`
checkpoints — the tests still passed, but by accident, not by design (an
unrelated stub drifting out of sync with `brain.ask_local_raw`'s real
signature). Both were fixed: the stub signatures now match
`brain.ask_local_raw`, and — since both files test something else entirely
(prompt compaction; orchestration/routing) — the pipeline is explicitly
disabled in both via `STOCK_ANALYSIS_RESEARCH_PIPELINE_ENABLED=false`, so
their single-call assertions are true for the reason they claim, not as a
side effect of a swallowed exception.

---

## 8. Honest residual limitations

- **Latency**: up to 7 sequential local LLM calls per analysis when the
  pipeline is enabled (1 facts call + 6 stages) versus 1 previously. Nothing
  in this phase parallelizes independent stages (e.g. bull/bear could run
  concurrently) — not requested, and the project's local-model call path is
  synchronous throughout.
- **No cross-process resumability.** `StageCheckpoint`/`ResearchPipelineResult`
  live only for the duration of one `synthesize_report()` call; a crash
  mid-pipeline loses that run's checkpoints (the deterministic facts/DCF
  underneath are unaffected and cheap to recompute from cache).
  True workflow persistence (e.g. resuming a partially-completed pipeline
  after a restart) would need a durable store and was not requested.
- **Confidence-vs-spread calibration is prompt-guided, not independently
  verified.** The `FinalInvestmentSynthesizer` is instructed to lower
  confidence when `dcf_scenario_spread` is wide, but nothing in code enforces
  that the reported `confidence` value actually correlates with the spread —
  it is research output, not a mechanically-derived score. (Confidence-vs-
  OMITTED-DATA calibration, by contrast, IS code-enforced since the
  corrective patch — see §9.4 — this residual limitation is narrower than it
  used to be.)
- **Numeric fidelity between a claim's prose and its cited evidence is not
  code-verified** (corrective patch, §9.3) — a deliberate scope decision, not
  an oversight: see §9.3 for why.
- **Still not investment advice.** Every research characterization is
  evidence-cited output about the business and its valuation, never a trade
  order, a position size, or a guarantee — stated in the rendered report
  section and unchanged from the Phase H.1 guardrail this stage sits
  alongside. Strengthened by the corrective patch: the schema itself no
  longer has a verdict-shaped field to misuse (§9.1), and free-text fields are
  screened for prohibited/unsupported language regardless (§9.1, §9.3).

---

## 9. Corrective patch (2026-08-06)

The live COST report (Costco Wholesale) exposed three classes of problem in
the H.2/H.3 pipeline that prompt instructions alone had not fully closed.
This section is the authoritative summary; code comments at each change site
cite it by "Problem N." Full regression at the time of writing: **1603
passed, 3 skipped**, 0 failed.

### 9.1 — No trade-advice output, anywhere, ever (Problem 1)

**Symptom:** the `FinalInvestmentSynthesizer` produced *"If you do not
currently hold a position: AVOID. If you already hold a position: SELL."*

**Fix:** removed every holding-dependent and action-oriented field from the
schema (see §4.1's current schema). Backstopped by a DETERMINISTIC scan
(`finance/content_policy.py`) for BUY/SELL/HOLD/AVOID/STRONG BUY/STRONG
SELL/position size/entry price/exit price/stop loss/target allocation/"if
you hold"/order instructions/price target — applied to every stage's
validated output (not just the final one; see §9.5) via a shared
`_validate_claim_fidelity` helper in `finance/research_pipeline.py`, and to
the fallback single-shot narrative report via
`finance/workflow.py::_ask_local_with_content_policy`. A violation in the
FinalInvestmentSynthesizer gets exactly ONE constrained repair attempt
(`_run_final_synthesizer_stage`) — the repair prompt includes the exact
violation text; if the repair is STILL in violation, the stage fails closed
and neither attempt's content is ever exposed. Every other stage fails
closed on the first violation, with no repair (matching how an unverifiable
evidence citation already behaved). This system produces **research, never
trade instructions** — full stop, in every code path, complete or reduced.

### 9.2 — CapEx/D&A/working-capital assumptions from reported history, never a margin difference (Problems 2-3)

**Symptom:** `capex_pct = abs(operating_margin - fcf_margin)` — financially
invalid (the gap between two unrelated margins also reflects taxes,
interest, and working-capital swings) — produced a 1.00% CapEx/revenue
assumption for COST against a real reported ratio near 2%.

**Fix:** the reported-history-first hierarchy documented in
`docs/PHASE_H1_STOCK_ANALYSIS.md`'s "Assumption provenance" section:
multi-year reported average → single reported year (labeled) → configured
default (explicit, never disguised as calculated). Verified against real,
live COST data: `tests/test_finance_cost_regression.py`.

### 9.3 — Semantic evidence validation: citing a real fact isn't the same as not overstating it (Problem 5)

**Symptom:** *"fortress balance sheet"* and *"industry-leading ROE"* each
cited a REAL, existing evidence ID — the citation-validation mechanism (§3)
guarantees the fact exists, not that the surrounding prose is faithful to
what it shows.

**Fix:** `finance/claim_validation.py`, applied through the same
`_validate_claim_fidelity` path as §9.1: superlative control (fortress,
industry-leading, best-in-class, exceptional, safest, guaranteed, certain
to, inevitable — deliberately NOT bare "best"/"leading", which are common,
neutral, non-promotional English/technical-analysis terms); causal-overreach
control (confirms, proves, protects from downside, eliminates/removes risk,
"X is due", "support is proven"); a consensus-language ban (this system has
no analyst-consensus data source at all, so any "consensus value/estimate"
claim is definitionally unsupported, not merely risky); future-tense
technical-signal certainty ("will reverse", "poised to rally" — technical
indicators describe history, never predict the future); and a
provenance-mismatch check (a claim naming a specific provider — SEC, Yahoo,
Alpha Vantage — that contributed NOTHING to this analysis, most damaging in
a reduced-mode report where the wrong provider gets credited). Qualified
language passes cleanly ("Net cash may provide financial flexibility", "The
MACD histogram is positive, indicating improving recent momentum").

**Deliberately NOT implemented:** free-text arithmetic/number matching
against cited evidence values ("numeric fidelity" in the fullest sense).
Bull/Bear commentary routinely derives a correct number that is not
literally present as a scalar in the evidence index (e.g. a YoY percentage
computed from two cited revenue figures); a naive matcher would reject
valid, well-supported claims, and five of the six pipeline stages have NO
repair attempt (only the FinalInvestmentSynthesizer does), so a false reject
silently falls the whole analysis back to the single-shot narrative path.
The existing evidence-ID citation requirement (§3) remains the structural
backstop: every specific claim must still name a real fact.

### 9.4 — Reduced-mode confidence is capped in code, not just by instruction (Problem 10)

**Symptom:** the prompts already told the model to lower its own confidence
when material datasets were omitted, but nothing enforced it.

**Fix:** `_has_material_omissions(index)` (true whenever any
`plan.omitted.*` evidence entry exists) drives two deterministic clamps —
never a rejection, only ever downward: `_cap_confidence_for_omissions`
(FinalInvestmentSynthesizer's 0.0-1.0 float, capped to
`RESEARCH_REDUCED_MODE_CONFIDENCE_CAP`, default 0.6) and
`_cap_confidence_enum_for_omissions` (Bull/Bear's low/medium/high, capped to
`RESEARCH_REDUCED_MODE_CONFIDENCE_ENUM_CAP`, default medium). Additionally,
`_require_omission_disclosure` REJECTS (with the same one-repair-attempt
path as §9.1) a FinalInvestmentSynthesizer response that doesn't actually
acknowledge the omission: `rationale` must cite a `plan.omitted.*` evidence
ID, and `key_uncertainties` must be non-empty. A reduced-data analysis may
never present itself with the same completeness, or the same silence about
what's missing, as a full one.

### 9.5 — Bull/bear/research-manager/risk-reviewer language tightened, not just the final stage (Problems 6-9)

The Research Manager's invented conflict-resolution doctrine was removed
(§2's table); its schema now only compares evidence strength (`evidence_
balance`, `supported_bull_points`, `supported_bear_points`, `unsupported_
points`, `shared_findings`, `key_disagreements`, `assumption_sensitive_
conclusions`, `data_gaps`, `balanced_assessment`) — never selects a
recommendation or declares a scenario authoritative. Bull/Bear researcher
role prompts were reworded to research-only framing ("bullish/bearish
INTERPRETATION", "modeled scenario", "assumption required") in place of
promotional language ("compelling investment case", "buy opportunity");
neither may call a scenario "achievable"/a "target" without naming the
assumptions it requires, and the bear side may never call itself a "short
case" or recommend shorting. Technical-indicator language is restricted to
descriptive, historical framing (disallowing "confirms a reversal",
"correction is due", etc. — §9.3's causal-overreach scan is the code-level
backstop for this same rule). DCF outputs are "bear/base/bull MODELED
value", never "price targets" or "consensus values"; market-price
comparisons say "appears above/below the base modeled value", never
"definitively overvalued by X%". Crucially, §9.1's and §9.3's scans are
applied to the validated output of EVERY stage — bull, bear, rebuttal,
research manager, risk reviewer, and the final synthesizer alike — not only
the last one, since a bull researcher's `thesis` or `key_points` flows into
the rendered report just as directly as the final characterization does.

### 9.6 — Prompt compaction, with instrumentation (Problem 11)

Problem 2's richer per-assumption provenance (a multi-year `derivation`
sentence, `source_periods`) pushed a real fixture's compact synthesis
payload over the previously-preferred 10,000-token threshold when repeated
per scenario. Fixed via real de-duplication, not by raising the threshold:
`_hoist_shared_assumption_provenance` states a provenance entry ONCE under
`dcf.shared_assumption_provenance` when it is dynamically verified
byte-identical across every scenario (CapEx/D&A/working-capital/tax-rate —
these come from history, not from a scenario's revenue/margin deltas), and
`_drop_empty_provenance_fields` omits None/empty fields from the compact
view. A redundant per-period statement `currency` (identical to the
top-level `financial_history_currency`) is similarly omitted, while a
genuinely differing one (e.g. a redomicile) still survives. Deliberately NOT
touched: DCF forecast rows and full assumption provenance stay WHOLE per
scenario (Phase H.1's own `test_compaction_does_not_drop_dcf_or_equity_
bridge_detail` requires this, and it is the single largest contributor to
payload size at ~40-45%) — compacting REQUIRED evidence to hit a token
target would violate this same corrective patch's own "don't remove
required evidence" instruction. New instrumentation:
`result.instrumentation["repeated_field_count"]` — a generic count of
cross-item-identical fields remaining in the compact payload's repeating
groups (DCF scenarios; annual-history periods), which stays accurate as a
regression signal even where a hoist has been deliberately deferred (e.g.
the DCF equity-bridge scalars, also identical across scenarios, were judged
not worth restructuring the per-scenario shape for at current fixture
sizes). Verified on real COST data: ~34.6KB / ~8.6k estimated tokens,
comfortably under both the 10k preferred and 20k hard thresholds
(`tests/test_finance_cost_regression.py`).

### 9.7 — A real, live-data regression fixture caught an integration bug the unit tests missed (Problem 4)

Building `tests/fixtures/cost_regression.json` (real, live-captured Yahoo +
SEC data for COST, trimmed to the XBRL concepts this codebase reads —
see the fixture's own `note` field) surfaced a genuine bug beyond the
CapEx-value fix itself: `finance/dcf.py::_validate_assumption_provenance`
rebuilt every `assumption_provenance` entry from a FIXED allowlist of only
the ORIGINAL field names (`source_period` singular, `reason`), silently
dropping the CURRENT ones (`source_periods` plural, `derivation`,
`source_evidence_ids`) that §9.2's richer audit trail actually populates. A
hand-built provenance dict in an earlier unit test never exercised the
plural/derivation fields, so the drop was invisible until real multi-year
history flowed through the full pipeline. Fixed additively (old field names
kept for backward compatibility, new ones now also survive) — see
`tests/test_finance_dcf.py::test_assumption_provenance_preserves_the_
current_reported_history_fields`.
