# Finance Correctness Specification

The source of truth for FullStockAnalysis correctness. Every future change to
the finance subsystem is measured against this document, and a new failing
stock is a reason to add or repair an invariant here — never to add a branch
for that stock.

**Status key.** Each invariant is marked `[ENFORCED]` (code prevents the
violation and a test proves it), `[PARTIAL]` (the mechanism exists but is not
wired to every consumer), or `[SPECIFIED]` (agreed rule, not yet built). The
value of this document depends on those labels being accurate; a rule marked
ENFORCED that isn't is worse than one honestly marked SPECIFIED.

---

## 0. Why this document exists

For several months every new stock exposed a *new variant of the same
correctness problem*, and each was fixed where it surfaced. The pattern
across those failures:

| Live symptom | Actual cause |
|---|---|
| −73% revenue growth | quarterly guidance divided by trailing-twelve-month revenue |
| "share count conflict" | current outstanding differenced against a weighted average |
| debt reconciliation warning | component sum compared against one of its own components |
| "strong free cash flow" for an insurer | OCF−CapEx read as owner cash |
| stale current ratio | derived metric not inheriting its sources' period |
| guidance "unavailable" | only revenue was scored; CapEx guidance existed |
| valuation "model invalid" | the model never applied to that business |
| pipeline COMPLETE beside a failed stage | COMPLETE meant "final stage ran" |

Not one is an arithmetic bug. In every case the arithmetic was correct and
**two numbers describing different things were combined as though they
described the same thing.** The code could not tell, because by the time
values reached the arithmetic they were plain floats.

The permanent fix is that a financial figure carries its identity until
canonical selection is complete, and one validator decides whether any two
figures may be combined for a given operation.

---

## 1. System boundaries

```
Providers
  ↓  finance/sec_normalization.py, yahoo_provider.py
NormalizedFinancialFact              §2
  ↓  finance/semantics.py
SemanticCompatibilityValidator       §3-4
  ↓  finance/freshness.py, ttm.py
CanonicalFinancialState              §5-7
  ↓  finance/canonical.py
DerivedMetricGraph                   §8
  ↓  finance/validity.py
Dependency invalidation              §9
  ↓  finance/business_model.py, metric_policy.py
BusinessModelSemanticProfile         §12
  ↓  finance/guidance.py, guidance_tables.py
GuidanceState                        §10-11
  ↓
ValidatedDCFInputPacket              §13
  ↓  finance/dcf.py (unchanged model)
Valuation gate                       §14
  ↓  finance/evidence.py
CanonicalResearchEvidence            §15
  ↓  finance/research_pipeline.py
Research roles → ValidatedResearchOutput  §16-17
  ↓  finance/workflow.py
Compact renderer                     §18
```

Each arrow is a boundary at which something may be refused. A refusal at an
early boundary must not be repairable at a later one.

---

## 2. Typed financial facts

**RULE.** A financially material value carries its semantic identity until
canonical selection completes. It does not become an anonymous float earlier.

Identity means: `metric_id`, `units`, `currency`, `entity_id`,
`consolidation_scope`, `accounting_basis`, `period_type`, `start/end/instant`,
`fiscal_year`, `fiscal_quarter`, `flow_or_instant`, `source_provider`,
`filing_type`, `accession`, `evidence_id`, `derivation_method`,
`freshness_status`, `validity_status`.

**RATIONALE.** A float remembers magnitude and nothing else — not the period
it covers, not the basis it was measured on, not whether it is a stock or a
flow. Every failure in §0 is downstream of that loss.

**IMPLEMENTATION.** `finance/semantics.py::SemanticFact`. `[ENFORCED]` at the
four boundaries in §4; `[PARTIAL]` elsewhere — `metrics.py` and
`normalization.py` still compute on floats.

---

## 3. Canonical metric identity

**RULE.** Two metrics are the same metric when their `metric_id` values are
equal. Never because their names look similar.

**FORBIDDEN.** Deciding that `LongTermDebt` can stand in for total debt
because both are "debt". Deciding that "subscription revenue" is revenue
because it contains the word.

**RATIONALE — the one that keeps recurring.** *A gap in the metric vocabulary
is not a neutral absence.* When no identity exists for a phrase, the nearest
**general** pattern claims it. `"subscription revenue growth of 11%–12%"` was
extracted as **consolidated** `revenue_growth` with `may_anchor=True` — a
component of revenue anchoring the whole company's DCF — purely because no
subscription identity existed. Absent identities do not produce missing data;
they produce wrong data.

**RULE (units and denominator are part of the identity).** A metric_id is
not just a name. "Operating income of $2.1 billion" and "operating income
equal to 21% of revenue" are different quantities, and so are "free cash flow
of $20 billion" and "free cash flow growth of 9% year-over-year". An absolute
metric carrying a PERCENTAGE resolves by its stated denominator:

```
% of revenue      → that metric's MARGIN identity
% year-over-year  → that metric's GROWTH identity
neither stated    → REFUSED
```

**RATIONALE.** Three live instances of one failure. A margin stored under an
absolute identity (0.21 as `operating_income`); a 5–6% EPS growth guide
stored as a $5–6 per-share figure; a 9–10% free-cash-flow growth guide stored
as $9–10 against a real $22B. Each was arithmetically faithful to the text
and named the wrong quantity. Counting the percentage cannot resolve this;
only the denominator can, and where none is stated the figure is not
identifiable at all.

**RULE (spelling is not a unit).** "5.0 to 6.0 percent" and "5.0% to 6.0%"
are one measurement. A unit written once at the end of a range distributes
over both ends — unless the other end carries a conflicting unit, which is
the separate case `6.6% to $25.3 billion` already refuses.

**IMPLEMENTATION.** `semantics.py::MetricIdentity`,
`guidance.py::GuidanceMetricName`, `resolve_percentage_identity`,
`percentage_denominator_stated`. `[ENFORCED]`.

---

## 4. Semantic compatibility

**RULE.** Compatibility is a property of a PAIR **and** an OPERATION, never
of a pair alone. Validation happens *before* arithmetic.

Operations: `COMPARE, SUM, SUBTRACT, RATIO, GROWTH, RECONCILE, ROLL_FORWARD,
PER_SHARE_CONVERSION, DCF_INPUT`.

**ALLOWED.** `COMPARE(Q3 guidance, TTM revenue)` — both belong in a report.
**FORBIDDEN.** `GROWTH(Q3 guidance, TTM revenue)` — one relation, two
answers; a validator returning a single verdict gets one of them wrong.

**RULE (forecast horizon).** Forward evidence against a forecast ASSUMPTION
is its own pair-and-operation. Evidence may set or be numerically compared
against an assumption's MAGNITUDE — and may establish
`DCF_MODEL_BOUND_CONFLICT` — only when the metric identity is compatible AND
the forecast horizon is compatible.

```
ASSUMPTION_COMPARABLE       same metric, same duration   → may set a magnitude
DIRECTIONAL_CORROBORATION   same metric, shorter window  → direction only
NOT_COMPARABLE              different metric, or an unplaceable horizon
```

A quarterly year-over-year growth rate may be entirely correct and is not the
same KIND of quantity as an annual year-1 assumption. It may support "growth
remains elevated"; it may not become the year-1 value, be averaged with an
annual rate, be compared against an annual bound, or change a bound
conflict's magnitude or severity.

**RATIONALE.** An annual bound of 25%, trailing twelve-month growth of 32%,
next-quarter guidance implying 84%. The conflict was real — 32% > 25% — and
was reported as though the model were 59 percentage points wrong, because the
quarterly figure supplied the magnitude. Right conclusion, wrong period.

**RULE (no over-correction).** The shorter-horizon figure is not discarded.
Its metric, target period, comparison period, derived rate and evidence ids
all survive, and it is attached to the conflict as `directional_support`.
Only the operations it is eligible for are restricted.

**IMPLEMENTATION.** `semantics.py::ForecastCompatibility`,
`forecast_compatibility`, `ForecastEligibility`;
`forward_assumptions.py::_bound_conflict_candidates`. `[ENFORCED]`.

**FAILURE STATUS.** `FORECAST_HORIZON_NOT_COMPARABLE`,
`PERIOD_FREQUENCY_MISMATCH`, `INCOMPATIBLE_PERIODS`,
`INCOMPATIBLE_METRICS`, `INCOMPATIBLE_SHARE_BASIS`, `INCOMPATIBLE_DEBT_BASIS`,
`INCOMPATIBLE_CURRENCY`, `INCOMPATIBLE_ACCOUNTING_BASIS`,
`INCOMPATIBLE_FLOW_INSTANT`, `INCOMPATIBLE_ENTITY_SCOPE`,
`SHARE_BASIS_NOT_COMPARABLE`.

An unrecognised operation is **refused**, not permitted. A validator that
allows what it does not understand guarantees nothing.

**IMPLEMENTATION.** `semantics.py::compatible_for`. `[ENFORCED]` at: guidance
→ assumption, share reconciliation, debt reconciliation, DCF gate.

---

## 5. Period semantics

Canonical types: `INSTANT, QUARTER, YTD_6M, YTD_9M, HALF_YEAR, ANNUAL, TTM,
MULTI_YEAR, UNKNOWN`.

**RULE.** A growth rate requires both sides to span the same number of
months. ANNUAL↔TTM is allowed (both twelve); QUARTER↔TTM is not.

A quarter compares only against **the same quarter** of another year. A
sequential quarter-on-quarter change is not a growth rate for a seasonal
business and is never the one a DCF wants.

**IMPLEMENTATION.** `semantics.py::PeriodFrequency`, `same_duration`.
`[ENFORCED]`.

---

## 6. TTM construction

**RULE.** Every flow metric constructs and validates its own TTM. There is no
global `ttm_valid`. Revenue TTM being valid says nothing about FCF TTM.

Constructions: four discrete quarters, or `latest FY − prior comparable YTD +
current YTD` with correct cumulative handling.

Each TTM preserves construction method, source periods, start/end, evidence
ids, validation status.

**IMPLEMENTATION.** `finance/ttm.py`. `[ENFORCED]`.

⚠️ **Two TTM builders exist** — `ttm.py::build_ttm` (`start_date`/`end_date`)
and `freshness.py::build_ttm` (`period_start`/`period_end`). Using the wrong
field pair raises, and an early version swallowed that in a bare `except`.
Consolidation candidate. `[PARTIAL]`.

---

## 7. Current vs historical

**RULE.** Current and historical evidence live in separate namespaces. A
historical figure may not silently answer a question about today.

```
current.ttm.revenue          historical.fy2025.revenue
current.ttm.revenue_growth_yoy   historical.5y.revenue_cagr
```

**FORBIDDEN.** A bare `revenue_growth` that either could satisfy. That
ambiguity let a Snapshot show a trailing-twelve-month base and, two rows
below, the prior fiscal year's growth.

**FAILURE STATUS.** `STALE_METRIC_CITATION`,
`CURRENT_GROWTH_USED_HISTORICAL_PERIOD`.

**IMPLEMENTATION.** `canonical.py`, `evidence.py`. `[ENFORCED]` — a
present-tense claim citing `fundamental.*` where `current.*` exists is
quarantined.

---

## 8. Derived metrics and freshness

**RULE.** A derived metric inherits the periods of its sources and states
them. Components from different instants are **refused**, not divided.

```
current_ratio = current_assets@D / current_liabilities@D   (same D)
```

**FORBIDDEN.** A Q2 numerator over an FY denominator. A stale annual ratio
displayed beside a quarterly balance-sheet date while that quarter's own
components sit unused.

**FAILURE STATUS.** `DERIVED_RATIO_PERIOD_MISMATCH`,
`DERIVED_METRIC_STALE_SOURCE`, `TTM_BASE_PERIOD_MISMATCH`,
`DCF_CANONICAL_PERIOD_CONFLICT`.

TTM alignment reports `ALIGNED / PARTIALLY_ALIGNED / MISALIGNED / UNKNOWN`
with the exact lagging metric, its actual end date, its construction method,
and whether it affects the valuation.

**IMPLEMENTATION.** `canonical.py`. `[ENFORCED]`.

---

## 9. Dependency invalidation

**RULE (global).** *If a canonical input is invalid, every material derived
value that depends on it becomes invalid.*

**RULE (equally important).** Invalidity propagates through **declared
dependencies only**. A debt conflict must not invalidate revenue, margin or
the technicals.

**RULE.** An INVALID metric exposes **no numeric value**. `ValidatedMetric.value`
returns `None`; the figure survives on `raw_value` for the audit view only.

**FORBIDDEN.** `value = 80.0B, warning = TOTAL_DEBT_CONFLICT` with downstream
code continuing to use 80.0B. Measured: `validation_status` existed on every
metric, was copied everywhere, and was checked nowhere.

Worked chain:
```
TOTAL_DEBT invalid → NET_DEBT → DEBT_TO_EQUITY → DCF_EQUITY_BRIDGE
→ MODELED_VALUE_PER_SHARE → MARKET_PRICE_COMPARISON
→ VALUATION_DERIVED_RISK → VALUATION_RECOMMENDATION_REASON
```

**IMPLEMENTATION.** `finance/validity.py` — `Validity`, `ValidatedMetric`,
`DEPENDENCIES`, `propagate()`, `root_causes()`. `[ENFORCED]` —
`workflow.py::_build_validity_graph` builds the graph from canonical
evidence, invalidates on an unresolved debt conflict and on an INVALID TTM
construction, calls `propagate()`, and does so **before** the valuation, so
`build_dcf_input_packet` can read it. A required input that is INVALID in the
graph refuses the packet by construction.

⚠️ One route to INVALID remains unreachable in production: `total_debt` is
always reconciled with `reported_is_authoritative=True` (§10), so a debt
conflict resolves rather than invalidating. Absence still refuses the packet;
a genuine unresolvable-total-debt case has no live producer today.

---

## 10. Conflict resolution

**RULE.** Compatibility **first**, authority **second**. A material conflict
either resolves by documented precedence or invalidates the metric. It is
never left as "whichever number was selected first".

**Precedence policy.** An issuer's own consolidated total outranks a sum this
system assembled from components, *once semantic identity is established* — a
sum assembled here can omit or double-count a component the issuer included.
This is a semantic rule, not a provider preference.

**FORBIDDEN.** Using precedence across incompatible identities:
`CURRENT_DEBT` overriding `TOTAL_DEBT`; `LONG_TERM_DEBT` substituting for
`TOTAL_DEBT`; current shares overriding a weighted average.

**IMPLEMENTATION.** `validity.py::resolve_or_invalidate`, wired in
`net_debt.py`. `[ENFORCED]` for debt.

---

## 11. Guidance

**RULE.** Issue period and target period are different fields and are never
collapsed. `issued_with_period = Q2 FY2026`, `target_period = FY2026`.

**RULE.** Forward information is typed: `CURRENT_QUARTER_GUIDANCE`,
`CURRENT_FY_GUIDANCE`, `NEXT_FY_GUIDANCE`, `MULTI_YEAR_GUIDANCE`,
`LONG_TERM_FRAMEWORK`, `MANAGEMENT_TARGET`, `ASPIRATIONAL_TARGET`, `OTHER`.
Only the first four may anchor a forecast for a named period.

**RULE (hard invariant).** These may **never** populate a DCF revenue-growth
assumption: `FREE_CASH_FLOW_GROWTH`, `OPERATING_CASH_FLOW_GROWTH`,
`EPS_GROWTH`, `NET_INCOME_GROWTH`, `EBITDA_GROWTH`, margins, and any revenue
**component** (subscription, service, product, segment). Only
`CONSOLIDATED_REVENUE_GROWTH`, or a period-checked derivation from
`CONSOLIDATED_REVENUE`.

The check **fails closed on unrecognised metrics** — a source never seen has
not been shown to be revenue growth.

**RULE.** Coverage is per metric. "No revenue guidance" ≠ "guidance
unavailable". Statuses: `COMPLETE_FOR_RELEVANT_METRICS / PARTIAL / MINIMAL /
UNAVAILABLE`; absence reasons distinguish `NO_GUIDANCE_EXISTS`,
`NO_REVENUE_GUIDANCE`, `GUIDANCE_EXTRACTION_FAILED`, `PARTIAL_GUIDANCE_ONLY`,
`NOT_APPLICABLE` (never searched).

**RULE (identity is the horizon CLASS, not the distance).** Guidance identity
is `(metric_id, target_period, horizon class, basis)`. `target_period_type`
distinguishes CURRENT_FISCAL_YEAR from NEXT_FISCAL_YEAR, and that difference
is a property of the ISSUE DATE, not of the target: one company restating one
full-year outlook in March and again in June produces both labels for the
same year. Collapsed to one class, so the two are recognised as one statement
and the older is SUPERSEDED. A long-term framework and an outlook for a named
year are still two statements; so are a quarter and a year, and GAAP and
adjusted.

**RATIONALE.** This is the same rule as "issue period and target period are
never collapsed", read in the other direction: information about when a
statement was made must not enter the identity of what it is about. Live, one
$90B FY2027 outlook occupied two identities and neither superseded the other.
The values agreed, so nothing was corrupted — but had the company RAISED its
outlook, the analysis would have carried the old figure and the new one side
by side as current guidance for one year.

**RULE (source qualification).** A number in a historical table is not
guidance. Proximity to a forward-looking word does not qualify a figure: an
earnings release is mostly condensed statements, reconciliations and
share-count tables, and those words are scattered through the prose around
them. A published GuidanceItem must carry validated prospective semantics —
forward-looking context *about this figure*, a target period, a target period
type, a metric identity, a prospective value or range, and an evidence
location — and must not sit inside a block the release itself presents as
reported results.

`GuidanceMetric.prospective_evidence` records the text that qualified the
figure rather than asserting a boolean, because a boolean would have recorded
the wrong verdict just as confidently. A live run published a historical
weighted-average diluted share count as guided share count.

**FAILURE STATUS.** `GUIDANCE_SOURCE_NOT_PROSPECTIVE`.

**RULE (tables).** A guidance table states two things per number: the ROW is
the metric, the COLUMN is the period. Parsing preserves both or returns
nothing — a table parsed half-right attaches real numbers to the wrong period
and is indistinguishable from correct guidance.

**FAILURE STATUS.** `GUIDANCE_METRIC_MISMATCH`,
`GUIDANCE_PERIOD_INCOMPATIBLE`, `GUIDANCE_NOT_PERIOD_COMMITMENT`.

**IMPLEMENTATION.** `guidance.py`, `guidance_tables.py`. `[ENFORCED]`.

**Supersession `[ENFORCED]`.** `guidance.py::guidance_identity` keys a
statement on `(metric, target period, target period type, basis)` and
`resolve_guidance_status` assigns CURRENT/SUPERSEDED within an identity only.
"Latest filing wins" was the wrong rule twice over: it dropped a still-current
full-year outlook because a later release guided only the next quarter, and it
kept a revised figure beside the figure it revised when both arrived in one
document.

---

## 12. Business-model suitability

**RULE.** Classification is evidence-driven — SEC SIC code plus the concepts
the issuer actually files. Never a ticker, name, or vendor sector label. (A
live broker-dealer's vendor sector read "Technology / Software"; its SIC was
6211.)

**RULE (precedence between the two signals).** Filed concepts may SUPPLY a
classification where no SIC code exists, and may REFINE one within financial
services. They may overturn a code that positively places the issuer OUTSIDE
financial services only on a concept that is **definitive** of the other
model — deposits for a bank, segregated customer cash for a broker-dealer,
policyholder benefits for an insurer.

**RATIONALE.** A large telecom (SIC 4813) was classified BANK on two tags —
loans receivable and a loss provision — filed for its device-payment book,
and its valuation was refused as not applicable to its business model. The
question is never how MANY financial concepts an issuer reports: anyone with
a captive financing operation books loans receivable, and only a bank takes
deposits. Counting harder cannot separate them; asking which concept can.

**RULE.** Metric suitability is a property of a metric **and a use**:
`PRIMARY / SUPPORTING / LOW_INFORMATION_VALUE / NOT_ECONOMICALLY_COMPARABLE /
NOT_APPLICABLE`.

**RULE.** Suitability propagates to Snapshot, Bull, Bear, Risk, valuation and
recommendation — not only the DCF. A metric rejected as owner FCF cannot
later support an owner-cash claim anywhere.

**ALLOWED.** Reporting an insurer's OCF−CapEx (relabelled "cash flow after
CapEx").
**FORBIDDEN.** "Strong free cash flow demonstrates cash generation" for that
same issuer; "current ratio below 1.0 indicates liquidity stress" where the
ratio is low-information.

**FAILURE STATUS.** `CASH_FLOW_SEMANTIC_MISUSE`,
`GENERIC_RATIO_INTERPRETATION_NOT_SUPPORTED`.

**IMPLEMENTATION.** `business_model.py`, `metric_policy.py`. `[ENFORCED]` —
violations are stubbed via quarantine, never fatal (see §17).

---

## 13. Validated DCF input

**RULE.** The model receives only inputs that passed: period alignment,
metric compatibility, business-model applicability, cash-flow suitability,
guidance compatibility, profitability normalization, net debt, share basis,
bounds.

**RULE (order).** raw evidence → semantic validation → valid derivation →
proposal → bound validation → optional clamp → DCF.
**FORBIDDEN.** bad derivation → nonsense value → clamp → DCF. *A clamp must
never repair a semantic error* — it turns a measurement of the wrong quantity
into a plausible one, which is harder to notice than the original mistake.

**IMPLEMENTATION.** `finance/dcf_packet.py::ValidatedDCFInputPacket`,
`build_dcf_input_packet`; wired in `workflow.py::_build_validated_dcf_packet`
and `_run_validated_dcf`. `[ENFORCED]` — there is exactly one call site for
`finance.dcf_model`, it is the adapter, and the adapter takes a packet and
nothing else: the engine's arguments live INSIDE the frozen packet, so a
caller cannot assemble them without passing validation.

---

## 14. Valuation status and gate

**RULE.** Separate arithmetic validity from economic applicability from
forecast-path validity.

`VALID_AND_APPLICABLE`, `VALID_BUT_NOT_APPLICABLE`,
`NOT_VALID_FOR_CURRENT_FORECAST_PATH`, `LIMITED`, `INVALID`.

**FORBIDDEN.** Rendering "model invalid" when the model never applied, or
when it correctly declined to grow a negative terminal cash flow into a
perpetuity. The model working is not the model failing.

**RULE.** Only valuation evidence that is valid for research may support
"overvalued", "undervalued", "market above bull", a premium/discount, a
modelled return, or a valuation-derived risk.

**IMPLEMENTATION.** `dcf_packet.py::ValuationStatus`, `classify_valuation`,
`build_valuation_research_evidence`; one status per run on
`facts["valuation_status"]` (`workflow.py::_valuation_status`), read by
`evidence.py::valuation_evidence_status` and by the report model.
`[ENFORCED]` — every numeric valuation id in the index is behind the gate,
and the compact renderer states the status BY CAUSE rather than calling a
working model invalid.

Note: a suitability that was never *assessed* is not a LIMITED one.
`SuitabilityAssessment.assessed` distinguishes them, because reporting the
second as the first withheld a usable valuation's evidence entirely.

---

## 15. Canonical research evidence

**RULE.** Every research role consumes the same packet. No role reaches into
raw provider facts. Namespaces stay separated: `current.*`, `historical.*`,
`business_model.*`, `dcf.*`, `technical.*`.

**IMPLEMENTATION.** `evidence.py::build_evidence_index`. `[ENFORCED]` for the
canonical current/historical split and the business-model packet.

The `fundamental_metrics` fallbacks now live in
`report_model.py::_build_snapshot` and every one of them is LABELLED — net
debt's included, which was the last unlabelled case. They fire only where no
canonical state exists (the non-SEC provider path). `[ENFORCED]`.

---

## 16. Research output validation

**RULE.** Empty and placeholder claims cannot satisfy a schema. `min_items` is
counted on entries that **survive** filtering, not on entries supplied.

**RULE.** ONE minimum-content rule, applied to every claim-bearing field.
Two had grown up side by side — `_string_list` dropped anything under
`_MIN_CLAIM_CHARS`, while `_str_field` (which validates every `claim`,
`risk`, `thesis` and `statement`, the fields a reader actually reads)
accepted anything that was not whitespace. A `claim` of "." passed the
schema, passed quarantine and rendered as a blank bullet.

`StockAnalysisReportModel` refuses construction with an empty claim, so a
future path that skips the schema still cannot publish one. The renderer does
NOT filter blanks: a blank silently dropped at render time is
indistinguishable from a claim never made, and the stage producing it goes on
producing them.

**RULE.** Claims are validated against evidence semantics — a historical
annual figure described as current, an insurer's simple FCF described as
owner FCF, a framework described as current-year guidance, a technical
indicator used as a prediction.

**IMPLEMENTATION.** `research_pipeline.py`, `claim_validation.py`,
`content_policy.py`. `[ENFORCED]`.

---

## 17. Consequence policy — block the sentence, not the stage

**RULE.** A semantic violation in role output is **stubbed**, never fatal.

**RATIONALE.** This was learned the expensive way. Cascade-fatal was tried
and cost a live insurer its entire final synthesis: the stage exhausted its
repairs still writing the same claim, and the report returned with no
recommendation at all — a worse outcome than the sentence it was blocking.
Plain OVERSTATEMENT does not work either, because these claims land in
`claims`, `rationale` and `key_risks`, whose policy is FAIL precisely because
*dropping* an element breaches a `min_items`.

`Severity.SEMANTIC_MISUSE` is therefore always stubbed: the element stays,
structure stays valid, the sentence is replaced. FABRICATION remains fatal.

**IMPLEMENTATION.** `content_policy.py`, `research_pipeline.py::apply_quarantine`.
`[ENFORCED]`.

---

## 18. Pipeline status, risk, and rendering

**RULE.** `COMPLETE` means **every required stage completed**. A report must
never say COMPLETE on the same page as a failed stage.
`COMPLETE / DEGRADED / PARTIAL / FAILED`.

**RULE.** A missing adversarial stage (bull, bear, rebuttal) caps confidence.
Not a forced HOLD — the recommendation stays the model's to choose from valid
evidence.

**RULE.** Company risk and analysis limitations are separate dimensions with
separate levels. "No DCF available" is a `VALUATION_METHOD_LIMITATION`, never
HIGH company risk. Reclassification is one-directional: it may move a finding
*out* of COMPANY_RISK, never into it.

**RULE.** Conditions must be future-looking, not already satisfied (unless
persistence is explicit), not built on internal model bounds, and must not
reference a diagnostic this analysis never raised.

**RULE (compact).** Compact mode shows validated conclusions only — no stage
names, schema errors, retry counts, evidence ids, blank bullets, duplicated
warnings, or modelled per-share values when the equity bridge is invalid.
Each limitation is stated **once**.

**RULE (root cause).** Report the cause, not the five things it caused. A
symptom whose specific cause is present in the same run is suppressed.

**RULE (three axes, never collapsed).** Arithmetic validity (did the model's
own checks pass — `dcf.validation_status`), valuation eligibility (may its
output be used as research evidence — `facts["valuation_status"]`) and
research readiness (how much weight this run can bear) are three questions
with three answers. "The DCF passed validation" answers only the first, and
the case that matters is where the first says yes and the second says no — a
model whose arithmetic is perfect resting on a forecast set by a configured
bound rather than by the company's economics. A readiness reason must name
which axis it is about.

**IMPLEMENTATION.** `finance/report_model.py` decides; `workflow.py`
formats. `[ENFORCED]` — the compact renderer makes no finance decision: it
reads a `StockAnalysisReportModel` and formats it, and a modelled value the
model did not publish is absent from the object rather than suppressed at
render time.

---

## 19. Test strategy

Four permanent categories. **Ticker fixtures prove a regression did not come
back; they cannot prove an architecture** — each exercises only the path its
own issuer happens to take, which is why every new stock kept finding a new
route to the same class of error.

1. **Semantic matrix** — period × metric × share × debt × guidance
   combinations, allow/reject asserted.
2. **Forbidden operation** — a named bad transition cannot occur.
3. **Property / metamorphic** — identical numbers, different metadata, and
   the behaviour must differ. This is what proves the system reasons about
   finance rather than floats.
4. **Canary fixtures** — one per business-model class.

Files: `test_finance_semantics.py`, `test_finance_semantic_boundaries.py`,
`test_finance_kernel_invariants.py`, `test_finance_validity_cascade.py`,
`test_finance_guidance_metric_identity.py`, `test_finance_guidance_tables.py`,
`test_finance_metric_policy.py`, `test_finance_business_model.py`,
`test_finance_derived_metrics.py`, `test_finance_runtime_boundaries.py`,
`test_finance_canaries.py`.

**Canaries.** `tests/fixtures/canary_company.py` builds a synthetic issuer's
SEC and Yahoo payloads; `canary_definitions.py` holds eight business-model
classes (mature profitable, high-growth profitable, loss-making growth,
insurer, broker-dealer, foreign private issuer, unusual-item company,
complex capital structure) plus two DEFECT classes used by the negative
runtime tests. Each canary runs the real `run_full_stock_analysis` and every
assertion crosses at least two boundaries.

---

## 20. Development rule

When a new stock exposes a bug:

```
new failure
  → identify the violated or MISSING general invariant
  → add a generalized failing test
  → fix the central kernel
  → run the semantic suite
  → run the canaries
  → rerun the live example
```

**Never** "fix XYZ". Ticker fixtures are permitted as regression evidence.
Production code stays ticker-neutral: no `if ticker ==`, no company-name
branch, no `special_case[...]`.

---

## 21. Division of responsibility

Deterministic (~80%): freshness, periods, TTM, metric identity, semantic
compatibility, reconciliation, derived metrics, business-model suitability,
guidance normalization, DCF input validity, dependency invalidation, evidence
eligibility, confidence caps, pipeline state.

LLM (~20%): bull/bear interpretation, rebuttal, risk explanation, final
synthesis, recommendation classification.

**The LLM may not repair accounting or data semantics through prose.** Where
policy says a metric cannot support a claim, no wording makes it able to.

---

## 22. Protected architecture

Router chooses only local vs Claude. ToolExecutor is the sole execution
authority. Role agents have no tools and no network. No silent provider
fallback, no browser fallback, no scraping, no brokerage, no trade execution,
no position sizing, no secret leakage. `router.py` and `tools/executor.py`
are unchanged and their hashes are verified before and after every phase.

---

## 23. Open gaps

Carried honestly so the next change starts from the real state:

1. **`total_debt` cannot become INVALID in production** (§9/§10). The cascade
   is wired and runs, and absence refuses the packet — but the only debt
   reconciliation calls `resolve_or_invalidate` with
   `reported_is_authoritative=True`, which always resolves. The INVALID
   branch is proven by unit test and by an injected graph, not by a live
   producer.
2. **Two TTM builders** with different field vocabularies (§6).
   `ttm.py::build_ttm` (`start_date`/`end_date`) and
   `freshness.py::build_ttm` (`period_start`/`period_end`). Consolidation
   candidate.
3. **An unusual item present in only one quarter of a trailing window cannot
   be measured** (§12). `build_unusual_items` needs a constructible TTM for
   the concept, and an issuer that tags a charge in one quarter and nowhere
   else produces none — so the charge is visible in the annual series and
   invisible to the trailing-window adjustment.
4. **The non-SEC provider path has no canonical state**, so its Snapshot runs
   on labelled `fundamental.*` fallbacks and its valuation suitability is
   `assessed=False`. Correct, and honestly labelled, but it is a second
   quality tier and the report does not say so in as many words.
5. `test_finance_yahoo_provider.py` (18 tests) has not run since a Windows
   Application Control policy began blocking a pandas DLL — environmental,
   unrelated to this code, but the suite is not fully verified.
6. **Live verification covers five issuers, deterministic path only.** The
   staged research pipeline was not driven live (it needs a local model), so
   the role-facing half of §15-§17 is verified by fixture and canary rather
   than against a live model's output.
