# Finance Extraction V2

A parallel extraction layer. It does not replace the canonical finance
architecture, and as of this writing it is **not** the production default —
`finance_extraction_mode` ships as `v1`.

## Why a second layer

V1 reads guidance out of an earnings release with regular expressions over
natural-language prose. That works for phrasings it has seen and fails
silently for phrasings it has not, and every failure has the same shape: a
number is extracted with the wrong *meaning* attached. A margin becomes an
amount. A historical table becomes an outlook. An analyst's estimate becomes
management's guidance. The number is right; the claim about it is false.

Adding a pattern fixes one sentence. The failure class is that **meaning is
not a property of a sentence's surface form**, and no amount of pattern
addition changes that.

V2 splits the job along the line where determinism actually helps:

| Stage | Layer | Why |
|---|---|---|
| Which document, which period | deterministic (`document_resolver.py`) | filing metadata is structured; there is nothing to interpret |
| What the sentence *claims* | model (`semantic_extractor.py`) | reading is the part regexes cannot do |
| Whether the claim is admissible | deterministic (`validator.py`) | a model must never be the last word on what enters the pipeline |

The model proposes; it never accepts. Every candidate crosses a deterministic
boundary of 15 checks before it becomes a `GuidanceMetric`, and the boundary
is what the benchmark measures.

## Modes

| Mode | Statements returned by | V2 runs | Use |
|---|---|---|---|
| `v1` (default) | V1 | no | production |
| `compare` | **V1** | yes | measurement — records disagreement, resolves nothing |
| `v2` | V2 | yes | evaluation; fails closed when the read fails |

`compare` returning V1's answer is the point, not a limitation. A compare mode
that preferred the newer answer would be V2-by-default wearing a diagnostic's
name.

### Wiring

`finance/extraction/runtime.py` is the seam. It is called from
`tools/finance_tools.py` and under `v1` performs exactly the call that was
there before.

The model client is injected at the **composition root**, `assistant.py`, at
import:

```python
finance_extraction_runtime.register_model_client(ask_local_raw)
```

That module already owns both halves — the configured local-model capability
and the tool layer that runs extraction — and `finance/` owns neither. A test
asserts no module under `finance/` imports `brain` or `router`.

Import-time registration is the point. Registering from inside a request
handler, or from `synthesize_report` where an `ask_local_fn` happens to be in
scope, would make the capability depend on call order — and since guidance
extraction runs *before* synthesis, a registration made there could only ever
take effect on a *later* analysis in the same process.

Registration is free: it stores a factory and calls nothing. Under `v1` no
extractor is constructed and no model is contacted.

### Model-call bounds

| Bound | Value | Source |
|---|---|---|
| sections per document | 4 | `finance_extraction_max_sections` |
| characters per section | 6000 | `finance_extraction_max_section_chars` |
| output tokens | 20000 | `finance_extraction_max_output_tokens` |
| wall clock | derived, ≥ budget ÷ 40 tok/s + 60s | `finance_extraction_timeout_seconds` |
| attempts | 1, plus at most 1 structured repair | `semantic_extractor.extract` |
| tools / network | none | `format: json`, no `tools` |

The timeout is **derived from** the budget, not configured beside it — a
configured value acts as a floor. This is the coupling
`research_stage_timeout_seconds` exists to enforce: two independent constants
drift, and then raising the budget to fix truncation silently converts
truncation failures into timeout failures.

The 20000-token budget is measured, not guessed. The configured model is a
reasoning model that spends budget on thinking tokens which never appear in
`message.content`. A 548-character section cost 7210 tokens; at an 8000 cap,
two consecutive real cases returned `completion_tokens=8002` with an **empty
body**, twice each including the repair. Truncation is now diagnosed
explicitly as `TRUNCATED_RESPONSE` rather than surfacing as a parse error —
without that flag, "the model ran out of budget while thinking" is
indistinguishable from "this document contains no guidance."

## What the model is not allowed to decide

The reader proposes measurements. It decides nothing:

| The model may propose | The model may not decide |
|---|---|
| metric, value/range, unit, denominator | canonical acceptance |
| target period, comparison period | DCF assumptions |
| basis, revision action | forecast compatibility, valuation eligibility |
| prospective status, evidence span | supersession authority |
| confidence | recommendation, risk, attractiveness |

`confidence` is the model's opinion of itself and carries no authority: a
candidate with `confidence: 1.0` whose cited sentence is not in the document
is refused before anything else is checked. The candidate schema deliberately
has no field for a recommendation, a rating or a DCF assumption — a field is
an invitation, and a test asserts none exists.

## Live semantic-reader benchmark

`python scripts/run_live_extraction_benchmark.py` — 12 cases, 21 expected
statements, read by the configured model. Model responses are cached on disk
so a scoring change re-scores for free.

Measured on `qwen3.5:397b-cloud`, extractor v2.0 / schema 2026-09-02:

| Measure | Result |
|---|---|
| semantic reader recall | 86% (18/21) |
| validator acceptance rate | **100% (18/18)** |
| end-to-end accepted recall | 86% (18/21) |
| accepted precision | **100% (10/10)** — exhaustive cases only |
| metric identity accuracy | 100% (19/19) |
| target-period accuracy | 100% (19/19) |
| unit accuracy | 100% (18/18) |
| basis accuracy | 100% (18/18) |
| revision-action accuracy | 0% (0/1) |
| economic dedup accuracy | 100% (28/28) |

Hard requirements, all met:

| | |
|---|---|
| critical false positives | **0** |
| unsupported accepted facts | **0** |
| wrong-unit accepted facts | **0** |
| wrong-target-period accepted facts | **0** |

31 candidates proposed, 28 accepted, 3 refused — every refusal correct: an
absolute metric proposed as a ratio (`UNIT_METRIC_MISMATCH`), a multi-year
framework with no resolvable fiscal period (`AMBIGUOUS_TARGET_PERIOD`), and a
leverage target stated as a multiple rather than a percentage
(`DENOMINATOR_NOT_IN_EVIDENCE`).

### Two caveats on these numbers

**The reader is not deterministic.** The same case returned 3 accepted
statements on one run and 4 on the next. A single pass is a sample, not a
fixed measurement, and the recall figure should be read with that in mind.
The acceptance boundary *is* deterministic, so the precision and
hard-requirement columns do not move.

**Precision is scored only on the 5 exhaustive cases.** On the real cases the
expected statements are the ones verified by reading, not everything the
document contains, so a correct extra statement there is not an error.

### Remaining failure classes — all reader, none validator

| Item | Class | What happened |
|---|---|---|
| `revenue@Q2 FY2027` | `VALUE_EXTRACTION` | "$91.0 billion, plus or minus 2%" read as the point 91.0 rather than 89.18–92.82. The tolerance clause is not applied. |
| `earnings_per_share@FY2026` | `MODEL_MISREAD` | the figures were in a section the model read; it reported the adjusted EPS and not the GAAP one |
| `free_cash_flow@FY2026` | `MODEL_MISREAD` | a reiterated "$18B+" floor, in a section the model read |
| `revenue@FY2027` action | `REVISION_ACTION` | "we confirm our prior revenue guidance" proposed with no `REAFFIRMED` action |

`SECTION_SELECTION` produced no misses: every expected statement was inside a
selected section, so the deterministic bounding is not what is costing recall.
All four remaining failures are reading failures. None is a reason to loosen
the acceptance boundary — no rule change could accept a candidate that was
never proposed.

### The one validator change this phase made

`denominator_is_grounded` refused `gross_margin` and `adjusted_gross_margin`
read out of "GAAP and non-GAAP gross margins are expected to be 74.9% and
75.0%, respectively, plus or minus 50 bps." Both candidates were semantically
correct and correctly grounded.

This met all four conditions of the change policy below. The spec's "neither
stated → REFUSED" governs an *absolute* metric carrying a percentage, where
21% under `operating_income` could be a margin or a growth rate. A metric
whose canonical taxonomy unit is `RATIO` has no such ambiguity — its name is
the denominator statement — and `resolve_identity` had always drawn that line
correctly. `denominator_is_grounded` restated the same decision without
consulting the taxonomy and so contradicted it.

The fix consults the same taxonomy both now use, so they cannot disagree.
Tests pin what it must *not* loosen: an absolute metric claiming a margin is
still refused, a ratio from a sentence stating no percentage is still refused,
and an unknown metric name is still refused.
See `tests/test_finance_extraction_denominator_identity.py`.

## `SECTION_SELECTION`: tabulated guidance

Fixed in `select_sections`. Found on a live QCOM compare run.

`select_sections` has two paths: a heading path when a release labels its
outlook on its own line, and a fallback that gathers sentences carrying
forward-looking vocabulary. An issuer that states guidance in a **table**
defeated the fallback. `html_to_text` flattens a table into fragments —
`Revenues $9.7B - $10.5B.` is its own "sentence" — and a table row carries
numbers, not vocabulary. It says *Revenues*, never *we expect revenues*.

So every row holding a figure was dropped, only the prose caption survived,
and the model was handed the safe-harbor boilerplate and asked to find
guidance in it. Measured on QCOM's July 2026 release: `Q4 FY26`, `9.7`,
`10.5`, `1.22`, `1.42`, `2.05`, `2.25` were all present in the document and
all absent from the section sent to the model.

**The rule.** A guidance sentence is often the *caption* of a table, so the
run of short fragments following a forward-looking sentence is absorbed with
it, ending at the first fragment long enough to be prose. This is a boundary
decision — the `[KEEP]` use of a pattern under the §24 classification. It
reads no meaning out of any row.

**What it must not do, and nearly did.** The first version kept any short
fragment carrying a digit, which absorbed `Shares outstanding were 139,933.`
following a guidance sentence — dragging reported history into a prompt
asking for forward-looking statements, which is how a results table becomes
guidance. A table is a *repetition* of short rows; one short sentence after a
paragraph is a sentence. The run is now kept only at three or more rows with
at least two carrying a figure. An existing test caught this
(`test_section_selection_falls_back_to_forward_sentences`) and it is now
pinned directly in `tests/test_finance_extraction_section_selection.py`.

The same work fixed a latent off-by-N in the `max_chars` bound: `used`
summed fragment lengths while the emitted body also carries a newline per
fragment, so a section could exceed its budget by one character per line.

**A results table is not a guidance table.** The first version of the run rule
walked out of an outlook and into the condensed income statement below it,
putting `Three Months Ended April 30 / Total revenue $4,571,779` into a prompt
asking for forward-looking statements. The benchmark caught it as a **critical
false positive** — the reader then published a component metric as
consolidated revenue growth. The run now stops at `REPORTED_RESULTS_MARKER`,
defined once in `schema.py` and used by both the reader and the boundary, so
the reader cannot send what the boundary would refuse to read. Two definitions
of one idea is how the denominator defect happened.

### `NOT_PROSPECTIVE` on table rows — fixed

The section fix is verified — the model now proposes all three QCOM
statements correctly (`revenue` 9.7–10.5, `earnings_per_share` 1.22–1.42,
`adjusted_earnings_per_share` 2.05–2.25 for Q4 FY26). **All three are then
rejected** with `NOT_PROSPECTIVE`: "nothing in the cited sentence establishes
the figure as forward-looking."

This is the same underlying property one layer down. `prospective_semantics`
tests the cited sentence alone, and the cited sentence is a table row, which
can never contain forward vocabulary. So tabulated guidance cannot currently
pass the boundary however well it is read.

The invariant is not wrong — it is what stops a historical results table
being read as guidance, V1's worst failure mode. What is missing is that
**prospectivity is a property of the block a figure sits in, not only of its
own sentence.** V1 already models this (`_OUTLOOK_BLOCK_RE` /
`_REPORTED_BLOCK_RE`); the V2 validator implements only the sentence-scoped
half. The governing evidence for a guidance row is its caption — here "The
following table summarizes GAAP and Non-GAAP guidance based on the current
outlook" — and a historical table's caption ("Three Months Ended…") would
still correctly govern its rows as reported.

**The fix.** `prospective_semantics` now accepts a second route:
`governing_caption` walks back from the row through the fragments before it
and returns the first prospective caption it reaches. It is not a proximity
rule — every fragment in between must be a table row, so prose between a
caption and a figure breaks the block. That is the "proximity is not
qualification" doctrine `finance/guidance.py` documents after a live run
published a historical share count as guidance because *expects* appeared in a
paragraph above the table.

It cannot admit history. A reported or analyst caption **ends** the walk
rather than being skipped, so it can never qualify anything; the sentence-level
`_HISTORICAL_TABLE` and `_ANALYST_ESTIMATE` refusals are untouched; a candidate
the model marked non-prospective is still refused; and a validator built
without document text has no block to consult and refuses. Ten refusal guards
are pinned in `tests/test_finance_extraction_prospective_block.py`, including
the identical table under a `Condensed Consolidated Statements` caption.

The caption, not the row, is recorded as `prospective_evidence` — a row does
not qualify itself, and recording it as its own evidence would assert
something the document never says.

**QCOM end to end:** all three Q4 FY2026 statements now proposed *and*
accepted — revenue 9.7–10.5, GAAP EPS 1.22–1.42, non-GAAP EPS 2.05–2.25.

### Residual risk, not yet addressed

A candidate with `action: WITHDRAWN` sets the metric's status to `WITHDRAWN`
without the validator checking that the cited sentence contains withdrawal
language. A model that hallucinated the action could therefore suppress real
guidance. No benchmark case exercises withdrawal and no live evidence shows
it happening, so under the change policy below the validator was left alone —
this is recorded as a known gap rather than fixed speculatively.

## Validator change policy

The deterministic boundary is not loosened to raise recall. A change requires
all four of:

1. the model proposed a semantically correct candidate,
2. its evidence grounding is correct,
3. the validator rejected it, and
4. the rejection violates an existing generalized invariant.

`NOT_PROPOSED` and `WRONG_PROPOSAL` are reader failures. No rule change could
have accepted a candidate that was never offered, so loosening validation in
response would trade precision away for nothing.

## §24 — Regex classification: KEEP vs DEPRECATE

The dividing line is not "regex bad". It is:

> **KEEP** a pattern that recognises *structure* — something the document's
> own shape determines, checkable without knowing what the sentence means.
>
> **DEPRECATE** a pattern that infers *meaning* — what a figure is, which
> period it belongs to, whether it is forward-looking, whose claim it is.
> These are semantic judgements that a surface form only correlates with.

A deprecated pattern is not deleted. Under V2 it moves from *deciding* to
*checking*: `validator.py` uses the same vocabulary to verify that a model's
claim is grounded in the cited sentence. Demotion from decision-maker to
witness is the whole migration.

### KEEP — structural

| Pattern | Recognises |
|---|---|
| `_FILING_INDEX_ROW`, `_FILING_INDEX_CELL` | HTML table structure in a filing index |
| `_SCALE_RE`, `_CURRENCY_OR_SCALE` | scale words and currency signs — lexical facts |
| `_YEAR_RE`, `_MULTI_YEAR_RE`, `_QUARTER_PERIOD_RE`, `_ANNUAL_PERIOD_RE` | date and period *tokens* (not which period a figure targets) |
| `_RANGE_PATTERNS`, `_TOLERANCE_PATTERN`, `_FLOOR_PATTERN`, `_MIDPOINT_PATTERN`, `_APPROXIMATE_PATTERN` | value *shape*: a range, a floor, a midpoint |
| `_CLAUSE_BOUNDARY`, `_COMPARISON_CLAUSE_RE` | sentence and clause segmentation |
| `_ADJUSTED_MARKERS` | the literal token "adjusted"/"non-GAAP" — a word, not a basis judgement |

### DEPRECATE — semantic

| Pattern | Infers | Failure class it produces |
|---|---|---|
| `_METRIC_PATTERNS` | which metric a number is | `UNIT_IDENTITY`, metric misattribution |
| `_FORWARD_MARKERS`, `_FORWARD_QUALIFIER`, `_REPORTED_ACTUAL_MARKERS` | whether a figure is forward-looking | historical table read as guidance |
| `_OUTLOOK_BLOCK_RE`, `_OUTLOOK_HEADER_RE`, `_OUTLOOK_TITLE_WORD`, `_REPORTED_BLOCK_RE`, `_REPORTED_BLOCK_TERMINATOR_RE` | where the outlook section is | `SECTION_SELECTION`, `TABLE_CONTAMINATION` |
| `_YEAR_GUIDANCE_RE`, `_REPORTING_PERIOD_RE` | which period a figure targets | `PERIOD_RESOLUTION` |
| `_PERCENT_OF_REVENUE`, `_PERCENT_YEAR_OVER_YEAR` | a percentage's denominator | margin read as growth and vice versa |
| `_WITHDRAWAL_MARKERS`, `_INCREMENT_MARKERS` | raise / lower / withdraw | `GUIDANCE_ACTION` |
| `_LONG_TERM_FRAMEWORK_RE`, `_MANAGEMENT_TARGET_RE`, `_ASPIRATIONAL_RE` | aspiration vs commitment | ambition read as guidance |
| `_BARE_FIGURE_PATTERN`, `_DUAL_BASIS_PATTERN` | attribution of an unqualified figure | analyst estimate read as management's |

A concrete illustration of the class, found while writing this document:
`_REPORTED_BLOCK_TERMINATOR_RE` contained literal `0x08` bytes where `\b` was
intended, so it had **never matched anything**. The reported-results guard it
terminates therefore reached its full span and could swallow a genuine
outlook section — the exact failure it was written to prevent — and the whole
suite passed throughout. A semantic pattern that silently stops working
produces plausible output, which is why the class is deprecated rather than
maintained. (Repaired; `_CURRENCY_OR_SCALE` had the same corruption.)

## §27 — V1 vs V2 over the benchmark

12 cases, 24 failure classes. `tests/test_finance_extraction_benchmark.py`.

| layer | cases | found (named) | accepted of offered | precision | critical FPs | wrong unit | wrong period | statements |
|---|---|---|---|---|---|---|---|---|
| V1 (regex) | 12 | 21/21 | 100% (21/21) | 91% | **2** | 0 | 1 | 36 |
| V2-validator | 12 | 9/21 | 100% (9/9) | **100%** | **0** | 0 | 0 | 9 |

V1's two critical false positives:

- `retail-fy27-outlook` — a same-store-sales figure read as `revenue_growth`
- `synthetic-analyst-estimate` — an analyst's estimate read as management's `revenue`

### What these numbers do and do not say

Read precision and critical false positives. Do not read either recall column
as a comparison:

- **V2 recall is not measured here.** The oracle proposes a candidate only
  where it can locate a sentence carrying the expected numbers, so `found
  (named)` for V2 scores the oracle's sentence-finder, not V2. What *is*
  measured is **acceptance rate**: of correct candidates actually offered,
  how many the validator accepted — 9/9. That number is the guard against the
  opposite failure, a boundary so strict it refuses everything and scores a
  meaningless 100% precision.
- **V1 recall is flattered.** On the real cases, *which* statements to verify
  was decided partly by looking at V1's output. Scoring 100% against a set
  chosen that way is partly a measurement of where the set came from. The
  precision and forbidden-statement columns carry no such bias — those were
  written from the documents against V1's behaviour.

Measuring V2's reading ability end to end requires a model call per case and
has not been run.

## Remaining failure classes

Tracked by the benchmark; none is a ticker-specific patch.

| Class | Status |
|---|---|
| `SECTION_SELECTION` | V2 addresses via `select_sections`; V1 open |
| `PERIOD_RESOLUTION` | V2 clean on benchmark; V1 has 1 |
| `UNIT_IDENTITY` | both clean on benchmark; V1's guard is pattern-shaped and fragile |
| `GUIDANCE_ACTION` | V2 models raise/lower/withdraw explicitly; V1 infers |
| `TABLE_CONTAMINATION` | V2 addresses via grounding checks; V1 open (see the `0x08` finding) |
| `ECONOMIC_DEDUP` | V2 dedups on economic identity; V1 dedups on name |
| `ACTUAL_SOURCE_PRECEDENCE` | `document_resolver.py` prefers the formal filing over the release; not yet exercised live |
| `EVIDENCE_GROUNDING` | V2-only concept; V1 has no notion of a cited sentence |

## Acceptance criteria for making V2 the default (§28)

Not met. Required first:

1. A live-model benchmark run measuring V2 recall — currently unmeasured.
2. A registered model client in the extraction path.
3. V2 recall at parity with V1 on the named set, with critical false
   positives still at zero.

Precision and the acceptance boundary are the parts that hold up today.
