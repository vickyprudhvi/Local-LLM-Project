# Financial Document Package Pipeline

Phase H.16. A new upstream layer, `finance/documents/`. It does not replace
the canonical finance architecture in `docs/finance_correctness_spec.md`,
and as of this writing it is **not** the production default —
`finance_document_pipeline_mode` ships as `v1`.

## Why a third layer, next to `finance/extraction/` and `finance/reported_actuals/`

Every live stock failure this project has hit has had the same root cause
named across three docs now: *meaning is not a property of a sentence's (or
a filing's) surface form, and no amount of pattern addition changes that.*
`finance/extraction/` fixed this for guidance prose. `finance/reported_actuals/`
fixed the narrower case of a deterministic row-alias reader for earnings-
release tables. Two gaps remained:

1. **Document selection was narrow.** Nothing in the codebase answered "what
   are the *latest relevant* financial documents for this issuer" across
   10-K/10-Q/8-K/20-F/6-K. `finance/extraction/document_resolver.py` — despite
   its name — is a period-completeness/precedence engine over already-fetched
   XBRL, not a document selector. `finance/reported_actuals/discovery.py`
   selects earnings-release *carriers* only.
2. **There was no reader for a financing/capital event's own text.**
   `finance/structural_breaks.py` classifies a post-balance-sheet event from
   its 8-K **item code and filing description only**, by design — its own
   docstring says the deterministic path never reads the event's body for an
   amount or a funded/committed status, because that classification decides
   nothing about the numbers and was never meant to.

## Modules

    package.py            FinancialDocumentResolver / FinancialDocumentPackage:
                           WHICH filings are relevant, classified. Never
                           reads a document's text.
    actuals_schema.py      ActualFactCandidate — one proposed REPORTED figure
    actuals_extractor.py   the LLM boundary: reads a deterministically-
                           parsed table grid, proposes line items
    actuals_validator.py   the acceptance boundary; produces the EXISTING
                           `finance.reported_actuals.facts.ActualFinancialFact`
    actuals_bridge.py      the controlled-fallback GATE, and the rejoin back
                           into `finance.reported_actuals.candidates`
    event_schema.py        EventCandidate — reuses
                           `finance.structural_breaks.PostBalanceSheetEventType`
    event_extractor.py     the LLM boundary for one financing 8-K's text
    event_validator.py     the acceptance boundary — the FUNDED-vs-COMMITTED
                           grounding check lives here
    event_resolver.py      aggregates validated events; Phase-1 boundary:
                           evidence/diagnostics only, no numeric wiring
    diagnostics.py         `DocumentPipelineDiagnostics`, spec section 23
    runtime.py             the `v1 | compare | v2` mode seam, and the two
                           model-client registries

## What is genuinely new vs. what is reused

| Capability | Status |
|---|---|
| 10-K/10-Q selection | new (`package.py`), thin — `finance.reported_actuals.discovery.PERIODIC_FORMS` split by `taxonomy.ANNUAL_FORMS` |
| Earnings-release + guidance-document selection | reused — `reported_actuals.discovery.find_reported_actual_filings`, unchanged, one document now carries both classifications |
| Financing-event *eligibility* | new (`package.py::FINANCING_EVENT_ITEMS`), same item-code vocabulary `structural_breaks.py` already uses |
| Foreign-private-issuer detection | new, evidence-driven (forms actually filed — 20-F/40-F vs. none), matching spec section 12's rule for business-model classification |
| Structured-actuals resolution | unchanged — `finance.extraction.document_resolver`, `finance.actualization` |
| Deterministic table-to-fact reading | unchanged — `finance.reported_actuals.tables`/`facts`/`candidates` |
| **LLM actual-table fallback** | new — only path that did not exist |
| **Financing-event amount/funded reading** | new — only path that did not exist |
| Guidance semantic extraction | unchanged — `finance.extraction`; the package's `guidance_documents` property is diagnostic only in this phase (see below) |

## Document classification (spec section 3)

`DocumentClass`: `ACTUAL_PERIODIC_ANNUAL`, `ACTUAL_PERIODIC_QUARTERLY`,
`ACTUAL_EARNINGS_RELEASE`, `GUIDANCE_UPDATE`, `FINANCING_EVENT`,
`CORPORATE_EVENT`, `UNKNOWN`. `DocumentRef.classifications` is a **tuple**:
an earnings-release 8-K routinely carries both `ACTUAL_EARNINGS_RELEASE` and
`GUIDANCE_UPDATE`, and both are attached to the one document rather than
forcing a choice.

Routine/administrative forms (3, 4, 5, 144, DEF 14A/DEFA14A, SC 13D/13G) are
classified `CORPORATE_EVENT` and are never offered to any extractor — spec
section 2's explicit list.

## Structured-first actuals, LLM as a controlled fallback (sections 5-6)

`actuals_bridge.should_attempt_fallback(target_period_end,
structured_completeness, reported_actuals_candidates)` is the single gate. It
returns `False` — the LLM extractor is never constructed and no model is
called — whenever **either** the XBRL structured path (via
`finance.extraction.document_resolver.classify_completeness`) **or** the
existing deterministic table reader already resolves the target period to
`StatementCompleteness.COMPLETE`. Most runs never reach the LLM path.

When it does run: `finance.reported_actuals.tables.parse_filing_tables`
parses the release's tables **structurally first** — caption, rows, period
columns, scale, currency, with zero financial interpretation. Only tables
already classified a reported statement (`StatementKind.REPORTED_STATEMENTS`
— never a guidance table, never a non-GAAP reconciliation) are rendered as a
plain-text grid and shown to the model. The model may only copy a cell it
was shown; it never computes, rescales, or invents a row.

## Deterministic validation boundary (`actuals_validator.py`)

In order: evidence grounding (the cited row is one the run showed) → value
grounding (the number is literally in that row) → metric identity (one of
the reviewed field names, the SAME vocabulary
`finance.reported_actuals.facts.ROW_ALIASES` resolves prose into) → scope
(consolidated only, checked against the candidate's own claim **and**
against `finance.reported_actuals.facts._COMPONENT_ROW` language in the
cited text — defense in depth, so a model cannot self-declare a segment row
consolidated) → basis (GAAP/IFRS only; an adjusted figure is refused, not
relabelled) → **not prospective** (a hard fail-safe against
`GUIDANCE_AS_ACTUAL`, even though `select_actual_sections` never sends a
guidance table) → currency → period resolution → period plausibility (must
already have ended) → confidence floor.

An accepted candidate becomes an
`finance.reported_actuals.facts.ActualFinancialFact` — the **exact** type
the deterministic reader already produces — and is handed to the *existing*
`finance.reported_actuals.candidates.build_candidates` /
`company_facts_overlay`. There is no second conflict-detection or
completeness-classification logic; an LLM-read fact and a regex-read fact
are indistinguishable to everything downstream of this boundary.

## Financing events: funded vs. committed (`event_validator.py`)

The one rule that matters most, and the direct implementation of the hard-
safety invariant `UNDRAWN_FACILITY_COUNTED_AS_FUNDED_DEBT = 0`:
`candidate.funded = True` is **refused outright** — never silently
downgraded to `False` — unless the candidate's own cited text contains
drawdown/proceeds-received language (`_FUNDED_LANGUAGE`: drew, borrowed,
issued and sold, received net proceeds, closed the offering, funded). "The
Company entered into a $2.0 billion revolving credit facility... no amounts
have been drawn" produces `funded=False, committed=True` — it does not
produce a value the validator "corrects."

A second check, `ITEM_CODE_MISMATCH`, refuses a claimed event type the
filing's own 8-K item code cannot support (a 3.02 filing cannot become an
`ACQUISITION`) unless it is one of the specific refinements
`finance.structural_breaks.classify_security_event` already performs itself
(a 2.03 debt filing whose text names a convertible, or names funding a
repurchase).

## Phase-1 boundary: events are evidence, not numeric inputs

`ResolvedFinancingEvent` is surfaced to `facts["document_pipeline"]`
diagnostics exactly as `finance.structural_breaks.PostBalanceSheetEvent`
already is. **It does not adjust `finance/net_debt.py` or any DCF input in
this phase.** Wiring a funded, amount-grounded event into the numeric debt
graph is deliberately deferred — spec section 35 forbids changing DCF
mathematics in this phase, and doing so correctly needs the same kind of
period/currency/entity compatibility check §4 of the correctness spec
requires for everything else, which is future work once this reader has a
live-shadow track record.

## Compare mode and diagnostics (sections 22-23)

One flag: `tools.config.finance_document_pipeline_mode()`
(`FINANCE_DOCUMENT_PIPELINE_MODE`, default `v1`), the exact `v1 | compare |
v2` convention `finance.extraction` and `finance.reported_actuals` already
use.

    v1        nothing runs. No package resolved, no filing fetched beyond
              what the existing pipeline fetches, no model called.
    compare   the package is resolved and both new extractors run where a
              model client is registered; candidates and their facts overlay
              are RECORDED in `facts["document_pipeline"]` and withheld from
              the resolver. Events are still returned (see above — they are
              pure evidence with no numeric effect in either mode).
    v2        validated candidates/overlay are merged alongside
              `finance.reported_actuals`'s own output and offered to
              `finance.actualization_runtime.resolve_actual_state` through
              the SAME `extra_candidates`/`extra_facts` parameters that
              module already accepts — no new resolver, no second seam.

`finance.documents.diagnostics.DocumentPipelineDiagnostics` carries: documents
selected per bucket, periods found, whether structured facts were already
available, LLM actuals/events candidates proposed/accepted/rejection codes,
and the events summary (funded count, committed-only count). It deliberately
does not carry full filing text.

## Registration (composition root)

`assistant.py`, alongside the existing
`finance_extraction_runtime.register_model_client` call:

```python
finance_documents_runtime.register_actuals_model_client(ask_local_raw)
finance_documents_runtime.register_event_model_client(ask_local_raw)
```

Two separate registries because either extractor may be registered without
the other, and because `finance/documents/` — like every other module under
`finance/` — must never import a model client itself.

## Regex classification addendum (§18/§24 of `finance_extraction_v2.md`)

New patterns this phase adds, classified by the same rule that document
already establishes: **KEEP** recognises structure, checkable without
knowing what a sentence means; **DEPRECATE** infers meaning.

### KEEP — structural

| Pattern | Recognises | Where |
|---|---|---|
| `FINANCING_EVENT_ITEMS` | 8-K item-code eligibility, a filing-metadata fact | `package.py` |
| `_ROUTINE_CORPORATE_FORMS` | a form code naming a routine filing type | `package.py` |
| `render_table` | a parsed `FilingTable`'s own grid, already structured by `reported_actuals/tables.py` | `actuals_extractor.py` |

### DEPRECATE (as primary authority) — semantic, used only as a witness

| Pattern | Infers | Used to CHECK, never to DECIDE |
|---|---|---|
| `_FUNDED_LANGUAGE` | whether money actually moved | `event_validator.py` grounds a model's `funded=True` claim against it; the model still decides funded/committed in the first instance |
| `COMPONENT_LANGUAGE` (reused from `reported_actuals.facts._COMPONENT_ROW`) | whether a row is a segment/component breakout | `actuals_validator.py` checks a model's self-declared `scope` against it |
| `PROSPECTIVE_LANGUAGE` (reused from `finance.extraction.validator._PROSPECTIVE`) | whether text is forward-looking | `actuals_validator.py` refuses an actual candidate whose own cited text reads as an outlook |

No new pattern in this phase is asked to decide metric identity, event type,
or a target period from prose alone — those questions are the model's, and
every one of the checks above only verifies a claim the model already made.

## Acceptance-criteria status (spec sections 28/32)

**Verdict: `NOT_READY` for shadow or default consideration.**

Met: the deterministic acceptance boundaries exist and are tested (unit
tests plus a 12-case initial Golden Document Benchmark), all hard-safety
positive controls added by this phase pass, `router.py`/`tools/executor.py`
are untouched, no ticker-specific production logic was added, and the
production default is unchanged (`v1`).

Not yet done, honestly: this phase did not run a live-model benchmark (no
model was called; see `docs/finance_extraction_v2.md`'s own live benchmark
for what that measurement looks like when it exists for this layer), the
Golden Document Benchmark is 12 cases against spec section 24's target of
25-40, and no live-shadow sample was run against real filings. See the
phase completion summary for the full breakdown.
