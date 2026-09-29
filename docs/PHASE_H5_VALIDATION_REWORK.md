# Validation layer rework

Spec for the staged research pipeline's output validation. Implement in phases.
Stop after each phase for review.

> **Amendments applied** (agreed 2026-08-13), marked **[A1]**–**[A5]** where they
> change the original text:
>
> * **A1** — Phase 4's imperative check excludes enum-valued fields.
> * **A2** — Phase 1's `caveats` drops unknown members with a warning; only
>   `confidence` rejects.
> * **A3** — Phase order is **2 → 3 → 5(metrics) → 1 → 4**.
> * **A4** — New Phase 0: persist stage outputs, because two acceptance
>   criteria assume a replay capability that does not exist.
> * **A5** — Phase 5 deletion criterion is run/ticker coverage, not elapsed
>   time.

## Working method

Before writing any code in a phase, explore the repo and report back. Then
propose a plan for that phase only. Do not start implementing until approved.

Each phase gets its own commit and its own tests. Do not bundle phases.

## Invariants

These hold at every commit. If a change would break one, stop and say so
instead of working around it.

1. The Phase H.3 output ("If you do not currently hold a position: AVOID")
   stays impossible to emit. A regression test asserts this specific string,
   and near variants, are rejected.
2. The pipeline fails closed. No invalid text reaches a completed checkpoint.
   This is a statement about what lands in the checkpoint, not about how much
   work gets discarded on the way there.
3. Determinism. Same input produces the same result. No sampling, no
   wall-clock, no unordered iteration in any validation path.
4. No per-ticker special-casing anywhere. No ticker symbols in the validation
   code or its config.

## Problem being solved

Six sequential stages emit JSON. Each stage's JSON is scanned by 47 literal
regex patterns. Any match fails the stage; a failed stage cascades, skipping
everything downstream.

The 47 patterns are two unrelated checks wearing the same mechanism:

* **8** guard fabrication (position size, entry/exit price, stop-loss, target
  allocation, "if you hold", order/trading instructions). Closed set. Regex is
  right.
* **39** guard overstatement (superlatives, "confirms", "intrinsic value",
  "achievable", "price target", consensus language). Unbounded English judged
  by substring match. Not enumerable, and five rounds of vocabulary patching
  have demonstrated that.

The failures are concentrated in four mechanisms:

1. Correct writing collides. A stage writes "these are modeled values, not a
   price target" and the matcher sees the banned phrase without the negation
   attached to it.
2. Repair amplifies. The repair path sends forbidden vocabulary back to the
   model, which then writes more disclaimers using more of those words.
   Observed live: 1 violation, then 4, then 3, then the stage died.
3. Stages duplicate the renderer. `_COMPACT_DISCLAIMER` already emits the
   caveat programmatically. Stages restating it inside their JSON adds nothing
   and causes most of the collisions.
4. One match is fatal to six stages.

---

## Phase 0 — persist stage outputs **[A4]**

Prerequisite, not a rework step. Two later acceptance criteria ("replay
existing runs", "replay the WM case") assume stored stage output. Nothing
persists it today: `logs/interactions.jsonl` records tool calls only.

* Write each `ResearchPipelineResult` — every checkpoint's status, error,
  validated output, token counts — to a run artifact on disk, keyed by
  symbol and timestamp.
* Opt-in by config, off by default; this is diagnostic capture, not a product
  feature.
* No behavior change to any validation path.

Acceptance: a completed run produces a replayable artifact containing every
stage's raw and validated output. A failed run produces one too, including the
failing text.

## Phase 2 — field-level quarantine *(runs first)* **[A3]**

Replace the pass/fail outcome with pass / quarantine / fail.

* Fabrication-class match: fail the stage, stop the cascade. Unchanged.
* Overstatement-class match: quarantine the offending field. Drop it or
  substitute a renderer-generated stub, keep the rest of the stage output,
  continue the pipeline.
* Every quarantine is recorded in the checkpoint with the field path, the
  pattern ID that fired, and the matched span. Part of the run artifact, not a
  log line.

**Fabrication class is 10 patterns, not 8.** `reader-directed investment
imperative` (×2) addresses the reader as an agent who should act — the H.3
harm mechanism — so it is cascade-fatal from this phase, not quarantinable.
This also makes Phase 4 a generalization of an existing rule rather than a
re-tightening of a relaxed one.

Load-bearing fields: `research_manager.balanced_assessment` is interpolated
into both the `risk_reviewer` and `final_investment_synthesizer` prompts.
Quarantining it degrades two downstream stages, not just the rendering. The
plan must state, per field, whether a stub is acceptable or the field forces
the fail path.

Acceptance: a stage with one overstatement match in one field completes, the
downstream stages run, and the checkpoint contains both the surviving output
and the quarantine record. The offending text appears nowhere in the rendered
report.

## Phase 3 — repair prompt

Rewrite it. The current path sends forbidden vocabulary back to the model,
which is what produces the escalation.

* Send only the single offending field, with the matched span redacted.
* Give a positive instruction: state what was computed, cite the evidence ID.
* Never name the banned words, never list categories of banned words.
* One retry, bounded. Second failure falls through to Phase 2 quarantine.

**The vocabulary originates in two places, not one.** The correction prompt is
one; `_validate_claim_fidelity`'s raised error message is the other, and that
message is interpolated into the correction. Both must be fixed or the leak
remains.

Acceptance: a test asserts neither the correction prompt nor the raised error
message contains any member of the pattern vocabulary. Replay the WM 1→4→3
case (captured in Phase 0) and show it does not escalate.

## Phase 5 — demote the 39 to advisory *(metrics half runs third)* **[A3]**

Split into two parts. The metrics half runs before Phase 1, because Phase 1
needs its data to size the caveat enums.

**5a — collect.** The 39 emit warnings into a metrics record attached to each
run: pattern ID, field path, matched span, and whether the run otherwise
passed. Gating behavior unchanged in this half.

**5b — demote and prune.** The 39 stop gating anything. A small script
aggregates metrics across runs and reports, per pattern, how many times it
fired and on what text.

Deletion criterion **[A5]**: a pattern is deletable when it has fired zero
times across at least 30 runs covering at least 10 distinct tickers. Not
elapsed time — this tool runs occasionally, and a calendar window says nothing
about coverage. Patterns with genuine hits stay and are reconsidered
individually. Nothing is deleted in 5a.

Acceptance: the aggregation script produces a per-pattern table. Zero gating
behavior from the 39 after 5b.

## Phase 1 — remove caveat prose from the scanned surface

Stage output stops carrying free-text caveats. Claims and structured
qualifiers only.

```json
{
  "claim": "DCF midpoint output is $142 at 9.1% WACC, 2.5% terminal growth",
  "evidence": ["EV-014", "EV-031"],
  "basis": "dcf_model",
  "confidence": "low",
  "caveats": ["model_output_not_forecast", "sensitivity_wide"]
}
```

* `confidence` is a closed enum, schema-validated, and **rejects** unknown
  members — it is genuinely small and closed.
* `caveats` is an open enum. Unknown members are **dropped with a recorded
  warning, never rejected** **[A2]**. The space of things worth caveating
  about a company is not enumerable, and a closed fatal vocabulary here would
  reintroduce the exact failure class this spec exists to remove — shorter
  list, same shape.
* Enum members are derived from the Phase 5a metrics, not from a manual
  sample. List what is found and propose the set before coding it.
* `basis` identifies which computation produced the claim.
* The renderer composes caveat prose from `caveats` and `confidence`,
  alongside the existing `_COMPACT_DISCLAIMER`. Stage prose never restates it.
* Update stage prompts to ask for claims and enum selections, never hedging
  language.

Acceptance: schema rejects an unknown `confidence`; schema drops and records an
unknown `caveat`. A stage whose model output contains a caveat sentence has
nowhere to put it. Runs replayed from Phase 0 artifacts show which of the 47
patterns stop firing.

## Phase 4 — redefine the fabrication class structurally

The 8 original patterns are topic bans. The harm mechanism in the H.3 incident
was that the output addressed the reader as an agent who holds a position and
should act. Ban the addressing directly:

* second person anywhere in a stage **prose** field: `you`, `your`, `yours`
* sentence-initial imperative from a closed verb list: buy, sell, hold, avoid,
  add, trim, exit, enter, wait, consider, take

Both are fabrication-class, so both are cascade-fatal. Keep the existing
patterns alongside them.

**Enum-valued fields are excluded from both checks** **[A1]**. The
`recommendation` field is a validated enum whose members include `buy`,
`hold`, `sell`, `avoid` — a bare imperative verb at position zero. A check
applied to every string field makes every non-`insufficient_evidence`
recommendation cascade-fatal, deleting a feature the project deliberately
built. The check runs on prose fields only, enumerated explicitly, never by
scanning all strings.

Acceptance: the H.3 string trips this on two independent rules. Word-boundary
matching, so "your" does not fire on substrings. Every enum field is tested to
confirm it is exempt, `recommendation: "buy"` specifically. Quoted material, if
exempted, is exempted structurally (a dedicated quote field excluded from the
scan), never by a regex carve-out — every regex carve-out in this codebase has
become the next bug.

---

## Out of scope

* No LLM-as-judge for modality. It breaks invariant 3 across model versions.
  If it returns it will be pinned, temperature 0, cached by content hash, and
  quarantine-only.
* No new vocabulary added to the 39. That approach is what this spec replaces.
* No changes to the evidence-ID citation check. It stays as-is and stays
  cascade-fatal.

## Reporting

At the end of each phase: what changed, what tests were added, which of the
four invariants each test covers, and anything found in the codebase that
contradicts the description above.
