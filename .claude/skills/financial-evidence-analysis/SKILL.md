---
name: financial-evidence-analysis
description: Evidence-bounded reasoning discipline for interpreting this project's finance pipeline output (CanonicalResearchEvidence, bull/bear cases, risk, valuation, recommendations). Use when asked to interpret, explain, challenge, or narrate stock-analysis evidence from finance/*.py — never for retrieving/extracting/calculating facts (SEC retrieval, XBRL mapping, guidance extraction, DCF math, technicals), which stay fully deterministic and are never this skill's job.
license: MIT (project-local wrapper); see docs/external_skills/buffett-investment-framework/ for the vendored design reference and its own MIT license
---

# Financial Evidence Analysis

This is this project's own evidence-discipline contract for reasoning about
finance-pipeline output. It draws selectively on
`docs/external_skills/buffett-investment-framework/` (vendored, pinned at tag
`v0.1.0` — see `PROVENANCE.md` there) for its evidence-boundary discipline —
**not** for its investment philosophy. See "What is and isn't adopted" below.

The authoritative, always-on enforcement of this contract lives in code:
`finance/claim_validation.py`, `finance/content_policy.py`,
`finance/metric_policy.py`, and `finance/evidence.py`. This skill is a second,
independent discipline for when a person asks Claude directly (in chat) to
interpret, narrate, or challenge finance-pipeline evidence — it is not a
substitute for those deterministic checks and does not change what the
pipeline itself validates.

## When to load this

Load it for interpretation tasks over already-computed finance evidence:
stock research narration, bull case / bear case construction, risk
interpretation, financial-strength characterization, valuation
interpretation, guidance interpretation, recommendation explanation,
upgrade/downgrade reasoning, reassessment triggers.

Do **not** load it for retrieval or calculation work: SEC/XBRL extraction,
table parsing, document normalization, period or currency resolution,
guidance extraction itself, DCF/metric calculation, or technical-indicator
computation. Those are deterministic (`finance/sec_*.py`, `finance/dcf.py`,
`finance/metrics.py`, `finance/guidance.py`, ...) and this skill has nothing
to add to them — loading it there would only risk narrating over facts that
are not yet settled.

## Canonical evidence is authoritative

1. `CanonicalResearchEvidence` / the evidence index (`finance/evidence.py`) is
   the only source of facts. Never recalculate, restate with a different
   number, or "correct" a canonical figure — if it looks wrong, say so as an
   open question, don't substitute your own arithmetic.
2. Never alter **period identity**: a `current.*` figure and a
   `fundamental.*` figure of the same name cover different periods and are
   not interchangeable; TTM, reported-annual, and guidance are three
   different kinds of number (see the "four kinds of evidence" rule already
   in `finance/research_pipeline.py`'s shared guardrails).
3. Never alter **metric identity**: a claim about revenue growth must cite
   revenue-growth evidence, not capex guidance, debt figures, or any other
   metric, even when they're nearby in the same filing.
4. Never invent a missing financial driver, segment breakdown, growth
   decomposition, or management expectation that was not supplied as
   evidence.
5. Never infer distress, strength, sustainability, or safety from a single
   metric in isolation — those are compound judgments that need the specific
   corroborating evidence the rulebook below names.

## Three levels of permitted financial meaning — nothing above them

1. **Direct fact** — read straight from canonical evidence
   (`FCF = $5.76B`, cited to its evidence ID).
2. **Derived fact** — a deterministic calculation over direct facts
   (`FCF margin = FCF / revenue`), never a new number you compute yourself in
   prose — if the pipeline didn't compute it, don't state it as a fact.
3. **Permitted interpretation** — an inference licensed by an explicit rule
   in the interpretation rulebook below, with its corroborating evidence
   present.

Anything else is **unresolved**. State what's missing and stop — do not fill
the gap with a plausible-sounding financial story. "The available evidence
does not establish the drivers of current revenue growth" is a complete,
correct answer; inventing a driver is not a more complete one.

## Interpretation rulebook (illustrative, not exhaustive — the deterministic
version of this lives in `finance/claim_validation.py` /
`finance/metric_policy.py` and is the real backstop)

| Evidence | Permits | Does NOT permit without separate evidence |
|---|---|---|
| `net_debt < 0` | "net cash position" | "debt stress", "debt may constrain capital allocation" |
| `current_ratio > 1` | "current assets exceed current liabilities" | "strong liquidity" |
| `current_ratio` near/below 1 | stating the ratio | "liquidity stress" |
| low `debt_to_equity` | "relatively low balance-sheet leverage" | "no debt-servicing risk" (needs interest coverage / maturities / liquidity / covenant evidence) |
| `revenue_growth > 0` | "revenue increased" | "sustainable growth" or "unsustainable growth" (without decomposition evidence, say the evidence doesn't establish the drivers) |
| `FCF > 0` | "positive free cash flow", with the number | "strong cash generation" (needs an explicit threshold/comparison) |
| gross debt > FCF, alone | stating both figures | "debt servicing risk" (needs coverage/maturity/liquidity/covenant/sustained-negative-FCF evidence) |
| management guidance | attributing a claim to guidance ONLY when the guided metric matches the claim's metric and the guided period matches the claim's period | citing one guided metric (e.g. capex) to support a claim about a different metric (e.g. revenue growth), or one guided period to support a claim about a different period |
| negative terminal FCFF / invalid DCF | "this forecast path does not support a valid perpetuity terminal value" | "the company will have cash-flow problems", "FCFF is unsuitable for this company" |
| missing valuation evidence | an analysis limitation | bearish/bullish directional evidence on its own |

Subjective adjectives (strong/weak/high/low/healthy/poor/concerning/robust/
sustainable/unsustainable/stretched/safe/modest/significant) require a
deterministic threshold, an explicit comparison, or sourced/attributed
language — otherwise use neutral, numeric phrasing instead.

## Bull / bear / risk / recommendation shape

- Bull claims: supported + favorable direction + real evidence IDs only.
  Render exactly as many as are genuinely supported — never pad to a round
  number, never leave a blank bullet.
- Bear claims: supported + unfavorable direction + real evidence IDs only.
- A risk needs a type, severity, evidence IDs, and a period. Missing data,
  an invalid DCF, or a missing decomposition is an **analysis limitation**,
  never automatically a **company risk**.
- Upgrade conditions must be favorable-direction evidence; downgrade
  conditions must be unfavorable-direction evidence. A debt decrease is
  never a downgrade condition. When direction can't be established, it's a
  reassessment trigger, not either side.
- A recommendation needs actual directional evidence. Missing valuation
  evidence alone is a limitation, not bearish evidence — it does not by
  itself justify AVOID or SELL.

## What is and isn't adopted from the vendored Buffett framework

Adopted: the evidence-boundary discipline — separating fact / calculation /
inference, "supported / challenged / unresolved" findings, preserving
counterevidence, naming a missing prerequisite and stopping that branch
rather than reasoning past it, citing the primary evidence behind a material
claim.

Not adopted: its investment philosophy as binding policy (moat-first quality
assessment, concentration, buy-and-hold horizon, owner-earnings as the
authoritative valuation method), its 65-card routing workflow, or anything
that would conflict with this project's existing recommendation/readiness
policy. This project's valuation number is always the deterministic FCFF DCF
in `finance/dcf.py` — this skill never recalculates or substitutes a value.

## If this skill (or the vendored reference) is unavailable

Nothing here is a runtime dependency of the finance pipeline. Degrade to
stricter, narrower claims and more "unresolved" findings — never to free-form
reasoning. The deterministic validators in `finance/*.py` enforce the same
contract independently and are never bypassed by this skill's absence.
