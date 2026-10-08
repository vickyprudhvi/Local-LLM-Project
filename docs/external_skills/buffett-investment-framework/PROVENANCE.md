# Vendored copy — buffett-investment-framework

Vendored as a **design reference** for Phase H.34 (deterministic evidence-bounded
research claims). This is a read-only snapshot, not a runtime dependency.

- Source: https://github.com/finterm-ai/investment-skills
- Skill: `skills/buffett-investment-framework`
- Pinned tag: `v0.1.0`
- Pinned commit: `f8fcdcbd64c70f30f5ee22f365224ae070628e4d`
- License: MIT (see `LICENSE.upstream`)
- Retrieved: 2026-10-08, via `npx skills add finterm-ai/investment-skills --skill buffett-investment-framework --yes`
  into an isolated temp directory, then re-cloned directly at the pinned tag to
  record exact provenance and vendor only the content files (not the
  interactive CLI router `scripts/framework.py` or `agents/openai.yaml`, which
  are built for an agentic chat session and are not invoked by this project's
  finance pipeline).

## What this is used for

- `finance/research_pipeline.py`'s evidence-discipline guardrail text (shared by
  the bull/bear/research-manager/risk-reviewer/final-synthesizer/rebuttal
  stages) distills concepts from this framework's evidence-boundary section
  (fact vs. calculation vs. inference, supported/challenged/unresolved,
  "do not invent missing inputs", "name the gap and stop that branch").
- `.claude/skills/financial-evidence-analysis/` (this repo's own Claude Code
  skill) cites it as a reference for the same reason, for interactive use.

## What was explicitly NOT adopted

- The 65-card routing framework itself, its CLI (`scripts/framework.py`), or
  its multi-tool-call workflow — this pipeline's stages are single structured
  LLM calls over a pre-rendered evidence index, not a multi-step agent session.
- Any Buffett-specific investment philosophy (moat-first business quality,
  concentration, buy-and-hold horizon, owner-earnings valuation as the
  authoritative method) as binding policy. This project's valuation output is
  the existing deterministic FCFF DCF (`finance/dcf.py`); the skill is never
  consulted to calculate or re-derive a value.
- Any restriction that would conflict with this project's existing research
  recommendation/readiness policy (`finance/workflow.py`, `finance/dcf_packet.py`).

## Runtime dependency status

None. If this vendored copy, the `.claude` skill, or the upstream package were
ever unavailable, `finance/research_pipeline.py`'s deterministic validators
(`finance/claim_validation.py`, `finance/content_policy.py`,
`finance/metric_policy.py`) remain the actual enforcement layer and are
untouched by this vendoring. See the "Deterministic validator is the
enforcement authority" note in `finance/research_pipeline.py`.
