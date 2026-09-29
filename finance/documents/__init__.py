"""Financial Document Package pipeline — the upstream target of Phase H.16.

WHAT THIS IS, AND WHAT IT REPLACES

Nothing downstream of a validated financial fact changes. This package is a
new UPSTREAM layer that answers two questions before any semantic parsing of
a stock's own filings happens:

    "What are the latest RELEVANT financial documents for this issuer?"
    "What financial facts do those documents actually say?"

and hands the answer to the *existing* canonical pipeline through the *same*
seam `finance/reported_actuals/` already uses (`extra_candidates` /
`extra_facts` on `finance.actualization_runtime.resolve_actual_state`). There
is exactly one canonical resolver, one guidance identity, one reporting-
currency resolver and one DCF gate; this package proposes candidates, never
canonical facts.

WHY A NEW PACKAGE RATHER THAN EXTENDING `finance/extraction/` OR
`finance/reported_actuals/` IN PLACE

Those two already do real, working, narrower jobs and stay exactly as they
are:

    finance/extraction/        guidance PROSE -> GuidanceCandidate  (LLM)
    finance/reported_actuals/  earnings-release TABLES -> ActualFinancialFact
                                (deterministic row-alias matching)

`finance/document_resolver.py` (inside `finance/extraction/`) is a period
completeness/precedence engine over already-fetched XBRL, not a document
selector across filing types -- despite its name, it never picks between a
10-K, a 10-Q, an 8-K and a 6-K.

This package is what was actually missing:

    finance/documents/package.py    WHICH filings, across 10-K/10-Q/8-K/
                                     20-F/6-K, are relevant to a current
                                     analysis (`FinancialDocumentPackage`)
    finance/documents/actuals_*.py  a CONTROLLED LLM FALLBACK for actual
                                     statement lines the deterministic table
                                     reader in `reported_actuals/` could not
                                     identify, converted back into the SAME
                                     `ActualFinancialFact` type so it rejoins
                                     the one canonical pipeline
    finance/documents/event_*.py    a genuinely new capability: reading a
                                     financing/capital 8-K's own text for
                                     amount, currency and FUNDED-VS-COMMITTED
                                     status, which `finance/structural_breaks.py`
                                     deliberately never attempted (it
                                     classifies from item codes and filing
                                     DESCRIPTIONS only, and says so)

MODE

One flag, `tools.config.finance_document_pipeline_mode()`
(`FINANCE_DOCUMENT_PIPELINE_MODE`, default `v1`), following the exact
`v1 | compare | v2` convention `finance/extraction/`,
`finance/reported_actuals/` and `finance/actualization.py` already use:

    v1        (default) nothing here runs. Byte-for-byte today's behaviour.
    compare   the package is resolved and the new extractors run; candidates
              and events are RECORDED in diagnostics and withheld from the
              resolver.
    v2        candidates and validated events are OFFERED to the existing
              canonical pipeline; a failure is a failure, never a silent
              fallback wearing v2's label.

PROTECTED BOUNDARY

The LLM proposes. It never performs arithmetic, never decides canonical
acceptance, never decides DCF eligibility, and never decides whether a
financing event is funded -- that is a deterministic grounding check in
`finance/documents/event_validator.py` against the event's own cited text.
"""

# Bumped when a candidate shape or acceptance rule changes in a way that
# invalidates a cached extraction. Mirrors `finance.extraction.EXTRACTOR_VERSION`.
DOCUMENT_PIPELINE_VERSION = "v1.0"
SCHEMA_VERSION = "2026-09-15"
