"""Finance Extraction V2 — a semantic reader in front of the existing kernel.

WHAT THIS REPLACES, AND WHAT IT DOES NOT

The finance architecture below validated candidates is unchanged and stays
that way:

    CandidateFinancialFacts
      -> semantic validation            finance/semantics.py
      -> CanonicalFinancialState        finance/freshness.py, canonical.py
      -> derived/dependency graph       finance/validity.py
      -> ValidatedDCFInputPacket        finance/dcf_packet.py
      -> ValuationEvidenceGate          finance/evidence.py
      -> CanonicalResearchEvidence      finance/evidence.py
      -> research pipeline              finance/research_pipeline.py
      -> StockAnalysisReportModel       finance/report_model.py
      -> renderer                       finance/workflow.py

V2 replaces one layer only: how a document becomes a CANDIDATE. Nothing here
is a second canonical finance system, and nothing here may write to any of
the objects above.

WHY

Guidance extraction had grown into a regex vocabulary that had to be widened
every time an issuer wrote a sentence a previous issuer had not. Each
widening was correct and each left the next spelling unhandled: "revenue
guidance of $90 billion" versus "guidance for revenue of $90 billion",
"expected" versus "expect", "68 percent" versus "68%". That is not a pattern
problem. Reading an English sentence and saying what quantity it commits the
company to is a semantic task, and a regex is the wrong instrument for it.

So the division is by KIND OF QUESTION rather than by convenience:

    DETERMINISTIC   numbers, units, currencies, dates, fiscal periods,
                    metric compatibility, accounting basis, target-period
                    validity, supersession, freshness, dependency
                    invalidation, forecast-horizon eligibility, canonical
                    state, DCF eligibility, and every acceptance decision

    SEMANTIC (LLM)  what an English sentence MEANS: which metric management
                    committed to, for which period, on which basis, and
                    whether this statement raises, reaffirms or withdraws an
                    earlier one

The model is a READER, not an authority. It proposes candidates; it never
accepts one. Every candidate is grounded against the source text and checked
against the same `finance.semantics` rules the rest of the system uses, and a
candidate that fails any check is refused rather than repaired.

MODULES

    schema.py             the candidate types and their vocabularies
    document_resolver.py  which document and period is CURRENT, deterministic
    semantic_extractor.py the LLM boundary, provider-neutral
    validator.py          the deterministic acceptance boundary
    compare.py            V1 vs V2, for measurement rather than for switching
"""

from finance.extraction.schema import (  # noqa: F401
    ExtractionOutcome,
    GuidanceCandidate,
    RejectionCode,
    ReportedActualCandidate,
    SourceType,
    StatementCompleteness,
    ValueType,
)

__all__ = [
    "ExtractionOutcome",
    "GuidanceCandidate",
    "RejectionCode",
    "ReportedActualCandidate",
    "SourceType",
    "StatementCompleteness",
    "ValueType",
]

# Bumped when the candidate shape or the acceptance rules change in a way that
# invalidates a cached extraction. Part of the cache key (section 21).
EXTRACTOR_VERSION = "v2.0"
SCHEMA_VERSION = "2026-09-02"
