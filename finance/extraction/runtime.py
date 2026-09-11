"""The seam where extraction mode is decided, and the only place V2 runs live.

WHY THIS MODULE EXISTS

Guidance extraction happens in the tool layer (`tools/finance_tools.py`),
which has no model client -- `run_full_stock_analysis` never receives an
`ask_local_fn`; only `synthesize_report` does. V2 reads documents with a
model. So there is no way for V2 to run in the live path unless something
that OWNS a model client hands one in. This module is that handoff, and
nothing more: a registry the owner populates and the tool reads.

WHAT IT DELIBERATELY DOES NOT DO

It does not choose. §15 and §28 are explicit that V2 must not become the
default by accident, and the most likely accident is exactly the shape this
module could have taken -- "use V2, fall back to V1 when it fails", which is
V2-by-default with a safety net and reports whichever layer answered last as
though it were the answer. So:

  * `v1` (the default) does not construct an extractor, does not import the
    semantic layer, and returns precisely what V1 returned before this module
    existed.
  * `compare` runs both and returns V1's answer. V2 is recorded, never used.
  * `v2` returns V2's answer and FAILS CLOSED when V2 cannot produce one --
    no statements and a stated reason, rather than V1's answer wearing V2's
    label. `finance_extraction_v1_fallback_enabled` (default False) is the
    one switch that changes this, and it labels what it did.

WHO REGISTERS

`assistant.py`, at import, from the `ask_local_raw` it already holds. That is
the composition root: it owns both the configured model capability and the
tool layer, and `finance/` owns neither. Import-time registration is the point
-- registering from inside a request handler would make the capability depend
on whether some earlier turn had run.

WHEN NO EXTRACTOR IS REGISTERED

A test that cleared it, or an embedding that never composed one. `v1` is
unaffected, and `compare`/`v2` record `NO_EXTRACTOR_REGISTERED` instead of
quietly behaving like `v1`. A mode that could not run is not the same as a
mode that ran and found nothing, and the observation says which.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from tools import config

from finance import guidance as gm
from finance.extraction import EXTRACTOR_VERSION, SCHEMA_VERSION


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_EXTRACTOR_FACTORY: Optional[Callable] = None
_SHARED_CACHE: Dict[str, object] = {}


def register_extractor_factory(factory: Optional[Callable]) -> None:
    """Give the extraction layer a way to build a reader.

    `factory()` returns an object satisfying `GuidanceSemanticExtractor`.
    Passing None clears it, which is what tests do to guarantee they are
    measuring V1.
    """
    global _EXTRACTOR_FACTORY
    _EXTRACTOR_FACTORY = factory


def register_model_client(ask_local_fn: Optional[Callable]) -> None:
    """The common case: register a factory built from a model client.

    Kept separate from `register_extractor_factory` so the caller does not
    have to import the semantic extractor to wire one up.
    """
    if ask_local_fn is None:
        register_extractor_factory(None)
        return

    def _factory():
        from finance.extraction.semantic_extractor import LocalModelGuidanceExtractor
        return LocalModelGuidanceExtractor(ask_local_fn, cache=_SHARED_CACHE)

    register_extractor_factory(_factory)


def extractor_registered() -> bool:
    return _EXTRACTOR_FACTORY is not None


# ---------------------------------------------------------------------------
# What a run reports about itself (§23)
# ---------------------------------------------------------------------------

@dataclass
class ExtractionObservation:
    """One document, one extraction, and what happened.

    Every field answers a question a reader would otherwise have to guess at:
    which layer produced the statements on screen, how many the other layer
    would have produced, and -- when V2 refused something -- under which code.
    Rejections are counted by code rather than logged as sentences, because a
    count by code is a failure CLASS and a list of sentences is not.
    """

    mode: str = "v1"
    layer_used: str = "v1"
    document_id: Optional[str] = None
    sections_read: int = 0
    v1_count: int = 0
    v2_count: Optional[int] = None
    v2_rejections: Dict[str, int] = field(default_factory=dict)
    v2_unsupported: int = 0
    disagreements_by_kind: Dict[str, int] = field(default_factory=dict)
    agreements: int = 0
    failure_code: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    extractor_version: str = EXTRACTOR_VERSION
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        payload = {
            "mode": self.mode,
            "layer_used": self.layer_used,
            "document_id": self.document_id,
            "sections_read": self.sections_read,
            "v1_statements": self.v1_count,
            "extractor_version": self.extractor_version,
            "schema_version": self.schema_version,
        }
        if self.v2_count is not None:
            payload["v2_statements"] = self.v2_count
            payload["v2_rejections_by_code"] = dict(self.v2_rejections)
            payload["v2_unsupported"] = self.v2_unsupported
        if self.disagreements_by_kind:
            payload["disagreements_by_kind"] = dict(self.disagreements_by_kind)
            payload["agreements"] = self.agreements
        if self.failure_code:
            payload["failure_code"] = self.failure_code
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------

def _v1_release(document_text, symbol, accession, document, filed, fiscal_year):
    return gm.extract_guidance_from_text(
        document_text, symbol, accession, document, filed,
        expected_fiscal_year=fiscal_year)


def _v2_metrics(document_text, filed, accession, observation):
    """Run the semantic layer. Returns None when it could not run at all.

    The distinction between "ran and produced nothing" and "could not run"
    is the whole point: the first is a document with no guidance in it, the
    second is a broken pipeline, and a caller that conflates them will ship
    the second as the first.
    """
    from finance.extraction.semantic_extractor import (
        ExtractionFailure, select_sections)
    from finance.extraction.validator import GuidanceCandidateValidator
    from finance.extraction.schema import RejectionCode

    sections = select_sections(document_text)
    observation.sections_read = len(sections)
    try:
        candidates = _EXTRACTOR_FACTORY().extract(
            sections, issued_at=filed, document_id=accession)
    except ExtractionFailure as failure:
        observation.failure_code = failure.code
        observation.notes.append(
            f"the semantic read did not complete ({failure.code}): {failure.message}")
        return None

    validator = GuidanceCandidateValidator(
        document_text=document_text, issued_at=filed)
    accepted, rejected = validator.validate_all(candidates)
    for _candidate, code, _reason in rejected:
        observation.v2_rejections[code] = observation.v2_rejections.get(code, 0) + 1
    observation.v2_unsupported = sum(
        1 for _c, code, _r in rejected if code in RejectionCode.UNSUPPORTED)
    observation.v2_count = len(accepted)
    return accepted


def extract_release(document_text, symbol, accession, document, filed,
                    fiscal_year=None, mode=None):
    """Extract one earnings release under the configured mode.

    Returns `(release, observation)`. The release is a `GuidanceRelease` in
    every mode, so callers downstream are unchanged -- the canonical pipeline
    below this point does not learn which layer read the document, which is
    what makes V2 a swappable extraction layer rather than a second system.
    """
    mode = (mode or config.finance_extraction_mode() or "v1").strip().lower()
    observation = ExtractionObservation(mode=mode, document_id=accession)

    release = _v1_release(document_text, symbol, accession, document, filed,
                          fiscal_year)
    v1_metrics = list(release.all_metrics or release.metrics.values())
    observation.v1_count = len(v1_metrics)

    if mode == "v1":
        return release, observation

    # An unrecognised mode is diagnosed BEFORE the wiring is. A typo in the
    # mode name reported as "no extractor registered" sends the reader off to
    # wire up a model client to fix a misspelling.
    if mode not in ("compare", "v2"):
        observation.failure_code = "UNKNOWN_MODE"
        observation.notes.append(
            f"'{mode}' is not a recognised extraction mode; V1 was used")
        return release, observation

    if not extractor_registered():
        # Not a silent degrade: the mode was asked for, could not run, and
        # says so. V1's answer is still returned because refusing to analyse
        # a stock over an unwired diagnostic mode would be the worse failure
        # -- but `layer_used` still reads v1, so nothing claims otherwise.
        observation.failure_code = "NO_EXTRACTOR_REGISTERED"
        observation.notes.append(
            f"mode '{mode}' needs a model-backed extractor and none is "
            "registered; V1 produced the statements below")
        return release, observation

    v2_metrics = _v2_metrics(document_text, filed, accession, observation)

    if mode == "compare":
        # Records, never resolves (§15).
        from finance.extraction.compare import compare
        if v2_metrics is not None:
            result = compare(v1_metrics, v2_metrics, observation.v2_rejections,
                             observation.v2_unsupported)
            observation.disagreements_by_kind = result.by_kind()
            observation.agreements = result.agreements
        return release, observation

    # mode == "v2"
    if v2_metrics is None:
        if config.finance_extraction_v1_fallback_enabled():
            observation.notes.append(
                "fell back to V1 because the V2 read failed and fallback "
                "is enabled; these statements are V1's")
            return release, observation
        # Fail closed. An empty release is a document we could not read, and
        # downstream treats absent guidance as absent -- which is true --
        # rather than inheriting V1's read under V2's name.
        observation.layer_used = "v2"
        observation.notes.append(
            "no statements: the V2 read failed and fallback is disabled")
        return release.with_metrics([]), observation

    observation.layer_used = "v2"
    return release.with_metrics(v2_metrics), observation
