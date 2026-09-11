"""The extraction seam: which layer ran, and what it is allowed to decide.

The risk this file exists to pin down is not that V2 reads badly -- the
benchmark covers that -- but that V2 becomes the production extractor by
accident. §28 says it must not, and the accidents are specific and testable:
a default that drifts, a compare mode that quietly prefers the newer answer,
a v2 mode that falls back to V1 and reports V1's statements as V2's.
"""

import pytest

from finance import guidance as gm
from finance.extraction import runtime
from finance.extraction.schema import RejectionCode
from finance.extraction.semantic_extractor import ExtractionFailure
from tools import config


# The "Reports ... Results" heading opens a reported-results block, and every
# figure after it is historical until a heading closes it -- so the outlook
# heading below is load-bearing, not decoration. Without it V1 correctly
# returns nothing and these tests would be asserting against an empty release.
DOC = (
    "Acme Corporation Reports Fourth Quarter Results. "
    "Revenue for the fourth quarter was $1.02 billion. "
    "Outlook. "
    "For the full year 2027, the company expects revenue of $4.10 billion to "
    "$4.30 billion. "
    "The company expects full year 2027 adjusted operating margin of 21% to 22%. "
)


@pytest.fixture(autouse=True)
def _no_registered_extractor():
    """Every test states its own wiring; none inherits another's."""
    runtime.register_extractor_factory(None)
    yield
    runtime.register_extractor_factory(None)


def _extract(mode=None):
    return runtime.extract_release(DOC, "ACME", "0000-1", "ex99.htm",
                                   "2027-02-11", fiscal_year=2027, mode=mode)


class _StubExtractor:
    """Returns whatever it was handed. Stands in for a model's reading."""

    def __init__(self, candidates=None, failure=None):
        self._candidates = candidates or []
        self._failure = failure
        self.calls = 0

    def extract(self, sections, *, issued_at=None, document_id=None):
        self.calls += 1
        if self._failure:
            raise self._failure
        return self._candidates


def _register(extractor):
    runtime.register_extractor_factory(lambda: extractor)
    return extractor


# ---------------------------------------------------------------------------
# The default
# ---------------------------------------------------------------------------

def test_the_shipped_default_is_v1(monkeypatch):
    """§28. The one line that decides whether V2 shipped by accident.

    Asserts the SHIPPED default -- what the code does with nothing configured
    -- not the ambient value. An operator switching this repository's own
    `.env` to `compare` is a deliberate, reversible choice; a default that
    drifted to a model-backed extractor is the accident this guards against,
    and only the unset case can tell them apart.
    """
    monkeypatch.delenv("FINANCE_EXTRACTION_MODE", raising=False)
    assert config.finance_extraction_mode() == "v1"


def test_an_unrecognised_configured_mode_falls_back_to_v1(monkeypatch):
    """A typo must not silently enable a model-backed extractor."""
    monkeypatch.setenv("FINANCE_EXTRACTION_MODE", "V2!")
    assert config.finance_extraction_mode() == "v1"


def test_v1_mode_never_builds_an_extractor():
    """Not merely 'V1 wins' -- V2 must not run at all.

    A default mode that constructs a model-backed reader and then discards
    its answer costs a model call per document on every ordinary run, which
    is a real cost paid for nothing.
    """
    stub = _register(_StubExtractor())
    release, observation = _extract("v1")
    assert stub.calls == 0
    assert observation.layer_used == "v1"
    assert observation.v2_count is None
    assert release.all_metrics


def test_v1_mode_returns_exactly_what_v1_returned_before_the_seam_existed():
    direct = gm.extract_guidance_from_text(
        DOC, "ACME", "0000-1", "ex99.htm", "2027-02-11", expected_fiscal_year=2027)
    through_seam, _observation = _extract("v1")
    assert through_seam.to_dict() == direct.to_dict()


# ---------------------------------------------------------------------------
# Compare records; it does not resolve
# ---------------------------------------------------------------------------

def test_compare_mode_returns_v1s_statements_even_when_v2_found_more():
    """§15. Compare is a measurement, not a selection.

    V2 here produces nothing while V1 produces statements; the returned
    release must still be V1's. The inverse -- V2 finding something V1 missed
    -- must equally not change the answer, which is what makes this mode safe
    to enable in a live run.
    """
    _register(_StubExtractor(candidates=[]))
    release, observation = _extract("compare")
    v1_direct = gm.extract_guidance_from_text(
        DOC, "ACME", "0000-1", "ex99.htm", "2027-02-11", expected_fiscal_year=2027)
    assert release.to_dict() == v1_direct.to_dict()
    assert observation.layer_used == "v1"
    assert observation.mode == "compare"
    assert observation.v2_count == 0


def test_compare_mode_records_the_disagreement_it_found():
    _register(_StubExtractor(candidates=[]))
    _release, observation = _extract("compare")
    assert observation.v1_count > 0
    payload = observation.to_dict()
    assert payload["v1_statements"] == observation.v1_count
    assert payload["v2_statements"] == 0
    # V1 found statements V2 did not: that is the disagreement, and it is
    # reported by KIND so a reader sees a failure class, not a sentence list.
    assert payload["disagreements_by_kind"].get("ONLY_V1") == observation.v1_count


# ---------------------------------------------------------------------------
# v2 mode fails closed
# ---------------------------------------------------------------------------

def test_v2_mode_produces_nothing_when_the_read_fails():
    """The failure that would be easiest to hide.

    A failed semantic read must not silently deliver V1's statements under
    V2's name. Empty is the honest answer: the document was not read.
    """
    _register(_StubExtractor(failure=ExtractionFailure("MODEL_ERROR", "no answer")))
    release, observation = _extract("v2")
    assert release.all_metrics == ()
    assert release.metrics == {}
    assert observation.layer_used == "v2"
    assert observation.failure_code == "MODEL_ERROR"


def test_v2_fallback_to_v1_is_off_by_default_and_says_so_when_on(monkeypatch):
    _register(_StubExtractor(failure=ExtractionFailure("MODEL_ERROR", "no answer")))
    assert config.finance_extraction_v1_fallback_enabled() is False

    monkeypatch.setattr(config, "finance_extraction_v1_fallback_enabled",
                        lambda: True)
    release, observation = _extract("v2")
    assert release.all_metrics, "fallback was enabled, so V1's answer stands"
    # ...and the observation does not let it pass as V2's work.
    assert observation.layer_used == "v1"
    assert any("V1" in note for note in observation.notes)


# ---------------------------------------------------------------------------
# Unwired and unknown modes
# ---------------------------------------------------------------------------

def test_an_unwired_compare_mode_is_reported_not_disguised_as_v1():
    """"Could not run" and "ran and found nothing" are different facts.

    With no extractor registered, `v2_statements` must be ABSENT rather than
    zero -- a zero would read as a document containing no guidance, which is
    a claim about the issuer rather than about our wiring.
    """
    _release, observation = _extract("compare")
    assert observation.failure_code == "NO_EXTRACTOR_REGISTERED"
    assert observation.v2_count is None
    assert "v2_statements" not in observation.to_dict()
    assert observation.layer_used == "v1"


def test_an_unknown_mode_falls_back_to_v1_and_names_itself():
    release, observation = _extract("v3-experimental")
    assert release.all_metrics
    assert observation.failure_code == "UNKNOWN_MODE"
    assert observation.layer_used == "v1"


# ---------------------------------------------------------------------------
# The release is a description of a filing, not of a reader
# ---------------------------------------------------------------------------

def test_with_metrics_rebuilds_both_views_together():
    """The name-keyed view is what every existing consumer reads.

    Replacing `all_metrics` alone would leave `metrics` holding the previous
    reader's answer, so the same release object would describe one filing two
    incompatible ways.
    """
    release = gm.extract_guidance_from_text(
        DOC, "ACME", "0000-1", "ex99.htm", "2027-02-11", expected_fiscal_year=2027)
    assert release.metrics

    emptied = release.with_metrics([])
    assert emptied.all_metrics == ()
    assert emptied.metrics == {}
    assert emptied.accession == release.accession
    assert emptied.filed == release.filed

    kept = [m for m in release.all_metrics][:1]
    single = release.with_metrics(kept)
    assert len(single.all_metrics) == 1
    assert set(single.metrics) == {kept[0].name}


def test_the_observation_carries_the_versions_that_produced_it():
    """§23. A recorded number is only comparable if you know what made it."""
    _release, observation = _extract("v1")
    payload = observation.to_dict()
    assert payload["extractor_version"]
    assert payload["schema_version"]


def test_rejection_codes_are_reported_by_code_not_by_sentence():
    """A count by code is a failure class; a list of sentences is not.

    Also keeps document text out of the observation, which is what makes it
    safe to log.
    """
    from finance.extraction.schema import GuidanceCandidate, ValueType

    bogus = GuidanceCandidate(
        metric_id="revenue", target_period="FY2027",
        value_type=ValueType.RANGE, low=99.0, high=99.0, unit="USD",
        source_sentence="A sentence that is not in the document.",
        confidence=0.9)
    _register(_StubExtractor(candidates=[bogus]))
    _release, observation = _extract("compare")

    assert observation.v2_count == 0
    assert observation.v2_rejections, "a refusal must be counted"
    for code, count in observation.v2_rejections.items():
        assert code in vars(RejectionCode).values()
        assert isinstance(count, int)
    assert "sentence" not in str(observation.to_dict()).lower()
