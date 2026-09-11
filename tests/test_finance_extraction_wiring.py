"""Composition wiring, model-call bounds, and every way a live read can fail.

The live benchmark (`scripts/run_live_extraction_benchmark.py`) measures how
WELL the model reads. This file measures what happens when it reads badly,
slowly, or not at all -- the cases a benchmark never produces on demand and
production produces eventually.

The governing rule throughout: a failed semantic read must never become an
accepted statement, and must never be dressed up as V1's answer.
"""

import json

import pytest

from finance import guidance as gm
from finance.extraction import runtime
from finance.extraction.schema import GuidanceCandidate, ValueType
from finance.extraction.semantic_extractor import (
    ExtractionFailure,
    LocalModelGuidanceExtractor,
    build_prompt,
    select_sections,
)
from tools import config

DOC = (
    "Acme Corporation Reports Fourth Quarter Results. "
    "Revenue for the fourth quarter was $1.02 billion. "
    "Outlook. "
    "For the full year 2027, the company expects revenue of $4.10 billion to "
    "$4.30 billion. "
)


@pytest.fixture(autouse=True)
def _clean_registry():
    runtime.register_extractor_factory(None)
    yield
    runtime.register_extractor_factory(None)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def test_the_composition_root_registers_the_extractor_at_import():
    """§2. Deterministic, and independent of run order.

    Importing the application must be sufficient. If registration happened
    inside a request handler instead, this capability would exist only after
    some unrelated turn had run -- and since guidance extraction happens
    BEFORE report synthesis, a registration made there could only ever take
    effect on a LATER analysis in the same process.
    """
    runtime.register_extractor_factory(None)
    assert not runtime.extractor_registered()

    import importlib

    import assistant
    importlib.reload(assistant)

    assert runtime.extractor_registered(), (
        "importing the composition root must be enough to wire the extractor")


def test_finance_does_not_import_a_model_client():
    """§2. The domain knows the interface, never a provider.

    Checked as source text rather than by patching, because the failure mode
    is someone adding `import brain` to make a test pass.
    """
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    for folder, _dirs, files in os.walk(os.path.join(root, "finance")):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    stripped = line.strip()
                    if stripped.startswith(("import brain", "from brain import")):
                        offenders.append(f"{path}:{number}")
                    if stripped.startswith(("import router", "from router import")):
                        offenders.append(f"{path}:{number}")
    assert not offenders, offenders


def test_registering_a_client_does_not_call_it():
    """Registration is free, so the composition root may always do it.

    Under the default v1 mode no model is contacted and none is loaded.
    """
    calls = []
    runtime.register_model_client(lambda *a, **k: calls.append(k) or {"ok": True})
    release, observation = runtime.extract_release(
        DOC, "ACME", "acc", "doc", "2027-02-11", fiscal_year=2027, mode="v1")
    assert calls == []
    assert release.all_metrics
    assert observation.layer_used == "v1"


# ---------------------------------------------------------------------------
# The model call is bounded (§3)
# ---------------------------------------------------------------------------

def _capture():
    """A fake client that records the call and returns valid empty output."""
    seen = {}

    def ask(messages, tools=None, timeout=None, options=None, response_format=None):
        seen["messages"] = messages
        seen["timeout"] = timeout
        seen["options"] = options
        seen["response_format"] = response_format
        seen["tools"] = tools
        return {"ok": True, "metrics": {},
                "message": {"content": json.dumps({"statements": []})}}

    return ask, seen


def test_the_call_is_bounded_and_toolless():
    ask, seen = _capture()
    sections = select_sections(DOC)
    LocalModelGuidanceExtractor(ask).extract(sections, issued_at="2027-02-11")

    assert seen["response_format"] == "json", "structured output is not optional"
    assert seen["tools"] is None, "the reader gets no tools and no network"
    assert seen["options"]["num_predict"] == config.finance_extraction_max_output_tokens()
    assert seen["timeout"] == config.finance_extraction_timeout_seconds()


def test_the_timeout_can_never_sit_below_the_token_budget(monkeypatch):
    """The bug `research_stage_timeout_seconds` was written to prevent.

    A budget and a timeout that move independently drift, and then raising
    the budget to stop truncation silently converts truncation into timeouts.
    """
    monkeypatch.setenv("FINANCE_EXTRACTION_MAX_OUTPUT_TOKENS", "40000")
    monkeypatch.setenv("FINANCE_EXTRACTION_TIMEOUT_SECONDS", "60")
    assert config.finance_extraction_timeout_seconds() > 60


def test_sections_and_characters_are_capped(monkeypatch):
    monkeypatch.setenv("FINANCE_EXTRACTION_MAX_SECTIONS", "2")
    monkeypatch.setenv("FINANCE_EXTRACTION_MAX_SECTION_CHARS", "200")
    huge = ("Outlook\n" + "The company expects revenue of $1 billion. " * 400)
    sections = select_sections(huge)
    assert len(sections) <= 2
    assert all(len(s.text) <= 200 for s in sections)


def test_the_prompt_carries_the_section_and_not_the_filing():
    section = select_sections(DOC)[0]
    system, user = build_prompt(section, "2027-02-11")
    assert section.text in user
    assert "JSON" in user
    assert len(user) < 20000


# ---------------------------------------------------------------------------
# Fail-closed (§5)
# ---------------------------------------------------------------------------

def _extractor(responses):
    """A client returning `responses` in order; a callable entry raises it."""
    queue = list(responses)

    def ask(messages, tools=None, timeout=None, options=None, response_format=None):
        item = queue.pop(0) if queue else {"ok": False}
        if isinstance(item, Exception):
            raise item
        return item

    return LocalModelGuidanceExtractor(ask)


def _ok(content, **metrics):
    return {"ok": True, "metrics": metrics, "message": {"content": content}}


def _run_v2(extractor):
    runtime.register_extractor_factory(lambda: extractor)
    return runtime.extract_release(DOC, "ACME", "acc", "doc", "2027-02-11",
                                   fiscal_year=2027, mode="v2")


@pytest.mark.parametrize("responses,expected_code", [
    # the client reported failure (network, timeout, refusal)
    ([{"ok": False, "message": {}}, {"ok": False, "message": {}}], "MODEL_ERROR"),
    # not JSON at all, twice
    ([_ok("I think revenue will be good."), _ok("Still not JSON.")], None),
    # empty body, twice
    ([_ok(""), _ok("")], "EMPTY_RESPONSE"),
    # budget consumed by thinking tokens, twice
    ([_ok("", truncated=True, completion_tokens=8000),
      _ok("", truncated=True, completion_tokens=8000)], "TRUNCATED_RESPONSE"),
])
def test_a_failed_read_yields_no_statements(responses, expected_code):
    release, observation = _run_v2(_extractor(responses))
    assert release.all_metrics == ()
    assert release.metrics == {}
    assert observation.layer_used == "v2"
    assert observation.failure_code
    if expected_code:
        assert observation.failure_code == expected_code


def test_truncation_is_diagnosed_as_truncation_not_as_bad_json():
    """They look identical downstream and need opposite corrections.

    A reasoning model can spend an entire budget on thinking and return an
    empty body, which is indistinguishable from "no guidance in this
    document" unless the truncation flag is read.
    """
    extractor = _extractor([_ok("", truncated=True, completion_tokens=8000),
                            _ok("", truncated=True, completion_tokens=8000)])
    _release, observation = _run_v2(extractor)
    assert observation.failure_code == "TRUNCATED_RESPONSE"


def test_one_repair_is_allowed_and_only_one():
    """Bounded retry: a second failure is a failure."""
    good = json.dumps({"statements": [{
        "metric_id": "revenue", "value_type": "range",
        "low": 4.10, "high": 4.30, "unit": "currency",
        "target_period": "FY2027", "target_period_type": "annual",
        "basis": "GAAP", "action": "NEW", "prospective": True,
        "source_sentence": "For the full year 2027, the company expects "
                           "revenue of $4.10 billion to $4.30 billion.",
        "confidence": 0.9}]})

    recovered = _extractor([_ok("not json"), _ok(good)])
    release, observation = _run_v2(recovered)
    assert release.all_metrics, "one repair must be allowed"
    assert observation.failure_code is None

    runtime.register_extractor_factory(None)
    never = _extractor([_ok("not json"), _ok("still not json"), _ok(good)])
    release, observation = _run_v2(never)
    assert release.all_metrics == (), "a second failure is a failure"


def test_an_empty_extraction_is_not_a_failure():
    """A document with no guidance in it is a fact, not an error.

    Conflating the two would make every quiet release look broken.
    """
    release, observation = _run_v2(_extractor([_ok(json.dumps({"statements": []}))]))
    assert release.all_metrics == ()
    assert observation.failure_code is None
    assert observation.v2_count == 0


def test_compare_still_returns_v1s_answer_when_the_read_fails():
    """§5. A broken diagnostic must not break production."""
    runtime.register_extractor_factory(
        lambda: _extractor([{"ok": False, "message": {}},
                            {"ok": False, "message": {}}]))
    release, observation = runtime.extract_release(
        DOC, "ACME", "acc", "doc", "2027-02-11", fiscal_year=2027, mode="compare")
    direct = gm.extract_guidance_from_text(
        DOC, "ACME", "acc", "doc", "2027-02-11", expected_fiscal_year=2027)
    assert release.to_dict() == direct.to_dict()
    assert observation.layer_used == "v1"


# ---------------------------------------------------------------------------
# What the model is not allowed to decide (§4)
# ---------------------------------------------------------------------------

def test_a_confident_ungrounded_claim_is_still_refused():
    """Confidence is the model's opinion of itself; it decides nothing."""
    invented = json.dumps({"statements": [{
        "metric_id": "revenue", "value_type": "range",
        "low": 99.0, "high": 99.0, "unit": "currency",
        "target_period": "FY2027", "target_period_type": "annual",
        "basis": "GAAP", "action": "NEW", "prospective": True,
        "source_sentence": "The company expects revenue of $99 billion.",
        "confidence": 1.0}]})
    release, observation = _run_v2(_extractor([_ok(invented)]))
    assert release.all_metrics == ()
    assert observation.v2_rejections, "an ungrounded claim must be refused"


def test_the_candidate_schema_carries_no_valuation_field():
    """§4. The reader proposes measurements, never conclusions.

    A field for a recommendation, a risk or a DCF assumption would be an
    invitation, so the schema must not have one.
    """
    import dataclasses

    fields = {f.name for f in dataclasses.fields(GuidanceCandidate)}
    forbidden = {"recommendation", "rating", "risk", "attractiveness",
                 "valuation", "dcf_assumption", "forecast", "eligible",
                 "supersedes", "accepted"}
    assert not (fields & forbidden), fields & forbidden


def test_below_threshold_confidence_is_refused():
    from finance.extraction.validator import GuidanceCandidateValidator

    grounded = ("For the full year 2027, the company expects revenue of "
                "$4.10 billion to $4.30 billion.")
    candidate = GuidanceCandidate(
        metric_id="revenue", value_type=ValueType.RANGE, low=4.10, high=4.30,
        unit="currency", target_period="FY2027", target_period_type="annual",
        prospective=True, source_sentence=grounded,
        confidence=config.finance_extraction_min_confidence() - 0.01)
    metric, code, _reason = GuidanceCandidateValidator(
        document_text=DOC, issued_at="2027-02-11").validate(candidate)
    assert metric is None and code


# ---------------------------------------------------------------------------
# Observability (§6)
# ---------------------------------------------------------------------------

def test_the_observation_is_metadata_only():
    """No document text, no prompts, no secrets -- so it is safe to log."""
    runtime.register_extractor_factory(
        lambda: _extractor([_ok(json.dumps({"statements": []}))]))
    _release, observation = runtime.extract_release(
        DOC, "ACME", "acc-123", "doc", "2027-02-11", fiscal_year=2027,
        mode="compare")

    blob = json.dumps(observation.to_dict())
    assert "Acme Corporation" not in blob
    assert "4.10" not in blob
    for value in ("api_key", "authorization", "bearer", "ollama_api_key"):
        assert value not in blob.lower()

    payload = observation.to_dict()
    assert payload["document_id"] == "acc-123"
    assert payload["extractor_version"] and payload["schema_version"]
    assert payload["sections_read"] >= 1


class _StubOutcome:
    def __init__(self, payload):
        self.payload = payload

    def freshness_dict(self, _now):
        return {"fetched_at": "2027-02-11T00:00:00Z"}


class _StubLedger:
    def _clock(self):
        return "2027-02-11T00:00:00Z"


class _StubCoordinator:
    """Serves one 8-K index and one exhibit. No network, no SEC."""

    ledger = _StubLedger()

    def fetch(self, dataset, _symbol, params=None):
        if dataset == "company_submissions":
            return _StubOutcome({"filings": {"recent": {
                "form": ["8-K"], "accessionNumber": ["0000000000-27-000001"],
                "filingDate": ["2027-02-11"], "primaryDocument": ["a.htm"],
                "items": ["2.02"], "reportDate": ["2027-02-11"]}}})
        document = (params or {}).get("document", "")
        if document.endswith("-index.html"):
            # Real SEC column order: Seq | Description | Document | Type | Size.
            return _StubOutcome({"document_text": (
                "<table><tr><td>1</td><td>8-K</td>"
                '<td><a href="/x/a.htm">a.htm</a></td>'
                "<td>8-K</td><td>1000</td></tr>"
                "<tr><td>2</td><td>Press release</td>"
                '<td><a href="/x/ex99.htm">ex99.htm</a></td>'
                "<td>EX-99.1</td><td>2000</td></tr></table>")})
        return _StubOutcome({"document_text": f"<p>{DOC}</p>"})


@pytest.fixture
def _stub_sec(monkeypatch):
    import tools.finance_tools as ft

    monkeypatch.setattr(ft, "get_sec_coordinator", lambda: _StubCoordinator())
    monkeypatch.setattr(ft, "resolve_cik", lambda _c, _s: ("0000000001", "Acme Corp"))
    monkeypatch.setattr(config, "sec_edgar_enabled", lambda: True)
    return ft


def _run_tool(ft):
    return ft.SecCurrentGuidanceTool().execute({"symbol": "ACME", "fiscal_year": 2027})


def test_v1_production_payload_gains_no_new_key(_stub_sec, monkeypatch):
    """§6. The default payload must be what it always was.

    An observability field that appears on every ordinary run is a change to
    production output, not observability.
    """
    monkeypatch.setattr(config, "finance_extraction_mode", lambda: "v1")
    payload = _run_tool(_stub_sec)
    assert "extraction" not in payload
    assert payload["guidance"] is not None


def test_compare_mode_adds_observability_and_still_returns_v1s_guidance(
        _stub_sec, monkeypatch):
    monkeypatch.setattr(config, "finance_extraction_mode", lambda: "compare")
    runtime.register_extractor_factory(
        lambda: _extractor([_ok(json.dumps({"statements": []}))]))

    payload = _run_tool(_stub_sec)
    assert "extraction" in payload, "compare mode must record what it saw"

    monkeypatch.setattr(config, "finance_extraction_mode", lambda: "v1")
    runtime.register_extractor_factory(None)
    baseline = _run_tool(_stub_sec)
    assert payload["guidance"] == baseline["guidance"], (
        "compare must not change the answer, only describe it")

    observation = payload["extraction"][0]
    assert observation["mode"] == "compare"
    assert observation["layer_used"] == "v1"
    assert DOC[:40] not in json.dumps(payload["extraction"])


def test_the_guidance_tool_clock_covers_the_work_the_mode_adds(monkeypatch):
    """The defect that made guidance vanish silently.

    `_SecTool` carried a flat 30-second timeout, correct while the tool only
    fetched and pattern-matched. Wiring a semantic read into it put a
    60-200-second model call inside that 30 seconds, so the tool timed out --
    and because `finance.sec.current_guidance` is deliberately non-fatal,
    every guidance statement disappeared from the analysis without a word.

    A tool allowed to do N seconds of work must be allowed the time to do it.
    The same coupling `research_stage_timeout_seconds` enforces, one layer up.
    """
    import tools.finance_tools as ft

    monkeypatch.setattr(config, "finance_extraction_mode", lambda: "v1")
    v1_clock = ft.SecCurrentGuidanceTool().timeout_seconds
    assert v1_clock == 30, "v1 does no extra work and must be untouched"

    for mode in ("compare", "v2"):
        monkeypatch.setattr(config, "finance_extraction_mode", lambda m=mode: m)
        clock = ft.SecCurrentGuidanceTool().timeout_seconds
        # It must cover at least ONE section read, or the first model call
        # can outlive the tool that made it.
        assert clock >= config.finance_extraction_timeout_seconds(), (
            f"{mode}: the tool clock ({clock}s) is shorter than one section "
            f"read ({config.finance_extraction_timeout_seconds()}s)")
        assert clock > v1_clock


def test_raising_the_extraction_timeout_raises_the_tool_clock(monkeypatch):
    """They cannot drift, which is the whole point of deriving one from the other."""
    import tools.finance_tools as ft

    monkeypatch.setattr(config, "finance_extraction_mode", lambda: "compare")
    before = ft.SecCurrentGuidanceTool().timeout_seconds
    monkeypatch.setenv("FINANCE_EXTRACTION_TIMEOUT_SECONDS", "2000")
    assert ft.SecCurrentGuidanceTool().timeout_seconds > before


def test_the_cache_key_names_the_model_but_holds_no_secret():
    section = select_sections(DOC)[0]
    key = LocalModelGuidanceExtractor(lambda *a, **k: None)._cache_key(
        section, "2027-02-11")
    assert config.finance_extraction_model_identity() in key
    assert "Acme" not in key, "the document text is hashed, not embedded"
