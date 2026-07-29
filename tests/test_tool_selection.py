"""Phase B — bounded tool selection.

The local prompt must stay ~constant in size as the registry grows: the local LLM
receives only a shortlist, never the whole registry, and each tool description is
truncated. These are pure-Python assertions (no live Ollama, no embeddings).
"""

import json

import tools.config as config
from tools.base import BaseTool
from tools.registry import ToolRegistry, bounded_ollama_schema


class _DummyTool(BaseTool):
    """A registerable tool with a bounded, uniform description for load tests."""

    def __init__(self, i):
        self.name = f"dummy.tool_{i:02d}"
        self.description = f"Dummy tool number {i:02d} for load testing. " + "detail " * 5
        self.input_schema = {"type": "object", "properties": {}}
        self.timeout_seconds = 5.0
        self.enabled = True
        self.llm_callable = True

    def execute(self, arguments):
        return {"ok": True}


def _registry_with(n):
    reg = ToolRegistry()
    for i in range(n):
        reg.register(_DummyTool(i))
    return reg


def _selection_prompt_size(reg, message, limit):
    shortlisted = reg.shortlist_tools(message, limit)
    schemas = [bounded_ollama_schema(d) for d in shortlisted]
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": message}]
    return len(json.dumps(messages)) + len(json.dumps(schemas)), shortlisted


def test_registry_can_hold_twenty_tools():
    reg = _registry_with(20)
    assert len(reg.enabled_definitions()) == 20
    assert reg.has("dummy.tool_19")


def test_shortlist_never_exceeds_limit():
    reg = _registry_with(20)
    limit = config.max_shortlist_tools()
    for message in ["do something with dummy tool 07", "totally unrelated request", ""]:
        shortlisted = reg.shortlist_tools(message, limit)
        assert len(shortlisted) <= limit


def test_selection_prompt_stays_below_cap():
    reg = _registry_with(20)
    size, shortlisted = _selection_prompt_size(reg, "please use dummy tool 03", config.max_shortlist_tools())
    assert size < config.max_selection_prompt_chars()
    assert len(shortlisted) <= config.max_shortlist_tools()


def test_prompt_does_not_grow_with_registry_size():
    # The core Phase B guarantee: 45 extra irrelevant tools must not inflate the
    # selection prompt, because only a fixed-size shortlist ever reaches it.
    limit = config.max_shortlist_tools()
    message = "please use dummy tool 03"
    size_small, short_small = _selection_prompt_size(_registry_with(limit), message, limit)
    size_large, short_large = _selection_prompt_size(_registry_with(50), message, limit)
    assert len(short_small) <= limit and len(short_large) <= limit
    # Same shortlist size and uniform descriptions -> effectively identical size.
    assert abs(size_large - size_small) < 50


def test_shortlist_is_deterministic():
    reg = _registry_with(20)
    limit = config.max_shortlist_tools()
    first = [d.name for d in reg.shortlist_tools("use dummy tool 11", limit)]
    second = [d.name for d in reg.shortlist_tools("use dummy tool 11", limit)]
    assert first == second


def test_relevant_tool_is_shortlisted_over_irrelevant_ones():
    reg = ToolRegistry()

    class _WeatherTool(_DummyTool):
        def __init__(self):
            super().__init__(0)
            self.name = "weather.forecast"
            self.description = "Get the weather forecast for a city."

    reg.register(_WeatherTool())
    for i in range(1, 20):
        reg.register(_DummyTool(i))
    names = [d.name for d in reg.shortlist_tools("what is the weather forecast today", config.max_shortlist_tools())]
    assert "weather.forecast" in names


def test_description_truncated_to_char_budget():
    reg = ToolRegistry()
    tool = _DummyTool(0)
    tool.description = "y" * 5000
    reg.register(tool)
    schema = bounded_ollama_schema(reg.enabled_definitions()[0])
    assert len(schema["function"]["description"]) <= config.max_tool_description_chars()


def test_small_registry_returns_all_without_scoring():
    # When candidates <= limit, everything is offered (no behavior change vs Phase A).
    reg = _registry_with(3)
    shortlisted = reg.shortlist_tools("anything at all", config.max_shortlist_tools())
    assert len(shortlisted) == 3
