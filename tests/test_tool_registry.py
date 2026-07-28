"""ToolRegistry behavior."""

import pytest

from tools.base import BaseTool
from tools.registry import ToolRegistry, default_registry


class _DummyTool(BaseTool):
    def __init__(self, name):
        self.name = name
        self.description = f"dummy {name}"
        self.input_schema = {"type": "object", "properties": {}}
        self.timeout_seconds = 5.0
        self.enabled = True

    def execute(self, arguments):
        return {"ok": True}


def test_register_and_get():
    reg = ToolRegistry()
    t = _DummyTool("a.tool")
    reg.register(t)
    assert reg.get("a.tool") is t
    assert reg.has("a.tool")


def test_duplicate_registration_rejected():
    reg = ToolRegistry()
    reg.register(_DummyTool("a.tool"))
    with pytest.raises(ValueError):
        reg.register(_DummyTool("a.tool"))


def test_unknown_get_raises_keyerror():
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


def test_disable_excludes_from_definitions_and_schemas():
    reg = default_registry()
    assert "system.echo" in [d.name for d in reg.enabled_definitions()]
    reg.disable("system.echo")
    names = [d.name for d in reg.enabled_definitions()]
    assert "system.echo" not in names
    assert "math.calculate" in names
    schema_names = [s["function"]["name"] for s in reg.enabled_ollama_schemas()]
    assert "system.echo" not in schema_names


def test_disable_then_enable():
    reg = default_registry()
    reg.disable("math.calculate")
    assert not reg.is_enabled("math.calculate")
    reg.enable("math.calculate")
    assert reg.is_enabled("math.calculate")


def test_enable_disable_unknown_raises():
    reg = default_registry()
    with pytest.raises(KeyError):
        reg.disable("nope")
    with pytest.raises(KeyError):
        reg.enable("nope")


def test_definitions_are_deterministically_ordered():
    reg = ToolRegistry()
    for n in ["z.tool", "a.tool", "m.tool"]:
        reg.register(_DummyTool(n))
    names = [d.name for d in reg.enabled_definitions()]
    assert names == ["a.tool", "m.tool", "z.tool"]


def test_default_registry_has_builtins():
    reg = default_registry()
    assert reg.has("system.echo")
    assert reg.has("math.calculate")
