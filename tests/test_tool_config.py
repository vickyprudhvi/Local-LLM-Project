"""Configuration: env parsing helpers and OLLAMA_URL default."""

import importlib
import os

import tool_loop


def test_env_bool_defaults_and_parsing(monkeypatch):
    monkeypatch.delenv("SOME_FLAG", raising=False)
    assert tool_loop._env_bool("SOME_FLAG", True) is True
    assert tool_loop._env_bool("SOME_FLAG", False) is False
    monkeypatch.setenv("SOME_FLAG", "false")
    assert tool_loop._env_bool("SOME_FLAG", True) is False
    monkeypatch.setenv("SOME_FLAG", "1")
    assert tool_loop._env_bool("SOME_FLAG", False) is True
    monkeypatch.setenv("SOME_FLAG", "on")
    assert tool_loop._env_bool("SOME_FLAG", False) is True


def test_env_int_defaults_and_bad_values(monkeypatch):
    monkeypatch.delenv("SOME_INT", raising=False)
    assert tool_loop._env_int("SOME_INT", 5) == 5
    monkeypatch.setenv("SOME_INT", "9")
    assert tool_loop._env_int("SOME_INT", 5) == 9
    monkeypatch.setenv("SOME_INT", "not-a-number")
    assert tool_loop._env_int("SOME_INT", 5) == 5


def test_ollama_url_defaults_to_localhost_when_absent(monkeypatch):
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    import brain
    importlib.reload(brain)
    assert brain.OLLAMA_URL == "http://localhost:11434"
    importlib.reload(brain)  # restore normal state


def test_ollama_url_uses_env_when_present(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://example.local:12345")
    import brain
    importlib.reload(brain)
    try:
        assert brain.OLLAMA_URL == "http://example.local:12345"
    finally:
        monkeypatch.delenv("OLLAMA_URL", raising=False)
        importlib.reload(brain)
