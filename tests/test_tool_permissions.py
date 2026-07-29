"""Phase C — tool permission model + confirmation enforcement.

Enforcement is centralized in ToolExecutor. Every test that must prove a handler
did NOT run uses a counting tool and asserts the call count stayed zero.
"""

import json
from unittest.mock import patch

import pytest

import confirmation
from tools.base import BaseTool
from tools.executor import ToolExecutor
from tools.models import (
    TOOL_CONFIRMATION_DECLINED,
    TOOL_CONFIRMATION_MISMATCH,
    TOOL_CONFIRMATION_REQUIRED,
    TOOL_PERMISSION_DENIED,
    TOOL_PERMISSION_INVALID,
    ToolCall,
    ToolConfirmation,
    ToolPermission,
    hash_arguments,
)
from tools.registry import ToolRegistry, bounded_ollama_schema, default_registry

READ, WRITE, DENIED = ToolPermission.READ, ToolPermission.WRITE, ToolPermission.DENIED


class _CountingTool(BaseTool):
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, name, permission):
        self.name = name
        self.permission = permission
        self.description = f"counting tool {name}"
        self.calls = 0

    def execute(self, arguments):
        self.calls += 1
        return {"ran": True}


def _ex(*tools):
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return ToolExecutor(reg)


def _call(name, args=None):
    return ToolCall("c1", name, args or {})


# ---- enum ----

def test_permission_enum_valid_values():
    assert ToolPermission("read") is READ
    assert ToolPermission("write") is WRITE
    assert ToolPermission("denied") is DENIED


def test_permission_coerce_fails_closed_on_invalid():
    for bad in ("bogus", "", None, 123, object()):
        assert ToolPermission.coerce(bad) is DENIED
    assert ToolPermission.coerce("read") is READ
    assert ToolPermission.coerce(WRITE) is WRITE


def test_definition_default_permission_is_denied():
    class _T(BaseTool):
        name = "x.y"
        input_schema = {"type": "object", "properties": {}}

        def execute(self, arguments):
            return {}

    assert _T().definition().permission is DENIED


# ---- read ----

def test_read_tool_runs_without_confirmation_exactly_once():
    tool = _CountingTool("t.read", READ)
    r = _ex(tool).execute(_call("t.read"))
    assert r.success is True
    assert tool.calls == 1


# ---- write ----

def test_write_tool_requires_confirmation_and_does_not_execute():
    tool = _CountingTool("t.write", WRITE)
    r = _ex(tool).execute(_call("t.write"))
    assert r.success is False
    assert r.error.code == TOOL_CONFIRMATION_REQUIRED
    assert "action_summary" in (r.error.details or {})
    assert tool.calls == 0  # the checkpoint: handler did not run


def test_write_tool_executes_after_matching_approval():
    tool = _CountingTool("t.write", WRITE)
    conf = ToolConfirmation(True, "t.write", hash_arguments({}))
    r = _ex(tool).execute(_call("t.write"), confirmation=conf)
    assert r.success is True
    assert tool.calls == 1


def test_write_tool_declined_does_not_execute():
    tool = _CountingTool("t.write", WRITE)
    conf = ToolConfirmation(False, "t.write", hash_arguments({}))
    r = _ex(tool).execute(_call("t.write"), confirmation=conf)
    assert r.error.code == TOOL_CONFIRMATION_DECLINED
    assert tool.calls == 0


def test_write_tool_mismatched_tool_name_does_not_execute():
    tool = _CountingTool("t.write", WRITE)
    conf = ToolConfirmation(True, "some.other.tool", hash_arguments({}))
    r = _ex(tool).execute(_call("t.write"), confirmation=conf)
    assert r.error.code == TOOL_CONFIRMATION_MISMATCH
    assert tool.calls == 0


def test_write_tool_changed_arguments_do_not_execute():
    tool = _CountingTool("t.write", WRITE)
    # Approval was captured for {"a": 1}; the call now carries different arguments.
    conf = ToolConfirmation(True, "t.write", hash_arguments({"a": 1}))
    r = _ex(tool).execute(_call("t.write", {"b": 2}), confirmation=conf)
    assert r.error.code == TOOL_CONFIRMATION_MISMATCH
    assert tool.calls == 0


def test_confirmation_for_one_action_cannot_approve_a_different_action():
    a = _CountingTool("t.a", WRITE)
    b = _CountingTool("t.b", WRITE)
    ex = _ex(a, b)
    conf_for_a = ToolConfirmation(True, "t.a", hash_arguments({}))
    r = ex.execute(_call("t.b"), confirmation=conf_for_a)  # approval belongs to t.a
    assert r.error.code == TOOL_CONFIRMATION_MISMATCH
    assert b.calls == 0 and a.calls == 0


# ---- denied / invalid ----

def test_denied_tool_never_executes():
    tool = _CountingTool("t.denied", DENIED)
    r = _ex(tool).execute(_call("t.denied"))
    assert r.error.code == TOOL_PERMISSION_DENIED
    assert tool.calls == 0


def test_missing_permission_defaults_to_denied():
    class _NoPerm(BaseTool):
        name = "t.noperm"
        input_schema = {"type": "object", "properties": {}}

        def __init__(self):
            self.calls = 0

        def execute(self, arguments):
            self.calls += 1
            return {}

    tool = _NoPerm()
    r = _ex(tool).execute(_call("t.noperm"))
    assert r.error.code == TOOL_PERMISSION_DENIED
    assert tool.calls == 0


def test_invalid_permission_value_fails_closed():
    class _Bad(BaseTool):
        name = "t.bad"
        permission = "whatever"  # not a valid ToolPermission
        input_schema = {"type": "object", "properties": {}}

        def __init__(self):
            self.calls = 0

        def execute(self, arguments):
            self.calls += 1
            return {}

    tool = _Bad()
    r = _ex(tool).execute(_call("t.bad"))
    assert r.error.code == TOOL_PERMISSION_INVALID
    assert tool.calls == 0


# ---- confirmation collection layer (confirmation.py) ----

def test_resolve_with_confirmation_approves_and_runs_once():
    tool = _CountingTool("t.write", WRITE)
    r = confirmation.resolve_with_confirmation(_ex(tool), _call("t.write"), confirmer=lambda s: True)
    assert r.success is True and tool.calls == 1


def test_resolve_with_confirmation_declines_without_running():
    tool = _CountingTool("t.write", WRITE)
    r = confirmation.resolve_with_confirmation(_ex(tool), _call("t.write"), confirmer=lambda s: False)
    assert r.error.code == TOOL_CONFIRMATION_DECLINED and tool.calls == 0


def test_resolve_with_confirmation_passes_deterministic_summary():
    from tools.memory_tools import RememberTool
    reg = ToolRegistry()
    reg.register(RememberTool())
    seen = []
    with patch("memory_store.remember", return_value="id"):
        confirmation.resolve_with_confirmation(
            _ex_registry(reg), _call("memory.remember", {"fact": "PostgreSQL is my db"}),
            confirmer=lambda s: (seen.append(s), True)[1],
        )
    assert seen and "PostgreSQL is my db" in seen[0]


def _ex_registry(reg):
    return ToolExecutor(reg)


# ---- classification: clone is write ----

def test_clone_repository_is_write():
    from tools.repo_clone import CloneRepositoryTool
    assert CloneRepositoryTool().permission is WRITE


# ---- full permission table (every registered tool) ----

EXPECTED_PERMISSIONS = {
    "system.echo": READ,
    "math.calculate": READ,
    "system.time": READ,
    "memory.recall": READ,
    "memory.remember": WRITE,
    "calendar.read": READ,
    "camera.look": READ,
    "camera.look_carefully": READ,
    "camera.capture": WRITE,
    "camera.scan": WRITE,
    "browser.search": READ,
    "browser.fetch_page": READ,
    "github.search_repositories": READ,
    "github.get_repository": READ,
    "github.read_file": READ,
    "github.list_directory": READ,
    "github.list_releases": READ,
    "github.clone_repository": WRITE,
    "repo.list_files": READ,
    "repo.read_file": READ,
    "repo.inspect": READ,
    "repo.security_scan": READ,
    "repo.capability_report": READ,
}


def test_every_registered_tool_has_expected_explicit_permission(monkeypatch):
    monkeypatch.setenv("REPOSITORY_CLONE_ENABLED", "true")  # register clone + repo tools
    reg = default_registry(include_internet=True, include_clone=True, include_repo=True)
    for name, expected in EXPECTED_PERMISSIONS.items():
        assert reg.has(name), f"{name} should be registered"
        perm = ToolPermission.coerce(reg.get(name).permission)
        assert perm is expected, f"{name}: expected {expected}, got {perm}"
        assert perm in (READ, WRITE), f"{name} must be read or write, never denied"


# ---- Phase B integration: permission metadata does not grow the selection prompt ----

def test_permission_does_not_appear_in_or_grow_offered_schema():
    reg = ToolRegistry()
    for i in range(50):
        t = _CountingTool(f"w.tool_{i:02d}", WRITE)
        t.description = "A write-class tool for testing. " + "detail " * 5
        reg.register(t)
    shortlisted = reg.shortlist_tools("do a write thing with tool 07", 5)
    schemas = [bounded_ollama_schema(d) for d in shortlisted]
    assert len(shortlisted) <= 5
    blob = json.dumps(schemas)
    assert "permission" not in blob  # permission never leaks into the offered schema
    assert len(blob) < 4000  # bounded regardless of 50 registered write tools
