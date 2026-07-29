"""Phase E — process, working-directory, and environment security controls."""

import os
import sys

import pytest

import mcp_layer.client as client_module
from mcp_layer.client import McpClient
from mcp_layer.config import build_config
from mcp_layer.environment import allowlisted_secret_names, build_child_environment
from mcp_layer.errors import McpError
from mcp_layer.external import (
    resolve_workspaces_root,
    validate_executable,
    validate_working_directory,
)
from tools.models import (
    MCP_CONFIGURATION_INVALID,
    MCP_EXECUTABLE_NOT_FOUND,
    MCP_WORKING_DIRECTORY_INVALID,
)


# ---- process launch safety ----

def test_launch_uses_shell_false_and_separate_argv(monkeypatch):
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise OSError("stop before real launch")

    monkeypatch.setattr(client_module.subprocess, "Popen", fake_popen)
    with pytest.raises(McpError):
        McpClient([sys.executable, "-m", "test_mcp_server", "a && b"]).start()
    assert captured["kwargs"]["shell"] is False
    assert isinstance(captured["args"], list)
    # Shell metacharacters remain a single literal argv element — never interpreted.
    assert captured["args"] == [sys.executable, "-m", "test_mcp_server", "a && b"]


def test_missing_executable_returns_executable_not_found():
    with pytest.raises(McpError) as e:
        validate_executable("definitely_not_a_real_command_zzz")
    assert e.value.code == MCP_EXECUTABLE_NOT_FOUND


def test_executable_with_null_or_newline_rejected():
    for bad in ("py\x00thon", "python\n"):
        with pytest.raises(McpError) as e:
            validate_executable(bad)
        assert e.value.code == MCP_CONFIGURATION_INVALID


def test_real_executable_resolves():
    assert validate_executable(sys.executable)


# ---- working directory isolation ----

def test_valid_workdir_under_approved_root(tmp_path):
    approved = tmp_path / "mcp_workspaces"
    (approved / "test").mkdir(parents=True)
    approved_abs = resolve_workspaces_root(str(approved))
    resolved = validate_working_directory(str(approved / "test"), approved_abs)
    assert resolved == os.path.realpath(str(approved / "test"))


def test_workdir_traversal_rejected(tmp_path):
    approved = tmp_path / "mcp_workspaces"
    approved.mkdir()
    approved_abs = resolve_workspaces_root(str(approved))
    with pytest.raises(McpError) as e:
        validate_working_directory(str(approved / ".." / ".." / "etc"), approved_abs)
    assert e.value.code == MCP_WORKING_DIRECTORY_INVALID


def test_absolute_outside_path_rejected(tmp_path):
    approved = tmp_path / "mcp_workspaces"
    approved.mkdir()
    approved_abs = resolve_workspaces_root(str(approved))
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(McpError):
        validate_working_directory(str(outside), approved_abs)


def test_repo_root_is_not_an_allowed_workdir(tmp_path):
    approved = tmp_path / "mcp_workspaces"
    approved.mkdir()
    approved_abs = resolve_workspaces_root(str(approved))
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(client_module.__file__)))
    with pytest.raises(McpError):
        validate_working_directory(repo_root, approved_abs)


def test_home_directory_is_not_an_allowed_workdir(tmp_path):
    approved = tmp_path / "mcp_workspaces"
    approved.mkdir()
    approved_abs = resolve_workspaces_root(str(approved))
    with pytest.raises(McpError):
        validate_working_directory(os.path.expanduser("~"), approved_abs)


# ---- environment isolation ----

def test_parent_secret_not_passed_without_allowlist(monkeypatch):
    monkeypatch.setenv("PHASE_E_SECRET", "do-not-expose")
    env = build_child_environment([])
    assert "PHASE_E_SECRET" not in env


def test_parent_secret_passed_only_when_allowlisted(monkeypatch):
    monkeypatch.setenv("PHASE_E_SECRET", "do-not-expose")
    env = build_child_environment(["PHASE_E_SECRET"])
    assert env.get("PHASE_E_SECRET") == "do-not-expose"


def test_full_parent_environment_is_not_inherited(monkeypatch):
    monkeypatch.setenv("PHASE_E_RANDOM_MARKER", "12345")
    env = build_child_environment([])
    assert "PHASE_E_RANDOM_MARKER" not in env


def test_known_secret_names_are_flagged_when_allowlisted():
    flagged = allowlisted_secret_names(["MY_MCP_API_KEY", "ANTHROPIC_API_KEY"])
    assert flagged == ["ANTHROPIC_API_KEY"]


# ---- PYTHONPATH is never injected (Phase E closeout Task 1) ----

def test_repo_root_not_in_child_pythonpath_by_default(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = build_child_environment([])
    assert "PYTHONPATH" not in env


def test_parent_pythonpath_not_inherited_by_default(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/some/parent/path")
    env = build_child_environment([])
    assert "PYTHONPATH" not in env


def test_pythonpath_passed_only_when_explicitly_allowlisted(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/some/parent/path")
    env = build_child_environment(["PYTHONPATH"])
    assert env.get("PYTHONPATH") == "/some/parent/path"


def test_internal_server_launches_by_absolute_script_not_dash_m():
    from mcp_layer.external import build_launch_argv, internal_test_server_script
    cfg = build_config({
        "enabled": True, "required": False, "server_id": "test", "transport": "stdio",
        "command": "python", "args": ["-m", "test_mcp_server"], "internal_test_server": True,
        "working_directory": "./mcp_workspaces/test",
        "startup_timeout_seconds": 5, "call_timeout_seconds": 5, "shutdown_timeout_seconds": 5,
        "environment_allowlist": [], "tool_policy": {"default_permission": "denied", "tools": {}},
    })
    argv = build_launch_argv(cfg, sys.executable)
    assert argv[0] == sys.executable
    assert "-m" not in argv                       # never uses `-m` (which would need PYTHONPATH)
    assert os.path.isabs(argv[1]) and argv[1].endswith(os.path.join("test_mcp_server", "server.py"))
    assert argv[1] == internal_test_server_script()


def test_external_server_uses_configured_args_verbatim():
    from mcp_layer.external import build_launch_argv
    cfg = build_config({
        "enabled": True, "required": False, "server_id": "ext", "transport": "stdio",
        "command": "myserver", "args": ["--stdio", "--flag"], "internal_test_server": False,
        "working_directory": "./mcp_workspaces/test",
        "startup_timeout_seconds": 5, "call_timeout_seconds": 5, "shutdown_timeout_seconds": 5,
        "environment_allowlist": [], "tool_policy": {"default_permission": "denied", "tools": {}},
    })
    assert build_launch_argv(cfg, "/usr/bin/myserver") == ["/usr/bin/myserver", "--stdio", "--flag"]


def test_config_stores_names_only_never_values(monkeypatch):
    monkeypatch.setenv("MY_MCP_API_KEY", "super-secret-value")
    raw = {
        "enabled": True, "required": False, "server_id": "test", "transport": "stdio",
        "command": "python", "args": [], "working_directory": "./mcp_workspaces/test",
        "startup_timeout_seconds": 5, "call_timeout_seconds": 5, "shutdown_timeout_seconds": 5,
        "environment_allowlist": ["MY_MCP_API_KEY"],
        "tool_policy": {"default_permission": "denied", "tools": {}},
    }
    cfg = build_config(raw)
    assert cfg.environment_allowlist == ("MY_MCP_API_KEY",)
    assert "super-secret-value" not in repr(cfg)
