"""Phase F closeout — effective-configuration resolution and template protection.

Precedence: MCP_CONFIG_PATH override -> enabled managed server -> committed template.
The committed `config/mcp_server.json` must stay portable, disabled, and never
written by provisioning.
"""

import json
import os

import pytest

from mcp_layer.config import load_config
from mcp_layer.config_resolver import (
    McpConfigSource,
    ResolvedMcpConfig,
    resolve_config,
)
from mcp_layer.errors import McpError
from mcp_management.registry import STATUS_DISABLED, STATUS_INSTALLED, InstalledServer, upsert
from tests.mcp_provisioning_helpers import manager_paths, write_template
from tools.models import MCP_CONFIGURATION_INVALID

MANAGED_ROOT = "app_data/mcp_servers"


def _managed_server(paths, enabled=True, status=STATUS_INSTALLED, server_id="filesystem"):
    """Create a managed install: registry entry + generated server.json."""
    base = paths["base_dir"]
    server_root = os.path.join(base, MANAGED_ROOT, server_id)
    os.makedirs(server_root, exist_ok=True)
    config_path = os.path.join(server_root, "server.json")
    document = {
        "enabled": enabled, "required": False, "server_id": server_id,
        "display_name": "Managed", "transport": "stdio", "command": "node",
        "args": [], "working_directory": "./mcp_workspaces/filesystem",
        "startup_timeout_seconds": 15, "call_timeout_seconds": 15,
        "shutdown_timeout_seconds": 5, "environment_allowlist": [],
        "tool_policy": {"default_permission": "denied", "tools": {}},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(document, f)
    upsert(server_id, InstalledServer(
        catalog_id="official-filesystem", installed_version="2026.7.10", status=status,
        install_directory=os.path.join(server_root, "versions", "2026.7.10"),
        configuration_path=config_path, installed_at="now",
    ), None, base, MANAGED_ROOT)
    return config_path


def _resolve(paths, override=""):
    return resolve_config(base_dir=paths["base_dir"], managed_root=MANAGED_ROOT,
                          override=override, template_path=paths["template_path"])


@pytest.fixture
def paths(tmp_path):
    return manager_paths(tmp_path)


# ---- precedence ----

def test_template_used_when_nothing_is_managed(paths):
    template = write_template(paths, enabled=False)
    resolved = _resolve(paths)
    assert resolved.source is McpConfigSource.DEFAULT_TEMPLATE
    assert str(resolved.path) == os.path.realpath(template)
    assert load_config(str(resolved.path)).enabled is False


def test_managed_active_beats_the_template(paths):
    write_template(paths, enabled=False)
    managed = _managed_server(paths, enabled=True)
    resolved = _resolve(paths)
    assert resolved.source is McpConfigSource.MANAGED_ACTIVE
    assert resolved.server_id == "filesystem"
    assert str(resolved.path) == os.path.realpath(managed)


def test_override_beats_everything(paths, tmp_path):
    write_template(paths, enabled=False)
    _managed_server(paths, enabled=True)
    override = tmp_path / "my_server.json"
    override.write_text(json.dumps({"enabled": False, "server_id": "custom",
                                    "transport": "stdio"}), encoding="utf-8")
    resolved = _resolve(paths, override=str(override))
    assert resolved.source is McpConfigSource.ENVIRONMENT_OVERRIDE
    assert str(resolved.path) == os.path.realpath(str(override))


def test_disabled_managed_config_falls_back_to_template(paths):
    template = write_template(paths, enabled=False)
    _managed_server(paths, enabled=False)  # generated config says enabled: false
    resolved = _resolve(paths)
    assert resolved.source is McpConfigSource.DEFAULT_TEMPLATE
    assert str(resolved.path) == os.path.realpath(template)


def test_disabled_registry_status_falls_back_to_template(paths):
    write_template(paths, enabled=False)
    _managed_server(paths, enabled=True, status=STATUS_DISABLED)
    assert _resolve(paths).source is McpConfigSource.DEFAULT_TEMPLATE


def test_missing_managed_file_falls_back_to_template(paths):
    write_template(paths, enabled=False)
    config_path = _managed_server(paths, enabled=True)
    os.unlink(config_path)
    assert _resolve(paths).source is McpConfigSource.DEFAULT_TEMPLATE


def test_no_template_and_no_managed_resolves_to_none(paths):
    assert _resolve(paths).source is McpConfigSource.NONE


def test_corrupt_registry_does_not_break_resolution(paths):
    write_template(paths, enabled=False)
    managed_dir = os.path.join(paths["base_dir"], MANAGED_ROOT)
    os.makedirs(managed_dir, exist_ok=True)
    with open(os.path.join(managed_dir, "installed_servers.json"), "w", encoding="utf-8") as f:
        f.write("{ corrupt")
    # Startup must still resolve something rather than crashing.
    assert _resolve(paths).source is McpConfigSource.DEFAULT_TEMPLATE


# ---- override validation ----

def test_override_must_exist(paths, tmp_path):
    with pytest.raises(McpError) as e:
        _resolve(paths, override=str(tmp_path / "nope.json"))
    assert e.value.code == MCP_CONFIGURATION_INVALID


def test_override_must_be_a_regular_file(paths, tmp_path):
    directory = tmp_path / "a_directory"
    directory.mkdir()
    with pytest.raises(McpError) as e:
        _resolve(paths, override=str(directory))
    assert e.value.code == MCP_CONFIGURATION_INVALID


@pytest.mark.parametrize("bad", ["with\x00null.json", "with\nnewline.json"])
def test_override_rejects_illegal_characters(paths, bad):
    with pytest.raises(McpError) as e:
        _resolve(paths, override=bad)
    assert e.value.code == MCP_CONFIGURATION_INVALID


def test_override_is_read_from_the_environment_not_the_llm(paths, tmp_path, monkeypatch):
    write_template(paths, enabled=False)
    override = tmp_path / "env_server.json"
    override.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MCP_CONFIG_PATH", str(override))
    resolved = resolve_config(base_dir=paths["base_dir"], managed_root=MANAGED_ROOT,
                              template_path=paths["template_path"])
    assert resolved.source is McpConfigSource.ENVIRONMENT_OVERRIDE


# ---- logging safety ----

def test_describe_reveals_only_source_and_basename(paths):
    template = write_template(paths, enabled=False)
    described = _resolve(paths).describe()
    assert described == "default_template:mcp_server.json"
    # The full path (which can contain a username) is not in the log line.
    assert os.path.dirname(template) not in described


def test_resolved_config_is_immutable():
    resolved = ResolvedMcpConfig(path=None, source=McpConfigSource.NONE)
    with pytest.raises(Exception):
        resolved.source = McpConfigSource.MANAGED_ACTIVE


# ---- the committed template itself (Task 2) ----

def test_committed_template_is_portable_and_disabled():
    """The tracked config must stay safe for source control."""
    with open("config/mcp_server.json", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["enabled"] is False, "the committed template must ship disabled"

    config = load_config("config/mcp_server.json")
    assert config is not None and config.enabled is False
    assert config.transport == "stdio"
    assert config.tool_policy.default_permission.value == "denied"

    # No machine-specific absolute paths may be committed.
    blob = json.dumps({k: v for k, v in raw.items() if k != "_comment"})
    assert "C:/" not in blob and "C:\\\\" not in blob
    assert "/Users/" not in blob and "/home/" not in blob
    for argument in raw.get("args", []):
        assert not os.path.isabs(argument)


def test_committed_template_has_no_machine_specific_state():
    """No usernames, drives, venvs, entrypoints, workspaces, or credentials."""
    with open("config/mcp_server.json", encoding="utf-8") as f:
        raw = json.load(f)
    payload = json.dumps({k: v for k, v in raw.items() if k != "_comment"}).lower()

    for forbidden in (
        "node_modules", "@modelcontextprotocol", "app_data", "venv",
        "site-packages", "program files", "users/", "appdata", "onedrive",
        "install-record", "installed_servers", ".exe",
    ):
        assert forbidden not in payload, f"{forbidden!r} must not appear in the template"

    assert os.environ.get("USERNAME", "\0nope").lower() not in payload
    assert os.environ.get("USER", "\0nope").lower() not in payload
    # Nothing that could be a credential or environment value.
    assert raw["environment_allowlist"] == []
    assert raw["tool_policy"]["tools"] == {}


def test_disabled_template_starts_nothing_and_resolves_no_executable(monkeypatch):
    """Loading the disabled template must not touch subprocess or executable lookup."""
    import shutil
    import subprocess

    import mcp_layer.client as client_module
    import mcp_layer.external as external_module
    from tools.registry import ToolRegistry

    monkeypatch.setattr(client_module.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("no subprocess may start when disabled"))
    monkeypatch.setattr(external_module.shutil, "which",
                        lambda *a, **k: pytest.fail("no executable lookup when disabled"))

    registry = ToolRegistry()
    session = external_module.bootstrap_from_config(
        registry, config_path="config/mcp_server.json")
    assert session.health.state.value == "disabled"
    assert session.client is None and session.tools == []
    # Built-in tools remain untouched and available.
    from tools.calculator import CalculatorTool

    registry.register(CalculatorTool())
    assert registry.has("math.calculate")
    session.shutdown()  # no-op, must not raise


def test_disabled_template_needs_no_working_directory(tmp_path):
    """The configured working directory need not exist while disabled."""
    from tools.registry import ToolRegistry

    import mcp_layer.external as external_module

    assert not os.path.isdir(os.path.join(os.getcwd(), "mcp_workspaces", "disabled"))
    session = external_module.bootstrap_from_config(
        ToolRegistry(), config_path="config/mcp_server.json")
    assert session.health.state.value == "disabled"
    # Still absent: nothing was created just by loading a disabled config.
    assert not os.path.isdir(os.path.join(os.getcwd(), "mcp_workspaces", "disabled"))


# ---- managed-state integrity (Task 5 F/K, Task 6) ----

def test_registry_config_path_outside_managed_root_is_ignored(paths, tmp_path):
    """A tampered registry cannot redirect startup to an arbitrary config file."""
    write_template(paths, enabled=False)
    rogue = tmp_path / "rogue.json"
    rogue.write_text(json.dumps({
        "enabled": True, "server_id": "filesystem", "transport": "stdio",
        "command": "node", "args": [], "working_directory": ".",
        "tool_policy": {"default_permission": "denied", "tools": {}},
    }), encoding="utf-8")

    _managed_server(paths, enabled=True)
    upsert("filesystem", InstalledServer(
        catalog_id="official-filesystem", installed_version="2026.7.10",
        status=STATUS_INSTALLED, install_directory="x",
        configuration_path=str(rogue),  # points OUTSIDE the managed root
        installed_at="now",
    ), None, paths["base_dir"], MANAGED_ROOT)

    resolved = _resolve(paths)
    assert resolved.path is None or str(rogue) != str(resolved.path), \
        "a configuration_path outside the managed root must never be selected"


def test_managed_config_with_mismatched_server_id_is_rejected(paths):
    """The generated document's server_id must match its registry key."""
    write_template(paths, enabled=False)
    config_path = _managed_server(paths, enabled=True)
    document = json.load(open(config_path, encoding="utf-8"))
    document["server_id"] = "someone-else"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(document, f)
    assert _resolve(paths).source is McpConfigSource.DEFAULT_TEMPLATE


def test_unknown_server_entry_does_not_crash_resolution(paths):
    write_template(paths, enabled=False)
    upsert("ghost", InstalledServer(
        catalog_id="official-filesystem", installed_version="1.0.0",
        status=STATUS_INSTALLED, install_directory="nowhere",
        configuration_path="nowhere/server.json", installed_at="now",
    ), None, paths["base_dir"], MANAGED_ROOT)
    assert _resolve(paths).source is McpConfigSource.DEFAULT_TEMPLATE


def test_malformed_managed_config_json_falls_back(paths):
    write_template(paths, enabled=False)
    config_path = _managed_server(paths, enabled=True)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("{ not json")
    assert _resolve(paths).source is McpConfigSource.DEFAULT_TEMPLATE


def test_override_with_malformed_json_fails_visibly(paths, tmp_path):
    """An explicit override must not be silently replaced by the template."""
    write_template(paths, enabled=False)
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    resolved = _resolve(paths, override=str(bad))
    # Resolution points at the operator's file; loading it raises rather than
    # silently falling back to the template.
    assert resolved.source is McpConfigSource.ENVIRONMENT_OVERRIDE
    with pytest.raises(McpError) as e:
        load_config(str(resolved.path))
    assert e.value.code == MCP_CONFIGURATION_INVALID


def test_override_with_invalid_phase_e_config_fails_visibly(paths, tmp_path):
    write_template(paths, enabled=False)
    bad = tmp_path / "invalid.json"
    bad.write_text(json.dumps({"enabled": True, "server_id": "bad id!",
                               "transport": "stdio"}), encoding="utf-8")
    resolved = _resolve(paths, override=str(bad))
    assert resolved.source is McpConfigSource.ENVIRONMENT_OVERRIDE
    with pytest.raises(McpError):
        load_config(str(resolved.path))
