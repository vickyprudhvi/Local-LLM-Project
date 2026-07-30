"""Phase F — generated configuration is Phase E-valid, absolute, and secret-free."""

import json
import os

import pytest

from mcp_layer.config import build_config, load_config
from mcp_layer.errors import McpError
from mcp_management.configuration_generator import (
    generate_config_dict,
    validate_generated,
    write_config,
    write_permissions_snapshot,
)
from mcp_management.planner import build_plan
from tests.mcp_provisioning_helpers import make_catalog, workspace_with_file
from tools.models import MCP_CONFIGURATION_GENERATION_FAILED, ToolPermission


@pytest.fixture
def plan(tmp_path):
    entry = make_catalog().get("official-filesystem")
    return build_plan(entry, requested_directories=[workspace_with_file(tmp_path)],
                      base_dir=str(tmp_path / "repo"))


def _generated(plan, tmp_path):
    entrypoint = tmp_path / "install" / "dist" / "index.js"
    entrypoint.parent.mkdir(parents=True, exist_ok=True)
    entrypoint.write_text("// server", encoding="utf-8")
    node = os.path.realpath(os.sys.executable)  # any real absolute executable
    return generate_config_dict(plan, node, str(entrypoint)), str(entrypoint), node


def test_generated_config_passes_the_phase_e_loader(plan, tmp_path):
    raw, _, _ = _generated(plan, tmp_path)
    config = build_config(raw)  # the real Phase E loader
    assert config.enabled is True
    assert config.transport == "stdio"
    assert config.server_id == "filesystem"


def test_paths_are_absolute(plan, tmp_path):
    raw, entrypoint, node = _generated(plan, tmp_path)
    assert os.path.isabs(raw["command"].replace("/", os.sep))
    assert os.path.isabs(raw["args"][0].replace("/", os.sep))
    assert os.path.isabs(raw["working_directory"].replace("/", os.sep))


def test_approved_directory_is_canonical_and_passed_to_the_server(plan, tmp_path):
    raw, _, _ = _generated(plan, tmp_path)
    approved = os.path.realpath(str(plan.requested_directories[0]))
    assert raw["args"][1].replace("/", os.sep) == approved


def test_working_directory_is_the_isolated_workspace(plan, tmp_path):
    raw, _, _ = _generated(plan, tmp_path)
    assert os.path.join("mcp_workspaces", "filesystem") in raw["working_directory"].replace("/", os.sep)


def test_relative_paths_are_rejected(plan):
    with pytest.raises(McpError) as e:
        generate_config_dict(plan, "node", "dist/index.js")
    assert e.value.code == MCP_CONFIGURATION_GENERATION_FAILED


def test_policy_comes_from_the_catalog_not_the_server(plan, tmp_path):
    raw, _, _ = _generated(plan, tmp_path)
    policy = raw["tool_policy"]
    assert policy["default_permission"] == "denied"
    assert policy["tools"]["write_file"]["permission"] == "write"
    assert policy["tools"]["read_text_file"]["permission"] == "read"
    assert policy["tools"]["move_file"]["permission"] == "denied"
    assert policy["tools"]["move_file"]["enabled"] is False


def test_unknown_tools_are_denied_by_default(plan, tmp_path):
    raw, _, _ = _generated(plan, tmp_path)
    config = build_config(raw)
    # A tool the server might advertise but the catalog never listed:
    assert "undocumented_extra_tool" not in config.tool_policy.tools
    assert config.tool_policy.default_permission is ToolPermission.DENIED


def test_safe_default_timeouts(plan, tmp_path):
    raw, _, _ = _generated(plan, tmp_path)
    assert raw["startup_timeout_seconds"] > 0
    assert raw["call_timeout_seconds"] > 0
    assert raw["shutdown_timeout_seconds"] > 0


def test_no_secret_values_in_the_configuration(plan, tmp_path, monkeypatch):
    monkeypatch.setenv("PHASE_F_SECRET", "do-not-expose")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    raw, _, _ = _generated(plan, tmp_path)
    blob = json.dumps(raw)
    assert "do-not-expose" not in blob and "sk-should-not-leak" not in blob
    # Environment handling is by NAME only.
    assert isinstance(raw["environment_allowlist"], list)


def test_written_config_round_trips_through_the_loader(plan, tmp_path):
    raw, _, _ = _generated(plan, tmp_path)
    target = tmp_path / "managed" / "server.json"
    write_config(raw, str(target))
    reloaded = load_config(str(target))
    assert reloaded is not None and reloaded.server_id == "filesystem"


def test_invalid_document_is_rejected_before_writing(tmp_path):
    with pytest.raises(McpError) as e:
        validate_generated({"enabled": True, "server_id": "bad id!", "transport": "stdio"})
    assert e.value.code == MCP_CONFIGURATION_GENERATION_FAILED


def test_permissions_snapshot_records_local_authority(plan, tmp_path):
    target = tmp_path / "permissions.json"
    write_permissions_snapshot(plan, str(target))
    data = json.load(open(target, encoding="utf-8"))
    assert data["source"] == "trusted_catalog"
    assert "write_file" in data["write_tools"]
    assert "move_file" in data["denied_tools"]
    assert data["default_permission"] == "denied"
    assert "ignored" in data["note"]
