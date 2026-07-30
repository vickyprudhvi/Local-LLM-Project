"""Phase F — post-install validation against a REAL server process, plus failure paths."""

import os

import pytest

from mcp_layer.errors import McpError
from mcp_management.configuration_generator import generate_config_dict, validate_generated
from mcp_management.npm_installer import install_package, validate_entrypoint
from mcp_management.planner import build_plan
from mcp_management.validator import validate_installation
from tests.mcp_provisioning_helpers import (
    FakeNpm,
    make_catalog,
    node_available,
    workspace_with_file,
)
from tools.models import MCP_POST_INSTALL_VALIDATION_FAILED


@pytest.fixture
def installed(tmp_path):
    """A staged install of the Node fixture plus its generated, validated config."""
    if not node_available():
        pytest.skip("node/npm not available")
    import shutil

    entry = make_catalog().get("official-filesystem")
    base_dir = str(tmp_path / "repo")
    os.makedirs(os.path.join(base_dir, "mcp_workspaces"), exist_ok=True)
    workspace = workspace_with_file(tmp_path)
    plan = build_plan(entry, requested_directories=[workspace], base_dir=base_dir)

    staging = str(tmp_path / "staging")
    install_package(plan, staging, npm_executable="npm", runner=FakeNpm(), timeout=30)
    entrypoint = validate_entrypoint(plan, staging)
    raw = generate_config_dict(plan, shutil.which("node"), entrypoint)
    config = validate_generated(raw)
    return {"plan": plan, "entry": entry, "config": config, "entrypoint": entrypoint,
            "base_dir": base_dir, "workspace": workspace}


# ---- real process ----

def test_all_checks_pass_against_the_real_server(installed):
    report = validate_installation(
        installed["config"], installed["plan"],
        expected_tools=installed["entry"].expected_tools,
        entrypoint=installed["entrypoint"], base_dir=installed["base_dir"],
    )
    assert report.ok is True
    names = {c["check"]: c["ok"] for c in report.checks}
    for check in ("entrypoint_exists", "server_starts", "initialize", "protocol_compatible",
                  "tools_list", "expected_tools_present", "policy_applied",
                  "allowed_directories", "read_smoke_test", "write_smoke_test"):
        assert names.get(check) is True, check
    assert report.protocol_version
    assert report.server_name == "fake-filesystem-server"


def test_local_policy_is_applied_not_server_advertised(installed):
    """The fixture advertises write_file as read-only; the catalog says WRITE."""
    report = validate_installation(
        installed["config"], installed["plan"],
        expected_tools=installed["entry"].expected_tools,
        entrypoint=installed["entrypoint"], base_dir=installed["base_dir"],
    )
    policy = installed["config"].tool_policy
    assert policy.tools["write_file"].permission.value == "write"
    # The undocumented tool the fixture advertises is denied, never registered.
    assert "undocumented_extra_tool" not in policy.tools
    assert report.denied_tool_count >= 1


def test_write_smoke_test_is_off_by_default_and_leaves_no_file(installed):
    validate_installation(
        installed["config"], installed["plan"],
        expected_tools=installed["entry"].expected_tools,
        entrypoint=installed["entrypoint"], base_dir=installed["base_dir"],
        run_write_smoke_test=False,
    )
    assert os.listdir(installed["workspace"]) == ["hello.txt"]


def test_optional_write_smoke_test_cleans_up_after_itself(installed):
    report = validate_installation(
        installed["config"], installed["plan"],
        expected_tools=installed["entry"].expected_tools,
        entrypoint=installed["entrypoint"], base_dir=installed["base_dir"],
        run_write_smoke_test=True,
    )
    assert report.ok is True
    # Only the disposable installer-owned file was used, and it is gone.
    assert sorted(os.listdir(installed["workspace"])) == ["hello.txt"]


def test_shutdown_leaves_no_orphan_process(installed):
    captured = {}
    from mcp_layer.external import start_server

    def capturing_start(config, **kwargs):
        client = start_server(config, **kwargs)
        captured["client"] = client
        return client

    validate_installation(
        installed["config"], installed["plan"],
        expected_tools=installed["entry"].expected_tools,
        entrypoint=installed["entrypoint"], base_dir=installed["base_dir"],
        start_server_fn=capturing_start,
    )
    assert captured["client"]._proc is None  # shut down by the validator


# ---- failure paths (no real process needed) ----

class _FakeClient:
    def __init__(self, tools=None, protocol="2024-11-05", call_results=None, fail_call=None):
        self._tools = tools if tools is not None else []
        self.initialize_result = {"protocolVersion": protocol,
                                  "serverInfo": {"name": "fake", "version": "0"}}
        self._call_results = call_results or {}
        self._fail_call = fail_call
        self.shutdown_called = False

    @property
    def protocol_version(self):
        return self.initialize_result.get("protocolVersion")

    @property
    def server_info(self):
        return self.initialize_result.get("serverInfo") or {}

    def list_tools(self, timeout=None):
        return self._tools

    def call_tool(self, name, arguments, timeout=None):
        if self._fail_call == name:
            raise McpError("MCP_CALL_FAILED", "forced failure")
        return self._call_results.get(name, {})

    def shutdown(self):
        self.shutdown_called = True


def _schema():
    return {"type": "object", "properties": {}}


def _tool(name):
    return {"name": name, "description": name, "inputSchema": _schema()}


def test_missing_entrypoint_fails_validation(installed):
    with pytest.raises(McpError) as e:
        validate_installation(installed["config"], installed["plan"],
                              entrypoint=str(installed["entrypoint"]) + ".missing",
                              base_dir=installed["base_dir"])
    assert e.value.code == MCP_POST_INSTALL_VALIDATION_FAILED


def test_missing_expected_tools_fails_validation(installed):
    client = _FakeClient(tools=[_tool("something_else")])
    with pytest.raises(McpError) as e:
        validate_installation(installed["config"], installed["plan"],
                              expected_tools=("read_text_file",),
                              entrypoint=installed["entrypoint"],
                              base_dir=installed["base_dir"],
                              start_server_fn=lambda config, **kw: client)
    assert e.value.code == MCP_POST_INSTALL_VALIDATION_FAILED
    assert client.shutdown_called is True  # always shut down


def test_server_reporting_extra_allowed_directories_fails(installed):
    tools = [_tool(n) for n in ("list_allowed_directories", "list_directory", "read_text_file")]
    client = _FakeClient(
        tools=tools,
        call_results={"list_allowed_directories": {"directories": [os.path.abspath(os.sep)]}},
    )
    with pytest.raises(McpError) as e:
        validate_installation(installed["config"], installed["plan"],
                              expected_tools=("read_text_file",),
                              entrypoint=installed["entrypoint"],
                              base_dir=installed["base_dir"],
                              start_server_fn=lambda config, **kw: client)
    assert e.value.code == MCP_POST_INSTALL_VALIDATION_FAILED
    assert "beyond the approved" in e.value.message


def test_missing_protocol_version_fails(installed):
    client = _FakeClient(tools=[_tool("read_text_file")], protocol=None)
    with pytest.raises(McpError) as e:
        validate_installation(installed["config"], installed["plan"],
                              entrypoint=installed["entrypoint"],
                              base_dir=installed["base_dir"],
                              start_server_fn=lambda config, **kw: client)
    assert e.value.code == MCP_POST_INSTALL_VALIDATION_FAILED


def test_allowed_directory_parser_handles_the_official_server_format(tmp_path):
    """The official server returns a newline-delimited string under `content`.

    If the parser missed this shape it would find zero roots and the security check
    would pass without verifying anything.
    """
    from mcp_management.validator import _extract_paths

    approved = str(tmp_path / "approved")
    os.makedirs(approved, exist_ok=True)
    official = {"content": f"Allowed directories:\n{approved}"}
    assert _extract_paths(official) == {os.path.realpath(approved)}

    # Other shapes servers use are still parsed.
    assert _extract_paths({"directories": [approved]}) == {os.path.realpath(approved)}
    assert _extract_paths({"content": [{"type": "text", "text": approved}]}) == {
        os.path.realpath(approved)}
    assert _extract_paths({"text": f"- {approved}"}) == {os.path.realpath(approved)}
    assert _extract_paths({}) == set()


def test_no_registrable_tool_fails(installed):
    # Discovered tools exist but none are in the local policy -> nothing registers.
    client = _FakeClient(tools=[_tool("totally_unknown_tool")])
    with pytest.raises(McpError) as e:
        validate_installation(installed["config"], installed["plan"],
                              entrypoint=installed["entrypoint"],
                              base_dir=installed["base_dir"],
                              start_server_fn=lambda config, **kw: client)
    assert e.value.code == MCP_POST_INSTALL_VALIDATION_FAILED
