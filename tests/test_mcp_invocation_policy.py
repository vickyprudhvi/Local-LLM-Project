"""Phase G.4 — invocation policy enforcement on McpTool.execute.

Tests the exact-file-uri policy without starting a real MCP server: a fake client
records the arguments it receives so we can prove the model-provided URI is
replaced by the trusted file:// URI and that authorization is consumed even on
failure.
"""

import os
from unittest.mock import MagicMock

import pytest

from mcp_layer.errors import McpError
from mcp_layer.tool import McpTool
from mcp_management.document_authorization import (
    DocumentAuthorizationStore,
    DocumentInputSnapshot,
)
from tools.base import ToolFailure
from tools.models import (
    MCP_DOCUMENT_AUTHORIZATION_REQUIRED,
    MCP_DOCUMENT_NORMALIZATION_FAILED,
    MCP_DOCUMENT_OUTPUT_TOO_LARGE,
    MCP_DOCUMENT_PATH_INVALID,
    MCP_DOCUMENT_PATH_RESTRICTED,
    ToolPermission,
)


@pytest.fixture(autouse=True)
def _reset_default_store():
    """Give every test a fresh default authorization store."""
    original = DocumentAuthorizationStore._default
    DocumentAuthorizationStore._default = DocumentAuthorizationStore()
    yield
    DocumentAuthorizationStore._default = original


def _make_auth(path: str) -> str:
    stat = os.stat(path)
    snap = DocumentInputSnapshot(
        source_uri=path,
        local_path=path,
        size_bytes=stat.st_size,
        ctime_ns=stat.st_ctime_ns,
        permission=ToolPermission.READ,
    )
    return DocumentAuthorizationStore.default().create_authorization(snap).auth_id


def test_exact_file_uri_replaces_model_path_with_trusted_uri(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("G4_VERIFY_TXT_2026", encoding="utf-8")
    auth_id = _make_auth(str(p))

    client = MagicMock()
    client.call_tool.return_value = {"text": "converted"}

    tool = McpTool(
        registry_name="mcp.markitdown.convert_to_markdown",
        remote_name="convert_to_markdown",
        description="test",
        input_schema={"type": "object", "properties": {"uri": {"type": "string"}}},
        permission=ToolPermission.READ,
        client=client,
        invocation_policy={"argument_mode": "exact_file_uri"},
    )

    result = tool.execute({"uri": str(p)})
    assert result == {"text": "converted"}
    called_uri = client.call_tool.call_args[0][1]["uri"]
    assert called_uri.startswith("file:///")
    assert called_uri.endswith("doc.txt")

    # Authorization is consumed after the call.
    auth = DocumentAuthorizationStore.default().get_authorization(auth_id)
    assert auth.consumed


def test_file_uri_argument_is_accepted_and_replaced(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("G4_VERIFY_TXT_2026", encoding="utf-8")
    auth_id = _make_auth(str(p))

    client = MagicMock()
    client.call_tool.return_value = {"text": "converted"}

    tool = McpTool(
        registry_name="mcp.markitdown.convert_to_markdown",
        remote_name="convert_to_markdown",
        description="test",
        input_schema={"type": "object", "properties": {"uri": {"type": "string"}}},
        permission=ToolPermission.READ,
        client=client,
        invocation_policy={"argument_mode": "exact_file_uri"},
    )

    tool.execute({"uri": f"file:///{str(p).replace(os.sep, '/')}"})
    called_uri = client.call_tool.call_args[0][1]["uri"]
    assert called_uri.startswith("file:///")
    assert DocumentAuthorizationStore.default().get_authorization(auth_id).consumed


def test_remote_uri_rejected_before_authorization_lookup():
    client = MagicMock()
    tool = McpTool(
        registry_name="mcp.markitdown.convert_to_markdown",
        remote_name="convert_to_markdown",
        description="test",
        input_schema={"type": "object", "properties": {"uri": {"type": "string"}}},
        permission=ToolPermission.READ,
        client=client,
        invocation_policy={"argument_mode": "exact_file_uri"},
    )
    with pytest.raises(ToolFailure) as exc:
        tool.execute({"uri": "https://example.com/file.pdf"})
    assert exc.value.code == MCP_DOCUMENT_PATH_RESTRICTED
    client.call_tool.assert_not_called()


def test_unc_path_rejected():
    client = MagicMock()
    tool = McpTool(
        registry_name="mcp.markitdown.convert_to_markdown",
        remote_name="convert_to_markdown",
        description="test",
        input_schema={"type": "object", "properties": {"uri": {"type": "string"}}},
        permission=ToolPermission.READ,
        client=client,
        invocation_policy={"argument_mode": "exact_file_uri"},
    )
    with pytest.raises(ToolFailure) as exc:
        tool.execute({"uri": "\\\\server\\share\\file.pdf"})
    assert exc.value.code == MCP_DOCUMENT_PATH_RESTRICTED


def test_unauthorized_local_path_rejected():
    client = MagicMock()
    tool = McpTool(
        registry_name="mcp.markitdown.convert_to_markdown",
        remote_name="convert_to_markdown",
        description="test",
        input_schema={"type": "object", "properties": {"uri": {"type": "string"}}},
        permission=ToolPermission.READ,
        client=client,
        invocation_policy={"argument_mode": "exact_file_uri"},
    )
    with pytest.raises(ToolFailure) as exc:
        tool.execute({"uri": "C:/Users/x/secret.pdf"})
    assert exc.value.code == MCP_DOCUMENT_AUTHORIZATION_REQUIRED


def test_non_uri_argument_rejected():
    client = MagicMock()
    tool = McpTool(
        registry_name="mcp.markitdown.convert_to_markdown",
        remote_name="convert_to_markdown",
        description="test",
        input_schema={"type": "object", "properties": {"uri": {"type": "string"}}},
        permission=ToolPermission.READ,
        client=client,
        invocation_policy={"argument_mode": "exact_file_uri"},
    )
    with pytest.raises(ToolFailure) as exc:
        tool.execute({"uri": ["multiple"]})
    assert exc.value.code == MCP_DOCUMENT_PATH_INVALID


def test_authorization_consumed_even_when_call_fails(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("G4_VERIFY_TXT_2026", encoding="utf-8")
    auth_id = _make_auth(str(p))

    client = MagicMock()
    client.call_tool.side_effect = McpError("MCP_CALL_FAILED", "server error")

    tool = McpTool(
        registry_name="mcp.markitdown.convert_to_markdown",
        remote_name="convert_to_markdown",
        description="test",
        input_schema={"type": "object", "properties": {"uri": {"type": "string"}}},
        permission=ToolPermission.READ,
        client=client,
        invocation_policy={"argument_mode": "exact_file_uri"},
    )

    with pytest.raises(ToolFailure):
        tool.execute({"uri": str(p)})
    assert DocumentAuthorizationStore.default().get_authorization(auth_id).consumed


def test_authorization_consumed_when_normalization_fails(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("G4_VERIFY_TXT_2026", encoding="utf-8")
    auth_id = _make_auth(str(p))

    client = MagicMock()
    client.call_tool.return_value = {"not_text": "bad shape"}

    tool = McpTool(
        registry_name="mcp.markitdown.convert_to_markdown",
        remote_name="convert_to_markdown",
        description="test",
        input_schema={"type": "object", "properties": {"uri": {"type": "string"}}},
        permission=ToolPermission.READ,
        client=client,
        invocation_policy={"argument_mode": "exact_file_uri"},
    )

    with pytest.raises(ToolFailure) as exc:
        tool.execute({"uri": str(p)})
    assert exc.value.code == MCP_DOCUMENT_NORMALIZATION_FAILED
    assert DocumentAuthorizationStore.default().get_authorization(auth_id).consumed


def test_oversized_result_rejected_without_truncation(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("G4_VERIFY_TXT_2026", encoding="utf-8")
    auth_id = _make_auth(str(p))

    client = MagicMock()
    client.call_tool.return_value = {"text": "x" * (2 * 1024 * 1024)}

    tool = McpTool(
        registry_name="mcp.markitdown.convert_to_markdown",
        remote_name="convert_to_markdown",
        description="test",
        input_schema={"type": "object", "properties": {"uri": {"type": "string"}}},
        permission=ToolPermission.READ,
        client=client,
        invocation_policy={"argument_mode": "exact_file_uri"},
    )

    with pytest.raises(ToolFailure) as exc:
        tool.execute({"uri": str(p)})
    assert exc.value.code == MCP_DOCUMENT_OUTPUT_TOO_LARGE
    assert DocumentAuthorizationStore.default().get_authorization(auth_id).consumed


def test_no_policy_passes_arguments_unchanged():
    client = MagicMock()
    client.call_tool.return_value = {"ok": True}

    tool = McpTool(
        registry_name="mcp.calc.add",
        remote_name="add",
        description="test",
        input_schema={"type": "object", "properties": {"a": {"type": "number"}}},
        permission=ToolPermission.READ,
        client=client,
    )

    result = tool.execute({"a": 1})
    assert result == {"ok": True}
    client.call_tool.assert_called_once_with("add", {"a": 1}, timeout=20.0)
