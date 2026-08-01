"""Phase G.4 — candidate validator registry and MarkItDown schema helpers.

Tests here do not start a real MCP process; they exercise the registry, schema
validation, and fixture path helpers deterministically.
"""

import os

import pytest

from mcp_layer.errors import McpError
from mcp_management.candidate_validators import (
    _advertised_extensions,
    _extract_text,
    _file_uri,
    _require_exact_schema,
    get_candidate_validator,
    markitdown_local_document_v1,
    register_candidate_validator,
)
from tools.models import (
    MCP_CANDIDATE_VALIDATOR_NOT_FOUND,
    MCP_EXPECTED_TOOL_MISSING,
)


def test_markitdown_validator_is_registered():
    validator = get_candidate_validator("markitdown_local_document_v1")
    assert validator is markitdown_local_document_v1


def test_unregistered_validator_raises():
    with pytest.raises(McpError) as exc:
        get_candidate_validator("does_not_exist")
    assert exc.value.code == MCP_CANDIDATE_VALIDATOR_NOT_FOUND


def test_register_and_retrieve_custom_validator():
    def _dummy(client, config, entry, base_dir):
        return None

    register_candidate_validator("test_dummy_v1", _dummy)
    assert get_candidate_validator("test_dummy_v1") is _dummy


def test_exact_schema_accepts_verified_shape():
    tool = {
        "name": "convert_to_markdown",
        "inputSchema": {
            "type": "object",
            "properties": {"uri": {"type": "string"}},
            "required": ["uri"],
        },
    }
    _require_exact_schema(tool)  # does not raise


def test_schema_rejects_missing_uri_property():
    tool = {
        "name": "convert_to_markdown",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
    with pytest.raises(McpError) as exc:
        _require_exact_schema(tool)
    assert exc.value.code == MCP_EXPECTED_TOOL_MISSING


def test_schema_rejects_extra_properties():
    tool = {
        "name": "convert_to_markdown",
        "inputSchema": {
            "type": "object",
            "properties": {"uri": {"type": "string"}, "extra": {"type": "string"}},
            "required": ["uri"],
        },
    }
    with pytest.raises(McpError) as exc:
        _require_exact_schema(tool)
    assert exc.value.code == MCP_EXPECTED_TOOL_MISSING


def test_schema_rejects_non_string_uri():
    tool = {
        "name": "convert_to_markdown",
        "inputSchema": {
            "type": "object",
            "properties": {"uri": {"type": "array"}},
            "required": ["uri"],
        },
    }
    with pytest.raises(McpError) as exc:
        _require_exact_schema(tool)
    assert exc.value.code == MCP_EXPECTED_TOOL_MISSING


def test_schema_rejects_optional_uri():
    tool = {
        "name": "convert_to_markdown",
        "inputSchema": {
            "type": "object",
            "properties": {"uri": {"type": "string"}},
            "required": [],
        },
    }
    with pytest.raises(McpError) as exc:
        _require_exact_schema(tool)
    assert exc.value.code == MCP_EXPECTED_TOOL_MISSING


def test_file_uri_is_absolute_file_uri(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("x", encoding="utf-8")
    uri = _file_uri(str(p))
    assert uri.startswith("file:///")
    assert os.path.splitdrive(uri.replace("file:///", ""))[1].endswith("sample.txt")


def test_extract_text_from_verified_shape():
    assert _extract_text({"text": "hello"}) == "hello"


def test_extract_text_from_content_list():
    assert _extract_text({"content": [{"type": "text", "text": "hello"}]}) == "hello"


def test_extract_text_returns_none_for_missing_text():
    assert _extract_text({"data": "hello"}) is None


def test_advertised_extensions_reads_catalog_hints():
    class _Hints:
        extensions = {"document_to_markdown": [".txt", ".pdf"]}

    class _Entry:
        selection_hints = _Hints()

    assert _advertised_extensions(_Entry()) == [".txt", ".pdf"]


def test_advertised_extensions_empty_when_no_hints():
    class _Entry:
        pass

    assert _advertised_extensions(_Entry()) == []
