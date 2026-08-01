"""Phase G.4 — versioned candidate validators for trusted catalog entries.

A candidate validator runs against the LIVE installed MCP process during
provisioning (before the candidate is promoted to the final install directory).
It validates exact tool schemas and performs real functional tests with
committed fixtures.  It must NEVER register tools into the production
ToolRegistry and must always shut down cleanly via the caller.
"""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Optional

from mcp_layer.client import McpClient
from mcp_layer.config import McpServerConfig
from mcp_layer.errors import McpError
from mcp_management.catalog import McpCatalogEntry
from tools.models import (
    MCP_CANDIDATE_VALIDATOR_NOT_FOUND,
    MCP_DOCUMENT_CONVERSION_FAILED,
    MCP_EXPECTED_TOOL_MISSING,
    MCP_INVALID_RESPONSE,
)

_CandidateValidator = Callable[[McpClient, McpServerConfig, McpCatalogEntry, str], None]

_EXPECTED_MARKITDOWN_SCHEMA = {
    "type": "object",
    "properties": {
        "uri": {
            "type": "string",
        },
    },
    "required": ["uri"],
}


class CandidateValidatorRegistry:
    """In-memory registry mapping validator names to their implementations."""

    def __init__(self):
        self._validators: Dict[str, _CandidateValidator] = {}

    def register(self, name: str, validator: _CandidateValidator) -> None:
        self._validators[name] = validator

    def get(self, name: str) -> Optional[_CandidateValidator]:
        return self._validators.get(name)

    def run(self, name: str, client: McpClient, config: McpServerConfig,
            catalog_entry: McpCatalogEntry, base_dir: str) -> None:
        validator = self.get(name)
        if validator is None:
            raise McpError(
                MCP_CANDIDATE_VALIDATOR_NOT_FOUND,
                f"Candidate validator {name!r} is not registered.",
            )
        validator(client, config, catalog_entry, base_dir)


# Global registry — populated at import time so the provisioning orchestrator
# can resolve validator names from the trusted catalog without circular imports.
_REGISTRY = CandidateValidatorRegistry()


def register_candidate_validator(name: str, validator: _CandidateValidator) -> None:
    _REGISTRY.register(name, validator)


def get_candidate_validator(name: str) -> _CandidateValidator:
    validator = _REGISTRY.get(name)
    if validator is None:
        raise McpError(
            MCP_CANDIDATE_VALIDATOR_NOT_FOUND,
            f"Candidate validator {name!r} is not registered.",
        )
    return validator


def _fixtures_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "tests", "fixtures")


def _file_uri(local_path: str) -> str:
    """Convert an absolute local path to a file:// URI accepted by MarkItDown."""
    return urllib.parse.urljoin("file:", urllib.request.pathname2url(os.path.abspath(local_path)))


def _require_exact_schema(tool) -> None:
    """Validate that `convert_to_markdown` exposes ONLY a single `uri` string argument."""
    name = tool.get("name")
    if name != "convert_to_markdown":
        raise McpError(MCP_EXPECTED_TOOL_MISSING,
                       f"Unexpected tool name in schema validation: {name!r}.")
    schema = tool.get("inputSchema") or tool.get("input_schema") or {}
    if schema.get("type") != "object":
        raise McpError(MCP_EXPECTED_TOOL_MISSING,
                       "convert_to_markdown input schema must be an object.")
    props = schema.get("properties", {})
    if set(props.keys()) != {"uri"}:
        raise McpError(MCP_EXPECTED_TOOL_MISSING,
                       "convert_to_markdown schema must contain exactly one property: 'uri'.")
    uri_prop = props.get("uri", {})
    if uri_prop.get("type") != "string":
        raise McpError(MCP_EXPECTED_TOOL_MISSING,
                       "convert_to_markdown 'uri' property must be a string.")
    required = schema.get("required", [])
    if "uri" not in required:
        raise McpError(MCP_EXPECTED_TOOL_MISSING,
                       "convert_to_markdown 'uri' property must be required.")


def _call_convert_to_markdown(client: McpClient, file_uri: str, timeout: float) -> str:
    """Call the live tool with a file:// URI and return the markdown text."""
    try:
        result = client.call_tool("convert_to_markdown", {"uri": file_uri}, timeout=timeout)
    except McpError as e:
        raise McpError(MCP_DOCUMENT_CONVERSION_FAILED,
                       f"convert_to_markdown failed for {file_uri} ({e.code}).") from e
    text = _extract_text(result)
    if text is None:
        raise McpError(MCP_INVALID_RESPONSE,
                       "convert_to_markdown returned a response without a text content item.")
    return text


def _extract_text(result) -> Optional[str]:
    """Extract text from the verified {'text': '<markdown>'} result shape."""
    if isinstance(result, dict):
        if "text" in result and isinstance(result["text"], str):
            return result["text"]
        # Some MCP clients wrap the tool result in a 'content' list.
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    return item["text"]
    return None


def _run_fixture_test(client: McpClient, fixtures_dir: str, filename: str, marker: str,
                      timeout: float) -> None:
    path = os.path.join(fixtures_dir, filename)
    if not os.path.isfile(path):
        raise McpError(MCP_DOCUMENT_CONVERSION_FAILED,
                       f"Fixture {filename!r} not found at {path}.")
    text = _call_convert_to_markdown(client, _file_uri(path), timeout)
    if marker not in text:
        raise McpError(MCP_DOCUMENT_CONVERSION_FAILED,
                       f"Fixture {filename!r} did not produce expected marker {marker!r}.")


def _advertised_extensions(catalog_entry: McpCatalogEntry) -> list[str]:
    """Return the document extensions the catalog entry advertises."""
    hints = getattr(catalog_entry, "selection_hints", None)
    if hints is None:
        return []
    extensions = getattr(hints, "extensions", {})
    return list(extensions.get("document_to_markdown", ()))


def markitdown_local_document_v1(client: McpClient, config: McpServerConfig,
                                 catalog_entry: McpCatalogEntry, base_dir: str) -> None:
    """Real-process validation for markitdown-mcp==0.0.1a4.

    Verifies:
      - convert_to_markdown exists with the exact upstream schema
      - every extension advertised in the catalog has a fixture that converts
        and contains its marker
    """
    tools = client.list_tools(timeout=config.startup_timeout_seconds)
    tool_by_name = {t.get("name"): t for t in tools if isinstance(t, dict)}
    tool = tool_by_name.get("convert_to_markdown")
    if tool is None:
        raise McpError(MCP_EXPECTED_TOOL_MISSING,
                       "convert_to_markdown is missing during candidate validation.")
    _require_exact_schema(tool)

    fixtures_dir = _fixtures_dir(base_dir)
    timeout = config.call_timeout_seconds

    # Phase G.4 — test every extension the catalog actually advertises.  A missing
    # fixture or a failed conversion is a validation failure, so the entry cannot
    # be enabled for that format.
    tested = 0
    for ext in _advertised_extensions(catalog_entry):
        filename = f"markitdown_sample{ext}"
        # Hyphenated markers avoid Markdown escaping of underscores in the
        # converted output (e.g. MarkItDown renders "G4_VERIFY" as "G4\_VERIFY").
        marker = f"G4-VERIFY-{ext.lstrip('.').upper()}-2026"
        _run_fixture_test(client, fixtures_dir, filename, marker, timeout)
        tested += 1

    if tested == 0:
        raise McpError(MCP_EXPECTED_TOOL_MISSING,
                       "No advertised document extensions to validate; catalog entry is misconfigured.")


register_candidate_validator("markitdown_local_document_v1", markitdown_local_document_v1)
