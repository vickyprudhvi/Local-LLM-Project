"""Phase G.1 Task 8/13 — ambiguity and multi-server requirement handling.

L: two approved servers, equal evidence, same capability -> AMBIGUOUS, no
   arbitrary winner.
M: a request needing capabilities only different servers can provide ->
   MULTI_SERVER_REQUIRED, no partial workflow.
N: a single fixture server supplying ALL required capabilities -> selected.
K: an explicitly named but untrusted/unknown server -> never selected.
"""

import pytest

from mcp_management.capabilities import (
    CapabilityEvidence,
    CapabilityEvidenceType,
    CapabilityRequirement,
    CapabilitySelectionStatus,
)
from mcp_management.capability_detector import McpCapabilityDetector
from mcp_management.catalog import build_catalog
from mcp_management.server_selector import McpServerSelector


def _catalog(entries):
    return build_catalog({"catalog_version": 1, "servers": entries})


def _entry(server_id, capabilities, selection_hints=None):
    return {
        "server_id": server_id,
        "display_name": server_id,
        "description": "test fixture",
        "capabilities": ["filesystem"],
        "risk_category": "local_filesystem",
        "transport": "stdio",
        "required_runtimes": ["node"],
        "installer": {"type": "npm", "package": f"@test/{server_id}", "version": "1.0.0",
                      "entrypoint": "dist/index.js"},
        "expected_tools": [],
        "default_tool_policy": {"default_permission": "denied", "tools": {}},
        "granular_capabilities": list(capabilities),
        "selection_hints": selection_hints or {},
    }


class _NoState:
    def is_installed(self, server_id):
        return False

    def is_disabled(self, server_id):
        return False

    def is_active(self, server_id):
        return False

    def get_health(self, server_id):
        return None


def _req(capability_id):
    ev = (CapabilityEvidence(CapabilityEvidenceType.LOCAL_PATH, "local_path", 40, "test"),)
    return CapabilityRequirement(capability_id=capability_id, confidence=0.8, evidence=ev)


# ---- L: ambiguous fixture, equal evidence ----

def test_two_equally_scored_servers_are_ambiguous_not_arbitrary():
    catalog = _catalog({
        "cat-a": _entry("filesystem-a", ["search_local_files"]),
        "cat-b": _entry("filesystem-b", ["search_local_files"]),
    })
    selection = McpServerSelector().select([_req("search_local_files")], catalog, _NoState(), _NoState())
    assert selection.status == CapabilitySelectionStatus.AMBIGUOUS
    assert selection.selected_server_id is None
    assert selection.error_code == "MCP_SERVER_SELECTION_AMBIGUOUS"


# ---- M: multi-server requirement, no partial workflow ----

def test_capabilities_split_across_servers_is_multi_server_required():
    catalog = _catalog({
        "cat-fs": _entry("filesystem", ["search_local_files"]),
        "cat-doc": _entry("document-test", ["document_to_markdown"]),
    })
    detector = McpCapabilityDetector()
    reqs = detector.detect(r"Find a PDF under C:\Documents and summarize it.", catalog)
    assert {r.capability_id for r in reqs} == {"search_local_files", "document_to_markdown"}

    selection = McpServerSelector().select(reqs, catalog, _NoState(), _NoState())
    assert selection.status == CapabilitySelectionStatus.MULTI_SERVER_REQUIRED
    assert selection.selected_server_id is None
    assert selection.error_code == "MCP_MULTI_SERVER_WORKFLOW_REQUIRED"


# ---- N: one fixture server supplies everything required ----

def test_single_server_supplying_all_required_capabilities_is_selected():
    catalog = _catalog({
        "cat-fs": _entry("filesystem", ["search_local_files"]),
        "cat-all": _entry("finance-test", ["search_local_files", "document_to_markdown"]),
    })
    selection = McpServerSelector().select(
        [_req("search_local_files"), _req("document_to_markdown")], catalog, _NoState(), _NoState())
    assert selection.status == CapabilitySelectionStatus.SELECTED
    assert selection.selected_server_id == "finance-test"


# ---- K: unknown explicit server is never selected ----

def test_unknown_explicit_server_is_never_selected_even_if_capability_is_available():
    catalog = _catalog({"cat-fs": _entry("filesystem", ["read_local_text_file"],
                                         {"actions": {"read_local_text_file": ["read"]}})})
    detector = McpCapabilityDetector()
    reqs = detector.detect(r"Use random-server MCP to read C:\approved\hello.txt", catalog)
    selection = McpServerSelector().select(reqs, catalog, _NoState(), _NoState())
    assert selection.status == CapabilitySelectionStatus.AMBIGUOUS
    assert selection.selected_server_id is None
    assert selection.error_code == "MCP_SERVER_SELECTION_AMBIGUOUS"
    assert "random-server" in selection.explanation
