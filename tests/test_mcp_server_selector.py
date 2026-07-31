"""Phase G.1 Task 7/9/13 — McpServerSelector candidate ranking.

Uses small, self-contained fixture catalogs (never the production catalog) so
ranking behavior is tested independent of what config/mcp_catalog.json happens
to contain. Installed/runtime status come from simple fakes — the selector must
never call anything beyond the read-only Protocol methods.
"""

import pytest

from mcp_management.capabilities import CapabilityEvidence, CapabilityEvidenceType, CapabilityRequirement
from mcp_management.capabilities import CapabilitySelectionStatus
from mcp_management.catalog import build_catalog
from mcp_management.server_selector import MAX_CANDIDATES, McpServerSelector


def _catalog(entries):
    return build_catalog({"catalog_version": 1, "servers": entries})


def _fs_entry(server_id, catalog_id=None, granular_capabilities=("read_local_text_file",)):
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
        "granular_capabilities": list(granular_capabilities),
    }


def _req(capability_id, evidence_score=40):
    ev = (CapabilityEvidence(CapabilityEvidenceType.LOCAL_PATH, "local_path", evidence_score, "test"),)
    return CapabilityRequirement(capability_id=capability_id, confidence=0.8, evidence=ev)


class _FakeInstalled:
    def __init__(self, installed=(), disabled=()):
        self._installed = set(installed)
        self._disabled = set(disabled)

    def is_installed(self, server_id):
        return server_id in self._installed

    def is_disabled(self, server_id):
        return server_id in self._disabled


class _FakeRuntime:
    def __init__(self, active=()):
        self._active = set(active)

    def is_active(self, server_id):
        return server_id in self._active

    def get_health(self, server_id):
        return "healthy" if server_id in self._active else None


def test_single_approved_server_is_selected():
    catalog = _catalog({"cat-a": _fs_entry("server-a")})
    selection = McpServerSelector().select(
        [_req("read_local_text_file")], catalog, _FakeInstalled(), _FakeRuntime())
    assert selection.status == CapabilitySelectionStatus.SELECTED
    assert selection.selected_server_id == "server-a"
    assert selection.selected_catalog_id == "cat-a"


def test_no_requirements_is_none_required():
    catalog = _catalog({"cat-a": _fs_entry("server-a")})
    selection = McpServerSelector().select([], catalog, _FakeInstalled(), _FakeRuntime())
    assert selection.status == CapabilitySelectionStatus.NONE_REQUIRED
    assert selection.candidates == ()


def test_no_provider_is_unsupported():
    catalog = _catalog({"cat-a": _fs_entry("server-a", granular_capabilities=("list_local_directory",))})
    selection = McpServerSelector().select(
        [_req("read_local_text_file")], catalog, _FakeInstalled(), _FakeRuntime())
    assert selection.status == CapabilitySelectionStatus.UNSUPPORTED
    assert selection.error_code == "MCP_CAPABILITY_UNAVAILABLE"


def test_entry_with_no_g1_metadata_is_never_a_candidate():
    """A catalog entry that has never declared granular_capabilities is simply
    invisible to the selector (backward compatibility), not an error."""
    entries = {"cat-a": _fs_entry("server-a", granular_capabilities=())}
    catalog = _catalog(entries)
    selection = McpServerSelector().select(
        [_req("read_local_text_file")], catalog, _FakeInstalled(), _FakeRuntime())
    assert selection.status == CapabilitySelectionStatus.UNSUPPORTED
    assert selection.candidates == ()


# ---- O: disabled entries are never selected ----

def test_disabled_entry_is_never_selected():
    catalog = _catalog({"cat-a": _fs_entry("server-a")})
    installed = _FakeInstalled(installed=["server-a"], disabled=["server-a"])
    selection = McpServerSelector().select(
        [_req("read_local_text_file")], catalog, installed, _FakeRuntime())
    assert selection.status == CapabilitySelectionStatus.UNSUPPORTED
    assert all(c.server_id != "server-a" for c in selection.candidates)


# ---- P: uninstalled approved entries may still be selected ----

def test_uninstalled_entry_may_be_selected_with_no_install_attempted():
    catalog = _catalog({"cat-a": _fs_entry("server-a")})
    selection = McpServerSelector().select(
        [_req("read_local_text_file")], catalog, _FakeInstalled(), _FakeRuntime())
    assert selection.status == CapabilitySelectionStatus.SELECTED
    assert selection.selected_server_id == "server-a"
    winner = next(c for c in selection.candidates if c.server_id == "server-a")
    assert winner.installed is False


# ---- installed/active are tie-breakers, never overrides ----

def test_installed_and_active_break_a_tie_between_equal_matches():
    catalog = _catalog({
        "cat-a": _fs_entry("server-a"),
        "cat-b": _fs_entry("server-b"),
    })
    installed = _FakeInstalled(installed=["server-b"])
    runtime = _FakeRuntime(active=["server-b"])
    selection = McpServerSelector().select(
        [_req("read_local_text_file")], catalog, installed, runtime)
    assert selection.status == CapabilitySelectionStatus.SELECTED
    assert selection.selected_server_id == "server-b"  # installed+active tie-breaker


def test_full_capability_coverage_beats_partial_regardless_of_install_status():
    catalog = _catalog({
        "cat-partial": _fs_entry("server-partial", granular_capabilities=("read_local_text_file",)),
        "cat-full": _fs_entry("server-full",
                              granular_capabilities=("read_local_text_file", "list_local_directory")),
    })
    # The partial server is installed+active; the full server is not. Full
    # coverage must still win — install/active status never overrides coverage.
    installed = _FakeInstalled(installed=["server-partial"])
    runtime = _FakeRuntime(active=["server-partial"])
    selection = McpServerSelector().select(
        [_req("read_local_text_file"), _req("list_local_directory")], catalog, installed, runtime)
    assert selection.status == CapabilitySelectionStatus.SELECTED
    assert selection.selected_server_id == "server-full"


# ---- bounded candidates + deterministic ordering ----

def test_candidates_bounded_to_maximum():
    entries = {f"cat-{i}": _fs_entry(f"server-{i}") for i in range(6)}
    catalog = _catalog(entries)
    selection = McpServerSelector().select(
        [_req("read_local_text_file")], catalog, _FakeInstalled(), _FakeRuntime())
    assert len(selection.candidates) <= MAX_CANDIDATES


def test_deterministic_ordering_across_repeated_calls():
    entries = {f"cat-{i}": _fs_entry(f"server-{i}") for i in range(4)}
    catalog = _catalog(entries)
    selector = McpServerSelector()
    first = selector.select([_req("read_local_text_file")], catalog, _FakeInstalled(), _FakeRuntime())
    for _ in range(5):
        again = selector.select([_req("read_local_text_file")], catalog, _FakeInstalled(), _FakeRuntime())
        assert again.status == first.status
        assert again.selected_server_id == first.selected_server_id
        assert [c.server_id for c in again.candidates] == [c.server_id for c in first.candidates]


def test_invalid_catalog_reports_invalid_catalog_status():
    selection = McpServerSelector().select([_req("read_local_text_file")], None,
                                           _FakeInstalled(), _FakeRuntime())
    assert selection.status == CapabilitySelectionStatus.INVALID_CATALOG
    assert selection.error_code == "MCP_CAPABILITY_CATALOG_INVALID"
