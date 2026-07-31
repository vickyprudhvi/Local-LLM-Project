"""Phase G.1 Task 3 — trusted catalog schema extension: granular_capabilities +
selection_hints.

A SEPARATE field from the existing `capabilities` list (see
tests/test_mcp_catalog.py, unmodified) — Phase F's coarse detect_capability()/
find_by_capability() keep working on entries with no Phase G.1 metadata at all.
"""

import pytest

from mcp_layer.errors import McpError
from mcp_management.catalog import McpSelectionHints, build_catalog, build_entry
from tests.mcp_provisioning_helpers import catalog_dict


def _entry_raw(**overrides):
    raw = catalog_dict()["servers"]["official-filesystem"]
    raw = dict(raw)
    raw.update(overrides)
    return raw


def _legacy_entry_raw(**overrides):
    """A pre-Phase-G.1 entry: no granular_capabilities/selection_hints at all."""
    raw = catalog_dict(granular_capabilities=())["servers"]["official-filesystem"]
    raw = dict(raw)
    del raw["granular_capabilities"]
    del raw["selection_hints"]
    raw.update(overrides)
    return raw


def test_entry_without_g1_metadata_loads_exactly_as_before():
    """Backward compatibility: an existing Phase F entry with no
    granular_capabilities/selection_hints still loads, just unselectable by G.1."""
    entry = build_entry("official-filesystem", _legacy_entry_raw())
    assert entry.granular_capabilities == ()
    assert entry.selection_hints == McpSelectionHints()
    assert entry.capabilities == ("filesystem", "read_files", "write_files")  # untouched


def test_granular_capabilities_and_hints_parse():
    raw = _entry_raw(
        granular_capabilities=["read_local_text_file", "list_local_directory"],
        selection_hints={
            "explicit_names": ["Filesystem", "filesystem MCP"],
            "actions": {"read_local_text_file": ["Read", "Open"]},
            "extensions": {"read_local_text_file": [".TXT", ".md"]},
        },
    )
    entry = build_entry("official-filesystem", raw)
    assert entry.granular_capabilities == ("read_local_text_file", "list_local_directory")
    # Normalized to lowercase for deterministic, case-insensitive matching.
    assert entry.selection_hints.explicit_names == ("filesystem", "filesystem mcp")
    assert entry.selection_hints.actions["read_local_text_file"] == ("read", "open")
    assert entry.selection_hints.extensions["read_local_text_file"] == (".txt", ".md")


def test_duplicate_granular_capability_fails_closed():
    raw = _entry_raw(granular_capabilities=["read_local_text_file", "read_local_text_file"])
    with pytest.raises(McpError) as exc:
        build_entry("official-filesystem", raw)
    assert exc.value.code == "MCP_CATALOG_INVALID"


def test_invalid_capability_id_shape_fails_closed():
    raw = _entry_raw(granular_capabilities=["Read Local Text File"])
    with pytest.raises(McpError):
        build_entry("official-filesystem", raw)


def test_empty_action_phrase_list_fails_closed():
    raw = _entry_raw(
        granular_capabilities=["read_local_text_file"],
        selection_hints={"actions": {"read_local_text_file": []}},
    )
    with pytest.raises(McpError):
        build_entry("official-filesystem", raw)


def test_hint_referencing_undeclared_capability_fails_closed():
    raw = _entry_raw(
        granular_capabilities=["read_local_text_file"],
        selection_hints={"actions": {"list_local_directory": ["list"]}},
    )
    with pytest.raises(McpError) as exc:
        build_entry("official-filesystem", raw)
    assert exc.value.code == "MCP_CATALOG_INVALID"


def test_malformed_extension_fails_closed():
    raw = _entry_raw(
        granular_capabilities=["read_local_text_file"],
        selection_hints={"extensions": {"read_local_text_file": ["txt"]}},  # missing leading dot
    )
    with pytest.raises(McpError):
        build_entry("official-filesystem", raw)


def test_unknown_selection_hints_field_fails_closed():
    raw = _entry_raw(
        granular_capabilities=["read_local_text_file"],
        selection_hints={"unexpected_field": ["x"]},
    )
    with pytest.raises(McpError):
        build_entry("official-filesystem", raw)


def test_too_many_granular_capabilities_rejected():
    raw = _entry_raw(granular_capabilities=[f"cap_{i}" for i in range(51)])
    with pytest.raises(McpError):
        build_entry("official-filesystem", raw)


def test_deterministic_ordering_preserved_from_source_list():
    raw = _entry_raw(granular_capabilities=["write_local_file", "read_local_text_file"],
                     selection_hints={})
    entry1 = build_entry("official-filesystem", raw)
    entry2 = build_entry("official-filesystem", raw)
    assert entry1.granular_capabilities == entry2.granular_capabilities == (
        "write_local_file", "read_local_text_file")


def test_catalog_with_mixed_g1_and_legacy_entries_loads():
    doc = catalog_dict()
    doc["servers"]["official-filesystem"]["granular_capabilities"] = ["read_local_text_file"]
    doc["servers"]["official-filesystem"]["selection_hints"] = {
        "actions": {"read_local_text_file": ["read"]},
    }
    catalog = build_catalog(doc)
    entry = catalog.get("official-filesystem")
    assert entry.granular_capabilities == ("read_local_text_file",)


def test_production_catalog_file_has_filesystem_granular_metadata():
    """The real config/mcp_catalog.json this branch ships must be selectable."""
    from mcp_management.catalog import load_catalog

    catalog = load_catalog()
    entry = catalog.get("official-filesystem")
    assert "read_local_text_file" in entry.granular_capabilities
    assert "filesystem" in entry.selection_hints.explicit_names
