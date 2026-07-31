"""Phase G.1 Task 4/6/13 — McpCapabilityDetector.

Named to avoid colliding with the existing tests/test_mcp_capability_detector.py
(Phase F's own, unmodified, coarse detect_capability()/validate_detection()
tests) — this file exercises the NEW, separate McpCapabilityDetector class added
to the same module.
"""

import os
from unittest.mock import patch

import pytest

from mcp_management.capabilities import CapabilityEvidenceType
from mcp_management.capability_detector import DOCUMENT_TO_MARKDOWN_CAPABILITY, McpCapabilityDetector
from mcp_management.catalog import load_catalog


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.fixture(scope="module")
def detector():
    return McpCapabilityDetector()


def _ids(reqs):
    return {r.capability_id for r in reqs}


# ---- A/B/C/H: filesystem intents from an absolute local path ----

def test_read_text_file_absolute_windows_path(detector, catalog):
    reqs = detector.detect(r"read 'C:\approved\hello.txt'", catalog)
    assert _ids(reqs) == {"read_local_text_file"}


def test_list_directory(detector, catalog):
    reqs = detector.detect(r"list files in C:\approved\folder", catalog)
    assert _ids(reqs) == {"list_local_directory"}


def test_search_files(detector, catalog):
    reqs = detector.detect(r"find statement.pdf under C:\Documents", catalog)
    assert _ids(reqs) == {"search_local_files"}


def test_explicit_local_readme_is_filesystem(detector, catalog):
    reqs = detector.detect(r"read C:\Projects\repo\README.md", catalog)
    assert _ids(reqs) == {"read_local_text_file"}


# ---- D/E/F: filesystem vs document-conversion, extension alone is not enough ----

def test_pdf_summary_is_document_to_markdown(detector, catalog):
    reqs = detector.detect(r"summarize C:\Documents\report.pdf", catalog)
    assert _ids(reqs) == {DOCUMENT_TO_MARKDOWN_CAPABILITY}


def test_pdf_copy_is_filesystem_not_document(detector, catalog):
    reqs = detector.detect(r"copy C:\Documents\report.pdf to C:\Archive", catalog)
    assert _ids(reqs) == {"manage_local_files"}
    assert DOCUMENT_TO_MARKDOWN_CAPABILITY not in _ids(reqs)


@pytest.mark.parametrize("text", [
    "What is a PDF?",
    "Explain the DOCX file format.",
    "Why are XLSX files zipped?",
])
def test_conceptual_questions_require_nothing(detector, catalog, text):
    assert detector.detect(text, catalog) == ()


# ---- G/I: do not hijack repository-relative or remote paths ----

def test_bare_relative_filename_is_not_filesystem(detector, catalog):
    assert detector.detect("read README.md", catalog) == ()


def test_github_url_is_not_local_filesystem(detector, catalog):
    reqs = detector.detect("read https://github.com/example/repo/blob/main/README.md", catalog)
    assert reqs == ()


def test_generic_url_document_is_unsupported_in_this_phase(detector, catalog):
    reqs = detector.detect("open https://example.com/report.pdf", catalog)
    assert reqs == ()


# ---- J/K: explicit server evidence ----

def test_explicit_known_server_is_recorded_as_evidence(detector, catalog):
    reqs = detector.detect(r"Use the filesystem MCP server to read C:\approved\hello.txt", catalog)
    assert len(reqs) == 1
    evidence_types = {e.evidence_type for e in reqs[0].evidence}
    assert CapabilityEvidenceType.EXPLICIT_SERVER in evidence_types
    explicit = next(e for e in reqs[0].evidence if e.evidence_type == CapabilityEvidenceType.EXPLICIT_SERVER)
    assert explicit.value == "filesystem"


def test_unknown_explicit_server_is_recorded_as_unknown(detector, catalog):
    reqs = detector.detect(r"Use random-server MCP to read C:\approved\hello.txt", catalog)
    assert len(reqs) == 1
    explicit = next(e for e in reqs[0].evidence if e.evidence_type == CapabilityEvidenceType.EXPLICIT_SERVER)
    assert explicit.value.startswith("unknown:")
    assert "random-server" in explicit.value


# ---- R: Windows path forms ----

@pytest.mark.parametrize("text", [
    r"read C:\approved\hello.txt",
    r"read C:/approved/hello.txt",
    r"read 'C:\approved\hello.txt'",
    r'read "C:\approved\hello.txt"',
    r"read \\myserver\share\folder\hello.txt",
])
def test_windows_and_unc_path_forms(detector, catalog, text):
    reqs = detector.detect(text, catalog)
    assert _ids(reqs) == {"read_local_text_file"}


# ---- S: POSIX path forms ----

@pytest.mark.parametrize("text", [
    "read /home/user/file.txt",
    "list /tmp/folder/",
    "read '/home/user/file.txt'",
    'read "/home/user/file.txt"',
])
def test_posix_path_forms(detector, catalog, text):
    reqs = detector.detect(text, catalog)
    assert reqs  # at least one filesystem requirement detected
    assert all(r.capability_id in ("read_local_text_file", "list_local_directory") for r in reqs)


# ---- T: no filesystem inspection whatsoever ----

def test_detector_never_touches_the_filesystem_or_network(detector, catalog):
    with patch("os.stat", side_effect=AssertionError("os.stat must never be called")), \
         patch("os.path.isfile", side_effect=AssertionError("os.path.isfile must never be called")), \
         patch("os.path.isdir", side_effect=AssertionError("os.path.isdir must never be called")), \
         patch("builtins.open", side_effect=AssertionError("open() must never be called")):
        reqs = detector.detect(r"read C:\approved\hello.txt and list C:\approved\folder", catalog)
        assert reqs


# ---- Q: determinism ----

def test_determinism_across_repeated_calls(detector, catalog):
    text = r"Use the filesystem MCP server to read C:\approved\hello.txt"
    first = detector.detect(text, catalog)
    for _ in range(5):
        again = detector.detect(text, catalog)
        assert again == first
