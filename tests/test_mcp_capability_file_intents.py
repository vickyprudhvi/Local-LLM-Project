"""Phase G.1 Task 5 — filesystem vs. document-conversion intent, end to end
through the selector (production catalog has no document provider yet)."""

import pytest

from mcp_management.capabilities import CapabilitySelectionStatus
from mcp_management.capability_detector import DOCUMENT_TO_MARKDOWN_CAPABILITY, McpCapabilityDetector
from mcp_management.catalog import load_catalog
from mcp_management.server_selector import (
    ActiveRuntimeStatusProvider,
    McpServerSelector,
    RegistryInstalledState,
)


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.fixture
def pipeline(catalog, tmp_path):
    detector = McpCapabilityDetector()
    selector = McpServerSelector()
    installed = RegistryInstalledState(base_dir=str(tmp_path), managed_root="app_data/mcp_servers")
    runtime_status = ActiveRuntimeStatusProvider(None)

    def _run(text):
        reqs = detector.detect(text, catalog)
        return reqs, selector.select(reqs, catalog, installed, runtime_status)

    return _run


@pytest.mark.parametrize("text", [
    "summarize C:\\Documents\\report.pdf",
    "review C:\\Documents\\presentation.pptx",
    "extract text from C:\\Documents\\contract.docx",
    "analyze C:\\Documents\\budget.xlsx",
])
def test_document_conversion_requests_are_unsupported_without_a_provider(pipeline, text):
    reqs, selection = pipeline(text)
    assert {r.capability_id for r in reqs} == {DOCUMENT_TO_MARKDOWN_CAPABILITY}
    assert selection.status == CapabilitySelectionStatus.UNSUPPORTED
    assert selection.selected_server_id is None
    assert selection.error_code == "MCP_CAPABILITY_UNAVAILABLE"
    assert "document_to_markdown" in selection.explanation


@pytest.mark.parametrize("text,expected_capability", [
    ("copy C:\\docs\\report.pdf to C:\\archive", "manage_local_files"),
    ("delete C:\\docs\\report.pdf", "manage_local_files"),
    ("write this text to C:\\notes\\todo.txt", "write_local_file"),
])
def test_filesystem_actions_on_document_extensions_stay_filesystem(pipeline, text, expected_capability):
    reqs, selection = pipeline(text)
    assert {r.capability_id for r in reqs} == {expected_capability}
    assert DOCUMENT_TO_MARKDOWN_CAPABILITY not in {r.capability_id for r in reqs}
    assert selection.status == CapabilitySelectionStatus.SELECTED
    assert selection.selected_server_id == "filesystem"


@pytest.mark.parametrize("text", [
    "What is a PDF?",
    "Explain the DOCX file format.",
    "Why are XLSX files zipped?",
])
def test_conceptual_questions_select_nothing(pipeline, text):
    reqs, selection = pipeline(text)
    assert reqs == ()
    assert selection.status == CapabilitySelectionStatus.NONE_REQUIRED
    assert selection.selected_server_id is None
