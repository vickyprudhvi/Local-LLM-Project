"""Phase F — capability detection: bounded, catalog-validated, fail-closed."""

import json

import pytest

from mcp_management.capability_detector import detect_capability, validate_detection
from tests.mcp_provisioning_helpers import make_catalog
from tools.models import MCP_CAPABILITY_UNAVAILABLE, MCP_SERVER_NOT_APPROVED


@pytest.fixture
def catalog():
    return make_catalog()


# ---- requests that SHOULD select the filesystem server ----

@pytest.mark.parametrize("request_text", [
    "Read notes.txt.",
    "List files in this folder.",
    "Search my documents for the project plan.",
    "Create a text file in my approved workspace.",
    "Read hello.txt from mcp_workspaces/user_files.",
    "Show me the files in my project notes folder.",
])
def test_filesystem_requests_select_official_filesystem(catalog, request_text):
    detection = detect_capability(request_text, catalog)
    assert detection.requires_mcp is True
    assert detection.capability == "filesystem"
    assert detection.recommended_catalog_id == "official-filesystem"
    assert detection.error_code is None
    assert 0.0 < detection.confidence <= 1.0


# ---- requests that need NO MCP at all ----

@pytest.mark.parametrize("request_text", [
    "What is the capital of France?",
    "Explain SQL joins.",
    "Write a poem.",
    "What time is it?",
    "Tell me about the Roman empire.",
    "Define recursion.",
    "",
])
def test_unrelated_requests_require_no_mcp(catalog, request_text):
    detection = detect_capability(request_text, catalog)
    assert detection.requires_mcp is False
    assert detection.recommended_catalog_id is None
    assert detection.error_code is None


# ---- capability recognized but not in the catalog ----

def test_github_request_is_capability_unavailable(catalog):
    detection = detect_capability("Check GitHub pull requests.", catalog)
    assert detection.requires_mcp is True
    assert detection.capability == "github"
    assert detection.recommended_catalog_id is None
    assert detection.error_code == MCP_CAPABILITY_UNAVAILABLE
    assert "trusted catalog" in detection.reason


def test_unavailable_capability_reason_is_safe(catalog):
    detection = detect_capability("Show me the docker containers running.", catalog)
    assert detection.error_code == MCP_CAPABILITY_UNAVAILABLE
    # No package names, commands, or paths leak into the explanation.
    for token in ("npm", "install", "node_modules", "@modelcontextprotocol"):
        assert token not in detection.reason


# ---- validation of arbitrary (e.g. future LLM) detector output ----

def test_detector_cannot_select_unknown_catalog_id(catalog):
    detection = validate_detection({
        "requires_mcp": True, "capability": "filesystem",
        "recommended_catalog_id": "evil-server", "confidence": 0.99,
    }, catalog)
    assert detection.error_code == MCP_SERVER_NOT_APPROVED
    assert detection.recommended_catalog_id is None


def test_detection_output_carries_no_execution_details(catalog):
    detection = validate_detection({
        "requires_mcp": True, "capability": "filesystem", "confidence": 0.9,
        # Extra keys a compromised detector might try to smuggle through:
        "command": "rm -rf /", "package": "malicious", "args": ["--evil"],
        "permission": "write", "url": "https://attacker.example",
    }, catalog)
    blob = json.dumps(detection.to_dict())
    for token in ("rm -rf", "malicious", "--evil", "attacker.example"):
        assert token not in blob
    assert set(detection.to_dict()) <= {
        "requires_mcp", "capability", "recommended_catalog_id", "confidence",
        "reason", "error_code",
    }


def test_malformed_detector_output_fails_closed(catalog):
    assert validate_detection("not a dict", catalog).error_code == MCP_SERVER_NOT_APPROVED
    assert validate_detection({"requires_mcp": True}, catalog).error_code == MCP_CAPABILITY_UNAVAILABLE


def test_confidence_is_clamped(catalog):
    high = validate_detection({"requires_mcp": True, "capability": "filesystem",
                               "confidence": 99}, catalog)
    low = validate_detection({"requires_mcp": True, "capability": "filesystem",
                              "confidence": -5}, catalog)
    bad = validate_detection({"requires_mcp": True, "capability": "filesystem",
                              "confidence": "high"}, catalog)
    assert high.confidence == 1.0 and low.confidence == 0.0 and bad.confidence == 0.0


def test_long_request_is_bounded(catalog):
    detection = detect_capability("Read notes.txt " + ("x" * 50000), catalog)
    assert detection.requires_mcp is True
    assert len(detection.reason) <= 300
