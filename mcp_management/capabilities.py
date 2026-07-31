"""Phase G.1 — typed models for capability detection and server selection.

These models sit strictly BETWEEN the router (which decides only local vs. Claude)
and Phase B (which shortlists exact tools). Nothing here is a tool, a command, or
an executable — a `McpServerSelection` names, at most, an approved catalog server
as a PREFERRED PROVIDER. It never names an exact tool, never carries a raw
model-generated command, and never carries a secret.

Capability ids are plain strings, not a closed enum, so a catalog entry can
introduce a new granular capability (e.g. a future document-conversion server)
without a code change here — `mcp_management/catalog.py` is the only place that
validates their shape (`^[a-z0-9_]+$`).

Every model is a frozen dataclass with a `to_dict()` for logging/test assertions;
`to_dict()` never includes raw request text, file contents, or secrets — only the
structured, already-bounded fields these types themselves carry.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class CapabilitySelectionStatus(str, Enum):
    NONE_REQUIRED = "none_required"
    SELECTED = "selected"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    MULTI_SERVER_REQUIRED = "multi_server_required"
    INVALID_CATALOG = "invalid_catalog"


class CapabilityEvidenceType(str, Enum):
    EXPLICIT_SERVER = "explicit_server"
    EXPLICIT_CAPABILITY = "explicit_capability"
    ACTION_OBJECT = "action_object"
    FILE_EXTENSION = "file_extension"
    LOCAL_PATH = "local_path"
    URL = "url"
    NEGATIVE_RULE = "negative_rule"


@dataclass(frozen=True)
class CapabilityEvidence:
    """One structural signal that contributed to a CapabilityRequirement.

    `value` is a short, bounded, non-sensitive label (e.g. an action verb, a
    normalized extension, or a server name) — never a raw file path or full
    request text; see mcp_management.capability_detector for what is placed here.
    """

    evidence_type: CapabilityEvidenceType
    value: str
    score: int
    reason: str

    def to_dict(self) -> dict:
        return {
            "evidence_type": self.evidence_type.value,
            "value": self.value,
            "score": self.score,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CapabilityRequirement:
    """One granular capability a request appears to need, with its evidence."""

    capability_id: str
    confidence: float
    evidence: Tuple[CapabilityEvidence, ...] = ()

    def to_dict(self) -> dict:
        return {
            "capability_id": self.capability_id,
            "confidence": round(self.confidence, 4),
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass(frozen=True)
class McpServerCandidate:
    """One approved catalog server considered as a provider for the requirement(s)."""

    server_id: str
    catalog_id: str
    capabilities: Tuple[str, ...]
    matched_capabilities: Tuple[str, ...]
    score: int
    installed: bool
    active: bool
    enabled: bool
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "server_id": self.server_id,
            "catalog_id": self.catalog_id,
            "capabilities": list(self.capabilities),
            "matched_capabilities": list(self.matched_capabilities),
            "score": self.score,
            "installed": self.installed,
            "active": self.active,
            "enabled": self.enabled,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class McpServerSelection:
    """The Phase G.1 selector's final, deterministic result for one request.

    Names, at most, an approved server as a PREFERRED PROVIDER — never an exact
    tool. `candidates` is bounded to a small number (see server_selector.py) and
    ordered deterministically (score desc, then server_id asc).
    """

    status: CapabilitySelectionStatus
    required_capabilities: Tuple[CapabilityRequirement, ...]
    selected_server_id: Optional[str]
    selected_catalog_id: Optional[str]
    candidates: Tuple[McpServerCandidate, ...]
    explanation: str
    error_code: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "required_capabilities": [r.to_dict() for r in self.required_capabilities],
            "selected_server_id": self.selected_server_id,
            "selected_catalog_id": self.selected_catalog_id,
            "candidates": [c.to_dict() for c in self.candidates],
            "explanation": self.explanation,
            "error_code": self.error_code,
        }


@dataclass(frozen=True)
class TurnMcpContext:
    """A later-phase integration point (Task 10): the capability preference
    computed for one turn, attached alongside the turn but never consumed by
    Phase B or the tool loop in Phase G.1 itself. Reused as-is by G.2/G.8."""

    preferred_mcp_server_id: Optional[str] = None
    required_mcp_capabilities: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "preferred_mcp_server_id": self.preferred_mcp_server_id,
            "required_mcp_capabilities": list(self.required_mcp_capabilities),
        }


NONE_REQUIRED_SELECTION = McpServerSelection(
    status=CapabilitySelectionStatus.NONE_REQUIRED,
    required_capabilities=(),
    selected_server_id=None,
    selected_catalog_id=None,
    candidates=(),
    explanation="No MCP capability is required for this request.",
)
