"""Phase F — bounded capability detection.

Decides whether a request needs an MCP capability and, if so, which TRUSTED
catalog entry could serve it. Deterministic by design (pattern matching over the
request plus the catalog's own capability list) so it is reproducible and testable,
and so nothing model-generated can steer provisioning.

The output is a typed CapabilityDetection — structurally incapable of carrying a
shell command, a package name, an executable path, a URL, an environment value, or
a permission override. `validate_detection` re-checks ANY detector output
(including a future LLM-assisted one) against the catalog: an unknown catalog id
becomes MCP_SERVER_NOT_APPROVED and a capability with no approved server becomes
MCP_CAPABILITY_UNAVAILABLE.
"""

import os
import re

from mcp_management.models import CapabilityDetection
from tools.models import MCP_CAPABILITY_UNAVAILABLE, MCP_SERVER_NOT_APPROVED

MAX_REQUEST_CHARS = 2000

# Path shapes we can extract deterministically from a request, most specific first.
_QUOTED_PATH_RE = re.compile(r"['\"]([^'\"\n]{2,300})['\"]")
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s'\"<>|]*")
# Anchored to a token boundary so a leading '/' inside a relative path
# ("Documents/project/plan.md") is not mistaken for an absolute path.
_POSIX_PATH_RE = re.compile(r"(?:^|(?<=[\s'\"]))/(?:[^\s'\"<>|/]+/)+[^\s'\"<>|/]*")
_RELATIVE_PATH_RE = re.compile(r"[\w.\-]+(?:[\\/][\w.\-]+)+")
_FILE_SUFFIX_RE = re.compile(r"\.[A-Za-z0-9]{1,8}$")
_TRAILING_JUNK = " \t'\"“”‘’.,;:!?)(][}{"


_MAX_PATH_TOKENS = 12


def _extend_across_spaces(text, start):
    """Longest existing path formed by joining tokens from `start`, or None.

    Real paths often contain spaces and arrive unquoted. Only a form that exists on
    disk is accepted, so trailing sentence words can never be absorbed into a path.
    """
    tokens = text[start:].split()
    if len(tokens) < 2:
        return None
    best = None
    for count in range(2, min(len(tokens), _MAX_PATH_TOKENS) + 1):
        candidate = " ".join(tokens[:count]).strip(_TRAILING_JUNK)
        if os.path.isdir(candidate) or os.path.isfile(candidate):
            best = candidate
    return best


def extract_directory_candidate(user_text):
    """Deterministically pull the directory a request refers to, or None.

    Used by the automatic provisioning flow so the DIRECTORY shown in the approval
    plan comes from the user's own words rather than a guess. A path naming a file
    resolves to its parent directory. Returning None means "don't offer to
    provision" — the flow never invents a broad directory, and whatever is found is
    still screened by `validate_approved_directory` and shown for approval.
    """
    if not isinstance(user_text, str) or not user_text.strip():
        return None
    text = user_text[:MAX_REQUEST_CHARS]

    candidates = []
    for match in _QUOTED_PATH_RE.finditer(text):
        candidates.append(match.group(1))
    for pattern in (_WINDOWS_PATH_RE, _POSIX_PATH_RE, _RELATIVE_PATH_RE):
        for match in pattern.finditer(text):
            # An unquoted path may contain spaces ("C:\Local LLM Project\notes").
            # Grow the match across following tokens and keep the longest form that
            # actually exists; otherwise fall back to the plain match.
            extended = _extend_across_spaces(text, match.start())
            if extended:
                candidates.append(extended)
            candidates.append(match.group(0))

    for raw in candidates:
        candidate = raw.strip(_TRAILING_JUNK)
        if not candidate or any(c in candidate for c in ("\x00", "\n", "\r")):
            continue
        if os.path.isdir(candidate):
            return candidate
        # A file (existing, or simply named with an extension) means its directory.
        if os.path.isfile(candidate) or _FILE_SUFFIX_RE.search(os.path.basename(candidate)):
            parent = os.path.dirname(candidate)
            if parent:
                return parent
            continue
        return candidate
    return None

# Knowledge-style questions never provision anything.
_KNOWLEDGE_RE = re.compile(
    r"^\s*(explain|define|what\s+is|what's|whats|who\s+is|tell\s+me\s+about|why\s|how\s+do(?:es)?\s)",
    re.IGNORECASE,
)

# A filesystem request needs an ACTION applied to a filesystem OBJECT — so
# "Write a poem" (no object) and "What is the capital of France?" never match.
_FS_ACTION_RE = re.compile(
    r"\b(read|open|list|show|display|search|find|grep|create|write|save|append|"
    r"make|copy|view|cat|browse|inspect)\b",
    re.IGNORECASE,
)
_FS_OBJECT_RE = re.compile(
    r"(\bfiles?\b|\bfolders?\b|\bdirector(?:y|ies)\b|\bsubfolders?\b|\bnotes?\b|"
    r"\bdocuments?\b|\bworkspace\b|\bfilesystem\b|\.txt\b|\.md\b|\.json\b|\.csv\b|"
    r"\.log\b|\.ya?ml\b|\.py\b)",
    re.IGNORECASE,
)
# A bare filename/path is itself strong evidence ("notes.txt", "docs/plan.md").
_PATH_LIKE_RE = re.compile(r"[\w.-]+\.(?:txt|md|json|csv|log|ya?ml|py|ini|cfg)\b", re.IGNORECASE)

# Capabilities we can RECOGNIZE but which may have no approved catalog entry.
# Matching one of these yields MCP_CAPABILITY_UNAVAILABLE rather than silence, so
# the user learns why nothing happened. Patterns use distinctive service nouns to
# avoid firing on general discussion (e.g. "Explain SQL joins" matches nothing).
_OTHER_CAPABILITY_PATTERNS = (
    ("github", re.compile(r"\b(github|pull\s+requests?|\bPRs?\b)\b", re.IGNORECASE)),
    ("database", re.compile(r"\b(postgres(?:ql)?|mysql|sqlite|database\s+table)\b", re.IGNORECASE)),
    ("docker", re.compile(r"\b(docker|container(?:s)?\s+running|docker-compose)\b", re.IGNORECASE)),
    ("messaging", re.compile(r"\b(slack|send\s+an?\s+email|smtp)\b", re.IGNORECASE)),
    ("browser_automation", re.compile(r"\b(puppeteer|playwright|browse\s+to\s+https?://)\b", re.IGNORECASE)),
)

FILESYSTEM_CAPABILITY = "filesystem"


def _no_mcp(reason):
    return CapabilityDetection(requires_mcp=False, reason=reason)


def detect_capability(user_text, catalog, installed_server_ids=()):
    """Return a validated CapabilityDetection for `user_text`.

    `installed_server_ids` lets the caller distinguish "already available" from
    "needs provisioning"; detection itself is unaffected.
    """
    if not isinstance(user_text, str) or not user_text.strip():
        return _no_mcp("Empty request.")
    text = user_text[:MAX_REQUEST_CHARS]

    path_like = bool(_PATH_LIKE_RE.search(text))
    if _KNOWLEDGE_RE.match(text) and not path_like:
        return _no_mcp("The request asks for an explanation, not a system action.")

    has_action = bool(_FS_ACTION_RE.search(text))
    has_object = bool(_FS_OBJECT_RE.search(text))
    if (has_action and has_object) or (path_like and has_action):
        confidence = 0.95 if (has_action and has_object and path_like) else 0.85
        return validate_detection(
            {
                "requires_mcp": True,
                "capability": FILESYSTEM_CAPABILITY,
                "confidence": confidence,
                "reason": "The request requires reading or writing files in a local directory.",
            },
            catalog,
        )

    for capability, pattern in _OTHER_CAPABILITY_PATTERNS:
        if pattern.search(text):
            return validate_detection(
                {
                    "requires_mcp": True,
                    "capability": capability,
                    "confidence": 0.8,
                    "reason": f"The request appears to need the '{capability}' capability.",
                },
                catalog,
            )

    return _no_mcp("No MCP capability is required for this request.")


def validate_detection(raw, catalog) -> CapabilityDetection:
    """Validate a raw detection dict against the trusted catalog (fail closed).

    Any recommended catalog id must exist in the catalog; a capability with no
    approved entry is reported as unavailable. This is the single gate every
    detector result passes through, so a future LLM-assisted detector cannot
    introduce an unapproved server.
    """
    if not isinstance(raw, dict):
        return CapabilityDetection(False, reason="Malformed detector output.",
                                   error_code=MCP_SERVER_NOT_APPROVED)
    if not bool(raw.get("requires_mcp")):
        return _no_mcp(str(raw.get("reason", ""))[:300])

    capability = raw.get("capability")
    if not isinstance(capability, str) or not capability.strip():
        return CapabilityDetection(False, reason="Detector did not name a capability.",
                                   error_code=MCP_CAPABILITY_UNAVAILABLE)
    capability = capability.strip()[:64]
    reason = str(raw.get("reason", ""))[:300]
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)

    recommended = raw.get("recommended_catalog_id")
    if recommended is not None:
        # An explicitly named entry must exist in the trusted catalog.
        if not isinstance(recommended, str) or not catalog.has(recommended):
            return CapabilityDetection(
                requires_mcp=True, capability=capability, confidence=confidence,
                reason="The recommended MCP server is not in the trusted catalog.",
                error_code=MCP_SERVER_NOT_APPROVED,
            )
        entry = catalog.get(recommended)
    else:
        entry = catalog.find_by_capability(capability)

    if entry is None:
        return CapabilityDetection(
            requires_mcp=True, capability=capability, confidence=confidence,
            reason=f"No approved MCP server in the trusted catalog provides '{capability}'.",
            error_code=MCP_CAPABILITY_UNAVAILABLE,
        )

    return CapabilityDetection(
        requires_mcp=True,
        capability=capability,
        recommended_catalog_id=entry.catalog_id,
        confidence=confidence,
        reason=reason or f"The request requires the '{capability}' capability.",
    )
