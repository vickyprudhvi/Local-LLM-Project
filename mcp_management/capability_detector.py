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

from mcp_management.capabilities import CapabilityEvidence, CapabilityEvidenceType, CapabilityRequirement
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


# ============================================================================
# Phase G.1 — multi-capability request analysis for server selection.
#
# Deliberately separate from detect_capability()/validate_detection() above:
# those decide "does Phase F need to OFFER TO INSTALL something" for a single
# coarse capability id matched against fixed patterns. McpCapabilityDetector
# below extracts zero or more GRANULAR CapabilityRequirement objects (see
# mcp_management.capabilities) for mcp_management.server_selector, scored using
# the CATALOG's own selection_hints — so a new catalog entry with metadata is
# selectable with no code change here. Neither code path calls the other, and
# nothing above this point is modified.
#
# Deterministic and side-effect-free: no os.stat/Path.exists/open, no network,
# no MCP call, no LLM call. A path is recognized by SHAPE alone; it need not
# exist on disk.
# ============================================================================

_UNC_PATH_RE = re.compile(r"\\\\[^\s'\"<>|\\]+\\[^\s'\"<>|]*")
_URL_RE = re.compile(r"\bhttps?://\S+", re.IGNORECASE)

# Filesystem action verbs, most-specific first — the FIRST one that matches
# decides the granular capability (Task 5/7: action takes precedence).
_G1_MANAGE_VERBS = re.compile(r"\b(copy|move|delete|rename)\b", re.IGNORECASE)
_G1_CREATE_DIR_RE = re.compile(
    r"\b(?:create|make)\b[^.]{0,20}\b(?:directory|folder)\b", re.IGNORECASE)
_G1_CREATE_FILE_RE = re.compile(r"\b(?:create)\b[^.]{0,20}\bfile\b", re.IGNORECASE)
_G1_WRITE_VERBS = re.compile(r"\b(write|save|append)\b", re.IGNORECASE)
_G1_SEARCH_VERBS = re.compile(r"\b(find|search|locate)\b", re.IGNORECASE)
_G1_LIST_VERBS = re.compile(r"\b(list|browse)\b", re.IGNORECASE)
_G1_METADATA_RE = re.compile(r"\bfile\s+(?:info|metadata|size|details)\b", re.IGNORECASE)
_G1_READ_VERBS = re.compile(r"\b(read|open|view|cat|display|show)\b", re.IGNORECASE)

# Document-conversion verbs are a DISJOINT set from the filesystem verbs above —
# summarizing/reviewing/analyzing/extracting is never also a plain filesystem op.
# Bare "explain"/"inspect" are deliberately EXCLUDED (too generic — "Explain the
# DOCX file format" must stay NONE_REQUIRED); they only count paired with
# "this document".
_G1_DOCUMENT_VERBS = re.compile(
    r"\b(summarize|summarise|review|analyze|analyse)\b"
    r"|extract\s+(?:text|tables?|content)"
    r"|convert\s+(?:it\s+)?to\s+markdown"
    r"|read\s+and\s+summari[sz]e"
    r"|(?:explain|inspect)\s+this\s+document",
    re.IGNORECASE,
)
_G1_DOCUMENT_NOUNS = re.compile(
    r"\b(pdf|docx?|pptx?|xlsx?|epub|ipynb|word\s+document|powerpoint(?:\s+\w+)?|excel\s+\w+|"
    r"presentation|notebook|e-?book)\b",
    re.IGNORECASE)
# "inspect"/"explain" alone are too generic (they also read as plain conversation
# verbs) — they only count as document-conversion evidence together with "this
# document" or an actual document object; enforced in _document_capability below.
_G1_GENERIC_DOCUMENT_VERBS = frozenset({"inspect", "explain"})
_G1_DOCUMENT_EXTENSIONS = frozenset({
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    ".html", ".htm", ".ipynb", ".epub",
})
# Recognized text extensions, mirrored from the production catalog's own
# read_local_text_file hints — used ONLY as a lexical path-extension terminator
# (Task 3), independent of whatever a specific catalog entry declares.
_G1_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".json", ".yaml", ".yml", ".py", ".sql", ".csv", ".log", ".ini", ".toml", ".xml",
})
_G1_KNOWN_PATH_EXTENSIONS = _G1_DOCUMENT_EXTENSIONS | _G1_TEXT_EXTENSIONS

DOCUMENT_TO_MARKDOWN_CAPABILITY = "document_to_markdown"

_EXPLICIT_SERVER_RE = re.compile(
    r"\b(?:use|using)\s+(?:the\s+)?(?P<name>[A-Za-z][A-Za-z0-9 _-]{0,40}?)\s+mcp(?:\s+server)?\b",
    re.IGNORECASE)


def _extension_of(path_text):
    m = _FILE_SUFFIX_RE.search(path_text)
    return m.group(0).lower() if m else None


def _extend_unquoted_path_to_known_extension(text, start):
    """Grow an UNQUOTED path beginning at `start` across space-separated tokens
    until the accumulated string ends with a recognized extension (Task 3):
    "C:\\Users\\...\\learn stuff\\report.pdf" is not truncated at the space
    before "stuff" merely because the initial regex match stopped there.

    Lexical only — never touches the filesystem (no os.stat/Path.exists/open).
    Gives up after `_MAX_PATH_TOKENS` tokens and returns None, so a directory
    path with spaces and no recognizable extension is left to the caller's
    original (unextended) match rather than guessed at.
    """
    tokens = text[start:].split()
    if not tokens:
        return None
    accumulated = tokens[0]
    if _extension_of(accumulated.strip(_TRAILING_JUNK)) in _G1_KNOWN_PATH_EXTENSIONS:
        return accumulated.strip(_TRAILING_JUNK)
    for tok in tokens[1:_MAX_PATH_TOKENS]:
        accumulated = f"{accumulated} {tok}"
        stripped = accumulated.strip(_TRAILING_JUNK)
        if _extension_of(stripped) in _G1_KNOWN_PATH_EXTENSIONS:
            return stripped
    return None


def _find_local_paths(text):
    """Return (paths, path_types, has_url) — deterministic path-SHAPE extraction.

    A match that starts inside a URL span is never treated as a local path
    (Task 6): `open https://example.com/report.pdf` must never be read as a
    local file. `has_url` is reported separately so callers can distinguish
    "no path evidence at all" from "the only path-like text was a URL".
    `path_types` is a same-length list of coarse, privacy-safe labels
    ("windows_absolute" / "unc" / "posix_absolute" / "quoted") for debug
    logging (Task 9) — never the path text itself.
    """
    url_spans = [(m.start(), m.end()) for m in _URL_RE.finditer(text)]

    def _inside_url(pos):
        return any(start <= pos < end for start, end in url_spans)

    paths = []
    path_types = []
    for pattern, label in ((_UNC_PATH_RE, "unc"), (_WINDOWS_PATH_RE, "windows_absolute"),
                           (_POSIX_PATH_RE, "posix_absolute")):
        for m in pattern.finditer(text):
            if _inside_url(m.start()):
                continue
            candidate = m.group(0).strip(_TRAILING_JUNK)
            if not candidate:
                continue
            # An unquoted Windows/UNC path may contain spaces
            # ("C:\Users\...\learn stuff\report.pdf") and the regex above always
            # stops at the first one; extend it lexically to the nearest
            # recognized extension rather than truncating silently.
            if pattern is not _POSIX_PATH_RE:
                extended = _extend_unquoted_path_to_known_extension(text, m.start())
                if extended and len(extended) > len(candidate):
                    candidate = extended
            paths.append(candidate)
            path_types.append(label)
    for m in _QUOTED_PATH_RE.finditer(text):
        if _inside_url(m.start()):
            continue
        candidate = m.group(1).strip(_TRAILING_JUNK)
        if candidate and (_WINDOWS_PATH_RE.match(candidate) or _UNC_PATH_RE.match(candidate)
                          or candidate.startswith("/") or "/" in candidate or "\\" in candidate):
            paths.append(candidate)
            path_types.append("quoted")

    # Order-preserving de-duplication, keeping paths/path_types aligned.
    seen = set()
    dedup_paths, dedup_types = [], []
    for p, t in zip(paths, path_types):
        if p in seen:
            continue
        seen.add(p)
        dedup_paths.append(p)
        dedup_types.append(t)
    return dedup_paths, dedup_types, bool(url_spans)


def _matches_any(text, phrases):
    lowered = text.lower()
    return next((p for p in phrases if p in lowered), None)


class McpCapabilityDetector:
    """Extracts zero or more granular CapabilityRequirements from `user_text`.

    `catalog` supplies the granular-capability vocabulary via each entry's
    `selection_hints` — this class holds no fixed capability list of its own
    beyond the built-in filesystem/document classification rules (Task 5).
    """

    def detect(self, user_text, catalog):
        if not isinstance(user_text, str) or not user_text.strip():
            return ()
        text = user_text[:MAX_REQUEST_CHARS]

        local_paths, path_types, has_url = _find_local_paths(text)
        has_local_path = bool(local_paths)
        path_extension = next((e for e in (_extension_of(p) for p in local_paths) if e), None)
        path_type = path_types[0] if path_types else None
        explicit_server = self._explicit_server(text, catalog)
        is_knowledge_question = bool(_KNOWLEDGE_RE.match(text))

        requirements = []
        fs_capability = self._filesystem_capability(text, has_local_path)
        if fs_capability is not None:
            requirements.append(self._build_requirement(
                fs_capability, text, catalog, has_local_path, path_type, path_extension,
                explicit_server, action_value=None))

        doc_capability, doc_action = self._document_capability(
            text, path_extension, has_url, has_local_path)
        if doc_capability is not None and not (is_knowledge_question and not has_local_path):
            requirements.append(self._build_requirement(
                doc_capability, text, catalog, has_local_path, path_type, path_extension,
                explicit_server, action_value=doc_action))

        return tuple(requirements)

    # ---- capability classification ----

    def _filesystem_capability(self, text, has_local_path):
        if not has_local_path:
            return None  # a bare relative name/extension is never enough alone
        if _G1_MANAGE_VERBS.search(text):
            return "manage_local_files"
        if _G1_CREATE_DIR_RE.search(text):
            return "create_local_directory"
        if _G1_CREATE_FILE_RE.search(text) or _G1_WRITE_VERBS.search(text):
            return "write_local_file"
        if _G1_SEARCH_VERBS.search(text):
            return "search_local_files"
        if _G1_LIST_VERBS.search(text):
            return "list_local_directory"
        if _G1_METADATA_RE.search(text):
            return "get_local_file_metadata"
        if _G1_READ_VERBS.search(text):
            return "read_local_text_file"
        return None

    def _document_capability(self, text, path_extension, has_url, has_local_path):
        """Return (capability_id, matched_verb_text) or (None, None)."""
        if has_url and not has_local_path:
            return None, None  # remote URL document handling is unsupported in this phase
        verb_match = _G1_DOCUMENT_VERBS.search(text)
        if not verb_match:
            return None, None
        looks_like_document = (
            (path_extension in _G1_DOCUMENT_EXTENSIONS) or bool(_G1_DOCUMENT_NOUNS.search(text)))
        if not looks_like_document:
            return None, None
        return DOCUMENT_TO_MARKDOWN_CAPABILITY, verb_match.group(0).strip().lower()

    # ---- explicit server-name detection ----

    def _explicit_server(self, text, catalog):
        """Return ("known", server_id) | ("unknown", raw_name) | None.

        Only a name that matches a catalog entry's own `selection_hints.
        explicit_names` (or its server_id) counts as "known" — an unrecognized
        name is reported, never guessed at or silently substituted.
        """
        m = _EXPLICIT_SERVER_RE.search(text)
        if not m:
            return None
        raw_name = m.group("name").strip().lower()
        if catalog is None:
            return ("unknown", raw_name)
        for entry in catalog.entries.values():
            if raw_name == entry.server_id or raw_name in entry.selection_hints.explicit_names:
                return ("known", entry.server_id)
        return ("unknown", raw_name)

    # ---- evidence + requirement assembly ----

    def _build_requirement(self, capability_id, text, catalog, has_local_path, path_type,
                           path_extension, explicit_server, action_value=None):
        evidence = []

        if has_local_path:
            # `value` is a coarse SHAPE label (Task 9 debug trace), never the
            # actual path text — the path itself is never stored on evidence.
            evidence.append(CapabilityEvidence(
                CapabilityEvidenceType.LOCAL_PATH, path_type or "local_path", 40,
                "An absolute local filesystem path was found in the request."))

        matched_action = action_value or self._matched_action_phrase(catalog, capability_id, text)
        if matched_action:
            evidence.append(CapabilityEvidence(
                CapabilityEvidenceType.ACTION_OBJECT, matched_action, 35,
                f"The phrase {matched_action!r} suggests {capability_id!r}."))

        if path_extension:
            if self._extension_matches(catalog, capability_id, path_extension):
                evidence.append(CapabilityEvidence(
                    CapabilityEvidenceType.FILE_EXTENSION, path_extension, 20,
                    f"The extension {path_extension!r} suggests {capability_id!r}."))
            elif capability_id == DOCUMENT_TO_MARKDOWN_CAPABILITY and path_extension in _G1_DOCUMENT_EXTENSIONS:
                evidence.append(CapabilityEvidence(
                    CapabilityEvidenceType.FILE_EXTENSION, path_extension, 20,
                    f"The extension {path_extension!r} suggests a convertible document."))

        if explicit_server is not None:
            kind, value = explicit_server
            if kind == "known":
                evidence.append(CapabilityEvidence(
                    CapabilityEvidenceType.EXPLICIT_SERVER, value, 100,
                    "The request explicitly named an approved MCP server."))
            else:
                evidence.append(CapabilityEvidence(
                    CapabilityEvidenceType.EXPLICIT_SERVER, f"unknown:{value}", 0,
                    "The request named a server that is not in the trusted catalog."))

        confidence = min(1.0, 0.5 + 0.1 * len(evidence))
        return CapabilityRequirement(capability_id=capability_id, confidence=confidence,
                                     evidence=tuple(evidence))

    def _matched_action_phrase(self, catalog, capability_id, text):
        if catalog is None:
            return None
        for entry in catalog.entries.values():
            phrases = entry.selection_hints.actions.get(capability_id)
            if phrases:
                found = _matches_any(text, phrases)
                if found:
                    return found
        return None

    def _extension_matches(self, catalog, capability_id, extension):
        if catalog is None:
            return False
        return any(extension in entry.selection_hints.extensions.get(capability_id, ())
                  for entry in catalog.entries.values())
