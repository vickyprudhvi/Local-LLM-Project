"""Phase F.1 — deterministically detect an MCP filesystem call blocked by its approved roots.

Classification does NOT rely on the server's error wording. The primary, structural signal
is: does the tool call's own PATH ARGUMENT resolve outside every currently approved root?
That question is answerable from the call's arguments and the locally-known approved-root
list alone, so it is robust to whatever text a given filesystem server happens to phrase its
denial with. Server error text is used only as a bounded, secondary signal (a failure with no
extractable path argument is never classified, regardless of wording) — this is also what
keeps the classifier from firing on unrelated failures (not-found inside an approved root,
timeouts, malformed arguments, permission errors, crashes): those either resolve INSIDE an
approved root (so they're something else) or carry no path at all.

Narrowest-root selection reuses `mcp_management.planner.validate_approved_directory` for all
restricted/broad-location screening, so a proposed root is never allowed to be a credential
store, a system directory, a drive root, or an unjustified broad location such as the repo
root or a user's entire home/Documents folder — exactly the same rules Phase F already
enforces when a server is first installed.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from mcp_layer.errors import McpError
from mcp_management.planner import validate_approved_directory
from tools.models import (
    INVALID_ARGUMENTS,
    MALFORMED_TOOL_CALL,
    MCP_DIRECTORY_NOT_APPROVED,
    MCP_INVALID_RESPONSE,
    MCP_SERVER_EXITED,
    MCP_TIMEOUT,
    TOOL_CONFIRMATION_DECLINED,
    TOOL_CONFIRMATION_MISMATCH,
    TOOL_CONFIRMATION_REQUIRED,
    TOOL_DISABLED,
    TOOL_EXECUTION_ERROR,
    TOOL_PERMISSION_DENIED,
    TOOL_PERMISSION_INVALID,
    TOOL_TIMEOUT,
    UNKNOWN_TOOL,
)

# Failure codes that are never an "outside approved root" problem — a generic
# execution/transport/permission/confirmation fault must not spawn an access plan.
_NON_ELIGIBLE_ERROR_CODES = frozenset({
    TOOL_TIMEOUT, MCP_TIMEOUT, MCP_SERVER_EXITED, TOOL_EXECUTION_ERROR,
    INVALID_ARGUMENTS, MALFORMED_TOOL_CALL, UNKNOWN_TOOL, TOOL_DISABLED,
    TOOL_PERMISSION_DENIED, TOOL_PERMISSION_INVALID, TOOL_CONFIRMATION_REQUIRED,
    TOOL_CONFIRMATION_DECLINED, TOOL_CONFIRMATION_MISMATCH, MCP_INVALID_RESPONSE,
})

_PATH_ARG_KEYS = ("path", "file_path", "source", "destination")
_PATH_LIST_ARG_KEYS = ("paths",)

# Remote MCP tool names whose `path` argument already names a DIRECTORY target
# (as opposed to read_text_file/get_file_info, whose `path` names a file).
_DIRECTORY_REMOTE_TOOLS = frozenset({
    "list_directory", "list_directory_with_sizes", "directory_tree",
    "search_files", "create_directory", "list_allowed_directories",
})

_FILE_SUFFIX_RE = re.compile(r"\.[A-Za-z0-9]{1,8}$")


@dataclass(frozen=True)
class RootProposal:
    ok: bool
    directory: Optional[str]
    restricted: bool
    reason: str


@dataclass(frozen=True)
class FilesystemAccessFailure:
    """A classified, eligible-for-a-plan outside-root failure."""

    requested_paths: Tuple[str, ...]
    proposed_root: Optional[str]
    restricted: bool
    eligible: bool
    reason: str


def _extract_argument_paths(arguments):
    if not isinstance(arguments, dict):
        return []
    paths = []
    for key in _PATH_ARG_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value.strip())
    for key in _PATH_LIST_ARG_KEYS:
        value = arguments.get(key)
        if isinstance(value, list):
            paths.extend(p.strip() for p in value if isinstance(p, str) and p.strip())
    return paths


def _is_within(path, allowed_roots):
    for root in allowed_roots:
        if path == root or path.startswith(root + os.sep):
            return True
    return False


def _directory_for(path, treat_as_directory):
    if treat_as_directory:
        return path
    base = os.path.basename(path)
    if os.path.isdir(path):
        return path
    if os.path.isfile(path) or _FILE_SUFFIX_RE.search(base):
        return os.path.dirname(path) or path
    return path


def _common_root(directories):
    directories = sorted(set(directories))
    if len(directories) == 1:
        return directories[0]
    try:
        common = os.path.commonpath(directories)
    except ValueError:
        return None  # different drives, or otherwise unrelated
    return common


def remote_tool_name(registered_tool_name):
    """"mcp.<server_id>.<remote_name>" -> "<remote_name>"; passthrough otherwise."""
    parts = registered_tool_name.split(".", 2)
    return parts[2] if len(parts) == 3 else registered_tool_name


def propose_root(candidate_paths, remote_name="", base_dir=None):
    """The narrowest directory that would cover every candidate path, screened
    against the same forbidden/broad-location rules Phase F install-time uses.
    Never proposes the requested file itself, the repo root, home, or a
    credential/system directory — those all fail the underlying screen."""
    if not candidate_paths:
        return RootProposal(False, None, False, "No path was found in the tool call.")
    treat_as_directory = remote_name in _DIRECTORY_REMOTE_TOOLS
    dirs = [_directory_for(p, treat_as_directory) for p in candidate_paths]
    common = _common_root(dirs)
    if common is None:
        return RootProposal(False, None, False,
                            "The requested paths have no common directory; approve each separately.")
    try:
        validated = validate_approved_directory(common, base_dir=base_dir,
                                                 allow_broad=False, allow_create=False)
    except McpError as e:
        restricted = e.code == MCP_DIRECTORY_NOT_APPROVED
        return RootProposal(False, None, restricted, e.message)
    return RootProposal(True, validated, False, "")


def classify_outside_root_failure(tool_name, arguments, result, allowed_roots, base_dir=None):
    """Return a FilesystemAccessFailure for a call structurally outside every
    approved root, or None when the failure is unrelated / ineligible.

    `tool_name` is the REGISTERED tool name (e.g. "mcp.filesystem.read_text_file").
    `result` is the ToolResult from ToolExecutor.execute(). `allowed_roots` is the
    server's currently approved directories (any iterable of path-like strings).
    """
    if result is None or result.success:
        return None
    if not isinstance(tool_name, str) or not tool_name.startswith("mcp."):
        return None  # only MCP tools can hit an "approved root" restriction
    error = result.error
    if error is None or error.code in _NON_ELIGIBLE_ERROR_CODES:
        return None

    raw_paths = _extract_argument_paths(arguments)
    if not raw_paths:
        return None  # nothing to classify — never guess from error text alone

    allowed = [os.path.realpath(str(r)) for r in (allowed_roots or ())]
    resolved = []
    for p in raw_paths:
        if any(c in p for c in ("\x00", "\n", "\r")):
            return None
        try:
            resolved.append(os.path.realpath(p))
        except (OSError, ValueError):
            return None

    outside = [p for p in resolved if not _is_within(p, allowed)]
    if not outside:
        return None  # every path IS inside an approved root: some other failure

    remote = remote_tool_name(tool_name)
    proposal = propose_root(outside, remote_name=remote, base_dir=base_dir)
    return FilesystemAccessFailure(
        requested_paths=tuple(outside),
        proposed_root=proposal.directory,
        restricted=proposal.restricted,
        eligible=proposal.ok,
        reason=proposal.reason,
    )


class FilesystemAccessFailureClassifier:
    """Thin, stateless wrapper so callers can hold one configured instance."""

    def __init__(self, base_dir=None):
        self.base_dir = base_dir

    def classify(self, tool_name, arguments, result, allowed_roots: Sequence) -> Optional[FilesystemAccessFailure]:
        return classify_outside_root_failure(tool_name, arguments, result, allowed_roots,
                                             base_dir=self.base_dir)
