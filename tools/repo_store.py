"""Controlled repository storage: root resolution, path containment, symlink safety.

Every cloned repo lives under one configured REPOSITORY_ROOT as `owner/repo`. The
LLM never chooses a destination or an absolute path. All repo.* file operations
resolve requested relative paths and assert they stay inside the selected repo,
and never follow symlinks out of it.
"""

import os
import re
import stat

import tools.config as config
from tools.base import ToolFailure
from tools.github_tools import parse_repository, validate_path  # reuse Phase 2A validators
from tools.models import (
    INVALID_REPOSITORY_REF,
    REPOSITORY_NOT_CLONED,
    REPOSITORY_PATH_ESCAPE,
)

_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")


def repository_root_abs():
    """Absolute, resolved REPOSITORY_ROOT (created if missing)."""
    root = os.path.realpath(os.path.abspath(config.repository_root()))
    os.makedirs(root, exist_ok=True)
    return root


def validate_clone_ref(ref):
    """Validate a branch/tag/ref for cloning. Returns the ref or None; raises on bad input."""
    if ref is None:
        return None
    if not isinstance(ref, str):
        raise ToolFailure(INVALID_REPOSITORY_REF, "The ref must be a string.")
    ref = ref.strip()
    if not ref:
        return None
    if len(ref) > 200:
        raise ToolFailure(INVALID_REPOSITORY_REF, "The ref is too long.")
    if ref.startswith("-"):
        raise ToolFailure(INVALID_REPOSITORY_REF, "The ref must not begin with '-'.")
    if "\x00" in ref or any(ord(c) < 32 for c in ref) or ".." in ref:
        raise ToolFailure(INVALID_REPOSITORY_REF, "The ref contains invalid characters.")
    if not _REF_RE.match(ref):
        raise ToolFailure(INVALID_REPOSITORY_REF, "The ref contains unsupported characters.")
    return ref


def target_path(owner, repo):
    """Absolute controlled destination for owner/repo, asserted inside the root."""
    root = repository_root_abs()
    candidate = os.path.realpath(os.path.join(root, owner, repo))
    if candidate != root and not candidate.startswith(root + os.sep):
        raise ToolFailure(REPOSITORY_PATH_ESCAPE, "The computed repository path escapes the root.")
    return candidate


def staging_root():
    root = repository_root_abs()
    staging = os.path.join(root, ".staging")
    os.makedirs(staging, exist_ok=True)
    return staging


def is_cloned(owner, repo):
    try:
        return os.path.isdir(target_path(owner, repo))
    except ToolFailure:
        return False


def require_cloned_repo(repository):
    """Resolve 'owner/repo' to its existing controlled dir, or raise REPOSITORY_NOT_CLONED."""
    owner, repo = parse_repository({"repository": repository})
    path = target_path(owner, repo)
    if not os.path.isdir(path):
        raise ToolFailure(REPOSITORY_NOT_CLONED,
                          f"Repository '{owner}/{repo}' has not been cloned.")
    return owner, repo, path


def resolve_within(repo_dir, rel_path, *, allow_empty):
    """Validate a relative path and return (abs_candidate, validated_rel) inside repo_dir.

    Rejects traversal/absolute/backslash/null/control chars (via validate_path) and
    verifies the realpath stays inside repo_dir (symlink escapes are caught here).
    Does NOT itself reject a final symlink — callers decide (list marks, read blocks).
    """
    validated_rel = validate_path(rel_path, allow_empty=allow_empty)
    candidate = repo_dir if not validated_rel else os.path.join(repo_dir, validated_rel)
    real_repo = os.path.realpath(repo_dir)
    real_candidate = os.path.realpath(candidate)
    if real_candidate != real_repo and not real_candidate.startswith(real_repo + os.sep):
        raise ToolFailure(REPOSITORY_PATH_ESCAPE, "The requested path escapes the repository.")
    return candidate, validated_rel


def is_symlink(path):
    try:
        return stat.S_ISLNK(os.lstat(path).st_mode)
    except OSError:
        return False


def measure_repository(repo_dir):
    """Return (size_bytes, file_count) using lstat; never follows symlinks."""
    total_size = 0
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(repo_dir, followlinks=False):
        for name in filenames:
            file_count += 1
            try:
                total_size += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total_size, file_count
