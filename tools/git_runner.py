"""The one dedicated Git runner for Phase 2B.

This is the ONLY subprocess in Phase 2B and it is never exposed as a tool. It runs
fixed, validated Git operations only — a shallow clone and a HEAD lookup — with:
  - an argument list (never a shell string), shell=False
  - only the configured Git executable (no user-controlled program)
  - a controlled working/destination directory (never LLM-chosen)
  - a sanitized environment (no prompts, no LFS smudge, https-only, no system config)
  - a timeout, captured output, and sanitized/truncated stderr

It accepts no arbitrary commands, flags, URLs, or env from the caller/LLM.
"""

import os
import shutil
import subprocess

import tools.config as config
from tools.base import ToolFailure
from tools.models import (
    GIT_CLONE_FAILED,
    GIT_CLONE_TIMEOUT,
    GIT_COMMIT_LOOKUP_FAILED,
    GIT_NOT_AVAILABLE,
)

_STDERR_MAX = 500


def _sanitize_stderr(text):
    """Truncate and strip control chars so git's stderr is safe to log/report."""
    if not text:
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = "".join(ch for ch in text if ch >= " ")
    text = " ".join(text.split())
    if len(text) > _STDERR_MAX:
        text = text[:_STDERR_MAX] + "…"
    return text


def _git_env():
    """A minimal, hardened environment for git invocations."""
    env = {
        "GIT_TERMINAL_PROMPT": "0",       # never prompt for credentials
        "GIT_LFS_SKIP_SMUDGE": "1",       # do not download LFS objects
        "GIT_ALLOW_PROTOCOL": "https",    # https transport only (blocks file/ssh/git)
        "GIT_CONFIG_NOSYSTEM": "1",       # ignore system git config
        "GCM_INTERACTIVE": "never",       # no credential-manager prompts
        "GIT_ASKPASS": "",
    }
    # Preserve just enough of the real environment for git to locate itself / TLS.
    for key in ("PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP",
                "HOMEDRIVE", "HOMEPATH", "PROGRAMFILES", "ProgramFiles",
                "APPDATA", "LOCALAPPDATA", "PATHEXT", "COMSPEC"):
        if key in os.environ:
            env[key] = os.environ[key]
    # Point git config at nothing so no user hooks/aliases/config are inherited.
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    return env


class GitRunner:
    def __init__(self, git_executable=None):
        self._git = git_executable

    @property
    def git(self):
        return self._git or config.git_executable()

    def ensure_available(self):
        """Confirm the configured git executable can be found, else GIT_NOT_AVAILABLE."""
        if shutil.which(self.git) is None and not os.path.isfile(self.git):
            raise ToolFailure(GIT_NOT_AVAILABLE,
                              "Git is not available; repository cloning cannot proceed.")

    def clone(self, https_url, ref, destination, timeout=None):
        """Shallow, single-branch, no-tags, no-submodule clone into `destination`.

        `https_url`, `ref`, and `destination` are all constructed/validated by the
        caller — never taken from the LLM. Returns None on success; raises ToolFailure.
        """
        self.ensure_available()
        argv = [self.git, "clone", "--depth", "1", "--single-branch", "--no-tags"]
        if ref:
            argv += ["--branch", ref]
        argv += ["--", https_url, str(destination)]

        try:
            proc = subprocess.run(
                argv,
                shell=False,
                env=_git_env(),
                capture_output=True,
                text=True,
                timeout=timeout if timeout is not None else config.git_clone_timeout(),
            )
        except subprocess.TimeoutExpired:
            raise ToolFailure(GIT_CLONE_TIMEOUT, "The clone operation timed out.", retryable=True)
        except (OSError, ValueError) as e:
            raise ToolFailure(GIT_CLONE_FAILED, f"The clone could not be started ({type(e).__name__}).")

        if proc.returncode != 0:
            raise ToolFailure(GIT_CLONE_FAILED,
                              f"git clone failed: {_sanitize_stderr(proc.stderr)}",
                              log_meta={"git_returncode": proc.returncode})
        return None

    def rev_parse_head(self, repo_dir, timeout=30):
        """Return the full commit SHA of HEAD in `repo_dir` (a fixed, safe command)."""
        self.ensure_available()
        argv = [self.git, "-C", str(repo_dir), "rev-parse", "HEAD"]
        try:
            proc = subprocess.run(
                argv, shell=False, env=_git_env(),
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise ToolFailure(GIT_COMMIT_LOOKUP_FAILED, "Commit lookup timed out.")
        except (OSError, ValueError) as e:
            raise ToolFailure(GIT_COMMIT_LOOKUP_FAILED, f"Commit lookup failed ({type(e).__name__}).")
        if proc.returncode != 0:
            raise ToolFailure(GIT_COMMIT_LOOKUP_FAILED,
                              f"Commit lookup failed: {_sanitize_stderr(proc.stderr)}")
        return proc.stdout.strip()
