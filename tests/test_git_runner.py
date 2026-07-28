"""The dedicated Git runner: fixed argv, shell=False, hardened env, timeouts."""

import subprocess
from unittest.mock import patch

import pytest

import tools.git_runner as gr
from tools.base import ToolFailure
from tools.git_runner import GitRunner, _sanitize_stderr


class FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture(autouse=True)
def _git_available(monkeypatch):
    monkeypatch.setattr(gr.shutil, "which", lambda name: "/usr/bin/git")


def test_clone_uses_argv_shell_false_and_safe_flags():
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return FakeCompleted(0)

    with patch.object(gr.subprocess, "run", fake_run):
        GitRunner(git_executable="git").clone("https://github.com/a/b.git", "main", "/tmp/staging")

    argv = captured["argv"]
    assert argv[0] == "git" and argv[1] == "clone"
    assert "--depth" in argv and "1" in argv
    assert "--single-branch" in argv and "--no-tags" in argv
    assert "--branch" in argv and "main" in argv
    assert "--" in argv  # separates options from url/dest
    assert argv[-2] == "https://github.com/a/b.git"
    assert captured["kwargs"]["shell"] is False
    # No recurse-submodules flag is ever added.
    assert not any("recurse-submodules" in a for a in argv)


def test_clone_env_hardening_and_no_token():
    captured = {}

    def fake_run(argv, **kwargs):
        captured["env"] = kwargs["env"]
        captured["argv"] = argv
        return FakeCompleted(0)

    with patch.object(gr.subprocess, "run", fake_run):
        GitRunner("git").clone("https://github.com/a/b.git", None, "/tmp/s")

    env = captured["env"]
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_LFS_SKIP_SMUDGE"] == "1"
    assert env["GIT_ALLOW_PROTOCOL"] == "https"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    # No token anywhere in argv or env.
    joined = " ".join(captured["argv"]) + " " + " ".join(f"{k}={v}" for k, v in env.items())
    assert "github_pat" not in joined and "Bearer" not in joined and "ghp_" not in joined


def test_clone_no_ref_omits_branch():
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return FakeCompleted(0)

    with patch.object(gr.subprocess, "run", fake_run):
        GitRunner("git").clone("https://github.com/a/b.git", None, "/tmp/s")
    assert "--branch" not in captured["argv"]


def test_clone_timeout_raises_controlled():
    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=1)
    with patch.object(gr.subprocess, "run", fake_run):
        with pytest.raises(ToolFailure) as e:
            GitRunner("git").clone("https://github.com/a/b.git", None, "/tmp/s")
    assert e.value.code == "GIT_CLONE_TIMEOUT"


def test_clone_nonzero_exit_sanitizes_stderr():
    with patch.object(gr.subprocess, "run",
                      lambda argv, **kw: FakeCompleted(128, stderr="fatal: repository not found\n")):
        with pytest.raises(ToolFailure) as e:
            GitRunner("git").clone("https://github.com/a/b.git", None, "/tmp/s")
    assert e.value.code == "GIT_CLONE_FAILED"
    assert "\n" not in e.value.message


def test_git_missing_raises(monkeypatch):
    monkeypatch.setattr(gr.shutil, "which", lambda name: None)
    monkeypatch.setattr(gr.os.path, "isfile", lambda p: False)
    with pytest.raises(ToolFailure) as e:
        GitRunner("git").clone("https://github.com/a/b.git", None, "/tmp/s")
    assert e.value.code == "GIT_NOT_AVAILABLE"


def test_rev_parse_head():
    with patch.object(gr.subprocess, "run", lambda argv, **kw: FakeCompleted(0, stdout="abc123\n")):
        sha = GitRunner("git").rev_parse_head("/tmp/repo")
    assert sha == "abc123"


def test_rev_parse_failure():
    with patch.object(gr.subprocess, "run", lambda argv, **kw: FakeCompleted(1, stderr="bad")):
        with pytest.raises(ToolFailure) as e:
            GitRunner("git").rev_parse_head("/tmp/repo")
    assert e.value.code == "GIT_COMMIT_LOOKUP_FAILED"


def test_sanitize_stderr_truncates_and_strips():
    out = _sanitize_stderr("line1\nline2\r\n" + "x" * 1000)
    assert "\n" not in out and "\r" not in out
    assert len(out) <= 501
