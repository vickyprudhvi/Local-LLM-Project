"""Phase F — the automatic provisioning step inside a live turn.

The unit pieces were already covered; these tests cover the WIRING that was
missing: a real request must reach detection, produce a plan, prompt for approval,
install, register the new tools, and leave the original request answerable by the
normal pipeline.
"""

import os
from types import SimpleNamespace

import pytest

import assistant
from mcp_layer.errors import McpError
from mcp_management.capability_detector import extract_directory_candidate
from tests.mcp_provisioning_helpers import (
    FakeNpm,
    make_manager,
    node_available,
    workspace_with_file,
)


# ---- deterministic directory extraction ----

def test_extracts_directory_from_a_windows_file_path(tmp_path):
    workspace = workspace_with_file(tmp_path)
    target = os.path.join(workspace, "hello.txt")
    assert extract_directory_candidate(f"Read hello.txt from {target}.") == workspace


def test_extracts_directory_when_the_path_is_a_directory(tmp_path):
    workspace = workspace_with_file(tmp_path)
    # Trailing sentence punctuation must not become part of the path.
    assert extract_directory_candidate(f"Read hello.txt from {workspace}.") == workspace


def test_extracts_unquoted_path_containing_spaces(tmp_path):
    """Regression: a real path like 'C:\\...\\Local LLM Project\\...' has spaces.

    A regex that stops at the first space yields a non-existent directory, which
    then fails approval — the exact failure seen in a live run.
    """
    spaced = tmp_path / "Local LLM Project" / "mcp_workspaces" / "user_files"
    spaced.mkdir(parents=True)
    (spaced / "hello.txt").write_text("hi", encoding="utf-8")

    assert extract_directory_candidate(f"Read hello.txt from {spaced}.") == str(spaced)
    assert extract_directory_candidate(
        f"Read {spaced / 'hello.txt'} please") == str(spaced)


def test_space_extension_never_absorbs_trailing_sentence_words(tmp_path):
    """Only an existing path may be grown across spaces."""
    workspace = workspace_with_file(tmp_path)
    found = extract_directory_candidate(
        f"Read hello.txt from {workspace} and then summarize it for me")
    assert found == workspace


def test_extracts_quoted_path(tmp_path):
    workspace = workspace_with_file(tmp_path)
    assert extract_directory_candidate(f'List the files in "{workspace}"') == workspace


def test_extracts_relative_path_and_strips_filename():
    assert extract_directory_candidate("Read notes.txt from Documents/project") == \
        "Documents/project"
    assert extract_directory_candidate("Open Documents/project/plan.md") == \
        "Documents/project"


def test_no_path_yields_none():
    for text in ("Read my notes.", "List the files.", "What time is it?", "", None):
        assert extract_directory_candidate(text) is None


def test_control_characters_never_produce_a_candidate():
    assert extract_directory_candidate("Read /tmp/a\x00b/file.txt") != "/tmp/a\x00b"


# ---- the live turn step ----

@pytest.fixture
def ctx(tmp_path, monkeypatch):
    if not node_available():
        pytest.skip("node/npm not available")
    manager, paths = make_manager(tmp_path)
    workspace = workspace_with_file(tmp_path)
    # Never touch the real repo config or start a real Phase E session in tests.
    monkeypatch.setattr(assistant, "_start_mcp", lambda: SimpleNamespace(
        restarted=True, shutdown=lambda: None))
    return SimpleNamespace(manager=manager, paths=paths, workspace=workspace,
                           text=f"Read hello.txt from {workspace}")


class _Session:
    def __init__(self):
        self.shutdown_called = False

    def shutdown(self):
        self.shutdown_called = True


def test_request_needing_a_capability_offers_a_plan_and_installs_on_approval(ctx, monkeypatch):
    shown = []
    monkeypatch.setattr("mcp_management.manager.install",
                        _recording_install(ctx, shown))
    session = _Session()

    new_session = assistant._provision_if_needed(
        ctx.manager, session, ctx.text, confirmer=lambda plan: shown.append(plan) or True)

    assert shown, "the plan must be shown before installing"
    plan = shown[0]
    # The plan names the directory taken from the user's own words.
    assert str(plan.requested_directories[0]) == os.path.realpath(ctx.workspace)
    assert plan.package_version == "2026.7.10"
    # The old session was replaced so the new tools can register.
    assert session.shutdown_called is True
    assert getattr(new_session, "restarted", False) is True


def test_declining_installs_nothing_and_keeps_the_session(ctx):
    session = _Session()
    new_session = assistant._provision_if_needed(
        ctx.manager, session, ctx.text, confirmer=lambda plan: False)
    assert new_session is session          # unchanged
    assert session.shutdown_called is False
    from mcp_management.registry import load_registry
    assert load_registry(None, ctx.paths["base_dir"], ctx.paths["managed_root"]) == {}


def test_no_directory_named_means_no_offer(ctx):
    session = _Session()
    called = []
    new_session = assistant._provision_if_needed(
        ctx.manager, session, "Read my notes please.",
        confirmer=lambda plan: called.append(plan) or True)
    assert called == [], "must never guess a directory to grant"
    assert new_session is session


def test_unrelated_request_is_untouched(ctx):
    session = _Session()
    called = []
    new_session = assistant._provision_if_needed(
        ctx.manager, session, "What is the capital of France?",
        confirmer=lambda plan: called.append(plan) or True)
    assert called == []
    assert new_session is session


def test_already_installed_capability_makes_no_offer(ctx, monkeypatch):
    from mcp_management.registry import STATUS_INSTALLED, InstalledServer, upsert

    upsert("filesystem", InstalledServer(
        catalog_id="official-filesystem", installed_version="2026.7.10",
        status=STATUS_INSTALLED, install_directory="x", configuration_path="x/server.json",
        installed_at="now"), None, ctx.paths["base_dir"], ctx.paths["managed_root"])

    session = _Session()
    called = []
    new_session = assistant._provision_if_needed(
        ctx.manager, session, ctx.text, confirmer=lambda plan: called.append(plan) or True)
    assert called == [], "an installed capability must not be re-provisioned"
    assert new_session is session


def test_provisioning_failure_keeps_builtins_working(ctx, monkeypatch):
    def failing_install(*args, **kwargs):
        raise McpError("MCP_INSTALLATION_FAILED", "forced")

    monkeypatch.setattr("mcp_management.manager.install", failing_install)
    session = _Session()
    new_session = assistant._provision_if_needed(
        ctx.manager, session, ctx.text, confirmer=lambda plan: True)
    # Failure is contained: the old session survives and the turn continues.
    assert new_session is session
    assert session.shutdown_called is False


def test_nonexistent_directory_reports_the_reason_not_just_a_code(ctx, capsys):
    """A path that does not exist must explain itself, not print a bare error code."""
    session = _Session()
    missing = os.path.join(str(ctx.paths["base_dir"]), "no-such-project", "user_files")
    new_session = assistant._provision_if_needed(
        ctx.manager, session, f"Read hello.txt from {missing}.",
        confirmer=lambda plan: pytest.fail("must not ask for approval"))

    output = capsys.readouterr().out
    assert "does not exist" in output, output
    assert "MCP_DIRECTORY_NOT_APPROVED" in output
    assert new_session is session


def test_no_manager_is_a_no_op():
    session = _Session()
    assert assistant._provision_if_needed(None, session, "Read a/b/c.txt") is session


def _recording_install(ctx, shown):
    """Install stub that records nothing extra but succeeds like the real one."""
    from mcp_management.installer import install as real_install

    def _install(plan, entry, approval, **kwargs):
        kwargs.setdefault("npm_runner", FakeNpm())
        return real_install(plan, entry, approval, **kwargs)

    return _install
