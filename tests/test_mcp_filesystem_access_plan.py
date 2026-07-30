"""Phase F.1 Task 2-4 — models, outside-root detection, and narrow-root selection."""

import os

import pytest

from mcp_layer.errors import McpError
from mcp_management.access_classifier import (
    classify_outside_root_failure,
    propose_root,
)
from mcp_management.filesystem_access import (
    FilesystemAccessOperation,
    FilesystemAccessPlan,
    PendingFilesystemAccessRequest,
    PendingFilesystemAccessState,
)
from tools.models import ToolCall, ToolResult

pytestmark = pytest.mark.filterwarnings("ignore")


# ---- plan hashing / expiry ----

def _plan(**overrides):
    base = dict(
        plan_id="fsplan_1",
        server_id="filesystem",
        catalog_id="official-filesystem",
        operation=FilesystemAccessOperation.ADD_ROOT,
        requested_directory="/approved/new",
        current_allowed_directories=("/approved/old",),
        proposed_allowed_directories=("/approved/new", "/approved/old"),
    )
    base.update(overrides)
    return FilesystemAccessPlan(**base).with_hash()


def test_plan_hash_changes_when_a_security_field_changes():
    a = _plan()
    b = _plan(proposed_allowed_directories=("/approved/old", "/approved/new", "/approved/extra"))
    assert a.plan_hash != b.plan_hash


def test_plan_hash_is_stable_for_identical_security_fields():
    a = _plan()
    b = _plan()
    assert a.plan_hash == b.plan_hash


def test_plan_is_not_expired_without_expires_at():
    assert _plan().is_expired() is False


def test_plan_expiry_is_respected():
    import datetime

    past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=1)).isoformat(
        timespec="seconds")
    assert _plan(expires_at=past).is_expired() is True


def test_pending_request_advances_immutably():
    req = PendingFilesystemAccessRequest(
        request_id="fsreq_1", original_user_text="read x", requested_path="x",
        proposed_root="/approved", server_id="filesystem",
    )
    advanced = req.advanced(PendingFilesystemAccessState.AWAITING_APPROVAL, plan_id="fsplan_1")
    assert req.state == PendingFilesystemAccessState.DETECTED  # original untouched
    assert advanced.state == PendingFilesystemAccessState.AWAITING_APPROVAL
    assert advanced.access_plan_id == "fsplan_1"
    assert advanced.provisioning_attempts == 0


# ---- outside-root detection ----

def _fail_result(tool_name="mcp.filesystem.read_text_file", code="MCP_CALL_FAILED",
                 message="Access denied - outside allowed directories"):
    return ToolResult.fail(tool_name, "call_1", code, message)


def test_path_outside_every_root_is_detected(tmp_path):
    target_dir = tmp_path / "outside"
    target_dir.mkdir()
    target = target_dir / "README.md"
    target.write_text("hi", encoding="utf-8")
    allowed = (str(tmp_path / "mcp_workspaces" / "filesystem"),)
    (tmp_path / "mcp_workspaces" / "filesystem").mkdir(parents=True)

    result = _fail_result()
    failure = classify_outside_root_failure(
        "mcp.filesystem.read_text_file", {"path": str(target)}, result, allowed,
        base_dir=str(tmp_path))
    assert failure is not None
    assert failure.eligible is True
    assert failure.proposed_root == os.path.realpath(str(target_dir))


def test_path_inside_an_approved_root_is_not_classified(tmp_path):
    """A 'file not found inside an approved root' style failure must not spawn a plan."""
    allowed_dir = tmp_path / "approved"
    allowed_dir.mkdir()
    target = allowed_dir / "missing.md"

    result = ToolResult.fail("mcp.filesystem.read_text_file", "call_1", "MCP_CALL_FAILED",
                             "ENOENT: no such file or directory")
    failure = classify_outside_root_failure(
        "mcp.filesystem.read_text_file", {"path": str(target)}, result, (str(allowed_dir),),
        base_dir=str(tmp_path))
    assert failure is None


@pytest.mark.parametrize("code", [
    "TOOL_TIMEOUT", "MCP_TIMEOUT", "MCP_SERVER_EXITED", "TOOL_EXECUTION_ERROR",
    "INVALID_ARGUMENTS", "MALFORMED_TOOL_CALL", "TOOL_CONFIRMATION_REQUIRED",
])
def test_unrelated_failure_codes_are_never_classified(tmp_path, code):
    target = tmp_path / "outside" / "file.txt"
    result = ToolResult.fail("mcp.filesystem.read_text_file", "call_1", code, "boom")
    failure = classify_outside_root_failure(
        "mcp.filesystem.read_text_file", {"path": str(target)}, result, (), base_dir=str(tmp_path))
    assert failure is None


def test_successful_result_is_never_classified():
    result = ToolResult.ok("mcp.filesystem.read_text_file", "call_1", {"content": "hi"})
    failure = classify_outside_root_failure(
        "mcp.filesystem.read_text_file", {"path": "/outside/x.txt"}, result, ())
    assert failure is None


def test_non_mcp_tool_is_never_classified():
    result = ToolResult.fail("github.read_file", "call_1", "GITHUB_FILE_NOT_FOUND", "boom")
    failure = classify_outside_root_failure(
        "github.read_file", {"path": "/outside/x.txt"}, result, ())
    assert failure is None


def test_missing_path_argument_is_never_classified():
    result = _fail_result()
    failure = classify_outside_root_failure("mcp.filesystem.list_directory", {}, result, ())
    assert failure is None


def test_call_arguments_are_not_attached_to_toolresult_but_are_captured_by_caller():
    """Documents the real ToolResult contract this classifier depends on: the
    caller (not ToolResult) must supply the original call arguments."""
    call = ToolCall(call_id="c1", tool_name="mcp.filesystem.read_text_file",
                    arguments={"path": "/outside/x.txt"})
    result = _fail_result()
    assert not hasattr(result, "arguments")
    failure = classify_outside_root_failure(call.tool_name, call.arguments, result, ())
    assert failure is not None


# ---- narrowest-root selection ----

def test_single_file_proposes_its_parent_directory(tmp_path):
    nested = tmp_path / "project" / "data" / "repo" / "chapter_pdfs"
    nested.mkdir(parents=True)
    (nested / "README.md").touch()
    proposal = propose_root([str(nested / "README.md")], remote_name="read_text_file",
                            base_dir=str(tmp_path))
    assert proposal.ok is True
    assert proposal.directory == os.path.realpath(str(nested))
    # never the project root or an intermediate ancestor
    assert proposal.directory != os.path.realpath(str(tmp_path / "project"))
    assert proposal.directory != os.path.realpath(str(tmp_path / "project" / "data"))


def test_directory_listing_request_proposes_the_directory_itself(tmp_path):
    target = tmp_path / "project" / "data" / "repo"
    target.mkdir(parents=True)
    proposal = propose_root([str(target)], remote_name="list_directory", base_dir=str(tmp_path))
    assert proposal.ok is True
    assert proposal.directory == os.path.realpath(str(target))


def test_multiple_files_under_one_parent_propose_the_common_parent(tmp_path):
    parent = tmp_path / "project" / "docs"
    parent.mkdir(parents=True)
    (parent / "a.md").touch()
    (parent / "b.md").touch()
    proposal = propose_root([str(parent / "a.md"), str(parent / "b.md")],
                            remote_name="read_multiple_files", base_dir=str(tmp_path))
    assert proposal.ok is True
    assert proposal.directory == os.path.realpath(str(parent))


def test_restricted_leaf_name_is_never_proposed(tmp_path):
    ssh_dir = tmp_path / "home" / ".ssh"
    ssh_dir.mkdir(parents=True)
    key = ssh_dir / "id_rsa"
    key.touch()
    proposal = propose_root([str(key)], remote_name="read_text_file", base_dir=str(tmp_path))
    assert proposal.ok is False
    assert proposal.restricted is True
    assert proposal.directory is None


def test_repo_root_is_not_proposed_without_justification(tmp_path):
    # A file directly at the repo root would otherwise propose the root itself.
    (tmp_path / "README.md").touch()
    proposal = propose_root([str(tmp_path / "README.md")], remote_name="read_text_file",
                            base_dir=str(tmp_path))
    assert proposal.ok is False


def test_unrelated_paths_on_different_drives_require_a_separate_plan(tmp_path):
    proposal = propose_root(["C:\\a\\b\\file.txt", "D:\\x\\y\\file.txt"],
                            remote_name="read_multiple_files")
    assert proposal.ok is False
    assert proposal.directory is None
