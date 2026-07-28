"""github.clone_repository: preflight, staging, limits, containment (git + client mocked)."""

import os

import pytest

from tools.base import ToolFailure
from tools.github_client import GitHubResponse
from tools.repo_clone import CloneRepositoryTool
from tools import repo_store


class FakeClient:
    def __init__(self, data, status=200):
        self._resp = GitHubResponse(status, data, {"remaining": 50, "reset_at": None})

    def get(self, path, params=None):
        return self._resp


class FakeRunner:
    """Simulates git by materializing a small repo tree in the staging dir."""
    def __init__(self, files=None, fail=None, commit="deadbeef"):
        self.files = files if files is not None else {"README.md": "# hi", "main.py": "print(1)"}
        self.fail = fail
        self.commit = commit
        self.cloned = False

    def clone(self, url, ref, destination, timeout=None):
        if self.fail:
            raise self.fail
        os.makedirs(destination, exist_ok=True)
        for rel, content in self.files.items():
            p = os.path.join(destination, rel)
            os.makedirs(os.path.dirname(p) or destination, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
        self.cloned = True

    def rev_parse_head(self, repo_dir, timeout=30):
        return self.commit


PUBLIC = {"private": False, "visibility": "public", "default_branch": "main", "size": 10}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOSITORY_ROOT", str(tmp_path / "repos"))
    monkeypatch.setenv("REPOSITORY_CLONE_ENABLED", "true")
    return tmp_path / "repos"


def _tool(data=PUBLIC, runner=None, status=200):
    return CloneRepositoryTool(client=FakeClient(data, status), runner=runner or FakeRunner())


def test_valid_public_clone(root):
    tool = _tool()
    data = tool.execute(tool.validate_arguments({"repository": "octocat/Hello-World"}))
    assert data["repository"] == "octocat/Hello-World"
    assert data["ref"] == "main"
    assert data["commit"] == "deadbeef"
    assert data["executed"] is False and data["installed"] is False
    assert data["shallow"] is True and data["submodules_initialized"] is False
    assert "relative_path" in data and data["relative_path"] == "octocat/Hello-World"
    # No absolute path leaked to the LLM payload.
    assert not any(isinstance(v, str) and (":" in v and os.sep in v) for k, v in data.items() if k != "_log_meta")
    assert os.path.isdir(root / "octocat" / "Hello-World")


def test_default_branch_used_when_no_ref(root):
    tool = _tool({"private": False, "visibility": "public", "default_branch": "trunk", "size": 5})
    data = tool.execute(tool.validate_arguments({"repository": "a/b"}))
    assert data["ref"] == "trunk"


def test_invalid_ref_rejected(root):
    tool = _tool()
    with pytest.raises(ToolFailure) as e:
        tool.validate_arguments({"repository": "a/b", "ref": "-evil"})
    assert e.value.code == "INVALID_REPOSITORY_REF"


def test_private_repo_rejected(root):
    tool = _tool({"private": True, "visibility": "private", "default_branch": "main", "size": 5})
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"repository": "a/b"}))
    assert e.value.code == "PRIVATE_REPOSITORY_NOT_SUPPORTED"


def test_not_found_rejected(root):
    tool = _tool(None, status=404)
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"repository": "a/missing"}))
    assert e.value.code == "GITHUB_REPOSITORY_NOT_FOUND"


def test_preflight_size_limit(root, monkeypatch):
    monkeypatch.setenv("MAX_REPOSITORY_PREFLIGHT_SIZE_KB", "5")
    tool = _tool({"private": False, "visibility": "public", "default_branch": "main", "size": 999})
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"repository": "a/b"}))
    assert e.value.code == "REPOSITORY_TOO_LARGE"


def test_post_clone_file_limit(root, monkeypatch):
    monkeypatch.setenv("MAX_CLONED_REPOSITORY_FILES", "1")
    runner = FakeRunner(files={"a.py": "x", "b.py": "y", "c.py": "z"})
    tool = _tool(runner=runner)
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"repository": "a/b"}))
    assert e.value.code == "REPOSITORY_FILE_LIMIT_EXCEEDED"
    # Staging cleaned; final path not created.
    assert not os.path.isdir(root / "a" / "b")
    assert not os.listdir(root / ".staging")


def test_post_clone_size_limit(root, monkeypatch):
    monkeypatch.setenv("MAX_CLONED_REPOSITORY_SIZE_MB", "0")
    runner = FakeRunner(files={"big.txt": "x" * 100})
    tool = _tool(runner=runner)
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"repository": "a/b"}))
    assert e.value.code == "REPOSITORY_TOO_LARGE"
    assert not os.path.isdir(root / "a" / "b")


def test_already_cloned(root):
    (root / "a" / "b").mkdir(parents=True)
    tool = _tool()
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"repository": "a/b"}))
    assert e.value.code == "REPOSITORY_ALREADY_CLONED"


def test_clone_failure_cleans_staging(root):
    runner = FakeRunner(fail=ToolFailure("GIT_CLONE_FAILED", "boom"))
    tool = _tool(runner=runner)
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"repository": "a/b"}))
    assert e.value.code == "GIT_CLONE_FAILED"
    assert not os.path.isdir(root / "a" / "b")
    assert not os.listdir(root / ".staging")


def test_invalid_repository_identifier(root):
    tool = _tool()
    with pytest.raises(ToolFailure) as e:
        tool.validate_arguments({"repository": "not-a-repo"})
    assert e.value.code == "INVALID_REPOSITORY"


def test_result_json_serializable(root):
    import json
    tool = _tool()
    data = tool.execute(tool.validate_arguments({"repository": "a/b"}))
    data.pop("_log_meta", None)
    json.dumps(data)
