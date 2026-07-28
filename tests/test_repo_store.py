"""repo_store: root resolution, path containment, ref validation, symlink-safe sizing."""

import os

import pytest

from tools.base import ToolFailure
from tools import repo_store


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "repos"
    monkeypatch.setenv("REPOSITORY_ROOT", str(r))
    return r


def _make_repo(root, owner="octocat", repo="Hello-World"):
    d = root / owner / repo
    (d / "src").mkdir(parents=True)
    (d / "README.md").write_text("hello", encoding="utf-8")
    (d / "src" / "main.py").write_text("print('hi')", encoding="utf-8")
    return d


def test_repository_root_absolute(root):
    abs_root = repo_store.repository_root_abs()
    assert os.path.isabs(abs_root)
    assert os.path.isdir(abs_root)


@pytest.mark.parametrize("ref", ["main", "v1.2.3", "release/1.0", "feature-x"])
def test_valid_refs(ref):
    assert repo_store.validate_clone_ref(ref) == ref


@pytest.mark.parametrize("ref", ["-x", "--upload-pack=evil", "a..b", "a\x00b", "a b;rm"])
def test_invalid_refs(ref):
    with pytest.raises(ToolFailure) as e:
        repo_store.validate_clone_ref(ref)
    assert e.value.code == "INVALID_REPOSITORY_REF"


def test_target_path_inside_root(root):
    p = repo_store.target_path("octocat", "Hello-World")
    assert p.startswith(repo_store.repository_root_abs() + os.sep)


def test_require_cloned_missing(root):
    with pytest.raises(ToolFailure) as e:
        repo_store.require_cloned_repo("octocat/Hello-World")
    assert e.value.code == "REPOSITORY_NOT_CLONED"


def test_require_cloned_present(root):
    _make_repo(root)
    owner, repo, path = repo_store.require_cloned_repo("octocat/Hello-World")
    assert (owner, repo) == ("octocat", "Hello-World")
    assert os.path.isdir(path)


def test_resolve_within_ok(root):
    repo_dir = str(_make_repo(root))
    abs_path, rel = repo_store.resolve_within(repo_dir, "src/main.py", allow_empty=False)
    assert rel == "src/main.py"
    assert os.path.isfile(abs_path)


def test_resolve_within_traversal_rejected(root):
    repo_dir = str(_make_repo(root))
    with pytest.raises(ToolFailure) as e:
        repo_store.resolve_within(repo_dir, "../../.env", allow_empty=False)
    assert e.value.code in ("INVALID_REPOSITORY_PATH", "REPOSITORY_PATH_ESCAPE")


@pytest.mark.skipif(os.name == "nt", reason="symlink creation may need privileges on Windows")
def test_resolve_within_symlink_escape_blocked(root, tmp_path):
    repo_dir = _make_repo(root)
    secret = tmp_path / "outside_secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")
    link = repo_dir / "escape"
    os.symlink(str(secret), str(link))
    with pytest.raises(ToolFailure) as e:
        repo_store.resolve_within(str(repo_dir), "escape", allow_empty=False)
    assert e.value.code == "REPOSITORY_PATH_ESCAPE"


def test_measure_repository(root):
    repo_dir = _make_repo(root)
    size, count = repo_store.measure_repository(str(repo_dir))
    assert count == 2  # README.md + src/main.py
    assert size > 0
