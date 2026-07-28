"""repo.list_files and repo.read_file: bounds, containment, symlink safety, binary/dir."""

import os

import pytest

from tools.base import ToolFailure
from tools.repo_tools import ListFilesTool, ReadFileTool


@pytest.fixture
def cloned(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOSITORY_ROOT", str(tmp_path / "repos"))
    d = tmp_path / "repos" / "octocat" / "Hello-World"
    (d / "src").mkdir(parents=True)
    (d / ".git").mkdir()
    (d / "README.md").write_text("# Hello\nWorld", encoding="utf-8")
    (d / "src" / "main.py").write_text("print('hi')", encoding="utf-8")
    (d / "src" / "util.py").write_text("x = 1", encoding="utf-8")
    (d / "logo.png").write_bytes(b"\x89PNG\x00\x01\x02binary")
    (d / ".git" / "config").write_text("[core]", encoding="utf-8")
    return d


def _list(**args):
    t = ListFilesTool()
    return t.execute(t.validate_arguments(args))


def _read(**args):
    t = ReadFileTool()
    return t.execute(t.validate_arguments(args))


def test_root_listing(cloned):
    data = _list(repository="octocat/Hello-World", path="")
    names = {e["name"] for e in data["entries"]}
    assert "README.md" in names and "src" in names
    assert ".git" not in names  # .git excluded
    assert data["untrusted_content"] is True and data["executed"] is False


def test_nested_non_recursive(cloned):
    data = _list(repository="octocat/Hello-World", path="src")
    names = {e["name"] for e in data["entries"]}
    assert names == {"main.py", "util.py"}


def test_recursive_bounded(cloned):
    data = _list(repository="octocat/Hello-World", path="", recursive=True, max_depth=5)
    paths = {e["path"] for e in data["entries"]}
    assert "src/main.py" in paths
    assert not any(p.startswith(".git") for p in paths)


def test_entry_limit(cloned):
    data = _list(repository="octocat/Hello-World", path="", recursive=True, limit=2)
    assert data["truncated"] is True
    assert len(data["entries"]) <= 2


def test_deterministic_sort(cloned):
    d1 = _list(repository="octocat/Hello-World", path="src")
    d2 = _list(repository="octocat/Hello-World", path="src")
    assert [e["path"] for e in d1["entries"]] == [e["path"] for e in d2["entries"]]


def test_list_repository_not_cloned(cloned):
    with pytest.raises(ToolFailure) as e:
        _list(repository="nobody/nope", path="")
    assert e.value.code == "REPOSITORY_NOT_CLONED"


def test_list_traversal_rejected(cloned):
    with pytest.raises(ToolFailure) as e:
        _list(repository="octocat/Hello-World", path="../../")
    assert e.value.code in ("INVALID_REPOSITORY_PATH", "REPOSITORY_PATH_ESCAPE")


def test_read_utf8(cloned):
    data = _read(repository="octocat/Hello-World", path="README.md")
    assert "# Hello" in data["text"] and "World" in data["text"]
    assert data["source_type"] == "cloned_repository_file"
    assert data["untrusted_content"] is True and data["executed"] is False


def test_read_truncation(cloned):
    data = _read(repository="octocat/Hello-World", path="README.md", max_chars=4)
    assert data["truncated"] is True
    assert data["text"] == "# He"


def test_read_binary_rejected(cloned):
    with pytest.raises(ToolFailure) as e:
        _read(repository="octocat/Hello-World", path="logo.png")
    assert e.value.code == "REPOSITORY_BINARY_FILE"


def test_read_directory_rejected(cloned):
    with pytest.raises(ToolFailure) as e:
        _read(repository="octocat/Hello-World", path="src")
    assert e.value.code == "INVALID_REPOSITORY_PATH"


def test_read_missing(cloned):
    with pytest.raises(ToolFailure) as e:
        _read(repository="octocat/Hello-World", path="nope.txt")
    assert e.value.code == "REPOSITORY_FILE_NOT_FOUND"


def test_read_traversal_rejected(cloned):
    with pytest.raises(ToolFailure) as e:
        _read(repository="octocat/Hello-World", path="../../.env")
    assert e.value.code in ("INVALID_REPOSITORY_PATH", "REPOSITORY_PATH_ESCAPE")


def test_read_absolute_rejected(cloned):
    with pytest.raises(ToolFailure) as e:
        _read(repository="octocat/Hello-World", path="/etc/passwd")
    assert e.value.code in ("INVALID_REPOSITORY_PATH", "REPOSITORY_PATH_ESCAPE")


def _try_symlink(link, target):
    try:
        os.symlink(target, link)
        return True
    except (OSError, NotImplementedError):
        return False


def test_read_symlink_rejected(cloned, tmp_path):
    secret = tmp_path / "host_secret.txt"
    secret.write_text("SECRET", encoding="utf-8")
    link = cloned / "link.txt"
    if not _try_symlink(str(link), str(secret)):
        pytest.skip("symlinks not permitted on this platform")
    with pytest.raises(ToolFailure) as e:
        _read(repository="octocat/Hello-World", path="link.txt")
    assert e.value.code in ("REPOSITORY_SYMLINK_BLOCKED", "REPOSITORY_PATH_ESCAPE")


def test_list_marks_symlink_not_followed(cloned, tmp_path):
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    link = cloned / "external"
    if not _try_symlink(str(link), str(outside)):
        pytest.skip("symlinks not permitted on this platform")
    data = _list(repository="octocat/Hello-World", path="", recursive=True, max_depth=5)
    sym = [e for e in data["entries"] if e["name"] == "external"]
    assert sym and sym[0]["type"] == "symlink" and sym[0]["followed"] is False
    # Never traversed into the symlinked dir.
    assert not any("external/" in e["path"] for e in data["entries"])
