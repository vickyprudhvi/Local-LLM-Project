"""The 5 github.* tools (GitHub client mocked)."""

import base64
import json

import pytest

from tools.base import ToolFailure, ToolValidationError
from tools.github_client import GitHubResponse
from tools.github_tools import (
    GetRepositoryTool,
    ListDirectoryTool,
    ListReleasesTool,
    ReadFileTool,
    SearchRepositoriesTool,
    parse_repository,
    validate_path,
)


class FakeClient:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def get(self, path, params=None):
        self.calls.append({"path": path, "params": params})
        return self._response


def _resp(data, status=200, rate=None):
    return GitHubResponse(status, data, rate or {"remaining": 50, "reset_at": None})


# ---- identifier / path validation ----

def test_parse_repository_forms():
    assert parse_repository({"repository": "ollama/ollama"}) == ("ollama", "ollama")
    assert parse_repository({"owner": "a", "repo": "b"}) == ("a", "b")


@pytest.mark.parametrize("bad", [{"repository": "noslash"}, {"repository": "a/b/c"},
                                 {"repository": "../x"}, {}])
def test_parse_repository_invalid(bad):
    with pytest.raises(ToolFailure) as e:
        parse_repository(bad)
    assert e.value.code == "INVALID_REPOSITORY"


@pytest.mark.parametrize("bad", ["../etc", "/abs", "a\\b", "a/../b", "x\x00y"])
def test_validate_path_rejects(bad):
    with pytest.raises(ToolFailure) as e:
        validate_path(bad, allow_empty=False)
    assert e.value.code == "INVALID_REPOSITORY_PATH"


def test_validate_path_absolute_rejected():
    with pytest.raises(ToolFailure):
        validate_path("/etc/passwd", allow_empty=False)


# ---- search_repositories ----

def test_search_repositories_success():
    items = {"items": [
        {"full_name": "a/b", "stargazers_count": 100, "archived": True, "language": "Python",
         "html_url": "https://github.com/a/b", "license": {"spdx_id": "MIT"}},
    ]}
    tool = SearchRepositoriesTool(client=FakeClient(_resp(items)))
    data = tool.execute(tool.validate_arguments({"query": "mcp language:Python", "limit": 5, "sort": "stars"}))
    assert data["result_count"] == 1
    assert data["repositories"][0]["full_name"] == "a/b"
    assert data["repositories"][0]["archived"] is True
    assert data["repositories"][0]["license"] == "MIT"
    assert data["untrusted_content"] is True


def test_search_repositories_sort_validation():
    tool = SearchRepositoriesTool(client=FakeClient(_resp({"items": []})))
    with pytest.raises(ToolValidationError):
        tool.validate_arguments({"query": "x", "sort": "downloads"})


def test_search_repositories_limit_enforced(monkeypatch):
    items = {"items": [{"full_name": f"a/{i}"} for i in range(20)]}
    tool = SearchRepositoriesTool(client=FakeClient(_resp(items)))
    data = tool.execute(tool.validate_arguments({"query": "x", "limit": 3}))
    assert data["result_count"] == 3


def test_search_repositories_empty():
    tool = SearchRepositoriesTool(client=FakeClient(_resp({"items": []})))
    data = tool.execute(tool.validate_arguments({"query": "x"}))
    assert data["result_count"] == 0


# ---- get_repository ----

def test_get_repository_success():
    payload = {"full_name": "ollama/ollama", "stargazers_count": 1000, "topics": ["ai"],
               "license": {"spdx_id": "MIT"}, "subscribers_count": 50}
    tool = GetRepositoryTool(client=FakeClient(_resp(payload)))
    data = tool.execute(tool.validate_arguments({"repository": "ollama/ollama"}))
    assert data["full_name"] == "ollama/ollama"
    assert data["stars"] == 1000
    assert data["topics"] == ["ai"]
    assert data["source_type"] == "github_repository"


def test_get_repository_invalid_identifier():
    tool = GetRepositoryTool(client=FakeClient(_resp({})))
    with pytest.raises(ToolFailure) as e:
        tool.validate_arguments({"repository": "not-a-repo"})
    assert e.value.code == "INVALID_REPOSITORY"


def test_get_repository_not_found():
    tool = GetRepositoryTool(client=FakeClient(_resp(None, status=404)))
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"repository": "a/missing"}))
    assert e.value.code == "GITHUB_REPOSITORY_NOT_FOUND"


# ---- read_file ----

def _file_payload(text, **over):
    payload = {
        "type": "file",
        "encoding": "base64",
        "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
        "size": len(text.encode("utf-8")),
        "sha": "abc",
        "html_url": "https://github.com/a/b/blob/main/README.md",
    }
    payload.update(over)
    return payload


def test_read_file_success():
    tool = ReadFileTool(client=FakeClient(_resp(_file_payload("# Hello\nWorld"))))
    data = tool.execute(tool.validate_arguments({"repository": "a/b", "path": "README.md", "ref": "main"}))
    assert data["text"] == "# Hello\nWorld"
    assert data["truncated"] is False
    assert data["source_type"] == "github_file"
    assert data["untrusted_content"] is True


def test_read_file_truncation():
    tool = ReadFileTool(client=FakeClient(_resp(_file_payload("abcdefghij"))))
    data = tool.execute(tool.validate_arguments({"repository": "a/b", "path": "f.txt", "max_chars": 4}))
    assert data["truncated"] is True
    assert data["text"] == "abcd"


def test_read_file_binary_rejected():
    payload = {"type": "file", "encoding": "base64", "size": 4,
               "content": base64.b64encode(b"\x00\x01\x02\x03").decode("ascii")}
    tool = ReadFileTool(client=FakeClient(_resp(payload)))
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"repository": "a/b", "path": "f.bin"}))
    assert e.value.code == "GITHUB_BINARY_FILE"


def test_read_file_too_large(monkeypatch):
    monkeypatch.setenv("GITHUB_MAX_FILE_BYTES", "5")
    tool = ReadFileTool(client=FakeClient(_resp(_file_payload("way too long content"))))
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"repository": "a/b", "path": "f.txt"}))
    assert e.value.code == "GITHUB_FILE_TOO_LARGE"


def test_read_file_path_traversal_rejected():
    tool = ReadFileTool(client=FakeClient(_resp(_file_payload("x"))))
    with pytest.raises(ToolFailure) as e:
        tool.validate_arguments({"repository": "a/b", "path": "../secret"})
    assert e.value.code == "INVALID_REPOSITORY_PATH"


def test_read_file_missing():
    tool = ReadFileTool(client=FakeClient(_resp(None, status=404)))
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"repository": "a/b", "path": "nope.md"}))
    assert e.value.code == "GITHUB_FILE_NOT_FOUND"


def test_read_file_directory_passed_instead():
    tool = ReadFileTool(client=FakeClient(_resp([{"name": "a"}])))  # list => directory
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"repository": "a/b", "path": "src"}))
    assert e.value.code == "INVALID_REPOSITORY_PATH"


# ---- list_directory ----

def test_list_directory_success():
    listing = [
        {"name": "README.md", "path": "README.md", "type": "file", "size": 10, "sha": "s1",
         "html_url": "u1"},
        {"name": "src", "path": "src", "type": "dir", "size": 0, "sha": "s2", "html_url": "u2"},
    ]
    tool = ListDirectoryTool(client=FakeClient(_resp(listing)))
    data = tool.execute(tool.validate_arguments({"repository": "a/b", "path": ""}))
    assert data["entry_count"] == 2
    types = {e["name"]: e["type"] for e in data["entries"]}
    assert types == {"README.md": "file", "src": "directory"}
    assert data["truncated"] is False


def test_list_directory_entry_limit(monkeypatch):
    monkeypatch.setenv("GITHUB_MAX_DIRECTORY_ENTRIES", "2")
    listing = [{"name": str(i), "path": str(i), "type": "file", "size": 1} for i in range(5)]
    tool = ListDirectoryTool(client=FakeClient(_resp(listing)))
    data = tool.execute(tool.validate_arguments({"repository": "a/b"}))
    assert data["entry_count"] == 2
    assert data["truncated"] is True


def test_list_directory_file_passed_instead():
    tool = ListDirectoryTool(client=FakeClient(_resp({"type": "file", "name": "x"})))
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"repository": "a/b", "path": "README.md"}))
    assert e.value.code == "INVALID_REPOSITORY_PATH"


def test_list_directory_missing():
    tool = ListDirectoryTool(client=FakeClient(_resp(None, status=404)))
    with pytest.raises(ToolFailure) as e:
        tool.execute(tool.validate_arguments({"repository": "a/b", "path": "nope"}))
    assert e.value.code == "GITHUB_FILE_NOT_FOUND"


# ---- list_releases ----

def test_list_releases_success():
    releases = [
        {"tag_name": "v1.0", "name": "One", "html_url": "u", "prerelease": False, "draft": False,
         "author": {"login": "bob"}, "body": "notes", "published_at": "2024-01-01T00:00:00Z",
         "assets": [{"name": "app.zip", "size": 100, "download_count": 5, "content_type": "application/zip"}]},
    ]
    tool = ListReleasesTool(client=FakeClient(_resp(releases)))
    data = tool.execute(tool.validate_arguments({"repository": "a/b", "limit": 5}))
    assert data["result_count"] == 1
    rel = data["releases"][0]
    assert rel["tag_name"] == "v1.0"
    assert rel["author"] == "bob"
    # Assets are metadata only — no download url / content.
    assert rel["assets"][0] == {"name": "app.zip", "size_bytes": 100,
                                 "download_count": 5, "content_type": "application/zip"}


def test_list_releases_limit_enforced():
    releases = [{"tag_name": f"v{i}"} for i in range(20)]
    tool = ListReleasesTool(client=FakeClient(_resp(releases)))
    data = tool.execute(tool.validate_arguments({"repository": "a/b", "limit": 3}))
    assert data["result_count"] == 3


def test_list_releases_notes_truncated():
    releases = [{"tag_name": "v1", "body": "x" * 5000}]
    tool = ListReleasesTool(client=FakeClient(_resp(releases)))
    data = tool.execute(tool.validate_arguments({"repository": "a/b"}))
    assert data["releases"][0]["notes_truncated"] is True
    assert len(data["releases"][0]["notes"]) == ListReleasesTool._NOTES_MAX


def test_list_releases_empty():
    tool = ListReleasesTool(client=FakeClient(_resp([])))
    data = tool.execute(tool.validate_arguments({"repository": "a/b"}))
    assert data["result_count"] == 0


def test_list_releases_draft_prerelease_preserved():
    releases = [{"tag_name": "v2", "prerelease": True, "draft": True, "body": ""}]
    tool = ListReleasesTool(client=FakeClient(_resp(releases)))
    data = tool.execute(tool.validate_arguments({"repository": "a/b"}))
    assert data["releases"][0]["prerelease"] is True
    assert data["releases"][0]["draft"] is True


def test_github_tool_result_json_serializable():
    tool = GetRepositoryTool(client=FakeClient(_resp({"full_name": "a/b"})))
    data = tool.execute(tool.validate_arguments({"repository": "a/b"}))
    data.pop("_log_meta", None)
    json.dumps(data)
