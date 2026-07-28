"""Phase 2A GitHub tools (read-only, public REST API).

github.search_repositories, github.get_repository, github.read_file,
github.list_directory, github.list_releases.

All are read-only: no cloning, no writes, no asset downloads, no code execution.
Every result is marked untrusted. Repository identifiers and file paths are
validated before any API call.
"""

import base64
import re

import tools.config as config
from tools.base import BaseTool, ToolFailure, ToolValidationError
from tools.github_client import GitHubClient
from tools.models import (
    GITHUB_BINARY_FILE,
    GITHUB_FILE_NOT_FOUND,
    GITHUB_FILE_TOO_LARGE,
    GITHUB_REPOSITORY_NOT_FOUND,
    INVALID_REPOSITORY,
    INVALID_REPOSITORY_PATH,
)

_OWNER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}$")
_REPO_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_REF_RE = re.compile(r"^[A-Za-z0-9._/-]{1,120}$")
_ALLOWED_SORTS = {"best_match", "stars", "updated"}
_SORT_API = {"best_match": None, "stars": "stars", "updated": "updated"}


def parse_repository(arguments):
    """Extract and validate 'owner/repo' from either {repository} or {owner, repo}."""
    repository = arguments.get("repository")
    owner = arguments.get("owner")
    repo = arguments.get("repo")
    if isinstance(repository, str) and repository.strip():
        parts = repository.strip().split("/")
        if len(parts) != 2:
            raise ToolFailure(INVALID_REPOSITORY, "Repository must be in 'owner/repo' form.")
        owner, repo = parts[0], parts[1]
    if not isinstance(owner, str) or not isinstance(repo, str):
        raise ToolFailure(INVALID_REPOSITORY, "A repository ('owner/repo') is required.")
    owner, repo = owner.strip(), repo.strip()
    if not _OWNER_RE.match(owner) or not _REPO_RE.match(repo):
        raise ToolFailure(INVALID_REPOSITORY, "The repository identifier is invalid.")
    return owner, repo


def validate_ref(ref):
    if ref is None:
        return None
    if not isinstance(ref, str) or not ref.strip():
        raise ToolFailure(INVALID_REPOSITORY, "The ref must be a non-empty string when provided.")
    ref = ref.strip()
    if not _REF_RE.match(ref) or ".." in ref:
        raise ToolFailure(INVALID_REPOSITORY, "The ref contains invalid characters.")
    return ref


def validate_path(path, *, allow_empty):
    """Validate a repository-relative path. Rejects traversal/absolute/control chars."""
    if path is None:
        path = ""
    if not isinstance(path, str):
        raise ToolFailure(INVALID_REPOSITORY_PATH, "The path must be a string.")
    path = path.strip()
    if not path:
        if allow_empty:
            return ""
        raise ToolFailure(INVALID_REPOSITORY_PATH, "A file path is required.")
    if path.startswith("/") or path.startswith("\\"):
        raise ToolFailure(INVALID_REPOSITORY_PATH, "Absolute paths are not allowed.")
    if "\\" in path:
        raise ToolFailure(INVALID_REPOSITORY_PATH, "Backslashes are not allowed in paths.")
    if "\x00" in path or any(ord(c) < 32 for c in path):
        raise ToolFailure(INVALID_REPOSITORY_PATH, "The path contains control characters.")
    segments = path.split("/")
    if any(seg == ".." for seg in segments):
        raise ToolFailure(INVALID_REPOSITORY_PATH, "Path traversal ('..') is not allowed.")
    return path.strip("/")


class _GitHubTool(BaseTool):
    timeout_seconds = 25.0
    requires_internet = True

    def __init__(self, client=None):
        self._client = client

    @property
    def client(self):
        return self._client if self._client is not None else GitHubClient()


class SearchRepositoriesTool(_GitHubTool):
    name = "github.search_repositories"
    description = "Search public GitHub repositories. Returns name, description, stars, language, etc."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "GitHub search query, e.g. 'mcp server language:Python'."},
            "limit": {"type": "integer", "description": "Max repositories (1-10, default 5)."},
            "sort": {"type": "string", "description": "One of: best_match, stars, updated."},
        },
        "required": ["query"],
    }

    def validate_arguments(self, arguments):
        arguments = super().validate_arguments(arguments)
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolValidationError("'query' must be a non-empty string.")
        if len(query) > config.max_search_query_chars():
            raise ToolValidationError("'query' is too long.")
        limit = arguments.get("limit", 5)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ToolValidationError("'limit' must be a positive integer.")
        sort = arguments.get("sort", "best_match")
        if sort not in _ALLOWED_SORTS:
            raise ToolValidationError(f"'sort' must be one of {sorted(_ALLOWED_SORTS)}.")
        return {"query": query.strip(), "limit": min(limit, config.search_max_results()), "sort": sort}

    def execute(self, arguments):
        params = {"q": arguments["query"], "per_page": arguments["limit"]}
        api_sort = _SORT_API[arguments["sort"]]
        if api_sort:
            params["sort"] = api_sort
        resp = self.client.get("/search/repositories", params=params)
        items = (resp.data or {}).get("items", []) if resp.data else []
        repositories = [_map_repo(item) for item in items[:arguments["limit"]] if isinstance(item, dict)]
        return {
            "query": arguments["query"],
            "repositories": repositories,
            "result_count": len(repositories),
            "rate_limit": resp.rate_limit,
            "untrusted_content": True,
            "source_type": "github_search",
            "_log_meta": {"result_count": len(repositories),
                          "rate_limit_remaining": resp.rate_limit.get("remaining")},
        }


class GetRepositoryTool(_GitHubTool):
    name = "github.get_repository"
    description = "Get metadata for one public GitHub repository (stars, language, license, topics, dates)."
    input_schema = {
        "type": "object",
        "properties": {"repository": {"type": "string", "description": "'owner/repo'."}},
        "required": ["repository"],
    }

    def validate_arguments(self, arguments):
        arguments = super().validate_arguments(arguments)
        owner, repo = parse_repository(arguments)
        return {"owner": owner, "repo": repo}

    def execute(self, arguments):
        owner, repo = arguments["owner"], arguments["repo"]
        resp = self.client.get(f"/repos/{owner}/{repo}")
        if resp.status_code == 404 or not resp.data:
            raise ToolFailure(GITHUB_REPOSITORY_NOT_FOUND, f"Repository '{owner}/{repo}' was not found.")
        data = _map_repo(resp.data, full=True)
        data.update({
            "untrusted_content": True,
            "source_type": "github_repository",
            "rate_limit": resp.rate_limit,
            "_log_meta": {"rate_limit_remaining": resp.rate_limit.get("remaining")},
        })
        return data


class ReadFileTool(_GitHubTool):
    name = "github.read_file"
    description = "Read a public text file from a GitHub repo via the Contents API. Text only; no binaries."
    input_schema = {
        "type": "object",
        "properties": {
            "repository": {"type": "string", "description": "'owner/repo'."},
            "path": {"type": "string", "description": "File path within the repo, e.g. 'README.md'."},
            "ref": {"type": "string", "description": "Optional branch/tag/commit."},
            "max_chars": {"type": "integer", "description": "Max characters of text to return."},
        },
        "required": ["repository", "path"],
    }

    def validate_arguments(self, arguments):
        arguments = super().validate_arguments(arguments)
        owner, repo = parse_repository(arguments)
        path = validate_path(arguments.get("path"), allow_empty=False)
        ref = validate_ref(arguments.get("ref"))
        max_chars = arguments.get("max_chars", config.github_max_file_chars())
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
            raise ToolValidationError("'max_chars' must be a positive integer.")
        return {"owner": owner, "repo": repo, "path": path, "ref": ref,
                "max_chars": min(max_chars, config.github_max_file_chars())}

    def execute(self, arguments):
        owner, repo, path = arguments["owner"], arguments["repo"], arguments["path"]
        params = {"ref": arguments["ref"]} if arguments["ref"] else None
        resp = self.client.get(f"/repos/{owner}/{repo}/contents/{path}", params=params)
        if resp.status_code == 404 or resp.data is None:
            raise ToolFailure(GITHUB_FILE_NOT_FOUND, f"File '{path}' was not found in '{owner}/{repo}'.")
        item = resp.data
        if isinstance(item, list):
            raise ToolFailure(INVALID_REPOSITORY_PATH, f"'{path}' is a directory, not a file.")
        if not isinstance(item, dict) or item.get("type") != "file":
            raise ToolFailure(GITHUB_FILE_NOT_FOUND, f"'{path}' is not a readable file.")

        size = item.get("size") or 0
        if size > config.github_max_file_bytes():
            raise ToolFailure(GITHUB_FILE_TOO_LARGE,
                              f"The file is too large to read ({size} bytes).")

        content_b64 = item.get("content")
        if item.get("encoding") != "base64" or not content_b64:
            # Files >1MB return no inline content via the Contents API.
            raise ToolFailure(GITHUB_FILE_TOO_LARGE, "The file is too large to read inline.")
        try:
            raw = base64.b64decode(content_b64)
        except (ValueError, TypeError):
            raise ToolFailure(GITHUB_BINARY_FILE, "The file content could not be decoded.")
        if b"\x00" in raw:
            raise ToolFailure(GITHUB_BINARY_FILE, "The file appears to be binary, not text.")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolFailure(GITHUB_BINARY_FILE, "The file is not valid UTF-8 text.")

        max_chars = arguments["max_chars"]
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]
        return {
            "repository": f"{owner}/{repo}",
            "path": path,
            "ref": arguments["ref"],
            "sha": item.get("sha"),
            "size_bytes": size,
            "encoding": "utf-8",
            "text": text,
            "truncated": truncated,
            "html_url": item.get("html_url"),
            "untrusted_content": True,
            "source_type": "github_file",
            "_log_meta": {"size_bytes": size, "truncated": truncated,
                          "rate_limit_remaining": resp.rate_limit.get("remaining")},
        }


class ListDirectoryTool(_GitHubTool):
    name = "github.list_directory"
    description = "List files/subdirectories at a path in a public GitHub repo (non-recursive)."
    input_schema = {
        "type": "object",
        "properties": {
            "repository": {"type": "string", "description": "'owner/repo'."},
            "path": {"type": "string", "description": "Directory path (empty for repo root)."},
            "ref": {"type": "string", "description": "Optional branch/tag/commit."},
        },
        "required": ["repository"],
    }

    def validate_arguments(self, arguments):
        arguments = super().validate_arguments(arguments)
        owner, repo = parse_repository(arguments)
        path = validate_path(arguments.get("path"), allow_empty=True)
        ref = validate_ref(arguments.get("ref"))
        return {"owner": owner, "repo": repo, "path": path, "ref": ref}

    def execute(self, arguments):
        owner, repo, path = arguments["owner"], arguments["repo"], arguments["path"]
        params = {"ref": arguments["ref"]} if arguments["ref"] else None
        resp = self.client.get(f"/repos/{owner}/{repo}/contents/{path}", params=params)
        if resp.status_code == 404 or resp.data is None:
            raise ToolFailure(GITHUB_FILE_NOT_FOUND, f"Directory '{path}' was not found in '{owner}/{repo}'.")
        items = resp.data
        if isinstance(items, dict):
            raise ToolFailure(INVALID_REPOSITORY_PATH, f"'{path}' is a file, not a directory.")
        if not isinstance(items, list):
            raise ToolFailure(GITHUB_FILE_NOT_FOUND, "The directory listing was invalid.")

        cap = config.github_max_directory_entries()
        entries = []
        for item in items[:cap]:
            if not isinstance(item, dict):
                continue
            entries.append({
                "name": item.get("name"),
                "path": item.get("path"),
                "type": "directory" if item.get("type") == "dir" else item.get("type"),
                "size_bytes": item.get("size", 0),
                "sha": item.get("sha"),
                "html_url": item.get("html_url"),
            })
        return {
            "repository": f"{owner}/{repo}",
            "path": path,
            "ref": arguments["ref"],
            "entries": entries,
            "entry_count": len(entries),
            "truncated": len(items) > cap,
            "untrusted_content": True,
            "source_type": "github_directory",
            "_log_meta": {"entry_count": len(entries),
                          "rate_limit_remaining": resp.rate_limit.get("remaining")},
        }


class ListReleasesTool(_GitHubTool):
    name = "github.list_releases"
    description = "List recent public releases for a GitHub repo (tags, notes, assets as metadata only)."
    input_schema = {
        "type": "object",
        "properties": {
            "repository": {"type": "string", "description": "'owner/repo'."},
            "limit": {"type": "integer", "description": "Max releases (1-10, default 5)."},
        },
        "required": ["repository"],
    }
    _NOTES_MAX = 2000

    def validate_arguments(self, arguments):
        arguments = super().validate_arguments(arguments)
        owner, repo = parse_repository(arguments)
        limit = arguments.get("limit", 5)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ToolValidationError("'limit' must be a positive integer.")
        return {"owner": owner, "repo": repo, "limit": min(limit, config.github_max_releases())}

    def execute(self, arguments):
        owner, repo, limit = arguments["owner"], arguments["repo"], arguments["limit"]
        resp = self.client.get(f"/repos/{owner}/{repo}/releases", params={"per_page": limit})
        if resp.status_code == 404 or resp.data is None:
            raise ToolFailure(GITHUB_REPOSITORY_NOT_FOUND, f"Repository '{owner}/{repo}' was not found.")
        items = resp.data if isinstance(resp.data, list) else []
        releases = []
        for item in items[:limit]:
            if not isinstance(item, dict):
                continue
            body = item.get("body") or ""
            notes = body[:self._NOTES_MAX]
            assets = [{
                "name": a.get("name"),
                "size_bytes": a.get("size"),
                "download_count": a.get("download_count"),
                "content_type": a.get("content_type"),
            } for a in (item.get("assets") or []) if isinstance(a, dict)]
            releases.append({
                "tag_name": item.get("tag_name"),
                "name": item.get("name"),
                "html_url": item.get("html_url"),
                "published_at": item.get("published_at"),
                "created_at": item.get("created_at"),
                "prerelease": item.get("prerelease", False),
                "draft": item.get("draft", False),
                "author": (item.get("author") or {}).get("login"),
                "notes": notes,
                "notes_truncated": len(body) > self._NOTES_MAX,
                "assets": assets,
            })
        return {
            "repository": f"{owner}/{repo}",
            "releases": releases,
            "result_count": len(releases),
            "rate_limit": resp.rate_limit,
            "untrusted_content": True,
            "source_type": "github_releases",
            "_log_meta": {"result_count": len(releases),
                          "rate_limit_remaining": resp.rate_limit.get("remaining")},
        }


def _map_repo(item, full=False):
    """Map a raw GitHub repo payload to a bounded, safe subset."""
    license_obj = item.get("license") or {}
    mapped = {
        "full_name": item.get("full_name"),
        "description": item.get("description"),
        "html_url": item.get("html_url"),
        "stars": item.get("stargazers_count"),
        "forks": item.get("forks_count"),
        "open_issues": item.get("open_issues_count"),
        "language": item.get("language"),
        "default_branch": item.get("default_branch"),
        "archived": item.get("archived", False),
        "fork": item.get("fork", False),
        "visibility": item.get("visibility", "public"),
        "updated_at": item.get("updated_at"),
        "license": license_obj.get("spdx_id") or license_obj.get("name"),
    }
    if full:
        mapped.update({
            "homepage": item.get("homepage"),
            "watchers": item.get("subscribers_count", item.get("watchers_count")),
            "topics": item.get("topics", []),
            "created_at": item.get("created_at"),
            "pushed_at": item.get("pushed_at"),
            "size": item.get("size"),
        })
    return mapped


ALL_GITHUB_TOOL_CLASSES = [
    SearchRepositoriesTool,
    GetRepositoryTool,
    ReadFileTool,
    ListDirectoryTool,
    ListReleasesTool,
]
