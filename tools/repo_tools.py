"""Phase 2B repo.* tools: static inspection of a previously cloned repository.

All are read-only and require the repository.read capability. Every result is
marked untrusted. Nothing from the repository is executed, imported, or installed.
"""

import os

import tools.config as config
from tools import repo_analysis, repo_security, repo_store
from tools.base import BaseTool, ToolFailure, ToolValidationError
from tools.models import (
    REPOSITORY_BINARY_FILE,
    REPOSITORY_FILE_NOT_FOUND,
    REPOSITORY_FILE_TOO_LARGE,
    REPOSITORY_INSPECTION_FAILED,
    REPOSITORY_SECURITY_SCAN_FAILED,
    REPOSITORY_SYMLINK_BLOCKED,
    INVALID_REPOSITORY_PATH,
)


class _RepoTool(BaseTool):
    timeout_seconds = 30.0
    required_capabilities = ("repository.read",)

    def _repo(self, arguments):
        repository = arguments.get("repository")
        if not isinstance(repository, str) or not repository.strip():
            raise ToolValidationError("'repository' ('owner/repo') is required.")
        return repo_store.require_cloned_repo(repository)


class ListFilesTool(_RepoTool):
    name = "repo.list_files"
    description = ("List files/directories in a cloned repository (bounded, non-recursive by "
                  "default). Symlinks are marked and never followed. Does not read contents.")
    input_schema = {
        "type": "object",
        "properties": {
            "repository": {"type": "string", "description": "'owner/repo' (already cloned)."},
            "path": {"type": "string", "description": "Relative directory path (empty = repo root)."},
            "recursive": {"type": "boolean", "description": "Recurse into subdirectories (bounded)."},
            "max_depth": {"type": "integer", "description": "Max recursion depth."},
            "limit": {"type": "integer", "description": "Max entries to return."},
        },
        "required": ["repository"],
    }

    def execute(self, arguments):
        owner, repo, repo_dir = self._repo(arguments)
        base_dir, rel = repo_store.resolve_within(repo_dir, arguments.get("path", ""), allow_empty=True)
        if not os.path.isdir(base_dir):
            raise ToolFailure(REPOSITORY_FILE_NOT_FOUND, f"Directory '{rel or ''}' was not found.")
        recursive = bool(arguments.get("recursive", False))
        max_depth = min(int(arguments.get("max_depth", 2) or 2), config.repo_max_list_depth()) if recursive else 1
        limit = min(int(arguments.get("limit", 100) or 100), config.repo_max_list_entries())

        entries = []
        truncated = False
        base_depth = base_dir.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(base_dir, followlinks=False):
            depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
            if depth >= max_depth:
                dirnames[:] = []
            dirnames[:] = [d for d in sorted(dirnames) if d != ".git"]
            for name in sorted(dirnames) + sorted(filenames):
                abs_path = os.path.join(dirpath, name)
                rel_path = os.path.relpath(abs_path, repo_dir).replace(os.sep, "/")
                if repo_store.is_symlink(abs_path):
                    entries.append({"name": name, "path": rel_path, "type": "symlink", "followed": False})
                elif os.path.isdir(abs_path):
                    entries.append({"name": name, "path": rel_path, "type": "directory", "size_bytes": None})
                else:
                    try:
                        size = os.lstat(abs_path).st_size
                    except OSError:
                        size = None
                    entries.append({"name": name, "path": rel_path, "type": "file", "size_bytes": size})
                if len(entries) >= limit:
                    truncated = True
                    break
            if truncated or not recursive:
                break

        entries.sort(key=lambda e: e["path"])
        return {
            "repository": f"{owner}/{repo}",
            "path": rel or "",
            "recursive": recursive,
            "entries": entries[:limit],
            "entry_count": len(entries[:limit]),
            "truncated": truncated,
            "untrusted_content": True,
            "executed": False,
            "_log_meta": {"repository": f"{owner}/{repo}", "entry_count": len(entries[:limit]),
                          "truncated": truncated},
        }


class ReadFileTool(_RepoTool):
    name = "repo.read_file"
    description = ("Read a bounded text file from a cloned repository. Rejects symlinks, "
                  "directories, binaries, and path traversal. Content is untrusted data.")
    input_schema = {
        "type": "object",
        "properties": {
            "repository": {"type": "string", "description": "'owner/repo' (already cloned)."},
            "path": {"type": "string", "description": "Relative file path, e.g. 'README.md'."},
            "max_chars": {"type": "integer", "description": "Max characters to return."},
        },
        "required": ["repository", "path"],
    }

    def execute(self, arguments):
        owner, repo, repo_dir = self._repo(arguments)
        abs_path, rel = repo_store.resolve_within(repo_dir, arguments.get("path"), allow_empty=False)
        if repo_store.is_symlink(abs_path):
            raise ToolFailure(REPOSITORY_SYMLINK_BLOCKED, f"'{rel}' is a symlink and cannot be read.")
        if os.path.isdir(abs_path):
            raise ToolFailure(INVALID_REPOSITORY_PATH, f"'{rel}' is a directory, not a file.")
        if not os.path.isfile(abs_path):
            raise ToolFailure(REPOSITORY_FILE_NOT_FOUND, f"File '{rel}' was not found.")

        size = os.path.getsize(abs_path)
        if size > config.repo_max_read_bytes():
            raise ToolFailure(REPOSITORY_FILE_TOO_LARGE, f"The file is too large to read ({size} bytes).")
        with open(abs_path, "rb") as f:
            raw = f.read(config.repo_max_read_bytes())
        if b"\x00" in raw:
            raise ToolFailure(REPOSITORY_BINARY_FILE, "The file appears to be binary, not text.")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("latin-1")
            except UnicodeDecodeError:
                raise ToolFailure(REPOSITORY_BINARY_FILE, "The file could not be decoded as text.")

        max_chars = min(int(arguments.get("max_chars", config.repo_max_read_chars()) or
                            config.repo_max_read_chars()), config.repo_max_read_chars())
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]
        return {
            "repository": f"{owner}/{repo}",
            "path": rel,
            "size_bytes": size,
            "encoding": "utf-8",
            "text": text,
            "truncated": truncated,
            "untrusted_content": True,
            "source_type": "cloned_repository_file",
            "executed": False,
            "_log_meta": {"repository": f"{owner}/{repo}", "size_bytes": size, "truncated": truncated},
        }


class InspectTool(_RepoTool):
    name = "repo.inspect"
    description = ("Static structural summary of a cloned repository: languages, manifests, "
                  "dependencies, docs/tests, likely entry points and integration type. No execution.")
    input_schema = {
        "type": "object",
        "properties": {"repository": {"type": "string", "description": "'owner/repo' (already cloned)."}},
        "required": ["repository"],
    }
    timeout_seconds = 60.0

    def execute(self, arguments):
        owner, repo, repo_dir = self._repo(arguments)
        try:
            result = repo_analysis.analyze(repo_dir)
        except Exception as e:  # noqa: BLE001
            raise ToolFailure(REPOSITORY_INSPECTION_FAILED, f"Inspection failed ({type(e).__name__}).")
        return {
            "repository": f"{owner}/{repo}",
            "summary": result["observed"],
            "inferred": result["inferred"],
            "limitations": result["limitations"],
            "untrusted_content": True,
            "executed": False,
            "_log_meta": {"repository": f"{owner}/{repo}",
                          "file_count": result["observed"].get("file_count")},
        }


class SecurityScanTool(_RepoTool):
    name = "repo.security_scan"
    description = ("Bounded static security pattern scan of a cloned repository (Python AST + text "
                  "patterns). Identifies code needing human review. Never proves safety.")
    input_schema = {
        "type": "object",
        "properties": {"repository": {"type": "string", "description": "'owner/repo' (already cloned)."}},
        "required": ["repository"],
    }
    timeout_seconds = 60.0

    def execute(self, arguments):
        owner, repo, repo_dir = self._repo(arguments)
        try:
            result = repo_security.scan(repo_dir)
        except Exception as e:  # noqa: BLE001
            raise ToolFailure(REPOSITORY_SECURITY_SCAN_FAILED, f"Security scan failed ({type(e).__name__}).")
        return {
            "repository": f"{owner}/{repo}",
            "risk_summary": result["risk_summary"],
            "findings": result["findings"],
            "truncated": result["truncated"],
            "limitations": result["limitations"],
            "untrusted_content": True,
            "executed": False,
            "_log_meta": {"repository": f"{owner}/{repo}",
                          "finding_count": len(result["findings"]), "truncated": result["truncated"]},
        }


class CapabilityReportTool(_RepoTool):
    name = "repo.capability_report"
    description = ("Combine static inspection + security scan into a bounded integration-readiness "
                  "report. Separates observed facts from inference. Never approves installation.")
    input_schema = {
        "type": "object",
        "properties": {"repository": {"type": "string", "description": "'owner/repo' (already cloned)."}},
        "required": ["repository"],
    }
    timeout_seconds = 90.0

    def execute(self, arguments):
        owner, repo, repo_dir = self._repo(arguments)
        try:
            analysis = repo_analysis.analyze(repo_dir)
            scan = repo_security.scan(repo_dir)
        except Exception as e:  # noqa: BLE001
            raise ToolFailure(REPOSITORY_INSPECTION_FAILED, f"Capability report failed ({type(e).__name__}).")

        observed = analysis["observed"]
        integrations = analysis["inferred"]["integration_indicators"]
        deps = observed.get("dependencies", {})
        risk = scan["risk_summary"]
        high = risk.get("high", 0)
        critical = risk.get("critical", 0)

        # Requirement inference (facts where available, else unknown).
        api_keys = _infer_api_keys(observed, repo_dir)
        network = any(i["type"] in ("possible_rest_service", "possible_mcp_server") for i in integrations) \
            or bool(set(deps.get("python", []) + deps.get("node", [])) &
                    {"requests", "httpx", "aiohttp", "urllib3", "axios", "node-fetch"})

        if critical or high:
            status, reason = "manual_review_required", "Static inspection found higher-risk patterns needing review."
        elif not integrations and not observed.get("manifests"):
            status, reason = "insufficient_information", "Not enough static signal to classify the repository."
        else:
            status, reason = "static_review_complete", "Static inspection completed; no execution was performed."

        return {
            "repository": f"{owner}/{repo}",
            "observed": {
                "languages": [l["name"] for l in observed.get("languages", [])],
                "manifests": observed.get("manifests", []),
                "package_managers": observed.get("package_managers", []),
                "documentation": observed.get("documentation", []),
                "license": observed.get("license"),
            },
            "possible_integrations": integrations,
            "requirements": {
                "runtime": _infer_runtime(observed),
                "dependencies_sample": (deps.get("python", []) + deps.get("node", []))[:20],
                "api_keys": api_keys,
                "network": network,
                "filesystem": "unknown",
                "subprocess": any(f["category"].startswith("process_execution") for f in scan["findings"]),
            },
            "security_summary": {
                "requires_review": bool(high or critical),
                "high_findings": high,
                "critical_findings": critical,
            },
            "recommendation": {"status": status, "reason": reason},
            "limitations": [
                "No dependencies were installed",
                "No code was executed",
                "No live integration was tested",
                "This is not a security approval",
            ],
            "untrusted_content": True,
            "executed": False,
            "_log_meta": {"repository": f"{owner}/{repo}", "recommendation": status},
        }


def _infer_runtime(observed):
    runtime = []
    langs = {l["name"] for l in observed.get("languages", [])}
    if "Python" in langs:
        runtime.append("Python")
    if "JavaScript" in langs or "TypeScript" in langs:
        runtime.append("Node.js")
    if "Go" in langs:
        runtime.append("Go")
    if "Rust" in langs:
        runtime.append("Rust")
    return runtime or ["unknown"]


def _infer_api_keys(observed, repo_dir):
    # Presence of an env example is a static signal that keys may be required.
    for doc in observed.get("documentation", []):
        pass
    signals = []
    for name in (".env.example", ".env.sample", ".env.template"):
        if os.path.isfile(os.path.join(repo_dir, name)):
            signals.append(f"Environment example present ({name}) — API keys may be required")
            break
    return signals


ALL_REPO_TOOL_CLASSES = [
    ListFilesTool,
    ReadFileTool,
    InspectTool,
    SecurityScanTool,
    CapabilityReportTool,
]
