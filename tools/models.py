"""Minimal Phase 1 tool data models.

Plain dataclasses (no new dependency — pydantic is only a transitive dep here).
Everything a tool returns must be JSON-serializable: never exceptions, stack
traces, tokens, headers, or other non-serializable Python values.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


# ---- Controlled error codes (the only ones the executor/loop may emit) ----
# Phase 1
UNKNOWN_TOOL = "UNKNOWN_TOOL"
TOOL_DISABLED = "TOOL_DISABLED"
INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
TOOL_TIMEOUT = "TOOL_TIMEOUT"
TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
INVALID_TOOL_OUTPUT = "INVALID_TOOL_OUTPUT"
MALFORMED_TOOL_CALL = "MALFORMED_TOOL_CALL"
TOOL_STEP_LIMIT_REACHED = "TOOL_STEP_LIMIT_REACHED"

# Phase 2A — permission / internet
INTERNET_DISABLED = "INTERNET_DISABLED"

# Phase 2A — web search
SEARCH_API_KEY_MISSING = "SEARCH_API_KEY_MISSING"
SEARCH_AUTHENTICATION_FAILED = "SEARCH_AUTHENTICATION_FAILED"
SEARCH_RATE_LIMITED = "SEARCH_RATE_LIMITED"
SEARCH_PROVIDER_ERROR = "SEARCH_PROVIDER_ERROR"

# Phase 2A — page fetch / SSRF
INVALID_URL = "INVALID_URL"
UNSUPPORTED_URL_SCHEME = "UNSUPPORTED_URL_SCHEME"
PRIVATE_NETWORK_BLOCKED = "PRIVATE_NETWORK_BLOCKED"
REDIRECT_BLOCKED = "REDIRECT_BLOCKED"
TOO_MANY_REDIRECTS = "TOO_MANY_REDIRECTS"
FETCH_TIMEOUT = "FETCH_TIMEOUT"
RESPONSE_TOO_LARGE = "RESPONSE_TOO_LARGE"
UNSUPPORTED_CONTENT_TYPE = "UNSUPPORTED_CONTENT_TYPE"
INVALID_RESPONSE = "INVALID_RESPONSE"

# Phase 2A — GitHub
GITHUB_AUTHENTICATION_FAILED = "GITHUB_AUTHENTICATION_FAILED"
GITHUB_RATE_LIMITED = "GITHUB_RATE_LIMITED"
GITHUB_REPOSITORY_NOT_FOUND = "GITHUB_REPOSITORY_NOT_FOUND"
GITHUB_FILE_NOT_FOUND = "GITHUB_FILE_NOT_FOUND"
GITHUB_FILE_TOO_LARGE = "GITHUB_FILE_TOO_LARGE"
GITHUB_BINARY_FILE = "GITHUB_BINARY_FILE"
GITHUB_API_ERROR = "GITHUB_API_ERROR"
INVALID_REPOSITORY = "INVALID_REPOSITORY"
INVALID_REPOSITORY_PATH = "INVALID_REPOSITORY_PATH"

# Phase 2B — clone + static repository inspection
REPOSITORY_CLONE_DISABLED = "REPOSITORY_CLONE_DISABLED"
REPOSITORY_INSPECTION_DISABLED = "REPOSITORY_INSPECTION_DISABLED"
PRIVATE_REPOSITORY_NOT_SUPPORTED = "PRIVATE_REPOSITORY_NOT_SUPPORTED"
INVALID_REPOSITORY_REF = "INVALID_REPOSITORY_REF"
REPOSITORY_TOO_LARGE = "REPOSITORY_TOO_LARGE"
REPOSITORY_FILE_LIMIT_EXCEEDED = "REPOSITORY_FILE_LIMIT_EXCEEDED"
REPOSITORY_ALREADY_CLONED = "REPOSITORY_ALREADY_CLONED"
REPOSITORY_NOT_CLONED = "REPOSITORY_NOT_CLONED"
REPOSITORY_PATH_ESCAPE = "REPOSITORY_PATH_ESCAPE"
REPOSITORY_SYMLINK_BLOCKED = "REPOSITORY_SYMLINK_BLOCKED"
REPOSITORY_FILE_NOT_FOUND = "REPOSITORY_FILE_NOT_FOUND"
REPOSITORY_BINARY_FILE = "REPOSITORY_BINARY_FILE"
REPOSITORY_FILE_TOO_LARGE = "REPOSITORY_FILE_TOO_LARGE"
REPOSITORY_SCAN_LIMIT_REACHED = "REPOSITORY_SCAN_LIMIT_REACHED"
GIT_NOT_AVAILABLE = "GIT_NOT_AVAILABLE"
GIT_CLONE_TIMEOUT = "GIT_CLONE_TIMEOUT"
GIT_CLONE_FAILED = "GIT_CLONE_FAILED"
GIT_COMMIT_LOOKUP_FAILED = "GIT_COMMIT_LOOKUP_FAILED"
REPOSITORY_STAGING_CLEANUP_FAILED = "REPOSITORY_STAGING_CLEANUP_FAILED"
REPOSITORY_INSPECTION_FAILED = "REPOSITORY_INSPECTION_FAILED"
REPOSITORY_SECURITY_SCAN_FAILED = "REPOSITORY_SECURITY_SCAN_FAILED"


@dataclass(frozen=True)
class ToolDefinition:
    """What the local LLM is told about a tool."""

    name: str
    description: str
    input_schema: dict
    timeout_seconds: float = 10.0
    enabled: bool = True

    def to_ollama_schema(self) -> dict:
        """Ollama /api/chat `tools` entry — same shape router.py already uses."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True)
class ToolCall:
    """A single tool request extracted from the assistant message."""

    call_id: str
    tool_name: str
    arguments: dict


@dataclass(frozen=True)
class ToolError:
    code: str
    message: str
    retryable: bool = False

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


@dataclass
class ToolResult:
    success: bool
    tool_name: str
    call_id: str
    data: dict = field(default_factory=dict)
    error: Optional[ToolError] = None
    execution_time_ms: Optional[float] = None
    # Safe, non-content metadata for interaction_log only (bytes_read, result_count,
    # http_status_category, rate-limit remaining, ...). NEVER serialized to the model
    # and must never contain secrets or raw content.
    log_meta: Optional[dict] = None

    def to_provider_dict(self) -> dict:
        """The dict serialized into the provider-facing `tool` message content.

        Matches the documented Phase 1 shape: success, tool_name, call_id, data,
        error (None or {code, message}). execution_time_ms and log_meta are kept
        off the wire — they are used for logging and the return value only.
        """
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "data": self.data,
            "error": self.error.to_dict() if self.error else None,
        }

    def to_provider_json(self) -> str:
        import json

        return json.dumps(self.to_provider_dict())

    @classmethod
    def ok(cls, tool_name: str, call_id: str, data: dict, execution_time_ms: Optional[float] = None,
           log_meta: Optional[dict] = None) -> "ToolResult":
        return cls(True, tool_name, call_id, data=data, error=None,
                   execution_time_ms=execution_time_ms, log_meta=log_meta)

    @classmethod
    def fail(
        cls,
        tool_name: str,
        call_id: str,
        code: str,
        message: str,
        execution_time_ms: Optional[float] = None,
        retryable: bool = False,
        log_meta: Optional[dict] = None,
    ) -> "ToolResult":
        return cls(
            False,
            tool_name,
            call_id,
            data={},
            error=ToolError(code, message, retryable),
            execution_time_ms=execution_time_ms,
            log_meta=log_meta,
        )
