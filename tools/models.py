"""Minimal Phase 1 tool data models.

Plain dataclasses (no new dependency — pydantic is only a transitive dep here).
Everything a tool returns must be JSON-serializable: never exceptions, stack
traces, tokens, headers, or other non-serializable Python values.
"""

import hashlib
import json as _json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---- Phase C: tool permission model ----
class ToolPermission(str, Enum):
    """Every tool's effective permission. Fail closed: anything unrecognized -> DENIED."""

    READ = "read"      # auto-execute
    WRITE = "write"    # require explicit one-turn user confirmation
    DENIED = "denied"  # never execute

    @classmethod
    def coerce(cls, value) -> "ToolPermission":
        """Map any value to a permission, defaulting unknown/missing to DENIED."""
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except (ValueError, TypeError):
            return cls.DENIED


def hash_arguments(arguments) -> str:
    """Stable hash of tool arguments, so a confirmation binds to the exact call.

    Deterministic (sorted keys); non-JSON values fall back to their string form.
    """
    serialized = _json.dumps(arguments, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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
# Phase G.1 hotfix — an MCP tool call whose name was never in the shortlist
# actually offered to the model this round (tool_loop.py). Distinct from
# UNKNOWN_TOOL (not registered at all): this can also reject a REGISTERED
# tool the model was simply never offered.
TOOL_NOT_IN_SHORTLIST = "TOOL_NOT_IN_SHORTLIST"

# Phase C — permission / confirmation
TOOL_PERMISSION_DENIED = "TOOL_PERMISSION_DENIED"
TOOL_PERMISSION_INVALID = "TOOL_PERMISSION_INVALID"
TOOL_CONFIRMATION_REQUIRED = "TOOL_CONFIRMATION_REQUIRED"
TOOL_CONFIRMATION_DECLINED = "TOOL_CONFIRMATION_DECLINED"
TOOL_CONFIRMATION_MISMATCH = "TOOL_CONFIRMATION_MISMATCH"

# Phase D — MCP (Model Context Protocol) layer. These surface as ToolResult error
# codes: an MCP tool's execute() raises ToolFailure(<code>) and the existing
# executor normalizes it — the executor itself needs no MCP awareness.
MCP_STARTUP_FAILED = "MCP_STARTUP_FAILED"
MCP_TIMEOUT = "MCP_TIMEOUT"
MCP_TOOL_NOT_FOUND = "MCP_TOOL_NOT_FOUND"
MCP_SERVER_EXITED = "MCP_SERVER_EXITED"
MCP_CALL_FAILED = "MCP_CALL_FAILED"
MCP_INVALID_RESPONSE = "MCP_INVALID_RESPONSE"

# Phase E — external single-server MCP configuration
MCP_CONFIGURATION_INVALID = "MCP_CONFIGURATION_INVALID"
MCP_DISABLED = "MCP_DISABLED"
MCP_EXECUTABLE_NOT_FOUND = "MCP_EXECUTABLE_NOT_FOUND"
MCP_WORKING_DIRECTORY_INVALID = "MCP_WORKING_DIRECTORY_INVALID"
MCP_ENVIRONMENT_INVALID = "MCP_ENVIRONMENT_INVALID"
MCP_INITIALIZATION_FAILED = "MCP_INITIALIZATION_FAILED"
MCP_DISCOVERY_FAILED = "MCP_DISCOVERY_FAILED"
MCP_TOOL_DENIED = "MCP_TOOL_DENIED"
MCP_SHUTDOWN_FAILED = "MCP_SHUTDOWN_FAILED"

# Phase F — automatic MCP provisioning (trusted catalog, plan, approval, install)
MCP_CAPABILITY_UNAVAILABLE = "MCP_CAPABILITY_UNAVAILABLE"
MCP_SERVER_NOT_APPROVED = "MCP_SERVER_NOT_APPROVED"
MCP_CATALOG_INVALID = "MCP_CATALOG_INVALID"
MCP_CATALOG_ENTRY_INVALID = "MCP_CATALOG_ENTRY_INVALID"
MCP_RUNTIME_MISSING = "MCP_RUNTIME_MISSING"
MCP_PROVISIONING_PLAN_INVALID = "MCP_PROVISIONING_PLAN_INVALID"
MCP_PROVISIONING_CONFIRMATION_REQUIRED = "MCP_PROVISIONING_CONFIRMATION_REQUIRED"
MCP_PROVISIONING_CONFIRMATION_MISMATCH = "MCP_PROVISIONING_CONFIRMATION_MISMATCH"
MCP_PROVISIONING_DECLINED = "MCP_PROVISIONING_DECLINED"
MCP_INSTALLATION_FAILED = "MCP_INSTALLATION_FAILED"
MCP_INSTALLATION_TIMEOUT = "MCP_INSTALLATION_TIMEOUT"
MCP_PACKAGE_INTEGRITY_FAILED = "MCP_PACKAGE_INTEGRITY_FAILED"
MCP_ENTRYPOINT_NOT_FOUND = "MCP_ENTRYPOINT_NOT_FOUND"
MCP_CONFIGURATION_GENERATION_FAILED = "MCP_CONFIGURATION_GENERATION_FAILED"
MCP_POST_INSTALL_VALIDATION_FAILED = "MCP_POST_INSTALL_VALIDATION_FAILED"
MCP_ACTIVATION_FAILED = "MCP_ACTIVATION_FAILED"
MCP_ALREADY_INSTALLED = "MCP_ALREADY_INSTALLED"
MCP_NOT_INSTALLED = "MCP_NOT_INSTALLED"
MCP_REPAIR_FAILED = "MCP_REPAIR_FAILED"
MCP_UNINSTALL_FAILED = "MCP_UNINSTALL_FAILED"
MCP_UPDATE_AVAILABLE = "MCP_UPDATE_AVAILABLE"
MCP_PENDING_REQUEST_EXPIRED = "MCP_PENDING_REQUEST_EXPIRED"
MCP_PROVISIONING_LOOP_PREVENTED = "MCP_PROVISIONING_LOOP_PREVENTED"
MCP_REGISTRY_CORRUPT = "MCP_REGISTRY_CORRUPT"
MCP_DIRECTORY_NOT_APPROVED = "MCP_DIRECTORY_NOT_APPROVED"

# Phase F.1 — expand an already-installed server's approved filesystem roots
MCP_FILESYSTEM_ACCESS_REQUIRED = "MCP_FILESYSTEM_ACCESS_REQUIRED"
MCP_FILESYSTEM_ACCESS_PLAN_INVALID = "MCP_FILESYSTEM_ACCESS_PLAN_INVALID"
MCP_FILESYSTEM_ACCESS_CONFIRMATION_REQUIRED = "MCP_FILESYSTEM_ACCESS_CONFIRMATION_REQUIRED"
MCP_FILESYSTEM_ACCESS_CONFIRMATION_MISMATCH = "MCP_FILESYSTEM_ACCESS_CONFIRMATION_MISMATCH"
MCP_FILESYSTEM_ACCESS_DECLINED = "MCP_FILESYSTEM_ACCESS_DECLINED"
MCP_FILESYSTEM_ACCESS_EXPIRED = "MCP_FILESYSTEM_ACCESS_EXPIRED"
MCP_FILESYSTEM_ACCESS_PATH_INVALID = "MCP_FILESYSTEM_ACCESS_PATH_INVALID"
MCP_FILESYSTEM_ACCESS_PATH_RESTRICTED = "MCP_FILESYSTEM_ACCESS_PATH_RESTRICTED"
MCP_FILESYSTEM_ACCESS_ALREADY_GRANTED = "MCP_FILESYSTEM_ACCESS_ALREADY_GRANTED"
MCP_FILESYSTEM_ACCESS_UPDATE_FAILED = "MCP_FILESYSTEM_ACCESS_UPDATE_FAILED"
MCP_FILESYSTEM_ACCESS_VALIDATION_FAILED = "MCP_FILESYSTEM_ACCESS_VALIDATION_FAILED"
MCP_FILESYSTEM_ACCESS_ROLLBACK_FAILED = "MCP_FILESYSTEM_ACCESS_ROLLBACK_FAILED"
MCP_FILESYSTEM_ACCESS_NOT_INSTALLED = "MCP_FILESYSTEM_ACCESS_NOT_INSTALLED"
MCP_FILESYSTEM_ACCESS_LOOP_PREVENTED = "MCP_FILESYSTEM_ACCESS_LOOP_PREVENTED"
MCP_FILESYSTEM_ROOT_NOT_FOUND = "MCP_FILESYSTEM_ROOT_NOT_FOUND"
MCP_FILESYSTEM_LAST_ROOT_REQUIRED = "MCP_FILESYSTEM_LAST_ROOT_REQUIRED"

# Phase F.1 hotfix — deterministic MCP runtime session replacement after an
# approved filesystem access change (mcp_layer/runtime_manager.py).
MCP_RUNTIME_RESTART_FAILED = "MCP_RUNTIME_RESTART_FAILED"
MCP_RUNTIME_REBIND_FAILED = "MCP_RUNTIME_REBIND_FAILED"
MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH = "MCP_FILESYSTEM_RUNTIME_ROOT_MISMATCH"
MCP_RUNTIME_ROLLBACK_FAILED = "MCP_RUNTIME_ROLLBACK_FAILED"
MCP_RESUME_ABORTED = "MCP_RESUME_ABORTED"

# Phase G.1 — deterministic MCP capability detection and server selection
# (mcp_management/capabilities.py, capability_detector.py, server_selector.py).
# MCP_CAPABILITY_UNAVAILABLE already exists above (Phase F).
MCP_SERVER_SELECTION_AMBIGUOUS = "MCP_SERVER_SELECTION_AMBIGUOUS"
MCP_MULTI_SERVER_WORKFLOW_REQUIRED = "MCP_MULTI_SERVER_WORKFLOW_REQUIRED"
MCP_CAPABILITY_CATALOG_INVALID = "MCP_CAPABILITY_CATALOG_INVALID"

# Phase G.2 — server-keyed multi-runtime manager and lazy activation
# (mcp_layer/runtime_manager.py: MultiMcpRuntimeManager; mcp_management/
# runtime_activation.py). Preserves every Phase F.1 code above unchanged.
MCP_SERVER_NOT_INSTALLED = "MCP_SERVER_NOT_INSTALLED"
MCP_SERVER_DISABLED = "MCP_SERVER_DISABLED"
MCP_RUNTIME_START_FAILED = "MCP_RUNTIME_START_FAILED"
MCP_RUNTIME_NOT_HEALTHY = "MCP_RUNTIME_NOT_HEALTHY"
MCP_RUNTIME_ALREADY_STARTING = "MCP_RUNTIME_ALREADY_STARTING"
MCP_RUNTIME_STOP_FAILED = "MCP_RUNTIME_STOP_FAILED"
MCP_EXPECTED_TOOL_MISSING = "MCP_EXPECTED_TOOL_MISSING"
MCP_SERVER_CONFIG_NOT_FOUND = "MCP_SERVER_CONFIG_NOT_FOUND"
MCP_SERVER_CONFIG_INVALID = "MCP_SERVER_CONFIG_INVALID"
MCP_RUNTIME_VALIDATION_FAILED = "MCP_RUNTIME_VALIDATION_FAILED"

# Phase G.3 — generalized trusted MCP provisioning: approval, deterministic
# candidate installation (npm / python_venv), candidate validation, atomic
# activation, and original-request resumption (mcp_management/auto_provisioning.py,
# mcp_management/installers/*).
MCP_PROVISIONING_APPROVAL_REQUIRED = "MCP_PROVISIONING_APPROVAL_REQUIRED"
MCP_PROVISIONING_PLAN_EXPIRED = "MCP_PROVISIONING_PLAN_EXPIRED"
MCP_PROVISIONING_ALREADY_IN_PROGRESS = "MCP_PROVISIONING_ALREADY_IN_PROGRESS"
MCP_INSTALLER_UNSUPPORTED = "MCP_INSTALLER_UNSUPPORTED"
MCP_INSTALLATION_INTEGRITY_FAILED = "MCP_INSTALLATION_INTEGRITY_FAILED"
MCP_PYTHON_VERSION_UNSUPPORTED = "MCP_PYTHON_VERSION_UNSUPPORTED"
MCP_LOCK_FILE_INVALID = "MCP_LOCK_FILE_INVALID"
MCP_EXECUTABLE_VALIDATION_FAILED = "MCP_EXECUTABLE_VALIDATION_FAILED"
MCP_CANDIDATE_START_FAILED = "MCP_CANDIDATE_START_FAILED"
MCP_UNEXPECTED_TOOL_EXPOSED = "MCP_UNEXPECTED_TOOL_EXPOSED"
MCP_CONFIGURATION_ACTIVATION_FAILED = "MCP_CONFIGURATION_ACTIVATION_FAILED"
MCP_INSTALLED_STATE_UPDATE_FAILED = "MCP_INSTALLED_STATE_UPDATE_FAILED"
MCP_PROVISIONING_ROLLBACK_FAILED = "MCP_PROVISIONING_ROLLBACK_FAILED"
MCP_PROVISIONING_RESUME_FAILED = "MCP_PROVISIONING_RESUME_FAILED"
# Reused from Phase F unchanged: MCP_PROVISIONING_DECLINED, MCP_PROVISIONING_PLAN_INVALID,
# MCP_INSTALLATION_FAILED, MCP_EXPECTED_TOOL_MISSING (Phase G.2).

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
    # Phase C: effective permission. Default DENIED so a tool without an explicit
    # classification can never run (fail closed). Enforcement lives in ToolExecutor.
    permission: "ToolPermission" = ToolPermission.DENIED

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
class ToolConfirmation:
    """A user's decision on a single pending write-tool call.

    Single-use and call-bound: it approves only the exact tool + arguments it was
    issued for. A previous 'yes' can never approve a different or later action.
    """

    approved: bool
    tool_name: str
    arguments_hash: str
    request_id: Optional[str] = None


@dataclass(frozen=True)
class ToolError:
    code: str
    message: str
    retryable: bool = False
    # Safe, structured, non-sensitive extras for the model/orchestrator (e.g. a
    # deterministic action_summary for a confirmation request). Never secrets,
    # tokens, raw repository content, or absolute paths.
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        out = {"code": self.code, "message": self.message}
        if self.details:
            out["details"] = self.details
        return out


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
        details: Optional[dict] = None,
    ) -> "ToolResult":
        return cls(
            False,
            tool_name,
            call_id,
            data={},
            error=ToolError(code, message, retryable, details=details),
            execution_time_ms=execution_time_ms,
            log_meta=log_meta,
        )
