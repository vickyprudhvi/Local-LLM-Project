"""Phase E — MCP session health, for diagnostics only.

Never carries environment values, raw stderr, secret arguments, or full server
output — only counts, state, and the last controlled error code.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class McpHealthState(str, Enum):
    DISABLED = "disabled"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class McpHealth:
    state: McpHealthState
    server_id: Optional[str] = None
    discovered_tool_count: int = 0
    registered_tool_count: int = 0
    denied_tool_count: int = 0
    skipped_tool_count: int = 0
    disabled_tool_count: int = 0
    last_error_code: Optional[str] = None
    # (tool_name, reason, category) for each non-registered discovered tool. Names
    # and reasons only — never secrets, arguments, or server output. For diagnostics
    # / sanitized application logs, not for the LLM.
    diagnostics: tuple = ()
