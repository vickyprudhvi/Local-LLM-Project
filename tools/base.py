"""BaseTool — a minimal synchronous tool abstraction.

A tool performs one bounded operation and returns structured, JSON-serializable
data. Tools know NOTHING about conversation history, ChromaDB, UI channels,
Ollama messages, routing, system prompts, or the final response wording.
"""

from abc import ABC, abstractmethod

from tools.models import ToolDefinition


class ToolValidationError(Exception):
    """Raised by validate_arguments/execute for a controlled invalid-argument case.

    The executor converts this into an INVALID_ARGUMENTS ToolResult — the message
    text is safe to surface to the local LLM (never a raw stack trace).
    """


class ToolFailure(Exception):
    """Raised by a tool's execute() to surface a controlled error CODE.

    The executor converts this into a ToolResult.fail(code, message) so Phase 2A
    tools can report codes like SEARCH_RATE_LIMITED or GITHUB_FILE_NOT_FOUND. The
    message must be safe for the local LLM (no stack traces, no secrets). Optional
    log_meta is safe, non-content metadata for the interaction log only.
    """

    def __init__(self, code: str, message: str, retryable: bool = False, log_meta: dict = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.log_meta = log_meta


class BaseTool(ABC):
    # Subclasses set these as class attributes.
    name: str = ""
    description: str = ""
    input_schema: dict = {}
    timeout_seconds: float = 10.0
    enabled: bool = True
    # Phase 2A: tools that reach the network set this True. The executor blocks
    # them with INTERNET_DISABLED when INTERNET_READ_ENABLED is off. Phase 1 tools
    # keep the default (False) and are unaffected.
    requires_internet: bool = False
    # Phase 2B: named capabilities gated at execution time (e.g. "repository.clone",
    # "repository.read"). The executor blocks the tool with the mapped controlled
    # error when the capability's config flag is off. Empty for Phase 1/2A tools.
    required_capabilities: tuple = ()
    # Phase A: whether this tool is offered to the local tool-calling loop (the
    # LLM chooses it itself). True for LLM-selectable tools (echo, calculate,
    # web/GitHub/repo). False for router-dispatched built-ins (memory/time/camera/
    # calendar): they are registered and executed through the same registry and
    # executor, but selected by the router, never offered to the local LLM — so
    # registering them here does NOT change what the local loop can call.
    llm_callable: bool = True

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            timeout_seconds=self.timeout_seconds,
            enabled=self.enabled,
        )

    def validate_arguments(self, arguments: dict) -> dict:
        """Validate/normalize arguments. Raise ToolValidationError if invalid.

        Default implementation checks that `arguments` is a dict and that every
        `required` property in input_schema is present. Subclasses may override
        for stricter checks and should return the (possibly normalized) args.
        """
        if not isinstance(arguments, dict):
            raise ToolValidationError("Arguments must be a JSON object.")
        for key in self.input_schema.get("required", []):
            if key not in arguments:
                raise ToolValidationError(f"Missing required argument: {key!r}.")
        return arguments

    @abstractmethod
    def execute(self, arguments: dict) -> dict:
        """Run the operation and return a JSON-serializable dict (the ToolResult.data)."""
        raise NotImplementedError
