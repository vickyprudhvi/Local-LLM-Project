"""system.echo — the simplest possible tool.

Proves tool registration, argument validation, result serialization, and that
the result returns to the local LLM. It does NOT generate a conversational answer.
"""

from tools.base import BaseTool, ToolValidationError
from tools.models import ToolPermission

MAX_ECHO_LEN = 2000


class EchoTool(BaseTool):
    name = "system.echo"
    description = "Echo back the provided text unchanged. Useful only for diagnostics."
    permission = ToolPermission.READ  # pure, side-effect-free diagnostic
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to echo back."}
        },
        "required": ["text"],
    }
    timeout_seconds = 5.0

    def validate_arguments(self, arguments: dict) -> dict:
        arguments = super().validate_arguments(arguments)
        text = arguments.get("text")
        if not isinstance(text, str):
            raise ToolValidationError("'text' must be a string.")
        if len(text) > MAX_ECHO_LEN:
            raise ToolValidationError(f"'text' exceeds {MAX_ECHO_LEN} characters.")
        return arguments

    def execute(self, arguments: dict) -> dict:
        return {"echo": arguments["text"]}
