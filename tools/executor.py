"""ToolExecutor — validates and runs a single ToolCall, synchronously.

Timeouts use concurrent.futures.ThreadPoolExecutor + future.result(timeout=...).

Thread-timeout limitation (Phase 1): a Python thread that exceeds its timeout may
keep running in the background — Python cannot forcibly kill it. This is acceptable
here because the only tools are bounded and side-effect-free (echo, calculate). We
do NOT claim the thread is terminated.

All failures are returned as structured ToolResult objects — never raw stack traces.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import tools.config as config
from interaction_log import log_tool_event
from tools.base import ToolFailure, ToolValidationError
from tools.models import (
    INTERNET_DISABLED,
    INVALID_ARGUMENTS,
    INVALID_TOOL_OUTPUT,
    REPOSITORY_CLONE_DISABLED,
    REPOSITORY_INSPECTION_DISABLED,
    TOOL_DISABLED,
    TOOL_EXECUTION_ERROR,
    TOOL_TIMEOUT,
    UNKNOWN_TOOL,
    ToolCall,
    ToolResult,
)

# Named capability -> (config check, controlled error code, message shown to the LLM).
# Kept tiny and explicit; not a general permission platform.
_CAPABILITY_GATES = {
    "repository.clone": (config.repository_clone_enabled, REPOSITORY_CLONE_DISABLED,
                         "Repository cloning is disabled."),
    "repository.read": (config.repository_inspection_enabled, REPOSITORY_INSPECTION_DISABLED,
                        "Repository inspection is disabled."),
}


class ToolExecutor:
    def __init__(self, registry):
        self.registry = registry

    def execute(self, call: ToolCall, step: int = 0) -> ToolResult:
        name = call.tool_name
        started = time.perf_counter()

        def elapsed_ms():
            return round((time.perf_counter() - started) * 1000, 3)

        # 1-2. Exists / enabled.
        if not self.registry.has(name):
            log_tool_event(name, call.call_id, step, "rejected", error_code=UNKNOWN_TOOL)
            return ToolResult.fail(name, call.call_id, UNKNOWN_TOOL,
                                   f"No tool named {name!r} is available.", elapsed_ms())
        if not self.registry.is_enabled(name):
            log_tool_event(name, call.call_id, step, "rejected", error_code=TOOL_DISABLED)
            return ToolResult.fail(name, call.call_id, TOOL_DISABLED,
                                   f"The tool {name!r} is currently disabled.", elapsed_ms())

        tool = self.registry.get(name)

        # 2b. Capability gate: internet tools require read-only internet access.
        if getattr(tool, "requires_internet", False) and not config.internet_read_enabled():
            log_tool_event(name, call.call_id, step, "rejected", elapsed_ms(), INTERNET_DISABLED)
            return ToolResult.fail(name, call.call_id, INTERNET_DISABLED,
                                   "Read-only internet access is disabled.", elapsed_ms())

        # 2c. Named capability gates (Phase 2B: repository.clone / repository.read).
        for capability in getattr(tool, "required_capabilities", ()):
            gate = _CAPABILITY_GATES.get(capability)
            if gate is not None and not gate[0]():
                log_tool_event(name, call.call_id, step, "rejected", elapsed_ms(), gate[1])
                return ToolResult.fail(name, call.call_id, gate[1], gate[2], elapsed_ms())

        # 3. Validate arguments (controlled).
        try:
            arguments = tool.validate_arguments(call.arguments)
        except ToolValidationError as e:
            log_tool_event(name, call.call_id, step, "rejected", elapsed_ms(), INVALID_ARGUMENTS)
            return ToolResult.fail(name, call.call_id, INVALID_ARGUMENTS, str(e), elapsed_ms())
        except ToolFailure as e:
            # A tool may raise a more specific coded error during validation
            # (e.g. INVALID_REPOSITORY, INVALID_REPOSITORY_PATH).
            log_tool_event(name, call.call_id, step, "rejected", elapsed_ms(), e.code)
            return ToolResult.fail(name, call.call_id, e.code, e.message, elapsed_ms())

        # 4-6. Execute with a hard timeout in a worker thread.
        log_tool_event(name, call.call_id, step, "start")
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(tool.execute, arguments)
            try:
                data = future.result(timeout=tool.timeout_seconds)
            except FuturesTimeout:
                log_tool_event(name, call.call_id, step, "timeout", elapsed_ms(), TOOL_TIMEOUT)
                return ToolResult.fail(
                    name, call.call_id, TOOL_TIMEOUT,
                    f"The tool timed out after {tool.timeout_seconds:g}s.", elapsed_ms(), retryable=True,
                )
            except ToolValidationError as e:
                # A tool may raise this during execute() for input-dependent invalidity.
                log_tool_event(name, call.call_id, step, "rejected", elapsed_ms(), INVALID_ARGUMENTS)
                return ToolResult.fail(name, call.call_id, INVALID_ARGUMENTS, str(e), elapsed_ms())
            except ToolFailure as e:
                # A controlled, coded failure from a Phase 2A tool (network/API/etc.).
                log_tool_event(name, call.call_id, step, "error", elapsed_ms(), e.code, extra=e.log_meta)
                return ToolResult.fail(name, call.call_id, e.code, e.message, elapsed_ms(),
                                       retryable=e.retryable, log_meta=e.log_meta)
            except Exception as e:  # noqa: BLE001 — unexpected: contain it, don't leak a trace.
                log_tool_event(name, call.call_id, step, "error", elapsed_ms(), TOOL_EXECUTION_ERROR)
                return ToolResult.fail(
                    name, call.call_id, TOOL_EXECUTION_ERROR,
                    f"The tool failed to execute ({type(e).__name__}).", elapsed_ms(),
                )

        # 7. Validate output: must be a JSON-serializable dict.
        if not isinstance(data, dict):
            log_tool_event(name, call.call_id, step, "error", elapsed_ms(), INVALID_TOOL_OUTPUT)
            return ToolResult.fail(name, call.call_id, INVALID_TOOL_OUTPUT,
                                   "The tool returned a non-object result.", elapsed_ms())
        # A tool may attach safe, non-content logging metadata under "_log_meta".
        # It is removed before the result reaches the model and logged separately.
        log_meta = data.pop("_log_meta", None)
        try:
            json.dumps(data)
        except (TypeError, ValueError):
            log_tool_event(name, call.call_id, step, "error", elapsed_ms(), INVALID_TOOL_OUTPUT)
            return ToolResult.fail(name, call.call_id, INVALID_TOOL_OUTPUT,
                                   "The tool returned a non-serializable result.", elapsed_ms())

        result = ToolResult.ok(name, call.call_id, data, elapsed_ms(), log_meta=log_meta)
        log_tool_event(name, call.call_id, step, "complete", result.execution_time_ms, extra=log_meta)
        return result
