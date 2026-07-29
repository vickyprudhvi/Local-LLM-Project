"""ToolExecutor: success, every controlled error, timeout, duration, logging."""

import json
import time
from unittest.mock import patch

import pytest

from tools.base import BaseTool, ToolValidationError
from tools.models import ToolCall, ToolPermission
from tools.registry import ToolRegistry, default_registry
from tools.executor import ToolExecutor


class _SlowTool(BaseTool):
    name = "test.slow"
    description = "sleeps past its timeout"
    input_schema = {"type": "object", "properties": {}}
    timeout_seconds = 0.2
    permission = ToolPermission.READ

    def execute(self, arguments):
        time.sleep(1.0)
        return {"done": True}


class _ExplodingTool(BaseTool):
    name = "test.boom"
    description = "raises an unexpected error"
    input_schema = {"type": "object", "properties": {}}
    permission = ToolPermission.READ

    def execute(self, arguments):
        raise RuntimeError("secret internal detail")


class _BadOutputTool(BaseTool):
    name = "test.badoutput"
    description = "returns a non-dict"
    input_schema = {"type": "object", "properties": {}}
    permission = ToolPermission.READ

    def execute(self, arguments):
        return [1, 2, 3]


def _executor_with(*tools):
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return ToolExecutor(reg), reg


def test_successful_execution_and_duration_and_serializable():
    ex = ToolExecutor(default_registry())
    r = ex.execute(ToolCall("c1", "math.calculate", {"expression": "(17 * 23) + 5"}), step=1)
    assert r.success is True
    assert r.data == {"expression": "(17 * 23) + 5", "result": 396}
    assert r.execution_time_ms is not None and r.execution_time_ms >= 0
    json.dumps(r.to_provider_dict())  # must not raise


def test_invalid_arguments():
    ex = ToolExecutor(default_registry())
    r = ex.execute(ToolCall("c1", "math.calculate", {"expression": "1/0"}), step=1)
    assert r.success is False
    assert r.error.code == "INVALID_ARGUMENTS"


def test_missing_required_argument_is_invalid():
    ex = ToolExecutor(default_registry())
    r = ex.execute(ToolCall("c1", "math.calculate", {}), step=1)
    assert r.error.code == "INVALID_ARGUMENTS"


def test_unknown_tool():
    ex = ToolExecutor(default_registry())
    r = ex.execute(ToolCall("c1", "does.not.exist", {}), step=1)
    assert r.error.code == "UNKNOWN_TOOL"


def test_disabled_tool():
    reg = default_registry()
    reg.disable("system.echo")
    ex = ToolExecutor(reg)
    r = ex.execute(ToolCall("c1", "system.echo", {"text": "x"}), step=1)
    assert r.error.code == "TOOL_DISABLED"


def test_timeout():
    ex, _ = _executor_with(_SlowTool())
    r = ex.execute(ToolCall("c1", "test.slow", {}), step=1)
    assert r.success is False
    assert r.error.code == "TOOL_TIMEOUT"


def test_unexpected_exception_is_contained_without_leaking_detail():
    ex, _ = _executor_with(_ExplodingTool())
    r = ex.execute(ToolCall("c1", "test.boom", {}), step=1)
    assert r.error.code == "TOOL_EXECUTION_ERROR"
    assert "secret internal detail" not in r.error.message


def test_invalid_tool_output():
    ex, _ = _executor_with(_BadOutputTool())
    r = ex.execute(ToolCall("c1", "test.badoutput", {}), step=1)
    assert r.error.code == "INVALID_TOOL_OUTPUT"


def test_controlled_validation_error_during_execute():
    class _V(BaseTool):
        name = "test.validate"
        description = "raises validation error in execute"
        input_schema = {"type": "object", "properties": {}}
        permission = ToolPermission.READ

        def execute(self, arguments):
            raise ToolValidationError("bad input at runtime")

    ex, _ = _executor_with(_V())
    r = ex.execute(ToolCall("c1", "test.validate", {}), step=1)
    assert r.error.code == "INVALID_ARGUMENTS"
    assert r.error.message == "bad input at runtime"


def test_interaction_log_receives_safe_metadata():
    ex = ToolExecutor(default_registry())
    with patch("tools.executor.log_tool_event") as mock_log:
        ex.execute(ToolCall("c1", "system.echo", {"text": "hello"}), step=3)
    # start + complete
    assert mock_log.call_count >= 2
    for call in mock_log.call_args_list:
        args, kwargs = call
        joined = json.dumps({"a": list(args), "k": kwargs})
        # No secrets / no raw tool args or output should ever be logged.
        assert "hello" not in joined
    # The safe metadata (tool, call_id, step) is present.
    first_args = mock_log.call_args_list[0][0]
    assert first_args[0] == "system.echo"
    assert first_args[1] == "c1"
    assert first_args[2] == 3
