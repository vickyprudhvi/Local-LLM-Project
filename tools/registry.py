"""ToolRegistry — a static, in-process registry of built-in tools.

No dynamic plugin discovery. Built-ins are registered by default_registry():
the Phase 1 tools always, and the Phase 2A read-only internet/GitHub tools when
INTERNET_TOOLS_ENABLED is on.
"""

from typing import List

import tools.config as config
from tools.base import BaseTool
from tools.calculator import CalculatorTool
from tools.echo import EchoTool
from tools.models import ToolDefinition


class ToolRegistry:
    def __init__(self):
        self._tools = {}  # name -> BaseTool
        self._enabled = {}  # name -> bool

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError("Tool must have a non-empty name.")
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name!r}.")
        self._tools[tool.name] = tool
        self._enabled[tool.name] = bool(tool.enabled)

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def is_enabled(self, name: str) -> bool:
        return self._enabled.get(name, False)

    def enable(self, name: str) -> None:
        if name not in self._tools:
            raise KeyError(name)
        self._enabled[name] = True

    def disable(self, name: str) -> None:
        if name not in self._tools:
            raise KeyError(name)
        self._enabled[name] = False

    def enabled_definitions(self) -> List[ToolDefinition]:
        """Enabled tool definitions in deterministic (name-sorted) order."""
        defs = []
        for name in sorted(self._tools):
            if self._enabled.get(name):
                d = self._tools[name].definition()
                # Reflect the registry's live enabled state on the definition.
                defs.append(ToolDefinition(d.name, d.description, d.input_schema, d.timeout_seconds, True))
        return defs

    def enabled_ollama_schemas(self) -> List[dict]:
        return [d.to_ollama_schema() for d in self.enabled_definitions()]


def default_registry(include_internet=None) -> ToolRegistry:
    """A fresh registry with the built-in tools registered.

    Phase 1 tools are always registered. The 7 Phase 2A internet/GitHub tools are
    registered when internet tools are enabled (override with include_internet for
    tests). All GitHub tools share one GitHubClient; browser.search shares one
    search provider — sessions are created here and owned by the registry, not as
    module-level globals.
    """
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(CalculatorTool())

    if include_internet is None:
        include_internet = config.internet_tools_enabled()
    if include_internet:
        # Imported lazily so Phase 1-only environments never import bs4/network code.
        from tools.browser import FetchPageTool, SearchTool
        from tools.github_client import GitHubClient
        from tools.github_tools import ALL_GITHUB_TOOL_CLASSES

        reg.register(SearchTool())
        reg.register(FetchPageTool())
        github_client = GitHubClient()
        for tool_cls in ALL_GITHUB_TOOL_CLASSES:
            reg.register(tool_cls(client=github_client))
    return reg
