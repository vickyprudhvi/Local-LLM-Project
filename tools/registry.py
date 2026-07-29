"""ToolRegistry — a static, in-process registry of built-in tools.

No dynamic plugin discovery. Built-ins are registered by default_registry():
the Phase 1 tools always, and the Phase 2A read-only internet/GitHub tools when
INTERNET_TOOLS_ENABLED is on.
"""

import re
from typing import List

import tools.config as config
from tools.base import BaseTool
from tools.calculator import CalculatorTool
from tools.echo import EchoTool
from tools.models import ToolDefinition

# ---- Phase B: lexical relevance shortlisting (no embeddings/RAG) ----
# A deterministic, dependency-free scorer: overlap between the user message's
# tokens and each tool's name + description tokens, with name matches weighted
# higher. This is intentionally simple — it only has to surface the handful of
# plausibly-relevant tools so the local LLM chooses among a bounded candidate set.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "do", "does", "did", "you", "me", "my", "your", "it", "that", "this", "what",
    "whats", "how", "can", "could", "would", "please", "with", "at", "be", "have",
    "has", "get", "got", "if", "about", "as", "from", "by", "was", "were", "will",
    "just", "tell", "show", "give", "want", "need", "use",
})


def _tokenize(text: str) -> set:
    if not text:
        return set()
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS}


def _definition_tokens(definition: ToolDefinition):
    name_tokens = set(_TOKEN_RE.findall(definition.name.lower()))
    all_tokens = name_tokens | _tokenize(definition.description or "")
    return all_tokens, name_tokens


def _relevance(query_tokens: set, definition: ToolDefinition) -> int:
    all_tokens, name_tokens = _definition_tokens(definition)
    return len(query_tokens & all_tokens) + len(query_tokens & name_tokens)


def bounded_ollama_schema(definition: ToolDefinition, max_description_chars=None) -> dict:
    """An Ollama tool schema whose description is truncated to the char budget.

    Applied before prompt construction so no single tool definition can bloat the
    selection prompt regardless of how verbose its registered description is.
    """
    if max_description_chars is None:
        max_description_chars = config.max_tool_description_chars()
    description = definition.description or ""
    if len(description) > max_description_chars:
        description = description[:max_description_chars].rstrip()
    return {
        "type": "function",
        "function": {
            "name": definition.name,
            "description": description,
            "parameters": definition.input_schema,
        },
    }


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
        """Definitions offered to the local LLM: enabled AND llm_callable, in
        deterministic (name-sorted) order.

        Router-dispatched built-ins (llm_callable=False) are registered and
        executable through the same registry/executor, but are deliberately
        excluded here so the local tool-calling loop never offers them — that
        keeps the local LLM's callable tool set unchanged.
        """
        defs = []
        for name in sorted(self._tools):
            tool = self._tools[name]
            if self._enabled.get(name) and getattr(tool, "llm_callable", True):
                d = tool.definition()
                # Reflect the registry's live enabled state on the definition.
                defs.append(ToolDefinition(d.name, d.description, d.input_schema, d.timeout_seconds, True))
        return defs

    def enabled_ollama_schemas(self) -> List[dict]:
        return [d.to_ollama_schema() for d in self.enabled_definitions()]

    def shortlist_tools(self, user_message: str, limit=None) -> List[ToolDefinition]:
        """Return at most `limit` candidate ToolDefinitions, most relevant first.

        Phase B: the local LLM never receives the full registry. This bounds the
        candidate set so the selection prompt stays ~constant in size as the
        registry grows. It only limits candidates — it does NOT force tool use;
        the local LLM may still choose no tool. Selection is deterministic
        (relevance desc, then name asc) so tests and logs are reproducible.
        """
        if limit is None:
            limit = config.max_shortlist_tools()
        candidates = self.enabled_definitions()  # llm_callable + enabled, name-sorted
        if len(candidates) <= limit:
            return candidates
        query_tokens = _tokenize(user_message)
        ranked = sorted(candidates, key=lambda d: (-_relevance(query_tokens, d), d.name))
        return ranked[:limit]


def default_registry(include_internet=None, include_clone=None, include_repo=None) -> ToolRegistry:
    """A fresh registry with the built-in tools registered.

    Phase 1 tools are always registered. Phase 2A internet/GitHub tools when internet
    tools are enabled. Phase 2B: github.clone_repository when internet + cloning are
    enabled; the 5 repo.* inspection tools when inspection (or cloning) is enabled.
    Overrides (include_*) are for tests. Shared GitHubClient / GitRunner are created
    here and owned by the registry — not module-level globals.
    """
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(CalculatorTool())
    _register_builtins(reg)

    if include_internet is None:
        include_internet = config.internet_tools_enabled()
    if include_clone is None:
        include_clone = config.repository_clone_enabled()
    if include_repo is None:
        include_repo = config.repository_inspection_enabled()

    github_client = None
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

    # Phase 2B: cloning needs internet (metadata preflight) + the clone capability.
    if include_internet and include_clone:
        from tools.git_runner import GitRunner
        from tools.repo_clone import CloneRepositoryTool
        if github_client is None:
            from tools.github_client import GitHubClient
            github_client = GitHubClient()
        reg.register(CloneRepositoryTool(client=github_client, runner=GitRunner()))

    # Phase 2B: static inspection of already-cloned repositories.
    if include_repo:
        from tools.repo_tools import ALL_REPO_TOOL_CLASSES
        for tool_cls in ALL_REPO_TOOL_CLASSES:
            reg.register(tool_cls())
    return reg


def _register_builtins(reg: ToolRegistry) -> None:
    """Register the router-dispatched built-in capabilities.

    These are the former hardcoded branches of assistant.dispatch (memory, time,
    camera, calendar). They are always registered and always executed through the
    shared ToolExecutor, but marked llm_callable=False so the local tool-calling
    loop never offers them. Imported lazily here so a Phase-1-only import of
    tools.registry stays light. See tool_dispatch.py for how the router selects them.
    """
    from tools.calendar_tools import CalendarReadTool
    from tools.camera_tools import CaptureCameraTool, LookCarefullyTool, LookTool, ScanRoomTool
    from tools.memory_tools import RecallTool, RememberTool
    from tools.system_tools import TimeTool

    for tool_cls in (
        TimeTool,
        RememberTool,
        RecallTool,
        CalendarReadTool,
        LookTool,
        LookCarefullyTool,
        CaptureCameraTool,
        ScanRoomTool,
    ):
        reg.register(tool_cls())
