"""Phase G.1 — deterministic MCP server candidate ranking and selection.

Selects an APPROVED CATALOG SERVER as the preferred provider for one or more
`CapabilityRequirement`s — never an exact tool (that remains Phase B's job) and
never anything outside the trusted catalog. Nothing here starts, stops,
installs, or reconfigures a server: `installed_state` and `runtime_status` are
strictly read-only views over state that already exists (the Phase F installed-
server registry, and whatever session `mcp_layer.runtime_manager.ActiveMcpRuntime`
currently holds).

Scoring is a simple, fully deterministic sum of evidence weights (see
`mcp_management.capability_detector`), broken only by (score desc, server_id
asc) — no randomness, no LLM tie-breaker. Two independent tie situations both
resolve to `AMBIGUOUS` rather than an arbitrary winner: two servers that fully
cover the requirement(s) with an equal top score, or an explicit reference to a
server name the trusted catalog does not recognize.
"""

from typing import Optional, Protocol, Sequence

from mcp_management.capabilities import (
    NONE_REQUIRED_SELECTION,
    CapabilityEvidenceType,
    CapabilitySelectionStatus,
    McpServerCandidate,
    McpServerSelection,
    ToolRequirement,
)
from tools.models import (
    MCP_CAPABILITY_CATALOG_INVALID,
    MCP_CAPABILITY_UNAVAILABLE,
    MCP_MULTI_SERVER_WORKFLOW_REQUIRED,
    MCP_SERVER_SELECTION_AMBIGUOUS,
)

MAX_CANDIDATES = 3

_EXPLICIT_SERVER_BONUS = 1000
_EXACT_CAPABILITY_BONUS = 500
_INSTALLED_BONUS = 10
_ENABLED_BONUS = 3
_ACTIVE_BONUS = 5


# ---- Task 9: read-only status abstractions ----

class McpInstalledStateProvider(Protocol):
    def is_installed(self, server_id: str) -> bool: ...
    def is_disabled(self, server_id: str) -> bool: ...


class McpRuntimeStatusProvider(Protocol):
    def is_active(self, server_id: str) -> bool: ...
    def get_health(self, server_id: str) -> Optional[str]: ...


class RegistryInstalledState:
    """Read-only view over the existing Phase F installed-server registry.

    Reuses `mcp_management.registry.load_registry` — no parallel installed-state
    store. Loaded once per instance (a selector is built fresh per request).
    """

    def __init__(self, base_dir=None, managed_root=None, registry_path=None):
        from mcp_management.registry import STATUS_DISABLED, load_registry

        self._servers = load_registry(registry_path, base_dir, managed_root)
        self._disabled_status = STATUS_DISABLED

    def is_installed(self, server_id: str) -> bool:
        return server_id in self._servers

    def is_disabled(self, server_id: str) -> bool:
        entry = self._servers.get(server_id)
        return entry is not None and entry.status == self._disabled_status


class ActiveRuntimeStatusProvider:
    """Read-only view over a runtime holder — either
    `mcp_layer.runtime_manager.MultiMcpRuntimeManager` (Phase G.2, preferred:
    exposes `.get_session(server_id)`) or the older single-slot
    `ActiveMcpRuntime` (exposes a single `.session`). Never calls a lifecycle
    method on either."""

    def __init__(self, runtime=None):
        self._runtime = runtime

    def _session_for(self, server_id):
        if self._runtime is None:
            return None
        get_session = getattr(self._runtime, "get_session", None)
        if callable(get_session):
            return get_session(server_id)
        return getattr(self._runtime, "session", None)

    def _health(self, server_id):
        session = self._session_for(server_id)
        health = getattr(session, "health", None)
        if health is not None and getattr(health, "server_id", None) == server_id:
            return health
        return None

    def is_active(self, server_id: str) -> bool:
        health = self._health(server_id)
        return bool(health is not None and health.state.value == "healthy")

    def get_health(self, server_id: str) -> Optional[str]:
        health = self._health(server_id)
        return health.state.value if health is not None else None


class _NullInstalledState:
    def is_installed(self, server_id):
        return False

    def is_disabled(self, server_id):
        return False


class _NullRuntimeStatus:
    def is_active(self, server_id):
        return False

    def get_health(self, server_id):
        return None


# ---- the selector ----

class McpServerSelector:
    """Ranks approved catalog servers for a set of CapabilityRequirements."""

    def select(
        self,
        requirements: Sequence,
        catalog,
        installed_state: Optional[McpInstalledStateProvider] = None,
        runtime_status: Optional[McpRuntimeStatusProvider] = None,
    ) -> McpServerSelection:
        requirements = tuple(requirements or ())
        if not requirements:
            return NONE_REQUIRED_SELECTION

        if catalog is None:
            return McpServerSelection(
                status=CapabilitySelectionStatus.INVALID_CATALOG,
                required_capabilities=requirements, selected_server_id=None, selected_catalog_id=None,
                candidates=(), explanation="No trusted MCP catalog is available.",
                error_code=MCP_CAPABILITY_CATALOG_INVALID,
            )

        installed_state = installed_state or _NullInstalledState()
        runtime_status = runtime_status or _NullRuntimeStatus()

        # An explicit reference to an unrecognized server always forces
        # clarification — never silently substituted for a different, approved
        # provider the user did not actually ask for.
        unknown_name = self._unknown_explicit_server(requirements)
        if unknown_name is not None:
            return McpServerSelection(
                status=CapabilitySelectionStatus.AMBIGUOUS,
                required_capabilities=requirements, selected_server_id=None, selected_catalog_id=None,
                candidates=(),
                explanation=(f"'{unknown_name}' is not an approved MCP server. Please specify which "
                            "approved server or capability you want to use."),
                error_code=MCP_SERVER_SELECTION_AMBIGUOUS,
            )

        required_ids = tuple(dict.fromkeys(r.capability_id for r in requirements))
        req_by_id = {r.capability_id: r for r in requirements}
        explicit_server_id = self._explicit_known_server(requirements)

        candidates = self._build_candidates(
            required_ids, req_by_id, catalog, installed_state, runtime_status, explicit_server_id)

        complete = [c for c in candidates if set(required_ids) <= set(c.matched_capabilities)]

        if complete:
            ranked = sorted(complete, key=lambda c: (-c.score, c.server_id))
            bounded = tuple(ranked[:MAX_CANDIDATES])
            top_score = ranked[0].score
            tied = [c for c in ranked if c.score == top_score]
            if len(tied) > 1:
                return McpServerSelection(
                    status=CapabilitySelectionStatus.AMBIGUOUS,
                    required_capabilities=requirements, selected_server_id=None,
                    selected_catalog_id=None, candidates=bounded,
                    explanation=("More than one approved MCP server could handle this request. "
                                "Please specify which server or capability you want to use."),
                    error_code=MCP_SERVER_SELECTION_AMBIGUOUS,
                )
            winner = ranked[0]
            return McpServerSelection(
                status=CapabilitySelectionStatus.SELECTED,
                required_capabilities=requirements, selected_server_id=winner.server_id,
                selected_catalog_id=winner.catalog_id, candidates=bounded,
                explanation=(f"Selected {winner.server_id!r} for capability match "
                            f"({', '.join(winner.matched_capabilities)})."),
                tool_requirement=ToolRequirement.REQUIRED,
                preferred_mcp_server_id=winner.server_id,
            )

        # No single approved server covers every required capability.
        ranked_partial = sorted(candidates, key=lambda c: (-c.score, c.server_id))
        bounded_partial = tuple(ranked_partial[:MAX_CANDIDATES])
        if len(required_ids) == 1:
            return McpServerSelection(
                status=CapabilitySelectionStatus.UNSUPPORTED,
                required_capabilities=requirements, selected_server_id=None, selected_catalog_id=None,
                candidates=bounded_partial,
                explanation=(f"This request requires the {required_ids[0]!r} capability, but no "
                            "approved MCP server currently provides it."),
                error_code=MCP_CAPABILITY_UNAVAILABLE,
            )
        return McpServerSelection(
            status=CapabilitySelectionStatus.MULTI_SERVER_REQUIRED,
            required_capabilities=requirements, selected_server_id=None, selected_catalog_id=None,
            candidates=bounded_partial,
            explanation=("This request requires a workflow across multiple MCP servers. "
                        "Multi-server workflows are not enabled yet."),
            error_code=MCP_MULTI_SERVER_WORKFLOW_REQUIRED,
        )

    # ---- internals ----

    def _unknown_explicit_server(self, requirements):
        for req in requirements:
            for ev in req.evidence:
                if ev.evidence_type == CapabilityEvidenceType.EXPLICIT_SERVER and ev.value.startswith("unknown:"):
                    return ev.value.split(":", 1)[1]
        return None

    def _explicit_known_server(self, requirements):
        for req in requirements:
            for ev in req.evidence:
                if ev.evidence_type == CapabilityEvidenceType.EXPLICIT_SERVER and not ev.value.startswith("unknown:"):
                    return ev.value
        return None

    def _build_candidates(self, required_ids, req_by_id, catalog, installed_state, runtime_status,
                          explicit_server_id):
        candidates = []
        for entry in catalog.entries.values():
            if not entry.granular_capabilities:
                continue  # no Phase G.1 metadata -> not yet selectable (backward compat)
            if not entry.enabled:
                continue  # catalog entry explicitly disabled
            matched = tuple(cid for cid in required_ids if cid in entry.granular_capabilities)
            if not matched:
                continue
            if installed_state.is_disabled(entry.server_id):
                continue  # never select a disabled entry

            installed = bool(installed_state.is_installed(entry.server_id))
            active = bool(runtime_status.is_active(entry.server_id))
            enabled = True  # reached only when not catalog- or state-disabled, above

            score = 0
            reasons = []
            if explicit_server_id == entry.server_id:
                score += _EXPLICIT_SERVER_BONUS
                reasons.append("explicit server reference")
            for cid in matched:
                score += _EXACT_CAPABILITY_BONUS
                reasons.append(f"capability match: {cid}")
                for ev in req_by_id[cid].evidence:
                    if ev.evidence_type in (CapabilityEvidenceType.ACTION_OBJECT,
                                            CapabilityEvidenceType.FILE_EXTENSION,
                                            CapabilityEvidenceType.LOCAL_PATH):
                        score += ev.score
                        reasons.append(f"{ev.evidence_type.value}: {ev.value}")
            if installed:
                score += _INSTALLED_BONUS
                reasons.append("installed")
            if enabled:
                score += _ENABLED_BONUS
            if active:
                score += _ACTIVE_BONUS
                reasons.append("active")

            candidates.append(McpServerCandidate(
                server_id=entry.server_id, catalog_id=entry.catalog_id,
                capabilities=entry.granular_capabilities, matched_capabilities=matched,
                score=score, installed=installed, active=active, enabled=enabled,
                reasons=tuple(reasons),
            ))
        return candidates
