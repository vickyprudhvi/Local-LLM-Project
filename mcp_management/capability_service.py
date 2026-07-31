"""Phase G.1 — the thin orchestration facade assistant.py calls.

Wires the detector and selector to the EXISTING trusted catalog, installed-
server registry, and active-runtime holder — no parallel state of its own.
Everything here is read-only: it never starts, stops, installs, or reconfigures
an MCP server (see mcp_management.server_selector's own docstring for why the
status providers it uses are structurally read-only).
"""

from mcp_management.capabilities import NONE_REQUIRED_SELECTION, McpServerSelection
from mcp_management.capability_detector import McpCapabilityDetector
from mcp_management.server_selector import (
    ActiveRuntimeStatusProvider,
    McpServerSelector,
    RegistryInstalledState,
)

_DETECTOR = McpCapabilityDetector()
_SELECTOR = McpServerSelector()


def select_for_request(user_text, catalog, base_dir=None, managed_root=None, registry_path=None,
                       runtime=None, detector=None, selector=None) -> McpServerSelection:
    """Detect required capabilities and select an approved provider, or
    NONE_REQUIRED. `catalog` must be the caller's already-loaded trusted
    McpCatalog — this function never loads one itself, so there is only ever one
    catalog-loading path in the process."""
    if catalog is None:
        return NONE_REQUIRED_SELECTION

    detector = detector or _DETECTOR
    selector = selector or _SELECTOR

    requirements = detector.detect(user_text, catalog)
    if not requirements:
        return NONE_REQUIRED_SELECTION

    installed_state = RegistryInstalledState(base_dir=base_dir, managed_root=managed_root,
                                             registry_path=registry_path)
    runtime_status = ActiveRuntimeStatusProvider(runtime)
    return selector.select(requirements, catalog, installed_state, runtime_status)
