"""Phase F — the trusted MCP catalog: the ONLY source of installable servers.

The catalog is application-maintained data, never model output. Every entry is
schema-validated at load time and fails closed: an unpinned version, an unknown
installer type, a non-stdio transport, a bad name, or a missing capability list
rejects the whole catalog with MCP_CATALOG_INVALID.

Because a catalog entry's `description` is displayed and may reach a prompt, it is
sanitized and length-bounded with the same helper the Phase E discovery path uses.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import tools.config as app_config
from mcp_layer.config import McpToolPolicy, McpToolPolicyEntry
from mcp_layer.discovery import sanitize_description
from mcp_layer.errors import McpError
from tools.models import MCP_CATALOG_INVALID, ToolPermission

CATALOG_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
SERVER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
CAPABILITY_RE = re.compile(r"^[a-z0-9_]+$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Exact semantic version only — never a range, tag, or wildcard.
EXACT_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")

SUPPORTED_INSTALLERS = ("npm",)
SUPPORTED_TRANSPORTS = ("stdio",)
SUPPORTED_INPUT_TYPES = ("directory", "environment_variable")
MAX_CATALOG_ENTRIES = 100
MAX_CAPABILITIES = 25
MAX_TOOLS_PER_POLICY = 100

# ---- Phase G.1: optional capability-selection metadata ----
# Deliberately a SEPARATE field from the existing `capabilities` list above (which
# Phase F's coarse detect_capability()/find_by_capability() already depend on —
# values like "filesystem", "read_files"). `granular_capabilities` uses the finer
# per-action vocabulary the Phase G.1 selector matches against (e.g.
# "read_local_text_file"), so adding it can never change what Phase F already
# matches. An entry with no `granular_capabilities` is simply invisible to the
# Phase G.1 selector — fully backward compatible.
MAX_GRANULAR_CAPABILITIES = 50
MAX_EXPLICIT_NAMES = 20
MAX_HINT_PHRASES_PER_CAPABILITY = 20
MAX_EXTENSIONS_PER_CAPABILITY = 30
_EXTENSION_RE = re.compile(r"^\.[a-z0-9]{1,10}$")


def _invalid(message: str) -> McpError:
    return McpError(MCP_CATALOG_INVALID, f"Invalid MCP catalog: {message}")


@dataclass(frozen=True)
class McpRequiredInput:
    name: str
    input_type: str
    required: bool
    user_approval_required: bool


@dataclass(frozen=True)
class McpSelectionHints:
    """Deterministic phrase/extension hints for the Phase G.1 server selector only.

    Never consulted by the installer, activator, or any runtime-lifecycle code —
    scoring data, not trust or execution data. `actions` and `extensions` map a
    granular capability id to the lowercase phrases/extensions that suggest it.
    """

    explicit_names: Tuple[str, ...] = ()
    actions: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    extensions: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "explicit_names": list(self.explicit_names),
            "actions": {k: list(v) for k, v in sorted(self.actions.items())},
            "extensions": {k: list(v) for k, v in sorted(self.extensions.items())},
        }


@dataclass(frozen=True)
class McpCatalogEntry:
    catalog_id: str
    server_id: str
    display_name: str
    description: str
    capabilities: Tuple[str, ...]
    risk_category: str
    transport: str
    required_runtimes: Tuple[str, ...]
    installer_type: str
    package_name: str
    package_version: str
    entrypoint_relative: str
    required_inputs: Tuple[McpRequiredInput, ...]
    expected_tools: Tuple[str, ...]
    default_tool_policy: McpToolPolicy
    # Phase G.1 — optional; absent (empty tuple / empty hints) means this entry is
    # not yet selectable by the capability selector, but loads exactly as before.
    granular_capabilities: Tuple[str, ...] = ()
    selection_hints: McpSelectionHints = field(default_factory=McpSelectionHints)

    @property
    def package_source(self) -> str:
        return f"{self.installer_type}:{self.package_name}@{self.package_version}"

    def requires_directory(self) -> bool:
        return any(i.input_type == "directory" and i.required for i in self.required_inputs)

    def required_environment_variables(self) -> Tuple[str, ...]:
        return tuple(i.name for i in self.required_inputs if i.input_type == "environment_variable")


@dataclass(frozen=True)
class McpCatalog:
    catalog_version: int
    entries: Dict[str, McpCatalogEntry] = field(default_factory=dict)

    def get(self, catalog_id) -> Optional[McpCatalogEntry]:
        return self.entries.get(catalog_id)

    def has(self, catalog_id) -> bool:
        return catalog_id in self.entries

    def capability_summaries(self) -> Tuple[dict, ...]:
        """Compact, bounded capability summaries (safe to place in a prompt)."""
        return tuple(
            {
                "catalog_id": e.catalog_id,
                "display_name": e.display_name,
                "capabilities": list(e.capabilities),
                "description": e.description[:200],
            }
            for e in sorted(self.entries.values(), key=lambda x: x.catalog_id)
        )

    def find_by_capability(self, capability) -> Optional[McpCatalogEntry]:
        for entry in sorted(self.entries.values(), key=lambda x: x.catalog_id):
            if capability in entry.capabilities:
                return entry
        return None


def _string(raw, key, required=True, max_len=200):
    value = raw.get(key)
    if value is None and not required:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise _invalid(f"'{key}' must be a non-empty string.")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise _invalid(f"'{key}' contains illegal characters.")
    if len(value) > max_len:
        raise _invalid(f"'{key}' exceeds {max_len} characters.")
    return value


def _is_safe_relative_path(value):
    """True only for a genuinely relative path with no '..' segment.

    `os.path.isabs` alone is not enough: on Windows (Python 3.13+) a leading-slash
    path like '/etc/passwd' is NOT reported as absolute, yet os.path.join would
    still escape the install root with it. So root-relative and drive-qualified
    forms are rejected explicitly.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or os.path.isabs(value):
        return False
    if re.match(r"^[A-Za-z]:", normalized):  # drive-qualified, e.g. C:foo
        return False
    return ".." not in normalized.split("/")


def _build_policy(raw, catalog_id) -> McpToolPolicy:
    if not isinstance(raw, dict):
        raise _invalid(f"{catalog_id}: 'default_tool_policy' must be an object.")
    default_permission = ToolPermission.coerce(raw.get("default_permission", "denied"))
    if default_permission is not ToolPermission.DENIED:
        # A catalog must not hand out blanket access to undeclared tools.
        raise _invalid(f"{catalog_id}: 'default_permission' must be 'denied'.")
    tools_raw = raw.get("tools", {})
    if not isinstance(tools_raw, dict):
        raise _invalid(f"{catalog_id}: 'default_tool_policy.tools' must be an object.")
    if len(tools_raw) > MAX_TOOLS_PER_POLICY:
        raise _invalid(f"{catalog_id}: too many tools in the default policy.")
    entries = {}
    for name, spec in tools_raw.items():
        if not isinstance(name, str) or not TOOL_NAME_RE.match(name):
            raise _invalid(f"{catalog_id}: tool name {name!r} is invalid.")
        if not isinstance(spec, dict):
            raise _invalid(f"{catalog_id}: tool {name!r} must map to an object.")
        entries[name] = McpToolPolicyEntry(
            enabled=bool(spec.get("enabled", False)),
            # Fail closed: an unknown permission string becomes DENIED.
            permission=ToolPermission.coerce(spec.get("permission", "denied")),
        )
    return McpToolPolicy(default_permission=default_permission, tools=entries)


def _build_required_inputs(raw, catalog_id) -> Tuple[McpRequiredInput, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise _invalid(f"{catalog_id}: 'required_inputs' must be a list.")
    inputs = []
    for item in raw:
        if not isinstance(item, dict):
            raise _invalid(f"{catalog_id}: each required input must be an object.")
        name = item.get("name")
        input_type = item.get("type")
        if not isinstance(name, str) or not name.strip():
            raise _invalid(f"{catalog_id}: a required input is missing 'name'.")
        if input_type not in SUPPORTED_INPUT_TYPES:
            raise _invalid(f"{catalog_id}: unsupported required-input type {input_type!r}.")
        if input_type == "environment_variable" and not ENV_NAME_RE.match(name):
            raise _invalid(f"{catalog_id}: environment variable name {name!r} is invalid.")
        inputs.append(McpRequiredInput(
            name=name,
            input_type=input_type,
            required=bool(item.get("required", False)),
            user_approval_required=bool(item.get("user_approval_required", True)),
        ))
    return tuple(inputs)


def _build_granular_capabilities(raw, catalog_id) -> Tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise _invalid(f"{catalog_id}: 'granular_capabilities' must be a list.")
    if len(raw) > MAX_GRANULAR_CAPABILITIES:
        raise _invalid(f"{catalog_id}: too many granular capabilities.")
    seen = set()
    out = []
    for cap in raw:
        if not isinstance(cap, str) or not CAPABILITY_RE.match(cap):
            raise _invalid(f"{catalog_id}: granular capability {cap!r} must match ^[a-z0-9_]+$.")
        if cap in seen:
            raise _invalid(f"{catalog_id}: duplicate granular capability {cap!r}.")
        seen.add(cap)
        out.append(cap)
    return tuple(out)


def _build_selection_hints(raw, catalog_id, granular_capabilities) -> McpSelectionHints:
    if raw is None:
        return McpSelectionHints()
    if not isinstance(raw, dict):
        raise _invalid(f"{catalog_id}: 'selection_hints' must be an object.")
    known = set(granular_capabilities)

    explicit_raw = raw.get("explicit_names", [])
    if not isinstance(explicit_raw, list):
        raise _invalid(f"{catalog_id}: 'selection_hints.explicit_names' must be a list.")
    if len(explicit_raw) > MAX_EXPLICIT_NAMES:
        raise _invalid(f"{catalog_id}: too many selection_hints.explicit_names.")
    explicit_names = []
    for name in explicit_raw:
        if not isinstance(name, str) or not name.strip():
            raise _invalid(f"{catalog_id}: an explicit_names entry must be a non-empty string.")
        normalized = name.strip().lower()
        if len(normalized) > 80:
            raise _invalid(f"{catalog_id}: an explicit_names entry is too long.")
        explicit_names.append(normalized)

    def _phrase_map(key, max_per_capability, validate_phrase):
        section = raw.get(key, {})
        if not isinstance(section, dict):
            raise _invalid(f"{catalog_id}: 'selection_hints.{key}' must be an object.")
        result = {}
        for cap, phrases in section.items():
            if cap not in known:
                # A hint for a capability the entry never declared is a malformed,
                # inconsistent catalog entry — fail closed rather than silently drop.
                raise _invalid(f"{catalog_id}: selection_hints.{key} references undeclared "
                               f"capability {cap!r}.")
            if not isinstance(phrases, list) or not phrases:
                raise _invalid(f"{catalog_id}: selection_hints.{key}[{cap!r}] must be a non-empty list.")
            if len(phrases) > max_per_capability:
                raise _invalid(f"{catalog_id}: too many selection_hints.{key}[{cap!r}] entries.")
            normalized_phrases = []
            for phrase in phrases:
                validate_phrase(phrase)
                normalized_phrases.append(phrase.strip().lower())
            result[cap] = tuple(normalized_phrases)
        return result

    def _validate_action_phrase(phrase):
        if not isinstance(phrase, str) or not phrase.strip() or len(phrase) > 80:
            raise _invalid(f"{catalog_id}: an action-hint phrase must be a short, non-empty string.")

    def _validate_extension(phrase):
        if not isinstance(phrase, str) or not _EXTENSION_RE.match(phrase.strip().lower()):
            raise _invalid(f"{catalog_id}: extension hint {phrase!r} must look like '.ext'.")

    actions = _phrase_map("actions", MAX_HINT_PHRASES_PER_CAPABILITY, _validate_action_phrase)
    extensions = _phrase_map("extensions", MAX_EXTENSIONS_PER_CAPABILITY, _validate_extension)

    # Reject any unrecognized top-level key so a future field cannot be silently
    # ignored — fail closed on malformed/unknown shape rather than load it partially.
    known_keys = {"explicit_names", "actions", "extensions"}
    unknown = set(raw) - known_keys
    if unknown:
        raise _invalid(f"{catalog_id}: selection_hints has unknown field(s): {sorted(unknown)}.")

    return McpSelectionHints(explicit_names=tuple(explicit_names), actions=actions, extensions=extensions)


def build_entry(catalog_id, raw) -> McpCatalogEntry:
    """Validate one catalog entry (fail closed)."""
    if not isinstance(catalog_id, str) or not CATALOG_ID_RE.match(catalog_id):
        raise _invalid(f"catalog id {catalog_id!r} must match ^[a-zA-Z0-9_-]+$.")
    if not isinstance(raw, dict):
        raise _invalid(f"{catalog_id}: entry must be an object.")

    server_id = _string(raw, "server_id", max_len=64)
    if not SERVER_ID_RE.match(server_id):
        raise _invalid(f"{catalog_id}: 'server_id' must match ^[a-zA-Z0-9_-]+$.")

    transport = raw.get("transport", "stdio")
    if transport not in SUPPORTED_TRANSPORTS:
        raise _invalid(f"{catalog_id}: transport {transport!r} is not supported (only 'stdio').")

    capabilities_raw = raw.get("capabilities")
    if not isinstance(capabilities_raw, list) or not capabilities_raw:
        raise _invalid(f"{catalog_id}: 'capabilities' must be a non-empty list.")
    if len(capabilities_raw) > MAX_CAPABILITIES:
        raise _invalid(f"{catalog_id}: too many capabilities.")
    capabilities = []
    for cap in capabilities_raw:
        if not isinstance(cap, str) or not CAPABILITY_RE.match(cap):
            raise _invalid(f"{catalog_id}: capability {cap!r} must match ^[a-z0-9_]+$.")
        capabilities.append(cap)

    installer = raw.get("installer")
    if not isinstance(installer, dict):
        raise _invalid(f"{catalog_id}: 'installer' must be an object.")
    installer_type = installer.get("type")
    if installer_type not in SUPPORTED_INSTALLERS:
        raise _invalid(f"{catalog_id}: installer type {installer_type!r} is not supported.")
    package_name = _string(installer, "package", max_len=214)
    version = installer.get("version")
    if not isinstance(version, str) or not EXACT_VERSION_RE.match(version):
        raise _invalid(f"{catalog_id}: installer version must be an EXACT pinned version "
                       f"(got {version!r}); ranges, tags, and wildcards are rejected.")
    entrypoint = _string(installer, "entrypoint", max_len=400)
    if not _is_safe_relative_path(entrypoint):
        raise _invalid(f"{catalog_id}: 'entrypoint' must be a relative path without '..'.")

    # npm lifecycle scripts are structurally disabled in Phase F. A catalog entry
    # may omit the key or set it to exactly false; anything else is rejected, so
    # there is no path — catalog, config, env, or LLM — that turns them on.
    if installer.get("allow_lifecycle_scripts", False) is not False:
        raise _invalid(f"{catalog_id}: npm lifecycle scripts cannot be enabled; "
                       "'allow_lifecycle_scripts' must be absent or false.")

    runtimes_raw = raw.get("required_runtimes", [])
    if not isinstance(runtimes_raw, list):
        raise _invalid(f"{catalog_id}: 'required_runtimes' must be a list.")
    runtimes = []
    for name in runtimes_raw:
        if not isinstance(name, str) or not name.strip() or any(
                c in name for c in ("\x00", "\n", "\r", os.sep, "/")):
            raise _invalid(f"{catalog_id}: runtime {name!r} is invalid.")
        runtimes.append(name)

    expected_raw = raw.get("expected_tools", [])
    if not isinstance(expected_raw, list):
        raise _invalid(f"{catalog_id}: 'expected_tools' must be a list.")
    expected = []
    for name in expected_raw:
        if not isinstance(name, str) or not TOOL_NAME_RE.match(name):
            raise _invalid(f"{catalog_id}: expected tool {name!r} is invalid.")
        expected.append(name)

    granular_capabilities = _build_granular_capabilities(raw.get("granular_capabilities"), catalog_id)
    selection_hints = _build_selection_hints(raw.get("selection_hints"), catalog_id, granular_capabilities)

    return McpCatalogEntry(
        catalog_id=catalog_id,
        server_id=server_id,
        display_name=_string(raw, "display_name", max_len=120),
        # Untrusted-ish display text: sanitized + bounded before it can reach a prompt.
        description=sanitize_description(raw.get("description", "")),
        capabilities=tuple(capabilities),
        risk_category=_string(raw, "risk_category", max_len=64),
        transport=transport,
        required_runtimes=tuple(runtimes),
        installer_type=installer_type,
        package_name=package_name,
        package_version=version,
        entrypoint_relative=entrypoint,
        required_inputs=_build_required_inputs(raw.get("required_inputs"), catalog_id),
        expected_tools=tuple(expected),
        default_tool_policy=_build_policy(raw.get("default_tool_policy"), catalog_id),
        granular_capabilities=granular_capabilities,
        selection_hints=selection_hints,
    )


def build_catalog(raw: dict) -> McpCatalog:
    if not isinstance(raw, dict):
        raise _invalid("the catalog root must be an object.")
    version = raw.get("catalog_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise _invalid("'catalog_version' must be a positive integer.")
    servers = raw.get("servers")
    if not isinstance(servers, dict) or not servers:
        raise _invalid("'servers' must be a non-empty object.")
    if len(servers) > MAX_CATALOG_ENTRIES:
        raise _invalid("the catalog has too many entries.")

    entries = {}
    seen_server_ids = {}
    for catalog_id, spec in servers.items():
        entry = build_entry(catalog_id, spec)
        if entry.server_id in seen_server_ids:
            raise _invalid(f"duplicate server_id {entry.server_id!r} "
                           f"({catalog_id} and {seen_server_ids[entry.server_id]}).")
        seen_server_ids[entry.server_id] = catalog_id
        entries[catalog_id] = entry
    return McpCatalog(catalog_version=version, entries=entries)


def default_catalog_path(base_dir=None):
    base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, app_config.mcp_catalog_path())


def load_catalog(path=None, base_dir=None) -> McpCatalog:
    """Load + validate the trusted catalog. Raises MCP_CATALOG_INVALID on any problem."""
    path = path or default_catalog_path(base_dir)
    if not os.path.isfile(path):
        raise _invalid("the catalog file was not found.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        raise _invalid("the catalog could not be read or parsed as JSON.") from e
    return build_catalog(raw)
