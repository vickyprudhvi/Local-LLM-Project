"""Phase F — the managed installed-server registry (application-owned state).

Separate from the trusted catalog: the catalog says what MAY be installed, this
says what IS installed. Written atomically (temp file -> fsync -> os.replace) so an
interrupted write cannot leave a half-written registry, and a corrupt registry
raises MCP_REGISTRY_CORRUPT rather than being silently overwritten.

The LLM never edits this file; only deterministic code here does.
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import tools.config as app_config
from mcp_layer.errors import McpError
from tools.models import MCP_REGISTRY_CORRUPT

REGISTRY_VERSION = 1
REGISTRY_FILENAME = "installed_servers.json"

STATUS_INSTALLED = "installed"
STATUS_DISABLED = "disabled"
STATUS_FAILED = "failed"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class InstalledServer:
    catalog_id: str
    installed_version: str
    status: str
    install_directory: str
    configuration_path: str
    installed_at: str
    last_validated_at: Optional[str] = None
    last_validation_result: Optional[str] = None
    approved_directories: tuple = ()
    # Phase G.3 (Task 12) — optional; absent on any registry entry written before
    # this phase (including the existing Filesystem install), so old state keeps
    # loading exactly as before with these simply unset.
    installer_type: Optional[str] = None
    catalog_entry_hash: Optional[str] = None
    lock_hash: Optional[str] = None
    expected_tools_hash: Optional[str] = None
    tool_policy_hash: Optional[str] = None
    last_known_good_version: Optional[str] = None

    def to_dict(self):
        data = asdict(self)
        data["approved_directories"] = list(self.approved_directories)
        return data


def managed_root(base_dir=None, root=None):
    base_dir = base_dir or _REPO_ROOT
    root = root or app_config.mcp_managed_root()
    return os.path.join(base_dir, str(root))


def registry_path(base_dir=None, root=None):
    return os.path.join(managed_root(base_dir, root), REGISTRY_FILENAME)


def atomic_write_json(path, payload):
    """Write JSON atomically: temp file in the same directory, fsync, then replace."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=directory, text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass  # best effort; not all filesystems support it
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_registry(path=None, base_dir=None, root=None) -> Dict[str, InstalledServer]:
    """Load the registry. Missing file -> empty. Corrupt file -> MCP_REGISTRY_CORRUPT."""
    path = path or registry_path(base_dir, root)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        raise McpError(MCP_REGISTRY_CORRUPT,
                       "The MCP server registry could not be read; it was left untouched.") from e
    if not isinstance(raw, dict) or not isinstance(raw.get("servers"), dict):
        raise McpError(MCP_REGISTRY_CORRUPT,
                       "The MCP server registry has an unexpected structure; it was left untouched.")

    servers = {}
    for server_id, spec in raw["servers"].items():
        if not isinstance(spec, dict):
            raise McpError(MCP_REGISTRY_CORRUPT, "A registry entry has an unexpected structure.")
        try:
            servers[server_id] = InstalledServer(
                catalog_id=spec["catalog_id"],
                installed_version=spec["installed_version"],
                status=spec.get("status", STATUS_INSTALLED),
                install_directory=spec["install_directory"],
                configuration_path=spec["configuration_path"],
                installed_at=spec.get("installed_at", ""),
                last_validated_at=spec.get("last_validated_at"),
                last_validation_result=spec.get("last_validation_result"),
                approved_directories=tuple(spec.get("approved_directories", ())),
                installer_type=spec.get("installer_type"),
                catalog_entry_hash=spec.get("catalog_entry_hash"),
                lock_hash=spec.get("lock_hash"),
                expected_tools_hash=spec.get("expected_tools_hash"),
                tool_policy_hash=spec.get("tool_policy_hash"),
                last_known_good_version=spec.get("last_known_good_version"),
            )
        except KeyError as e:
            raise McpError(MCP_REGISTRY_CORRUPT,
                           f"A registry entry is missing required field {e.args[0]!r}.") from e
    return servers


def save_registry(servers: Dict[str, InstalledServer], path=None, base_dir=None, root=None):
    path = path or registry_path(base_dir, root)
    atomic_write_json(path, {
        "registry_version": REGISTRY_VERSION,
        "servers": {sid: entry.to_dict() for sid, entry in sorted(servers.items())},
    })
    return path


def get_installed(server_id, path=None, base_dir=None, root=None) -> Optional[InstalledServer]:
    return load_registry(path, base_dir, root).get(server_id)


def upsert(server_id, entry: InstalledServer, path=None, base_dir=None, root=None):
    servers = load_registry(path, base_dir, root)
    servers[server_id] = entry
    save_registry(servers, path, base_dir, root)
    return entry


def remove(server_id, path=None, base_dir=None, root=None):
    servers = load_registry(path, base_dir, root)
    removed = servers.pop(server_id, None)
    save_registry(servers, path, base_dir, root)
    return removed


def set_status(server_id, status, path=None, base_dir=None, root=None,
               validation_result=None):
    from dataclasses import replace

    servers = load_registry(path, base_dir, root)
    entry = servers.get(server_id)
    if entry is None:
        return None
    updated = replace(
        entry,
        status=status,
        last_validated_at=utc_now() if validation_result else entry.last_validated_at,
        last_validation_result=validation_result or entry.last_validation_result,
    )
    servers[server_id] = updated
    save_registry(servers, path, base_dir, root)
    return updated
