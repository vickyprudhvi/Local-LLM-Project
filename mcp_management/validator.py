"""Phase F — post-install validation of a freshly installed MCP server.

Reuses the Phase E machinery rather than duplicating it: `external.start_server`
performs executable + working-directory validation, the isolated-environment
launch (shell=False), `initialize`, and `notifications/initialized`;
`discovery.plan_registration` applies schema limits and the LOCAL tool policy.

Every check is recorded so a failure names the reason without leaking server
output. The write smoke test is OFF by default and, when enabled, only touches a
disposable installer-owned file.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

from mcp_layer.client import PROTOCOL_VERSION
from mcp_layer.discovery import plan_registration
from mcp_layer.errors import McpError
from mcp_layer.external import start_server
from tools.models import MCP_POST_INSTALL_VALIDATION_FAILED

SMOKE_TEST_FILENAME = ".mcp_install_smoke_test.tmp"


@dataclass
class ValidationReport:
    ok: bool = False
    checks: list = field(default_factory=list)
    discovered_tool_count: int = 0
    registered_tool_count: int = 0
    denied_tool_count: int = 0
    protocol_version: Optional[str] = None
    server_name: Optional[str] = None
    failure: Optional[str] = None

    def record(self, name, passed, detail=""):
        self.checks.append({"check": name, "ok": bool(passed), "detail": detail[:200]})
        return passed

    def summary(self):
        return {
            "ok": self.ok,
            "checks": list(self.checks),
            "discovered_tool_count": self.discovered_tool_count,
            "registered_tool_count": self.registered_tool_count,
            "denied_tool_count": self.denied_tool_count,
            "protocol_version": self.protocol_version,
            "server_name": self.server_name,
            "failure": self.failure,
        }


def _fail(report, name, detail):
    report.record(name, False, detail)
    report.ok = False
    report.failure = detail
    raise McpError(MCP_POST_INSTALL_VALIDATION_FAILED, detail)


def validate_installation(config, plan, expected_tools=(), entrypoint=None,
                          base_dir=None, run_write_smoke_test=False,
                          start_server_fn=None):
    """Start the installed server, validate it, then shut it down cleanly.

    `config` is the generated (already Phase E-validated) McpServerConfig.
    Returns a ValidationReport; raises MCP_POST_INSTALL_VALIDATION_FAILED on any
    failed check. The server is always shut down, so no orphan process remains.
    """
    report = ValidationReport()
    start = start_server_fn or start_server

    # 1. Entrypoint exists (the argv target the generated config points at).
    if entrypoint is not None and not os.path.isfile(entrypoint):
        _fail(report, "entrypoint_exists", "The installed entrypoint is missing.")
    report.record("entrypoint_exists", True)

    client = None
    try:
        # 2-3-5. Launch (shell=False), initialize, notifications/initialized.
        try:
            client = start(config, base_dir=base_dir, allow_create=True)
        except McpError as e:
            _fail(report, "server_starts", f"The server failed to start ({e.code}).")
        report.record("server_starts", True)
        report.record("initialize", True)

        # 4. Protocol compatibility.
        report.protocol_version = client.protocol_version
        report.server_name = (client.server_info or {}).get("name")
        if client.protocol_version and client.protocol_version != PROTOCOL_VERSION:
            # Different but present: accept with a recorded note rather than failing,
            # since MCP servers may negotiate an older/newer dated revision.
            report.record("protocol_compatible", True,
                          f"server={client.protocol_version} client={PROTOCOL_VERSION}")
        elif not client.protocol_version:
            _fail(report, "protocol_compatible", "The server did not report a protocol version.")
        else:
            report.record("protocol_compatible", True)

        # 6. tools/list.
        try:
            raw_tools = client.list_tools(timeout=config.startup_timeout_seconds)
        except McpError as e:
            _fail(report, "tools_list", f"tools/list failed ({e.code}).")
        report.record("tools_list", True, f"{len(raw_tools)} tool(s)")

        # 7. At least one expected core tool is present.
        discovered_names = {t.get("name") for t in raw_tools if isinstance(t, dict)}
        expected = tuple(expected_tools or ())
        if expected and not (set(expected) & discovered_names):
            _fail(report, "expected_tools_present",
                  "The server does not expose the expected core tools.")
        report.record("expected_tools_present", True,
                      ", ".join(sorted(set(expected) & discovered_names)) if expected else "none required")

        # 8-9. Schema limits + LOCAL policy (unknown tools land in diagnostics as denied).
        registrations, diagnostics = plan_registration(raw_tools, config)
        report.discovered_tool_count = len(registrations) + len(diagnostics)
        report.registered_tool_count = len(registrations)
        report.denied_tool_count = sum(1 for _, _, cat in diagnostics if cat == "denied")
        if not registrations:
            _fail(report, "policy_applied", "No discovered tool passed the local policy.")
        report.record("policy_applied", True,
                      f"registered={len(registrations)} denied={report.denied_tool_count}")

        registered_names = {r["remote_name"] for r in registrations}

        # 10-11. The server's allowed roots must match the approved directories.
        approved = {os.path.realpath(str(d)) for d in plan.requested_directories}
        if approved and "list_allowed_directories" in registered_names:
            try:
                result = client.call_tool("list_allowed_directories", {},
                                          timeout=config.call_timeout_seconds)
            except McpError as e:
                _fail(report, "allowed_directories", f"list_allowed_directories failed ({e.code}).")
            reported = _extract_paths(result)
            if reported:
                extra = {p for p in reported if p not in approved}
                if extra:
                    _fail(report, "allowed_directories",
                          "The server reports access beyond the approved directories.")
            report.record("allowed_directories", True, f"{len(reported)} root(s)")
        else:
            report.record("allowed_directories", True, "not applicable")

        # 12. Read smoke test (a read-only listing of an approved root).
        if approved and "list_directory" in registered_names:
            target = sorted(approved)[0]
            try:
                client.call_tool("list_directory", {"path": target},
                                 timeout=config.call_timeout_seconds)
            except McpError as e:
                _fail(report, "read_smoke_test", f"The read smoke test failed ({e.code}).")
            report.record("read_smoke_test", True)
        else:
            report.record("read_smoke_test", True, "not applicable")

        # 13. Write smoke test: opt-in only, and only a disposable installer-owned file.
        if run_write_smoke_test and approved and "write_file" in registered_names:
            target = os.path.join(sorted(approved)[0], SMOKE_TEST_FILENAME)
            try:
                client.call_tool("write_file",
                                 {"path": target, "content": "mcp install smoke test"},
                                 timeout=config.call_timeout_seconds)
            except McpError as e:
                _fail(report, "write_smoke_test", f"The write smoke test failed ({e.code}).")
            finally:
                try:
                    os.unlink(target)
                except OSError:
                    pass
            report.record("write_smoke_test", True)
        else:
            report.record("write_smoke_test", True, "skipped (not enabled)")

        report.ok = True
        return report
    finally:
        # 14. Always shut down — no orphan process, even on a failed check.
        if client is not None:
            try:
                client.shutdown()
            except Exception:  # noqa: BLE001 — never mask the validation outcome
                pass


def _extract_paths(result):
    """Canonical directory paths from a list_allowed_directories result.

    Servers report this differently: a list under `directories`/`roots`/`paths`, or
    (as the official filesystem server does) a newline-delimited string under
    `content`/`text` preceded by an "Allowed directories:" header. All of those are
    parsed, because this feeds a security check — silently finding nothing would
    make the check pass without verifying anything.
    """
    paths = set()
    if not isinstance(result, dict):
        return paths

    candidates = []
    for key in ("directories", "allowed_directories", "roots", "paths"):
        value = result.get(key)
        if isinstance(value, list):
            candidates.extend(value)
        elif isinstance(value, str):
            candidates.extend(value.splitlines())
    for key in ("content", "text"):
        value = result.get(key)
        if isinstance(value, str):
            candidates.extend(value.splitlines())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    candidates.extend(item["text"].splitlines())
                elif isinstance(item, str):
                    candidates.extend(item.splitlines())

    for item in candidates:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().strip("-").strip()
        if not cleaned:
            continue
        # Skip header/label lines such as "Allowed directories:".
        if cleaned.endswith(":") or cleaned.lower().startswith("allowed director"):
            continue
        paths.add(os.path.realpath(cleaned))
    return paths
