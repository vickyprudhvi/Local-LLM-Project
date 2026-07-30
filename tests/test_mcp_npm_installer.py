"""Phase F — the npm installer: exact pins, isolation, shell=False, bounded output."""

import json
import os
import subprocess

import pytest

from mcp_layer.errors import McpError
from mcp_management import npm_installer
from mcp_management.npm_installer import (
    build_npm_argv,
    check_runtimes,
    install_package,
    lockfile_hash,
    resolve_runtime,
    sanitize_output,
    validate_entrypoint,
)
from mcp_management.planner import build_plan
from tests.mcp_provisioning_helpers import FakeNpm, make_catalog, workspace_with_file
from tools.models import (
    MCP_ENTRYPOINT_NOT_FOUND,
    MCP_INSTALLATION_FAILED,
    MCP_INSTALLATION_TIMEOUT,
    MCP_RUNTIME_MISSING,
)


@pytest.fixture
def plan(tmp_path):
    entry = make_catalog().get("official-filesystem")
    return build_plan(entry, requested_directories=[workspace_with_file(tmp_path)],
                      base_dir=str(tmp_path / "repo"))


# ---- argv construction ----

def test_argv_pins_the_exact_version(plan):
    argv = build_npm_argv(plan, "/usr/bin/npm")
    assert argv[:3] == ["/usr/bin/npm", "install",
                        f"{plan.package_name}@{plan.package_version}"]
    assert "--save-exact" in argv


def test_argv_never_global(plan):
    argv = build_npm_argv(plan, "/usr/bin/npm")
    assert "-g" not in argv and "--global" not in argv


def test_argv_always_ignores_lifecycle_scripts(plan):
    """Unconditional: Phase F never runs package lifecycle scripts."""
    assert "--ignore-scripts" in build_npm_argv(plan, "npm")


def test_catalog_cannot_enable_lifecycle_scripts(tmp_path):
    """A catalog entry requesting lifecycle scripts is rejected outright."""
    from mcp_layer.errors import McpError
    from tools.models import MCP_CATALOG_INVALID

    with pytest.raises(McpError) as e:
        make_catalog(allow_lifecycle_scripts=True)
    assert e.value.code == MCP_CATALOG_INVALID
    # Explicit false (or omission) remains acceptable.
    assert make_catalog(allow_lifecycle_scripts=False).get("official-filesystem")
    assert make_catalog().get("official-filesystem")


def test_plan_has_no_lifecycle_script_switch(plan):
    """There is no field an attacker or caller could flip."""
    assert not hasattr(plan, "allow_lifecycle_scripts")
    assert "allow_lifecycle_scripts" not in plan.security_fields()


def test_argv_contains_no_shell_metacharacters(plan):
    argv = build_npm_argv(plan, "/usr/bin/npm")
    joined = " ".join(argv)
    for token in ("&&", "||", ";", "|", ">", "<", "$(", "`"):
        assert token not in joined


# ---- runtime discovery ----

def test_missing_runtime_reports_runtime_missing():
    with pytest.raises(McpError) as e:
        resolve_runtime("definitely_not_a_runtime_zzz")
    assert e.value.code == MCP_RUNTIME_MISSING
    # It must be clear nothing was installed on the user's behalf.
    assert "not be installed automatically" in e.value.message


def test_runtime_name_with_illegal_characters_rejected():
    for bad in ("node\x00", "node\n"):
        with pytest.raises(McpError) as e:
            resolve_runtime(bad)
        assert e.value.code == MCP_RUNTIME_MISSING


def test_check_runtimes_resolves_node_and_npm(plan):
    resolved = check_runtimes(plan)
    assert set(resolved) == {"node", "npm"}
    assert all(os.path.isabs(p) for p in resolved.values())


# ---- installation ----

def test_install_uses_isolated_directory_and_seeds_package_json(plan, tmp_path):
    npm = FakeNpm()
    target = tmp_path / "staging"
    install_package(plan, str(target), npm_executable="npm", runner=npm, timeout=30)
    call = npm.calls[0]
    assert call["cwd"] == str(target)
    manifest = json.load(open(target / "package.json", encoding="utf-8"))
    # A private manifest keeps npm from walking up into another project.
    assert manifest["private"] is True
    assert manifest["name"].startswith("mcp-managed-")


def test_install_returns_bounded_sanitized_output(plan, tmp_path):
    npm = FakeNpm(stdout="ok\x07" + "y" * 50000, stderr="warn\x00" + "z" * 50000)
    record = install_package(plan, str(tmp_path / "staging"), npm_executable="npm",
                             runner=npm, timeout=30)
    assert len(record["stdout"]) <= 8 * 1024
    assert len(record["stderr"]) <= 8 * 1024
    assert "\x07" not in record["stdout"] and "\x00" not in record["stderr"]


def test_install_record_omits_the_resolved_executable(plan, tmp_path):
    npm = FakeNpm()
    record = install_package(plan, str(tmp_path / "staging"),
                             npm_executable="/opt/secret-path/npm", runner=npm, timeout=30)
    assert "/opt/secret-path/npm" not in json.dumps(record["argv_tail"])


def test_install_failure_raises_and_does_not_retry(plan, tmp_path):
    npm = FakeNpm(fail=True, returncode=1)
    with pytest.raises(McpError) as e:
        install_package(plan, str(tmp_path / "staging"), npm_executable="npm",
                        runner=npm, timeout=30)
    assert e.value.code == MCP_INSTALLATION_FAILED
    assert len(npm.calls) == 1  # never a second attempt with another version/tag


def test_install_timeout_is_normalized(plan, tmp_path):
    npm = FakeNpm(timeout=True)
    with pytest.raises(McpError) as e:
        install_package(plan, str(tmp_path / "staging"), npm_executable="npm",
                        runner=npm, timeout=1)
    assert e.value.code == MCP_INSTALLATION_TIMEOUT


def test_default_runner_uses_shell_false(plan, tmp_path, monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        raise OSError("stop before launching npm")

    monkeypatch.setattr(npm_installer.subprocess, "run", fake_run)
    with pytest.raises(McpError):
        install_package(plan, str(tmp_path / "staging"), npm_executable="npm", timeout=5)
    assert captured["kwargs"]["shell"] is False
    assert isinstance(captured["argv"], list)


def test_child_environment_excludes_secrets(plan, tmp_path, monkeypatch):
    monkeypatch.setenv("PHASE_F_SECRET", "do-not-expose")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    npm = FakeNpm()
    install_package(plan, str(tmp_path / "staging"), npm_executable="npm",
                    runner=npm, timeout=30)
    env = npm.calls[0]["env"]
    assert "PHASE_F_SECRET" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "do-not-expose" not in json.dumps(env)


# ---- entrypoint validation ----

def test_entrypoint_validation_succeeds_after_install(plan, tmp_path):
    target = tmp_path / "staging"
    install_package(plan, str(target), npm_executable="npm", runner=FakeNpm(), timeout=30)
    entrypoint = validate_entrypoint(plan, str(target))
    assert os.path.isfile(entrypoint)
    assert lockfile_hash(str(target))


def test_missing_entrypoint_reports_not_found(plan, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(McpError) as e:
        validate_entrypoint(plan, str(empty))
    assert e.value.code == MCP_ENTRYPOINT_NOT_FOUND


def test_sanitize_output_bounds_and_strips_control_characters():
    assert "\x00" not in sanitize_output("a\x00b")
    assert len(sanitize_output("x" * 100000)) <= 8 * 1024
    assert sanitize_output(None) == ""
