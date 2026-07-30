"""Phase F — deterministic npm installation into an isolated managed directory.

The LLM never contributes to the command line. `build_npm_argv` derives every
argument from the validated plan: the exact pinned `package@version`, never a
range or tag, never `-g`, never a shell string. The install runs with
`shell=False` inside the plan's own directory (seeded with a private package.json
so npm cannot walk up into another project), with a bounded timeout and bounded,
sanitized output.

Lifecycle scripts are disabled (`--ignore-scripts`) unless the trusted catalog
entry opted in for that package.
"""

import json
import os
import re
import shutil
import subprocess

import tools.config as app_config
from mcp_layer.environment import build_child_environment
from mcp_layer.errors import McpError
from mcp_management.models import McpProvisioningPlan
from tools.models import (
    MCP_ENTRYPOINT_NOT_FOUND,
    MCP_INSTALLATION_FAILED,
    MCP_INSTALLATION_TIMEOUT,
    MCP_RUNTIME_MISSING,
)

MAX_OUTPUT_CHARS = 8 * 1024
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# npm needs its cache/config locations; these carry no secrets.
_NPM_ENV_ALLOWLIST = ("APPDATA", "LOCALAPPDATA", "npm_config_cache", "npm_config_registry")


def sanitize_output(text):
    """Bounded, control-character-free output for diagnostics/logs."""
    if not text:
        return ""
    return _CONTROL_RE.sub("", str(text))[-MAX_OUTPUT_CHARS:]


def resolve_runtime(name):
    """Resolve a required runtime executable, or raise MCP_RUNTIME_MISSING.

    Never installs or downloads a runtime, and never substitutes another program.
    """
    if not isinstance(name, str) or not name.strip():
        raise McpError(MCP_RUNTIME_MISSING, "A required runtime was not named.")
    if any(c in name for c in ("\x00", "\n", "\r")):
        raise McpError(MCP_RUNTIME_MISSING, "A required runtime name is invalid.")
    resolved = shutil.which(name)
    if not resolved:
        raise McpError(
            MCP_RUNTIME_MISSING,
            f"The required runtime {name!r} was not found. Install it separately and retry; "
            "it will not be installed automatically.",
        )
    return resolved


def check_runtimes(plan: McpProvisioningPlan):
    """Resolve every runtime the plan requires. Returns {name: absolute path}."""
    return {name: resolve_runtime(name) for name in plan.required_runtimes}


def build_npm_argv(plan: McpProvisioningPlan, npm_executable):
    """The exact argv for the install. Deterministic; no shell, no globals, no ranges."""
    spec = f"{plan.package_name}@{plan.package_version}"
    argv = [
        npm_executable, "install", spec,
        "--save-exact",   # never rewrite the pin into a range
        "--omit=dev",     # runtime dependencies only
        "--no-audit",
        "--no-fund",
        # Always: Phase F never runs package lifecycle scripts. There is no catalog,
        # config, environment, or model-supplied way to turn this off.
        "--ignore-scripts",
    ]
    return argv


def _seed_package_json(target_dir, plan):
    """A private package.json so npm treats `target_dir` as an isolated project."""
    manifest = {
        "name": f"mcp-managed-{plan.server_id}",
        "version": "0.0.0",
        "private": True,
        "description": f"Managed install of {plan.package_name} for MCP server "
                       f"{plan.server_id}. Generated; do not edit.",
    }
    with open(os.path.join(target_dir, "package.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def default_npm_runner(argv, cwd, env, timeout):
    """Run npm with shell=False and bounded output. Returns (rc, stdout, stderr)."""
    try:
        completed = subprocess.run(
            argv, cwd=cwd, env=env, shell=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise McpError(MCP_INSTALLATION_TIMEOUT,
                       f"The installation exceeded {timeout:g}s and was stopped.") from e
    except (OSError, ValueError) as e:
        raise McpError(MCP_INSTALLATION_FAILED, "The package manager could not be started.") from e
    return completed.returncode, completed.stdout, completed.stderr


def install_package(plan: McpProvisioningPlan, target_dir, npm_executable=None,
                    runner=None, timeout=None):
    """Install the plan's exact package into `target_dir`. Returns a bounded record.

    `target_dir` is created by the caller (a temporary staging directory during a
    transaction), so a failure never touches a promoted version directory.
    """
    runner = runner or default_npm_runner
    timeout = float(timeout or app_config.mcp_install_timeout())
    npm_executable = npm_executable or resolve_runtime("npm")

    os.makedirs(target_dir, exist_ok=True)
    _seed_package_json(target_dir, plan)

    argv = build_npm_argv(plan, npm_executable)
    env = build_child_environment(_NPM_ENV_ALLOWLIST)
    try:
        returncode, stdout, stderr = runner(argv, target_dir, env, timeout)
    except subprocess.TimeoutExpired as e:
        # Normalized here too, so a custom runner needn't know the error mapping.
        raise McpError(MCP_INSTALLATION_TIMEOUT,
                       f"The installation exceeded {timeout:g}s and was stopped.") from e

    record = {
        "argv_tail": argv[1:],  # omit the resolved executable path
        "returncode": returncode,
        "stdout": sanitize_output(stdout),
        "stderr": sanitize_output(stderr),
    }
    if returncode != 0:
        # Never retry with a different version, tag, or package.
        raise McpError(
            MCP_INSTALLATION_FAILED,
            f"Installing {plan.package_name}@{plan.package_version} failed "
            f"(exit code {returncode}).",
        )
    return record


def validate_entrypoint(plan: McpProvisioningPlan, install_dir):
    """Confirm the catalog's declared entrypoint exists inside the install directory."""
    root = os.path.realpath(install_dir)
    entry = os.path.realpath(os.path.join(root, plan.entrypoint_relative))
    if entry != root and not entry.startswith(root + os.sep):
        raise McpError(MCP_ENTRYPOINT_NOT_FOUND,
                       "The declared entrypoint escapes the install directory.")
    if not os.path.isfile(entry):
        raise McpError(MCP_ENTRYPOINT_NOT_FOUND,
                       "The installed package does not contain the expected entrypoint.")
    return entry


def lockfile_hash(install_dir):
    """SHA-256 of package-lock.json when present (recorded for audit)."""
    import hashlib

    path = os.path.join(install_dir, "package-lock.json")
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
