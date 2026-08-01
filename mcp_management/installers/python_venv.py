"""Phase G.3 Task 7 — the isolated Python virtual-environment installer backend.

Every candidate gets its OWN venv under
`<server_root>/candidates/<transaction_id>/venv`, created with `python -m venv`
from a trusted LOCAL interpreter (`sys.executable` — the interpreter already
running this process; never downloaded, never model-provided). Dependencies come
ONLY from a lock file the trusted catalog entry names (a path inside this
repository, committed, hash-bound into the plan) and are installed with
`pip install --require-hashes`, so every package — direct or transitive — must
carry a verified hash or the install fails closed. No editable installs, no
extras, no Git/path dependencies, no global or user site packages
(`PYTHONNOUSERSITE=1`), no arbitrary pip arguments from the model.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

import tools.config as app_config
from mcp_layer.environment import build_child_environment
from mcp_layer.errors import McpError
from mcp_management.installers.base import CandidateInstallation, McpLaunchSpec, ProvisioningTransaction
from tools.models import (
    MCP_EXECUTABLE_VALIDATION_FAILED,
    MCP_INSTALLATION_FAILED,
    MCP_INSTALLATION_TIMEOUT,
    MCP_LOCK_FILE_INVALID,
    MCP_PYTHON_VERSION_UNSUPPORTED,
)

INSTALLER_TYPE = "python_venv"
_VENV_DIRNAME = "venv"
MAX_OUTPUT_CHARS = 8 * 1024


def _venv_python(venv_dir):
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def _venv_console_script(venv_dir, script_name):
    """Path to a console-script shim inside the isolated venv."""
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", f"{script_name}.exe")
    return os.path.join(venv_dir, "bin", script_name)


def resolve_trusted_interpreter():
    """The ONLY interpreter this backend will ever use: the one already running
    this process. Never downloaded, never resolved from PATH by name, never
    supplied by the model."""
    return sys.executable


def check_python_constraint(interpreter, constraint):
    """Raise MCP_PYTHON_VERSION_UNSUPPORTED unless `interpreter`'s version
    satisfies the catalog's exact constraint string."""
    version_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    try:
        spec = SpecifierSet(constraint)
    except InvalidSpecifier as e:
        raise McpError(MCP_PYTHON_VERSION_UNSUPPORTED,
                       f"The catalog's python_constraint {constraint!r} is malformed.") from e
    if not spec.contains(version_str):
        raise McpError(MCP_PYTHON_VERSION_UNSUPPORTED,
                       f"Python {version_str} does not satisfy the required {constraint!r}.")
    return version_str


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lock_file_hash(lock_file_path):
    """SHA-256 of the committed lock file's exact content (bound into the plan
    hash — Task 3: changing the lock file invalidates any prior approval)."""
    if not os.path.isfile(lock_file_path):
        raise McpError(MCP_LOCK_FILE_INVALID, "The catalog's lock file was not found.")
    return _sha256_file(lock_file_path)


def _run(argv, cwd, env, timeout, error_code):
    try:
        completed = subprocess.run(
            argv, cwd=cwd, env=env, shell=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise McpError(MCP_INSTALLATION_TIMEOUT,
                       f"The step exceeded {timeout:g}s and was stopped.") from e
    except (OSError, ValueError) as e:
        raise McpError(error_code, "The command could not be started.") from e
    if completed.returncode != 0:
        stderr = (completed.stderr or "")[-MAX_OUTPUT_CHARS:]
        raise McpError(error_code, f"Command failed (exit code {completed.returncode}): {stderr}")
    return completed


class PythonVenvInstaller:
    installer_type = INSTALLER_TYPE

    def prepare_candidate(self, plan, catalog_entry, transaction: ProvisioningTransaction) -> CandidateInstallation:
        shutil.rmtree(transaction.candidate_directory, ignore_errors=True)
        os.makedirs(transaction.candidate_directory, exist_ok=True)

        interpreter = resolve_trusted_interpreter()
        check_python_constraint(interpreter, catalog_entry.python_constraint)

        venv_dir = os.path.join(transaction.candidate_directory, _VENV_DIRNAME)
        env = build_child_environment((), extra={"PYTHONNOUSERSITE": "1"})
        _run([interpreter, "-m", "venv", venv_dir],
             cwd=transaction.candidate_directory, env=env,
             timeout=app_config.mcp_install_timeout(), error_code=MCP_INSTALLATION_FAILED)

        venv_python = _venv_python(venv_dir)
        if not os.path.isfile(venv_python):
            raise McpError(MCP_EXECUTABLE_VALIDATION_FAILED,
                           "The candidate virtual environment has no interpreter.")
        return CandidateInstallation(
            transaction=transaction, install_directory=transaction.candidate_directory,
            extra={"venv_dir": venv_dir, "venv_python": venv_python})

    def install_candidate(self, candidate: CandidateInstallation, plan, catalog_entry) -> CandidateInstallation:
        lock_path = os.path.join(_repo_root(), catalog_entry.lock_file_relative)
        computed_hash = lock_file_hash(lock_path)
        if plan is not None and plan.lock_file_hash and plan.lock_file_hash != computed_hash:
            raise McpError(MCP_LOCK_FILE_INVALID,
                           "The lock file changed since this plan was approved.")

        self._enforce_lock_environment(catalog_entry)

        venv_python = candidate.extra["venv_python"]
        env = build_child_environment((), extra={"PYTHONNOUSERSITE": "1"})
        argv = [
            venv_python, "-m", "pip", "install",
            "--require-hashes", "--no-input", "--disable-pip-version-check",
            "--no-cache-dir",
            "-r", lock_path,
        ]
        if catalog_entry.install_options.get("no_deps"):
            argv.append("--no-deps")
        # A relative wheel path inside the lock file (portable across clones —
        # never a hardcoded absolute path) resolves against THIS cwd, per pip's
        # documented behavior for bare local-path requirements.
        _run(argv, cwd=_repo_root(), env=env,
             timeout=app_config.mcp_install_timeout(), error_code=MCP_INSTALLATION_FAILED)

        return CandidateInstallation(
            transaction=candidate.transaction, install_directory=candidate.install_directory,
            lock_hash=computed_hash, extra=dict(candidate.extra))

    def _enforce_lock_environment(self, catalog_entry) -> None:
        """Verify the current interpreter/platform matches the reviewed lock environment.

        The lock file is platform-specific; installation on an unreviewed platform
        is refused to prevent silent hash/environment skew.
        """
        lock_env = catalog_entry.lock_environment
        if not lock_env:
            return
        expected_python = lock_env.get("python_version")
        expected_platform = lock_env.get("platform")
        if expected_python:
            current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
            # Allow exact major.minor match; micro/patch is intentionally not bound.
            if current_python != expected_python:
                raise McpError(
                    MCP_PYTHON_VERSION_UNSUPPORTED,
                    f"Lock environment requires Python {expected_python}; running {current_python}.",
                )
        if expected_platform:
            current_platform = sysconfig.get_platform()
            # Normalize the two common spellings of the Windows platform tag:
            # sysconfig may report 'win-amd64' while pip/PEP 425 uses 'win_amd64'.
            normalized_current = current_platform.replace("-", "_")
            normalized_expected = expected_platform.replace("-", "_")
            if normalized_expected not in normalized_current and normalized_current not in normalized_expected:
                raise McpError(
                    MCP_PYTHON_VERSION_UNSUPPORTED,
                    f"Lock environment requires platform {expected_platform}; running {current_platform}.",
                )

    def validate_artifacts(self, candidate: CandidateInstallation, plan, catalog_entry) -> None:
        venv_python = candidate.extra.get("venv_python") or _venv_python(
            os.path.join(candidate.install_directory, _VENV_DIRNAME))
        if not os.path.isfile(venv_python):
            raise McpError(MCP_EXECUTABLE_VALIDATION_FAILED,
                           "The candidate interpreter is missing after installation.")
        # No executable this installer will ever launch resolves outside the
        # candidate's own venv directory.
        venv_dir_real = os.path.realpath(os.path.dirname(os.path.dirname(venv_python))
                                         if sys.platform == "win32"
                                         else os.path.dirname(os.path.dirname(venv_python)))
        if not os.path.realpath(venv_python).startswith(venv_dir_real + os.sep):
            raise McpError(MCP_EXECUTABLE_VALIDATION_FAILED,
                           "The candidate interpreter resolves outside its own environment.")

        check = subprocess.run(
            [venv_python, "-c",
             "import importlib.metadata as m, sys; "
             "sys.stdout.write(m.version(sys.argv[1]))", catalog_entry.package_name],
            cwd=candidate.transaction.candidate_directory,
            env=build_child_environment((), extra={"PYTHONNOUSERSITE": "1"}),
            shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=30, check=False,
        )
        if check.returncode != 0:
            raise McpError(MCP_EXECUTABLE_VALIDATION_FAILED,
                           f"The installed package {catalog_entry.package_name!r} could not be verified.")
        installed_version = check.stdout.strip()
        try:
            if Version(installed_version) != Version(catalog_entry.package_version):
                raise McpError(MCP_EXECUTABLE_VALIDATION_FAILED,
                               f"Installed version {installed_version!r} does not exactly match "
                               f"the catalog's pinned {catalog_entry.package_version!r}.")
        except InvalidVersion as e:
            raise McpError(MCP_EXECUTABLE_VALIDATION_FAILED,
                           "The installed package reported an unparseable version.") from e

    def build_launch_spec(self, candidate: CandidateInstallation, catalog_entry) -> McpLaunchSpec:
        venv_dir = candidate.extra.get("venv_dir") or os.path.join(candidate.install_directory, _VENV_DIRNAME)
        venv_python = candidate.extra.get("venv_python") or _venv_python(venv_dir)
        if catalog_entry.launch_entrypoint_type == "console_script":
            command = _venv_console_script(venv_dir, catalog_entry.console_script)
            if not os.path.isfile(command):
                raise McpError(MCP_EXECUTABLE_VALIDATION_FAILED,
                               f"Console-script entrypoint {catalog_entry.console_script!r} is missing.")
            args = tuple(catalog_entry.launch_arguments)
            return McpLaunchSpec(command=command, args=args)
        args = ("-m", catalog_entry.launch_module, *catalog_entry.launch_arguments)
        return McpLaunchSpec(command=venv_python, args=args)

    def cleanup_candidate(self, candidate: CandidateInstallation) -> None:
        if candidate.install_directory != candidate.transaction.final_directory:
            shutil.rmtree(candidate.install_directory, ignore_errors=True)


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
