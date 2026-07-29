"""github.clone_repository — controlled shallow clone of a validated PUBLIC GitHub repo.

Only `{repository, ref?}` is accepted. The clone URL and destination are computed
internally; no destination/URL/flags/token/env come from the LLM. Public repos only,
confirmed via the existing GitHub API client. Nothing is executed or installed.
"""

import os
import shutil
import uuid

import tools.config as config
from tools.base import BaseTool, ToolFailure, ToolValidationError
from tools.git_runner import GitRunner
from tools.github_client import GitHubClient
from tools.github_tools import parse_repository
from tools.models import (
    GITHUB_REPOSITORY_NOT_FOUND,
    PRIVATE_REPOSITORY_NOT_SUPPORTED,
    REPOSITORY_ALREADY_CLONED,
    REPOSITORY_FILE_LIMIT_EXCEEDED,
    REPOSITORY_TOO_LARGE,
    ToolPermission,
)
from tools import repo_store


class CloneRepositoryTool(BaseTool):
    name = "github.clone_repository"
    description = (
        "Clone a PUBLIC GitHub repository into the controlled local workspace for static "
        "inspection. Read-only: nothing is executed, installed, or started. Input is just "
        "the 'owner/repo' (and optional ref)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "repository": {"type": "string", "description": "'owner/repo' (public GitHub only)."},
            "ref": {"type": "string", "description": "Optional branch/tag; defaults to the repo's default branch."},
        },
        "required": ["repository"],
    }
    timeout_seconds = 180.0
    requires_internet = True
    required_capabilities = ("repository.clone",)
    # WRITE: creates files in the controlled workspace (downloads a repository to
    # disk), so it requires explicit user confirmation.
    permission = ToolPermission.WRITE

    def confirmation_summary(self, arguments: dict) -> str:
        # Deterministic: built only from the validated repository argument, never
        # from repository content. `repository` is 'owner/repo'.
        repository = arguments.get("repository")
        repo_label = repository if isinstance(repository, str) and repository.strip() else "the requested repository"
        return f"Clone {repo_label} into the controlled repository workspace."

    def __init__(self, client=None, runner=None):
        self._client = client
        self._runner = runner

    @property
    def client(self):
        return self._client if self._client is not None else GitHubClient()

    @property
    def runner(self):
        return self._runner if self._runner is not None else GitRunner()

    def validate_arguments(self, arguments):
        arguments = super().validate_arguments(arguments)
        owner, repo = parse_repository(arguments)
        ref = repo_store.validate_clone_ref(arguments.get("ref"))
        return {"owner": owner, "repo": repo, "ref": ref}

    def execute(self, arguments):
        owner, repo, ref = arguments["owner"], arguments["repo"], arguments["ref"]

        # 1. Preflight metadata via the existing GitHub client (confirms public + size).
        resp = self.client.get(f"/repos/{owner}/{repo}")
        if resp.status_code == 404 or not resp.data:
            raise ToolFailure(GITHUB_REPOSITORY_NOT_FOUND,
                              f"Repository '{owner}/{repo}' was not found or is not accessible.")
        meta = resp.data
        if meta.get("private") or meta.get("visibility") not in (None, "public"):
            raise ToolFailure(PRIVATE_REPOSITORY_NOT_SUPPORTED,
                              "Only public repositories can be cloned in this phase.")
        default_branch = meta.get("default_branch")
        preflight_kb = meta.get("size") or 0  # GitHub reports repo size in KB
        if preflight_kb > config.max_repository_preflight_size_kb():
            raise ToolFailure(REPOSITORY_TOO_LARGE,
                              f"The repository is too large to clone ({preflight_kb} KB reported).")

        effective_ref = ref or default_branch

        # 2. Existing clone → do not overwrite/update.
        if repo_store.is_cloned(owner, repo):
            raise ToolFailure(REPOSITORY_ALREADY_CLONED,
                              f"Repository '{owner}/{repo}' is already cloned.")

        final_path = repo_store.target_path(owner, repo)
        https_url = f"https://github.com/{owner}/{repo}.git"
        staging = os.path.join(repo_store.staging_root(), f"{owner}__{repo}__{uuid.uuid4().hex[:12]}")

        try:
            # 3. Shallow clone into staging.
            self.runner.clone(https_url, effective_ref, staging)

            # 4. Post-clone size / file-count enforcement (metadata is only an estimate).
            size_bytes, file_count = repo_store.measure_repository(staging)
            if file_count > config.max_cloned_repository_files():
                raise ToolFailure(REPOSITORY_FILE_LIMIT_EXCEEDED,
                                  f"The cloned repository has too many files ({file_count}).")
            if size_bytes > config.max_cloned_repository_size_mb() * 1_000_000:
                raise ToolFailure(REPOSITORY_TOO_LARGE,
                                  f"The cloned repository is too large ({size_bytes} bytes).")

            commit = self.runner.rev_parse_head(staging)

            # 5. Move the validated clone into its controlled final path.
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            os.replace(staging, final_path)
        except ToolFailure:
            _safe_cleanup(staging)
            raise
        except Exception as e:  # noqa: BLE001 — contain unexpected errors, clean staging.
            _safe_cleanup(staging)
            from tools.models import GIT_CLONE_FAILED
            raise ToolFailure(GIT_CLONE_FAILED, f"The clone could not be completed ({type(e).__name__}).")

        return {
            "repository": f"{owner}/{repo}",
            "ref": effective_ref,
            "default_branch": default_branch,
            "local_repository_id": f"{owner}/{repo}",
            "relative_path": f"{owner}/{repo}",
            "commit": commit,
            "file_count": file_count,
            "size_bytes": size_bytes,
            "shallow": True,
            "submodules_initialized": False,
            "lfs_downloaded": False,
            "executed": False,
            "installed": False,
            "untrusted_content": True,
            "source_type": "cloned_repository",
            "_log_meta": {"repository": f"{owner}/{repo}", "size_bytes": size_bytes,
                          "file_count": file_count},
        }


def _safe_cleanup(staging):
    """Remove only the controlled staging directory; never touch anything else."""
    try:
        root = repo_store.repository_root_abs()
        real = os.path.realpath(staging)
        if real.startswith(os.path.join(root, ".staging") + os.sep) and os.path.isdir(real):
            shutil.rmtree(real, ignore_errors=True)
    except OSError:
        pass
