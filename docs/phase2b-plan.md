# Phase 2B Plan — Controlled Clone + Static Repository Inspection

Build on Phase 1 (tool framework) and Phase 2A (read-only web/GitHub tools) to let an explicitly
enabled assistant clone a validated **public** GitHub repo into a controlled directory and inspect
it **statically**. Nothing from a cloned repo is executed, imported, or installed. Disabled by
default. No new dependencies (git + `tomllib`/`ast`/`subprocess`/`shutil` are all available).

## Existing Phase 1 / 2A architecture (reused, not duplicated)

- `tools/base.py` `BaseTool`/`ToolValidationError`/`ToolFailure`; `tools/executor.py` `ToolExecutor`
  (ThreadPoolExecutor timeout, capability gate, `log_meta`); `tools/registry.py` `default_registry()`;
  `tools/models.py` `ToolResult.ok/fail` + error codes; `tools/config.py` getter pattern.
- `tools/github_client.py` `GitHubClient` (metadata preflight); `tools/github_tools.py`
  `parse_repository` / `validate_ref` / `validate_path`.
- `tool_loop.py` (MAX_TOOL_STEPS, exhaustion, safety block); `interaction_log.log_tool_event`;
  `assistant.dispatch` (mode=="local"). Tool messages never persist → ChromaDB exclusion is automatic.

Prerequisite: `pytest` green (282) before implementation.

## Exact extension points

- New tools subclass `BaseTool`, register through `default_registry()`, execute through the existing
  `ToolExecutor` and local tool loop — no new registry/executor/loop.
- New named-capability gate in the executor (`required_capabilities` → `repository.clone`,
  `repository.read`) sitting beside the existing `requires_internet` gate.
- The clone tool reuses `GitHubClient` for its public/size preflight.

## Files to create

`tools/git_runner.py` (the one dedicated Git subprocess), `tools/repo_store.py` (root/containment/
symlink/size), `tools/repo_clone.py` (`github.clone_repository`), `tools/repo_analysis.py`
(analyzers + dependency parsing), `tools/repo_security.py` (static scanner), `tools/repo_tools.py`
(the 5 `repo.*` tools); tests `test_git_runner/repo_store/repo_clone/repo_files/repo_inspect/
repo_security/repo_capability/phase2b_integration.py`; docs `phase2b-plan.md`,
`phase2b-repository-inspection.md`.

## Files to modify (additive)

- `tools/models.py` — 22 Phase 2B error codes.
- `tools/base.py` — `required_capabilities: tuple = ()`.
- `tools/executor.py` — named-capability gate map (`repository.clone`→`REPOSITORY_CLONE_DISABLED`,
  `repository.read`→`REPOSITORY_INSPECTION_DISABLED`).
- `tools/config.py` — Phase 2B getters (all with safe defaults).
- `tools/registry.py` — register `github.clone_repository` when internet + cloning enabled; the 5
  `repo.*` tools when inspection (or cloning) enabled; share `GitHubClient`/`GitRunner`.
- `tool_loop.py` — extend the untrusted-content safety block for cloned repos + static-only statement.
- `.env.example`, `.gitignore` (add `data/`), `README.md`.

## Clone safety model

Public GitHub only. Input `{repository, ref?}` — no destination/URL/flags/token/env. Preflight via
`GitHubClient` (reject private/404, resolve default branch, reject > preflight size). Destination
computed as `REPOSITORY_ROOT/owner/repo`, resolved, asserted inside root. Clone into a staging dir,
enforce timeout, post-clone size/file-count enforcement, `rev-parse HEAD`, then `os.replace` into
the final path. Existing clone → `REPOSITORY_ALREADY_CLONED` (never update/overwrite). Failures
clean only the staging dir.

## Controlled-directory model

One configured `REPOSITORY_ROOT` (default `data/repositories`, gitignored), resolved to an absolute
path. Targets built only from validated `owner`/`repo`; realpath containment rejects traversal,
absolute components, and symlink escapes (`REPOSITORY_PATH_ESCAPE`). A `.staging/` subdir holds
in-progress clones; cleanup only ever touches `.staging/`.

## Git invocation design

`tools/git_runner.py` — the only Phase 2B subprocess, never a tool. `subprocess.run(argv,
shell=False, env=hardened, cwd/dest controlled, timeout, capture_output)`. Clone argv:
`git clone --depth 1 --single-branch --no-tags [--branch <ref>] -- <https-url> <staging>`. Env:
`GIT_TERMINAL_PROMPT=0`, `GIT_LFS_SKIP_SMUDGE=1`, `GIT_ALLOW_PROTOCOL=https`, `GIT_CONFIG_NOSYSTEM=1`,
`GIT_CONFIG_GLOBAL=os.devnull`, cleared askpass/credential-manager. No token in URL/args. Verifies
git exists (`GIT_NOT_AVAILABLE`); maps timeout/nonzero/rev-parse to controlled codes; stderr
sanitized + truncated. Never `sh/bash/cmd/powershell`, never `shell=True`.

## Repository path validation design

Reuse `github_tools.validate_path` (rejects `..`/absolute/backslash/null/control chars). Clone ref
validated separately (`validate_clone_ref`: non-empty, ≤200 chars, no null/control/`..`, must not
start with `-`, safe charset) → `INVALID_REPOSITORY_REF`. All repo.* path inputs resolved via
`repo_store.resolve_within` (realpath containment).

## Symlink-handling design

`os.scandir`/`os.walk(followlinks=False)` + `lstat`. Symlinks are marked (`type:"symlink",
followed:false`), never recursed into, and rejected by `repo.read_file`
(`REPOSITORY_SYMLINK_BLOCKED`). Realpath containment additionally blocks a symlink that resolves
outside the repo (`REPOSITORY_PATH_ESCAPE`). Windows junctions/reparse points are treated like
symlinks. `.git` is skipped by default.

## Static inspection design

`repo_analysis.analyze` (languages, extension distribution, manifests, package managers, docs/tests/
license, container/CI files, submodule/LFS/binary/vendored flags, dependency parsing via
`tomllib`/`json`/text, entry-point + integration inference) and `repo_security.scan` (Python `ast`
for eval/exec/subprocess/shell-true/pickle/ctypes/network/dynamic-import + text patterns for
curl-pipe-shell/powershell/docker-privileged/host-mount/encoded blobs/hardcoded-secret with
redaction). Everything reads files as data — manifests and `setup.py`/`package.json` scripts are
parsed/read as text, never executed. Output separates observed facts from inference.

## Security-scan limitations

Pattern/AST based → false positives and false negatives are possible; a clean scan means only that
no configured pattern matched. It never asserts safe/trusted/approved. Bounded by
files/file-bytes/total-bytes/depth/findings caps (`REPOSITORY_SCAN_LIMIT_REACHED` + truncation flag).

## Automated testing plan

All network/clone mocked (fake `GitHubClient`/`GitRunner`, `tmp_path` repo fixtures incl. a
symlink-escape fixture); never touches the real `REPOSITORY_ROOT`. Covers git-runner argv/env/
shell=False/timeouts, clone preflight/staging/limits/containment/rejections, list/read containment +
symlink + binary/dir, inspect + dependency parsing, security detections + redaction + limits,
capability inference + no-approval, and fake-LLM integration (clone→inspect→scan→capability, errors
to LLM, MAX_TOOL_STEPS, memory exclusion, Claude unchanged, config toggles). Full suite before + after.

## Optional live testing plan

With `REPOSITORY_CLONE_ENABLED=true` and a disposable `REPOSITORY_ROOT`: `octocat/Hello-World` →
get_repository → clone → list_files → read_file → inspect → security_scan → capability_report. Report
skips/failures honestly; never execute repo code.

## Token-overhead considerations

Phase 2B adds 6 tool schemas. Measured off/P1/P1+2A/P1+2A+2B. Schemas kept concise; outputs bounded
(inspect/capability are summaries, scan caps findings). Phase 2B is OFF by default so it adds nothing
unless enabled. Tool-selection routing is deferred to a later phase.

## Explicit Phase 2B exclusions

Plugin install/enable, dynamic registration from cloned code, MCP start/connect, package installs,
builds, Docker run/compose, shell/PS/bat execution, test execution, importing/running repo modules
or setup.py, submodules, LFS, hooks, private repos, SSH, arbitrary hosts, updates/pull/checkout after
clone, repo writes/edits/commits, auto-remediation, any safe/trusted/approved claim, asyncio, and any
second registry/executor/loop/GitHub-client.
