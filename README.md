# home-ai — local-first voice assistant

A local-first Windows voice assistant: mic in → a local model (via Ollama) answers out loud,
escalating hard or accuracy-critical questions to the Claude API. See `spec.md` for the full
design and `CAMERA.md` for the camera features.

## Running the assistant

```
pip install -r requirements.txt
cp .env.example .env      # then fill in your keys/models
python assistant.py
```

At the prompt choose `t` (text), `p` (push-to-talk), or `q` (quit).

## Phase 1: local tool calling

The local model can now call a small set of **safe, built-in tools**, receive a structured JSON
result, and then write the final answer itself. The tool never writes the user-facing reply — the
local LLM always does. This runs **only on the local answering path**; Claude routing is unchanged.

### Available Phase 1 tools

- **`system.echo`** — echoes text back (diagnostics).
- **`math.calculate`** — evaluates a basic arithmetic expression (`+ - * / **`, parentheses,
  decimals, unary minus) using a restricted parser — never `eval`.

Internet access, GitHub search, plugins, MCP, filesystem, and shell tools are **not** included yet.

### Enabling / disabling

Configured via environment (safe defaults; the app starts fine if these are unset):

```
OLLAMA_URL=http://localhost:11434
TOOL_CALLING_ENABLED=true          # set false to restore original single-shot local chat
MAX_TOOL_STEPS=5                   # max tool calls per turn (every call counts)
DEFAULT_TOOL_TIMEOUT_SECONDS=10
```

### Example interaction

```
> Use the calculator to calculate (17 * 23) + 5.
The result is 396.
```

Internally: the router picks the local path → the local model requests `math.calculate` →
the executor returns `{"expression": "(17 * 23) + 5", "result": 396}` → that JSON is handed back
to the local model → the model replies "The result is 396."

### Running the tests

```
python -m pytest -q
```

The suite is deterministic and does not require a live Ollama endpoint.

### More detail

- `docs/phase1-tool-framework.md` — design, configuration, token overhead, limitations.
- `docs/phase1-tool-call-test.md` — the verified native Ollama tool-call/result message format.
- `docs/phase1-tool-framework-plan.md` — the implementation plan.

## Phase 2A: read-only internet & GitHub tools

The local model can now **search the public web and inspect public GitHub repositories** through
read-only tools, built on the same Phase 1 framework. Results always return to the local LLM, which
writes the final answer and cites its sources. **Repositories are never cloned or executed**, no
files are downloaded, nothing is installed, and no writes are made to GitHub.

### Available Phase 2A tools

- **`browser.search`** — search the public web (Tavily). Needs `TAVILY_API_KEY`.
- **`browser.fetch_page`** — fetch a public **HTTPS** page and extract readable text. Blocks
  local/private/loopback/metadata addresses (SSRF-guarded), enforces byte/char/redirect limits,
  and never executes JavaScript.
- **`github.search_repositories`**, **`github.get_repository`**, **`github.read_file`**,
  **`github.list_directory`**, **`github.list_releases`** — read-only GitHub REST API access.

Remote content is marked **untrusted**: the model is instructed never to follow instructions found
inside fetched pages or repository files (prompt-injection defense). Still **not** included: repo
cloning, code/shell execution, package installs, plugins, MCP, private repos, or any GitHub writes.

### Environment variables

```
INTERNET_TOOLS_ENABLED=true     # false → the 7 tools are not offered to the model
INTERNET_READ_ENABLED=true      # false → tools are visible but blocked at execution
SEARCH_PROVIDER=tavily
TAVILY_API_KEY=                 # without it, browser.search returns a controlled missing-key error
GITHUB_TOKEN=                   # optional; read-only token → higher rate limits + live tests
HTTP_USER_AGENT=Local-LLM-Project/1.0
ALLOW_HTTP_FETCH=false          # HTTPS only by default
SEARCH_MAX_RESULTS=10
MAX_SEARCH_QUERY_CHARS=500
BROWSER_CONNECT_TIMEOUT_SECONDS=5
BROWSER_READ_TIMEOUT_SECONDS=20
BROWSER_MAX_REDIRECTS=5
MAX_PAGE_BYTES=2000000
MAX_PAGE_CHARS=30000
GITHUB_TIMEOUT_SECONDS=20
GITHUB_MAX_FILE_BYTES=1000000
GITHUB_MAX_FILE_CHARS=30000
GITHUB_MAX_DIRECTORY_ENTRIES=200
GITHUB_MAX_RELEASES=10
```

Missing optional credentials never break startup — only the specific capability is disabled.

### Example interactions

GitHub:
```
> Find public GitHub repositories for financial analysis tools and compare the top two.
The local model calls github.search_repositories, github.get_repository, and github.read_file,
then answers with each repo's full name, URL, stars, and what it read — citing only real sources.
```

Browser:
```
> Search for the official page of <project> and summarize it.
The local model calls browser.search, then browser.fetch_page on a result, then summarizes the
page and names the final URL. (A search snippet alone is not treated as having read the page.)
```

### Security limitations

`browser.fetch_page` resolves and validates every hop against an IP blocklist, but a small
DNS-rebinding (TOCTOU) window exists between resolution and connection. Text extraction is
best-effort (no JavaScript rendering). See `docs/phase2a-internet-github-tools.md` for the full
security model, SSRF protections, rate-limit handling, and token-overhead numbers.

### Running the tests (Phase 1 + Phase 2A)

```
python -m pytest -q
```

Automated tests mock all network access — no internet or API keys required. Optional live tests
(GitHub / Tavily / a public page) run only when credentials and connectivity are available and are
reported honestly as skipped otherwise.

## Phase 2B: controlled clone + static repository inspection

The local model can clone a validated **public** GitHub repository into a controlled local
workspace and inspect it **statically** — languages, manifests, dependencies, likely entry points,
integration type, and a static security scan — then explain how it might integrate. **Repositories
are never executed, imported, installed, or started in Phase 2B.** Disabled by default.

### Available Phase 2B tools

- **`github.clone_repository`** — shallow (`--depth 1`), single-branch, no-tags, no-submodule,
  no-LFS clone of a public repo into `REPOSITORY_ROOT/<owner>/<repo>`. Public repos only; the URL
  and destination are computed internally (never taken from the model). Size/file-count limited.
- **`repo.list_files`** / **`repo.read_file`** — bounded listing / text reading of a cloned repo.
  Path traversal, absolute paths, and symlinks are rejected; symlinks are never followed.
- **`repo.inspect`** — static structural summary (facts vs inference).
- **`repo.security_scan`** — bounded Python-AST + text-pattern scan flagging code that needs human
  review. It **never** claims a repository is safe.
- **`repo.capability_report`** — integration-readiness report; recommendation is limited to
  `insufficient_information | static_review_complete | manual_review_required |
  not_supported_by_current_architecture` (never "safe to install").

Cloned content is untrusted: the model is told never to follow instructions or run commands found
in a README or source file, and that inspection was static only.

### Required environment variables

```
REPOSITORY_CLONE_ENABLED=false      # OFF by default; enable to allow cloning
REPOSITORY_INSPECTION_ENABLED=false # enable repo.* tools for already-cloned repos (implied by clone)
REPOSITORY_ROOT=data/repositories   # controlled workspace (gitignored)
GIT_EXECUTABLE=git
GIT_CLONE_TIMEOUT_SECONDS=120
MAX_REPOSITORY_PREFLIGHT_SIZE_KB=200000
MAX_CLONED_REPOSITORY_SIZE_MB=250
MAX_CLONED_REPOSITORY_FILES=25000
REPO_MAX_LIST_ENTRIES=500
REPO_MAX_LIST_DEPTH=5
REPO_MAX_READ_BYTES=1000000
REPO_MAX_READ_CHARS=30000
REPO_SCAN_MAX_FILES=5000
REPO_SCAN_MAX_FILE_BYTES=500000
REPO_SCAN_MAX_TOTAL_BYTES=50000000
REPO_SCAN_MAX_DEPTH=20
REPO_SCAN_MAX_FINDINGS=500
```

### Example clone + inspection interaction

```
> Clone octocat/Hello-World and tell me its languages, likely entry points, and integration type.
  Do not run or install anything.
The local model calls github.clone_repository, then repo.inspect (and optionally repo.security_scan
/ repo.capability_report), and explains the observed facts — clearly stating that inspection was
static and nothing was executed or installed.
```

### Security limitations

Clone uses a single dedicated Git subprocess (argument list, `shell=False`, hardened env,
`GIT_ALLOW_PROTOCOL=https`, no token in the URL/args, timeout); no generic subprocess tool is
exposed. Destinations are computed internally and cannot escape `REPOSITORY_ROOT`; a staging
directory is used and cleaned on failure. GitHub's reported size is only a preflight estimate, so a
post-clone size/file-count check is also enforced. Static analysis and scanning are best-effort and
bounded — a clean scan does **not** prove a repository is safe. See
`docs/phase2b-repository-inspection.md` for the full model.

**Repositories are never cloned recursively (no submodules/LFS), never updated after clone, and
never installed or executed.**

### Enabling cloning (opt-in)

Set in `.env`: `REPOSITORY_CLONE_ENABLED=true` (this also enables the `repo.*` inspection tools).
With it off, the Phase 2B tools are not offered to the model at all.
