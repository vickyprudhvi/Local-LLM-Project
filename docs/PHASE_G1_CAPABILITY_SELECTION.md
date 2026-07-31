# Phase G.1 — MCP Capability Catalog and Deterministic Server Selection

> Capability selection does not execute tools, start servers, install packages,
> or modify permissions.

> A server may be selected as the approved provider even when it is not
> installed. Installation is handled by a later approval-driven provisioning
> phase.

> Public MCP directories are discovery sources only. They are never runtime
> trust sources.

## Purpose

Phase F/F.1 answered "does this request need an MCP capability, and is the
Filesystem server installed/expanded to serve it?" — but only for one coarse
capability (`"filesystem"`) matched against a fixed pattern list. Phase G.1
generalizes this into a typed, catalog-driven layer that can answer, for ANY
approved server with capability metadata:

1. Does this request require an MCP capability?
2. What capability or capabilities, specifically?
3. Which approved server can provide it?
4. Is that server installed, active, disabled, or ambiguous against another
   approved provider?

It does **not** pick an exact tool, and it does not touch MCP runtime lifecycle
in any way — no `bootstrap_from_config`, no `McpRuntimeManager`, no install.

## Architecture: five layers, five responsibilities

```
Router                Capability selector          Phase B            Local LLM        ToolExecutor
local or Claude   ->   required MCP capability  ->  bounded exact- ->  one exact   ->   permission +
only                   + approved server            tool candidate     tool or          execution
                       provider (never a tool)       shortlist          none             authority
```

- **Router** (`router.py`) is completely unaware Phase G.1 exists. It still
  decides only `local` vs. `claude` (plus its own small fixed built-in set) —
  nothing was added to `RouteDecision` or to the router's tool contract.
- **Capability selector** (this phase) runs strictly AFTER the router picks
  `local` and BEFORE Phase B. It names, at most, an approved **server** as a
  preferred capability provider.
- **Phase B** (`tools/registry.py::shortlist_tools`) is completely unchanged —
  Phase G.1's selection is not yet consumed by it (a later phase, G.8, does
  that). The bounded exact-tool shortlist behaves byte-for-byte as before.
- **Local LLM** still picks one exact tool (or none) from Phase B's shortlist.
- **ToolExecutor** (`tools/executor.py`, unmodified) remains the sole
  permission/execution authority.

## Catalog capability metadata

`mcp_management/catalog.py`'s `McpCatalogEntry` gains two OPTIONAL fields,
populated from `config/mcp_catalog.json`:

```json
{
  "granular_capabilities": ["read_local_text_file", "list_local_directory", "..."],
  "selection_hints": {
    "explicit_names": ["filesystem", "filesystem mcp", "local files"],
    "actions": {"read_local_text_file": ["read", "open", "..."]},
    "extensions": {"read_local_text_file": [".txt", ".md", "..."]}
  }
}
```

**Why a separate field from the existing `capabilities` list.** Phase F's own
`detect_capability()`/`McpCatalog.find_by_capability()` already depend on the
existing `capabilities` list using coarse values (`"filesystem"`,
`"read_files"`, ...). Reusing that same key for Phase G.1's much finer
vocabulary (`"read_local_text_file"`) would either collide with or silently
redefine what Phase F already matches. `granular_capabilities` is therefore
new, additive, and never touched by any Phase F code path — `tests/
test_mcp_catalog.py` (Phase F's own catalog tests) passes completely
unmodified.

**Backward compatibility.** An entry with no `granular_capabilities` loads
exactly as before (`entry.granular_capabilities == ()`,
`entry.selection_hints == McpSelectionHints()`); it is simply invisible to the
Phase G.1 selector until metadata is added — never an error.

**Validation (fail closed).** `mcp_management/catalog.py` rejects: a
capability id that doesn't match `^[a-z0-9_]+$`, a duplicate capability id
within one entry, a `selection_hints` phrase/extension referencing a
capability the entry never declared, a malformed extension (must look like
`.ext`), and any unrecognized top-level key under `selection_hints`. Loading
never executes code or touches the network — it is pure JSON parsing +
dataclass construction, exactly like every other catalog field.

## Detection rules (`mcp_management/capability_detector.py::McpCapabilityDetector`)

Deliberately a **separate class**, added to the existing capability_detector.py
module rather than replacing `detect_capability()`/`validate_detection()`
(Phase F's own, unmodified — still used by `assistant._provision_if_needed`).
`McpCapabilityDetector.detect(user_text, catalog)` returns zero or more
`CapabilityRequirement`s. No filesystem access, no network access, no LLM call
— `os.stat`, `Path.exists`, and `open()` are never invoked; a path is
recognized by shape alone.

- **Local-path detection**: Windows (`C:\...`, `C:/...`), UNC
  (`\\server\share\...`), and POSIX (`/home/...`) forms, quoted or unquoted.
  Only an ABSOLUTE path counts as strong local-path evidence — a bare relative
  filename (`README.md`) never does (see "repository-relative" below).
- **URL handling**: a match that starts inside an `https://...` span is never
  treated as a local path. `open https://example.com/report.pdf` produces NO
  requirement at all — remote-document handling is out of scope for this
  phase, and a URL must never be misread as a local file.
- **Repository-relative paths**: `read README.md` (no absolute path, no
  explicit filesystem wording) produces no filesystem requirement, so the
  existing repository/GitHub/browser tools remain free to compete in Phase B
  exactly as before. `read C:\Projects\repo\README.md` DOES produce one,
  because the user supplied an explicit local path.
- **Action vs. extension — action wins**: two disjoint verb sets decide the
  category. Filesystem verbs (`read/open/list/find/search/write/save/create/
  copy/move/delete/...`) require an absolute local path and map to a specific
  granular capability (`copy`/`delete` → `manage_local_files`,
  `list`/`browse` → `list_local_directory`, `find`/`search` →
  `search_local_files`, etc.). Document-conversion verbs
  (`summarize/review/analyze/extract text`) are a **disjoint** set — `copy
  report.pdf` is `manage_local_files` (filesystem), `summarize report.pdf` is
  `document_to_markdown`, even though both name the same file extension.
- **Extension/noun alone is never enough**: `What is a PDF?` matches neither
  verb set, so it produces no requirement at all — extension or document-noun
  presence only matters once a real action verb from one of the two sets has
  already fired.
- **Explicit server naming**: `Use the filesystem MCP server to ...` is
  recognized structurally and checked against every catalog entry's
  `selection_hints.explicit_names`. A match becomes `EXPLICIT_SERVER` evidence
  (decisive in scoring); a name that matches **no** catalog entry is recorded
  as `"unknown:<name>"` — never silently guessed at or substituted for a
  different, approved server.

## Selection algorithm (`mcp_management/server_selector.py::McpServerSelector`)

1. No requirements → `NONE_REQUIRED` (existing behavior unchanged, by
   construction — this is the exact object every non-MCP request returns).
2. An explicit reference to an unrecognized server → `AMBIGUOUS` /
   `MCP_SERVER_SELECTION_AMBIGUOUS` immediately — never silently redirected to
   whatever approved server happens to also match.
3. Build one `McpServerCandidate` per catalog entry that (a) has
   `granular_capabilities` at all, (b) matches at least one required
   capability id, and (c) is not administratively disabled. Score = sum of:
   explicit-server bonus (1000, only for the named server) + 500 per matched
   capability id + the matched requirement's own evidence scores
   (local-path/action/extension, shared identically by every candidate
   providing that capability — this is what keeps "more accurate match" from
   ever being outranked by mere install/active status) + small installed
   (+10) / enabled (+3) / active (+5) bonuses, which can only break a tie
   between otherwise-EQUAL candidates.
4. Candidates that cover **every** required capability id are ranked first
   (`score` desc, `server_id` asc — no randomness). A single top scorer →
   `SELECTED`. A tie at the top → `AMBIGUOUS`.
5. No candidate covers everything: exactly one distinct capability was
   required → `UNSUPPORTED` / `MCP_CAPABILITY_UNAVAILABLE`. More than one
   distinct capability was required → `MULTI_SERVER_REQUIRED` /
   `MCP_MULTI_SERVER_WORKFLOW_REQUIRED` (Phase G.1 never begins a partial
   workflow — a later phase may add multi-server orchestration).
6. Returned candidates are bounded to 3, always in the same deterministic
   order.

## Read-only server status (Task 9)

`McpInstalledStateProvider`/`McpRuntimeStatusProvider` are small `Protocol`s.
`RegistryInstalledState` wraps the EXISTING `mcp_management.registry.
load_registry` (no parallel installed-state store). `ActiveRuntimeStatusProvider`
wraps whatever `mcp_layer.runtime_manager.ActiveMcpRuntime` currently holds and
only ever reads `.session.health` — it never calls `ensure_started`,
`replace_active_session`, `stop`, `install`, `provision`, or `restart`. This is
the abstraction a completed Phase G.0 multi-runtime manager would implement
instead; nothing in `server_selector.py` assumes there is only one runtime.

## Integration point (`assistant.py`)

```python
if decision.mode == "local":
    reply, extra_metrics, pending_fs_request_id = _run_local_turn_with_capability_selection(
        provisioning_manager, runtime, user_text, prompt, history, system_prompt,
        attempted_fs_requests)
```

`_run_local_turn_with_capability_selection` calls
`mcp_management.capability_service.select_for_request` (a thin facade over the
detector + selector + the two read-only providers above) using
`provisioning_manager.catalog` — the SAME already-loaded trusted catalog Phase
F uses; there is no second catalog-loading path. `NONE_REQUIRED` and
`SELECTED` both fall straight through to the pre-existing `_run_local_turn`
(the Phase F.1 hotfix path) completely unchanged — this is what makes
invariant 15 ("existing behavior unchanged when no MCP capability is
required") mechanically true rather than just documented. `UNSUPPORTED`,
`AMBIGUOUS`, and `MULTI_SERVER_REQUIRED` return a normalized message and skip
Phase B entirely for that turn — zero LLM calls, zero MCP calls.

## Ambiguity and unsupported-capability behavior

No LLM tie-breaker exists in Phase G.1 — ambiguity always resolves to a fixed,
deterministic clarification message, never an arbitrary winner. An unsupported
capability (most notably `document_to_markdown` — MarkItDown is not installed
or approved in this phase) always returns the same normalized message; the
model never invents or selects an unapproved public repository as a
substitute provider.

## Privacy-safe diagnostics (Task 12)

Nothing is printed on a normal request. `MCP_CAPABILITY_DEBUG=true` enables
`assistant._log_capability_selection`, which prints only capability ids,
confidence, the selection status, and (when selected) the server id — never
raw request text, a file path, file contents, or a candidate's full reasoning
list. Startup output is unchanged; the catalog being loaded does not, by
itself, print anything new.

## Integration points for later phases

- `mcp_management.capabilities.TurnMcpContext` — a typed
  `(preferred_mcp_server_id, required_mcp_capabilities)` pair Phase G.1
  computes but does not yet attach anywhere Phase B or the tool loop reads
  from. G.8 (server-aware shortlisting) is expected to consume it.
- `McpRuntimeStatusProvider`/`McpInstalledStateProvider` — implemented today
  by thin adapters over the single-runtime holder; a completed Phase G.0
  multi-runtime manager can implement the same two-method Protocols with no
  change to `server_selector.py`.
- `McpServerSelection.candidates` already carries `installed`/`active` per
  candidate, which G.2 (lazy activation) and G.3 (generalized provisioning)
  can act on directly instead of re-deriving it.

## Hotfix: document-analysis requests bypassing capability selection

A real CLI request —
`summarize C:\Users\Prudhvi\OneDrive\learn stuff\Hands_on_LLM.pdf` — reached
Phase B with browser/GitHub tools shortlisted, the model then hallucinated
`filesystem.read_file`, and a Filesystem access plan was incorrectly offered.
Investigation found two confirmed architectural gaps (not a bug in this exact
string's detection, which the existing detector actually classified correctly
in isolation) that made this class of failure possible:

1. **Resumption bypassed capability selection.** `_restart_mcp_and_resume` —
   used by BOTH the cross-turn "yes" approval and a mid-turn `access.add`
   restart — called `_run_local_turn` directly on the resumed text, skipping
   the Phase G.1 wrapper entirely on the second pass. Fixed by introducing
   ONE authoritative entrypoint, `assistant._process_local_request_with_
   capability_selection`, that every local request and every resumption now
   calls; `_run_local_turn` is called from nowhere else in production code.
2. **Unquoted Windows paths containing spaces were silently truncated.**
   `_WINDOWS_PATH_RE` stops at the first whitespace, so
   `C:\Users\...\learn stuff\Hands_on_LLM.pdf` extracted as
   `C:\Users\...\learn` — the wrong path, and wrong LOCAL_PATH evidence, even
   though a lucky noun-regex match happened to still classify this specific
   sentence correctly. Fixed with a lexical (never filesystem-touching)
   extension: an unquoted match is grown token-by-token until the
   accumulated string ends in a recognized extension (Task 3).

Two further hardening measures closed the remaining gap between "detection is
correct" and "the wrong path never gets taken even if it weren't":

3. **Shortlist-membership enforcement** (`tool_loop.py`): an `mcp.`-namespaced
   tool call naming anything outside the exact set of tool schemas offered
   THIS round is rejected (`TOOL_NOT_IN_SHORTLIST`) before `ToolExecutor` ever
   runs — closing the "hallucinated tool" gap the same way `router.py`'s
   `_OFFERED_FUNCTION_NAMES` already closes it for routing. Scoped to `mcp.*`
   only, so the small fixed built-in set (calculator, echo, ...) — already
   exercised by hundreds of pre-existing tests that don't shortlist-tune their
   FakeLLM scripts — is unaffected.
4. **Broadened document-conversion vocabulary** (Task 2): additional verbs
   (`extract tables/content`, `convert to markdown`, `read and summarize`,
   `explain/inspect this document`) and extensions (`.html`, `.htm`, `.ipynb`,
   `.epub`) so more real phrasings resolve to `document_to_markdown` instead
   of falling through to NONE_REQUIRED.

## Explicitly out of scope for Phase G.1

MarkItDown installation, Python/uv/Playwright/finance-MCP provisioning, remote
HTTP MCP, OAuth/Composio, multi-runtime lifecycle management, lazy startup,
idle shutdown, automatic installation, multi-server workflow execution,
server-aware Phase B filtering, and any public MCP-directory discovery. See
the task's own "OUT OF SCOPE" list for the complete set — none of it changed.

## Tests

```
pytest -q tests/test_mcp_capability_catalog.py
pytest -q tests/test_mcp_capability_requirement_detection.py
pytest -q tests/test_mcp_capability_file_intents.py
pytest -q tests/test_mcp_server_selector.py
pytest -q tests/test_mcp_capability_ambiguity.py
pytest -q tests/test_assistant_capability_selection.py
pytest -q   # full suite
```

Note: the suggested test filename `tests/test_mcp_capability_detector.py` was
NOT reused for the new detector — that file already exists and tests Phase F's
own `detect_capability()`/`validate_detection()`. The new tests live in
`tests/test_mcp_capability_requirement_detection.py` instead, so neither file
was overwritten or had to change.
