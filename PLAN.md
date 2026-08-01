# Phase G.4 Implementation Plan — Production MarkItDown MCP Provider

**Status:** READY  
**Prerequisites resolved.**

---

## Goal

Add Microsoft `markitdown-mcp==0.0.1a4` as the first production provider for the `document_to_markdown` capability. Use the existing Phase G.1–G.3 framework and Phase B bounded shortlisting. Preserve all prior security boundaries: router decides only local or Claude, capability detection does not select an exact tool, Phase B selects the candidate tool set, the local LLM selects one offered tool, `ToolExecutor` remains unchanged, no remote URL conversion, no public MCP discovery, no arbitrary package installation, and no Filesystem MCP access inheritance.

---

## Mandatory sequencing correction (post-approval)

Document authorization must follow this exact sequence:

1. `DocumentInputAuthorization` is created only after runtime activation and successful revalidation of the `DocumentInputSnapshot`.
2. The authorization is **atomically reserved** before the single MCP `convert_to_markdown` call.
3. The MCP server is called **once** under that reservation.
4. Immediately after the MCP call returns — regardless of whether the remote conversion succeeded or failed — the authorization is **marked consumed**.
5. Result normalization, output-size validation, and LLM summarization all happen **after** the authorization is consumed.
6. If normalization, output-size validation, or summarization fails, the authorization **remains consumed**.
7. The same authorization is **never** reused for an automatic retry.
8. Any retry requires a new `DocumentInputSnapshot` capture, revalidation, and a new `DocumentInputAuthorization`.

This correction is binding and will be enforced in `DocumentAuthorizationStore`, `LocalDocumentExactFilePolicy`, `DocumentConversionResult` normalization, and the assistant/tool-execution pipeline.

---

## What has changed from the previous plan

1. **Split authorization into two immutable models**: `DocumentInputSnapshot` (bound to the provisioning plan) and `DocumentInputAuthorization` (short-lived, created only after runtime activation).
2. **Expanded plan hash to cover complete file identity and trust context**: file path, URI, size, SHA-256, modified time, plus package identity, lock hashes, policies, launch config, and environment.
3. **Added atomic single-use authorization with separate state store and explicit lifecycle**, now including the mandatory reserve-before-call / consume-immediately-after-call sequencing.
4. **Specified that the invocation policy ignores the model URI and replaces it with the trusted file URI**, while still rejecting malicious URIs before any MCP client call.
5. **Added Python lock environment pinning** with fail-closed behavior.
6. **Removed any allowance for placeholder or partial hashes**; stated explicit stop-and-report blocker policy.
7. **Replaced generic `candidate_validator: "markitdown"` with versioned `"markitdown_local_document_v1"`** and a registry.
8. **Added requirement to verify actual upstream package behavior** and document real artifacts.
9. **Restricted result normalization to verified shapes** and added `MCP_DOCUMENT_RESULT_INVALID`.
10. **Added granular error codes** for authorization, path/URI, file-change, input-size, timeout, and conversion failures.
11. **Added real PDF fixture requirement** for candidate validation.
12. **Clarified separation between discovery and Phase B shortlisting**.
13. **Added explicit honest network security language**.
14. **Added expanded security test list and regression target**.

---

## Verified upstream facts

All four prerequisites are resolved.

### Artifact hashes

- Wheel: `markitdown_mcp-0.0.1a4-py3-none-any.whl`
  - SHA-256: `7fb06fff7d722ec108d08752704dc8f313f7fd267e00dff546131ec229645230`
- Source distribution: `markitdown_mcp-0.0.1a4.tar.gz`
  - SHA-256: `309c94dc883311e6909d849382a6c7bc402dfb2692dab448c136c6864c6bf49e`
- Transitive library: `markitdown-0.1.7-py3-none-any.whl`
  - SHA-256: `4eca912c87c6aa6897284a7f4bf6769a23bccf8544530f5d8b175fbe3797c916`

### Verified runtime behavior

- Default transport: **stdio**
- HTTP/SSE transport: available upstream, **disabled in this project**
- Console entrypoint: `markitdown-mcp = markitdown_mcp.__main__:main`
- Venv entrypoint path: `venv/Scripts/markitdown-mcp.exe` (Windows)
- Protocol version: `2024-11-05`
- Server name: `markitdown`
- Server version: `1.8.1` (from `mcp` dependency)
- Exact tool name: `convert_to_markdown`
- Input schema:
  ```json
  {
    "properties": {"uri": {"title": "Uri", "type": "string"}},
    "required": ["uri"],
    "title": "convert_to_markdownArguments",
    "type": "object"
  }
  ```
- Result shape:
  ```json
  {"text": "<markdown string>"}
  ```
- Resources: `{"resources": []}`
- Prompts: `{"prompts": []}`
- Environment variables read by server: **none identified**
- Plugin behavior: upstream declares `markitdown[all]`; production lock **omits** `[all]` extras

### Reviewed environment

- Implementation: CPython
- Python major/minor: **3.13**
- Platform: **win_amd64**
- Operating system: Windows 11

### Dependency lock

- File: `config/mcp_locks/markitdown-mcp-0.0.1a4.txt`
- Packages: 60
- Hash-locked: every artifact has a verified SHA-256
- Lock file SHA-256: `b94ace40f56fa7ab74c632105ad4b1c0195c7e2c0dd6294e0c7d88e302491222`
- Installation flags: `pip install --require-hashes --no-deps -r <file>`
- Verified by clean-venv install: **yes**

### Fixtures

- `tests/fixtures/markitdown_sample.txt` — verified conversion
- `tests/fixtures/markitdown_sample.html` — verified conversion
- `tests/fixtures/markitdown_sample.pdf` — genuine parseable PDF, verified conversion

### Critical upstream security finding

The upstream `convert_to_markdown` tool accepts and will fetch `http:`, `https:`, and `data:` URIs. The server does **not** enforce local-only conversion. Remote/UNC/traversal URI denial and exact-file replacement are the responsibility of this project's invocation policy, enforced before `client.call_tool`.

---

## Mandatory architecture

### 1. Two immutable document models

`mcp_management/document_authorization.py` will contain:

```python
@dataclass(frozen=True)
class DocumentInputSnapshot:
    request_id: str
    original_path: str
    canonical_path: str
    file_uri: str
    extension: str
    size_bytes: int
    sha256: str
    modified_time_ns: int
    captured_at: datetime

@dataclass(frozen=True)
class DocumentInputAuthorization:
    authorization_id: str
    request_id: str
    server_id: str
    remote_tool_name: str
    canonical_path: str
    trusted_file_uri: str
    extension: str
    expected_size_bytes: int
    expected_sha256: str
    expected_modified_time_ns: int
    created_at: datetime
    expires_at: datetime
    single_use: bool = True

class DocumentAuthorizationState(Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    CONSUMED = "consumed"
    EXPIRED = "expired"
```

`DocumentAuthorizationStore` will:
- store state keyed by `(request_id, authorization_id)`
- provide atomic `reserve(request_id, authorization_id)` and `consume(request_id, authorization_id)` operations
- support snapshot capture and revalidation
- generate canonical local `file:///` URIs
- enforce the reserve → call → consume sequence

### 2. Plan hash covers complete identity

`AutoProvisioningPlan` will carry `document_snapshots: Tuple[DocumentInputSnapshot, ...]`.

The plan hash will include, in canonical order:
- request ID
- server ID
- catalog ID
- capability ID
- for each snapshot: canonical_path, file_uri, extension, size_bytes, sha256, modified_time_ns, captured_at
- package name
- exact package version
- artifact hash
- dependency lock hash
- lock environment tuple (implementation, python_major_minor, platform)
- launch configuration
- candidate_validator ID
- expected tools
- tool-policy hash
- invocation-policy hash
- installation destination
- runtime network-policy statement

### 3. Invocation policy

`mcp_layer/invocation_policy.py` will contain:

- `McpInvocationPolicy` protocol
- `NoOpInvocationPolicy`
- `LocalDocumentExactFilePolicy`

The policy will:
1. Require exactly one `uri` argument.
2. Parse it structurally.
3. Reject: `http`, `https`, `data:`, `ftp:`, other unsupported schemes, UNC/network shares, `file://server/share`, relative paths, traversal paths, directories, globs, arrays/lists, multiple inputs, and any local file different from the authorization.
4. Accept either an absolute local path or an equivalent local `file://` URI that resolves to the authorized canonical path.
5. Replace the model-provided value with `trusted_file_uri`.
6. Return sanitized arguments.

Policy failures occur **before** `client.call_tool`.

### 4. Atomic single-use authorization flow

1. Capture `DocumentInputSnapshot` before showing the provisioning plan.
2. Include the snapshot in the plan hash.
3. User approves.
4. Install in isolated Python environment.
5. Run candidate validator `markitdown_local_document_v1`.
6. Atomically activate installed state.
7. G.2 starts MarkItDown lazily.
8. Revalidate the file identity (path, regular-file status, size, modified time, SHA-256).
9. If unchanged, create a short-lived single-use `DocumentInputAuthorization`.
10. **Reserve the authorization atomically.**
11. Call the MCP server with `trusted_file_uri`.
12. **Immediately mark the authorization consumed** after the single MCP call attempt, regardless of remote success or failure.
13. Normalize and size-check the result.
14. Local LLM summarizes complete Markdown.
15. Return the summary.

If the file changed after approval or after authorization: return `MCP_DOCUMENT_CHANGED`.

### 5. Python lock environment pinning

`config/mcp_locks/markitdown-mcp-0.0.1a4.txt` will be accompanied by metadata declaring:

```python
{
  "implementation": "CPython",
  "python_major_minor": "3.13",
  "platform": "win_amd64"
}
```

The installer will:
- fail closed if the running interpreter does not match the reviewed lock environment, OR
- support multiple reviewed lock files, one per approved Python/platform combination.
- pass `--no-deps` when `install_options.no_deps` is true, because the production lock intentionally omits the upstream `[all]` extras.

No dynamic pip resolution is permitted.

### 6. No placeholder or partial hashes

The lock file must contain:
- exact versions for all direct and transitive dependencies
- verified hashes for every installable artifact
- no editable, Git, or local path dependencies
- no wildcard versions or `latest`
- no placeholder or temporary hashes

If exact artifacts and hashes cannot be obtained: stop, report the blocker, do not weaken security, and do not add the production catalog entry as enabled.

### 7. Versioned candidate validator

Catalog entry uses:

```json
"candidate_validator": "markitdown_local_document_v1"
```

`mcp_management/candidate_validators.py` will contain:
- `CandidateValidator` protocol
- `validator_registry = {"markitdown_local_document_v1": validate_markitdown_local_document_v1}`
- The generic provisioner calls only the validator selected by the catalog entry.

Unknown validator IDs fail closed.

### 8. Verified upstream behavior

Before implementation, inspect the installed artifact for `markitdown-mcp==0.0.1a4` and document:
- exact console entrypoint
- transport behavior
- exact remote tool name
- exact argument schema
- exact result schema
- resources or prompts exposed, if any
- environment variables actually read
- plugin behavior in this pinned version
- startup behavior
- runtime network behavior
- supported formats
- Windows compatibility

No invented environment variables, result fields, launch arguments, tool names, or controls.

If plugins are disabled by default:
- do not install plugin packages
- do not pass plugin-enabling options
- validate only the exact reviewed tool surface

### 9. Result normalization

`mcp_layer/document_conversion.py` will contain:

```python
@dataclass(frozen=True)
class DocumentConversionResult:
    source_name: str
    source_extension: str
    source_size_bytes: int
    source_sha256: str
    markdown: str
    markdown_char_count: int
    server_id: str
    remote_tool_name: str
    installed_version: str
```

It will:
- inspect the pinned package and document the actual MCP tool result shape
- normalize only verified result forms
- return `MCP_DOCUMENT_RESULT_INVALID` for unexpected shapes
- exclude server stderr, venv paths, environment values, dependency locations, and internal config paths

### 10. No silent truncation

If Markdown exceeds the configured inline limit:
- return `MCP_DOCUMENT_OUTPUT_TOO_LARGE`
- include metadata only: source name, source size, Markdown character count, configured maximum
- do not send partial content to the LLM

Chunking, summarization, indexing, and RAG are out of scope.

### 11. Candidate validation with real PDF

Validation must include:
- `tests/fixtures/markitdown_sample.txt` or `markitdown_sample.html`
- `tests/fixtures/markitdown_sample.pdf` — a genuinely parseable real PDF with known text

Steps:
1. Start the real candidate process.
2. Initialize it.
3. Call `tools/list`.
4. Verify exact expected remote tools.
5. Call `convert_to_markdown` on the TXT/HTML fixture and verify known content.
6. Call `convert_to_markdown` on the PDF fixture and verify known PDF text appears in normalized Markdown.
7. Reject an HTTP URI through the project invocation policy.
8. Shut down the candidate.
9. Confirm no orphan process.
10. Avoid production `ToolRegistry` registration.

### 12. Discovery and Phase B remain separate

- `mcp_layer/discovery.py` discovers the exact remote tool, creates the namespaced registry entry, attaches ownership, READ permission, and invocation-policy identity.
- Phase B receives `preferred_mcp_server_id="markitdown"` from capability selection and boosts tools owned by that server, including `mcp.markitdown.convert_to_markdown`, within the shortlist bound.
- No Phase B logic lives in discovery, candidate validation, capability selection, or `ToolExecutor`.

### 13. Tool permission

`convert_to_markdown` is classified as `READ`.

No second generic confirmation is added for each explicitly requested document conversion.

User gates:
- first-time provisioning approval
- request-bound exact-file authorization

### 14. Error codes

Add to `tools/models.py`:

- `MCP_DOCUMENT_AUTHORIZATION_REQUIRED`
- `MCP_DOCUMENT_AUTHORIZATION_EXPIRED`
- `MCP_DOCUMENT_AUTHORIZATION_CONSUMED`
- `MCP_DOCUMENT_PATH_INVALID`
- `MCP_DOCUMENT_PATH_MISMATCH`
- `MCP_DOCUMENT_CHANGED`
- `MCP_DOCUMENT_URI_SCHEME_DENIED`
- `MCP_DOCUMENT_EXTENSION_DENIED`
- `MCP_DOCUMENT_INPUT_TOO_LARGE`
- `MCP_DOCUMENT_NOT_FOUND`
- `MCP_DOCUMENT_NOT_LOCAL`
- `MCP_DOCUMENT_OUTPUT_TOO_LARGE`
- `MCP_DOCUMENT_RESULT_INVALID`
- `MCP_DOCUMENT_CONVERSION_TIMEOUT`
- `MCP_MARKITDOWN_CONVERSION_FAILED`
- `MCP_CANDIDATE_FUNCTIONAL_VALIDATION_FAILED`

Each code must have deterministic trigger conditions. Policy failures occur before `client.call_tool`.

### 15. Honest network security language

Documentation and plan summaries will state:

- Remote document URI schemes are denied by host invocation policy.
- Only one exact local file URI is sent to the MCP tool.
- No runtime credential variables are supplied.
- Third-party plugins are not installed or enabled.
- **OS-level outbound networking is not sandboxed.**
- **The local MarkItDown process may technically have network access.**
- **The project does not claim firewall-level isolation.**

---

## File list

### New files

- `docs/security/MARKITDOWN_MCP_REVIEW.md`
- `docs/PHASE_G4_MARKITDOWN_PROVIDER.md`
- `config/mcp_locks/markitdown-mcp-0.0.1a4.txt`
- `mcp_management/document_authorization.py`
- `mcp_management/candidate_validators.py`
- `mcp_layer/invocation_policy.py`
- `mcp_layer/document_conversion.py`
- `tests/fixtures/markitdown_sample.txt`
- `tests/fixtures/markitdown_sample.html`
- `tests/fixtures/markitdown_sample.pdf`
- `tests/test_mcp_document_authorization.py`
- `tests/test_mcp_invocation_policy.py`
- `tests/test_mcp_markitdown_catalog.py`
- `tests/test_mcp_markitdown_integration.py`
- `tests/test_assistant_markitdown_provisioning.py`
- `scripts/manual_verify_g4_markitdown.py`

### Modified files

- `config/mcp_catalog.json`
- `mcp_management/catalog.py`
- `mcp_management/provisioning_models.py`
- `mcp_management/installers/python_venv.py`
- `mcp_management/configuration_generator.py`
- `mcp_management/auto_provisioning.py`
- `mcp_layer/discovery.py`
- `mcp_layer/tool.py`
- `mcp_layer/external.py`
- `tools/models.py`
- `assistant.py`
- `tests/test_mcp_capability_file_intents.py`
- `tests/test_mcp_candidate_validation.py`
- `tests/test_mcp_python_venv_installer.py`
- `tests/test_mcp_configuration_generation.py`

---

## Catalog entry design

```json
{
  "catalog_id": "official-markitdown",
  "server_id": "markitdown",
  "display_name": "Microsoft MarkItDown MCP",
  "capabilities": {
    "document_to_markdown": {
      "evidence": {
        "tool_names": ["convert_to_markdown"],
        "extensions": [".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".html", ".htm"],
        "paths": []
      },
      "required": true
    }
  },
  "installer": {
    "type": "python_venv",
    "package_spec": "markitdown-mcp==0.0.1a4",
    "lock_file": "config/mcp_locks/markitdown-mcp-0.0.1a4.txt",
    "install_options": {
      "no_deps": true,
      "require_hashes": true
    },
    "lock_environment": {
      "implementation": "CPython",
      "python_major_minor": "3.13",
      "platform": "win_amd64"
    }
  },
  "launch": {
    "entrypoint_type": "console_script",
    "console_script": "markitdown-mcp"
  },
  "candidate_validator": "markitdown_local_document_v1",
  "expected_tools": ["convert_to_markdown"],
  "default_tool_policy": {
    "tools": {
      "convert_to_markdown": {
        "enabled": true,
        "permission": "READ"
      }
    },
    "default_permission": "denied"
  },
  "invocation_policy": {
    "id": "local_document_exact_file_v1",
    "allowed_extensions": [".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".html", ".htm"],
    "max_input_bytes": 52428800,
    "max_output_chars": 200000,
    "conversion_timeout_seconds": 60
  }
}
```

Catalog validation will reject:
- missing invocation policy
- missing candidate validator
- unknown policy or validator
- non-exact version
- lock environment mismatch
- missing lock hash
- wildcard version
- placeholder hashes
- unexpected enabled tools
- unknown permission values

---

## First-use flow

User: `summarize C:\Users\Prudhvi\OneDrive\learn stuff\Hands_on_LLM.pdf`

1. Router proposes local or Claude.
2. G.1 detects `document_to_markdown`.
3. Local-capability routing selects effective local mode.
4. Trusted catalog selects `markitdown`.
5. Capture `DocumentInputSnapshot`.
6. MarkItDown is not installed.
7. G.3 creates a provisioning plan bound to package, lock, environment, policies, and complete document snapshot.
8. User approves.
9. Revalidate plan and snapshot.
10. Install in isolated Python environment.
11. Run `markitdown_local_document_v1` candidate validation.
12. Atomically activate installed state.
13. G.2 starts MarkItDown lazily.
14. Revalidate the snapshot again.
15. Create short-lived single-use `DocumentInputAuthorization`.
16. **Reserve the authorization atomically.**
17. Phase B includes `mcp.markitdown.convert_to_markdown`.
18. Local LLM selects the offered tool.
19. Invocation policy validates the model URI against the authorization.
20. Invocation policy replaces it with `trusted_file_uri`.
21. Call the MCP server once.
22. **Immediately consume the authorization** after the MCP call attempt.
23. Normalize only the verified result shape.
24. Enforce output limit without truncation.
25. Local LLM summarizes complete Markdown.
26. Return the summary.

---

## Reuse flow

For a later document request:

1. G.1 selects MarkItDown.
2. G.3 sees it is installed; no provisioning plan.
3. G.2 starts or reuses the healthy runtime.
4. Capture and validate a new `DocumentInputSnapshot`.
5. Create a new single-use `DocumentInputAuthorization`.
6. Reserve → call once → consume.
7. Convert and summarize.

The previous file authorization must not be reusable.

---

## Security tests to add

Prove no MCP client call occurs for:

- `http://example.com/report.pdf`
- `https://example.com/report.pdf`
- `data:text/plain,secret`
- `ftp://example.com/report.pdf`
- `\\server\share\report.pdf`
- `file://server/share/report.pdf`
- `relative/path/report.pdf`
- `../report.pdf`
- `C:\different\secret.pdf`
- directory paths
- wildcard paths
- arrays or multiple inputs
- another request’s authorization
- expired authorization
- consumed authorization
- authorization reserved concurrently
- file changed after approval
- file changed after authorization
- file deleted before conversion
- file enlarged beyond limit
- unsupported extension substitution
- malformed result shape
- oversized Markdown output

Also test:

- valid absolute Windows path
- equivalent trusted local file URI
- path with spaces
- OneDrive-local hydrated file
- repeated request creates a new authorization
- concurrent calls cannot both consume one authorization

---

## Regression requirements

Preserve:

- `tools/executor.py` unchanged
- `router.py` remains local/Claude only
- Phase B bound unchanged
- Filesystem MCP behavior unchanged
- no Filesystem root-access inheritance
- no binary-PDF fallback through Filesystem MCP
- no raw local path sent to Claude
- no installation before approval
- no candidate tools registered into production registry
- no public MCP discovery
- no orphan processes

Current baseline:

- `1060 passed, 3 skipped`

Final target:

- higher total, zero regressions

---

## Checkpoint / approval gates

1. **Plan approved** — this document.
2. **Upstream verification complete** ✅ — exact tool schema, result schema, entrypoint, environment, hashes, PDF fixture. Readiness is now READY.
3. **Catalog + lock file + installer/config support** — run catalog, installer, config tests.
4. **Authorization + invocation policy** — run policy and authorization tests.
5. **Validator registry + candidate validation** — run candidate validation tests with PDF fixture.
6. **Runtime integration + Phase B shortlisting** — run manual verification and integration tests.
7. **Full suite green** — implementation report.

---

## Changes made from the previous plan

1. Replaced single `DocumentInputAuthorization` with two immutable models: `DocumentInputSnapshot` and `DocumentInputAuthorization`.
2. Expanded plan hash to include complete file identity and trust context, not just paths.
3. Added atomic single-use authorization with separate state store and explicit lifecycle.
4. **Added mandatory sequencing correction**: reserve before MCP call, consume immediately after the single conversion attempt, before normalization or summarization; authorization remains consumed on failure; no automatic retry with same authorization.
5. Specified that the invocation policy ignores the model URI and replaces it with the trusted file URI, while still rejecting malicious URIs before any MCP client call.
6. Added Python lock environment pinning and fail-closed behavior.
7. Removed any allowance for placeholder or partial hashes; stated explicit stop-and-report blocker policy.
8. Replaced generic validator ID with versioned `"markitdown_local_document_v1"` and a registry.
9. Added requirement to verify actual upstream package behavior and document real artifacts.
10. Restricted result normalization to verified shapes and added `MCP_DOCUMENT_RESULT_INVALID`.
11. Added many more granular error codes.
12. Added real PDF fixture requirement for candidate validation.
13. Clarified separation between discovery and Phase B shortlisting.
14. Added explicit honest network security language.
15. Added expanded security test list and regression target.

---

## Verified upstream blockers or uncertainties

All prerequisites are resolved. No blockers remain.

Minor residual risks (documented in `docs/security/MARKITDOWN_MCP_REVIEW.md`):
- Transitive `requests` and `httpx` remain installed; OS-level outbound networking is not sandboxed.
- Native-code dependencies (`onnxruntime`, `pypdfium2`, etc.) are hash-verified but not behaviorally audited.
- `magika` uses an ONNX model for content-type detection.

---

## Implementation readiness decision

**READY**

All four prerequisites are satisfied:
1. `markitdown-mcp==0.0.1a4` inspected directly.
2. Exact entrypoint, tool schema, result schema, environment, and plugin behavior documented.
3. Complete hash-locked dependency set produced for CPython 3.13 / `win_amd64`.
4. Genuine parseable PDF fixture prepared and validated.

Main Phase G.4 production integration may now begin, checkpoint by checkpoint.

---

## What will NOT happen without explicit approval

- `tools/executor.py` will not be modified.
- `router.py` will not be modified.
- Phase B bound or scoring will not be weakened.
- Existing Filesystem MCP behavior will not be changed.
- No lock file will be committed.
- No hash will be weakened or placeholder-replaced.
- No `ToolExecutor`, router, or Phase B logic will be modified.
