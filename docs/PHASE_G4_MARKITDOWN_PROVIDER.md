# Phase G.4 Implementation Report — Production MarkItDown MCP Provider

**Status:** COMPLETE  
**Date:** 2026-07-31  
**Reviewed environment:** CPython 3.13 / Windows `win_amd64`

---

## Goal

Add Microsoft `markitdown-mcp==0.0.1a4` as the first production provider for the
`document_to_markdown` capability, using only the existing Phase G.1–G.3
framework, Phase B bounded shortlisting, and `ToolExecutor` without modification.

---

## What was implemented

### 1. Trusted catalog entry

File: `config/mcp_catalog.json`

- Entry ID: `official-markitdown`
- Server ID: `markitdown`
- Capability: `document_to_markdown`
- Package: `markitdown-mcp==0.0.1a4`
- Installer: `python_venv` with `--require-hashes --no-deps`
- Launch: `python_module: markitdown_mcp` (see §5 for the console-script → module decision)
- Candidate validator: `markitdown_local_document_v1`
- Invocation policy: `argument_mode: exact_file_uri`
- Status: **enabled** after all gating validations passed

### 2. Hash-locked dependency set

File: `config/mcp_locks/markitdown-mcp-0.0.1a4.txt`

- 60 packages
- Every artifact has at least one verified SHA-256
- No editable/Git/URL/wildcard/placeholder dependencies
- Installed with exact production flags:
  `pip install --require-hashes --no-deps --no-input --disable-pip-version-check -r config/mcp_locks/markitdown-mcp-0.0.1a4.txt`
- Lock file SHA-256: `b94ace40f56fa7ab74c632105ad4b1c0195c7e2c0dd6294e0c7d88e302491222`
- The lock intentionally omits upstream `[all]` extras (`azure-ai-*`, `pydub`,
  `speechrecognition`, `youtube-transcript-api`).

### 3. Lock-environment fail-closed check

File: `mcp_management/installers/python_venv.py`

- Added platform-tag normalization so `win_amd64` and `win-amd64` are treated as
  equivalent.
- Fails closed when the running interpreter does not match the reviewed
  `python_version`/`platform`.

### 4. Document authorization and plan binding

Files: `mcp_management/document_authorization.py`,
`mcp_management/provisioning_models.py`,
`mcp_management/auto_provisioning.py`, `assistant.py`

- `DocumentInputSnapshot` is captured from the user's text before the plan is
  shown and bound into the provisioning plan hash.
- `AutoProvisioningPlan` carries `document_snapshots` and includes them in
  `security_fields()` / `compute_hash()`.
- On resumption, each snapshot is revalidated and a short-lived
  `DocumentInputAuthorization` is created.
- Authorization is single-use: reserved before the MCP call, consumed
  immediately after the single conversion attempt, and stays consumed if
  conversion/normalization/summarization fails.

### 5. Launch-method decision: console script → python module

The upstream package declares a console script `markitdown-mcp`. Initial testing
used it, but after venv relocation from the candidate directory to the final
`versions/<version>/venv` directory, the Windows `.exe` shim broke silently
(`exit 1`, no stderr).

Fix: the catalog entry now launches with `python.exe -m markitdown_mcp`. This:

- uses the same upstream `markitdown_mcp.__main__:main` entrypoint,
- is robust to venv relocation,
- still runs inside the isolated venv,
- requires no ToolExecutor or framework changes.

This is the only implementation deviation from the literal catalog snippet in
the original plan.

### 6. Candidate validator

File: `mcp_management/candidate_validators.py`

- `markitdown_local_document_v1` verifies the exact `convert_to_markdown` schema
  (single required string `uri`).
- It dynamically tests every extension advertised in the catalog against real
  committed fixtures.
- Markers use hyphenated form `G4-VERIFY-<EXT>-2026` because MarkItDown escapes
  underscores in Markdown output.

### 7. Fixtures

Files under `tests/fixtures/`:

- `markitdown_sample.txt`
- `markitdown_sample.html`
- `markitdown_sample.htm`
- `markitdown_sample.pdf` (genuine parseable PDF)
- `markitdown_sample.docx` (minimal valid DOCX package)
- `markitdown_sample.pptx` (created with `python-pptx`)
- `markitdown_sample.xls` (created via Excel COM/pywin32)
- `markitdown_sample.xlsx` (created with `openpyxl`)

### 8. Invocation policy and result normalization

File: `mcp_layer/tool.py`, tests: `tests/test_mcp_invocation_policy.py`

- `LocalDocumentExactFilePolicy` (activated by `argument_mode: "exact_file_uri"`):
  - rejects remote schemes, UNC/network shares, relative paths, traversal,
    directories, globs, arrays, and any local path that does not match the
    active authorization;
  - replaces the model-provided URI with the trusted `file://` URI.
- Result normalization accepts only the verified `{"text": "..."}` shape and
  enforces an output-size limit without truncation.
- Authorization is consumed even when the MCP call or normalization fails.

### 9. Version-validation tightening

File: `mcp_management/catalog.py`

- The catalog version regex was tightened so that a trailing separator such as
  `2.0.0-` is rejected, while valid pre-release suffixes like `0.0.1a4` and
  `1.2.3-rc.1` remain accepted.

### 10. Regression isolation for G.1 tests

Files: `tests/test_capability_gating_regression.py`,
`tests/test_mcp_capability_file_intents.py`

- Existing tests that verify the "no document provider installed" path now load
  a fixture copy of the production catalog with `official-markitdown` disabled.
- This preserves the original test semantics after the production entry is
  enabled.

---

## Verification results

### Manual integration script

```text
venv/Scripts/python.exe scripts/manual_verify_g4_markitdown.py
```

Result: **PASS**

- Clean venv created.
- Lock installed with exact production flags.
- All 8 advertised extensions converted and their markers found:
  `.txt`, `.html`, `.htm`, `.pdf`, `.docx`, `.pptx`, `.xls`, `.xlsx`.
- Runtime activated healthy with 1 tool registered.

### Full regression suite

```text
venv/Scripts/python.exe -m pytest tests -q --tb=short
```

Result: **1116 passed, 3 skipped, 0 failed**.

Protected files confirmed unchanged:
- `tools/executor.py`
- `router.py`

---

## Security and behavior notes

- Remote URI conversion is blocked by host invocation policy; the upstream
  server itself accepts `http:`, `https:`, and `data:` URIs.
- Only one exact local `file://` URI is ever sent to the MCP tool.
- No runtime credential variables are supplied.
- Third-party cloud/audio/YouTube plugins are not installed or enabled.
- OS-level outbound networking is **not sandboxed**; the MarkItDown child
  process runs with the same network privileges as the parent.
- The project does **not** claim firewall-level isolation.

See `docs/security/MARKITDOWN_MCP_REVIEW.md` for the complete upstream behavior
review.

---

## Files changed

- `config/mcp_catalog.json` — added and enabled `official-markitdown`
- `mcp_management/catalog.py` — python_venv/installer support, version regex
  tightening
- `mcp_management/installers/python_venv.py` — lock-environment platform
  normalization
- `mcp_management/provisioning_models.py` — `document_snapshots` bound into plan
- `mcp_management/auto_provisioning.py` — snapshot carriage and resume
  authorization creation
- `mcp_management/document_authorization.py` — snapshot/authorization models and
  store
- `mcp_management/candidate_validators.py` — `markitdown_local_document_v1`
- `mcp_layer/tool.py` — exact-file invocation policy + result normalization
- `assistant.py` — capture document snapshots for `document_to_markdown`
- `mcp_management/runtime_activation.py` — updated not-installed message
- `docs/security/MARKITDOWN_MCP_REVIEW.md` — noted production launch method
- `docs/PHASE_G4_MARKITDOWN_PROVIDER.md` — this report
- `scripts/manual_verify_g4_markitdown.py` — real-process integration script
- `tests/*` — authorization, invocation-policy, candidate-validator,
  provisioning-plan-hash, document-auth, and regression-isolation tests
- `tests/fixtures/markitdown_sample.*` — real fixtures for every advertised
  extension

---

## Residual risks

Documented in `docs/security/MARKITDOWN_MCP_REVIEW.md`:

1. `requests`/`httpx` remain loaded as transitive dependencies.
2. Native-code dependencies (`onnxruntime`, `pypdfium2`) are hash-verified but
   not behaviorally audited.
3. `magika` uses a bundled ONNX model for content-type guessing.
4. OS-level outbound network isolation is not implemented.

---

## Conclusion

Phase G.4 is implemented, verified, and the production MarkItDown MCP catalog
entry is enabled. All gating criteria from the approved plan are satisfied and
the full regression suite is green.
