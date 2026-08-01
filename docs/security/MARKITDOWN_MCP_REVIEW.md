# Security and Behavior Review — Microsoft MarkItDown MCP 0.0.1a4

**Review date:** 2026-07-31  
**Reviewed artifacts:**
- Wheel: `markitdown_mcp-0.0.1a4-py3-none-any.whl`
  - SHA-256: `7fb06fff7d722ec108d08752704dc8f313f7fd267e00dff546131ec229645230`
- Source distribution: `markitdown_mcp-0.0.1a4.tar.gz`
  - SHA-256: `309c94dc883311e6909d849382a6c7bc402dfb2692dab448c136c6864c6bf49e`
- Transitive library: `markitdown-0.1.7-py3-none-any.whl`
  - SHA-256: `4eca912c87c6aa6897284a7f4bf6769a23bccf8544530f5d8b175fbe3797c916`

**Reviewed environment:**
- Implementation: CPython
- Python major/minor: 3.13
- Operating system: Windows 11
- Architecture/platform tag: `win_amd64`

**Scope:** This review covers the exact pinned version `markitdown-mcp==0.0.1a4` as distributed by PyPI. It does **not** cover the upstream `main` branch or any later version.

---

## 1. Console entrypoint and launch behavior

The package declares one console entrypoint:

```ini
[console_scripts]
markitdown-mcp = markitdown_mcp.__main__:main
```

Inside the isolated venv this resolves to `venv/Scripts/markitdown-mcp.exe` on Windows.

Default transport is **stdio**. The server also supports `--http` (Streamable HTTP + SSE) and `--sse` (deprecated alias), bound by default to `127.0.0.1:3001`. This project uses **only the stdio transport**.

**Production launch method:** although the package declares a console script, the
catalog is configured to launch the server with `python.exe -m markitdown_mcp`.
This avoids the Windows console-script `.exe` shim breaking when the venv is
relocated during the atomic candidate-to-final promotion, and it still uses the
same `markitdown_mcp.__main__:main` entrypoint.

Command-line surface:
- `--http` / `--sse` — disabled in this project
- `--host`, `--port` — rejected by argparse when not using HTTP/SSE
- No other arguments

No environment variables are read by `markitdown_mcp.__main__` in the reviewed code.

---

## 2. MCP tool surface

`tools/list` returns exactly one tool:

```json
{
  "name": "convert_to_markdown",
  "inputSchema": {
    "properties": {
      "uri": {
        "title": "Uri",
        "type": "string"
      }
    },
    "required": ["uri"],
    "title": "convert_to_markdownArguments",
    "type": "object"
  }
}
```

Tool result shape (verified against local `file://` URIs):

```json
{
  "text": "<markdown string>"
}
```

The tool is declared `async def convert_to_markdown(uri: str) -> str`. The server wraps the returned `str` as a text content item.

---

## 3. Upstream URI policy

The upstream tool docstring states:

> Convert a resource described by an http:, https:, file: or data: URI to markdown

The implementation is:

```python
return MarkItDown().convert_uri(uri).markdown
```

**Critical finding:** the upstream server itself accepts and will fetch `http:`, `https:`, and `data:` URIs. It does **not** enforce local-only conversion. Any remote-URI restriction must be applied by the host (this project) before calling the MCP server.

---

## 4. Resources and prompts

`resources/list` returns `{"resources": []}`.

`prompts/list` returns `{"prompts": []}`.

No resources or prompts are exposed by the reviewed version.

---

## 5. Plugin behavior

The upstream package declares:

```text
Requires-Dist: markitdown[all]<0.2.0,>=0.1.1
```

`markitdown[all]` installs the following optional, network-capable or cloud-backed extras:
- `azure-ai-contentunderstanding`
- `azure-ai-documentintelligence`
- `azure-identity`
- `pydub`
- `speechrecognition`
- `youtube-transcript-api`

For this local-first project, those plugins are **not installed**. The production lock file intentionally omits the `[all]` extra and installs only the curated local-format dependencies (`docx`, `pdf`, `pptx`, `xls`, `xlsx`) plus the base runtime.

No plugin-enabling environment variable is passed. No plugin-enabling command-line option is used.

**Important:** even with the reduced dependency set, the installed environment still contains `requests` and `httpx` as transitive dependencies of `markitdown` and `mcp`. Therefore the server process still has network-capable libraries loaded. OS-level outbound network isolation is not enforced by this project.

---

## 6. Dependency metadata

Production lock file:
- Path: `config/mcp_locks/markitdown-mcp-0.0.1a4.txt`
- Packages: 60
- Hash-locked: yes, every package has a verified SHA-256
- Lock file SHA-256: `b94ace40f56fa7ab74c632105ad4b1c0195c7e2c0dd6294e0c7d88e302491222`
- Installation flags: `pip install --require-hashes --no-deps -r <this-file>`
- Reason for `--no-deps`: `markitdown-mcp` declares `markitdown[all]` as a dependency. The production lock intentionally omits the `[all]` extras, so dependency resolution must be disabled.
- Reviewed environment: CPython 3.13 / `win_amd64`

The lock file intentionally does **not** include:
- editable dependencies
- Git dependencies
- wildcard versions
- `latest`
- placeholder hashes
- untrusted local path dependencies

---

## 7. Network behavior

- **Startup:** the stdio server does not perform network operations during startup in the reviewed code.
- **Runtime local conversion:** reads only the local file URI supplied in the `uri` argument.
- **Runtime remote conversion:** the upstream server will fetch remote URIs if passed. This project prevents that by host invocation policy.
- **OS-level outbound networking:** **not sandboxed**. The MarkItDown child process runs with the same network privileges as the parent process.

---

## 8. Supported local formats (with reduced dependency set)

Verified working:
- Plain text (`.txt`)
- HTML (`.html`, `.htm`)
- PDF (`.pdf`) — via `pdfplumber` / `pdfminer.six`
- DOCX (`.docx`) — via `mammoth` + `lxml`
- XLSX (`.xlsx`) — via `openpyxl` + `pandas`
- XLS (`.xls`) — via `xlrd` + `pandas`
- PPTX (`.pptx`) — via `python-pptx`

Audio transcription, YouTube transcription, and Azure cloud conversion are **not installed** and therefore not supported.

---

## 9. Windows stdio behavior

- The console entrypoint produces a Windows `.exe` shim in `venv/Scripts/markitdown-mcp.exe`.
- The shim launches `python.exe -m markitdown_mcp`.
- stdio transport uses the parent process's stdin/stdout pipes.
- First startup can take several seconds due to heavy imports (`magika`, `onnxruntime`, `pandas`, `pydantic`, etc.). A 60-second startup timeout is recommended for first use.

---

## 10. Security controls provided by this project

- **Catalog control:** only `markitdown-mcp==0.0.1a4` from the trusted catalog may be installed.
- **Hash-locked install:** every package hash is verified by `pip --require-hashes`.
- **Reduced dependency surface:** `[all]` extras are omitted; only local-format dependencies are installed.
- **Exact-file authorization:** before conversion the assistant captures a `DocumentInputSnapshot`, revalidates it, and creates a single-use `DocumentInputAuthorization`.
- **Invocation policy:** `LocalDocumentExactFilePolicy` runs before `client.call_tool`. It rejects remote schemes, UNC paths, relative paths, directories, globs, arrays, multiple inputs, and any local file that does not match the active authorization. It replaces the model-provided URI with the trusted `file_uri`.
- **READ permission:** `convert_to_markdown` is classified as READ; no additional generic confirmation is added per document.
- **No truncation:** oversized Markdown results return `MCP_DOCUMENT_OUTPUT_TOO_LARGE` without sending partial content to the LLM.
- **Atomic single-use authorization:** reserved before the MCP call and consumed immediately after the single conversion attempt.

---

## 11. Honest network security language

- Runtime remote-document access: **denied by host invocation policy**.
- OS-level outbound networking: **not sandboxed**.
- The local MarkItDown process may technically have network access.
- This project does **not** claim firewall-level isolation.

---

## 12. Unresolved risks

1. The server process contains `requests` and `httpx`. A bug or prompt-injection bypass in the invocation policy could allow a remote URI to reach the server.
2. Some transitive dependencies (e.g., `pypdfium2`, `onnxruntime`) download or load native code. Their integrity is covered by hash verification but their runtime behavior is not audited.
3. `magika` uses an ONNX model for content-type guessing. The model file is bundled in the wheel and hash-verified, but its predictions are a black box.
4. OS-level outbound network isolation is not implemented. A separate firewall or sandbox is recommended for high-assurance deployments.

---

## 13. Verification summary

- Wheel hash verified independently: ✅
- Sdist hash verified independently: ✅
- Exact tool name verified: `convert_to_markdown` ✅
- Exact input schema verified: single required string `uri` ✅
- Exact result shape verified: `{"text": "..."}` ✅
- Resources/prompts verified empty: ✅
- Local TXT conversion verified: ✅
- Local PDF conversion verified: ✅
- Clean lock install with `--require-hashes --no-deps` verified: ✅
- Reduced dependency set (no `[all]` extras) verified: ✅
