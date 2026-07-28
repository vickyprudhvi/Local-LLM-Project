"""Bounded static security scan of a cloned repository.

Python files are analyzed with the `ast` module (parsed, never executed); other
files are checked with conservative text patterns. This is NOT a security
certification: a clean scan means only that no configured pattern matched. It
never claims a repository is safe, trusted, or malware-free. Possible secrets are
redacted from evidence. Bounded by file count, file size, total bytes, and findings.
"""

import ast
import os
import re

import tools.config as config
from tools import repo_store
from tools.repo_analysis import collect_files, read_text

# ---- secret detection + redaction ----
# (pattern, severity) — used both to emit a "credentials" finding and to redact evidence.
_CREDENTIAL_PATTERNS = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "high"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "high"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "high"),
    (re.compile(r"sk-[A-Za-z0-9\-]{20,}"), "high"),
    (re.compile(r"tvly-[A-Za-z0-9\-]{10,}"), "high"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "high"),
    (re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{6,}['\"]"), "medium"),
]
_SECRET_PATTERNS = [pat for pat, _ in _CREDENTIAL_PATTERNS]


def _redact(text):
    if not text:
        return ""
    for pat in _SECRET_PATTERNS:
        text = pat.sub("***REDACTED***", text)
    return text[:200]


# ---- Python AST checks: dotted-call name -> (category, severity, reason) ----
_CALL_RULES = {
    "eval": ("code_execution", "high", "Evaluates code at runtime."),
    "exec": ("code_execution", "high", "Executes code at runtime."),
    "compile": ("code_execution", "medium", "Compiles code at runtime."),
    "__import__": ("dynamic_import", "medium", "Imports modules dynamically."),
    "importlib.import_module": ("dynamic_import", "medium", "Imports modules dynamically."),
    "os.system": ("process_execution", "high", "Runs a shell command."),
    "os.popen": ("process_execution", "high", "Runs a shell command."),
    "subprocess.run": ("process_execution", "high", "Starts an external process."),
    "subprocess.call": ("process_execution", "high", "Starts an external process."),
    "subprocess.Popen": ("process_execution", "high", "Starts an external process."),
    "subprocess.check_output": ("process_execution", "high", "Starts an external process."),
    "subprocess.check_call": ("process_execution", "high", "Starts an external process."),
    "pickle.load": ("unsafe_deserialization", "high", "Deserializes untrusted data."),
    "pickle.loads": ("unsafe_deserialization", "high", "Deserializes untrusted data."),
    "marshal.loads": ("unsafe_deserialization", "high", "Deserializes untrusted data."),
    "dill.loads": ("unsafe_deserialization", "high", "Deserializes untrusted data."),
    "shutil.rmtree": ("filesystem", "medium", "Recursively deletes files."),
    "os.remove": ("filesystem", "low", "Deletes files."),
    "os.chmod": ("filesystem", "low", "Changes file permissions."),
    "os.symlink": ("filesystem", "low", "Creates symbolic links."),
    "ctypes.CDLL": ("native_access", "medium", "Loads native libraries."),
    "socket.socket": ("network", "low", "Opens network sockets."),
}
_IMPORT_RULES = {
    "requests": ("network", "low", "Performs network requests."),
    "httpx": ("network", "low", "Performs network requests."),
    "urllib": ("network", "low", "Performs network requests."),
    "aiohttp": ("network", "low", "Performs network requests."),
    "socket": ("network", "low", "Uses network sockets."),
    "ctypes": ("native_access", "medium", "Accesses native code."),
    "cffi": ("native_access", "medium", "Accesses native code."),
    "pickle": ("unsafe_deserialization", "low", "Imports pickle (deserialization)."),
    "marshal": ("unsafe_deserialization", "low", "Imports marshal (deserialization)."),
}

# ---- text patterns for all files ----
_TEXT_PATTERNS = [
    (re.compile(r"(?:curl|wget)\s+[^\n|]*\|\s*(?:sudo\s+)?(?:sh|bash)\b"),
     "install_lifecycle", "high", "Downloads and pipes a script to a shell."),
    (re.compile(r"(?i)base64\s*\.?\s*(?:b64)?decode\s*\([^)]*\)\s*\)?\s*(?:,\s*)?"),
     "obfuscation", "medium", "Decodes base64 content (possible obfuscation)."),
    (re.compile(r"(?i)(?:powershell|pwsh)(?:\.exe)?\s+.*-enc(?:odedcommand)?\b"),
     "process_execution", "high", "Runs an encoded PowerShell command."),
    (re.compile(r"(?i)Invoke-Expression|\biex\b"),
     "process_execution", "medium", "Uses PowerShell Invoke-Expression."),
    (re.compile(r"(?i)privileged\s*:\s*true"),
     "container_privileged", "high", "Runs a container in privileged mode."),
    (re.compile(r"(?m)^\s*-\s*(?:/|\$\{?PWD).*:.*$|-v\s+/[^:]*:/"),
     "host_mount", "medium", "Mounts a host filesystem path into a container."),
]
_ENCODED_BLOB = re.compile(r"[A-Za-z0-9+/]{400,}={0,2}")
_SEVERITIES = ("critical", "high", "medium", "low", "informational")


def _call_name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = []
        node = func
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _scan_python(rel, text, add):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return  # unparseable; text patterns still apply separately
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            rule = _CALL_RULES.get(name)
            if name and rule is None and name.split(".")[-1] in ("system", "popen"):
                rule = ("process_execution", "high", "Runs a shell command.")
            if rule:
                cat, sev, reason = rule
                # subprocess with shell=True is notably higher risk.
                if cat == "process_execution" and _has_shell_true(node):
                    sev, reason = "high", "Starts a process via the shell (shell=True)."
                    cat = "process_execution_shell"
                line = node.lineno
                evidence = lines[line - 1].strip() if 0 < line <= len(lines) else name
                add(cat, sev, rel, line, evidence, "high", reason)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                _import_finding(alias.name, rel, node.lineno, add)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                _import_finding(node.module, rel, node.lineno, add)


def _has_shell_true(call):
    for kw in call.keywords:
        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
            return True
    return False


def _import_finding(module, rel, line, add):
    top = module.split(".")[0]
    rule = _IMPORT_RULES.get(top)
    if rule:
        cat, sev, reason = rule
        add(cat, sev, rel, line, f"import {module}", "medium", reason)


def _scan_text(rel, text, add):
    lines = text.splitlines()
    for pat, cat, sev, reason in _TEXT_PATTERNS:
        for m in pat.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            evidence = lines[line - 1].strip() if 0 < line <= len(lines) else m.group(0)
            add(cat, sev, rel, line, evidence, "medium", reason)
    for pat, sev in _CREDENTIAL_PATTERNS:
        for m in pat.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            evidence = lines[line - 1].strip() if 0 < line <= len(lines) else m.group(0)
            add("credentials", sev, rel, line, evidence, "medium",
                "Possible hardcoded credential/secret in the repository.")
    if _ENCODED_BLOB.search(text):
        add("obfuscation", "medium", rel, None,
            "Large encoded payload/blob detected", "low",
            "Large encoded content can hide code or data.")


def scan(repo_dir):
    """Static scan. Returns {risk_summary, findings, limitations, truncated, ...}."""
    max_files = config.repo_scan_max_files()
    max_file_bytes = config.repo_scan_max_file_bytes()
    max_total = config.repo_scan_max_total_bytes()
    max_depth = config.repo_scan_max_depth()
    max_findings = config.repo_scan_max_findings()

    findings = []
    truncated = False
    total_read = 0
    files_scanned = 0

    def add(category, severity, path, line, evidence, confidence, reason):
        nonlocal truncated
        if len(findings) >= max_findings:
            truncated = True
            return
        findings.append({
            "category": category,
            "severity": severity if severity in _SEVERITIES else "informational",
            "path": path,
            "line": line,
            "evidence": _redact(evidence),
            "confidence": confidence,
            "review_reason": reason,
        })

    # Request one beyond the cap so we can detect (and flag) truncation accurately.
    for rel, abs_path, is_link in collect_files(repo_dir, max_files + 1, max_depth):
        if is_link:
            continue
        if files_scanned >= max_files or total_read >= max_total:
            truncated = True
            break
        base = os.path.basename(rel)
        ext = os.path.splitext(base)[1].lower()
        text = read_text(abs_path, max_file_bytes)
        if text is None:
            continue
        total_read += len(text)
        files_scanned += 1
        if ext == ".py":
            _scan_python(rel, text, add)
        _scan_text(rel, text, add)
        if base == "package.json":
            _scan_package_json(rel, text, add)

    risk_summary = {sev: 0 for sev in _SEVERITIES}
    for f in findings:
        risk_summary[f["severity"]] = risk_summary.get(f["severity"], 0) + 1

    return {
        "risk_summary": risk_summary,
        "findings": findings,
        "files_scanned": files_scanned,
        "truncated": truncated,
        "limitations": [
            "Static scan only",
            "Pattern matches may be false positives",
            "Absence of findings does not establish safety",
            "No code was executed",
        ],
    }


def _scan_package_json(rel, text, add):
    import json
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return
    scripts = data.get("scripts") or {}
    for name in ("preinstall", "install", "postinstall", "prepare", "prepublish"):
        if name in scripts:
            add("install_lifecycle", "medium", rel, None,
                f"{name}: {scripts[name]}", "high",
                "npm lifecycle scripts run automatically on install.")
