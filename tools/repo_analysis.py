"""Static, read-only analyzers for a cloned repository.

Everything here reads files as data only. It NEVER imports, executes, or evaluates
repository code or manifests (pyproject/setup.py/package.json are parsed or read as
text; scripts are reported, never run). Output separates observed facts from
inference. Bounded to keep token/context cost predictable.
"""

import json
import os
import tomllib

import tools.config as config
from tools import repo_store

_LANG_BY_EXT = {
    ".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".jsx": "JavaScript", ".go": "Go",
    ".rs": "Rust", ".java": "Java", ".kt": "Kotlin", ".rb": "Ruby", ".php": "PHP",
    ".c": "C", ".h": "C", ".cpp": "C++", ".cc": "C++", ".hpp": "C++", ".cs": "C#",
    ".swift": "Swift", ".sh": "Shell", ".ps1": "PowerShell", ".r": "R", ".scala": "Scala",
    ".md": "Markdown", ".rst": "reStructuredText", ".toml": "TOML", ".yaml": "YAML",
    ".yml": "YAML", ".json": "JSON", ".html": "HTML", ".css": "CSS",
}
_CODE_LANGS = {"Python", "JavaScript", "TypeScript", "Go", "Rust", "Java", "Kotlin",
               "Ruby", "PHP", "C", "C++", "C#", "Swift", "Shell", "PowerShell", "Scala"}

_PY_MANIFESTS = {"pyproject.toml", "requirements.txt", "setup.py", "setup.cfg",
                 "Pipfile", "poetry.lock", "uv.lock"}
_JS_MANIFESTS = {"package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
                 "bun.lock", "tsconfig.json"}
_GO_MANIFESTS = {"go.mod", "go.sum"}
_RUST_MANIFESTS = {"Cargo.toml", "Cargo.lock"}
_JAVA_MANIFESTS = {"pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle"}
_CONTAINER_FILES = {"Dockerfile", "docker-compose.yml", "docker-compose.yaml",
                    "compose.yml", "compose.yaml"}
_INTEGRATION_MANIFESTS = {"mcp.json", "server.json", "plugin.yaml", "plugin.yml",
                          "plugin.json", "manifest.json", "openapi.json", "openapi.yaml",
                          "swagger.json", "swagger.yaml"}
_LIFECYCLE_SCRIPTS = {"preinstall", "install", "postinstall", "prepare", "prepublish"}
_BINARY_EXTS = {".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".class", ".jar",
                ".wasm", ".pyc", ".zip", ".tar", ".gz", ".7z", ".png", ".jpg", ".jpeg",
                ".gif", ".pdf", ".mp4", ".mp3", ".woff", ".woff2", ".ttf"}
_VENDORED_DIRS = {"node_modules", "vendor", "dist", "build", "target", ".venv", "venv",
                  "__pycache__", "site-packages", "bower_components"}


def read_text(abs_path, max_bytes):
    """Read a bounded text file as str, or None if binary/oversized/unreadable."""
    try:
        if os.path.getsize(abs_path) > max_bytes:
            return None
        with open(abs_path, "rb") as f:
            raw = f.read(max_bytes)
    except OSError:
        return None
    if b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("latin-1")
        except UnicodeDecodeError:
            return None


def collect_files(repo_dir, max_files, max_depth):
    """Yield (rel_path, abs_path, is_symlink) under repo_dir, skipping .git, bounded,
    never following symlinked directories."""
    root_depth = repo_dir.rstrip(os.sep).count(os.sep)
    count = 0
    for dirpath, dirnames, filenames in os.walk(repo_dir, followlinks=False):
        depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
        if depth >= max_depth:
            dirnames[:] = []
        # Do not descend into .git or symlinked directories.
        dirnames[:] = [d for d in sorted(dirnames)
                       if d != ".git" and not repo_store.is_symlink(os.path.join(dirpath, d))]
        for name in sorted(filenames):
            abs_path = os.path.join(dirpath, name)
            rel = os.path.relpath(abs_path, repo_dir).replace(os.sep, "/")
            yield rel, abs_path, repo_store.is_symlink(abs_path)
            count += 1
            if count >= max_files:
                return


def _pkg_managers(names, pyproject_data, has_poetry):
    managers = set()
    if names & {"requirements.txt", "setup.py", "setup.cfg"} or "pyproject.toml" in names:
        managers.add("pip")
    if "poetry.lock" in names or has_poetry:
        managers.add("poetry")
    if "uv.lock" in names:
        managers.add("uv")
    if "Pipfile" in names:
        managers.add("pipenv")
    if "package-lock.json" in names or "package.json" in names:
        managers.add("npm")
    if "pnpm-lock.yaml" in names:
        managers.add("pnpm")
    if "yarn.lock" in names:
        managers.add("yarn")
    if "bun.lock" in names:
        managers.add("bun")
    if names & _GO_MANIFESTS:
        managers.add("go modules")
    if names & _RUST_MANIFESTS:
        managers.add("cargo")
    if "pom.xml" in names:
        managers.add("maven")
    if names & {"build.gradle", "build.gradle.kts", "settings.gradle"}:
        managers.add("gradle")
    return sorted(managers)


def parse_dependencies(repo_dir, names, abs_by_name):
    """Parse dependency manifests defensively (never executed). Returns a dict."""
    deps = {"python": [], "node": [], "go": [], "rust": [], "lifecycle_scripts": [],
            "node_scripts_present": False, "parse_errors": []}
    max_bytes = config.repo_scan_max_file_bytes()

    if "pyproject.toml" in names:
        text = read_text(abs_by_name["pyproject.toml"], max_bytes)
        if text is not None:
            try:
                data = tomllib.loads(text)
                proj = data.get("project", {})
                deps["python"].extend(_names_only(proj.get("dependencies", [])))
                poetry = data.get("tool", {}).get("poetry", {})
                deps["python"].extend(list(poetry.get("dependencies", {}).keys()))
            except (tomllib.TOMLDecodeError, AttributeError, TypeError):
                deps["parse_errors"].append("pyproject.toml")
    for rname in [n for n in names if n == "requirements.txt" or n.startswith("requirements")]:
        text = read_text(abs_by_name.get(rname, ""), max_bytes)
        if text:
            for line in text.splitlines():
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    deps["python"].append(line.split("==")[0].split(">=")[0].split("[")[0].strip())

    if "package.json" in names:
        text = read_text(abs_by_name["package.json"], max_bytes)
        if text is not None:
            try:
                data = json.loads(text)
                deps["node"].extend(list((data.get("dependencies") or {}).keys()))
                deps["node"].extend(list((data.get("devDependencies") or {}).keys()))
                scripts = data.get("scripts") or {}
                deps["node_scripts_present"] = bool(scripts)
                deps["lifecycle_scripts"] = sorted(set(scripts) & _LIFECYCLE_SCRIPTS)
            except (json.JSONDecodeError, AttributeError, TypeError):
                deps["parse_errors"].append("package.json")

    if "go.mod" in names:
        text = read_text(abs_by_name["go.mod"], max_bytes)
        if text:
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("require ") and " " in line[8:]:
                    deps["go"].append(line[8:].split()[0])

    if "Cargo.toml" in names:
        text = read_text(abs_by_name["Cargo.toml"], max_bytes)
        if text is not None:
            try:
                data = tomllib.loads(text)
                deps["rust"].extend(list((data.get("dependencies") or {}).keys()))
            except (tomllib.TOMLDecodeError, AttributeError, TypeError):
                deps["parse_errors"].append("Cargo.toml")

    for key in ("python", "node", "go", "rust"):
        deps[key] = sorted({d for d in deps[key] if d})[:100]
    return deps


def _names_only(items):
    out = []
    for item in items or []:
        if isinstance(item, str):
            out.append(item.split("==")[0].split(">=")[0].split("[")[0].split(";")[0].strip())
    return out


def analyze(repo_dir):
    """Produce a bounded static structural summary (facts + inference + limitations)."""
    max_files = config.repo_scan_max_files()
    max_depth = config.repo_scan_max_depth()

    ext_counts = {}
    lang_counts = {}
    names = set()
    abs_by_name = {}
    documentation = []
    tests = []
    license_name = None
    ci_files = []
    has_binaries = False
    vendored = []
    entry_evidence = []       # (rel, evidence)
    text_cache = {}

    size_bytes, total_files = repo_store.measure_repository(repo_dir)

    for rel, abs_path, is_link in collect_files(repo_dir, max_files, max_depth):
        if is_link:
            continue
        base = os.path.basename(rel)
        ext = os.path.splitext(base)[1].lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        lang = _LANG_BY_EXT.get(ext)
        if lang in _CODE_LANGS or lang in ("Markdown", "reStructuredText"):
            if lang in _CODE_LANGS:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
        if ext in _BINARY_EXTS:
            has_binaries = True
        names.add(base)
        abs_by_name.setdefault(base, abs_path)

        low = rel.lower()
        if base.lower() in ("readme.md", "readme.rst", "readme.txt", "readme") or low.startswith("docs/"):
            documentation.append(rel)
        if low.startswith("test") or "/test" in low or low.startswith("tests/"):
            tests.append(rel)
        if base.lower().startswith("license") or base.lower().startswith("copying"):
            license_name = license_name or base
        if low.startswith(".github/workflows/") or base in (".gitlab-ci.yml", "azure-pipelines.yml"):
            ci_files.append(rel)
        top = rel.split("/")[0]
        if top in _VENDORED_DIRS and top not in vendored:
            vendored.append(top)

        # Entry-point evidence: __main__ blocks (Python) — text scan only.
        if ext == ".py" and base not in ("__init__.py",):
            text = read_text(abs_path, config.repo_scan_max_file_bytes())
            if text is not None:
                text_cache[rel] = text
                if "__main__" in text and "if __name__" in text:
                    entry_evidence.append((rel, "Contains a __main__ block"))

    manifests = sorted(names & (_PY_MANIFESTS | _JS_MANIFESTS | _GO_MANIFESTS |
                                _RUST_MANIFESTS | _JAVA_MANIFESTS | _INTEGRATION_MANIFESTS))
    container_files = sorted(names & _CONTAINER_FILES)
    has_pyproject_poetry = False
    if "pyproject.toml" in names:
        pt = read_text(abs_by_name["pyproject.toml"], config.repo_scan_max_file_bytes())
        has_pyproject_poetry = bool(pt and "[tool.poetry]" in pt)
    package_managers = _pkg_managers(names, None, has_pyproject_poetry)
    deps = parse_dependencies(repo_dir, names, abs_by_name)

    languages = [{"name": n, "file_count": c}
                 for n, c in sorted(lang_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    possible_entrypoints = [{"path": rel, "type": "possible_service_or_cli",
                             "confidence": "medium", "evidence": ev}
                            for rel, ev in entry_evidence[:20]]
    integration = _integration_indicators(names, deps, documentation, repo_dir, abs_by_name,
                                          container_files, possible_entrypoints)

    return {
        "observed": {
            "file_count": total_files,
            "size_bytes": size_bytes,
            "languages": languages,
            "extension_distribution": dict(sorted(ext_counts.items(), key=lambda kv: -kv[1])[:20]),
            "manifests": manifests,
            "package_managers": package_managers,
            "documentation": sorted(set(documentation))[:20],
            "tests": sorted(set(_top_dirs(tests)))[:20],
            "license": license_name,
            "container_files": container_files,
            "ci_files": sorted(ci_files)[:20],
            "has_submodules": ".gitmodules" in names,
            "has_lfs_pointers": _has_lfs(repo_dir, abs_by_name),
            "has_binaries": has_binaries,
            "vendored_directories": vendored,
            "dependencies": deps,
        },
        "inferred": {
            "possible_entrypoints": possible_entrypoints,
            "integration_indicators": integration,
        },
        "limitations": [
            "Static inspection only",
            "No code was executed",
            "No dependencies were installed",
            "Inferred entry points and integration types are not confirmed",
        ],
    }


def _top_dirs(paths):
    out = set()
    for p in paths:
        out.add(p.split("/")[0] + "/" if "/" in p else p)
    return out


def _has_lfs(repo_dir, abs_by_name):
    if ".gitattributes" not in abs_by_name:
        return False
    text = read_text(abs_by_name[".gitattributes"], 200_000)
    return bool(text and "filter=lfs" in text)


def _integration_indicators(names, deps, documentation, repo_dir, abs_by_name,
                            container_files, entrypoints):
    indicators = []
    all_deps = set(deps["python"]) | set(deps["node"]) | set(deps["rust"])
    readme_text = ""
    for doc in documentation:
        if os.path.basename(doc).lower().startswith("readme"):
            readme_text = (read_text(os.path.join(repo_dir, doc), 200_000) or "").lower()
            break

    mcp_ev = []
    if any("mcp" in d.lower() or "modelcontextprotocol" in d.lower() for d in all_deps):
        mcp_ev.append("Dependency references MCP")
    if names & {"mcp.json", "server.json"}:
        mcp_ev.append("MCP/server manifest present")
    if "mcp server" in readme_text or "model context protocol" in readme_text:
        mcp_ev.append("README mentions an MCP server")
    if mcp_ev:
        indicators.append({"type": "possible_mcp_server",
                           "confidence": "high" if len(mcp_ev) >= 2 else "medium", "evidence": mcp_ev})

    rest_ev = []
    if all_deps & {"fastapi", "flask", "uvicorn", "starlette", "express", "aiohttp", "django"}:
        rest_ev.append("Web framework dependency present")
    if names & {"openapi.json", "openapi.yaml", "swagger.json", "swagger.yaml"}:
        rest_ev.append("OpenAPI/Swagger spec present")
    if rest_ev:
        indicators.append({"type": "possible_rest_service",
                           "confidence": "medium", "evidence": rest_ev})

    if container_files:
        indicators.append({"type": "possible_docker_service", "confidence": "medium",
                           "evidence": [f"Container file present: {', '.join(container_files)}"]})

    cli_ev = []
    if all_deps & {"click", "typer", "argparse"}:
        cli_ev.append("CLI framework dependency present")
    if "package.json" in names and deps.get("node_scripts_present"):
        cli_ev.append("package.json defines scripts")
    if cli_ev:
        indicators.append({"type": "possible_cli", "confidence": "medium", "evidence": cli_ev})

    if not indicators and (deps["python"] or deps["node"] or deps["rust"]):
        indicators.append({"type": "possible_library", "confidence": "low",
                           "evidence": ["Has a dependency manifest but no clear service/CLI entry point"]})
    return indicators
