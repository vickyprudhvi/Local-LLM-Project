"""repo.inspect + dependency parsing (static, no execution)."""

import os

import pytest

from tools.repo_tools import InspectTool


def _write(root, rel, content):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOSITORY_ROOT", str(tmp_path / "repos"))
    d = tmp_path / "repos" / "acme" / "proj"
    _write(d, "README.md", "# Proj\nAn MCP server for finance.\n")
    _write(d, "pyproject.toml",
           "[project]\nname='proj'\nrequires-python='>=3.11'\ndependencies=['mcp>=1.0','fastapi','click']\n")
    _write(d, "requirements.txt", "requests==2.0\nhttpx\n# comment\n")
    _write(d, "src/server.py", "def main():\n    pass\n\nif __name__ == '__main__':\n    main()\n")
    _write(d, "tests/test_x.py", "def test_x():\n    assert True\n")
    _write(d, "Dockerfile", "FROM python:3.11\n")
    _write(d, ".github/workflows/ci.yml", "on: push\n")
    _write(d, "LICENSE", "MIT License\n")
    _write(d, "docs/guide.md", "docs\n")
    return d


def _inspect(repo="acme/proj"):
    t = InspectTool()
    return t.execute(t.validate_arguments({"repository": repo}))


def test_language_detection(repo):
    data = _inspect()
    langs = {l["name"] for l in data["summary"]["languages"]}
    assert "Python" in langs


def test_manifest_and_package_manager_detection(repo):
    data = _inspect()
    assert "pyproject.toml" in data["summary"]["manifests"]
    assert "pip" in data["summary"]["package_managers"]


def test_readme_license_docs_tests(repo):
    s = _inspect()["summary"]
    assert any("README" in d or "readme" in d.lower() for d in s["documentation"])
    assert s["license"] and "LICENSE" in s["license"]
    assert any("test" in t.lower() for t in s["tests"])
    assert any("docs" in d.lower() for d in s["documentation"])


def test_docker_and_ci_detection(repo):
    s = _inspect()["summary"]
    assert "Dockerfile" in s["container_files"]
    assert any("workflows" in c for c in s["ci_files"])


def test_entry_point_detection(repo):
    entries = _inspect()["inferred"]["possible_entrypoints"]
    assert any(e["path"] == "src/server.py" for e in entries)


def test_mcp_indicator_detection(repo):
    indicators = _inspect()["inferred"]["integration_indicators"]
    types = {i["type"] for i in indicators}
    assert "possible_mcp_server" in types


def test_rest_indicator_detection(repo):
    indicators = _inspect()["inferred"]["integration_indicators"]
    assert any(i["type"] == "possible_rest_service" for i in indicators)


def test_facts_and_inference_separated(repo):
    data = _inspect()
    assert "summary" in data and "inferred" in data
    assert "limitations" in data
    assert data["executed"] is False


def test_dependency_parsing(repo):
    deps = _inspect()["summary"]["dependencies"]
    assert "mcp" in deps["python"] or any(d.startswith("mcp") for d in deps["python"])
    assert "requests" in deps["python"]


def test_setup_py_treated_as_text(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOSITORY_ROOT", str(tmp_path / "repos"))
    d = tmp_path / "repos" / "a" / "b"
    # A setup.py that would raise if executed — must be inspected as text only.
    _write(d, "setup.py", "raise SystemExit('should never run')\n")
    _write(d, "README.md", "hi")
    data = InspectTool().execute({"repository": "a/b"})  # must not raise
    assert data["executed"] is False


def test_package_json_lifecycle_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOSITORY_ROOT", str(tmp_path / "repos"))
    d = tmp_path / "repos" / "a" / "b"
    _write(d, "package.json",
           '{"name":"x","dependencies":{"express":"^4"},"scripts":{"postinstall":"node evil.js"}}')
    data = InspectTool().execute({"repository": "a/b"})
    deps = data["summary"]["dependencies"]
    assert "express" in deps["node"]
    assert "postinstall" in deps["lifecycle_scripts"]


def test_malformed_manifest_handled(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOSITORY_ROOT", str(tmp_path / "repos"))
    d = tmp_path / "repos" / "a" / "b"
    _write(d, "package.json", "{ this is not json ")
    _write(d, "pyproject.toml", "this = = = broken")
    data = InspectTool().execute({"repository": "a/b"})  # must not raise
    assert "package.json" in data["summary"]["dependencies"]["parse_errors"]


def test_go_and_cargo_detection(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOSITORY_ROOT", str(tmp_path / "repos"))
    d = tmp_path / "repos" / "a" / "b"
    _write(d, "go.mod", "module x\nrequire github.com/foo/bar v1.0.0\n")
    _write(d, "Cargo.toml", "[dependencies]\nserde='1.0'\n")
    data = InspectTool().execute({"repository": "a/b"})
    assert "go modules" in data["summary"]["package_managers"]
    assert "cargo" in data["summary"]["package_managers"]
