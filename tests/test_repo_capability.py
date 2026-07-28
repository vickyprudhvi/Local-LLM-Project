"""repo.capability_report: integration inference, facts vs inference, no install approval."""

import json

import pytest

from tools.repo_tools import CapabilityReportTool


def _write(root, rel, content):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOSITORY_ROOT", str(tmp_path / "repos"))
    return tmp_path / "repos"


def _report(root, files, repo="a/b"):
    d = root / "a" / "b"
    for rel, content in files.items():
        _write(d, rel, content)
    t = CapabilityReportTool()
    return t.execute(t.validate_arguments({"repository": repo}))


def test_mcp_server_indicators(root):
    r = _report(root, {
        "README.md": "# MCP server for finance\n",
        "pyproject.toml": "[project]\nname='x'\ndependencies=['mcp>=1.0']\nrequires-python='>=3.11'\n",
        "server.py": "if __name__ == '__main__':\n    pass\n",
    })
    types = {i["type"] for i in r["possible_integrations"]}
    assert "possible_mcp_server" in types


def test_python_library_and_runtime(root):
    r = _report(root, {"pyproject.toml": "[project]\nname='x'\nrequires-python='>=3.11'\ndependencies=['attrs']\n",
                       "lib.py": "x = 1\n"})
    assert "Python" in r["requirements"]["runtime"]


def test_rest_service_indicator(root):
    r = _report(root, {"pyproject.toml": "[project]\nname='x'\ndependencies=['fastapi','uvicorn']\n",
                       "app.py": "x=1\n"})
    assert any(i["type"] == "possible_rest_service" for i in r["possible_integrations"])
    assert r["requirements"]["network"] is True


def test_docker_service_indicator(root):
    r = _report(root, {"Dockerfile": "FROM python\n", "README.md": "hi"})
    assert any(i["type"] == "possible_docker_service" for i in r["possible_integrations"])


def test_unknown_integration(root):
    r = _report(root, {"notes.txt": "just some text\n"})
    assert r["recommendation"]["status"] in (
        "insufficient_information", "static_review_complete")


def test_api_key_inference(root):
    r = _report(root, {"README.md": "hi", ".env.example": "API_KEY=\n",
                       "pyproject.toml": "[project]\nname='x'\ndependencies=['requests']\n"})
    assert r["requirements"]["api_keys"]  # non-empty signal


def test_security_summary_reused(root):
    r = _report(root, {"runner.py": "import subprocess\nsubprocess.run('x', shell=True)\n",
                       "pyproject.toml": "[project]\nname='x'\ndependencies=['click']\n"})
    assert r["security_summary"]["requires_review"] is True
    assert r["recommendation"]["status"] == "manual_review_required"
    assert r["requirements"]["subprocess"] is True


def test_facts_vs_inference_separated(root):
    r = _report(root, {"README.md": "hi", "pyproject.toml": "[project]\nname='x'\ndependencies=['click']\n"})
    assert "observed" in r and "possible_integrations" in r
    assert r["executed"] is False


def test_no_safe_to_install_recommendation(root):
    r = _report(root, {"README.md": "hi", "pyproject.toml": "[project]\nname='x'\n"})
    status = r["recommendation"]["status"]
    assert status in ("insufficient_information", "static_review_complete",
                      "manual_review_required", "not_supported_by_current_architecture")
    assert status not in ("safe_to_install", "trusted", "approved", "malware_free")


def test_result_serializable(root):
    r = _report(root, {"README.md": "hi", "pyproject.toml": "[project]\nname='x'\n"})
    r.pop("_log_meta", None)
    json.dumps(r)
