"""Phase G.2 Task 11/16(K/L) — per-server locking and idempotence.

Real threads calling `ensure_started` concurrently: for the SAME server_id only
one bootstrap must happen and every caller gets the same healthy session; for
DIFFERENT server_ids, starts proceed independently with no identity/tool
collision.
"""

import os
import threading

import pytest

from mcp_layer.runtime_manager import MultiMcpRuntimeManager
from tests.mcp_multi_runtime_helpers import manager_paths, node_available, write_fixture_server_config
from tools.registry import ToolRegistry

pytestmark = pytest.mark.skipif(not node_available(), reason="node/npm not available")


@pytest.fixture
def env(tmp_path):
    paths = manager_paths(tmp_path)
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()
    return paths, str(root_a), str(root_b)


def test_concurrent_ensure_started_for_the_same_server_bootstraps_once(env):
    paths, root_a, _root_b = env
    write_fixture_server_config(paths, "filesystem-test", [root_a])
    reg = ToolRegistry()
    rm = MultiMcpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])

    results = [None] * 8
    errors = []

    def _call(i):
        try:
            results[i] = rm.ensure_started("filesystem-test", expected_allowed_roots=(os.path.realpath(root_a),))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=_call, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors
    sessions = {id(s) for s in results if s is not None}
    assert len(sessions) == 1  # exactly one bootstrap; every caller got the SAME session
    assert results[0].health.state.value == "healthy"
    results[0].shutdown()


def test_concurrent_ensure_started_for_different_servers_is_independent(env):
    paths, root_a, root_b = env
    write_fixture_server_config(paths, "filesystem-test", [root_a])
    write_fixture_server_config(paths, "document-test", [root_b])
    reg = ToolRegistry()
    rm = MultiMcpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])

    outcome = {}
    errors = []

    def _start(server_id, root):
        try:
            outcome[server_id] = rm.ensure_started(server_id, expected_allowed_roots=(os.path.realpath(root),))
        except Exception as e:  # noqa: BLE001
            errors.append((server_id, e))

    t1 = threading.Thread(target=_start, args=("filesystem-test", root_a))
    t2 = threading.Thread(target=_start, args=("document-test", root_b))
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert not errors
    assert outcome["filesystem-test"] is not outcome["document-test"]
    assert outcome["filesystem-test"].session_id != outcome["document-test"].session_id
    assert reg.has("mcp.filesystem-test.read_text_file")
    assert reg.has("mcp.document-test.read_text_file")
    # No cross-registration: each tool is owned by its OWN session only.
    fs_tool = reg.get("mcp.filesystem-test.read_text_file")
    doc_tool = reg.get("mcp.document-test.read_text_file")
    assert fs_tool.session_owner == outcome["filesystem-test"].session_id
    assert doc_tool.session_owner == outcome["document-test"].session_id

    outcome["filesystem-test"].shutdown()
    outcome["document-test"].shutdown()


def test_repeated_ensure_started_is_idempotent(env):
    paths, root_a, _root_b = env
    write_fixture_server_config(paths, "filesystem-test", [root_a])
    reg = ToolRegistry()
    rm = MultiMcpRuntimeManager(reg, base_dir=paths["base_dir"], managed_root=paths["managed_root"])

    sessions = [rm.ensure_started("filesystem-test", expected_allowed_roots=(os.path.realpath(root_a),))
               for _ in range(5)]
    assert len({id(s) for s in sessions}) == 1
    sessions[0].shutdown()
