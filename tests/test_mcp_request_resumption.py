"""Phase F — the original request survives provisioning and resumes via the NORMAL pipeline."""

import os

import pytest

from mcp_layer.errors import McpError
from mcp_management.models import PendingRequestState, ProvisioningApproval
from tests.mcp_provisioning_helpers import (
    FakeNpm,
    make_manager,
    node_available,
    workspace_with_file,
)
from tools.models import (
    MCP_PROVISIONING_CONFIRMATION_REQUIRED,
    MCP_PROVISIONING_DECLINED,
    MCP_PROVISIONING_LOOP_PREVENTED,
)

pytestmark = pytest.mark.skipif(not node_available(), reason="node/npm not available")


@pytest.fixture
def ctx(tmp_path):
    manager, paths = make_manager(tmp_path)
    workspace = workspace_with_file(tmp_path)
    return {"manager": manager, "paths": paths, "workspace": workspace,
            "text": f"Read hello.txt from {workspace}"}


def _approve(plan):
    return ProvisioningApproval(True, plan.plan_id, plan.compute_hash())


def test_request_is_retained_through_provisioning(ctx):
    manager = ctx["manager"]
    detection, request = manager.begin_request(ctx["text"])
    assert detection.recommended_catalog_id == "official-filesystem"
    assert request.original_user_text == ctx["text"]
    assert request.state is PendingRequestState.CAPABILITY_DETECTED

    plan = manager.prepare_plan("official-filesystem", [ctx["workspace"]],
                                request_id=request.request_id)
    assert manager.pending(request.request_id).state is PendingRequestState.AWAITING_APPROVAL

    manager.provision(plan, approval=_approve(plan), request_id=request.request_id,
                      npm_runner=FakeNpm())
    assert manager.pending(request.request_id).state is PendingRequestState.READY

    # Resumption hands the ORIGINAL text back for the normal pipeline to re-run.
    assert manager.resume(request.request_id) == ctx["text"]
    assert manager.pending(request.request_id).state is PendingRequestState.RESUMED


def test_resume_returns_none_before_ready(ctx):
    manager = ctx["manager"]
    _, request = manager.begin_request(ctx["text"])
    assert manager.resume(request.request_id) is None


def test_failed_provisioning_does_not_resume(ctx):
    manager = ctx["manager"]
    _, request = manager.begin_request(ctx["text"])
    plan = manager.prepare_plan("official-filesystem", [ctx["workspace"]],
                                request_id=request.request_id)
    with pytest.raises(McpError):
        manager.provision(plan, approval=_approve(plan), request_id=request.request_id,
                          npm_runner=FakeNpm(fail=True))
    assert manager.pending(request.request_id).state is PendingRequestState.FAILED
    assert manager.resume(request.request_id) is None


def test_declined_provisioning_marks_declined_and_installs_nothing(ctx):
    manager = ctx["manager"]
    _, request = manager.begin_request(ctx["text"])
    plan = manager.prepare_plan("official-filesystem", [ctx["workspace"]],
                                request_id=request.request_id)
    with pytest.raises(McpError) as e:
        manager.provision(plan, request_id=request.request_id,
                          confirmer=lambda p: False, npm_runner=FakeNpm())
    assert e.value.code == MCP_PROVISIONING_DECLINED
    assert manager.pending(request.request_id).state is PendingRequestState.DECLINED
    assert not os.path.isdir(str(plan.install_directory))
    assert manager.resume(request.request_id) is None


def test_no_approval_and_no_confirmer_installs_nothing(ctx):
    manager = ctx["manager"]
    _, request = manager.begin_request(ctx["text"])
    plan = manager.prepare_plan("official-filesystem", [ctx["workspace"]],
                                request_id=request.request_id)
    with pytest.raises(McpError) as e:
        manager.provision(plan, request_id=request.request_id, npm_runner=FakeNpm())
    assert e.value.code == MCP_PROVISIONING_CONFIRMATION_REQUIRED
    assert not os.path.isdir(str(plan.install_directory))


def test_only_one_provisioning_attempt_per_request(ctx):
    manager = ctx["manager"]
    _, request = manager.begin_request(ctx["text"])
    plan = manager.prepare_plan("official-filesystem", [ctx["workspace"]],
                                request_id=request.request_id)
    manager.provision(plan, approval=_approve(plan), request_id=request.request_id,
                      npm_runner=FakeNpm())
    # A second attempt for the SAME original request is refused (no loop).
    with pytest.raises(McpError) as e:
        manager.provision(plan, approval=_approve(plan), request_id=request.request_id,
                          npm_runner=FakeNpm())
    assert e.value.code == MCP_PROVISIONING_LOOP_PREVENTED


def test_already_installed_server_needs_no_new_request(ctx):
    manager = ctx["manager"]
    _, request = manager.begin_request(ctx["text"])
    plan = manager.prepare_plan("official-filesystem", [ctx["workspace"]],
                                request_id=request.request_id)
    manager.provision(plan, approval=_approve(plan), request_id=request.request_id,
                      npm_runner=FakeNpm())

    # The same capability request now resolves without opening a pending request.
    detection, second = manager.begin_request(ctx["text"])
    assert detection.requires_mcp is True
    assert second is None


def test_provisioning_activates_a_config_the_phase_e_loader_accepts(ctx):
    """Activation writes the MANAGED config and the resolver selects it."""
    from mcp_layer.config import load_config
    from mcp_layer.config_resolver import McpConfigSource, resolve_config

    from tests.mcp_provisioning_helpers import managed_config_file, write_template

    manager, paths = ctx["manager"], ctx["paths"]
    write_template(paths, enabled=False)
    _, request = manager.begin_request(ctx["text"])
    plan = manager.prepare_plan("official-filesystem", [ctx["workspace"]],
                                request_id=request.request_id)
    manager.provision(plan, approval=_approve(plan), request_id=request.request_id,
                      npm_runner=FakeNpm())

    config = load_config(managed_config_file(paths))
    assert config is not None and config.enabled is True
    assert config.server_id == "filesystem"
    assert config.tool_policy.tools["write_file"].permission.value == "write"

    resolved = resolve_config(base_dir=paths["base_dir"], managed_root=paths["managed_root"],
                              override="", template_path=paths["template_path"])
    assert resolved.source is McpConfigSource.MANAGED_ACTIVE
    assert str(resolved.path) == os.path.realpath(managed_config_file(paths))


def test_provisioning_never_writes_the_committed_template(ctx):
    from tests.mcp_provisioning_helpers import write_template

    manager, paths = ctx["manager"], ctx["paths"]
    template = write_template(paths, enabled=False)
    with open(template, "rb") as f:
        before = f.read()

    _, request = manager.begin_request(ctx["text"])
    plan = manager.prepare_plan("official-filesystem", [ctx["workspace"]],
                                request_id=request.request_id)
    manager.provision(plan, approval=_approve(plan), request_id=request.request_id,
                      npm_runner=FakeNpm())

    with open(template, "rb") as f:
        assert f.read() == before, "Phase F must never write config/mcp_server.json"


def test_installer_never_executes_the_new_tool_itself(ctx):
    """Provisioning returns; it does not call the newly discovered MCP tool."""
    manager = ctx["manager"]
    _, request = manager.begin_request(ctx["text"])
    plan = manager.prepare_plan("official-filesystem", [ctx["workspace"]],
                                request_id=request.request_id)
    result = manager.provision(plan, approval=_approve(plan), request_id=request.request_id,
                               npm_runner=FakeNpm())
    # Only validation ran (a read-only listing); the user's actual request is not
    # answered here — it must go back through routing/shortlist/executor.
    assert "note" not in result or "original request" in str(result.get("note", ""))
    assert manager.pending(request.request_id).state is PendingRequestState.READY
    assert os.listdir(ctx["workspace"]) == ["hello.txt"]
