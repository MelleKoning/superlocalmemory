"""A workspace that requires a login refuses command-line writes too.

Two settings mean "this workspace requires a login": ``deployment.mode`` in
config.toml, and ``require_login`` in the workspace's own role settings, which
the dashboard toggle writes. Both must reach every transport. When only the
first did, turning the toggle on refused writes over HTTP and admitted the same
write from the command line, as the machine owner.

The second half of this file covers which workspace the role is read from. A
user can hold different roles in different workspaces; the one that decides a
write is the role held on the workspace the write lands in.
"""

from __future__ import annotations

import json
import pathlib

import pytest


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """A store with two users, two workspaces, and a login requirement."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("SLM_USER_SESSION", raising=False)

    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.infra.data_root import state_path
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.migration_runner import apply_all

    config = SLMConfig.load()
    db = DatabaseManager(config.db_path)
    db.initialize(schema)
    db.close()
    apply_all(pathlib.Path(state_path("learning.db")), pathlib.Path(config.db_path))

    from superlocalmemory.access.rbac import RbacEngine

    rbac = RbacEngine(str(config.db_path))
    boss = rbac.create_user("boss", "correct-horse-battery", "Boss")
    watcher = rbac.create_user("watcher", "staple-battery-horse", "Watcher")
    rbac.set_membership("default", boss["user_id"], "admin")
    rbac.set_membership("team", boss["user_id"], "admin")
    rbac.set_membership("default", watcher["user_id"], "admin")
    rbac.set_membership("team", watcher["user_id"], "viewer")
    rbac.set_require_login(True)

    # The command line writes into whichever workspace is active.
    (tmp_path / "profiles.json").write_text(json.dumps({"active_profile": "team"}), encoding="utf-8")

    return {"rbac": rbac, "boss": boss, "watcher": watcher, "path": tmp_path}


def _gate():
    from superlocalmemory.core.admission import gate_cli_mutation
    from superlocalmemory.core.operation_request import OperationKind

    return gate_cli_mutation, OperationKind


def test_the_role_settings_alone_put_the_workspace_in_company_mode(workspace):
    """config.toml says personal; the role settings say a login is required."""
    from superlocalmemory.core.admission import _company_mode_active, _resolve_deployment

    deployment = _resolve_deployment()
    assert deployment.is_enterprise is False
    assert workspace["rbac"].require_login() is True
    assert _company_mode_active(deployment) is True


def test_an_unauthenticated_command_line_write_is_refused(workspace):
    """No session token, so no principal, so no write."""
    gate_cli_mutation, OperationKind = _gate()
    with pytest.raises(SystemExit) as exit_info:
        gate_cli_mutation(OperationKind.REMEMBER)
    assert exit_info.value.code == 1


def test_the_role_is_read_from_the_workspace_the_write_lands_in(workspace, monkeypatch):
    """Admin in one workspace, viewer in another. The active one decides."""
    from superlocalmemory.core.admission import _session_principal, _target_profile

    token = workspace["rbac"].create_session(workspace["watcher"]["user_id"])
    monkeypatch.setenv("SLM_USER_SESSION", token)

    assert _target_profile("") == "team"

    _, _, on_target = _session_principal(_target_profile(""))
    _, _, on_default = _session_principal("default")
    assert [role.value for role in on_target] == ["viewer"]
    assert [role.value for role in on_default] == ["admin"]
    assert on_target != on_default, (
        "the two workspaces must not resolve to the same role, or this test "
        "cannot tell which one the gate consulted"
    )


def test_a_viewer_in_the_active_workspace_cannot_write_from_the_command_line(
    workspace, monkeypatch,
):
    """The user is an admin somewhere else. That does not help them here."""
    gate_cli_mutation, OperationKind = _gate()
    token = workspace["rbac"].create_session(workspace["watcher"]["user_id"])
    monkeypatch.setenv("SLM_USER_SESSION", token)
    with pytest.raises(SystemExit) as exit_info:
        gate_cli_mutation(OperationKind.REMEMBER)
    assert exit_info.value.code == 1


def test_an_admin_in_the_active_workspace_is_admitted(workspace, monkeypatch):
    """The control. A login requirement must not lock out the people who have one."""
    gate_cli_mutation, OperationKind = _gate()
    token = workspace["rbac"].create_session(workspace["boss"]["user_id"])
    monkeypatch.setenv("SLM_USER_SESSION", token)
    gate_cli_mutation(OperationKind.REMEMBER)  # must not raise


def test_switching_workspace_is_decided_by_the_role_in_the_destination(
    workspace, monkeypatch,
):
    """`profile switch team` is decided by the role held on ``team``."""
    gate_cli_mutation, OperationKind = _gate()
    token = workspace["rbac"].create_session(workspace["watcher"]["user_id"])
    monkeypatch.setenv("SLM_USER_SESSION", token)
    with pytest.raises(SystemExit):
        gate_cli_mutation(OperationKind.PROFILE_SWITCH, profile="team")


def test_a_personal_workspace_is_still_frictionless(tmp_path, monkeypatch):
    """No roles configured anywhere means no login, and no prompt for one."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("SLM_USER_SESSION", raising=False)

    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    config = SLMConfig.load()
    db = DatabaseManager(config.db_path)
    db.initialize(schema)
    db.close()

    gate_cli_mutation, OperationKind = _gate()
    gate_cli_mutation(OperationKind.REMEMBER)  # must not raise


def test_the_read_scope_is_clamped_by_the_role_settings_too(workspace):
    """The same toggle governs reads.

    A client that asks to read across every workspace must be answered by the
    same rule that governs writing to them. When this path read only
    config.toml, a request for cross-workspace results was left unclamped over
    one transport and refused over another.
    """
    from superlocalmemory.core.admission import enforce_read_scope
    from superlocalmemory.core.operation_policy_registry import OperationPolicyRegistry
    from superlocalmemory.core.operation_request import OperationKind

    registry = OperationPolicyRegistry.default()
    policy = registry._policies.get(OperationKind.RECALL)
    assert policy is not None and policy.allow_cross_profile is False, (
        "this test is only meaningful while cross-workspace reads are off by "
        "default; if that default changes, this test must change with it"
    )

    assert enforce_read_scope(True, True, registry=registry) == (False, False)
    # An unspecified flag is left alone so the server default still applies.
    assert enforce_read_scope(None, None, registry=registry) == (None, None)


def test_a_personal_workspace_reads_across_its_own_scopes_unclamped(
    tmp_path, monkeypatch,
):
    """The control for the clamp: nothing to enforce means nothing enforced."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    from superlocalmemory.core.admission import enforce_read_scope

    assert enforce_read_scope(True, True) == (True, True)


def test_a_present_but_uninterpretable_config_does_not_grant_owner_access(
    tmp_path, monkeypatch,
):
    """An unreadable answer is not a "yes".

    A config file that exists but cannot be interpreted leaves the workspace's
    trust level unknown. Unknown must not resolve to the most permissive
    answer, because on an enterprise store that hands out owner access.
    """
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    (tmp_path / "config.toml").write_text('[deployment]\nmode = "enterprise"\n', encoding="utf-8")

    import superlocalmemory.core.config as config_module
    from superlocalmemory.core.admission import _resolve_deployment

    def _explode(*args, **kwargs):
        raise RuntimeError("simulated failure inside the loader")

    monkeypatch.setattr(config_module, "load_deployment_config", _explode)
    assert _resolve_deployment().is_enterprise is True


def test_no_config_at_all_is_still_a_personal_install(tmp_path, monkeypatch):
    """The control for the fail-closed rule: a fresh install stays frictionless."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    from superlocalmemory.core.admission import _resolve_deployment

    assert (tmp_path / "config.toml").exists() is False
    assert _resolve_deployment().is_enterprise is False
