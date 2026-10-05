# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Cancel and every start-up message tell the truth at every stage (L1-03).

Once a restore has started writing, the store may already be the copy and the
request's export is the only record of the memories written after it. Cancel
must then be refused, and no message may say "your memories were not changed".
"""

from __future__ import annotations

import io
from argparse import Namespace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.cli import upgrade_cmd
from superlocalmemory.core.file_lock import exclusive_lock
from superlocalmemory.server.routes import upgrade_restore as routes
from superlocalmemory.storage import _restore_boot as rb
from superlocalmemory.storage import backup
from superlocalmemory.storage import upgrade_restore as ur
from superlocalmemory.storage._durable_json import read_json

from ._upgrade_store import add_memory, current_store, memory_ids


def _killed_after_the_write(tmp_path: Path, monkeypatch):
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m_old", ["Deploys go out on Tuesdays"])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    add_memory(memory_db, "m_new", ["The new office is in Pune"])
    ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path,
                       memory_db=memory_db)

    def killed(*_a, **_k):
        raise SystemExit("process killed after the store was written")
    with monkeypatch.context() as m:
        m.setattr(rb, "_finish", killed)
        with pytest.raises(SystemExit):
            ur.perform_pending_restore(tmp_path, memory_db, learning_db)
    intent = read_json(tmp_path / "restore-intent.json")
    assert intent["stage"] == "writing" and memory_ids(memory_db) == {"m_old"}
    return learning_db, memory_db, point, Path(intent["final_delta_dir"])


def test_cancel_is_refused_once_writing_began(tmp_path, monkeypatch) -> None:
    learning_db, memory_db, _point, final = _killed_after_the_write(tmp_path, monkeypatch)

    with pytest.raises(ur.RestoreRefusedError) as refused:
        ur.cancel_restore(tmp_path)

    assert "part-way through" in str(refused.value) and "Restart" in str(refused.value)
    assert (tmp_path / "restore-intent.json").exists()
    assert "m_new" in (final / "memories.jsonl").read_text(encoding="utf-8"), "the export is kept"
    outcome = ur.perform_pending_restore(tmp_path, memory_db, learning_db)
    assert outcome.status == "restored" and outcome.reimport_pending
    assert "m_new" in (Path(outcome.delta_dir) / "memories.jsonl").read_text(encoding="utf-8")


def test_refusal_at_writing_stage_does_not_claim_nothing_changed(tmp_path, monkeypatch) -> None:
    learning_db, memory_db, _point, _final = _killed_after_the_write(tmp_path, monkeypatch)
    lease = memory_db.resolve().with_name(memory_db.name + ".writer.lock")

    with exclusive_lock(lease):
        outcome = ur.perform_pending_restore(tmp_path, memory_db, learning_db)

    assert outcome.status == "refused"
    assert "not changed" not in outcome.message
    assert "may already hold the restored copy" in outcome.message
    assert (tmp_path / "restore-intent.json").exists()


def test_refusal_before_writing_still_says_nothing_changed(tmp_path) -> None:
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays"])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path,
                       memory_db=memory_db)
    lease = memory_db.resolve().with_name(memory_db.name + ".writer.lock")

    with exclusive_lock(lease):
        outcome = ur.perform_pending_restore(tmp_path, memory_db, learning_db)

    assert outcome.status == "refused" and "were not changed" in outcome.message
    assert ur.cancel_restore(tmp_path) is True, "a request that has not started can go"


def test_a_copy_lost_mid_restore_hands_the_export_to_the_reimport(tmp_path, monkeypatch) -> None:
    learning_db, memory_db, point, final = _killed_after_the_write(tmp_path, monkeypatch)
    point.memory_snapshot.write_bytes(b"")                    # the copy is gone

    outcome = ur.perform_pending_restore(tmp_path, memory_db, learning_db)

    assert outcome.status == "snapshot_unusable"
    assert outcome.reimport_pending and Path(outcome.delta_dir) == final
    assert "not changed" not in outcome.message
    assert "m_new" in (final / "memories.jsonl").read_text(encoding="utf-8")


def test_dashboard_and_cli_refuse_cancel_while_writing(tmp_path, monkeypatch, capsys) -> None:
    _l, _m, _p, final = _killed_after_the_write(tmp_path, monkeypatch)
    monkeypatch.setattr(routes, "_require_credential", lambda _r: None)
    monkeypatch.setattr(routes, "_require_manage", lambda _r: {"username": "owner"})
    app = FastAPI()
    routes.register(app, data_root=tmp_path)

    r = TestClient(app).post("/api/upgrade/restore/cancel")

    assert r.status_code == 409 and "part-way through" in r.json()["error"]
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    code = upgrade_cmd.cmd_db_restore(Namespace(data_root=tmp_path, json=False, yes=False,
                                                cancel=True, no_reimport=False, point_id=None))
    assert code == 1 and "part-way through" in capsys.readouterr().out
    assert final.is_dir() and (tmp_path / "restore-intent.json").exists()
