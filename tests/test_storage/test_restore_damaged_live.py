# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A live store SQLite cannot read never blocks a restore point (L1-02).

The store is damaged after the copy was taken (a disk error, a sync client, a
crash in a file copy). The preview, the request and the start-up restore must
all still work: a full restore with nothing carried over, said plainly first,
the damaged file kept aside, and never a 500, a traceback, or a request set
aside as "failed" because the old store could not be read.
"""

from __future__ import annotations

import io
import json
import sqlite3
from argparse import Namespace
from contextlib import closing
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.cli import upgrade_cmd
from superlocalmemory.server.routes import upgrade_restore as routes
from superlocalmemory.storage import backup
from superlocalmemory.storage import upgrade_restore as ur

from ._upgrade_store import add_memory, current_store, fact_ids, integrity


def _damaged(tmp_path: Path):
    learning_db, memory_db = current_store(tmp_path)
    for i in range(60):
        add_memory(memory_db, f"m{i}", [f"Fact number {i} about the deployment pipeline " * 4],
                   fact_ids=[f"f{i}"])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    return learning_db, memory_db, point


def _damage(memory_db: Path) -> None:
    with closing(sqlite3.connect(memory_db)) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    size = memory_db.stat().st_size
    with open(memory_db, "r+b") as fh:
        fh.truncate(size // 2)


def test_preview_and_request_work_on_a_damaged_store(tmp_path) -> None:
    learning_db, memory_db, point = _damaged(tmp_path)
    _damage(memory_db)

    preview = ur.preview_restore(point.point_id, data_root=tmp_path, memory_db=memory_db)

    assert preview.verified and not preview.live_store_readable and not preview.problems
    assert preview.facts_in_snapshot == 60 and preview.added_memories == 0
    assert any("cannot be read" in w and "replaces it entirely" in w for w in preview.warnings)
    ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path,
                       memory_db=memory_db)

    outcome = ur.perform_pending_restore(tmp_path, memory_db, learning_db)

    assert outcome.status == "restored", outcome.message
    assert "could not be carried over" in outcome.message
    assert integrity(memory_db) == "ok" and len(fact_ids(memory_db)) == 60
    kept = list((tmp_path / "pre-restore").glob("memory-*-damaged-before-restore.db"))
    assert len(kept) == 1 and kept[0].stat().st_size > 0
    assert not (tmp_path / "restore-intent.json").exists()


def test_a_request_made_before_the_damage_still_restores_and_adds_back(tmp_path) -> None:
    learning_db, memory_db, point = _damaged(tmp_path)
    add_memory(memory_db, "later", ["Written after the copy was taken"], fact_ids=["later-f"])
    ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path,
                       memory_db=memory_db)
    _damage(memory_db)                                  # damaged before the restart

    outcome = ur.perform_pending_restore(tmp_path, memory_db, learning_db)

    assert outcome.status == "restored", outcome.message
    assert not list(tmp_path.glob("restore-intent.failed-*.json")), "never set aside"
    added = [json.loads(line)["memory_id"] for line in
             (Path(outcome.delta_dir) / "memories.jsonl").read_text(encoding="utf-8").splitlines()]
    assert added == ["later"] and outcome.reimport_pending, "the request-time export is used"
    assert integrity(memory_db) == "ok"


def test_a_missing_store_file_is_restored_in_full(tmp_path) -> None:
    learning_db, memory_db, point = _damaged(tmp_path)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{memory_db}{suffix}").unlink(missing_ok=True)

    preview = ur.preview_restore(point.point_id, data_root=tmp_path, memory_db=memory_db)
    ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path,
                       memory_db=memory_db)
    outcome = ur.perform_pending_restore(tmp_path, memory_db, learning_db)

    assert not preview.live_store_readable and outcome.status == "restored", outcome.message
    assert len(fact_ids(memory_db)) == 60


def test_cli_warns_and_does_not_crash(tmp_path, monkeypatch, capsys) -> None:
    _learning_db, memory_db, point = _damaged(tmp_path)
    _damage(memory_db)
    monkeypatch.setattr(upgrade_cmd, "_daemon_running", lambda: False)
    monkeypatch.setattr(upgrade_cmd, "_start_daemon", lambda: True)
    monkeypatch.setattr(upgrade_cmd, "_load_config", lambda: None)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    args = Namespace(data_root=tmp_path, json=False, yes=True, cancel=False,
                     no_reimport=False, point_id=point.point_id)

    code = upgrade_cmd.cmd_db_restore(args)

    out = capsys.readouterr().out
    assert code == 0, out
    assert "warning:" in out and "cannot be read" in out and "Traceback" not in out
    assert len(fact_ids(memory_db)) == 60


def test_dashboard_preview_answers_instead_of_500(tmp_path, monkeypatch) -> None:
    _learning_db, memory_db, point = _damaged(tmp_path)
    _damage(memory_db)
    monkeypatch.setattr(routes, "_require_credential", lambda _r: None)
    monkeypatch.setattr(routes, "_require_manage", lambda _r: {"username": "owner"})
    app = FastAPI()
    routes.register(app, data_root=tmp_path)

    r = TestClient(app).post("/api/upgrade/restore/preview",
                             json={"restore_point_id": point.point_id})

    assert r.status_code == 200, r.text
    assert r.json()["live_store_readable"] is False and r.json()["verified"] is True
