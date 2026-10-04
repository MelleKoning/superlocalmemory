# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Answer Check history under GDPR: erased with the person, exported in full,
removed with a deleted workspace — and NOT wiped by "Reset learning data"."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from superlocalmemory.compliance.gdpr import GDPRCompliance
from superlocalmemory.core import answer_check_history as h
from superlocalmemory.core import answer_check_history_store as store
from superlocalmemory.storage import migration_runner as mr
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, MemoryRecord

from ..test_core.test_answer_check_history import make_response


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    store._reset_for_testing()
    h._reset_for_testing()
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(real_schema)
    mgr.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('alice', 'Alice')")
    mgr.store_memory(MemoryRecord(memory_id="m1", profile_id="alice", content="Alice info"))
    mgr.store_fact(AtomicFact(fact_id="f1", memory_id="m1", profile_id="alice",
                              content="Alice fact"))
    result = mr.apply_all(tmp_path / "learning.db", tmp_path / "memory.db")
    assert "M053_answer_check_history" in result["applied"]
    yield tmp_path
    store._reset_for_testing()
    h._reset_for_testing()


def _seed(root: Path, profile: str, n: int) -> None:
    now = int(time.time() * 1000) - 60_000
    events = [h.event_from_response(make_response(), profile, now_ms=now + i, origin_name="")
              for i in range(n)]
    conn = store.connect(root / "learning.db", readonly=False)
    for i in range(0, n, 128):
        store.insert_batch(conn, events[i:i + 128])
    conn.close()


def _rows(root: Path, profile: str) -> int:
    with sqlite3.connect(root / "learning.db") as conn:
        return conn.execute("SELECT COUNT(*) FROM answer_check_events WHERE profile_id=?",
                            (profile,)).fetchone()[0]


def _gdpr(root: Path) -> GDPRCompliance:
    return GDPRCompliance(DatabaseManager(root / "memory.db"), data_root=root)


def test_gdpr_erase_removes_history(root) -> None:
    _seed(root, "alice", 20)
    _seed(root, "bob", 5)
    h.enable(True)
    h.record_recall_verdict(make_response(), profile_id="alice")   # unsaved, in the ring
    counts = _gdpr(root).forget_profile("alice")
    assert counts["answer_check_history"] == 20
    assert _rows(root, "alice") == 0 and _rows(root, "bob") == 5
    assert h.recent("alice", after_seq=0, limit=50)[0] == []
    with sqlite3.connect(root / "learning.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM answer_check_erasures "
                            "WHERE profile_id='alice'").fetchone()[0] == 1


def test_gdpr_export_includes_all_rows_at_cap(root) -> None:
    _seed(root, "alice", store.MAX_ROWS_CEILING)
    exported = _gdpr(root).export_profile_data("alice")
    rows = exported["learning_signals"]["answer_check_events"]
    assert len(rows) == store.MAX_ROWS_CEILING
    assert store.clamp_settings(30, 10**9)[1] <= len(rows), \
        "retention may never keep more rows than an export carries"


def test_profile_delete_removes_history(root, monkeypatch) -> None:
    from superlocalmemory.server.routes import helpers

    _seed(root, "alice", 7)
    monkeypatch.setattr(helpers, "DB_PATH", root / "memory.db")
    helpers.delete_profile_from_db("alice")
    assert _rows(root, "alice") == 0


def test_learning_reset_keeps_history(root) -> None:
    from superlocalmemory.learning.database import LearningDatabase

    _seed(root, "alice", 4)
    LearningDatabase(root / "learning.db").reset("alice")
    assert _rows(root, "alice") == 4


def test_erase_failure_aborts_before_memory_rows(root, monkeypatch) -> None:
    def broken(*_a, **_k):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(store, "erase_profile_everywhere", broken)
    gdpr = _gdpr(root)
    with pytest.raises(RuntimeError, match="learning receipt purge failed"):
        gdpr.forget_profile("alice")
    mgr = DatabaseManager(root / "memory.db")
    assert mgr.execute("SELECT 1 FROM profiles WHERE profile_id='alice'")
    assert mgr.execute("SELECT 1 FROM atomic_facts WHERE profile_id='alice'")
