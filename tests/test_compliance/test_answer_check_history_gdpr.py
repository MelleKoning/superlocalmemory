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


def test_gdpr_export_is_never_capped(root) -> None:
    """Between sweeps a profile holds more than the retention ceiling (an agent
    recalling every 2 s adds 300 rows in 10 minutes). The export carries all of
    them, and so for every learning table, not just this one."""
    _seed(root, "alice", store.MAX_ROWS_CEILING + 300)
    with sqlite3.connect(root / "learning.db") as conn:
        conn.execute("CREATE TABLE zz_signals (profile_id TEXT, n INTEGER)")
        conn.executemany("INSERT INTO zz_signals VALUES ('alice', ?)",
                         [(i,) for i in range(25_001)])
    signals = _gdpr(root).export_profile_data("alice")["learning_signals"]
    rows = signals["answer_check_events"]
    assert len(rows) == store.MAX_ROWS_CEILING + 300 == _rows(root, "alice")
    assert len({r["event_id"] for r in rows}) == len(rows)
    assert len(signals["zz_signals"]) == 25_001


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


# -- an erasure made on an older version is applied at the next start (A-F5) ----------

def _seed_now(root: Path, profile: str, n: int) -> None:
    """Checks made after the profile was created (``_seed`` dates them 60 s back)."""
    now = int(time.time() * 1000) + 1_000
    events = [h.event_from_response(make_response(), profile, now_ms=now + i, origin_name="")
              for i in range(n)]
    conn = store.connect(root / "learning.db", readonly=False)
    store.insert_batch(conn, events)
    conn.close()


def _erase_like_4_1_19(root: Path, profile: str) -> None:
    """What 4.1.18/4.1.19's erasure does: learning reset, then the profile's
    rows and the profile itself. It has never heard of the history tables."""
    from superlocalmemory.learning.database import LearningDatabase

    LearningDatabase(root / "learning.db").reset(profile)
    with sqlite3.connect(root / "memory.db") as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        for table in ("atomic_facts", "memories"):
            conn.execute(f"DELETE FROM {table} WHERE profile_id=?", (profile,))  # noqa: S608
        conn.execute("DELETE FROM profiles WHERE profile_id=?", (profile,))


def _tombstoned(root: Path, profile: str) -> bool:
    with sqlite3.connect(root / "learning.db") as conn:
        return conn.execute("SELECT 1 FROM answer_check_erasures WHERE profile_id=?",
                            (profile,)).fetchone() is not None


def test_an_erasure_on_an_older_version_is_applied_when_the_writer_starts(root) -> None:
    _seed_now(root, "alice", 20)
    _seed_now(root, "default", 5)
    _erase_like_4_1_19(root, "alice")
    assert _rows(root, "alice") == 20            # what the older version leaves behind
    store.start_writer(root / "learning.db", memory_db=root / "memory.db")
    try:
        deadline = time.monotonic() + 10
        while _rows(root, "alice") and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        store.stop_writer()
    assert _rows(root, "alice") == 0 and _tombstoned(root, "alice")
    assert _rows(root, "default") == 5


def test_history_from_before_a_profile_was_created_again_is_erased(root) -> None:
    now = int(time.time() * 1000)
    _seed(root, "alice", 7)                       # the first alice, 60 s ago
    _erase_like_4_1_19(root, "alice")
    with sqlite3.connect(root / "memory.db") as conn:  # the same name, created again
        conn.execute("INSERT INTO profiles (profile_id, name, created_at) "
                     "VALUES ('alice', 'Alice', datetime('now'))")
    later = [h.event_from_response(make_response(), "alice", now_ms=now + 5_000 + i,
                                   origin_name="") for i in range(3)]
    conn = store.connect(root / "learning.db", readonly=False)
    store.insert_batch(conn, later)               # the new alice's own checks
    conn.close()
    assert store.reconcile_with_profiles(root / "learning.db", root / "memory.db") == {
        "profiles": 1, "rows": 7}
    assert _rows(root, "alice") == 3


def test_reconcile_erases_nothing_it_cannot_be_sure_of(root, tmp_path) -> None:
    _seed(root, "alice", 4)
    nowhere = tmp_path / "missing" / "memory.db"
    assert store.reconcile_with_profiles(root / "learning.db", nowhere)["rows"] == 0
    with sqlite3.connect(root / "memory.db") as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DELETE FROM profiles")
    assert store.reconcile_with_profiles(root / "learning.db", root / "memory.db")["rows"] == 0
    assert _rows(root, "alice") == 4


def test_reconcile_keeps_every_live_profiles_history(root) -> None:
    _seed_now(root, "alice", 9)
    _seed_now(root, "default", 2)
    # A check 4 s older than its profile's row: a clock nudged back, not a person.
    with sqlite3.connect(root / "memory.db") as conn:
        created = conn.execute("SELECT created_at FROM profiles WHERE profile_id='alice'"
                               ).fetchone()[0]
    nudged = store._created_ms(created) - 4_000
    conn = store.connect(root / "learning.db", readonly=False)
    store.insert_batch(conn, [h.event_from_response(make_response(), "alice", now_ms=nudged,
                                                    origin_name="")])
    conn.close()
    assert store.reconcile_with_profiles(root / "learning.db", root / "memory.db") == {
        "profiles": 0, "rows": 0}
    assert _rows(root, "alice") == 10 and _rows(root, "default") == 2
