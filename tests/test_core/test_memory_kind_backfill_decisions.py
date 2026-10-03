# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Classification runs: the per-fact decision, bounded waiting, and short writes.

Split from test_memory_kind_backfill.py (same fixtures). What is pinned here:
a strong rules cue is not replaced by a disagreeing model; a background wait
under steady recall load always ends; every batch is one short write whose
lock hold is measured without SQLite's automatic checkpoint, and the batch
size follows that measurement (median of three, never on one spike).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from superlocalmemory.core import recall_gate
from superlocalmemory.core.memory_kind_backfill_plan import decide, plan_changes
from superlocalmemory.core.memory_kind_config import MemoryKindConfig
from superlocalmemory.encoding.memory_kind_recipe import KINDS_V1
from superlocalmemory.encoding.memory_kind_rules import suggest_by_rules
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.memory_kind_store import KindCandidate, MemoryKindStore
from superlocalmemory.storage.memory_kinds import KindAssignment, KindSource, MemoryKind
from tests.test_core.test_memory_kind_backfill import TEXTS, _cfg, _engine, _fact, _runner


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    """A real store with the real M052 migration applied."""
    from superlocalmemory.storage import schema as real_schema
    from superlocalmemory.storage.migrations import M052_memory_kinds as m052

    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(real_schema)
    with mgr.raw_connection() as conn:
        m052.apply(conn)
    return mgr

# ---------------------------------------------------------------------------
# The per-fact decision (rules vs model)
# ---------------------------------------------------------------------------


class TestDecision:
    def _model(self, kind: MemoryKind, conf: float = 0.9) -> KindAssignment:
        return KindAssignment(kind, KindSource.MODEL_LAYA, conf, KINDS_V1.recipe_id)

    def test_strong_cue_beats_a_disagreeing_model(self) -> None:
        rules = suggest_by_rules("Never push to main without a review.", "semantic")
        assert rules.kind is MemoryKind.RULE
        chosen = decide(self._model(MemoryKind.OPINION), rules, MemoryKind.RULE)
        assert chosen == rules

    def test_model_fills_in_where_no_cue_fired(self) -> None:
        rules = suggest_by_rules("The office is in Pune.", "semantic")
        model = self._model(MemoryKind.STATUS)
        assert decide(model, rules, None) == model

    def test_agreement_keeps_the_model_confidence(self) -> None:
        rules = suggest_by_rules("We decided to use SQLite.", "semantic")
        model = self._model(MemoryKind.DECISION, 0.7)
        assert decide(model, rules, MemoryKind.DECISION) == model

    def test_correction_cue_defers_to_the_verify_check(self) -> None:
        rules = suggest_by_rules("Correction: the port is 8765, not 8767.", "semantic")
        assert rules.kind is MemoryKind.CORRECTION
        model = self._model(MemoryKind.SEMANTIC)   # verify < 0.8 already applied
        assert decide(model, rules, MemoryKind.CORRECTION) == model

    def test_rules_and_off_pass_through(self) -> None:
        rules = suggest_by_rules("x", "semantic")
        assert decide(rules, rules, None) == rules
        assert decide(None, rules, None) is None


def test_idle_wait_is_bounded_under_constant_recall_load() -> None:
    recall_gate.begin_recall()
    try:
        done = threading.Event()

        def background() -> None:
            with recall_gate.background_work(), \
                    recall_gate.idle_wait_deadline(time.monotonic() + 0.3):
                recall_gate.wait_for_foreground_idle()
            done.set()

        t = threading.Thread(target=background)
        t.start()
        assert done.wait(2.0), "the bounded wait never returned"
        t.join()
    finally:
        recall_gate.end_recall()


def test_unbounded_wait_is_unchanged_for_other_callers() -> None:
    recall_gate.begin_recall()
    done = threading.Event()

    def background() -> None:
        with recall_gate.background_work():
            recall_gate.wait_for_foreground_idle()
        done.set()

    t = threading.Thread(target=background, daemon=True)
    t.start()
    try:
        assert not done.wait(0.4)
    finally:
        recall_gate.end_recall()
    assert done.wait(2.0)
    t.join()


class _SlowDiskStore(MemoryKindStore):
    """Writes hold the lock for ``write_s``; records each read's batch size."""

    def __init__(self, db: DatabaseManager, write_s: float) -> None:
        super().__init__(db)
        self.write_s = write_s
        self.limits: list[int] = []

    def select_batch(self, profile_id, *, after_rowid, limit, mode):
        self.limits.append(limit)
        return super().select_batch(profile_id, after_rowid=after_rowid, limit=limit,
                                    mode=mode)

    def apply_batch(self, *a, **k):
        import dataclasses

        result = super().apply_batch(*a, **k)
        return dataclasses.replace(result, write_ms=self.write_s * 1000.0 or 0.5)


def test_batch_shrinks_when_writes_hold_the_lock_too_long(db: DatabaseManager) -> None:
    for i in range(260):
        _fact(db, f"fact number {i}")
    store = _SlowDiskStore(db, write_s=0.08)          # 80 ms: over the 50 ms target
    runner = _runner(_engine(db, cfg=_cfg(batch_size={"rules": 40})), store=store)
    runner.create_run("default", mode="untyped", requested_by="test")
    for _ in range(9):
        runner.run_once()
    # Decided on the median of the last three writes, never below the floor.
    assert store.limits[:9] == [40, 40, 40, 20, 20, 20, 10, 10, 10]
    store.write_s = 0.0                               # a fast disk: grow back
    for _ in range(4):
        runner.run_once()
    assert store.limits[-1] == 20 and max(store.limits) <= 40


def test_one_slow_write_does_not_shrink_the_batch(db: DatabaseManager) -> None:
    for i in range(200):
        _fact(db, f"fact number {i}")
    store = _SlowDiskStore(db, write_s=0.0)
    runner = _runner(_engine(db, cfg=_cfg(batch_size={"rules": 20})), store=store)
    runner.create_run("default", mode="untyped", requested_by="test")
    for i in range(6):
        store.write_s = 0.08 if i == 2 else 0.0      # a checkpoint-like spike
        runner.run_once()
    assert set(store.limits) == {20}


def test_a_batch_is_one_write_transaction(db: DatabaseManager, monkeypatch) -> None:
    for t in TEXTS:
        _fact(db, t)
    runner = _runner(_engine(db, cfg=_cfg(batch_size={"rules": 2})))
    run = runner.create_run("default", mode="untyped", requested_by="test")
    runner.run_once()                                  # queued -> running + first batch
    writes: list[str] = []
    real = db.execute

    def spy(sql, params=()):
        if sql.lstrip().upper().startswith(("UPDATE", "INSERT", "DELETE")):
            writes.append(sql.split()[1] if len(sql.split()) > 1 else sql)
        return real(sql, params)

    monkeypatch.setattr(db, "execute", spy)
    assert runner.run_once().action == "batch"
    assert writes == []        # facts, history and run counters: all in apply_batch
    got = runner._store_for(runner._engine()).get_run(run["run_id"])
    assert got["processed"] == 4 and got["changed"] == 4


def test_a_batch_reports_its_lock_hold_and_checkpoints_after(db: DatabaseManager) -> None:
    from superlocalmemory.storage.memory_kind_store import KindChange

    fid = _fact(db, TEXTS[0])
    store = MemoryKindStore(db)
    run = store.create_run("default", backend="rules", recipe_id="r", mode="untyped",
                           requested_by="t", total_estimate=1)
    assert store.set_run_status(run["run_id"], "running", expected=["queued"])
    rowid = int(db.execute("SELECT rowid FROM atomic_facts WHERE fact_id=?", (fid,))[0]["rowid"])
    change = KindChange(rowid, fid, None, None, None, "semantic",
                        suggest_by_rules(TEXTS[0], "semantic"))
    result = store.apply_batch(run["run_id"], "default", [change], new_cursor=rowid,
                               actor="t", examined=1)
    assert result.applied == 1 and result.write_ms > 0.0
    # SQLite's automatic checkpoint was kept out of the timed write and run
    # afterwards: the main database file (read here ignoring the WAL) already
    # holds the batch.
    conn = sqlite3.connect(f"file:{db.db_path}?immutable=1", uri=True)
    try:
        kind = conn.execute("SELECT memory_kind FROM atomic_facts WHERE fact_id = ?",
                            (fid,)).fetchone()[0]
    finally:
        conn.close()
    assert kind == "rule"
    # The manager's normal setting is untouched for every other connection.
    assert int(dict(db.execute("PRAGMA wal_autocheckpoint")[0])["wal_autocheckpoint"]) == 400


def test_default_batch_keeps_a_real_shaped_write_short() -> None:
    # Measured on a copy of a real 640 MB store (rows carry two 3 KB vectors):
    # 200 facts held the write lock ~115 ms, 100 ~52 ms, 50 ~30 ms (max 40).
    assert MemoryKindConfig().batch_size["rules"] <= 50


def test_plan_changes_drops_no_ops_and_lower_authority() -> None:
    cands = [
        KindCandidate(1, "a", "default", "x", "semantic", "rule", "rules"),
        KindCandidate(2, "b", "default", "y", "semantic", "rule", "user"),
        KindCandidate(3, "c", "default", "z", "semantic", None, None),
    ]
    same = KindAssignment(MemoryKind.RULE, KindSource.RULES, None, "kinds-rules-v1")
    changes, unchanged = plan_changes(cands, [same, same, same], {})
    assert [c.fact_id for c in changes] == ["c"]
    assert unchanged == 2
