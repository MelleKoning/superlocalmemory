# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The same question on the same store gets the same bridges, on any machine.

Bridge discovery used to stop at a 0.4 s wall-clock deadline and walked the
candidate entities in set order, which Python randomises per process. On a busy
machine it simply did less, and which part it did changed from run to run:
measured on the author's store, the same 50 questions ranked differently on two
runs of the same code. These tests fix the clock, the load and the hash seed
in turn and require one answer.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import RetrievalConfig
from superlocalmemory.retrieval import bridge_discovery as bd
from superlocalmemory.retrieval.bridge_discovery import BridgeDiscovery
from superlocalmemory.retrieval.engine import RetrievalEngine
from superlocalmemory.storage import entity_index
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, CanonicalEntity, MemoryRecord

_SEEDS = [f"seed-{i}" for i in range(6)]


def _entities_of_seed(i: int) -> list[str]:
    return [f"E{i * 5 + k:02d}" for k in range(5)]


def build_store(path: Path) -> DatabaseManager:
    """Six seeds with disjoint entities, so every neighbouring pair needs bridges."""
    db = DatabaseManager(path)
    db.initialize(real_schema)
    for n in range(30):
        db.store_entity(CanonicalEntity(entity_id=f"E{n:02d}", canonical_name=f"e{n}"))
    db.store_memory(MemoryRecord(memory_id="m0", content="parent"))
    for i, sid in enumerate(_SEEDS):
        db.store_fact(AtomicFact(fact_id=sid, memory_id="m0", content=sid,
                                 canonical_entities=_entities_of_seed(i)))
    for n in range(30):
        for k in range(3):
            # Some bridges also name an entity of the next seed, so their
            # overlap - and score - differs and the sort has work to do.
            extra = [f"E{(n + 5) % 30:02d}"] if k == 0 else []
            db.store_fact(AtomicFact(
                fact_id=f"b-{n:02d}-{k}", memory_id="m0", content=f"bridge {n} {k}",
                canonical_entities=[f"E{n:02d}", *extra],
                created_at=f"2026-01-{k + 1:02d}T00:00:00+00:00",
            ))
    return db


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    return build_store(tmp_path / "memory.db")


class _SlowClock:
    """Every reading is one second after the last: a machine under heavy load."""

    def __init__(self) -> None:
        self._now = time.monotonic()

    def monotonic(self) -> float:
        self._now += 1.0
        return self._now

    def __getattr__(self, name):
        return getattr(time, name)


def _discover(db: DatabaseManager, **kw):
    return BridgeDiscovery(db).discover(list(_SEEDS), "default", **kw)


class TestSameAnswerEveryTime:
    def test_five_runs_agree(self, db) -> None:
        runs = [_discover(db) for _ in range(5)]
        assert runs[0], "the fixture must produce bridges"
        assert all(r == runs[0] for r in runs)

    def test_a_slow_clock_changes_nothing(self, db, monkeypatch) -> None:
        expected = _discover(db)
        monkeypatch.setattr(bd, "time", _SlowClock(), raising=False)
        assert [_discover(db) for _ in range(5)] == [expected] * 5

    def test_slow_lookups_change_nothing(self, db, monkeypatch) -> None:
        expected = _discover(db)
        real = entity_index.facts_for_entity

        def slow(*a, **kw):
            time.sleep(0.05)  # 10 lookups x 50 ms passes the old 0.4 s deadline
            return real(*a, **kw)

        real_scan = db.get_facts_by_entity

        def slow_scan(*a, **kw):
            time.sleep(0.05)
            return real_scan(*a, **kw)

        # Both lookup paths, so this holds whichever one discovery uses.
        monkeypatch.setattr(entity_index, "facts_for_entity", slow)
        monkeypatch.setattr(db, "get_facts_by_entity", slow_scan)
        assert _discover(db) == expected

    def test_a_recall_ranks_the_same_five_times_under_a_slow_clock(
        self, db, monkeypatch,
    ) -> None:
        monkeypatch.setattr(bd, "time", _SlowClock(), raising=False)
        semantic = MagicMock()
        semantic.search.return_value = [(s, 0.9 - i * 0.01) for i, s in enumerate(_SEEDS)]
        embedder = MagicMock()
        embedder.embed.return_value = [0.1, 0.2, 0.3]
        # Bridges carry no channel score of their own; on a real store the
        # entity-graph boost lets them past the evidence floor. The floor is
        # not what this test is about, so it is off.
        engine = RetrievalEngine(
            db=db, config=RetrievalConfig(evidence_floor_enabled=False),
            channels={"semantic": semantic},
            embedder=embedder, bridge_discovery=BridgeDiscovery(db),
        )
        try:
            runs = [
                [r.fact.fact_id for r in engine.recall(
                    "which project uses the parser", "default", limit=20).results]
                for _ in range(5)
            ]
        finally:
            engine.close()
        assert any(fid.startswith("b-") for fid in runs[0]), runs[0]
        assert all(r == runs[0] for r in runs)


_HASH_SEED_SCRIPT = """
import sys
from pathlib import Path
sys.path.insert(0, {tests!r})
from test_retrieval.test_bridge_determinism import build_store, _SEEDS
from superlocalmemory.retrieval.bridge_discovery import BridgeDiscovery
db = build_store(Path({path!r}))
print(BridgeDiscovery(db).discover(list(_SEEDS), "default", max_lookups=4))
"""


def test_a_partial_walk_is_the_same_under_any_hash_seed(tmp_path: Path) -> None:
    """When the work budget stops the walk early, it stops at the same place.

    Set iteration order differs between processes with different hash seeds;
    the walk is sorted, so it must not.
    """
    tests = str(Path(__file__).resolve().parents[1])
    outputs = []
    for seed in ("1", "2", "3"):
        script = _HASH_SEED_SCRIPT.format(tests=tests, path=str(tmp_path / f"{seed}.db"))
        env = {**os.environ, "PYTHONHASHSEED": seed}
        done = subprocess.run([sys.executable, "-c", script], env=env,
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr[-2000:]
        outputs.append(done.stdout.strip().splitlines()[-1])
    assert outputs[0] != "[]"
    assert len(set(outputs)) == 1, outputs


class TestTheEntityIndex:
    def test_every_stored_fact_is_indexed(self, db) -> None:
        stored = {(r["fact_id"], r["entity_id"]) for r in map(dict, db.execute(
            "SELECT fact_id, entity_id FROM fact_entity_associations"))}
        assert ("seed-0", "E00") in stored and ("b-29-0", "E04") in stored
        assert len(stored) == 6 * 5 + 30 * 3 + 30

    def test_indexed_and_scanned_lookups_agree(self, db) -> None:
        for n in range(30):
            eid = f"E{n:02d}"
            assert (
                entity_index.facts_for_entity(db, eid, "default", limit=5, indexed=True)
                == entity_index.facts_for_entity(db, eid, "default", limit=5, indexed=False)
            ), eid

    def test_backfill_indexes_older_facts_once(self, tmp_path: Path) -> None:
        store = build_store(tmp_path / "old.db")
        store.execute("DELETE FROM fact_entity_associations")  # an older store
        counts = dict(store.execute("SELECT entity_id, fact_count FROM canonical_entities"))
        first = entity_index.backfill(tmp_path / "old.db", batch_size=50, max_batches=100)
        again = entity_index.backfill(tmp_path / "old.db", batch_size=50, max_batches=100)
        assert first["complete"] and first["inserted"] == 6 * 5 + 30 * 3 + 30
        assert again["inserted"] == 0
        assert entity_index.is_complete(store)
        # It records that a pair exists; it never counts a fact again.
        assert dict(store.execute(
            "SELECT entity_id, fact_count FROM canonical_entities")) == counts
