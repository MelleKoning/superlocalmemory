# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The daemon's in-memory vector index decodes stored vectors in a child process.

Decoding 16k JSON embeddings took ~2 s of interpreter time on the warm-up
thread of a 22k-fact store, and a recall 7.5 s after ready took 3.7 s behind
it (retrieval/vector_index_build.py). The child must build the same index.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from superlocalmemory.retrieval import canonical_vector_index as cvi
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.schema import create_all_tables

DIM = 4


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(path))
    create_all_tables(conn)
    conn.execute("ALTER TABLE atomic_facts ADD COLUMN archive_status TEXT DEFAULT 'live'")
    conn.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES ('m','default','s')")
    rng = np.random.default_rng(9)
    for i in range(50):
        v = rng.normal(size=DIM)
        raw = json.dumps(v.tolist()) if i % 2 else v.astype(np.float32).tobytes()
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, embedding, "
            "quarantined, archive_status) VALUES (?,?,?,?,?,?,?)",
            (f"f{i:02d}", "m", "default", f"t{i}", raw, int(i == 7),
             "archived" if i == 9 else "live"))
    conn.commit()
    conn.close()
    manager = DatabaseManager(path)
    yield manager
    manager.close()


def _part(index, profile="default"):
    with index._lock:
        return index._fresh(profile)


def test_the_child_builds_exactly_the_in_place_index(db) -> None:
    here = cvi.CanonicalVectorIndex(db, DIM)
    there = cvi.CanonicalVectorIndex(db, DIM)
    there.build_in_child = True
    a, b = _part(here), _part(there)
    assert a.ids == b.ids and len(a.ids) == 48  # withheld and archived excluded
    assert np.array_equal(a.matrix[: a.size], b.matrix[: b.size])
    q = np.ones(DIM)
    assert here.search(q, 5, "default") == there.search(q, 5, "default")


def test_the_daemon_index_never_decodes_on_this_interpreter(db, monkeypatch) -> None:
    index = cvi.CanonicalVectorIndex(db, DIM)
    index.build_in_child = True
    monkeypatch.setattr(cvi, "partition_arrays",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("decoded here")))
    assert _part(index).count == 48


def test_a_child_that_cannot_run_falls_back_to_building_in_place(db, monkeypatch) -> None:
    from superlocalmemory.retrieval import vector_index_build as vib

    monkeypatch.setattr(vib, "child_arrays", None)
    index = cvi.CanonicalVectorIndex(db, DIM)
    index.build_in_child = True
    assert _part(index).count == 48


def test_daemon_start_turns_it_on(monkeypatch) -> None:
    from superlocalmemory.retrieval import adjacency_rcu
    from superlocalmemory.retrieval import entity_graph_warmup as egw
    from superlocalmemory.server import recall_warmup

    index = SimpleNamespace(build_in_child=False)
    engine = SimpleNamespace(_retrieval_engine=SimpleNamespace(_kind_vectors=index),
                             profile_id="default")
    monkeypatch.setattr(egw, "warm", lambda *a, **k: None)
    monkeypatch.setattr(adjacency_rcu, "prefer_background_builds", lambda *a, **k: None)
    monkeypatch.setattr(recall_warmup, "_state", {"phase": "pending"})
    thread = recall_warmup.begin(engine)
    if thread is not None:
        thread.join(timeout=30)
    assert index.build_in_child is True
