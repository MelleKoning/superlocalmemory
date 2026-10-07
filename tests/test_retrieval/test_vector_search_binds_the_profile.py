# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Vector reads bind the profile instead of joining on the vector table's partition.

``em.profile_id = fe.profile_id`` makes sqlite-vec run a nested statement for
every candidate row to read the partition value. Each one allocates, and this
SQLite build takes one process-wide lock per allocation, so a single search
held up every other database read in the daemon (native samples during slow
recalls on a 22k-fact store). With the profile bound the rows are the same.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from superlocalmemory.retrieval.vector_store import VectorStore, VectorStoreConfig
from superlocalmemory.storage import schema as real_schema

DIM = 4
_PARTITION_JOIN = ("em.profile_id = fe.profile_id", "fe.profile_id = em.profile_id")


def _store(tmp_path: Path) -> VectorStore:
    path = tmp_path / "v.db"
    conn = sqlite3.connect(str(path))
    real_schema.create_all_tables(conn)
    conn.commit()
    conn.close()
    store = VectorStore(path, VectorStoreConfig(dimension=DIM, enabled=True))
    if not store.available:
        pytest.skip("sqlite-vec not loadable here")
    rng = np.random.default_rng(3)
    for i in range(40):
        for pid in ("default", "other"):
            v = rng.normal(size=DIM)
            store.upsert(f"{pid}-{i:02d}", pid, (v / np.linalg.norm(v)).tolist())
    return store


def _traced(store: VectorStore, monkeypatch) -> list[str]:
    seen: list[str] = []
    real = store._connect

    def connect():
        conn = real()
        conn.set_trace_callback(seen.append)
        return conn

    monkeypatch.setattr(store, "_connect", connect)
    return seen


def test_search_answers_exactly_what_the_partition_join_answered(tmp_path: Path) -> None:
    store = _store(tmp_path)
    q = [0.3, -0.2, 0.9, 0.1]
    got = store.search(q, top_k=10, profile_id="default")
    with store._managed_connection() as conn:  # the previous statement, as an oracle
        rows = conn.execute(
            "SELECT fe.distance, em.fact_id FROM fact_embeddings AS fe JOIN embedding_metadata AS em "
            "ON em.vec_rowid = fe.rowid AND em.profile_id = fe.profile_id "
            "WHERE fe.embedding MATCH ? AND fe.profile_id = ? AND fe.k = ?",
            (store._serialize_f32(q), "default", 10)).fetchall()
    oracle = sorted(((str(r["fact_id"]), max(0.0, 1.0 - r["distance"])) for r in rows),
                    key=lambda x: (-x[1], x[0]))
    assert got == oracle
    assert all(fid.startswith("default-") for fid, _ in got)
    assert store.indexed_fact_ids("other") == {f"other-{i:02d}" for i in range(40)}
    assert store.count("default") == 40
    assert store.is_searchable_by_meaning("other-07", "other")
    assert not store.is_searchable_by_meaning("other-07", "default")


def test_no_profile_read_joins_on_the_partition(tmp_path: Path, monkeypatch) -> None:
    store = _store(tmp_path)
    seen = _traced(store, monkeypatch)
    store.search([0.3, -0.2, 0.9, 0.1], top_k=5, profile_id="default")
    store.indexed_fact_ids("default")
    store.count("default")
    store.is_searchable_by_meaning("default-01", "default")
    joined = [s for s in seen if any(j in s for j in _PARTITION_JOIN)]
    assert joined == [], joined
