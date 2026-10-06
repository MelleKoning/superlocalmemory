# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The unreferenced-vector sweep removes only vectors no fact can reach."""

from __future__ import annotations

import sqlite3

import pytest

sqlite_vec = pytest.importorskip("sqlite_vec")


@pytest.fixture
def store(tmp_path):
    from superlocalmemory.retrieval.vector_store import VectorStore, VectorStoreConfig

    vs = VectorStore(tmp_path / "memory.db", VectorStoreConfig(dimension=4))
    if not vs.available:
        pytest.skip("sqlite-vec cannot load on this Python")
    for i in range(5):
        assert vs.upsert(f"f{i}", "default", [float(i + 1), 0.0, 1.0, 0.5])
    return vs, tmp_path / "memory.db"


def _vec_rows(db_path) -> int:
    from superlocalmemory.storage.vector_residue import vec_connection

    with vec_connection(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()[0])


def _orphan(db_path, fact_id: str) -> None:
    """What a delete with foreign keys off (or the metadata GC) leaves behind."""
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM embedding_metadata WHERE fact_id = ?", (fact_id,))
    conn.execute("DELETE FROM vector_row_map WHERE fact_id = ?", (fact_id,))
    conn.commit()
    conn.close()


def test_sweep_removes_only_unreferenced_vectors(store):
    from superlocalmemory.storage.vector_residue import (
        sweep_unreferenced_vectors,
        unreferenced_rowids,
        vec_connection,
    )

    vs, db_path = store
    _orphan(db_path, "f1")
    _orphan(db_path, "f3")
    with vec_connection(db_path) as conn:
        assert len(unreferenced_rowids(conn)) == 2

    assert sweep_unreferenced_vectors(db_path) == 2
    assert _vec_rows(db_path) == 3
    assert vs.indexed_fact_ids("default") == {"f0", "f2", "f4"}
    assert sweep_unreferenced_vectors(db_path) == 0  # idempotent


def test_sweep_is_bounded_per_pass(store):
    from superlocalmemory.storage.vector_residue import sweep_unreferenced_vectors

    _vs, db_path = store
    for fact_id in ("f0", "f1", "f2"):
        _orphan(db_path, fact_id)
    assert sweep_unreferenced_vectors(db_path, limit=2) == 2
    assert sweep_unreferenced_vectors(db_path, limit=2) == 1


def test_delete_rechecks_references_under_the_write_lock(store):
    """A rowid that became referenced after it was listed is never removed."""
    from superlocalmemory.storage.vector_residue import (
        delete_rowids,
        unreferenced_rowids,
        vec_connection,
    )

    _vs, db_path = store
    with vec_connection(db_path) as conn:
        referenced = [int(r[0]) for r in conn.execute("SELECT vec_rowid FROM embedding_metadata")]
        assert delete_rowids(conn, db_path, referenced) == 0
        assert unreferenced_rowids(conn) == []
    assert _vec_rows(db_path) == 5
