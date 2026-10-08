"""Checking a save's vectors reads only that save's facts.

The vector owner's fingerprint step asked for EVERY indexed fact of the
profile and then kept the save's few ids: a join of all metadata rows to the
vec0 table, 2.4 s on a 22k-vector store, three times per save. That was 43%
of the background enrichment's time while recalls ran. The answer for the
asked ids must be exactly the whole-profile answer narrowed to them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from superlocalmemory.core.transactions.concrete_owners import VectorOwner
from superlocalmemory.core.transactions.owners import OperationContext
from superlocalmemory.retrieval.vector_store import VectorStore, VectorStoreConfig
from superlocalmemory.storage import schema as real_schema

DIM = 4


def _store(tmp_path: Path) -> VectorStore:
    path = tmp_path / "v.db"
    conn = sqlite3.connect(str(path))
    real_schema.create_all_tables(conn)
    conn.commit()
    conn.close()
    store = VectorStore(path, VectorStoreConfig(dimension=DIM, enabled=True))
    if not store.available:
        pytest.skip("sqlite-vec not loadable here")
    rng = np.random.default_rng(5)
    for i in range(30):
        for pid in ("default", "other"):
            v = rng.normal(size=DIM)
            store.upsert(f"{pid}-{i:02d}", pid, (v / np.linalg.norm(v)).tolist())
    return store


@pytest.mark.parametrize("ids", [
    ("default-03",),
    ("default-00", "default-29", "missing", "other-04"),
    tuple(f"default-{i:02d}" for i in range(30)) + ("default-03",),
    (),
])
def test_narrowed_answer_equals_whole_profile_answer(tmp_path: Path, ids) -> None:
    store = _store(tmp_path)
    whole = store.indexed_fact_ids("default")
    assert store.indexed_among("default", ids) == whole & set(ids)


def test_vector_owner_reads_only_the_operation_facts(tmp_path: Path) -> None:
    store = _store(tmp_path)

    def whole_profile(_profile_id):
        raise AssertionError("fingerprints read every indexed fact of the profile")

    store.indexed_fact_ids = whole_profile  # type: ignore[method-assign]

    class _Db:
        def execute(self, sql, params=()):
            return []

    owner = VectorOwner(_Db(), vector_store=store)
    ctx = OperationContext(operation_id="op1", profile_id="default", subject_id="m1",
                           fact_ids=("default-01", "missing"))
    assert owner._fingerprints(ctx) == {}
