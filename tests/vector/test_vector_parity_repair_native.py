# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm db repair`` vector parity against a REAL Lance table (native lane).

tests/storage/test_vector_parity_repair.py covers the rules with a fake
projection; this proves the real one: ids listed without reading vectors, a
preview that creates nothing, orphans removed in one table version, and the
repair finding the projection by itself when SLM is stopped.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3

import numpy as np
import pytest

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(importlib.util.find_spec("lancedb") is None,
                       reason="LanceDB optional dependency not installed"),
]


def _vec(seed: int) -> list[float]:
    rng = np.random.RandomState(seed)
    v = rng.randn(768).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _store(engine, text: str) -> str:
    from superlocalmemory.core.engine_ingestion import canonical_store, local_trusted_actor_id

    receipt = canonical_store(engine, text, source_type="python-api",
                              trusted_actor_id=local_trusted_actor_id("python-api"),
                              require_complete=True, return_receipt=True)
    return list(receipt.final_fact_ids)[0]


@pytest.fixture
def promoted(engine_with_mock_deps):
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.mutations import delete_fact_authorized
    from superlocalmemory.vector.lancedb_backend import LanceDBVectorBackend

    engine = engine_with_mock_deps
    ids = {"ok": _store(engine, "Synthetic ok memory about the orchard row one."),
           "soft": _store(engine, "Synthetic soft deleted memory about the orchard row two."),
           "erased": _store(engine, "Synthetic erased memory about the north gate.")}
    assert delete_fact_authorized(engine, ids["erased"],
                                  trusted_actor_id=local_trusted_actor_id("python-api"),
                                  source_agent_id="test").get("ok")
    db_path = engine._db.db_path
    engine.close()

    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE atomic_facts SET archive_status = 'archived' WHERE fact_id = ?",
                 (ids["soft"],))
    conn.execute("INSERT OR REPLACE INTO backend_status (backend_name, status, record_count, "
                 "error_message) VALUES ('lancedb', 'active', 3, '')")
    conn.commit()
    conn.close()
    root = db_path.parent
    (root / "config.json").write_text(json.dumps(
        {"scale_engine_state": "promoted", "vector_backend": "lancedb"}), encoding="utf-8")
    backend = LanceDBVectorBackend(str(root / "lance"))
    quoted = "gh'ost"
    backend.add_vectors([ids["ok"], ids["soft"], ids["erased"], quoted],
                        [_vec(1), _vec(2), _vec(3), _vec(4)], ["active"] * 4)
    backend.close()
    return {"db": db_path, "root": root, "ids": ids, "quoted": quoted}


def _plan(db):
    from superlocalmemory.storage.integrity_scan import plan

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return plan(conn)
    finally:
        conn.close()


def test_the_scan_finds_the_projection_on_disk_and_changes_nothing(promoted):
    files = sorted(str(p) for p in promoted["root"].joinpath("lance").rglob("*"))
    lance = _plan(promoted["db"])["vector_parity"]["lance"]

    assert lance == {"state": "active", "rows": 4, "orphans": 3}
    assert sorted(str(p) for p in promoted["root"].joinpath("lance").rglob("*")) == files


def test_a_store_with_no_projection_is_not_opened_or_created(tmp_path):
    from superlocalmemory.vector.lancedb_backend import LanceDBVectorBackend

    assert LanceDBVectorBackend.read_fact_ids(str(tmp_path / "lance")) is None
    assert not (tmp_path / "lance").exists()


def test_apply_finds_the_projection_by_itself_and_removes_in_one_version(promoted):
    from superlocalmemory.storage.integrity_repair import Limits, Repair
    from superlocalmemory.vector.lancedb_backend import LanceDBVectorBackend

    lance_dir = str(promoted["root"] / "lance")
    probe = LanceDBVectorBackend(lance_dir)
    versions_before = probe._manifest_count()
    probe.close()

    summary = Repair(promoted["db"], limits=Limits(pause_s=0, confirm_s=0)).apply()

    assert summary["status"] == "finished"
    assert summary["done"]["vector_parity.lance_orphans_removed"] == 3
    assert LanceDBVectorBackend.read_fact_ids(lance_dir) == [promoted["ids"]["ok"]]
    probe = LanceDBVectorBackend(lance_dir)
    assert probe._manifest_count() == versions_before + 1  # three ids, one write
    probe.close()
    assert summary["after"]["vector_parity"]["lance"]["orphans"] == 0
