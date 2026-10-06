# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Search by meaning stays exact, fresh and fast without the sqlite-vec extension.

On a Python that cannot load SQLite extensions the semantic channel and the
spreading-activation seeds read every fact on every recall (5.6-6.7 s each on a
21,739-fact store) and were abandoned at the 8 s guard on every recall. The
in-memory index that replaces those scans must give the same candidates a full
scan of the store gives, and must never be stale: a fact written, embedded,
withheld, moved or erased by any writer is reflected on the next search.
"""

from __future__ import annotations

import numpy as np
import pytest

from superlocalmemory.retrieval import canonical_vector_index as cvi
from superlocalmemory.retrieval.semantic_channel import SemanticChannel
from superlocalmemory.retrieval.spreading_activation import SpreadingActivation
from superlocalmemory.storage import fact_search_changes as changes
from tests.test_retrieval.cross_scope_fixture import DIM, REQ, build_store


@pytest.fixture(params=[False, True], ids=["blob", "json-text"])
def store(tmp_path, request):
    return build_store(tmp_path / "v.db", json_text=request.param)


def _brute(store, q: np.ndarray, k: int, profile: str = REQ) -> list[str]:
    rows = store.db.execute(
        "SELECT fact_id FROM atomic_facts WHERE profile_id = ?"
        f"{store.db.visible_fact_clause()}", (profile,))
    ids = [dict(r)["fact_id"] for r in rows]
    qn = q / np.linalg.norm(q)
    scored = sorted(((-float(np.dot(store.embs[f] / np.linalg.norm(store.embs[f]), qn)), f)
                     for f in ids))
    return [f for _, f in scored[:k]]


def _insert(store, fid: str, vec: np.ndarray | None, profile: str = REQ) -> None:
    with store.db.raw_connection() as conn:
        conn.execute("INSERT INTO memories (memory_id, profile_id, scope, content)"
                     " VALUES (?, ?, 'personal', 'x')", (f"m_{fid}", profile))
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, scope, content,"
            " fact_type, confidence, importance, evidence_count, access_count,"
            " embedding, created_at) VALUES (?, ?, ?, 'personal', 'new', 'semantic',"
            " 0.9, 0.5, 1, 0, ?, datetime('now'))",
            (fid, f"m_{fid}", profile, None if vec is None else vec.astype(np.float32).tobytes()))
    if vec is not None:
        store.embs[fid] = vec.astype(np.float32)


def test_the_index_returns_what_a_full_scan_of_the_store_returns(store) -> None:
    index = cvi.CanonicalVectorIndex(store.db, DIM)
    for q in store.queries[:5]:
        got = [f for f, _ in index.search(q.tolist(), top_k=20, profile_id=REQ)]
        assert got == _brute(store, q, 20)
    assert index.builds == 1


def test_scores_are_the_vector_store_scale_and_ties_are_ordered_by_id(store) -> None:
    index = cvi.CanonicalVectorIndex(store.db, DIM)
    q = store.embs[store.ids("L")[0]]
    _insert(store, "L_TWIN_B", q)
    _insert(store, "L_TWIN_A", q)
    hits = index.search(q.tolist(), top_k=3, profile_id=REQ)
    assert [f for f, _ in hits[:3]] == sorted([store.ids("L")[0], "L_TWIN_A", "L_TWIN_B"])
    assert all(0.0 <= s <= 1.0 + 1e-6 for _, s in hits)


def test_a_fact_written_after_the_build_is_found_without_a_rebuild(store) -> None:
    index = cvi.CanonicalVectorIndex(store.db, DIM)
    q = store.queries[0]
    index.search(q.tolist(), top_k=5, profile_id=REQ)
    _insert(store, "LNEW0", q)
    assert index.search(q.tolist(), top_k=1, profile_id=REQ)[0][0] == "LNEW0"
    assert index.builds == 1 and index.deltas_applied >= 1


def test_a_vector_added_to_an_existing_fact_is_found(store) -> None:
    index = cvi.CanonicalVectorIndex(store.db, DIM)
    q = store.queries[1]
    _insert(store, "LLATE", None)
    assert "LLATE" not in {f for f, _ in index.search(q.tolist(), top_k=50, profile_id=REQ)}
    store.db.execute("UPDATE atomic_facts SET embedding = ? WHERE fact_id = 'LLATE'",
                     (q.astype(np.float32).tobytes(),))
    assert index.search(q.tolist(), top_k=1, profile_id=REQ)[0][0] == "LLATE"


@pytest.mark.parametrize("change", [
    "UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?",
    "DELETE FROM atomic_facts WHERE fact_id = ?",
    "UPDATE atomic_facts SET profile_id = 'owner' WHERE fact_id = ?",
])
def test_a_withheld_erased_or_moved_fact_leaves_before_the_next_search(store, change) -> None:
    index = cvi.CanonicalVectorIndex(store.db, DIM)
    top = store.ids("L")[3]
    q = store.embs[top]
    assert index.search(q.tolist(), top_k=1, profile_id=REQ)[0][0] == top
    store.db.execute(change, (top,))
    assert top not in {f for f, _ in index.search(q.tolist(), top_k=200, profile_id=REQ)}


def test_reading_a_fact_does_not_touch_the_log(store) -> None:
    head = changes.log_bounds(store.db)[0]
    store.db.execute("UPDATE atomic_facts SET access_count = access_count + 1")
    assert changes.log_bounds(store.db)[0] == head


def test_a_reader_behind_the_retained_log_rebuilds_and_stays_correct(store) -> None:
    index = cvi.CanonicalVectorIndex(store.db, DIM)
    q = store.queries[2]
    index.search(q.tolist(), top_k=5, profile_id=REQ)
    _insert(store, "LGAP", q)
    head = changes.log_bounds(store.db)[0]
    store.db.execute(f"DELETE FROM {changes.TABLE} WHERE seq <= ?", (head,))
    _insert(store, "LGAP2", -q)  # the retained log now starts past the reader
    assert index.search(q.tolist(), top_k=1, profile_id=REQ)[0][0] == "LGAP"
    assert index.builds == 2


def test_a_store_whose_log_went_backwards_is_rebuilt(store) -> None:
    index = cvi.CanonicalVectorIndex(store.db, DIM)
    q = store.queries[3]
    index.search(q.tolist(), top_k=5, profile_id=REQ)
    index._seq += 10_000  # what a restored, older store file looks like
    assert [f for f, _ in index.search(q.tolist(), 10, REQ)] == _brute(store, q, 10)
    assert index.builds == 2


def test_without_the_log_the_channels_keep_their_full_scan(store) -> None:
    for trigger in changes.trigger_names():
        store.db.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    store.db.execute(f"DROP TABLE {changes.TABLE}")
    assert cvi.candidate_vector_source(store.db, None, DIM) is None


def test_semantic_channel_gives_the_same_top_answers_as_its_full_scan(store) -> None:
    source = cvi.candidate_vector_source(store.db, None, DIM)
    fast = SemanticChannel(store.db, vector_store=source)
    slow = SemanticChannel(store.db, vector_store=None)
    for q in store.queries[:5]:
        a = fast.search(q.tolist(), REQ, top_k=10)
        b = slow.search(q.tolist(), REQ, top_k=10)
        assert [f for f, _ in a] == [f for f, _ in b]
        assert np.allclose([s for _, s in a], [s for _, s in b], atol=1e-5)


def test_spreading_activation_seeds_are_the_full_scan_seeds(store) -> None:
    source = cvi.candidate_vector_source(store.db, None, DIM)
    fast = SpreadingActivation(store.db, source)
    slow = SpreadingActivation(store.db, None)
    for q in store.queries[:5]:
        a = fast._seed_search(q.tolist(), REQ, include_global=False, include_shared=False)
        b = slow._seed_search(q.tolist(), REQ, include_global=False, include_shared=False)
        assert [f for f, _ in a] == [f for f, _ in b]


def test_an_empty_exhaustive_answer_does_not_fall_back_to_a_full_scan(store, monkeypatch) -> None:
    source = cvi.candidate_vector_source(store.db, None, DIM)
    ch = SemanticChannel(store.db, vector_store=source)
    monkeypatch.setattr(ch, "_search_full_scan", lambda *a, **k: pytest.fail("full scan"))
    assert ch.search(store.queries[0].tolist(), "nobody", top_k=5) == []
