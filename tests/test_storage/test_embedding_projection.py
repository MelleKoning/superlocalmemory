# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""#147: the channels read id + embedding only, and nothing about recall moves.

The contract (GitHub #147): the visible-fact predicate stays byte-identical,
the same candidates come back in the same order with the same scores, rows
with a missing or wrong-width embedding are still skipped, and no channel
hydrates a full fact to score a candidate.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from superlocalmemory.math.hopfield import HopfieldConfig
from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel
from superlocalmemory.retrieval.semantic_channel import SemanticChannel
from superlocalmemory.retrieval.spreading_activation import SpreadingActivation
from superlocalmemory.storage.database import DatabaseManager
from tests.test_retrieval.cross_scope_fixture import (
    OWN,
    REQ,
    PartitionedVS,
    build_store,
)

SCOPES = [(True, True), (True, False), (False, True)]


@pytest.fixture(autouse=True)
def _no_background_writes(monkeypatch):
    import superlocalmemory.storage.deferred_writes as dw
    monkeypatch.setattr(dw, "submit_background", lambda fn: None)


@pytest.fixture(params=["blob", "json"])
def store(request, tmp_path):
    s = build_store(
        tmp_path / "p.db", n_local=40, n_global=60, n_shared=20, n_denied=10,
        degree=3, json_text=request.param == "json",
    )
    _add_odd_rows(s.db)
    return s


def _add_odd_rows(db: DatabaseManager) -> None:
    """NULL, wrong-width, withheld and archived external rows."""
    cols = {dict(r)["name"] for r in db.execute("PRAGMA table_info(atomic_facts)")}
    if "archive_status" not in cols:
        db.execute(
            "ALTER TABLE atomic_facts ADD COLUMN archive_status TEXT DEFAULT 'live'"
        )
    short = np.ones(384, dtype=np.float32)
    odd = [
        ("X_null", None),
        ("X_short_blob", short.tobytes()),
        ("X_short_json", "[" + ",".join(["0.5"] * 384) + "]"),
        ("X_withheld", np.ones(768, dtype=np.float32).tobytes()),
        ("X_archived", np.ones(768, dtype=np.float32).tobytes()),
    ]
    for i, (fid, emb) in enumerate(odd):
        db.execute(
            "INSERT INTO memories (memory_id, profile_id, scope, content) "
            "VALUES (?,?, 'global', ?)", (f"m_{fid}", OWN, fid),
        )
        db.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, scope, content,"
            " fact_type, confidence, importance, evidence_count, access_count,"
            " embedding, created_at) VALUES (?,?,?, 'global', ?, 'semantic', 0.9,"
            " 0.5, 1, 0, ?, datetime('2025-06-01', ?))",
            (fid, f"m_{fid}", OWN, fid, emb, f"+{i} seconds"),
        )
    db.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = 'X_withheld'")
    db.execute(
        "UPDATE atomic_facts SET archive_status = 'archived' WHERE fact_id = 'X_archived'"
    )


def _hydrated_pairs(facts) -> list[tuple[str, np.ndarray]]:
    """What the channels computed from a hydrated fact before #147."""
    return [
        (f.fact_id, np.array(f.embedding, dtype=np.float32))
        for f in facts if f.embedding is not None
    ]


def _same_pairs(a, b) -> bool:
    return [x for x, _ in a] == [x for x, _ in b] and all(
        u.dtype == v.dtype and u.shape == v.shape and u.tobytes() == v.tobytes()
        for (_, u), (_, v) in zip(a, b)
    )


# ---------------------------------------------------------------------------
# The two readers agree with the hydrating readers they replace
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("include_global,include_shared", SCOPES)
def test_external_projection_matches_hydration_bit_for_bit(
    store, include_global, include_shared,
) -> None:
    kw = dict(include_global=include_global, include_shared=include_shared)
    hydrated = _hydrated_pairs(store.db.get_external_visible_facts(REQ, **kw))
    projected = store.db.get_external_visible_embeddings(REQ, **kw)

    assert _same_pairs(projected, hydrated)
    ids = {fid for fid, _ in projected}
    assert not ids & {"X_withheld", "X_archived", "X_null"}
    assert not any(fid[0] in "LDP" for fid in ids), "own or forbidden row leaked"
    if include_global:
        assert {"X_short_blob", "X_short_json"} <= ids  # callers skip on width


def test_personal_scope_projection_is_empty(store) -> None:
    assert store.db.get_external_visible_embeddings(REQ) == []


@pytest.mark.parametrize("include_global,include_shared", SCOPES)
def test_by_ids_projection_matches_hydration_bit_for_bit(
    store, include_global, include_shared,
) -> None:
    kw = dict(include_global=include_global, include_shared=include_shared)
    wanted = sorted(store.embs) + ["X_null", "X_withheld", "X_archived",
                                   "X_short_blob", "missing"]
    hydrated = _hydrated_pairs(store.db.get_facts_by_ids(wanted, REQ, **kw))
    projected = store.db.get_fact_embeddings_by_ids(wanted, REQ, **kw)
    assert _same_pairs(projected, hydrated)
    assert len(projected) > 40


def test_an_unreadable_row_is_skipped_not_fatal(store, caplog) -> None:
    store.db.execute(
        "UPDATE atomic_facts SET embedding = ? WHERE fact_id = 'G00003'", (b"\x00" * 7,),
    )
    store.db.execute(
        "UPDATE atomic_facts SET embedding = '[1, 2' WHERE fact_id = 'G00004'",
    )
    with caplog.at_level(logging.WARNING):
        got = store.db.get_external_visible_embeddings(REQ, include_global=True)
    ids = [fid for fid, _ in got]
    assert "G00003" not in ids and "G00004" not in ids
    assert "G00005" in ids, "one bad row took its neighbours with it"
    assert "G00003" in caplog.text and "G00004" in caplog.text


# ---------------------------------------------------------------------------
# Each channel returns exactly what it returned with hydrated facts
# ---------------------------------------------------------------------------

class _HydratingDB:
    """The store as the channels saw it before #147: projections via hydration."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def __getattr__(self, name):
        return getattr(self._db, name)

    def get_external_visible_embeddings(self, profile_id, **kw):
        return _hydrated_pairs(self._db.get_external_visible_facts(profile_id, **kw))

    def get_fact_embeddings_by_ids(self, fact_ids, profile_id, include_global=False,
                                   include_shared=False):
        return _hydrated_pairs(self._db.get_facts_by_ids(
            fact_ids, profile_id, include_global=include_global,
            include_shared=include_shared,
        ))


def _channels(db, store):
    local_vs = PartitionedVS(store.embs, store.ids("L"))
    return {
        "spreading": SpreadingActivation(db, local_vs),
        "semantic": SemanticChannel(db, vector_store=local_vs),
        "hopfield": HopfieldChannel(
            db=db, vector_store=local_vs,
            config=HopfieldConfig(prefilter_candidates=10),
        ),
    }


@pytest.mark.parametrize("include_global,include_shared", SCOPES)
def test_every_channel_answers_identically(store, include_global, include_shared) -> None:
    new = _channels(store.db, store)
    old = _channels(_HydratingDB(store.db), store)
    kw = dict(include_global=include_global, include_shared=include_shared)
    for name in new:
        for q in store.queries[:4]:
            got = new[name].search(q.tolist(), REQ, top_k=10, **kw)
            want = old[name].search(q.tolist(), REQ, top_k=10, **kw)
            assert got == want, f"{name} changed its answer"
            assert got, f"{name} returned nothing; the comparison proves nothing"


def test_no_channel_hydrates_a_fact_to_score_it(store, monkeypatch) -> None:
    """Deterministic cost check: rows hydrated during a cross-scope search.

    Counting rows instead of timing them keeps this exact on any CI machine.
    """
    hydrated = {"n": 0}
    real = DatabaseManager._row_to_fact

    def counting(self, row):
        hydrated["n"] += 1
        return real(self, row)

    monkeypatch.setattr(DatabaseManager, "_row_to_fact", counting)
    chans = _channels(store.db, store)
    q = store.queries[0].tolist()
    for name in ("spreading", "hopfield"):
        hydrated["n"] = 0
        assert chans[name].search(q, REQ, top_k=10, include_global=True,
                                  include_shared=True)
        assert hydrated["n"] == 0, f"{name} hydrated {hydrated['n']} rows"
    # Semantic hydrates only its final top_k*2 candidates (it needs their
    # Fisher vectors), never the external supplement.
    hydrated["n"] = 0
    assert chans["semantic"].search(q, REQ, top_k=10, include_global=True,
                                    include_shared=True)
    assert hydrated["n"] <= 20, f"semantic hydrated {hydrated['n']} rows"
