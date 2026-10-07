# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Exact nearest-neighbour search over the stored vectors, kept in memory.

WHY
---
When the sqlite-vec extension cannot load (a Python built without loadable
SQLite extensions, as on some managed macOS installs), the semantic channel,
the spreading-activation seeds and Hopfield had no vector index. Semantic and
the seed search each read every fact (``SELECT *``) and decoded its embedding
and both Fisher vectors on every recall: measured on a copy of a 21,739-fact
store, 5.6-6.7 s of CPU per channel. Together they passed the 8 s hang guard on
every recall, so every answer came back without meaning-based search, and the
abandoned scans kept the channel pool busy into the next recall.

WHAT THIS IS
------------
The answer sqlite-vec gives -- exact (not approximate) cosine nearest
neighbours within one owner profile, withheld and soft-deleted facts excluded,
score ``max(0, cosine)`` -- from an L2-normalised float32 matrix in memory. A
search is one matrix-vector product: about 1 ms for 21,739 x 768 (67 MB).

It is a candidate source only, like the extension: every channel still loads
and authorises its candidates through the canonical scope predicate. Kind-
scoped recall also uses it to search only one kind's facts (``search_within``),
whichever platform it is on.

Freshness is the shared rule in ``retrieval/change_tracked``: a vector written
by any process is searchable on the next recall, and an erased, withheld or
re-scoped fact is gone before the search that could have returned it.

Ties are broken on ``fact_id`` and the cut is taken after ranking, so the same
question over the same store returns the same candidates every time.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from superlocalmemory.retrieval.change_tracked import ChangeTrackedCache
from superlocalmemory.storage.embedding_projection import decode_embedding_array

logger = logging.getLogger(__name__)

#: ids per ``IN (...)`` read; well inside SQLITE_MAX_VARIABLE_NUMBER.
_CHUNK = 800


@dataclass
class _Partition:
    """One owner profile's vectors. A row index never moves except on compaction."""

    dim: int
    ids: list[str] = field(default_factory=list)
    row_of: dict[str, int] = field(default_factory=dict)
    matrix: np.ndarray | None = None
    live: np.ndarray | None = None
    size: int = 0

    def __post_init__(self) -> None:
        if self.matrix is None:
            self.matrix = np.zeros((0, self.dim), dtype=np.float32)
        if self.live is None:
            self.live = np.zeros(0, dtype=bool)

    @property
    def count(self) -> int:
        return len(self.row_of)


class CanonicalVectorIndex(ChangeTrackedCache[_Partition]):
    """The in-memory stand-in for sqlite-vec's ``VectorStore`` read side."""

    WHAT = "v"
    #: Its answer covers every visible vector: an empty result means none.
    exhaustive = True

    def __init__(self, db, dimension: int) -> None:
        super().__init__(db)
        self._dim = int(dimension)

    # -- the VectorStore read interface ----------------------------------

    def search(
        self, query_embedding, top_k: int = 30, profile_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Top-k ``(fact_id, max(0, cosine))``, the scale ``VectorStore`` uses."""
        return self.search_within(query_embedding, top_k, profile_id, None)

    def search_within(
        self, query_embedding, top_k: int, profile_id: str | None,
        allowed: Iterable[str] | None,
    ) -> list[tuple[str, float]]:
        """``search`` restricted to ``allowed`` fact ids (None: no restriction)."""
        q = self._query(query_embedding)
        if q is None or profile_id is None or top_k <= 0 or not self.available:
            return []
        with self._lock:
            part = self._fresh(profile_id)
            if part.count == 0:
                return []
            mask = part.live[: part.size]
            if allowed is not None:
                rows = [part.row_of[f] for f in allowed if f in part.row_of]
                if not rows:
                    return []
                only = np.zeros(part.size, dtype=bool)
                only[rows] = True
                mask = mask & only
            scores = np.where(mask, part.matrix[: part.size] @ q, -np.inf)
            return _top_k(scores, part.ids, top_k)

    def count(self, profile_id: str | None = None) -> int:
        if profile_id is None or not self.available:
            return 0
        with self._lock:
            return self._fresh(profile_id).count

    def warm(self, profile_id: str) -> int:
        """Build a profile now (daemon start), so no recall pays for it."""
        return self.count(profile_id)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "profiles": len(self._parts),
                "vectors": sum(p.count for p in self._parts.values()),
                "builds": self.builds,
                "deltas_applied": self.deltas_applied,
                "last_change": self._seq,
            }

    # -- ChangeTrackedCache ----------------------------------------------

    def _build(self, profile_id: str) -> _Partition:
        t0 = time.monotonic()
        rows = self._db.execute(
            "SELECT fact_id, embedding FROM atomic_facts WHERE profile_id = ?"
            f"{self._db.visible_fact_clause()} ORDER BY fact_id",
            (profile_id,),
        )
        part = _Partition(self._dim)
        ids: list[str] = []
        vecs: list[np.ndarray] = []
        for r in rows:
            d = dict(r)
            vec = self._decode(d["embedding"], d["fact_id"])
            if vec is not None:
                ids.append(str(d["fact_id"]))
                vecs.append(vec)
        if vecs:
            part.matrix = np.vstack(vecs).astype(np.float32, copy=False)
            part.ids = ids
            part.row_of = {fid: i for i, fid in enumerate(ids)}
            part.live = np.ones(len(ids), dtype=bool)
            part.size = len(ids)
        logger.info(
            "in-memory vector index: %d vectors for profile %s in %.0f ms",
            part.count, profile_id, (time.monotonic() - t0) * 1000.0,
        )
        return part

    def _apply(self, fact_ids: list[str]) -> None:
        visible = self._db.visible_fact_clause()
        current: dict[str, tuple[str, np.ndarray | None]] = {}
        for start in range(0, len(fact_ids), _CHUNK):
            batch = fact_ids[start:start + _CHUNK]
            marks = ",".join("?" for _ in batch)
            for r in self._db.execute(
                f"SELECT fact_id, profile_id, embedding FROM atomic_facts "
                f"WHERE fact_id IN ({marks}){visible}", tuple(batch),
            ):
                d = dict(r)
                current[str(d["fact_id"])] = (
                    str(d["profile_id"]), self._decode(d["embedding"], d["fact_id"]),
                )
        for fact_id in fact_ids:
            owner, vec = current.get(fact_id, (None, None))
            for pid, part in self._parts.items():
                if pid == owner and vec is not None:
                    _upsert(part, fact_id, vec)
                else:
                    _remove(part, fact_id)
        for part in self._parts.values():
            _compact_if_sparse(part)

    # -- helpers -----------------------------------------------------------

    def _query(self, query_embedding) -> np.ndarray | None:
        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        if q.shape[0] != self._dim:
            return None
        norm = float(np.linalg.norm(q))
        return q / norm if norm >= 1e-10 else None

    def _decode(self, raw, fact_id) -> np.ndarray | None:
        try:
            vec = decode_embedding_array(raw, fact_id=str(fact_id))
        except (ValueError, TypeError) as exc:
            logger.warning("vector index skips fact %s: unreadable embedding (%s)",
                           fact_id, exc)
            return None
        if vec is None or vec.ndim != 1 or vec.shape[0] != self._dim:
            return None
        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype(np.float32) if norm > 1e-10 else None


def _top_k(scores: np.ndarray, ids: list[str], k: int) -> list[tuple[str, float]]:
    """Rank first, then cut, ties on fact_id: the same k every time."""
    finite = np.isfinite(scores)
    n = int(finite.sum())
    if n == 0:
        return []
    k = min(k, n)
    kth = np.partition(scores[finite], n - k)[n - k]
    picked = np.nonzero(finite & (scores >= kth))[0]
    ranked = sorted(
        ((ids[i], max(0.0, float(scores[i]))) for i in picked),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[:k]


def _upsert(part: _Partition, fact_id: str, vec: np.ndarray) -> None:
    row = part.row_of.get(fact_id)
    if row is None:
        if part.size == part.matrix.shape[0]:
            grow = max(64, part.size // 2)
            part.matrix = np.vstack([part.matrix, np.zeros((grow, part.dim), np.float32)])
            part.live = np.concatenate([part.live, np.zeros(grow, dtype=bool)])
            part.ids.extend([""] * grow)
        row = part.size
        part.size += 1
        part.row_of[fact_id] = row
        part.ids[row] = fact_id
    part.matrix[row] = vec
    part.live[row] = True


def _remove(part: _Partition, fact_id: str) -> None:
    row = part.row_of.pop(fact_id, None)
    if row is not None:
        part.live[row] = False
        part.matrix[row] = 0.0


def _compact_if_sparse(part: _Partition) -> None:
    """Drop dead rows once they outnumber the live ones (bounded memory)."""
    if part.size - part.count <= max(1024, part.count):
        return
    keep = [i for i in range(part.size) if part.live[i]]
    part.matrix = part.matrix[keep].copy()
    part.ids = [part.ids[i] for i in keep]
    part.row_of = {fid: i for i, fid in enumerate(part.ids)}
    part.live = np.ones(len(keep), dtype=bool)
    part.size = len(keep)


_SHARED: "weakref.WeakKeyDictionary[object, dict[int, CanonicalVectorIndex]]" = (
    weakref.WeakKeyDictionary())
_SHARED_LOCK = threading.Lock()


def shared_index(db, dimension: int) -> CanonicalVectorIndex:
    """One in-memory index per database and dimension in this process.

    Recall and consolidation read the same vectors: two instances would hold
    the matrix twice and each pay the first full read.
    """
    with _SHARED_LOCK:
        per_db = _SHARED.setdefault(db, {})
        index = per_db.get(int(dimension))
        if index is None:
            index = per_db[int(dimension)] = CanonicalVectorIndex(db, dimension)
        return index


def candidate_vector_source(db, vector_store, dimension: int, index=None):
    """The vector candidate source the retrieval channels read.

    sqlite-vec when it loaded; otherwise the exact in-memory index, so a
    platform without loadable SQLite extensions searches by meaning in
    milliseconds instead of re-reading every fact on every recall.
    """
    if vector_store is not None and getattr(vector_store, "available", False):
        return vector_store
    index = index if index is not None else shared_index(db, dimension)
    return index if index.available else vector_store


__all__ = ["CanonicalVectorIndex", "candidate_vector_source", "shared_index"]
