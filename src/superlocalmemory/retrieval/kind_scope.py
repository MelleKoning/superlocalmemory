# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Kind-filtered recall searches inside the kind, not only filters after it.

WHY
---
``kind`` is a hard filter (``retrieval/facets``), applied to the fused
candidate pool. The pool is built without knowing the kind, so a memory of the
asked-for kind that the unfiltered search ranked below the pool was never a
candidate at all, and the filtered answer came back short or EMPTY although the
memory was there -- even for its exact words once enrichment had added
near-identical facts of other kinds. Measured on a copy of a 21,739-fact store:
kind=decision recalls returned 0 results for 3 of 10 everyday questions.

WHAT THIS DOES
--------------
Before fusion, two extra searches run restricted to the facts whose DISPLAYED
kind (``storage.memory_kinds.kind_fields`` -- the one serializer every surface
uses) is the one asked for:

* meaning: exact cosine neighbours among that kind's vectors
  (``CanonicalVectorIndex.search_within``), scored by the semantic channel's
  own rescoring, so their scores are on the semantic channel's scale;
* words: the BM25 channel's own FTS query, restricted in SQL to rows that can
  display as that kind, then to the exact members.

Their results are merged into the semantic and BM25 channel lists (same ids
keep their best score) and fusion proceeds as before. The post-fusion kind
filter still runs and still decides what is shown: this only makes sure the
memories it is looking for are in front of it.

Which facts are of which kind is held in memory per profile and kept current
by the ``fact_search_changes`` log (``retrieval/change_tracked``), so a kind
written, corrected or backfilled by any process counts from the next recall.
Personal scope only: an opted-in shared or global fact of another profile is
still found by the ordinary channels and still filtered after fusion.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from superlocalmemory.retrieval.change_tracked import ChangeTrackedCache
from superlocalmemory.storage.memory_kinds import (
    ALIASES,
    LEGACY_TO_KIND,
    MemoryKind,
    kind_fields,
)

logger = logging.getLogger(__name__)

_CHUNK = 800
_KIND_READ = ("memory_kind", "memory_kind_source", "memory_kind_confidence", "fact_type")


@dataclass
class _Kinds:
    """One profile: each visible fact's raw kind columns, plus derived sets."""

    raw: dict[str, tuple] = field(default_factory=dict)
    by_threshold: dict[float, dict[str, frozenset[str]]] = field(default_factory=dict)


class KindMembership(ChangeTrackedCache[_Kinds]):
    """Which visible facts of a profile display as which kind."""

    WHAT = "k"

    def ids_of(self, profile_id: str, kind: str, threshold: float) -> frozenset[str] | None:
        """Displayed-``kind`` fact ids, or None when this cannot be known."""
        if not self.available:
            return None
        with self._lock:
            part = self._fresh(profile_id)
            sets = part.by_threshold.get(threshold)
            if sets is None:
                grouped: dict[str, set[str]] = {}
                for fid, row in part.raw.items():
                    shown = kind_fields(dict(zip(_KIND_READ, row)),
                                        display_min_confidence=threshold)["memory_kind"]
                    if shown:
                        grouped.setdefault(str(shown), set()).add(fid)
                sets = {k: frozenset(v) for k, v in grouped.items()}
                part.by_threshold[threshold] = sets
            return sets.get(kind, frozenset())

    def _columns(self) -> str:
        present = {str(dict(r).get("name")) for r in
                   self._db.execute("PRAGMA table_info(atomic_facts)")}
        return ", ".join(c if c in present else f"NULL AS {c}" for c in _KIND_READ)

    def _build(self, profile_id: str) -> _Kinds:
        t0 = time.monotonic()
        rows = self._db.execute(
            f"SELECT fact_id, {self._columns()} FROM atomic_facts "
            f"WHERE profile_id = ?{self._db.visible_fact_clause()}",
            (profile_id,),
        )
        part = _Kinds()
        for r in rows:
            d = dict(r)
            part.raw[str(d["fact_id"])] = tuple(d[c] for c in _KIND_READ)
        logger.info("kind index: %d facts for profile %s in %.0f ms",
                    len(part.raw), profile_id, (time.monotonic() - t0) * 1000.0)
        return part

    def _apply(self, fact_ids: list[str]) -> None:
        current: dict[str, tuple[str, tuple]] = {}
        cols = self._columns()
        visible = self._db.visible_fact_clause()
        for start in range(0, len(fact_ids), _CHUNK):
            batch = fact_ids[start:start + _CHUNK]
            for r in self._db.execute(
                f"SELECT fact_id, profile_id, {cols} FROM atomic_facts "
                f"WHERE fact_id IN ({','.join('?' * len(batch))}){visible}",
                tuple(batch),
            ):
                d = dict(r)
                current[str(d["fact_id"])] = (
                    str(d["profile_id"]), tuple(d[c] for c in _KIND_READ))
        for pid, part in self._parts.items():
            for fid in fact_ids:
                owner, row = current.get(fid, (None, None))
                if owner == pid:
                    part.raw[fid] = row
                else:
                    part.raw.pop(fid, None)
            part.by_threshold.clear()


def kind_superset_clause(kind: str, prefix: str = "af") -> tuple[str, list[str]]:
    """SQL matching every row that CAN display as ``kind`` (a superset).

    A row displays as K only if its own kind parses to K or its legacy
    ``fact_type`` maps to K, so this never excludes a member; callers then keep
    the exact members. Spellings are normalised the way ``parse_kind`` does.
    """
    target = MemoryKind(kind)
    spellings = [target.value, *(a for a, k in ALIASES.items() if k is target)]
    legacy = [t for t, k in LEGACY_TO_KIND.items() if k is target]
    t = f"{prefix}." if prefix else ""
    parts = [f"replace(lower(trim({t}memory_kind)), '_', '-') IN "
             f"({','.join('?' * len(spellings))})"]
    params = list(spellings)
    if legacy:
        parts.append(f"lower(trim({t}fact_type)) IN ({','.join('?' * len(legacy))})")
        params.extend(legacy)
    return " AND (" + " OR ".join(parts) + ")", params


class _Within:
    """A vector source restricted to one set of ids (for the semantic rescoring)."""

    available = True

    def __init__(self, index: Any, allowed: frozenset[str]) -> None:
        self._index, self._allowed = index, allowed

    def search(self, query_embedding, top_k: int = 30, profile_id: str | None = None):
        return self._index.search_within(query_embedding, top_k, profile_id, self._allowed)


def _merge(base: list[tuple[str, float]], extra: list[tuple[str, float]]):
    best: dict[str, float] = {}
    for fid, score in (*base, *extra):
        best[fid] = max(score, best.get(fid, float("-inf")))
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))


def supplement(
    engine: Any, ch_results: dict[str, list[tuple[str, float]]], *, query: str,
    query_embedding: list[float] | None, profile_id: str, kind: str,
    stage_ms: dict[str, float] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """``ch_results`` with in-kind semantic and BM25 hits merged in.

    Never raises: a supplement that cannot run leaves the ordinary channels'
    answer exactly as it was (the kind filter after fusion still applies) and
    says so in the log.
    """
    t0 = time.monotonic()
    out = dict(ch_results)
    try:
        members = engine._kind_membership.ids_of(
            profile_id, kind, engine._display_min_confidence)
        if members is None:
            logger.warning("kind-scoped search unavailable; kind=%s is filtered "
                           "after fusion only", kind)
            return out
        if not members:
            return out
        semantic = engine._semantic
        index = engine._kind_vectors
        if semantic is not None and query_embedding is not None and index is not None:
            import numpy as np

            q_vec = np.array(query_embedding, dtype=np.float32)
            hits = semantic._search_via_vector_store(
                query_embedding, q_vec, profile_id, engine._config.semantic_top_k,
                source=_Within(index, members),
            )
            if hits:
                out["semantic"] = _merge(out.get("semantic", []), hits)
        bm25 = engine._bm25
        if bm25 is not None and hasattr(bm25, "_fts5_search"):
            k = engine._config.bm25_top_k
            hits = bm25._fts5_search(query, profile_id, k * 2,
                                     extra_where=kind_superset_clause(kind))
            hits = [h for h in hits if h[0] in members][:k]
            if hits:
                out["bm25"] = _merge(out.get("bm25", []), hits)
    except Exception as exc:  # noqa: BLE001 -- fall back to the ordinary answer
        logger.warning("kind-scoped search failed (%s); kind=%s is filtered after "
                       "fusion only", type(exc).__name__, kind)
        return dict(ch_results)
    finally:
        if stage_ms is not None:
            stage_ms["kind_scope"] = round((time.monotonic() - t0) * 1000.0, 1)
    return out


__all__ = ["KindMembership", "attach", "kind_superset_clause", "supplement", "warm"]


def attach(engine: Any, db: Any, vectors: Any, dimension: int) -> None:
    """Give a RetrievalEngine what kind-scoped recall reads.

    The in-memory vector index is the channels' own source when sqlite-vec is
    unavailable; with sqlite-vec a second, kind-scoping-only copy is built on
    first use (or at daemon start, ``warm``) because vec0 cannot search inside
    an arbitrary id set.
    """
    from superlocalmemory.retrieval.canonical_vector_index import CanonicalVectorIndex

    index = vectors if isinstance(vectors, CanonicalVectorIndex) else CanonicalVectorIndex(
        db, dimension)
    engine._kind_vectors = index if index.available else None
    membership = KindMembership(db)
    engine._kind_membership = membership if membership.available else None


def warm(engine: Any, profile_id: str) -> None:
    """Build both in-memory indexes for a profile before anyone recalls."""
    retrieval = getattr(engine, "_retrieval_engine", engine)
    for name in ("_kind_vectors", "_kind_membership"):
        cache = getattr(retrieval, name, None)
        try:
            if name == "_kind_vectors" and cache is not None:
                cache.warm(profile_id)
            elif cache is not None:
                cache.ids_of(profile_id, MemoryKind.DECISION.value,
                             retrieval._display_min_confidence)
        except Exception as exc:  # noqa: BLE001 -- a recall will build it instead
            logger.warning("warming %s failed (%s)", name, type(exc).__name__)
