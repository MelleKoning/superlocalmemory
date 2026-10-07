# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tag-filtered recall searches inside the tag set, not only filters after it.

WHY
---
``tags`` is a hard filter (``retrieval/facets``), applied to the fused
candidate pool. The pool is built without knowing about tags, so a memory
carrying the asked-for tag that the unfiltered search ranked below the pool
was never a candidate at all — the filtered answer comes back short or EMPTY
although the memory is there, exactly the defect ``retrieval/kind_scope.py``
fixed for ``kind`` (measured there: 3 of 10 everyday kind-filtered questions
returned nothing on a 21,739-fact store).

WHAT THIS DOES
--------------
Before fusion, two extra searches run restricted to the facts whose parent
memory's tags satisfy the requested labels/match:

* meaning: exact cosine neighbours among that set's vectors
  (``CanonicalVectorIndex.search_within`` — the same index ``kind_scope``
  attaches at daemon start, reused here as-is: it restricts to an arbitrary
  id set, nothing about it is kind-specific);
* words: the BM25 channel's own FTS query, then filtered in Python to the
  exact members (no SQL superset clause — tags are stored as a CSV string, a
  JSON-array string, or a real list, so there is no single SQL shape to
  narrow on the way kind's ``kind_superset_clause`` does for one column).

Their results are merged into the semantic and BM25 channel lists (same ids
keep their best score) and fusion proceeds as before. The post-fusion tag
filter (``retrieval/facets``) still runs and still decides what is shown:
this only makes sure the memories it is looking for are in front of it.

WHY MEMBERSHIP IS NOT CACHED (unlike ``kind_scope.KindMembership``)
--------------------------------------------------------------------
Tags live on ``memories.metadata_json``, a table ``storage.fact_search_changes``
does not watch — only ``atomic_facts`` column changes are logged. Extending
that trigger-based log to a second table, for one filter, was judged not
worth the risk in this lane (4.1.22 G05; no schema migration ships with this
gap either — see ``core.tag_identity``). Membership is instead read fresh on
every call that needs it. Measured on the real M5 store, read-only: a full
scan of the default profile's 12,775 tagged facts takes ~75 ms warm and
~1.7 s on the first cold scan — against a 3.0 s recall ceiling
(``slm-quality-budget.md``) that leaves headroom. Personal scope only, like
``kind_scope``: an opted-in shared or global fact of another profile is still
found by the ordinary channels and still filtered after fusion.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def tag_members(
    db: Any, profile_id: str, tags: tuple[str, ...], match: str,
) -> frozenset[str] | None:
    """Visible fact ids of this profile whose parent memory satisfies
    ``tags``/``match``, or None when membership could not be read (the
    caller then leaves the ordinary channels untouched — never raises)."""
    from superlocalmemory.core.tag_identity import tag_key, tag_keys

    wanted = frozenset(k for k in (tag_key(t) for t in tags) if k is not None)
    if not wanted:
        return frozenset()
    try:
        rows = db.execute(
            "SELECT f.fact_id AS fact_id, "
            "json_extract(m.metadata_json, '$.tags') AS tags "
            "FROM atomic_facts f JOIN memories m ON m.memory_id = f.memory_id "
            f"WHERE f.profile_id = ?{db.visible_fact_clause('f')}",
            (profile_id,),
        )
    except Exception as exc:  # noqa: BLE001 — membership unavailable, not an error
        logger.warning("tag membership scan failed (%s)", type(exc).__name__)
        return None
    keep: set[str] = set()
    for row in rows:
        d = dict(row)
        have = frozenset(tag_keys(d.get("tags")))
        if not have:
            continue
        ok = bool(wanted & have) if match == "any" else wanted <= have
        if ok:
            keep.add(str(d["fact_id"]))
    return frozenset(keep)


class _Within:
    """A vector source restricted to one set of ids (semantic rescoring).

    Small, deliberate duplicate of ``retrieval.kind_scope._Within``: the two
    modules stay independently readable and neither breaks if the other's
    private shape changes.
    """

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
    query_embedding: list[float] | None, profile_id: str, tags: tuple[str, ...],
    match: str, stage_ms: dict[str, float] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """``ch_results`` with in-tag-set semantic and BM25 hits merged in.

    Never raises: a supplement that cannot run leaves the ordinary channels'
    answer exactly as it was (the post-fusion tag filter still applies) and
    says so in the log.
    """
    t0 = time.monotonic()
    out = dict(ch_results)
    try:
        members = tag_members(engine._db, profile_id, tags, match)
        if members is None:
            logger.warning("tag-scoped search unavailable; tags=%r is filtered "
                           "after fusion only", tags)
            return out
        if not members:
            return out
        semantic = getattr(engine, "_semantic", None)
        # Reused as-is from kind_scope.attach(): a generic "cosine search
        # restricted to an arbitrary id set" index, not kind-specific.
        index = getattr(engine, "_kind_vectors", None)
        if semantic is not None and query_embedding is not None and index is not None:
            import numpy as np

            q_vec = np.array(query_embedding, dtype=np.float32)
            hits = semantic._search_via_vector_store(
                query_embedding, q_vec, profile_id, engine._config.semantic_top_k,
                source=_Within(index, members),
            )
            if hits:
                out["semantic"] = _merge(out.get("semantic", []), hits)
        bm25 = getattr(engine, "_bm25", None)
        if bm25 is not None and hasattr(bm25, "_fts5_search"):
            k = engine._config.bm25_top_k
            hits = bm25._fts5_search(query, profile_id, k * 4)
            hits = [h for h in hits if h[0] in members][:k]
            if hits:
                out["bm25"] = _merge(out.get("bm25", []), hits)
    except Exception as exc:  # noqa: BLE001 — fall back to the ordinary answer
        logger.warning("tag-scoped search failed (%s); tags=%r is filtered after "
                       "fusion only", type(exc).__name__, tags)
        return dict(ch_results)
    finally:
        if stage_ms is not None:
            stage_ms["tag_scope"] = round((time.monotonic() - t0) * 1000.0, 1)
    return out


__all__ = ["supplement", "tag_members"]
