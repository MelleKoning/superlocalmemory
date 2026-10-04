# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""A fixed, seeded cross-scope store for the #146 / #147 retrieval tests.

A requester owns ``L*`` facts.  Another profile owns:
  * ``G*``  scope='global'                       -- visible with include_global
  * ``S*``  scope='shared', shared with requester -- visible with include_shared
  * ``D*``  scope='shared', shared with someone else -- never visible
  * ``P*``  scope='personal'                      -- never visible

Embeddings are clustered (so cosine has real structure, not all ~0), every
fact has random graph edges, and everything is seeded so a run is repeatable.
Ported and scaled down from .backup/4.1.20/analysis/bench_146_147.py.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager

REQ = "requester"
OWN = "owner"
DIM = 768
N_CENTERS = 24


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


@dataclass
class CrossScopeStore:
    db: DatabaseManager
    embs: dict[str, np.ndarray]
    queries: list[np.ndarray] = field(default_factory=list)

    def ids(self, prefixes: str) -> list[str]:
        return sorted(f for f in self.embs if f[0] in prefixes)


def build_store(
    path: Path,
    *,
    n_local: int = 120,
    n_global: int = 300,
    n_shared: int = 60,
    n_denied: int = 40,
    degree: int = 6,
    json_text: bool = False,
    seed: int = 42,
) -> CrossScopeStore:
    """Build the store. ``json_text=True`` writes legacy JSON-text embeddings."""
    rng = np.random.default_rng(seed)
    rnd = random.Random(seed)
    centers = [_unit(rng.normal(size=DIM)) for _ in range(N_CENTERS)]

    def near(c: int, noise: float = 0.9) -> np.ndarray:
        return _unit(centers[c] + noise * _unit(rng.normal(size=DIM)))

    rows: list[tuple[str, str, str, str | None]] = []
    rows += [(f"L{i:05d}", REQ, "personal", None) for i in range(n_local)]
    rows += [(f"G{i:05d}", OWN, "global", None) for i in range(n_global)]
    rows += [(f"S{i:05d}", OWN, "shared", json.dumps([REQ])) for i in range(n_shared)]
    rows += [(f"D{i:05d}", OWN, "shared", json.dumps(["someone_else"]))
             for i in range(n_denied)]
    rows += [(f"P{i:05d}", OWN, "personal", None) for i in range(n_denied)]

    db = DatabaseManager(path)
    embs: dict[str, np.ndarray] = {}
    with db.raw_connection() as conn:
        schema.create_all_tables(conn)
        for p in (REQ, OWN):
            conn.execute(
                "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?,?)", (p, p),
            )
        for i, (fid, prof, scope, shared_with) in enumerate(rows):
            e = near(rnd.randrange(N_CENTERS))
            embs[fid] = e
            stored = json.dumps([float(x) for x in e]) if json_text else e.tobytes()
            fisher = rng.normal(size=DIM).astype(np.float32).tobytes()
            conn.execute(
                "INSERT INTO memories (memory_id, profile_id, scope, shared_with, content)"
                " VALUES (?,?,?,?,?)",
                (f"m_{fid}", prof, scope, shared_with, fid),
            )
            # Distinct created_at per row: ORDER BY created_at is then total.
            conn.execute(
                "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, scope,"
                " shared_with, content, fact_type, confidence, importance,"
                " evidence_count, access_count, embedding, fisher_mean,"
                " fisher_variance, created_at) VALUES (?,?,?,?,?,?, 'semantic',"
                " 0.9, 0.5, 1, 0, ?, ?, ?, datetime('2026-01-01', ?))",
                (fid, f"m_{fid}", prof, scope, shared_with, f"content {fid}",
                 stored, fisher, fisher, f"+{i} seconds"),
            )
        ids = [r[0] for r in rows]
        owner = {r[0]: (r[1], r[2]) for r in rows}
        k = 0
        for fid in ids:
            for _ in range(degree):
                tgt = rnd.choice(ids)
                if tgt == fid:
                    continue
                prof, scope = owner[fid]
                conn.execute(
                    "INSERT INTO graph_edges (edge_id, profile_id, scope, source_id,"
                    " target_id, edge_type, weight, created_at) VALUES"
                    " (?,?,?,?,?,'semantic',?,'2026-01-01')",
                    (f"e{k}", prof, scope, fid, tgt, rnd.random()),
                )
                k += 1
    queries = [near(c, 0.6) for c in range(0, N_CENTERS, 2)]
    return CrossScopeStore(db=db, embs=embs, queries=queries)


def cosine(q: np.ndarray, v: np.ndarray) -> float:
    """The channel's own per-row cosine arithmetic (kept per row on purpose)."""
    q32 = np.asarray(q, dtype=np.float32)
    return float(np.dot(q32, v) / (float(np.linalg.norm(q32)) * float(np.linalg.norm(v))))


class PartitionedVS:
    """Owner-partitioned KNN with the real vec0 score convention max(0, cos).

    ``ids`` decides which facts the index holds.  The per-row arithmetic is the
    channel's own, so an oracle built on it is comparable bit for bit.
    """

    available = True

    def __init__(self, embs: dict[str, np.ndarray], ids: list[str]) -> None:
        self._embs = embs
        self._ids = list(ids)
        self.calls = 0

    def search(self, q, top_k: int = 10, profile_id: str | None = None):
        self.calls += 1
        out = [(f, max(0.0, cosine(q, self._embs[f]))) for f in self._ids]
        out.sort(key=lambda x: (-x[1], x[0]))
        return out[:top_k]
