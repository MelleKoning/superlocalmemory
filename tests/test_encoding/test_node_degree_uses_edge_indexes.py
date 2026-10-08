"""A node's degree is counted from the edge indexes, not by walking the profile.

The hub filter asks for the degree of many neighbour facts on every save. The
"source = id OR target = id" form made SQLite read every edge of the profile
per call (33 ms warm, 0.8 s cold on a 486k-edge store). The count must stay
exactly the same: out-edges + in-edges, a self-loop counted once, other
profiles never counted.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from superlocalmemory.encoding.graph_builder import GraphBuilder
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.schema import create_all_tables


def _store(tmp_path: Path) -> DatabaseManager:
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(path))
    create_all_tables(conn)
    conn.commit()
    conn.close()
    db = DatabaseManager(path)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('other', 'other')")
    edges = [
        ("e1", "default", "a", "b"), ("e2", "default", "c", "a"), ("e3", "default", "a", "a"),
        ("e4", "default", "b", "c"), ("e5", "other", "a", "b"), ("e6", "default", "a", "d"),
    ]
    for edge_id, profile, src, tgt in edges:
        db.execute("INSERT INTO graph_edges "
                   "(edge_id, profile_id, source_id, target_id, edge_type, weight) "
                   "VALUES (?, ?, ?, ?, 'entity', 1.0)", (edge_id, profile, src, tgt))
    return db


def test_degree_counts_are_exact(tmp_path: Path) -> None:
    builder = GraphBuilder(_store(tmp_path))
    cache: dict[str, int] = {}
    assert builder._node_degree("a", "default", cache) == 4  # e1, e2, e3 (self-loop once), e6
    assert builder._node_degree("b", "default", cache) == 2
    assert builder._node_degree("d", "default", cache) == 1
    assert builder._node_degree("zz", "default", cache) == 0
    assert builder._node_degree("a", "other", {}) == 1


def test_degree_query_probes_by_source_and_target(tmp_path: Path) -> None:
    db = _store(tmp_path)
    seen: list[tuple[str, tuple]] = []
    real = db.execute

    def spy(sql, params=()):
        seen.append((sql, tuple(params)))
        return real(sql, params)

    db.execute = spy  # type: ignore[method-assign]
    GraphBuilder(db)._node_degree("a", "default", {})
    db.execute = real  # type: ignore[method-assign]
    sql, params = seen[-1]
    plan = [str(dict(r)["detail"]) for r in db.execute("EXPLAIN QUERY PLAN " + sql, params)]
    searches = [step for step in plan if "graph_edges" in step]
    assert searches, plan
    # Every probe pins the node id; none walks the whole profile.
    assert all("source_id=?" in step or "target_id=?" in step for step in searches), plan


def test_a_facts_edges_are_written_on_one_connection(tmp_path: Path) -> None:
    """All edges of one new fact commit together, on one connection.

    Per-edge commits opened, configured and closed a connection for every edge
    (tens per saved fact) while holding the store's write lock. The rows must be
    the ones edge-by-edge writing produces.
    """
    from superlocalmemory.storage.models import AtomicFact, EdgeType, GraphEdge, MemoryRecord

    db = _store(tmp_path)
    for mid in ["m_n"] + [f"m_o{i}" for i in range(6)]:
        db.store_memory(MemoryRecord(memory_id=mid, content="parent"))
    for i in range(6):
        db.store_fact(AtomicFact(fact_id=f"o{i}", memory_id=f"m_o{i}", content=f"other {i}"))
    new = AtomicFact(fact_id="n", memory_id="m_n", content="new")
    db.store_fact(new)
    builder = GraphBuilder(db)
    planned = [GraphEdge(profile_id="default", source_id="n", target_id=f"o{i}",
                         edge_type=EdgeType.ENTITY, weight=1.0) for i in range(6)]
    builder._build_entity_edges = lambda *_a, **_k: list(planned)  # type: ignore[method-assign]
    opened: list[int] = []
    real_connect = db._connect

    def counting_connect(*a, **k):
        opened.append(1)
        return real_connect(*a, **k)

    db._connect = counting_connect  # type: ignore[method-assign]
    try:
        edges = builder.build_edges(new, "default")
    finally:
        db._connect = real_connect  # type: ignore[method-assign]
    assert len(edges) >= 6
    assert len(opened) == 1, f"{len(opened)} connections for {len(edges)} edges"
    stored = {(dict(r)["source_id"], dict(r)["target_id"]) for r in db.execute(
        "SELECT source_id, target_id FROM graph_edges "
        "WHERE source_id = 'n' AND edge_type = 'entity'")}
    assert stored == {("n", f"o{i}") for i in range(6)}
