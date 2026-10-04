# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A data export carries every code-graph row, and the dashboard export
carries the code graph and learning signals at all."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from superlocalmemory.compliance.gdpr import GDPRCompliance
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager

_OVER_OLD_CAP = 10_000 + 2_345


def _memory_db(root: Path) -> DatabaseManager:
    mgr = DatabaseManager(root / "memory.db")
    mgr.initialize(real_schema)
    mgr.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('alice', 'Alice')")
    return mgr


def _code_graph(root: Path, n: int) -> None:
    conn = sqlite3.connect(root / "code_graph.db")
    conn.execute("CREATE TABLE graph_nodes (node_id TEXT PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE graph_edges (src TEXT, dst TEXT)")
    conn.executemany("INSERT INTO graph_nodes VALUES (?, ?)",
                     [(f"n{i}", f"symbol_{i}") for i in range(n)])
    conn.executemany("INSERT INTO graph_edges VALUES (?, ?)",
                     [(f"n{i}", f"n{i + 1}") for i in range(n)])
    conn.commit()
    conn.close()


def test_code_graph_export_holds_every_row_past_the_old_cap(tmp_path: Path) -> None:
    mgr = _memory_db(tmp_path)
    _code_graph(tmp_path, _OVER_OLD_CAP)
    graph = GDPRCompliance(mgr, data_root=tmp_path).export_profile_data("alice")["code_graph"]
    assert len(graph["graph_nodes"]) == _OVER_OLD_CAP
    assert len(graph["graph_edges"]) == _OVER_OLD_CAP
    assert len({r["node_id"] for r in graph["graph_nodes"]}) == _OVER_OLD_CAP


def test_export_without_explicit_root_still_reads_the_side_databases(tmp_path: Path) -> None:
    """The dashboard builds the exporter from the engine's database alone. The
    files next to that database belong to the same installation and must be in
    the export, as they are in the erasure."""
    mgr = _memory_db(tmp_path)
    _code_graph(tmp_path, 3)
    with sqlite3.connect(tmp_path / "learning.db") as conn:
        conn.execute("CREATE TABLE zz_signals (profile_id TEXT, n INTEGER)")
        conn.executemany("INSERT INTO zz_signals VALUES (?, ?)",
                         [("alice", 1), ("alice", 2), ("bob", 3)])
    data = GDPRCompliance(mgr).export_profile_data("alice")
    assert len(data["code_graph"]["graph_nodes"]) == 3
    assert [r["n"] for r in data["learning_signals"]["zz_signals"]] == [1, 2]
