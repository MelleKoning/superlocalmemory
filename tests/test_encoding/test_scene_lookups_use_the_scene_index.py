"""Scene assignment looks scene members up by scene, not by walking the profile.

Every enriched fact asks which recent scenes still have a live member, and
loads one anchor embedding per candidate scene. Both queries joined
``scene_fact_members`` on (scene_id, profile_id); SQLite chose the
(profile_id, fact_id, scene_id) index and so read every membership row of the
profile for each scene: 6.1 s per call on an 11.6k-scene / 20k-member store
(the background enrichment of ONE memory spent most of its time there, holding
a core while recalls ran), 28 ms when the lookup goes by scene_id. Same rows.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from superlocalmemory.encoding.scene_builder import SceneBuilder
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.schema import create_all_tables

_PROFILE = "default"


def _store(tmp_path: Path) -> DatabaseManager:
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(path))
    create_all_tables(conn)
    conn.commit()
    conn.close()
    db = DatabaseManager(path)
    db.execute("INSERT OR IGNORE INTO memories (memory_id, profile_id, content) VALUES ('m1', ?, 'x')",
               (_PROFILE,))
    for i in range(12):
        db.execute("INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, embedding) "
                   "VALUES (?, 'm1', ?, ?, ?)", (f"f{i}", _PROFILE, f"fact {i}", "[0.1, 0.2]"))
    for s in range(4):
        members = [f"f{s * 3 + k}" for k in range(3)]
        db.execute("INSERT INTO memory_scenes (scene_id, profile_id, theme, fact_ids_json, last_updated) "
                   "VALUES (?, ?, ?, ?, ?)",
                   (f"s{s}", _PROFILE, f"theme {s}", str(members).replace("'", '"'), f"2026-01-0{s + 1}"))
    return db


def _captured_sql(db: DatabaseManager, call) -> list[tuple[str, tuple]]:
    seen: list[tuple[str, tuple]] = []
    real = db.execute

    def spy(sql, params=()):
        seen.append((sql, tuple(params)))
        return real(sql, params)

    db.execute = spy  # type: ignore[method-assign]
    try:
        call()
    finally:
        db.execute = real  # type: ignore[method-assign]
    return [(s, p) for s, p in seen if "scene_fact_members" in s]


def _member_lookups(db: DatabaseManager, sql: str, params: tuple) -> list[str]:
    plan = [str(dict(r)["detail"]) for r in db.execute("EXPLAIN QUERY PLAN " + sql, params)]
    return [step for step in plan if "member" in step]


def test_recent_scene_candidates_probe_members_by_scene(tmp_path: Path) -> None:
    db = _store(tmp_path)
    builder = SceneBuilder(db)
    queries = _captured_sql(db, lambda: builder._get_assignment_scenes(_PROFILE, [0.1, 0.2]))
    assert queries, "the candidate query must read scene membership"
    steps = _member_lookups(db, *queries[0])
    assert steps and all("(scene_id=?)" in step for step in steps), steps


def test_live_scene_anchors_probe_members_by_scene(tmp_path: Path) -> None:
    db = _store(tmp_path)
    builder = SceneBuilder(db)
    queries = _captured_sql(db, lambda: builder._load_live_scene_embeddings(_PROFILE, ("s0", "s1", "s2", "s3")))
    assert queries
    steps = _member_lookups(db, *queries[0])
    assert steps and all("(scene_id=?)" in step for step in steps), steps


def test_same_scenes_and_anchors(tmp_path: Path) -> None:
    db = _store(tmp_path)
    builder = SceneBuilder(db)
    scenes = builder._get_assignment_scenes(_PROFILE, [0.1, 0.2])
    assert [s.scene_id for s in scenes] == ["s3", "s2", "s1", "s0"]  # newest first
    anchors = builder._load_live_scene_embeddings(_PROFILE, ("s0", "s2", "missing"))
    assert set(anchors) == {"s0", "s2"}
