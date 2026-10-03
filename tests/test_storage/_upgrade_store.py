# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Stores shaped like a user's, for the upgrade / restore / downgrade tests.

``current_store`` is what a 4.1.19 engine leaves: the real schema with every
migration recorded. ``as_4118`` takes M052 back out of it, which is exactly a
4.1.18 store (every migration up to M051, stamp 51, no kind columns).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

KIND_COLUMNS = (
    "memory_kind", "memory_kind_source", "memory_kind_confidence",
    "memory_kind_recipe", "memory_kind_at",
)


def _conn(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(path), isolation_level=None)


def current_store(root: Path) -> tuple[Path, Path]:
    """(learning_db, memory_db) fully migrated; the copy the runner took is removed."""
    from superlocalmemory.storage import migration_runner as mr
    from superlocalmemory.storage import schema

    root.mkdir(parents=True, exist_ok=True)
    memory_db, learning_db = root / "memory.db", root / "learning.db"
    with closing(_conn(memory_db)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        schema.create_all_tables(conn)
    assert mr.apply_all(learning_db, memory_db)["failed"] == []
    assert mr.apply_deferred(learning_db, memory_db)["failed"] == []
    clear_snapshots(root)
    return learning_db, memory_db


def clear_snapshots(root: Path) -> None:
    """Remove the copies this test's own setup took (explicit paths only)."""
    snaps = root / "pre-migration-snapshots"
    if snaps.is_dir():
        for child in sorted(snaps.iterdir()):
            if child.is_file() and not child.is_symlink():
                child.unlink()


def as_4118(learning_db: Path, memory_db: Path) -> None:
    """Undo M052 so the store is exactly what 4.1.18 left behind."""
    with closing(_conn(memory_db)) as conn:
        conn.execute("DROP INDEX IF EXISTS idx_facts_memory_kind")
        conn.execute("DROP TABLE IF EXISTS memory_kind_runs")
        conn.execute("DROP TABLE IF EXISTS memory_kind_history")
        for name in KIND_COLUMNS:
            conn.execute(f"ALTER TABLE atomic_facts DROP COLUMN {name}")
        conn.execute("DELETE FROM migration_log WHERE name='M052_memory_kinds'")
    for db in (learning_db, memory_db):
        with closing(_conn(db)) as conn:
            conn.execute("UPDATE slm_schema_version SET version=51 WHERE id=1")


def add_profile(memory_db: Path, profile_id: str) -> None:
    with closing(_conn(memory_db)) as conn:
        conn.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
                     (profile_id, profile_id.title()))


def add_memory(memory_db: Path, memory_id: str, facts: list[str], *,
               profile: str = "default", fact_ids: list[str] | None = None) -> list[str]:
    ids = fact_ids or [f"{memory_id}-f{i}" for i in range(len(facts))]
    with closing(_conn(memory_db)) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "INSERT INTO memories (memory_id, profile_id, content, session_id, metadata_json) "
            "VALUES (?, ?, ?, 'sess-1', ?)",
            (memory_id, profile, " ".join(facts), json.dumps({"source": "test"})))
        for fid, text in zip(ids, facts):
            conn.execute(
                "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
                "VALUES (?, ?, ?, ?)", (fid, memory_id, profile, text))
    return ids


def delete_fact(memory_db: Path, fact_id: str) -> None:
    with closing(_conn(memory_db)) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("DELETE FROM atomic_facts WHERE fact_id=?", (fact_id,))


def delete_memory(memory_db: Path, memory_id: str) -> None:
    with closing(_conn(memory_db)) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))


def confirm_kind(memory_db: Path, fact_id: str, kind: str, source: str = "user") -> None:
    from superlocalmemory.storage.memory_kinds import COARSE, MemoryKind

    with closing(_conn(memory_db)) as conn:
        conn.execute(
            "UPDATE atomic_facts SET memory_kind=?, memory_kind_source=?, "
            "memory_kind_at='2026-10-03T00:00:00Z', fact_type=? WHERE fact_id=?",
            (kind, source, COARSE[MemoryKind(kind)], fact_id))


def record_erasure(memory_db: Path, *, profile_id: str, subject_type: str,
                   subject_id: str, fact_ids: list[str], memory_ids: dict[str, str]) -> str:
    """What a completed erasure leaves in the live store: receipt + tombstones."""
    erasure_id = uuid.uuid4().hex
    with closing(_conn(memory_db)) as conn:
        conn.execute(
            "INSERT INTO erasure_receipts (erasure_id, profile_id, subject_type, subject_id, "
            "requested_by, fact_count, state, all_erased, owner_evidence_json, audit_hash, "
            "requested_at, completed_at) VALUES (?, ?, ?, ?, 'owner', ?, 'COMPLETE', 1, '[]', "
            "'h', 1.0, 2.0)",
            (erasure_id, profile_id, subject_type, subject_id, len(fact_ids)))
        for fid in fact_ids:
            conn.execute(
                "INSERT INTO projection_tombstones (profile_id, fact_id, erasure_id, memory_id, "
                "created_at) VALUES (?, ?, ?, ?, 2.0)",
                (profile_id, fid, erasure_id, memory_ids.get(fid)))
    return erasure_id


def fact_ids(memory_db: Path, profile: str | None = None) -> set[str]:
    with closing(sqlite3.connect(f"file:{memory_db}?mode=ro", uri=True)) as conn:
        if profile is None:
            return {r[0] for r in conn.execute("SELECT fact_id FROM atomic_facts")}
        return {r[0] for r in conn.execute(
            "SELECT fact_id FROM atomic_facts WHERE profile_id=?", (profile,))}


def memory_ids(memory_db: Path) -> set[str]:
    with closing(sqlite3.connect(f"file:{memory_db}?mode=ro", uri=True)) as conn:
        return {r[0] for r in conn.execute("SELECT memory_id FROM memories")}


def columns(memory_db: Path, table: str = "atomic_facts") -> set[str]:
    with closing(sqlite3.connect(f"file:{memory_db}?mode=ro", uri=True)) as conn:
        return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")}


def logical_digest(memory_db: Path) -> str:
    """A digest of what the store holds, independent of page layout."""
    digest = hashlib.sha256()
    with closing(sqlite3.connect(f"file:{memory_db}?mode=ro", uri=True)) as conn:
        for table in ("memories", "atomic_facts"):
            for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid"):
                digest.update(repr(row).encode())
    return digest.hexdigest()


def integrity(memory_db: Path) -> str:
    with closing(sqlite3.connect(str(memory_db))) as conn:
        return conn.execute("PRAGMA integrity_check").fetchone()[0]


def file_digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
