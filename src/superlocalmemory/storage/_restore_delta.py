# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What changed since a safety copy was taken, and carrying it across a restore.

Restoring a copy taken before an update must not cost the user anything they
did after it. Before the live store is replaced, its difference from the copy
is written out as plain JSON lines (``restore-delta/<ts>/``):

  memories.jsonl    memories added since the copy   -> added back after the restore
  deleted.jsonl     facts / memories deleted since  -> deleted again in the copy
  kind_edits.jsonl  kinds a person confirmed        -> applied again after the restore
  erasures.jsonl    erasure receipts and tombstones -> erased again in the copy
  profiles.jsonl    profiles created since          -> created in the copy
  meta.json         the live profile list and counts

Deletions and erasures are applied to a STAGING copy of the snapshot, never to
the live store, so erased data never reappears even for a moment and the live
store is replaced in one transaction by an already-correct copy.

Every read of the live store goes through ``mode=ro`` and every read of the
copy through ``immutable=1``: computing a preview changes no byte of either.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

from superlocalmemory.storage._durable_json import (
    read_json, read_jsonl, write_json_atomic, write_jsonl_atomic,
)

logger = logging.getLogger(__name__)

_CONFIRMED = ("user", "caller")
_KIND_COLS = ("memory_kind", "memory_kind_source", "memory_kind_confidence",
              "memory_kind_recipe", "memory_kind_at")
#: Ledgers that record an erasure; never deleted by an erasure replay.
_LEDGERS = frozenset({"erasure_receipts", "projection_tombstones", "profiles"})


def open_compare(memory_db: Path, snapshot: Path) -> sqlite3.Connection:
    """Live store as ``main`` (read-only) with the copy attached as ``snap``."""
    conn = sqlite3.connect(f"{Path(memory_db).absolute().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("ATTACH DATABASE ? AS snap",
                 (f"{Path(snapshot).absolute().as_uri()}?mode=ro&immutable=1",))
    return conn


def _tables(conn: sqlite3.Connection, schema: str = "main") -> set[str]:
    return {r[0] for r in conn.execute(
        f"SELECT name FROM {schema}.sqlite_master WHERE type='table'")}


def _cols(conn: sqlite3.Connection, table: str, schema: str = "main") -> list[str]:
    try:
        return [str(r[1]) for r in conn.execute(f"PRAGMA {schema}.table_info('{table}')")]
    except sqlite3.Error:
        return []


def _both(conn: sqlite3.Connection, table: str) -> bool:
    return table in _tables(conn, "main") and table in _tables(conn, "snap")


def _count(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0] or 0)


_ADDED_MEMORIES = ("FROM main.memories m WHERE NOT EXISTS "
                   "(SELECT 1 FROM snap.memories s WHERE s.memory_id = m.memory_id)")
_DELETED_FACTS = ("FROM snap.atomic_facts s WHERE NOT EXISTS "
                  "(SELECT 1 FROM main.atomic_facts f WHERE f.fact_id = s.fact_id)")
_DELETED_MEMORIES = (
    "FROM snap.memories s WHERE NOT EXISTS "
    "(SELECT 1 FROM main.memories m WHERE m.memory_id = s.memory_id) "
    "AND NOT EXISTS (SELECT 1 FROM snap.atomic_facts sf JOIN main.atomic_facts lf "
    "ON lf.fact_id = sf.fact_id WHERE sf.memory_id = s.memory_id)")


def _kind_edit_sql(conn: sqlite3.Connection) -> str | None:
    if "memory_kind_source" not in _cols(conn, "atomic_facts"):
        return None
    return ("FROM main.atomic_facts f WHERE f.memory_kind_source IN ('user','caller') "
            "AND f.memory_kind IS NOT NULL AND EXISTS "
            "(SELECT 1 FROM snap.atomic_facts s WHERE s.fact_id = f.fact_id)")


def _new_rows_sql(conn: sqlite3.Connection, table: str, keys: Iterable[str]) -> str | None:
    """Rows of ``table`` in the live store that the copy does not have."""
    if table not in _tables(conn, "main"):
        return None
    if table not in _tables(conn, "snap"):
        return f"FROM main.{table} l"
    match = " AND ".join(f"s.{k} = l.{k}" for k in keys)
    return f"FROM main.{table} l WHERE NOT EXISTS (SELECT 1 FROM snap.{table} s WHERE {match})"


def delta_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Counts for the preview. Reads only."""
    kinds = _kind_edit_sql(conn)
    corrections = _new_rows_sql(conn, "correction_cases", ("case_id",))
    receipts = _new_rows_sql(conn, "erasure_receipts", ("erasure_id",))
    tombstones = _new_rows_sql(conn, "projection_tombstones", ("profile_id", "fact_id"))
    return {
        "facts_now": _count(conn, "SELECT COUNT(*) FROM main.atomic_facts"),
        "facts_in_snapshot": _count(conn, "SELECT COUNT(*) FROM snap.atomic_facts"),
        "memories_now": _count(conn, "SELECT COUNT(*) FROM main.memories"),
        "memories_in_snapshot": _count(conn, "SELECT COUNT(*) FROM snap.memories"),
        "added_memories": _count(conn, f"SELECT COUNT(*) {_ADDED_MEMORIES}"),
        "deleted_facts": _count(conn, f"SELECT COUNT(*) {_DELETED_FACTS}"),
        "deleted_memories": _count(conn, f"SELECT COUNT(*) {_DELETED_MEMORIES}"),
        "kind_edits": _count(conn, f"SELECT COUNT(*) {kinds}") if kinds else 0,
        "corrections_lost": _count(conn, f"SELECT COUNT(*) {corrections}") if corrections else 0,
        "erasures": max(
            _count(conn, "SELECT COUNT(*) " + _and(receipts, "l.state='COMPLETE'"))
            if receipts else 0,
            _count(conn, f"SELECT COUNT(*) {tombstones}") if tombstones else 0),
    }


def _and(sql: str, condition: str) -> str:
    return f"{sql} {'AND' if ' WHERE ' in sql else 'WHERE'} {condition}"


def _kind_of_memory(conn: sqlite3.Connection, memory_id: str) -> str | None:
    """The kind every fact of this memory was confirmed as, else None."""
    if "memory_kind_source" not in _cols(conn, "atomic_facts"):
        return None
    rows = conn.execute(
        "SELECT memory_kind, memory_kind_source FROM main.atomic_facts WHERE memory_id=?",
        (memory_id,)).fetchall()
    if not rows or any(r[1] not in _CONFIRMED or not r[0] for r in rows):
        return None
    kinds = {r[0] for r in rows}
    return kinds.pop() if len(kinds) == 1 else None


def export_delta(conn: sqlite3.Connection, delta_dir: Path) -> dict[str, int]:
    """Write the delta files (0600). Returns the counts written."""
    delta_dir = Path(delta_dir)
    delta_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    memories = []
    for row in conn.execute(f"SELECT m.* {_ADDED_MEMORIES} ORDER BY m.rowid"):
        item = dict(row)
        item["memory_kind"] = _kind_of_memory(conn, item["memory_id"])
        memories.append(item)
    deleted = [{"type": "fact", **dict(r)} for r in conn.execute(
        f"SELECT s.fact_id, s.profile_id, s.memory_id {_DELETED_FACTS}")]
    deleted += [{"type": "memory", **dict(r)} for r in conn.execute(
        f"SELECT s.memory_id, s.profile_id {_DELETED_MEMORIES}")]
    kinds_sql = _kind_edit_sql(conn)
    kind_edits = [dict(r) for r in conn.execute(
        f"SELECT f.fact_id, f.profile_id, f.fact_type, "
        f"{', '.join('f.' + c for c in _KIND_COLS)} {kinds_sql}")] if kinds_sql else []
    erasures: list[dict[str, Any]] = []
    receipts = _new_rows_sql(conn, "erasure_receipts", ("erasure_id",))
    if receipts:
        erasures += [{"type": "receipt", **dict(r)} for r in conn.execute(
            f"SELECT l.* {receipts}")]
    tombstones = _new_rows_sql(conn, "projection_tombstones", ("profile_id", "fact_id"))
    if tombstones:
        erasures += [{"type": "tombstone", **dict(r)} for r in conn.execute(
            f"SELECT l.* {tombstones}")]
    new_profiles = _new_rows_sql(conn, "profiles", ("profile_id",))
    profiles = [dict(r) for r in conn.execute(f"SELECT l.* {new_profiles}")] if new_profiles else []
    write_jsonl_atomic(delta_dir / "memories.jsonl", memories)
    write_jsonl_atomic(delta_dir / "deleted.jsonl", deleted)
    write_jsonl_atomic(delta_dir / "kind_edits.jsonl", kind_edits)
    write_jsonl_atomic(delta_dir / "erasures.jsonl", erasures)
    write_jsonl_atomic(delta_dir / "profiles.jsonl", profiles)
    live_profiles = sorted(r[0] for r in conn.execute("SELECT profile_id FROM main.profiles"))
    counts = {"memories": len(memories), "deleted": len(deleted),
              "kind_edits": len(kind_edits), "erasures": len(erasures),
              "profiles": len(profiles)}
    write_json_atomic(delta_dir / "meta.json", {"live_profiles": live_profiles, "counts": counts})
    return counts


# ---------------------------------------------------------------------------
# Applying the delta to a staging copy (never the live store)
# ---------------------------------------------------------------------------

def _load_vec(conn: sqlite3.Connection) -> bool:
    try:
        from superlocalmemory.storage.sqlite_vectors import load_sqlite_vec_extension

        load_sqlite_vec_extension(conn)
        conn.execute("SELECT vec_version()").fetchone()
        return True
    except Exception:  # noqa: BLE001 - absent extension: vector rows are left
        return False


def _layout(conn: sqlite3.Connection) -> tuple[list[str], set[str]]:
    """(regular tables, virtual table names). Shadow tables are excluded."""
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
    virtual = {n for n, sql in rows if (sql or "").upper().startswith("CREATE VIRTUAL")}
    regular = [n for n, _sql in rows
               if n not in virtual and not n.startswith("sqlite_")
               and not any(n.startswith(v + "_") for v in virtual)]
    return regular, virtual


def _insert_rows(conn: sqlite3.Connection, table: str, rows: list[dict[str, Any]]) -> int:
    have = set(_cols(conn, table))
    done = 0
    for row in rows:
        cols = [c for c in row if c in have]
        if not cols:
            continue
        cur = conn.execute(
            f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)})", [row[c] for c in cols])
        done += max(cur.rowcount, 0)
    return done


def _erase_profiles(conn: sqlite3.Connection, profiles: set[str], live: set[str],
                    vec: bool) -> int:
    """Remove every row of each erased profile (foreign keys off, one transaction)."""
    if not profiles:
        return 0
    regular, virtual = _layout(conn)
    if "fact_embeddings" in virtual and not vec:
        raise RuntimeError("cannot remove an erased profile's vectors: the vector "
                           "extension did not load")
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("BEGIN IMMEDIATE")
    try:
        for pid in sorted(profiles):
            if "fact_embeddings" in virtual:
                conn.execute("DELETE FROM fact_embeddings WHERE profile_id=?", (pid,))
            for table in regular:
                if table in _LEDGERS or "profile_id" not in _cols(conn, table):
                    continue
                conn.execute(f"DELETE FROM {table} WHERE profile_id=?", (pid,))
            if pid not in live:
                conn.execute("DELETE FROM profiles WHERE profile_id=?", (pid,))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return len(profiles)


def _remove_fact(conn: sqlite3.Connection, fact_id: str, *, erasure: bool,
                 fact_tables: list[str], virtual: set[str], vec: bool) -> bool:
    conn.execute("SAVEPOINT fact")
    try:
        if vec and "fact_embeddings" in virtual and "embedding_metadata" in fact_tables:
            conn.execute("DELETE FROM fact_embeddings WHERE rowid IN (SELECT vec_rowid "
                         "FROM embedding_metadata WHERE fact_id=?)", (fact_id,))
        if "fact_expansion_fts" in virtual:
            conn.execute("DELETE FROM fact_expansion_fts WHERE fact_id=?", (fact_id,))
        if erasure and "correction_cases" in fact_tables:
            # The ledger holds no text, but it RESTRICTs the fact's deletion; an
            # erasure in the live store already removed these rows with the fact.
            cases = ("SELECT case_id FROM correction_cases WHERE predecessor_fact_id=? "
                     "OR successor_fact_id=?")
            if "correction_events" in _tables(conn):
                conn.execute(f"DELETE FROM correction_events WHERE case_id IN ({cases})",
                             (fact_id, fact_id))
            conn.execute("DELETE FROM correction_cases WHERE predecessor_fact_id=? "
                         "OR successor_fact_id=?", (fact_id, fact_id))
        for table in fact_tables:
            if table not in ("correction_cases", "graph_edges"):
                conn.execute(f"DELETE FROM {table} WHERE fact_id=?", (fact_id,))
        if "graph_edges" in fact_tables:
            conn.execute("DELETE FROM graph_edges WHERE source_id=? OR target_id=?",
                         (fact_id, fact_id))
        conn.execute("DELETE FROM atomic_facts WHERE fact_id=?", (fact_id,))
        conn.execute("RELEASE fact")
        return True
    except sqlite3.IntegrityError:
        conn.execute("ROLLBACK TO fact")
        conn.execute("RELEASE fact")
        return False


def _fact_tables(conn: sqlite3.Connection) -> tuple[list[str], set[str]]:
    regular, virtual = _layout(conn)
    out = [t for t in regular
           if t not in ("atomic_facts", "projection_tombstones")
           and "fact_id" in _cols(conn, t)]
    if "correction_cases" in regular:
        out.append("correction_cases")
    if "graph_edges" in regular:
        out.append("graph_edges")
    return out, virtual


def _pending_obligation_profiles(conn: sqlite3.Connection, data_root: Path) -> set[str]:
    """Profiles the GDPR obligation ledger says must not come back."""
    if not (Path(data_root) / "backup_obligations.db").exists():
        return set()
    from superlocalmemory.infra.backup_obligations import BackupObligationStore

    store = BackupObligationStore(Path(data_root))
    present = {r[0] for r in conn.execute("SELECT DISTINCT profile_id FROM atomic_facts")}
    return {pid for pid in present if pid and store.list_pending_for_profile(pid)}


def apply_delta_to_staging(staging: Path, delta_dir: Path, *, data_root: Path) -> dict[str, int]:
    """Erasures, deletions and new profiles applied to the staging copy."""
    delta_dir = Path(delta_dir)
    erasures = read_jsonl(delta_dir / "erasures.jsonl")
    deleted = read_jsonl(delta_dir / "deleted.jsonl")
    profiles = read_jsonl(delta_dir / "profiles.jsonl")
    live = set((read_json(delta_dir / "meta.json") or {}).get("live_profiles") or [])
    receipts = [e for e in erasures if e.get("type") == "receipt"]
    tombstones = [e for e in erasures if e.get("type") == "tombstone"]
    with closing(sqlite3.connect(str(staging), isolation_level=None)) as conn:
        vec = _load_vec(conn)
        erased_profiles = {r["profile_id"] for r in receipts
                           if r.get("subject_type") == "profile" and r.get("state") == "COMPLETE"}
        erased_profiles |= _pending_obligation_profiles(conn, data_root)
        n_profiles = _erase_profiles(conn, erased_profiles, live, vec)
        conn.execute("PRAGMA foreign_keys=ON")
        fact_tables, virtual = _fact_tables(conn)
        counts = {"profiles_erased": n_profiles, "facts_erased": 0, "facts_deleted": 0,
                  "memories_removed": 0, "kept_protected": 0, "profiles_added": 0}
        conn.execute("BEGIN IMMEDIATE")
        try:
            counts["profiles_added"] = _insert_rows(
                conn, "profiles", [{k: v for k, v in p.items() if k != "type"} for p in profiles])
            plan = [(t["fact_id"], True) for t in tombstones]
            plan += [(d["fact_id"], False) for d in deleted if d.get("type") == "fact"]
            for fact_id, erasure in plan:
                if not conn.execute("SELECT 1 FROM atomic_facts WHERE fact_id=?",
                                    (fact_id,)).fetchone():
                    continue
                if _remove_fact(conn, fact_id, erasure=erasure, fact_tables=fact_tables,
                                virtual=virtual, vec=vec):
                    counts["facts_erased" if erasure else "facts_deleted"] += 1
                else:
                    counts["kept_protected"] += 1
            memory_ids = [d["memory_id"] for d in deleted if d.get("type") == "memory"]
            memory_ids += [t["memory_id"] for t in tombstones if t.get("memory_id")]
            for memory_id in memory_ids:
                cur = conn.execute(
                    "DELETE FROM memories WHERE memory_id=? AND NOT EXISTS "
                    "(SELECT 1 FROM atomic_facts WHERE memory_id=?)", (memory_id, memory_id))
                counts["memories_removed"] += max(cur.rowcount, 0)
            for table, rows in (("erasure_receipts", receipts),
                                ("projection_tombstones", tombstones)):
                if rows and table in _tables(conn):
                    _insert_rows(conn, table, [{k: v for k, v in r.items() if k != "type"}
                                               for r in rows])
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("the prepared copy failed its check")
    return counts


def load_memories(delta_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(Path(delta_dir) / "memories.jsonl")


def load_kind_edits(delta_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(Path(delta_dir) / "kind_edits.jsonl")


def metadata_of(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata_json") or "{}"
    try:
        meta = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        meta = {}
    return meta if isinstance(meta, dict) else {}


__all__ = ["apply_delta_to_staging", "delta_counts", "export_delta", "load_kind_edits",
           "load_memories", "metadata_of", "open_compare"]
