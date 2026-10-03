# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""After a restore: add back what was written since the copy, exactly once.

Runs once the engine is up (a daemon background thread, or ``slm db restore``).
Every memory written after the copy goes back in through the same canonical
ingestion as any other write -- its own profile, scope, sharing and session --
under the idempotency key ``restore-reimport:<memory_id>``, so running this
twice, or resuming it after a crash, never makes a duplicate. A kind is carried
only when every fact of that memory had the same confirmed kind.

Kinds a person confirmed on facts that the copy already had are put back with
their original source, and only over a kind nobody confirmed since.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from superlocalmemory.storage._durable_json import read_json, write_json_atomic
from superlocalmemory.storage._restore_delta import load_kind_edits, load_memories, metadata_of
from superlocalmemory.storage._restore_types import OUTCOME_NAME, ReimportReport

logger = logging.getLogger(__name__)

SOURCE_TYPE = "restore-reimport"
_KIND_SQL = (
    "UPDATE atomic_facts SET memory_kind=?, memory_kind_source=?, memory_kind_confidence=?, "
    "memory_kind_recipe=?, memory_kind_at=?, fact_type=? WHERE fact_id=? AND profile_id=? "
    "AND COALESCE(memory_kind_source, '') NOT IN ('user', 'caller')")


def _shared_with(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return [s.strip() for s in str(value).split(",") if s.strip()]
    return [str(v) for v in parsed] if isinstance(parsed, list) else []


def _already(conn: sqlite3.Connection, profile_id: str, key: str) -> bool:
    try:
        return conn.execute(
            "SELECT 1 FROM ingestion_operations WHERE profile_id=? AND source_type=? "
            "AND idempotency_key=?", (profile_id, SOURCE_TYPE, key)).fetchone() is not None
    except sqlite3.Error:
        return False


def _readonly(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{Path(db_path).absolute().as_uri()}?mode=ro", uri=True)


def _reimport_memories(engine: Any, rows: list[dict[str, Any]], db_path: Path,
                       counts: dict[str, Any]) -> None:
    from superlocalmemory.core.engine_ingestion import canonical_store, local_trusted_actor_id
    from superlocalmemory.core.ingestion_command import UnknownProfileError
    from superlocalmemory.storage.memory_kinds import METADATA_KEY, parse_kind

    actor = local_trusted_actor_id(SOURCE_TYPE)
    for row in rows:
        key = f"{SOURCE_TYPE}:{row['memory_id']}"
        profile_id = str(row.get("profile_id") or "default")
        with closing(_readonly(db_path)) as conn:
            if _already(conn, profile_id, key):
                counts["already_present"] += 1
                continue
        meta = {k: v for k, v in metadata_of(row).items() if not str(k).startswith("_slm")}
        kind = parse_kind(row.get("memory_kind"))
        if kind is not None:
            meta[METADATA_KEY] = kind.value
        try:
            result = canonical_store(
                engine, str(row.get("content") or ""), source_type=SOURCE_TYPE,
                trusted_actor_id=actor, metadata=meta,
                scope=str(row.get("scope") or "personal"),
                shared_with=_shared_with(row.get("shared_with")),
                session_id=str(row.get("session_id") or ""),
                session_date=row.get("session_date") or None,
                speaker=str(row.get("speaker") or ""), role=str(row.get("role") or "user"),
                idempotency_key=key, require_complete=False, profile_id=profile_id)
        except UnknownProfileError:
            counts["skipped_unknown_profile"] += 1
            continue
        except Exception as exc:  # noqa: BLE001 - reported per memory, never fatal
            counts["failed"] += 1
            counts["errors"].append(f"{row.get('memory_id')}: {type(exc).__name__}")
            logger.warning("[SLM] Could not add back memory %s: %s", row.get("memory_id"), exc)
            continue
        if result == []:
            counts["skipped_rejected"] += 1
        else:
            counts["added"] += 1


def _reapply_kinds(db_path: Path, edits: list[dict[str, Any]], counts: dict[str, Any]) -> None:
    from superlocalmemory.storage.memory_kinds import COARSE, parse_kind
    from superlocalmemory.storage.write_lock import get_write_lock

    if not edits:
        return
    with get_write_lock(db_path), closing(sqlite3.connect(str(db_path), timeout=30)) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(atomic_facts)")}
        if "memory_kind_source" not in cols:
            counts["kinds_skipped"] += len(edits)
            return
        history = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                               "AND name='memory_kind_history'").fetchone() is not None
        now = datetime.now(UTC).isoformat(timespec="seconds")
        for edit in edits:
            kind = parse_kind(edit.get("memory_kind"))
            if kind is None or edit.get("memory_kind_source") not in ("user", "caller"):
                counts["kinds_skipped"] += 1
                continue
            cur = conn.execute(_KIND_SQL, (
                kind.value, edit["memory_kind_source"], edit.get("memory_kind_confidence"),
                edit.get("memory_kind_recipe"), edit.get("memory_kind_at") or now,
                COARSE[kind], edit["fact_id"], edit["profile_id"]))
            if cur.rowcount != 1:
                counts["kinds_skipped"] += 1
                continue
            counts["kinds_reapplied"] += 1
            if history:
                conn.execute(
                    "INSERT INTO memory_kind_history (fact_id, profile_id, run_id, origin, "
                    "new_kind, new_source, new_fact_type, actor, changed_at) "
                    "VALUES (?, ?, NULL, 'restore', ?, ?, ?, 'restore', ?)",
                    (edit["fact_id"], edit["profile_id"], kind.value,
                     edit["memory_kind_source"], COARSE[kind], now))
        conn.commit()


def reimport_delta(engine: Any, delta_dir: Path, *, data_root: Path | None = None
                   ) -> ReimportReport:
    """Add back memories and confirmed kinds from ``delta_dir``. Safe to repeat."""
    db_path = Path(engine._db.db_path)
    counts: dict[str, Any] = {k: 0 for k in (
        "added", "already_present", "skipped_unknown_profile", "skipped_rejected",
        "failed", "kinds_reapplied", "kinds_skipped")}
    counts["errors"] = []
    memories = load_memories(delta_dir)
    if memories:
        _reimport_memories(engine, memories, db_path, counts)
    _reapply_kinds(db_path, load_kind_edits(delta_dir), counts)
    report = ReimportReport(**{**counts, "errors": counts["errors"][:20]})
    if data_root is not None:
        outcome = read_json(Path(data_root) / OUTCOME_NAME)
        if outcome is not None:
            write_json_atomic(Path(data_root) / OUTCOME_NAME, {
                **outcome, "reimport_pending": report.failed > 0, "reimport": report.as_dict()})
    logger.info("[SLM] After the restore: %s", report.as_dict())
    return report


def run_pending_reimport(engine: Any, data_root: Path) -> ReimportReport | None:
    """Daemon hook: re-import once when the last restore left work to do."""
    outcome = read_json(Path(data_root) / OUTCOME_NAME)
    if not outcome or not outcome.get("reimport_pending") or not outcome.get("delta_dir"):
        return None
    return reimport_delta(engine, Path(outcome["delta_dir"]), data_root=Path(data_root))


__all__ = ["SOURCE_TYPE", "reimport_delta", "run_pending_reimport"]
