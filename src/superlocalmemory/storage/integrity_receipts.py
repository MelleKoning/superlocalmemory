# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Receipts and undo copies for ``slm db repair``.

One receipt per action: what, why, before, after (counts and ids only, never
memory text). An undo copy keeps the whole removed row so ``--undo RUN_ID``
can put it back; an action on an erased subject keeps no undo copy, because a
copy would keep the erased words. Both are written in the same transaction as
the change they describe, so a repair interrupted at any point leaves either
the change and its receipt or neither.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import time
from typing import Any

DDL = """
CREATE TABLE IF NOT EXISTS integrity_repair_runs (
    run_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    finished_at REAL,
    status TEXT NOT NULL CHECK (status IN ('running', 'finished', 'stopped', 'failed', 'undone')),
    summary_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS integrity_repair_receipts (
    receipt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT NOT NULL,
    reason TEXT NOT NULL,
    before_json TEXT NOT NULL,
    after_json TEXT NOT NULL,
    undoable INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_integrity_receipts_run ON integrity_repair_receipts (run_id);
CREATE TABLE IF NOT EXISTS integrity_repair_undo (
    undo_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    target TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('row', 'update')),
    row_json TEXT NOT NULL,
    restored_at REAL
);
CREATE INDEX IF NOT EXISTS idx_integrity_undo_run ON integrity_repair_undo (run_id);
"""


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)


def _encode(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"b64": base64.b64encode(bytes(value)).decode("ascii")}
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"b64"}:
        return base64.b64decode(value["b64"])
    return value


def row_json(columns: list[str], row: tuple) -> str:
    return json.dumps({c: _encode(v) for c, v in zip(columns, row)})


def receipt(conn: sqlite3.Connection, run_id: str, action: str, target: str, reason: str,
            before: dict, after: dict, *, undoable: bool) -> None:
    conn.execute(
        "INSERT INTO integrity_repair_receipts (run_id, action, target, reason, before_json, "
        "after_json, undoable, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, action, target, reason, json.dumps(before, sort_keys=True),
         json.dumps(after, sort_keys=True), int(undoable), time.time()))


def keep_row(conn: sqlite3.Connection, run_id: str, target: str, payload: str,
             kind: str = "row") -> None:
    conn.execute("INSERT INTO integrity_repair_undo (run_id, target, kind, row_json) "
                 "VALUES (?, ?, ?, ?)", (run_id, target, kind, payload))


def start_run(conn: sqlite3.Connection, run_id: str) -> None:
    conn.execute("INSERT INTO integrity_repair_runs (run_id, started_at, status) "
                 "VALUES (?, ?, 'running')", (run_id, time.time()))
    conn.commit()


def finish_run(conn: sqlite3.Connection, run_id: str, status: str, summary: dict) -> None:
    conn.execute("UPDATE integrity_repair_runs SET finished_at = ?, status = ?, summary_json = ? "
                 "WHERE run_id = ?", (time.time(), status, json.dumps(summary, sort_keys=True),
                                      run_id))
    conn.commit()


def restore_rows(conn: sqlite3.Connection, run_id: str) -> dict[str, int]:
    """Put back every undo copy of a run that is not restored yet. Idempotent.

    A removed row goes back with its own rowid (``INSERT OR IGNORE``: a row
    that exists again is left as it is); an update is reversed only where the
    row still holds the value the repair wrote.
    """
    restored: dict[str, int] = {}
    pending = conn.execute("SELECT undo_id, target, kind, row_json FROM integrity_repair_undo "
                           "WHERE run_id = ? AND restored_at IS NULL ORDER BY undo_id DESC",
                           (run_id,)).fetchall()
    for undo_id, target, kind, payload in pending:
        data = {k: _decode(v) for k, v in json.loads(payload).items()}
        if kind == "row":
            cols = list(data)
            conn.execute(f"INSERT OR IGNORE INTO {target} ({', '.join(cols)}) "  # noqa: S608
                         f"VALUES ({', '.join('?' * len(cols))})", tuple(data[c] for c in cols))
        else:
            key, written, old = data["key"], data["written"], data["old"]
            sets = ", ".join(f"{c} = ?" for c in old)
            guard = " AND ".join(f"{c} IS ?" for c in written)
            conn.execute(f"UPDATE {target} SET {sets} WHERE {key[0]} = ? AND {guard}",  # noqa: S608
                         (*old.values(), key[1], *written.values()))
        conn.execute("UPDATE integrity_repair_undo SET restored_at = ? WHERE undo_id = ?",
                     (time.time(), undo_id))
        restored[target] = restored.get(target, 0) + 1
    return restored


__all__ = ["ensure_tables", "finish_run", "keep_row", "receipt", "restore_rows", "row_json",
           "start_run"]
