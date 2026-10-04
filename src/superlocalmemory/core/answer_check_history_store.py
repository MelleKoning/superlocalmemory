# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Answer Check history, the saved side: the daemon's writer and the SQL.

The recall path never reaches this module. ``answer_check_history`` keeps the
ring; a single thread here ("slm-answer-check-history", daemon process only)
saves it to learning.db every ``FLUSH_INTERVAL_S`` seconds, or sooner when
``BATCH_MAX`` entries are waiting. Each save is one short ``BEGIN IMMEDIATE``
transaction of at most ``BATCH_MAX`` rows; pruning deletes in chunks of
``PRUNE_CHUNK`` so no transaction holds learning.db's write lock for long.

Erasure is safe across processes. ``erase_profile_everywhere`` writes a
tombstone and deletes the rows in one transaction, and every insert is
conditional on there being no tombstone at or after the event's own time — so
a batch already in flight in the daemon when ``slm gdpr`` erased the profile in
another process inserts nothing. The daemon also purges its ring from new
tombstones on every tick. In this process, ``_flush_lock`` makes erase and
flush mutually exclusive.

A tombstone never refuses an entry recorded after this process had applied it
(``answer_check_history.erasure_known_at``): the wall clock can step back, and
a recall made after an erasure is not erased by it. Every entry a tombstone
does refuse is counted (``erased_unsaved``), so "recorded" always equals what
was saved plus what is accounted for.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from superlocalmemory.core import answer_check_history as history

logger = logging.getLogger(__name__)

FLUSH_INTERVAL_S = 2.0
BATCH_MAX = 128
SWEEP_INTERVAL_S = 600.0
PRUNE_CHUNK = 500
TOMBSTONE_TTL_MS = 86_400_000
CONNECT_TIMEOUT_S = 5.0
BUSY_TIMEOUT_MS = 5000
_LOG_EVERY_S = 60.0

#: A check this much older than its profile's row counts as the previous
#: profile's: room for a clock that was nudged back, never for a person.
RECREATED_MARGIN_MS = 5_000

DEFAULT_RETENTION_DAYS = 30
DEFAULT_MAX_ROWS = 10_000
#: Most rows kept per profile once a sweep has run. Between sweeps (every
#: ``SWEEP_INTERVAL_S``) a profile can briefly hold more; a data export
#: (compliance/gdpr.py) returns every row there is either way.
MAX_ROWS_CEILING = 10_000
MAX_ROWS_FLOOR = 1_000
RETENTION_DAYS_RANGE = (1, 365)

COLUMNS = ("event_id", "profile_id", "occurred_ms", "status", "detail", "backend",
           "origin", "abstained", "abstention_reason", "answer_confidence", "threshold",
           "reordered", "result_count", "query_type", "retrieval_ms", "judge_ms",
           "total_ms", "calibration_id")
INSERT_SQL = (
    f"INSERT OR IGNORE INTO answer_check_events ({','.join(COLUMNS)}) "
    f"SELECT {','.join('?' * len(COLUMNS))} WHERE NOT EXISTS ("
    "SELECT 1 FROM answer_check_erasures e WHERE e.profile_id = ? AND e.erased_at_ms >= ? "
    "AND e.erased_at_ms > ?)"
)

_flush_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop = threading.Event()
_conn: sqlite3.Connection | None = None
_state: dict[str, Any] = {"learning_db": None, "memory_db": None,
                          "retention_days": DEFAULT_RETENTION_DAYS,
                          "max_rows": DEFAULT_MAX_ROWS, "last_saved_ms": None,
                          "tombstones_seen_ms": 0, "last_sweep": 0.0, "last_log": 0.0}


# -- connections and SQL -----------------------------------------------------------

def connect(db: Path, *, readonly: bool) -> sqlite3.Connection:
    if readonly:
        return sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
    conn = sqlite3.connect(str(db), timeout=CONNECT_TIMEOUT_S, isolation_level=None,
                           check_same_thread=False)
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    return conn


def _row(ev: history.VerdictEvent, known: int = 0) -> tuple:
    values = tuple(getattr(ev, name) for name in COLUMNS)
    values = tuple(int(v) if isinstance(v, bool) else v for v in values)
    return values + (ev.profile_id, ev.occurred_ms, int(known))


def _refused(ev: history.VerdictEvent, known: int, tombstones: Mapping[str, int]) -> bool:
    """Whether a tombstone refuses ``ev`` — the same rule as ``INSERT_SQL``."""
    erased_at = tombstones.get(ev.profile_id)
    return erased_at is not None and erased_at >= ev.occurred_ms and erased_at > known


def _in_txn(conn: sqlite3.Connection, fn) -> Any:
    conn.execute("BEGIN IMMEDIATE")
    try:
        out = fn()
        conn.execute("COMMIT")
        return out
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def save_batch(conn: sqlite3.Connection, events: list[history.VerdictEvent],
               known: list[int] | None = None) -> tuple[int, int]:
    """Insert in one transaction: (rows inserted, rows an erasure refused).

    ``known[i]`` is the latest erasure this process had applied when
    ``events[i]`` was recorded; such an erasure does not refuse it. A row that
    is already saved (same event id) is neither: it is simply there.
    """
    marks = list(known) if known is not None else [0] * len(events)
    if len(marks) != len(events):
        raise ValueError("known must have one entry per event")

    def go() -> tuple[int, int]:
        profiles = sorted({ev.profile_id for ev in events})
        tombstones: dict[str, int] = {}
        if profiles:
            tombstones = {pid: int(at) for pid, at in conn.execute(
                "SELECT profile_id, erased_at_ms FROM answer_check_erasures "  # noqa: S608
                f"WHERE profile_id IN ({','.join('?' * len(profiles))})", tuple(profiles))}
        rows, refused = [], 0
        for ev, mark in zip(events, marks):
            if _refused(ev, mark, tombstones):
                refused += 1
            else:
                rows.append(_row(ev, mark))
        before = conn.total_changes
        conn.executemany(INSERT_SQL, rows)
        return conn.total_changes - before, refused
    return _in_txn(conn, go)


def insert_batch(conn: sqlite3.Connection, events: list[history.VerdictEvent],
                 known: list[int] | None = None) -> int:
    """Insert in one transaction; returns rows actually inserted."""
    return save_batch(conn, events, known)[0]


def _delete_chunk(conn: sqlite3.Connection, where: str, params: tuple, limit: int) -> int:
    sql = (f"DELETE FROM answer_check_events WHERE rowid IN (SELECT rowid FROM "  # noqa: S608
           f"answer_check_events WHERE {where} ORDER BY occurred_ms ASC LIMIT ?)")
    return _in_txn(conn, lambda: conn.execute(sql, params + (limit,)).rowcount)


def prune(conn: sqlite3.Connection, *, now_ms: int, retention_days: int,
          max_rows: int) -> int:
    """Age, then per-profile count, then old tombstones. Chunked transactions."""
    removed = 0
    cutoff = now_ms - retention_days * 86_400_000
    while True:
        n = _delete_chunk(conn, "occurred_ms < ?", (cutoff,), PRUNE_CHUNK)
        removed += n
        if n < PRUNE_CHUNK:
            break
    over = conn.execute("SELECT profile_id, COUNT(*) FROM answer_check_events "
                        "GROUP BY profile_id HAVING COUNT(*) > ?", (max_rows,)).fetchall()
    for profile_id, count in over:
        excess = count - max_rows
        while excess > 0:
            n = _delete_chunk(conn, "profile_id = ?", (profile_id,), min(PRUNE_CHUNK, excess))
            if n == 0:
                break
            removed += n
            excess -= n
    _in_txn(conn, lambda: conn.execute(
        "DELETE FROM answer_check_erasures WHERE erased_at_ms < ?",
        (now_ms - TOMBSTONE_TTL_MS,)))
    return removed


def erase_profile_rows(conn: sqlite3.Connection, profile_id: str, *, now_ms: int) -> int:
    """Tombstone and delete in one transaction; returns rows deleted."""
    def go() -> int:
        conn.execute("INSERT OR REPLACE INTO answer_check_erasures (profile_id, erased_at_ms) "
                     "VALUES (?, ?)", (profile_id, now_ms))
        return conn.execute("DELETE FROM answer_check_events WHERE profile_id = ?",
                            (profile_id,)).rowcount
    return _in_txn(conn, go)


def apply_remote_tombstones(conn: sqlite3.Connection) -> int:
    """Purge the ring of profiles another process erased. Returns entries purged."""
    rows = conn.execute("SELECT profile_id, erased_at_ms FROM answer_check_erasures "
                        "WHERE erased_at_ms > ?", (_state["tombstones_seen_ms"],)).fetchall()
    purged = 0
    for profile_id, erased_at in rows:
        purged += history.forget_profile(profile_id, occurred_before_ms=int(erased_at),
                                         erased_at_ms=int(erased_at))
        _state["tombstones_seen_ms"] = max(_state["tombstones_seen_ms"], int(erased_at))
    return purged


# -- the writer --------------------------------------------------------------------

def _log_once(message: str, exc: BaseException) -> None:
    now = time.monotonic()
    if now - _state["last_log"] >= _LOG_EVERY_S:
        _state["last_log"] = now
        logger.warning(message, type(exc).__name__)


def _writer_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = connect(_state["learning_db"], readonly=False)
    return _conn


def flush_once() -> int:
    """Save up to ``BATCH_MAX`` unsaved entries. Returns how many were taken."""
    with _flush_lock:
        batch = history.snapshot_unsaved(BATCH_MAX)
        if not batch:
            return 0
        known = [history.erasure_known_at(ev.profile_id, seq) for seq, ev in batch]
        try:
            inserted, refused = save_batch(_writer_conn(), [ev for _, ev in batch], known)
        except sqlite3.Error as exc:
            history.mark_failed(batch)
            _log_once("Answer Check history: a save failed (%s); will retry", exc)
            return 0
        history.mark_saved(batch[-1][0], inserted, erased=refused)
        _state["last_saved_ms"] = int(time.time() * 1000)
        return len(batch)


def _tick() -> None:
    while flush_once() >= BATCH_MAX and not _stop.is_set():
        pass
    with _flush_lock:
        apply_remote_tombstones(_writer_conn())
    if time.monotonic() - _state["last_sweep"] >= SWEEP_INTERVAL_S:
        _state["last_sweep"] = time.monotonic()
        prune(_writer_conn(), now_ms=int(time.time() * 1000),
              retention_days=_state["retention_days"], max_rows=_state["max_rows"])


def _loop() -> None:
    try:
        reconcile_with_profiles(_state["learning_db"], _state["memory_db"])
    except Exception as exc:  # noqa: BLE001 — retried at the next start
        _log_once("Answer Check history: could not check for erasures made by an "
                  "older version (%s); will look again at the next start", exc)
    wake = history.wake_event()
    while not _stop.is_set():
        wake.wait(FLUSH_INTERVAL_S)
        wake.clear()
        if _stop.is_set():
            break
        try:
            _tick()
        except Exception as exc:  # noqa: BLE001 — the writer must outlive one bad tick
            _log_once("Answer Check history: writer tick failed (%s)", exc)


def clamp_settings(retention_days: Any, max_rows: Any) -> tuple[int, int]:
    def as_int(value: Any, default: int) -> int:
        return value if isinstance(value, int) and not isinstance(value, bool) else default
    low, high = RETENTION_DAYS_RANGE
    days = min(high, max(low, as_int(retention_days, DEFAULT_RETENTION_DAYS)))
    rows = min(MAX_ROWS_CEILING, max(MAX_ROWS_FLOOR, as_int(max_rows, DEFAULT_MAX_ROWS)))
    return days, rows


def start_writer(learning_db: Path, *, retention_days: Any = DEFAULT_RETENTION_DAYS,
                 max_rows: Any = DEFAULT_MAX_ROWS, memory_db: Path | None = None) -> None:
    """Start the single writer for this process (idempotent) and enable recording.

    Its first act is ``reconcile_with_profiles``: an erasure made while an older
    version ran is applied before anything else is saved. ``memory_db``
    defaults to the ``memory.db`` beside ``learning_db``.
    """
    global _thread
    days, rows = clamp_settings(retention_days, max_rows)
    learning = Path(learning_db)
    _state.update(learning_db=learning, retention_days=days, max_rows=rows, last_sweep=0.0,
                  memory_db=Path(memory_db) if memory_db else learning.with_name("memory.db"))
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    history.enable(True)
    _thread = threading.Thread(target=_loop, name="slm-answer-check-history", daemon=True)
    _thread.start()


def stop_writer(*, timeout_s: float = 2.0) -> int:
    """Stop recording, save what is left (≤ 1 s), close. Returns entries unsaved."""
    global _thread, _conn
    history.enable(False)
    _stop.set()
    history.wake_event().set()
    if _thread is not None:
        _thread.join(timeout_s)
        _thread = None
    deadline = time.monotonic() + 1.0
    if _state["learning_db"] is not None:
        while time.monotonic() < deadline and flush_once() > 0:
            pass
    with _flush_lock:
        if _conn is not None:
            _conn.close()
            _conn = None
    return history.counters()["unsaved"]


def writer_info() -> dict[str, Any]:
    return {"running": _thread is not None and _thread.is_alive(),
            "last_saved_ms": _state["last_saved_ms"],
            "retention_days": _state["retention_days"], "max_rows": _state["max_rows"]}


# -- erasure (GDPR, profile deletion, the tab's Clear button) -----------------------

def _has_tables(conn: sqlite3.Connection) -> bool:
    found = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
        "('answer_check_events', 'answer_check_erasures')")}
    return len(found) == 2


def erase_profile_everywhere(learning_db: Path, profile_id: str) -> int:
    """Erase a profile's history from this process's ring and from learning.db.

    Safe from any process. Returns the number of saved rows deleted. Raises on
    a database error so a caller (GDPR erasure) can abort before going further.
    """
    with _flush_lock:
        now_ms = int(time.time() * 1000)
        history.forget_profile(profile_id, erased_at_ms=now_ms)
        db = Path(learning_db)
        if not db.exists():
            return 0
        conn = connect(db, readonly=False)
        try:
            if not _has_tables(conn):
                return 0
            return erase_profile_rows(conn, profile_id, now_ms=now_ms)
        finally:
            conn.close()


clear_profile = erase_profile_everywhere


# -- erasures made while an older version ran (4.1.18 / 4.1.19) --------------------

def _created_ms(created_at: Any) -> int | None:
    """``profiles.created_at`` (SQLite ``datetime('now')``, UTC) in ms; None if unreadable."""
    if not isinstance(created_at, str) or not created_at.strip():
        return None
    try:
        moment = datetime.fromisoformat(created_at.strip().replace(" ", "T", 1))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return int(moment.timestamp() * 1000)


def _profiles(memory_db: Path) -> dict[str, int | None] | None:
    """Every profile in ``memory_db`` with its creation time, or None when that
    cannot be read with certainty (then nothing is erased on its word)."""
    if not Path(memory_db).exists():
        return None
    try:
        conn = connect(Path(memory_db), readonly=True)
        try:
            rows = conn.execute("SELECT profile_id, created_at FROM profiles").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if not rows:
        return None  # no profile at all is not a store anyone erased everything from
    return {pid: _created_ms(created) for pid, created in rows if isinstance(pid, str)}


def reconcile_with_profiles(learning_db: Path | None, memory_db: Path | None, *,
                            now_ms: int | None = None) -> dict[str, int]:
    """Erase the history of profiles erased or deleted while an older version ran.

    4.1.18 and 4.1.19 do not know this history exists: a privacy erasure or a
    profile deletion run on them (after "prepare to go back") removes the
    profile and leaves its checks behind. Both end by deleting the profile's
    row in memory.db, and memory.db is the source of truth for profiles. So,
    each time this version's writer starts:

    * a profile with history but no row any more is erased here, exactly as an
      erasure in this version would have done (rows and tombstone);
    * a profile whose row is newer than some of its history was removed and
      created again under the same name; the history from before its row was
      created belonged to the one that was removed, and is deleted.

    Nothing is erased when the profiles cannot be read. Returns
    ``{"profiles": n, "rows": n}`` for what was removed.
    """
    done = {"profiles": 0, "rows": 0}
    if learning_db is None or memory_db is None or not Path(learning_db).exists():
        return done
    profiles = _profiles(Path(memory_db))
    if profiles is None:
        return done
    stamp = int(time.time() * 1000) if now_ms is None else int(now_ms)
    with _flush_lock:
        conn = connect(Path(learning_db), readonly=False)
        try:
            if not _has_tables(conn):
                return done
            seen = conn.execute("SELECT profile_id, MIN(occurred_ms) FROM answer_check_events "
                                "GROUP BY profile_id").fetchall()
            for profile_id, oldest in seen:
                removed = _reconcile_one(conn, profile_id, oldest, profiles, stamp)
                if removed:
                    done["profiles"] += 1
                    done["rows"] += removed
        finally:
            conn.close()
    if done["rows"]:
        logger.info("Answer Check history: removed %d checks of %d profiles erased or "
                    "deleted while an older version was running", done["rows"],
                    done["profiles"])
    return done


def _reconcile_one(conn: sqlite3.Connection, profile_id: str, oldest: int,
                   profiles: dict[str, int | None], now_ms: int) -> int:
    if profile_id not in profiles:
        history.forget_profile(profile_id, erased_at_ms=now_ms)
        return erase_profile_rows(conn, profile_id, now_ms=now_ms)
    created = profiles[profile_id]
    if created is None or oldest is None:
        return 0
    before = created - RECREATED_MARGIN_MS
    if int(oldest) >= before:
        return 0
    removed = 0
    while True:
        n = _delete_chunk(conn, "profile_id = ? AND occurred_ms < ?", (profile_id, before),
                          PRUNE_CHUNK)
        removed += n
        if n < PRUNE_CHUNK:
            return removed


# -- readers (dashboard routes; never the recall path) ------------------------------

def _read(learning_db: Path, sql: str, params: tuple) -> list[dict[str, Any]]:
    db = Path(learning_db)
    if not db.exists():
        return []
    conn = connect(db, readonly=True)
    try:
        conn.row_factory = sqlite3.Row
        if not _has_tables(conn):
            return []
        return [dict(row) for row in conn.execute(sql, params)]
    finally:
        conn.close()


def read_window(learning_db: Path, profile_id: str, *, since_ms: int,
                include_dashboard: bool) -> list[dict[str, Any]]:
    origin_clause = "" if include_dashboard else " AND origin != 'dashboard'"
    return _read(learning_db,
                 f"SELECT {','.join(COLUMNS)} FROM answer_check_events "  # noqa: S608
                 f"WHERE profile_id = ? AND occurred_ms >= ?{origin_clause}",
                 (profile_id, int(since_ms)))


def read_page(learning_db: Path, profile_id: str, *, cursor: tuple[int, str] | None,
              limit: int, status: str | None, since_ms: int,
              include_dashboard: bool = True) -> tuple[list[dict[str, Any]], str | None]:
    """Newest first, keyset-paginated on (occurred_ms, event_id)."""
    clauses, params = ["profile_id = ?", "occurred_ms >= ?"], [profile_id, int(since_ms)]
    if cursor is not None:
        clauses.append("(occurred_ms < ? OR (occurred_ms = ? AND event_id < ?))")
        params += [cursor[0], cursor[0], cursor[1]]
    if status:
        clauses.append("status = ?")
        params.append(status)
    if not include_dashboard:
        clauses.append("origin != 'dashboard'")
    rows = _read(learning_db,
                 f"SELECT {','.join(COLUMNS)} FROM answer_check_events "  # noqa: S608
                 f"WHERE {' AND '.join(clauses)} "
                 "ORDER BY occurred_ms DESC, event_id DESC LIMIT ?",
                 tuple(params) + (limit + 1,))
    if len(rows) <= limit:
        return rows, None
    last = rows[limit - 1]
    return rows[:limit], f"{last['occurred_ms']}:{last['event_id']}"


def event_as_row(ev: history.VerdictEvent) -> Mapping[str, Any]:
    return {name: getattr(ev, name) for name in COLUMNS}


def _reset_for_testing() -> None:
    stop_writer(timeout_s=1.0)
    _state.update(learning_db=None, memory_db=None, retention_days=DEFAULT_RETENTION_DAYS,
                  max_rows=DEFAULT_MAX_ROWS, last_saved_ms=None, tombstones_seen_ms=0,
                  last_sweep=0.0, last_log=0.0)


__all__ = ["BATCH_MAX", "COLUMNS", "FLUSH_INTERVAL_S", "MAX_ROWS_CEILING", "apply_remote_tombstones",
           "clamp_settings", "clear_profile", "connect", "erase_profile_everywhere",
           "erase_profile_rows", "event_as_row", "flush_once", "insert_batch", "prune",
           "reconcile_with_profiles", "save_batch",
           "read_page", "read_window", "start_writer", "stop_writer", "writer_info"]
