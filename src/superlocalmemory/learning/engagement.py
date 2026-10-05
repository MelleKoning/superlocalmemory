# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""
EngagementTracker -- Local-only engagement metrics for V3 learning.

Tracks user engagement events per profile:
    - recall_count, store_count, delete_count
    - session_count, active_days
    - composite engagement_score
    - health classification: active / warm / cold / inactive

All data stays local.  Uses direct sqlite3 with a self-contained
``engagement_events`` table.  NOT coupled to V3 DatabaseManager.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("superlocalmemory.learning.engagement")

# Valid event types
VALID_EVENT_TYPES = frozenset({"recall", "store", "delete", "session_start"})

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS engagement_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id   TEXT    NOT NULL,
    event_type   TEXT    NOT NULL,
    created_at   TEXT    NOT NULL
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_engagement_profile
    ON engagement_events (profile_id, event_type)
"""

# Health thresholds (events in last 7 days)
_ACTIVE_THRESHOLD = 10
_WARM_THRESHOLD = 3


def _utcnow_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _today_iso() -> str:
    """Return today's date as ISO-8601 string."""
    return date.today().isoformat()


class EngagementTracker:
    """
    Tracks per-profile engagement events for the V3 learning system.

    Each instance owns a sqlite3 database at *db_path*.
    Thread-safe: all writes serialised through a lock.

    Args:
        db_path: Path to the sqlite3 database file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
        except BaseException:
            # Setup failed: close it now, or the file stays open (Windows
            # then cannot delete or replace it) for as long as the error lives.
            conn.close()
            raise
        return conn

    # ------------------------------------------------------------------
    # Public API: record events
    # ------------------------------------------------------------------

    def record_event(self, profile_id: str, event_type: str) -> Optional[int]:
        """
        Record an engagement event.

        Args:
            profile_id: Profile that generated the event.
            event_type: One of ``recall``, ``store``, ``delete``,
                        ``session_start``.

        Returns:
            Row ID of the inserted record, or None on invalid input.
        """
        if not profile_id:
            return None
        if event_type not in VALID_EVENT_TYPES:
            logger.warning("Invalid event type: %r", event_type)
            return None

        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "INSERT INTO engagement_events "
                    "(profile_id, event_type, created_at) VALUES (?, ?, ?)",
                    (profile_id, event_type, now),
                )
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Public API: stats
    # ------------------------------------------------------------------

    def get_stats(self, profile_id: str) -> Dict[str, Any]:
        """
        Return engagement statistics for a profile.

        Returns:
            Dict with keys: recall_count, store_count, delete_count,
            session_count, active_days, total_events, engagement_score.
        """
        conn = self._connect()
        try:
            # Count by event type
            rows = conn.execute(
                "SELECT event_type, COUNT(*) AS cnt "
                "FROM engagement_events WHERE profile_id = ? "
                "GROUP BY event_type",
                (profile_id,),
            ).fetchall()
            counts: Dict[str, int] = {r["event_type"]: r["cnt"] for r in rows}

            recall_count = counts.get("recall", 0)
            store_count = counts.get("store", 0)
            delete_count = counts.get("delete", 0)
            session_count = counts.get("session_start", 0)
            total_events = sum(counts.values())

            # Active days: count distinct dates
            active_days = self._count_active_days(conn, profile_id)

            # Composite engagement score
            score = self._compute_engagement_score(
                recall_count=recall_count,
                store_count=store_count,
                session_count=session_count,
                active_days=active_days,
            )

            return {
                "recall_count": recall_count,
                "store_count": store_count,
                "delete_count": delete_count,
                "session_count": session_count,
                "active_days": active_days,
                "total_events": total_events,
                "engagement_score": round(score, 4),
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Public API: health
    # ------------------------------------------------------------------

    def get_health(self, profile_id: str) -> str:
        """
        Classify engagement health for a profile.

        Uses events in the last 7 days:
            - **active**: >= 10 events
            - **warm**: >= 3 events
            - **cold**: >= 1 event
            - **inactive**: 0 events

        Args:
            profile_id: Profile to check.

        Returns:
            One of ``"active"``, ``"warm"``, ``"cold"``, ``"inactive"``.
        """
        recent = self._count_recent_events(profile_id, days=7)

        if recent >= _ACTIVE_THRESHOLD:
            return "active"
        if recent >= _WARM_THRESHOLD:
            return "warm"
        if recent >= 1:
            return "cold"
        return "inactive"

    # ------------------------------------------------------------------
    # Public API: weekly summary
    # ------------------------------------------------------------------

    def get_weekly_summary(self, profile_id: str) -> Dict[str, Any]:
        """
        Summarise the last 7 days of engagement.

        Returns:
            Dict with period_start, period_end, total_events,
            by_type, active_days.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=7)
        ).isoformat()

        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT event_type, COUNT(*) AS cnt "
                "FROM engagement_events "
                "WHERE profile_id = ? AND created_at >= ? "
                "GROUP BY event_type",
                (profile_id, cutoff),
            ).fetchall()
            by_type = {r["event_type"]: r["cnt"] for r in rows}
            total = sum(by_type.values())

            # Active days in the window
            day_rows = conn.execute(
                "SELECT COUNT(DISTINCT SUBSTR(created_at, 1, 10)) "
                "FROM engagement_events "
                "WHERE profile_id = ? AND created_at >= ?",
                (profile_id, cutoff),
            ).fetchone()
            active_days = day_rows[0] if day_rows else 0

            return {
                "period_start": (date.today() - timedelta(days=6)).isoformat(),
                "period_end": _today_iso(),
                "total_events": total,
                "by_type": by_type,
                "active_days": active_days,
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_active_days(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
    ) -> int:
        """Count distinct dates with at least one event."""
        row = conn.execute(
            "SELECT COUNT(DISTINCT SUBSTR(created_at, 1, 10)) "
            "FROM engagement_events WHERE profile_id = ?",
            (profile_id,),
        ).fetchone()
        return row[0] if row else 0

    def _count_recent_events(self, profile_id: str, days: int = 7) -> int:
        """Count events in the last *days* days."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat()

        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM engagement_events "
                "WHERE profile_id = ? AND created_at >= ?",
                (profile_id, cutoff),
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    @staticmethod
    def _compute_engagement_score(
        recall_count: int,
        store_count: int,
        session_count: int,
        active_days: int,
    ) -> float:
        """
        Compute a composite engagement score in [0.0, 1.0].

        Weighted sum normalised by a soft ceiling:
            raw = 0.4 * recall + 0.3 * store + 0.2 * session + 0.1 * days
            score = raw / (raw + 20)   (sigmoid-like saturation at ~20)

        This gives:
            - 0 activity  -> 0.0
            - 20 events   -> ~0.5
            - 100 events  -> ~0.83
        """
        raw = (
            0.4 * recall_count
            + 0.3 * store_count
            + 0.2 * session_count
            + 0.1 * active_days
        )
        if raw <= 0:
            return 0.0
        return raw / (raw + 20.0)


# ---------------------------------------------------------------------------
# Derive-on-read engagement — zero hot-path cost (Invariants I1, I3)
# ---------------------------------------------------------------------------

def derive_engagement_from_dbs(
    memory_db_path: "Path | str",
    learning_db_path: "Path | str",
    profile_id: str,
) -> "Dict[str, Any]":
    """Derive engagement metrics from tables that already exist — no writes.

    Source tables
    -------------
    memory.db : atomic_facts
        store_count   — live facts (lifecycle in active/warm/cold)
        days_active   — distinct calendar days with at least one live fact
        recent_7d     — facts created in the last 7 days (drives health)
    learning.db : learning_signals
        recall_count  — COUNT(DISTINCT query) for the profile.
                        This is a proxy: one distinct query = one recall
                        session.  If the same query was run N times, it
                        counts as 1.  The table is populated by the
                        post_tool_outcome_hook when Claude Code surfaces
                        recall results; it may be empty in environments
                        without that hook, in which case recall_count = 0
                        while store_count still reflects real activity.

    Invariant compliance
    --------------------
    I1 — zero writes, no lock acquisition on the recall/remember hot path.
    I3 — no new table or row is ever created; bounded by the existing
         lifecycle-management and retention systems (atomic_facts rows age
         through active → warm → cold → archived via Langevin dynamics;
         learning_signals rows are pruned by the retention sweep).
    I6 — every field carries a named source.

    health_status (plain language, non-technical)
    ----------------------------------------------
    "ACTIVE"   — 10 or more memories added in the last 7 days
    "WARM"     —  3–9 memories added in the last 7 days
    "COLD"     —  1–2 memories added in the last 7 days
    "INACTIVE" —  no new memories in the last 7 days
    """
    memory_db_path = Path(str(memory_db_path))
    learning_db_path = Path(str(learning_db_path))

    store_count: int = 0
    days_active: int = 0
    recent_7d: int = 0
    recall_count: int = 0

    # ── memory.db : atomic_facts ──────────────────────────────────────────
    if memory_db_path.exists():
        try:
            # read-only URI; never creates the file, never acquires write lock
            _uri = f"{memory_db_path.resolve().as_uri()}?mode=ro"
            _conn = sqlite3.connect(_uri, uri=True, timeout=1.0)
            _conn.execute("PRAGMA query_only=ON")
            try:
                _tables = {
                    r[0]
                    for r in _conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                if "atomic_facts" in _tables:
                    # Only live facts (archived = soft-deleted / forgotten).
                    # CRIT-fix-3: lifecycle filter avoids counting deleted facts
                    # as "stored"; a profile with 1 000 facts all archived
                    # should not report store_count=1000 to a non-technical user.
                    _r = _conn.execute(
                        "SELECT COUNT(*) FROM atomic_facts "
                        "WHERE profile_id=? AND lifecycle IN ('active','warm','cold')",
                        (profile_id,),
                    ).fetchone()
                    store_count = _r[0] if _r else 0

                    _r = _conn.execute(
                        "SELECT COUNT(DISTINCT SUBSTR(created_at, 1, 10)) "
                        "FROM atomic_facts "
                        "WHERE profile_id=? AND lifecycle IN ('active','warm','cold')",
                        (profile_id,),
                    ).fetchone()
                    days_active = _r[0] if _r else 0

                    _r = _conn.execute(
                        "SELECT COUNT(*) FROM atomic_facts "
                        "WHERE profile_id=? "
                        "AND lifecycle IN ('active','warm','cold') "
                        "AND created_at >= datetime('now', '-7 days')",
                        (profile_id,),
                    ).fetchone()
                    recent_7d = _r[0] if _r else 0
            finally:
                _conn.close()
        except Exception:
            # Any failure (locked, missing, corrupt) → keep zeros;
            # health stays INACTIVE which is the honest fallback.
            pass

    # ── learning.db : learning_signals (recall proxy) ─────────────────────
    if learning_db_path.exists():
        try:
            _uri = f"{learning_db_path.resolve().as_uri()}?mode=ro"
            _conn = sqlite3.connect(_uri, uri=True, timeout=1.0)
            _conn.execute("PRAGMA query_only=ON")
            try:
                _tables = {
                    r[0]
                    for r in _conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                if "learning_signals" in _tables:
                    # CRIT-fix-1: label is "distinct queries" not raw row count.
                    # One query that surfaces 10 facts writes 10 rows; COUNT(*)
                    # would inflate the number by 10×. COUNT(DISTINCT query)
                    # approximates "sessions where you asked for something"
                    # which is the intent of recall_count for non-technical users.
                    _r = _conn.execute(
                        "SELECT COUNT(DISTINCT query) FROM learning_signals "
                        "WHERE profile_id=?",
                        (profile_id,),
                    ).fetchone()
                    recall_count = _r[0] if _r else 0
            finally:
                _conn.close()
        except Exception:
            pass

    # ── health derivation (plain language) ───────────────────────────────
    if recent_7d >= _ACTIVE_THRESHOLD:
        health_status = "ACTIVE"
    elif recent_7d >= _WARM_THRESHOLD:
        health_status = "WARM"
    elif recent_7d >= 1:
        health_status = "COLD"
    else:
        health_status = "INACTIVE"

    total_events = store_count  # recalls add to proxy separately via recall_count
    memories_per_day = (
        round(store_count / days_active, 1) if days_active > 0 else 0
    )
    raw = (
        0.4 * recall_count
        + 0.3 * store_count
        + 0.1 * days_active
    )
    score = (raw / (raw + 20.0)) if raw > 0 else 0.0

    return {
        "health_status": health_status,
        "days_active": days_active,
        "memories_per_day": memories_per_day,
        "total_events": total_events,
        "recall_count": recall_count,
        "store_count": store_count,
        "session_count": 0,          # not derivable without dedicated writes
        "engagement_score": round(score, 4),
        # I6 provenance — every figure is traceable to a named source table
        "source": "memory.db:atomic_facts,learning.db:learning_signals",
    }
