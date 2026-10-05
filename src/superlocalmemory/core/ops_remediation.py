# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Operational Recovery & Admin Remediation helpers (resilience slice).

Provides two primary functions used by HTTP endpoints, MCP tools, and CLI:

    list_failed_operations(db_path, profile_id=None) -> dict
        - dead_letter: ingestion ops that exhausted automatic retries (M031)
        - degraded_manifests: completion_manifests in DEGRADED state
        - exhausted_obligations: projection_obligations FAILED with attempts >= MAX

    resolve_operation(db_path, engine, operation_id, action) -> dict
        action ∈ {"retry", "force_reconcile", "cancel"}
        - retry: re-enqueue a dead-lettered ingestion op via IngestionCommand.retry()
        - force_reconcile: immediate obligation redrive (bypasses 30s throttle)
        - cancel: mark op terminally cancelled and remove from failure surfaces

Design constraints (NON-NEGOTIABLE):
    - No schema migration — surface via existing tables only
    - Pure reads in list_failed_operations (no mutation)
    - Additive & backward-compatible — healthy-path unaffected
    - Immutable return dicts; explicit error handling; no silent swallowing

Part of SuperLocalMemory V4 | Operational Recovery
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Maximum automatic materialization/apply attempts (mirrors service.py + ingestion_command.py)
_MAX_ATTEMPTS = 10

_VALID_ACTIONS = frozenset({"retry", "force_reconcile", "cancel"})

# A failed obligation is exhausted when its apply/erase attempts ran out, or --
# for an erasure, which is closed by proof rather than by retrying -- when it
# has been re-proven and found unconfirmed that many times. Two ? = _MAX_ATTEMPTS.
_EXHAUSTED_PREDICATE = (
    "(attempts >= ? OR (kind = 'erase' AND verify_attempts >= ?))"
)

_SYNC_EXHAUSTED = (
    "Background sync failed after maximum retries — use Force Re-sync or Cancel."
)
_ERASE_UNCONFIRMED = (
    "A deletion could not be confirmed: the memory or one of its search "
    "indexes still holds it. Delete the memory again, then use Reconcile; "
    "or Cancel to dismiss."
)


# ---------------------------------------------------------------------------
# Public API — list_failed_operations
# ---------------------------------------------------------------------------


def list_failed_operations(
    db_path: str | Path,
    profile_id: str | None = None,
) -> dict[str, Any]:
    """Collect all failure-surface items from memory.db.

    Returns an immutable-shaped dict:
    {
        "dead_letter": [{"category": "dead_letter", "operation_id": ..., ...}],
        "degraded_manifests": [...],
        "exhausted_obligations": [...],
        "total": int,
    }

    Each entry carries at minimum: category, operation_id, profile_id, error/state,
    attempts (where applicable), and when (unix epoch float).

    Raises nothing — all errors are logged and returned as empty lists.
    """
    db_path = Path(db_path)
    dead_letter: list[dict] = []
    degraded_manifests: list[dict] = []
    exhausted_obligations: list[dict] = []

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            dead_letter = _fetch_dead_letter(conn, profile_id)
            degraded_manifests = _fetch_degraded_manifests(conn, profile_id)
            exhausted_obligations = _fetch_exhausted_obligations(conn, profile_id)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.error("list_failed_operations: DB query failed: %s", exc)

    total = len(dead_letter) + len(degraded_manifests) + len(exhausted_obligations)
    return {
        "dead_letter": dead_letter,
        "degraded_manifests": degraded_manifests,
        "exhausted_obligations": exhausted_obligations,
        "total": total,
    }


def _fetch_dead_letter(
    conn: sqlite3.Connection,
    profile_id: str | None,
) -> list[dict]:
    """Read dead_letter_operations rows (M031 table)."""
    try:
        if profile_id is not None:
            rows = conn.execute(
                "SELECT original_op_id, error, attempt_count, profile_id, "
                "dead_lettered_at FROM dead_letter_operations WHERE profile_id = ? "
                "ORDER BY dead_lettered_at DESC LIMIT 200",
                (profile_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT original_op_id, error, attempt_count, profile_id, "
                "dead_lettered_at FROM dead_letter_operations "
                "ORDER BY dead_lettered_at DESC LIMIT 200",
            ).fetchall()
    except sqlite3.Error:
        # Table may not exist on older DBs — fail gracefully
        return []

    result: list[dict] = []
    for row in rows:
        entry = {
            "category": "dead_letter",
            "operation_id": row["original_op_id"],
            "error": row["error"] or "",
            "attempts": row["attempt_count"] or 0,
            "profile_id": row["profile_id"] or "",
            "when": row["dead_lettered_at"],
            "what_happened": "Ingestion failed after maximum retries — needs manual action.",
        }
        result.append(entry)
    return result


def _fetch_degraded_manifests(
    conn: sqlite3.Connection,
    profile_id: str | None,
) -> list[dict]:
    """Read completion_manifests rows in DEGRADED state (M033 table)."""
    try:
        if profile_id is not None:
            rows = conn.execute(
                "SELECT operation_id, profile_id, state, updated_at "
                "FROM completion_manifests WHERE state = 'DEGRADED' AND profile_id = ? "
                "ORDER BY updated_at DESC LIMIT 200",
                (profile_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT operation_id, profile_id, state, updated_at "
                "FROM completion_manifests WHERE state = 'DEGRADED' "
                "ORDER BY updated_at DESC LIMIT 200",
            ).fetchall()
    except sqlite3.Error:
        return []

    result: list[dict] = []
    for row in rows:
        entry = {
            "category": "degraded_manifest",
            "operation_id": row["operation_id"],
            "state": row["state"],
            "profile_id": row["profile_id"] or "",
            "when": row["updated_at"],
            "what_happened": "Operation completed partially — some projections did not apply.",
        }
        result.append(entry)
    return result


def _fetch_exhausted_obligations(
    conn: sqlite3.Connection,
    profile_id: str | None,
) -> list[dict]:
    """Read projection_obligations rows that are FAILED and exhausted (M033 table).

    Exhausted = state='failed' AND NOT admin-cancelled AND either attempts >=
    _MAX_ATTEMPTS, or (for an erasure) re-proofs >= _MAX_ATTEMPTS. An erasure
    is closed by proof rather than by retrying, so its re-proof count is what
    says it has been unconfirmed for too long. Admin-cancelled obligations
    (detail contains 'admin_cancel') are excluded so they don't resurface.
    """
    where = f"state = 'failed' AND {_EXHAUSTED_PREDICATE}"
    params: list[Any] = [_MAX_ATTEMPTS, _MAX_ATTEMPTS]
    if profile_id is not None:
        where += " AND profile_id = ?"
        params.append(profile_id)
    try:
        rows = conn.execute(
            "SELECT operation_id, MAX(profile_id) AS profile_id, "
            "MAX(attempts) AS attempts, MAX(updated_at) AS updated_at, "
            "GROUP_CONCAT(DISTINCT kind) AS kinds, "
            "SUM(CASE WHEN detail LIKE '%admin_cancel%' THEN 1 ELSE 0 END) AS cancelled "
            f"FROM projection_obligations WHERE {where} "
            "GROUP BY operation_id "
            "ORDER BY updated_at DESC LIMIT 200",
            tuple(params),
        ).fetchall()
    except sqlite3.Error:
        return []

    result: list[dict] = []
    for row in rows:
        # Skip admin-cancelled obligations
        if row["cancelled"]:
            continue
        kinds = set((row["kinds"] or "").split(","))
        erase_only = kinds == {"erase"}
        entry = {
            "category": "exhausted_obligation",
            "operation_id": row["operation_id"],
            "kind": "erase" if erase_only else ",".join(sorted(kinds)),
            "attempts": row["attempts"] or 0,
            "profile_id": row["profile_id"] or "",
            "when": row["updated_at"],
            "what_happened": _ERASE_UNCONFIRMED if erase_only else _SYNC_EXHAUSTED,
        }
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Public API — resolve_operation
# ---------------------------------------------------------------------------


def resolve_operation(
    db_path: str | Path,
    engine: Any,
    operation_id: str,
    action: str,
) -> dict[str, Any]:
    """Resolve a failed/stuck operation.

    Parameters
    ----------
    db_path
        Path to memory.db (used for DLQ + obligation queries).
    engine
        The running engine instance (may be None for cancel-only operations).
    operation_id
        The operation to act upon.
    action
        One of: "retry", "force_reconcile", "cancel".

    Returns a dict: {"success": bool, "action": str, "operation_id": str, ...}
    Raises ValueError for invalid action.
    """
    if action not in _VALID_ACTIONS:
        raise ValueError(
            f"invalid action {action!r}; must be one of {sorted(_VALID_ACTIONS)}"
        )

    db_path = Path(db_path)

    if action == "cancel":
        return _action_cancel(db_path, operation_id)
    if action == "retry":
        return _action_retry(db_path, engine, operation_id)
    if action == "force_reconcile":
        return _action_force_reconcile(db_path, engine, operation_id)

    # Unreachable — guarded by _VALID_ACTIONS check above
    raise ValueError(f"unhandled action {action!r}")  # pragma: no cover


def _action_cancel(db_path: Path, operation_id: str) -> dict:
    """Cancel an operation: remove from DLQ, mark obligations as admin-cancelled."""
    found = False
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            # Remove from dead_letter_operations
            cursor = conn.execute(
                "DELETE FROM dead_letter_operations WHERE original_op_id = ?",
                (operation_id,),
            )
            if cursor.rowcount > 0:
                found = True

            # Mark exhausted obligations as cancelled (so they don't resurface)
            import json
            cancel_detail = json.dumps({"admin_cancel": True, "cancelled_at": time.time()})
            cursor2 = conn.execute(
                "UPDATE projection_obligations SET state = 'failed', "
                "detail = ?, attempts = MAX(attempts, ?) "
                "WHERE operation_id = ? AND state = 'failed'",
                (cancel_detail, _MAX_ATTEMPTS, operation_id),
            )
            if cursor2.rowcount > 0:
                found = True

            # Mark degraded manifest if present
            cursor3 = conn.execute(
                "UPDATE completion_manifests SET state = 'FAILED' "
                "WHERE operation_id = ? AND state = 'DEGRADED'",
                (operation_id,),
            )
            if cursor3.rowcount > 0:
                found = True

            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.error("resolve_operation cancel %s: %s", operation_id, exc)
        return {
            "success": False,
            "action": "cancel",
            "operation_id": operation_id,
            "reason": str(exc),
        }

    if not found:
        return {
            "success": False,
            "action": "cancel",
            "operation_id": operation_id,
            "reason": "not_found — operation not in any failure surface",
        }

    return {
        "success": True,
        "action": "cancel",
        "operation_id": operation_id,
        "message": "Operation cancelled. It will no longer appear in the attention list.",
    }


def _action_retry(db_path: Path, engine: Any, operation_id: str) -> dict:
    """Retry a dead-lettered ingestion operation via IngestionCommand.retry()."""
    # Check that the operation exists in DLQ
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT id FROM dead_letter_operations WHERE original_op_id = ? LIMIT 1",
                (operation_id,),
            ).fetchone()
        finally:
            conn.close()
    except Exception as exc:
        logger.error("resolve_operation retry DLQ check %s: %s", operation_id, exc)
        return {
            "success": False,
            "action": "retry",
            "operation_id": operation_id,
            "reason": str(exc),
        }

    if row is None:
        return {
            "success": False,
            "action": "retry",
            "operation_id": operation_id,
            "reason": "not_found — operation not in dead-letter queue",
        }

    # Attempt retry via IngestionCommand
    if engine is not None:
        try:
            # Build a fully-wired ingestion command from the live engine
            # (db + queryable writer + materializer). Constructing
            # IngestionCommand directly would miss the required
            # write_queryable/materialize collaborators.
            from superlocalmemory.core.engine_ingestion import (
                build_engine_ingestion_command,
            )
            from superlocalmemory.core.ingestion_command import IngestionState
            cmd = build_engine_ingestion_command(engine)
            operation = cmd.retry(operation_id)
            # G-10: only remove from DLQ if the retry actually moved the
            # operation out of FAILED state.  If it is still FAILED, leave
            # the DLQ row so it surfaces again on the next /ops list call
            # and the operator can investigate.
            if operation.state is IngestionState.FAILED:
                return {
                    "success": False,
                    "action": "retry",
                    "operation_id": operation_id,
                    "reason": (
                        "retry_still_failed — operation remains in FAILED state;"
                        " DLQ row preserved for further investigation"
                    ),
                }
            # Operation moved out of FAILED → remove from DLQ
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    "DELETE FROM dead_letter_operations WHERE original_op_id = ?",
                    (operation_id,),
                )
                conn.commit()
            finally:
                conn.close()
            return {
                "success": True,
                "action": "retry",
                "operation_id": operation_id,
                "message": f"Operation re-queued for retry (state: {operation.state.value}).",
            }
        except Exception as exc:
            logger.error("resolve_operation retry %s: %s", operation_id, exc)
            return {
                "success": False,
                "action": "retry",
                "operation_id": operation_id,
                "reason": str(exc),
            }
    else:
        # No engine — best effort: just clear the DLQ row so it doesn't resurface
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    "DELETE FROM dead_letter_operations WHERE original_op_id = ?",
                    (operation_id,),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("resolve_operation retry (no engine) %s: %s", operation_id, exc)
        return {
            "success": True,
            "action": "retry",
            "operation_id": operation_id,
            "message": "DLQ entry cleared. Daemon will retry on next cycle.",
        }


def _action_force_reconcile(db_path: Path, engine: Any, operation_id: str) -> dict:
    """Force an immediate projection reconcile for this operation_id."""
    if engine is None:
        return {
            "success": False,
            "action": "force_reconcile",
            "operation_id": operation_id,
            "reason": "engine not available — daemon must be running for force_reconcile",
        }

    try:
        from superlocalmemory.core.transactions.concrete_owners import (
            build_transaction_service,
        )
        from superlocalmemory.server.unified_daemon import _context_for_operation

        db_obj = getattr(engine, "_db", None)
        if db_obj is None:
            return {
                "success": False,
                "action": "force_reconcile",
                "operation_id": operation_id,
                "reason": "engine._db not available",
            }

        # An erasure id has no ingestion row by design: re-prove it instead of
        # reporting "no canonical record". Erase work goes first so a manifest
        # written by the apply reconcile below already reflects it.
        erasure = (
            _force_reconcile_erasure(engine, operation_id)
            if _has_erase_obligations(db_obj, operation_id) else None
        )

        context = _context_for_operation(engine, operation_id)
        if context is None:
            return erasure or {
                "success": False,
                "action": "force_reconcile",
                "operation_id": operation_id,
                "reason": "not_found — no canonical record for this operation",
            }

        service = build_transaction_service(engine)
        service.reconcile_operation(db_obj, context)
        if erasure is not None and not erasure["success"]:
            return erasure
        return {
            "success": True,
            "action": "force_reconcile",
            "operation_id": operation_id,
            "message": "Reconcile triggered immediately (30s throttle bypassed).",
        }
    except Exception as exc:
        logger.error("resolve_operation force_reconcile %s: %s", operation_id, exc)
        return {
            "success": False,
            "action": "force_reconcile",
            "operation_id": operation_id,
            "reason": str(exc),
        }


def _has_erase_obligations(db_obj: Any, operation_id: str) -> bool:
    return bool(db_obj.execute(
        "SELECT 1 FROM projection_obligations "
        "WHERE operation_id = ? AND kind = 'erase' LIMIT 1",
        (operation_id,),
    ))


def _force_reconcile_erasure(engine: Any, operation_id: str) -> dict:
    """Re-prove an erasure now. It closes only if nothing of the memory remains."""
    from superlocalmemory.core.transactions.erase_redrive import (
        reconcile_erase_operation,
    )

    outcome = reconcile_erase_operation(engine, operation_id)
    if outcome.closed:
        return {
            "success": True,
            "action": "force_reconcile",
            "operation_id": operation_id,
            "message": "Deletion confirmed: nothing of the memory remains.",
        }
    return {
        "success": False,
        "action": "force_reconcile",
        "operation_id": operation_id,
        "reason": f"deletion not confirmed: {outcome.reason}",
    }


# ---------------------------------------------------------------------------
# Counts helpers (for /health and /status surface)
# ---------------------------------------------------------------------------


def get_failure_counts(db_path: str | Path) -> dict[str, int]:
    """Fast count query for /health and /status surface.

    Returns {"dead_letter_count": int, "degraded_operations": int,
             "exhausted_obligations": int}.
    All counts default to 0 on error (never raises).
    """
    db_path = Path(db_path)
    counts = {"dead_letter_count": 0, "degraded_operations": 0, "exhausted_obligations": 0}
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            try:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM dead_letter_operations"
                ).fetchone()
                counts["dead_letter_count"] = int(row["c"]) if row else 0
            except sqlite3.Error:
                pass

            try:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM completion_manifests WHERE state = 'DEGRADED'"
                ).fetchone()
                counts["degraded_operations"] = int(row["c"]) if row else 0
            except sqlite3.Error:
                pass

            try:
                row = conn.execute(
                    "SELECT COUNT(DISTINCT operation_id) AS c "
                    "FROM projection_obligations "
                    f"WHERE state = 'failed' AND {_EXHAUSTED_PREDICATE} "
                    "AND (detail IS NULL OR detail NOT LIKE '%admin_cancel%')",
                    (_MAX_ATTEMPTS, _MAX_ATTEMPTS),
                ).fetchone()
                counts["exhausted_obligations"] = int(row["c"]) if row else 0
            except sqlite3.Error:
                pass
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.debug("get_failure_counts: %s", exc)

    return counts


__all__ = [
    "list_failed_operations",
    "resolve_operation",
    "get_failure_counts",
]
