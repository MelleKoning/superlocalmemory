# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Background redrive for erase obligations.

The 30 s projection redrive used to treat every pending operation as an
ingestion. Erasure ids have no ingestion row by design, so their obligations
were failed as orphans and, after ten passes, reported as exhausted. This
module owns the erase side of that redrive.

An erase obligation is closed by proof, never by retrying: it becomes ERASED
only when the memory is gone from the canonical store, its tombstone exists,
and the owning projection holds nothing for it. Proof is read-only; nothing
here deletes or restores data. When proof fails the obligation is marked
FAILED with the reason, and once that has happened ``MAX_REPROOFS`` times the
erasure is reported by ``slm ops status`` instead of sitting silently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from superlocalmemory.core.transactions.erasure import MAX_ERASE_ATTEMPTS
from superlocalmemory.core.transactions.obligations import Obligation, ObligationLedger
from superlocalmemory.core.transactions.owners import (
    ObligationKind,
    ObligationState,
    OperationContext,
    is_terminal_success,
)

logger = logging.getLogger(__name__)

# Failed re-proofs after which an unfinished erasure counts as exhausted.
MAX_REPROOFS = MAX_ERASE_ATTEMPTS

REASON_STILL_STORED = "the memory is still stored"
REASON_NO_TOMBSTONE = "no deletion record exists for it"


@dataclass(frozen=True, slots=True)
class EraseOutcome:
    """Result of re-proving one erasure: closed, or the first reason it is not."""

    operation_id: str
    closed: bool
    reason: str = ""


def reconcile_pending_erasures(engine: Any, *, limit: int = 20) -> int:
    """Re-prove this profile's unfinished erasures. Returns how many closed."""
    db = getattr(engine, "_db", None)
    profile_id = getattr(engine, "_profile_id", None)
    if db is None or not profile_id:
        return 0
    ledger = ObligationLedger()
    with db.raw_connection() as conn:
        op_ids = ledger.unfinished_erase_operation_ids(
            conn, profile_id=profile_id, limit=limit,
        )
    closed = 0
    for operation_id in op_ids:
        try:
            closed += int(reconcile_erase_operation(engine, operation_id).closed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("erase redrive failed for %s: %s", operation_id, exc)
    return closed


def reconcile_erase_operation(engine: Any, operation_id: str) -> EraseOutcome:
    """Re-prove every unfinished erase obligation of one operation."""
    from superlocalmemory.core.transactions.concrete_owners import (
        build_erasure_service,
    )

    db = engine._db
    profile_id = engine._profile_id
    ledger = ObligationLedger()
    with db.raw_connection() as conn:
        obligations = ledger.fetch(conn, operation_id)
    unfinished = [o for o in obligations if _needs_proof(o, profile_id)]
    if not unfinished:
        mine = [
            o for o in obligations
            if o.kind is ObligationKind.ERASE and o.profile_id == profile_id
        ]
        if mine and all(is_terminal_success(o.state) for o in mine):
            return EraseOutcome(operation_id, closed=True)
        return EraseOutcome(
            operation_id, closed=False,
            reason="it is not an open deletion in this profile",
        )
    has_apply = any(o.kind is ObligationKind.APPLY for o in obligations)
    service = build_erasure_service(engine)
    # One erase request normally names one fact; group by subject so a group
    # that cannot be proven never blocks one that can.
    groups: dict[str, list[Obligation]] = {}
    for ob in unfinished:
        groups.setdefault(ob.subject_id, []).append(ob)
    reasons = [
        _reconcile_group(db, ledger, service, operation_id, profile_id, obs, has_apply)
        for obs in groups.values()
    ]
    first_reason = next((r for r in reasons if r), "")
    return EraseOutcome(operation_id, closed=not first_reason, reason=first_reason)


def _needs_proof(ob: Obligation, profile_id: str) -> bool:
    return (
        ob.kind is ObligationKind.ERASE
        and ob.profile_id == profile_id
        and not is_terminal_success(ob.state)
        and not _admin_cancelled(ob)
    )


def _admin_cancelled(ob: Obligation) -> bool:
    detail = ob.detail or {}
    return bool(detail.get("admin_cancel") or detail.get("admin_cancelled"))


def _reconcile_group(
    db: Any, ledger: ObligationLedger, service: Any, operation_id: str,
    profile_id: str, obs: list[Obligation], has_apply: bool,
) -> str:
    """Close one subject's obligations, or record why not. Returns the reason."""
    subject_id = obs[0].subject_id
    # Fact-shaped erasures only: entity and profile erasures name a subject that
    # is not a fact id, so the tombstone check fails them closed with a reason.
    context = OperationContext(
        operation_id=operation_id,
        profile_id=profile_id,
        subject_id=subject_id,
        fact_ids=(subject_id,),
    )
    reason, proven = _unproven_reason(db, service, context, obs)
    if reason:
        _record_unproven(db, ledger, obs, reason)
        return reason
    _close(db, ledger, operation_id, profile_id, proven, has_apply)
    return ""


def _unproven_reason(
    db: Any, service: Any, context: OperationContext, obs: list[Obligation],
) -> tuple[str, list[tuple[str, str | None]]]:
    """Read-only proof. Returns ``("", proofs)`` or ``(reason, [])``."""
    from superlocalmemory.core.transactions.erasure import is_tombstoned

    if db.execute(
        "SELECT 1 FROM atomic_facts WHERE fact_id = ? AND profile_id = ? LIMIT 1",
        (context.subject_id, context.profile_id),
    ):
        return REASON_STILL_STORED, []
    with db.raw_connection() as conn:
        if not is_tombstoned(conn, context.profile_id, context.subject_id):
            return REASON_NO_TOMBSTONE, []
    proven: list[tuple[str, str | None]] = []
    for ob in obs:
        proof = service.prove_erased(context, ob.owner)
        if not proof.erased:
            return f"the {ob.owner} index still holds it", []
        proven.append((ob.owner, proof.checksum or None))
    return "", proven


def _record_unproven(
    db: Any, ledger: ObligationLedger, obs: list[Obligation], reason: str,
) -> None:
    """Mark each obligation FAILED with the reason, unless it changed meanwhile.

    Bumps ``verify_attempts`` (a re-proof), never ``attempts`` (an erase): the
    redrive did not try to erase anything. The compare-and-set matters while a
    delete is still in flight: if the erasure service closed an owner after
    the snapshot was read, its ERASED stands.
    """
    detail = {"phase": "redrive", "error": f"deletion not confirmed: {reason}"}
    with db.raw_connection() as conn:
        for ob in obs:
            ledger.mark_if_unchanged(
                conn, ob, ObligationState.FAILED,
                detail=detail, bump_verify_attempts=True,
            )


def _close(
    db: Any, ledger: ObligationLedger, operation_id: str, profile_id: str,
    proven: list[tuple[str, str | None]], has_apply: bool,
) -> None:
    """Mark the proven owners ERASED and re-seal an existing manifest, atomically.

    No manifest is created: manifests describe ingestions, and an erasure's
    durable record is its receipt. One already present is re-sealed in the
    same transaction so it cannot disagree with the ledger -- for an
    operation that also carries apply work, and for the FAILED manifest the
    4.1.17-4.1.20 orphan path wrote for erasure ids.
    """
    from superlocalmemory.core.transactions.reconciler import Reconciler

    with db.raw_connection() as conn:
        for owner, checksum in proven:
            ledger.mark(
                conn, operation_id, owner, ObligationKind.ERASE,
                ObligationState.ERASED,
                checksum=checksum,
                detail={"phase": "redrive-proven"},
            )
        has_manifest = conn.execute(
            "SELECT 1 FROM completion_manifests WHERE operation_id = ?",
            (operation_id,),
        ).fetchone() is not None
        if has_manifest:
            # Erase-only: the canonical side of an erasure is the delete, and
            # it was just proven. With apply work, ingestion state decides.
            Reconciler(ledger).reconcile(
                conn, operation_id, profile_id,
                canonical_committed=None if has_apply else True,
            )


__all__ = [
    "MAX_REPROOFS",
    "EraseOutcome",
    "reconcile_erase_operation",
    "reconcile_pending_erasures",
]
