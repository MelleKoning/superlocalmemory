# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Background redrive for erase obligations.

The 30 s projection redrive used to treat every pending operation as an
ingestion. Erasure ids have no ingestion row by design, so their obligations
were failed as orphans and, after ten passes, reported as exhausted. This
module owns the erase side of that redrive.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def pending_obligation_kinds(db, operation_id: str) -> set[str]:
    """Kinds with non-terminal obligations for one operation.

    The redrive fans out by kind: ``apply`` belongs to the ingestion
    reconciler, ``erase`` to the erasure reconciler below. Reading kinds
    first is what stops an erasure id (no ingestion row by design) from
    being misread as an orphaned ingestion.
    """
    from superlocalmemory.core.transactions.owners import ObligationState

    try:
        rows = db.execute(
            "SELECT DISTINCT kind FROM projection_obligations "
            "WHERE operation_id = ? AND state NOT IN (?, ?)",
            (operation_id, str(ObligationState.VERIFIED), str(ObligationState.ERASED)),
        )
    except Exception as exc:  # noqa: BLE001
        # Fail open for visibility: an empty set skips this op for one pass
        # only (it stays pending and is retried in 30s), but must be loud.
        logger.warning(
            "erase/apply kind lookup failed for %s: %s", operation_id, exc,
        )
        return set()
    return {str(dict(r).get("kind")) for r in rows}


def reconcile_erase_operation(engine, db, ledger, operation_id: str) -> int:
    """Drive one erase-kind obligation set toward ERASED without failing it.

    Read-only re-proof: an obligation is marked ERASED (no attempt bump)
    only when its fact is canonically absent, its tombstone is present, and
    its owner proves no residue. Anything else is left pending for the
    erasure service — this path never marks FAILED and never bumps attempts,
    so a slow purge cannot be mistaken for an orphan.
    """
    from superlocalmemory.core.transactions.concrete_owners import (
        build_erasure_service,
    )
    from superlocalmemory.core.transactions.owners import (
        ObligationKind,
        ObligationState,
    )

    with db.raw_connection() as conn:
        pending = [
            o for o in ledger.fetch(conn, operation_id)
            if o.kind == ObligationKind.ERASE
            and o.state not in (ObligationState.VERIFIED, ObligationState.ERASED)
        ]
    if not pending:
        return 0
    service = build_erasure_service(engine)
    done = 0
    # Group by (profile, subject): one erase request normally names one fact,
    # but entity/profile erasures fan out. A group that cannot be proven is
    # left pending without affecting the groups that can.
    groups: dict[tuple[str, str], list] = {}
    for ob in pending:
        groups.setdefault((ob.profile_id, ob.subject_id), []).append(ob)
    for (profile_id, subject_id), obs in groups.items():
        done += _reconcile_erase_group(
            db, ledger, service, operation_id, profile_id, subject_id, obs,
        )
    return 1 if done == len(groups) and groups else 0


def _reconcile_erase_group(
    db, ledger, service, operation_id: str,
    profile_id: str, subject_id: str, obs: list,
) -> int:
    """Prove and close one erase group. Returns 1 iff the group completed."""
    from superlocalmemory.core.transactions.erasure import is_tombstoned
    from superlocalmemory.core.transactions.owners import (
        ObligationKind,
        ObligationState,
        OperationContext,
    )

    # NOTE: fact-shaped erasures only. Entity/profile erasures name subjects
    # that are not fact_ids and carry multi-fact contexts this redrive cannot
    # reconstruct; the tombstone gate below fails those closed (pending).
    if db.execute(
        "SELECT 1 FROM atomic_facts WHERE fact_id = ? AND profile_id = ? LIMIT 1",
        (subject_id, profile_id),
    ):
        return 0
    with db.raw_connection() as conn:
        if not is_tombstoned(conn, profile_id, subject_id):
            return 0
    context = OperationContext(
        operation_id=operation_id,
        profile_id=profile_id,
        subject_id=subject_id,
        fact_ids=(subject_id,),
    )
    # Collect proofs before opening the write transaction: proofs are
    # read-only, and a mark must never interleave with a later proof read.
    proven: list[tuple[str, str | None]] = []
    for ob in obs:
        try:
            proof = service.prove_erased(context, ob.owner)
        except Exception:  # noqa: BLE001
            return 0
        if not proof.erased:
            return 0
        proven.append((ob.owner, proof.checksum or None))
    with db.raw_connection() as conn:
        for owner, checksum in proven:
            ledger.mark(
                conn, operation_id, owner, ObligationKind.ERASE,
                ObligationState.ERASED,
                checksum=checksum,
                detail={"phase": "redrive-proven"},
            )
        # Close the loop for the missing-manifest feed: erase ops never had
        # a manifest writer, so without this they would be re-fetched every
        # 30s forever and crowd out ingestion redrives.
        from superlocalmemory.core.transactions.reconciler import Reconciler

        Reconciler(ledger).reconcile(
            conn, operation_id, profile_id, canonical_committed=False,
        )
    return 1


__all__ = ["pending_obligation_kinds", "reconcile_erase_operation"]
