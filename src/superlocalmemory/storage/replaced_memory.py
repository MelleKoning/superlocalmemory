# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Facts written for a memory its caller already replaced are retired as they are written.

A caller who saves an update with ``replaces`` naming a whole memory
(``core/remember_replaces.py``) retires every fact that memory has at that
moment. Its background enrichment may not have run yet - an agent typically
saves a checkpoint and seconds later saves the update - and enrichment then
derives more facts from the replaced memory. Without this module every one of
them would surface in recall and session context.

WHY HERE. Every fact is written through ``DatabaseManager.store_fact`` - the
materializer, the consolidator, the command line - and that method writes the
fact row, its temporal anchor and its projection intent in one transaction.
It calls ``retire_if_memory_replaced`` inside that same transaction, so a fact
of a replaced memory is never visible un-retired, whenever it is written. A
check on the read side would have to be repeated on every read path; one at the
write is one place.

HOW. Each such fact gets its own correction case (reason ``replaced_by_caller``,
same successor) through the same ledger the replacement used, so it is marked
and snapshotted exactly like the facts retired at replacement time. The
replacement's cases carry a group key in their idempotency key
(``replaces:memory:<group>:``); late cases carry the same group
(``replaces:late:<group>:``). When the last applied case of a replacement is
rolled back, ``restore_late_facts`` rolls its late cases back in the same
transaction, so undoing the replacement brings every derived fact back too.
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Callable

from superlocalmemory.storage.correction_cases import (
    CALLER_REPLACEMENT_REASON,
    CorrectionActor,
    CorrectionAuthorizationError,
    CorrectionCase,
    propose_on_connection,
    transition_on_connection,
)

WHOLE_MEMORY = "replaces:memory:"
#: Reason recorded when the ledger itself refuses a late retirement (for
#: example the successor was re-scoped after the whole-memory case was
#: applied) and this module falls back to retiring the fact directly,
#: without a correction case. Distinct from CALLER_REPLACEMENT_REASON so a
#: reader can tell the two routes apart.
DIRECT_RETIRE_REASON = "replaced_by_caller:direct (ledger refused the late case)"
LATE = "replaces:late:"
SINGLE_FACT = "replaces:fact:"


def group_key(profile_id: str, memory_id: str, successor_fact_id: str) -> str:
    """The id shared by one whole-memory replacement and its late facts."""
    return hashlib.sha256(
        f"{profile_id}\0{memory_id}\0{successor_fact_id}".encode("utf-8")).hexdigest()[:32]


def case_id_for(profile_id: str, fact_id: str, successor_fact_id: str) -> str:
    """Deterministic case id: replaying the same retirement names the same case."""
    return uuid.uuid5(uuid.NAMESPACE_URL,
                      f"slm-replaces:{profile_id}:{fact_id}:{successor_fact_id}").hex


def _group_of(idempotency_key: str, prefix: str) -> str | None:
    if not idempotency_key.startswith(prefix):
        return None
    group, _, _rest = idempotency_key[len(prefix):].partition(":")
    return group or None


def _owner_checks(profile_id: str, actor: CorrectionActor):
    def is_profile(candidate: str) -> bool:
        return candidate == profile_id

    def is_actor(candidate: CorrectionActor) -> bool:
        return candidate == actor

    return is_profile, is_actor


def _direct_retire(conn: sqlite3.Connection, *, fact_id: str, profile_id: str,
                   successor_fact_id: str) -> None:
    """Retire ``fact_id`` on its temporal row directly, bypassing the ledger.

    Used only when the ledger itself refuses the late case -- for example the
    successor was re-scoped after the whole-memory replacement was applied,
    so the case's recorded scope no longer matches. No correction case is
    recorded for this retirement (there is nothing a reviewer could roll
    back to a different outcome), but the fact still carries
    ``system_expired_at`` and so is excluded from recall and session
    context, which is the one guarantee this module exists to make: a fact
    of a replaced memory is never left current.

    ``store_temporal_validity`` always runs before this hook in the same
    transaction (``DatabaseManager.store_fact``), so the row this updates
    already exists with ``system_expired_at IS NULL``.
    """
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "UPDATE fact_temporal_validity SET system_expired_at=?, invalidated_by=?, "
        "invalidation_reason=? WHERE fact_id=? AND profile_id=? AND system_expired_at IS NULL",
        (now, successor_fact_id, DIRECT_RETIRE_REASON, fact_id, profile_id),
    )


def _whole_memory_case(conn: sqlite3.Connection, fact_id: str, memory_id: str,
                       profile_id: str) -> sqlite3.Row | None:
    # Cheap first: does any other fact of this memory carry a caller's mark?
    # Indexed on atomic_facts(memory_id) and the temporal primary key; this is
    # all an ordinary write ever pays.
    marked = [str(r[0]) for r in conn.execute(
        "SELECT tv.fact_id FROM atomic_facts p JOIN fact_temporal_validity tv "
        "ON tv.fact_id = p.fact_id AND tv.profile_id = p.profile_id "
        "WHERE p.memory_id = ? AND p.profile_id = ? AND p.fact_id != ? "
        "AND tv.system_expired_at IS NOT NULL AND tv.invalidation_reason = ?",
        (memory_id, profile_id, fact_id, CALLER_REPLACEMENT_REASON)).fetchall()]
    if not marked:
        return None
    return conn.execute(
        "SELECT successor_fact_id, scope, reviewed_by_actor_id, idempotency_key "
        "FROM correction_cases WHERE profile_id = ? AND status = 'applied' "
        "AND reason_code = ? AND idempotency_key LIKE ? "
        f"AND predecessor_fact_id IN ({','.join('?' for _ in marked)}) "
        "ORDER BY applied_at DESC LIMIT 1",
        (profile_id, CALLER_REPLACEMENT_REASON, WHOLE_MEMORY + "%", *marked)).fetchone()


def retire_if_memory_replaced(conn: sqlite3.Connection, *, fact_id: str, memory_id: str,
                              profile_id: str) -> CorrectionCase | None:
    """Retire a just-written fact if its memory was replaced by its caller.

    Runs inside the transaction that wrote the fact, so it commits or rolls
    back with it. Returns the applied case, or None when nothing applies, or
    when the ledger refused the case and the fact was retired directly
    instead (L1-14): a ledger refusal must never fail the write, and a fact
    of a replaced memory is never left current either way.
    """
    if not memory_id:
        return None
    row = _whole_memory_case(conn, fact_id, memory_id, profile_id)
    if row is None or row["successor_fact_id"] == fact_id:
        return None
    already = conn.execute(
        "SELECT 1 FROM correction_cases WHERE profile_id = ? AND predecessor_fact_id = ? "
        "AND status IN ('proposed', 'applied') LIMIT 1", (profile_id, fact_id)).fetchone()
    if already is not None:  # a re-store of a fact that already has its case
        return None
    group = _group_of(str(row["idempotency_key"]), WHOLE_MEMORY)
    successor = str(row["successor_fact_id"])
    actor = CorrectionActor(actor_id=str(row["reviewed_by_actor_id"]),
                            actor_kind="host_authenticated", trust_tier="trusted")
    is_profile, is_actor = _owner_checks(profile_id, actor)
    case_id = case_id_for(profile_id, fact_id, successor)
    try:
        propose_on_connection(
            conn, case_id=case_id, profile_id=profile_id, scope=str(row["scope"]),
            predecessor_fact_id=fact_id, successor_fact_id=successor,
            reason_code=CALLER_REPLACEMENT_REASON, actor=actor,
            idempotency_key=f"{LATE}{group}:{case_id}",
            is_profile_active=is_profile, is_actor_trusted=is_actor,
        )
    except CorrectionAuthorizationError:
        # Nothing was proposed; retire the fact directly and stop.
        _direct_retire(conn, fact_id=fact_id, profile_id=profile_id,
                       successor_fact_id=successor)
        return None
    try:
        return transition_on_connection(
            conn, case_id=case_id, expected_version=0, actor=actor,
            operation_id=f"replaces:late-apply:{case_id}", from_status="proposed",
            to_status="applied", mutate_temporal=True,
            is_profile_active=is_profile, is_actor_trusted=is_actor,
        )
    except CorrectionAuthorizationError:
        # The case was proposed but the ledger refused to apply it -- most
        # commonly the successor was re-scoped after the whole-memory
        # replacement, so the case's recorded scope no longer matches
        # (correction_cases.py:_apply_predecessor_temporal). Close the
        # orphaned 'proposed' case cleanly (no temporal mutation, so this
        # transition cannot hit the same scope check) rather than leaving it
        # stuck forever, then retire the fact by the one route that does
        # not depend on the ledger's scope agreement.
        transition_on_connection(
            conn, case_id=case_id, expected_version=0, actor=actor,
            operation_id=f"replaces:late-reject:{case_id}", from_status="proposed",
            to_status="rejected", mutate_temporal=False,
            is_profile_active=is_profile, is_actor_trusted=is_actor,
        )
        _direct_retire(conn, fact_id=fact_id, profile_id=profile_id,
                       successor_fact_id=successor)
        return None


def restore_late_facts(conn: sqlite3.Connection, case: CorrectionCase, *,
                       actor: CorrectionActor, operation_id: str,
                       is_profile_active: Callable[[str], bool],
                       is_actor_trusted: Callable[[CorrectionActor], bool]) -> int:
    """After ``case`` was rolled back: once its replacement has no applied case
    left, roll back the facts retired late on its behalf. Same transaction."""
    group = _group_of(case.idempotency_key, WHOLE_MEMORY)
    if group is None:
        return 0
    still_replaced = conn.execute(
        "SELECT 1 FROM correction_cases WHERE profile_id = ? AND status = 'applied' "
        "AND idempotency_key LIKE ? LIMIT 1",
        (case.profile_id, f"{WHOLE_MEMORY}{group}:%")).fetchone()
    if still_replaced is not None:
        return 0
    late = conn.execute(
        "SELECT case_id, version FROM correction_cases WHERE profile_id = ? "
        "AND status = 'applied' AND idempotency_key LIKE ? ORDER BY case_id",
        (case.profile_id, f"{LATE}{group}:%")).fetchall()
    for row in late:
        late_id = str(row[0])
        transition_on_connection(
            conn, case_id=late_id, expected_version=int(row[1]), actor=actor,
            operation_id="replaces:undo:" + hashlib.sha256(
                f"{operation_id}\0{late_id}".encode("utf-8")).hexdigest()[:48],
            from_status="applied", to_status="rolled_back", mutate_temporal=True,
            is_profile_active=is_profile_active, is_actor_trusted=is_actor_trusted,
        )
    return len(late)


__all__ = ["LATE", "SINGLE_FACT", "WHOLE_MEMORY", "case_id_for", "group_key",
           "restore_late_facts", "retire_if_memory_replaced"]
