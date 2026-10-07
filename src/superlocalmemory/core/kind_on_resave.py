# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A kind declared on a save whose words are already stored.

Identical words are one fact (``DatabaseManager.store_fact`` folds a second
save onto the fact already holding them). 4.1.21 then dropped the kind the
second save declared and reported success, so a caller who declared "rule"
was told it was saved while the memory stayed a "status" - silently.

What happens now, decided for 4.1.22:

* the stored fact has NO confirmed kind (untyped, a legacy label, or a machine
  suggestion): the caller's declaration confirms it, exactly as if it had been
  declared the first time. The change is recorded in ``memory_kind_history``
  with the caller as actor and ``caller`` as the new source, so it is
  auditable and never mistaken for a run's suggestion;
* the stored fact already has the SAME confirmed kind: nothing changes;
* the stored fact already has a DIFFERENT confirmed kind: the first
  confirmation is kept (one caller cannot silently retype what another
  confirmed) and the save says so with ``kind_conflict: {kept, requested}``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from superlocalmemory.storage.memory_kinds import (
    COARSE,
    KindSource,
    MemoryKind,
    is_confirmed,
    parse_kind,
)

logger = logging.getLogger(__name__)

_HISTORY_SQL = (
    "INSERT INTO memory_kind_history (fact_id, profile_id, run_id, origin, old_kind, "
    "old_source, old_confidence, old_fact_type, old_recipe, old_at, new_kind, new_source, "
    "new_confidence, new_fact_type, actor, changed_at) "
    "VALUES (?,?,NULL,'user_edit',?,?,?,?,?,?,?,?,NULL,?,?,?)"
)


def _stored_kind(db: Any, fact_id: str, profile_id: str) -> dict[str, Any] | None:
    rows = db.execute(
        "SELECT memory_kind, memory_kind_source, memory_kind_confidence, "
        "memory_kind_recipe, memory_kind_at, fact_type FROM atomic_facts "
        "WHERE fact_id = ? AND profile_id = ?",
        (fact_id, profile_id),
    )
    return dict(rows[0]) if rows else None


def confirm_on_resave(
    db: Any, *, fact_id: str, profile_id: str, declared: MemoryKind, actor: str,
) -> None:
    """Confirm ``declared`` on the folded fact ``fact_id`` if nothing confirmed it.

    Runs inside the save's own write transaction, so the confirmation commits
    with the save or not at all. A fact that already carries a confirmed kind
    (the same or another) is left exactly as it is.
    """
    stored = _stored_kind(db, fact_id, profile_id)
    if stored is None or is_confirmed(stored.get("memory_kind_source")):
        return
    now = datetime.now(UTC).isoformat()
    new_fact_type = COARSE[declared]
    db.execute(
        "UPDATE atomic_facts SET memory_kind = ?, memory_kind_source = ?, "
        "memory_kind_confidence = NULL, memory_kind_recipe = 'caller', memory_kind_at = ?, "
        "fact_type = ? WHERE fact_id = ? AND profile_id = ?",
        (declared.value, KindSource.CALLER.value, now, new_fact_type, fact_id, profile_id),
    )
    db.execute(_HISTORY_SQL, (
        fact_id, profile_id,
        stored.get("memory_kind"), stored.get("memory_kind_source"),
        stored.get("memory_kind_confidence"), stored.get("fact_type"),
        stored.get("memory_kind_recipe"), stored.get("memory_kind_at"),
        declared.value, KindSource.CALLER.value, new_fact_type, actor or "caller", now,
    ))


def kind_conflict(
    db: Any, *, fact_ids: list[str], profile_id: str, declared: str | None,
) -> dict[str, str] | None:
    """``{"kept", "requested"}`` when the saved fact keeps another confirmed kind.

    Read back after the save, so it reports what is stored, not what was
    meant to happen. ``None`` when no kind was declared, nothing was saved,
    or the stored kind is the declared one.
    """
    requested = parse_kind(declared) if declared else None
    if requested is None or not fact_ids:
        return None
    try:
        stored = _stored_kind(db, str(fact_ids[0]), profile_id)
    except Exception as exc:  # noqa: BLE001 - a read-back never fails a saved write
        logger.warning("could not read back the saved kind (%s)", type(exc).__name__)
        return None
    if stored is None or not is_confirmed(stored.get("memory_kind_source")):
        return None
    kept = str(stored.get("memory_kind") or "")
    if kept == requested.value:
        return None
    return {"kept": kept, "requested": requested.value}


__all__ = ["confirm_on_resave", "kind_conflict"]
