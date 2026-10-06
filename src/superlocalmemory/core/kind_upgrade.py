# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Classify an upgraded store's older memories once, on this device, by itself.

A store from before 4.1.19 has facts with no memory kind. 4.1.21 typed them
only when someone started "Classify my memories" by hand. Now the runner
thread queues that run itself, at most ONCE per profile, ever:

* never when the profile already had a run of any kind and state - one that
  is running or paused is resumed or left as it is (observed, not duplicated);
  a completed one is never repeated; a cancelled or undone one stays the
  person's decision;
* never when typing would send memory text off the device (that always needs
  the person's confirmation), or when memory kinds are turned off;
* never when nothing is untyped.

The run is the ordinary resumable, pausable, undoable run, so everything the
dashboard and ``slm kinds backfill`` offer applies to it.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

REQUESTED_BY = "automatic-upgrade"

#: What each outcome means, as the status card says it.
MEANINGS = {
    "queued": "Older memories are being classified on this device, in the background.",
    "observed": "A classification run for this profile is already in progress; it continues.",
    "already_classified": "This profile's memories were already classified; it is not repeated.",
    "not_needed": "Every memory already has a kind.",
    "needs_confirmation": "Classifying would send memory text online; start it from the "
                          "dashboard to confirm.",
    "disabled": "Memory kinds are turned off in Settings.",
    "schema_not_ready": "This update's database step has not run yet.",
    "failed": "The automatic classification could not be queued; it is retried at the next "
              "start.",
}


def _profiles(db: Any) -> list[str]:
    try:
        return [str(dict(r)["profile_id"]) for r in db.execute(
            "SELECT profile_id FROM profiles ORDER BY profile_id")]
    except Exception:  # noqa: BLE001 - a store without the profiles table
        return ["default"]


def ensure_profile_typed(db: Any, store: Any, choice: Any, enabled: bool,
                         profile_id: str) -> dict[str, Any]:
    """Queue this profile's one automatic run if it needs one; say what happened."""
    from superlocalmemory.core import memory_kind_backfill_plan as plan
    from superlocalmemory.core import memory_kind_runs as runs

    def said(state: str, run: dict | None = None, **extra: Any) -> dict[str, Any]:
        return {"state": state, "run_id": (run or {}).get("run_id"),
                "meaning": MEANINGS[state], **extra}

    if not db.has_memory_kind_columns():
        return said("schema_not_ready")
    if not enabled or choice.active == "off":
        return said("disabled")
    active = runs.active_run(db, profile_id)
    if active is not None:
        return said("observed", active, status=active["status"])
    previous = store.recent_runs(profile_id, limit=1)
    if previous:
        return said("already_classified", previous[0], status=previous[0]["status"])
    untyped = plan.count_candidates(db, profile_id, "untyped")
    if untyped == 0:
        return said("not_needed")
    if choice.leaves_device:
        return said("needs_confirmation", untyped=untyped)
    run = store.create_run_once(profile_id, backend=choice.active,
                                recipe_id=plan.recipe_for(choice.active),
                                requested_by=REQUESTED_BY, total_estimate=untyped)
    if run is None:  # another process or a person queued one first
        current = runs.active_run(db, profile_id) or (store.recent_runs(profile_id, limit=1)
                                                      or [None])[0]
        return said("observed" if current else "failed", current)
    logger.info("Queued the one automatic classification of %d older memories", untyped)
    return said("queued", run, untyped=untyped)


def ensure_store_typed(db: Any, store: Any, choice: Any, enabled: bool) -> dict[str, dict]:
    """``ensure_profile_typed`` for every profile; a failure is a state, never raised."""
    out: dict[str, dict] = {}
    for profile_id in _profiles(db):
        try:
            out[profile_id] = ensure_profile_typed(db, store, choice, enabled, profile_id)
        except Exception as exc:  # noqa: BLE001 - the runner must keep running
            logger.warning("Automatic classification check failed: %s", type(exc).__name__)
            out[profile_id] = {"state": "failed", "run_id": None, "meaning": MEANINGS["failed"]}
    return out


__all__ = ["MEANINGS", "REQUESTED_BY", "ensure_profile_typed", "ensure_store_typed"]
