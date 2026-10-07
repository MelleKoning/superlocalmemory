# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Correction cases a user's action overtook: see them, put one back.

When a user deletes, replaces or edits a memory, a correction that SLM itself
proposed and nobody reviewed no longer stands in the way: it is closed as
"overtaken by a user action" (core/overtaken_cases.py). These routes are the
undo surface for that: ``slm corrections overtaken`` and
``slm corrections restore-overtaken CASE_ID`` call them, and the MCP
``list_corrections`` tool gets the same list through ``GET /api/corrections``.

Authorized exactly like listing and reviewing corrections: the caller's role
on the profile named (or the active one), before that profile's existence is
revealed. Another profile's case is "not found", like a missing one. Ids and
lifecycle metadata only; no memory text is ever stored or returned.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["corrections"])

_MAX_LIMIT = 500


def overtaken_for(engine: Any, profile_id: str, limit: int = 100) -> list[dict]:
    """The profile's overtaken cases, newest first (shared with /api/corrections)."""
    from superlocalmemory.core.overtaken_cases import listing

    return listing(engine._db, profile_id, max(1, min(int(limit), _MAX_LIMIT)))


@router.get("/api/overtaken-corrections")
async def list_overtaken(request: Request, limit: int = 100, profile_id: str = ""):
    from superlocalmemory.server.routes import memories as m

    try:
        profile = m._routed_profile(profile_id)
        engine, target, _ctx = m._authorize_memory_mutation(
            request, "update", "overtaken-list", run_pre_hook=False, profile=profile)
        return {"success": True, "profile_id": target,
                "overtaken": overtaken_for(engine, target, limit)}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Overtaken correction list error")


@router.post("/api/overtaken-corrections/{case_id}/restore")
async def restore_overtaken(request: Request, case_id: str):
    from superlocalmemory.core.overtaken_cases import RestoreRefused, restore
    from superlocalmemory.server.routes import memories as m

    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        profile = m._routed_profile(body.get("profile_id") if isinstance(body, dict) else None)
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", case_id, run_pre_hook=False, profile=profile)
        try:
            with engine._db.transaction():
                result = restore(engine._db, case_id, profile_id=target,
                                 actor_id=str(ctx.get("agent_id") or ""))
        except RestoreRefused as exc:
            raise HTTPException(409, detail=f"{exc}. Nothing was changed.") from None
        from superlocalmemory.core.mutations import purge_profile_context_cache

        purge_profile_context_cache(engine, target)  # a pending case withholds its successor
        m._announce("memory.updated", {"case_id": case_id, "profile_id": target,
                                       "status": "restored"}, ctx["agent_id"])
        logger.info("overtaken correction case %s restored in %s", case_id[:12], target)
        return {"success": True, "profile_id": target, **result}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Overtaken correction restore error")


__all__ = ["overtaken_for", "router"]
