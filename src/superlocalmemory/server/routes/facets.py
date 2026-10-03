# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""GET /api/v3/facets - the projects and agents memories were saved under.

What a person or an agent can narrow a recall by (``project``, ``saved_by``),
with how many memories each holds, for the active profile. Reads only.

Permission: READ on the active profile, like every other content-bearing
read route (memories, facts, recall). Without this gate an anonymous or
under-permissioned caller in a company-mode workspace could enumerate every
project name and every agent id that ever saved a memory here — exactly the
cross-tenant metadata the other sensitive read routes already withhold.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from superlocalmemory.access.rbac import Permission
from superlocalmemory.retrieval.facets import list_facets

router = APIRouter(prefix="/api/v3/facets", tags=["facets"])


@router.get("")
def get_facets(request: Request, limit: int = 50):
    from superlocalmemory.server.rbac_enforce import require_permission

    require_permission(request, Permission.READ)
    engine = getattr(request.app.state, "engine", None)
    db = getattr(engine, "_db", None) if engine is not None else None
    if db is None:
        raise HTTPException(503, detail="memory engine is not ready; retry shortly")
    profile_id = getattr(engine, "profile_id", None) or getattr(engine, "_profile_id", "default")
    return {"profile": profile_id, **list_facets(db, profile_id, limit=max(1, min(int(limit), 200)))}
