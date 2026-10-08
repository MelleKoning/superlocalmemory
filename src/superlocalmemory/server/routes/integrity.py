# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Store integrity over HTTP: the health picture and ``slm db repair``.

The repair runs here when SLM is running, so the daemon stays the only writer
of its store. It refuses unless the caller names this daemon's own data root:
a repair must never land on a store nobody meant.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/integrity", tags=["integrity"])


class RepairRequest(BaseModel):
    root: str = Field(..., min_length=1, max_length=4096)
    undo_run_id: str = Field("", max_length=64)
    batch_size: int = Field(100, ge=1, le=5000)
    pause_ms: int = Field(50, ge=0, le=10_000)
    max_seconds: float | None = Field(None, gt=0)


def _db_path(request: Request) -> Path:
    engine = getattr(request.app.state, "engine", None)
    db = getattr(engine, "_db", None)
    if db is not None:
        return Path(db.db_path)
    from superlocalmemory.server.routes.helpers import DB_PATH

    return Path(DB_PATH)


def _manage(request: Request) -> None:
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server import write_identity
    from superlocalmemory.server.rbac_enforce import require_permission

    write_identity.require_write_actor(
        request, getattr(request.app.state, "daemon_descriptor", None),
        actor_kind="integrity-repair")
    require_permission(request, Permission.MANAGE)


@router.get("")
def get_integrity(request: Request, pages: bool = False) -> dict[str, Any]:
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.rbac_enforce import require_permission
    from superlocalmemory.storage.integrity_health import health

    require_permission(request, Permission.READ)
    return health(_db_path(request), pages=pages)


@router.post("/repair")
def post_repair(request: Request, body: RepairRequest) -> dict[str, Any]:
    from superlocalmemory.infra.data_root import canonical_data_root
    from superlocalmemory.storage.integrity_repair import Limits, Repair, RepairBusy

    _manage(request)
    own_root = canonical_data_root().resolve()
    if Path(body.root).expanduser().resolve() != own_root:
        raise HTTPException(409, detail=f"this SLM serves {own_root}, not {body.root}; "
                                        "nothing was changed")
    repair = Repair(_db_path(request), limits=Limits(
        batch_size=body.batch_size, pause_s=body.pause_ms / 1000.0,
        max_seconds=body.max_seconds), engine=getattr(request.app.state, "engine", None))
    try:
        if body.undo_run_id:
            return {"undone": body.undo_run_id, "restored": repair.undo(body.undo_run_id),
                    "ran_in": "daemon"}
        return {**repair.apply(), "ran_in": "daemon"}
    except RepairBusy as exc:
        raise HTTPException(409, detail=str(exc)) from None
    except Exception:
        logger.exception("integrity repair failed")
        raise HTTPException(500, detail="repair failed; what finished is receipted, run it "
                                        "again to continue") from None


def _store_check_args(request: Request) -> tuple[Path, Path, str]:
    from superlocalmemory import __version__
    from superlocalmemory.infra.data_root import canonical_data_root

    return _db_path(request), canonical_data_root(), __version__


@router.get("/summary")
def get_store_check(request: Request) -> dict[str, Any]:
    """The last store check and repair, for the dashboard. Reads a small file."""
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.rbac_enforce import require_permission
    from superlocalmemory.storage import store_check

    require_permission(request, Permission.READ)
    return {**store_check.read_state(_store_check_args(request)[1]),
            "labels": store_check.FINDINGS}


@router.post("/check", status_code=202)
def post_store_check(request: Request) -> dict[str, Any]:
    """Check the store again in the background. Read-only."""
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.rbac_enforce import require_permission
    from superlocalmemory.storage import store_check

    require_permission(request, Permission.READ)
    if not store_check.start_check(*_store_check_args(request)):
        raise HTTPException(409, detail="a check or repair is already running")
    return {"started": "check"}


@router.post("/repair-now", status_code=202)
def post_store_repair(request: Request) -> dict[str, Any]:
    """The dashboard's Repair now: backup copy, repair, check again."""
    from superlocalmemory.storage import store_check

    _manage(request)
    db_path, root, version = _store_check_args(request)
    if not store_check.start_repair(db_path, root, version,
                                    engine=getattr(request.app.state, "engine", None)):
        raise HTTPException(409, detail="a check or repair is already running")
    return {"started": "repair"}

