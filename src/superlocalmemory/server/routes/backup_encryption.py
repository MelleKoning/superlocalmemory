# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Cloud backup encryption routes.

  GET  /api/backup/encryption              key-free status and notices (READ)
  POST /api/backup/recovery-key            show the recovery key (MANAGE)
  POST /api/backup/encryption/acknowledge  dismiss the old-backups notice (MANAGE)

Every /api response is sent with ``Cache-Control: no-store`` by the security
middleware. The recovery key is only returned from a POST, never logged.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger("superlocalmemory.routes.backup_encryption")
router = APIRouter()


def _require_read(request: Request) -> None:
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.rbac_enforce import require_permission

    require_permission(request, Permission.READ)


def _require_manage(request: Request) -> None:
    from superlocalmemory.server.rbac_enforce import require_manage

    require_manage(request)


@router.get("/api/backup/encryption")
def encryption_status_route(request: Request) -> dict:
    _require_read(request)
    from superlocalmemory.infra.backup_keys import encryption_status

    return encryption_status()


@router.post("/api/backup/recovery-key")
def show_recovery_key_route(request: Request) -> dict:
    _require_manage(request)
    from superlocalmemory.infra.backup_keys import (
        BackupKeyError,
        reveal_recovery_key,
    )

    try:
        return reveal_recovery_key()
    except BackupKeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/backup/encryption/acknowledge")
def acknowledge_notice_route(request: Request) -> dict:
    _require_manage(request)
    from superlocalmemory.infra.backup_keys import acknowledge_legacy_notice

    acknowledge_legacy_notice()
    return {"success": True}
