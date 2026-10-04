# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Removing what SLM installed for the on-device check — and only that.

``remove()`` deletes SLM's own install under ``runtime_dir()``: a finished one
(``.slm-managed``) and equally one that never finished (``.install-steps.json``
or ``install.lock`` with a partial environment and download). The second case
used to answer "No managed install to remove." and leave the half-download in
place with no way to clear it from the dashboard.

An install someone else made is never deleted. Its record (``adopted.json``,
which lives in the same folder) is kept when it still names a real install
outside the folder, so removing SLM's copy does not forget theirs;
``forget_adopted()`` is the separate, explicit way to stop using it.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger("superlocalmemory.core.laya_runtime")

#: Any one of these says the folder holds a setup SLM started.
_SLM_EVIDENCE = (".slm-managed", ".install-steps.json", "install.lock")
_ADOPTED = "adopted.json"


def _runtime():
    from superlocalmemory.core import laya_runtime

    return laya_runtime


def _refusal(run_dir: Path, base: Path) -> str:
    """Why ``run_dir`` must not be deleted from, or "" when it may be."""
    try:
        resolved_run_dir = run_dir.resolve(strict=False)
        resolved_base = base.resolve(strict=False)
    except OSError as exc:
        return f"Couldn't resolve the install path: {exc}"
    try:
        resolved_run_dir.relative_to(resolved_base)
    except ValueError:
        return "Refusing to remove: the install path is outside the data directory."
    if resolved_run_dir.parts[-2:] != ("runtimes", "laya"):
        return "Refusing to remove: unexpected install path."
    check = run_dir
    while True:
        if check.is_symlink():
            return "Refusing to remove: a symlink is in the way."
        if check == base or check.parent == check:
            return ""
        check = check.parent


def _adopted_worth_keeping(run_dir: Path) -> bool:
    """An adopted record that names a real install outside SLM's folder."""
    lr = _runtime()
    record = lr._read_json(run_dir / _ADOPTED)
    python = str((record or {}).get("python") or "")
    return bool(python) and Path(python).exists() and not lr._inside(python, run_dir)


def remove(slm_home: Path | None = None):
    """Delete SLM's own install, finished or not. Never an adopted one."""
    lr = _runtime()
    run_dir = lr.runtime_dir(slm_home)
    if not any((run_dir / name).exists() for name in _SLM_EVIDENCE):
        return lr.LayaRuntimeStatus(state=lr.STATE_FAILED, error="No managed install to remove.")

    base = Path(slm_home) if slm_home is not None else lr.canonical_data_root()
    refused = _refusal(run_dir, base)
    if refused:
        return lr.LayaRuntimeStatus(state=lr.STATE_FAILED, error=refused)

    try:
        if not _adopted_worth_keeping(run_dir):
            shutil.rmtree(run_dir)
            return lr.LayaRuntimeStatus(state=lr.STATE_NOT_INSTALLED)
        for entry in run_dir.iterdir():
            if entry.name == _ADOPTED:
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    except OSError as exc:
        return lr.LayaRuntimeStatus(state=lr.STATE_FAILED,
                                    error=f"Couldn't remove the install: {exc}")
    return lr.LayaRuntimeStatus(state=lr.STATE_NOT_INSTALLED)


def forget_adopted(slm_home: Path | None = None) -> bool:
    """Stop using an install someone else made: drop SLM's record of it.

    Only the record goes; the install's own files are never touched. True when
    there was a record to drop.
    """
    lr = _runtime()
    record = lr.runtime_dir(slm_home) / _ADOPTED
    if record.is_symlink() or not record.is_file():
        return False
    try:
        record.unlink()
    except OSError:
        logger.warning("Laya: couldn't forget the adopted install record")
        return False
    return True


__all__ = ["forget_adopted", "remove"]
