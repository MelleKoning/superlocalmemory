# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``slm db restore-points | restore | prepare-downgrade`` -- the dashboard's
"Updates & restore" card, from a terminal. Same functions, same records.

``slm db restore <id>`` stops SuperLocalMemory, restores, and starts it again
(the start adds back what was written after the copy). If SuperLocalMemory
cannot be stopped, the restore is left waiting and runs at its next start.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from superlocalmemory.storage import upgrade_restore as ur
from superlocalmemory.storage._restore_public import public_view
from superlocalmemory.storage._restore_types import DowngradeRefusedError, RestoreRefusedError


def _paths(args: Namespace) -> tuple[Path, Path, Path]:
    root = getattr(args, "data_root", None)
    if root is None:
        from superlocalmemory.infra.data_root import canonical_data_root

        config = _load_config()
        root = canonical_data_root(configured_base_dir=getattr(config, "base_dir", None))
    root = Path(root)
    return root, root / "memory.db", root / "learning.db"


def _load_config() -> Any | None:
    try:
        from superlocalmemory.core.config import SLMConfig

        return SLMConfig.load()
    except Exception:  # noqa: BLE001 - restore still works; projections are settled at start
        return None


def _daemon_running() -> bool:
    from superlocalmemory.cli.daemon import is_daemon_running

    return is_daemon_running()


def _stop_daemon() -> bool:
    from superlocalmemory.cli.daemon import stop_daemon

    return stop_daemon()


def _start_daemon() -> bool:
    from superlocalmemory.cli.daemon import ensure_daemon

    return ensure_daemon()


def _emit(args: Namespace, payload: dict[str, Any], text: str, root: Path | None = None) -> None:
    """JSON paths are data-directory-relative (L3-21)."""
    if getattr(args, "json", False):
        print(json.dumps(public_view(payload, root) if root else payload, default=str))
    else:
        print(text)


def _confirmed(args: Namespace, word: str) -> bool:
    if getattr(args, "yes", False):
        return True
    if not sys.stdin.isatty():
        # L3-14: this is a refusal, not a prompt -- --json must still get
        # valid JSON here, not plain text with no stdout for a caller parsing
        # it as JSON.
        _emit(args, {"confirmed": False,
                     "message": f"Nothing was changed. Add --yes to confirm "
                               f"(or type {word} when asked)."},
              f"Nothing was changed. Add --yes to confirm (or type {word} when asked).")
        return False
    return input(f"Type {word} to continue: ").strip() == word


def _mb(n: int | None) -> str:
    return f"{(n or 0) / 1_048_576:,.0f} MB"


def cmd_db_restore_points(args: Namespace) -> int:
    root, _m, _l = _paths(args)
    points = ur.list_restore_points(root)
    lines = ["Restore points (copies from the last two updates are kept):"] if points else [
        "No restore points yet. One is made automatically before every update."]
    for p in points:
        versions = f"{p.from_version or '?'} -> {p.to_version or '?'}"
        note = "  (older copy, no checksum)" if p.legacy else ""
        if not p.restorable:
            note += "  (made by a newer version: cannot be restored by this one)"
        lines.append(f"  {p.point_id}  {p.created_at or ''}  {p.reason:<13} {versions:<18} "
                     f"{_mb(p.size_bytes):>9}  {p.facts if p.facts is not None else '?'} "
                     f"memories{note}")
    _emit(args, {"restore_points": [p.as_dict() for p in points]}, "\n".join(lines), root)
    return 0


def _describe(preview: ur.RestorePreview) -> str:
    lines = [f"Restore point {preview.point_id}:",
             f"  memories in the copy: {preview.facts_in_snapshot:,}   now: {preview.facts_now:,}",
             f"  added since (kept aside, added back after): {preview.added_memories:,}",
             f"  deleted since (stay deleted): {preview.deleted_facts:,}",
             f"  erased since (stay erased): {preview.erasures:,}",
             f"  kinds you confirmed (applied again): {preview.kind_edits:,}"]
    if preview.returning_facts:
        lines.append(f"  missing now with no deletion recorded (come back): "
                     f"{preview.returning_facts:,}")
    if preview.learning_records_lost:
        lines.append(f"  learning records since the copy (not carried over): "
                     f"{preview.learning_records_lost:,}")
    if preview.corrections_lost:
        lines.append(f"  corrections opened since (cannot be carried over): "
                     f"{preview.corrections_lost:,}")
    if preview.repair:
        lines.append(f"  also undoes the one-time repair after the update: "
                     f"{preview.repair.get('memories_restored', 0)} memories' text, "
                     f"{preview.repair.get('replaced_marks_undone', 0)} wrong 'replaced' marks")
    lines += [f"  warning: {w}" for w in preview.warnings]
    lines += [f"  problem: {p}" for p in preview.problems]
    return "\n".join(lines)


def cmd_db_restore(args: Namespace) -> int:
    root, memory_db, learning_db = _paths(args)
    if getattr(args, "cancel", False):
        try:
            cancelled = ur.cancel_restore(root)
        except RestoreRefusedError as exc:          # part-way through (L1-03)
            _emit(args, {"cancelled": False, "message": str(exc)}, str(exc), root)
            return 1
        _emit(args, {"cancelled": cancelled},
              "The waiting restore was cancelled." if cancelled else "No restore was waiting.")
        return 0
    point_id = getattr(args, "point_id", None)
    usage = "Usage: slm db restore <restore point> [--yes] [--no-reimport] | --cancel"
    if not point_id:
        _emit(args, {"error": "point_id is required", "usage": usage}, usage)
        return 2
    try:
        preview = ur.preview_restore(point_id, data_root=root, memory_db=memory_db)
    except RestoreRefusedError as exc:
        _emit(args, {"restored": False, "message": str(exc)}, str(exc))
        return 1
    if not getattr(args, "json", False):
        print(_describe(preview))
    if not preview.verified or not preview.disk_ok or not preview.restorable:
        _emit(args, preview.as_dict(), "Nothing was changed.", root)
        return 1
    if not _confirmed(args, "RESTORE"):
        return 2
    try:
        ur.request_restore(point_id, requested_by="cli",
                           reimport=not getattr(args, "no_reimport", False),
                           data_root=root, memory_db=memory_db)
    except RestoreRefusedError as exc:
        _emit(args, {"restored": False, "message": str(exc)}, str(exc))
        return 1
    if _daemon_running() and (not _stop_daemon() or _daemon_running()):
        _emit(args, {"status": "waiting_for_restart"},
              "SuperLocalMemory could not be stopped; the restore runs at its next start.")
        return 0
    outcome = ur.perform_pending_restore(root, memory_db, learning_db, config=_load_config())
    started = _start_daemon()
    payload = {**(outcome.as_dict() if outcome else {"status": "nothing_waiting"}),
               "restarted": started}
    text = outcome.message if outcome else "No restore was waiting."
    if outcome and outcome.reimport_pending and not started:
        text += " Newer memories are added back the next time SuperLocalMemory starts."
    _emit(args, payload, text, root)
    return 0 if outcome and outcome.status == "restored" else 1


def cmd_db_prepare_downgrade(args: Namespace) -> int:
    root, memory_db, learning_db = _paths(args)
    if getattr(args, "cancel", False):
        cancelled = ur.cancel_downgrade(root)
        _emit(args, {"cancelled": cancelled},
              "Cancelled; this version takes the store back at its next start."
              if cancelled else "Nothing was prepared.")
        return 0
    if not _confirmed(args, "DOWNGRADE"):
        return 2
    try:
        report = ur.prepare_downgrade(requested_by="cli", data_root=root,
                                      memory_db=memory_db, learning_db=learning_db)
    except DowngradeRefusedError as exc:
        _emit(args, {"prepared": False, "message": str(exc)}, str(exc))
        return 1
    _emit(args, report.as_dict(), report.message, root)
    return 0


__all__ = ["cmd_db_prepare_downgrade", "cmd_db_restore", "cmd_db_restore_points"]
