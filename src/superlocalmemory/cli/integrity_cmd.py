# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm db integrity`` and ``slm db repair``.

    slm db integrity [--pages] [--json]            health, five sections, read-only
    slm db repair [--json]                         what a repair would do, read-only
    slm db repair --apply --root PATH [--batch-size N] [--pause-ms MS] [--max-seconds S]
    slm db repair --undo RUN_ID --root PATH

``--apply`` and ``--undo`` change the store, so they insist on ``--root``: the
data folder you mean, which must be the one this command resolves to (from
``SLM_DATA_DIR`` or the default). Anything else is refused before a byte is
written. With SLM running the repair runs inside it (it stays the only
writer); otherwise here. Counts and ids only are printed, never memory text.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any


def register_db_integrity_parsers(db_sub: Any) -> None:
    """Attach ``slm db integrity`` and ``slm db repair``. Called from cli/main.py."""
    h = db_sub.add_parser("integrity", help="Store health: pages, relations, fidelity, "
                                            "projections, repair (read-only)")
    h.add_argument("--pages", action="store_true", help="Also check every page (slow)")
    h.add_argument("--json", action="store_true", help="machine-readable output")
    r = db_sub.add_parser("repair", help="Fix orphan rows, erased-text leftovers and settled "
                                         "obligations, with receipts (preview by default)")
    r.add_argument("--apply", action="store_true", help="Make the changes the preview lists")
    r.add_argument("--undo", default="", metavar="RUN_ID", help="Put back what a run changed")
    r.add_argument("--root", default="", help="The data folder to change (required to change)")
    r.add_argument("--batch-size", type=int, default=200, help="Rows per short write")
    r.add_argument("--pause-ms", type=int, default=50, help="Pause between writes")
    r.add_argument("--max-seconds", type=float, default=None,
                   help="Stop after this long; run again to continue")
    r.add_argument("--json", action="store_true", help="machine-readable output")


def _root() -> Path:
    from superlocalmemory.infra.data_root import canonical_data_root

    return canonical_data_root().resolve()


def _emit(command: str, data: dict, use_json: bool, render) -> None:
    if use_json:
        from superlocalmemory.cli.json_output import json_print

        json_print(command, data=data)
    else:
        render(data)


def _fail(command: str, code: str, message: str, use_json: bool) -> int:
    if use_json:
        from superlocalmemory.cli.json_output import json_print

        json_print(command, error={"code": code, "message": message, "retryable": False})
    else:
        print(f"Refused: {message}", file=sys.stderr)
    return 2 if code == "WRONG_ROOT" else 1


def _render_health(data: dict) -> None:
    for section, body in data.items():
        print(f"{section}:")
        for key, value in body.items():
            print(f"  {key}: {json.dumps(value, sort_keys=True)}")


def _render_plan(data: dict) -> None:
    print(f"Store: {data['root']}")
    for o in data["plan"]["orphans"]:
        if o["rows"]:
            print(f"  {o['action']:6} {o['rows']:>7}  {o['table']} -> {o['parent']}: {o['why']}")
    for key in ("parentless_facts", "unreachable_vectors", "erased_text", "keyword_index",
                "failed_obligations"):
        print(f"  {key}: {json.dumps(data['plan'][key], sort_keys=True)}")
    if "summary" in data:
        s = data["summary"]
        print(f"Run {s['run_id']}: {s['status']} in {s['seconds']} s, {s['batches']} writes, "
              f"longest write {s['max_lock_hold_ms']} ms")
        print(f"  done: {json.dumps(s['done'], sort_keys=True)}")
        print(f"  undo with: slm db repair --undo {s['run_id']} --root {data['root']}")


def cmd_db_integrity(args: Namespace) -> int:
    from superlocalmemory.storage.integrity_health import health

    db_path = _root() / "memory.db"
    if not db_path.exists():
        return _fail("db integrity", "NO_STORE", f"no store at {db_path}", args.json)
    _emit("db integrity", health(db_path, pages=bool(args.pages)), args.json, _render_health)
    return 0


def _via_daemon(body: dict) -> dict | None:
    from superlocalmemory.cli.daemon import daemon_request, is_daemon_running

    if not is_daemon_running():
        return None
    result = daemon_request("POST", "/api/integrity/repair", body, timeout_seconds=6 * 3600,
                            preserve_conflict=True)
    if not isinstance(result, dict):
        raise RuntimeError("SLM did not answer the repair request")
    return result


def cmd_db_repair(args: Namespace) -> int:
    import sqlite3

    from superlocalmemory.storage.integrity_scan import plan

    root = _root()
    db_path = root / "memory.db"
    if not db_path.exists():
        return _fail("db repair", "NO_STORE", f"no store at {db_path}", args.json)
    if not (args.apply or args.undo):
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
        try:
            data = {"root": str(root), "plan": plan(conn), "changed": False}
        finally:
            conn.close()
        _emit("db repair", data, args.json, _render_plan)
        return 0
    named = Path(args.root).expanduser().resolve() if args.root else None
    if named != root:
        return _fail("db repair", "WRONG_ROOT",
                     f"this would change {root}; pass --root {root} to confirm "
                     f"(got {args.root or 'no --root'}). Nothing was changed.", args.json)
    body = {"root": str(root), "undo_run_id": args.undo, "batch_size": args.batch_size,
            "pause_ms": args.pause_ms, "max_seconds": args.max_seconds}
    from superlocalmemory.cli.daemon import DaemonConflict

    try:
        result = _via_daemon(body)
    except DaemonConflict as exc:
        return _fail("db repair", "WRONG_ROOT", exc.detail, args.json)
    if result is None:
        from superlocalmemory.storage.integrity_repair import Limits, Repair

        repair = Repair(db_path, limits=Limits(args.batch_size, args.pause_ms / 1000.0,
                                               args.max_seconds))
        result = ({"undone": args.undo, "restored": repair.undo(args.undo)} if args.undo
                  else repair.apply())
        result["ran_in"] = "this process (SLM is not running)"
    if args.undo:
        _emit("db repair", {"root": str(root), **result}, args.json,
              lambda d: print(f"Undone {d['undone']}: {json.dumps(d['restored'], sort_keys=True)}"))
        return 0
    data = {"root": str(root), "plan": result["after"], "summary": result, "changed": True}
    _emit("db repair", data, args.json, _render_plan)
    return 0 if result["status"] in ("finished", "stopped") else 1


__all__ = ["cmd_db_integrity", "cmd_db_repair", "register_db_integrity_parsers"]
