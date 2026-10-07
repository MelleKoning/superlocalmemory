# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm corrections overtaken`` and ``slm corrections restore-overtaken``.

    slm corrections overtaken [--limit N] [--profile P] [--json]
    slm corrections restore-overtaken CASE_ID [--profile P] [--json]

When you delete, replace or edit a memory, a correction SLM proposed by itself
and nobody reviewed no longer blocks you: it is closed as "overtaken by a user
action" and kept. The first command lists those cases (ids, what you did, when,
and whether each can still come back); the second puts one back exactly as it
was. A case cannot come back once one of its memories is deleted, or while the
memory has another open correction (your own edit, until you reject it).

Both run inside SLM (it stays the only writer of its store), like
``slm review-correction``. Ids only are printed, never memory text.
"""

from __future__ import annotations

import sys
import urllib.parse
from argparse import Namespace
from typing import Any


def register_corrections_parsers(sub: Any) -> None:
    """Attach ``slm corrections``. Called from cli/main.py."""
    p = sub.add_parser("corrections", help="Corrections a delete, replace or edit of yours "
                                           "closed: list them, put one back")
    csub = p.add_subparsers(dest="corrections_command")
    ls = csub.add_parser("overtaken", help="List correction cases your own actions closed")
    ls.add_argument("--limit", type=int, default=50, help="How many, newest first (1-500)")
    ls.add_argument("--profile", default="", help="Profile (default: the active one)")
    ls.add_argument("--json", action="store_true", help="machine-readable output")
    rs = csub.add_parser("restore-overtaken", help="Put an overtaken correction case back")
    rs.add_argument("case_id", help="Case id from slm corrections overtaken")
    rs.add_argument("--profile", default="", help="Profile (default: the active one)")
    rs.add_argument("--json", action="store_true", help="machine-readable output")


def _out(command: str, use_json: bool, *, data: dict | None = None,
         error: dict | None = None) -> None:
    from superlocalmemory.cli.json_output import json_print

    if use_json:
        json_print(command, data=data, error=error)


def _refused(command: str, message: str, use_json: bool, code: str = "CONFLICT") -> int:
    if use_json:
        _out(command, True, error={"code": code, "message": message, "retryable": False})
    else:
        print(f"Refused: {message}", file=sys.stderr)
    return 1


def _render(rows: list[dict]) -> None:
    if not rows:
        print("No correction case was overtaken by your actions.")
        return
    for r in rows:
        state = ("restored" if r.get("restored_at") else
                 "can be restored" if r.get("restorable") else
                 f"cannot be restored: {r.get('not_restorable_because')}")
        print(f"{r['case_id']}  {r['user_action']:7}  {r['overtaken_at']}  "
              f"{r['predecessor_fact_id']} -> {r['successor_fact_id']}  ({state})")


def _daemon() -> bool:
    from superlocalmemory.cli.daemon import ensure_daemon, is_daemon_running

    return bool(is_daemon_running() or ensure_daemon())


def cmd_corrections(args: Namespace) -> int:
    sub = getattr(args, "corrections_command", None)
    use_json = bool(getattr(args, "json", False))
    if sub not in ("overtaken", "restore-overtaken"):
        print("Usage: slm corrections overtaken | slm corrections restore-overtaken CASE_ID",
              file=sys.stderr)
        return 2
    from superlocalmemory.cli.commands import _daemon_unavailable
    from superlocalmemory.cli.daemon import DaemonConflict, DaemonNotFound, daemon_request

    command = f"corrections {sub}"
    profile = (getattr(args, "profile", "") or "").strip()
    if sub == "overtaken" and not 1 <= int(args.limit) <= 500:
        return _refused(command, "--limit must be from 1 to 500", use_json, "INVALID")
    if sub == "restore-overtaken":
        from superlocalmemory.core.admission import gate_cli_mutation
        from superlocalmemory.core.operation_request import OperationKind

        gate_cli_mutation(OperationKind.CORRECT)
    if not _daemon():
        _daemon_unavailable(command, use_json)
        return 1
    try:
        if sub == "overtaken":
            path = f"/api/overtaken-corrections?limit={int(args.limit)}"
            if profile:
                path += "&profile_id=" + urllib.parse.quote(profile, safe="")
            result = daemon_request("GET", path, preserve_not_found=True)
        else:
            path = ("/api/overtaken-corrections/" + urllib.parse.quote(args.case_id, safe="")
                    + "/restore")
            result = daemon_request("POST", path, {"profile_id": profile} if profile else {},
                                    preserve_conflict=True, preserve_not_found=True)
    except DaemonConflict as exc:
        return _refused(command, exc.detail, use_json)
    except DaemonNotFound as exc:  # an unknown --profile
        return _refused(command, exc.error_message or exc.message, use_json,
                        (exc.error_code or exc.code).upper())
    if not isinstance(result, dict) or not result.get("success"):
        _daemon_unavailable(command, use_json)
        return 1
    if use_json:
        _out(command, True, data=result)
    elif sub == "overtaken":
        _render(list(result.get("overtaken") or []))
    else:
        print(f"Restored correction case {result['restored']} ({result['events']} event(s)); "
              "it is waiting for review again.")
    return 0


def run(args: Namespace) -> None:
    """The ``slm corrections`` entry point: exits with the command's status."""
    rc = cmd_corrections(args)
    if rc:
        sys.exit(rc)


__all__ = ["cmd_corrections", "register_corrections_parsers", "run"]
