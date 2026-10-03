# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``slm kinds`` — memory kinds from the terminal, through the running daemon.

Every subcommand is a thin client of ``/api/memory-kinds`` (the routes the
dashboard uses), so the CLI and the dashboard cannot disagree:

    slm kinds status
    slm kinds settings [--enable|--disable] [--backend B] [--jev-consent yes|no]
    slm kinds backfill start [--mode untyped|refresh] [--yes]
    slm kinds backfill pause|resume|cancel|revert RUN_ID
    slm kinds backfill status

A run that would send memory text online is refused by the daemon until it is
confirmed; here that confirmation is ``--yes``. ``--json`` everywhere.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, NoReturn

from superlocalmemory.cli.daemon import DaemonConflict, DaemonNotFound, daemon_request

_BASE = "/api/memory-kinds"
_NOT_RUNNING = "The SLM daemon is not running. Start it with: slm serve"


def register_kinds_parser(sub: Any) -> None:
    """Attach the ``kinds`` parser. Called from cli/main.py."""
    p = sub.add_parser("kinds", help="Memory kinds: status, settings, classify my memories")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    ksub = p.add_subparsers(dest="kinds_command", title="kinds subcommands")

    s = ksub.add_parser("status", help="kinds per memory, the backend in use, runs")
    s.add_argument("--json", action="store_true")

    st = ksub.add_parser("settings", help="show or change the memory-kind settings")
    on_off = st.add_mutually_exclusive_group()
    on_off.add_argument("--enable", action="store_true", help="turn memory kinds on")
    on_off.add_argument("--disable", action="store_true", help="turn memory kinds off")
    st.add_argument("--backend", choices=["auto", "rules", "laya", "jev", "llm", "off"])
    st.add_argument("--jev-consent", choices=["yes", "no"],
                    help="allow Jev to type memories (sends memory text online)")
    st.add_argument("--standing-rules", choices=["on", "off"],
                    help="give confirmed standing rules to every new session")
    st.add_argument("--json", action="store_true")

    b = ksub.add_parser("backfill", help="classify existing memories (undoable)")
    bsub = b.add_subparsers(dest="backfill_command", title="backfill actions")
    start = bsub.add_parser("start", help="start a classification run")
    start.add_argument("--mode", choices=["untyped", "refresh"], default="untyped",
                       help="untyped: only memories without a kind; refresh: also redo "
                            "suggestions (never a kind you confirmed)")
    start.add_argument("--yes", action="store_true",
                       help="confirm a run that sends memory text online")
    start.add_argument("--json", action="store_true")
    for action in ("pause", "resume", "cancel", "revert"):
        a = bsub.add_parser(action, help=f"{action} a classification run")
        a.add_argument("run_id")
        a.add_argument("--json", action="store_true")
    bs = bsub.add_parser("status", help="the run in progress, if any")
    bs.add_argument("--json", action="store_true")


@dataclass(frozen=True, slots=True)
class _Out:
    as_json: bool
    command: str

    def fail(self, message: str) -> NoReturn:
        if self.as_json:
            from superlocalmemory.cli.json_output import json_print

            json_print(self.command, error={"message": message})
        else:
            print(message)
        sys.exit(1)

    def emit(self, data: dict, text: str) -> None:
        if self.as_json:
            from superlocalmemory.cli.json_output import json_print

            json_print(self.command, data=data)
        else:
            print(text)


def _request(out: _Out, method: str, path: str, body: dict | None = None) -> dict:
    """The daemon's answer; prints and exits on a refusal or when it is absent."""
    try:
        result = daemon_request(method, _BASE + path, body, preserve_conflict=True,
                                preserve_not_found=True)
    except DaemonConflict as exc:
        needs_yes = path == "/backfill" and "Confirm to continue" in exc.detail
        out.fail(exc.detail + (" Run again with --yes to confirm." if needs_yes else ""))
    except DaemonNotFound:
        out.fail("No such classification run in this profile.")
    if result is None:
        out.fail(_NOT_RUNNING)
    return result


def _run_line(run: dict | None) -> str:
    if not run:
        return "No classification run in progress."
    eta = run.get("eta_seconds")
    eta_text = f", about {int(eta)} s left" if isinstance(eta, (int, float)) else ""
    return (f"Run {run['run_id']} ({run['backend']}, {run['mode']}): {run['status']}, "
            f"{run['processed']}/{run['total_estimate']} looked at, {run['changed']} "
            f"typed{eta_text}.")


def _status_text(status: dict) -> str:
    if not status.get("schema_ready"):
        return status.get("reason") or "Memory kinds are not available yet."
    backend = status["backend"]
    lines = [f"Memory kinds: {'on' if status['enabled'] else 'off'}.",
             f"Backend: {backend['active']} - {backend['reason']}"]
    counts = status["counts"]
    for kind, c in sorted(counts["kind"].items()):
        if c["confirmed"] or c["suggested"]:
            lines.append(f"  {kind}: {c['confirmed']} confirmed, {c['suggested']} suggested")
    by_source = counts.get("by_source") or {}
    if by_source:
        lines.append("  with a kind, by who set it: " + ", ".join(
            f"{src} {n}" for src, n in sorted(by_source.items())))
    lines.append(f"  shown by their old type: {counts['legacy']}, untyped: {counts['untyped']}")
    lines.append(_run_line(status.get("active_run")))
    return "\n".join(lines)


def _settings_body(args: Namespace) -> dict:
    body: dict[str, Any] = {}
    if args.enable or args.disable:
        body["enabled"] = bool(args.enable)
    if args.backend:
        body["backend"] = args.backend
    if args.jev_consent:
        body["jev_consent"] = args.jev_consent == "yes"
    if args.standing_rules:
        body["standing_rules_in_session"] = args.standing_rules == "on"
    return body


def cmd_kinds(args: Namespace) -> None:
    sub = getattr(args, "kinds_command", None) or "status"
    if sub == "backfill":
        _backfill(args)
        return
    out = _Out(bool(getattr(args, "json", False)), f"kinds {sub}")
    if sub == "settings":
        body = _settings_body(args)
        settings = (_request(out, "POST", "/settings", body) if body
                    else _request(out, "GET", "/settings"))
        out.emit(settings, "\n".join(f"{k}: {v}" for k, v in sorted(settings.items())
                                     if not isinstance(v, dict)))
        return
    status = _request(out, "GET", "/status")
    out.emit(status, _status_text(status))


def _backfill(args: Namespace) -> None:
    action = getattr(args, "backfill_command", None) or "status"
    out = _Out(bool(getattr(args, "json", False)), f"kinds backfill {action}")
    if action == "status":
        status = _request(out, "GET", "/status")
        out.emit({"active_run": status.get("active_run")}, _run_line(status.get("active_run")))
        return
    if action == "start":
        run = _request(out, "POST", "/backfill",
                       {"mode": args.mode, "confirm_data_leaves_device": bool(args.yes)})
        out.emit(run, f"Started run {run['run_id']} ({run['backend']}): "
                      f"{run['total_estimate']} memories to look at. Undo it any time "
                      f"with: slm kinds backfill revert {run['run_id']}")
        return
    run = _request(out, "POST", f"/backfill/{args.run_id}/{action}")
    out.emit(run, _run_line(run))


__all__ = ["cmd_kinds", "register_kinds_parser"]
