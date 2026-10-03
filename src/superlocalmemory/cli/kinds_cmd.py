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
    slm kinds set FACT_ID KIND
    slm kinds review [--kind K] [--limit N]
    slm kinds confirm FACT_ID[=KIND] ...

A run that would send memory text online is refused by the daemon until it is
confirmed; here that confirmation is ``--yes``. ``--json`` everywhere.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, NoReturn

from superlocalmemory.cli.daemon import (
    DaemonConflict,
    DaemonNotFound,
    DaemonUnprocessable,
    daemon_request,
)
from superlocalmemory.core.kind_query import InvalidKind, resolve_kind

_BASE = "/api/memory-kinds"
_NOT_RUNNING = "The SLM daemon is not running. Start it with: slm serve"


def _json_flag(parser: Any) -> None:
    """``--json`` that never clobbers an already-set parent value.

    L3-14: argparse writes every action's default into the namespace before
    parsing a subparser's own arguments, so a bare ``store_true`` default of
    False on a nested parser overwrote the top level's True the moment a
    caller wrote the global flag before the subcommand
    (``slm kinds --json status``) rather than after it
    (``slm kinds status --json``) — even though both are documented as
    equivalent. ``default=SUPPRESS`` means "say nothing" instead of "say
    False" when this particular parser's own flag was not given, so parsing
    continues up the chain to whatever the enclosing parser already set.
    """
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="machine-readable output")


def register_kinds_parser(sub: Any) -> None:
    """Attach the ``kinds`` parser. Called from cli/main.py."""
    p = sub.add_parser("kinds", help="Memory kinds: status, settings, classify my memories")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    ksub = p.add_subparsers(dest="kinds_command", title="kinds subcommands")

    s = ksub.add_parser("status", help="kinds per memory, the backend in use, runs")
    _json_flag(s)

    st = ksub.add_parser("settings", help="show or change the memory-kind settings")
    on_off = st.add_mutually_exclusive_group()
    on_off.add_argument("--enable", action="store_true", help="turn memory kinds on")
    on_off.add_argument("--disable", action="store_true", help="turn memory kinds off")
    st.add_argument("--backend", choices=["auto", "rules", "laya", "jev", "llm", "off"])
    st.add_argument("--jev-consent", choices=["yes", "no"],
                    help="allow Jev to type memories (sends memory text online)")
    st.add_argument("--standing-rules", choices=["on", "off"],
                    help="give confirmed standing rules to every new session")
    _json_flag(st)

    b = ksub.add_parser("backfill", help="classify existing memories (undoable)")
    # L3-14: ``backfill`` itself had no --json of its own, so
    # ``slm kinds backfill --json`` (no action word -- _backfill() already
    # treats a missing one as "status") was an argparse usage error, not a
    # recognised invocation -- exactly the second argv Hermes generates.
    _json_flag(b)
    bsub = b.add_subparsers(dest="backfill_command", title="backfill actions")
    start = bsub.add_parser("start", help="start a classification run")
    start.add_argument("--mode", choices=["untyped", "refresh"], default="untyped",
                       help="untyped: only memories without a kind; refresh: also redo "
                            "suggestions (never a kind you confirmed)")
    start.add_argument("--yes", action="store_true",
                       help="confirm a run that sends memory text online")
    _json_flag(start)
    for action in ("pause", "resume", "cancel", "revert"):
        a = bsub.add_parser(action, help=f"{action} a classification run")
        a.add_argument("run_id")
        _json_flag(a)
    bs = bsub.add_parser("status", help="the run in progress, if any")
    _json_flag(bs)

    set_p = ksub.add_parser("set", help="set (confirm) one memory's kind")
    set_p.add_argument("fact_id", help="exact fact id, from recall or list")
    set_p.add_argument("kind", help="one of the nine memory kinds, or a known alias")
    _json_flag(set_p)

    review_p = ksub.add_parser("review", help="suggestions awaiting confirmation")
    review_p.add_argument("--kind", default="", help="only suggestions of this kind")
    review_p.add_argument("--limit", type=int, default=20)
    _json_flag(review_p)

    confirm_p = ksub.add_parser("confirm", help="confirm kinds for 1-200 facts at once")
    confirm_p.add_argument(
        "items", nargs="*",
        help="FACT_ID or FACT_ID=KIND (bare FACT_ID accepts the stored suggestion)",
    )
    _json_flag(confirm_p)


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
                                preserve_not_found=True, preserve_unprocessable=True)
    except DaemonConflict as exc:
        needs_yes = path == "/backfill" and "Confirm to continue" in exc.detail
        out.fail(exc.detail + (" Run again with --yes to confirm." if needs_yes else ""))
    except DaemonNotFound as exc:
        # L3-11: say what was actually not found (e.g. "Memory not found" for
        # an unknown fact_id) -- not a fixed phrase written for one caller
        # (an unknown classification run) and reused for every 404 here.
        out.fail(exc.message)
    except DaemonUnprocessable as exc:
        # L3-11: a 422 is invalid input, refused before any work -- exit 2,
        # never collapsed to None and reported as "the daemon is not
        # running" (exit 1).
        _unprocessable_exit(out, exc.code, exc.message)
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


def _invalid_kind_exit(out: "_Out", message: str) -> NoReturn:
    """Exit 2: the same code `remember`/`recall`/`list` use for a kind that
    never parsed. Refused before any daemon request — never a reason to
    retry."""
    if out.as_json:
        from superlocalmemory.cli.json_output import json_print

        json_print(out.command, error={"code": "INVALID_KIND", "message": message})
    else:
        print(message, file=sys.stderr)
    sys.exit(2)


def _invalid_item_exit(out: "_Out", message: str) -> NoReturn:
    if out.as_json:
        from superlocalmemory.cli.json_output import json_print

        json_print(out.command, error={"code": "INVALID_ITEMS", "message": message})
    else:
        print(message, file=sys.stderr)
    sys.exit(2)


def _unprocessable_exit(out: "_Out", code: str, message: str) -> NoReturn:
    """Exit 2: the daemon's own 422 (L3-11) — invalid input, refused before
    any work, never "the daemon is not running" (what a collapsed-to-None
    422 used to print, exit 1)."""
    if out.as_json:
        from superlocalmemory.cli.json_output import json_print

        json_print(out.command, error={"code": code or "INVALID_REQUEST", "message": message})
    else:
        print(message, file=sys.stderr)
    sys.exit(2)


def _set(args: Namespace) -> None:
    out = _Out(bool(getattr(args, "json", False)), "kinds set")
    try:
        parsed = resolve_kind(args.kind)
    except InvalidKind as exc:
        _invalid_kind_exit(out, str(exc))
    if parsed is None:
        _invalid_kind_exit(out, "a kind is required")
    from urllib.parse import quote

    fact = _request(out, "PATCH", "/fact/" + quote(args.fact_id, safe=""), {"kind": parsed})
    out.emit(fact, "\n".join(f"{k}: {v}" for k, v in sorted(fact.items())))


def _review(args: Namespace) -> None:
    out = _Out(bool(getattr(args, "json", False)), "kinds review")
    try:
        parsed = resolve_kind(getattr(args, "kind", ""))
    except InvalidKind as exc:
        _invalid_kind_exit(out, str(exc))
    from urllib.parse import quote

    qs = f"?limit={int(getattr(args, 'limit', 20))}"
    if parsed:
        qs += f"&kind={quote(parsed)}"
    result = _request(out, "GET", "/suggestions" + qs)
    items = result.get("items") or []
    lines = [
        f"{item.get('fact_id')}: {item.get('memory_kind')} ({item.get('memory_kind_state')})"
        for item in items
    ]
    out.emit(result, "\n".join(lines) if lines else "No suggestions.")


def _parse_confirm_item(out: "_Out", raw: str) -> dict:
    fact_id, _, kind_raw = raw.partition("=")
    fact_id = fact_id.strip()
    if not fact_id:
        _invalid_item_exit(out, f"not a FACT_ID[=KIND]: {raw!r}")
    entry: dict[str, Any] = {"fact_id": fact_id}
    if kind_raw.strip():
        try:
            entry["kind"] = resolve_kind(kind_raw)
        except InvalidKind as exc:
            _invalid_kind_exit(out, str(exc))
    return entry


def _confirm(args: Namespace) -> None:
    out = _Out(bool(getattr(args, "json", False)), "kinds confirm")
    raw_items = list(getattr(args, "items", None) or [])
    if not raw_items:
        _invalid_item_exit(
            out, "confirm needs at least one FACT_ID or FACT_ID=KIND",
        )
    parsed_items = [_parse_confirm_item(out, raw) for raw in raw_items]
    result = _request(out, "POST", "/confirm", {"items": parsed_items})
    items = result.get("items") or []
    lines = [
        f"{item.get('fact_id')}: "
        + ("ok (" + str(item.get("memory_kind")) + ")" if item.get("ok")
           else item.get("error", "failed"))
        for item in items
    ]
    out.emit(result, "\n".join(lines) if lines else "No items.")


def cmd_kinds(args: Namespace) -> None:
    sub = getattr(args, "kinds_command", None) or "status"
    if sub == "backfill":
        _backfill(args)
        return
    if sub == "set":
        _set(args)
        return
    if sub == "review":
        _review(args)
        return
    if sub == "confirm":
        _confirm(args)
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
