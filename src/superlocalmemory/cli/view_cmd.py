# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""``slm view`` — saved views from the terminal, through the running daemon.

A saved view is a named recall query (issue #113). Every subcommand is a thin
client of ``/api/v3/views`` — the routes the dashboard uses — so the terminal
and the dashboard validate, authorise and run a view in exactly one way:

    slm view list
    slm view create NAME "QUERY" [--kind K] [--window 7d] [--as-of ISO] [--limit N]
    slm view run NAME            (also: slm view show NAME)
    slm view rename NAME NEW_NAME
    slm view delete NAME

Every result is printed with the id of its memory, the id ``slm delete`` and
the MCP ``fetch`` tool take. ``--json`` everywhere. Exit 2 for input the daemon
refused before doing anything, 1 for any other refusal.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from typing import Any, NoReturn
from urllib.parse import quote

from superlocalmemory.cli.daemon import (
    DaemonConflict,
    DaemonNotFound,
    DaemonUnprocessable,
    daemon_request,
)

_BASE = "/api/v3/views"
_NOT_RUNNING = "The SLM daemon is not running. Start it with: slm serve"
_CONTENT_CHARS = 140


class _Out:
    def __init__(self, as_json: bool, command: str) -> None:
        self.as_json = as_json
        self.command = command

    def fail(self, message: str, code: str = "", exit_code: int = 1) -> NoReturn:
        if self.as_json:
            from superlocalmemory.cli.json_output import json_print

            json_print(self.command, error={"code": code or "VIEW_REFUSED",
                                            "message": message})
        else:
            print(message, file=sys.stderr)
        sys.exit(exit_code)

    def emit(self, data: dict, text: str) -> None:
        if self.as_json:
            from superlocalmemory.cli.json_output import json_print

            json_print(self.command, data=data)
        else:
            print(text)


def _request(out: _Out, method: str, path: str, body: dict | None = None) -> dict:
    """The daemon's answer; prints the refusal and exits when there is one."""
    try:
        result = daemon_request(method, _BASE + path, body, preserve_conflict=True,
                                preserve_not_found=True, preserve_unprocessable=True)
    except DaemonConflict as exc:
        out.fail(exc.detail)
    except DaemonNotFound as exc:
        out.fail(exc.message, exc.code)
    except DaemonUnprocessable as exc:
        out.fail(exc.message, exc.code, exit_code=2)
    if result is None:
        out.fail(_NOT_RUNNING)
    return result


def _describe(view: dict) -> str:
    filters = ", ".join(f"{k}={v}" for k, v in sorted((view.get("filters") or {}).items()))
    extra = f"  [{filters}]" if filters else ""
    return f"{view['name']}: \"{view['query']}\"{extra}  (up to {view['limit']} results)"


def _one_line(text: str) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= _CONTENT_CHARS else flat[:_CONTENT_CHARS - 1] + "…"


def _run_text(run: dict) -> str:
    lines = [f"View {_describe(run['view'])}", ""]
    if not run["results"]:
        lines.append("  Nothing in your memory matches this view right now.")
    for row in run["results"]:
        score = row.get("score") or 0
        lines.append(f"  {row['rank']}. [{score:.2f}] {_one_line(row.get('content'))}")
        lines.append(f"     memory id: {row.get('fact_id', '')}")
    if run.get("no_confident_match") and run["results"]:
        lines += ["", "  None of these is a confident match; treat them as leads."]
    return "\n".join(lines)


def _list(args: Namespace, out: _Out) -> None:
    data = _request(out, "GET", "")
    views = data.get("views") or []
    if not views:
        text = ("No saved views yet. Save one with:\n"
                "  slm view create \"Work log\" \"what did I ship this week\" --window 7d")
    else:
        text = "\n".join([f"Saved views ({len(views)}):"]
                         + [f"  {_describe(v)}" for v in views])
    out.emit(data, text)


def _create(args: Namespace, out: _Out) -> None:
    filters = {k: v for k, v in (("kind", args.kind), ("window", args.window),
                                 ("as_of", args.as_of)) if v}
    body: dict[str, Any] = {"name": args.name, "query": args.query, "filters": filters}
    if args.limit is not None:
        body["limit"] = args.limit
    data = _request(out, "POST", "", body)
    out.emit(data, f"{data['message']}\n  {_describe(data['view'])}\n"
                   f"Run it with: slm view run \"{data['view']['name']}\"")


def _run(args: Namespace, out: _Out) -> None:
    data = _request(out, "GET", f"/run?name={quote(args.name, safe='')}&via=cli")
    out.emit(data, _run_text(data))


def _rename(args: Namespace, out: _Out) -> None:
    data = _request(out, "POST", "/rename", {"name": args.name, "new_name": args.new_name})
    out.emit(data, data["message"])


def _delete(args: Namespace, out: _Out) -> None:
    data = _request(out, "POST", "/delete", {"name": args.name})
    out.emit(data, data["message"])


_HANDLERS = {"list": _list, "create": _create, "run": _run, "show": _run,
             "rename": _rename, "delete": _delete}


def cmd_view(args: Namespace) -> None:
    """Dispatch ``slm view <subcommand>``."""
    sub = getattr(args, "view_command", None)
    handler = _HANDLERS.get(sub or "")
    if handler is None:
        print("Usage: slm view {list | create | run | show | rename | delete}")
        print()
        print("  slm view list                              your saved views")
        print("  slm view create NAME \"QUERY\" [--window 7d] [--kind K] [--limit N]")
        print("  slm view run NAME                          recall's answer, with memory ids")
        print("  slm view rename NAME NEW_NAME")
        print("  slm view delete NAME                       no memory is touched")
        return
    handler(args, _Out(bool(getattr(args, "json", False)), f"view {sub}"))


def _json_flag(parser: Any) -> None:
    # SUPPRESS: a nested parser must not overwrite `slm view --json run X`.
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="machine-readable output")


def register_view_parser(sub: Any) -> None:
    """Attach the ``view`` parser. Called from cli/main.py."""
    p = sub.add_parser("view", help="Saved views: named recall queries you can re-run")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    vsub = p.add_subparsers(dest="view_command", title="view subcommands")

    _json_flag(vsub.add_parser("list", help="your saved views"))

    c = vsub.add_parser("create", help="save a named recall query")
    c.add_argument("name", help="a short name, e.g. \"Work log\"")
    c.add_argument("query", help="what to look for, as you would ask recall")
    c.add_argument("--kind", default="", help="only memories of this kind")
    c.add_argument("--window", default="",
                   help="only this time range: 24h, 7d, 30d, or 2026-07-01..2026-07-31")
    c.add_argument("--as-of", dest="as_of", default="",
                   help="what was true at this instant (ISO-8601)")
    c.add_argument("--limit", type=int, default=None, help="results to show (1-50)")
    _json_flag(c)

    for name, text in (("run", "run a view: recall's answer, with memory ids"),
                       ("show", "same as run")):
        r = vsub.add_parser(name, help=text)
        r.add_argument("name", help="the view's name")
        _json_flag(r)

    rn = vsub.add_parser("rename", help="give a view a new name")
    rn.add_argument("name")
    rn.add_argument("new_name")
    _json_flag(rn)

    d = vsub.add_parser("delete", help="delete a view (no memory is touched)")
    d.add_argument("name")
    _json_flag(d)
