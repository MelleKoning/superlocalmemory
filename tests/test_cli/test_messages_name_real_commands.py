# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Every ``slm <command>`` a user is told to run must be a command that exists.

A daemon error told users to run ``slm start``; the command is
``slm serve start``. Advice that fails with "invalid choice" is a dead end at
exactly the moment someone needs help, so this scans every user-facing string
in the package (Python string literals, dashboard JS and HTML) and the docs.
"""

from __future__ import annotations

import ast
import io
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "src" / "superlocalmemory"

# A command a user is told to type: quoted in backticks / <code>, after
# "run", or at the start of a shell line.
_ADVICE = re.compile(r"(?:`|<code>|\b[Rr]un:? |\$ |^)slm ([a-z][a-z0-9_-]*)", re.M)
# In code, a literal that merely starts with "slm ..." is usually a process
# name to match or sample text, not advice, so only quoted forms count there.
_ADVICE_IN_CODE = re.compile(r"(?:`|<code>|\b[Rr]un:? |\$ )slm ([a-z][a-z0-9_-]*)")

# `slm hook <name>` is dispatched before argparse, so it is not a subparser.
_PRE_ARGPARSE = frozenset({"hook"})

# Docs that name a command only to say it does not exist.
_NAMED_AS_ABSENT = {
    "auto-memory.md": {"patterns", "useful", "learning"},
    "cli-reference.md": {"consistency"},
    "company-mode.md": {"user", "role", "company-mode"},
    "compliance.md": {"audit", "retention"},
}


def _top_level_commands(monkeypatch: pytest.MonkeyPatch) -> frozenset[str]:
    from superlocalmemory.cli import main as cli_main

    monkeypatch.setattr(sys, "argv", ["slm", "--no-such-command--"])
    buf = io.StringIO()
    with redirect_stderr(buf), pytest.raises(SystemExit):
        cli_main.main()
    # The usage line lists every subcommand as {a,b,c}.
    match = re.search(r"\{([a-z0-9,-]+)\}", buf.getvalue())
    assert match, buf.getvalue()
    return frozenset(match.group(1).split(",")) | _PRE_ARGPARSE


def _docstring_nodes(tree: ast.AST) -> set[int]:
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, owners) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                found.add(id(first.value))
    return found


def _package_advice() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for path in sorted(_PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        skip = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in skip):
                where = f"{path.relative_to(_REPO)}:{node.lineno}"
                found += [(where, m) for m in _ADVICE_IN_CODE.findall(node.value)]
    for pattern in ("*.js", "*.html"):
        for path in sorted(_PKG.rglob(pattern)):
            text = path.read_text(encoding="utf-8", errors="replace")
            found += [(str(path.relative_to(_REPO)), m)
                      for m in _ADVICE_IN_CODE.findall(text)]
    return found


def _docs_advice() -> list[tuple[str, str]]:
    paths = sorted((_REPO / "docs").glob("*.md")) + [_REPO / "README.md"]
    found: list[tuple[str, str]] = []
    for path in paths:
        if not path.is_file():
            continue
        absent = _NAMED_AS_ABSENT.get(path.name, set())
        text = path.read_text(encoding="utf-8")
        found += [(str(path.relative_to(_REPO)), m)
                  for m in _ADVICE.findall(text) if m not in absent]
    return found


def test_messages_in_the_package_only_name_real_commands(monkeypatch) -> None:
    known = _top_level_commands(monkeypatch)
    wrong = [(where, cmd) for where, cmd in _package_advice() if cmd not in known]
    assert not wrong, f"user-facing text names commands that do not exist: {wrong}"


def test_docs_only_name_real_commands(monkeypatch) -> None:
    known = _top_level_commands(monkeypatch)
    wrong = [(where, cmd) for where, cmd in _docs_advice() if cmd not in known]
    assert not wrong, f"docs name commands that do not exist: {wrong}"


def test_the_daemon_start_hint_names_serve_start() -> None:
    from superlocalmemory.cli import daemon

    hints = [h for _, _, h in daemon._LIVENESS_DIAGNOSIS.values()]
    assert any("`slm serve start`" in h for h in hints)
