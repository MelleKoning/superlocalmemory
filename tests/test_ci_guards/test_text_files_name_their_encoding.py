# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""CI guard: every text file SLM reads or writes says it is UTF-8.

Without ``encoding=``, Python uses the locale's code page. That is UTF-8 on
macOS and Linux, so nothing shows there, but on a Windows machine it is
usually cp1252: the dashboard's ``index.html`` (which contains characters
outside cp1252) then failed to load with ``UnicodeDecodeError``, and anything
written there was unreadable by a UTF-8 reader.

The check is syntactic, so it runs the same on every platform: a call to
``read_text()``, ``write_text(text)``, a text-mode ``open()`` or a
``Path.open("w")``-style call with no
``encoding`` argument fails it.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory"


def _mode(call: ast.Call) -> str | None:
    """The literal mode of an ``open()`` call, ``"r"`` by default, None if computed."""
    node = call.args[1] if len(call.args) >= 2 else None
    for keyword in call.keywords:
        if keyword.arg == "mode":
            node = keyword.value
    if node is None:
        return "r"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _missing_encoding(call: ast.Call) -> bool:
    names = {keyword.arg for keyword in call.keywords}
    if "encoding" in names or None in names:  # None: **kwargs, cannot tell
        return False
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr == "read_text":
        return not call.args
    if isinstance(func, ast.Attribute) and func.attr == "write_text":
        return len(call.args) == 1
    if isinstance(func, ast.Name) and func.id == "open" and len(call.args) < 4:
        mode = _mode(call)
        return mode is not None and "b" not in mode
    if isinstance(func, ast.Attribute) and func.attr == "open" and call.args:
        # Path.open("w"): the mode is the first argument.
        first = call.args[0]
        return (
            isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and first.value.rstrip("t+") in {"r", "w", "a", "x"}
        )
    return False


def find_unnamed_encodings(root: Path) -> list[str]:
    found = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _missing_encoding(node):
                found.append(f"{path.relative_to(root.parent)}:{node.lineno}: {ast.unparse(node)[:80]}")
    return found


def test_every_text_read_and_write_in_the_product_names_utf8():
    found = find_unnamed_encodings(SRC)
    assert not found, (
        "These calls use the locale's encoding, which is not UTF-8 on Windows. "
        "Pass encoding=\"utf-8\":\n  " + "\n  ".join(found)
    )


def test_the_guard_catches_each_form(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "m.py").write_text(
        "from pathlib import Path\n"
        "p = Path('x')\n"
        "p.read_text()\n"
        "p.write_text('t')\n"
        "open('x')\n"
        "open('x', 'w')\n"
        "open('x', 'rb')\n"
        "p.read_text(encoding='utf-8')\n"
        "p.write_bytes(b'')\n"
        "p.open('w')\n"
        "p.open('rb')\n",
        encoding="utf-8",
    )
    found = find_unnamed_encodings(package)
    assert [line.split(":")[1] for line in found] == ["3", "4", "5", "6", "10"]
