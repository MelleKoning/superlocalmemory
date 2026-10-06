# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""CI guard: ``with sqlite3.connect(...)`` is never used to close a connection.

A connection used as a context manager commits or rolls back on exit, and
stays OPEN. On macOS and Linux nothing shows. On Windows an open database file
cannot be deleted or replaced, so the embedding migration aborted after
doing its work: its temporary folder could not be removed while the staging
database was still open (``WinError 32``). Use
``with closing(sqlite3.connect(...)) as conn, conn:`` to keep the transaction
and close the file.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory"


def find_unclosed(root: Path) -> list[str]:
    found = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.With, ast.AsyncWith)):
                continue
            for item in node.items:
                call = item.context_expr
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "connect"
                        and isinstance(call.func.value, ast.Name)
                        and "sqlite" in call.func.value.id):
                    found.append(f"{path.relative_to(root.parent)}:{node.lineno}")
    return found


def test_the_product_never_leaves_a_with_block_connection_open():
    found = find_unclosed(SRC)
    assert not found, (
        "`with sqlite3.connect(...)` leaves the database open (Windows then cannot "
        "delete or replace it). Use `with closing(sqlite3.connect(...)) as conn, conn:`:\n  "
        + "\n  ".join(found)
    )


def test_the_premise_a_with_block_does_not_close(tmp_path):
    with sqlite3.connect(tmp_path / "a.db") as conn:
        conn.execute("CREATE TABLE t (x)")
    conn.execute("SELECT 1")  # still open: would raise ProgrammingError if closed
    conn.close()


def test_the_guard_finds_it(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "m.py").write_text(
        "import sqlite3\nfrom contextlib import closing\n"
        "with sqlite3.connect('a') as c:\n    pass\n"
        "with closing(sqlite3.connect('a')) as c, c:\n    pass\n",
        encoding="utf-8",
    )
    assert [line.split(":")[1] for line in find_unclosed(package)] == ["3"]
