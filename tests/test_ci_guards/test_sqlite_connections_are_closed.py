# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""CI guard: no SQLite connection is left open by mistake.

On macOS and Linux an open database file can still be deleted or replaced,
so nothing shows. On Windows it cannot (``WinError 32``), and two shapes of
code left files open there:

1. ``with sqlite3.connect(...) as conn`` commits or rolls back on exit and
   stays OPEN. The embedding migration aborted after doing its work: its
   temporary folder could not be removed. Use
   ``with closing(sqlite3.connect(...)) as conn, conn:``.
2. A helper that opens a connection, configures it (PRAGMAs, ATTACH, an
   extension) and returns or stores it. When the configuring statement
   raised (a damaged file, "database is locked"), the connection was never
   closed, and anything keeping the exception (a log record does) kept the
   file open: the restore of a damaged store then could not replace it. The
   statements between ``connect`` and the hand-over must sit in a ``try``
   that closes the connection on failure.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory"

#: Setting these attributes cannot raise, so it may sit before the ``try``.
_SAFE_ATTRIBUTES = frozenset({"row_factory", "text_factory", "isolation_level"})


def _is_connect(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "connect" and isinstance(node.func.value, ast.Name)
            and "sqlite" in node.func.value.id)


def _closes(statements: list[ast.stmt], target: str) -> bool:
    """Whether ``statements`` call ``<target>.close()``."""
    for statement in statements:
        for node in ast.walk(statement):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "close"
                    and ast.unparse(node.func.value) == target):
                return True
    return False


def _guarded_try(statement: ast.stmt, target: str) -> bool:
    if not isinstance(statement, ast.Try):
        return False
    handlers = [line for handler in statement.handlers for line in handler.body]
    return _closes(handlers, target) or _closes(statement.finalbody, target)


def _is_safe(statement: ast.stmt, target: str) -> bool:
    return (isinstance(statement, ast.Assign) and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Attribute)
            and ast.unparse(statement.targets[0].value) == target
            and statement.targets[0].attr in _SAFE_ATTRIBUTES)


def _hands_over(statement: ast.stmt, target: str) -> bool:
    """``return target`` (the connection leaves the function)."""
    return (isinstance(statement, ast.Return) and statement.value is not None
            and ast.unparse(statement.value) == target)


def _unguarded_after_connect(block: list[ast.stmt]) -> list[int]:
    """Line numbers of connections in ``block`` that can leak on setup failure.

    A connection that leaves the function (returned, or stored on an object)
    leaks if a statement that can raise runs between ``connect`` and that
    hand-over outside a ``try`` that closes it.
    """
    leaks = []
    for index, statement in enumerate(block):
        if not (isinstance(statement, (ast.Assign, ast.AnnAssign))
                and _is_connect(statement.value)):
            continue
        target_node = statement.targets[0] if isinstance(statement, ast.Assign) else statement.target
        target = ast.unparse(target_node)
        stored = isinstance(target_node, (ast.Attribute, ast.Subscript))
        rest = block[index + 1:]
        escapes = stored or any(_hands_over(s, target) for s in rest)
        if not escapes:
            continue
        for later in rest:
            if _is_safe(later, target):
                continue
            if _hands_over(later, target) or _guarded_try(later, target):
                break
            if not any(target in ast.unparse(n) for n in ast.walk(later)
                       if isinstance(n, ast.Call)):
                break  # does not touch the connection
            leaks.append(statement.lineno)
            break
    return leaks


def _blocks(tree: ast.AST):
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                yield block


def find_leaks(root: Path) -> list[str]:
    found = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        where = path.relative_to(root.parent)
        for node in ast.walk(tree):
            if isinstance(node, (ast.With, ast.AsyncWith)):
                found += [f"{where}:{node.lineno} with-block connection stays open"
                          for item in node.items if _is_connect(item.context_expr)]
        for block in _blocks(tree):
            found += [f"{where}:{line} not closed if its setup fails"
                      for line in _unguarded_after_connect(block)]
    return sorted(set(found))


def test_the_product_closes_every_connection_it_opens():
    found = find_leaks(SRC)
    assert not found, (
        "These can leave a database open, which Windows then cannot delete or "
        "replace. Use `with closing(sqlite3.connect(...)) as conn, conn:`, or put "
        "the setup after connect in `try: ... except BaseException: conn.close(); raise`:\n  "
        + "\n  ".join(found)
    )


def test_the_premise_a_with_block_does_not_close(tmp_path):
    with sqlite3.connect(tmp_path / "a.db") as conn:
        conn.execute("CREATE TABLE t (x)")
    conn.execute("SELECT 1")  # still open: would raise ProgrammingError if closed
    conn.close()


def test_the_guard_finds_each_shape(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "m.py").write_text(
        "import sqlite3\n"
        "from contextlib import closing\n"
        "with sqlite3.connect('a') as c:\n"                 # 3: with-block
        "    pass\n"
        "with closing(sqlite3.connect('a')) as c, c:\n"     # fine
        "    pass\n"
        "def leaky():\n"
        "    conn = sqlite3.connect('a')\n"                 # 8: leaks
        "    conn.execute('PRAGMA journal_mode=WAL')\n"
        "    return conn\n"
        "def guarded():\n"
        "    conn = sqlite3.connect('a')\n"                 # fine
        "    conn.row_factory = sqlite3.Row\n"
        "    try:\n"
        "        conn.execute('PRAGMA journal_mode=WAL')\n"
        "    except BaseException:\n"
        "        conn.close()\n"
        "        raise\n"
        "    return conn\n"
        "class Holder:\n"
        "    def get(self):\n"
        "        if self._conn is None:\n"
        "            self._conn = sqlite3.connect('a')\n"  # 23: stored, leaks
        "            self._conn.execute('PRAGMA x')\n"
        "        return self._conn\n"
        "def used_and_closed():\n"
        "    conn = sqlite3.connect('a')\n"                 # fine: never escapes
        "    try:\n"
        "        return conn.execute('SELECT 1').fetchall()\n"
        "    finally:\n"
        "        conn.close()\n",
        encoding="utf-8",
    )
    lines = sorted(int(entry.split(":")[1].split()[0]) for entry in find_leaks(package))
    assert lines == [3, 8, 23]
