# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What the restore records look like to a reader of the dashboard or the CLI (L3-21).

The records on disk keep absolute paths: the start-up restore and the re-import
need them. A person reading the status -- possibly a viewer on the LAN in
company mode -- must not learn the OS account name, the layout of the data
directory beyond its own folders, or raw exception text. So every surface
passes a record through ``public_view`` first:

  * a path inside the data directory becomes data-directory-relative
    (``pre-restore/memory-...-before-restore.db``),
  * any other absolute path becomes its file name only,
  * a sentence that still embeds the data directory or the home directory
    (an outcome written by an earlier build) has it replaced by a placeholder.

Messages are written in plain words at their source; this is the backstop.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePath
from typing import Any

_DATA_DIR = "<data directory>"
_HOME = "~"
#: An absolute POSIX or Windows path appearing inside a sentence.
_EMBEDDED = re.compile(r"(?:(?<=\s)|(?<=\()|^)(?:/|[A-Za-z]:\\)[^\s()'\"]+")


def _is_path(value: str) -> bool:
    return len(value) > 1 and (value.startswith("/") or bool(re.match(r"^[A-Za-z]:\\", value)))


def _relative(value: str, roots: list[str]) -> str:
    for root in roots:
        if value == root:
            return "."
        if value.startswith(root + os.sep) or value.startswith(root + "/"):
            return PurePath(value[len(root) + 1:]).as_posix()
    return PurePath(value).name


def _scrub(text: str, roots: list[str]) -> str:
    for root in roots:
        text = text.replace(root, _DATA_DIR)
    home = str(Path.home())
    if len(home) > 1:
        text = text.replace(home, _HOME)
    return _EMBEDDED.sub(lambda m: PurePath(m.group(0)).name, text)


def _roots(data_root: Path) -> list[str]:
    raw = str(Path(data_root))
    resolved = str(Path(data_root).resolve())
    return sorted({raw, resolved}, key=len, reverse=True)


def public_view(value: Any, data_root: Path) -> Any:
    """``value`` (a record, a list, a string) with no absolute path left in it."""
    roots = _roots(data_root)

    def walk(item: Any) -> Any:
        if isinstance(item, dict):
            return {k: walk(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [walk(v) for v in item]
        if isinstance(item, Path):
            item = str(item)
        if isinstance(item, str):
            return _relative(item, roots) if _is_path(item) else _scrub(item, roots)
        return item

    return walk(value)


__all__ = ["public_view"]
