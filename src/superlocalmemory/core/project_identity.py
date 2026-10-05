# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What project a value names - the one rule every project comparison uses.

Three writers record a project today, each in a different shape:

* ``remember(project=...)`` stores whatever the agent passed, which in practice
  is a bare name ("superlocalmemory") or a full working directory;
* the tool-event hook stores the full working directory
  (``tool_events.project_path``, e.g. "/Users/x/work/superlocalmemory");
* the session-end hook writes "[superlocalmemory] session ended ..." - the
  directory's last component, in the content itself.

The only thing all three share is the directory's last component, so that is
the identity: the last non-empty part of a path (either slash), Unicode
NFC-normalised (macOS can hand back decomposed names), compared without regard
to case (the default file systems of macOS and Windows ignore case, and two
projects that differ only in case on Linux are not a real-world case worth a
second rule). Surrounding spaces and trailing slashes never matter.

The cost, stated rather than hidden: two different directories with the same
last component ("/a/app" and "/b/app") are the same project here. A bare name
cannot be told apart from either, and two of the three writers only ever had
the name, so a finer rule would make most stored projects unmatchable.

Write, backfill, boost, filter and summary all call ``project_key``; nothing
else in the product decides whether two project values are the same.
"""

from __future__ import annotations

import re
import unicodedata

#: Longest project value kept. A path longer than any real one is not a project.
MAX_PROJECT_CHARS = 1024

_SEPARATORS = re.compile(r"[\\/]+")
_DRIVE = re.compile(r"[A-Za-z]:")
_NOT_A_NAME = frozenset({".", "..", "~"})

#: "[name] session ended 2026-10-04 18:20" - the Stop hook's summary prefix.
_SESSION_END = re.compile(r"\A\[([^\]\n]{1,200})\] session ended \d{4}-\d{2}-\d{2}")


def project_key(value: object) -> str | None:
    """The identity of the project ``value`` names, or None when it names none.

    "SuperLocalMemory", " superlocalmemory ", "/Users/x/superlocalmemory/" and
    "C:\\\\work\\\\SuperLocalMemory" all give "superlocalmemory". Empty text, a
    bare root ("/"), a drive ("C:"), "~", "." and ".." name no project.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()[:MAX_PROJECT_CHARS]
    if not text:
        return None
    parts = [p.strip() for p in _SEPARATORS.split(text) if p.strip()]
    if not parts:
        return None
    name = parts[-1]
    if name in _NOT_A_NAME or _DRIVE.fullmatch(name):
        return None
    return unicodedata.normalize("NFC", name).casefold()


def storable_project(value: object) -> str:
    """The project value to save with a memory: trimmed, bounded, and empty
    when it names no project (so "/" or "   " is never stored as one)."""
    if project_key(value) is None:
        return ""
    return str(value).strip()[:MAX_PROJECT_CHARS]


def project_name(value: object) -> str | None:
    """The project's name as written (last path component, spaces trimmed,
    case kept), or None when ``value`` names no project. For display and for
    building a query; compare with ``project_key``."""
    if project_key(value) is None:
        return None
    text = str(value).strip()[:MAX_PROJECT_CHARS]
    return [p.strip() for p in _SEPARATORS.split(text) if p.strip()][-1]


def session_context_query(project_path: object) -> str:
    """The query a session start runs when it was given only a project.

    It used to be "project context <full path>": every directory name in the
    path is a search term, so memories that merely mention paths (session
    trivia from other repositories) outranked the project's own. The
    project's name carries the meaning; the path's other parts do not.
    """
    name = project_name(project_path)
    return f"project context {name}" if name else "recent important decisions"


def session_end_project(content: object) -> str | None:
    """The project named by a session-end summary's "[name] session ended"
    prefix, or None when ``content`` is not such a summary."""
    if not isinstance(content, str):
        return None
    match = _SESSION_END.match(content)
    if match is None:
        return None
    name = match.group(1).strip()
    return name if project_key(name) is not None else None


__all__ = ["MAX_PROJECT_CHARS", "project_key", "project_name", "session_context_query",
           "session_end_project", "storable_project"]
