# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Which safety copies to keep, across every name they have been written under.

Three namings exist in users' snapshot folders:

  ``memory-20260819-120000-123456-pre-migration.db``   current; UTC; a ``-N``
                                                       suffix may follow the
                                                       microseconds on collision
  ``memory-20260819-120000-pre-migration.db``          the first release; UTC
  ``memory-20260823-144350-pre-4.1.0.db``              copies named for the
                                                       version they precede;
                                                       LOCAL time, zone unknown

The last kind was never recognised, so it was never pruned: 1.28 GB on the
owner's machine, with the ``-shm`` and ``-wal`` SQLite left beside each one.
The second kind was recognised by name but not by time, and sorted AFTER every
current name -- so it was kept forever and a newer generation pruned instead.

A generation is the set of copies taken together (``memory`` and ``learning``
share a timestamp) and is only useful whole. Retention keeps the ``keep``
newest generations. A legacy time could be anywhere from fourteen hours before
to twelve hours after the same digits read as UTC, so a generation is pruned
only when at least ``keep`` others are newer than it under EVERY reading. Where
the order is genuinely uncertain, one more copy is kept rather than one lost.

Deletion is only ever by explicit full path, of a regular file directly inside
the folder; a symlink is neither followed, counted nor removed, and a folder
that is itself a symlink is left alone.
"""

from __future__ import annotations

import calendar
import logging
import os
import re
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("superlocalmemory.storage.backup")

_STAMP = r"(?P<date>\d{8})-(?P<time>\d{6})(?:-(?P<us>\d{6}))?(?:-(?P<n>\d+))?"
_CURRENT = re.compile(rf"^(?P<stem>.+)-{_STAMP}-pre-migration\.db$")
_LEGACY = re.compile(
    rf"^(?P<stem>[A-Za-z][A-Za-z0-9_.]*)-{_STAMP}"
    r"-pre-(?P<version>\d[0-9A-Za-z.+_-]*)\.db$"
)
_SIDE_SUFFIXES = ("-wal", "-shm", "-journal")
_STAGING_SUFFIX = ".partial"

#: A copy is written under ``<name>.partial`` and renamed when verified. One
#: that a hard crash interrupted keeps that name -- the size of the store --
#: for ever. A copy being written right now touches its file continuously, so
#: one untouched for this long is abandoned, not in progress.
_STAGING_ABANDONED_AFTER_S = 3600.0

#: The widest offsets from UTC in use: UTC-12 and UTC+14.
_LOCAL_EARLIEST = 14 * 3600
_LOCAL_LATEST = 12 * 3600


@dataclass
class _Generation:
    earliest: float
    latest: float
    files: list[Path] = field(default_factory=list)


def _parse(name: str) -> tuple[tuple, float, float] | None:
    """(generation key, earliest, latest) in UTC seconds, or None if not a copy."""
    for pattern, legacy in ((_CURRENT, False), (_LEGACY, True)):
        match = pattern.match(name)
        if match is None:
            continue
        date, clock = match["date"], match["time"]
        try:
            # timegm normalises an out-of-range day (a hand-made "...0100...")
            # into the neighbouring one, so such a name still orders by its
            # digits; a month that does not exist is not a time at all.
            moment = calendar.timegm((
                int(date[:4]), int(date[4:6]), int(date[6:]),
                int(clock[:2]), int(clock[2:4]), int(clock[4:]),
            )) + int(match["us"] or 0) / 1_000_000
        except ValueError:
            return None
        if legacy:
            key = ("legacy", match["date"], match["time"], match["version"])
            return key, moment - _LOCAL_EARLIEST, moment + _LOCAL_LATEST
        key = ("current", match["date"], match["time"], match["us"])
        return key, moment, moment
    return None


def _list(root: Path) -> tuple[dict[str, Path], set[str]]:
    """Regular files directly in ``root`` by name, and every name present."""
    with os.scandir(root) as entries:
        listed = list(entries)
    regular = {
        entry.name: root / entry.name
        for entry in listed
        if not entry.is_symlink() and entry.is_file(follow_symlinks=False)
    }
    return regular, {entry.name for entry in listed}


def _side_files(main: Path, children: dict[str, Path]) -> list[Path]:
    return [
        children[main.name + suffix]
        for suffix in _SIDE_SUFFIXES
        if main.name + suffix in children
    ]


def _orphaned_side_files(children: dict[str, Path], everything: set[str]) -> list[Path]:
    """Side files of a copy that is gone -- a prune interrupted half-way."""
    orphans = []
    for name, path in children.items():
        for suffix in _SIDE_SUFFIXES:
            base = name[: -len(suffix)]
            if name.endswith(suffix) and _parse(base) and base not in everything:
                orphans.append(path)
    return orphans


def _abandoned_staging(children: dict[str, Path], now: float) -> list[Path]:
    """Staging files of copies a crash interrupted, with their side files."""
    found = []
    for name, path in children.items():
        stem = name
        for suffix in _SIDE_SUFFIXES:
            if stem.endswith(_STAGING_SUFFIX + suffix):
                stem = stem[: -len(suffix)]
                break
        if not stem.endswith(_STAGING_SUFFIX) or _parse(stem[: -len(_STAGING_SUFFIX)]) is None:
            continue
        try:
            age = now - os.lstat(path).st_mtime
        except OSError:
            continue
        if age > _STAGING_ABANDONED_AFTER_S:
            found.append(path)
    return sorted(found)


def plan(root: Path, keep: int, protect: frozenset[str] = frozenset()) -> list[Path]:
    """Paths to delete, main files of a generation before its side files.

    A generation with a file named in ``protect`` -- the copy this run just
    took -- is never deleted, and counts as newer than every other: it IS the
    newest, whatever time the clock wrote into its name. A clock running behind
    the newest existing copy otherwise named it "oldest", and keeping the two
    newest by name deleted it the moment it was taken.
    """
    children, everything = _list(root)
    generations: dict[tuple, _Generation] = {}
    for name, path in children.items():
        parsed = _parse(name)
        if parsed is None:
            continue
        key, earliest, latest = parsed
        generations.setdefault(key, _Generation(earliest, latest)).files.append(path)

    protected = {
        key for key, gen in generations.items()
        if any(path.name in protect for path in gen.files)
    }
    doomed: list[Path] = []
    for key, gen in sorted(generations.items(), key=lambda kv: kv[1].latest):
        if key in protected:
            continue
        certainly_newer = sum(
            1 for other_key, other in generations.items()
            if other_key != key
            and (other_key in protected or other.earliest > gen.latest)
        )
        if certainly_newer < keep:
            continue
        mains = sorted(gen.files)
        doomed.extend(mains)
        for main in mains:
            doomed.extend(_side_files(main, children))
    doomed.extend(sorted(_orphaned_side_files(children, everything)))
    doomed.extend(_abandoned_staging(children, time.time()))
    return doomed


def prune(root: Path, keep: int, protect: frozenset[str] = frozenset()) -> None:
    """Delete every generation older than the ``keep`` newest. Never raises.

    ``protect`` names files of the copy just taken; see ``plan``.
    """
    if not root.exists():
        return
    if root.is_symlink():
        logger.warning(
            "[SLM] Not pruning safety copies in %s: the folder is a symlink, and "
            "nothing is deleted through one", root,
        )
        return
    try:
        doomed = plan(root, keep, protect)
    except OSError as exc:
        logger.warning("[SLM] Could not list safety copies in %s: %s", root, exc)
        return
    for target in doomed:
        try:
            info = os.lstat(target)
        except FileNotFoundError:
            continue
        if target.parent != root or not stat.S_ISREG(info.st_mode):
            continue
        logger.info("[SLM] Removing old pre-migration snapshot: %s", target)
        try:
            target.unlink()
        except OSError as exc:
            logger.warning("[SLM] Could not remove %s: %s", target, exc)
