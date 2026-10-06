# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Copies of memory and remote secrets stay readable by their owner only.

A restore point, a backup or a learning copy holds the same memories as the
live store, which is ``0600``; the configuration and the answer-check and
feedback keys are secrets of their own. Releases before 4.1.20 wrote some of them with
the default mode (``0644``), so another account on the same computer could read
them whenever the data directory itself was not private.

* :func:`create_private_file`, :func:`copy_private` and
  :func:`make_private_dir` are what every writer of such a copy uses, so a new
  copy is owner-only before a single byte goes into it.
* :func:`tighten_private_files` runs once at every daemon start and makes the
  copies an older release left behind owner-only. It is idempotent: a file
  that is already private is not touched. POSIX: group and other access is
  removed from files and folders (nothing is ever added, so a ``0400`` file
  stays ``0400``). Windows: the owner-only access list of
  :mod:`infra.owner_only_acl`, files only.

Symbolic links are never followed, and a file owned by another account is
reported, not changed.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: Folders under the data root whose every file is a copy of memory.
PRIVATE_DIRS: tuple[str, ...] = (
    "pre-migration-snapshots",   # restore points (storage/backup.py)
    "pre-restore",               # the store as it was before each restore
    "restore-delta",             # memories carried across a restore
    "backups",                   # routine backups and backup sets (infra/backup.py)
    "secrets",                   # answer-check provider keys (core/judge_keys.py)
)
#: Single files under the data root that hold secrets or copies.
PRIVATE_FILES: tuple[tuple[str, ...], ...] = (
    ("remote_keys.json",),
    ("remote", "remote.json"),
    ("remote", "tls", "server.key"),
    ("remote", "tls", "ca.key"),
    # Tightened again here, not only when next saved or loaded: an older
    # release's 0644 copy stayed readable by other accounts until then.
    ("config.json",),          # can hold provider API keys
    (".feedback-hash-key",),   # keys the feedback query hashes
)
#: File-name prefixes in the data root itself (dashboard learning backups).
PRIVATE_PREFIXES: tuple[str, ...] = ("learning.db.backup_",)

_FILE_MODE = 0o600
_DIR_MODE = 0o700
_WINDOWS = os.name == "nt"


# -- writing new copies ----------------------------------------------------------------


def _restrict(path: Path, mode: int) -> None:
    if _WINDOWS:
        from superlocalmemory.infra.owner_only_acl import restrict_to_owner

        restrict_to_owner(path)
    else:
        os.chmod(path, mode)


def _strip_others(path: Path, info: os.stat_result) -> None:
    """Remove group and other access; never add any (a 0400 file stays 0400)."""
    if _WINDOWS:
        _restrict(path, _FILE_MODE)
    else:
        os.chmod(path, stat.S_IMODE(info.st_mode) & 0o700)


def create_private_file(path: Path) -> None:
    """Create (or truncate) ``path`` readable and writable by its owner only."""
    path = Path(path)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                 | getattr(os, "O_BINARY", 0), _FILE_MODE)
    os.close(fd)
    # An existing file keeps its old mode, and 0600 means nothing on Windows.
    _restrict(path, _FILE_MODE)


def make_private_dir(path: Path) -> None:
    """Create ``path`` (and parents); the folder itself is owner-only on POSIX."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    if not _WINDOWS:
        os.chmod(path, _DIR_MODE)


def copy_private(source: Path, destination: Path) -> None:
    """Copy ``source`` byte for byte into a new owner-only ``destination``.

    Unlike :func:`shutil.copy2` / :func:`shutil.copyfile`, the destination is
    owner-only before any byte is written and never takes the source's mode.
    """
    create_private_file(destination)
    with open(source, "rb") as src, open(destination, "r+b") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())


# -- tightening what older releases left behind ------------------------------------------


@dataclass(frozen=True)
class TightenReport:
    tightened: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    failed: tuple[str, ...] = field(default=())

    def __bool__(self) -> bool:
        return bool(self.tightened or self.failed or self.skipped)


def _needs_tightening(path: Path, info: os.stat_result) -> bool:
    if _WINDOWS:  # mode bits say nothing there; read the access list
        from superlocalmemory.infra.owner_only_acl import is_owner_only

        return not is_owner_only(path)
    return bool(stat.S_IMODE(info.st_mode) & 0o077)


def _owned_by_me(info: os.stat_result) -> bool:
    getuid = getattr(os, "getuid", None)
    return getuid is None or info.st_uid == getuid()


def _tighten_one(path: Path, tightened: list[str], skipped: list[str],
                 failed: list[str]) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        failed.append(f"{path}: {exc}")
        return
    if stat.S_ISLNK(info.st_mode):
        return  # never follow a link out of the data folder
    is_dir = stat.S_ISDIR(info.st_mode)
    if not (is_dir or stat.S_ISREG(info.st_mode)):
        return
    if is_dir and _WINDOWS:
        return  # folder access lists are inherited; files are set one by one
    try:
        if not _needs_tightening(path, info):
            return
        if not _owned_by_me(info):
            skipped.append(str(path))
            return
        _strip_others(path, info)
    except OSError as exc:
        failed.append(f"{path}: {exc}")
        return
    tightened.append(str(path))


def _walk(root: Path, tightened: list[str], skipped: list[str], failed: list[str]) -> None:
    """``root`` and everything below it, without following links."""
    _tighten_one(root, tightened, skipped, failed)
    try:
        if stat.S_ISLNK(os.lstat(root).st_mode) or not root.is_dir():
            return
    except OSError:
        return
    for current, dirs, files in os.walk(root, followlinks=False):
        here = Path(current)
        for name in dirs:
            _tighten_one(here / name, tightened, skipped, failed)
        for name in files:
            _tighten_one(here / name, tightened, skipped, failed)


def tighten_tree(root: Path) -> TightenReport:
    """Make ``root`` and everything below it owner-only (a freshly copied folder)."""
    tightened: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    _walk(Path(root), tightened, skipped, failed)
    return TightenReport(tuple(tightened), tuple(skipped), tuple(failed))


def tighten_private_files(data_root: Path) -> TightenReport:
    """Make every copy and remote secret under ``data_root`` owner-only.

    Never raises for one file: problems are collected and logged, so a start
    is never stopped by a file it cannot change.
    """
    data_root = Path(data_root)
    tightened: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for name in PRIVATE_DIRS:
        _walk(data_root / name, tightened, skipped, failed)
    for parts in PRIVATE_FILES:
        _tighten_one(data_root.joinpath(*parts), tightened, skipped, failed)
    try:
        entries = sorted(data_root.iterdir()) if data_root.is_dir() else []
    except OSError as exc:
        failed.append(f"{data_root}: {exc}")
        entries = []
    for entry in entries:
        if entry.name.startswith(PRIVATE_PREFIXES):
            _tighten_one(entry, tightened, skipped, failed)
    report = TightenReport(tuple(tightened), tuple(skipped), tuple(failed))
    if report.tightened:
        logger.info("Made %d older backup or secret file(s) readable only by you.",
                    len(report.tightened))
    for path in report.skipped:
        logger.warning("%s belongs to another account, so it was left as it is; "
                       "other accounts may be able to read it.", path)
    for problem in report.failed:
        logger.warning("Could not make a backup or secret file private: %s", problem)
    return report


__all__ = [
    "PRIVATE_DIRS",
    "PRIVATE_FILES",
    "PRIVATE_PREFIXES",
    "TightenReport",
    "copy_private",
    "create_private_file",
    "make_private_dir",
    "tighten_private_files",
    "tighten_tree",
]
