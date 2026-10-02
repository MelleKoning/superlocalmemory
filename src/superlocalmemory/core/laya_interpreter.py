# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Which Python interpreters SLM will run for the on-device answer check.

"Use an existing install" takes a path from the dashboard and SLM then runs it
— once to check it, and again at every engine start. So the path must be a
Python interpreter that nobody but this user (or the system) can change:

* a regular, executable file (not a folder, not a missing path);
* inside a Python environment (``<env>/bin/python`` with ``<env>/pyvenv.cfg``)
  or the very interpreter SLM itself runs on;
* that file, its folder, and the environment's folder, ``bin`` folder and
  ``pyvenv.cfg`` all owned by this user or root and writable by no one else;
* when a person hands the path in (``strict_location``), not in a temporary
  folder and not under ``/Users/Shared``, which every account can write to.

Folders above the environment are deliberately not walked: Homebrew's own
prefix is group-writable by the admin group on most Macs, and refusing every
Homebrew Python would refuse the most common install there is. Anyone in that
group can already change any Homebrew program.

``refusal()`` returns a plain-language reason, or "" when the path is fine.
Stdlib only; it never runs anything.
"""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path

#: Folders every account on the Mac can write into.
SHARED_ROOTS: tuple[Path, ...] = (Path("/Users/Shared"),)

REASON_NOT_ABSOLUTE = "Enter the full path to the Python interpreter, starting with /."
REASON_NOT_FOUND = "That Python can't be found."
REASON_TEMPORARY = "That looks like a temporary location, not a real install."
REASON_SHARED = ("That Python is in a folder every account on this Mac can change, "
                 "so SLM won't run it.")
REASON_NOT_INTERPRETER = "That path isn't a Python interpreter."
REASON_NOT_ENVIRONMENT = ("Choose the python inside a Python environment "
                          "(the folder that has a pyvenv.cfg file).")
REASON_WRITABLE = ("That Python can be changed by other accounts on this Mac, "
                   "so SLM won't run it.")


def _under(path: Path, roots: tuple[Path, ...]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _environment_root(python: Path) -> Path | None:
    """``<env>`` when ``python`` is ``<env>/bin/<name>`` and ``<env>/pyvenv.cfg``
    is a regular file; the ``bin`` folder is resolved, the file name is not
    (an environment's python is normally a link to the base interpreter)."""
    try:
        bin_dir = python.parent.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if bin_dir.name != "bin":
        return None
    cfg = bin_dir.parent / "pyvenv.cfg"
    try:
        return bin_dir.parent if stat.S_ISREG(cfg.lstat().st_mode) else None
    except OSError:
        return None


def _safely_owned(path: Path) -> bool:
    try:
        st = path.stat()
    except OSError:
        return False
    getuid = getattr(os, "getuid", None)
    if getuid is not None and st.st_uid not in (getuid(), 0):
        return False
    return not st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


def refusal(python: str, *, strict_location: bool,
            in_temp_dir: Callable[[Path], bool]) -> str:
    """Why ``python`` must not be run, or "" when it may be."""
    if not isinstance(python, str) or not python or not os.path.isabs(python):
        return REASON_NOT_ABSOLUTE
    given = Path(python)
    try:
        real = given.resolve(strict=True)
    except (OSError, RuntimeError):
        return REASON_NOT_FOUND
    if strict_location:
        # Folders, not files: an environment's python is a link, and where the
        # link sits matters as much as where it points.
        for folder in (given.parent, real.parent):
            if in_temp_dir(folder):
                return REASON_TEMPORARY
            if _under(folder, SHARED_ROOTS) or _under(folder.resolve(), SHARED_ROOTS):
                return REASON_SHARED
    try:
        mode = real.stat().st_mode
    except OSError:
        return REASON_NOT_FOUND
    if not stat.S_ISREG(mode) or not os.access(real, os.X_OK):
        return REASON_NOT_INTERPRETER
    env = _environment_root(given)
    running = real == Path(sys.executable).resolve()
    if env is None and not running:
        return REASON_NOT_ENVIRONMENT
    guarded = [real, real.parent]
    if env is not None:
        guarded += [env, env / "bin", env / "pyvenv.cfg"]
    if not all(_safely_owned(p) for p in guarded):
        return REASON_WRITABLE
    return ""


__all__ = [
    "REASON_NOT_ABSOLUTE",
    "REASON_NOT_ENVIRONMENT",
    "REASON_NOT_FOUND",
    "REASON_NOT_INTERPRETER",
    "REASON_SHARED",
    "REASON_TEMPORARY",
    "REASON_WRITABLE",
    "SHARED_ROOTS",
    "refusal",
]
