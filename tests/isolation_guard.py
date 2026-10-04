# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Keep tests out of the user's live SuperLocalMemory data.

Two layers, because each covers a hole in the other:

* :class:`LiveRootGuard` is installed as a process audit hook by
  ``tests/conftest.py``. It refuses every open, create, rename, delete, copy
  and SQLite connect under a live data root, and RECORDS each refusal, so a
  refusal that product code swallows (many state writers catch ``OSError``)
  still fails the test instead of passing silently.
* :func:`explicit_slm_root` is an autouse fixture that a module which writes
  runtime state imports directly. It pins ``SLM_DATA_DIR`` to the test's own
  ``tmp_path``, so the module stays inside it even when the root conftest is
  not loaded -- ``--noconftest`` (which this repo's CI uses for some files) or
  a ``--confcutdir`` below ``tests/``. That is how
  ``test_laya_runtime.py`` once wrote ``runtimes/laya/adopted.json`` into the
  real ``~/.superlocalmemory`` (2026-10-03 10:38): without the conftest,
  ``runtime_dir()`` resolved the default home root.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

_ROOT_ENV_KEYS = ("SLM_DATA_DIR", "SL_MEMORY_PATH", "SLM_HOME")

#: audit event -> indexes of its path arguments
_PATH_EVENTS: dict[str, tuple[int, ...]] = {
    "open": (0,),
    "os.mkdir": (0,),
    "os.rename": (0, 1),          # also raised by os.replace
    "os.remove": (0,),            # also raised by os.unlink
    "os.rmdir": (0,),
    "os.symlink": (0, 1),
    "os.link": (0, 1),
    "os.truncate": (0,),
    "os.chmod": (0,),
    "os.utime": (0,),
    "shutil.copyfile": (0, 1),
    "shutil.copytree": (0, 1),
    "shutil.move": (0, 1),
    "shutil.rmtree": (0,),
    "sqlite3.connect": (0,),
}


def _resolve(value: str | os.PathLike) -> Path:
    return Path(os.fsdecode(value)).expanduser().resolve(strict=False)


def _account_home() -> Path | None:
    """The account's home from the password database, whatever HOME says."""
    try:
        import pwd
    except ImportError:  # Windows
        return None
    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, OSError):
        return None


def live_data_roots(env: Mapping[str, str] | None = None) -> frozenset[Path]:
    """Every root that may hold live SLM state for the invoking user.

    Must be called before the test session rewrites HOME and SLM_DATA_DIR.
    The account home is always included, so pre-setting HOME or SLM_DATA_DIR
    in the invoking shell no longer moves the guard off the real store.
    """
    env = os.environ if env is None else env
    roots = {_resolve(env[key]) for key in _ROOT_ENV_KEYS if env.get(key)}
    homes = {Path.home(), _account_home()}
    roots.update(_resolve(home / ".superlocalmemory") for home in homes if home)
    return frozenset(roots)


class LiveRootGuard:
    """Audit hook that refuses, and remembers, any touch of a live root."""

    def __init__(self, protected: Iterable[Path]) -> None:
        self.protected = tuple(sorted({_resolve(p) for p in protected}))
        self._violations: list[str] = []
        self._lock = threading.Lock()

    def covers(self, path: str | os.PathLike) -> bool:
        candidate = _resolve(path)
        return any(candidate == root or candidate.is_relative_to(root)
                   for root in self.protected)

    def audit(self, event: str, args: tuple) -> None:
        indexes = _PATH_EVENTS.get(event)
        if not indexes:
            return
        for index in indexes:
            if index >= len(args):
                continue
            value = args[index]
            if isinstance(value, int) or not isinstance(value, (str, bytes, os.PathLike)):
                continue  # file descriptors and dir_fd-relative calls carry no path
            if event == "sqlite3.connect" and os.fsdecode(value) in ("", ":memory:"):
                continue
            if self.covers(value):
                message = f"pytest denied live SLM state ({event}): {_resolve(value)}"
                with self._lock:
                    self._violations.append(message)
                raise PermissionError(message)

    def drain(self) -> list[str]:
        """Refusals since the last call (each test checks and clears them)."""
        with self._lock:
            found, self._violations = self._violations, []
        return found


@pytest.fixture(autouse=True)
def explicit_slm_root(tmp_path, monkeypatch):
    """Pin this module's data root to ``tmp_path``, with or without conftest.

    The same root the conftest's per-test fixture selects, so a module behaves
    identically either way; the conftest is simply no longer what keeps it out
    of the live store.
    """
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SL_MEMORY_PATH", str(tmp_path / "wrong-legacy-alias"))
    monkeypatch.setenv("SLM_HOME", str(tmp_path / "wrong-hook-alias"))
    return tmp_path
