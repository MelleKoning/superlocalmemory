# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The safety copy a start has already taken, and when a later pass may rely on it.

A start runs two migration passes against the same store: the eager pass
(``apply_all``) before the engine exists, and the deferred pass
(``apply_deferred``) inside ``MemoryEngine.initialize()`` once the engine has
bootstrapped its tables. Each used to copy the whole store, seconds apart. On a
2 GB store that is 4 GB per start, and because retention keeps two generations,
one start filled both slots and evicted the copy taken before the previous
upgrade -- the only rollback point older than this start.

Between the two passes the daemon start path writes nothing but the upgrade
itself (``server/unified_daemon.py`` lifespan, in order):

  1. ``apply_all``                        -- the copy is taken here
  2. stale-artifact reap, expired mesh-lease cleanup, orphan-process reap
  3. ``MemoryEngine(config).initialize()`` -- schema bootstrap, then
     ``apply_all`` and ``apply_deferred`` back to back (``core/engine.py``)
  4. writer lease, journal replay, background writers  -- ordinary writes

The lifespan runs synchronously before uvicorn creates its server, so no request
is served in that window, and the pending materializer waits for the engine to
be published after step 4. The only writes in step 2 are the deletion of mesh
leases that have already expired -- state the next start re-derives. So the
copy from step 1 is a faithful "before" for everything step 3 changes.

The copy is NOT reused when:

  * another live daemon is writing to the same data directory, at either pass
    -- by its pid file, or by the store's writer lease naming a live process;
  * a deferred pass already ran for it -- the first one claims it, used or not.
    The daemon's own deferred pass in step 4 runs after the writer lease and the
    journal replay, so it always copies for itself if it has anything to apply;
  * the process that took it is not this one (a fork inherits module state);
  * any of its files is no longer on disk.

Writers in OTHER processes that are not daemons (a direct-engine CLI write, the
Python API) are not fenced by any of this -- nor by the second copy it replaces,
which could equally be taken a moment before such a write.

One slot, not a map: the state is bounded by construction.
"""

from __future__ import annotations

import json
import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path

_COPY_SUFFIX = "-pre-migration.db"


@dataclass(frozen=True, slots=True)
class BootCopy:
    """A generation the eager pass wrote in this process."""

    learning_db: str
    memory_db: str
    directory: Path
    files: tuple[Path, ...]
    pid: int


_lock = threading.Lock()
_held: BootCopy | None = None


def _key(path: Path) -> str:
    """One spelling per file. The daemon passes resolved paths and the engine
    joins them from config, so the same store can arrive named two ways."""
    return os.path.normcase(str(Path(path).expanduser().resolve(strict=False)))


def copy_names(directory: Path) -> frozenset[str]:
    """Names of the safety copies directly inside ``directory``."""
    try:
        with os.scandir(directory) as entries:
            return frozenset(
                entry.name for entry in entries
                if entry.name.endswith(_COPY_SUFFIX)
                and entry.is_file(follow_symlinks=False)
            )
    except OSError:
        return frozenset()


def remember(
    learning_db: Path, memory_db: Path, directory: Path, before: frozenset[str],
) -> None:
    """Record the generation the eager pass just wrote (the names new since ``before``)."""
    written = copy_names(directory) - before
    record = BootCopy(
        learning_db=_key(learning_db),
        memory_db=_key(memory_db),
        directory=Path(directory),
        files=tuple(Path(directory) / name for name in sorted(written)),
        pid=os.getpid(),
    ) if written else None
    global _held
    with _lock:
        _held = record


def forget() -> None:
    """Drop any recorded copy, so the next deferred pass takes its own."""
    global _held
    with _lock:
        _held = None


def claim(learning_db: Path, memory_db: Path) -> BootCopy | None:
    """Take the recorded copy for this pair of databases. Clears it either way."""
    global _held
    with _lock:
        held, _held = _held, None
    if held is None or held.pid != os.getpid():
        return None
    if (held.learning_db, held.memory_db) != (_key(learning_db), _key(memory_db)):
        return None
    return held


#: ``lease_holder`` when the lease is held but its holder cannot be read.
UNKNOWN_HOLDER = -1


def another_writer_holds(memory_db: Path) -> bool:
    """A live process other than this one holds the store's writer lease."""
    return lease_holder(memory_db) is not None


def lease_holder(memory_db: Path) -> int | None:
    """The pid of a live process other than this one holding the writer lease.

    None when nobody else holds it; ``UNKNOWN_HOLDER`` when it is held but the
    record cannot be read.

    ``daemon.pid`` cannot answer this during a start: the new daemon writes its
    own pid there before its lifespan runs, so an old daemon still serving on
    another port is invisible in it. The lease file the write coordinator keeps
    beside the store records its holder, and is read here without locking it --
    taking the lock, even briefly, could make a daemon that is claiming it at
    that instant believe another one is serving and exit. A stale record (the
    holder died) names a pid that is no longer alive. A record that cannot be
    read because it is locked (Windows) counts as held.
    """
    lease = Path(memory_db).with_name(Path(memory_db).name + ".writer.lock")
    try:
        raw = lease.read_bytes()
    except FileNotFoundError:
        return None
    except PermissionError:
        return UNKNOWN_HOLDER
    except OSError:
        return None
    try:
        pid = int(json.loads(raw)["pid"])
    except (ValueError, KeyError, TypeError):
        return None
    if pid <= 0 or pid == os.getpid():
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except OSError:
        return pid    # alive, owned by someone else
    return pid


def still_on_disk(copy: BootCopy) -> bool:
    """Every file of the copy is still a regular file where it was written."""
    for path in copy.files:
        try:
            if not stat.S_ISREG(os.lstat(path).st_mode):
                return False
        except OSError:
            return False
    return bool(copy.files)
