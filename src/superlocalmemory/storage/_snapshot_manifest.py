# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What a safety copy is, proven: its manifest, its checksum, and its contents.

A pre-migration copy used to be two files and a log line. Nothing recorded
which version it was taken before, how many memories it held, or what its bytes
were -- so nothing could later tell a user "this is the copy from before the
4.1.19 update, it holds 6,560 memories, and it is intact".

Each generation now gets ``manifest-<stamp>-pre-migration.json`` beside it,
written only AFTER both copies are renamed into place, so a manifest never
describes a copy that does not exist. Its checksums let a restore refuse a copy
whose bytes changed since it was taken (a failing disk, a sync client, a hand
edit) before anything is overwritten.

The manifest name does not end in ``.db``, so retention (``_snapshot_retention``)
never mistakes it for a copy; ``prune_orphan_manifests`` removes a manifest once
retention has removed the copies it describes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import stat
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from superlocalmemory.storage._durable_json import read_json, write_json_atomic

logger = logging.getLogger("superlocalmemory.storage.backup")

SNAPSHOT_SUFFIX = "-pre-migration.db"
MANIFEST_PREFIX = "manifest-"
MANIFEST_SUFFIX = "-pre-migration.json"
MANIFEST_FORMAT = 1
_STEMS = ("memory", "learning")
_PARTIAL_ABANDONED_AFTER_S = 3600.0
_CHUNK = 1 << 20


def package_version() -> str:
    """The installed SuperLocalMemory version, or "unknown"."""
    try:
        from importlib.metadata import version

        return version("superlocalmemory")
    except Exception:  # noqa: BLE001 - a missing dist-info is not an error here
        return "unknown"


def stamp_of(snapshot: Path) -> str | None:
    """``memory-<stamp>-pre-migration.db`` -> ``<stamp>``; None for other names."""
    name = Path(snapshot).name
    for stem in _STEMS:
        prefix = f"{stem}-"
        if name.startswith(prefix) and name.endswith(SNAPSHOT_SUFFIX):
            stamp = name[len(prefix):-len(SNAPSHOT_SUFFIX)]
            return stamp or None
    return None


def manifest_path(backups_root: Path, stamp: str) -> Path:
    return Path(backups_root) / f"{MANIFEST_PREFIX}{stamp}{MANIFEST_SUFFIX}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_immutable(path: Path) -> sqlite3.Connection:
    """Read a quiescent copy without creating -wal/-shm beside it."""
    return sqlite3.connect(f"{Path(path).absolute().as_uri()}?mode=ro&immutable=1", uri=True)


def _count(conn: sqlite3.Connection, sql: str) -> int | None:
    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.Error:
        return None
    return int(row[0]) if row and row[0] is not None else 0


def store_counts(path: Path, *, immutable: bool = True) -> dict[str, int | None]:
    """facts / memories / max_fact_rowid of a memory database (None if absent)."""
    if not Path(path).exists():
        return {"facts": None, "memories": None, "max_fact_rowid": None}
    conn = (_open_immutable(path) if immutable
            else sqlite3.connect(f"{Path(path).absolute().as_uri()}?mode=ro", uri=True))
    try:
        return {
            "facts": _count(conn, "SELECT COUNT(*) FROM atomic_facts"),
            "memories": _count(conn, "SELECT COUNT(*) FROM memories"),
            "max_fact_rowid": _count(conn, "SELECT MAX(rowid) FROM atomic_facts"),
        }
    finally:
        conn.close()


def verify_snapshot(
    snapshot: Path,
    expected_sha256: str | None,
    *,
    require_table: str | None = None,
) -> None:
    """Prove ``snapshot`` is the intact copy it claims to be, reading only.

    Raises ``SnapshotUnusableError`` when it is missing, empty, its bytes no
    longer match ``expected_sha256``, it fails SQLite's own check, it holds no
    tables, or it lacks ``require_table``.
    """
    from superlocalmemory.storage.backup import SnapshotUnusableError

    snapshot = Path(snapshot)
    if not snapshot.is_file() or snapshot.stat().st_size == 0:
        raise SnapshotUnusableError(f"safety copy is missing or empty: {snapshot.name}")
    if expected_sha256:
        actual = sha256_file(snapshot)
        if actual != expected_sha256:
            raise SnapshotUnusableError(
                f"safety copy {snapshot.name} changed after it was taken "
                f"(checksum {actual[:12]}... expected {expected_sha256[:12]}...)")
    try:
        conn = _open_immutable(snapshot)
        try:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if not tables:
                raise SnapshotUnusableError(f"safety copy holds no tables: {snapshot.name}")
            if require_table and require_table not in tables:
                raise SnapshotUnusableError(
                    f"safety copy {snapshot.name} has no {require_table} table")
            if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise SnapshotUnusableError(f"safety copy failed its check: {snapshot.name}")
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise SnapshotUnusableError(
            f"safety copy is not a readable database: {snapshot.name}") from exc


def check_copy_holds_the_store(
    snapshot: Path, before: int | None, after: int | None,
) -> dict[str, int | None]:
    """Refuse a memory copy whose fact count the source cannot explain.

    ``before`` and ``after`` are the live store's fact counts read just before
    and just after the copy. With nothing writing in between they are equal and
    the copy must hold exactly that many. A copy that passed SQLite's check but
    holds fewer (an empty or truncated database) is the failure this exists
    for: it would let the upgrade run with nothing to go back to.
    """
    from superlocalmemory.storage.backup import SnapshotUnusableError

    counts = store_counts(snapshot)
    known = [n for n in (before, after) if n is not None]
    if not known:
        return counts
    got = counts["facts"]
    low, high = min(known), max(known)
    if got is None or not (low <= got <= high):
        raise SnapshotUnusableError(
            f"safety copy {Path(snapshot).name} holds {got} memories but the store "
            f"held {low}" + (f"-{high}" if high != low else "")
            + "; the update will not run without a complete copy")
    return counts


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def write_generation_manifest(
    backups_root: Path,
    stamp: str,
    pairs: list[tuple[Path, Path]],
    *,
    reason: str,
    counts: dict[str, int | None] | None = None,
    checksums: dict[str, str] | None = None,
) -> Path:
    """Write ``manifest-<stamp>-pre-migration.json`` for copies already in place."""
    from superlocalmemory.storage._schema_version import read_schema_version

    checksums = dict(checksums or {})
    files: list[dict[str, Any]] = []
    for snapshot, db in pairs:
        snapshot = Path(snapshot)
        digest = checksums.get(snapshot.name) or sha256_file(snapshot)
        files.append({
            "db": Path(db).name, "db_path": str(db), "snapshot": snapshot.name,
            "size": snapshot.stat().st_size, "sha256": digest,
        })
    dbs = [Path(db) for _snap, db in pairs]
    data_root = dbs[0].parent if dbs else Path(backups_root).parent
    manifest = {
        "format": MANIFEST_FORMAT,
        "stamp": stamp,
        "reason": reason,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "from_version": _read_text(data_root / ".last_version"),
        "to_version": package_version(),
        "schema_version_before": max((read_schema_version(db) for db in dbs), default=0),
        "files": files,
        "counts": counts or {"facts": None, "memories": None, "max_fact_rowid": None},
    }
    return write_json_atomic(manifest_path(backups_root, stamp), manifest)


def read_manifest(backups_root: Path, stamp: str) -> dict[str, Any] | None:
    return read_json(manifest_path(backups_root, stamp))


def update_manifest(backups_root: Path, stamp: str, updates: dict[str, Any]) -> bool:
    """Merge ``updates`` into an existing manifest; False when there is none."""
    current = read_manifest(backups_root, stamp)
    if current is None:
        return False
    write_json_atomic(manifest_path(backups_root, stamp), {**current, **updates})
    return True


def _regular_children(root: Path) -> list[os.DirEntry]:
    with os.scandir(root) as entries:
        return [e for e in entries if not e.is_symlink() and e.is_file(follow_symlinks=False)]


def cleanup_stale_partials(backups_root: Path, *, now: float | None = None) -> list[Path]:
    """Unlink ``*-pre-migration.db.partial`` (+ -wal/-shm) untouched for an hour.

    A copy interrupted by a crash keeps its staging name for ever. Only regular
    files directly in ``backups_root`` whose name is exactly that shape are
    removed, each by its explicit path.
    """
    root = Path(backups_root)
    if not root.is_dir() or root.is_symlink():
        return []
    now = time.time() if now is None else now
    removed: list[Path] = []
    for entry in _regular_children(root):
        name = entry.name
        base = name
        for side in ("-wal", "-shm"):
            if base.endswith(".partial" + side):
                base = base[: -len(side)]
                break
        if not (base.endswith(SNAPSHOT_SUFFIX + ".partial")
                and stamp_of(Path(base[: -len(".partial")]))):
            continue
        path = root / name
        try:
            info = os.lstat(path)
        except OSError:
            continue
        if not stat.S_ISREG(info.st_mode) or now - info.st_mtime <= _PARTIAL_ABANDONED_AFTER_S:
            continue
        try:
            path.unlink()
            removed.append(path)
            logger.info("[SLM] Removed an interrupted safety copy: %s", path)
        except OSError as exc:
            logger.warning("[SLM] Could not remove %s: %s", path, exc)
    return removed


def prune_orphan_manifests(backups_root: Path) -> list[Path]:
    """Remove each manifest whose copies retention has already removed."""
    root = Path(backups_root)
    if not root.is_dir() or root.is_symlink():
        return []
    names = {e.name for e in _regular_children(root)}
    removed: list[Path] = []
    for name in sorted(names):
        if not (name.startswith(MANIFEST_PREFIX) and name.endswith(MANIFEST_SUFFIX)):
            continue
        stamp = name[len(MANIFEST_PREFIX):-len(MANIFEST_SUFFIX)]
        if any(f"{stem}-{stamp}{SNAPSHOT_SUFFIX}" in names for stem in _STEMS):
            continue
        path = root / name
        try:
            path.unlink()
            removed.append(path)
        except OSError as exc:
            logger.warning("[SLM] Could not remove %s: %s", path, exc)
    return removed


__all__ = [
    "MANIFEST_FORMAT", "SNAPSHOT_SUFFIX", "check_copy_holds_the_store",
    "cleanup_stale_partials", "manifest_path", "package_version",
    "prune_orphan_manifests", "read_manifest", "sha256_file", "stamp_of",
    "store_counts", "update_manifest", "verify_snapshot", "write_generation_manifest",
]
