# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Small records that must survive a power cut: written whole or not at all.

Every intent, outcome, marker and manifest the upgrade and restore flow keeps
is written here: to a temporary sibling, flushed to disk, renamed over the
final name (atomic within one directory), and the directory flushed so the
rename itself survives. The file is readable by its owner only (0600) because
it names memory files and, for a restore, counts of a person's memories.

A reader never sees half a record: it sees the old one, the new one, or none.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return  # some platforms cannot open a directory; the file is durable
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    """Write ``payload`` to ``path`` durably and atomically, mode 0600."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    data = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)
    return path


def read_json(path: Path) -> dict[str, Any] | None:
    """The record at ``path``, or None when it is absent or not a JSON object."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> Path:
    """One JSON object per line, written with the same guarantees."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        for row in rows:
            os.write(fd, json.dumps(row, sort_keys=True, default=str).encode("utf-8") + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSON-lines file; a missing file is no rows. A bad line raises."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


__all__ = ["read_json", "read_jsonl", "write_json_atomic", "write_jsonl_atomic"]
