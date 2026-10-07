# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Keys of hosted embedding models that are not the live one, kept 0600.

A re-index job row never holds a key. While a switch to a hosted model runs,
that model's key has to survive a restart, and after the switch the previous
model's key has to survive until ``slm embedder forget-previous`` so a
rollback can still use it. config.json holds only the live model's key (as it
always has); these two roles live here, beside it, with the same protection.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

FILE_NAME = "embedding-reindex-secrets.json"
TARGET = "target"
PREVIOUS = "previous"


def _path(data_root: Path) -> Path:
    return Path(data_root) / FILE_NAME


def _read(data_root: Path) -> dict:
    path = _path(data_root)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write(data_root: Path, data: dict) -> None:
    path = _path(data_root)
    if not data:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{FILE_NAME}.")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def get(data_root: Path, role: str) -> str:
    return str(_read(data_root).get(role) or "")


def put(data_root: Path, role: str, key: str) -> None:
    data = _read(data_root)
    if key:
        data[role] = key
    else:
        data.pop(role, None)
    _write(data_root, data)


def drop(data_root: Path, role: str) -> None:
    put(data_root, role, "")


__all__ = ["FILE_NAME", "PREVIOUS", "TARGET", "drop", "get", "put"]
