# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Answer-check thresholds measured on YOUR store, one per judge.

The built-in thresholds in ``judge_recipe.CALIBRATIONS`` were measured on one
real store (Laya) and on public synthetic sets (Jev). A judge can be far more
or less cautious on another person's memories, so a threshold measured on
this store, by ``scripts/quality/answer_quality_eval.py judges``, may replace
the built-in one for one (judge, recipe) pair:

    <data dir>/answer_check_calibration.json
    {"schema": "superlocalmemory.answer-check-calibration/v1",
     "entries": [{"backend": "laya", "recipe_id": "sufficiency-v1",
                  "threshold": 0.45, "answered": 38, "unanswered": 14,
                  "measured_at": "2026-10-06"}]}

An entry is used only when it is complete and was measured on enough of both
kinds of question to say anything (``MIN_ANSWERED`` / ``MIN_UNANSWERED``). A
file that cannot be used is reported in the log and ignored as a whole, so a
typo never half-applies: the built-in thresholds stay in force.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

FILE_NAME = "answer_check_calibration.json"
SCHEMA = "superlocalmemory.answer-check-calibration/v1"
BACKENDS = frozenset({"laya", "jev", "jev-listwise"})
#: Fewer than this of either kind and a threshold is a guess, not a measurement.
MIN_ANSWERED = 20
MIN_UNANSWERED = 10
STATUS = "measured_on_this_store"


class CalibrationFileError(ValueError):
    """The calibration file cannot be used as written."""


def _count(entry: dict, key: str, minimum: int, where: str) -> int:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CalibrationFileError(
            f"{where}: {key} must be a whole number of at least {minimum}")
    return value


def parse_entries(payload: object) -> dict[tuple[str, str], float]:
    """(backend, recipe_id) -> threshold, or CalibrationFileError."""
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise CalibrationFileError(f"schema must be {SCHEMA!r}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise CalibrationFileError("entries must be a non-empty list")
    out: dict[tuple[str, str], float] = {}
    for index, entry in enumerate(entries):
        where = f"entries[{index}]"
        if not isinstance(entry, dict):
            raise CalibrationFileError(f"{where} must be an object")
        backend = entry.get("backend")
        recipe = entry.get("recipe_id")
        if backend not in BACKENDS:
            raise CalibrationFileError(
                f"{where}: backend must be one of {', '.join(sorted(BACKENDS))}")
        if not isinstance(recipe, str) or not recipe:
            raise CalibrationFileError(f"{where}: recipe_id is required")
        threshold = entry.get("threshold")
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0):
            raise CalibrationFileError(f"{where}: threshold must be between 0 and 1")
        _count(entry, "answered", MIN_ANSWERED, where)
        _count(entry, "unanswered", MIN_UNANSWERED, where)
        if (backend, recipe) in out:
            raise CalibrationFileError(f"{where}: {backend}/{recipe} appears twice")
        out[(backend, recipe)] = float(threshold)
    return out


_cache_lock = threading.Lock()
_cache: dict[Path, tuple[float, dict[tuple[str, str], float]]] = {}


def _default_path() -> Path | None:
    try:
        from superlocalmemory.infra.data_root import state_path
        return state_path(FILE_NAME)
    except (ImportError, OSError, ValueError) as exc:
        logger.warning("answer-check calibration file location unknown: %s", exc)
        return None


def measured_thresholds(path: Path | None = None) -> dict[tuple[str, str], float]:
    """The thresholds the file at ``path`` (default: the data folder) sets.

    Empty when there is no file. Re-read when the file changes; an unusable
    file is logged and contributes nothing.
    """
    target = path if path is not None else _default_path()
    if target is None:
        return {}
    try:
        mtime = target.stat().st_mtime
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("answer-check calibration file unreadable (%s): %s", target, exc)
        return {}
    with _cache_lock:
        cached = _cache.get(target)
        if cached is not None and cached[0] == mtime:
            return dict(cached[1])
    try:
        parsed = parse_entries(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, CalibrationFileError) as exc:
        logger.warning(
            "answer-check calibration file %s is not used, built-in thresholds "
            "stay in force: %s", target, exc)
        parsed = {}
    with _cache_lock:
        _cache[target] = (mtime, parsed)
    return dict(parsed)


__all__ = [
    "BACKENDS",
    "CalibrationFileError",
    "FILE_NAME",
    "MIN_ANSWERED",
    "MIN_UNANSWERED",
    "SCHEMA",
    "STATUS",
    "measured_thresholds",
    "parse_entries",
]
