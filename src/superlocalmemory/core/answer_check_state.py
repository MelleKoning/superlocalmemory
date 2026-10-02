# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The answer check's settings: one file, whatever the operating mode.

Every ``retrieval.sufficiency_*`` setting — which check runs, the hosted
check's consent, its reordering and that consent, the provider, where the
on-device model lives — is the person's choice, not a property of Mode A, B
or C. Kept inside the per-mode config copies (``mode_a/b/c.json``) it was
restored by every mode switch, so a consent withdrawn in one mode came back
on switching back, and the dashboard routes could write one file while every
decision read another.

So, once this file exists, it is the only place those settings are read
from: ``SLMConfig.load()`` and ``SLMConfig.for_mode()`` lay it over whatever
the config file says, and ``SLMConfig.save()`` stops writing copies of them.
Before it exists (an install that never touched the answer check), the
config file's own values are used exactly as before.

Damage fails closed: an unreadable or malformed file is read as "no consent,
no reordering" — never as a yes — so a corrupt file can only ever turn the
hosted check off.

Stdlib only, and no import of ``core.config`` at module level (config imports
this module); the field list is read from ``RetrievalConfig`` when needed.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

STATE_FILE = "answer_check.json"
PREFIX = "sufficiency_"

#: What a damaged file is read as. Only consents are forced: every one of them
#: must be the boolean True to send anything anywhere, so False is the safe value.
FAIL_CLOSED: Mapping[str, bool] = {
    "sufficiency_jev_consent": False,
    "sufficiency_jev_rerank": False,
    "sufficiency_jev_rerank_consent": False,
}

_PROCESS_LOCK = threading.RLock()


def state_path(base_dir: Path) -> Path:
    return Path(base_dir) / STATE_FILE


def exists(base_dir: Path) -> bool:
    return state_path(base_dir).is_file()


def field_defaults() -> dict[str, Any]:
    """Every answer-check setting and its shipped default."""
    from superlocalmemory.core.config import RetrievalConfig

    default = RetrievalConfig()
    return {f.name: getattr(default, f.name)
            for f in fields(RetrievalConfig) if f.name.startswith(PREFIX)}


def _type_ok(value: Any, default: Any) -> bool:
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, type(default))


def sanitize(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Only known settings, each of the type its default has. A hand-edited
    ``"true"`` for a consent is dropped, never read as a yes."""
    defaults = field_defaults()
    return {name: raw[name] for name in defaults
            if name in raw and _type_ok(raw[name], defaults[name])}


def snapshot(retrieval: Any) -> dict[str, Any]:
    """The answer-check settings of one ``RetrievalConfig``, as a plain dict."""
    return {name: getattr(retrieval, name, default)
            for name, default in field_defaults().items()}


def _load(base_dir: Path) -> tuple[dict[str, Any] | None, bool]:
    """(settings, damaged). (None, False) when the file does not exist yet."""
    path = state_path(base_dir)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, False
    except OSError as exc:
        logger.warning("Answer check settings unreadable (%s); online check kept off",
                       type(exc).__name__)
        return dict(FAIL_CLOSED), True
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        logger.warning("Answer check settings file is damaged; online check kept off")
        return dict(FAIL_CLOSED), True
    clean = sanitize(data)
    # A consent that is missing, or not the boolean True/False, is a no.
    return {**clean, **{k: False for k in FAIL_CLOSED if k not in clean}}, False


def read(base_dir: Path) -> dict[str, Any] | None:
    """The stored settings, or None when the file does not exist yet.

    Lock-free: every write replaces the file atomically, so a reader sees the
    old file or the new one, never half of either.
    """
    return _load(base_dir)[0]


def overlay(retrieval: Any, base_dir: Path) -> Any:
    """``retrieval`` with the stored settings laid over it — a new object.

    Unchanged when nothing is stored yet.
    """
    stored = read(base_dir)
    if not stored:
        return retrieval
    return replace(retrieval, **stored)


@contextmanager
def _locked(base_dir: Path) -> Iterator[None]:
    lock_path = Path(base_dir) / f".{STATE_FILE}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _PROCESS_LOCK, lock_path.open("a+b") as handle:
        try:
            import fcntl
        except ImportError:  # Windows: one process at a time is the best we get
            yield
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                             dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(values), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def update(
    base_dir: Path,
    change: Callable[[dict[str, Any]], Mapping[str, Any]],
    *,
    seed: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply ``change`` under the cross-process lock; returns what is now stored.

    ``change`` receives a copy of the current settings and returns the new
    ones (a dict or any mapping); it must not rely on mutating its argument.
    ``seed`` supplies the starting point when nothing is stored yet — the
    settings in effect right now, so creating the file changes nothing by
    itself. A damaged file starts from ``seed`` with every consent forced off.
    The file always holds every setting, so no field can ever fall back to a
    stale per-mode copy.
    """
    with _locked(base_dir):
        path = state_path(base_dir)
        stored, damaged = _load(base_dir)
        base = field_defaults()
        if stored is None or damaged:
            base = {**base, **sanitize(seed())}
        if stored is not None:
            base = {**base, **stored}
        result = {**base, **sanitize(change(dict(base)))}
        _write(path, result)
    if stored is None:
        _strip_copies(base_dir)
    return dict(result)


#: The config files that carried copies of these settings before they had a home.
_COPY_HOLDERS = ("config.json", "mode_a.json", "mode_b.json", "mode_c.json")


def _strip_copies(base_dir: Path) -> None:
    """Once, when this file is first created: take the old copies out of the
    config files, so nothing can ever fall back to a consent kept there.
    Best effort — a copy left behind is ignored while this file exists."""
    for name in _COPY_HOLDERS:
        path = Path(base_dir) / name
        try:
            _strip_one(path)
        except (OSError, ValueError) as exc:
            logger.warning("Answer check: could not tidy %s (%s); it is ignored",
                           name, type(exc).__name__)


def _strip_one(path: Path) -> None:
    if not path.is_file():
        return
    from superlocalmemory.server.config_file import update_config

    def drop(data: dict) -> None:
        retrieval = data.get("retrieval")
        if isinstance(retrieval, dict):
            for key in [k for k in retrieval if k.startswith(PREFIX)]:
                del retrieval[key]

    data = json.loads(path.read_text(encoding="utf-8"))
    retrieval = data.get("retrieval") if isinstance(data, dict) else None
    if isinstance(retrieval, dict) and any(k.startswith(PREFIX) for k in retrieval):
        update_config(path, drop)


def ensure(base_dir: Path, *, seed: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
    """Create the file from ``seed`` when it does not exist yet; never changes
    an existing one. Returns what is stored."""
    if exists(base_dir):
        return read(base_dir) or {}
    return update(base_dir, lambda current: current, seed=seed)


# -- what core/config.py calls ---------------------------------------------------

def overlay_safely(retrieval: Any, base_dir: Path) -> Any:
    """``overlay()`` that never fails a config load: any error reads as "no consent"."""
    try:
        return overlay(retrieval, Path(base_dir))
    except Exception as exc:  # noqa: BLE001 — a config load must never fail on this
        logger.warning("Answer check settings could not be read (%s); online check kept off",
                       type(exc).__name__)
        return replace(retrieval, **FAIL_CLOSED)


def remember_baseline(config: Any) -> None:
    """What the answer check was set to when this config object was built, so a
    later save can tell an explicit change from a stale copy."""
    config._answer_check_baseline = snapshot(config.retrieval)


def retrieval_to_save(config: Any, base_dir: Path, *, mode_change: bool) -> dict[str, Any]:
    """The ``retrieval`` section ``SLMConfig.save()`` writes, keeping this file right.

    A setting changed on ``config`` since it was loaded (the setup wizard
    choosing the on-device check, say) is written here. Nothing else is: a copy
    loaded before someone withdrew consent and saved later for an unrelated
    reason must not undo the withdrawal, and a mode switch (``mode_change``)
    never changes these settings at all. Once this file exists, config files
    carry no copy of them.
    """
    section = asdict(config.retrieval)
    baseline = getattr(config, "_answer_check_baseline", None)
    current = snapshot(config.retrieval)
    changed = {} if (mode_change or baseline is None) else {
        k: v for k, v in current.items() if baseline.get(k) != v}
    if changed:
        update(base_dir, lambda stored: {**stored, **changed}, seed=lambda: baseline)
        config._answer_check_baseline = {**baseline, **changed}
    if exists(base_dir):
        section = {k: v for k, v in section.items() if not k.startswith(PREFIX)}
    return section


__all__ = [
    "FAIL_CLOSED",
    "PREFIX",
    "STATE_FILE",
    "ensure",
    "exists",
    "field_defaults",
    "overlay",
    "overlay_safely",
    "read",
    "remember_baseline",
    "retrieval_to_save",
    "sanitize",
    "snapshot",
    "state_path",
    "update",
]
