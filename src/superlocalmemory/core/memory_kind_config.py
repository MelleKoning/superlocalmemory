# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Memory-kind settings: one file for every mode, strict about consent.

What the settings decide: whether memories get a kind suggestion at all, which
already-running model may suggest one (``auto`` / ``rules`` / ``laya`` /
``jev`` / ``llm`` / ``off``), whether Jev may be used for typing, the
confidence below which a suggestion is shown as "not sure", whether confirmed
standing rules are given to every new session, and how fast a "Classify my
memories" run may go.

Why a file of its own (``memory_kinds.json`` beside ``config.json``). The
answer check learned this in 4.1.18: settings kept in the per-mode config
copies (``mode_a/b/c.json``) are restored by every mode switch, so a consent
withdrawn in one mode came back after switching away and back. Jev typing
consent sends every memory off the device, so it must not be resurrectable
that way. Once the file exists it is the only authority; before that a
``memory_kinds`` section in the config file is read (hand-edited installs).

Damage fails closed: an unreadable file reads as the defaults with no Jev
consent. ``jev_consent`` is only ever the literal boolean ``True`` — the same
rule the answer check's own consent follows — so ``"true"`` or ``1`` is "no".

Stdlib only, no import of ``core.config`` (config imports this module).
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATE_FILE = "memory_kinds.json"

#: Every value ``backend`` may hold. ``auto`` never resolves to Jev.
BACKENDS: tuple[str, ...] = ("auto", "rules", "laya", "jev", "llm", "off")

#: Facts per second a classification run may type, per backend. Rules cost
#: about 2 ms a memory; Laya about 55 ms a fact on the worker the answer check
#: shares; Jev sends one request of 8 memories every 2 s. The LLM entry is kept
#: for completeness: in 4.1.19 the local or cloud model types memories only as
#: they are saved (inside the extraction call), so a run over existing
#: memories uses the rules there.
_DEFAULT_RATES = {"rules": 500.0, "laya": 8.0, "jev": 4.0, "llm": 3.0}
#: Facts per batch (one batch = one write transaction). Rules: 50, measured on
#: a copy of a real 640 MB store whose rows carry two 3 KB vectors, so every
#: rewritten row moves ~10 KB into the WAL: 200 facts held the write lock
#: ~115 ms, 100 ~52 ms, 50 ~30 ms (max 40) against a 50 ms target. The runner
#: also halves a batch whose write ran over target (slower disks). Model
#: batches are 8: the classifier asks a model about at most 8 facts per call.
_DEFAULT_BATCH = {"rules": 50, "laya": 8, "jev": 8, "llm": 8}
_RATE_RANGE = (0.05, 10_000.0)
_BATCH_RANGE = (1, 500)

_LOCK = threading.RLock()


class FrozenMap(Mapping):
    """A read-only mapping that ``copy.deepcopy`` and ``dataclasses.asdict`` accept.

    ``types.MappingProxyType`` cannot be deep-copied, and ``SLMConfig`` holds
    this config; a config that breaks ``deepcopy`` breaks every caller that
    clones one.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", dict(data))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("FrozenMap is read-only")

    def __copy__(self) -> "FrozenMap":
        return self

    def __deepcopy__(self, memo: dict) -> "FrozenMap":
        return self

    def __reduce__(self):
        return (FrozenMap, (dict(self._data),))

    def __repr__(self) -> str:
        return f"FrozenMap({self._data!r})"


@dataclass(frozen=True)
class MemoryKindConfig:
    """The memory-kind settings. Read by attribute, with defaults, by WP-3/4/7."""

    enabled: bool = True
    backend: str = "auto"
    jev_consent: bool = False
    display_min_confidence: float = 0.20
    standing_rules_in_session: bool = True
    llm_extract_kinds: bool = True
    rate_per_second: Mapping[str, float] = field(
        default_factory=lambda: FrozenMap(_DEFAULT_RATES))
    batch_size: Mapping[str, int] = field(
        default_factory=lambda: FrozenMap(_DEFAULT_BATCH))


_BOOL_FIELDS = ("enabled", "standing_rules_in_session", "llm_extract_kinds")
#: What the settings route may change. Rates and batch sizes are tuning, not
#: a user choice, so they are read from the file but never offered.
USER_SETTINGS = ("enabled", "backend", "jev_consent", "standing_rules_in_session",
                 "display_min_confidence", "llm_extract_kinds")


def _number(value: Any, low: float, high: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) and low <= f <= high else None


def _rates(raw: Any) -> FrozenMap:
    out = dict(_DEFAULT_RATES)
    if isinstance(raw, Mapping):
        for key in out:
            value = _number(raw.get(key), *_RATE_RANGE)
            if value is not None:
                out[key] = value
    return FrozenMap(out)


def _batches(raw: Any) -> FrozenMap:
    out = dict(_DEFAULT_BATCH)
    if isinstance(raw, Mapping):
        for key in out:
            value = raw.get(key)
            if isinstance(value, int) and not isinstance(value, bool) \
                    and _BATCH_RANGE[0] <= value <= _BATCH_RANGE[1]:
                out[key] = value
    return FrozenMap(out)


def memory_kind_config_from(data: Any) -> MemoryKindConfig:
    """A validated config from any mapping. Unknown keys and bad values are
    dropped (bad values fall back to the default). Never raises."""
    if not isinstance(data, Mapping):
        return MemoryKindConfig()
    default = MemoryKindConfig()
    kwargs: dict[str, Any] = {}
    for name in _BOOL_FIELDS:
        value = data.get(name)
        kwargs[name] = value if isinstance(value, bool) else getattr(default, name)
    kwargs["jev_consent"] = data.get("jev_consent") is True
    backend = data.get("backend")
    backend = backend.strip().lower() if isinstance(backend, str) else ""
    kwargs["backend"] = backend if backend in BACKENDS else default.backend
    threshold = _number(data.get("display_min_confidence"), 0.0, 1.0)
    kwargs["display_min_confidence"] = (threshold if threshold is not None
                                        else default.display_min_confidence)
    kwargs["rate_per_second"] = _rates(data.get("rate_per_second"))
    kwargs["batch_size"] = _batches(data.get("batch_size"))
    return MemoryKindConfig(**kwargs)


def settings_dict(cfg: MemoryKindConfig) -> dict[str, Any]:
    """JSON-safe dict of every field."""
    out: dict[str, Any] = {}
    for f in fields(MemoryKindConfig):
        value = getattr(cfg, f.name)
        out[f.name] = dict(value) if isinstance(value, Mapping) else value
    return out


def state_path(base_dir: Path) -> Path:
    return Path(base_dir) / STATE_FILE


def _read_file(base_dir: Path) -> tuple[dict[str, Any] | None, bool]:
    """(stored values or None when absent, damaged?)."""
    path = state_path(base_dir)
    if not path.is_file():
        return None, False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, True
    if not isinstance(raw, dict):
        return None, True
    return raw, False


def load_memory_kind_config(base_dir: Path | None,
                            section: Any = None) -> MemoryKindConfig:
    """The settings in force: the settings file, else the config section.

    Never raises: any failure reads as the defaults without Jev consent.
    """
    try:
        if base_dir is not None:
            stored, damaged = _read_file(Path(base_dir))
            if damaged:
                logger.warning("Memory kind settings could not be read; "
                               "using the defaults with Jev typing off")
                return MemoryKindConfig()
            if stored is not None:
                return memory_kind_config_from(stored)
        return memory_kind_config_from(section)
    except Exception as exc:  # noqa: BLE001 — a config load must never fail on this
        logger.warning("Memory kind settings unavailable (%s); using the defaults",
                       type(exc).__name__)
        return MemoryKindConfig()


def _write(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(values), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_memory_kind_settings(base_dir: Path, changes: Mapping[str, Any],
                              *, seed: Any = None) -> MemoryKindConfig:
    """Apply ``changes`` (USER_SETTINGS keys only) and write the file atomically.

    ``seed`` is what to start from when no file exists yet (the config
    section). A damaged file is replaced by defaults plus the changes — never
    by a consent nobody gave.
    """
    with _LOCK:
        stored, damaged = _read_file(Path(base_dir))
        base = stored if stored is not None else ({} if damaged else (seed or {}))
        current = settings_dict(memory_kind_config_from(base))
        current.update({k: v for k, v in changes.items() if k in USER_SETTINGS})
        cfg = memory_kind_config_from(current)
        _write(state_path(Path(base_dir)), settings_dict(cfg))
        return cfg


__all__ = [
    "BACKENDS",
    "STATE_FILE",
    "USER_SETTINGS",
    "FrozenMap",
    "MemoryKindConfig",
    "load_memory_kind_config",
    "memory_kind_config_from",
    "save_memory_kind_settings",
    "settings_dict",
    "state_path",
]
