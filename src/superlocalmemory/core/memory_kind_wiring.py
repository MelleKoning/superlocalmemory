# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""How the engine's kind classifier sees what is running *now*.

The classifier is built once per engine, but three things it depends on change
while the daemon runs: the answer-check judge (a dashboard switch swaps
``retrieval_engine._sufficiency_judge``), the retrieval engine itself (a mode
change rebuilds it), and the memory-kind settings (the dashboard writes them).
So the classifier is handed suppliers and a live view, never snapshots: a
switch takes effect on the next memory with nothing rebuilt.

Nothing here builds a judge, a worker or a client (LLD I5). No judge means
the rules.
"""

from __future__ import annotations

import logging
from typing import Any

from superlocalmemory.core.memory_kind_config import MemoryKindConfig
from superlocalmemory.storage.models import Mode

logger = logging.getLogger(__name__)


class LiveKindConfig:
    """Attribute reads go to the engine's current ``memory_kinds`` settings."""

    __slots__ = ("_engine",)

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def __getattr__(self, name: str) -> Any:
        return getattr(current_config(self._engine), name)


def current_config(engine: Any) -> MemoryKindConfig:
    """The engine's memory-kind settings, or the defaults. Never raises."""
    cfg = getattr(getattr(engine, "_config", None), "memory_kinds", None)
    return cfg if isinstance(cfg, MemoryKindConfig) else MemoryKindConfig()


def live_judge(engine: Any) -> Any | None:
    """The answer-check judge running right now, or None."""
    retrieval = getattr(engine, "_retrieval_engine", None)
    return getattr(retrieval, "_sufficiency_judge", None)


def engine_mode(engine: Any) -> Mode:
    mode = getattr(getattr(engine, "_config", None), "mode", Mode.A)
    return mode if isinstance(mode, Mode) else Mode.A


def llm_available(engine: Any) -> bool:
    return getattr(engine, "_llm", None) is not None


def build_kind_classifier(engine: Any) -> Any | None:
    """The engine's ``KindClassifier``, or None if it cannot be built.

    ``None`` keeps every write working exactly as 4.1.18 did (I1): the
    materializer then stores only kinds a caller declared.
    """
    try:
        from superlocalmemory.encoding.memory_kind_classifier import KindClassifier

        return KindClassifier(
            config=LiveKindConfig(engine),
            mode=engine_mode(engine),
            judge_supplier=lambda: live_judge(engine),
            llm_available=llm_available(engine),
        )
    except Exception as exc:  # noqa: BLE001 — typing is optional, saving is not
        logger.warning("Memory kind suggestions unavailable (%s)", type(exc).__name__)
        return None


__all__ = ["LiveKindConfig", "build_kind_classifier", "current_config", "engine_mode",
           "live_judge", "llm_available"]
