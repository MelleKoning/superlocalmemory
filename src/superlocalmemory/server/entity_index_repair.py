# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The daemon's background fill of the entity index (storage/entity_index.py).

Runs after the M028 backfill, in the same task, one short batch per tick, and
publishes durable progress on ``application.state.entity_index_status`` for the
status report. Lookups keep using the full scan until it reports complete, so
the store answers exactly as before while this runs.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_MIN_RETRY_SECONDS = 0.05
_MAX_RETRY_SECONDS = 30.0


def _publish(application, status: dict, **extra) -> None:
    application.state.entity_index_status = {
        **status, "source": "startup_background_repair", **extra,
    }


async def run_entity_index_backfill(
    application, memory_db_path: Path, *, batch_size: int, tick_seconds: float,
) -> None:
    """Fill the index to completion; retry with backoff on a database error."""
    from superlocalmemory.storage import entity_index

    failures = 0
    while True:
        try:
            await asyncio.to_thread(
                entity_index.backfill, Path(memory_db_path),
                batch_size=batch_size, max_batches=1,
            )
            status = await asyncio.to_thread(entity_index.status, Path(memory_db_path))
        except sqlite3.Error as exc:
            failures += 1
            delay = min(_MAX_RETRY_SECONDS,
                        max(_MIN_RETRY_SECONDS, tick_seconds) * (2 ** min(failures - 1, 10)))
            logger.warning("Entity index backfill will retry in %.1fs: %s", delay, exc)
            _publish(application, getattr(application.state, "entity_index_status", {}) or {},
                     state="retrying", last_error=type(exc).__name__,
                     retry_attempt=failures, retry_delay_seconds=delay)
            await asyncio.sleep(delay)
            continue
        failures = 0
        _publish(application, status, retry_attempt=0, retry_delay_seconds=0.0)
        if status["state"] == "complete":
            return
        await asyncio.sleep(max(0.0, float(tick_seconds)))


__all__ = ["run_entity_index_backfill"]
