# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The daemon's background fill of the entity index (storage/entity_index.py).

Runs after the M028 backfill, in the same task, one short batch per tick, and
publishes durable progress on ``application.state.entity_index_status`` for the
status report. Lookups keep using the full scan until it reports complete, so
the store answers exactly as before while this runs.

Q7 (2026-10-06): ``run_entity_index_backfill`` still returns once that
one-time backfill reports complete -- an existing, tested contract
(``app.state.entity_index_status["state"] == "complete"`` after it returns)
that must keep meaning exactly that. The daemon's startup task chains
``run_entity_index_gap_sweep`` after it, which never returns (short of
cancellation): it runs ``storage/entity_index.py``'s never-terminal gap
sweep for the life of the process, so a fact written after the backfill's
snapshot by a path that bypasses the real-time indexing hook (an older
version reached by a downgrade, a restore, a direct import) is found and
repaired whenever it was introduced, not only if it existed at this one
startup. Its progress is published separately, on
``application.state.entity_index_gap_status``, never touching
``entity_index_status``.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_MIN_RETRY_SECONDS = 0.05
_MAX_RETRY_SECONDS = 30.0

#: Pause between batches, at most. Until the fill completes, link lookups scan
#: the facts table (exact, but 60 ms or more per entity on a large store), so
#: the fill should finish in minutes; a batch holds the write lock for about
#: 50 ms, so a 0.25 s pause still leaves the store free four fifths of the time.
MAX_TICK_SECONDS = 0.25

#: Q7: once a gap-sweep lap finds nothing to repair, back off exponentially
#: (capped at an hour) so an otherwise-healthy store does not pay a
#: continuous full-table background scan forever. Finding and fixing even
#: one fact resets the backoff to the normal tick pace immediately, so an
#: actual gap is still chased quickly.
_GAP_IDLE_BACKOFF_FLOOR_SECONDS = 1.0
_GAP_IDLE_BACKOFF_CEILING_SECONDS = 3600.0


def _publish(application, status: dict, **extra) -> None:
    application.state.entity_index_status = {
        **status, "source": "startup_background_repair", **extra,
    }


def _publish_gap(application, status: dict, **extra) -> None:
    application.state.entity_index_gap_status = {
        **status, "source": "background_gap_sweep", **extra,
    }


def gap_sweep_delay(consecutive_empty_laps: int, tick_seconds: float) -> float:
    """Seconds to sleep before the next gap-sweep batch.

    Pure function of the idle streak so the backoff policy is testable
    without a database or an event loop.
    """
    if consecutive_empty_laps <= 0:
        return min(MAX_TICK_SECONDS, max(0.0, float(tick_seconds)))
    return min(
        _GAP_IDLE_BACKOFF_CEILING_SECONDS,
        _GAP_IDLE_BACKOFF_FLOOR_SECONDS * (2 ** min(consecutive_empty_laps - 1, 16)),
    )


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
        await asyncio.sleep(min(MAX_TICK_SECONDS, max(0.0, float(tick_seconds))))


async def run_entity_index_gap_sweep(
    application, memory_db_path: Path, *, batch_size: int, tick_seconds: float,
) -> None:
    """Run the never-terminal coverage-gap sweep forever (Q7).

    Meant to be chained after ``run_entity_index_backfill`` returns, in the
    same background task, for the life of the process: returns only on
    cancellation (daemon shutdown) or a non-``sqlite3.Error`` exception.
    """
    from superlocalmemory.storage import entity_index

    gap_failures = 0
    consecutive_empty_laps = 0
    while True:
        try:
            result = await asyncio.to_thread(
                entity_index.repair_coverage_gap, Path(memory_db_path),
                batch_size=batch_size, max_batches=1,
            )
            gap_status = await asyncio.to_thread(
                entity_index.gap_sweep_status, Path(memory_db_path),
            )
        except sqlite3.Error as exc:
            gap_failures += 1
            delay = min(_MAX_RETRY_SECONDS,
                        max(_MIN_RETRY_SECONDS, tick_seconds) * (2 ** min(gap_failures - 1, 10)))
            logger.warning("Entity index gap sweep will retry in %.1fs: %s", delay, exc)
            _publish_gap(
                application, getattr(application.state, "entity_index_gap_status", {}) or {},
                state="retrying", last_error=type(exc).__name__,
                retry_attempt=gap_failures, retry_delay_seconds=delay,
            )
            await asyncio.sleep(delay)
            continue
        gap_failures = 0
        lap_complete = result["laps_completed"] > 0
        if result["inserted"] > 0:
            consecutive_empty_laps = 0
        elif lap_complete:
            consecutive_empty_laps += 1
        _publish_gap(application, gap_status, retry_attempt=0, retry_delay_seconds=0.0,
                     consecutive_empty_laps=consecutive_empty_laps)
        # The backoff paces how soon the NEXT lap starts once a whole lap in
        # a row found nothing -- it must not also slow down the batches
        # WITHIN an active lap, or a lap that still has real work left would
        # crawl at the same backed-off pace instead of finishing promptly.
        delay = (
            gap_sweep_delay(consecutive_empty_laps, tick_seconds) if lap_complete
            else min(MAX_TICK_SECONDS, max(0.0, float(tick_seconds)))
        )
        await asyncio.sleep(delay)


__all__ = ["gap_sweep_delay", "run_entity_index_backfill", "run_entity_index_gap_sweep"]
