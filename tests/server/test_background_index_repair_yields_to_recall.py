# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The entity-index sweep waits while a person's recall is running.

After every start it re-reads the whole store in quick laps (250-fact windows,
0.25 s apart, then 1, 2, 4 s ...). Sampled on a copy of a 22k-fact store, it
was busy 40-70% of every 10 s for the first two minutes, beside recalls that
took 2-10 s. Other background work (the materializer, the embedding backfill)
already waits for recalls; this one did not.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from superlocalmemory.core import recall_gate
from superlocalmemory.server.entity_index_repair import (
    run_entity_index_backfill,
    run_entity_index_gap_sweep,
)


def _drive(runner, tmp_path: Path, name: str) -> list[tuple[str, int]]:
    calls: list[tuple[str, int]] = []

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append((fn.__name__, recall_gate.in_flight()))
        if fn.__name__ == name:
            raise asyncio.CancelledError  # one batch is enough to see when it ran
        return {"state": "running"}

    async def fake_sleep(seconds: float) -> None:
        if recall_gate.in_flight():
            recall_gate.end_recall()  # the person's recall finishes meanwhile

    recall_gate.begin_recall()
    try:
        with patch("asyncio.sleep", side_effect=fake_sleep), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            try:
                asyncio.run(runner(SimpleNamespace(state=SimpleNamespace()), tmp_path / "x.db",
                                   batch_size=50, tick_seconds=0.05))
            except asyncio.CancelledError:
                pass
    finally:
        while recall_gate.in_flight():
            recall_gate.end_recall()
    return calls


def test_the_gap_sweep_starts_a_batch_only_once_the_recall_is_done(tmp_path: Path) -> None:
    calls = _drive(run_entity_index_gap_sweep, tmp_path, "repair_coverage_gap")
    assert calls and calls[0] == ("repair_coverage_gap", 0), calls


def test_the_backfill_waits_for_recalls_too(tmp_path: Path) -> None:
    calls = _drive(run_entity_index_backfill, tmp_path, "backfill")
    assert calls and calls[0] == ("backfill", 0), calls
