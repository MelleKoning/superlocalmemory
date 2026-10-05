# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""A maintenance pass must not move a memory to the archive by itself.

4.1.15 (#136) placed every NEW memory at radius 1 - R(t), its retention. The
pass that runs afterwards on every tick still advanced each memory one step of
the Langevin simulation and wrote the tier that step produced. That simulation
has one stable place, the edge of the ball: near the edge its pull inward
fades to nothing while its outward push does not. Measured: a memory used
1,000 times at full importance, saved today, reaches radius 0.99 (archived)
within about 800 passes; an ordinary one within about 200.

Archived memories are removed from every recall. On a real five-month-old
store this left 67% of memories unreachable, including ones saved the week
before and ones used often, while their own retention read 0.8-1.0.

The tier a pass writes is now the one the memory's retention gives — the same
relationship the seed already uses — however many passes run.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.core.maintenance import (
    _LANGEVIN_DIM,
    _langevin_pass,
    _retention_radius,
    run_maintenance,
)
from superlocalmemory.math.langevin import _RADIUS_COLD, LangevinDynamics
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.schema import create_all_tables

_PROFILE = "default"


def _edge_position(seed: int) -> list[float]:
    """Where the old pass left memories: at the edge of the ball (0.99)."""
    d = np.random.default_rng(seed).standard_normal(_LANGEVIN_DIM)
    return (d / np.linalg.norm(d) * 0.99).tolist()


@pytest.fixture()
def store(tmp_path: Path) -> DatabaseManager:
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(path))
    create_all_tables(conn)
    conn.execute("INSERT INTO memories (memory_id, profile_id, content) "
                 "VALUES ('m1', ?, 'source')", (_PROFILE,))
    week_ago = (datetime.now(UTC) - timedelta(days=7)).isoformat()
    for i, (fid, access) in enumerate((("recent", 0), ("used", 40))):
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
            " lifecycle, access_count, importance, created_at, langevin_position)"
            " VALUES (?, 'm1', ?, ?, 'archived', ?, 0.5, ?, ?)",
            (fid, _PROFILE, f"synthetic fact {fid}", access, week_ago,
             str(_edge_position(i))),
        )
        conn.execute(
            "INSERT INTO fact_retention (fact_id, profile_id, lifecycle_zone,"
            " retention_score) VALUES (?, ?, 'archive', 0.95)", (fid, _PROFILE))
    conn.commit()
    conn.close()
    return DatabaseManager(str(path))


def _zone(db: DatabaseManager, fact_id: str) -> str:
    rows = db.execute("SELECT lifecycle_zone FROM fact_retention WHERE fact_id=?",
                      (fact_id,))
    return str(dict(rows[0])["lifecycle_zone"])


class TestAMaintenancePassFollowsRetention:
    def test_a_week_old_memory_left_at_the_edge_comes_back(self, store) -> None:
        config = SLMConfig.default()
        run_maintenance(store, config, _PROFILE)
        assert _zone(store, "recent") != "archive"
        assert _zone(store, "used") != "archive"

    @pytest.mark.parametrize("access,importance,age", [
        (0, 0.0, 30.0), (5, 0.5, 1.0), (1000, 1.0, 0.0)])
    def test_many_passes_never_push_a_memory_out(self, access, importance, age) -> None:
        ld = LangevinDynamics(dim=_LANGEVIN_DIM, dt=0.005, temperature=0.3)
        fact = {"fact_id": "f", "position": _edge_position(7), "access_count": access,
                "age_days": age, "importance": importance}
        for _ in range(400):
            (result,) = _langevin_pass(ld, [fact])
            fact = {**fact, "position": result["position"]}
        radius = float(np.linalg.norm(fact["position"]))
        assert radius == pytest.approx(_retention_radius(access, age, importance), abs=1e-9)
        assert radius < _RADIUS_COLD
        assert result["lifecycle"] != "archived"

    def test_a_memory_unused_for_years_is_still_archived(self) -> None:
        ld = LangevinDynamics(dim=_LANGEVIN_DIM, dt=0.005, temperature=0.3)
        fact = {"fact_id": "old", "position": _edge_position(3), "access_count": 0,
                "age_days": 3 * 365.0, "importance": 0.0}
        (result,) = _langevin_pass(ld, [fact])
        assert result["lifecycle"] == "archived"

    def test_the_same_memory_gets_the_same_tier_every_pass(self) -> None:
        ld = LangevinDynamics(dim=_LANGEVIN_DIM, dt=0.005, temperature=0.3)
        fact = {"fact_id": "f", "position": _edge_position(1), "access_count": 2,
                "age_days": 90.0, "importance": 0.3}
        tiers = set()
        for _ in range(50):
            (result,) = _langevin_pass(ld, [fact])
            fact = {**fact, "position": result["position"]}
            tiers.add(result["lifecycle"])
        assert len(tiers) == 1
