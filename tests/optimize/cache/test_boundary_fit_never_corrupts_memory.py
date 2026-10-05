# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""The cache-boundary fit is the one place SLM still reaches native LAPACK.

SciPy's L-BFGS-B factorizes a small matrix with ``dpotrf`` on every step, and
on macOS that is Accelerate's ``dpotrf``. Accelerate's ``dpotrf`` has a
reported out-of-bounds fault for large matrices (scipy/scipy#26145, seen at
order 2048 and 3000). Here the matrix order is the number of stored
corrections, so it is bounded by ``maxcor`` no matter how many samples come
in. These tests pin that bound and run the real fit under Guard Malloc.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from superlocalmemory.optimize.cache import boundary_store
from tests.helpers.native_guard import (
    GUARD_MALLOC_AVAILABLE,
    SKIP_REASON,
    assert_no_stray_write,
    run_under_guard_malloc,
)

_SMALL_ORDER_LIMIT = 10


def _samples(count: int, seed: int) -> list[tuple[float, int]]:
    rng = random.Random(seed)
    return [(rng.uniform(0.5, 1.0), rng.randint(0, 1)) for _ in range(count)]


def test_the_fit_bounds_the_matrix_lapack_factorizes(monkeypatch: pytest.MonkeyPatch) -> None:
    real_minimize = boundary_store._sp_minimize
    assert real_minimize is not None, "scipy is a core dependency"
    calls: list[dict] = []

    def _recording_minimize(*args, **kwargs):
        calls.append(kwargs)
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(boundary_store, "_sp_minimize", _recording_minimize)
    boundary_store._fit_logistic_mle(_samples(500, seed=1), 0.95, 10.0)

    assert len(calls) == 1
    assert calls[0]["method"] == "L-BFGS-B"
    assert len(calls[0]["x0"]) == 2
    max_corrections = calls[0]["options"].get("maxcor")
    assert max_corrections is not None, "the factorized matrix order must be bounded explicitly"
    assert max_corrections <= _SMALL_ORDER_LIMIT


@pytest.mark.skipif(not GUARD_MALLOC_AVAILABLE, reason=SKIP_REASON)
def test_fitting_the_cache_boundary_makes_no_stray_memory_write(tmp_path: Path) -> None:
    done = run_under_guard_malloc(
        """
        import random
        from superlocalmemory.optimize.cache.boundary_store import _fit_logistic_mle
        rng = random.Random(0)
        for count in (1, 2, 5, 20, 200, 2000):
            for _ in range(3):
                samples = [(rng.uniform(0.5, 1.0), rng.randint(0, 1)) for _ in range(count)]
                _fit_logistic_mle(samples, 0.95, 10.0)
        print("fitted")
        """,
        tmp_path,
    )
    assert_no_stray_write(done, "fitted")
