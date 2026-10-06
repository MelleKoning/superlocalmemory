# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""When every outcome agrees, the semantic cache's boundary is the same on
every platform, and the same cache state always makes the same choice.

With only hits (or only misses) the fit has no interior optimum, and the
optimiser stopped wherever its tolerances tripped: a 1e-9 change in the input
moved gamma from 45.9 to 37.9 and the chance of skipping the cache at
similarity 1.0 from 0 to 8%. Windows' floating point landed on the second, and
an unseeded draw then made the cache miss now and then.
"""

from __future__ import annotations

import pytest

from superlocalmemory.optimize.cache.boundary_store import (
    PerItemBoundaryRecord,
    _fit_logistic_mle,
)

_HITS = [0.97, 0.98, 0.99, 0.96, 0.95]


def _learned(shift: float, correct: bool) -> PerItemBoundaryRecord:
    record = PerItemBoundaryRecord(entry_id="sem:agree")
    for similarity in _HITS:
        record = record.add_sample(similarity + shift, correct)
    return record


@pytest.mark.parametrize("shift", [0.0, 1e-9, 1e-7, 1e-5])
def test_all_hits_fit_the_lower_boundary_whatever_the_float_noise(shift) -> None:
    record = _learned(shift, correct=True)
    assert record.gamma_hat == 100.0
    assert record.t_hat == pytest.approx(min(_HITS) + shift, abs=0.0)
    # The serve/explore probability no longer depends on the noise.
    assert record.compute_tau(1.0, delta=0.02, return_threshold=0.98) == 0.0
    assert record.compute_tau(0.98, delta=0.02, return_threshold=0.98) == 0.0


@pytest.mark.parametrize("shift", [0.0, 1e-9])
def test_all_misses_fit_the_upper_boundary(shift) -> None:
    record = _learned(shift, correct=False)
    assert record.gamma_hat == 100.0
    assert record.t_hat == pytest.approx(max(_HITS) + shift, abs=0.0)


def test_both_fit_paths_agree_on_the_corner(monkeypatch) -> None:
    from superlocalmemory.optimize.cache import boundary_store

    samples = [(s, 1) for s in _HITS]
    with_scipy = _fit_logistic_mle(samples, 0.95, 10.0)
    monkeypatch.setattr(boundary_store, "_SCIPY_AVAILABLE", False)
    assert _fit_logistic_mle(samples, 0.95, 10.0) == with_scipy == (0.95, 100.0)


def test_mixed_outcomes_still_go_to_the_optimiser() -> None:
    t, gamma = _fit_logistic_mle([(0.99, 1), (0.97, 1), (0.60, 0)], 0.95, 10.0)
    assert (t, gamma) != (0.6, 100.0)
    assert 0.60 <= t <= 0.99


def test_the_same_state_makes_the_same_choice_every_time() -> None:
    record = PerItemBoundaryRecord(
        entry_id="sem:mixed", t_hat=0.95, gamma_hat=20.0,
        samples=[(0.97, 1), (0.96, 1), (0.93, 0), (0.98, 1)],
    )
    tau = record.compute_tau(0.97, delta=0.02, return_threshold=0.98)
    assert 0.0 < tau < 1.0, "pick a state whose choice is genuinely uncertain"
    first = record.should_explore(0.97, delta=0.02, return_threshold=0.98)
    for _ in range(200):
        assert record.should_explore(0.97, delta=0.02, return_threshold=0.98) is first


def test_an_injected_draw_decides() -> None:
    record = PerItemBoundaryRecord(
        entry_id="sem:mixed", t_hat=0.95, gamma_hat=20.0,
        samples=[(0.97, 1), (0.96, 1), (0.93, 0), (0.98, 1)],
    )
    tau = record.compute_tau(0.97, delta=0.02, return_threshold=0.98)
    assert record.should_explore(0.97, delta=0.02, return_threshold=0.98, draw=0.0) is True
    assert record.should_explore(0.97, delta=0.02, return_threshold=0.98,
                                 draw=min(1.0, tau + 1e-6)) is False


def test_different_queries_and_states_draw_independently() -> None:
    record = PerItemBoundaryRecord(entry_id="sem:mixed", samples=[(0.97, 1), (0.93, 0)])
    draws = {record.explore_draw(q / 1000) for q in range(900, 1000)}
    assert len(draws) == 100
    later = record.add_sample(0.95, True)
    assert later.explore_draw(0.97) != record.explore_draw(0.97)
