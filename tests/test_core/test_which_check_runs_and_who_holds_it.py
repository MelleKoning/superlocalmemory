# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Which answer check runs, with which weights, and which engines hold it.

* "auto" turns the on-device check on only for an install that was set up and
  passed its check (the dashboard or setup path). A library that merely happens
  to be importable from SLM's own environment does not turn it on.
* Weights named by repo id are pinned to the measured revision; if that
  revision is not in the cache, the check stays off rather than load another.
* A malformed reorder count means no reordering, never "twenty memories".
* A switch that races a hot reconfigure never leaves the published engine
  holding a stopped check while the process runs none.
* The check is built without loading anything; loading starts on first use.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from superlocalmemory.core import engine_wiring, judge_selection
from superlocalmemory.core.laya_runtime import LAYA_MODEL_REVISION

# The selection fixture fakes platform, runtime, key store and both judges.
from tests.test_core.test_judge_selection import (  # noqa: F401
    _MANAGED_PY,
    _SNAPSHOT,
    _cfg,
    _jev_cfg,
    world,
)


def _snapshot(root: Path) -> Path:
    path = root / "hub" / "models--aac6fef--laya-mlx" / "snapshots" / LAYA_MODEL_REVISION
    path.mkdir(parents=True)
    return path


# ---------------------------------------------------------------------------
# M-2: what "auto" means
# ---------------------------------------------------------------------------

class TestAutoNeedsACheckedInstall:
    def test_an_importable_library_alone_does_not_turn_it_on(self, world, tmp_path) -> None:  # noqa: F811
        _snapshot(tmp_path)
        world.detect = "not_installed"
        world.has_laya = {sys.executable}
        cfg = _cfg("auto", sufficiency_hf_home=str(tmp_path))
        assert judge_selection.resolve_judge_mode(cfg) == "off"
        assert engine_wiring.init_sufficiency_judge(cfg) is None
        assert world.events == []

    def test_a_named_interpreter_that_was_never_checked_does_not_either(self, world) -> None:  # noqa: F811
        world.detect = "failed"
        world.has_laya = {"/opt/laya/bin/python"}
        cfg = _cfg("auto", sufficiency_python="/opt/laya/bin/python")
        assert engine_wiring.init_sufficiency_judge(cfg) is None

    def test_a_checked_install_does(self, world) -> None:  # noqa: F811
        world.detect = "ready"
        judge = engine_wiring.init_sufficiency_judge(_cfg("auto"))
        assert judge_selection.backend_of(judge) == "laya"
        assert judge.kwargs["python"] == _MANAGED_PY
        assert judge.kwargs["model"] == _SNAPSHOT

    def test_choosing_laya_by_name_still_finds_slms_own_library(self, world, tmp_path) -> None:  # noqa: F811
        snapshot = _snapshot(tmp_path)
        world.detect = "not_installed"
        world.has_laya = {sys.executable}
        judge = engine_wiring.init_sufficiency_judge(
            _cfg("laya", sufficiency_hf_home=str(tmp_path)))
        assert judge.kwargs["python"] == sys.executable
        assert judge.kwargs["model"] == str(snapshot)


class TestARepoIdIsPinnedToTheMeasuredRevision:
    def test_the_cached_measured_snapshot_is_what_loads(self, world, tmp_path) -> None:  # noqa: F811
        snapshot = _snapshot(tmp_path)
        world.has_laya = {"/opt/laya/bin/python"}
        judge = engine_wiring.init_sufficiency_judge(_cfg(
            "laya", sufficiency_python="/opt/laya/bin/python",
            sufficiency_hf_home=str(tmp_path)))
        assert judge.kwargs["model"] == str(snapshot)

    def test_without_the_measured_snapshot_it_stays_off(self, world, tmp_path, caplog) -> None:  # noqa: F811
        other = tmp_path / "hub" / "models--aac6fef--laya-mlx" / "snapshots" / ("0" * 40)
        other.mkdir(parents=True)
        world.has_laya = {"/opt/laya/bin/python"}
        with caplog.at_level(logging.INFO, logger=judge_selection.__name__):
            judge = engine_wiring.init_sufficiency_judge(_cfg(
                "laya", sufficiency_python="/opt/laya/bin/python",
                sufficiency_hf_home=str(tmp_path)))
        assert judge is None, "another revision was loaded under the measured name"
        assert any("measured" in r.getMessage() for r in caplog.records)

    def test_another_repo_by_name_is_refused(self, world, tmp_path) -> None:  # noqa: F811
        world.has_laya = {"/opt/laya/bin/python"}
        assert engine_wiring.init_sufficiency_judge(_cfg(
            "laya", sufficiency_python="/opt/laya/bin/python",
            sufficiency_model="someone/else", sufficiency_hf_home=str(tmp_path))) is None

    def test_a_named_folder_is_used_as_it_is(self, world, tmp_path) -> None:  # noqa: F811
        world.has_laya = {"/opt/laya/bin/python"}
        folder = tmp_path / "my-weights"
        folder.mkdir()
        judge = engine_wiring.init_sufficiency_judge(_cfg(
            "laya", sufficiency_python="/opt/laya/bin/python", sufficiency_model=str(folder)))
        assert judge.kwargs["model"] == str(folder)


# ---------------------------------------------------------------------------
# F10: a malformed reorder count means no reordering
# ---------------------------------------------------------------------------

class TestAMalformedReorderCountMeansOff:
    @pytest.mark.parametrize("value", ["lots", "20", 20.0, [20], None, True, 0, -5])
    def test_it_is_off_with_one_warning_naming_the_key(self, value, caplog) -> None:
        cfg = _cfg("jev", sufficiency_jev_rerank=True, sufficiency_jev_rerank_consent=True,
                   sufficiency_jev_rerank_k=value)
        judge_selection._rerank_k_warned = False
        with caplog.at_level(logging.WARNING, logger=judge_selection.__name__):
            assert judge_selection.jev_rerank_k(cfg) == 0
            assert judge_selection.jev_rerank_k(cfg) == 0
        warnings = [r.getMessage() for r in caplog.records]
        assert len(warnings) == 1, "one warning, not one per recall or status poll"
        assert "sufficiency_jev_rerank_k" in warnings[0]
        assert "lots" not in warnings[0]

    @pytest.mark.parametrize("value,expected", [(20, 20), (3, 5), (99, 30), (12, 12)])
    def test_a_whole_number_is_clamped_as_before(self, value, expected) -> None:
        cfg = _cfg("jev", sufficiency_jev_rerank=True, sufficiency_jev_rerank_consent=True,
                   sufficiency_jev_rerank_k=value)
        assert judge_selection.jev_rerank_k(cfg) == expected

    def test_the_hosted_judge_is_built_without_reordering(self, world) -> None:  # noqa: F811
        judge = engine_wiring.init_sufficiency_judge(_jev_cfg(
            sufficiency_jev_rerank=True, sufficiency_jev_rerank_consent=True,
            sufficiency_jev_rerank_k="lots"))
        assert judge.kwargs["rerank_k"] == 0


# ---------------------------------------------------------------------------
# M-11: a switch racing a hot reconfigure
# ---------------------------------------------------------------------------

class _Engine:
    """Stands in for a RetrievalEngine: weakly referenceable, holds a judge."""

    def __init__(self, judge=None) -> None:
        self._sufficiency_judge = judge


def _built(cfg) -> _Engine:
    engine = _Engine(engine_wiring.init_sufficiency_judge(cfg))
    judge_selection.register_engine(engine)
    return engine


def _live_judge():
    live = judge_selection._live
    return None if live is None else live.judge


class TestASwitchRacingAReconfigure:
    def test_the_published_engine_keeps_a_live_check(self, world) -> None:  # noqa: F811
        """The auditor's interleaving: E2 (reconfigure) shares E1's Laya, the
        switch targets the still-published E1, then E1 is closed."""
        e1 = _built(_cfg("laya"))
        e2 = _built(_cfg("laya"))
        assert e2._sufficiency_judge is e1._sufficiency_judge
        engine_wiring.attach_sufficiency_judge(e1, _jev_cfg())
        engine_wiring.release_sufficiency_judge(e1, final=True)
        judge = e2._sufficiency_judge
        assert judge is not None and judge.closed is False, \
            "the published engine holds a stopped check"
        assert judge is _live_judge()
        assert judge_selection.backend_of(judge) == "jev"
        assert world.max_live == 1

    def test_a_switch_between_building_and_publishing_is_caught(self, world) -> None:  # noqa: F811
        e1 = _built(_cfg("laya"))
        unregistered = _Engine(engine_wiring.init_sufficiency_judge(_cfg("laya")))
        engine_wiring.attach_sufficiency_judge(e1, _jev_cfg())
        judge_selection.register_engine(unregistered)
        engine_wiring.release_sufficiency_judge(e1, final=True)
        assert unregistered._sufficiency_judge is _live_judge()
        assert unregistered._sufficiency_judge.closed is False

    def test_switching_off_reaches_every_engine_holding_the_check(self, world) -> None:  # noqa: F811
        e1 = _built(_jev_cfg())
        e2 = _Engine(e1._sufficiency_judge)
        judge_selection.register_engine(e2)
        engine_wiring.attach_sufficiency_judge(e1, _cfg("off"))
        assert e1._sufficiency_judge is None and e2._sufficiency_judge is None, \
            "the check was switched off but a published engine kept it"
        assert world.live == 0 and _live_judge() is None


# ---------------------------------------------------------------------------
# L-5: nothing loads until the check is first used
# ---------------------------------------------------------------------------

def test_the_on_device_check_is_built_without_loading(world) -> None:  # noqa: F811
    judge = engine_wiring.init_sufficiency_judge(_cfg("laya"))
    assert judge.kwargs.get("start") is False


def test_a_switch_starts_loading_the_new_check_at_once(world, monkeypatch) -> None:  # noqa: F811
    """Built idle so a one-shot command never loads it — but a switch happens
    in the long-running daemon, where the person expects it to be ready soon."""
    started: list = []
    real = judge_selection._construct_laya

    def constructed(plan):
        judge = real(plan)
        judge.start_warmup = lambda: started.append(judge)
        return judge

    monkeypatch.setattr(judge_selection, "_construct_laya", constructed)
    engine = _Engine()
    assert engine_wiring.attach_sufficiency_judge(engine, _cfg("laya")) == "laya"
    assert started == [engine._sufficiency_judge]


def test_the_online_check_is_told_where_the_choice_is_stored(world) -> None:  # noqa: F811
    """So it can re-read consent before every request, in every process."""
    from superlocalmemory.infra.data_root import canonical_data_root

    judge = engine_wiring.init_sufficiency_judge(_jev_cfg())
    assert judge.kwargs["state_dir"] == canonical_data_root()
