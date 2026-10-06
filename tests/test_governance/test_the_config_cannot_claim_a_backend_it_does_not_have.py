# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What the settings say is active must match what is on disk.

THE STATE THIS REPAIRS

On a real store the settings read ``graph_backend='cozo'``,
``vector_backend='lancedb'``, ``scale_engine_state='verified'`` -- and neither
the ``cozo/`` nor the ``lance/`` directory existed, with no promotion journal to
explain it. Something had written the selection a completed promotion writes,
without a promotion having completed.

Nothing corrected it, and nothing would have. The startup recovery acts only
when a promotion journal exists, so with no journal it returns immediately and
the claim survives every restart. The dashboard then reports the configured
backend while retrieval reads SQLite — a disagreement between two surfaces that
a person notices last and trusts first.

The repair is narrow on purpose: remove the claim, disable nothing. ``auto``
still detects and initialises both projections when their libraries are present.
"""

from __future__ import annotations

import pytest


class _Config:
    """The three settings this touches, plus a save that records being called."""

    def __init__(self, graph="auto", vector="auto", state="", base_dir=None):
        self.graph_backend = graph
        self.vector_backend = vector
        self.scale_engine_state = state
        self.base_dir = base_dir
        self.saves = 0
        self.mode = type("M", (), {"value": "b"})()

    def save(self):
        self.saves += 1


def _reconcile(config, data_dir):
    """Run only the reconciliation, with the manager pointed at ``data_dir``.

    ``base_dir`` is set here because the scale manager reads that attribute
    directly rather than resolving the data root itself — a stub without it
    makes the reconciliation skip, and every assertion below would then pass or
    fail for a reason that has nothing to do with the code under test.
    """
    from superlocalmemory.core.backend_orchestrator import BackendOrchestrator

    config.base_dir = str(data_dir)
    orchestrator = object.__new__(BackendOrchestrator)
    orchestrator._config = config
    BackendOrchestrator._reconcile_backend_selection(orchestrator)


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    return tmp_path


class TestAClaimWithNothingBehindItIsRemoved:
    def test_a_named_graph_backend_with_no_directory_falls_back_to_auto(
        self, data_dir,
    ) -> None:
        config = _Config(graph="cozo", vector="lancedb", state="verified")

        _reconcile(config, data_dir)

        assert config.graph_backend == "auto"
        assert config.vector_backend == "auto"
        assert config.saves == 1

    def test_verified_is_left_alone(self, data_dir) -> None:
        """It means "parity checked, not yet promoted", which is a real state."""
        config = _Config(graph="cozo", vector="lancedb", state="verified")

        _reconcile(config, data_dir)

        assert config.scale_engine_state == "verified"

    def test_promoted_with_nothing_on_disk_is_reset(self, data_dir) -> None:
        """That one asserts a directory swap that plainly did not happen."""
        config = _Config(graph="cozo", vector="lancedb", state="promoted")

        _reconcile(config, data_dir)

        assert config.scale_engine_state == "local_core"


class TestARealPromotionIsNotTouched:
    def test_a_backend_whose_directory_exists_is_left_as_it_is(
        self, data_dir,
    ) -> None:
        (data_dir / "cozo").mkdir()
        (data_dir / "lance").mkdir()
        config = _Config(graph="cozo", vector="lancedb", state="promoted")

        _reconcile(config, data_dir)

        assert config.graph_backend == "cozo"
        assert config.vector_backend == "lancedb"
        assert config.scale_engine_state == "promoted"
        assert config.saves == 0, "an unchanged config must not be rewritten"

    def test_one_backend_present_and_one_absent_corrects_only_the_absent_one(
        self, data_dir,
    ) -> None:
        """The two projections are promoted together but can be installed apart."""
        (data_dir / "cozo").mkdir()
        config = _Config(graph="cozo", vector="lancedb", state="promoted")

        _reconcile(config, data_dir)

        assert config.graph_backend == "cozo"
        assert config.vector_backend == "auto"
        assert config.scale_engine_state == "promoted", (
            "one directory present means a swap did happen; do not undo the state"
        )

    def test_an_open_promotion_journal_defers_to_the_recovery_path(
        self, data_dir,
    ) -> None:
        """Mid-promotion is the recovery path's business, not this one's."""
        from superlocalmemory.core.scale_engine import ScaleEngineManager

        journal = data_dir / ScaleEngineManager.PROMOTION_JOURNAL
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text('{"state": "intent"}', encoding="utf-8")
        config = _Config(graph="cozo", vector="lancedb", state="promoted")

        _reconcile(config, data_dir)

        assert config.graph_backend == "cozo"
        assert config.saves == 0


class TestItIsSafeToRunEveryStartup:
    def test_an_already_consistent_config_is_not_rewritten(self, data_dir) -> None:
        config = _Config(graph="auto", vector="auto", state="local_core")

        _reconcile(config, data_dir)

        assert config.saves == 0

    def test_running_it_twice_changes_nothing_the_second_time(
        self, data_dir,
    ) -> None:
        config = _Config(graph="cozo", vector="lancedb", state="promoted")

        _reconcile(config, data_dir)
        first = config.saves
        _reconcile(config, data_dir)

        assert config.saves == first

    def test_a_config_that_cannot_be_saved_does_not_stop_startup(
        self, data_dir,
    ) -> None:
        """A projection is derived data; the store must still serve SQLite."""
        class _Unsaveable(_Config):
            def save(self):
                raise OSError("read-only file system")

        config = _Unsaveable(graph="cozo", vector="lancedb", state="promoted")

        _reconcile(config, data_dir)  # must not raise

        assert config.graph_backend == "auto"


class TestItActuallyRunsAtStartup:
    """A repair nothing calls is the same as no repair."""

    @pytest.mark.parametrize(
        "state", ["local_core", "prepared", "verified", "promoted"],
    )
    def test_bringing_the_backends_up_reconciles_whatever_state_it_is_in(
        self, data_dir, state,
    ) -> None:
        """It must run before anything reads the selection it is correcting,
        and it must run for every store — not only one already promoted.

        The stores that need it are precisely the ones that do NOT reach the
        promoted path. The real case is a store whose settings name a graph and
        a vector backend, whose state is ``verified``, and where neither
        directory exists: startup returned early, the reconciliation never ran,
        and every restart preserved the claim.

        This was asserted by comparing where two strings appeared in the source
        of the startup method, which is true whether or not the method returns
        between them.
        """
        from superlocalmemory.core.backend_orchestrator import BackendOrchestrator

        calls: list[str] = []

        class _Orchestrator:
            _config = _StartupConfig(
                graph_backend="cozo",
                vector_backend="lancedb",
                scale_engine_state=state,
            )
            _db = None
            _drain = _NoOpDrain()

            def _recover_interrupted_scale_promotion(self):
                calls.append("recover")

            def _reconcile_backend_selection(self):
                calls.append("reconcile")

            def _maybe_schedule_auto_promote(self):
                calls.append("auto_promote")

            def _detect_cozo(self):
                calls.append("detect_cozo")
                return False

            def _detect_lancedb(self):
                calls.append("detect_lance")
                return False

            def __getattr__(self, name):
                def _noop(*args, **kwargs):
                    calls.append(name)
                return _noop

        try:
            BackendOrchestrator.on_daemon_start(_Orchestrator())
        except AttributeError:
            # The promoted path goes on to bring real backends up, which this
            # stand-in cannot provide. What is being checked happens before
            # that, and `calls` records whether it happened.
            pass

        assert "reconcile" in calls, (
            f"startup with state={state!r} never reconciled what the settings "
            f"claim against what is on disk"
        )
        if "detect_cozo" in calls:
            assert calls.index("reconcile") < calls.index("detect_cozo"), (
                "the selection was read before it was corrected"
            )


class _StartupConfig:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __getattr__(self, name):
        return None


class _NoOpDrain:
    running = False

    def start(self):
        return None

    def stop(self):
        return None
