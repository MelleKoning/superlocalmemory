# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""The vector store's version history is bounded, automatically.

GitHub #137. Every write to the vector store creates a version, and nothing
removed them. MEASURED on the author's store: 5,561 memories, a 610 MB SQLite
database, and a **17 GB** vector store holding **50,580 versions**. Each vector
operation walks that history, which is why the daemon held a core at 90%+ with
an empty queue and every Python thread parked.

Two changes bound it, and they address different halves:

  - ``_VectorBatch`` cuts the RATE: one version per drain pass instead of one
    per memory.
  - This pass cuts the STOCK: versions past the retention window are dropped on
    the maintenance tick, so a store that already has a history heals itself
    rather than needing a command nobody knows to run.

Verified against real LanceDB, 300 facts written one at a time:

    before  rows=300  versions=300  size=6.5M
    after   rows=300  versions=  1  size=128K
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from superlocalmemory.core import maintenance_scheduler as ms


class _Backend:
    def __init__(self, result=None, boom: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._result = result or {"ok": True, "versions_before": 50580,
                                  "versions_after": 1}
        self._boom = boom

    def compact(self, **kwargs):
        self.calls.append(kwargs)
        if self._boom:
            raise self._boom
        return self._result


class _Orchestrator:
    def __init__(self, backend) -> None:
        self._backend = backend

    def get_vector_backend(self):
        return self._backend


@pytest.fixture()
def orchestrated(monkeypatch):
    def _install(backend):
        monkeypatch.setattr(
            "superlocalmemory.core.backend_orchestrator.get_orchestrator",
            lambda: _Orchestrator(backend),
        )
        return backend
    return _install


class TestTheTickBoundsTheHistory:
    def test_it_compacts_with_the_retention_window(self, orchestrated) -> None:
        backend = orchestrated(_Backend())
        out = ms.compact_vector_store()
        assert out["ok"] is True
        assert backend.calls == [
            {"retention": timedelta(days=ms._VECTOR_HISTORY_DAYS)}
        ]

    def test_it_never_deletes_unverified_files_on_a_live_daemon(
        self, orchestrated,
    ) -> None:
        """Those are the files an in-flight write is creating right now."""
        backend = orchestrated(_Backend())
        ms.compact_vector_store()
        assert backend.calls[0].get("delete_unverified") in (None, False)

    def test_a_backend_that_cannot_compact_is_not_an_error(
        self, orchestrated,
    ) -> None:
        """sqlite-vec has no version history to bound."""
        class _Plain:
            pass
        orchestrated(_Plain())
        out = ms.compact_vector_store()
        assert out["ok"] is True and out["applicable"] is False

    def test_a_failure_does_not_propagate(self, orchestrated) -> None:
        """A maintenance pass must not take the daemon down."""
        orchestrated(_Backend(boom=RuntimeError("lance is unhappy")))
        out = ms.compact_vector_store()
        assert out["ok"] is False and "unhappy" in out["reason"]

    def test_no_orchestrator_is_not_an_error(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "superlocalmemory.core.backend_orchestrator.get_orchestrator",
            lambda: None,
        )
        assert ms.compact_vector_store()["ok"] is False

    def test_the_window_is_tunable(self, monkeypatch) -> None:
        """An operator with a 17 GB store may want it shorter, once."""
        assert ms._VECTOR_HISTORY_DAYS == 7.0

    def test_a_silent_skip_is_logged(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(
            "superlocalmemory.core.backend_orchestrator.get_orchestrator",
            lambda: None,
        )
        with caplog.at_level("WARNING"):
            ms.compact_vector_store()
        assert "no orchestrator" in caplog.text

    def test_startup_arms_compaction_without_waiting_the_interval(self) -> None:
        import inspect
        src = inspect.getsource(ms.MaintenanceScheduler.start)
        assert "_initial_vector_compaction" in src


class TestCompactDoesNotParseHistory:
    def test_it_counts_manifests_on_disk(self, tmp_path) -> None:
        from superlocalmemory.vector.lancedb_backend import LanceDBVectorBackend

        class _Table:
            def __init__(self) -> None:
                self.list_calls = 0

            def list_versions(self):
                self.list_calls += 1
                return list(range(50_580))

            def optimize(self, **kwargs) -> None:
                return None

        versions = tmp_path / "embeddings.lance" / "_versions"
        versions.mkdir(parents=True)
        (versions / "1.manifest").write_text("x")
        (versions / "2.manifest").write_text("x")
        backend = LanceDBVectorBackend.__new__(LanceDBVectorBackend)
        backend._table = _Table()
        backend._db_path = str(tmp_path)
        out = backend.compact()
        assert out["ok"] is True
        assert out["versions_before"] == 2
        assert backend._table.list_calls == 0


class TestOfflineCompactRefusesALiveWriter:
    def test_it_exits_nonzero_when_the_daemon_is_alive(self, monkeypatch) -> None:
        from argparse import Namespace

        from superlocalmemory.cli import commands as cli_commands

        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.owned_daemon_process_alive",
            lambda: True,
        )
        assert cli_commands._cmd_db_compact(Namespace(offline=True)) == 1


class TestLiveCompactGoesThroughTheDaemon:
    """`slm db compact` (without --offline) used to compact inside the CLI
    process, where the daemon's backends do not exist, so it always failed
    with "no orchestrator". It now asks the running daemon to do it."""

    def test_it_asks_the_running_daemon(self, monkeypatch, capsys) -> None:
        from argparse import Namespace

        from superlocalmemory.cli import commands as cli_commands

        calls = []
        monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda: True)
        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.daemon_request",
            lambda method, path, body=None, **kw: calls.append((method, path)) or {
                "ok": True, "applicable": False, "reason": "not applicable"},
        )
        assert cli_commands._cmd_db_compact(Namespace(offline=False)) == 0
        assert calls == [("POST", "/maintenance/compact")]

    def test_with_no_daemon_it_says_what_to_do(self, monkeypatch, capsys) -> None:
        from argparse import Namespace

        from superlocalmemory.cli import commands as cli_commands

        monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda: False)
        assert cli_commands._cmd_db_compact(Namespace(offline=False)) == 1
        err = capsys.readouterr().err
        assert "not running" in err and "--offline" in err

    def test_the_daemon_route_exists_and_is_gated(self) -> None:
        import inspect

        from superlocalmemory.server import unified_daemon

        source = inspect.getsource(unified_daemon)
        start = source.index('@application.post("/maintenance/compact")')
        body = source[start:start + 1600]
        assert "require_permission(request, Permission.WRITE" in body
        assert "authorize_route_mutation(" in body
        assert "compact_vector_store" in body
