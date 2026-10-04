# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The two slow Laya jobs the dashboard starts and then polls.

Setting up the on-device model (download + check) and adopting an existing
install (load the model + check) both take from seconds to minutes. Run inside
a request, either one held the daemon's event loop for that long, freezing
recall, remember, MCP and the dashboard together. Here each runs on its own
thread, at most one of each at a time, and reports progress as a status
snapshot the dashboard reads.

``on_done(status)`` runs on the job's thread once the result is recorded —
the daemon uses it to save the paths of a finished install and switch the
check on, so that happens even when nobody is watching the page, and a status
read never has to. ``before_verify()`` runs right before the model is loaded
for its check, so the daemon can stop its own running copy first: two models
must never be resident together.

Status types come from ``core.laya_runtime``, imported when a job runs, so
this module never imports it at load time (laya_runtime imports this one).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

DoneFn = Callable[[Any], None]
HookFn = Callable[[], None]


def _runtime():
    from superlocalmemory.core import laya_runtime

    return laya_runtime


class _LayaJob:
    """One background job of one kind; a second start while it runs is refused."""

    _instance: "_LayaJob | None" = None
    _instance_lock = threading.Lock()
    _thread_name = "laya-job"

    def __init__(self) -> None:
        lr = _runtime()
        self._lock = threading.Lock()
        self._running = False
        self._status = lr.LayaRuntimeStatus(state=lr.STATE_NOT_INSTALLED)
        self._thread: threading.Thread | None = None

    @classmethod
    def instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def status(self):
        with self._lock:
            return self._status

    def forget(self) -> None:
        """Drop the last finished result (after Remove), so it is not
        reported against an install that no longer exists."""
        lr = _runtime()
        with self._lock:
            if not self._running:
                self._status = lr.LayaRuntimeStatus(state=lr.STATE_NOT_INSTALLED)

    def _launch(self, first_step: str, work: Callable[[Callable[[float, str], None]], Any],
                on_done: DoneFn | None) -> bool:
        lr = _runtime()
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._status = lr.LayaRuntimeStatus(state=lr.STATE_INSTALLING, progress=0.0,
                                                step=first_step)

        def _progress(fraction: float, step: str) -> None:
            with self._lock:
                self._status = lr.LayaRuntimeStatus(
                    state=lr.STATE_INSTALLING, progress=fraction, step=step)

        def _runner() -> None:
            try:
                result = work(_progress)
            except Exception as exc:  # noqa: BLE001 — the job must always finish.
                logger.exception("Laya background job crashed")
                result = lr.LayaRuntimeStatus(state=lr.STATE_FAILED,
                                              error=f"Unexpected error: {exc}")
            with self._lock:
                self._status = result
                self._running = False
            if on_done is not None:
                try:
                    on_done(result)
                except Exception:  # noqa: BLE001 — logged; the result stands
                    logger.exception("Laya job: applying the finished result failed")

        self._thread = threading.Thread(target=_runner, daemon=True, name=self._thread_name)
        self._thread.start()
        return True


class LayaInstallJob(_LayaJob):
    """The one background install the dashboard starts and polls."""

    _instance: "LayaInstallJob | None" = None
    _instance_lock = threading.Lock()
    _thread_name = "laya-install"

    def start(self, *, on_done: DoneFn | None = None,
              before_verify: HookFn | None = None) -> bool:
        """False when an install is already running."""
        extra = {"before_verify": before_verify} if before_verify is not None else {}

        def _work(progress):
            return _runtime().install(progress=progress, **extra)

        return self._launch("Starting…", _work, on_done)

    def cancel(self) -> bool:
        """Stop the running setup at its next check (within about a second).
        False when nothing is running. What was downloaded is kept."""
        from superlocalmemory.core import laya_process

        if not self.running:
            return False
        laya_process.CANCEL.set()
        return True


class LayaAdoptJob(_LayaJob):
    """Checks an install someone already has ("Use an existing install")."""

    _instance: "LayaAdoptJob | None" = None
    _instance_lock = threading.Lock()
    _thread_name = "laya-adopt"

    def start(self, python: str, hf_home: str, model_path: str = "", *,
              on_done: DoneFn | None = None,
              before_verify: HookFn | None = None) -> bool:
        """False when a check is already running."""

        def _work(progress):
            progress(0.5, "Checking that install works")
            if before_verify is not None:
                before_verify()
            return _runtime().adopt(python, hf_home, model_path)

        return self._launch("Checking that install works", _work, on_done)


class LayaTestJob(_LayaJob):
    """"Test" for an install that is set up: load it and ask the check question.

    The same check the install passed when it was set up, run again now, so a
    person can see it still works — and how long it took.
    """

    _instance: "LayaTestJob | None" = None
    _instance_lock = threading.Lock()
    _thread_name = "laya-test"

    def start(self, python: str, hf_home: str, model_path: str, *,
              on_done: DoneFn | None = None,
              before_verify: HookFn | None = None) -> bool:
        """False when a test is already running."""

        def _work(progress):
            lr = _runtime()
            progress(0.5, "Testing the on-device check")
            if before_verify is not None:
                before_verify()
            started = time.monotonic()
            ok, reason = lr.verify(python, hf_home, model_path)
            seconds = round(time.monotonic() - started, 1)
            return lr.LayaRuntimeStatus(
                state=lr.STATE_READY if ok else lr.STATE_FAILED, python=python,
                hf_home=hf_home, model_path=model_path, progress=1.0,
                step=f"{seconds}", error="" if ok else reason)

        return self._launch("Testing the on-device check", _work, on_done)


__all__ = ["LayaAdoptJob", "LayaInstallJob", "LayaTestJob"]
