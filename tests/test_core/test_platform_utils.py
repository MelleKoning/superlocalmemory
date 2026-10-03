# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for platform_utils — cross-platform subprocess and resource utilities.

V3.4.24: Validates popen_platform_kwargs, get_rss_mb, is_pid_alive,
kill_process, and start_parent_watchdog across platforms.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.core.platform_utils import (
    cancel_blocking_read,
    get_rss_mb,
    is_pid_alive,
    kill_process,
    popen_platform_kwargs,
    start_parent_watchdog,
)


class TestPopenPlatformKwargs:
    """Platform-appropriate subprocess kwargs."""

    def test_returns_dict(self) -> None:
        result = popen_platform_kwargs()
        assert isinstance(result, dict)

    def test_posix_has_start_new_session(self) -> None:
        with patch("superlocalmemory.core.platform_utils.sys") as mock_sys:
            mock_sys.platform = "darwin"
            result = popen_platform_kwargs()
            assert result == {"start_new_session": True}

    def test_linux_has_start_new_session(self) -> None:
        with patch("superlocalmemory.core.platform_utils.sys") as mock_sys:
            mock_sys.platform = "linux"
            result = popen_platform_kwargs()
            assert result == {"start_new_session": True}

    def test_win32_has_create_no_window(self) -> None:
        with patch("superlocalmemory.core.platform_utils.sys") as mock_sys:
            mock_sys.platform = "win32"
            result = popen_platform_kwargs()
            assert "creationflags" in result
            assert result["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    def test_kwargs_can_unpack_into_popen(self) -> None:
        kwargs = popen_platform_kwargs()
        assert len(kwargs) == 1


class TestGetRssMb:
    """RSS memory measurement."""

    def test_returns_float(self) -> None:
        result = get_rss_mb()
        assert isinstance(result, float)

    def test_returns_positive_on_current_platform(self) -> None:
        result = get_rss_mb()
        assert result > 0.0

    def test_returns_zero_on_resource_failure(self) -> None:
        with patch("superlocalmemory.core.platform_utils.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch.dict("sys.modules", {"resource": None}):
                result = get_rss_mb()
                assert result == 0.0


class TestIsPidAlive:
    """Process liveness check."""

    def test_current_process_is_alive(self) -> None:
        assert is_pid_alive(os.getpid()) is True

    def test_zero_pid_is_not_alive(self) -> None:
        assert is_pid_alive(0) is False

    def test_negative_pid_is_not_alive(self) -> None:
        assert is_pid_alive(-1) is False

    def test_nonexistent_pid_is_not_alive(self) -> None:
        assert is_pid_alive(99999999) is False

    def test_parent_process_is_alive(self) -> None:
        assert is_pid_alive(os.getppid()) is True

    @pytest.mark.skipif(sys.platform == "win32", reason="EPERM semantics are POSIX")
    def test_unsignalable_but_existing_pid_is_alive(self) -> None:
        # PID 1 (init/launchd) exists but a normal user cannot signal it ->
        # os.kill(1, 0) raises EPERM. is_pid_alive must treat EPERM as ALIVE,
        # not dead (regression: it used to catch all OSError and return False,
        # which also made the parent-pid test flaky under reparenting to init).
        assert is_pid_alive(1) is True


class TestKillProcess:
    """Process termination."""

    def test_zero_pid_returns_false(self) -> None:
        assert kill_process(0) is False

    def test_negative_pid_returns_false(self) -> None:
        assert kill_process(-1) is False

    def test_kill_real_subprocess(self) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pid = proc.pid
        assert is_pid_alive(pid) is True
        result = kill_process(pid)
        assert result is True
        proc.wait(timeout=5)


class TestCancelBlockingRead:
    """Unblock a thread wedged in a synchronous pipe read before closing it.

    Windows gotcha this guards: closing a pipe HANDLE while another thread
    has a synchronous ReadFile in flight on that same handle does not
    cancel the read — the close() call itself blocks until the read
    completes, which (no writer, e.g. a dead/silent embedding worker) is
    forever. CancelSynchronousIo() aborts the specific thread's pending
    I/O so the blocked read returns immediately and the handle can then be
    closed safely. POSIX needs none of this: closing the fd already wakes
    a concurrent blocked read.
    """

    def test_posix_is_a_noop(self) -> None:
        with patch("superlocalmemory.core.platform_utils.sys") as mock_sys, \
             patch("superlocalmemory.core.platform_utils.ctypes") as mock_ctypes:
            mock_sys.platform = "darwin"
            thread = threading.Thread(target=lambda: None)
            cancel_blocking_read(thread)
            mock_ctypes.windll.kernel32.OpenThread.assert_not_called()

    def test_thread_without_native_id_is_a_noop(self) -> None:
        with patch("superlocalmemory.core.platform_utils.sys") as mock_sys, \
             patch("superlocalmemory.core.platform_utils.ctypes") as mock_ctypes:
            mock_sys.platform = "win32"
            thread = MagicMock()
            thread.native_id = None
            cancel_blocking_read(thread)
            mock_ctypes.windll.kernel32.OpenThread.assert_not_called()

    def test_windows_cancels_then_closes_the_thread_handle(self) -> None:
        with patch("superlocalmemory.core.platform_utils.sys") as mock_sys, \
             patch("superlocalmemory.core.platform_utils.ctypes") as mock_ctypes:
            mock_sys.platform = "win32"
            kernel32 = mock_ctypes.windll.kernel32
            kernel32.OpenThread.return_value = 4242
            thread = MagicMock()
            thread.native_id = 9999
            cancel_blocking_read(thread)
            kernel32.OpenThread.assert_called_once_with(1, False, 9999)
            kernel32.CancelSynchronousIo.assert_called_once_with(4242)
            kernel32.CloseHandle.assert_called_once_with(4242)

    def test_windows_openthread_failure_skips_cancel(self) -> None:
        with patch("superlocalmemory.core.platform_utils.sys") as mock_sys, \
             patch("superlocalmemory.core.platform_utils.ctypes") as mock_ctypes:
            mock_sys.platform = "win32"
            kernel32 = mock_ctypes.windll.kernel32
            kernel32.OpenThread.return_value = 0
            thread = MagicMock()
            thread.native_id = 9999
            cancel_blocking_read(thread)
            kernel32.CancelSynchronousIo.assert_not_called()
            kernel32.CloseHandle.assert_not_called()

    def test_windows_never_raises_even_if_the_win32_call_fails(self) -> None:
        with patch("superlocalmemory.core.platform_utils.sys") as mock_sys, \
             patch("superlocalmemory.core.platform_utils.ctypes") as mock_ctypes:
            mock_sys.platform = "win32"
            kernel32 = mock_ctypes.windll.kernel32
            kernel32.OpenThread.side_effect = OSError("boom")
            thread = MagicMock()
            thread.native_id = 9999
            cancel_blocking_read(thread)  # must not raise


class TestStartParentWatchdog:
    """Parent watchdog thread."""

    def test_does_not_crash(self) -> None:
        stop = threading.Event()
        # Patch os.getppid so the function always enters the "start thread"
        # branch regardless of real process parentage (e.g. CI reparenting,
        # PID-namespace containers, or suites that affect ppid indirectly).
        with patch("os.getppid", return_value=os.getpid()):
            thread = start_parent_watchdog(stop_event=stop)
        assert thread is not None
        stop.set()
        thread.join(timeout=1)
        assert not thread.is_alive()

    def test_creates_daemon_thread(self) -> None:
        stop = threading.Event()
        with patch("os.getppid", return_value=os.getpid()):
            thread = start_parent_watchdog(stop_event=stop)
        assert thread is not None
        assert thread.daemon
        assert thread.name == "parent-watchdog"
        stop.set()
        thread.join(timeout=1)
        assert not thread.is_alive()
