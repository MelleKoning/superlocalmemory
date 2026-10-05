# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Cross-platform utilities for subprocess management and resource monitoring.

V3.4.24: Consolidates Windows/POSIX branching from 10+ files into one module.
Replaces the Unix-only ``resource`` module with ``psutil`` on Windows.
Inspired by community PR #14 (GuillaumeG / Tyrin451).
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading

# Minimal Win32 access right for OpenThread — enough for CancelSynchronousIo,
# nothing more (see cancel_blocking_read).
_WIN32_THREAD_TERMINATE = 0x0001


def popen_platform_kwargs() -> dict:
    """Platform-appropriate kwargs for subprocess.Popen.

    POSIX: ``start_new_session=True`` — prevents terminal signals bleeding.
    Windows: ``CREATE_NO_WINDOW`` — prevents console window popup.
    """
    if sys.platform == "win32":
        # CREATE_NO_WINDOW = 0x08000000 — only defined on Windows.
        flag = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        return {"creationflags": flag}
    return {"start_new_session": True}


def get_rss_mb() -> float:
    """Current process RSS in megabytes.

    POSIX: ``resource.getrusage`` (stdlib). Windows: ``psutil``.
    Returns 0.0 if measurement is unavailable.
    """
    if sys.platform != "win32":
        try:
            import resource
            ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return ru_maxrss / 1024 / 1024  # macOS: bytes
            return ru_maxrss / 1024  # Linux: kilobytes
        except Exception:
            return 0.0
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def is_pid_alive(pid: int) -> bool:
    """Check whether a process with *pid* is alive. The one place SLM asks.

    POSIX: ``os.kill(pid, 0)`` — signal 0 checks existence.
    Windows: ``psutil.pid_exists()`` (a declared dependency), never
    ``os.kill``: there signal 0 is ``CTRL_C_EVENT``, so ``os.kill(pid, 0)``
    either fails with WinError 87 or sends Ctrl+C to the process group.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import psutil
        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False  # ESRCH — no such process
    except PermissionError:
        return True   # EPERM — process EXISTS, we just can't signal it
    except OSError:
        return False


def current_account() -> str:
    """The account this process runs as, qualified where the OS qualifies it.

    POSIX: the login name. Windows: ``DOMAIN\\user``, the form psutil gives
    for any process, so the two compare directly. Read from the process's
    own token, not from ``USERNAME``: a process started with a stripped
    environment (a service, an editor's MCP host) has no ``USERNAME``, and
    ``getpass.getuser()`` then raises OSError.
    """
    if sys.platform == "win32":
        import psutil
        return str(psutil.Process(os.getpid()).username())
    import getpass
    return getpass.getuser()


def current_user_name() -> str:
    """The bare user name (no domain) of :func:`current_account`."""
    return current_account().rsplit("\\", 1)[-1]


def kill_process(pid: int) -> bool:
    """Send SIGTERM (POSIX) or taskkill /F /T (Windows).

    Returns True if the signal was sent successfully.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            subprocess.call(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False
    try:
        import signal
        os.kill(pid, signal.SIGTERM)
        return True
    except OSError:
        return False


def cancel_blocking_read(thread: threading.Thread) -> None:
    """Unblock a thread wedged in a synchronous read, before closing its fd.

    POSIX: no-op. Closing the fd/stream from another thread already wakes a
    concurrent blocked ``read()`` there (how the embedding worker's readline
    timeout unblocks its reader thread on macOS/Linux).

    Windows: closing a pipe HANDLE while another thread has a synchronous
    ``ReadFile`` in flight on that same handle does **not** cancel the read.
    The close() call itself blocks until that read completes — forever, for
    a worker pipe with no writer (confirmed via faulthandler stack dump on
    GH Actions windows-latest: the closing thread wedged inside ``close()``,
    the reader thread wedged inside ``readline()``, same handle, deadlock).
    ``CancelSynchronousIo`` is the documented Win32 call for exactly this: it
    aborts the pending I/O issued by the given thread, so the blocked read
    returns immediately and it is then safe to close the handle.
    """
    if sys.platform != "win32":
        return
    native_id = getattr(thread, "native_id", None)
    if not native_id:
        return
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenThread(_WIN32_THREAD_TERMINATE, False, native_id)
        if not handle:
            return
        try:
            kernel32.CancelSynchronousIo(handle)
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        pass


def start_parent_watchdog(
    *, stop_event: threading.Event | None = None,
) -> threading.Thread | None:
    """Self-terminate when the parent process dies.

    Prevents orphaned workers (500+ MB each) after parent crash/kill.
    V3.3.7 origin: 33 GB consumed by orphaned workers.
    V3.4.24: Consolidated from 3 separate worker files.
    """
    try:
        parent_pid = os.getppid()
    except AttributeError:
        return
    if parent_pid <= 1:
        return

    stop = stop_event or threading.Event()

    def _watch() -> None:
        while not stop.wait(5):
            if not is_pid_alive(parent_pid):
                os._exit(0)

    t = threading.Thread(target=_watch, daemon=True, name="parent-watchdog")
    t.start()
    return t
