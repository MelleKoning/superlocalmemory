# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Talking to, and stopping, one model worker process over its pipes."""

from __future__ import annotations

import sys
import threading
from typing import Any


def readline_with_timeout(stream: Any, timeout_seconds: float) -> str:
    """Read a line from stream with timeout. Returns '' on timeout.

    Prefer a deadline-driven selector poll of the stream's file descriptor
    (POSIX pipes). That path never spawns a helper thread, so a hung
    worker cannot leak reader threads or pin the pipe FD across timeouts.
    A thread fallback remains only for streams without a usable fileno
    (unit-test mocks) and for Windows, where selectors cannot wait on
    pipes.
    """
    import selectors

    timeout_seconds = max(0.0, float(timeout_seconds))
    fd: int | None
    try:
        raw_fd = stream.fileno()
        fd = raw_fd if isinstance(raw_fd, int) else None
    except (AttributeError, OSError, ValueError, TypeError):
        fd = None

    # Windows select()/selectors only accept sockets, not subprocess pipes.
    if fd is not None and sys.platform != "win32":
        try:
            with selectors.DefaultSelector() as sel:
                sel.register(fd, selectors.EVENT_READ)
                events = sel.select(timeout=timeout_seconds)
            if not events:
                return ""
            line = stream.readline()
            return line if line else ""
        except (OSError, ValueError):
            return ""

    result_container: list[str] = []
    error_container: list[Exception] = []

    def _read() -> None:
        try:
            result_container.append(stream.readline())
        except Exception as exc:
            error_container.append(exc)

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    reader.join(timeout=timeout_seconds)

    if reader.is_alive():
        return ""
    if error_container:
        raise error_container[0]
    return result_container[0] if result_container else ""


def _reap_later(proc: Any) -> None:
    """Wait for a killed worker on a daemon thread, so it never stays a zombie.

    A worker stuck in uninterruptible I/O dies only when that I/O ends; until
    then nobody else would collect its exit status. Never raises.
    """
    def _wait() -> None:
        try:
            proc.wait()
        except Exception:
            pass

    try:
        threading.Thread(target=_wait, name="slm-worker-reaper", daemon=True).start()
    except Exception:
        pass


def stop_process(proc: Any, timeout: float = 3.0) -> None:
    """Ask one worker process to quit, kill it if it will not, close its pipes."""
    try:
        proc.stdin.write('{"cmd":"quit"}\n')
        proc.stdin.flush()
        proc.wait(timeout=max(0.0, timeout))
    except Exception:
        try:
            returncode = proc.poll()
        except Exception:
            returncode = None
        if returncode is None or not isinstance(returncode, int):
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=max(0.0, timeout))
            except Exception:
                _reap_later(proc)
    finally:
        # Explicit close prevents TextIOWrapper from flushing a dead
        # child's stdin later from an unraisable object finalizer.
        for stream_name in ("stdin", "stdout", "stderr"):
            stream = getattr(proc, stream_name, None)
            if stream is not None:
                try:
                    stream.close()
                except (BrokenPipeError, OSError, ValueError):
                    pass
