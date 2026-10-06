# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""One enforceable deadline for a native host hook.

A host (Codex, Claude Code) kills a hook that overruns its ``timeout`` and
throws its output away. Before 4.1.22 ``slm hook codex-start`` blocked in
``readline()`` on an ``slm mcp`` child, only checked its own deadline between
lines, and then ran a second child for up to 10 s: the internal deadline was
not enforceable, so the host's 15 s limit could fire first and the session
started with no SLM context and no explanation (handoff §13).

The pieces here keep a hook inside its budget on every path:

* :class:`HookDeadline` — one end-to-end clock every step reads from;
* :func:`run_bounded` — a child process that is killed at the deadline, with
  its own process group on POSIX so its helpers go with it (never a
  name-pattern kill);
* :class:`McpStdio` — JSON-RPC over an ``slm mcp`` child read by a thread, so
  the hook waits on a queue with a timeout instead of blocking in ``readline``;
* :class:`Watchdog` — if anything still overruns, it writes the degraded
  output exactly once, stops the children it owns, and exits the process.

Windows: pipes are read by threads (no ``select`` on pipes) and a timed-out
child is ended with ``Popen.kill``; grandchildren are not reached there,
because the shared daemon a child may have spawned is one of them.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

_KILL_WAIT_S = 1.0


class HookDeadline:
    """A monotonic end time shared by every step of one hook run."""

    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)
        self._end = time.monotonic() + self.seconds

    def remaining(self, reserve: float = 0.0) -> float:
        return max(0.0, self._end - time.monotonic() - reserve)

    def expired(self) -> bool:
        return self.remaining() <= 0.0


def _popen_group_kwargs() -> dict[str, Any]:
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def stop_child(proc: subprocess.Popen) -> None:
    """End a child this hook started, bounded; its process group on POSIX."""
    if proc.poll() is not None:
        return
    try:
        if sys.platform != "win32":
            os.killpg(proc.pid, signal.SIGKILL)  # the group this hook created
        else:
            proc.kill()
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=_KILL_WAIT_S)
    except subprocess.TimeoutExpired:
        pass


class ChildRegistry:
    """Children this hook owns, so a watchdog can stop exactly those."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._children: list[subprocess.Popen] = []

    def add(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._children.append(proc)

    def discard(self, proc: subprocess.Popen) -> None:
        with self._lock:
            if proc in self._children:
                self._children.remove(proc)

    def stop_all(self) -> None:
        with self._lock:
            children = list(self._children)
        for proc in children:
            stop_child(proc)


@dataclass(frozen=True)
class BoundedResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool


def run_bounded(
    cmd: list[str],
    *,
    deadline: HookDeadline,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
    registry: ChildRegistry | None = None,
    reserve: float = 0.3,
) -> BoundedResult:
    """Run ``cmd`` and return by the deadline, killing it if it overruns."""
    limit = deadline.remaining(reserve)
    if limit <= 0:
        return BoundedResult(None, "", "", True)
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", env=env,
            **_popen_group_kwargs(),
        )
    except OSError as exc:
        return BoundedResult(None, "", str(exc), False)
    if registry is not None:
        registry.add(proc)
    try:
        out, err = proc.communicate(input=input_text, timeout=limit)
        return BoundedResult(proc.returncode, out or "", err or "", False)
    except subprocess.TimeoutExpired:
        stop_child(proc)
        try:
            out, err = proc.communicate(timeout=_KILL_WAIT_S)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            out, err = "", ""
        return BoundedResult(None, out or "", err or "", True)
    except Exception as exc:  # noqa: BLE001 - a hook step never raises
        stop_child(proc)
        return BoundedResult(None, "", str(exc), False)
    finally:
        if registry is not None:
            registry.discard(proc)


class McpStdio:
    """A deadline-bound JSON-RPC client for one ``slm mcp`` child."""

    def __init__(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        registry: ChildRegistry | None = None,
    ) -> None:
        self._registry = registry
        self._lines: "queue.Queue[str | None]" = queue.Queue()
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            errors="replace", env=env, **_popen_group_kwargs(),
        )
        if registry is not None:
            registry.add(self.proc)
        threading.Thread(target=self._pump, name="slm-hook-mcp-reader", daemon=True).start()

    def _pump(self) -> None:
        stream = self.proc.stdout
        try:
            while True:  # this thread may block; the hook never does
                line = stream.readline()
                if not line:  # EOF, including after a kill
                    break
                self._lines.put(line)
        except (OSError, ValueError, StopIteration):
            pass
        finally:
            self._lines.put(None)

    def send(self, message: dict) -> bool:
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()
            return True
        except (OSError, ValueError, AssertionError):
            return False

    def reply(self, request_id: int, deadline: HookDeadline) -> dict | None:
        """The response with ``request_id``, or ``None`` at the deadline/EOF."""
        while True:
            limit = deadline.remaining()
            if limit <= 0:
                return None
            try:
                line = self._lines.get(timeout=min(limit, 0.5))
            except queue.Empty:
                continue
            if line is None:
                return None
            try:
                message = json.loads(line)
            except ValueError:
                continue  # malformed line: skip, never crash the hook
            if isinstance(message, dict) and message.get("id") == request_id:
                return message

    def close(self) -> None:
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except OSError:
            pass
        stop_child(self.proc)
        if self._registry is not None:
            self._registry.discard(self.proc)


class Watchdog:
    """Guarantees one well-formed output and an exit by a hard deadline."""

    def __init__(
        self,
        seconds: float,
        degraded: Callable[[], str],
        *,
        registry: ChildRegistry | None = None,
        exit_code: int = 0,
    ) -> None:
        self._lock = threading.Lock()
        self._emitted = False
        self._degraded = degraded
        self._registry = registry
        self._exit_code = exit_code
        self._timer = threading.Timer(max(0.0, seconds), self._fire)
        self._timer.daemon = True

    def start(self) -> "Watchdog":
        self._timer.start()
        return self

    def emit(self, text: str) -> bool:
        """Write the hook's output once; later calls (or the watchdog) are no-ops."""
        with self._lock:
            if self._emitted:
                return False
            self._emitted = True
            _write_stdout(text)
            return True

    def cancel(self) -> None:
        self._timer.cancel()

    def _fire(self) -> None:
        if self._registry is not None:
            self._registry.stop_all()
        self.emit(self._degraded())
        os._exit(self._exit_code)


def _write_stdout(text: str) -> None:
    data = (text if text.endswith("\n") else text + "\n").encode("utf-8", "replace")
    try:
        sys.stdout.flush()
    except (OSError, ValueError):
        pass
    try:
        os.write(sys.stdout.fileno(), data)
    except (OSError, ValueError, AttributeError):
        try:
            sys.stdout.write(data.decode("utf-8"))
            sys.stdout.flush()
        except (OSError, ValueError):
            pass
