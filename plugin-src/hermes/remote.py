"""Remote mode for the Hermes SLM plugin: /slm commands over Hermes's own MCP client.

Selected only by ``plugins.entries.superlocalmemory.settings.connection: remote``.
This module opens no sockets and runs no programs: every call goes through
``ctx.call_mcp("superlocalmemory", ...)``, i.e. the MCP server the user configured
in Hermes (URL, TLS verification and key all live in Hermes's config). It imports
no SuperLocalMemory package.

A save that the SLM server did not confirm is reported as ``NOT SAVED`` - never as
success. Automatic captures pause while the server is unreachable and are counted,
so ``/slm status`` shows what was skipped. Nothing is kept on this computer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger("superlocalmemory.hermes.remote")

SERVER = "superlocalmemory"
CIRCUIT_SECONDS = 30.0
VERSION_CACHE_SECONDS = 300.0
MAX_CONTENT = 50_000
READ_ONLY_TAG = "[remote_key_read_only]"
#: Lifecycle tools that write. A read-only key is refused these; once that is
#: seen the plugin stops sending them (and counts them as skipped).
LIFECYCLE_WRITES = frozenset({"session_init", "log_tool_event", "observe",
                              "settle_session_outcomes", "close_session"})
#: Skills that work against a remote SLM (the rest manage the SLM computer).
REMOTE_SKILLS = ("slm-cache", "slm-compress", "slm-recall", "slm-remember", "slm-scope",
                 "slm-session", "slm-status")
HOST_ONLY_ROLES = frozenset({"governance", "loop"})
HOST_ONLY_MESSAGE = ("'/slm {cmd}' manages the SLM computer and is not available over a "
                     "remote connection. Run 'slm {cmd}' on the SLM computer.")
_TIMEOUTS = {"recall": 8, "remember": 15, "delete_memory": 15, "update_memory": 15}


def _bounded(value: Any, limit: int = 300) -> str:
    text = "" if value is None else str(value)
    return text[:limit] + ("… [truncated]" if len(text) > limit else "")


@dataclass(frozen=True)
class Envelope:
    ok: bool
    payload: Any = None
    reason: str = ""
    truncated: bool = False

    @classmethod
    def fail(cls, code: str, reason: str) -> "Envelope":
        return cls(False, None, f"{reason} ({code})" if code else reason)

    @classmethod
    def from_hermes(cls, raw: Any) -> "Envelope":
        if not isinstance(raw, dict):
            return cls.fail("bad_envelope", "Hermes returned an unexpected result")
        if raw.get("ok") is not True:
            return cls(False, None, _bounded(raw.get("error") or "the SLM server did not answer"))
        payload = raw.get("structuredContent")
        if payload is None:
            result = raw.get("result")
            if isinstance(result, str):
                try:
                    payload = json.loads(result)
                except ValueError:
                    payload = result
            else:
                payload = result
        return cls(True, payload, "", raw.get("truncated") is True)


class RemoteHealth:
    """Outage circuit and counters for automatic captures. Thread-safe."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self.failures_total = 0
        self.dropped_while_open = 0
        self.dropped_read_only = 0
        self.last_error = ""
        self.last_ok_at: str | None = None
        self.read_only_key = False
        self._open_until = 0.0
        self._down = False
        self._version: tuple[float, str | None] | None = None

    def record(self, ok: bool, reason: str = "") -> None:
        with self._lock:
            if ok:
                if self._down:
                    logger.info("SLM remote reachable again")
                self._down = False
                self._open_until = 0.0
                self.last_ok_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
                return
            self.failures_total += 1
            self.last_error = _bounded(reason, 300)
            self._open_until = self._clock() + CIRCUIT_SECONDS
            if not self._down:
                self._down = True
                logger.warning("SLM remote unreachable: %s; automatic capture paused for %d s; "
                               "explicit /slm remember still tries every time",
                               self.last_error, int(CIRCUIT_SECONDS))

    def mark_read_only(self) -> None:
        with self._lock:
            if not self.read_only_key:
                logger.warning("SLM remote key is read-only: automatic capture is off; "
                               "recall still works")
            self.read_only_key = True

    def lifecycle_allowed(self, tool: str) -> bool:
        with self._lock:
            if self.read_only_key and tool in LIFECYCLE_WRITES:
                self.dropped_read_only += 1
                return False
            if self._clock() < self._open_until:
                self.dropped_while_open += 1
                return False
            return True

    def status_line(self) -> str:
        with self._lock:
            parts = [
                "connection: remote",
                f"failures since start: {self.failures_total}",
                f"lifecycle captures skipped while unreachable: {self.dropped_while_open}",
            ]
            if self.read_only_key:
                parts.append(f"read-only key, captures not sent: {self.dropped_read_only}")
            parts.append(f"last error: {self.last_error or 'none'}")
            parts.append(f"last success: {self.last_ok_at or 'never'}")
            return " · ".join(parts)

    def cached_version(self) -> str | None | bool:
        """The cached server version, or ``False`` when the cache is stale."""
        with self._lock:
            if self._version is None or self._clock() - self._version[0] > VERSION_CACHE_SECONDS:
                return False
            return self._version[1]

    def remember_version(self, version: str | None) -> None:
        with self._lock:
            self._version = (self._clock(), version)


def is_read_only_refusal(text: Any) -> bool:
    return READ_ONLY_TAG in str(text or "")


def call(ctx: Any, tool: str, args: dict[str, Any], timeout: float) -> Envelope:
    try:
        raw = ctx.call_mcp(SERVER, tool, args, timeout=timeout)
    except PermissionError:
        return Envelope.fail("grant_missing", "Hermes has not granted the superlocalmemory MCP "
                             "server to this plugin (plugins.entries.superlocalmemory."
                             "mcp_allowlist)")
    except Exception as exc:  # noqa: BLE001 — a transport failure is a reason, not a crash
        return Envelope.fail("transport", _bounded(exc))
    return Envelope.from_hermes(raw)


# -- argument builders -------------------------------------------------------------------


class _Usage(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # never exit the Hermes process
        raise _Usage(message)

    def exit(self, status: int = 0, message: str | None = None) -> None:
        raise _Usage(message or "")


def _parser(prog: str) -> _Parser:
    return _Parser(prog=f"/slm {prog}", add_help=False, exit_on_error=False)


def _int_range(low: int, high: int):
    def convert(text: str) -> int:
        value = int(text)
        if not low <= value <= high:
            raise ValueError
        return value
    convert.__name__ = f"integer {low}-{high}"
    return convert


def _no_args(argv: list[str], _sid: str) -> dict | str:
    return {} if not argv else "This command takes no arguments over a remote connection."


def _recall_args(argv: list[str], _sid: str) -> dict | str:
    p = _parser("recall")
    p.add_argument("query", nargs="+")
    p.add_argument("--limit", type=_int_range(1, 50), default=10)
    p.add_argument("--kind", default="")
    ns = p.parse_args(argv)
    args = {"query": " ".join(ns.query), "limit": ns.limit, "agent_id": "hermes"}
    return {**args, "kind": ns.kind} if ns.kind else args


def _remember_args(argv: list[str], session_id: str) -> dict | str:
    p = _parser("remember")
    p.add_argument("content", nargs="+")
    p.add_argument("--tags", default="")
    p.add_argument("--kind", default="")
    p.add_argument("--replaces", default=None)
    ns = p.parse_args(argv)
    content = " ".join(ns.content)
    if len(content) > MAX_CONTENT:
        return f"NOT SAVED: the memory is longer than {MAX_CONTENT} characters."
    args: dict[str, Any] = {"content": content, "tags": ns.tags, "agent_id": "hermes"}
    if ns.kind:
        args["kind"] = ns.kind
    if ns.replaces:
        args["replaces"] = ns.replaces
    seed = json.dumps({"session": session_id, **args}, sort_keys=True)
    args["idempotency_key"] = hashlib.sha256(
        ("hermes-remote\0" + seed).encode("utf-8")).hexdigest()[:32]
    return args


def _list_args(argv: list[str], _sid: str) -> dict | str:
    p = _parser("list")
    p.add_argument("-n", "--limit", type=_int_range(1, 200), default=20)
    p.add_argument("--kind", default="")
    ns = p.parse_args(argv)
    return {"limit": ns.limit, **({"kind": ns.kind} if ns.kind else {})}


def _fact_id_args(argv: list[str], _sid: str) -> dict | str:
    p = _parser("delete")
    p.add_argument("fact_id")
    return {"fact_id": p.parse_args(argv).fact_id, "agent_id": "hermes"}


def _update_args(argv: list[str], _sid: str) -> dict | str:
    p = _parser("update")
    p.add_argument("fact_id")
    p.add_argument("content", nargs="+")
    ns = p.parse_args(argv)
    return {"fact_id": ns.fact_id, "content": " ".join(ns.content), "agent_id": "hermes"}


def _summary_args(argv: list[str], _sid: str) -> dict | str:
    p = _parser("summary")
    p.add_argument("kind", nargs="?", default="day")
    p.add_argument("target", nargs="?", default="")
    ns = p.parse_args(argv)
    return {"kind": ns.kind, "target": ns.target}


def _trace_args(argv: list[str], _sid: str) -> dict | str:
    p = _parser("trace")
    p.add_argument("query", nargs="+")
    p.add_argument("--limit", type=_int_range(1, 50), default=10)
    ns = p.parse_args(argv)
    return {"query": " ".join(ns.query), "limit": ns.limit}


# -- renderers ---------------------------------------------------------------------------


def _render_json(env: Envelope) -> str:
    if not env.ok:
        return f"SLM server error: {env.reason}"
    text = env.payload if isinstance(env.payload, str) else json.dumps(env.payload, indent=2,
                                                                      sort_keys=True)
    return _bounded(text, 8_000)


def _render_remember(env: Envelope) -> str:
    if not env.ok:
        return f"NOT SAVED: {env.reason}"
    payload = env.payload
    if not isinstance(payload, dict) or payload.get("success") is not True:
        detail = (payload or {}) if isinstance(payload, dict) else {}
        reason = detail.get("error") or detail.get("code") or \
            "the SLM server did not confirm the write"
        return f"NOT SAVED: {_bounded(reason)}"
    count = len(payload.get("fact_ids") or [])
    if payload.get("pending"):
        return f"Saved ({count} fact(s); queryable now, enrichment still running)."
    return f"Saved ({count} fact(s))."


def _render_mutation(env: Envelope) -> str:
    if not env.ok:
        return f"NOT CHANGED: {env.reason}"
    payload = env.payload
    if not isinstance(payload, dict) or payload.get("success") is not True:
        detail = payload if isinstance(payload, dict) else {}
        return f"NOT CHANGED: {_bounded(detail.get('error') or 'the SLM server did not confirm it')}"
    return "Done."


def remote_help() -> str:
    names = ", ".join(sorted(REMOTE_COMMANDS))
    return ("Remote connection: these /slm commands run on the SLM server: " + names +
            ". Commands that manage the SLM computer run there with 'slm <command>'.")


@dataclass(frozen=True)
class RemoteSpec:
    tool: str | None
    build: Callable[[list[str], str], dict | str]
    render: Callable[[Envelope], str]


REMOTE_COMMANDS: dict[str, RemoteSpec] = {
    "status": RemoteSpec("get_status", _no_args, _render_json),
    "health": RemoteSpec("health", _no_args, _render_json),
    "recall": RemoteSpec("recall", _recall_args, _render_json),
    "remember": RemoteSpec("remember", _remember_args, _render_remember),
    "list": RemoteSpec("list_recent", _list_args, _render_json),
    "delete": RemoteSpec("delete_memory", _fact_id_args, _render_mutation),
    "update": RemoteSpec("update_memory", _update_args, _render_mutation),
    "summary": RemoteSpec("get_memory_summary", _summary_args, _render_json),
    "trace": RemoteSpec("recall_trace", _trace_args, _render_json),
    "kinds": RemoteSpec("memory_kinds_status", _no_args, _render_json),
    "help": RemoteSpec(None, _no_args, lambda _env: remote_help()),
}


def version_problem(ctx: Any, health: RemoteHealth, release: tuple[int, int, int]) -> str | None:
    """The plugin runs only with the SLM release it ships with, remote or local."""
    wanted = ".".join(str(p) for p in release)
    cached = health.cached_version()
    if cached is False:
        env = call(ctx, "get_status", {}, timeout=8)
        if not env.ok:
            health.record(False, env.reason)
            return f"SLM server unavailable: {env.reason}"
        health.record(True)
        version = env.payload.get("version") if isinstance(env.payload, dict) else None
        health.remember_version(str(version) if version else None)
        cached = health.cached_version()
    if cached != wanted:
        return (f"Hermes SLM plugin requires exactly SLM {wanted} on the SLM server; the server "
                f"reports {cached or 'no version'}. Upgrade one side so they match.")
    return None


def run_remote(ctx: Any, argv: list[str], session_id: str, health: RemoteHealth,
               release: tuple[int, int, int]) -> str:
    command = argv[0]
    spec = REMOTE_COMMANDS.get(command)
    if spec is None:
        return HOST_ONLY_MESSAGE.format(cmd=command)
    if spec.tool is None:
        return remote_help()
    try:
        args = spec.build(argv[1:], session_id)
    except (_Usage, argparse.ArgumentError) as exc:
        return f"Usage error for /slm {command}: {exc}"
    if isinstance(args, str):
        return args
    problem = version_problem(ctx, health, release)
    if problem:
        return f"NOT SAVED: {problem}" if command == "remember" else problem
    env = call(ctx, spec.tool, args, timeout=_TIMEOUTS.get(spec.tool, 8))
    if env.ok:
        health.record(True)
    elif is_read_only_refusal(env.reason):
        health.mark_read_only()
    else:
        health.record(False, env.reason)
    rendered = spec.render(env)
    if command == "status":
        rendered = f"{rendered}\n{health.status_line()}"
    return rendered


__all__ = ["Envelope", "HOST_ONLY_MESSAGE", "HOST_ONLY_ROLES", "LIFECYCLE_WRITES",
           "REMOTE_COMMANDS", "REMOTE_SKILLS", "RemoteHealth", "call", "is_read_only_refusal",
           "remote_help", "run_remote", "version_problem"]
