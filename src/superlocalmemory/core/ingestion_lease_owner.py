# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Who holds an enrichment lease, and is that process still alive?

A save is enriched under a lease (900 s, renewed every 30 s while work runs).
When the service is killed mid-enrichment the lease outlives its owner, and
the restarted service had to wait out the whole 900 s before it could finish
that save. The owner token now names the machine, the process id and the
process's start time, so a lease whose owner is provably gone is released at
once. Anything uncertain — an older token, another machine, a live process,
a start time that cannot be read — keeps its lease and expires as before.

Token: ``ingestion-worker:<host>:<pid>@<start>:<random>``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PREFIX = "ingestion-worker:"
#: Start times read twice for one process differ by far less than this.
_START_TOLERANCE_S = 1.0


@dataclass(frozen=True)
class Owner:
    host: str
    pid: int
    started: float


def host_tag() -> str:
    """A short, stable tag for this machine (never the host name itself)."""
    return hashlib.sha256(socket.gethostname().encode("utf-8", "replace")).hexdigest()[:10]


def _start_time(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:  # noqa: BLE001 — unknown start time means "cannot tell"
        return None


def owner_token() -> str:
    pid = os.getpid()
    started = _start_time(pid)
    stamp = f"{started:.3f}" if started is not None else "unknown"
    return f"{PREFIX}{host_tag()}:{pid}@{stamp}:{uuid.uuid4().hex}"


def parse_owner(token: str) -> Owner | None:
    """The owner named by ``token``, or None when it does not name one."""
    if not isinstance(token, str) or not token.startswith(PREFIX):
        return None
    try:
        host, rest = token[len(PREFIX):].split(":", 1)
        pid_text, rest = rest.split("@", 1)
        started_text, _ = rest.split(":", 1)
        return Owner(host=host, pid=int(pid_text), started=float(started_text))
    except ValueError:
        return None


def owner_is_dead(token: str) -> bool:
    """True only when the owner is provably gone on this machine."""
    from superlocalmemory.core.platform_utils import is_pid_alive

    owner = parse_owner(token)
    if owner is None or owner.host != host_tag() or owner.pid <= 0:
        return False
    if not is_pid_alive(owner.pid):
        return True
    started = _start_time(owner.pid)
    return started is not None and abs(started - owner.started) > _START_TOLERANCE_S


def release_dead_owner_leases(db, *, now: float | None = None) -> list[str]:
    """Make the leases of dead owners due now; compare-and-swap per operation."""
    cutoff = time.time() if now is None else float(now)
    released: list[str] = []
    rows = db.execute(
        "SELECT operation_id, lease_owner FROM ingestion_operations "
        "WHERE state='enriching' AND lease_expires_at > ?", (cutoff,))
    for row in rows:
        data = dict(row)
        if not owner_is_dead(str(data["lease_owner"])):
            continue
        updated = db.execute(
            "UPDATE ingestion_operations SET lease_expires_at=0, "
            "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE operation_id=? AND state='enriching' AND lease_owner=? "
            "RETURNING operation_id",
            (data["operation_id"], data["lease_owner"]))
        if updated:
            released.append(str(data["operation_id"]))
    if released:
        logger.info("released %d enrichment lease(s) held by a process that has exited",
                    len(released))
    return released
