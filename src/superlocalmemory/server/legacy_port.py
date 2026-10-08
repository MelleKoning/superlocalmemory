# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The deprecated port-8767 TCP redirect, and who may take it.

Pre-descriptor clients (V3.6 and older) dial ``127.0.0.1:8767`` and know
nothing about data roots. The redirect forwards them to the daemon's real
port. Before 4.1.22 EVERY daemon tried to take 8767 unless
``SLM_DISABLE_LEGACY_PORT=1`` — so a second data root (a test, a team store,
a second profile) started first could capture it and send old clients'
reads and writes to the wrong store.

Now only a daemon serving the *default* data root takes it: the root a
process with no environment selection resolves to
(:func:`superlocalmemory.infra.data_root.is_implicit_default_root`), which is
the only root an old client can mean.
"""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

LEGACY_PORT = 8767


def legacy_redirect_decision() -> tuple[bool, str]:
    """``(take_it, reason)`` for this process's legacy-port redirect."""
    if os.environ.get("SLM_DISABLE_LEGACY_PORT", "").lower() in ("1", "true"):
        return False, "disabled by SLM_DISABLE_LEGACY_PORT"
    from superlocalmemory.infra.data_root import canonical_data_root, is_implicit_default_root

    if not is_implicit_default_root():
        return False, f"this daemon serves {canonical_data_root()}, not the default data root"
    return True, "default data root"


async def start_legacy_redirect(
    primary_port: int, legacy_port: int, status: dict | None = None,
) -> None:
    """Byte-level TCP redirect from ``legacy_port`` to ``primary_port``.

    Runs as a task in the daemon's own loop. A port already in use (an older
    daemon, another service) is logged and skipped.

    ``status``, when given, is updated in place once the bind attempt
    resolves: ``{"bound": bool, "port": int | None, "reason": str}``. This is
    how a status endpoint reports the *truth* of whether this daemon holds
    the legacy port, rather than assuming a policy decision always succeeds.
    """
    _deprecation_warned = False

    async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        nonlocal _deprecation_warned
        if not _deprecation_warned:
            logger.warning(
                "Request on deprecated port %d. Update config to use port %d.",
                legacy_port, primary_port,
            )
            _deprecation_warned = True
        try:
            upstream_r, upstream_w = await asyncio.open_connection("127.0.0.1", primary_port)
            await asyncio.gather(_pipe(reader, upstream_w), _pipe(upstream_r, writer))
        except Exception:
            pass
        finally:
            writer.close()

    async def _pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter):
        try:
            while True:
                data = await src.read(8192)
                if not data:
                    break
                dst.write(data)
                await dst.drain()
        except Exception:
            pass
        finally:
            try:
                dst.close()
            except Exception:
                pass

    try:
        server = await asyncio.start_server(_handle_client, "127.0.0.1", legacy_port)
        logger.info("Legacy redirect: port %d → %d (deprecated)", legacy_port, primary_port)
        if status is not None:
            status["bound"] = True
            status["port"] = legacy_port
            status["reason"] = "default data root"
        await server.serve_forever()
    except OSError as exc:
        logger.info("Port %d in use (old daemon?), skipping legacy redirect", legacy_port)
        if status is not None:
            status["bound"] = False
            status["port"] = None
            status["reason"] = f"port {legacy_port} already in use: {exc}"


def maybe_start_legacy_redirect(
    primary_port: int, legacy_port: int = LEGACY_PORT, status: dict | None = None,
):
    """Start the redirect task when this daemon may own the legacy port.

    Returns the task, or ``None`` when the port is left alone. Must be called
    from inside the daemon's running event loop.

    ``status``, when given, is populated in place immediately with the policy
    decision (``bound: False`` + the reason) and then, if the policy says to
    take the port, updated again once the real bind attempt resolves (see
    :func:`start_legacy_redirect`). Callers that only care about the policy
    decision get an answer synchronously; callers that care about the truth
    of whether the socket is actually listening should read ``status`` again
    after a short delay.
    """
    take_it, reason = legacy_redirect_decision()
    if status is not None:
        status["bound"] = False
        status["port"] = None
        status["reason"] = reason
    if not take_it:
        logger.info("Legacy port %d not taken: %s", legacy_port, reason)
        return None
    return asyncio.create_task(start_legacy_redirect(primary_port, legacy_port, status=status))
