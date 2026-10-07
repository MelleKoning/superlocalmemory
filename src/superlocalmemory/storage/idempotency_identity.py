# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Which caller an idempotency key belongs to.

A save is retried with the same idempotency key so that it is stored once.
The key is bound to the caller who first used it, so nobody else can replay
it and receive (or overwrite) that caller's receipt. That binding must name a
caller that is STABLE across daemon restarts, or every retry after a restart
looks like a different caller and is refused.

The tool interface, the command line and the proxy authenticate with the
daemon's private capability. That capability is regenerated on every start,
and its actor id (``daemon-capability:<fingerprint>``) changed with it, so
4.1.21 refused every retry after a restart as "a different request".

Who can present a daemon capability is the same set of processes on every
start: whoever can read this data folder's owner-only daemon descriptor. All
starts of this data folder's daemon are therefore one caller, and the key is
bound to that caller. The journal and the ingestion record live inside the
same data folder, so the binding cannot reach another folder's daemon.

Every other actor id (an API key, the local install token) is already stable
and is bound exactly as before, byte for byte: a key used by one of them is
still refused to every other.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Actor ids minted from the per-start daemon capability.
_PER_START_PREFIX = "daemon-capability:"

#: The one caller every start of this data folder's daemon client stands for.
DAEMON_CLIENT_PRINCIPAL = "daemon-capability"

_ACTOR_FIELD = "trusted_actor_id"


def idempotency_principal(actor_id: object) -> str:
    """The caller an idempotency key is bound to, for ``actor_id``."""
    actor = str(actor_id or "")
    if actor.startswith(_PER_START_PREFIX):
        return DAEMON_CLIENT_PRINCIPAL
    return actor


def same_principal(first: object, second: object) -> bool:
    """Whether two actor ids are the same caller for idempotency."""
    return idempotency_principal(first) == idempotency_principal(second)


def stable_request_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """``payload`` with its actor replaced by the stable caller it stands for.

    Hash THIS, never the raw payload. Unchanged for every actor that was
    already stable, so their request hashes are byte-identical to before.
    """
    stable = dict(payload)
    if _ACTOR_FIELD in stable:
        stable[_ACTOR_FIELD] = idempotency_principal(stable[_ACTOR_FIELD])
    return stable


__all__ = [
    "DAEMON_CLIENT_PRINCIPAL",
    "idempotency_principal",
    "same_principal",
    "stable_request_payload",
]
