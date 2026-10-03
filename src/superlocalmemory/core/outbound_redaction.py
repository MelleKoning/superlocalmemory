# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Credentials stay in SLM; they never travel to another machine.

SLM stores a memory exactly as written, credentials included - keeping them is
part of what it is for. The boundary is the network: text sent to a hosted
model or a cloud embedding service has every recognized credential replaced
first (``retrieval.hosted_redaction``, the screen the online answer check
already uses). A service on this machine - a local Ollama, a local
OpenAI-compatible server - sees the text as it is.

"On this machine" means a loopback host: ``localhost`` (or a name under it),
``127.0.0.0/8`` or ``::1``. Anything else, including a LAN address or a URL
that cannot be read, is treated as another machine.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from superlocalmemory.retrieval.hosted_redaction import redact_for_hosted_judge


def is_local_endpoint(url: str) -> bool:
    """True only for a URL whose host is this machine (loopback)."""
    if not isinstance(url, str) or "://" not in url:
        return False
    try:
        host = (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def for_endpoint(text: str, url: str) -> str:
    """``text`` as it may be sent to ``url``: unchanged for this machine,
    credentials replaced for anywhere else. Never raises."""
    if not isinstance(text, str):
        return ""
    if is_local_endpoint(url):
        return text
    return redact_for_hosted_judge(text)


__all__ = ["for_endpoint", "is_local_endpoint"]
