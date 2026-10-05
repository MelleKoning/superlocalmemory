# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""LLM/embedding provider endpoint trust validation (#112 part 2).

Pure functions, no I/O. Applies the SAME endpoint-trust posture already
established for the remote reranker (``retrieval/remote_reranker_config.py``,
issue #112 part 1) to a second kind of operator-supplied endpoint: a custom
OpenAI-compatible LLM provider (Mode B's local model, or Mode C's own
endpoint) configured from the CLI.

Posture, unchanged from the reranker's:
  1. Loopback (127.x, ::1, localhost) — always allowed over plain HTTP.
  2. A NUMERIC RFC1918/ULA/link-local address — allowed over plain HTTP only
     when ``trust_plain_http_lan`` is True (the default). Bare hostnames are
     never trusted as proof of locality, even if they currently resolve to a
     private IP — DNS is mutable and not a trust boundary.
  3. Everything else (public IPs, bare hostnames) — always requires HTTPS.

This module does not change that posture. It reuses the same classification
helpers (``_is_loopback_host`` / ``_is_private_lan_host``) so the two
endpoints can never silently drift apart; only the error wording differs,
via ``label``, so a CLI user sees "the LLM endpoint" rather than
"retrieval.cross_encoder_endpoint".

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

from urllib.parse import urlparse

from superlocalmemory.retrieval.remote_reranker_config import (
    _is_loopback_host,
    _is_private_lan_host,
)


def validate_provider_endpoint_url(
    endpoint: str,
    *,
    trust_plain_http_lan: bool = True,
    label: str = "the configured endpoint",
) -> str | None:
    """Return an actionable error string, or ``None`` when the URL is safe.

    Args:
        endpoint: The operator-supplied URL. Blank is not an error here —
            callers that require a non-blank endpoint check that separately,
            with their own message naming the missing field.
        trust_plain_http_lan: When True (the default), a numeric private-LAN
            address may use plain HTTP. Read from
            ``config.retrieval.trust_plain_http_lan`` — the same flag the
            reranker uses, so one setting governs both.
        label: Names the field in error messages (e.g. "the LLM endpoint").
    """
    endpoint = (endpoint or "").strip()
    if not endpoint:
        return None

    try:
        parsed = urlparse(endpoint)
    except ValueError as exc:
        return f"{label} is not a valid URL: {exc}"
    if parsed.scheme not in ("http", "https"):
        return (
            f"{label} must use http or https, got "
            f"{parsed.scheme or '(none)'!r}."
        )
    if not parsed.hostname:
        return (
            f"{label} has no host; expected something like "
            f"\"http://127.0.0.1:8080/v1\"."
        )
    if parsed.query or parsed.fragment:
        return f"{label} must not include a query string or fragment."
    if parsed.username or parsed.password:
        # httpx logs "HTTP Request: POST <url>" at INFO using str(url), which
        # renders an embedded password in full — refuse at the door instead
        # of trusting every caller down the line to redact it.
        return (
            f"{label} must not embed credentials (user:password@host) — the "
            f"HTTP client logs request URLs in full. Use the API key option "
            f"instead."
        )
    if parsed.scheme == "http":
        hostname = parsed.hostname
        if _is_loopback_host(hostname):
            return None  # loopback always allowed regardless of trust flag
        if trust_plain_http_lan and _is_private_lan_host(hostname):
            # Numeric private address on an operator-trusted LAN. Threat
            # model: an attacker already on the same physical LAN can still
            # MITM plain HTTP (ARP spoofing) — this is allowed because the
            # LAN is assumed to be under the operator's control.
            return None
        if not _is_private_lan_host(hostname):
            return (
                f"{label} must use HTTPS for this host. Plain HTTP is "
                f"allowed only for loopback (127.x/::1/localhost) and "
                f"numeric private-LAN addresses (RFC1918: 10.x, 172.16-31.x, "
                f"192.168.x; IPv6 ULA fc00::/7; link-local 169.254.x/fe80::). "
                f"Bare hostnames are not trusted even if they resolve to a "
                f"private IP — use a numeric address or configure HTTPS."
            )
        # Private-LAN address but trust_plain_http_lan is False (hardened).
        return (
            f"{label} uses plain HTTP to a private-LAN address. HTTPS is "
            f"required because retrieval.trust_plain_http_lan is set to "
            f"false. Either put a TLS-terminating proxy in front of it, or "
            f"set retrieval.trust_plain_http_lan=true to permit plain HTTP "
            f"within your private network (default for new installs)."
        )
    return None
