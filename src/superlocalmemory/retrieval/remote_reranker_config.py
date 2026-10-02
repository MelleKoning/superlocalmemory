# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Remote reranker configuration: which backend, which URL, and whether it is safe.

Pure functions, no I/O. Split out of ``remote_reranker`` (which re-exports
every name here) so that module stays within the size limit.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse, urlunparse

# Backend tokens that select the remote path. "openai" is what issue #105
# asked for and matches ``embedding.provider == "openai"``, the established
# repo token for "any OpenAI-compatible HTTP endpoint". It is a slight misnomer
# — OpenAI has no rerank API and these endpoints are usually llama-server or
# TEI — so "remote" is accepted as a truthful alias.
REMOTE_CROSS_ENCODER_BACKENDS = ("openai", "remote")


def is_remote_cross_encoder_backend(backend: str) -> bool:
    """True when ``backend`` selects the remote reranker."""
    return (backend or "").strip().lower() in REMOTE_CROSS_ENCODER_BACKENDS


def validate_remote_reranker_config(
    backend: str,
    endpoint: str,
    trust_plain_http_lan: bool = True,
) -> str | None:
    """Return an actionable error string, or None when the pair is coherent.

    Covers the issue-#103 leftover directly: an endpoint configured against a
    LOCAL backend used to be dropped on the floor by ``SLMConfig.load``. It now
    produces a named error naming both keys and the exact edit to make.

    Args:
        backend:            Value of ``retrieval.cross_encoder_backend``.
        endpoint:           Value of ``retrieval.cross_encoder_endpoint``.
        trust_plain_http_lan: When True (the default), numeric RFC1918/ULA/
            link-local addresses may use plain HTTP — the same security posture
            as the local reranker, where memory text only crosses loopback.
            Set to False in hardened deployments (zero-trust networks, shared
            colocation) to require HTTPS for all non-loopback hosts.

    Threat model note: trusting a private-LAN address does NOT prevent a
    MITM attack by an adversary on the same physical LAN (e.g. via ARP
    spoofing). This flag means "the LAN is under my control and I accept that
    risk." It is not a claim that RFC1918 traffic is cryptographically secure.
    """
    backend = (backend or "").strip()
    endpoint = (endpoint or "").strip()
    remote = is_remote_cross_encoder_backend(backend)

    if remote and not endpoint:
        return (
            f"retrieval.cross_encoder_backend={backend!r} selects the remote "
            f"reranker but retrieval.cross_encoder_endpoint is empty. Set the "
            f"endpoint (e.g. \"http://127.0.0.1:8041/v1/rerank\"), or set "
            f"cross_encoder_backend to \"\" (PyTorch) / \"onnx\" to rerank "
            f"locally."
        )
    if endpoint and not remote:
        return (
            f"retrieval.cross_encoder_endpoint is set to {endpoint!r} but "
            f"retrieval.cross_encoder_backend={backend!r} is a LOCAL backend, "
            f"so the endpoint would be ignored. Set cross_encoder_backend to "
            f"\"openai\" to use the endpoint, or remove cross_encoder_endpoint "
            f"to rerank locally."
        )
    if not remote:
        return None
    return _validate_endpoint_url(endpoint, trust_plain_http_lan=trust_plain_http_lan)


def _validate_endpoint_url(
    endpoint: str,
    trust_plain_http_lan: bool = True,
) -> str | None:
    """Scheme/host allow-listing for the operator-supplied rerank URL.

    Plain-HTTP allowances (most-to-least trusted):
      1. Loopback (127.x, ::1, localhost) — always allowed.
      2. Numeric RFC1918/ULA/link-local addresses — allowed when
         ``trust_plain_http_lan`` is True (the default).  Only numeric
         addresses qualify; bare hostnames are never trusted because DNS is
         mutable and not a trust boundary.
      3. Everything else (public IPs, bare hostnames) — always requires HTTPS.
    """
    try:
        parsed = urlparse(endpoint)
    except ValueError as exc:
        return f"retrieval.cross_encoder_endpoint is not a valid URL: {exc}"
    if parsed.scheme not in ("http", "https"):
        return (
            f"retrieval.cross_encoder_endpoint must use http or https, got "
            f"{parsed.scheme or '(none)'!r}. SuperLocalMemory will not open "
            f"file/ftp/other schemes for reranking."
        )
    if not parsed.hostname:
        return (
            "retrieval.cross_encoder_endpoint has no host; expected something "
            "like \"http://127.0.0.1:8041/v1/rerank\"."
        )
    if parsed.query or parsed.fragment:
        return (
            "retrieval.cross_encoder_endpoint must not include a query string "
            "or fragment. Put bearer credentials in "
            "SLM_CROSS_ENCODER_API_KEY and configure a clean endpoint URL."
        )
    if parsed.username or parsed.password:
        # httpx logs "HTTP Request: POST <url>" at INFO using str(url), which
        # renders an embedded password in full. This module never logs the raw
        # URL, but it does not own the httpx logger — so credentials are
        # refused at the door instead of being trusted to stay redacted.
        return (
            "retrieval.cross_encoder_endpoint must not embed credentials "
            "(user:password@host) — the HTTP client logs request URLs in "
            "full. Put the token in SLM_CROSS_ENCODER_API_KEY (preferred) or "
            "retrieval.cross_encoder_api_key; it is sent as a Bearer header "
            "and never logged."
        )
    if parsed.scheme == "http":
        hostname = parsed.hostname
        if _is_loopback_host(hostname):
            return None  # loopback always allowed regardless of trust flag
        if trust_plain_http_lan and _is_private_lan_host(hostname):
            # Numeric private address on an operator-trusted LAN. Threat model:
            # an attacker on the same physical LAN can still MITM plain HTTP
            # (ARP spoofing). This is allowed because the LAN is assumed to be
            # under the operator's control. Set trust_plain_http_lan=False in
            # hardened/zero-trust environments.
            return None
        if not _is_private_lan_host(hostname):
            # Public IP, CGNAT, or a bare hostname (DNS not trusted as a
            # proof of locality). Bare hostnames that happen to resolve to
            # private IPs are NOT trusted: DNS can be poisoned or changed,
            # so only provably-private numeric addresses are accepted.
            return (
                "retrieval.cross_encoder_endpoint must use HTTPS for this "
                "host. Plain HTTP is allowed only for loopback "
                "(127.x/::1/localhost) and numeric private-LAN addresses "
                "(RFC1918: 10.x, 172.16-31.x, 192.168.x; IPv6 ULA fc00::/7; "
                "link-local 169.254.x/fe80::). "
                "Bare hostnames are not trusted even if they resolve to a "
                "private IP — use a numeric address or configure HTTPS."
            )
        # Private-LAN address but trust_plain_http_lan is False (hardened mode)
        return (
            "retrieval.cross_encoder_endpoint uses plain HTTP to a "
            "private-LAN address. HTTPS is required because "
            "retrieval.trust_plain_http_lan is set to false. "
            "Either configure a TLS-terminating proxy on the reranker, or "
            "set retrieval.trust_plain_http_lan=true to permit plain HTTP "
            "within your private network (default for new installs)."
        )
    return None


def _is_loopback_host(hostname: str) -> bool:
    """Return True only for literal loopback names/addresses (no DNS trust)."""
    host = (hostname or "").rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_private_lan_host(hostname: str) -> bool:
    """True only for numeric private-range addresses (RFC1918, ULA, link-local).

    Deliberate non-DNS: bare hostnames (e.g. ``my-reranker.lan``) return False
    even if they currently resolve to a private IP. DNS is mutable and not a
    trust boundary — an adversary who can influence DNS resolution can redirect
    the endpoint to a public host, defeating the locality check. Only numeric
    addresses are provably bound to a private range at configuration time.

    Accepted ranges (Python 3.11+ ``ipaddress.is_private``):
      IPv4  RFC1918: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
      IPv4  link-local: 169.254.0.0/16
      IPv6  ULA: fc00::/7 (includes fd00::/8)
      IPv6  link-local: fe80::/10

    Excluded ranges (not accepted for plain HTTP):
      CGNAT 100.64.0.0/10 — ISP-shared address space, not operator-controlled
      172.15.0.0/8 and 172.32.0.0/8 — outside the 172.16.0.0/12 boundary
      Public unicast addresses

    IPv4-mapped IPv6 addresses (``::ffff:192.168.1.1``) are unwrapped to their
    IPv4 equivalent before the range check, so they are handled consistently.
    """
    host = (hostname or "").rstrip(".").lower()
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        # Not a numeric address — bare hostname, not provably private
        return False
    # Unwrap IPv4-mapped IPv6 (::ffff:192.168.1.1 → 192.168.1.1) so the
    # RFC1918 check applies to the IPv4 portion.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr.is_private


def normalize_rerank_endpoint(endpoint: str) -> str:
    """Append ``/rerank`` when the URL stops at the API root.

    Mirrors the embedding path's ``/embeddings`` suffixing so a user can paste
    either ``http://host:8041/v1`` or ``http://host:8041/v1/rerank``.
    """
    url = (endpoint or "").strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.path.endswith("/rerank"):
        return url
    return f"{url}/rerank"


def redact_endpoint(endpoint: str) -> str:
    """Drop any ``user:password@`` userinfo before an endpoint reaches a log.

    Defence in depth. ``_validate_endpoint_url`` already refuses credentialed
    URLs, so this should never have anything to strip in a configured install
    — it exists so that any future caller constructing a reranker directly
    still cannot put a password in the log.
    """
    try:
        parsed = urlparse(endpoint)
    except ValueError:
        return "<unparseable endpoint>"
    if not parsed.hostname:
        return endpoint
    netloc = parsed.hostname
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username or parsed.password:
        netloc = f"***@{netloc}"
    return urlunparse(parsed._replace(netloc=netloc, query="", fragment=""))
