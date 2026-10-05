# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for core.provider_endpoint_trust (#112 part 2).

This module reuses the reranker's private-LAN/loopback classifiers verbatim
(tests/test_retrieval/test_lan_endpoint_security.py already proves every
boundary of those), so these tests cover only the wrapper's own behavior:
scheme/credential/fragment refusal, the loopback/trust_plain_http_lan/public
branches, and the ``label`` substitution used to produce a message naming
the right field (the LLM endpoint, not the reranker's).
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.provider_endpoint_trust import validate_provider_endpoint_url


class TestBlankAndMalformed:
    def test_blank_is_not_an_error(self) -> None:
        """Emptiness is a separate, caller-specific check."""
        assert validate_provider_endpoint_url("") is None
        assert validate_provider_endpoint_url("   ") is None

    def test_non_http_scheme_refused(self) -> None:
        err = validate_provider_endpoint_url("ftp://example.com/v1")
        assert err is not None
        assert "http or https" in err

    def test_no_host_refused(self) -> None:
        err = validate_provider_endpoint_url("http://")
        assert err is not None
        assert "no host" in err

    def test_query_string_refused(self) -> None:
        err = validate_provider_endpoint_url("https://example.com/v1?key=abc")
        assert err is not None
        assert "query string" in err

    def test_fragment_refused(self) -> None:
        err = validate_provider_endpoint_url("https://example.com/v1#frag")
        assert err is not None

    def test_embedded_credentials_refused(self) -> None:
        err = validate_provider_endpoint_url("http://user:pass@127.0.0.1:8080/v1")
        assert err is not None
        assert "credentials" in err


class TestLoopbackAlwaysAllowed:
    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:8080/v1",
        "http://localhost:8080/v1",
        "http://[::1]:8080/v1",
    ])
    def test_plain_http_loopback_allowed(self, url: str) -> None:
        assert validate_provider_endpoint_url(url, trust_plain_http_lan=False) is None


class TestPrivateLanRespectsTrustFlag:
    def test_allowed_when_trust_flag_true(self) -> None:
        assert validate_provider_endpoint_url(
            "http://192.168.1.50:8041/v1", trust_plain_http_lan=True,
        ) is None

    def test_refused_when_trust_flag_false(self) -> None:
        err = validate_provider_endpoint_url(
            "http://192.168.1.50:8041/v1", trust_plain_http_lan=False,
        )
        assert err is not None
        assert "trust_plain_http_lan" in err


class TestPublicAndBareHostnamesRequireHttps:
    def test_public_ip_plain_http_refused(self) -> None:
        err = validate_provider_endpoint_url("http://8.8.8.8/v1", trust_plain_http_lan=True)
        assert err is not None
        assert "HTTPS" in err

    def test_bare_hostname_plain_http_refused_even_if_it_resolves_locally(self) -> None:
        """DNS is not a trust boundary — a hostname is never treated as
        provably local, even one that happens to resolve to a private IP."""
        err = validate_provider_endpoint_url(
            "http://my-server.lan:8041/v1", trust_plain_http_lan=True,
        )
        assert err is not None
        assert "Bare hostnames" in err

    def test_https_to_a_public_host_allowed(self) -> None:
        assert validate_provider_endpoint_url("https://api.example.com/v1") is None

    def test_https_to_a_bare_hostname_allowed(self) -> None:
        assert validate_provider_endpoint_url("https://my-server.lan/v1") is None


class TestLabelNamesTheRightField:
    def test_label_appears_in_error(self) -> None:
        err = validate_provider_endpoint_url(
            "http://8.8.8.8/v1", label="the LLM endpoint",
        )
        assert err is not None
        assert err.startswith("the LLM endpoint")

    def test_default_label_is_generic(self) -> None:
        err = validate_provider_endpoint_url("http://8.8.8.8/v1")
        assert err is not None
        assert err.startswith("the configured endpoint")
