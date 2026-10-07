# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``ollama_reachability``: the bounded, cached check behind
``LLMBackbone.is_available()`` for provider ``"ollama"``.

4.1.22: that method used to report True unconditionally for Ollama, so a
saved Mode B with nothing listening never degraded to Mode A. These tests
pin the replacement contract directly against the module (no real network,
no real Ollama): reachable/unreachable both read correctly, the TTL cache
means a second call inside the window never probes again, and a probe that
hangs past its own timeout still reads as unreachable rather than blocking.
"""

from __future__ import annotations

import time

import httpx
import pytest

from superlocalmemory.llm import ollama_reachability as r


@pytest.fixture(autouse=True)
def _clean_cache():
    r.clear_cache()
    yield
    r.clear_cache()


def test_reports_reachable_when_the_probe_answers_200(monkeypatch):
    calls: list[str] = []

    def fake_get(url, timeout=None):
        calls.append(url)
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    assert r.ollama_reachable("http://127.0.0.1:11434") is True
    assert calls == ["http://127.0.0.1:11434/api/tags"]


def test_reports_unreachable_on_a_non_200(monkeypatch):
    monkeypatch.setattr(
        httpx, "get",
        lambda url, timeout=None: httpx.Response(500, request=httpx.Request("GET", url)),
    )
    assert r.ollama_reachable("http://127.0.0.1:11434") is False


def test_reports_unreachable_when_nothing_is_listening(monkeypatch):
    def fake_get(url, timeout=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", fake_get)
    assert r.ollama_reachable("http://127.0.0.1:11434") is False


def test_reports_unreachable_on_a_slow_server_instead_of_blocking(monkeypatch):
    """A hung server times out as unreachable; it never raises or blocks the
    caller past the given timeout."""
    def fake_get(url, timeout=None):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx, "get", fake_get)
    assert r.ollama_reachable("http://127.0.0.1:11434", timeout=0.05) is False


def test_a_second_call_inside_the_ttl_never_probes_again(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, timeout=None):
        calls["n"] += 1
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    assert r.ollama_reachable("http://127.0.0.1:11434", cache_ttl=60.0) is True
    assert r.ollama_reachable("http://127.0.0.1:11434", cache_ttl=60.0) is True
    assert calls["n"] == 1, "the cached answer must not trigger a second probe"


def test_a_call_after_the_ttl_expires_probes_again(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, timeout=None):
        calls["n"] += 1
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    assert r.ollama_reachable("http://127.0.0.1:11434", cache_ttl=0.05) is True
    time.sleep(0.1)
    assert r.ollama_reachable("http://127.0.0.1:11434", cache_ttl=0.05) is True
    assert calls["n"] == 2


def test_a_trailing_slash_on_the_host_does_not_double_up(monkeypatch):
    seen: list[str] = []

    def fake_get(url, timeout=None):
        seen.append(url)
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    r.ollama_reachable("http://127.0.0.1:11434/")
    assert seen == ["http://127.0.0.1:11434/api/tags"]


def test_two_different_hosts_are_cached_independently(monkeypatch):
    answers = {"http://a/api/tags": 200, "http://b/api/tags": 500}

    def fake_get(url, timeout=None):
        return httpx.Response(answers[url], request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    assert r.ollama_reachable("http://a") is True
    assert r.ollama_reachable("http://b") is False
