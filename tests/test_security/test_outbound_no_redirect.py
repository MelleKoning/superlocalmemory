# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Every exit in ``core/outbound_http`` refuses to follow a redirect.

The attack: a program squatting a port the hook posts to answers with a
redirect to another origin. Before 4.1.20 the stdlib ``urlopen`` path followed
it and re-sent ``X-SLM-Hook-Token`` to the second origin. Each test runs two
real HTTP servers on 127.0.0.1: the first redirects, the second records
whatever reaches it. The second must never be reached.
"""

from __future__ import annotations

import asyncio
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from superlocalmemory.core import outbound_http

_TOKEN = "hook-token-must-not-travel"


class _Recorder:
    def __init__(self) -> None:
        self.requests: list[dict[str, str]] = []


def _serve(handler_cls: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture()
def servers() -> Iterator[tuple[str, _Recorder]]:
    recorder = _Recorder()

    class Collector(BaseHTTPRequestHandler):
        def _record(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            recorder.requests.append({k.lower(): v for k, v in self.headers.items()})
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        do_GET = do_POST = _record

        def log_message(self, *args: object) -> None:  # quiet
            pass

    collector = _serve(Collector)
    target = f"http://localhost:{collector.server_address[1]}/stolen"

    class Squatter(BaseHTTPRequestHandler):
        def _redirect(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = do_POST = _redirect

        def log_message(self, *args: object) -> None:
            pass

    squatter = _serve(Squatter)
    try:
        yield f"http://127.0.0.1:{squatter.server_address[1]}/internal/prewarm", recorder
    finally:
        squatter.shutdown()
        collector.shutdown()


def test_gated_urlopen_does_not_follow_a_redirect_to_another_origin(servers) -> None:
    url, recorder = servers
    req = urllib.request.Request(
        url, data=b'{"tool":"x"}', method="POST",
        headers={"Content-Type": "application/json", "X-SLM-Hook-Token": _TOKEN},
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        outbound_http.urlopen(req, timeout=5)
    assert caught.value.code == 302
    assert recorder.requests == [], "the redirect target received a request"


def test_gated_urlopen_refuses_redirects_even_when_a_proxy_is_configured(
    servers, monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, recorder = servers
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")  # dead; loopback bypasses it
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    req = urllib.request.Request(url, headers={"X-SLM-Hook-Token": _TOKEN})
    with pytest.raises(urllib.error.HTTPError) as caught:
        outbound_http.urlopen(req, timeout=5)
    assert caught.value.code == 302
    assert recorder.requests == []


def test_gated_urlopen_still_reaches_loopback_directly_with_a_proxy_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits: list[str] = []

    class Ok(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args: object) -> None:
            pass

    server = _serve(Ok)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/ok", data=b"{}", method="POST",
        )
        with outbound_http.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
    finally:
        server.shutdown()
    assert hits == ["/ok"]


def test_post_json_returns_the_redirect_instead_of_following_it(servers) -> None:
    url, recorder = servers
    resp = outbound_http.post_json(url, {"q": 1}, headers={"X-SLM-Hook-Token": _TOKEN},
                                   timeout=5)
    assert resp.status_code == 302
    assert recorder.requests == []


def test_gated_client_returns_the_redirect_instead_of_following_it(servers) -> None:
    url, recorder = servers
    client = outbound_http.GatedClient(timeout=5)
    try:
        resp = client.post(url, json={"q": 1}, headers={"X-SLM-Hook-Token": _TOKEN})
        assert resp.status_code == 302
        with client.stream("POST", url, json={"q": 1},
                           headers={"X-SLM-Hook-Token": _TOKEN}) as streamed:
            assert streamed.status_code == 302
    finally:
        client.close()
    assert recorder.requests == []


def test_async_stream_returns_the_redirect_instead_of_following_it(servers) -> None:
    url, recorder = servers

    async def _run() -> int:
        async with outbound_http.astream_json(
            "POST", url, {"q": 1}, headers={"X-SLM-Hook-Token": _TOKEN}, timeout=5,
        ) as resp:
            return resp.status_code

    assert asyncio.run(_run()) == 302
    assert recorder.requests == []
