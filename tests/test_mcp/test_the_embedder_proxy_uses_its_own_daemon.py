# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""An MCP engine embeds through its own data root's daemon only (C-8).

The LIGHT (MCP) engine attached an embedder proxy to whatever answered on the
configured daemon port. With two installations on one machine -- a second
data root, a test run, a dev checkout -- an MCP process for one root sent its
memories to the other root's daemon and stored the vectors that came back,
possibly from a different model. The proxy must find the daemon that owns its
own root (its descriptor), prove it with ``/health``, and present that
daemon's capability on every request.

A small local HTTP server stands in for "a daemon"; nothing touches 8765.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from superlocalmemory.core.mcp_embedder_proxy import McpEmbedderProxy
from superlocalmemory.infra.daemon_identity import build_descriptor, write_descriptor


def _lower(headers) -> dict:
    return {k.lower(): v for k, v in headers.items()}


class _FakeDaemon:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict]] = []
        self.health: dict = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a) -> None:  # quiet
                pass

            def _reply(self, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                outer.requests.append(("GET", self.path, _lower(self.headers)))
                self._reply(outer.health if self.path == "/health" else {"ok": True})

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(("POST", self.path, _lower(self.headers)))
                self._reply({"embeddings": [[0.25] * 4 for _ in body.get("texts", [])]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, name="fake-daemon", daemon=True,
        )
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture()
def daemon():
    d = _FakeDaemon()
    yield d
    d.close()


def test_a_daemon_this_root_does_not_own_is_never_used(daemon, mode_a_config) -> None:
    """The MCP engine's own attach path, with the configured port occupied."""
    from superlocalmemory.core.engine import MemoryEngine

    mode_a_config.daemon_port = daemon.port  # "whatever is on 8765"
    engine = MemoryEngine(mode_a_config)
    engine._try_init_proxy()

    assert not isinstance(engine._embedder, McpEmbedderProxy), (
        "the MCP engine attached to a daemon that does not own its data root"
    )
    assert McpEmbedderProxy().embed("a private memory") is None
    assert not any(path.startswith("/api/v3/embed") for _, path, _ in daemon.requests), (
        "a memory was sent to a daemon that does not own this data root"
    )


def test_the_owned_daemon_is_used_with_its_capability(daemon) -> None:
    root = os.environ["SLM_DATA_DIR"]
    descriptor = build_descriptor(
        data_root=root, port=daemon.port, version="test", state="ready",
    )
    write_descriptor(descriptor, data_root=root)
    daemon.health = descriptor.public_health_fields()

    proxy = McpEmbedderProxy()
    assert proxy.is_available() is True
    assert proxy.embed_batch(["a", "b"]) == [[0.25] * 4, [0.25] * 4]

    posts = [h for method, path, h in daemon.requests if method == "POST"]
    assert posts, "the owned daemon was not asked"
    assert posts[0].get("x-slm-daemon-capability") == descriptor.capability
    assert posts[0].get("x-slm-target-instance") == descriptor.instance_id


def test_a_descriptor_whose_health_does_not_match_is_refused(daemon) -> None:
    root = os.environ["SLM_DATA_DIR"]
    descriptor = build_descriptor(
        data_root=root, port=daemon.port, version="test", state="ready",
    )
    write_descriptor(descriptor, data_root=root)
    daemon.health = {**descriptor.public_health_fields(), "instance_id": "someone-else"}

    assert McpEmbedderProxy().is_available() is False
    assert not any(path.startswith("/api/v3/embed") for _, path, _ in daemon.requests)
