# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The SLM API key never travels to a second origin through a redirect.

``X-SLM-API-Key`` is a custom header: HTTP clients that follow redirects
forward it, unlike ``Authorization``. So:

1. No product client sends it except through the gated exit, which refuses
   every redirect (two real servers on 127.0.0.1 prove the second origin -
   other port, other scheme - is never reached).
2. Every file that names the header is a reviewed server-side reader; a new
   client that sends it fails this test until its redirect behaviour is reviewed.
3. The daemon never answers ``/mcp`` with a redirect, and accepts the API key
   as ``Authorization: Bearer`` so MCP clients can use the header they strip.
"""

from __future__ import annotations

import re
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from superlocalmemory.core import outbound_http

REPO = Path(__file__).resolve().parents[2]
_KEY = "api-key-must-not-travel"

#: Files that READ the header on the server side (reviewed), plus the one
#: reviewed client, listed last.
_REVIEWED_READERS = {
    "src/superlocalmemory/infra/auth_middleware.py",
    "src/superlocalmemory/server/access_gate.py",
    "src/superlocalmemory/server/api.py",
    "src/superlocalmemory/server/remote_access.py",
    "src/superlocalmemory/server/ui.py",
    "src/superlocalmemory/server/unified_daemon.py",
    "src/superlocalmemory/server/write_identity.py",
    # The one reviewed CLIENT. It forwards a relayed MCP request to this machine's
    # own daemon and may attach the key (supplied by the local installation, never
    # by the relay) to that request. It accepts only http://127.0.0.1:<port>/mcp
    # as origin (anything else throws invalid_local_origin), sends with
    # `redirect: 'error'` so any 3xx fails the request instead of being followed,
    # and a frame's own headers cannot override the key.
    # integrations/remote-gateway/tests/local-connector.test.mjs drives real
    # 301/302/303/307/308 responses and proves the redirect target is never contacted.
    "integrations/remote-gateway/src/local-connector.ts",
    # That client's own test: it names the header only to prove the key is never
    # sent anywhere but the loopback origin and never follows a redirect.
    "integrations/remote-gateway/tests/local-connector.test.mjs",
}
_CLIENT_ROOTS = ("src", "plugin-src", "hermes-plugin", "integrations", "ide", "scripts",
                 "bin", "npm", "packages")


def _serve(handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture()
def collector() -> Iterator[tuple[int, list[dict]]]:
    seen: list[dict] = []

    class Collector(BaseHTTPRequestHandler):
        def _record(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            seen.append({k.lower(): v for k, v in self.headers.items()})
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        do_GET = do_POST = _record

        def log_message(self, *args: object) -> None:
            pass

    server = _serve(Collector)
    try:
        yield server.server_address[1], seen
    finally:
        server.shutdown()


def _squatter(status: int, location: str) -> ThreadingHTTPServer:
    class Squatter(BaseHTTPRequestHandler):
        def _redirect(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            self.send_response(status)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = do_POST = _redirect

        def log_message(self, *args: object) -> None:
            pass

    return _serve(Squatter)


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("target", ["other-port", "other-host", "other-scheme"])
def test_gated_exit_never_forwards_the_api_key_through_a_redirect(collector, status,
                                                                   target, method) -> None:
    port, seen = collector
    location = {"other-port": f"http://127.0.0.1:{port}/stolen",
                "other-host": f"http://localhost:{port}/stolen",
                "other-scheme": f"https://127.0.0.1:{port}/stolen"}[target]
    squatter = _squatter(status, location)
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{squatter.server_address[1]}/mcp/hermes",
            data=b"{}" if method == "POST" else None, method=method,
            headers={"X-SLM-API-Key": _KEY, "Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as err:
            outbound_http.urlopen(request, timeout=5)
        assert err.value.code == status
    finally:
        squatter.shutdown()
    assert seen == [], "the redirect target was contacted"


def test_every_file_naming_the_header_is_a_reviewed_server_reader() -> None:
    pattern = re.compile(r"x-slm-api-key", re.IGNORECASE)
    found = set()
    for root in _CLIENT_ROOTS:
        base = REPO / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".js", ".mjs", ".ts", ".sh",
                                                  ".ps1", ".json", ".yaml", ".yml"}:
                if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                    found.add(path.relative_to(REPO).as_posix())
    assert found - _REVIEWED_READERS == set(), (
        "A new file names X-SLM-API-Key. If it SENDS the header, it must go through "
        "core.outbound_http (no redirects) or send Authorization: Bearer instead; then "
        f"add it to the reviewed list: {sorted(found - _REVIEWED_READERS)}")


def test_no_doc_example_follows_redirects_with_the_api_key() -> None:
    for path in [*REPO.glob("*.md"), *(REPO / "docs").rglob("*.md")]:
        for block in re.findall(r"```.*?```", path.read_text(encoding="utf-8"), re.S):
            if re.search(r"x-slm-api-key", block, re.I):
                assert not re.search(r"(?:^|\s)(?:-L|--location)\b|follow_redirects=True",
                                     block), path


def test_the_api_key_works_as_a_bearer_token_for_remote_mcp() -> None:
    from superlocalmemory.infra.auth_middleware import API_KEY_FILE
    from superlocalmemory.server.remote_access import authenticate_remote

    Path(API_KEY_FILE).write_text(_KEY + "\n", encoding="utf-8")
    principal = authenticate_remote({"authorization": f"Bearer {_KEY}"})
    assert principal is not None and principal.kind == "legacy-api-key"
    assert principal.scope == "write"
    assert authenticate_remote({"authorization": "Bearer wrong"}) is None
    assert authenticate_remote({"x-slm-api-key": _KEY}) is not None
