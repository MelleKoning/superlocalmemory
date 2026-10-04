# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The one door memory text leaves this machine through: ``core.outbound_http``.

Credentials stay in SLM. Every HTTP body that can carry memory text is screened
here, decided on the URL it is actually sent to: another machine (a cloud
provider, a LAN Ollama) never receives a credential; this machine sees the text
as written and is reached directly, never through an environment proxy.

No test here touches a non-loopback host: remote URLs go to an in-process
``httpx.MockTransport`` or a patched client, and the proxy tests use a fake
proxy on 127.0.0.1.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import urllib.request

import httpx
import pytest

from superlocalmemory.core import outbound_http
from superlocalmemory.core.outbound_redaction import is_local_endpoint

_KEY = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0"  # fake, test-only
_TEXT = f"Prod OpenAI key is {_KEY} and the db password=Hunt3rTwo!x9"
_REMOTE = "https://llm.example.com/v1/chat/completions"
_LAN = "http://192.168.1.50:11434/api/embed"
_LOCAL = "http://127.0.0.1:11434/api/embed"


def _payload() -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": _TEXT}],
            "input": [_TEXT, "plain"], "n": 3, "stream": False}


# -- the decision ----------------------------------------------------------


@pytest.mark.parametrize("url", [_REMOTE, _LAN])
def test_a_body_for_another_machine_loses_every_credential(url: str) -> None:
    out = outbound_http.outbound_json(_payload(), url)
    blob = json.dumps(out)
    assert _KEY not in blob and "Hunt3rTwo" not in blob
    assert out["model"] == "m" and out["n"] == 3 and out["stream"] is False
    assert out["input"][1] == "plain"
    assert "Prod OpenAI key is" in out["messages"][0]["content"]


def test_a_body_for_this_machine_is_sent_as_written() -> None:
    assert outbound_http.outbound_json(_payload(), _LOCAL) == _payload()


def test_the_callers_payload_is_never_mutated() -> None:
    original = _payload()
    outbound_http.outbound_json(original, _REMOTE)
    assert original == _payload()


def test_raw_json_bytes_for_another_machine_are_screened() -> None:
    raw = json.dumps(_payload()).encode()
    out = outbound_http.outbound_content(raw, _LAN)
    assert _KEY.encode() not in out
    assert json.loads(out)["model"] == "m"


def test_bytes_that_need_no_change_are_sent_byte_for_byte() -> None:
    raw = b'{"a":"nothing secret here","b":[1,2]}'
    assert outbound_http.outbound_content(raw, _REMOTE) is raw


def test_a_body_it_cannot_read_is_refused_for_another_machine() -> None:
    with pytest.raises(outbound_http.OutboundBlocked):
        outbound_http.outbound_content(b"password=Hunt3rTwo!x9", _REMOTE)


def test_a_body_it_cannot_read_still_goes_to_this_machine() -> None:
    raw = b"password=Hunt3rTwo!x9"
    assert outbound_http.outbound_content(raw, _LOCAL) is raw


# -- the httpx paths ---------------------------------------------------------


def test_post_json_screens_the_body_and_never_follows_a_redirect(monkeypatch) -> None:
    seen: dict = {}

    def _fake_post(url, **kw):
        seen.update(kw, url=url)
        return httpx.Response(200, json={}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", _fake_post)
    outbound_http.post_json(_LAN, _payload(), headers={"X": "1"}, timeout=5.0)
    assert _KEY not in json.dumps(seen["json"])
    assert seen["follow_redirects"] is False
    assert seen["trust_env"] is True  # another machine: the user's proxy applies
    assert seen["headers"] == {"X": "1"}


def test_post_json_to_this_machine_ignores_environment_proxies(monkeypatch) -> None:
    seen: dict = {}

    def _fake_post(url, **kw):
        seen.update(kw)
        return httpx.Response(200, json={}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", _fake_post)
    outbound_http.post_json(_LOCAL, _payload(), timeout=5.0)
    assert seen["trust_env"] is False
    assert _KEY in json.dumps(seen["json"])


def _capturing_transport(bodies: list[bytes]) -> httpx.MockTransport:
    def _handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return httpx.Response(200, json={"ok": True})
    return httpx.MockTransport(_handler)


def test_the_gated_client_screens_post_and_stream_bodies() -> None:
    bodies: list[bytes] = []
    client = outbound_http.GatedClient(transport=_capturing_transport(bodies))
    try:
        client.post(_REMOTE, json=_payload())
        with client.stream("POST", _REMOTE, json=_payload()) as resp:
            resp.read()
        with client.stream("POST", _REMOTE,
                           content=json.dumps(_payload()).encode()) as resp:
            resp.read()
    finally:
        client.close()
    assert len(bodies) == 3
    for body in bodies:
        assert _KEY.encode() not in body


def test_a_closed_gated_client_refuses_to_send() -> None:
    client = outbound_http.GatedClient(transport=_capturing_transport([]))
    client.close()
    with pytest.raises(RuntimeError):
        client.post(_REMOTE, json={"a": "b"})


def test_the_async_stream_screens_the_body() -> None:
    bodies: list[bytes] = []

    async def _run() -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            bodies.append(request.content)
            return httpx.Response(200, text="data: [DONE]\n")

        async with outbound_http.astream_json(
            "POST", _REMOTE, _payload(), transport=httpx.MockTransport(_handler),
        ) as resp:
            await resp.aread()

    asyncio.run(_run())
    assert bodies and _KEY.encode() not in bodies[0]


# -- the urllib path -----------------------------------------------------------


def test_urlopen_screens_a_json_body_for_another_machine(monkeypatch) -> None:
    seen: list[urllib.request.Request] = []

    def _fake_urlopen(req, timeout=None):
        seen.append(req)
        raise OSError("stop")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open",
                        lambda self, req, *a, **kw: _fake_urlopen(req))
    req = urllib.request.Request(_LAN, data=json.dumps(_payload()).encode(),
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(OSError):
        outbound_http.urlopen(req, timeout=5)
    assert seen and _KEY.encode() not in seen[0].data
    assert seen[0].get_header("Content-type") == "application/json"
    assert req.data == json.dumps(_payload()).encode()  # caller's request untouched


# -- L2-10: this machine is never reached through an environment proxy ----------


class _FakeProxy:
    """A one-shot 'proxy' on 127.0.0.1 that records whether anything reached it."""

    def __init__(self) -> None:
        self.saw: list[bytes] = []
        self._srv = socket.socket()
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(1)
        self._srv.settimeout(1)
        self.url = f"http://127.0.0.1:{self._srv.getsockname()[1]}"
        self._t = threading.Thread(target=self._accept, daemon=True)
        self._t.start()

    def _accept(self) -> None:
        try:
            conn, _ = self._srv.accept()
        except OSError:
            return
        self.saw.append(conn.recv(65536))
        conn.sendall(b"HTTP/1.1 502 Bad\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        conn.close()

    def close(self) -> None:
        self._srv.close()
        self._t.join(4)


@pytest.fixture()
def env_proxy(monkeypatch):
    proxy = _FakeProxy()
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
                 "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, proxy.url)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    yield proxy
    proxy.close()


def _closed_loopback_url() -> str:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here: a direct connection is refused at once
    return f"http://127.0.0.1:{port}/api/chat"


def test_httpx_post_to_this_machine_bypasses_the_proxy(env_proxy) -> None:
    with pytest.raises(httpx.HTTPError):
        outbound_http.post_json(_closed_loopback_url(), {"content": _TEXT}, timeout=2.0)
    env_proxy.close()
    assert env_proxy.saw == []


def test_gated_client_to_this_machine_bypasses_the_proxy(env_proxy) -> None:
    client = outbound_http.GatedClient(timeout=2.0)
    try:
        with pytest.raises(httpx.HTTPError):
            client.post(_closed_loopback_url(), json={"content": _TEXT})
    finally:
        client.close()
    env_proxy.close()
    assert env_proxy.saw == []


def test_urlopen_to_this_machine_bypasses_the_proxy(env_proxy) -> None:
    req = urllib.request.Request(_closed_loopback_url(),
                                 data=json.dumps({"content": _TEXT}).encode(),
                                 method="POST")
    with pytest.raises(OSError):
        outbound_http.urlopen(req, timeout=2)
    env_proxy.close()
    assert env_proxy.saw == []


# -- L2-14: a *.localhost name is another machine until proven otherwise --------


@pytest.mark.parametrize("url", [
    "http://my-box.localhost:1234/v1",
    "http://evil.localhost/v1",
    "http://a.b.localhost./v1",
])
def test_a_name_under_localhost_is_treated_as_another_machine(url: str) -> None:
    assert not is_local_endpoint(url)
    assert _KEY not in json.dumps(outbound_http.outbound_json(_payload(), url))


@pytest.mark.parametrize("url", [
    "http://localhost:11434/api/chat", "http://LOCALHOST./v1",
    "http://127.0.0.1:8080/v1", "http://127.9.9.9/v1", "http://[::1]:9000/v1",
])
def test_loopback_itself_is_still_this_machine(url: str) -> None:
    assert is_local_endpoint(url)
