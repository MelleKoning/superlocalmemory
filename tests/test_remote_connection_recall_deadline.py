# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The laptop tells the daemon how long a relayed call may still take.

Every relayed request gets ``x-slm-deadline-ms``: the moment the relay gives up
(never more than RELAY_DEADLINE_MS from now) minus a margin for the keyword
fallback, serialisation and the trip back. A caller cannot set it itself.
"""
import base64

import pytest

from superlocalmemory.remote_connections import session
from superlocalmemory.remote_connections.codec import FrameError
from superlocalmemory.remote_connections.credentials import ConnectorCredential
from superlocalmemory.remote_connections.origin import CanonicalMcpOrigin, relay_request_headers

NOW_S = 1_760_000_000.0
NOW_MS = int(NOW_S * 1000)


def credential():
    return ConnectorCredential("install-a", "owner-a", "default", "a" * 32, 1,
                               NOW_MS + 600_000, "a" * 64, "slmr_" + "b" * 43)


def frame(remaining_ms, extra_headers=()):
    return {"v": 1, "kind": "request", "id": "x", "generation": 1,
            "deadlineAt": NOW_MS + remaining_ms,
            "headers": [["content-type", "application/json"], *map(list, extra_headers)],
            "bodyBase64": base64.b64encode(
                b'{"jsonrpc":"2.0","id":1,"method":"tools/list"}').decode()}


async def observe(remaining_ms, extra_headers=()):
    seen = []

    async def app(scope, receive, send):
        seen.append(dict(scope["headers"]))
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    origin = CanonicalMcpOrigin(app, clock=lambda: NOW_S)
    await origin(frame(remaining_ms, extra_headers), credential())
    return seen[0]


def test_the_margin_is_named_next_to_the_relay_deadline():
    assert session.RECALL_DEADLINE_MARGIN_MS == 4000
    assert session.RELAY_DEADLINE_MS == 25000


@pytest.mark.parametrize("remaining, expected_ms", [
    (25_000, NOW_MS + 21_000),
    (12_000, NOW_MS + 8_000),
    (4_000, NOW_MS),
    (1_000, NOW_MS - 3_000),
    # Beyond the budget (a relay clock ahead of this one): never longer than 25 s.
    (29_000, NOW_MS + 21_000),
])
def test_the_deadline_is_the_relay_deadline_minus_the_margin(remaining, expected_ms):
    assert session.recall_deadline_ms(NOW_MS + remaining, NOW_MS) == expected_ms


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining, expected_ms",
                         [(25_000, NOW_MS + 21_000), (9_000, NOW_MS + 5_000)])
async def test_origin_stamps_every_request_with_the_deadline(remaining, expected_ms):
    headers = await observe(remaining)
    assert headers[b"x-slm-deadline-ms"] == str(expected_ms).encode()


@pytest.mark.asyncio
async def test_origin_never_stamps_a_deadline_beyond_the_relay_bound():
    headers = await observe(29_000)
    bound = NOW_MS + session.RELAY_DEADLINE_MS - session.RECALL_DEADLINE_MARGIN_MS
    assert int(headers[b"x-slm-deadline-ms"]) <= bound


def test_a_value_the_caller_supplied_is_replaced_whatever_its_case():
    pairs = [["Content-Type", "application/json"],
             ["X-SLM-Deadline-MS", "9999999999999"], ["x-slm-deadline-ms", "1"]]
    out = relay_request_headers(pairs, deadline_at_ms=NOW_MS + 12_000, now_ms=NOW_MS)
    stamped = [(k, v) for k, v in out.items() if k.lower() == "x-slm-deadline-ms"]
    assert stamped == [("x-slm-deadline-ms", str(NOW_MS + 8_000))]
    assert out["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_a_relayed_frame_cannot_carry_the_header_at_all():
    # The wire format does not allow it, so it is refused before any app runs.
    with pytest.raises(FrameError):
        await observe(10_000, [["x-slm-deadline-ms", "9999999999999"]])
