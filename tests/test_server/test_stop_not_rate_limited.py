# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Regression test for Q10 (2026-10-06): ``POST /stop`` must never be
rate-limited, against the REAL production app (``unified_daemon.create_app``).

Reproduced live during 4.1.22 seeding (see
``.backup/4.1.22/g02-final-measurements.md`` §1b): a burst of writes
(``seed_bulk.py``) exhausted the loopback write budget (300 requests / 60s,
``RateLimiter`` via ``_lb_write_limiter``). The very next ``slm serve stop``
sent ``POST /stop`` over loopback -- and ``rate_limit_middleware`` counted it
as just another write (``is_write = request.method in (POST, PUT, DELETE,
PATCH)`` has no path exception), so it ALSO got HTTP 429. The CLI's
``daemon_request()`` has no special handling for 429 (``cli/daemon.py``'s
``except urllib.error.HTTPError`` branch only special-cases 401/403/409/404/
422; 429 falls through to ``return None``, indistinguishable from "no
daemon"), so ``stop_daemon()`` reported failure even though ``lsof`` proved
the daemon was still listening on the port -- the exact "Daemon was not
running" false report this fixes.

``/stop`` is authenticated by a private per-instance capability header
(``_require_daemon_actor`` -> ``write_identity.require_daemon_actor``), not
by request volume, so exempting it from the rate limiter loses no security
guarantee: the install's own except-clause for a missing rate limiter notes
"rate limiting is not a security boundary" (unlike auth, which fails
closed).
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from superlocalmemory.infra import rate_limiter as rl
from superlocalmemory.server.unified_daemon import create_app

_LOOPBACK_PEER = ("127.0.0.1", 50000)


@pytest.fixture()
def app():
    rl.reset_managed()
    application = create_app()
    yield application
    rl.reset_managed()


def _client(app) -> TestClient:
    return TestClient(
        app, client=_LOOPBACK_PEER,
        base_url="http://127.0.0.1:8765",
        raise_server_exceptions=False,
    )


def _shrink_loopback_write_budget(max_requests: int) -> None:
    """Reconfigure the live ``lb_write`` limiter the middleware consults.

    The public ``set_limits()`` floors the loopback write budget at 300
    (``_loopback_write``: ``max(300, write * 10)``) by design, so a runaway
    local agent is still eventually throttled -- that floor cannot be
    configured below 300 through the public API, and 300+ real HTTP round
    trips would make this test slow. Reconfiguring the registered limiter
    object directly reproduces "budget exhausted" deterministically.
    """
    shrunk = False
    for role, limiter in list(rl._MANAGED):
        if role == "lb_write":
            limiter.configure(max_requests=max_requests, window_seconds=60)
            shrunk = True
    assert shrunk, "lb_write limiter was not registered by create_app()"


class TestStopExemptFromRateLimit:
    def test_stop_not_blocked_after_loopback_write_budget_is_exhausted(
        self, app,
    ) -> None:
        _shrink_loopback_write_budget(1)
        client = _client(app)

        # Exhaust the (shrunk) loopback write budget on an ordinary write path.
        first = client.post("/remember", json={"content": "x"})
        second = client.post("/remember", json={"content": "y"})
        assert 429 in (first.status_code, second.status_code), (
            "setup failed to exhaust the loopback write budget: "
            f"{first.status_code} {first.text[:200]}, "
            f"{second.status_code} {second.text[:200]}"
        )

        # /stop must never be answered with 429, regardless of the exhausted
        # write budget. No daemon capability header is sent, so the request
        # still fails -- but on the route's own identity check (403, since
        # application.state.daemon_descriptor is unset in this harness),
        # never on rate limiting. That distinction is exactly what this
        # test proves.
        resp = client.post("/stop")
        assert resp.status_code != 429, (
            "POST /stop was rate-limited; it must be exempt from the write "
            f"limiter (got {resp.status_code}: {resp.text[:200]})"
        )

    def test_health_is_still_rate_limited_as_a_control(self, app) -> None:
        """Anti-tautology: the limiter must still fire for an ordinary path,
        so a middleware removed entirely (rather than given a path
        exception) cannot make the test above pass for the wrong reason."""
        _shrink_loopback_write_budget(1)
        client = _client(app)

        first = client.post("/remember", json={"content": "x"})
        second = client.post("/remember", json={"content": "y"})
        assert 429 in (first.status_code, second.status_code), (
            f"{first.status_code}, {second.status_code}"
        )
