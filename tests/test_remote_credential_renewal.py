"""The laptop renews its gateway credential before it expires, recovers when the gateway
already holds a newer one, and never races its own reconnect."""
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from superlocalmemory.remote_connections.credentials import ConnectorCredential
from superlocalmemory.remote_connections.gateway_provider import CloudGatewayProvider
from superlocalmemory.remote_connections.native_enrollment import NativeEnrollmentStore
from superlocalmemory.remote_connections.runtime import (
    RENEWAL_CHECK_S,
    RENEWAL_WINDOW_MS,
    renewal_delay_s,
)
from tests.test_remote_connection_runtime import enrolled_runtime
from tests.test_remote_native_enrollment_store import Backend, record

DAY_MS = 24 * 3600 * 1000
ORIGIN_KEY = "slmr_" + "k" * 43


def _proof_target(proof):
    import base64
    import json
    payload = proof.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))["htu"]


def _delivery(row, generation, token="b" * 64, expires_ms=None):
    return {
        "device_token": token,
        "expires_at_ms": expires_ms or int(time.time() * 1000) + 30 * DAY_MS,
        "generation": generation,
        "connection_id": row.connection_id,
        "profile_id": row.profile,
    }


# --- gateway client -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_renewal_sends_the_held_generation_with_an_endpoint_bound_proof(tmp_path):
    requests = []
    row = replace(record(), access_token="owner-token", completed=True)

    async def http(path, **kwargs):
        requests.append((path, kwargs))
        return _delivery(row, 4)

    provider = CloudGatewayProvider(
        NativeEnrollmentStore(tmp_path, backend=Backend()), redirect_uri=row.redirect_uri, http=http
    )
    delivered = await provider.renew(row, 3)
    assert delivered["generation"] == 4
    ((path, kwargs),) = requests
    assert path == "/owner/renew"
    assert kwargs["json"] == {"expected_generation": 3}
    assert kwargs["headers"]["Authorization"] == "Bearer owner-token"
    assert _proof_target(kwargs["headers"]["DPoP"]).endswith("/owner/renew")


@pytest.mark.asyncio
async def test_a_malformed_renewal_delivery_is_refused(tmp_path):
    row = replace(record(), access_token="owner-token", completed=True)

    async def http(path, **kwargs):
        return {**_delivery(row, 4), "connection_id": "f" * 32}

    provider = CloudGatewayProvider(
        NativeEnrollmentStore(tmp_path, backend=Backend()), redirect_uri=row.redirect_uri, http=http
    )
    with pytest.raises(ValueError, match="invalid_device_delivery"):
        await provider.renew(row, 3)


def test_renewal_answers_are_told_apart_and_nothing_else_changes():
    assert CloudGatewayProvider._renewal_error("/owner/renew", 409) == "renewal_conflict"
    assert CloudGatewayProvider._renewal_error("/owner/renew", 403) == "connection_unavailable"
    assert CloudGatewayProvider._renewal_error("/owner/renew", 503) is None
    assert CloudGatewayProvider._renewal_error("/owner/connections", 409) is None


# --- when to renew ----------------------------------------------------------------------

def test_renewal_waits_are_short_and_measured_on_the_wall_clock():
    expires = 100 * DAY_MS
    assert RENEWAL_WINDOW_MS == 15 * DAY_MS
    assert renewal_delay_s(expires, 0) == RENEWAL_CHECK_S == 3600
    assert renewal_delay_s(expires, expires - RENEWAL_WINDOW_MS) == 0
    # A computer that slept through the due time renews as soon as it looks again.
    assert renewal_delay_s(expires, expires + 10 * DAY_MS) == 0
    assert 0 < renewal_delay_s(expires, expires - RENEWAL_WINDOW_MS - 60_000) <= 60


# --- renewing on the laptop ---------------------------------------------------------------

class Companion:
    def __init__(self):
        self.events = []
        self._running = True

    async def start(self):
        self.events.append("start")
        self._running = True

    async def stop(self):
        self.events.append("stop")
        self._running = False


def _renewing_runtime(tmp_path, *, generation=1, expires_ms=None):
    """A completed connection holding `generation`. Tests script the gateway per call."""
    runtime, row = enrolled_runtime(tmp_path)
    row = replace(row, completed=True, origin_key=ORIGIN_KEY)
    runtime.store.save(row)
    held = ConnectorCredential(
        row.installation_id, row.owner, row.profile, row.connection_id, generation,
        expires_ms or int(time.time() * 1000) + 10 * DAY_MS, "a" * 64, ORIGIN_KEY, row.private_key,
    )
    runtime.vault().save(held)
    companion = Companion()
    runtime._companions[row.connection_id] = companion
    calls = []

    async def exchange(latest, code):
        calls.append("refresh")
        return latest

    async def unexpected(latest, *args):
        raise AssertionError("gateway call not scripted by this test")

    runtime.provider = SimpleNamespace(exchange=exchange, renew=unexpected, provision=unexpected)
    return runtime, row, companion, calls


def _held(runtime, row):
    return runtime.vault().load(row.installation_id, row.owner, row.profile, row.connection_id)


@pytest.mark.asyncio
async def test_renewal_pauses_the_link_saves_the_new_credential_then_reconnects(tmp_path):
    runtime, row, companion, calls = _renewing_runtime(tmp_path)
    new = _delivery(row, 2, token="c" * 64)
    runtime.provider.renew = _recording(calls, "renew", new)
    assert await runtime.renew_credential(row) == "renewed"
    assert calls[0] == "refresh" and calls[1][0] == "renew" and calls[1][1] == 1
    assert companion.events == ["stop", "start"]
    held = _held(runtime, row)
    expected = (2, "c" * 64, new["expires_at_ms"])
    assert (held.generation, held.device_token, held.expires_at_ms) == expected
    assert held.origin_key == ORIGIN_KEY
    assert runtime.store.by_connection(row.connection_id).expires_at_ms == new["expires_at_ms"]


def _recording(calls, name, value):
    async def method(latest, *args):
        calls.append((name, *args))
        if isinstance(value, Exception):
            raise value
        return value
    return method


@pytest.mark.asyncio
async def test_when_another_renewal_already_won_the_laptop_takes_the_current_credential(tmp_path):
    runtime, row, companion, calls = _renewing_runtime(tmp_path)
    runtime.provider.renew = _recording(calls, "renew", ValueError("renewal_conflict"))
    current = _delivery(row, 3, token="d" * 64)
    runtime.provider.provision = _recording(calls, "provision", current)
    assert await runtime.renew_credential(row) == "renewed"
    assert _held(runtime, row).generation == 3
    assert companion.events == ["stop", "start"]


@pytest.mark.asyncio
async def test_a_renewal_that_is_not_due_yet_changes_nothing(tmp_path):
    runtime, row, companion, calls = _renewing_runtime(tmp_path)
    runtime.provider.renew = _recording(calls, "renew", ValueError("renewal_conflict"))
    runtime.provider.provision = _recording(calls, "provision", _delivery(row, 1, token="a" * 64))
    assert await runtime.renew_credential(row) == "not_due"
    held = _held(runtime, row)
    assert (held.generation, held.device_token) == (1, "a" * 64)
    assert companion.events == ["stop", "start"]


@pytest.mark.asyncio
async def test_an_outage_keeps_the_current_credential_and_reconnects(tmp_path):
    runtime, row, companion, calls = _renewing_runtime(tmp_path)
    runtime.provider.renew = _recording(calls, "renew", ValueError("remote_gateway_unavailable"))
    assert await runtime.renew_credential(row) == "unavailable"
    assert _held(runtime, row).generation == 1
    assert companion.events == ["stop", "start"]


@pytest.mark.asyncio
async def test_a_removed_connection_asks_the_owner_to_sign_in_again(tmp_path):
    runtime, row, companion, calls = _renewing_runtime(tmp_path)
    runtime.provider.renew = _recording(calls, "renew", ValueError("connection_unavailable"))
    assert await runtime.renew_credential(row) == "authorization_required"
    assert runtime._states[row.connection_id] == "authorization_required"
    assert companion.events == ["stop"]


@pytest.mark.asyncio
async def test_an_expired_credential_is_renewed_using_its_stored_generation(tmp_path):
    runtime, row, companion, calls = _renewing_runtime(tmp_path, generation=5)
    vault = runtime.vault()
    expired = replace(vault.load(row.installation_id, row.owner, row.profile, row.connection_id))
    assert expired.generation == 5
    later = SimpleNamespace(now=time.time() + 40 * 24 * 3600)
    vault._clock = lambda: later.now  # the stored credential has expired by now
    assert vault.load(row.installation_id, row.owner, row.profile, row.connection_id) is None
    assert vault.generation(row.installation_id, row.owner, row.profile, row.connection_id) == 5
    fresh = _delivery(row, 6, token="e" * 64, expires_ms=int(later.now * 1000) + 30 * DAY_MS)
    runtime.provider.renew = _recording(calls, "renew", fresh)
    runtime.vault = lambda: vault
    assert await runtime.renew_credential(row) == "renewed"
    assert ("renew", 5) in calls


@pytest.mark.asyncio
async def test_a_laptop_refused_by_the_gateway_recovers_a_newer_credential_once(tmp_path):
    runtime, row, companion, calls = _renewing_runtime(tmp_path)
    runtime.provider.renew = _recording(calls, "renew", ValueError("renewal_conflict"))
    runtime.provider.provision = _recording(calls, "provision", _delivery(row, 2, token="f" * 64))
    assert await runtime.recover(row) == "renewed"
    assert _held(runtime, row).generation == 2
    # A second refusal on the same generation does not loop.
    runtime.provider.provision = _recording(calls, "provision", _delivery(row, 2, token="f" * 64))
    assert await runtime.recover(row) == "not_due"
