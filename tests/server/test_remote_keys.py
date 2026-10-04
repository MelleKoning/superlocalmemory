# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Named remote keys: hashed at rest, revocable, fail closed on a tampered store."""

from __future__ import annotations

import hmac
import json
import os
import stat

import pytest

from superlocalmemory.server import remote_keys
from superlocalmemory.server.remote_keys import RemoteKeyError, RemoteKeyStore


@pytest.fixture()
def store(tmp_path):
    return RemoteKeyStore(tmp_path / "remote_keys.json")


def test_add_verify_revoke_list_round_trip(store) -> None:
    record, secret = store.add("hermes-laptop", "write")
    assert secret.startswith("slmr_") and len(secret) == 5 + 43
    assert store.verify(secret) == record
    assert store.verify(secret[:-1] + ("A" if secret[-1] != "A" else "B")) is None
    revoked = store.revoke("hermes-laptop")
    assert revoked.revoked_at is not None
    assert store.verify(secret) is None
    listed = store.list()
    assert [k.name for k in listed] == ["hermes-laptop"] and not listed[0].active


def test_only_a_domain_separated_digest_is_stored(store) -> None:
    import hashlib

    _, secret = store.add("a", "read")
    raw = store.path.read_text()
    assert secret not in raw
    assert hashlib.sha256(secret.encode()).hexdigest() not in raw
    assert remote_keys.digest_secret(secret) in raw


def test_revoke_by_key_id(store) -> None:
    record, secret = store.add("a", "read")
    store.revoke(record.key_id)
    assert store.verify(secret) is None


def test_duplicate_active_name_refused_but_reusable_after_revoke(store) -> None:
    store.add("a", "read")
    with pytest.raises(RemoteKeyError) as err:
        store.add("a", "write")
    assert err.value.code == "duplicate_name"
    store.revoke("a")
    store.add("a", "write")


@pytest.mark.parametrize("name", ["", "A", "-x", "a b", "a/b", "x" * 49, "név"])
def test_invalid_names_refused(store, name) -> None:
    with pytest.raises(RemoteKeyError):
        store.add(name, "read")


def test_invalid_scope_refused(store) -> None:
    with pytest.raises(RemoteKeyError):
        store.add("a", "admin")


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_atomic_write_leaves_0600(store) -> None:
    store.add("a", "read")
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert not [p for p in store.path.parent.iterdir() if p.name.endswith(".tmp")]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
@pytest.mark.parametrize("mode", [0o620, 0o602, 0o640, 0o604])
def test_group_or_world_accessible_store_fails_closed(store, mode, caplog) -> None:
    _, secret = store.add("a", "write")
    os.chmod(store.path, mode)
    assert store.verify(secret) is None
    assert "Remote keys are disabled" in caplog.text
    with pytest.raises(RemoteKeyError) as err:
        store.add("b", "read")
    assert err.value.code == "store_untrusted"


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership")
def test_wrong_owner_fails_closed(store, monkeypatch) -> None:
    _, secret = store.add("a", "write")
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
    assert store.verify(secret) is None


def test_unknown_version_or_corrupt_store_fails_closed(store) -> None:
    _, secret = store.add("a", "write")
    data = json.loads(store.path.read_text())
    data["version"] = 99
    store.path.write_text(json.dumps(data))
    os.chmod(store.path, 0o600)
    assert store.verify(secret) is None
    store.path.write_text("{not json")
    assert store.verify(secret) is None


def test_verify_checks_every_record(store, monkeypatch) -> None:
    secrets = [store.add(f"k{i}", "read")[1] for i in range(5)]
    calls = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append(1)
        return real(a, b)

    monkeypatch.setattr(remote_keys.hmac, "compare_digest", spy)
    assert store.verify(secrets[0]) is not None
    assert len(calls) == 5


@pytest.mark.parametrize("presented", ["", "slmr_", "Bearer x", "slmr_" + "A" * 42,
                                       "slmr_" + "A" * 44, "xxxxx" + "A" * 43, None, 7])
def test_malformed_presented_keys_are_refused(store, presented) -> None:
    store.add("a", "write")
    assert store.verify(presented) is None


def test_revocation_takes_effect_without_a_new_store_object(store) -> None:
    """Another process (the CLI) revokes; this store object sees it on the next call."""
    _, secret = store.add("a", "write")
    assert store.verify(secret) is not None
    RemoteKeyStore(store.path).revoke("a")
    assert store.verify(secret) is None
