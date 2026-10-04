# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The remote TLS private key and the remote key store are owner-only on every
platform (audit 4.1.20 L12).

``os.open(..., 0o600)`` is the whole protection on macOS and Linux, and no
protection at all on Windows, where a new file inherits its folder's access
list. Both writers now apply an owner-only access list to the new file before
a single secret byte is written into it.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from superlocalmemory.cli import remote_commands
from superlocalmemory.infra import owner_only_acl
from superlocalmemory.server.remote_keys import RemoteKeyStore


def _record_calls(monkeypatch):
    calls: list[tuple[Path, int]] = []
    real = owner_only_acl.restrict_to_owner

    def spy(path):
        calls.append((Path(path), Path(path).stat().st_size))
        real(path)

    monkeypatch.setattr(owner_only_acl, "restrict_to_owner", spy)
    return calls


def test_key_store_is_restricted_before_the_secret_is_written(tmp_path, monkeypatch) -> None:
    calls = _record_calls(monkeypatch)
    store_path = tmp_path / "remote_keys.json"
    RemoteKeyStore(store_path).add("hermes-laptop", "read", profile="default")
    assert calls, "the key store writer never restricted its file"
    assert all(size == 0 for _p, size in calls), "restricted only after data was written"
    assert all(p.parent == tmp_path and p != store_path for p, _s in calls), \
        "the temporary file is restricted, so the final name never appears open"
    assert store_path.exists()


def test_tls_private_key_is_restricted_before_the_key_is_written(tmp_path, monkeypatch) -> None:
    calls = _record_calls(monkeypatch)
    key = tmp_path / "tls" / "server.key"
    remote_commands._write_private(key, b"-----BEGIN PRIVATE KEY-----\n")
    assert [size for _p, size in calls] == [0]
    assert key.read_bytes().startswith(b"-----BEGIN")


def test_a_failed_restriction_leaves_nothing_behind(tmp_path, monkeypatch) -> None:
    def refuse(_path):
        raise OSError("cannot restrict")

    monkeypatch.setattr(owner_only_acl, "restrict_to_owner", refuse)
    key = tmp_path / "server.key"
    with pytest.raises(OSError):
        remote_commands._write_private(key, b"secret")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_posix_files_are_0600(tmp_path) -> None:
    store_path = tmp_path / "remote_keys.json"
    RemoteKeyStore(store_path).add("hermes-laptop", "read", profile="default")
    key = tmp_path / "server.key"
    remote_commands._write_private(key, b"k")
    for path in (store_path, key):
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


@pytest.mark.skipif(sys.platform != "win32", reason="Windows access lists")
def test_windows_files_have_an_owner_only_protected_dacl(tmp_path) -> None:
    import ntsecuritycon
    import win32api
    import win32con
    import win32security

    from superlocalmemory.optimize.proxy.capture import (
        _windows_dacl_is_owner_only, _windows_owner_dacl,
    )

    store_path = tmp_path / "remote_keys.json"
    RemoteKeyStore(store_path).add("hermes-laptop", "read", profile="default")
    key = tmp_path / "tls" / "server.key"
    remote_commands._write_private(key, b"k")
    owner_sid, _dacl = _windows_owner_dacl(win32api, win32con, win32security)
    for path in (store_path, key):
        descriptor = win32security.GetNamedSecurityInfo(
            str(path), win32security.SE_FILE_OBJECT,
            win32security.OWNER_SECURITY_INFORMATION
            | win32security.DACL_SECURITY_INFORMATION)
        assert _windows_dacl_is_owner_only(descriptor, owner_sid, ntsecuritycon,
                                           win32security), path.name
