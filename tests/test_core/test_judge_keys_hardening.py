# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The stored key is read exactly as checked, and only ever as a clean key.

* A key file edited by hand (``echo KEY > file`` leaves a line break) still
  works; anything that is not a usable key raises a clear key-store error that
  never quotes the key, instead of being sent as a broken header.
* The file that was checked is the file that is read: swapping in a symlink or
  another file between the check and the read is refused, not read through.
"""

from __future__ import annotations

import os
import stat
import sys

import pytest

from superlocalmemory.core import judge_keys
from superlocalmemory.core.judge_keys import JudgeKeyStore, JudgeKeyStoreError

VALID_KEY = "sk-live-" + "a1B2c3D4" * 4
POSIX = os.name == "posix"


@pytest.fixture
def store(tmp_path) -> JudgeKeyStore:
    return JudgeKeyStore(slm_home=tmp_path)


def _key_file(store: JudgeKeyStore, provider: str = "typesafe"):
    return store._key_path(provider)


# ---------------------------------------------------------------------------
# F2 — a hand-edited key file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("framing", ["{k}\n", "{k}\r\n", "  {k}  \n", "\n{k}\n\n", "\t{k}"])
def test_surrounding_whitespace_from_a_hand_edit_is_ignored(store, framing):
    store.set_key("typesafe", VALID_KEY)
    _key_file(store).write_text(framing.format(k=VALID_KEY), encoding="ascii")
    assert store.load("typesafe") == VALID_KEY
    assert store.masked("typesafe") == "****" + VALID_KEY[-4:]


@pytest.mark.parametrize("content", [
    "valid-key-12345678\n injected\n",
    "valid-key-12345678\r\nX-Injected: 1",
    "valid key 12345678",
    "short",
    "a" * 5000,
    "café-key-123456789",
], ids=["newline-then-injected", "crlf-header-injection", "inner-space", "too-short",
        "too-long", "non-ascii"])
def test_anything_else_is_a_clear_key_store_error_that_never_quotes_the_key(store, content):
    store.set_key("typesafe", VALID_KEY)
    _key_file(store).write_bytes(content.encode("utf-8"))
    with pytest.raises(JudgeKeyStoreError) as excinfo:
        store.load("typesafe")
    message = str(excinfo.value)
    assert "valid-key" not in message and "injected" not in message.lower()
    assert "Save it again" in message or "Save the key again" in message


def test_an_unusable_key_never_reaches_a_request_header(store):
    """The reproduced bytes: the header must never be built from them."""
    store.set_key("typesafe", VALID_KEY)
    _key_file(store).write_text("valid-key-12345678\n injected\n", encoding="ascii")
    with pytest.raises(JudgeKeyStoreError):
        store.load("typesafe")


def test_an_empty_key_file_reads_as_no_key(store):
    store.set_key("typesafe", VALID_KEY)
    _key_file(store).write_text("\n", encoding="ascii")
    assert store.load("typesafe") is None
    assert store.has_key("typesafe") is False


# ---------------------------------------------------------------------------
# F1 — the file checked is the file read
# ---------------------------------------------------------------------------

def _swap_after_check(monkeypatch, swap):
    real = judge_keys._refuse_unsafe

    def checked_then_swapped(path, *, expect):
        result = real(path, expect=expect)
        if expect == "file":
            swap(path)
        return result

    monkeypatch.setattr(judge_keys, "_refuse_unsafe", checked_then_swapped)


@pytest.mark.skipif(not POSIX, reason="symlinks and O_NOFOLLOW are POSIX here")
def test_a_symlink_swapped_in_after_the_check_is_never_read_through(store, tmp_path, monkeypatch):
    store.set_key("typesafe", VALID_KEY)
    decoy = tmp_path / "decoy.txt"
    decoy.write_text("attacker-key-00000000", encoding="ascii")

    def to_symlink(path):
        path.unlink()
        path.symlink_to(decoy)

    _swap_after_check(monkeypatch, to_symlink)
    with pytest.raises(JudgeKeyStoreError):
        store.load("typesafe")


@pytest.mark.skipif(not POSIX, reason="inode identity is checked on POSIX")
def test_a_different_file_renamed_in_after_the_check_is_refused(store, tmp_path, monkeypatch):
    store.set_key("typesafe", VALID_KEY)
    other = tmp_path / "secrets" / "other.key"
    other.write_text("attacker-key-00000000", encoding="ascii")
    os.chmod(other, 0o600)

    _swap_after_check(monkeypatch, lambda path: os.replace(other, path))
    with pytest.raises(JudgeKeyStoreError):
        store.load("typesafe")


@pytest.mark.skipif(not POSIX, reason="POSIX permissions")
def test_a_key_file_other_users_can_change_is_refused(store):
    store.set_key("typesafe", VALID_KEY)
    os.chmod(_key_file(store), 0o666)
    with pytest.raises(JudgeKeyStoreError) as excinfo:
        store.load("typesafe")
    assert VALID_KEY not in str(excinfo.value)


@pytest.mark.skipif(not POSIX, reason="POSIX symlinks")
def test_a_secrets_folder_that_is_a_symlink_is_refused_on_read(store, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    (elsewhere / "jev-typesafe.key").write_text(VALID_KEY, encoding="ascii")
    os.chmod(elsewhere / "jev-typesafe.key", 0o600)
    (tmp_path / "secrets").symlink_to(elsewhere)
    with pytest.raises(JudgeKeyStoreError):
        store.load("typesafe")


@pytest.mark.skipif(not POSIX, reason="POSIX symlinks")
def test_set_key_never_writes_through_a_folder_swapped_for_a_symlink(store, tmp_path, monkeypatch):
    """The folder is checked, then held open: a swap after the check cannot
    redirect the write somewhere else."""
    store.set_key("typesafe", VALID_KEY)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    real = judge_keys._refuse_unsafe
    swapped = {"done": False}

    def checked_then_swapped(path, *, expect):
        result = real(path, expect=expect)
        if expect == "dir" and not swapped["done"]:
            swapped["done"] = True
            moved = tmp_path / "moved-secrets"
            os.rename(path, moved)
            path.symlink_to(elsewhere)
        return result

    monkeypatch.setattr(judge_keys, "_refuse_unsafe", checked_then_swapped)
    try:
        store.set_key("typesafe", "sk-live-" + "z9Y8x7W6" * 4)
    except (JudgeKeyStoreError, OSError):
        pass
    assert list(elsewhere.iterdir()) == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_a_fresh_key_is_still_owner_only(store):
    store.set_key("typesafe", VALID_KEY)
    mode = stat.S_IMODE(_key_file(store).stat().st_mode)
    assert mode == 0o600
    assert store.load("typesafe") == VALID_KEY


@pytest.mark.parametrize(("key", "usable"), [
    (VALID_KEY, True),
    (VALID_KEY + "\n", False),
    ("valid-key-12345678\r\nX-Injected: 1", False),
    ("short", False),
    (None, False),
])
def test_is_usable_key_is_the_header_guard_for_keys_from_elsewhere(key, usable):
    assert judge_keys.is_usable_key(key) is usable
