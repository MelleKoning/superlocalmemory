# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The cloud backup encryption key and its recovery key.

One 256-bit key per machine encrypts every cloud backup. It lives in the same
credential store as the GitHub token and Drive credentials (OS keychain first,
then the existing owner-only file fallback) and never beside the backup files.

The user sees it as a *recovery key* (``SLMBK1-XXXX-...``): base32 with a
checksum so a typo is reported as a typo rather than as a wrong key. It is
shown once when it is created, can be shown again on this machine, and can be
imported on a new machine.

Safety rules enforced here:
  * A key is never silently replaced. If the state file records a key but the
    credential store has lost it, every operation fails closed and tells the
    user how to put it back.
  * A freshly generated key is read back from the store before it is used, so
    nothing is ever encrypted with a key that was not persisted.
  * Creation is serialised across threads and processes, so concurrent first
    backups converge on one key.
  * Key material is never logged and never placed in an exception message.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from superlocalmemory.infra import cloud_backup as _cb
from superlocalmemory.infra.backup_crypto import (
    KEY_BYTES,
    BackupCryptoError,
    key_id_hex,
)

logger = logging.getLogger("superlocalmemory.cloud_backup.keys")

CREDENTIAL_NAME = "cloud_backup_key"
RECOVERY_PREFIX = "SLMBK1"
_STORED_PREFIX = "slmbk1:"
_STATE_FILE = "cloud-backup-encryption.json"
_LOCK_FILE = ".cloud-backup-key.lock"
_CHECKSUM_BYTES = 3
_CHECKSUM_INFO = b"superlocalmemory:recovery-key:v1"
_GROUP = 4
_BASE32_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
_LOOKALIKES = str.maketrans({"0": "O", "1": "I", "8": "B"})
_ENCODED_LENGTH = 56  # (32 key + 3 checksum bytes) * 8 / 5

_THREAD_LOCK = threading.RLock()

RECOVERY_NOTICE = (
    "Cloud backups are now encrypted on this computer before they are uploaded. "
    "A recovery key was created. You need it to restore these backups on any "
    "other computer. Show it with `slm backup recovery-key` (or Backup > Show "
    "recovery key in the dashboard) and keep it somewhere safe off this computer."
)
LEGACY_NOTICE = (
    "Backups uploaded before version 4.1.19 were not encrypted and are still in "
    "your cloud storage; SuperLocalMemory did not delete them. GitHub: only the "
    "newest 5 backup releases are kept, so the old unencrypted ones are removed "
    "as new backups are made; to remove them now, delete the older releases on "
    "the repository's Releases page. Google Drive: in the SLM-Backup folder, "
    "delete the files whose names end in .db (encrypted backups end in .slmenc)."
)


class BackupKeyError(BackupCryptoError):
    """A backup key operation could not be completed."""


class BackupKeyUnavailableError(BackupKeyError):
    """The backup key cannot be loaded or saved; nothing was encrypted with it."""


class RecoveryKeyFormatError(BackupKeyError):
    """The text entered is not a valid recovery key."""


# ---------------------------------------------------------------------------
# Recovery key encoding
# ---------------------------------------------------------------------------


def format_recovery_key(key: bytes) -> str:
    if len(key) != KEY_BYTES:
        raise ValueError("backup key must be 32 bytes")
    body = base64.b32encode(bytes(key) + _checksum(key)).decode("ascii")
    groups = [body[i : i + _GROUP] for i in range(0, len(body), _GROUP)]
    return "-".join([RECOVERY_PREFIX, *groups])


def parse_recovery_key(text: str) -> bytes:
    cleaned = "".join(str(text or "").split()).replace("-", "").upper()
    if cleaned.startswith(RECOVERY_PREFIX):
        cleaned = cleaned[len(RECOVERY_PREFIX):]
    cleaned = cleaned.translate(_LOOKALIKES)
    if len(cleaned) != _ENCODED_LENGTH or not set(cleaned) <= _BASE32_ALPHABET:
        raise RecoveryKeyFormatError(
            "That is not a recovery key. A recovery key starts with SLMBK1- "
            "followed by 14 groups of 4 letters and digits."
        )
    try:
        raw = base64.b32decode(cleaned)
    except (binascii.Error, ValueError) as exc:
        raise RecoveryKeyFormatError("That is not a valid recovery key.") from exc
    key, check = raw[:KEY_BYTES], raw[KEY_BYTES:]
    if _checksum(key) != check:
        raise RecoveryKeyFormatError(
            "The recovery key looks mistyped (its checksum does not match). "
            "Check each group of characters and try again."
        )
    return key


def _checksum(key: bytes) -> bytes:
    return hashlib.sha256(_CHECKSUM_INFO + bytes(key)).digest()[:_CHECKSUM_BYTES]


# ---------------------------------------------------------------------------
# Key storage
# ---------------------------------------------------------------------------


def state_path() -> Path:
    return _cb._memory_dir() / _STATE_FILE


def load_backup_key() -> bytes | None:
    """Return this machine's backup key, ``None`` if none was ever created.

    Raises ``BackupKeyUnavailableError`` when a key was created here but the
    credential store no longer returns it (or returns something corrupt).
    """
    stored = _read_stored_key()
    if stored is not None:
        return stored
    recorded = _read_state().get("key_id")
    if recorded:
        raise BackupKeyUnavailableError(_missing_key_message(str(recorded)))
    return None


def ensure_backup_key(*, origin: str, legacy_uploads: bool = False) -> tuple[bytes, bool]:
    """Return ``(key, created)``, creating and persisting a key on first use."""
    with _key_lock():
        existing = load_backup_key()
        if existing is not None:
            _repair_state(existing)
            return existing, False
        key = os.urandom(KEY_BYTES)
        if not _cb._store_credential(CREDENTIAL_NAME, _encode_stored(key)):
            raise BackupKeyUnavailableError(
                "The backup encryption key could not be saved in the credential "
                "store, so nothing was uploaded. Check that the OS keychain or the "
                "SuperLocalMemory data folder is writable."
            )
        if _read_stored_key() != key:
            raise BackupKeyUnavailableError(
                "The credential store did not keep the backup encryption key, so "
                "nothing was uploaded. Check the OS keychain configuration."
            )
        _write_state({
            "version": 1,
            "key_id": key_id_hex(key),
            "created_at": datetime.now(UTC).isoformat(),
            "origin": origin,
            "recovery_key_shown": False,
            "legacy_plaintext_uploads": bool(legacy_uploads),
        })
        logger.warning(
            "Cloud backups are now encrypted on this computer (key %s). A recovery "
            "key was created; show it with `slm backup recovery-key` and keep it "
            "somewhere safe off this computer.",
            key_id_hex(key),
        )
        return key, True


def mark_recovery_key_shown() -> None:
    with _key_lock():
        _update_state(recovery_key_shown=True)


def reveal_recovery_key() -> dict[str, Any]:
    """Return the recovery key for display to the user. Never log the result."""
    with _key_lock():
        key = load_backup_key()
        if key is None:
            raise BackupKeyUnavailableError(
                "No backup encryption key exists on this computer yet. One is "
                "created when you connect a cloud destination or make the first "
                "cloud backup."
            )
        _repair_state(key)
        _update_state(recovery_key_shown=True)
    return {
        "recovery_key": format_recovery_key(key),
        "key_id": key_id_hex(key),
        "advice": (
            "Keep this recovery key somewhere safe off this computer (a password "
            "manager or printed copy). Anyone with it and your cloud account can "
            "read your backups; without it, they cannot be restored elsewhere."
        ),
    }


def install_recovery_key(text: str) -> dict[str, Any]:
    """Put a recovery key into this machine's credential store.

    Refuses to replace a different key that is already in use: backups made
    with it would become unreadable here.
    """
    key = parse_recovery_key(text)
    with _key_lock():
        current = _read_stored_key()
        if current is not None and current != key:
            raise BackupKeyError(
                f"A different backup key ({key_id_hex(current)}) is already in use "
                "on this computer, so it was not replaced. To restore a backup made "
                "with another key, pass that recovery key to `slm backup decrypt`."
            )
        if current is None:
            if not _cb._store_credential(CREDENTIAL_NAME, _encode_stored(key)):
                raise BackupKeyUnavailableError("The recovery key could not be saved.")
            if _read_stored_key() != key:
                raise BackupKeyUnavailableError("The credential store did not keep the key.")
        previous = _read_state()
        _write_state({
            "version": 1,
            "key_id": key_id_hex(key),
            "created_at": previous.get("created_at") or datetime.now(UTC).isoformat(),
            "origin": "import",
            "recovery_key_shown": True,
            "legacy_plaintext_uploads": bool(previous.get("legacy_plaintext_uploads")),
            "legacy_notice_acknowledged": bool(previous.get("legacy_notice_acknowledged")),
        })
    logger.info("Backup encryption key %s installed from a recovery key", key_id_hex(key))
    return {"key_id": key_id_hex(key), "installed": current is None}


def acknowledge_legacy_notice() -> None:
    with _key_lock():
        _update_state(legacy_notice_acknowledged=True)


def encryption_status() -> dict[str, Any]:
    """Public, key-free status for the dashboard and CLI."""
    error = None
    try:
        key = load_backup_key()
    except BackupKeyUnavailableError as exc:
        key, error = None, str(exc)
    state = _read_state()
    pending = key is not None and not state.get("recovery_key_shown", False)
    legacy = bool(state.get("legacy_plaintext_uploads")) and not state.get(
        "legacy_notice_acknowledged", False
    )
    notices = [n for n, on in ((error, bool(error)), (RECOVERY_NOTICE, pending),
                               (LEGACY_NOTICE, legacy)) if on]
    return {
        "uploads_encrypted": True,
        "enabled": key is not None,
        "key_id": key_id_hex(key) if key is not None else state.get("key_id"),
        "created_at": state.get("created_at"),
        "recovery_key_pending": pending,
        "legacy_plaintext_uploads": legacy,
        "key_error": error,
        "notices": notices,
    }


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _missing_key_message(kid: str) -> str:
    return (
        f"The backup encryption key {kid} is missing from this computer's credential "
        "store. No new key was created, because backups already made with it would "
        "then be unreadable here, and nothing was uploaded. Run `slm backup "
        "recovery-key --import` and enter your recovery key to put it back. If that "
        f"recovery key is lost, delete {state_path()} to start over with a new key; "
        f"backups made with key {kid} can then no longer be restored."
    )


def _encode_stored(key: bytes) -> str:
    return _STORED_PREFIX + base64.urlsafe_b64encode(bytes(key)).decode("ascii").rstrip("=")


def _read_stored_key() -> bytes | None:
    raw = _cb._get_credential(CREDENTIAL_NAME)
    if raw is None:
        return None
    if not raw.startswith(_STORED_PREFIX):
        raise BackupKeyUnavailableError(
            "The stored backup encryption key is unreadable. It was left unchanged; "
            "nothing was uploaded. Import your recovery key to repair it."
        )
    body = raw[len(_STORED_PREFIX):]
    try:
        key = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
    except (binascii.Error, ValueError) as exc:
        raise BackupKeyUnavailableError("The stored backup encryption key is unreadable.") from exc
    if len(key) != KEY_BYTES:
        raise BackupKeyUnavailableError("The stored backup encryption key is unreadable.")
    return key


def _read_state() -> dict[str, Any]:
    path = state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Backup encryption state unreadable (%s); treating as empty", type(exc).__name__)
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(data: dict[str, Any]) -> None:
    _cb._atomic_write_creds(state_path(), data)


def _update_state(**changes: Any) -> None:
    current = _read_state()
    if not current:
        return
    _write_state({**current, **changes})


def _repair_state(key: bytes) -> None:
    """A key with no state record (state file lost): record it, ask to view it."""
    state = _read_state()
    if state.get("key_id") == key_id_hex(key):
        return
    _write_state({
        **state,
        "version": 1,
        "key_id": key_id_hex(key),
        "created_at": state.get("created_at") or datetime.now(UTC).isoformat(),
        "recovery_key_shown": False,
    })


@contextmanager
def _key_lock() -> Iterator[None]:
    """Serialise key creation across threads and processes."""
    lock_path = _cb._memory_dir() / _LOCK_FILE
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCK, open(lock_path, "a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
