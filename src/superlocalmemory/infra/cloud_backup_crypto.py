# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Cloud backup encryption: what the upload and restore paths call.

Upload: ``prepare_upload_key`` makes sure the machine has a persisted backup
key, then ``encrypted_upload_copy`` writes an encrypted copy of one backup
file into a private temporary directory, checks it, yields it for upload and
removes it afterwards. Uploaders never see the plaintext path's bytes.

Restore: ``restore_backup_file`` turns a downloaded backup back into a usable
SQLite file, with this machine's key or with a recovery key. Backups made by
4.1.18 and earlier are plain SQLite; they are recognised by their header and
copied as they are.
"""

from __future__ import annotations

import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from superlocalmemory.infra import backup_crypto as bc
from superlocalmemory.infra import backup_keys as bk
from superlocalmemory.infra import cloud_backup as _cb

logger = logging.getLogger("superlocalmemory.cloud_backup.crypto")

UPLOAD_MIMETYPE = "application/octet-stream"

_SHOW_ONCE_MESSAGE = (
    "Backups are encrypted on this computer before upload. Save this recovery key "
    "now, somewhere safe off this computer: it is shown only once, and you need it "
    "to restore your backups on another computer. `slm backup recovery-key` shows "
    "it again on this computer."
)
_EXISTING_KEY_MESSAGE = (
    "Backups are encrypted on this computer before upload, with the key this "
    "computer already has. `slm backup recovery-key` shows its recovery key."
)
_ENCRYPTED_NOTE = (
    "Uploads are encrypted on this computer before they leave it. "
    "`slm backup recovery-key` shows the recovery key."
)


# ---------------------------------------------------------------------------
# Upload side
# ---------------------------------------------------------------------------


def previous_uploads_exist(db_path: Path | None = None) -> bool:
    """Did any destination sync before encryption existed (4.1.18 uploads)?"""
    try:
        destinations = _cb.get_destinations(db_path)
    except Exception as exc:  # status only: never block a backup on it
        logger.debug("Could not read destinations: %s", type(exc).__name__)
        return False
    return any(d.get("last_sync_at") for d in destinations)


def prepare_upload_key(db_path: Path | None = None) -> tuple[bytes, bool]:
    """Return ``(key, created)``. Raises if no persisted key can be had."""
    return bk.ensure_backup_key(
        origin="backup", legacy_uploads=previous_uploads_exist(db_path),
    )


@contextmanager
def encrypted_upload_copy(path: Path, key: bytes) -> Iterator[Path]:
    """Yield an encrypted copy of ``path``; it is deleted when the block exits."""
    workdir = Path(tempfile.mkdtemp(prefix="slm-cloud-enc-"))
    out = workdir / (Path(path).name + bc.ENCRYPTED_SUFFIX)
    try:
        bc.encrypt_file(Path(path), out, key)
        _check_ready(out, key)
        yield out
    finally:
        out.unlink(missing_ok=True)
        try:
            workdir.rmdir()
        except OSError as exc:
            logger.warning("Could not remove temporary backup folder: %s", exc)


def begin_encrypted_sync(db_path: Path | None = None) -> dict[str, Any]:
    """Make sure a key exists before a sync, and describe it (key-free)."""
    try:
        key, created = prepare_upload_key(db_path)
    except Exception as exc:
        logger.error("Cloud backup not uploaded: %s", exc)
        return {"enabled": False, "recovery_key_created": False, "error": str(exc)}
    return {
        "enabled": True,
        "key_id": bc.key_id_hex(key),
        "recovery_key_created": created,
        "message": bk.RECOVERY_NOTICE if created else _ENCRYPTED_NOTE,
        "notices": bk.encryption_status()["notices"],
    }


def attach_encryption(result: dict[str, Any]) -> dict[str, Any]:
    """Add encryption details to a connect response; the key appears once."""
    if "error" in result:
        return result
    try:
        key, created = bk.ensure_backup_key(
            origin="connect", legacy_uploads=previous_uploads_exist(),
        )
    except Exception as exc:
        logger.error("Backup encryption key unavailable: %s", exc)
        return {**result, "encryption": {"enabled": False, "error": str(exc)}}
    info: dict[str, Any] = {"enabled": True, "key_id": bc.key_id_hex(key)}
    if created:
        bk.mark_recovery_key_shown()
        info = {**info, "recovery_key": bk.format_recovery_key(key),
                "message": _SHOW_ONCE_MESSAGE}
    else:
        info = {**info, "message": _EXISTING_KEY_MESSAGE}
    return {**result, "encryption": info}


def recovery_key_page_text(result: dict[str, Any]) -> str:
    """Sentence appended to the connect success page (HTML-escaped by caller)."""
    enc = result.get("encryption") or {}
    if enc.get("recovery_key"):
        return (
            " Backups are encrypted on this computer before upload. Your recovery "
            "key, shown only once (save it now; you need it to restore on another "
            f"computer): {enc['recovery_key']}"
        )
    if enc.get("enabled"):
        return " " + _EXISTING_KEY_MESSAGE
    if enc.get("error"):
        return " " + str(enc["error"])
    return ""


def _check_ready(out: Path, key: bytes) -> None:
    """Refuse to hand over anything but ciphertext made with the stored key."""
    with open(out, "rb") as fh:
        header = bc.read_header(fh)
    if header.key_id != bc.key_id(key) or bk.load_backup_key() != key:
        raise bk.BackupKeyError(
            "The backup encryption key changed while preparing the upload; "
            "nothing was uploaded."
        )


# ---------------------------------------------------------------------------
# Restore side
# ---------------------------------------------------------------------------


def restore_backup_file(
    src: Path,
    dest: Path | None = None,
    *,
    recovery_key: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Turn a downloaded cloud backup into a plain SQLite file at ``dest``.

    Raises a ``BackupCryptoError`` subclass with a user-facing message on any
    failure; ``dest`` is never left partly written.
    """
    src = Path(src)
    fmt = bc.detect_format(src)
    if fmt == bc.FORMAT_UNKNOWN:
        raise bc.BackupFormatError(
            "This file is not a SuperLocalMemory backup (it is neither an encrypted "
            "backup nor a SQLite database). Check that the download completed."
        )
    target = Path(dest) if dest is not None else _default_output(src, fmt)
    if fmt == bc.FORMAT_SQLITE:
        bc.copy_file_atomic(src, target, overwrite=overwrite)
        return {
            "format": "plaintext",
            "output": str(target),
            "key_source": None,
            "message": "This backup was made before 4.1.19 and is not encrypted; "
                       "it was copied as it is.",
        }
    key, source = _restore_key(recovery_key)
    bc.decrypt_file(src, target, key, overwrite=overwrite)
    return {
        "format": "encrypted",
        "output": str(target),
        "key_source": source,
        "key_id": bc.key_id_hex(key),
        "message": "The backup was decrypted and passed its integrity check.",
    }


def _restore_key(recovery_key: str | None) -> tuple[bytes, str]:
    if recovery_key is not None:
        return bk.parse_recovery_key(recovery_key), "recovery key"
    key = bk.load_backup_key()
    if key is None:
        raise bk.BackupKeyUnavailableError(
            "This backup is encrypted and this computer has no backup key. Restore "
            "it with the recovery key that was shown when cloud backup was set up: "
            "`slm backup decrypt FILE --recovery-key-stdin`."
        )
    return key, "this computer"


def _default_output(src: Path, fmt: str) -> Path:
    if fmt == bc.FORMAT_ENCRYPTED and src.name.endswith(bc.ENCRYPTED_SUFFIX):
        return src.with_name(src.name[: -len(bc.ENCRYPTED_SUFFIX)])
    return src.with_name(f"{src.stem}-restored{src.suffix or '.db'}")
