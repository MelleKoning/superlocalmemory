# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Client-side encryption for cloud backup files.

Cloud backups carry the whole memory store, credentials included, so every
file is encrypted on this machine before it is uploaded. The provider only
ever stores ciphertext.

File format, version 1::

    header  = magic "SLMBKENC" (8) | version (1) | key id (8)
              | chunk size, big-endian uint32 (4) | file nonce (12)
    body    = chunk_0 || chunk_1 || ... || chunk_n
    chunk_i = AES-256-GCM(file_key, nonce_i, plaintext_i, aad=header)

``file_key`` is derived per file with HKDF-SHA256 from the backup key, salted
with the random 96-bit file nonce, so no (key, nonce) pair ever repeats across
files. ``nonce_i`` is ``file_nonce[:7] || i (uint32) || last-flag (1)``: the
counter authenticates chunk order and the flag authenticates the end of the
stream, so reordering, splicing, truncation and appended data all fail. The
header is the associated data of every chunk, so it cannot be altered either.
This is the STREAM construction used by Tink's streaming AEAD and by ``age``.

Plaintext is processed one chunk at a time: a multi-gigabyte ``memory.db`` is
never held in memory.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MAGIC = b"SLMBKENC"
FORMAT_VERSION = 1
ENCRYPTED_SUFFIX = ".slmenc"
KEY_BYTES = 32
KEY_ID_BYTES = 8
FILE_NONCE_BYTES = 12
TAG_BYTES = 16
DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB
MIN_CHUNK_SIZE = 256
MAX_CHUNK_SIZE = 16 << 20
SQLITE_MAGIC = b"SQLite format 3\x00"

FORMAT_ENCRYPTED = "encrypted"
FORMAT_SQLITE = "sqlite"
FORMAT_UNKNOWN = "unknown"

_HEADER = struct.Struct(">8sB8sI12s")
HEADER_SIZE = _HEADER.size
_PREFIX_BYTES = 7
_MAX_CHUNKS = 1 << 32
_FILE_KEY_INFO = b"superlocalmemory:cloud-backup:v1:file-key"
_KEY_ID_INFO = b"superlocalmemory:cloud-backup:key-id"


class BackupCryptoError(Exception):
    """Base class for backup encryption errors. Messages never carry key material."""


class BackupFormatError(BackupCryptoError):
    """The file is not an encrypted backup this version can read."""


class BackupIntegrityError(BackupCryptoError):
    """The ciphertext was altered, truncated, reordered, or the key is wrong."""


class BackupKeyMismatchError(BackupCryptoError):
    """The backup was encrypted with a different key than the one supplied."""

    def __init__(self, found_key_id: str, supplied_key_id: str) -> None:
        self.found_key_id = found_key_id
        self.supplied_key_id = supplied_key_id
        super().__init__(
            f"This backup was encrypted with key {found_key_id}, but the key "
            f"supplied is {supplied_key_id}. Use the recovery key that was shown "
            "when the backup was set up on the machine that made it."
        )


@dataclass(frozen=True)
class BackupHeader:
    version: int
    key_id: bytes
    chunk_size: int
    file_nonce: bytes
    raw: bytes


def key_id(key: bytes) -> bytes:
    """A short public fingerprint of a backup key (reveals nothing about it)."""
    _check_key(key)
    return hmac.new(key, _KEY_ID_INFO, hashlib.sha256).digest()[:KEY_ID_BYTES]


def key_id_hex(key: bytes) -> str:
    return key_id(key).hex()


def detect_format(path: Path) -> str:
    """Classify a backup file by its first bytes."""
    with open(path, "rb") as fh:
        head = fh.read(max(len(MAGIC), len(SQLITE_MAGIC)))
    if head.startswith(MAGIC):
        return FORMAT_ENCRYPTED
    if head.startswith(SQLITE_MAGIC):
        return FORMAT_SQLITE
    return FORMAT_UNKNOWN


def read_header(src: BinaryIO) -> BackupHeader:
    raw = _read_exact(src, HEADER_SIZE)
    if len(raw) < HEADER_SIZE or not raw.startswith(MAGIC):
        raise BackupFormatError("Not an encrypted SuperLocalMemory backup file.")
    _magic, version, kid, chunk_size, nonce = _HEADER.unpack(raw)
    if version != FORMAT_VERSION:
        raise BackupFormatError(
            f"Encrypted backup format version {version} is not supported by this "
            "version of SuperLocalMemory. Upgrade SuperLocalMemory to restore it."
        )
    if not MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE:
        raise BackupFormatError("Encrypted backup header is invalid (chunk size).")
    return BackupHeader(version, kid, chunk_size, nonce, raw)


def encrypt_stream(
    src: BinaryIO, dst: BinaryIO, key: bytes, *, chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    """Encrypt ``src`` into ``dst``. Returns the number of bytes written."""
    _check_key(key)
    if not MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE:
        raise ValueError("chunk_size out of range")
    nonce = os.urandom(FILE_NONCE_BYTES)
    header = _HEADER.pack(MAGIC, FORMAT_VERSION, key_id(key), chunk_size, nonce)
    aead = AESGCM(_file_key(key, nonce))
    dst.write(header)
    written = len(header)
    for index, chunk, last in _chunks(src, chunk_size):
        sealed = aead.encrypt(_chunk_nonce(nonce, index, last), chunk, header)
        dst.write(sealed)
        written += len(sealed)
    return written


def decrypt_stream(src: BinaryIO, dst: BinaryIO, key: bytes) -> int:
    """Decrypt ``src`` into ``dst``. Returns the number of plaintext bytes.

    Raises ``BackupKeyMismatchError`` when the header names another key and
    ``BackupIntegrityError`` on any authentication failure. ``dst`` may hold
    a prefix of the plaintext when this raises; ``decrypt_file`` discards it.
    """
    _check_key(key)
    header = read_header(src)
    if not hmac.compare_digest(header.key_id, key_id(key)):
        raise BackupKeyMismatchError(header.key_id.hex(), key_id_hex(key))
    aead = AESGCM(_file_key(key, header.file_nonce))
    for index, sealed, last in _chunks(src, header.chunk_size + TAG_BYTES):
        if len(sealed) < TAG_BYTES:
            raise BackupIntegrityError(_INTEGRITY_MESSAGE)
        try:
            plain = aead.decrypt(
                _chunk_nonce(header.file_nonce, index, last), sealed, header.raw,
            )
        except InvalidTag as exc:
            raise BackupIntegrityError(_INTEGRITY_MESSAGE) from exc
        dst.write(plain)
    return dst.tell() if dst.seekable() else -1


def encrypt_file(
    src_path: Path, dst_path: Path, key: bytes, *, chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Path:
    with open(src_path, "rb") as src, _atomic_output(Path(dst_path), True) as dst:
        encrypt_stream(src, dst, key, chunk_size=chunk_size)
    return Path(dst_path)


def decrypt_file(
    src_path: Path, dst_path: Path, key: bytes, *, overwrite: bool = False,
) -> Path:
    """Decrypt to ``dst_path`` atomically: on any failure nothing is left behind."""
    with open(src_path, "rb") as src, _atomic_output(Path(dst_path), overwrite) as dst:
        decrypt_stream(src, dst, key)
    return Path(dst_path)


def copy_file_atomic(src_path: Path, dst_path: Path, *, overwrite: bool = False) -> Path:
    """Copy a plaintext backup with the same no-partial-file guarantee."""
    with open(src_path, "rb") as src, _atomic_output(Path(dst_path), overwrite) as dst:
        while True:
            block = src.read(DEFAULT_CHUNK_SIZE)
            if not block:
                break
            dst.write(block)
    return Path(dst_path)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

_INTEGRITY_MESSAGE = (
    "The encrypted backup failed its integrity check: it was changed, cut short, "
    "or is incomplete. Download it again; nothing was restored."
)


def _check_key(key: bytes) -> None:
    if not isinstance(key, (bytes, bytearray)) or len(key) != KEY_BYTES:
        raise ValueError("backup key must be 32 bytes")


def _file_key(key: bytes, file_nonce: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(), length=KEY_BYTES, salt=file_nonce, info=_FILE_KEY_INFO,
    ).derive(bytes(key))


def _chunk_nonce(file_nonce: bytes, index: int, last: bool) -> bytes:
    if index >= _MAX_CHUNKS:
        raise BackupCryptoError("backup file is too large for this format")
    return file_nonce[:_PREFIX_BYTES] + index.to_bytes(4, "big") + (b"\x01" if last else b"\x00")


def _read_exact(src: BinaryIO, size: int) -> bytes:
    parts: list[bytes] = []
    remaining = size
    while remaining > 0:
        block = src.read(remaining)
        if not block:
            break
        parts.append(block)
        remaining -= len(block)
    return b"".join(parts)


def _chunks(src: BinaryIO, size: int) -> Iterator[tuple[int, bytes, bool]]:
    """Yield (index, block, is_last) with one block of look-ahead.

    Always yields at least one block (possibly empty) so an empty input still
    produces an authenticated final chunk.
    """
    current = _read_exact(src, size)
    index = 0
    while True:
        following = _read_exact(src, size) if len(current) == size else b""
        last = not following
        yield index, current, last
        if last:
            return
        current = following
        index += 1


@contextmanager
def _atomic_output(dst: Path, overwrite: bool) -> Iterator[BinaryIO]:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists; choose another path")
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dst.parent, prefix=".slm-partial-")
    try:
        with os.fdopen(fd, "wb") as out:
            yield out
            out.flush()
            os.fsync(out.fileno())
        if dst.exists() and not overwrite:
            raise FileExistsError(f"{dst} already exists; choose another path")
        os.replace(tmp_name, dst)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
