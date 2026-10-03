# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Client-side encryption format for cloud backups.

Every file sent to GitHub or Google Drive is encrypted on this machine first.
These tests pin the file format: a round trip returns the exact bytes, and any
change to the ciphertext (a flipped byte, a cut, a reordered chunk, a different
key) fails loudly without leaving a partial plaintext file behind.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from superlocalmemory.infra import backup_crypto as bc

_KEY = bytes(range(32))
_OTHER_KEY = bytes(range(1, 33))
_SMALL_CHUNK = 256


def _encrypt(data: bytes, key: bytes = _KEY, chunk: int = _SMALL_CHUNK) -> bytes:
    out = io.BytesIO()
    bc.encrypt_stream(io.BytesIO(data), out, key, chunk_size=chunk)
    return out.getvalue()


def _decrypt(blob: bytes, key: bytes = _KEY) -> bytes:
    out = io.BytesIO()
    bc.decrypt_stream(io.BytesIO(blob), out, key)
    return out.getvalue()


@pytest.mark.parametrize("size", [0, 1, 255, 256, 257, 512, 4096 + 17])
def test_round_trip_returns_exact_bytes(size: int) -> None:
    data = os.urandom(size)
    assert _decrypt(_encrypt(data)) == data


def test_large_input_round_trips_through_files(tmp_path: Path) -> None:
    src = tmp_path / "memory-20260101-000000.db"
    data = os.urandom(3 * bc.DEFAULT_CHUNK_SIZE + 12345)
    src.write_bytes(data)
    enc = tmp_path / "out.slmenc"
    dec = tmp_path / "restored.db"
    bc.encrypt_file(src, enc, _KEY)
    bc.decrypt_file(enc, dec, _KEY)
    assert dec.read_bytes() == data
    # One tag per chunk plus the header: the size is predictable.
    chunks = 4
    assert enc.stat().st_size == bc.HEADER_SIZE + len(data) + chunks * bc.TAG_BYTES


def test_header_carries_magic_version_key_id_and_nonce() -> None:
    blob = _encrypt(b"hello")
    header = bc.read_header(io.BytesIO(blob))
    assert blob.startswith(bc.MAGIC)
    assert header.version == bc.FORMAT_VERSION
    assert header.key_id == bc.key_id(_KEY)
    assert len(header.file_nonce) == 12
    assert header.chunk_size == _SMALL_CHUNK


def test_each_file_gets_a_fresh_nonce_and_different_ciphertext() -> None:
    data = b"same plaintext" * 50
    first, second = _encrypt(data), _encrypt(data)
    assert bc.read_header(io.BytesIO(first)).file_nonce != \
        bc.read_header(io.BytesIO(second)).file_nonce
    assert first[bc.HEADER_SIZE:] != second[bc.HEADER_SIZE:]


def test_ciphertext_does_not_contain_the_plaintext() -> None:
    secret = b"ghp_PLANTEDcredential0123456789abcdef"
    blob = _encrypt(b"x" * 300 + secret + b"y" * 300)
    assert secret not in blob
    assert b"PLANTED" not in blob


# ---- tamper detection -------------------------------------------------------


def test_flipped_body_byte_is_rejected() -> None:
    blob = bytearray(_encrypt(os.urandom(1000)))
    blob[bc.HEADER_SIZE + 10] ^= 0x01
    with pytest.raises(bc.BackupIntegrityError):
        _decrypt(bytes(blob))


def test_flipped_header_byte_is_rejected() -> None:
    blob = bytearray(_encrypt(os.urandom(1000)))
    blob[bc.HEADER_SIZE - 1] ^= 0x01  # last byte of the file nonce
    with pytest.raises(bc.BackupIntegrityError):
        _decrypt(bytes(blob))


def test_truncation_at_a_chunk_boundary_is_rejected() -> None:
    blob = _encrypt(os.urandom(_SMALL_CHUNK * 3))
    one_chunk = _SMALL_CHUNK + bc.TAG_BYTES
    with pytest.raises(bc.BackupIntegrityError):
        _decrypt(blob[: bc.HEADER_SIZE + 2 * one_chunk])


def test_truncation_inside_a_chunk_is_rejected() -> None:
    blob = _encrypt(os.urandom(_SMALL_CHUNK * 3))
    with pytest.raises(bc.BackupIntegrityError):
        _decrypt(blob[:-5])


def test_header_only_file_is_rejected() -> None:
    blob = _encrypt(b"abc")
    with pytest.raises(bc.BackupIntegrityError):
        _decrypt(blob[: bc.HEADER_SIZE])


def test_swapped_chunks_are_rejected() -> None:
    blob = _encrypt(os.urandom(_SMALL_CHUNK * 3))
    size = _SMALL_CHUNK + bc.TAG_BYTES
    head, body = blob[: bc.HEADER_SIZE], blob[bc.HEADER_SIZE:]
    chunks = [body[i : i + size] for i in range(0, len(body), size)]
    swapped = head + chunks[1] + chunks[0] + b"".join(chunks[2:])
    with pytest.raises(bc.BackupIntegrityError):
        _decrypt(swapped)


def test_appended_data_is_rejected() -> None:
    blob = _encrypt(os.urandom(_SMALL_CHUNK * 2))
    with pytest.raises(bc.BackupIntegrityError):
        _decrypt(blob + os.urandom(_SMALL_CHUNK + bc.TAG_BYTES))


def test_chunk_from_another_file_is_rejected() -> None:
    first = _encrypt(os.urandom(_SMALL_CHUNK * 2))
    second = _encrypt(os.urandom(_SMALL_CHUNK * 2))
    spliced = first[: bc.HEADER_SIZE] + second[bc.HEADER_SIZE:]
    with pytest.raises(bc.BackupIntegrityError):
        _decrypt(spliced)


def test_wrong_key_reports_a_key_mismatch_naming_both_ids() -> None:
    blob = _encrypt(b"secret")
    with pytest.raises(bc.BackupKeyMismatchError) as info:
        _decrypt(blob, key=_OTHER_KEY)
    message = str(info.value)
    assert bc.key_id_hex(_KEY) in message
    assert bc.key_id_hex(_OTHER_KEY) in message


def test_wrong_key_with_forged_key_id_still_fails_integrity() -> None:
    blob = bytearray(_encrypt(b"secret" * 100))
    forged = bc.key_id(_OTHER_KEY)
    start = len(bc.MAGIC) + 1
    blob[start : start + len(forged)] = forged
    with pytest.raises(bc.BackupIntegrityError):
        _decrypt(bytes(blob), key=_OTHER_KEY)


def test_unsupported_version_is_a_format_error() -> None:
    blob = bytearray(_encrypt(b"secret"))
    blob[len(bc.MAGIC)] = 99
    with pytest.raises(bc.BackupFormatError):
        _decrypt(bytes(blob))


def test_absurd_chunk_size_in_header_is_a_format_error() -> None:
    blob = bytearray(_encrypt(b"secret"))
    offset = len(bc.MAGIC) + 1 + 8
    blob[offset : offset + 4] = (1 << 31).to_bytes(4, "big")
    with pytest.raises(bc.BackupFormatError):
        _decrypt(bytes(blob))


def test_key_must_be_32_bytes() -> None:
    with pytest.raises(ValueError):
        _encrypt(b"x", key=b"short")


# ---- file level: no partial output -----------------------------------------


def test_failed_decrypt_leaves_no_output_and_no_temp_file(tmp_path: Path) -> None:
    enc = tmp_path / "b.slmenc"
    blob = bytearray(_encrypt(os.urandom(_SMALL_CHUNK * 4)))
    blob[-3] ^= 0xFF  # the first chunks are valid, the last one is not
    enc.write_bytes(bytes(blob))
    out = tmp_path / "restore" / "memory.db"
    out.parent.mkdir()
    with pytest.raises(bc.BackupIntegrityError):
        bc.decrypt_file(enc, out, _KEY)
    assert not out.exists()
    assert list(out.parent.iterdir()) == []


def test_decrypt_refuses_to_overwrite_unless_asked(tmp_path: Path) -> None:
    src = tmp_path / "a.db"
    src.write_bytes(b"data")
    enc = tmp_path / "a.db.slmenc"
    bc.encrypt_file(src, enc, _KEY)
    out = tmp_path / "existing.db"
    out.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        bc.decrypt_file(enc, out, _KEY)
    assert out.read_bytes() == b"keep me"
    bc.decrypt_file(enc, out, _KEY, overwrite=True)
    assert out.read_bytes() == b"data"


def test_detect_format(tmp_path: Path) -> None:
    sqlite_file = tmp_path / "legacy.db"
    sqlite_file.write_bytes(bc.SQLITE_MAGIC + b"\x00" * 100)
    enc = tmp_path / "new.slmenc"
    bc.encrypt_file(sqlite_file, enc, _KEY)
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"not a backup at all")
    assert bc.detect_format(sqlite_file) == bc.FORMAT_SQLITE
    assert bc.detect_format(enc) == bc.FORMAT_ENCRYPTED
    assert bc.detect_format(junk) == bc.FORMAT_UNKNOWN
