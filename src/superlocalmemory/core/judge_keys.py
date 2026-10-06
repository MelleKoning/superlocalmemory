# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Keys for the hosted answer check (Jev), kept out of config.json.

One owner-only file per provider under ``<SLM_HOME>/secrets/``. A key can be
set, tested, cleared and shown as its last four characters. Nothing returns it
except ``load``, which only the Jev client calls.

Every write is atomic (temp file in the same directory, then ``os.replace``),
and every read or write refuses a key file that is not a regular file owned by
the current user — a symlink there could point the write (or the read)
somewhere this module never intended to touch.

The checks hold for what is actually opened, not just for what the path named
a moment earlier: the folder is held open while the key is written, the key
file is opened without following a link, and the open file must be the one
that was checked (same device and inode). A key read back is stripped of the
whitespace a hand edit leaves and validated again; a file that is not a usable
key raises ``JudgeKeyStoreError`` rather than being sent as a broken header.
"""

from __future__ import annotations

import contextlib
import os
import secrets as _secrets
import stat
import tempfile
from pathlib import Path

PROVIDERS = ("typesafe", "openrouter")

_SECRETS_DIRNAME = "secrets"
_MIN_KEY_LEN = 8
_MAX_KEY_LEN = 4096
# Printable, visible ASCII only: 0x21 '!' .. 0x7E '~'. Excludes the space
# character (0x20) and every control character, so "no whitespace" and
# "printable ASCII" are the same bound.
_MIN_PRINTABLE = 0x21
_MAX_PRINTABLE = 0x7E
#: Read at most this much: a longer file cannot hold a valid key.
_READ_LIMIT = _MAX_KEY_LEN + 64

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
#: A FIFO swapped in for the key file must not block the read.
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
#: Folder-relative calls (POSIX). Elsewhere the path-based checks are used.
_DIR_FD = bool(_O_NOFOLLOW and _O_DIRECTORY and os.open in os.supports_dir_fd)


class JudgeKeyStoreError(RuntimeError):
    """A stored key file failed a safety check and will not be used.

    Raised instead of silently treating the key as absent: a symlink, a
    non-regular file, or a file owned by someone else in this location is
    evidence of tampering, not an ordinary "not configured yet" state.
    """


def _validate_provider(provider: str) -> None:
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}. Choose one of: "
            f"{', '.join(PROVIDERS)}."
        )


def _validate_key(key: str) -> None:
    if not isinstance(key, str) or not (_MIN_KEY_LEN <= len(key) <= _MAX_KEY_LEN):
        raise ValueError(
            f"The key must be {_MIN_KEY_LEN} to {_MAX_KEY_LEN} characters long."
        )
    if any(not (_MIN_PRINTABLE <= ord(ch) <= _MAX_PRINTABLE) for ch in key):
        raise ValueError(
            "The key must contain only printable characters, with no "
            "spaces, tabs, or line breaks."
        )


def is_usable_key(key: object) -> bool:
    """Whether ``key`` can go into a request header as it is.

    For callers that receive a key from elsewhere (the connection test): a
    key with a line break or a space inside must never become a header.
    """
    try:
        _validate_key(key)  # type: ignore[arg-type]
    except ValueError:
        return False
    return True


def _refuse_unsafe(path: Path, *, expect: str) -> os.stat_result:
    """Raise unless ``path`` is a regular file/dir owned by the current user.

    ``path.lstat()`` is used (never ``stat()``) so a symlink is judged on
    what it IS, not on whatever it points at. Returns that ``lstat`` so a
    caller can prove the thing it later opens is the thing checked here.
    """
    st = path.lstat()
    _refuse_stat(st, path.name, expect=expect)
    return st


def _refuse_stat(st: os.stat_result, name: str, *, expect: str) -> None:
    if stat.S_ISLNK(st.st_mode):
        raise JudgeKeyStoreError(
            f"{name} is a symlink; refusing to use it."
        )
    if expect == "file" and not stat.S_ISREG(st.st_mode):
        raise JudgeKeyStoreError(
            f"{name} is not a regular file; refusing to use it."
        )
    if expect == "dir" and not stat.S_ISDIR(st.st_mode):
        raise JudgeKeyStoreError(
            f"{name} is not a directory; refusing to use it."
        )
    try:
        current_uid = os.getuid()
    except AttributeError:
        return  # Windows: no POSIX ownership to check.
    if st.st_uid != current_uid:
        raise JudgeKeyStoreError(
            f"{name} is not owned by the current user; refusing to use it."
        )


def _refuse_shared_write(st: os.stat_result, name: str) -> None:
    """Another account that can change the key decides where memories go."""
    if os.name == "posix" and st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise JudgeKeyStoreError(
            f"{name} can be changed by other users on this computer; refusing "
            "to use it. Save the key again in Settings → Answer check."
        )


def _same_file(before: os.stat_result, after: os.stat_result, name: str) -> None:
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise JudgeKeyStoreError(
            f"{name} changed while it was being read; refusing to use it."
        )


def _clean_key(raw: bytes, provider: str) -> str | None:
    """The stored key without a hand edit's surrounding whitespace.

    None when the file holds nothing but whitespace (nothing saved). Anything
    else that is not a valid key raises — the message never quotes the file.
    """
    unusable = JudgeKeyStoreError(
        f"The saved key for {provider} is not usable: it has spaces or line "
        "breaks inside it, characters other than plain letters, digits and "
        "symbols, or the wrong length. Save it again in Settings → Answer check."
    )
    if len(raw) > _READ_LIMIT:
        raise unusable
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        raise unusable from None
    key = text.strip()
    if not key:
        return None
    try:
        _validate_key(key)
    except ValueError:
        raise unusable from None
    return key


def _read_fd(fd: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total <= _READ_LIMIT:
        chunk = os.read(fd, _READ_LIMIT + 1 - total)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


class JudgeKeyStore:
    """Owner-only, provider-keyed storage for the hosted answer check's key."""

    def __init__(self, slm_home: Path | None = None) -> None:
        if slm_home is not None:
            self._home = Path(slm_home)
        else:
            from superlocalmemory.infra.data_root import canonical_data_root

            self._home = canonical_data_root()

    # -- paths --------------------------------------------------------------

    def _secrets_dir(self) -> Path:
        return self._home / _SECRETS_DIRNAME

    def _key_path(self, provider: str) -> Path:
        return self._secrets_dir() / f"jev-{provider}.key"

    def _ensure_secrets_dir(self) -> tuple[Path, os.stat_result]:
        secrets_dir = self._secrets_dir()
        if not (secrets_dir.exists() or secrets_dir.is_symlink()):
            secrets_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        st = _refuse_unsafe(secrets_dir, expect="dir")
        return secrets_dir, st

    @contextlib.contextmanager
    def _held_dir(self, secrets_dir: Path, checked: os.stat_result):
        """The checked folder, held open so a later swap cannot redirect us."""
        try:
            dfd = os.open(secrets_dir, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC)
        except OSError as exc:
            raise JudgeKeyStoreError(
                f"{secrets_dir.name} changed while it was being used; refusing to use it."
            ) from exc
        try:
            _same_file(checked, os.fstat(dfd), secrets_dir.name)
            yield dfd
        finally:
            os.close(dfd)

    # -- writes ---------------------------------------------------------

    def set_key(self, provider: str, key: str) -> None:
        """Validate and store. Raises ValueError with a plain-language reason."""
        _validate_provider(provider)
        _validate_key(key)
        secrets_dir, dir_st = self._ensure_secrets_dir()
        final_path = secrets_dir / f"jev-{provider}.key"
        if not _DIR_FD:  # pragma: no cover — Windows
            self._set_key_by_path(secrets_dir, final_path, provider, key)
            return
        with self._held_dir(secrets_dir, dir_st) as dfd:
            os.fchmod(dfd, 0o700)
            try:
                existing = os.stat(final_path.name, dir_fd=dfd, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                _refuse_stat(existing, final_path.name, expect="file")
            tmp_name = f".jev-{provider}.{_secrets.token_hex(8)}.tmp"
            fd = os.open(
                tmp_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC,
                0o600,
                dir_fd=dfd,
            )
            try:
                with os.fdopen(fd, "w", encoding="ascii") as handle:
                    handle.write(key)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, final_path.name, src_dir_fd=dfd, dst_dir_fd=dfd)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name, dir_fd=dfd)
                raise
            with contextlib.suppress(OSError):
                os.fsync(dfd)

    def _set_key_by_path(self, secrets_dir: Path, final_path: Path,
                         provider: str, key: str) -> None:  # pragma: no cover — Windows
        os.chmod(secrets_dir, 0o700)
        if final_path.exists() or final_path.is_symlink():
            _refuse_unsafe(final_path, expect="file")
        fd, tmp_name = tempfile.mkstemp(
            dir=secrets_dir, prefix=f".jev-{provider}.", suffix=".tmp"
        )
        try:
            # Windows ignores 0o600: give the empty file an owner-only access
            # list before the key goes in. Refuses (raises) if it cannot.
            from superlocalmemory.infra.owner_only_acl import restrict_to_owner

            try:
                restrict_to_owner(Path(tmp_name))
            except BaseException:
                os.close(fd)  # so the clean-up below can delete it on Windows
                raise
            with os.fdopen(fd, "w", encoding="ascii") as handle:
                handle.write(key)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, final_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

    def clear(self, provider: str) -> None:
        """Remove only this provider's key file.

        ``unlink`` never follows a symlink for its final path component, and
        the folder is checked and held open first, so this only ever removes
        the entry ``jev-<provider>.key`` inside SLM's own secrets folder.
        """
        _validate_provider(provider)
        secrets_dir = self._secrets_dir()
        if not (secrets_dir.exists() or secrets_dir.is_symlink()):
            return
        dir_st = _refuse_unsafe(secrets_dir, expect="dir")
        if not _DIR_FD:  # pragma: no cover — Windows
            with contextlib.suppress(FileNotFoundError):
                self._key_path(provider).unlink()
            return
        with self._held_dir(secrets_dir, dir_st) as dfd:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(f"jev-{provider}.key", dir_fd=dfd)

    # -- reads ------------------------------------------------------------

    def load(self, provider: str) -> str | None:
        """The key itself. Only the Jev client may call this."""
        _validate_provider(provider)
        path = self._key_path(provider)
        secrets_dir = path.parent
        if not (secrets_dir.exists() or secrets_dir.is_symlink()):
            return None
        dir_st = _refuse_unsafe(secrets_dir, expect="dir")
        _refuse_shared_write(dir_st, secrets_dir.name)
        if not path.exists() and not path.is_symlink():
            return None
        checked = _refuse_unsafe(path, expect="file")
        try:
            raw = self._read_checked(secrets_dir, dir_st, path, checked)
        except JudgeKeyStoreError:
            raise
        except OSError as exc:
            raise JudgeKeyStoreError(
                f"Could not read the stored key for {provider}."
            ) from exc
        return _clean_key(raw, provider)

    def _read_checked(self, secrets_dir: Path, dir_st: os.stat_result,
                      path: Path, checked: os.stat_result) -> bytes:
        flags = os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC | getattr(os, "O_BINARY", 0)
        if _DIR_FD:
            with self._held_dir(secrets_dir, dir_st) as dfd:
                fd = os.open(path.name, flags, dir_fd=dfd)
        else:  # pragma: no cover — Windows
            fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            _refuse_stat(opened, path.name, expect="file")
            _same_file(checked, opened, path.name)
            _refuse_shared_write(opened, path.name)
            return _read_fd(fd)
        finally:
            os.close(fd)

    def has_key(self, provider: str) -> bool:
        return self.load(provider) is not None

    def masked(self, provider: str) -> str:
        """"****abcd", or "" when no key is stored."""
        key = self.load(provider)
        if not key:
            return ""
        if len(key) > 8:
            return "****" + key[-4:]
        return "****"


__all__ = ["PROVIDERS", "JudgeKeyStore", "JudgeKeyStoreError", "is_usable_key"]
