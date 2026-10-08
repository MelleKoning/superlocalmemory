# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — RBAC / teams (C3)

"""Role-based access control for the memory tenancy model.

The tenancy boundary is the *profile* (individual / team / company). This
engine adds real, persisted user identity and role grants on top of that
boundary, following the Cognee reference model (Tenant -> User -> Role).

Design principles
-----------------
* **Additive / self-hosting-correct.** With zero users the daemon stays
  single-operator: the machine operator (install token / daemon capability
  over loopback) is the implicit *owner* with every permission, so individual
  use has no login friction. Once an org creates users + memberships from the
  dashboard, logged-in users are enforced against their role.
* **Enforced, not decorative.** ``has_permission`` is called from the write /
  delete / share / manage routes. This module is the single source of truth
  for the role -> permission matrix.
* **No plaintext secrets.** Passwords are hashed with ``hashlib.scrypt`` and a
  per-user random salt; session tokens are stored only as SHA-256 hashes and
  compared in constant time.

All state lives in memory.db (tables from migration M024). Connections are
short-lived (opened per call) with a bounded busy timeout, matching the mesh
broker's concurrency-safe pattern.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from superlocalmemory.storage import store_signature
from superlocalmemory.storage.memory_write import memory_write

logger = logging.getLogger("superlocalmemory.access.rbac")

# Match the daemon's default (SLM_DB_BUSY_TIMEOUT_MS, default 10 000 ms).
# Reads use this via _conn(); writes go through memory_write() which reads the
# same env var internally, so in-process writers never see SQLITE_BUSY.
_BUSY_TIMEOUT_MS: int = max(0, int(os.environ.get("SLM_DB_BUSY_TIMEOUT_MS", "10000")))
_SESSION_TTL_HOURS = 12

# scrypt cost parameters (OWASP-recommended interactive-login range).
_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32


class Role(str, Enum):
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"
    MANAGE = "manage"  # manage users, roles, profile settings


# Role -> permission set. This is the single source of truth for authorization.
_ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.ADMIN: frozenset(Permission),
    Role.MEMBER: frozenset({Permission.READ, Permission.WRITE, Permission.SHARE}),
    Role.VIEWER: frozenset({Permission.READ}),
}


def permissions_for_role(role: Role) -> frozenset[Permission]:
    return _ROLE_PERMISSIONS.get(role, frozenset())


class RbacError(ValueError):
    """Raised for invalid RBAC operations (bad role, duplicate user, ...)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
    return uuid.uuid4().hex[:16]


def _hash_password(password: str) -> str:
    """scrypt$N$r$p$salt_hex$dk_hex — self-describing so params can evolve."""
    if not password or len(password) < 8:
        raise RbacError("Password must be at least 8 characters.")
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(
        password.encode("utf-8"), salt=salt,
        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_hex, dk_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        dk = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p), dklen=len(bytes.fromhex(dk_hex)),
        )
        return hmac.compare_digest(dk.hex(), dk_hex)
    except (ValueError, TypeError):
        return False


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


# Constant dummy hash so verify_credentials spends ~equal time on unknown and
# known usernames (defeats the username-enumeration timing oracle). Computed
# once at import.
_DUMMY_PASSWORD_HASH = _hash_password("__slm_dummy_sentinel__never_a_real_pw__")


class RbacEngine:
    """Persisted user / membership / session store with role enforcement."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        #: key -> (store signature when read, value). See ``get_policy``.
        self._policy_memo: dict[str, tuple[Any, str]] = {}

    # -- connection -------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
        try:
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            conn.row_factory = sqlite3.Row
        except BaseException:
            # Setup failed: close it now, or the file stays open (Windows
            # then cannot delete or replace it) for as long as the error lives.
            conn.close()
            raise
        return conn

    # -- users ------------------------------------------------------------

    def user_count(self) -> int:
        conn = self._conn()
        try:
            return int(conn.execute("SELECT COUNT(*) FROM rbac_users").fetchone()[0])
        finally:
            conn.close()

    def create_user(self, username: str, password: str, display_name: str = "",
                    created_by: str = "owner") -> dict:
        username = (username or "").strip()
        if not username:
            raise RbacError("Username is required.")
        pw_hash = _hash_password(password)
        user_id = _uid()
        # memory_write: process write lock + busy_timeout.
        with memory_write(self._db_path) as conn:
            # The UNIQUE(username) constraint is the source of truth; the SELECT
            # is a fast path. Two concurrent creates can both pass the SELECT, so
            # the loser's INSERT hits the constraint — convert that to RbacError
            # (409) instead of leaking a raw IntegrityError as a 500.
            if conn.execute(
                "SELECT 1 FROM rbac_users WHERE username=?", (username,)
            ).fetchone():
                raise RbacError(f"User '{username}' already exists.")
            try:
                conn.execute(
                    "INSERT INTO rbac_users (user_id, username, display_name, "
                    "password_hash, status, created_at, created_by) "
                    "VALUES (?, ?, ?, ?, 'active', ?, ?)",
                    (user_id, username, display_name or username, pw_hash,
                     _now(), created_by),
                )
            except sqlite3.IntegrityError:
                raise RbacError(f"User '{username}' already exists.")
        logger.info("RBAC: created user '%s' (%s)", username, user_id)
        return {"user_id": user_id, "username": username,
                "display_name": display_name or username, "status": "active"}

    def set_password(self, user_id: str, password: str) -> None:
        pw_hash = _hash_password(password)
        with memory_write(self._db_path) as conn:
            cur = conn.execute(
                "UPDATE rbac_users SET password_hash=? WHERE user_id=?",
                (pw_hash, user_id),
            )
            if cur.rowcount == 0:
                raise RbacError("User not found.")
            # A password change invalidates every existing session.
            conn.execute("DELETE FROM rbac_sessions WHERE user_id=?", (user_id,))

    def set_status(self, user_id: str, status: str) -> None:
        if status not in ("active", "disabled"):
            raise RbacError("status must be 'active' or 'disabled'.")
        with memory_write(self._db_path) as conn:
            cur = conn.execute(
                "UPDATE rbac_users SET status=? WHERE user_id=?", (status, user_id)
            )
            if cur.rowcount == 0:
                raise RbacError("User not found.")
            if status == "disabled":
                conn.execute("DELETE FROM rbac_sessions WHERE user_id=?", (user_id,))

    def delete_user(self, user_id: str) -> None:
        with memory_write(self._db_path) as conn:
            conn.execute("DELETE FROM rbac_sessions WHERE user_id=?", (user_id,))
            conn.execute("DELETE FROM rbac_memberships WHERE user_id=?", (user_id,))
            cur = conn.execute("DELETE FROM rbac_users WHERE user_id=?", (user_id,))
            if cur.rowcount == 0:
                raise RbacError("User not found.")

    def get_user(self, user_id: str) -> dict | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT user_id, username, display_name, status, created_at, created_by "
                "FROM rbac_users WHERE user_id=?", (user_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_users(self) -> list[dict]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT user_id, username, display_name, status, created_at, created_by "
                "FROM rbac_users ORDER BY username"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # -- credentials / sessions ------------------------------------------

    def verify_credentials(self, username: str, password: str) -> dict | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT user_id, username, display_name, password_hash, status "
                "FROM rbac_users WHERE username=?", ((username or "").strip(),)
            ).fetchone()
        finally:
            conn.close()
        if not row or row["status"] != "active":
            # Run a verify against a constant dummy hash so an unknown/disabled
            # username costs the same wall-time as a wrong password — no timing
            # oracle for username enumeration.
            _verify_password(password, _DUMMY_PASSWORD_HASH)
            return None
        if not _verify_password(password, row["password_hash"]):
            return None
        return {"user_id": row["user_id"], "username": row["username"],
                "display_name": row["display_name"]}

    def create_session(self, user_id: str, ttl_hours: int = _SESSION_TTL_HOURS) -> str:
        """Mint an opaque session token; only its hash is stored. Returns the
        raw token (shown to the client once)."""
        raw = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires = (now + timedelta(hours=ttl_hours)).isoformat()
        with memory_write(self._db_path) as conn:
            conn.execute(
                "INSERT INTO rbac_sessions (token_hash, user_id, created_at, "
                "expires_at, last_seen) VALUES (?, ?, ?, ?, ?)",
                (_hash_token(raw), user_id, now.isoformat(), expires, now.isoformat()),
            )
        return raw

    def resolve_session(self, raw_token: str) -> dict | None:
        """Return the active user for a session token, or None if invalid /
        expired / disabled. Updates last_seen. Constant-time by hash lookup.

        Concurrency design
        ------------------
        Phase 1 (read): WAL allows concurrent readers without blocking the
        writer — ``_conn()`` with ``busy_timeout`` only.  The write lock is
        NOT taken here so a burst of concurrent requests doesn't serialise
        through the single writer on every call.

        Phase 2 (conditional write): only if the session is expired/stale.
        ``memory_write()`` is opened AFTER the read connection is closed
        (lock-ordering invariant: get_write_lock is outermost).
        """
        if not raw_token:
            return None
        token_hash = _hash_token(raw_token)
        now = datetime.now(timezone.utc)

        # -- Phase 1: read-only lookup ----------------------------------------
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT s.user_id, s.expires_at, s.last_seen, "
                "u.username, u.display_name, u.status "
                "FROM rbac_sessions s JOIN rbac_users u ON u.user_id = s.user_id "
                "WHERE s.token_hash=?", (token_hash,)
            ).fetchone()
            if not row:
                return None
            try:
                expired = datetime.fromisoformat(row["expires_at"]) <= now
            except ValueError:
                expired = True
            row_status = row["status"]
            row_last_seen = row["last_seen"]
            user_dict = {
                "user_id": row["user_id"],
                "username": row["username"],
                "display_name": row["display_name"],
            }
        finally:
            conn.close()

        # -- Phase 2: write if needed (read conn closed above) ----------------
        if expired or row_status != "active":
            with memory_write(self._db_path) as wconn:
                wconn.execute(
                    "DELETE FROM rbac_sessions WHERE token_hash=?", (token_hash,)
                )
            return None

        # Debounce last_seen: only write when stale (>60 s), so a burst of
        # authenticated requests does not serialise through the writer every call.
        stale = True
        try:
            stale = (now - datetime.fromisoformat(row_last_seen)).total_seconds() > 60
        except (ValueError, TypeError, KeyError):
            stale = True
        if stale:
            with memory_write(self._db_path) as wconn:
                wconn.execute(
                    "UPDATE rbac_sessions SET last_seen=? WHERE token_hash=?",
                    (now.isoformat(), token_hash),
                )
        return user_dict

    def revoke_session(self, raw_token: str) -> None:
        with memory_write(self._db_path) as conn:
            conn.execute(
                "DELETE FROM rbac_sessions WHERE token_hash=?",
                (_hash_token(raw_token),),
            )

    def purge_expired_sessions(self) -> int:
        with memory_write(self._db_path) as conn:
            cur = conn.execute(
                "DELETE FROM rbac_sessions WHERE expires_at <= ?", (_now(),)
            )
        return cur.rowcount or 0

    # -- memberships ------------------------------------------------------

    def set_membership(self, profile_id: str, user_id: str, role: str,
                       added_by: str = "owner") -> dict:
        try:
            role_enum = Role(role)
        except ValueError:
            raise RbacError(f"Invalid role '{role}'. Use admin/member/viewer.")
        with memory_write(self._db_path) as conn:
            if not conn.execute(
                "SELECT 1 FROM rbac_users WHERE user_id=?", (user_id,)
            ).fetchone():
                raise RbacError("User not found.")
            conn.execute(
                "INSERT INTO rbac_memberships (profile_id, user_id, role, added_at, added_by) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(profile_id, user_id) DO UPDATE SET role=excluded.role, "
                "added_at=excluded.added_at, added_by=excluded.added_by",
                (profile_id, user_id, role_enum.value, _now(), added_by),
            )
        return {"profile_id": profile_id, "user_id": user_id, "role": role_enum.value}

    def remove_membership(self, profile_id: str, user_id: str) -> None:
        with memory_write(self._db_path) as conn:
            conn.execute(
                "DELETE FROM rbac_memberships WHERE profile_id=? AND user_id=?",
                (profile_id, user_id),
            )

    def get_role(self, user_id: str, profile_id: str) -> Role | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT role FROM rbac_memberships WHERE user_id=? AND profile_id=?",
                (user_id, profile_id),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        try:
            return Role(row["role"])
        except ValueError:
            return None

    def list_members(self, profile_id: str) -> list[dict]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT m.user_id, u.username, u.display_name, m.role, m.added_at "
                "FROM rbac_memberships m JOIN rbac_users u ON u.user_id = m.user_id "
                "WHERE m.profile_id=? ORDER BY u.username", (profile_id,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def list_user_profiles(self, user_id: str) -> list[dict]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT profile_id, role FROM rbac_memberships WHERE user_id=?",
                (user_id,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # -- policy settings --------------------------------------------------

    def get_policy(self, key: str, default: str = "") -> str:
        """The stored value, re-read only when something was committed since.

        Every request asks ``require_login()``. Opening a connection for each
        ask put a database open on the service's request loop, so one slow open
        froze every request in flight. A commit by any process changes the
        store's signature (storage/store_signature), read BEFORE the value, so
        a changed policy is seen on the very next ask; an unreadable signature
        means no reuse.
        """
        sig = store_signature.of(self._db_path)
        hit = self._policy_memo.get(key)
        if sig is not None and hit is not None and hit[0] == sig:
            return hit[1] if hit[1] is not None else default
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT value FROM rbac_settings WHERE key=?", (key,)
            ).fetchone()
        finally:
            conn.close()
        value = row["value"] if row else None
        if sig is not None:
            self._policy_memo = {**self._policy_memo, key: (sig, value)}
        return value if value is not None else default

    def set_policy(self, key: str, value: str) -> None:
        with memory_write(self._db_path) as conn:
            conn.execute(
                "INSERT INTO rbac_settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def require_login(self) -> bool:
        """Company mode: mutations require a valid user session (no owner
        bypass). Default off (personal / single-operator use)."""
        return self.get_policy("require_login", "0") == "1"

    def set_require_login(self, enabled: bool) -> None:
        self.set_policy("require_login", "1" if enabled else "0")

    # -- correction review policy -----------------------------------------

    # Key prefix used in rbac_settings.  One entry per profile_id.
    _CORRECTION_POLICY_PREFIX = "correction_review_policy:"

    def set_correction_review_policy(
        self,
        profile_id: str,
        policy: dict,
        *,
        authorizer_user_id: str,
    ) -> dict:
        """Attach a correction-case review policy for ``profile_id``.

        Authorization rules
        -------------------
        * PERSONAL install (no users yet / user_count == 0): the machine
          owner is the implicit admin and may attach a policy without a
          role check.  This matches the additive / self-hosting-correct
          principle in the RBAC design.
        * TEAM / ENTERPRISE install (user_count > 0): the authorizer MUST
          hold Role.ADMIN for the target profile.  Any other role raises
          RbacError.  This prevents a MEMBER or VIEWER from escalating their
          own correction authority.

        Governance invariants (CRIT-hardened)
        --------------------------------------
        C1  Non-admins cannot attach a policy (role check above).
        C2  ``automatic_application`` is never silently promoted to True.
            The field defaults to False; an authorizer must set it
            explicitly, and it is recorded with the authorizer's user_id and
            a timestamp so the action is auditable.
        C3  Personal installs do NOT auto-create a policy.  The machine
            owner must call this method explicitly.  Until they do,
            ``get_correction_review_policy`` returns None and BrainTruth
            reports the policy as not_configured.

        Returns the stored policy dict.
        """
        import json as _json

        if not profile_id:
            raise RbacError("profile_id must be non-empty.")
        if not authorizer_user_id:
            raise RbacError("authorizer_user_id must be non-empty.")

        # CRIT-C1: enforce admin role in multi-user (team/enterprise) mode.
        # In personal mode (zero registered users) the machine owner is the
        # implicit admin — deny-by-default still applies to defined users.
        user_count = self.user_count()
        if user_count > 0:
            role = self.get_role(authorizer_user_id, profile_id)
            if role != Role.ADMIN:
                raise RbacError(
                    f"Attaching a correction review policy requires Role.ADMIN. "
                    f"User '{authorizer_user_id}' has role={role!r} on "
                    f"profile '{profile_id}'."
                )

        # CRIT-C2: automatic_application is never silently set to True.
        # The authorizer must explicitly include it in the policy dict;
        # it is still coerced to bool so a truthy non-bool value doesn't slip through.
        auto_apply = bool(policy.get("automatic_application", False))

        safe_policy = {
            "policy_id": str(policy.get("policy_id") or _uid()),
            "authorized_by": authorizer_user_id,
            "authorized_at": _now(),
            "automatic_application": auto_apply,
            "enabled": bool(policy.get("enabled", True)),
            "applies_to_profile": str(policy.get("applies_to_profile", profile_id)),
        }
        key = f"{self._CORRECTION_POLICY_PREFIX}{profile_id}"
        self.set_policy(key, _json.dumps(safe_policy))
        logger.info(
            "RBAC: correction review policy attached for profile '%s' by '%s' "
            "(automatic_application=%s)",
            profile_id, authorizer_user_id, auto_apply,
        )
        return safe_policy

    def get_correction_review_policy(self, profile_id: str) -> dict | None:
        """Return the attached correction review policy for ``profile_id``.

        Returns None when no policy has been attached.  The caller (host
        integration) should pass this to
        ``BrainTruthService(review_policy=...)``.

        This method is safe to call in personal mode (no users configured).
        """
        import json as _json

        key = f"{self._CORRECTION_POLICY_PREFIX}{profile_id}"
        raw = self.get_policy(key)
        if not raw:
            return None
        try:
            return _json.loads(raw)
        except (ValueError, TypeError):
            logger.warning(
                "RBAC: malformed correction review policy for profile '%s' — "
                "returning None (treat as not_configured)",
                profile_id,
            )
            return None

    # -- authorization ----------------------------------------------------

    def has_permission(self, user_id: str, profile_id: str,
                       permission: Permission | str) -> bool:
        """True iff the user's role on the profile grants ``permission``.

        A user with no membership on the profile has NO access (deny-by-default).
        The implicit machine owner is handled by the caller, not here.
        """
        perm = permission if isinstance(permission, Permission) else Permission(permission)
        role = self.get_role(user_id, profile_id)
        if role is None:
            return False
        return perm in permissions_for_role(role)
