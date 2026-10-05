# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""Saved views on disk: the ``saved_views`` table in learning.db (M054).

PROFILE SCOPE IS IN EVERY STATEMENT
-----------------------------------
Every read and write names ``profile_id`` in its WHERE clause; there is no
method that reads across profiles except :meth:`ViewStore.erase_profile` and
:meth:`ViewStore.export_profile`, which take one profile too. A view saved in
one profile cannot be listed, run, renamed or deleted from another, and a name
lookup in profile B for a view that exists only in A answers "not found", the
same as for a name nobody used.

The store never creates the table or the database file: the migration owns
the schema. On a store the migration has not reached, every call raises
``views_unavailable`` rather than inventing a second, unversioned schema.

Writes are short ``BEGIN IMMEDIATE`` transactions, so the CLI, the daemon and
an MCP process can share the file; SQLite serialises them.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from superlocalmemory.views.model import (
    MAX_VIEWS_PER_PROFILE,
    TOO_MANY_VIEWS,
    VIEW_EXISTS,
    VIEW_NOT_FOUND,
    VIEWS_UNAVAILABLE,
    SavedView,
    ViewError,
    name_key,
    validate_filters,
    validate_limit,
    validate_name,
    validate_query,
)

_COLUMNS = ("view_id, profile_id, name, query, filters_json, result_limit, "
            "created_at, updated_at")
_UNAVAILABLE = ("Saved views are not ready on this computer yet. Restart SLM once "
                "(slm restart) so it can finish updating, then try again.")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _profile(profile_id: object) -> str:
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("profile_id must be a non-empty string")
    return profile_id.strip()


def _row_to_view(row: tuple) -> SavedView:
    view_id, profile_id, name, query, filters_json, limit, created, updated = row
    # Written only by this module from validated values; read defensively so a
    # hand-edited row degrades to "no filters" rather than breaking the list.
    try:
        stored = json.loads(filters_json or "{}")
    except ValueError:
        stored = {}
    if not isinstance(stored, dict):
        stored = {}
    filters = tuple(sorted((str(k), str(v)) for k, v in stored.items()))
    return SavedView(view_id=view_id, profile_id=profile_id, name=name, query=query,
                     filters=filters, limit=int(limit), created_at=created,
                     updated_at=updated)


def _not_found(name: str) -> ViewError:
    return ViewError(VIEW_NOT_FOUND, f"There is no saved view called {name!r}.",
                     field="name")


class ViewStore:
    """Profile-scoped CRUD over ``saved_views``. Returns immutable views."""

    def __init__(self, learning_db: str | Path) -> None:
        self._path = Path(learning_db)

    # -- plumbing -------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if not self._path.exists():
            raise ViewError(VIEWS_UNAVAILABLE, _UNAVAILABLE)
        conn = sqlite3.connect(str(self._path), timeout=10.0, isolation_level=None)
        conn.execute("PRAGMA busy_timeout=10000")
        present = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='saved_views'"
        ).fetchone()
        if not present:
            conn.close()
            raise ViewError(VIEWS_UNAVAILABLE, _UNAVAILABLE)
        return conn

    @staticmethod
    def _find(conn: sqlite3.Connection, profile: str, name: str) -> SavedView | None:
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM saved_views WHERE profile_id = ? AND name_key = ?",
            (profile, name_key(name)),
        ).fetchone()
        return _row_to_view(row) if row else None

    # -- reads ----------------------------------------------------------------

    def list(self, profile_id: str) -> tuple[SavedView, ...]:
        """Every view in the profile, in name order (case-insensitive), stable."""
        profile = _profile(profile_id)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM saved_views WHERE profile_id = ? "
                "ORDER BY name_key ASC, view_id ASC", (profile,),
            ).fetchall()
        return tuple(_row_to_view(r) for r in rows)

    def get(self, profile_id: str, name: object) -> SavedView:
        profile, clean = _profile(profile_id), validate_name(name)
        with closing(self._connect()) as conn:
            view = self._find(conn, profile, clean)
        if view is None:
            raise _not_found(clean)
        return view

    # -- writes ---------------------------------------------------------------

    def create(self, profile_id: str, *, name: object, query: object,
               filters: object = None, limit: object = None) -> SavedView:
        """Save a new view. Refuses a name already used in this profile."""
        profile = _profile(profile_id)
        clean_name = validate_name(name)
        clean_query = validate_query(query)
        clean_filters = validate_filters(filters)
        clean_limit = validate_limit(limit)
        stamp = _now()
        view = SavedView(view_id=uuid.uuid4().hex, profile_id=profile, name=clean_name,
                         query=clean_query, filters=clean_filters, limit=clean_limit,
                         created_at=stamp, updated_at=stamp)
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                count = conn.execute("SELECT COUNT(*) FROM saved_views WHERE profile_id = ?",
                                     (profile,)).fetchone()[0]
                if count >= MAX_VIEWS_PER_PROFILE:
                    raise ViewError(TOO_MANY_VIEWS,
                                    f"A profile can keep at most {MAX_VIEWS_PER_PROFILE} "
                                    "saved views. Delete one you no longer use first.")
                if self._find(conn, profile, clean_name) is not None:
                    raise ViewError(VIEW_EXISTS,
                                    f"A view called {clean_name!r} already exists. "
                                    "Pick another name or rename the old one.",
                                    field="name")
                conn.execute(
                    "INSERT INTO saved_views (profile_id, view_id, name, name_key, query, "
                    "filters_json, result_limit, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (profile, view.view_id, view.name, name_key(view.name), view.query,
                     json.dumps(view.filter_map, sort_keys=True), view.limit,
                     view.created_at, view.updated_at),
                )
                conn.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                # The unique index is the last word on names, should a writer
                # that did not take this lock ever get in between.
                conn.execute("ROLLBACK")
                raise ViewError(VIEW_EXISTS, f"A view called {clean_name!r} already exists.",
                                field="name") from exc
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return view

    def rename(self, profile_id: str, name: object, new_name: object) -> SavedView:
        """Give a view a new name. Its query, filters and id are unchanged."""
        profile = _profile(profile_id)
        old, new = validate_name(name), validate_name(new_name)
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                current = self._find(conn, profile, old)
                if current is None:
                    raise _not_found(old)
                clash = self._find(conn, profile, new)
                if clash is not None and clash.view_id != current.view_id:
                    raise ViewError(VIEW_EXISTS, f"A view called {new!r} already exists.",
                                    field="new_name")
                stamp = _now()
                conn.execute(
                    "UPDATE saved_views SET name = ?, name_key = ?, updated_at = ? "
                    "WHERE profile_id = ? AND view_id = ?",
                    (new, name_key(new), stamp, profile, current.view_id),
                )
                conn.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                conn.execute("ROLLBACK")
                raise ViewError(VIEW_EXISTS, f"A view called {new!r} already exists.",
                                field="new_name") from exc
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return SavedView(view_id=current.view_id, profile_id=profile, name=new,
                         query=current.query, filters=current.filters, limit=current.limit,
                         created_at=current.created_at, updated_at=stamp)

    def delete(self, profile_id: str, name: object) -> SavedView:
        """Remove a view (only the saved query; no memory is touched)."""
        profile, clean = _profile(profile_id), validate_name(name)
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                current = self._find(conn, profile, clean)
                if current is None:
                    raise _not_found(clean)
                conn.execute("DELETE FROM saved_views WHERE profile_id = ? AND view_id = ?",
                             (profile, current.view_id))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return current

    # -- data rights (GDPR) ---------------------------------------------------

    def export_profile(self, profile_id: str) -> list[dict]:
        """Every view of one profile, as plain data, for an access request."""
        return [v.to_dict() for v in self.list(profile_id)]

    def erase_profile(self, profile_id: str) -> int:
        """Delete every view of one profile. Returns how many were removed.

        A store without the table has nothing to erase, so that is 0, not an
        error: erasure must not fail on a store that never had views.
        """
        profile = _profile(profile_id)
        try:
            conn = self._connect()
        except ViewError:
            return 0
        with closing(conn):
            cursor = conn.execute("DELETE FROM saved_views WHERE profile_id = ?", (profile,))
            return int(cursor.rowcount or 0)


def default_store() -> ViewStore:
    """The store for this installation: learning.db under the data root."""
    from superlocalmemory.infra.data_root import state_path

    return ViewStore(state_path("learning.db"))


__all__ = ["ViewStore", "default_store"]
