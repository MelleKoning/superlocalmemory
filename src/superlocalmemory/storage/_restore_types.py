# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The records the upgrade / restore / downgrade flow hands to every surface.

Each has ``as_dict()`` with plain JSON types, so the dashboard, the CLI's
``--json`` and any tool see the same fields under the same names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Files and folders the flow owns inside the data directory. Nothing outside
#: these, ``pre-migration-snapshots/`` and ``pre-restore/`` is ever removed.
INTENT_NAME = "restore-intent.json"
OUTCOME_NAME = "restore-outcome.json"
DELTA_DIR = "restore-delta"
SAFETY_DIR = "pre-restore"
SNAPSHOTS_DIR = "pre-migration-snapshots"
DOWNGRADE_MARKER = ".downgrade-prepared"
REPAIR_RECORD = "upgrade-repair.json"


class RestoreRefusedError(RuntimeError):
    """A restore was not started; nothing was changed. The message is for a person."""


class DowngradeRefusedError(RuntimeError):
    """Prepare-for-downgrade was refused; nothing was changed."""


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class _AsDict:
    def as_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))  # type: ignore[call-overload]


@dataclass(frozen=True)
class RestorePoint(_AsDict):
    """One snapshot generation: the copies taken together before one change."""

    point_id: str
    created_at: str
    reason: str
    from_version: str | None
    to_version: str | None
    memory_snapshot: Path
    learning_snapshot: Path | None
    size_bytes: int
    sha256: dict[str, str] | None
    facts: int | None
    legacy: bool                      # True = no manifest (taken by an older build)
    repair: dict[str, Any] | None = None   # one-time repair that ran after this copy
    schema_version: int | None = None      # the store version stamped in the copy
    restorable: bool = True                # False: this build cannot open the copy
    not_restorable_reason: str | None = None


@dataclass(frozen=True)
class RestorePreview(_AsDict):
    point_id: str
    verified: bool
    facts_in_snapshot: int
    facts_now: int
    memories_in_snapshot: int
    memories_now: int
    added_memories: int               # kept aside, added back after the restore
    deleted_facts: int                # forgotten or erased after the copy; stay deleted
    deleted_memories: int
    kind_edits: int                   # confirmed kinds; applied again
    corrections_lost: int             # cannot be replayed: shown before confirming
    erasures: int                     # erased after the copy; stay erased
    repair: dict[str, Any] | None
    disk_needed_bytes: int
    disk_free_bytes: int
    disk_ok: bool
    problems: list[str] = field(default_factory=list)   # any of these blocks the restore
    returning_facts: int = 0          # missing now, no deletion recorded: they come back
    returning_memories: int = 0
    restorable: bool = True           # False: the copy is newer than this build
    live_store_readable: bool = True  # False: full restore, nothing carried over
    learning_records_lost: int = 0    # learning recorded since the copy; not carried over
    warnings: list[str] = field(default_factory=list)   # shown; do not block


@dataclass(frozen=True)
class RestoreIntent(_AsDict):
    point_id: str
    requested_by: str
    requested_at: str
    reimport: bool
    delta_dir: Path
    intent_path: Path
    preview: dict[str, Any]


@dataclass(frozen=True)
class RestoreOutcome(_AsDict):
    status: str        # restored | refused | snapshot_unusable | failed | invalid
    point_id: str | None
    message: str
    safety_copies: list[str] = field(default_factory=list)
    delta_dir: str | None = None
    reimport_pending: bool = False
    applied: dict[str, int] = field(default_factory=dict)
    projections: str = "not_checked"


@dataclass(frozen=True)
class ReimportReport(_AsDict):
    added: int = 0
    already_present: int = 0
    skipped_unknown_profile: int = 0
    skipped_rejected: int = 0
    failed: int = 0
    kinds_reapplied: int = 0
    kinds_skipped: int = 0
    kinds_waiting: int = 0            # its memory is still enriching; tried at next start
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DowngradeReport(_AsDict):
    prepared: bool
    target_schema: int
    point_id: str | None
    message: str
    marker: str | None = None
    #: The release to install, and the exact commands, when known.
    target_version: str | None = None
    install_commands: list[str] = field(default_factory=list)
