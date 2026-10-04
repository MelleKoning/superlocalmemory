# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A fully constructed remember runtime whose writer coordinator is a test double.

The claim-loop tests used to build the runtime with ``__new__`` and set two
attributes by hand. Every field ``start()`` later needs (the journal, the
background committer) was missing, so each new step in ``start()`` broke them.
Constructing the real object and swapping only the coordinator keeps them
honest about what ``start()`` does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def runtime_with_coordinator(tmp_path: Path, coordinator: Any):
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage.database import DatabaseManager

    runtime = CanonicalRememberRuntime(
        db=DatabaseManager(tmp_path / "memory.db"),
        profile_id="default",
        writer=lambda _request, _operation_id: [],
        journal_path=tmp_path / "admission_journal.db",
    )
    runtime.coordinator = coordinator
    return runtime
