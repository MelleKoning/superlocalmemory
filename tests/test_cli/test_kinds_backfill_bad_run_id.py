# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Issue #148, expanded scope: ``slm kinds backfill pause|resume|cancel|revert RUN_ID``.

``_backfill`` interpolates ``args.run_id`` straight into
``f"/backfill/{args.run_id}/{action}"`` -- the same unvalidated-path-segment
bug as ``_cmd_ops_resolve``. ``daemon_request`` already has a catch-all
``except Exception: return None``, so this site never surfaced a raw
traceback, but a bad run ID silently produced the misleading "SLM daemon is
not running" message instead of naming the real problem. A real backfill
run_id is ``uuid.uuid4().hex[:16]`` (``storage/memory_kind_store.py``), so it
is validated the same way an operation ID is.
"""

from __future__ import annotations

from argparse import Namespace

import pytest

from superlocalmemory.cli.daemon_paths import InvalidDaemonId
from superlocalmemory.cli.kinds_cmd import _backfill


class _NeverCalledDaemon:
    def __call__(self, *args, **kwargs):
        raise AssertionError("daemon_request must not be called for an invalid run_id")


@pytest.mark.parametrize("bad_run_id", ["…", "a/b c?"])
def test_invalid_run_id_rejected_before_any_daemon_call(bad_run_id, monkeypatch, capsys):
    monkeypatch.setattr("superlocalmemory.cli.kinds_cmd.daemon_request", _NeverCalledDaemon())
    args = Namespace(backfill_command="cancel", run_id=bad_run_id, json=False)

    with pytest.raises(SystemExit) as exc_info:
        _backfill(args)

    assert exc_info.value.code != 0
    out = capsys.readouterr().out
    # The raw bad value must never be echoed unescaped (repr-style instead,
    # e.g. '…' rather than a literal U+2026); static English prose in
    # the message may still use ordinary punctuation like an em dash, same
    # as the rest of this CLI's user-facing text.
    assert "…" not in out
    assert "invalid run ID" in out


def test_error_message_names_invalid_run_id(monkeypatch, capsys):
    monkeypatch.setattr("superlocalmemory.cli.kinds_cmd.daemon_request", _NeverCalledDaemon())
    args = Namespace(backfill_command="cancel", run_id="…", json=False)

    with pytest.raises(SystemExit):
        _backfill(args)

    out = capsys.readouterr().out
    assert "invalid run ID" in out


def test_invalid_daemon_id_class_importable_for_run_id_validation():
    # Guards the design: the same helper used for operation_id is reused
    # here, rather than a parallel, drifting regex.
    from superlocalmemory.cli.daemon_paths import validate_daemon_id

    with pytest.raises(InvalidDaemonId):
        validate_daemon_id("…", label="run ID")
