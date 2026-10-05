# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Issue #148, expanded scope: every daemon-path call site, not just ops resolve.

``_switch_profile_runtime`` interpolates ``profile_name`` straight into
``f"/api/profiles/{profile_name}/switch"`` the same way ``_cmd_ops_resolve``
interpolated ``operation_id`` -- the same class of bug. ``daemon_request``
(``cli/daemon.py``) already has a catch-all ``except Exception: return
None``, so this particular site never surfaced a traceback, but a bad name
silently reached (or tried to reach) the daemon and came back as the
misleading "resident daemon did not acknowledge the profile switch" instead
of naming the real problem. This test asserts the bad value is rejected
before any daemon call is attempted, with a friendly, ASCII-safe message.
"""

from __future__ import annotations

import pytest

from superlocalmemory.cli.commands import _switch_profile_runtime
from superlocalmemory.cli.daemon_paths import InvalidDaemonId


class _NeverCalledDaemon:
    """Fails the test if the CLI ever tries to reach a daemon."""

    def __call__(self, *args, **kwargs):
        raise AssertionError("daemon_request must not be called for an invalid profile name")


@pytest.mark.parametrize("bad_name", ["…", "a/b c?", ""])
def test_invalid_profile_name_rejected_before_any_daemon_call(bad_name, monkeypatch):
    monkeypatch.setattr(
        "superlocalmemory.cli.daemon.is_daemon_running",
        lambda: (_ for _ in ()).throw(AssertionError("must not even check daemon state")),
    )
    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", _NeverCalledDaemon())

    with pytest.raises(InvalidDaemonId):
        _switch_profile_runtime(config=None, profile_name=bad_name)


def test_error_message_is_ascii_safe_and_names_the_field():
    with pytest.raises(InvalidDaemonId) as exc_info:
        _switch_profile_runtime(config=None, profile_name="…")

    message = str(exc_info.value)
    message.encode("ascii")  # must never raise -- no raw non-ASCII leaked
    assert "profile name" in message
