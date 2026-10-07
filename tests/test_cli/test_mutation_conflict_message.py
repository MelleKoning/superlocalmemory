# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A daemon's refusal (HTTP 409) is printed as that refusal, never as an outage.

4.1.21: deleting a memory protected by correction history was refused with
409 and ``slm delete`` printed DAEMON_UNAVAILABLE with a restart hint. The
real daemon path is tests/test_integration/test_erasure_integrity_e2e.py.
"""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch

import pytest

REASON = "fact is protected by correction history: it is part of correction case c1 (rejected)"


def _refused(*_args, **kwargs):
    from superlocalmemory.cli.daemon import DaemonConflict

    assert kwargs.get("preserve_conflict") is True
    raise DaemonConflict(REASON)


@pytest.mark.parametrize("use_json", [False, True])
def test_delete_prints_the_daemon_reason(capsys, use_json) -> None:
    from superlocalmemory.cli.commands import cmd_delete

    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch("superlocalmemory.cli.daemon.daemon_request", side_effect=_refused),
        pytest.raises(SystemExit) as exited,
    ):
        cmd_delete(Namespace(fact_id="fact-1", yes=True, json=use_json))

    assert exited.value.code == 1
    out = capsys.readouterr()
    assert "DAEMON_UNAVAILABLE" not in out.out + out.err
    if use_json:
        error = json.loads(out.out)["error"]
        assert error == {"code": "CONFLICT", "message": REASON, "retryable": False}
    else:
        assert out.err.strip() == f"Refused: {REASON}"
