"""CLI mutations must use the resident daemon as the single SQLite writer."""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch


def test_delete_routes_to_resident_daemon(capsys) -> None:
    from superlocalmemory.cli.commands import cmd_delete

    args = Namespace(fact_id="fact-1", yes=True, json=True)
    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch(
            "superlocalmemory.cli.daemon.daemon_request",
            return_value={"success": True, "deleted": "fact-1"},
        ) as request,
        patch(
            "superlocalmemory.core.engine.MemoryEngine.initialize",
            side_effect=AssertionError("must not cold-start a second engine"),
        ),
    ):
        cmd_delete(args)

    request.assert_called_once_with(
        "DELETE", "/api/memories/fact-1", preserve_not_found=True,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["data"]["deleted"] == "fact-1"


def test_update_routes_to_resident_daemon(capsys) -> None:
    from superlocalmemory.cli.commands import cmd_update

    args = Namespace(fact_id="fact 1", content="Updated content.", json=True)
    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch(
            "superlocalmemory.cli.daemon.daemon_request",
            return_value={
                "success": True,
                "predecessor_fact_id": "fact 1",
                "successor_fact_id": "successor-1",
                "correction_case": {"case_id": "case-1", "status": "proposed", "version": 0},
                "review_required": True,
            },
        ) as request,
        patch(
            "superlocalmemory.core.engine.MemoryEngine.initialize",
            side_effect=AssertionError("must not cold-start a second engine"),
        ),
    ):
        cmd_update(args)

    request.assert_called_once_with(
        "PATCH",
        "/api/memories/fact%201",
        {"content": "Updated content."},
        preserve_not_found=True,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["data"]["predecessor_fact_id"] == "fact 1"
    assert payload["data"]["correction_case"]["status"] == "proposed"
    assert payload["data"]["review_required"] is True


def test_review_correction_routes_to_resident_daemon(capsys) -> None:
    from superlocalmemory.cli.commands import cmd_review_correction

    args = Namespace(
        case_id="case 1",
        action="apply",
        expected_version=0,
        event_valid_until="2026-08-16T00:00:00Z",
        json=True,
    )
    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch(
            "superlocalmemory.cli.daemon.daemon_request",
            return_value={"success": True, "correction_case": {"status": "applied"}},
        ) as request,
    ):
        cmd_review_correction(args)

    request.assert_called_once_with(
        "POST",
        "/api/corrections/case%201/apply",
        {"expected_version": 0, "event_valid_until": "2026-08-16T00:00:00Z"},
    )
    assert json.loads(capsys.readouterr().out)["data"]["correction_case"]["status"] == "applied"
