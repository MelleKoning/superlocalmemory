# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""On a shared Windows computer, another account's daemon is not ours.

Windows has no uid, and the account check answered "this account" for every
process there. So a stale ``daemon.pid`` naming another account's old daemon
was adopted (seen on the Windows CI runner). Windows names the account a
process runs as (``DOMAIN\\user``); that is compared instead. Runs anywhere by
removing ``os.getuid`` the way Windows lacks it.
"""

from __future__ import annotations

import getpass
import os

import pytest

from superlocalmemory.cli import daemon as cli_daemon


class _Process:
    def __init__(self, username) -> None:
        self._username = username

    def username(self):
        if isinstance(self._username, Exception):
            raise self._username
        return self._username


@pytest.fixture()
def windows_account(monkeypatch):
    """No uid, and this process runs as ``OFFICE-PC\\<this user>``."""
    monkeypatch.delattr(os, "getuid", raising=False)
    monkeypatch.setenv("USERDOMAIN", "OFFICE-PC")
    return f"OFFICE-PC\\{getpass.getuser()}"


def test_a_process_of_this_account_is_ours(windows_account):
    assert cli_daemon._process_is_this_account(_Process(windows_account)) is True
    # Windows account names are not case-sensitive.
    assert cli_daemon._process_is_this_account(_Process(windows_account.upper())) is True


@pytest.mark.parametrize("other", ["OFFICE-PC\\someone-else", "OTHER-PC\\{me}"])
def test_another_accounts_process_is_not_ours(windows_account, other):
    other = other.format(me=getpass.getuser())
    assert cli_daemon._process_is_this_account(_Process(other)) is False


def test_a_process_whose_account_cannot_be_read_is_not_ours(windows_account):
    import psutil

    denied = psutil.AccessDenied(pid=4242)
    assert cli_daemon._process_is_this_account(_Process(denied)) is False
