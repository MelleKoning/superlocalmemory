# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The upgrade log line states no invented fact count (audit 4.1.20 L8)."""

from __future__ import annotations

import re
from pathlib import Path

from superlocalmemory.server.unified_daemon import upgrade_banner

SRC = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory" / "server" / "unified_daemon.py"


def test_banner_names_both_versions_and_no_count() -> None:
    line = upgrade_banner("4.1.19", "4.1.20")
    assert "4.1.19 → 4.1.20" in line
    assert "restore point" in line
    assert not re.search(r"\d+k\+|\d[\d,]* (atomic )?facts", line)


def test_no_hardcoded_fact_count_remains_in_the_daemon() -> None:
    assert "18k+" not in SRC.read_text(encoding="utf-8")
