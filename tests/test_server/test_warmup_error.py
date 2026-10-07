# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``sanitized_warmup_error``: the reason text ``/health`` and ``slm warmup``
surface for a stuck embedding warmup (4.1.22)."""

from __future__ import annotations

from pathlib import Path

from superlocalmemory.server.warmup_error import sanitized_warmup_error


def test_keeps_the_exception_type_and_message():
    out = sanitized_warmup_error(ValueError("no cached model, offline"))
    assert out == "ValueError: no cached model, offline"


def test_strips_this_machines_home_directory():
    home = str(Path.home())
    out = sanitized_warmup_error(OSError(f"cache miss under {home}/.cache/huggingface"))
    assert home not in out
    assert "<HOME>" in out


def test_truncates_a_very_long_message():
    out = sanitized_warmup_error(RuntimeError("x" * 500))
    assert len(out) <= 200
