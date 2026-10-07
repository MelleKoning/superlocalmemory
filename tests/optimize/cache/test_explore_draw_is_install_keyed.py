# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""W3: which cached answers are re-checked is keyed by this install's secret.

The draw stays a pure function of the record and the query on one install
(repeatable), but someone who knows a record's state and lacks the install's
token cannot compute it, the same property learning/bandit_draw.py gives the
recall draw.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core import security_primitives
from superlocalmemory.optimize.cache import boundary_store
from superlocalmemory.optimize.cache.boundary_store import PerItemBoundaryRecord


def _draws(rec: PerItemBoundaryRecord) -> list[float]:
    return [rec.explore_draw(q / 100) for q in range(80, 100)]


@pytest.fixture()
def token_file(tmp_path, monkeypatch):
    path = tmp_path / ".install_token"
    monkeypatch.setattr(security_primitives, "_install_token_path", lambda: path)
    monkeypatch.setattr(boundary_store, "_explore_keys", {}, raising=False)
    return path


def test_same_install_same_draws(token_file) -> None:
    token_file.write_text("a" * 64)
    rec = PerItemBoundaryRecord(entry_id="w3", samples=[(0.9, 1), (0.7, 0)])
    assert _draws(rec) == _draws(rec)


def test_another_install_draws_differently(token_file, monkeypatch) -> None:
    rec = PerItemBoundaryRecord(entry_id="w3", samples=[(0.9, 1), (0.7, 0)])
    token_file.write_text("a" * 64)
    first = _draws(rec)
    monkeypatch.setattr(boundary_store, "_explore_keys", {}, raising=False)
    token_file.write_text("b" * 64)
    assert _draws(rec) != first


def test_the_unkeyed_digest_no_longer_predicts_the_draw(token_file) -> None:
    import hashlib
    import json

    token_file.write_text("c" * 64)
    rec = PerItemBoundaryRecord(entry_id="w3", samples=[(0.9, 1)])
    material = json.dumps([rec.entry_id, repr(float(rec.t_hat)), repr(float(rec.gamma_hat)),
                           [[repr(float(s)), int(c)] for s, c in rec.samples], "0.950000"],
                          ensure_ascii=True)
    unkeyed = hashlib.sha256(b"slm:vcache-explore:v1|" + material.encode()).digest()
    assert rec.explore_draw(0.95) != int.from_bytes(unkeyed[:8], "big") / 2.0**64
