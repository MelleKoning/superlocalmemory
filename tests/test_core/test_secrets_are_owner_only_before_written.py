# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Each secret file is made owner-only while it is still empty.

``0o600`` means nothing on Windows, so these writers left an answer-check key,
the feedback hashing key and ``config.json`` (which can hold API keys) with the
folder's inherited access list there. Each now goes through
``infra.owner_only_acl.restrict_to_owner`` — chmod 600 on POSIX, an owner-only
access list on Windows — before the secret is written. This records the size
of the file at that moment, on any platform.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.infra import owner_only_acl


@pytest.fixture()
def restricted(monkeypatch) -> list[tuple[str, int]]:
    seen: list[tuple[str, int]] = []
    real = owner_only_acl.restrict_to_owner

    def recording(path) -> None:
        path = Path(path)
        seen.append((path.name, path.stat().st_size))
        real(path)

    monkeypatch.setattr(owner_only_acl, "restrict_to_owner", recording)
    return seen


def test_the_answer_check_key_on_windows(restricted, tmp_path):
    from superlocalmemory.core.judge_keys import JudgeKeyStore

    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    final = secrets_dir / "jev-typesafe.key"
    JudgeKeyStore(slm_home=tmp_path)._set_key_by_path(
        secrets_dir, final, "typesafe", "sk-live-" + "a1B2c3D4" * 4)
    assert [size for _name, size in restricted] == [0]
    assert final.read_text(encoding="ascii").startswith("sk-live-")


def test_the_feedback_hashing_key(restricted, tmp_path):
    from superlocalmemory.learning.feedback import _load_or_create_hash_key

    key = _load_or_create_hash_key(tmp_path / "learning.db")
    assert restricted == [(".feedback-hash-key", 0)]
    assert (tmp_path / ".feedback-hash-key").read_bytes() == key


def test_config_json(restricted, tmp_path):
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.models import Mode

    SLMConfig.for_mode(Mode.B).save(tmp_path / "config.json")
    assert [size for _name, size in restricted] == [0]
    assert (tmp_path / "config.json").stat().st_size > 0
